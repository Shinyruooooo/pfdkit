import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict, field

import ase
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter, ExpCellFilter
from pfd.constants import ase_log_name, ase_traj_name



@dataclass
class MDParameters:
    """MD simulation parameters that can be serialized."""
    # For bot optimization and MD
    ensemble: str = "nvt"  # "nvt", "npt", or "both"
    # for MD
    temp: float = 300.0  # K
    press: Optional[float] = None  # Bar (None for NVT)
    dt: float = 2.0  # fs
    nsteps: int = 30000  # NVT production steps
    traj_freq: int = 100  # frames
    log_freq: int = 100  # steps
    tau_t: float = 100.0  # temperature coupling time in fs
    tau_p: float = 1000  # pressure coupling time in fs (NPT)
    compressibility: Optional[float] = None  # 1/bar (NPT), material-dependent, required for NPT (no default)
    seed: Optional[int] = None  # random seed for velocity initialization; None keeps non-deterministic behavior
    no_pbc: bool = False  # disable periodic boundary conditions
    vol_tol: Optional[float] = 0.2  # allowed relative cell volume change in stability monitor; None disables the volume check
    max_force: Optional[float] = 50.0  # max per-atom force threshold in eV/Angstrom in stability monitor; None disables the force check
    custom_config: Dict[str, Any] = field(default_factory=dict)   # Custom configuration
    output_prefix: str = "md"
    ## for optimization
    max_step: int =1000 # maximum steps in optimization
    scalar_pressure: float = 0.0  # target scalar pressure for optimization
    fmax: float = 0.01  # force convergence criterion for optimization
    constant_volume: bool = False  # whether to keep volume constant during optimization
    filter_config: Dict[str, Any] = field(default_factory=dict)  # Custom filter configuration
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MDParameters':
        return cls(**json.loads(json_str))
    
    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'MDParameters':
        with open(filename, 'r') as f:
            return cls.from_json(f.read())
        

class MDStabilityError(RuntimeError):
    """MD terminated because the trajectory became physically unstable."""


def _json_value(x):
    """Convert a value to a JSON-safe native Python scalar; None if unavailable/non-finite."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


class MDStabilityMonitor:
    """Watchdog for MLIP-driven MD: aborts on blow-up instead of running on.

    Attached to the dynamics with ``dyn.attach(monitor.check, interval=1)``.
    Raises ``MDStabilityError`` when any of the following is detected:

    - temperature above ``max_temp`` (K)
    - cell volume outside ``(1 +- vol_tol) * V0`` where V0 is the initial volume
      (skipped when ``vol_tol`` is None)
    - max per-atom force norm above ``max_force`` (eV/Angstrom)
      (skipped when ``max_force`` is None)
    - NaN or inf in energy, forces, stress (if available) or positions

    On violation the current structure is written to ``fail_file`` and a JSON
    diagnostic summary to ``diag_file`` for later analysis before the
    exception is raised. Failures while saving diagnostics never mask the
    original ``MDStabilityError``.
    """

    def __init__(
        self,
        atoms: Atoms,
        fail_file: str = "md_failed.extxyz",
        diag_file: str = "md_failed.json",
        max_temp: float = 5000.0,
        vol_tol: Optional[float] = 0.2,
        max_force: Optional[float] = 50.0,
        total_steps: Optional[int] = None,
    ):
        if max_force is not None and max_force <= 0:
            raise ValueError(
                f"max_force must be positive (eV/Angstrom) or None, got {max_force}"
            )
        self.atoms = atoms
        self.fail_file = fail_file
        self.diag_file = diag_file
        self.max_temp = max_temp
        self.vol_tol = vol_tol
        self.max_force = max_force
        self.total_steps = total_steps
        self.v0 = atoms.get_volume()
        # number of MD steps completed; observers are also called once at the
        # initial state (step 0), so the counter is incremented after each check
        self.step = 0

    def check(self) -> None:
        reasons = []

        positions = self.atoms.get_positions()
        if not np.isfinite(positions).all():
            reasons.append("NaN/inf in positions")

        energy = None
        try:
            energy = self.atoms.get_potential_energy()
            if not np.isfinite(energy):
                reasons.append("NaN/inf in energy")
        except Exception as e:
            reasons.append(f"energy evaluation failed: {e}")

        fmax = None
        try:
            forces = self.atoms.get_forces()
            if not np.isfinite(forces).all():
                reasons.append("NaN/inf in forces")
            else:
                fmax = np.linalg.norm(forces, axis=1).max()
                if self.max_force is not None and fmax > self.max_force:
                    reasons.append(
                        f"max force {fmax:.2f} eV/Ang exceeds limit {self.max_force} eV/Ang"
                    )
        except Exception as e:
            reasons.append(f"force evaluation failed: {e}")

        try:
            stress = self.atoms.get_stress()
            if not np.isfinite(stress).all():
                reasons.append("NaN/inf in stress")
        except Exception:
            pass  # calculator does not provide stress; skip

        temp = self.atoms.get_temperature()
        if not np.isfinite(temp) or temp > self.max_temp:
            reasons.append(f"temperature {temp:.1f} K exceeds limit {self.max_temp} K")

        volume = self.atoms.get_volume()
        volume_ratio = None
        if not np.isfinite(volume) or self.v0 <= 0:
            reasons.append("invalid cell volume")
        else:
            volume_ratio = volume / self.v0
            if self.vol_tol is not None and (
                volume_ratio > 1.0 + self.vol_tol or volume_ratio < 1.0 - self.vol_tol
            ):
                reasons.append(
                    f"cell volume changed by {(volume_ratio - 1.0) * 100:.1f}% "
                    f"(allowed +/-{self.vol_tol * 100:.0f}%)"
                )

        step = self.step
        self.step += 1
        if reasons:
            self._save_diagnostics(reasons, step, temp, energy, fmax, volume, volume_ratio)
            raise MDStabilityError(
                "MD stability check failed: " + "; ".join(reasons)
            )

    def _save_diagnostics(self, reasons, step, temp, energy, fmax, volume, volume_ratio) -> None:
        """Best-effort dump of the failed structure and a JSON summary.

        Never raises: diagnostics must not mask the original MDStabilityError.
        """
        try:
            write(self.fail_file, self.atoms, format="extxyz")
        except Exception:
            pass  # structure may be unwritable (e.g. NaN positions)
        diag = {
            "reason": "; ".join(reasons),
            "step": int(step),
            "temperature_K": _json_value(temp),
            "energy_eV": _json_value(energy),
            "fmax_eV_per_A": _json_value(fmax),
            "volume_A3": _json_value(volume),
            "initial_volume_A3": _json_value(self.v0),
            "volume_ratio": _json_value(volume_ratio),
            "requested_nsteps": int(self.total_steps) if self.total_steps is not None else None,
            "completed_steps": int(step),
            "early_stopped": True,
        }
        try:
            with open(self.diag_file, "w") as f:
                json.dump(diag, f, indent=2)
        except Exception:
            pass


class MDRunner:
    """
    MD simulation runner that wraps an ASE Atoms object.
    
    Examples
    --------
    >>> # Create from structure file
    >>> md_runner = MDRunner.from_file("structure.cif")
    >>> md_runner.atoms.set_calculator(calculator)
    >>> 
    >>> # Run MD with parameters from JSON
    >>> md_runner.run_md_from_json("md_params.json")
    >>> 
    >>> # Or run with direct parameters
    >>> params = MDParameters(temperature=500.0, nsteps_nvt=50000)
    >>> md_runner.run_md(params)
    """

    def __init__(self, atoms: Atoms):
        """Initialize MDRunner with an Atoms object."""
        self.atoms = atoms
        self.md_history = []
        self.logger = self._setup_logger()
    
    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self.atoms)
    
    def set_calculator(self, calc) -> None:
        """Set calculator for the atoms."""
        self.atoms.set_calculator(calc)
    
    @property
    def calc(self):
        """Get the calculator."""
        return self.atoms.calc
    
    @calc.setter
    def calc(self, calc):
        """Set the calculator."""
        self.atoms.calc = calc

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for MD simulation."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'MDRunner':
        """Create MDRunner from structure file."""
        atoms_data = read(filename, index=0)
        # Ensure we have a single Atoms object, not a list
        if isinstance(atoms_data, list):
            atoms = atoms_data[0]
        else:
            atoms = atoms_data
        return cls(atoms)
    
    @classmethod
    def from_atoms(cls, atoms: Atoms) -> 'MDRunner':
        """Create MDRunner from existing Atoms object."""
        return cls(atoms)
    
    def initialize_velocities(self, temperature: float, seed: Optional[int] = None) -> None:
        """Initialize Maxwell-Boltzmann velocity distribution.

        Uses a local random number generator when ``seed`` is given, so the
        process-global ``np.random`` state is left untouched. ``seed=None``
        keeps the non-deterministic default behavior.
        """
        rng = np.random.RandomState(seed) if seed is not None else None
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature, rng=rng)
        self.logger.info(f"Initialized velocities at {temperature} K")
    
    def run_npt(self, 
                params: MDParameters,
                log_file: Optional[str] = ase_log_name,
                traj_file: Optional[str] = ase_traj_name
                ) -> None:
        """Run NPT simulation."""

        if params.press is None:
            raise ValueError("Pressure must be specified for NPT simulation")
        if params.compressibility is None:
            raise ValueError(
                "NPT ensemble requires compressibility (1/bar). "
                "Please provide material-dependent value."
            )
        timestep = params.dt * units.fs
        
        # Initialize velocities
        self.initialize_velocities(params.temp, seed=params.seed)
        
        # Setup NPT dynamics
        dyn = NPTBerendsen(
            self.atoms,
            timestep,
            temperature_K=params.temp,
            pressure_au=params.press * units.bar,
            taut=params.tau_t * units.fs,
            taup=params.tau_p * units.fs,
            compressibility_au=params.compressibility/units.bar,
            logfile=log_file,
            loginterval=params.log_freq,
            **params.custom_config
        )
        traj = Trajectory(traj_file, 'w', atoms=self.atoms)
        dyn.attach(traj.write, interval=params.traj_freq)
        # stability watchdog: abort on blow-up / NaN instead of running on garbage
        monitor = MDStabilityMonitor(
            self.atoms, vol_tol=params.vol_tol, max_force=params.max_force,
            total_steps=params.nsteps,
        )
        dyn.attach(monitor.check, interval=1)
            # Run NPT
        self.logger.info("#### Starting MD...")
        start_time = time.time()
        try:
            dyn.run(params.nsteps)
        finally:
            traj.close()
        elapsed = time.time() - start_time
        self.logger.info(f"#### MD simulation completed in {elapsed:.2f} s")
        
        # Store history
        self.md_history.append({
            'type': 'NPT',
            'steps': params.nsteps,
            'temperature': params.temp,
            'pressure': params.press,
            'duration': elapsed
        })
    
    def run_nvt(
        self, 
        params: MDParameters,
        log_file: Optional[str] = ase_log_name,
        traj_file: Optional[str] = ase_traj_name
        ) -> None:
        """Run NVT simulation."""
        timestep = params.dt * units.fs
        tdamp = params.tau_t * units.fs  # temperature coupling time in fs, same convention as NPT
        
        # Initialize velocities if not already done
        if not hasattr(self, '_velocities_initialized'):
            self.initialize_velocities(params.temp, seed=params.seed)
            self._velocities_initialized = True
        
        # Setup NVT dynamics
        dyn = NoseHooverChainNVT(
            self.atoms,
            timestep,
            temperature_K=params.temp,
            tdamp=tdamp,
            logfile=log_file,
            loginterval=params.log_freq,
            **params.custom_config
        )
        traj = Trajectory(traj_file, 'w', atoms=self.atoms)
        dyn.attach(traj.write, interval=params.traj_freq)
        # stability watchdog: abort on blow-up / NaN instead of running on garbage
        monitor = MDStabilityMonitor(
            self.atoms, vol_tol=params.vol_tol, max_force=params.max_force,
            total_steps=params.nsteps,
        )
        dyn.attach(monitor.check, interval=1)

        # Run NVT
        self.logger.info("#### Starting NVT simulation...")
        start_time = time.time()
        try:
            dyn.run(params.nsteps)
        finally:
            traj.close()
        elapsed = time.time() - start_time
        self.logger.info(f"#### NVT simulation completed in {elapsed:.2f} s")
        # Store history
        self.md_history.append({
            'type': 'NVT',
            'steps': params.nsteps,
            'temperature': params.temp,
            'duration': elapsed
        })
    
    def run_md(self, params: MDParameters,**kwargs) -> None:
        """Run MD simulation based on ensemble parameter."""
        if not hasattr(self.atoms, 'calc') or self.atoms.calc is None:
            raise ValueError("Calculator must be set before running MD")
        
        if params.no_pbc:
            self.atoms.set_pbc(False)
            
        self.logger.info(f"Starting MD simulation with {len(self.atoms)} atoms")
        self.logger.info(f"Ensemble: {params.ensemble.upper()}")
        self.logger.info(f"Temperature: {params.temp} K")
        
        total_start = time.time()
        
        if params.ensemble.lower() == "npt":
            self.run_npt(params)
        elif params.ensemble.lower() == "nvt":
            self.run_nvt(params)

        elif params.ensemble.lower() == "lbfgs":
            self.run_opt_LBFGS(params,**kwargs)
        else:
            raise ValueError(f"Unknown ensemble: {params.ensemble}")
        
        total_elapsed = time.time() - total_start
        self.logger.info(f"#### Total MD simulation completed in {total_elapsed:.2f} s")

    def run_opt_LBFGS(self, 
        params: MDParameters,
        log_file: str = ase_log_name,
        traj_file: str = ase_traj_name) -> None:
        """Run MD simulation using parameters from dictionary."""
        
        # add unitcell filters
        ucf = UnitCellFilter(
            self.atoms,
            scalar_pressure=params.scalar_pressure*units.GPa,
            constant_volume=params.constant_volume,
            **params.filter_config
            )
        dyn = LBFGS(
            ucf, 
            trajectory=traj_file, 
            logfile=log_file,
            #maxstep= params.max_step,
            **params.custom_config
            )
        traj = Trajectory(traj_file, 'w', atoms=self.atoms)
        dyn.attach(traj.write, interval=params.traj_freq)
        
        self.logger.info("#### Starting LBFGS optimization...")
        start_time = time.time()
        dyn.run(fmax=params.fmax,
                steps=params.max_step)
        elapsed = time.time() - start_time
        self.logger.info(f"#### LBFGS optimization completed in {elapsed:.2f} s")
        # Store history
        self.md_history.append({
            'type': 'opt-LBFGS',
            #'steps': params.nsteps,
            'duration': elapsed
        })
        #params = MDParameters(**config)
        #self.run_md(params)
    
    def run_md_from_json(
        self, 
        json_file: Union[str, Path],
        **kwargs
        ) -> None:
        """Run MD simulation using parameters from JSON file."""
        params = MDParameters.from_file(json_file)
        self.logger.info(f"Loaded MD parameters from {json_file}")
        self.run_md(params, **kwargs)

    def run_md_from_dict(self, params_dict: Dict[str, Any], **kwargs) -> None:
        """Run MD simulation using parameters from dictionary."""
        params = MDParameters(**params_dict)
        self.run_md(params)
    
    def get_md_summary(self) -> Dict[str, Any]:
        """Get summary of completed MD runs."""
        return {
            'total_runs': len(self.md_history),
            'runs': self.md_history,
            'total_steps': sum(run['steps'] for run in self.md_history),
            'total_duration': sum(run['duration'] for run in self.md_history)
        }

