import unittest
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT

from pfd.exploration.md.ase import MDRunner, MDParameters
from pfd.exploration.md.ase_calc import CalculatorWrapper, EMTCalculatorWrapper


class TestMDParameters(unittest.TestCase):
    """Test MDParameters dataclass functionality."""
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)
    
    def tearDown(self):
        shutil.rmtree(self.tmpdir)
    
    def test_md_parameters_creation(self):
        """Test MDParameters creation with defaults."""
        params = MDParameters()
        self.assertEqual(params.temp, 300.0)
        self.assertEqual(params.ensemble, "nvt")
        self.assertEqual(params.nsteps, 30000)
        self.assertIsNone(params.press)
    
    def test_md_parameters_custom(self):
        """Test MDParameters with custom values."""
        params = MDParameters(
            temp=500.0,
            press=1.0,
            dt=1.0,
            nsteps=1000,
            ensemble="npt"
        )
        self.assertEqual(params.temp, 500.0)
        self.assertEqual(params.press, 1.0)
        self.assertEqual(params.dt, 1.0)
        self.assertEqual(params.nsteps, 1000)
        self.assertEqual(params.ensemble, "npt")
    
    def test_md_parameters_json_serialization(self):
        """Test JSON serialization and deserialization."""
        params = MDParameters(temp=400.0, nsteps=5000)
        
        # Test to_json
        json_str = params.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("400.0", json_str)
        self.assertIn("5000", json_str)
        
        # Test from_json
        params_restored = MDParameters.from_json(json_str)
        self.assertEqual(params_restored.temp, 400.0)
        self.assertEqual(params_restored.nsteps, 5000)
        self.assertEqual(params_restored.ensemble, "nvt")
    
    def test_md_parameters_file_io(self):
        """Test file I/O operations."""
        params = MDParameters(temp=350.0, nsteps=2000)
        json_file = self.test_dir / "params.json"
        
        # Write to file
        with open(json_file, 'w') as f:
            f.write(params.to_json())
        
        # Read from file
        params_loaded = MDParameters.from_file(json_file)
        self.assertEqual(params_loaded.temp, 350.0)
        self.assertEqual(params_loaded.nsteps, 2000)


class TestCalculatorWrapper(unittest.TestCase):
    """Test calculator wrapper functionality."""
    
    def test_emt_calculator_registration(self):
        """Test EMT calculator is properly registered."""
        calc_names = CalculatorWrapper.get_all_calculator()
        self.assertIn('emt', calc_names)
    
    def test_emt_calculator_creation(self):
        """Test EMT calculator creation."""
        calc_wrapper = CalculatorWrapper.get_calculator('emt')
        calc = calc_wrapper().create()
        
        # Check it's the right type
        self.assertIsInstance(calc, EMT)
    
    def test_unknown_calculator(self):
        """Test error handling for unknown calculator."""
        with self.assertRaises(RuntimeError):
            CalculatorWrapper.get_calculator('nonexistent_calc')


class TestMDRunner(unittest.TestCase):
    """Test MDRunner functionality with EMT calculator."""
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)
        
        # Create test atoms - simple copper cluster
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        self.atoms = self.atoms * (2, 2, 2)  # 32 atoms
        
        # Perturb positions slightly to make dynamics interesting
        np.random.seed(42)
        positions = self.atoms.get_positions()
        positions += np.random.normal(0, 0.1, positions.shape)
        self.atoms.set_positions(positions)
    
    def tearDown(self):
        shutil.rmtree(self.tmpdir)
    
    def test_mdrunner_creation(self):
        """Test MDRunner creation from atoms."""
        md_runner = MDRunner(self.atoms.copy())
        self.assertEqual(len(md_runner), 32)
        self.assertIsInstance(md_runner.atoms, Atoms)
    
    def test_mdrunner_from_atoms(self):
        """Test MDRunner.from_atoms class method."""
        md_runner = MDRunner.from_atoms(self.atoms.copy())
        self.assertEqual(len(md_runner), 32)
    
    def test_mdrunner_from_file(self):
        """Test MDRunner.from_file class method."""
        # Write atoms to file
        structure_file = self.test_dir / "test_structure.cif"
        self.atoms.write(structure_file)
        
        # Create MDRunner from file
        md_runner = MDRunner.from_file(structure_file)
        self.assertEqual(len(md_runner), 32)
        self.assertEqual(md_runner.atoms.get_chemical_symbols(), 
                        self.atoms.get_chemical_symbols())
    
    def test_calculator_setting(self):
        """Test calculator setting and property access."""
        md_runner = MDRunner(self.atoms.copy())
        calc = EMT()
        
        # Test set_calculator method
        md_runner.set_calculator(calc)
        self.assertIsInstance(md_runner.calc, EMT)
        
        # Test calc property setter
        calc2 = EMT()
        md_runner.calc = calc2
        self.assertIs(md_runner.calc, calc2)
    
    def test_velocity_initialization(self):
        """Test velocity initialization."""
        md_runner = MDRunner(self.atoms.copy())
        
        # Initialize velocities
        md_runner.initialize_velocities(300.0, seed=42)
        
        # Check velocities were set
        velocities = md_runner.atoms.get_velocities()
        self.assertIsNotNone(velocities)
        self.assertEqual(velocities.shape, (32, 3))
        
        # Check they're not all zero
        self.assertGreater(np.abs(velocities).sum(), 0.0)
    
    def test_short_nvt_simulation(self):
        """Test short NVT MD simulation with EMT."""
        md_runner = MDRunner(self.atoms.copy())
        
        # Set EMT calculator
        calc = EMT()
        md_runner.set_calculator(calc)
        
        # Create short simulation parameters
        params = MDParameters(
            temp=300.0,
            dt=1.0,  # 1 fs timestep
            nsteps=10,  # Very short simulation
            traj_freq=5,  # Write every 5 steps
            log_freq=5,
            ensemble="nvt"
        )
        
        # Run simulation
        original_dir = Path.cwd()
        try:
            import os
            os.chdir(self.test_dir)
            
            md_runner.run_nvt(
                params,
                log_file="test_nvt.log",
                traj_file="test_nvt.traj"
            )
            
            # Check files were created
            self.assertTrue((self.test_dir / "test_nvt.log").exists())
            self.assertTrue((self.test_dir / "test_nvt.traj").exists())
            
            # Check MD history
            self.assertEqual(len(md_runner.md_history), 1)
            history = md_runner.md_history[0]
            self.assertEqual(history['type'], 'NVT')
            self.assertEqual(history['steps'], 10)
            self.assertEqual(history['temperature'], 300.0)
            self.assertGreater(history['duration'], 0.0)
            
        finally:
            os.chdir(original_dir)
    
    def test_short_npt_simulation(self):
        """Test short NPT MD simulation with EMT."""
        md_runner = MDRunner(self.atoms.copy())
        
        # Set EMT calculator
        calc = EMT()
        md_runner.set_calculator(calc)
        
        # Create short NPT simulation parameters
        params = MDParameters(
            temp=300.0,
            press=1.0,  # 1 bar
            dt=1.0,
            nsteps=10,
            traj_freq=5,
            log_freq=5,
            ensemble="npt",
            compressibility=4.5e-5,  # 1/bar, required for NPT
        )
        
        # Run simulation
        original_dir = Path.cwd()
        try:
            import os
            os.chdir(self.test_dir)
            
            md_runner.run_npt(
                params,
                log_file="test_npt.log",
                traj_file="test_npt.traj"
            )
            
            # Check files were created
            self.assertTrue((self.test_dir / "test_npt.log").exists())
            self.assertTrue((self.test_dir / "test_npt.traj").exists())
            
            # Check MD history
            self.assertEqual(len(md_runner.md_history), 1)
            history = md_runner.md_history[0]
            self.assertEqual(history['type'], 'NPT')
            self.assertEqual(history['pressure'], 1.0)
            
        finally:
            os.chdir(original_dir)
    
    def test_run_md_from_parameters(self):
        """Test run_md method with MDParameters."""
        md_runner = MDRunner(self.atoms.copy())
        md_runner.set_calculator(EMT())
        
        params = MDParameters(
            temp=250.0,
            dt=1.0,
            nsteps=5,
            ensemble="nvt"
        )
        
        original_dir = Path.cwd()
        try:
            import os
            os.chdir(self.test_dir)
            md_runner.run_md(params)
            
            # Check simulation ran
            self.assertEqual(len(md_runner.md_history), 1)
            
        finally:
            os.chdir(original_dir)
    
    def test_run_md_from_json(self):
        """Test run_md_from_json method."""
        md_runner = MDRunner(self.atoms.copy())
        md_runner.set_calculator(EMT())
        
        # Create JSON parameter file
        params = MDParameters(
            temp=200.0,
            dt=1.0,
            nsteps=5,
            ensemble="nvt"
        )
        
        json_file = self.test_dir / "md_params.json"
        with open(json_file, 'w') as f:
            f.write(params.to_json())
        
        original_dir = Path.cwd()
        try:
            import os
            os.chdir(self.test_dir)
            md_runner.run_md_from_json(json_file)
            
            # Check simulation ran with correct temperature
            self.assertEqual(len(md_runner.md_history), 1)
            self.assertEqual(md_runner.md_history[0]['temperature'], 200.0)
            
        finally:
            os.chdir(original_dir)
    
    def test_run_md_from_dict(self):
        """Test run_md_from_dict method."""
        md_runner = MDRunner(self.atoms.copy())
        md_runner.set_calculator(EMT())
        
        params_dict = {
            'temp': 150.0,
            'dt': 1.0,
            'nsteps': 5,
            'ensemble': 'nvt'
        }
        
        original_dir = Path.cwd()
        try:
            import os
            os.chdir(self.test_dir)
            md_runner.run_md_from_dict(params_dict)
            
            # Check simulation ran
            self.assertEqual(len(md_runner.md_history), 1)
            self.assertEqual(md_runner.md_history[0]['temperature'], 150.0)
            
        finally:
            os.chdir(original_dir)
    
    def test_md_summary(self):
        """Test MD summary functionality."""
        md_runner = MDRunner(self.atoms.copy())
        md_runner.set_calculator(EMT())
        
        # Run multiple short simulations
        original_dir = Path.cwd()
        try:
            import os
            os.chdir(self.test_dir)
            
            params1 = MDParameters(temp=300.0, nsteps=5, ensemble="nvt")
            params2 = MDParameters(temp=400.0, nsteps=3, ensemble="nvt")
            
            md_runner.run_md(params1)
            md_runner.run_md(params2)
            
            # Check summary
            summary = md_runner.get_md_summary()
            self.assertEqual(summary['total_runs'], 2)
            self.assertEqual(summary['total_steps'], 8)  # 5 + 3
            self.assertGreater(summary['total_duration'], 0.0)
            self.assertEqual(len(summary['runs']), 2)
            
        finally:
            os.chdir(original_dir)
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        md_runner = MDRunner(self.atoms.copy())
        
        # Test running MD without calculator
        params = MDParameters(nsteps=5)
        with self.assertRaises(ValueError):
            md_runner.run_md(params)
        
        # Test NPT without pressure
        md_runner.set_calculator(EMT())
        npt_params = MDParameters(ensemble="npt", press=None)
        with self.assertRaises(ValueError):
            md_runner.run_npt(npt_params)
        
        # Test unknown ensemble
        bad_params = MDParameters(ensemble="unknown")
        with self.assertRaises(ValueError):
            md_runner.run_md(bad_params)


class TestMDStabilityMonitor(unittest.TestCase):
    """Test NPT compressibility requirement, stability monitor, and velocity seed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (2, 2, 2)
        self.atoms.set_calculator(EMT())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_npt_requires_compressibility(self):
        """NPT without compressibility must raise ValueError."""
        from pfd.exploration.md.ase import MDRunner
        md_runner = MDRunner(self.atoms.copy())
        md_runner.set_calculator(EMT())
        params = MDParameters(
            temp=300.0, press=1.0, dt=1.0, nsteps=5, ensemble="npt",
            compressibility=None,
        )
        with self.assertRaises(ValueError) as ctx:
            md_runner.run_npt(params)
        self.assertIn("compressibility", str(ctx.exception))

    def test_monitor_cell_expansion(self):
        """Cell volume expansion beyond tolerance raises MDStabilityError and dumps diagnostics."""
        import json
        from pfd.exploration.md.ase import MDStabilityMonitor, MDStabilityError
        atoms = self.atoms.copy()
        atoms.set_calculator(EMT())
        fail_file = str(self.test_dir / "md_failed.extxyz")
        diag_file = str(self.test_dir / "md_failed.json")
        monitor = MDStabilityMonitor(atoms, fail_file=fail_file, diag_file=diag_file)
        # healthy state passes
        monitor.check()
        # expand cell by 30% in volume (> 20% tolerance)
        atoms.set_cell(atoms.cell * (1.3 ** (1 / 3)), scale_atoms=True)
        with self.assertRaises(MDStabilityError) as ctx:
            monitor.check()
        self.assertIn("volume", str(ctx.exception))
        self.assertTrue((self.test_dir / "md_failed.extxyz").exists())
        self.assertTrue((self.test_dir / "md_failed.json").exists())
        with open(diag_file) as f:
            diag = json.load(f)
        for key in ("reason", "step", "temperature_K", "energy_eV",
                    "fmax_eV_per_A", "volume_A3", "initial_volume_A3",
                    "volume_ratio"):
            self.assertIn(key, diag)
        self.assertIn("volume", diag["reason"])
        self.assertEqual(diag["step"], 1)  # second check call
        self.assertAlmostEqual(diag["volume_ratio"], 1.3, places=3)
        self.assertIsInstance(diag["temperature_K"], float)
        self.assertIsInstance(diag["energy_eV"], float)

    def test_monitor_nan_detection(self):
        """NaN positions trigger MDStabilityError; unwritable values become null in JSON."""
        import json
        from pfd.exploration.md.ase import MDStabilityMonitor, MDStabilityError
        atoms = self.atoms.copy()
        atoms.set_calculator(EMT())
        fail_file = str(self.test_dir / "md_failed.extxyz")
        diag_file = str(self.test_dir / "md_failed.json")
        monitor = MDStabilityMonitor(atoms, fail_file=fail_file, diag_file=diag_file)
        positions = atoms.get_positions()
        positions[0, 0] = np.nan
        atoms.set_positions(positions)
        with self.assertRaises(MDStabilityError):
            monitor.check()
        with open(diag_file) as f:
            diag = json.load(f)
        self.assertIn("NaN", diag["reason"])
        # energy/forces cannot be evaluated with NaN positions -> null
        self.assertIsNone(diag["energy_eV"])
        self.assertIsNone(diag["fmax_eV_per_A"])

    def test_diagnostics_write_failure_does_not_mask_error(self):
        """Unwritable diagnostic paths must not mask the MDStabilityError."""
        from pfd.exploration.md.ase import MDStabilityMonitor, MDStabilityError
        atoms = self.atoms.copy()
        atoms.set_calculator(EMT())
        monitor = MDStabilityMonitor(
            atoms,
            fail_file="/nonexistent_dir_xyz/md_failed.extxyz",
            diag_file="/nonexistent_dir_xyz/md_failed.json",
        )
        positions = atoms.get_positions()
        positions[0, 0] = np.nan
        atoms.set_positions(positions)
        with self.assertRaises(MDStabilityError):
            monitor.check()

    def test_monitor_step_counting(self):
        """Monitor records the number of checks (MD steps) performed."""
        from pfd.exploration.md.ase import MDStabilityMonitor
        atoms = self.atoms.copy()
        atoms.set_calculator(EMT())
        monitor = MDStabilityMonitor(atoms)
        self.assertEqual(monitor.step, 0)
        monitor.check()
        monitor.check()
        self.assertEqual(monitor.step, 2)

    def test_velocity_seed_reproducibility(self):
        """Fixed seed gives identical initial velocities twice."""
        from pfd.exploration.md.ase import MDRunner
        runner1 = MDRunner(self.atoms.copy())
        runner1.initialize_velocities(300.0, seed=12345)
        v1 = runner1.atoms.get_velocities().copy()

        runner2 = MDRunner(self.atoms.copy())
        runner2.initialize_velocities(300.0, seed=12345)
        v2 = runner2.atoms.get_velocities().copy()

        np.testing.assert_array_equal(v1, v2)

    def test_velocity_different_seeds_differ(self):
        """Different seeds give different initial velocities."""
        from pfd.exploration.md.ase import MDRunner
        runner1 = MDRunner(self.atoms.copy())
        runner1.initialize_velocities(300.0, seed=12345)
        v1 = runner1.atoms.get_velocities().copy()

        runner2 = MDRunner(self.atoms.copy())
        runner2.initialize_velocities(300.0, seed=54321)
        v2 = runner2.atoms.get_velocities().copy()

        self.assertFalse(np.allclose(v1, v2))

    def test_seed_does_not_touch_global_rng(self):
        """Seeded velocity init must not change the global np.random state."""
        from pfd.exploration.md.ase import MDRunner
        np.random.seed(2026)
        expected = np.random.random(5)

        np.random.seed(2026)
        runner = MDRunner(self.atoms.copy())
        runner.initialize_velocities(300.0, seed=12345)
        actual = np.random.random(5)

        np.testing.assert_array_equal(actual, expected)

    def test_vol_tol_none_disables_volume_check(self):
        """vol_tol=None disables the volume check; other checks still apply."""
        from pfd.exploration.md.ase import MDStabilityMonitor
        atoms = self.atoms.copy()
        atoms.set_calculator(EMT())
        monitor = MDStabilityMonitor(atoms, vol_tol=None)
        # expand cell by 30% in volume: would fail with default vol_tol=0.2
        atoms.set_cell(atoms.cell * (1.3 ** (1 / 3)), scale_atoms=True)
        monitor.check()  # must not raise

    def test_vol_tol_configurable(self):
        """A relaxed vol_tol allows volume changes beyond the default 20%."""
        from pfd.exploration.md.ase import MDStabilityMonitor
        atoms = self.atoms.copy()
        atoms.set_calculator(EMT())
        monitor = MDStabilityMonitor(atoms, vol_tol=0.5)
        atoms.set_cell(atoms.cell * (1.3 ** (1 / 3)), scale_atoms=True)
        monitor.check()  # 30% < 50% tolerance: must not raise

    def _compressed_atoms_with_force(self):
        """Atoms with a large but finite force (one atom pushed toward a neighbor)."""
        atoms = self.atoms.copy()
        atoms.set_calculator(EMT())
        pos = atoms.get_positions()
        pos[0] += [1.4 / np.sqrt(2), 1.4 / np.sqrt(2), 0.0]  # toward nearest neighbor
        atoms.set_positions(pos)
        fmax = np.linalg.norm(atoms.get_forces(), axis=1).max()
        self.assertGreater(fmax, 0.0)
        return atoms, fmax

    def test_max_force_default_50(self):
        """Default max_force is 50 eV/Ang; a larger force triggers."""
        from pfd.exploration.md.ase import MDStabilityMonitor, MDStabilityError
        atoms, fmax = self._compressed_atoms_with_force()
        self.assertGreater(fmax, 50.0)  # EMT at 0.7x cell exceeds 50 eV/Ang
        monitor = MDStabilityMonitor(atoms, vol_tol=None)  # default max_force=50
        with self.assertRaises(MDStabilityError) as ctx:
            monitor.check()
        self.assertIn("force", str(ctx.exception))

    def test_max_force_custom_value(self):
        """Custom max_force: Fmax below it passes, above it triggers."""
        from pfd.exploration.md.ase import MDStabilityMonitor, MDStabilityError
        atoms, fmax = self._compressed_atoms_with_force()
        relaxed = MDStabilityMonitor(atoms, vol_tol=None, max_force=fmax * 1.1)
        relaxed.check()  # fmax < limit: must not raise
        strict = MDStabilityMonitor(atoms, vol_tol=None, max_force=fmax * 0.9)
        with self.assertRaises(MDStabilityError):
            strict.check()

    def test_max_force_none_disables(self):
        """max_force=None disables the force check; other checks still work."""
        from pfd.exploration.md.ase import MDStabilityMonitor
        atoms, fmax = self._compressed_atoms_with_force()
        monitor = MDStabilityMonitor(atoms, vol_tol=None, max_force=None)
        monitor.check()  # huge force must not raise when disabled

    def test_max_force_invalid(self):
        """max_force <= 0 is rejected at construction."""
        from pfd.exploration.md.ase import MDStabilityMonitor
        with self.assertRaises(ValueError):
            MDStabilityMonitor(self.atoms.copy(), max_force=0)
        with self.assertRaises(ValueError):
            MDStabilityMonitor(self.atoms.copy(), max_force=-5.0)

    def test_diag_early_stop_fields(self):
        """Diagnostics include requested/completed steps and early_stopped flag."""
        import json
        from pfd.exploration.md.ase import MDStabilityMonitor, MDStabilityError
        atoms, fmax = self._compressed_atoms_with_force()
        diag_file = str(self.test_dir / "md_failed.json")
        monitor = MDStabilityMonitor(
            atoms, vol_tol=None, diag_file=diag_file,
            fail_file=str(self.test_dir / "md_failed.extxyz"),
            total_steps=80000,
        )
        with self.assertRaises(MDStabilityError):
            monitor.check()
        with open(diag_file) as f:
            diag = json.load(f)
        self.assertEqual(diag["requested_nsteps"], 80000)
        self.assertEqual(diag["completed_steps"], diag["step"])
        self.assertTrue(diag["early_stopped"])

    def test_normalize_config_explicit_null(self):
        """Explicit JSON null for vol_tol/max_force/seed/compressibility/press
        passes dargs validation and reaches set_md as None."""
        from pfd.exploration.task.ase_task_group import AseTaskGroup
        conf = {
            "conf_idx": [0], "n_sample": 1, "ens": "npt", "dt": 0.5,
            "nsteps": 100, "temps": [1500], "press": [1], "trj_freq": 10,
            "tau_t": 500, "tau_p": 2000, "compressibility": 5e-6,
            "seed": None, "vol_tol": None, "max_force": None,
        }
        data = AseTaskGroup.normalize_config(conf, strict=False)
        self.assertIsNone(data["vol_tol"])
        self.assertIsNone(data["max_force"])
        self.assertIsNone(data["seed"])
        tg = AseTaskGroup()
        tg.set_md(
            temps=data["temps"], press=data["press"],
            **{k: v for k, v in data.items()
               if k not in ("conf_idx", "n_sample", "temps", "press")},
        )
        self.assertIsNone(tg.vol_tol)
        self.assertIsNone(tg.max_force)
        # NVT without press: explicit null must also be accepted
        data_nvt = AseTaskGroup.normalize_config(
            {"temps": [300], "ens": "nvt", "press": None}, strict=False)
        self.assertIsNone(data_nvt["press"])


class TestIntegrationWithCalculatorWrapper(unittest.TestCase):
    """Test integration between MDRunner and CalculatorWrapper."""
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (2, 2, 2)
    
    def tearDown(self):
        shutil.rmtree(self.tmpdir)
    
    def test_integration_with_calculator_wrapper(self):
        """Test MDRunner with calculator from CalculatorWrapper."""
        md_runner = MDRunner(self.atoms.copy())
        
        # Get calculator from wrapper
        calc_wrapper = CalculatorWrapper.get_calculator('emt')
        calc = calc_wrapper().create()
        md_runner.set_calculator(calc)
        
        # Run short simulation
        params = MDParameters(temp=300.0, nsteps=5, ensemble="nvt")
        
        original_dir = Path.cwd()
        try:
            import os
            os.chdir(self.test_dir)
            md_runner.run_md(params)
            
            # Verify simulation completed
            self.assertEqual(len(md_runner.md_history), 1)
            
        finally:
            os.chdir(original_dir)


if __name__ == '__main__':
    unittest.main()
