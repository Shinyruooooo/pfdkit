import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dargs import Argument
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    FatalError,
    TransientError,
)

from pfd.constants import lmp_conf_name, lmp_input_name, lmp_log_name, lmp_traj_name
from pfd.exploration.task.lmp_template_task_group import MODEL_PLACEHOLDER
from pfd.fp.run_fp import _link_input
from pfd.utils import run_command, set_directory


def _parse_thermo(log_text: str) -> Tuple[List[dict], bool]:
    """Parse thermo rows from a LAMMPS log.

    Returns (rows, nan_detected). Each row is a dict keyed by the thermo
    header names. Multiple run blocks are concatenated.
    """
    rows: List[dict] = []
    nan_detected = False
    lines = log_text.split("\n")
    ii = 0
    while ii < len(lines):
        tokens = lines[ii].split()
        if tokens and tokens[0] == "Step" and "Temp" in tokens:
            header = tokens
            ii += 1
            while ii < len(lines):
                row_tokens = lines[ii].split()
                if len(row_tokens) != len(header):
                    break
                try:
                    row = {k: float(v) for k, v in zip(header, row_tokens)}
                except ValueError:
                    if any("nan" in v.lower() for v in row_tokens):
                        nan_detected = True
                    break
                if any(np.isnan(v) or np.isinf(v) for v in row.values()):
                    nan_detected = True
                rows.append(row)
                ii += 1
        ii += 1
    return rows, nan_detected


def _requested_nsteps(input_text: str) -> Optional[int]:
    """Sum the steps of all `run` commands, substituting numeric variables."""
    variables: Dict[str, float] = {}
    for line in input_text.split("\n"):
        tokens = line.split("#")[0].split()
        if len(tokens) >= 4 and tokens[0] == "variable" and tokens[2] in (
            "equal",
            "index",
            "loop",
        ):
            try:
                variables[tokens[1]] = float(tokens[3])
            except ValueError:
                pass
    total = 0
    found = False
    for line in input_text.split("\n"):
        tokens = line.split("#")[0].split()
        if len(tokens) >= 2 and tokens[0] == "run":
            raw = tokens[1]
            m = re.fullmatch(r"\$\{?(\w+)\}?", raw)
            if m and m.group(1) in variables:
                total += int(variables[m.group(1)])
                found = True
            else:
                try:
                    total += int(float(raw))
                    found = True
                except ValueError:
                    pass
    return total if found else None


def _count_dump_frames(traj_path: Path) -> int:
    if not traj_path.is_file():
        return 0
    count = 0
    with open(traj_path) as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                count += 1
    return count


class RunLmp(OP):
    r"""Execute a LAMMPS exploration task with a user-written input template.

    The task directory contains ``in.lammps`` (the revised template) and
    ``conf.lmp`` (the sampled structure). The model is provided as
    ``pfd_model.pt2`` (the placeholder used in the template's pair_style
    line).
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
                "models": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIO(
            {
                "log": Artifact(Path),
                "traj": Artifact(Path),
                # declared for Slices compatibility with PrepRunExpl; never set
                "optional_output": Artifact(Path, optional=True),
                "md_failed": Artifact(Path, optional=True),
                "md_diag": Artifact(Path, optional=True),
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        config = ip["config"] if ip["config"] is not None else {}
        config = RunLmp.normalize_config(config)
        command = config["command"]
        max_temp = config["max_temp"]
        type_map = config.get("type_map")
        task_name = ip["task_name"]
        task_path = Path(ip["task_path"])
        models = ip["models"]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            # clean stale diagnostics from a previous attempt
            for fname in ("md_failed.extxyz", "md_failed.json"):
                stale = Path(fname)
                if stale.is_file():
                    stale.unlink()
            for ii in task_path.resolve().iterdir():
                _link_input(ii.name, ii)
            # provide the model under the placeholder name used in the
            # template's pair_style line
            model_src = Path(models[0]).resolve()
            _link_input(MODEL_PLACEHOLDER, model_src)

            if not Path(lmp_input_name).is_file():
                raise FatalError(f"missing {lmp_input_name} in the task directory")

            cmd = command.split()
            if "-in" not in cmd:
                cmd += ["-in", lmp_input_name]
            if "-log" not in cmd:
                cmd += ["-log", lmp_log_name]
            if "-screen" not in cmd:
                cmd += ["-screen", "lmp_stdout.log"]
            ret, out, err = run_command(" ".join(cmd), shell=True)
            if not Path(lmp_log_name).is_file():
                raise FatalError(
                    f"LAMMPS did not produce {lmp_log_name} (return code {ret}); "
                    f"stderr: {err[-500:]}"
                )
            self._check_and_handle(
                ret=ret,
                work_dir=Path("."),
                max_temp=max_temp,
                type_map=type_map,
            )
            # the traj output must exist even if the run died before the
            # first dump: fall back to a single-frame dump of the input conf
            if _count_dump_frames(Path(lmp_traj_name)) == 0:
                import dpdata

                dpdata.System(lmp_conf_name, fmt="lammps/lmp").to(
                    "lammps/dump", lmp_traj_name
                )
        ret_dict = {
            "log": work_dir / lmp_log_name,
            "traj": work_dir / lmp_traj_name,
        }
        for key, fname in (("md_failed", "md_failed.extxyz"), ("md_diag", "md_failed.json")):
            fpath = work_dir / fname
            if fpath.exists():
                ret_dict[key] = fpath
        return OPIO(ret_dict)

    def _check_and_handle(
        self,
        ret: int,
        work_dir: Path,
        max_temp: float,
        type_map: Optional[List[str]],
    ) -> None:
        """Inspect the LAMMPS log; apply controlled-early-stop semantics."""
        log_text = (work_dir / lmp_log_name).read_text(errors="replace")
        rows, nan_detected = _parse_thermo(log_text)
        lost_atoms = "Lost atoms" in log_text
        error_in_log = "ERROR" in log_text
        temps = [r["Temp"] for r in rows if "Temp" in r]
        peak_temp = max(temps) if temps else None
        completed_steps = int(rows[-1]["Step"]) if rows else 0
        requested = _requested_nsteps((work_dir / lmp_input_name).read_text())

        unstable_reason = None
        if nan_detected:
            unstable_reason = "NaN or Inf detected in thermo output"
        elif lost_atoms:
            unstable_reason = "LAMMPS lost atoms (structure blown apart)"
        elif peak_temp is not None and peak_temp > max_temp:
            unstable_reason = (
                f"temperature {peak_temp:.1f} K exceeds limit {max_temp} K"
            )

        if unstable_reason is not None:
            logging.warning(
                "LAMMPS exploration stopped early (returning partial "
                "trajectory and diagnostics): %s",
                unstable_reason,
            )
            self._dump_diagnostics(
                work_dir=work_dir,
                reason=unstable_reason,
                rows=rows,
                completed_steps=completed_steps,
                requested=requested,
                type_map=type_map,
            )
            return

        if ret != 0 or error_in_log:
            if error_in_log:
                # LAMMPS prints ERROR for deterministic input/config errors
                raise FatalError(
                    "LAMMPS reported ERROR (deterministic input/config problem); "
                    "check lmp_stdout.log"
                )
            raise TransientError(
                f"LAMMPS exited abnormally (ret={ret}) without instability "
                "signature; check lmp_stdout.log"
            )
        logging.info("LAMMPS exploration finished: %d steps", completed_steps)

    def _dump_diagnostics(
        self,
        work_dir: Path,
        reason: str,
        rows: List[dict],
        completed_steps: int,
        requested: Optional[int],
        type_map: Optional[List[str]],
    ) -> None:
        """Save the last trajectory frame and a JSON diagnostic."""
        import dpdata

        traj_path = work_dir / lmp_traj_name
        n_frames = _count_dump_frames(traj_path)
        # md_failed.extxyz: last frame of the partial trajectory; fall back
        # to the input structure if the dump is empty
        try:
            if n_frames > 0:
                sys = dpdata.System(str(traj_path), fmt="lammps/dump", type_map=type_map)
                last = sys[-1]
            else:
                last = dpdata.System(str(work_dir / lmp_conf_name), fmt="lammps/lmp")
            last.to("ase", str(work_dir / "md_failed.extxyz"))
        except Exception as e:
            logging.warning("failed to save md_failed.extxyz: %s", e)

        last_row = rows[-1] if rows else {}
        volume = last_row.get("Volume") or last_row.get("Vol")
        volumes = [r.get("Volume") or r.get("Vol") for r in rows]
        volumes = [v for v in volumes if v is not None]
        diag = {
            "reason": reason,
            "step": completed_steps,
            "temperature_K": last_row.get("Temp"),
            "energy_eV": last_row.get("PotEng") or last_row.get("TotEng"),
            "fmax_eV_per_A": None,
            "volume_A3": volume,
            "initial_volume_A3": volumes[0] if volumes else None,
            "volume_ratio": (volume / volumes[0]) if volumes and volume else None,
            "requested_nsteps": requested,
            "completed_steps": completed_steps,
            "early_stopped": True,
        }
        with open(work_dir / "md_failed.json", "w") as f:
            json.dump(diag, f, indent=2)

    @staticmethod
    def lmp_args():
        doc_command = (
            "The LAMMPS command prefix, e.g. 'lmp -k on g 1 -sf kk'. "
            "'-in', '-log' and '-screen' are added automatically if absent."
        )
        doc_max_temp = (
            "Thermo temperature ceiling in K; the task is stopped early "
            "(controlled early stop) when exceeded."
        )
        doc_type_map = (
            "Element order of the LAMMPS data file (same as the template "
            "pair_coeff order); used to read back the trajectory."
        )
        return [
            Argument("command", str, optional=True, default="lmp", doc=doc_command),
            Argument(
                "max_temp",
                [float, int],
                optional=True,
                default=5000.0,
                doc=doc_max_temp,
            ),
            Argument(
                "type_map",
                [list, type(None)],
                optional=True,
                default=None,
                doc=doc_type_map,
            ),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = RunLmp.lmp_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=False)
        return data
