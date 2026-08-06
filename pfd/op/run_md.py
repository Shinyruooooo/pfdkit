import glob
import json
import logging
import os
import random
import re
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

import ase
import numpy as np
from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    TransientError,
    FatalError,
)

from pfd.constants import (
    ase_conf_name,
    ase_input_name,
    ase_log_name,
    ase_traj_name
)

from pfd.fp.run_fp import (
    _link_input,
)

from pfd.exploration import md
from pfd.exploration.md import (
    MDRunner,
    CalculatorWrapper,
    MDStabilityError,
)

from pfd.utils import (
    BinaryFileInput,
    set_directory,
)



class RunASE(OP):
    r"""Execute a ASE MD task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The LAMMPS
    command is exectuted from directory `task_name`. The trajectory
    and the model deviation will be stored in files `op["traj"]` and
    `op["model_devi"]`, respectively.

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
        return OPIOSign(
            {
                "log": Artifact(Path),
                "traj": Artifact(Path),
                "optional_output": Artifact(Path, optional=True),
                "md_failed": Artifact(Path, optional=True),
                "md_diag": Artifact(Path, optional=True),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:

            - `config`: (`dict`) The config of lmp task. Check `RunLmp.lmp_args` for definitions.
            - `task_name`: (`str`) The name of the task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepLmp`.
            - `models`: (`Artifact(List[Path])`) The frozen model to estimate the model deviation. The first model with be used to drive molecular dynamics simulation.

        Returns
        -------
        Any
            Output dict with components:
            - `log`: (`Artifact(Path)`) The log file of LAMMPS.
            - `traj`: (`Artifact(Path)`) The output trajectory.
            - `model_devi`: (`Artifact(Path)`) The model deviation. The order of recorded model deviations should be consistent with the order of frames in `traj`.

        Raises
        ------
        FatalError
            On deterministic failures: invalid configuration/parameters
            (ValueError etc.). Not retried.
        TransientError
            On other failures that may recover after a retry.

        Notes
        -----
        ``MDStabilityError`` (the stability monitor stopped the trajectory)
        is NOT an error here: the exploration entered an unreliable region
        of the model, so the task returns normally with the partial
        trajectory, the log and the failure diagnostics (controlled early
        stop). The remote script exits 0, so neither dpdispatcher nor
        dflow/argo will retry the task.
        """
        config = ip["config"] if ip["config"] is not None else {}
        ## what the config should be like?
        # {"calculator": "mace"}
        config = RunASE.normalize_config(config)
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        models = ip["models"]
        input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        model_files = [Path(ii).resolve() for ii in models]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            # remove stale stability diagnostics from a previous attempt so
            # they are not mistaken for this run's output
            for fname in ("md_failed.extxyz", "md_failed.json"):
                stale = Path(fname)
                if stale.is_file():
                    stale.unlink()
            # link input files
            for ii in input_files:
                iname = ii.name
                _link_input(iname, ii)
            # instantiate calculator
            calc_style = config.pop("calculator", "mace")
            calc = CalculatorWrapper.get_calculator(calc_style)
            calc = calc().create(model_path=str(model_files[0]), **config)

            # instantiate MDRunner
            md_runner = MDRunner.from_file(
                filename=ase_conf_name
            )
            md_runner.set_calculator(calc)
            try:
                md_runner.run_md_from_json(
                    json_file=ase_input_name,
                )
            except MDStabilityError as e:
                # controlled early stop: the trajectory entered an unreliable
                # region of the model. Return the partial trajectory and the
                # diagnostics as normal outputs; do not fail the task.
                logging.warning(
                    "ASE MD exploration stopped early by stability monitor "
                    "(returning partial trajectory and diagnostics): %s", e,
                )
            except (ValueError, TypeError, KeyError) as e:
                # configuration/parameter errors are deterministic
                raise FatalError(
                    f"ASE MD failed due to deterministic error: {e}"
                ) from e
            except Exception as e:
                raise TransientError(f"ASE MD/relax failed: {e}")
        ret_dict = {
            "log": work_dir / ase_log_name,
            "traj": work_dir / ase_traj_name
        }
        # return stability diagnostics when the monitor stopped the MD early
        for key, fname in (("md_failed", "md_failed.extxyz"), ("md_diag", "md_failed.json")):
            fpath = work_dir / fname
            if fpath.exists():
                ret_dict[key] = fpath
        return OPIO(ret_dict)

    @staticmethod
    def ase_args():
        doc_calc_type = "The type of calculator to use, e.g., 'mace', 'mattersim'."
        return [
            Argument("calculator", str,  default="mattersim", doc=doc_calc_type, alias=['calc']),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = RunASE.ase_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=False)
        return data