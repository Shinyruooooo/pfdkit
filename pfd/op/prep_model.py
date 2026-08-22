import logging
import os
from pathlib import Path
from typing import Optional

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

from pfd.utils import run_command, set_directory

FROZEN_MODEL_NAME = "compressed_model.pt2"


class PrepModelFreeze(OP):
    r"""Freeze and compress a trained model checkpoint for LAMMPS exploration.

    Runs (on the compute resource via the dispatcher executor):

        DP_CUDA_INFER=2 dp --pt-expt freeze -c <ckpt> -o frozen_model --lower-kind graph
        DP_CUDA_INFER=2 dp --pt-expt compress -i frozen_model.pt2 -o compressed_model.pt2

    The checkpoint preference is ``model_ema.ckpt.pt`` (EMA weights, as in
    the reference freeze.slurm) over ``model.ckpt.pt``.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "model": Artifact(Path),  # directory containing checkpoints
                "config": BigParameter(dict),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "frozen_model": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        config = ip["config"] if ip["config"] is not None else {}
        config = self.normalize_config(config)
        use_ema = config["use_ema"]
        model_dir = Path(ip["model"])

        ema = model_dir / "model_ema.ckpt.pt"
        plain = model_dir / "model.ckpt.pt"
        if model_dir.is_file():
            # a bare checkpoint file (e.g. the base model of the first
            # iteration) rather than a training output directory
            ckpt = model_dir
        elif use_ema and ema.is_file():
            ckpt = ema
        elif plain.is_file():
            ckpt = plain
        elif ema.is_file():
            ckpt = ema
        else:
            raise FatalError(
                f"no model checkpoint found in {model_dir} "
                "(looked for model_ema.ckpt.pt and model.ckpt.pt)"
            )

        # pin OMP threads like the reference freeze.slurm does; an unpinned
        # PyTorch otherwise spawns threads per core and the AOTI make_fx
        # tracing gets OOM-killed on multi-core nodes
        omp = os.environ.get("SLURM_CPUS_PER_TASK", "8")
        env_prefix = f"OMP_NUM_THREADS={omp} DP_CUDA_INFER=2"
        ret1, _, err1 = run_command(
            f"{env_prefix} dp --pt-expt freeze -c {ckpt} -o frozen_model --lower-kind graph",
            shell=True,
        )
        if ret1 != 0 or not Path("frozen_model.pt2").is_file():
            raise FatalError(f"dp freeze failed (ret={ret1}): {err1[-500:]}")
        ret2, _, err2 = run_command(
            f"{env_prefix} dp --pt-expt compress -i frozen_model.pt2 -o {FROZEN_MODEL_NAME}",
            shell=True,
        )
        out = Path(FROZEN_MODEL_NAME)
        if ret2 != 0 or not out.is_file():
            raise FatalError(f"dp compress failed (ret={ret2}): {err2[-500:]}")
        logging.info("model frozen and compressed: %s (from %s)", out, ckpt.name)
        return OPIO({"frozen_model": out})

    @staticmethod
    def freeze_args():
        doc_use_ema = (
            "Prefer model_ema.ckpt.pt (EMA weights) over model.ckpt.pt "
            "when both exist."
        )
        return [
            Argument("use_ema", bool, optional=True, default=True, doc=doc_use_ema),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = PrepModelFreeze.freeze_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=False)
        return data
