import json
import logging
from pathlib import Path
from typing import Dict, List

from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# default reason keywords that mark the model as unreliable; reasons matching
# `ignored_reasons` (default: volume) are reported but do not count against
# convergence. Unknown/corrupted diagnostics are counted conservatively.
DEFAULT_IGNORED_REASONS = ["volume"]


class ExplStabilityOP(OP):
    """Aggregate MD early-stop diagnostics of one exploration iteration.

    Counts how many exploration slices were stopped early by the stability
    monitor and decides whether the exploration is considered dynamically
    stable, based on the `expl_stability` section of the evaluate config:

    {
        "enabled": true,
        "max_failed_slices": 0,
        "max_lost_fraction": 0.0,
        "ignored_reasons": ["volume"]
    }
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                # one traj per slice -> total number of slices
                "trajs": Artifact(List[Path]),
                # one diag per early-stopped slice (optional output of RunASE)
                "md_diags": Artifact(List[Path], optional=True),
                # the full evaluate config dict; the "expl_stability"
                # sub-section is extracted from it. Must be a BigParameter:
                # the flow passes evaluate_config as a dflow big parameter
                # (artifact-backed), like ModelTestOP does.
                "config": BigParameter(dict, default={}),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "report": Parameter(Dict),
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        trajs = ip.get("trajs") or []
        md_diags = ip.get("md_diags") or []
        config = ip.get("config") or {}
        stab_config = config.get("expl_stability") or {}

        total_slices = len(trajs)
        # the check is enabled by the mere presence of the "expl_stability"
        # section (even an empty dict), unless explicitly disabled
        enabled = "expl_stability" in config and stab_config.get("enabled", True)
        max_failed_slices = int(stab_config.get("max_failed_slices", 0))
        # None means the lost-fraction check is not applied
        max_lost_fraction = stab_config.get("max_lost_fraction", None)
        if max_lost_fraction is not None:
            max_lost_fraction = float(max_lost_fraction)
        ignored_reasons = [
            str(k).lower()
            for k in stab_config.get("ignored_reasons", DEFAULT_IGNORED_REASONS)
        ]

        diags = []
        for path in md_diags:
            p = Path(path)
            # dflow materializes clean slices (no optional output) as
            # placeholder entries: empty dirs, empty files, or small JSON
            # markers like {"path_list": [{"dflow_list_item": null, ...}]}
            if p.is_dir():
                inner = p / "md_failed.json"
                if not inner.is_file():
                    continue  # placeholder for a clean slice
                p = inner
            try:
                text = p.read_text()
                if not text.strip():
                    continue  # empty placeholder
                diag = json.loads(text)
            except Exception as e:
                logging.warning("Corrupted diagnostics file %s: %s", path, e)
                diags.append({"reason": "corrupted diagnostics"})
                continue
            if not isinstance(diag, dict) or "reason" not in diag:
                continue  # dflow placeholder, not a real diagnostic
            diags.append(diag)

        counted = []
        ignored = []
        for diag in diags:
            reason = str(diag.get("reason", "unknown")).lower()
            if any(k in reason for k in ignored_reasons):
                ignored.append(diag)
            else:
                counted.append(diag)

        lost_steps = 0.0
        for diag in counted:
            requested = diag.get("requested_nsteps")
            completed = diag.get("completed_steps")
            try:
                if requested and completed is not None and float(requested) > 0:
                    lost_steps += 1.0 - float(completed) / float(requested)
            except (TypeError, ValueError):
                pass
        lost_fraction = lost_steps / total_slices if total_slices > 0 else 0.0

        reasons = {}
        for diag in counted:
            key = str(diag.get("reason", "unknown"))
            reasons[key] = reasons.get(key, 0) + 1

        stable = True
        if enabled:
            stable = len(counted) <= max_failed_slices and (
                max_lost_fraction is None or lost_fraction <= max_lost_fraction
            )

        report = {
            "enabled": enabled,
            "stable": stable,
            "total_slices": total_slices,
            "failed_slices": len(counted),
            "ignored_slices": len(ignored),
            "lost_fraction": lost_fraction,
            "max_failed_slices": max_failed_slices,
            "max_lost_fraction": max_lost_fraction,
            "ignored_reasons": ignored_reasons,
            "reasons": reasons,
        }
        logging.info(
            "Exploration stability: %d/%d slices stopped early (%d ignored), "
            "lost step fraction %.4f, stable=%s (enabled=%s)",
            len(counted),
            total_slices,
            len(ignored),
            lost_fraction,
            stable,
            enabled,
        )
        return OPIO({"report": report})
