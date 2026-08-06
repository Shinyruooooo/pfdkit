import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from dflow.python import OPIO, FatalError, TransientError

from pfd.constants import ase_conf_name, ase_input_name
from pfd.exploration.md import CalculatorWrapper, MDParameters
from pfd.op.run_md import RunASE
from pfd.utils import set_directory


@CalculatorWrapper.register("emt_test")
class EMTTestWrapper(CalculatorWrapper):
    """Lightweight EMT calculator for RunASE tests (no MLIP needed)."""

    def create(self, model_path=None, **kwargs):
        return EMT()


class TestRunASEErrorClassification(unittest.TestCase):
    """MDStabilityError/config errors must be FatalError (no dflow retry)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)
        self.task_path = self.test_dir / "input"
        self.task_path.mkdir()
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        write(self.task_path / ase_conf_name, atoms, format="extxyz")
        self.model = self.test_dir / "model.pth"
        self.model.write_text("dummy")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _make_ip(self, params: MDParameters) -> OPIO:
        (self.task_path / ase_input_name).write_text(params.to_json())
        return OPIO(
            {
                "config": {"calculator": "emt_test"},
                "task_name": "task.000000",
                "task_path": self.task_path,
                "models": [self.model],
            }
        )

    def _execute(self, ip):
        # RunASE works in a relative task dir; run inside the tmp dir
        with set_directory(self.test_dir):
            return RunASE().execute(ip)

    def test_instability_raises_fatal_error_and_dumps_diagnostics(self):
        """A blown-up MD (T > 5000 K) -> FatalError + md_failed files, no retry."""
        params = MDParameters(
            temp=6000.0, dt=1.0, nsteps=5, ensemble="nvt", traj_freq=1,
            log_freq=1, seed=12345,
        )
        with self.assertRaises(FatalError) as ctx:
            self._execute(self._make_ip(params))
        self.assertIn("instability", str(ctx.exception))
        task_dir = self.test_dir / "task.000000"
        self.assertTrue((task_dir / "md_failed.extxyz").exists())
        self.assertTrue((task_dir / "md_failed.json").exists())
        with open(task_dir / "md_failed.json") as f:
            diag = json.load(f)
        for key in ("reason", "step", "temperature_K", "energy_eV",
                    "fmax_eV_per_A", "volume_A3", "initial_volume_A3",
                    "volume_ratio"):
            self.assertIn(key, diag)
        self.assertIn("temperature", diag["reason"])

    def test_config_error_raises_fatal_error(self):
        """NPT without compressibility (ValueError) -> FatalError, not TransientError."""
        params = MDParameters(
            temp=300.0, press=1.0, dt=1.0, nsteps=5, ensemble="npt",
            compressibility=None,
        )
        with self.assertRaises(FatalError) as ctx:
            self._execute(self._make_ip(params))
        self.assertIn("compressibility", str(ctx.exception))

    def test_successful_run_returns_log_and_traj(self):
        """A healthy MD run succeeds; md_failed/md_diag outputs are absent."""
        params = MDParameters(
            temp=300.0, dt=1.0, nsteps=3, ensemble="nvt", traj_freq=1,
            log_freq=1, seed=12345,
        )
        out = self._execute(self._make_ip(params))
        task_dir = self.test_dir / "task.000000"
        self.assertEqual(out["log"], Path("task.000000") / "ase.log")
        self.assertTrue((task_dir / "ase.log").exists())
        self.assertTrue((task_dir / "traj.traj").exists())
        self.assertIsNone(out.get("md_failed"))
        self.assertIsNone(out.get("md_diag"))

    def test_other_errors_remain_transient(self):
        """Non-deterministic errors are still wrapped as TransientError."""
        params = MDParameters(temp=300.0, dt=1.0, nsteps=5, ensemble="nvt")
        ip = self._make_ip(params)
        with mock.patch(
            "pfd.op.run_md.MDRunner.run_md_from_json",
            side_effect=RuntimeError("infrastructure boom"),
        ):
            with self.assertRaises(TransientError):
                self._execute(ip)


if __name__ == "__main__":
    unittest.main()
