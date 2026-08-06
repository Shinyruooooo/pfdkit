import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write
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
    """MDStabilityError -> controlled early stop; config errors -> FatalError."""

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

    def test_instability_returns_partial_trajectory_and_diagnostics(self):
        """A blown-up MD (T > 5000 K) -> controlled early stop: normal return,
        partial trajectory + diagnostics, no FatalError/TransientError."""
        params = MDParameters(
            temp=6000.0, dt=1.0, nsteps=50, ensemble="nvt", traj_freq=1,
            log_freq=1, seed=12345,
        )
        out = self._execute(self._make_ip(params))  # must not raise
        task_dir = self.test_dir / "task.000000"
        # partial trajectory exists and is readable, shorter than requested
        traj_path = task_dir / "traj.traj"
        self.assertEqual(out["traj"], Path("task.000000") / "traj.traj")
        frames = read(traj_path, index=":")
        self.assertGreaterEqual(len(frames), 1)
        self.assertLess(len(frames), 50)
        for atoms in frames:
            self.assertEqual(len(atoms), 32)
            self.assertTrue(atoms.cell.shape == (3, 3))
        # diagnostics returned as outputs
        self.assertTrue((task_dir / "md_failed.extxyz").exists())
        self.assertTrue((task_dir / "md_failed.json").exists())
        self.assertEqual(out["md_failed"], Path("task.000000") / "md_failed.extxyz")
        self.assertEqual(out["md_diag"], Path("task.000000") / "md_failed.json")
        with open(task_dir / "md_failed.json") as f:
            diag = json.load(f)
        for key in ("reason", "step", "temperature_K", "energy_eV",
                    "fmax_eV_per_A", "volume_A3", "initial_volume_A3",
                    "volume_ratio", "requested_nsteps", "completed_steps",
                    "early_stopped"):
            self.assertIn(key, diag)
        self.assertIn("temperature", diag["reason"])
        self.assertEqual(diag["requested_nsteps"], 50)
        self.assertEqual(diag["completed_steps"], diag["step"])
        self.assertTrue(diag["early_stopped"])
        # downstream renderer handles the partial trajectory as-is
        from pfd.exploration.render.traj_render_ase import TrajRenderASE
        confs = TrajRenderASE().get_confs([traj_path])
        self.assertEqual(len(confs), len(frames))

    def test_step0_stop_still_returns_readable_traj(self):
        """Monitor triggers at the initial state: traj still has frame 0."""
        # one atom pushed toward its nearest neighbor -> huge initial force
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        pos = atoms.get_positions()
        pos[0] += [1.4 / np.sqrt(2), 1.4 / np.sqrt(2), 0.0]
        atoms.set_positions(pos)
        write(self.task_path / ase_conf_name, atoms, format="extxyz")
        params = MDParameters(
            temp=300.0, dt=1.0, nsteps=10, ensemble="nvt", traj_freq=1,
            log_freq=1, seed=12345,
        )
        out = self._execute(self._make_ip(params))  # must not raise
        task_dir = self.test_dir / "task.000000"
        frames = read(task_dir / "traj.traj", index=":")
        self.assertGreaterEqual(len(frames), 1)
        self.assertTrue((task_dir / "md_failed.json").exists())
        self.assertEqual(out["md_diag"], Path("task.000000") / "md_failed.json")

    def test_stale_diagnostics_removed_before_run(self):
        """Stale md_failed files from a previous attempt are cleaned up."""
        task_dir = self.test_dir / "task.000000"
        task_dir.mkdir()
        (task_dir / "md_failed.extxyz").write_text("stale")
        (task_dir / "md_failed.json").write_text("stale")
        params = MDParameters(
            temp=300.0, dt=1.0, nsteps=3, ensemble="nvt", traj_freq=1,
            log_freq=1, seed=12345,
        )
        out = self._execute(self._make_ip(params))
        self.assertIsNone(out.get("md_failed"))
        self.assertIsNone(out.get("md_diag"))
        self.assertFalse((task_dir / "md_failed.extxyz").exists())
        self.assertFalse((task_dir / "md_failed.json").exists())

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
