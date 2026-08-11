import json
import shutil
import tempfile
import unittest
from pathlib import Path

from dflow.python import OPIO

from pfd.op.expl_stability import ExplStabilityOP


class TestExplStabilityOP(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)
        # one traj file per slice; only existence/count matters
        self.trajs = []
        for ii in range(4):
            p = self.test_dir / f"traj{ii}.traj"
            p.write_text("dummy")
            self.trajs.append(p)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_diag(self, name, reason, completed=100, requested=1000):
        p = self.test_dir / name
        p.write_text(
            json.dumps(
                {
                    "reason": reason,
                    "step": completed,
                    "completed_steps": completed,
                    "requested_nsteps": requested,
                    "early_stopped": True,
                }
            )
        )
        return p

    def _run(self, md_diags=None, config=None):
        ip = OPIO(
            {
                "trajs": self.trajs,
                "md_diags": md_diags,
                "config": config or {},
            }
        )
        return ExplStabilityOP().execute(ip)["report"]

    def test_no_early_stop_is_stable(self):
        report = self._run(md_diags=None, config={"expl_stability": {}})
        self.assertTrue(report["stable"])
        self.assertEqual(report["total_slices"], 4)
        self.assertEqual(report["failed_slices"], 0)
        self.assertEqual(report["lost_fraction"], 0.0)

    def test_temperature_explosion_counts_as_failure(self):
        diag = self._write_diag(
            "d0.json", "temperature 6644.6 K exceeds limit 5000.0 K"
        )
        report = self._run(md_diags=[diag], config={"expl_stability": {}})
        self.assertFalse(report["stable"])
        self.assertEqual(report["failed_slices"], 1)
        self.assertAlmostEqual(report["lost_fraction"], (1 - 100 / 1000) / 4)

    def test_volume_reason_ignored_by_default(self):
        diag = self._write_diag(
            "d0.json", "volume ratio 1.345 outside tolerance 0.2"
        )
        report = self._run(md_diags=[diag], config={"expl_stability": {}})
        self.assertTrue(report["stable"])
        self.assertEqual(report["failed_slices"], 0)
        self.assertEqual(report["ignored_slices"], 1)

    def test_corrupted_diag_counted_conservatively(self):
        bad = self.test_dir / "bad.json"
        bad.write_text("{not valid json")
        report = self._run(md_diags=[bad], config={"expl_stability": {}})
        self.assertFalse(report["stable"])
        self.assertEqual(report["failed_slices"], 1)

    def test_tolerance_max_failed_slices(self):
        diag = self._write_diag(
            "d0.json", "max force 74.8 eV/Ang exceeds limit 50.0 eV/Ang"
        )
        report = self._run(
            md_diags=[diag],
            config={"expl_stability": {"max_failed_slices": 1}},
        )
        self.assertTrue(report["stable"])

    def test_tolerance_max_lost_fraction(self):
        diag = self._write_diag(
            "d0.json",
            "temperature 6644.6 K exceeds limit 5000.0 K",
            completed=990,
            requested=1000,
        )
        # lost = (1 - 0.99) / 4 = 0.0025 <= 0.01 -> stable
        report = self._run(
            md_diags=[diag],
            config={
                "expl_stability": {"max_failed_slices": 1, "max_lost_fraction": 0.01}
            },
        )
        self.assertTrue(report["stable"])
        self.assertAlmostEqual(report["lost_fraction"], 0.0025)

    def test_disabled_by_enabled_false(self):
        diag = self._write_diag(
            "d0.json", "temperature 6644.6 K exceeds limit 5000.0 K"
        )
        report = self._run(
            md_diags=[diag], config={"expl_stability": {"enabled": False}}
        )
        self.assertTrue(report["stable"])
        self.assertFalse(report["enabled"])
        # failures are still reported for visibility
        self.assertEqual(report["failed_slices"], 1)

    def test_no_expl_stability_section_disables(self):
        diag = self._write_diag(
            "d0.json", "temperature 6644.6 K exceeds limit 5000.0 K"
        )
        report = self._run(md_diags=[diag], config={"converge": {}})
        self.assertTrue(report["stable"])
        self.assertFalse(report["enabled"])

    def test_custom_ignored_reasons(self):
        diag = self._write_diag(
            "d0.json", "temperature 6644.6 K exceeds limit 5000.0 K"
        )
        report = self._run(
            md_diags=[diag],
            config={"expl_stability": {"ignored_reasons": ["temperature"]}},
        )
        self.assertTrue(report["stable"])
        self.assertEqual(report["ignored_slices"], 1)

    def test_dflow_placeholders_are_skipped(self):
        """dflow materializes clean slices as null-item placeholders; they
        must not count as corrupted/failed diagnostics."""
        marker = self.test_dir / "marker"
        marker.write_text('{"path_list": [{"dflow_list_item": null, "order": 1}]}')
        empty_dir = self.test_dir / "empty_task"
        empty_dir.mkdir()
        empty_file = self.test_dir / "empty.json"
        empty_file.write_text("")
        diag = self._write_diag(
            "d0.json", "temperature 8128.1 K exceeds limit 5000.0 K"
        )
        report = self._run(
            md_diags=[marker, empty_dir, empty_file, diag],
            config={"expl_stability": {}},
        )
        self.assertEqual(report["failed_slices"], 1)
        self.assertFalse(report["stable"])  # 1 real failure > default 0
        self.assertNotIn("corrupted diagnostics", report["reasons"])

    def test_dir_entry_with_real_diag_is_loaded(self):
        """If dflow passes the task directory, pick up md_failed.json inside."""
        task_dir = self.test_dir / "task.000000"
        task_dir.mkdir()
        (task_dir / "md_failed.json").write_text(
            json.dumps({"reason": "max force 74.8 eV/Ang exceeds limit 50.0 eV/Ang"})
        )
        report = self._run(md_diags=[task_dir], config={"expl_stability": {}})
        self.assertEqual(report["failed_slices"], 1)


class TestExplStabilityArgs(unittest.TestCase):
    """The expl_stability section passes dargs validation of evaluate_args."""

    def _normalize(self, conf):
        from dargs import Argument
        from pfd.entrypoint.args import evaluate_args

        base = Argument("base", dict, evaluate_args())
        data = base.normalize_value(conf, trim_pattern="_*")
        base.check_value(data, strict=False)
        return data

    def test_present_section_gets_defaults(self):
        data = self._normalize(
            {
                "converge": {"type": "energy_rmse"},
                "expl_stability": {"max_failed_slices": 1},
            }
        )
        stab = data["expl_stability"]
        self.assertTrue(stab["enabled"])
        self.assertEqual(stab["max_failed_slices"], 1)
        self.assertIsNone(stab["max_lost_fraction"])
        self.assertEqual(stab["ignored_reasons"], ["volume"])
        self.assertEqual(stab["consecutive_clean_iters"], 1)

    def test_absent_section_defaults_to_none(self):
        data = self._normalize({"converge": {"type": "energy_rmse"}})
        self.assertIsNone(data["expl_stability"])


if __name__ == "__main__":
    unittest.main()
