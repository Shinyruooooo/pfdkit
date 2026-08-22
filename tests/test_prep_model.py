import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dflow.python import OPIO, FatalError

from pfd.op.prep_model import PrepModelFreeze, FROZEN_MODEL_NAME
from pfd.utils import set_directory


class TestPrepModelFreeze(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "model.ckpt.pt").write_text("plain")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _mock_run_command(self, ret_freeze=0, ret_compress=0):
        def fake(cmd, shell=False):
            if " freeze " in cmd:
                if ret_freeze == 0:
                    Path("frozen_model.pt2").write_text("frozen")
                return ret_freeze, "", "freeze failed"
            if " compress " in cmd:
                if ret_compress == 0:
                    Path(FROZEN_MODEL_NAME).write_text("compressed")
                return ret_compress, "", "compress failed"
            return 1, "", "unknown command"

        return fake

    def test_plain_ckpt(self):
        with set_directory(self.root):
            with mock.patch("pfd.op.prep_model.run_command", self._mock_run_command()):
                out = PrepModelFreeze().execute(
                    OPIO({"model": self.model_dir, "config": {}})
                )
        self.assertEqual(Path(out["frozen_model"]).name, FROZEN_MODEL_NAME)
        self.assertTrue((self.root / FROZEN_MODEL_NAME).is_file())

    def test_ema_preferred(self):
        (self.model_dir / "model_ema.ckpt.pt").write_text("ema")
        seen = []

        def fake(cmd, shell=False):
            seen.append(cmd)
            if " freeze " in cmd:
                Path("frozen_model.pt2").write_text("frozen")
            if " compress " in cmd:
                Path(FROZEN_MODEL_NAME).write_text("compressed")
            return 0, "", ""

        with set_directory(self.root):
            with mock.patch("pfd.op.prep_model.run_command", fake):
                PrepModelFreeze().execute(OPIO({"model": self.model_dir, "config": {}}))
        self.assertIn("model_ema.ckpt.pt", seen[0])

    def test_ema_disabled(self):
        (self.model_dir / "model_ema.ckpt.pt").write_text("ema")
        seen = []

        def fake(cmd, shell=False):
            seen.append(cmd)
            if " freeze " in cmd:
                Path("frozen_model.pt2").write_text("frozen")
            if " compress " in cmd:
                Path(FROZEN_MODEL_NAME).write_text("compressed")
            return 0, "", ""

        with set_directory(self.root):
            with mock.patch("pfd.op.prep_model.run_command", fake):
                PrepModelFreeze().execute(
                    OPIO({"model": self.model_dir, "config": {"use_ema": False}})
                )
        self.assertIn("model.ckpt.pt", seen[0])
        self.assertNotIn("model_ema.ckpt.pt", seen[0])

    def test_missing_ckpt_fatal(self):
        empty = self.root / "empty"
        empty.mkdir()
        with set_directory(self.root):
            with self.assertRaises(FatalError):
                PrepModelFreeze().execute(OPIO({"model": empty, "config": {}}))

    def test_freeze_failure_fatal(self):
        with set_directory(self.root):
            with mock.patch(
                "pfd.op.prep_model.run_command", self._mock_run_command(ret_freeze=1)
            ):
                with self.assertRaises(FatalError):
                    PrepModelFreeze().execute(
                        OPIO({"model": self.model_dir, "config": {}})
                    )

    def test_compress_failure_fatal(self):
        with set_directory(self.root):
            with mock.patch(
                "pfd.op.prep_model.run_command", self._mock_run_command(ret_compress=1)
            ):
                with self.assertRaises(FatalError):
                    PrepModelFreeze().execute(
                        OPIO({"model": self.model_dir, "config": {}})
                    )

    def test_bare_ckpt_file(self):
        """A bare .pt file (e.g. the base model of the first iteration) is
        used directly."""
        bare = self.root / "base_model.pt"
        bare.write_text("bare")
        with set_directory(self.root):
            with mock.patch("pfd.op.prep_model.run_command", self._mock_run_command()):
                out = PrepModelFreeze().execute(OPIO({"model": bare, "config": {}}))
        self.assertTrue((self.root / FROZEN_MODEL_NAME).is_file())


if __name__ == "__main__":
    unittest.main()
