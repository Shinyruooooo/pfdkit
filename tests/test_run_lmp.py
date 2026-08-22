import json
import shutil
import tempfile
import unittest
from pathlib import Path

from dflow.python import OPIO, FatalError, TransientError

from pfd.op.run_lmp import RunLmp, _parse_thermo, _requested_nsteps
from pfd.utils import set_directory

MOCK_LMP = r"""#!/bin/bash
# mock lammps: writes log.lammps + traj.dump according to $MOCK_SCENARIO
LOG=log.lammps
while [ $# -gt 0 ]; do
  case "$1" in
    -log) LOG="$2"; shift 2;;
    *) shift;;
  esac
done
case "$MOCK_SCENARIO" in
  success)
    cat > "$LOG" <<EOF
Step Time Temp PotEng TotEng Volume
0 0 300 -900 -900 5857
500 0.5 310 -899 -898 5857
1000 1.0 305 -899 -898 5857
Loop time of 12.3 on 1 procs
EOF
    cp traj.dump.template traj.dump
    exit 0
    ;;
  temp_explosion)
    cat > "$LOG" <<EOF
Step Time Temp PotEng TotEng Volume
0 0 300 -900 -900 5857
500 0.5 5000 -890 -880 5857
600 0.6 8300 -100 3000 5857
EOF
    cp traj.dump.template traj.dump
    exit 1
    ;;
  nan_thermo)
    cat > "$LOG" <<EOF
Step Time Temp PotEng TotEng Volume
0 0 300 -900 -900 5857
500 0.5 nan nan nan 5857
EOF
    cp traj.dump.template traj.dump
    exit 1
    ;;
  crash_no_signature)
    echo "some random failure" > "$LOG"
    exit 1
    ;;
  error_in_log)
    cat > "$LOG" <<EOF
Step Time Temp PotEng TotEng Volume
0 0 300 -900 -900 5857
ERROR: Illegal pair_coeff command
EOF
    exit 1
    ;;
  no_log)
    exit 1
    ;;
esac
"""

IN_LMP = """units metal
atom_style atomic
read_data conf.lmp
pair_style deepmd pfd_model.pt2
pair_coeff * * Ge Te
variable NSTEPS equal 1000
dump pfd_traj all custom 50 traj.dump id type element x y z fx fy fz
run ${NSTEPS}
"""

TRAJ_TEMPLATE = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z fx fy fz
1 1 0.0 0.0 0.0 0.1 0.0 0.0
2 2 2.5 2.5 2.5 -0.1 0.0 0.0
ITEM: TIMESTEP
500
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z fx fy fz
1 1 0.1 0.0 0.0 0.2 0.0 0.0
2 2 2.6 2.5 2.5 -0.2 0.0 0.0
"""


class TestRunLmp(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)
        self.mock = self.root / "mock_lmp.sh"
        self.mock.write_text(MOCK_LMP)
        self.mock.chmod(0o755)
        self.task_path = self.root / "prep"
        self.task_path.mkdir()
        (self.task_path / "in.lammps").write_text(IN_LMP)
        (self.task_path / "conf.lmp").write_text("# fake conf")
        (self.task_path / "traj.dump.template").write_text(TRAJ_TEMPLATE)
        self.model = self.root / "model.pt2"
        self.model.write_text("fake model")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _run(self, scenario):
        ip = OPIO(
            {
                "config": {"command": f"bash {self.mock}", "type_map": ["Ge", "Te"]},
                "task_name": "task.000000",
                "task_path": self.task_path,
                "models": [self.model],
            }
        )
        import os

        env = dict(os.environ)
        with set_directory(self.root):
            os.environ["MOCK_SCENARIO"] = scenario
            try:
                return RunLmp().execute(ip)
            finally:
                os.environ.pop("MOCK_SCENARIO", None)

    def test_success(self):
        out = self._run("success")
        self.assertTrue((self.root / "task.000000" / "log.lammps").is_file())
        self.assertTrue((self.root / "task.000000" / "traj.dump").is_file())
        self.assertIn("traj", out.keys())
        # optional outputs are padded with None by exec_sign_check
        self.assertIsNone(out["md_failed"])
        self.assertIsNone(out["md_diag"])

    def test_temp_explosion_controlled_stop(self):
        out = self._run("temp_explosion")
        task = self.root / "task.000000"
        self.assertIn("md_failed", out.keys())
        self.assertIn("md_diag", out.keys())
        diag = json.loads((task / "md_failed.json").read_text())
        self.assertIn("temperature", diag["reason"])
        self.assertEqual(diag["completed_steps"], 600)
        self.assertEqual(diag["requested_nsteps"], 1000)
        self.assertTrue(diag["early_stopped"])
        # partial trajectory still returned
        self.assertTrue((task / "traj.dump").is_file())

    def test_nan_controlled_stop(self):
        out = self._run("nan_thermo")
        diag = json.loads((self.root / "task.000000" / "md_failed.json").read_text())
        self.assertIn("NaN", diag["reason"])

    def test_crash_without_signature_is_transient(self):
        with self.assertRaises(TransientError):
            self._run("crash_no_signature")

    def test_error_in_log_is_fatal(self):
        with self.assertRaises(FatalError):
            self._run("error_in_log")

    def test_missing_log_is_fatal(self):
        with self.assertRaises(FatalError):
            self._run("no_log")


class TestLogParsing(unittest.TestCase):
    def test_parse_thermo(self):
        log = "Step Time Temp PotEng\n0 0 300 -900\n500 0.5 310 -899\nLoop time 1.0\n"
        rows, nan = _parse_thermo(log)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["Temp"], 310)
        self.assertFalse(nan)

    def test_requested_nsteps_variable_substitution(self):
        self.assertEqual(_requested_nsteps(IN_LMP), 1000)
        ramp = "variable A equal 500\nrun $A\nrun 2000\n"
        self.assertEqual(_requested_nsteps(ramp), 2500)


if __name__ == "__main__":
    unittest.main()
