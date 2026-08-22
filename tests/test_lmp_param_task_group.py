import unittest
from io import StringIO

from ase.build import bulk
from ase.io import read, write

from pfd.exploration.task.lmp_param_task_group import (
    LmpParamTaskGroup,
    make_lmp_input_param,
)


def _conf_str(atoms=None):
    if atoms is None:
        atoms = bulk("GeTe", "rocksalt", a=6.0, cubic=True) * (2, 2, 2)
    buf = StringIO()
    write(buf, atoms, format="extxyz")
    return buf.getvalue()


class TestMakeLmpInputParam(unittest.TestCase):
    def test_npt_generates_valid_script(self):
        s = make_lmp_input_param(
            conf_file="conf.lmp", ens="npt", nsteps=50000, dt=1.0,
            trj_freq=100, temp=1200.0, tau_t=500.0, press=10000.0,
            tau_p=2000.0, seed=12345, type_map=["Ge", "Te"],
        )
        self.assertIn("pair_style      deepmd pfd_model.pt2", s)
        self.assertIn("pair_coeff      * * Ge Te", s)
        self.assertIn("read_data       conf.lmp", s)
        self.assertIn("atom_modify     map yes", s)
        # tau_t 500 fs -> 0.5 ps; press 10000 bar; tau_p 2000 fs -> 2 ps
        self.assertIn("fix             pfd_npt all npt temp 1200.000000 1200.000000 0.500000 iso 10000.000000 10000.000000 2.000000", s)
        # dt 1 fs -> 0.001 ps
        self.assertIn("timestep        0.001000", s)
        # seed used
        self.assertIn("velocity        all create 1200.000000 12345", s)
        # dump with forces + canonical name
        self.assertIn("dump            pfd_traj all custom 100 traj.dump id type element x y z fx fy fz", s)
        # masses from ase (Ge 72.630, Te 127.6)
        self.assertIn("mass            1 72.630000", s)
        self.assertIn("mass            2 127.600000", s)

    def test_nvt(self):
        s = make_lmp_input_param(
            conf_file="conf.lmp", ens="nvt", nsteps=1000, dt=0.5,
            trj_freq=10, temp=300.0, tau_t=100.0, type_map=["Ge", "Te"],
        )
        self.assertIn("fix             pfd_nvt all nvt temp 300.000000 300.000000 0.100000", s)
        self.assertIn("timestep        0.000500", s)

    def test_nve(self):
        s = make_lmp_input_param(
            conf_file="conf.lmp", ens="nve", nsteps=1000, dt=1.0,
            trj_freq=10, temp=1000.0, tau_t=100.0, type_map=["Ge", "Te"],
        )
        self.assertIn("fix             pfd_nve all nve", s)

    def test_npt_requires_press(self):
        with self.assertRaises(ValueError):
            make_lmp_input_param(
                conf_file="conf.lmp", ens="npt", nsteps=100, dt=1.0,
                trj_freq=10, temp=300.0, tau_t=100.0, press=None,
                type_map=["Ge", "Te"],
            )

    def test_unknown_ensemble(self):
        with self.assertRaises(ValueError):
            make_lmp_input_param(
                conf_file="conf.lmp", ens="nph", nsteps=100, dt=1.0,
                trj_freq=10, temp=300.0, tau_t=100.0, press=1.0,
                type_map=["Ge", "Te"],
            )

    def test_missing_type_map(self):
        with self.assertRaises(ValueError):
            make_lmp_input_param(
                conf_file="conf.lmp", ens="nvt", nsteps=100, dt=1.0,
                trj_freq=10, temp=300.0, tau_t=100.0, type_map=None,
            )


class TestLmpParamTaskGroup(unittest.TestCase):
    def _make(self, **kw):
        tg = LmpParamTaskGroup()
        tg.set_conf(conf_list=[_conf_str()], n_sample=1)
        defaults = dict(temps=[1000], ens="nvt", nsteps=100, trj_freq=10, type_map=["Ge", "Te"])
        defaults.update(kw)
        tg.set_md(**defaults)
        return tg

    def test_task_files_contract(self):
        tg = self._make()
        tg.make_task()
        self.assertEqual(len(tg), 1)
        files = tg[0].files()
        self.assertIn("in.lammps", files)
        self.assertIn("conf.lmp", files)
        script = files["in.lammps"]
        self.assertIn("pfd_model.pt2", script)
        self.assertIn("read_data       conf.lmp", script)
        self.assertIn("pfd_traj", script)
        # conf is valid lammps-data
        atoms = read(StringIO(files["conf.lmp"]), format="lammps-data", style="atomic", units="metal")
        self.assertEqual(len(atoms), 64)

    def test_temp_press_product(self):
        tg = self._make(temps=[300, 1200], press=[1, 10000], ens="npt")
        tg.make_task()
        # 1 conf x 2 temps x 2 press = 4 tasks
        self.assertEqual(len(tg), 4)

    def test_seed_reproducibility(self):
        tg1 = self._make(seed=12345)
        tg1.make_task()
        tg2 = self._make(seed=12345)
        tg2.make_task()
        self.assertEqual(tg1[0].files()["in.lammps"], tg2[0].files()["in.lammps"])

    def test_different_seed_differs(self):
        tg1 = self._make(seed=12345)
        tg1.make_task()
        tg2 = self._make(seed=999)
        tg2.make_task()
        self.assertNotEqual(tg1[0].files()["in.lammps"], tg2[0].files()["in.lammps"])

    def test_normalize_config_defaults(self):
        data = LmpParamTaskGroup.normalize_config({"temps": [1000], "ens": "nvt"}, strict=False)
        self.assertEqual(data["dt"], 1.0)
        self.assertIsNone(data["seed"])
        self.assertIsNone(data["press"])

    def test_generated_script_passes_template_validation(self):
        """The generated script must satisfy the same contract the RunLmp
        path expects (pair_style placeholder, read_data, dump with forces)."""
        from pfd.exploration.task.lmp_template_task_group import LmpTemplateTaskGroup
        tg = self._make()
        tg.make_task()
        script = tg[0].files()["in.lammps"]
        # reuse the template validator as the contract check
        checker = LmpTemplateTaskGroup()
        checker._validate_template(script.split("\n"))  # should not raise


if __name__ == "__main__":
    unittest.main()
