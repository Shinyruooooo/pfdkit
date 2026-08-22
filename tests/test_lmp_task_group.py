import unittest
from io import StringIO

from ase.build import bulk
from ase.io import read, write

from pfd.exploration.task.lmp_template_task_group import (
    DUMP_ID,
    MODEL_PLACEHOLDER,
    LmpTemplateTaskGroup,
    parse_pair_coeff_elements,
    revise_variable_lines,
)

TEMPLATE = f"""
# user-written template
units           metal
dimension       3
boundary        p p p
atom_style      atomic

read_data       structure.lmp

pair_style      deepmd {MODEL_PLACEHOLDER}
pair_coeff      * * Ge Te

variable        TSTART equal 300.0
variable        TSTOP  equal 2000.0
variable        SEED   equal 12345

velocity        all create ${{TSTART}} ${{SEED}} mom yes rot yes dist gaussian
fix             melt all npt temp ${{TSTART}} ${{TSTOP}} 0.1 iso 1.0 1.0 1.0

dump            {DUMP_ID} all custom 1000 mytraj.lammpstrj id type element x y z fx fy fz
dump_modify     {DUMP_ID} element Ge Te
dump_modify     {DUMP_ID} sort id

run             100000
"""


def _conf_str(atoms=None):
    if atoms is None:
        atoms = bulk("GeTe", "rocksalt", a=6.0, cubic=True) * (2, 2, 2)
    buf = StringIO()
    write(buf, atoms, format="extxyz")
    return buf.getvalue()


def _make_group(revisions=None, template=TEMPLATE, type_map=None):
    tg = LmpTemplateTaskGroup()
    tg.set_conf(conf_list=[_conf_str()], n_sample=1)
    tg.set_lmp(
        template, revisions=revisions or {}, trj_freq=50, type_map=type_map
    )
    return tg


class TestTemplateValidation(unittest.TestCase):
    def _expect_error(self, template, part_of_msg):
        tg = LmpTemplateTaskGroup()
        tg.set_conf(conf_list=[_conf_str()], n_sample=1)
        with self.assertRaises(ValueError) as cm:
            tg.set_lmp(template)
        self.assertIn(part_of_msg, str(cm.exception))

    def test_missing_pair_style(self):
        self._expect_error(
            TEMPLATE.replace("pair_style      deepmd", "# pair_style deepmd"),
            "pair_style",
        )

    def test_missing_model_placeholder(self):
        self._expect_error(
            TEMPLATE.replace(MODEL_PLACEHOLDER, "my_model.pt2"), "placeholder"
        )

    def test_missing_read_data(self):
        self._expect_error(
            TEMPLATE.replace("read_data", "# read_data"), "read_data"
        )

    def test_dump_without_forces(self):
        self._expect_error(
            TEMPLATE.replace("fx fy fz", "vx vy vz"), "fx fy fz"
        )

    def test_missing_dump(self):
        self._expect_error(
            TEMPLATE.replace(f"dump            {DUMP_ID}", "dump            other"),
            "dump",
        )

    def test_pair_coeff_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            _make_group(type_map=["Te", "Ge"])
        self.assertIn("pair_coeff", str(cm.exception))

    def test_parse_pair_coeff(self):
        elements = parse_pair_coeff_elements(TEMPLATE.split("\n"))
        self.assertEqual(elements, ["Ge", "Te"])


class TestRevisions(unittest.TestCase):
    def test_revise_variables(self):
        lines = TEMPLATE.split("\n")
        out = revise_variable_lines(lines, {"TSTOP": 1500, "SEED": 999})
        self.assertIn("variable        TSTOP  equal 1500", out)
        self.assertIn("variable        SEED   equal 999", out)
        # untouched variable
        self.assertIn("variable        TSTART equal 300.0", out)

    def test_revision_applied_in_group(self):
        tg = _make_group(revisions={"TSTOP": 1500})
        self.assertIn("variable        TSTOP  equal 1500", tg.lmp_template)


class TestMakeTask(unittest.TestCase):
    def test_task_files(self):
        tg = _make_group()
        tg.make_task()
        self.assertEqual(len(tg), 1)
        files = tg[0].files()
        self.assertIn("in.lammps", files)
        self.assertIn("conf.lmp", files)
        # model placeholder kept; RunLmp provides the file at run time
        self.assertIn(MODEL_PLACEHOLDER, files["in.lammps"])
        # read_data rewritten to conf.lmp
        self.assertIn("read_data conf.lmp", files["in.lammps"])
        # dump rewritten to canonical name/freq
        self.assertIn("dump pfd_traj all custom 50 traj.dump", files["in.lammps"])
        # dump_modify id untouched
        self.assertIn("dump_modify     pfd_traj element Ge Te", files["in.lammps"])

    def test_conf_lmp_is_valid_lammps_data(self):
        tg = _make_group()
        tg.make_task()
        conf = tg[0].files()["conf.lmp"]
        atoms = read(StringIO(conf), format="lammps-data", style="atomic", units="metal")
        self.assertEqual(len(atoms), 64)
        self.assertEqual(sorted(set(atoms.get_chemical_symbols())), ["Ge", "Te"])

    def test_triclinic_conf(self):
        atoms = bulk("GeTe", "rocksalt", a=6.0)  # non-cubic cell
        tg = LmpTemplateTaskGroup()
        tg.set_conf(conf_list=[_conf_str(atoms)], n_sample=1)
        tg.set_lmp(TEMPLATE, trj_freq=50)
        tg.make_task()
        conf = tg[0].files()["conf.lmp"]
        self.assertIn("xy xz yz", conf)

    def test_type_map_order_preserved(self):
        tg = _make_group(type_map=["Ge", "Te"])
        tg.make_task()
        conf = tg[0].files()["conf.lmp"]
        atoms = read(StringIO(conf), format="lammps-data", style="atomic", units="metal")
        self.assertEqual(len(atoms), 64)

    def test_atom_modify_map_auto_inserted(self):
        """graph-lowered .pt2 models need atom_modify map yes; it is added
        automatically when missing."""
        self.assertNotIn("atom_modify", TEMPLATE)
        tg = _make_group()
        self.assertIn("atom_modify     map yes", tg.lmp_template)

    def test_atom_modify_map_not_duplicated(self):
        tg = _make_group(
            template=TEMPLATE.replace(
                "atom_style      atomic",
                "atom_style      atomic\natom_modify     map yes",
            )
        )
        self.assertEqual(tg.lmp_template.count("atom_modify"), 1)


class TestNormalizeConfig(unittest.TestCase):
    def test_defaults(self):
        data = LmpTemplateTaskGroup.normalize_config(
            {"input_lmp_template": TEMPLATE}, strict=False
        )
        self.assertEqual(data["trj_freq"], 50)
        self.assertEqual(data["revisions"], {})
        self.assertIsNone(data["type_map"])

    def test_explicit_null_type_map(self):
        data = LmpTemplateTaskGroup.normalize_config(
            {"input_lmp_template": TEMPLATE, "type_map": None}, strict=False
        )
        self.assertIsNone(data["type_map"])


if __name__ == "__main__":
    unittest.main()
