import shutil
import tempfile
import unittest
from pathlib import Path

from ase import Atoms

from pfd.exploration.render.traj_render_lammps import TrajRenderLammps

DUMP = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0.0 12.0
0.0 12.0
0.0 12.0
ITEM: ATOMS id type element x y z fx fy fz
1 1 Ge 0.0 0.0 0.0 0.1 0.0 0.0
2 1 Ge 3.0 3.0 3.0 -0.1 0.0 0.0
3 2 Te 1.5 1.5 1.5 0.0 0.1 0.0
4 2 Te 4.5 4.5 4.5 0.0 -0.1 0.0
ITEM: TIMESTEP
50
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0.0 12.0
0.0 12.0
0.0 12.0
ITEM: ATOMS id type element x y z fx fy fz
1 1 Ge 0.1 0.0 0.0 0.2 0.0 0.0
2 1 Ge 3.1 3.0 3.0 -0.2 0.0 0.0
3 2 Te 1.6 1.5 1.5 0.0 0.2 0.0
4 2 Te 4.6 4.5 4.5 0.0 -0.2 0.0
"""


class TestTrajRenderLammps(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.traj = Path(self.tmpdir) / "traj.dump"
        self.traj.write_text(DUMP)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_read_dump_to_atoms(self):
        render = TrajRenderLammps(type_map=["Ge", "Te"])
        atoms_list = render.get_confs([self.traj])
        self.assertEqual(len(atoms_list), 2)
        self.assertIsInstance(atoms_list[0], Atoms)
        self.assertEqual(sorted(set(atoms_list[0].get_chemical_symbols())), ["Ge", "Te"])
        self.assertEqual(len(atoms_list[0]), 4)

    def test_keyword_call_contract(self):
        """The selector calls get_confs with keyword args; a ConfFilters-like
        object must never land in a type_map position."""
        render = TrajRenderLammps(type_map=["Ge", "Te"])

        class FakeFilters:
            def check(self, atoms_list):
                return atoms_list[:1]

        atoms_list = render.get_confs(
            [self.traj], conf_filters=FakeFilters(), optional_outputs=None
        )
        self.assertEqual(len(atoms_list), 1)


if __name__ == "__main__":
    unittest.main()
