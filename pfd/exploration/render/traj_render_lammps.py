from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
)
import dpdata
from ase import Atoms

from .traj_render import (
    TrajRender,
)

if TYPE_CHECKING:
    from pfd.exploration.selector import (
        ConfFilters,
    )


@TrajRender.register("lmp")
@TrajRender.register("lammps")
class TrajRenderLammps(TrajRender):
    def __init__(
        self,
        nopbc: bool = False,
        use_ele_temp: int = 0,
        type_map: Optional[List[str]] = None,
    ):
        self.nopbc = nopbc
        self.use_ele_temp = use_ele_temp
        # element order of the LAMMPS data/dump files; keyword-only source
        # so positional callers passing conf_filters cannot clobber it
        self.type_map = type_map

    def get_confs(
        self,
        trajs: List[Path],
        conf_filters: Optional["ConfFilters"] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> List[Atoms]:
        """Read LAMMPS dump trajectories into a list of ASE Atoms.

        Mirrors TrajRenderASE.get_confs: the downstream selector consumes
        lists of ASE Atoms, so dpdata systems are converted frame by frame.
        The element order (``type_map``) comes from the constructor.
        """
        from pfd.utils.ase2xyz import dpdata2ase

        ntraj = len(trajs)
        if optional_outputs:
            assert ntraj == len(optional_outputs)

        atoms_list = []
        for ii in range(ntraj):
            ss = dpdata.System(trajs[ii], fmt="lammps/dump", type_map=self.type_map)
            ss.nopbc = self.nopbc
            atoms_list.extend(dpdata2ase(ss))
        if conf_filters is not None:
            atoms_list = conf_filters.check(atoms_list)
        return atoms_list
