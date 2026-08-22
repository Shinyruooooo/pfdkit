import logging
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import dargs
import numpy as np
from ase.data import chemical_symbols
from ase.io import read, write
from dargs import Argument

from pfd.constants import lmp_conf_name, lmp_input_name, lmp_traj_name

from .conf_sampling_task_group import ConfSamplingTaskGroup
from .task import ExplorationTask

# placeholder tokens the user's template must use
MODEL_PLACEHOLDER = "pfd_model.pt2"
DUMP_ID = "pfd_traj"


def _find_line(lines: List[str], keywords: List[str]) -> int:
    """Find the index of the only line containing all keywords (as tokens).

    Comments (everything after '#') are stripped before matching.
    """
    found = [
        ii
        for ii, line in enumerate(lines)
        if all(kk in line.split("#")[0].split() for kk in keywords)
    ]
    if len(found) == 0:
        raise ValueError(
            f"LAMMPS template misses a line containing {' '.join(keywords)}"
        )
    if len(found) > 1:
        raise ValueError(
            f"LAMMPS template has more than one line containing {' '.join(keywords)}"
        )
    return found[0]


def revise_variable_lines(lines: List[str], revisions: Dict) -> List[str]:
    """Override `variable KEY equal/index/... RHS` lines with revisions."""
    lines = list(lines)
    for key, value in revisions.items():
        pattern = re.compile(rf"^(\s*variable\s+{re.escape(key)}\s+\S+\s+).*$")
        hit = False
        for ii, line in enumerate(lines):
            m = pattern.match(line)
            if m:
                lines[ii] = m.group(1) + str(value)
                hit = True
        if not hit:
            logging.warning(
                "revision key %s not found as a `variable` line in the template; "
                "left unchanged",
                key,
            )
    return lines


def parse_pair_coeff_elements(lines: List[str]) -> List[str]:
    """Parse element symbols from the `pair_coeff * * ...` line."""
    idx = _find_line(lines, ["pair_coeff"])
    tokens = lines[idx].split("#")[0].split()
    elements = [t for t in tokens[1:] if t in chemical_symbols]
    return elements


class LmpTemplateTaskGroup(ConfSamplingTaskGroup):
    """Exploration task group running user-written LAMMPS input templates.

    The user provides a complete LAMMPS input script; PFD only rewrites a
    small, well-defined set of things:

    - the model file name in the `pair_style deepmd ...` line
      (placeholder ``pfd_model.pt2``);
    - the structure file name in the `read_data` line (rewritten to
      ``conf.lmp``);
    - the dump frequency and file name of the dump with id ``pfd_traj``;
    - `variable` definitions listed in ``revisions``.

    Everything else (ensemble, ramp/hold stages, fixes, computes) is fully
    controlled by the user's template.
    """

    def __init__(self):
        super().__init__()
        self.lmp_set = False

    def set_lmp(
        self,
        lmp_template: str,
        revisions: Optional[Dict] = None,
        trj_freq: int = 50,
        type_map: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        lmp_template : str
            Content of the user-written LAMMPS input script.
        revisions : dict, optional
            `variable` overrides, e.g. {"TSTOP": 2000, "SEED": 12345}.
        trj_freq : int
            Trajectory dump frequency in MD steps.
        type_map : list of str, optional
            Element order used when writing the LAMMPS data file. Must match
            the element order of the template's `pair_coeff` line. Defaults
            to the elements of the sampled configurations.
        """
        lines = [ll for ll in lmp_template.split("\n")]
        self._validate_template(lines)
        # graph-lowered .pt2 models fold ghost neighbours onto their owners
        # and require atom mapping; auto-insert after atom_style if missing
        if not any(
            "atom_modify" in ll.split("#")[0].split()
            and "map" in ll.split("#")[0].split()
            for ll in lines
        ):
            idx = _find_line(lines, ["atom_style"])
            lines.insert(idx + 1, "atom_modify     map yes")
        template_elements = parse_pair_coeff_elements(lines)
        if type_map is not None and template_elements:
            if list(type_map) != template_elements:
                raise ValueError(
                    f"type_map {list(type_map)} does not match the element order "
                    f"of the template pair_coeff line {template_elements}"
                )
        self.type_map = list(type_map) if type_map is not None else None
        self._pair_coeff_elements = template_elements

        # rewrite placeholders: the pair_style line keeps the
        # MODEL_PLACEHOLDER file name; RunLmp provides the model under this
        # exact name in the task directory at run time
        idx = _find_line(lines, ["read_data"])
        tokens = lines[idx].split()
        tokens[1] = lmp_conf_name
        lines[idx] = " ".join(tokens)

        idx = _find_line(lines, ["dump", DUMP_ID])
        tokens = lines[idx].split("#")[0].split()
        # dump ID group-ID style N file args...
        if len(tokens) < 6:
            raise ValueError(f"invalid dump line: {lines[idx]}")
        tokens[4] = str(trj_freq)
        tokens[5] = lmp_traj_name
        lines[idx] = " ".join(tokens)

        lines = revise_variable_lines(lines, revisions or {})

        self.lmp_template = "\n".join(lines)
        self.trj_freq = trj_freq
        self.revisions = dict(revisions or {})
        self.lmp_set = True

    def _validate_template(self, lines: List[str]) -> None:
        """Fail fast at submission time if the template misses the contract."""
        idx = _find_line(lines, ["pair_style"])
        tokens = lines[idx].split()
        if "deepmd" not in " ".join(tokens):
            raise ValueError(
                "the pair_style line of the LAMMPS template must use a deepmd "
                "pair style (deepmd or dpa4spin)"
            )
        if MODEL_PLACEHOLDER not in tokens:
            raise ValueError(
                f"the pair_style line must use the model placeholder "
                f"'{MODEL_PLACEHOLDER}', e.g. 'pair_style deepmd {MODEL_PLACEHOLDER}'"
            )
        _find_line(lines, ["read_data"])
        idx = _find_line(lines, ["dump", DUMP_ID])
        tokens = lines[idx].split()
        if not {"fx", "fy", "fz"}.issubset(set(tokens)):
            raise ValueError(
                f"the dump '{DUMP_ID}' must contain fx fy fz columns so that "
                "forces can be read back from the trajectory"
            )
        _find_line(lines, ["units"])
        _find_line(lines, ["run"])

    def make_task(self) -> "LmpTemplateTaskGroup":
        if not self.conf_set:
            raise RuntimeError("confs are not set")
        if not self.lmp_set:
            raise RuntimeError("LAMMPS template is not set")
        self.clear()
        for conf_str in self._sample_confs():
            atoms = read(StringIO(conf_str), format="extxyz")
            conf_lmp = self._atoms_to_lmp_data(atoms)
            task = ExplorationTask()
            task.add_file(lmp_input_name, self.lmp_template).add_file(
                lmp_conf_name, conf_lmp
            )
            self.add_task(task)
        return self

    def _atoms_to_lmp_data(self, atoms) -> str:
        symbols = atoms.get_chemical_symbols()
        specorder = self.type_map
        if specorder is None:
            specorder = sorted(set(symbols), key=lambda s: symbols.index(s))
        if self._pair_coeff_elements and specorder != self._pair_coeff_elements:
            raise ValueError(
                f"configuration elements order {specorder} does not match the "
                f"template pair_coeff order {self._pair_coeff_elements}"
            )
        buf = StringIO()
        write(
            buf,
            atoms,
            format="lammps-data",
            units="metal",
            atom_style="atomic",
            masses=True,
            specorder=specorder,
        )
        return buf.getvalue()

    @classmethod
    def args(cls) -> List[dargs.Argument]:
        doc_template = "Content of the user-written LAMMPS input template."
        doc_revisions = (
            "Overrides of `variable` definitions in the template, "
            "e.g. {\"TSTOP\": 2000, \"SEED\": 12345}."
        )
        doc_trj_freq = "Trajectory dump frequency in MD steps."
        doc_type_map = (
            "Element order for the LAMMPS data file; must match the element "
            "order of the template pair_coeff line. Defaults to the element "
            "order of the sampled configurations."
        )
        return [
            Argument("input_lmp_template", str, optional=False, doc=doc_template),
            Argument("revisions", dict, optional=True, default={}, doc=doc_revisions),
            Argument("trj_freq", int, optional=True, default=50, doc=doc_trj_freq),
            Argument(
                "type_map", [list, type(None)], optional=True, default=None, doc=doc_type_map
            ),
        ]

    @classmethod
    def make_task_grp_from_conf(
        cls, task_grp_config: Dict, init_confs: List[str], *args, **kwargs
    ) -> "List[LmpTemplateTaskGroup]":
        """Create LAMMPS template task groups from the workflow config."""
        confs_idx = task_grp_config.pop("conf_idx")
        n_sample = task_grp_config.pop("n_sample")
        task_grp_ls = []
        for ii in confs_idx:
            atoms_ls = read(init_confs[ii], index=":")
            if not isinstance(atoms_ls, list):
                atoms_ls = [atoms_ls]
            atoms_ls_str = []
            for atoms in atoms_ls:
                buf = StringIO()
                write(buf, atoms, format="extxyz")
                atoms_ls_str.append(buf.getvalue())
            task_grp = cls()
            task_grp.set_conf(
                conf_list=atoms_ls_str, n_sample=n_sample, random_sample=True
            )
            config = cls.normalize_config(task_grp_config, strict=False)
            template = config.pop("input_lmp_template")
            task_grp.set_lmp(lmp_template=template, **config)
            task_grp_ls.append(task_grp)
        return task_grp_ls

    @classmethod
    def normalize_config(cls, data: Dict, strict: bool = True) -> Dict:
        ta = cls.args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=strict)
        return data
