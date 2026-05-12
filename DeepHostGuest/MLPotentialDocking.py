"""Dock host-guest complexes with ASE-compatible machine-learning potentials.

This module mirrors the optimisation strategy used by
``DockingFunction_withPenalty.py``: the host conformation is kept fixed, the
initial guest conformation is randomised, and differential evolution optimises
6 + n variables (Euler rotations, xyz translation, and n rotatable-bond
angles).  The objective is an atomistic machine-learning potential energy
provided by an ASE calculator, for example MACE-OFF.
"""

import copy
import importlib.util
import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.optimize import Bounds, differential_evolution
from scipy.spatial import distance



def get_torsions(mol_list):
    atom_counter = 0
    torsion_list = []
    for mol in mol_list:
        torsion_query = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        matches = mol.GetSubstructMatches(torsion_query)
        for match in matches:
            idx2, idx3 = match[0], match[1]
            bond = mol.GetBondBetweenAtoms(idx2, idx3)
            j_atom = mol.GetAtomWithIdx(idx2)
            k_atom = mol.GetAtomWithIdx(idx3)
            for b1 in j_atom.GetBonds():
                if b1.GetIdx() == bond.GetIdx():
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in k_atom.GetBonds():
                    if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    if idx4 == idx1:
                        continue
                    if (mol.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (mol.GetAtomWithIdx(idx4).GetAtomicNum() == 1):
                        continue
                    if mol.GetAtomWithIdx(idx4).IsInRing():
                        torsion_list.append((idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    torsion_list.append((idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                    break
                break
        atom_counter += mol.GetNumAtoms()
    return torsion_list


def get_transformation_matrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    return np.array([
        [cz * cy, (cz * sy * sx) - (sz * cx), (cz * sy * cx) + (sz * sx), disp_x],
        [sz * cy, (sz * sy * sx) + (cz * cx), (sz * sy * cx) - (cz * sx), disp_y],
        [-sy, cy * sx, cy * cx, disp_z],
        [0, 0, 0, 1],
    ], dtype=np.double)


def apply_changes(mol: Chem.Mol, values, rotable_bonds):
    opt_mol = copy.copy(mol)
    conf = opt_mol.GetConformer()
    for r, tors in enumerate(rotable_bonds):
        rdMolTransforms.SetDihedralRad(conf, tors[0], tors[1], tors[2], tors[3], values[6 + r])
    rdMolTransforms.TransformConformer(conf, get_transformation_matrix(values[:6]))
    return opt_mol


def get_random_conformation(mol: Chem.Mol, rotable_bonds=None, seed=None, canonicalize=True):
    work_mol = _ensure_conformer(mol, add_hs=False, seed=seed or 1000)
    if seed is not None:
        np.random.seed(seed)
    if rotable_bonds is None:
        rotable_bonds = get_torsions([work_mol])
    rand_vec = np.random.rand(len(rotable_bonds) + 6) * 10.0
    new_conf = apply_changes(work_mol, rand_vec, rotable_bonds)
    if canonicalize:
        rdMolTransforms.CanonicalizeConformer(new_conf.GetConformer())
    return new_conf


MACE_OFF_SUPPORTED_ELEMENTS = {"H", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}


def _require_module(module_name: str, package_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Optional dependency '{module_name}' is required for ML-potential docking. "
            f"Install it with: {package_hint}"
        )


def mace_off_calculator(model: str = "medium", device: str = "cpu", **kwargs):
    """Create a MACE-OFF ASE calculator.

    Parameters
    ----------
    model
        MACE-OFF model size or checkpoint path accepted by ``mace.calculators.mace_off``.
    device
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    **kwargs
        Extra keyword arguments forwarded to ``mace_off``.
    """

    _require_module("mace", "pip install mace-torch")
    from mace.calculators import mace_off

    return mace_off(model=model, device=device, **kwargs)


def _ensure_conformer(mol: Chem.Mol, *, add_hs: bool = True, seed: int = 1000) -> Chem.Mol:
    if not isinstance(mol, Chem.Mol):
        raise TypeError("Expected an RDKit Chem.Mol instance.")

    work_mol = copy.deepcopy(mol)
    if add_hs:
        work_mol = Chem.AddHs(work_mol, addCoords=True)

    if work_mol.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed) if seed is not None else -1
        status = AllChem.EmbedMolecule(work_mol, params)
        if status != 0:
            status = AllChem.EmbedMolecule(work_mol, randomSeed=int(seed) if seed is not None else -1)
        if status != 0:
            raise ValueError("RDKit failed to embed a 3D conformer for the molecule.")
        AllChem.MMFFOptimizeMolecule(work_mol)
    return work_mol


def _symbols_and_positions(mol: Chem.Mol) -> Tuple[list, np.ndarray]:
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = np.asarray(conf.GetPositions(), dtype=np.float64)
    return symbols, positions


def validate_supported_elements(mols: Iterable[Chem.Mol], supported_elements: Sequence[str], potential_name: str) -> None:
    """Raise a helpful error when molecules contain elements unsupported by a potential."""

    supported = set(supported_elements)
    present = {atom.GetSymbol() for mol in mols for atom in mol.GetAtoms()}
    unsupported = sorted(present - supported)
    if unsupported:
        raise ValueError(
            f"{potential_name} does not support element(s): {', '.join(unsupported)}. "
            f"Supported elements are: {', '.join(sorted(supported))}."
        )


def rdkit_mol_to_ase_atoms(mol: Chem.Mol):
    """Convert one RDKit molecule with a conformer to an ASE ``Atoms`` object."""

    _require_module("ase", "pip install ase")
    from ase import Atoms

    symbols, positions = _symbols_and_positions(mol)
    return Atoms(symbols=symbols, positions=positions)


def rdkit_complex_to_ase_atoms(host_mol: Chem.Mol, guest_mol: Chem.Mol):
    """Convert a fixed host and transformed guest to one ASE ``Atoms`` object."""

    _require_module("ase", "pip install ase")
    from ase import Atoms

    host_symbols, host_positions = _symbols_and_positions(host_mol)
    guest_symbols, guest_positions = _symbols_and_positions(guest_mol)
    return Atoms(
        symbols=host_symbols + guest_symbols,
        positions=np.vstack([host_positions, guest_positions]),
    )


def combine_rdkit_mols_with_conformers(host_mol: Chem.Mol, guest_mol: Chem.Mol) -> Chem.Mol:
    """Create an RDKit complex molecule from host and guest conformers."""

    complex_mol = Chem.CombineMols(host_mol, guest_mol)
    conf = Chem.Conformer(complex_mol.GetNumAtoms())
    host_positions = np.asarray(host_mol.GetConformer().GetPositions(), dtype=np.float64)
    guest_positions = np.asarray(guest_mol.GetConformer().GetPositions(), dtype=np.float64)
    positions = np.vstack([host_positions, guest_positions])
    for atom_idx, position in enumerate(positions):
        conf.SetAtomPosition(atom_idx, tuple(float(x) for x in position))
    complex_mol.RemoveAllConformers()
    complex_mol.AddConformer(conf, assignId=True)
    return complex_mol


def _estimate_guest_length(guest_mol: Chem.Mol, seed: int = 1000, sel_conformers: int = 50) -> float:
    copy_mol = copy.deepcopy(guest_mol)
    mol_lengths = []
    try:
        cids = AllChem.EmbedMultipleConfs(copy_mol, numConfs=1000, randomSeed=int(seed), numThreads=0)
        props = AllChem.MMFFGetMoleculeProperties(copy_mol)
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(copy_mol, props, confId=cid) if props is not None else None
            energy = 0.0
            if ff is not None:
                ff.Minimize(maxIts=200)
                energy = ff.CalcEnergy()
            mol_lengths.append((energy, cid))
        mol_lengths.sort(key=lambda item: item[0])
        coords_sets = [copy_mol.GetConformer(cid).GetPositions() for _, cid in mol_lengths[:sel_conformers]]
    except Exception as exc:
        print(exc)
        coords_sets = [guest_mol.GetConformer().GetPositions()]

    lengths = []
    for coords in coords_sets:
        coords = np.asarray(coords, dtype=np.float64)
        if len(coords) > 1:
            lengths.append(float(np.max(distance.cdist(coords, coords))))
    if not lengths:
        raise ValueError("Error estimating guest molecule size for docking bounds.")
    return max(lengths)


class ASEPotentialConformationOptimizer:
    """Score transformed guest conformations with an ASE calculator."""

    def __init__(
        self,
        guest_mol: Chem.Mol,
        host_mol: Chem.Mol,
        calculator,
        *,
        seed: int = 1000,
        canonicalize_guest: bool = True,
        subtract_host_energy: bool = False,
        subtract_guest_energy: bool = False,
    ):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.host_mol = host_mol
        self.calculator = calculator
        self.rotable_bonds = get_torsions([guest_mol])
        self.init_guest_mol = guest_mol
        self.mol = get_random_conformation(
            guest_mol,
            rotable_bonds=self.rotable_bonds,
            seed=seed,
            canonicalize=canonicalize_guest,
        )
        self.subtract_host_energy = subtract_host_energy
        self.subtract_guest_energy = subtract_guest_energy
        self._host_energy = self._calculate_energy(rdkit_mol_to_ase_atoms(host_mol)) if subtract_host_energy else 0.0

    def _calculate_energy(self, atoms) -> float:
        atoms.calc = self.calculator
        return float(atoms.get_potential_energy())

    def score_conformation(self, values) -> float:
        if len(np.shape(values)) < 2:
            values = np.expand_dims(values, axis=0)

        guest = apply_changes(self.mol, values[0], self.rotable_bonds)
        complex_atoms = rdkit_complex_to_ase_atoms(self.host_mol, guest)
        energy = self._calculate_energy(complex_atoms)

        if self.subtract_host_energy:
            energy -= self._host_energy
        if self.subtract_guest_energy:
            energy -= self._calculate_energy(rdkit_mol_to_ase_atoms(guest))
        return energy

    def get_adaptive_bounds(self, sel_conformers: int = 50) -> float:
        return _estimate_guest_length(self.init_guest_mol, seed=self.seed, sel_conformers=sel_conformers)


def dock_compound_with_ase_potential(
    guest_mol: Chem.Mol,
    host_mol: Chem.Mol,
    calculator,
    *,
    seed: int = 1000,
    savepath: Optional[str] = None,
    output_guest_path: Optional[str] = None,
    output_complex_path: Optional[str] = None,
    canonicalize_guest: bool = True,
    add_hs: bool = True,
    subtract_host_energy: bool = False,
    subtract_guest_energy: bool = False,
    popsize: int = 15,
    revise_popsize: bool = False,
    bounds_padding: float = 0.0,
    **kwargs,
):
    """Optimise a host-guest pose with an ASE-compatible ML potential.

    Parameters mirror ``dock_compound`` where possible.  The host is read from
    ``host_mol`` rather than from a PLY mesh because ML interatomic potentials
    require atom identities and atom coordinates.

    Returns
    -------
    opt_complex_mol, opt_guest_mol, starting_guest_mol, docking_result
    """

    if seed is not None:
        np.random.seed(seed)

    host_mol = _ensure_conformer(host_mol, add_hs=add_hs, seed=seed)
    guest_mol = _ensure_conformer(guest_mol, add_hs=add_hs, seed=seed)

    opt = ASEPotentialConformationOptimizer(
        guest_mol=guest_mol,
        host_mol=host_mol,
        calculator=calculator,
        seed=seed,
        canonicalize_guest=canonicalize_guest,
        subtract_host_energy=subtract_host_energy,
        subtract_guest_energy=subtract_guest_energy,
    )

    host_coords = np.asarray(host_mol.GetConformer().GetPositions(), dtype=np.float64)
    center_of_mass = np.mean(host_coords, axis=0)
    guest_length = opt.get_adaptive_bounds(50) + float(bounds_padding)
    max_coord = center_of_mass + guest_length
    min_coord = center_of_mass - guest_length

    max_bound = np.concatenate([[np.pi] * 3, max_coord, [np.pi] * len(opt.rotable_bonds)], axis=0)
    min_bound = np.concatenate([[-np.pi] * 3, min_coord, [-np.pi] * len(opt.rotable_bonds)], axis=0)
    bounds = Bounds(lb=min_bound, ub=max_bound, keep_feasible=True)

    print(f"Number of Optimized Parameter: {len(max_bound)}")
    print(f"Number of Rotatable Bonds: {len(opt.rotable_bonds)}")

    optimization_history = []

    def callback(xk, convergence):
        optimization_history.append(xk)

    if revise_popsize:
        new_popsize = int(popsize + 15 * np.log(len(opt.rotable_bonds) + 1))
    else:
        new_popsize = int(np.ceil(popsize / (len(opt.rotable_bonds) + 6)))

    result = differential_evolution(
        opt.score_conformation,
        bounds=bounds,
        callback=callback,
        seed=seed,
        popsize=new_popsize,
        **kwargs,
    )

    opt_guest_mol = apply_changes(opt.mol, result["x"], opt.rotable_bonds)
    opt_complex_mol = combine_rdkit_mols_with_conformers(host_mol, opt_guest_mol)

    if savepath:
        np.savetxt(savepath, optimization_history)
    if output_guest_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_guest_path)), exist_ok=True)
        AllChem.MolToMolFile(opt_guest_mol, output_guest_path)
    if output_complex_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_complex_path)), exist_ok=True)
        AllChem.MolToMolFile(opt_complex_mol, output_complex_path)

    docking_result = {
        "num_atoms": opt_guest_mol.GetNumHeavyAtoms(),
        "num_rotbonds": len(opt.rotable_bonds),
        "rotbonds": opt.rotable_bonds,
        "success": bool(result["success"]),
        "fun": float(result["fun"]),
        "message": str(result["message"]),
        "nit": int(result["nit"]),
        "nfev": int(result["nfev"]),
        "x": np.asarray(result["x"], dtype=np.float64),
        "energy_unit": "eV (ASE calculator default)",
    }

    return opt_complex_mol, opt_guest_mol, opt.mol, docking_result


def dock_compound_with_mace_off(
    guest_mol: Chem.Mol,
    host_mol: Chem.Mol,
    *,
    model: str = "medium",
    device: str = "cpu",
    validate_elements: bool = True,
    mace_kwargs: Optional[dict] = None,
    **dock_kwargs,
):
    """Convenience wrapper for MACE-OFF docking.

    MACE-OFF is intended for neutral organic systems containing H, C, N, O, P,
    S, F, Cl, Br and I.  For unsupported chemistries, pass a different ASE
    calculator to ``dock_compound_with_ase_potential``.
    """

    host_checked = _ensure_conformer(host_mol, add_hs=dock_kwargs.get("add_hs", True), seed=dock_kwargs.get("seed", 1000))
    guest_checked = _ensure_conformer(guest_mol, add_hs=dock_kwargs.get("add_hs", True), seed=dock_kwargs.get("seed", 1000))
    if validate_elements:
        validate_supported_elements(
            [host_checked, guest_checked],
            MACE_OFF_SUPPORTED_ELEMENTS,
            "MACE-OFF",
        )

    calculator = mace_off_calculator(model=model, device=device, **(mace_kwargs or {}))
    return dock_compound_with_ase_potential(
        guest_mol=guest_checked,
        host_mol=host_checked,
        calculator=calculator,
        **dock_kwargs,
    )
