"""Dock host-guest complexes with ASE-compatible machine-learning potentials.

This module mirrors the optimisation strategy used by
``DockingFunction_withPenalty.py``: the host conformation is kept fixed, the
initial guest conformation is randomised, and differential evolution optimises
6 + n variables (Euler rotations, xyz translation, and n rotatable-bond
angles).  The objective is an atomistic machine-learning potential energy
provided by an ASE calculator, for example MACE-OFF or Meta FAIRChem UMA.
"""

import copy
import importlib.util
import os
from types import SimpleNamespace
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


def _ensure_torch_compiler_is_compiling() -> None:
    """Provide a compatibility shim required by some MACE builds.

    Recent MACE versions call ``torch.compiler.is_compiling()``.  Some
    supported PyTorch 2.x environments expose ``torch.compiler`` but do not yet
    provide that helper, which otherwise raises an ``AttributeError`` during
    every MACE energy evaluation.
    """

    if importlib.util.find_spec("torch") is None:
        return

    import torch

    if not hasattr(torch, "compiler"):
        torch.compiler = SimpleNamespace()

    if hasattr(torch.compiler, "is_compiling"):
        return

    dynamo = getattr(torch, "_dynamo", None)
    dynamo_is_compiling = getattr(dynamo, "is_compiling", None)
    if callable(dynamo_is_compiling):
        torch.compiler.is_compiling = dynamo_is_compiling
    else:
        torch.compiler.is_compiling = lambda: False


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

    _ensure_torch_compiler_is_compiling()
    _require_module("mace", "pip install mace-torch")
    from mace.calculators import mace_off

    return mace_off(model=model, device=device, **kwargs)


def fairchem_uma_calculator(
    model: str = "uma-s-1p1",
    device: str = "cpu",
    task_name: str = "omol",
    inference_settings="default",
    seed: int = 41,
    predictor_kwargs: Optional[dict] = None,
):
    """Create a Meta FAIRChem UMA ASE calculator.

    UMA is exposed through FAIRChem's ASE-compatible ``FAIRChemCalculator``.
    The returned calculator maps the optimised guest variables directly to the
    host-guest complex potential energy used by ``differential_evolution``.

    Parameters
    ----------
    model
        UMA checkpoint name accepted by ``fairchem.core.pretrained_mlip``, such
        as ``"uma-s-1p1"``, or a local checkpoint path supported by FAIRChem.
    device
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    task_name
        UMA task/head name.  ``"omol"`` is the default for finite molecular
        host-guest complexes; use another FAIRChem task such as ``"omc"`` when
        it better matches the target system.
    inference_settings
        FAIRChem inference settings object or preset string (for example
        ``"default"`` or ``"turbo"``).
    seed
        Random seed forwarded to ``FAIRChemCalculator``.
    predictor_kwargs
        Extra keyword arguments forwarded to
        ``pretrained_mlip.get_predict_unit``.
    """

    _require_module("fairchem", "pip install fairchem-core")
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    kwargs = dict(predictor_kwargs or {})
    predictor = pretrained_mlip.get_predict_unit(
        model,
        device=device,
        inference_settings=inference_settings,
        **kwargs,
    )
    return FAIRChemCalculator(predictor, task_name=task_name, seed=seed)


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


def _set_ase_charge_and_spin(
    atoms,
    charge: Optional[int] = None,
    spin_multiplicity: Optional[int] = None,
):
    """Attach total molecular charge and spin multiplicity metadata to ASE atoms."""

    if charge is not None:
        atoms.info["charge"] = int(charge)
    if spin_multiplicity is not None:
        atoms.info["spin"] = int(spin_multiplicity)
    return atoms


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


def rdkit_mol_to_ase_atoms(
    mol: Chem.Mol,
    charge: Optional[int] = None,
    spin_multiplicity: Optional[int] = None,
):
    """Convert one RDKit molecule with a conformer to an ASE ``Atoms`` object."""

    _require_module("ase", "pip install ase")
    from ase import Atoms

    symbols, positions = _symbols_and_positions(mol)
    atoms = Atoms(symbols=symbols, positions=positions)
    return _set_ase_charge_and_spin(atoms, charge=charge, spin_multiplicity=spin_multiplicity)


def rdkit_complex_to_ase_atoms(
    host_mol: Chem.Mol,
    guest_mol: Chem.Mol,
    charge: Optional[int] = None,
    spin_multiplicity: Optional[int] = None,
):
    """Convert a fixed host and transformed guest to one ASE ``Atoms`` object."""

    _require_module("ase", "pip install ase")
    from ase import Atoms

    host_symbols, host_positions = _symbols_and_positions(host_mol)
    guest_symbols, guest_positions = _symbols_and_positions(guest_mol)
    atoms = Atoms(
        symbols=host_symbols + guest_symbols,
        positions=np.vstack([host_positions, guest_positions]),
    )
    return _set_ase_charge_and_spin(atoms, charge=charge, spin_multiplicity=spin_multiplicity)


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


def _conformer_centroid(mol: Chem.Mol) -> np.ndarray:
    """Return the arithmetic centroid of an RDKit conformer."""

    return np.mean(np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64), axis=0)


def _center_conformer_at_origin(mol: Chem.Mol) -> None:
    """Translate a molecule conformer so rotations keep its centroid fixed."""

    conf = mol.GetConformer()
    centroid = _conformer_centroid(mol)
    for atom_idx, position in enumerate(np.asarray(conf.GetPositions(), dtype=np.float64)):
        conf.SetAtomPosition(atom_idx, tuple(float(x) for x in position - centroid))


def _estimate_guest_size(guest_mol: Chem.Mol, seed: int = 1000, sel_conformers: int = 50) -> Tuple[float, float]:
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
    centroid_radii = []
    for coords in coords_sets:
        coords = np.asarray(coords, dtype=np.float64)
        if len(coords) > 1:
            lengths.append(float(np.max(distance.cdist(coords, coords))))
        centroid = np.mean(coords, axis=0)
        centroid_radii.append(float(np.max(np.linalg.norm(coords - centroid, axis=1))))
    if not lengths:
        raise ValueError("Error estimating guest molecule size for docking bounds.")
    return max(lengths), max(centroid_radii)


def _estimate_guest_length(guest_mol: Chem.Mol, seed: int = 1000, sel_conformers: int = 50) -> float:
    return _estimate_guest_size(guest_mol, seed=seed, sel_conformers=sel_conformers)[0]


def _guest_center_bounds(
    host_coords: np.ndarray,
    guest_length: float,
    guest_radius: float,
    bounds_padding: float = 0.0,
    center_bound_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build translation bounds that keep the guest centroid near the host centroid.

    The translation variables are interpreted as the guest-centroid target because
    the optimiser recentres the starting guest conformer at the origin.  By
    default the allowed centroid displacement is the smaller of the old adaptive
    bound (guest diameter) and the host radial size after subtracting the guest
    radius.  This avoids sampling poses whose guest centroid is far outside the
    host while still allowing callers to override the radius explicitly.
    """

    host_coords = np.asarray(host_coords, dtype=np.float64)
    host_centroid = np.mean(host_coords, axis=0)
    host_radius = float(np.max(np.linalg.norm(host_coords - host_centroid, axis=1)))
    padding = float(bounds_padding)

    if center_bound_radius is None:
        bound_radius = min(float(guest_length), max(host_radius - float(guest_radius), 0.0))
    else:
        bound_radius = float(center_bound_radius)

    bound_radius = max(bound_radius + padding, 0.0)
    max_coord = host_centroid + bound_radius
    min_coord = host_centroid - bound_radius
    return min_coord, max_coord, host_centroid, bound_radius


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
        complex_charge: Optional[int] = None,
        complex_spin_multiplicity: Optional[int] = None,
        host_charge: Optional[int] = None,
        host_spin_multiplicity: Optional[int] = None,
        guest_charge: Optional[int] = None,
        guest_spin_multiplicity: Optional[int] = None,
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
        _center_conformer_at_origin(self.mol)
        self.subtract_host_energy = subtract_host_energy
        self.subtract_guest_energy = subtract_guest_energy
        self.complex_charge = complex_charge
        self.complex_spin_multiplicity = complex_spin_multiplicity
        self.host_charge = host_charge
        self.host_spin_multiplicity = host_spin_multiplicity
        self.guest_charge = guest_charge
        self.guest_spin_multiplicity = guest_spin_multiplicity
        self._host_energy = (
            self._calculate_energy(
                rdkit_mol_to_ase_atoms(
                    host_mol,
                    charge=host_charge,
                    spin_multiplicity=host_spin_multiplicity,
                )
            )
            if subtract_host_energy
            else 0.0
        )

    def _calculate_energy(self, atoms) -> float:
        _ensure_torch_compiler_is_compiling()
        atoms.calc = self.calculator
        return float(atoms.get_potential_energy())

    def score_conformation(self, values) -> float:
        if len(np.shape(values)) < 2:
            values = np.expand_dims(values, axis=0)

        guest = apply_changes(self.mol, values[0], self.rotable_bonds)
        complex_atoms = rdkit_complex_to_ase_atoms(
            self.host_mol,
            guest,
            charge=self.complex_charge,
            spin_multiplicity=self.complex_spin_multiplicity,
        )
        energy = self._calculate_energy(complex_atoms)

        if self.subtract_host_energy:
            energy -= self._host_energy
        if self.subtract_guest_energy:
            energy -= self._calculate_energy(
                rdkit_mol_to_ase_atoms(
                    guest,
                    charge=self.guest_charge,
                    spin_multiplicity=self.guest_spin_multiplicity,
                )
            )
        return energy

    def get_adaptive_bounds(self, sel_conformers: int = 50) -> float:
        return _estimate_guest_length(self.init_guest_mol, seed=self.seed, sel_conformers=sel_conformers)

    def get_adaptive_guest_size(self, sel_conformers: int = 50) -> Tuple[float, float]:
        return _estimate_guest_size(self.init_guest_mol, seed=self.seed, sel_conformers=sel_conformers)


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
    complex_charge: Optional[int] = None,
    complex_spin_multiplicity: Optional[int] = None,
    host_charge: Optional[int] = None,
    host_spin_multiplicity: Optional[int] = None,
    guest_charge: Optional[int] = None,
    guest_spin_multiplicity: Optional[int] = None,
    popsize: int = 15,
    revise_popsize: bool = False,
    bounds_padding: float = 0.0,
    center_bound_radius: Optional[float] = None,
    **kwargs,
):
    """Optimise a host-guest pose with an ASE-compatible ML potential.

    Parameters mirror ``dock_compound`` where possible.  The host is read from
    ``host_mol`` rather than from a PLY mesh because ML interatomic potentials
    require atom identities and atom coordinates.  ``complex_charge`` and
    ``complex_spin_multiplicity`` are written to ``ase.Atoms.info`` as the total
    charge and spin multiplicity for calculators such as UMA/OMOL.  When
    subtracting isolated host or guest energies, use the corresponding
    ``host_*`` or ``guest_*`` arguments to describe those isolated systems.
    ``bounds_padding`` expands or shrinks the guest-centroid translation box,
    while ``center_bound_radius`` can be used to explicitly set the half-width
    around the host centroid.

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
        complex_charge=complex_charge,
        complex_spin_multiplicity=complex_spin_multiplicity,
        host_charge=host_charge,
        host_spin_multiplicity=host_spin_multiplicity,
        guest_charge=guest_charge,
        guest_spin_multiplicity=guest_spin_multiplicity,
    )

    host_coords = np.asarray(host_mol.GetConformer().GetPositions(), dtype=np.float64)
    guest_length, guest_radius = opt.get_adaptive_guest_size(50)
    min_coord, max_coord, host_centroid, effective_center_bound_radius = _guest_center_bounds(
        host_coords=host_coords,
        guest_length=guest_length,
        guest_radius=guest_radius,
        bounds_padding=bounds_padding,
        center_bound_radius=center_bound_radius,
    )

    max_bound = np.concatenate([[np.pi] * 3, max_coord, [np.pi] * len(opt.rotable_bonds)], axis=0)
    min_bound = np.concatenate([[-np.pi] * 3, min_coord, [-np.pi] * len(opt.rotable_bonds)], axis=0)
    bounds = Bounds(lb=min_bound, ub=max_bound, keep_feasible=True)

    print(f"Number of Optimized Parameter: {len(max_bound)}")
    print(f"Number of Rotatable Bonds: {len(opt.rotable_bonds)}")
    print(f"Guest centroid translation bounds: {min_coord} to {max_coord}")

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
        "complex_charge": complex_charge,
        "complex_spin_multiplicity": complex_spin_multiplicity,
        "host_charge": host_charge,
        "host_spin_multiplicity": host_spin_multiplicity,
        "guest_charge": guest_charge,
        "guest_spin_multiplicity": guest_spin_multiplicity,
        "host_centroid": np.asarray(host_centroid, dtype=np.float64),
        "guest_length": float(guest_length),
        "guest_radius": float(guest_radius),
        "center_bound_radius": float(effective_center_bound_radius),
        "translation_min_bound": np.asarray(min_coord, dtype=np.float64),
        "translation_max_bound": np.asarray(max_coord, dtype=np.float64),
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


def dock_compound_with_fairchem_uma(
    guest_mol: Chem.Mol,
    host_mol: Chem.Mol,
    *,
    model: str = "uma-s-1p1",
    device: str = "cpu",
    task_name: str = "omol",
    inference_settings="default",
    seed: int = 41,
    charge: int = 0,
    spin_multiplicity: int = 1,
    host_charge: Optional[int] = None,
    host_spin_multiplicity: Optional[int] = None,
    guest_charge: Optional[int] = None,
    guest_spin_multiplicity: Optional[int] = None,
    uma_kwargs: Optional[dict] = None,
    **dock_kwargs,
):
    """Dock a host-guest complex with the Meta FAIRChem UMA potential.

    This convenience wrapper keeps the host coordinates fixed, randomly
    initialises the guest, and optimises the guest ``6 + n`` variables with the
    same differential-evolution workflow as ``dock_compound_with_ase_potential``.
    The optimisation objective is the UMA/FAIRChem ASE calculator energy of the
    transformed host-guest complex.  ``charge`` and ``spin_multiplicity`` are
    the total charge and spin multiplicity of the host-guest complex and are
    stored on each ASE ``Atoms`` object as ``atoms.info["charge"]`` and
    ``atoms.info["spin"]``.

    Returns
    -------
    opt_complex_mol, opt_guest_mol, starting_guest_mol, docking_result
        The predicted low-energy complex, the guest conformation in that
        complex, the randomised starting guest, and optimisation metadata.
    """

    dock_kwargs.setdefault("seed", seed)
    calculator = fairchem_uma_calculator(
        model=model,
        device=device,
        task_name=task_name,
        inference_settings=inference_settings,
        seed=seed,
        predictor_kwargs=uma_kwargs,
    )
    return dock_compound_with_ase_potential(
        guest_mol=guest_mol,
        host_mol=host_mol,
        calculator=calculator,
        complex_charge=charge,
        complex_spin_multiplicity=spin_multiplicity,
        host_charge=host_charge,
        host_spin_multiplicity=host_spin_multiplicity,
        guest_charge=guest_charge,
        guest_spin_multiplicity=guest_spin_multiplicity,
        **dock_kwargs,
    )
