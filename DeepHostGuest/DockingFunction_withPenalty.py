import os
import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, AllChem
from scipy.spatial import distance, cKDTree
from scipy.optimize import Bounds, differential_evolution
from torch_geometric.data import Batch
from collections import OrderedDict
from functools import partial
import torch

from DeepHostGuest.utils.data import *


def _logpdf_gaussian_mixture(y, mu, sigma, pi):
    # y: (...,1), mu/sigma/pi: (N,) aligned to model outputs along axis=0
    # Compute log N(y|mu, sigma) = -0.5*((y-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
    inv_sigma = 1.0 / np.clip(sigma, 1e-8, None)
    z = (y - mu) * inv_sigma
    log_norm = -0.5 * (z * z) - np.log(np.clip(sigma, 1e-8, None)) - 0.5 * np.log(2.0 * np.pi)
    return log_norm + np.log(np.clip(pi, 1e-12, None))


def calculate_probablity_fast(pi, sigma, mu, y):  # drop-in replacement
    logprob = _logpdf_gaussian_mixture(y, mu, sigma, pi)
    # Sum across components per pair
    return np.exp(logprob).sum(axis=1)


class _PenaltyCalculator:
    __slots__ = (
        "host_coords",
        "host_tree",
        "guest_template",
        "guest_nonbond_pairs",
        "host_guest_rm",
        "guest_guest_rm",
    )

    def __init__(self, host_mol: Chem.Mol, guest_template: Chem.Mol,
                 *, removeHs: bool, host_guest_rm: float, guest_guest_rm: float):

        host_ref = Chem.RemoveHs(host_mol, sanitize=False)
        self.host_coords = np.asarray(host_ref.GetConformer().GetPositions(), dtype=np.float64)
        self.host_tree = cKDTree(self.host_coords)

        self.guest_template = Chem.RemoveHs(guest_template)
        self.guest_nonbond_pairs = self._build_guest_nonbond_pairs(self.guest_template)

        self.host_guest_rm = float(host_guest_rm)
        self.guest_guest_rm = float(guest_guest_rm)

    @staticmethod
    def _build_guest_nonbond_pairs(guest: Chem.Mol):
        bonds = set()
        for b in guest.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bonds.add((min(i, j), max(i, j)))
        N = guest.GetNumAtoms()
        idx_i, idx_j = [], []
        for i in range(N - 1):
            for j in range(i + 1, N):
                if (i, j) not in bonds:
                    idx_i.append(i)
                    idx_j.append(j)
        return np.asarray(idx_i, dtype=np.int32), np.asarray(idx_j, dtype=np.int32)

    @staticmethod
    def _lj_like(dist: np.ndarray):
        r = np.clip(dist, 1e-12, None)
        inv = 3.0 / r
        inv2 = inv * inv
        inv6 = inv2 * inv2 * inv2
        inv12 = inv6 * inv6
        return inv12 - inv6

    def penalty(self, guest_mol_current: Chem.Mol):
        gm = Chem.RemoveHs(guest_mol_current)
        gcoords = np.asarray(gm.GetConformer().GetPositions(), dtype=np.float64)

        host_guest_terms = 0.0
        if self.host_guest_rm > 0:
            for p in gcoords:
                nb = self.host_tree.query_ball_point(p, r=self.host_guest_rm)
                if not nb:
                    continue
                diffs = self.host_coords[nb] - p  # (k,3)
                d = np.linalg.norm(diffs, axis=1)  # (k,)
                mask = (d > 0.0) & (d < self.host_guest_rm)
                if np.any(mask):
                    host_guest_terms += np.sum(self._lj_like(d[mask]))

        gg_terms = 0.0
        if self.guest_guest_rm > 0:
            i, j = self.guest_nonbond_pairs
            if i.size:
                vec = gcoords[j] - gcoords[i]
                d = np.linalg.norm(vec, axis=1)
                mask = (d > 0.0) & (d < self.guest_guest_rm)
                if np.any(mask):
                    gg_terms = np.sum(self._lj_like(d[mask]))

        return abs(host_guest_terms), abs(gg_terms)


def score_compound(guest_mol, host_ply, model, removeHs=False, dist_threshold=5.,
                   seed=1000, device='cpu', host_mol=None,
                   penalty=False, host_guest_rm=3., guest_guest_rm=1.5):
    if penalty and not host_mol:
        raise Exception('host_mol is required for penalty')

    if seed:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    if isinstance(guest_mol, Chem.Mol):
        try:
            guest_mol.GetConformer()
        except Exception:
            guest_mol = Chem.AddHs(guest_mol)
            AllChem.EmbedMolecule(guest_mol)
            AllChem.MMFFOptimizeMolecule(guest_mol)
        guest = from_networkx(mol2graph.mol_to_nx(guest_mol, removeHs=removeHs))
        guest = Batch.from_data_list([guest])
    else:
        raise Exception('mol should be an RDKIT molecule')

    if not isinstance(host_ply, Batch):
        if isinstance(host_ply, str):
            host = Cartesian()(FaceToEdge()(read_ply(host_ply)))
            host = Batch.from_data_list([host])
        else:
            raise Exception('host should be a string with the ply file path or a Batch instance')

    model.eval()
    guest, host = guest.to(device), host.to(device)
    pi_t, sigma_t, mu_t, dist_t, atom_types, bond_types, batch = model(guest, host)
    pi = pi_t.detach().cpu().numpy()
    sigma = sigma_t.detach().cpu().numpy()
    mu = mu_t.detach().cpu().numpy()
    dist = dist_t.detach().cpu().numpy()

    # [SPEED-UP] use fast probability
    prob = calculate_probablity_fast(pi, sigma, mu, dist)
    if dist_threshold:
        prob[torch.where(dist_t > dist_threshold)[0]] = 0.0
    score = -np.sum(prob, axis=0)

    if penalty:
        # [SPEED-UP] cached penalty
        pen = _PenaltyCalculator(host_mol, guest_mol, removeHs=removeHs,
                                 host_guest_rm=host_guest_rm, guest_guest_rm=guest_guest_rm)
        p_hg, p_gg = pen.penalty(guest_mol)
        score += p_hg + p_gg
    return score


def count_num_opt_parameters(guest_mol):
    return 6 + len(get_torsions([guest_mol]))


def dock_compound(guest_mol, host_ply, model, removeHs=True, dist_threshold=5.,
                  seed=1000, device='cpu', savepath=None, host_mol=None, canonicalize_guest=True,
                  host_guest_rm=3., guest_guest_rm=1.5, **kwargs):
    """
    Perform docking of a compound molecule to a host polymer using a specified model and method.
    (Signature and behavior preserved.)
    """
    if not host_mol:
        raise Exception('host_mol is required for penalty')

    if seed:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    if isinstance(guest_mol, Chem.Mol):
        try:
            guest_mol.GetConformer()
        except Exception:
            guest_mol = Chem.AddHs(guest_mol)
            AllChem.EmbedMolecule(guest_mol)
            AllChem.MMFFOptimizeMolecule(guest_mol)
        guest = from_networkx(mol2graph.mol_to_nx(guest_mol, removeHs=removeHs))
        guest = Batch.from_data_list([guest])
    else:
        raise Exception('mol should be an RDKIT molecule')

    if not isinstance(host_ply, Batch):
        if isinstance(host_ply, str):
            host = Cartesian()(FaceToEdge()(read_ply(host_ply)))
            host = Batch.from_data_list([host])
        else:
            raise Exception('host should be a string with the ply file path or a Batch instance')

    model.eval()
    guest, host = guest.to(device), host.to(device)
    pi_t, sigma_t, mu_t, dist_t, atom_types, bond_types, batch = model(guest, host)
    pi = pi_t.detach().cpu().numpy()
    sigma = sigma_t.detach().cpu().numpy()
    mu = mu_t.detach().cpu().numpy()
    host_coords = host.pos.detach().cpu().numpy()

    # [SPEED-UP] build cached penalty calculator once
    pen_calc = _PenaltyCalculator(host_mol, guest_mol, removeHs=removeHs,
                                  host_guest_rm=host_guest_rm, guest_guest_rm=guest_guest_rm)

    opt = OptimizeConformation(
        guest_mol=guest_mol,
        host_coords=host_coords,
        n_particles=1,
        pi=pi,
        mu=mu,
        sigma=sigma,
        removeHs=removeHs,
        dist_threshold=dist_threshold,
        seed=seed,
        host_mol=host_mol,
        canonicalize_guest=canonicalize_guest,
        # [SPEED-UP] pass cached penalty calc
        penalty_calc=pen_calc,
    )

    center_of_mass = np.mean(host_coords, axis=0)

    guest_length = opt.get_adaptive_bounds(50)
    max_coord = center_of_mass + guest_length
    min_coord = center_of_mass - guest_length

    max_bound = np.concatenate([[np.pi] * 3, max_coord, [np.pi] * len(opt.rotable_bonds)], axis=0)
    min_bound = np.concatenate([[-np.pi] * 3, min_coord, [-np.pi] * len(opt.rotable_bonds)], axis=0)

    bounds = Bounds(lb=min_bound, ub=max_bound, keep_feasible=True)

    print(f'Number of Optimized Parameter: {len(max_bound)}')
    print(f'Number of Rotatable Bonds: {len(opt.rotable_bonds)}')

    starting_mol = opt.mol

    optimization_history = []

    def callback(xk, convergence):
        optimization_history.append(xk)

    revise_popsize = kwargs.get('revise_popsize', False)
    popsize = kwargs.get('popsize', 15)
    kwargs.pop('popsize', None)
    kwargs.pop('revise_popsize', None)

    if revise_popsize:
        new_popsize = int(popsize + 15 * np.log(len(opt.rotable_bonds) + 1))
    else:
        new_popsize = int(np.ceil(popsize / (len(opt.rotable_bonds) + 6)))

    fixed_score_conformation = partial(opt.score_conformation,
                                       host_guest_rm=host_guest_rm,
                                       guest_guest_rm=guest_guest_rm)

    result = differential_evolution(
        fixed_score_conformation,
        bounds=bounds,
        callback=callback,
        seed=seed,
        popsize=new_popsize,
        **kwargs,
    )

    opt_mol = apply_changes(starting_mol, result['x'], opt.rotable_bonds)
    if savepath:
        np.savetxt(savepath, optimization_history)

    docking_result = {
        'num_atoms': opt_mol.GetNumHeavyAtoms(),
        'num_rotbonds': len(opt.rotable_bonds),
        'rotbonds': opt.rotable_bonds,
        'success': result['success'],
        'fun': result['fun'],
    }

    return opt_mol, starting_mol, docking_result

class OptimizeConformation:
    def __init__(self, guest_mol, host_coords, n_particles, pi, mu, sigma, removeHs=True, save_molecules=False,
                 dist_threshold=5, seed=1000, host_mol=None, canonicalize_guest=True, penalty_calc: _PenaltyCalculator = None):
        super(OptimizeConformation, self).__init__()
        if seed:
            np.random.seed(seed)
        self.seed = seed
        self.removeHs = removeHs
        self.opt_mols = []
        self.n_particles = n_particles
        self.rotable_bonds = get_torsions([guest_mol])
        self.save_molecules = save_molecules
        self.dist_threshold = dist_threshold
        self.init_guest_mol = guest_mol
        self.canonicalize_guest = canonicalize_guest
        self.mol = get_random_conformation(guest_mol, rotable_bonds=self.rotable_bonds, seed=seed,
                                           canonicalize=self.canonicalize_guest)

        # Host, MDN params
        self.hostCoords = np.stack([host_coords for _ in range(n_particles)]).astype(np.float64)
        self.pi = np.concatenate([pi for _ in range(n_particles)], axis=0)
        self.sigma = np.concatenate([sigma for _ in range(n_particles)], axis=0)
        self.mu = np.concatenate([mu for _ in range(n_particles)], axis=0)

        # Hydrogen masking
        self.noHidx = [idx for idx in range(self.mol.GetNumAtoms()) if self.mol.GetAtomWithIdx(idx).GetAtomicNum() != 1]
        self.host_mol = host_mol

        self._penalty = penalty_calc

    def score_conformation(self, values, host_guest_rm=3., guest_guest_rm=1.5):
        if len(values.shape) < 2:
            values = np.expand_dims(values, axis=0)

        # Build a working copy once
        m = copy.copy(self.mol)

        # apply rotations (dihedral rotation)
        conf = m.GetConformer()
        base = 6
        for r, tors in enumerate(self.rotable_bonds):
            rdMolTransforms.SetDihedralRad(conf, tors[0], tors[1], tors[2], tors[3], values[0, base + r])

        rdMolTransforms.TransformConformer(conf, GetTransformationMatrix(values[0, :6]))

        # Guest coords (w/ or w/o H)
        if self.removeHs:
            guestCoords = np.array(m.GetConformer().GetPositions()[self.noHidx], dtype=np.float64)
        else:
            guestCoords = np.array(m.GetConformer().GetPositions(), dtype=np.float64)

        # Distances to host
        dist = distance.cdist(guestCoords.reshape(-1, 3), self.hostCoords.reshape(-1, 3)).flatten().reshape(-1, 1)

        prob = calculate_probablity_fast(self.pi, self.sigma, self.mu, dist)
        if self.dist_threshold:
            # indices where dist > threshold (using torch index pattern from original)
            mask_idx = np.where(dist.ravel() > float(self.dist_threshold))[0]
            if mask_idx.size:
                prob[mask_idx] = 0.0
        prob = np.sum(prob, axis=0)

        if self.save_molecules:
            self.opt_mols.append(m)

        if self._penalty is not None:
            penalty_host_guest, penalty_inner_guest = self._penalty.penalty(m)
        else:
            # Fallback to original path if not provided (rare)
            penalty_host_guest, penalty_inner_guest = calculate_penalty_all(
                self.host_mol, m, host_guest_rm=host_guest_rm, guest_guest_rm=guest_guest_rm, scale_factor=1
            )

        score = -prob + penalty_host_guest + penalty_inner_guest
        return score

    def get_adaptive_bounds(self, sel_conformers=50):
        copy_mol = copy.deepcopy(self.init_guest_mol)
        mol_length = []
        try:
            cids = AllChem.EmbedMultipleConfs(copy_mol, numConfs=1000, randomSeed=self.seed, numThreads=0)
            ff = AllChem.MMFFGetMoleculeForceField(copy_mol, AllChem.MMFFGetMoleculeProperties(copy_mol))
            energies = []
            for cid in cids:
                ff.Initialize()
                ff.Minimize(maxIts=200)
                energy = ff.CalcEnergy()
                energies.append(energy)

            # Sort conformers by energy.
            sorted_indices = sorted(range(len(energies)), key=lambda k: energies[k])
            top_indices = sorted_indices[:sel_conformers]
            top_conformers = [copy_mol.GetConformer(cid) for cid in [cids[i] for i in top_indices]]
            for i, conf in enumerate(top_conformers):
                conformer = Chem.Mol(copy_mol, True)
                conformer.AddConformer(copy_mol.GetConformer(id=i), assignId=True)
                coords = np.array(conformer.GetConformer().GetPositions())
                host_guest_mol_dist = distance.cdist(coords, coords)
                mol_length.append(np.max(host_guest_mol_dist))
        except Exception as e:
            print(e)
            coords = np.array(self.init_guest_mol.GetConformer().GetPositions())
            host_guest_mol_dist = distance.cdist(coords, coords)
            mol_length.append(np.max(host_guest_mol_dist))
        if len(mol_length) == 0:
            raise ValueError('Error Conformers Generation')
        return max(mol_length).item()


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def GetTransformationMatrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    # Same matrix as original, computed with fewer temporaries
    return np.array([
        [cz * cy, (cz * sy * sx) - (sz * cx), (cz * sy * cx) + (sz * sx), disp_x],
        [sz * cy, (sz * sy * sx) + (cz * cx), (sz * sy * cx) - (cz * sx), disp_y],
        [-sy,      cy * sx,                     cy * cx,                     disp_z],
        [0, 0, 0, 1],
    ], dtype=np.double)


def calculate_probablity(pi, sigma, mu, y):  # kept for compatibility
    return calculate_probablity_fast(pi, sigma, mu, y)


def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.copy(mol)
    conf = opt_mol.GetConformer()
    for r, tors in enumerate(rotable_bonds):
        rdMolTransforms.SetDihedralRad(conf, tors[0], tors[1], tors[2], tors[3], values[6 + r])
    rdMolTransforms.TransformConformer(conf, GetTransformationMatrix(values[:6]))
    return opt_mol


def get_torsions(mol_list):
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2, idx3 = match[0], match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if b1.GetIdx() == bond.GetIdx():
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    if idx4 == idx1:  # skip 3-rings
                        continue
                    if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1):
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append((idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append((idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break
        atom_counter += m.GetNumAtoms()
    return torsionList


def get_random_conformation(mol, rotable_bonds=None, seed=None, canonicalize=True):
    if isinstance(mol, Chem.Mol):
        try:
            mol.GetConformer()
        except Exception:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
    else:
        raise Exception('mol should be an RDKIT molecule')
    if seed:
        np.random.seed(seed)
    if rotable_bonds is None:
        rotable_bonds = get_torsions([mol])
    # Fix small bug in original random vector shape
    rand_vec = np.random.rand(len(rotable_bonds) + 6) * 10.0
    new_conf = apply_changes(mol, rand_vec, rotable_bonds)
    if canonicalize:
        Chem.rdMolTransforms.CanonicalizeConformer(new_conf.GetConformer())
    return new_conf



def show_opt_conformations(guest, prefix, save_path, numpy_txt, seed=None, n_particles=1, canonicalize=True):
    tors_list = get_torsions([guest])
    mol = get_random_conformation(guest, tors_list, seed=seed, canonicalize=canonicalize)

    optimization_history = np.loadtxt(numpy_txt)
    opt_tuple = [tuple(row) for row in optimization_history]
    unique_dict = OrderedDict.fromkeys(opt_tuple)
    unique_opt = np.array(list(unique_dict.keys()))

    os.makedirs(save_path, exist_ok=True)
    for i, opt in enumerate(unique_opt):
        if len(opt.shape) < 2:
            opt = np.expand_dims(opt, axis=0)
        m = copy.copy(mol)
        conf = m.GetConformer()
        for r, tors in enumerate(tors_list):
            rdMolTransforms.SetDihedralRad(conf, tors[0], tors[1], tors[2], tors[3], opt[0, 6 + r])
        rdMolTransforms.TransformConformer(conf, GetTransformationMatrix(opt[0, :6]))
        AllChem.MolToMolFile(m, os.path.join(save_path, f'{prefix}_opt_{i}.mol'))
    AllChem.MolToMolFile(mol, os.path.join(save_path, f'{prefix}_init.mol'))
    return None


def show_opt_process(prefix, basedir, removeHs=False, save_path=None):
    os.chdir(basedir)
    guest_conf_list = [i for i in os.listdir(basedir) if ('_opt' in i and i.endswith('.mol'))]
    init_mol = Chem.MolFromMolFile(f'{prefix}_init.mol', removeHs=removeHs, sanitize=True)
    init_positions = init_mol.GetConformer().GetPositions()
    xyz_blocks = [f'{init_mol.GetNumAtoms()}\n', ' generated by shot_opt_process\n']
    for j, atom in enumerate(init_mol.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y, z = init_positions[j]
        xyz_blocks.append(f' {symbol:>2} {x:>15.4f} {y:>15.4f} {z:>15.4f}\n')

    for i in range(len(guest_conf_list)):
        guest_file = f'{prefix}_opt_{i}.mol'
        guest = Chem.MolFromMolFile(guest_file, removeHs=removeHs, sanitize=True)
        xyz_blocks.append(f'{guest.GetNumAtoms()}\n')
        xyz_blocks.append(' generated by shot_opt_process\n')
        positions = guest.GetConformer().GetPositions()
        for j, atom in enumerate(guest.GetAtoms()):
            symbol = atom.GetSymbol()
            x, y, z = positions[j]
            xyz_blocks.append(f' {symbol:>2} {x:>15.4f} {y:>15.4f} {z:>15.4f}\n')
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.writelines(xyz_blocks)
