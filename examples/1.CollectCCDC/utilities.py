import warnings

import numpy as np
import os
import re
import shutil
from sugar.molecule import HostMolecule
from ccdc import io
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter, OrderedDict
from UFF4MOF_constants import UFF4MOF_atom_parameters
import subprocess

metal_element = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                 'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th',
                 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh',
                 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv']

# {Symbol:[(oxidation_state1, geometry1), (oxidation_state2, geometry2), ...]}
metal_oxidation_state = {'Li': [(2, 3)], 'Be': [(2, 3)], 'Na': [(2, 3), (2, 4)], 'Mg': [(2, 3), (3, 6)],
                         'Al': [(2, 3), (3, 6)], 'K': [(2, 3), (2, 4)], 'Ca': [(2, 3), (1, 1), (2, 6)],
                         'Sc': [(3, 3), (3, 6)], 'Ti': [(2, 4), (4, 6), (4, 8), (4, 3)],
                         'V': [(5, 3), (2, 4), (2, 3), (3, 6)], 'Cr': [(2, 4), (3, 6)],
                         'Mn': [(2, 4), (1, 1), (2, 3), (2, 6), (4, 8), (3, 6)],
                         'Fe': [(2, 4), (1, 1), (2, 3), (2, 6), (3, 6)], 'Co': [(2, 3), (2, 4), (2, 6), (1, 1)],
                         'Ni': [(2, 4)], 'Cu': [(2, 4), (1, 1), (2, 3), (2, 2), (1, 3)],
                         'Zn': [(2, 3), (2, 4), (2, 2), (1, 1)], 'Ga': [(2, 3), (3, 3), (3, 6)], 'Ge': [], 'Rb': [],
                         'Sr': [(2, 6), (4, 8)], 'Y': [(3, 3), (4, 8), (3, 6)], 'Zr': [(4, 8), (4, 3)],
                         'Nb': [(5, 3), (4, 8)], 'Mo': [(2, 4), (2, 3), (4, 8), (6, 6), (6, 3)], 'Tc': [(5, 6)],
                         'Ru': [(2, 4), (2, 6)], 'Rh': [(3, 6)], 'Pd': [(2, 4), (3, 6)],
                         'Ag': [(2, 3), (1, 1), (2, 4), (2, 2)], 'Cd': [(2, 3), (1, 1), (2, 4), (4, 8)],
                         'In': [(2, 3), (3, 3), (4, 8), (3, 6)], 'Sn': [], 'Sb': [(3, 3)], 'Cs': [],
                         'Ba': [(2, 3), (2, 6)],
                         'La': [(3, 3), (4, 8)], 'Ce': [(4, 8), (3, 6)], 'Pr': [(4, 8), (3, 6)], 'Nd': [(4, 8), (3, 6)],
                         'Pm': [(3, 6)], 'Sm': [(4, 8), (3, 6)], 'Eu': [(4, 8), (3, 6)], 'Gd': [(4, 8), (3, 6)],
                         'Tb': [(4, 8), (3, 6)], 'Dy': [(4, 8), (3, 6)], 'Ho': [(4, 8), (3, 6)], 'Er': [(4, 8), (3, 6)],
                         'Tm': [(4, 8), (3, 6)], 'Yb': [(4, 8), (3, 6)], 'Lu': [(1, 1), (4, 8), (3, 6)],
                         'Hf': [(4, 8), (4, 3)], 'Ta': [(5, 3)], 'W': [(2, 4), (4, 3), (4, 8), (6, 6), (6, 3)],
                         'Re': [(3, 6), (5, 6), (7, 3)], 'Os': [(6, 6), (2, 4)], 'Ir': [(3, 6)], 'Pt': [(2, 4)],
                         'Au': [(1, 1), (3, 4)], 'Hg': [(2, 3), (2, 1)], 'Tl': [(3, 3)], 'Pb': [(2, 4)], 'Bi': [(3, 3)],
                         'Po': [(2, 3)], 'Fr': [], 'Ra': [(2, 6)], 'Ac': [(3, 6)], 'Th': [(4, 6)], 'Pa': [(4, 6)],
                         'U': [(4, 6), (4, 8), (3, 6)], 'Np': [(4, 6)], 'Pu': [(4, 6)], 'Am': [(4, 6)], 'Cm': [(3, 6)],
                         'Bk': [(3, 6)], 'Cf': [(3, 6)], 'Es': [(3, 6)], 'Fm': [(3, 6)], 'Md': [(3, 6)], 'No': [(3, 6)],
                         'Lr': [(3, 6)], 'Rf': [], 'Db': [], 'Sg': [], 'Bh': [], 'Hs': [], 'Mt': [], 'Ds': [], 'Rg': [],
                         'Cn': [], 'Uut': [], 'Fl': [], 'Uup': [], 'Lv': []}

valence_rules = {
    'H': 1,
    'C': 4,
    # 'N': 3,
    # 'O': 2,
    'F': 1,
    'Cl': 1,
    'Br': 1,
    'I': 1
}


def write_components(components, basedir, code):
    code = code.lower()
    for i, comp in enumerate(components):
        outpath = f'{basedir}/{code}_{i}.mol'
        with io.MoleculeWriter(outpath) as mol_writer:
            mol_writer.write(comp)


# def identify_host_guest(components, basedir, code, threshold=2.):
#     """
#     Identify host and guest mol, write out host and guest mol file respectively.
#
#     Note:
#         For cases which contain only 2 components.
#     """
#     code = code.lower()
#     os.chdir(basedir)
#     formula_list = [comp.formula for comp in components]
#     num_components = len(formula_list)
#     pore_diameter_list = []
#     centroid_list = []
#     molfiles = [f'{code}_{i}.mol' for i in range(num_components)]
#
#     for molfile in molfiles:
#         sugar_mol = HostMolecule.init_from_mol_file(molfile)
#         rdmol = Chem.MolFromMolFile(molfile, removeHs=False)
#         has_hydrogen = any(atom.GetAtomicNum() == 1.preprocessing.generate_structural_data for atom in rdmol.GetAtoms())
#         has_bond = rdmol.GetNumBonds() > 0
#         if not has_bond or not has_hydrogen:
#             remove_exception_files_all(components, basedir, code)
#             return None
#         centroid_list.append(sugar_mol.get_centroid_remove_h())
#         try:
#             pore_diameter = sugar_mol.cal_pore_diameter_opt()
#             pore_diameter_list.append(pore_diameter)
#             # print(f'{molfile}: {pore_diameter}')
#         except:
#             pore_diameter_list.append(0)
#             # print(f'{molfile} does not have a pore.')
#
#     formula_count = Counter(formula_list)
#     formula0, formula1, pore0, pore1 = list(formula_count.keys())[0], list(formula_count.keys())[1.preprocessing.generate_structural_data], 0, 0
#     for i, formula in enumerate(formula_list):
#         if formula == formula0:
#             pore0 += pore_diameter_list[i]
#         elif formula == formula1:
#             pore1 += pore_diameter_list[i]
#     pore0 /= formula_count[formula0]
#     pore1 /= formula_count[formula1]
#
#     # Identify Host or Guest based on the pore diameter.
#     if pore0 < threshold and pore1 < threshold:
#         remove_exception_files_all(components, basedir, code)
#         return None
#     elif pore1 > threshold and pore1 > pore0:
#         is_host_or_guest = ['Host' if formula == formula1 else 'Guest' for formula in formula_list]
#         closest_host_index, closest_guest_index, min_dist = identify_index(is_host_or_guest, centroid_list)
#         if min_dist > pore1 / 2:
#             remove_exception_files_all(components, basedir, code)
#         else:
#             remove_exception_files_partial(components, basedir, code, (closest_host_index, closest_guest_index))
#             get_rename_host_guest_files(closest_host_index, closest_guest_index, code)
#         return None
#     elif pore0 > threshold and pore0 > pore1:
#         is_host_or_guest = ['Host' if formula == formula0 else 'Guest' for formula in formula_list]
#         closest_host_index, closest_guest_index, min_dist = identify_index(is_host_or_guest, centroid_list)
#         if min_dist > pore0 / 2:
#             remove_exception_files_all(components, basedir, code)
#         else:
#             remove_exception_files_partial(components, basedir, code, (closest_host_index, closest_guest_index))
#             get_rename_host_guest_files(closest_host_index, closest_guest_index, code)
#         return None
#     else:
#         remove_exception_files_all(components, basedir, code)
#         return None


def identify_host_guest_multiple(components, basedir, code, threshold=2.,dist_threshold=2):
    """
    Identify host and guest mol, write out host and guest mol file respectively.

    Note:
        For cases which contain more than 2 components.
    """
    code = code.lower()
    os.chdir(basedir)
    formula_list = [comp.formula for comp in components]
    num_components = len(formula_list)
    pore_diameter_list = []
    centroid_list = []
    molfiles = [f'{code}_{i}.mol' for i in range(num_components)]
    filtered_formula_list = []
    filtered_molfile_list = []

    # print(Counter(formula_list))

    for i, molfile in enumerate(molfiles):
        rdkit_mol = Chem.MolFromMolFile(molfile, sanitize=False, removeHs=False)
        try:
            if any(atom.GetSymbol() in metal_element for atom in rdkit_mol.GetAtoms()):
                ox_dict, fromatoms = get_metal_oxidation_state(rdkit_mol)
                revised_mol = revise_metal_mol(molfile, metal=ox_dict, from_atoms=fromatoms,
                                               sanitize=True, kekulize=False, reset_charge=True, reset_N=True)
                Chem.MolToMolFile(revised_mol, molfile)
                sugar_mol = HostMolecule.init_from_rdkit(revised_mol)
            else:
                sugar_mol = HostMolecule.init_from_mol_file(molfile)
            rdmol = Chem.MolFromMolFile(molfile, removeHs=False)
            has_hydrogen = any(atom.GetAtomicNum() == 1 for atom in rdmol.GetAtoms())
            has_bond = rdmol.GetNumBonds() > 0
        except:
            os.remove(os.path.join(basedir, f'{code}_{i}.mol'))
            continue

        # If the component does not contain H or chemical bonds,
        # remove it as it is not a potential guest molecule.
        if not has_bond or not has_hydrogen:
            # if not has_bond:
            os.remove(os.path.join(basedir, f'{code}_{i}.mol'))
            continue
        centroid_list.append(sugar_mol.get_centroid_remove_h())
        try:
            pore_diameter = sugar_mol.cal_pore_diameter_opt()
            pore_diameter_list.append(pore_diameter)
            # print(f'{molfile}: {pore_diameter}')
        except:
            pore_diameter_list.append(0)
            # print(f'{molfile} does not have a pore.')
        filtered_formula_list.append(formula_list[i])
        filtered_molfile_list.append(f'{code}_{i}.mol')

    formula_count = Counter(filtered_formula_list)
    unique_formulas = {key: 0 for key in formula_count.keys()}
    for i, formula in enumerate(filtered_formula_list):
        unique_formulas[formula] += pore_diameter_list[i]
    for key in unique_formulas.keys():
        unique_formulas[key] /= formula_count[key]

    # Identify Host or Guest based on the pore diameter.
    if all(value < threshold for value in unique_formulas.values()):
        remove_exception_files_all(components, basedir, code)
        return None
    else:
        # Set as "Host" if the pore size is greater than the threshold,
        # as guests typically do not have a pore.
        is_host_or_guest_dict = {'Host': [key for key, value in unique_formulas.items() if value > threshold],
                                 'Guest': [key for key, value in unique_formulas.items() if value <= threshold]}
        is_host_or_guest = ['Host' if formula in is_host_or_guest_dict['Host'] else 'Guest' for formula in
                            filtered_formula_list]
        closest_host_index, closest_guest_index, min_dist = identify_index(is_host_or_guest, centroid_list)

        # If the distance between the centroids is greater than half of the host pore diameter,
        # it means the guest is not encapsulated in the host and can be removed.
        # However, the {dist_threshold} can be smaller than 2 to take cavity error into account.
        if min_dist > unique_formulas[filtered_formula_list[closest_host_index]] / dist_threshold:
            # print(f'Too far of host and guest, distance:{min_dist}')
            remove_exception_files_all(components, basedir, code)
        else:
            # print(
            #     f'Host: {filtered_molfile_list[closest_host_index]}, Guest: {filtered_molfile_list[closest_guest_index]}')
            # print(f'MinDistance: {min_dist}')
            host_index = int(re.search(r'_(\d+)\.', filtered_molfile_list[closest_host_index]).group(1))
            guest_index = int(re.search(r'_(\d+)\.', filtered_molfile_list[closest_guest_index]).group(1))
            remove_exception_files_partial(components, basedir, code, (host_index, guest_index))
            get_rename_host_guest_files(host_index, guest_index, code)
        return None


def sort_key(file_name):
    return int(file_name.split('_')[-1].split('.')[0])


def remove_exception_files_all(components, basedir, code):
    code = code.lower()
    for i, comp in enumerate(components):
        outpath = f'{basedir}/{code}_{i}.mol'
        try:
            os.remove(outpath)
        except:
            pass


def remove_exception_files_partial(components, basedir, code, index):
    code = code.lower()
    for i, comp in enumerate(components):
        if i not in index:
            outpath = f'{basedir}/{code}_{i}.mol'
            try:
                os.remove(outpath)
            except:
                continue


def identify_index(index_list, centroid_list):
    host_indices = [i for i, val in enumerate(index_list) if val == 'Host']
    guest_indices = [i for i, val in enumerate(index_list) if val == 'Guest']
    distances = {(i, j): np.linalg.norm(centroid_list[i] - centroid_list[j]) for i in host_indices for j in
                 guest_indices}
    # print(distances)
    min_distance_key = min(distances, key=distances.get)
    closest_host_index, closest_guest_index = min_distance_key
    return closest_host_index, closest_guest_index, min(distances.values())


def get_rename_host_guest_files(host_index, guest_index, code):
    code = code.lower()
    shutil.move(f'{code}_{host_index}.mol', f'{code}_host.mol')
    shutil.move(f'{code}_{guest_index}.mol', f'{code}_guest.mol')
    shutil.move(f'{code}_host.mol', f'{code}_1.mol')
    shutil.move(f'{code}_guest.mol', f'{code}_2.mol')


def move_need_check_file(components, basedir, check_dir, code):
    code = code.lower()
    for i, comp in enumerate(components):
        inpath = f'{basedir}/{code}_{i}.mol'
        outpath = f'{check_dir}/{code}_{i}.mol'
        try:
            shutil.move(inpath, outpath)
        except:
            pass


def run_command(command):
    """
    Return the state of command line.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def convert_molfile_format(obabel, pdb_file, mol_file):
    output, error = run_command(f"{obabel} {pdb_file} -O {mol_file}")
    return output, error


def revise_metal_mol(
        file_path,
        metal=None,
        from_atoms=(7, 8),
        sanitize=True,
        kekulize=False,
        reset_charge=False,
        reset_N=True
):
    """
    Preprocessing molecule with wrong bond type and formal charge.

    For example, a metal cage structure which is taken from .cif file, but there's no charge information about the metal
    atoms and the bond type between metal atoms and ligands are all single bond.
    Through this function, you can specify the correct formal charge for the metal atoms.

    Parameters
    ------------
    file_path : 'str'
        Path of the molecule file.
    metal: 'dict'
        In the form of {'metal1': new_formal_charge, 'metal2': new_formal_charge, ...}
    from_atoms: 'tuple'
        Replaces bonds between metals and atoms with atomic numbers in from_atoms.
    sanitize: 'bool'
        Whether sanitize the molecule or not. Default to be 'True'.
        If your molecule can not to be kekulized, you should set this parameter to be 'False'.
    kekulize: 'bool'
        Whether to kukulize the molecule or not. Default to be 'True'.
    reset_charge: 'bool'
        Whether reset the formal charge of the molecule or not. Default to be 'True'.
        After correcting the bond type or formal charge, the formal charge for non-metal atoms maybe wrong,
        so you should set this parameter to be 'True' to reset the formal charge = 0.
    reset_N: 'bool'
        Whether reset the formal charge of the N atom. Default to be 'True'.

    Adapt from rdkit Cookbook:
            https://github.com/rdkit/rdkit/blob/8f4d4a624c69df297552cabb2d912b2ac7c39e84/Docs/Book/Cookbook.rst#L2174
    """

    def is_transition_metal(at):
        n = at.GetAtomicNum()
        return (n >= 22 and n <= 29) or (n >= 40 and n <= 47) or (n >= 72 and n <= 79)

    if metal is None or not isinstance(metal, dict):
        raise ValueError(f"Metal input should be a dict.")
    temp_mol = AllChem.MolFromMolFile(file_path, sanitize=False, removeHs=False)
    # temp_mol = AllChem.AddHs(temp_mol)
    pt = Chem.GetPeriodicTable()

    rw_mol = Chem.RWMol(temp_mol)
    rw_mol.UpdatePropertyCache(strict=False)
    metals = [at for at in rw_mol.GetAtoms() if is_transition_metal(at)]

    for m in metals:
        m.SetFormalCharge(metal[m.GetSymbol()])
        for nbr in m.GetNeighbors():
            if nbr.GetAtomicNum() in from_atoms and \
                    nbr.GetExplicitValence() > pt.GetDefaultValence(nbr.GetAtomicNum()) and \
                    rw_mol.GetBondBetweenAtoms(nbr.GetIdx(), m.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rw_mol.RemoveBond(nbr.GetIdx(), m.GetIdx())
                rw_mol.AddBond(nbr.GetIdx(), m.GetIdx(), Chem.BondType.DATIVE)
    if reset_charge:
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() not in metal_element:
                atom.SetFormalCharge(0)
    if reset_N:
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == 'N':
                n_connections = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
                if n_connections == 4:
                    atom.SetFormalCharge(1)
    if sanitize:
        AllChem.SanitizeMol(rw_mol)
    if kekulize:
        AllChem.Kekulize(rw_mol)
    return rw_mol


def get_uff4mof_atomtypes():
    return list(UFF4MOF_atom_parameters.keys())


def get_metal_atoms(mol):
    """
    mol = Chem.MolFromMolFile(molpath, sanitize=False, removeHs=False)
    """
    metal_atoms = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() in metal_element]
    return list(set(metal_atoms))


def _get_metal_neighbors(mol):
    neighbors = [neighbor.GetAtomicNum() for atom in mol.GetAtoms() for neighbor in atom.GetNeighbors() if
                 atom.GetSymbol() in metal_element]
    return tuple(set(neighbors))


def _get_metal_conn(mol):
    metal_atoms = get_metal_atoms(mol)
    metal_conn = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_atoms:
            neighbors = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
            metal_conn.append((atom.GetSymbol(), len(neighbors)))
    return metal_conn


def get_metal_oxidation_state(mol):
    """
    mol = Chem.MolFromMolFile(molpath, sanitize=False, removeHs=False)
    """
    metal_atoms = get_metal_atoms(mol)
    if len(metal_atoms) > 1:
        raise ValueError(f'The current version does not support '
                         f'molecules having two or more different metals.')
    elif len(metal_atoms) == 0:
        raise ValueError(f'No metal atom was found.')
    else:
        metal_conn = _get_metal_conn(mol)
        if len(set(metal_conn)) > 1:
            raise ValueError(f'Different geometries were found for element {metal_conn[0][0]}')
        metal_conn = list(set(metal_conn))[0]
        geometry_hit = 0
        state = 0
        for value in metal_oxidation_state[metal_conn[0]]:
            if metal_conn[1] == value[1]:
                state = value[0]
                geometry_hit += 1

        if geometry_hit > 0:
            warnings.warn(
                f'Element {metal_conn[0]} has multiple oxidation states in '
                f'geometry {metal_conn[1]}, currently os being {state}')
        elif geometry_hit == 0:
            if len(metal_oxidation_state[metal_conn[0]]) == 1:
                state = metal_oxidation_state[metal_conn[0]][0][0]
            else:
                states = [i[0] for i in metal_oxidation_state[metal_conn[0]]]
                counter = Counter(states)
                state = counter.most_common(1)[0][0]
            warnings.warn(
                f'Element {metal_conn[0]} does not have a '
                f'matching geometry: {metal_conn[1]} in UFF4MOF')
        return {metal_conn[0]: state}, _get_metal_neighbors(mol)


def check_mol_structure(mol):
    atom_valence = {}

    for atom in mol.GetAtoms():
        bonds = atom.GetBonds()
        total_bond_count = sum(bond.GetBondTypeAsDouble() for bond in bonds)
        atom_symbol = atom.GetSymbol()
        atom_idx = atom.GetIdx()
        atom_valence[atom_idx] = (atom_symbol, total_bond_count)

    invalid_atoms = []

    for idx, (symbol, valence) in atom_valence.items():
        if symbol in valence_rules:
            if valence != valence_rules[symbol]:
                invalid_atoms.append((idx, symbol, valence))
    return invalid_atoms


def format_molecular_formula(formula):
    """
    To compare the molecular formula in the mol file with the one provided by CCDC,
    if they are not consistent, it indicates an error in the structure.
    """
    formatted_formula = ''
    pt = Chem.GetPeriodicTable()
    elements = [pt.GetElementSymbol(i) for i in range(1, 119)]

    for element in elements:
        count = re.search(rf'{element}\d*', formula)
        if count:
            if re.search(rf'{element}$', count.group()):
                formatted_formula += count.group() + '1.preprocessing.generate_structural_data' + ' '
            else:
                formatted_formula += count.group() + ' '

    return formatted_formula.strip()


def get_pore_windows(sugarmol):
    """
    sugarmol = HostMolecule.init_from_mol_file(molfile)

    ###BUG in window calculation
    """
    pore_diameter = sugarmol.cal_pore_diameter_opt()
    # windows = sugarmol.cal_windows()
    # return round(pore_diameter, 4), windows
    return round(pore_diameter, 4), None

def get_com_distance(sugarmol1, sugarmol2):
    """
    sugarmol = HostMolecule.init_from_mol_file(molfile)
    """
    com1 = sugarmol1.get_centroid_remove_h()
    com2 = sugarmol2.get_centroid_remove_h()
    return np.linalg.norm(com1 - com2)


def get_torsions(rdmol):
    rdmol_list = [rdmol]
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in rdmol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3.use_deepdock-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                            or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList


def get_mol_formula(molfile):
    with open(molfile, 'r') as f:
        mol_contents = f.readlines()
    num_atoms = mol_contents[3][:3].rstrip()
    num_atoms = int(num_atoms)
    atom_contents = mol_contents[4:4 + num_atoms]
    elements = [i.split()[3] for i in atom_contents]
    counts = Counter(elements)
    formula = ''
    for k, v in counts.items():
        formula += k + str(v)
    return formula


def concat_mol_file(file1, file2):
    """
    Merge two .mol files into one file.
    """
    pass


class XTBCalculation:
    """
    Generate molden.input and convert it into mesh file.

    Example: (Can be seen in /examples/2.preprocessing)
    -------------------------------------------------
    1.preprocessing.generate_structural_data. Execute xtb to generate molden.input
        from DeepDockHostGuest.1.preprocessing.preprocessing.generate_mesh import *
        from tqdm import tqdm

        xtb = ESP('/path/to/xtb', '/path/to/Multiwfn)
        host_path = '/path/to/Host'
        guest_path = '/path/to/Guest'
        names = os.listdir(host_path)
        outdir = '/path/to/xtb_output'

        for name in tqdm(names):
            print(f"======Processing {name}======")
            host_files = [i for i in os.listdir(os.path.join(host_path, name)) if i.endswith('.xyz')]
            files_prefix = [i.rstrip('_host.xyz') for i in host_files]
            for prefix in tqdm(files_prefix):
                try:
                    host_xtb, _ = xtb.run_xtb(
                        os.path.join(host_path, name, f"{prefix}_host.xyz"),
                        outpath=outdir,
                        name=name)
                    guest_xtb, _ = xtb.run_xtb(
                        os.path.join(guest_path, name, f"{prefix}_guest.xyz"),
                        outpath=outdir,
                        name=name)
                except Exception as e:
                    print(e)

    """

    def __init__(self, xtb, multiwfn):
        self.xtb = xtb
        self.multiwfn = multiwfn
        self.multiwfn_settings = multiwfn.rstrip('Multiwfn') + 'settings.ini'

    def run_xtb(self, xyzfile, outpath=None, verbose=False):
        """
        :param xyzpath: path of the .xyz file
        :param outpath: outdir of the calculation
        """
        os.makedirs(outpath, exist_ok=True)
        # os.chdir(outpath)
        if 'molden.input' in os.listdir(outpath):
            if verbose:
                print(f"{xyzfile.split('/')[-1]} has been calculated!!")
            return 0, 0
        else:
            if xyzfile.split('/')[-1] not in os.listdir(outpath):
                shutil.copy(xyzfile, outpath)
            os.chdir(outpath)
            out, errors = run_command(f"{self.xtb} {os.path.join(outpath, xyzfile.split('/')[-1])} --molden --esp")
            return out, errors

    def run_molden_to_fch(self, molden_path, workdir, verbose=False):
        """
        Convert molden.input to molden.fch file.

        The molden file should be the default value "molden.input"

        :param molden_path: the path of molden.input
        :param workdir: the working directory of Multiwfn runs.
                        it should be the folder of molden_path.
        """
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        if 'molden.fch' in os.listdir(workdir):
            if verbose:
                print(f"{molden_path} has been converted!!")
            return 0, 0
        current_directory = os.getcwd()
        molden_to_fch_txt = ['100\n', '2\n', '7\n', '\n']
        with open(os.path.join(workdir, 'molden2fch.txt'), 'w') as f:
            f.writelines(molden_to_fch_txt)
        if 'molden.input' not in os.listdir(workdir):
            shutil.copy(molden_path, workdir)
        os.chdir(workdir)
        if 'settings.ini' not in os.listdir(workdir):
            shutil.copy(self.multiwfn_settings, workdir)
        out, errors = run_command(f"{self.multiwfn} {molden_path} < molden2fch.txt |tee "
                                  f"molden2fch.log")
        os.remove('molden2fch.txt')
        os.remove('settings.ini')
        os.chdir(current_directory)
        return out, errors

    def run_fch_to_esp(self, fchpath, workdir, grid_points_spacing=0.25, verbose=False):
        """
        The .fch file name shoule be the default value "molden.fch".
        """
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        if 'vtx.pdb' in os.listdir(workdir) and 'surf.cub' in os.listdir(workdir):
            if verbose:
                print(f"{fchpath} has been calculated to vtx.pdb!!")
            return 0, 0
        current_directory = os.getcwd()
        fch_to_esp_txt = ['12\n', '3.use_deepdock\n', f'{grid_points_spacing}\n', '0\n', '-2\n', '\n', '66\n', '\n']
        with open(os.path.join(workdir, 'fch2esp.txt'), 'w') as f:
            f.writelines(fch_to_esp_txt)
        if 'molden.fch' not in os.listdir(workdir):
            shutil.copy(fchpath, workdir)
        os.chdir(workdir)
        if 'settings.ini' not in os.listdir(workdir):
            shutil.copy(self.multiwfn_settings, workdir)
        out, errors = run_command(f"{self.multiwfn} {fchpath} < fch2esp.txt |tee fch2esp.log")
        os.remove('fch2esp.txt')
        os.remove('settings.ini')
        os.chdir(current_directory)
        return out, errors

    def run_fch_to_loba(self, fchpath, workdir, threshold=50, verbose=False):
        if 'fch2loba.log' in os.listdir(workdir):
            if verbose:
                print(f"{fchpath} has been calculated!!")
            return 0, 0
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        if 'molden.fch' not in os.listdir(workdir):
            shutil.copy(fchpath, workdir)
        os.chdir(workdir)
        if 'settings.ini' not in os.listdir(workdir):
            shutil.copy(self.multiwfn_settings, workdir)
        fch_to_loba_txt = ['19\n', '1.preprocessing.generate_structural_data\n', '8\n', '100\n', f'{threshold}\n']
        with open(os.path.join(workdir, 'fch2loba.txt'), 'w') as f:
            f.writelines(fch_to_loba_txt)
        command = f'{self.multiwfn} {fchpath} < fch2loba.txt |tee fch2loba.log'
        out, errors = run_command(command)
        os.remove('settings.ini')
        os.remove('fch2loba.txt')
        # os.chdir(current_directory)
        return out, errors

    @staticmethod
    def get_loba_content(workdir):
        log_path = os.path.join(workdir, 'fch2loba.log')
        if not os.path.exists(log_path):
            raise ValueError(f'{workdir} does not contain LOBA result!')
        with open(log_path, 'r') as f:
            log_contents = f.readlines()
        loba_contents = []
        for line in log_contents:
            if line.startswith(' Oxidation state'):
                loba_contents.append(line.rstrip())
            elif line.startswith('The sum of oxidation states:'):
                loba_contents.append(line.rstrip())
                break
        if len(loba_contents) <= 1:
            print('No Content with LOBA')
            return None, None
        total_oxidation_states = int(loba_contents[-1].split()[-1])
        oxidation_states_dict = {}
        for content in loba_contents[:-1]:
            index = content.split('(')[0].split()[-1]
            ele_symbol = content.split('(')[-1].split(')')[0]
            state = int(content.split()[-1])
            if ele_symbol in metal_element:
                oxidation_states_dict[index + '_' + ele_symbol] = state
        return oxidation_states_dict, total_oxidation_states
