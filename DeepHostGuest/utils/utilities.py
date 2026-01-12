import warnings

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.rdForceFieldHelpers import (MMFFHasAllMoleculeParams,
                                            MMFFGetMoleculeProperties, MMFFGetMoleculeForceField,
                                            UFFGetMoleculeForceField)
from rdkit.Chem.AllChem import (MMFFGetMoleculeForceField, EmbedMultipleConfs, UFFOptimizeMolecule,
                                MMFFGetMoleculeProperties, MMFFOptimizeMolecule)
from pydockrmsd.dockrmsd import PyDockRMSD
import pydockrmsd.hungarian as hungarian
import numpy as np
import os
import networkx as nx
import warnings
import subprocess
import shutil
import re
from sugar.molecule import HostMolecule
from sugar.atom import _metal_element


def kekulize_mol_file(molfile, kekulized_molfile):
    mol = Chem.MolFromMolFile(molfile, removeHs=False)
    AllChem.Kekulize(mol)
    Chem.MolToMolFile(mol, kekulized_molfile)
    return None


def run_command(command):
    """
    Return the state of command line.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def convert_molfile(obabel, molfile, mol2file):
    """

    """
    output, error = run_command(f"{obabel} {molfile} -O {mol2file}")
    return output, error


def revise_mol2_file(mol2file, revised_mol2file):
    with open(mol2file, 'r') as f:
        mol2file_contents = f.readlines()
    bond_flag = 0
    atom_attr_flag = 0
    for i, content in enumerate(mol2file_contents):
        if content.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            atom_attr_flag = i
        elif content.startswith('@<TRIPOS>BOND'):
            bond_flag = i
            break
    if atom_attr_flag == 0:
        with open(revised_mol2file, 'w') as f:
            f.writelines(mol2file_contents)
    else:
        revised_mol2file_contents = mol2file_contents[:atom_attr_flag] + mol2file_contents[bond_flag:]
        with open(revised_mol2file, 'w') as f:
            f.writelines(revised_mol2file_contents)
    return None


def calculate_rmsd(mol2file1, mol2file2, decimals=2):
    if not os.path.splitext(mol2file1)[1] == '.mol2' or not os.path.splitext(mol2file2)[1] == '.mol2':
        raise KeyError(f'Not a .mol2 file!')
    else:
        dr = PyDockRMSD(mol2file1, mol2file2)
        # print(dr.total_of_possible_mappings)
        # print(dr.optimal_mapping)
        # print(dr.error)
        # print(hungarian(mol2file1, mol2file2))
        return round(dr.rmsd, decimals)


def is_align(mol_file1, mol_file2):
    """
    Determine whether the atom order in optimized .mol file is consistent with the original structure.
    """
    mol1 = Chem.MolFromMolFile(mol_file1, removeHs=False, sanitize=True)
    mol2 = Chem.MolFromMolFile(mol_file2, removeHs=False, sanitize=True)

    atoms1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
    atoms2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]

    if atoms1 == atoms2:
        return True
    elif atoms1 != atoms2:
        return False


def calculate_euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def calculate_rmsd_from_file(mol_file1, mol_file2, removeHs=True):
    if is_align(mol_file1, mol_file2):
        mol1 = Chem.MolFromMolFile(mol_file1, removeHs=removeHs, sanitize=True)
        mol2 = Chem.MolFromMolFile(mol_file2, removeHs=removeHs, sanitize=True)

        atomCoords1 = mol1.GetConformer().GetPositions()
        atomCoords2 = mol2.GetConformer().GetPositions()

        differences = atomCoords1 - atomCoords2
        squared_diffs = np.square(differences)
        mean_squared_diffs = np.mean(squared_diffs)
        rmsd = np.sqrt(mean_squared_diffs)
        return rmsd

    else:
        warnings.warn(f'{mol_file1} does not align with {mol_file2}')
        return None


def calculate_mse_from_file(mol_file1, mol_file2, removeHs=True):
    mol1 = Chem.MolFromMolFile(mol_file1, removeHs=removeHs, sanitize=True)
    mol2 = Chem.MolFromMolFile(mol_file2, removeHs=removeHs, sanitize=True)

    atomCoords1 = mol1.GetConformer().GetPositions()
    atomCoords2 = mol2.GetConformer().GetPositions()

    differences = atomCoords1 - atomCoords2
    N = differences.shape[0]
    return np.sum(np.square(differences)) / N


def calculate_mae_from_file(mol_file1, mol_file2, removeHs=True):
    mol1 = Chem.MolFromMolFile(mol_file1, removeHs=removeHs, sanitize=True)
    mol2 = Chem.MolFromMolFile(mol_file2, removeHs=removeHs, sanitize=True)

    atomCoords1 = mol1.GetConformer().GetPositions()
    atomCoords2 = mol2.GetConformer().GetPositions()
    distances = [calculate_euclidean_distance(atomCoords1[i], atomCoords2[i]) for i in range(atomCoords1.shape[0])]
    return np.mean(distances)


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


def concat_ply_file(ply_file1, ply_file2, out_ply, len_header=15):
    """
    Concat 2 ply files into 1 ply file.

    """
    with open(ply_file1, 'r') as f:
        ply_contents1 = f.readlines()
    with open(ply_file2, 'r') as f:
        ply_contents2 = f.readlines()

    num_vertex_1, num_face_1 = 0, 0
    for content in ply_contents1:
        if content.startswith('element vertex'):
            num_vertex_1 = int(content.split()[-1])
        elif content.startswith('element face'):
            num_face_1 = int(content.split()[-1])

    num_vertex_2, num_face_2 = 0, 0
    for content in ply_contents2:
        if content.startswith('element vertex'):
            num_vertex_2 = int(content.split()[-1])
        elif content.startswith('element face'):
            num_face_2 = int(content.split()[-1])
    total_ply_contens = ['ply\n', 'format ascii 1.0\n', 'comment VTK generated PLY File\n',
                         'obj_info vtkPolyData points and polygons: vtk4.0\n',
                         f'element vertex {num_vertex_1 + num_vertex_2}\n',
                         'property float x\n', 'property float y\n', 'property float z\n', 'property float nx\n',
                         'property float ny\n', 'property float nz\n', 'property float ESP \n',
                         f'element face {num_face_1 + num_face_2}\n',
                         'property list uchar int vertex_indices\n', 'end_header\n']

    ply_vertex_contents1 = ply_contents1[len_header:len_header + num_vertex_1]
    ply_face_contents1 = ply_contents1[len_header + num_vertex_1:len_header + num_vertex_1 + num_face_1]
    ply_vertex_contents2 = ply_contents2[len_header:len_header + num_vertex_2]
    ply_face_contents2 = ply_contents2[len_header + num_vertex_2:len_header + num_vertex_2 + num_face_2]

    total_ply_contens.extend(ply_vertex_contents1)
    total_ply_contens.extend(ply_vertex_contents2)
    total_ply_contens.extend(ply_face_contents1)

    # Re-index the vertex indices in every face
    for face_content in ply_face_contents2:
        face_info = face_content.split()
        total_ply_contens.append(
            f'{face_info[0]} {int(face_info[1]) + num_vertex_1} '
            f'{int(face_info[2]) + num_vertex_1} {int(face_info[3]) + num_vertex_1}\n'
        )

    with open(out_ply, 'w') as f:
        f.writelines(total_ply_contens)
    return None


def concat_molecules(mol1_file, mol2_file):
    mol1 = Chem.MolFromMolFile(mol1_file, removeHs=False)
    mol2 = Chem.MolFromMolFile(mol2_file, removeHs=False)
    concat_mol = rdmolops.CombineMols(mol1, mol2)
    return concat_mol


def concat_molecules_withmol(mol1, mol2):
    concat_mol = rdmolops.CombineMols(mol1, mol2)
    return concat_mol


def run_xtb_opt(struc_path, calculation_dir, xtb_path, extra_section=None):
    command = f'cd {calculation_dir} && {xtb_path} {struc_path} --opt'
    if extra_section is not None:
        command += f' {extra_section}'
    xtb_opt_out, xtb_opt_err = run_command(command)
    return xtb_opt_out, xtb_opt_err


def choose_ff(mol, interfrag=True, mmff_variant='MMFF94s', must_use_uff=False):
    """
    interfrag: 控制RDKit力场是否考虑分子间相互作用
                True：考虑（主客体结构优化必须为True）
                False：不考虑
    """
    if must_use_uff:
        ff = UFFGetMoleculeForceField(mol, ignoreInterfragInteractions=(not interfrag))
        return ff
    if MMFFHasAllMoleculeParams(mol):
        properties = MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
        ff = MMFFGetMoleculeForceField(mol, properties, nonBondedThresh=100.0,
                                       ignoreInterfragInteractions=(not interfrag))
    else:
        ff = UFFGetMoleculeForceField(mol, ignoreInterfragInteractions=(not interfrag))
    return ff


def write_mol(mol, out_path):
    Chem.MolToMolFile(mol, out_path)
    return None


def optimize_total_complex(mol, out_path, max_iter=200, interfrag=True,
                           mmff_variant='MMFF94s', two_stage_relax=True,
                           must_use_uff=False):
    if two_stage_relax:
        ff1 = choose_ff(mol, interfrag, mmff_variant, must_use_uff)
        heavy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        for idx in heavy_atoms:
            ff1.AddFixedPoint(idx)
        ff1.Minimize(maxIts=max_iter)

    ff2 = choose_ff(mol, interfrag, mmff_variant, must_use_uff)
    ff2.Minimize(maxIts=max_iter)
    write_mol(mol, out_path)
    return mol


def optimize_guest_only(mol, out_path, host_atom_count=None, guest_atom_indices=None,
                        max_iter=200, interfrag=True, mmff_variant='MMFF94s', must_use_uff=False):
    if guest_atom_indices is not None:
        guest_index_set = list(guest_atom_indices)
    elif host_atom_count is not None:
        frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        guest_index_set = []

        if len(frags) != 2:
            raise ValueError(f'Number of Fragments in is not 2: Actually {len(frags)}')

        # 确定host的index
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx())
        # Bonds are the edges of the graph.
        for bond in mol.GetBonds():
            e = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            graph.add_edge(*e)
        mols = []
        for component in nx.connected_components(graph):
            atoms = [atom for atom in mol.GetAtoms() if atom.GetIdx() in component]
            if len(atoms) != host_atom_count:
                for atom in atoms:
                    guest_index_set.append(atom.GetIdx())

    ff = choose_ff(mol, interfrag, mmff_variant, must_use_uff)

    for atom in mol.GetAtoms():
        if atom.GetIdx() not in guest_index_set:
            ff.AddFixedPoint(atom.GetIdx())
    ff.Minimize(maxIts=max_iter)
    write_mol(mol, out_path)
    return mol


def generate_conformers(rdmol, sel_conformers=10, num_conformers=800,
                        random_seed=123):
    cids = EmbedMultipleConfs(
        rdmol, numConfs=num_conformers, randomSeed=random_seed, numThreads=0
    )

    energies = []
    props = MMFFGetMoleculeProperties(rdmol, mmffVariant="MMFF94s")
    for cid in cids:
        ff = MMFFGetMoleculeForceField(rdmol, props, confId=cid)
        ff.Minimize(maxIts=200)
        energies.append(ff.CalcEnergy())

    # 能量排序
    sorted_indices = sorted(range(len(energies)), key=lambda k: energies[k])
    top_indices = sorted_indices[:sel_conformers]

    mol_list = []
    for i in top_indices:
        cid = cids[i]
        conformer = Chem.Mol(rdmol, True)
        conf = rdmol.GetConformer(cid)

        conformer.RemoveAllConformers()
        conformer.AddConformer(Chem.Conformer(conf), assignId=True)

        mol_list.append(conformer)

    return mol_list


def get_disconnected_mols(mol):
    # Get the connected components as separate Mol objects
    fragments = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    return fragments


def get_xtb_free_energy(struc_file, calculation_dir, xtb_path,
                        shermo_main_folder, opt=False, olevel='normal',
                        charge=0, other_keywords=' --alpb water'):
    os.makedirs(calculation_dir, exist_ok=True)
    struc_file_basename = os.path.basename(struc_file)
    os.chdir(calculation_dir)
    if not os.path.exists(struc_file_basename):
        shutil.copy(struc_file, calculation_dir)

    if 'sp.out' and 'g98.out' in os.listdir(calculation_dir):
        print(f'{struc_file_basename} has been calculated')
        pass
    else:
        xtb_hess_command = f'{xtb_path} {calculation_dir}/{struc_file_basename} -c {charge}'
        if opt:
            xtb_hess_command += f' --ohess {olevel}'
            sp_struc_file = f'{calculation_dir}/xtbopt{os.path.splitext(struc_file)[1]}'
        else:
            xtb_hess_command += ' --hess'
            sp_struc_file = f'{calculation_dir}/{struc_file_basename}'
        if other_keywords:
            xtb_hess_command += other_keywords

        xtb_ohess_out, xtb_ohess_error = run_command(xtb_hess_command)

        xtb_sp_command = f'{xtb_path} {sp_struc_file} -c {charge}'
        if other_keywords:
            xtb_sp_command += other_keywords
        xtb_sp_command += ' > sp.out'
        xtb_sp_out, xtb_sp_error = run_command(xtb_sp_command)

    shermo_executable = os.path.join(shermo_main_folder, 'Shermo')
    shermo_settings = os.path.join(shermo_main_folder, 'settings.ini')
    shutil.copy(shermo_settings, calculation_dir)

    with open('sp.out', 'r') as f:
        sp_out = f.read()

    match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", sp_out)
    if match:
        total_energy = float(match.group(1))
    else:
        raise ValueError('No total energy found in sp.out')

    shermo_out, shermo_errors = run_command(f"{shermo_executable} g98.out -E {total_energy} |tee shermo.log")
    with open('shermo.log', 'r') as f:
        shermo_log = f.readlines()
    G_corr, H_corr, U_corr, G, H, U = 0, 0, 0, 0, 0, 0
    for line in shermo_log:
        if line.startswith(' Thermal correction to U'):
            U_corr = float(line.split()[8])
        elif line.startswith(' Thermal correction to G'):
            G_corr = float(line.split()[8])
        elif line.startswith(' Thermal correction to H'):
            H_corr = float(line.split()[8])
        elif line.startswith(' Sum of electronic energy and thermal correction to U:'):
            U = float(line.split()[9])
        elif line.startswith(' Sum of electronic energy and thermal correction to G:'):
            G = float(line.split()[9])
        elif line.startswith(' Sum of electronic energy and thermal correction to H'):
            H = float(line.split()[9])
    if 0 in (U_corr, G_corr, H_corr, G, H, U):
        raise ValueError('Missing Values for one of (G_corr, H_corr, U_corr, G, H, U)')
    convertor = 627.510
    return G_corr * convertor, H_corr * convertor, U_corr * convertor, G * convertor, H * convertor, U * convertor


def get_free_energy_with_shermo(shermo_main_folder, out_file, energy, sclZPE=1.0, sclheat=1.0, sclS=1.0,
                                ilowfreq='2', intpvib=100):
    shermo_executable = os.path.join(shermo_main_folder, 'Shermo')
    shermo_settings = os.path.join(shermo_main_folder, 'settings.ini')

    with open(shermo_settings, 'r') as f:
        shermo_contents = f.readlines()
    shermo_contents[0] = f'E= {energy}\n'
    shermo_contents[4] = f'sclZPE= {sclZPE}\n'
    shermo_contents[5] = f'sclheat= {sclheat}\n'
    shermo_contents[6] = f'sclS= {sclS}\n'
    shermo_contents[8] = f'ilowfreq= {ilowfreq}\n'
    shermo_contents[10] = f'intpvib= {intpvib}\n'
    with open(shermo_settings, 'w') as f:
        f.writelines(shermo_contents)

    shermo_out, shermo_errors = run_command(f"{shermo_executable} {out_file} |tee shermo.log")
    with open('shermo.log', 'r') as f:
        shermo_log = f.readlines()
    G_corr, H_corr, U_corr, G, H, U = 0, 0, 0, 0, 0, 0
    for line in shermo_log:
        if line.startswith(' Thermal correction to U'):
            U_corr = float(line.split()[8])
        elif line.startswith(' Thermal correction to G'):
            G_corr = float(line.split()[8])
        elif line.startswith(' Thermal correction to H'):
            H_corr = float(line.split()[8])
        elif line.startswith(' Sum of electronic energy and thermal correction to U:'):
            U = float(line.split()[9])
        elif line.startswith(' Sum of electronic energy and thermal correction to G:'):
            G = float(line.split()[9])
        elif line.startswith(' Sum of electronic energy and thermal correction to H'):
            H = float(line.split()[9])
    if 0 in (U_corr, G_corr, H_corr, G, H, U):
        raise ValueError('Missing Values for one of (G_corr, H_corr, U_corr, G, H, U)')
    convertor = 627.5095
    return G_corr * convertor, H_corr * convertor, U_corr * convertor, G * convertor, H * convertor, U * convertor


def get_gaussian_hf_energy(logfile, kcal=True, tail_lines=100):
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()[-tail_lines:]

    flat = "".join(line.strip() for line in lines)

    # 匹配 \HF= 后面的能量值
    match = re.search(r"\\HF=([-\d\.EeDd]+)", flat)
    if match:
        energy = float(match.group(1).replace("D", "E"))
        energy_kcal = round(energy * 627.509, 2)
        if kcal:
            return energy_kcal
        else:
            return energy
    raise ValueError('Could not find HF energy from logfile')


def get_gaussian_counterpoise_corrected_energy(logfile, kcal=False, tail_lines=300):
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()[-tail_lines:]

    for line in lines:
        if line.startswith(' Counterpoise corrected energy'):
            line.strip().rstrip()
            energy = float(line.split()[-1])
            energy_kcal = round(energy * 627.509, 2)
            if kcal:
                return energy_kcal
            else:
                return energy
    raise ValueError('Could not find Counterpoise energy from logfile')


def get_gaussian_freq(logfile):
    with open(logfile, "r", encoding="utf-8") as f:
        lines = f.readlines()
    frequencies = []
    for line in lines:
        if line.startswith(' Frequencies'):
            freq = list(map(float, line.split()[-3:]))
            frequencies.extend(freq)
    return frequencies


def get_disconnected_rdmols(mol):
    # Get the connected components as separate Mol objects
    fragments = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    return fragments


def is_gaussian_terminate(logfile):
    with open(logfile, 'r') as log:
        log_contents = log.readlines()

    is_normal = False
    for content in log_contents[-50:]:
        if 'Normal termination of' in content:
            is_normal = True
            break
        elif 'Error termination' in content:
            is_normal = False
            break
    return is_normal


def convert_fchk2xyz(fchk, outdir, multiwfn_main_path):
    current_folder = os.getcwd()
    if os.path.basename(fchk) not in os.listdir(outdir):
        shutil.copy(fchk, outdir)
    os.chdir(outdir)

    convert_xyz_message = ['100\n', '2\n', '1\n', '\n']
    with open('fchk_to_xyz.txt', 'w') as f:
        f.writelines(convert_xyz_message)
    if 'settings.ini' not in os.listdir(outdir):
        shutil.copy(os.path.join(multiwfn_main_path, 'settings.ini'), outdir)

    out, errors = run_command(f"{os.path.join(multiwfn_main_path, 'Multiwfn')} {os.path.basename(fchk)} "
                              f"< fchk_to_xyz.txt |tee "
                              f"fchk_to_xyz.log")
    os.remove('fchk_to_xyz.txt')
    os.remove('settings.ini')
    os.chdir(current_folder)
    return out, errors


def get_metals_from_molfile(molfile_path):
    """
    读取 mol 文件，返回其中出现的金属元素符号列表（去重后）。
    """
    mol = HostMolecule.init_from_mol_file(molfile_path)

    # 用集合去重，再转成排序列表
    metals = {
        atom.get_element()
        for atom in mol.get_atoms()
        if atom.get_element() in _metal_element
    }
    return sorted(metals)

def get_elements_from_molfile(molfile_path):
    mol = HostMolecule.init_from_mol_file(molfile_path)
    elements = set()
    for atom in mol.get_atoms():
        elements.add(atom.get_element())
    return sorted(elements)

def preprocessing(
        file_path,
        metal=None,
        from_atoms=(7, 8),
        sanitize=False,
        kekulize=True,
        reset_charge=True
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
            if atom.GetSymbol() not in _metal_element:
                atom.SetFormalCharge(0)
    if sanitize:
        AllChem.SanitizeMol(rw_mol)
    if kekulize:
        AllChem.Kekulize(rw_mol)
    return rw_mol