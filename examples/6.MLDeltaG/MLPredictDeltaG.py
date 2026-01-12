"""
Used for extracting various features.
"""
import os
import numpy as np
import pyvista as pv
from rdkit import Chem
import prolif as plf
import sys
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, DataStructs, rdFingerprintGenerator
from collections import Counter
from rdkit.Chem import AllChem
from sugar.molecule import HostMolecule
import pywindow as pw
from rdkit.Chem.rdMolDescriptors import GetUSR, GetUSRCAT
from rdkit.Chem import MACCSkeys
from prolif.interactions.interactions import VdWContact

sys.path.append('/path/to/DeepHostGuest/Parent/Folder')

import sugar.pywindow.utilities as util

_old_optimise_z = util.optimise_z


def _optimise_z_patched(z, *args, **kwargs):
    """
    Wrapper around _optimise_z
    """
    z = float(np.asarray(z).ravel()[0])
    return _old_optimise_z(z, *args, **kwargs)


def extract_prop_from_ply(plyfile):
    """
    Extract property defined in the .ply file.

    Returns
    numpy.ndarray
    """
    with open(plyfile, 'r') as f:
        plycontent = f.readlines()
    prop = []
    start_index = None
    vertex_number = None
    for i, content in enumerate(plycontent):
        if content.startswith('element vertex'):
            vertex_number = content.rstrip('\n').split()[-1]
        elif content.startswith('end_header'):
            start_index = i
            break
    for i in range(start_index + 1, start_index + int(vertex_number) + 1):
        prop.append(float(plycontent[i].rstrip('\n').split()[-1]))

    return np.array(prop)


def show_point_cloud(ply_file, prop_name='Property'):
    point_cloud = pv.read(ply_file)

    prop = extract_prop_from_ply(ply_file)
    point_cloud[prop_name] = prop

    print(f"{ply_file} has {point_cloud.n_points} points, {point_cloud.n_cells} cells")

    plotter = pv.Plotter()
    alphas = np.where((prop >= 0) & (prop <= 1e-5), 0.0, 1.0)

    plotter.add_mesh(
        point_cloud,
        scalars=prop_name,
        opacity=alphas,
        render_points_as_spheres=True
    )
    plotter.show()


def show_host_guest_point_cloud(host_ply_file, guest_ply_file, prop_name='Property'):
    host_point_cloud = pv.read(host_ply_file)
    guest_point_cloud = pv.read(guest_ply_file)

    host_prop = extract_prop_from_ply(host_ply_file)
    host_point_cloud[prop_name] = host_prop
    guest_prop = extract_prop_from_ply(guest_ply_file)
    guest_point_cloud[prop_name] = guest_prop

    print(f"{host_ply_file} has {host_point_cloud.n_points} points, {host_point_cloud.n_cells} cells")
    print(f"{guest_ply_file} has {guest_point_cloud.n_points} points, {guest_point_cloud.n_cells} cells")

    plotter = pv.Plotter()
    host_alphas = np.where((host_prop >= 0) & (host_prop <= 1e-5), 0.0, 1.0)
    guest_alphas = np.where((guest_prop >= 0) & (guest_prop <= 1e-5), 0.0, 1.0)

    vmin = min(host_prop.min(), guest_prop.min())
    vmax = max(host_prop.max(), guest_prop.max())
    clim = (vmin, vmax)

    plotter.add_mesh(
        host_point_cloud,
        scalars=prop_name,
        opacity=host_alphas,
        render_points_as_spheres=True,
        clim=clim
    )
    plotter.add_mesh(
        guest_point_cloud,
        scalars=prop_name,
        opacity=guest_alphas,
        render_points_as_spheres=True,
        clim=clim
    )
    plotter.show()


def extract_coord_from_ply(plyfile):
    point_cloud = pv.read(plyfile)
    return np.array(point_cloud.points)


class AlignPointCloud:
    '''
    IMPORTANT NOTE:
    Always verify the original coordinate system of the ESP or ED point cloud
    (e.g., units in Å or Bohr, coordinate origin, and any prior centering or
    transformations). Failing to confirm this may lead to misalignment,
    incorrect scaling, or other coordinate inconsistencies when interpolating
    the point cloud into the new uniform grid.

    A class used to project (interpolate) a property-defined point cloud
    onto a new uniformly sampled 3D grid.

    Parameters
    ----------
    cube_center : array-like of shape (3,)
        The (x, y, z) coordinates of the center of the new uniform cube.

    num_points : int, default=128
        Number of grid points along each axis of the cube.

    step_size : float, default=1.0
        Spacing between adjacent grid points along each axis.

    Methods
    -------
    align_point_cloud(point_cloud, prop):
        Interpolates the input point cloud and its associated property values
        onto the new uniform 3D grid defined by (cube_center, num_points, step_size).
    '''

    def __init__(self, cube_center=[0., 0., 0.], num_points=128, step_size=0.6):
        self.num_points = num_points
        self.step_size = step_size / 0.5918
        self.cube_center = cube_center

        # Calculate the box to align the esp and ed.
        mid = (self.num_points - 1) / 2
        x = self.cube_center[0] + (np.arange(self.num_points) - mid) * self.step_size
        y = self.cube_center[1] + (np.arange(self.num_points) - mid) * self.step_size
        z = self.cube_center[2] + (np.arange(self.num_points) - mid) * self.step_size
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        self.cube = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    def align_point_cloud(self, prop_file, save_path, prop_name='Property'):
        ref_data = np.loadtxt(prop_file)
        ref_points = ref_data[:, :3]  # shape (N, 3)
        ref_prop = ref_data[:, 3]  # shape (N,)

        ref_point_cloud = pv.PolyData(ref_points)
        ref_point_cloud[prop_name] = ref_prop

        # Pyvista only accepts numpy.uint8 as property in .ply file.
        aligned_point_cloud = pv.PolyData(self.cube)
        aligned = aligned_point_cloud.interpolate(ref_point_cloud)
        aligned_prop = aligned[prop_name]

        aligned.save(save_path, binary=False)
        with open(save_path, 'r') as f:
            ply_content = f.readlines()
        num_vertices = aligned.number_of_points
        start_index = None
        for i, content in enumerate(ply_content):
            if content.startswith('end_header'):
                ply_content.insert(i - 2, f'property float {prop_name} \n')
                start_index = i
                break
        # property line has been inserted into index i,
        # so the range should be start_index + 2
        for i, index in enumerate(range(start_index + 2, num_vertices + start_index + 2)):
            ply_content[index] = ply_content[index].rstrip('\n') + ' ' + str(aligned_prop[i]) + '\n'
        os.remove(save_path)
        with open(save_path, 'w') as f:
            f.writelines(ply_content)
        return aligned, aligned_prop


def ply_to_xyz_prop_txt(plyfile, out_txt, prop_col=-1):
    """
    Convert ASCII PLY (vertex-based) to xyz+property txt.

    Parameters
    ----------
    prop_col : int
        Which column in vertex line is the property (default: last column).
    """
    with open(plyfile, "r") as f:
        lines = f.readlines()

    n_vertex = None
    start = None
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_vertex = int(line.split()[-1])
        elif line.startswith("end_header"):
            start = i + 1
            break

    if n_vertex is None or start is None:
        raise ValueError("Invalid PLY header")

    data = []
    for i in range(n_vertex):
        parts = lines[start + i].split()
        x, y, z = map(float, parts[:3])
        prop = float(parts[prop_col])
        data.append([x, y, z, prop])

    np.savetxt(out_txt, np.array(data), fmt="%.8f")


def get_interaction_fp_plf(host_mol_file, guest_mol_file, interactions=None):
    """
    Get Host-guest interaction fingerprints using prolif.
    Returns:
        ndarray
        ['Hydrophobic', 'HBAcceptor', 'HBDonor', 'XBAcceptor', 'XBDonor', 'Cationic',
        'Anionic', 'CationPi', 'PiCation', 'FaceToFace', 'EdgeToFace', 'PiStacking', 'VdWContact']
    """
    if interactions is None: interactions = ['Anionic', 'CationPi', 'Cationic', 'EdgeToFace', 'FaceToFace',
                                             'HBAcceptor', 'HBDonor', 'Hydrophobic', 'PiCation', 'PiStacking',
                                             'XBAcceptor', 'XBDonor', 'VdWContact']
    host_mol = Chem.MolFromMolFile(host_mol_file, removeHs=False)
    plf_host_mol = plf.Molecule.from_rdkit(host_mol, resname='HST', resnumber=1, chain="A")
    guest_mol = Chem.MolFromMolFile(guest_mol_file, removeHs=False)
    plf_guest_mol = plf.Molecule.from_rdkit(guest_mol, resname='GST', resnumber=1, chain="B")

    fp = plf.Fingerprint(interactions=interactions, count=True)
    ifp = fp.generate(plf_guest_mol, plf_host_mol, metadata=False)

    return next(iter(ifp.values()))


def get_surface_info(log_file, decimals=4):
    """
    Read surface analysis results from the log file after converting the ESP file with Multiwfn.

        volume (Angstrom^3): Van der Waals volume of the molecule.
        min_esp_value (kcal/mol): Minimum value of ESP.
        max_esp_value (kcal/mol): Maximum value of ESP.
        overall_sa (Angstrom^2): Overall surface area of the molecule.
        positive_sa (Angstrom^2): Surface area of regions with positive ESP values.
        negative_sa (Angstrom^2): Surface area of regions with positive ESP values.
        overall_average (kcal/mol)：Average ESP value of the overall region.
        positive_average (kcal/mol)：Average ESP value of the positive region.
        negative_average (kcal/mol)：Average ESP value of the negative region.
        overall_variance ((kcal/mol)^2): Variance of ESP in the overall region.
        positive_variance ((kcal/mol)^2): Variance of ESP in the positive region.
        negative_variance ((kcal/mol)^2): Variance of ESP in the negative region.
        molecular_polarity_index (kcal/mol): Molecular polarity index (MPI).
        nonpolar_sa (Angstrom^2): Polar surface area (|ESP| > 10 kcal/mol).
        polar_sa (Angstrom^2): Nonpolar surface area (|ESP| <= 10 kcal/mol).
    """
    with open(log_file, 'r') as f:
        log_contents = f.readlines()

    key_matching_dict = {
        ' Volume:': None,
        ' Minimal value:': [None, None],
        ' Overall surface area:': None,
        ' Positive surface area:': None,
        ' Negative surface area:': None,
        ' Overall average value:': None,
        ' Positive average value:': None,
        ' Negative average value:': None,
        ' Overall variance': None,
        ' Positive variance:': None,
        ' Negative variance:': None,
        ' Molecular polarity index (MPI):': None,
        ' Nonpolar surface area (|ESP| <= 10 kcal/mol):': None,
        ' Polar surface area (|ESP| > 10 kcal/mol):': None,
    }

    for log_line in log_contents:
        for key, value in key_matching_dict.items():
            if log_line.startswith(key):
                if key in [' Nonpolar surface area (|ESP| <= 10 kcal/mol):',
                           ' Polar surface area (|ESP| > 10 kcal/mol):']:
                    splited_line = log_line.split('(')[-2].split()
                    key_matching_dict[key] = round(float(splited_line[-2]), decimals)
                elif key == ' Minimal value:':
                    key_matching_dict[key] = [round(float(log_line.split()[2]), decimals),
                                              round(float(log_line.split()[6]), decimals)]
                else:
                    content = log_line.split()[-2]
                    if '(' in content:
                        content = content.strip('(')
                    key_matching_dict[key] = round(float(content), decimals)

    return {'volume': key_matching_dict[' Volume:'],
            'min_esp_value': key_matching_dict[' Minimal value:'][0],
            'max_esp_value': key_matching_dict[' Minimal value:'][1],
            'overall_sa': key_matching_dict[' Overall surface area:'],
            'positive_sa': key_matching_dict[' Positive surface area:'],
            'negative_sa': key_matching_dict[' Negative surface area:'],
            'overall_average': key_matching_dict[' Overall average value:'],
            'positive_average': key_matching_dict[' Positive average value:'],
            'negative_average': key_matching_dict[' Negative average value:'],
            'overall_variance': key_matching_dict[' Overall variance'],
            'positive_variance': key_matching_dict[' Positive variance:'],
            'negative_variance': key_matching_dict[' Negative variance:'],
            'molecular_polarity_index': key_matching_dict[' Molecular polarity index (MPI):'],
            'nonpolar_sa': key_matching_dict[' Nonpolar surface area (|ESP| <= 10 kcal/mol):'],
            'polar_sa': key_matching_dict[' Polar surface area (|ESP| > 10 kcal/mol):']}


def get_logp(rdmol):
    logP = Crippen.MolLogP(rdmol)
    return logP


def get_tpsa(rdmol):
    return rdMolDescriptors.CalcTPSA(rdmol)


def get_heavy_atom_count(rdmol, elements=None, halogens=None):
    """
    Count atoms by element type in an RDKit Mol.
    Halogens (F, Cl, Br, I) are summed into one feature: halogen_count.

    Returns
    -------
    dict
        {
          'C': nC,
          'O': nO,
          'N': nN,
          'S': nS,
          'halogen_count': nX,
          'heavy_atom_count': N
        }
    """
    if elements is None:
        elements = ['C', 'H', 'O', 'N', 'S']
    if halogens is None:
        halogens = ['F', 'Cl', 'Br', 'I']

    counts = {el: 0 for el in elements}
    halogen_count = 0

    for atom in rdmol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
        elif sym in halogens:
            halogen_count += 1

    # heavy atom count
    heavy_atom_count = sum(
        v for k, v in counts.items() if k != 'H'
    ) + halogen_count

    result = {k: v for k, v in counts.items() if k != 'H'}
    result["halogen_count"] = halogen_count
    # result["heavy_atom_count"] = heavy_atom_count

    return result


def get_pore_info(xyz_file):
    """
    Returns:
        [pore_diameter, pore_volume, [windows]]

    """
    molsys = pw.MolecularSystem.load_file(xyz_file)
    mol = molsys.system_to_molecule()
    pore_diameter = mol.calculate_pore_diameter_opt()
    pore_volume = mol.calculate_pore_volume_opt()
    windows = mol.calculate_windows()
    return pore_diameter, pore_volume, windows


def get_mf2_bits(mol: Chem.Mol, radius: int = 1, nbits: int = 64) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr  # shape (64,)


def get_mol_distance(sugar_mol1, sugar_mol2):
    com1 = sugar_mol1.get_centroid_remove_h()
    com2 = sugar_mol2.get_centroid_remove_h()

    dis = np.linalg.norm(com1 - com2)
    return dis


def get_morgan_count(mol, radius=2, nBits=2048):
    fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        arr[idx] = v
    return arr


def get_usr(mol):
    return GetUSR(mol)


def get_usrcat(mol):
    return GetUSRCAT(mol)


def get_atom_pair_fp(mol, fpsize=2048):
    """Compute the RDKit Shape-Based fingerprints of the given molecules."""
    shapefp_gen = rdFingerprintGenerator.GetAtomPairGenerator(
        fpSize=fpsize,
    )

    fp = shapefp_gen.GetFingerprint(mol)
    shapefp_array = np.zeros((fp.GetNumOnBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, shapefp_array)

    return shapefp_array

def get_maccs_fp(mol):
    """
    Calculate the RDKit MACCS Keys fingerprint, returning a NumPy array of length 167 (0/1),
    where bit 0 is unused and bits 1–166 correspond to the MACCS keys.
    """
    fp = MACCSkeys.GenMACCSKeys(mol)  # ExplicitBitVect，len = 167
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
