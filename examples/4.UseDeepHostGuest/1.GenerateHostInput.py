"""
Convert the single host molecular structure into a ply mesh file required by DeepHostGuest.
Accept input in formats such as mol, pdb, or xyz, and generate the output as vtx_down.ply.

Running in Terminal:
python 1.GenerateHostInput.py
"""
import sys

sys.path.append(f'/path/to/your/DeepHostGuest/Parent/Directory')
from DeepHostGuest.data_augmentation.generate_mesh import *
from DeepHostGuest.data_augmentation.from_vtx_to_mesh import get_polydata_connectivity_remove, surf_to_mesh, \
    downsampling
from DeepHostGuest.utils.utilities import convert_molfile
import pyvista as pv

host_mol_file = '/path/to/your/host_mol_file'
xtb_calculation_dir = '/path/to/your/xTBCaluclation_dir'
obabel_path = '/path/to/your/obabel'

host_xyz_file = '/path/to/your/host_xyz_file'
convert_molfile(obabel_path, host_mol_file, host_xyz_file)

xtb = ESP('/path/to/your/xtb',
          '/path/to/your/Multiwfn')

# 1.Run xtb to get molden.input
xtb_out, xtb_error = xtb.run_xtb_single(host_xyz_file, xtb_calculation_dir)

# 2.Convert molden.input to vtx.pdb
out1, errors1 = xtb.run_molden_to_fch(
    molden_path=os.path.join(xtb_calculation_dir, 'molden.input'),
    workdir=xtb_calculation_dir)
out2, errors2 = xtb.run_fch_to_esp(
    fchpath=os.path.join(xtb_calculation_dir, 'molden.fch'),
    workdir=xtb_calculation_dir, grid_points_spacing=1)

# 3.Convert vtx.pdb to vtx.ply file
pdb_file = os.path.join(xtb_calculation_dir, 'vtx.pdb')
ply_file = os.path.join(xtb_calculation_dir, 'vtx.ply')
surf = get_polydata_connectivity_remove(pdb_file)
esp = surf_to_mesh(surf, ply_file)

# 4.Downsampling of vtx.ply file
ply_file_down = os.path.join(xtb_calculation_dir, 'vtx_down.ply')
down_mesh, down_values = downsampling(ply_file, ply_file_down, target_reduction=0.9)

show_esp = True
if show_esp:
    point_cloud = pv.read(ply_file_down)
    point_cloud['ESP'] = down_values
    print(f"{ply_file} has {point_cloud.n_points} points, {point_cloud.n_cells} cells")
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars='ESP', show_edges=True)
    plotter.show()
