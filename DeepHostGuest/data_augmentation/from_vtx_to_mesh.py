"""
Post-process vtx.pdb (point cloud) to surface with desired property.

Example:
-------------------------------------------------
A. Read the vtx.pdb and perform surface reconstruction to get the mesh file
    1.preprocessing.generate_structural_data. Convert the vtx.pdb file to example.ply format, including ESP as a property.
        esp = pointcloud_to_mesh(vtx.pdb, example.ply)
        print(esp)  # pointcloud_to_mesh returns ESP values as list

    2.train_deepdock. Visualize the surface and its property using Pyvista.
        ply_file = 'pointcloud_to_mesh.ply'
        point_cloud = pv.read(ply_file)
        prop = extract_prop_from_ply(ply_file)  # Pyvista cannot read property in the generated .ply file.
                                                # Pyvista only accepts numpy.uint8 as property in .ply file.
        point_cloud['ESP'] = prop
        plotter = pv.Plotter()
        plotter.add_mesh(point_cloud, scalars='ESP')
        plotter.show()
B. Read the vtx.pdb with 'CONNECT' lines and directly construct mesh from it (---RECOMMENDED---).
    from DeepDockHostGuest.1.preprocessing.preprocessing.from_vtx_to_mesh import *
    import os

    pdbfile = '/path/to/your/vtx.pdb'
    savefile = '/path/to/your/savepath'
    surf = get_polydata_connectivity_remove(pdbfile)
    esp = surf_to_mesh(surf, savefile)
"""

import os
import numpy as np
import pyvista as pv
from collections import Counter
from itertools import combinations


def vtx_to_nodefile(infile, outfile):
    """
    Convert Multiwfn vtx.pdb file to .node file
    -------------------------------------------------
    Example:
        # Node count, dim, attribute (ESP), no boundary marker
        129703 0
        # Node index, node coordinates, ESP
        -7.261  -8.699      -3.96
    """
    with open(infile, 'r') as f:
        vtx = f.readlines()

    node_content = ['# Node count, dim, attribute, no boundary marker\n',
                    '# Node index, node coordinates\n']
    node_count = 0
    for line in vtx:
        if line.startswith('HETATM'):
            node_count += 1
            node_info = [str(node_count).ljust(10),
                         str(round(float(line[29:38].strip()), 3)).rjust(10),
                         str(round(float(line[38:46].strip()), 3)).rjust(10),
                         str(round(float(line[46:54].strip()), 3)).rjust(10),
                         str(round(float(line[60:66].strip()), 2)).rjust(10), '\n']
            node_content.append(' '.join(node_info))
    node_content.insert(1, f'{node_count} 0\n')
    with open(outfile, 'w') as f:
        f.writelines(node_content)
    return None


def vtx_to_coord(infile, downsample_size=1):
    """
    Read Multiwfn vtx.pdb file to numpy.ndarray (n*3.use_deepdock)
    -------------------------------------------------
    Example:
        -7.261     -1.314     -8.699
    """
    with open(infile, 'r') as f:
        vtx = f.readlines()
    coords = []
    esp_vals = []
    for line in vtx:
        if line.startswith('HETATM'):
            coords.append([
                round(float(line[29:38].strip()), 3),
                round(float(line[38:46].strip()), 3),
                round(float(line[46:54].strip()), 3),
            ])
            esp_vals.append(round(float(line[60:66].strip()), 2))
    coords = np.array(coords)
    esp_vals = np.array(esp_vals)
    return coords[::downsample_size], esp_vals[::downsample_size]


def vtx_to_pointcloud_pyvista(infile, downsample_size=1, check=True, show=False):
    """
    Read The Multiwfn vtx.pdb or output.txt file and Return pyvista.PolyData

    It is recommended to use the following steps to generate vtx.pdb file:
        1.preprocessing.generate_structural_data. Convert xtb molden.input to molden.fch (100,  2.train_deepdock, 7, 'Enter')
        2.train_deepdock. Restart or reset Multiwfn              (0, 'r')
        3.use_deepdock. Input 'molden.fch' and calculate ESP   (12, 0, 6)
    """
    with open(infile, 'r') as f:
        vtx = f.readlines()
    coords = []
    esp_vals = []
    for line in vtx:
        if line.startswith('HETATM'):
            coords.append([
                round(float(line[29:38].strip()), 3),
                round(float(line[38:46].strip()), 3),
                round(float(line[46:54].strip()), 3),
            ])
            esp_vals.append(round(float(line[60:66].strip()), 2))
    coords = np.array(coords)
    esp_vals = np.array(esp_vals)
    coords = coords[::downsample_size]
    esp_vals = esp_vals[::downsample_size]
    point_cloud = pv.PolyData(coords)
    if check:
        np.allclose(coords, point_cloud.points)
    point_cloud['ESP'] = esp_vals
    if show:
        point_cloud.plot(render_points_as_spheres=True)
    return point_cloud


def pointcloud_to_mesh(pointcloud, outfile, **kwargs):
    mesh = pointcloud.reconstruct_surface(progress_bar=True, **kwargs)
    interpolated_values = mesh.interpolate(pointcloud, pass_cell_data=False)
    mesh['ESP'] = interpolated_values['ESP']
    esp = mesh['ESP'].tolist()
    mesh.save(outfile, binary=False)
    with open(outfile, 'r') as f:
        ply_content = f.readlines()
    num_vertices = mesh.number_of_points
    start_index = None
    for i, content in enumerate(ply_content):
        if content.startswith('end_header'):
            ply_content.insert(i - 2, 'property float ESP \n')
            start_index = i
            break
    # property line has been inserted into index "i-2.train_deepdock",
    # so the range should be start_index + 2.train_deepdock
    for i, index in enumerate(range(start_index + 2, num_vertices + start_index + 2)):
        ply_content[index] = ply_content[index].rstrip('\n') + ' ' + str(esp[i]) + '\n'
    os.remove(outfile)
    with open(outfile, 'w') as f:
        f.writelines(ply_content)
    return esp


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

    return prop


def _xtb_esp_to_pointcloud(infile):
    with open(infile, 'r') as f:
        xtb_file = f.readlines()
    coords = [i.split()[0:3] for i in xtb_file]
    coords = list(map(lambda sublist: list(map(float, sublist)), coords))
    potential = [float(i.split()[-1]) for i in xtb_file]
    return np.array(coords), np.array(potential)


def xtb_esp_to_pointcloud(infile, check=True, show=False):
    coords, potential = _xtb_esp_to_pointcloud(infile)
    point_cloud = pv.PolyData(coords)
    if check:
        np.allclose(coords, point_cloud.points)
    point_cloud['ESP'] = potential
    if show:
        point_cloud.plot(render_points_as_spheres=True)
    return point_cloud


def read_vtx_connect(infile):
    """
    index_symbols[index] = [symbol, coords, esp_values]
    The index is range
    """
    with open(infile, 'r') as f:
        vtx_contents = f.readlines()
    index_symbols = {}
    connectivity = []
    for i, line in enumerate(vtx_contents):
        if line.startswith('HETATM'):
            element_symbol = line.rstrip().split()[-1]
            index_symbols[i - 1] = []
            index_symbols[i - 1].append(element_symbol)
            index_symbols[i - 1].append([
                round(float(line[29:38].strip()), 3),
                round(float(line[38:46].strip()), 3),
                round(float(line[46:54].strip()), 3),
            ])
            index_symbols[i - 1].append(round(float(line[60:66].strip()), 2))

        if line.startswith('CONECT'):
            conn_line = line.rstrip()[6:]
            connectivity.append(
                [int(conn_line[i:i + 6].strip()) for i in range(0, len(conn_line), 6)]
            )
    return index_symbols, connectivity


def get_polydata_connectivity_remove(infile, show=False):
    index_symbols, connectivity = read_vtx_connect(infile)
    faces = []
    for sublist in connectivity:
        if len(sublist) > 2:
            for face in combinations(sublist[1:], 2):
                # modified_face = [point - 1.preprocessing.generate_structural_data for point in face]
                # faces.append(sorted(modified_face))
                faces.append(sorted([face[0], face[1], sublist[0]]))
    counts = Counter(tuple(sorted(sublist)) for sublist in faces)
    final_faces = [list(key) for key, value in counts.items() if value > 1]

    # Get the needed indices
    C_indices = [i[0] for i in connectivity]
    # Create a dict
    indices_dict = {j: i for i, j in enumerate(C_indices)}
    coords = np.array([index_symbols[i][1] for i in indices_dict.keys()])
    esp = np.array([index_symbols[i][2] for i in indices_dict.keys()])
    # Replace the origin index to indices_dict.values()
    for i in range(len(final_faces)):
        final_faces[i] = [indices_dict[index] for index in final_faces[i]]

    for face in final_faces:
        face.insert(0, 3)
    faces = np.hstack(final_faces)
    surf = pv.PolyData(coords, faces)
    surf['ESP'] = esp
    if show:
        plotter = pv.Plotter()
        plotter.add_mesh(surf, scalars='ESP', show_edges=True)
        plotter.show()
    return surf


def surf_to_mesh(surface, outfile):
    esp = surface['ESP'].round(4).tolist()
    surface.save(outfile, binary=False)
    with open(outfile, 'r') as f:
        ply_content = f.readlines()
    num_vertices = surface.number_of_points
    start_index = None
    for i, content in enumerate(ply_content):
        if content.startswith('end_header'):
            ply_content.insert(i - 2, 'property float ESP \n')
            start_index = i
            break
    # property line has been inserted into index "i-2",
    # so the range should be start_index + 2
    for i, index in enumerate(range(start_index + 2, num_vertices + start_index + 2)):
        ply_content[index] = ply_content[index].rstrip('\n') + ' ' + str(esp[i]) + '\n'
    os.remove(outfile)
    with open(outfile, 'w') as f:
        f.writelines(ply_content)
    return esp


def downsampling(plyfile, outfile, target_reduction=0.8):
    point_cloud = pv.read(plyfile)
    prop = extract_prop_from_ply(plyfile)  # Pyvista cannot read property in the generated .ply file.
    # Pyvista only accepts numpy.uint8 as property in .ply file.
    point_cloud['ESP'] = prop
    decimated = point_cloud.decimate(target_reduction).interpolate(point_cloud)
    esp = decimated['ESP']
    decimated.save(outfile, binary=False)
    with open(outfile, 'r') as f:
        ply_content = f.readlines()
    num_vertices = decimated.number_of_points
    start_index = None
    for i, content in enumerate(ply_content):
        if content.startswith('end_header'):
            ply_content.insert(i - 2, 'property float ESP \n')
            start_index = i
            break
    # property line has been inserted into index "i-2",
    # so the range should be start_index + 2
    for i, index in enumerate(range(start_index + 2, num_vertices + start_index + 2)):
        ply_content[index] = ply_content[index].rstrip('\n') + ' ' + str(esp[i]) + '\n'
    os.remove(outfile)
    with open(outfile, 'w') as f:
        f.writelines(ply_content)
    return decimated, esp
