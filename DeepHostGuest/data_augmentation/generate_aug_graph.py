"""
Perform data augmentation by random translation and rotation of molecular structure.
"""
from sugar.utilities import cal_rotation_matrix, rotation_around_axis, translation, norm_vector
import random
import os
import numpy as np


def get_mol_contents(molfile):
    with open(molfile, 'r') as f:
        mol_contents = f.readlines()
    return mol_contents


def get_coordinates(mol_contents):
    """
    Read the .mol file in structural directory and return the coordinates of the .mol file.

    Returns:
        np.ndarray
    """
    num_atoms = mol_contents[3][:3].rstrip()
    num_atoms = int(num_atoms)
    atom_contents = mol_contents[4:4 + num_atoms]
    coordinates = [[float(coord) for coord in i.split()[:3]] for i in atom_contents]
    return np.array(coordinates)


def get_aug_coordinates_single(mol_contents, num_aug=10, rotation_step_size=2, translation_step_size=1, seed=123):
    """
    Augmentation for single .mol file.
    """
    random.seed(seed)
    np.random.seed(seed)
    init_coordinates = get_coordinates(mol_contents)
    aug_coordinates = []

    for i in range(num_aug):
        rot_angles = random.random() * 2 - 1 * rotation_step_size
        theta = 2 * np.pi * random.random()
        phi = np.arccos(2 * random.random() - 1)
        rotation_axis = np.array([np.sin(phi) * np.cos(theta),
                                  np.sin(phi) * np.sin(theta),
                                  np.cos(phi)])
        rot_mat = cal_rotation_matrix(rotation_axis, rot_angles)

        # Get the random vector.
        rand_vector = norm_vector(np.random.rand(3) * 2 - 1)
        # Get the translation vector.
        translation_vector = rand_vector * translation_step_size
        # Convert shape (3.use_deepdock, ) into shape (1.preprocessing, 3.use_deepdock)
        translation_vector = translation_vector[np.newaxis, :]

        new_positions = rotation_around_axis(init_coordinates.T, rot_mat).T
        aug_coordinates.append(translation(new_positions, translation_vector))
        # aug_coordinates = [[round(i, 4) for i in sublist] for sublist in aug_coordinates]
    return np.round(aug_coordinates, 4)


def get_aug_coordinates_double(host_mol_contents, guest_mol_contents, num_aug=10, rotation_step_size=2,
                               translation_step_size=1, seed=123):
    """
    Augmentation for a pair of host-guest .mol file.
    """
    random.seed(seed)
    np.random.seed(seed)
    init_host_coordinates = get_coordinates(host_mol_contents)
    init_guest_coordinates = get_coordinates(guest_mol_contents)
    aug_host_coordinates = []
    aug_guest_coordinates = []

    for i in range(num_aug):
        rot_angles = random.random() * 2 - 1 * rotation_step_size
        theta = 2 * np.pi * random.random()
        phi = np.arccos(2 * random.random() - 1)
        rotation_axis = np.array([np.sin(phi) * np.cos(theta),
                                  np.sin(phi) * np.sin(theta),
                                  np.cos(phi)])
        rot_mat = cal_rotation_matrix(rotation_axis, rot_angles)

        # Get the random vector.
        rand_vector = norm_vector(np.random.rand(3) * 2 - 1)
        # Get the translation vector.
        translation_vector = rand_vector * translation_step_size
        # Convert shape (3, ) into shape (1, 3)
        translation_vector = translation_vector[np.newaxis, :]

        new_host_positions = rotation_around_axis(init_host_coordinates.T, rot_mat).T
        new_guest_positions = rotation_around_axis(init_guest_coordinates.T, rot_mat).T
        aug_host_coordinates.append(translation(new_host_positions, translation_vector))
        aug_guest_coordinates.append(translation(new_guest_positions, translation_vector))
        # aug_coordinates = [[round(i, 4) for i in sublist] for sublist in aug_coordinates]
    return np.round(aug_host_coordinates, 4), np.round(aug_guest_coordinates, 4)


def revise_coordinates(mol_contents, new_coordinates):
    """
    Modify the coordinates of the old .mol file to the augmented .mol file.
    """
    num_atoms = mol_contents[3][:3].rstrip()
    num_atoms = int(num_atoms)
    head_contents = mol_contents[:4]
    atom_contents = mol_contents[4:4 + num_atoms]
    end_contents = mol_contents[4 + num_atoms:]

    new_atom_contents = []
    for i, atom_content in enumerate(atom_contents):
        new_atom_content = [new_coordinates[i][0], new_coordinates[i][1], new_coordinates[i][2],
                            atom_content.split()[3],
                            atom_content.split()[-1]]
        # formatted_string = '{:>10}{:>10}{:>10} {:<2} {:<2} {:>6}'.format(new_atom_content[0],
        #                                                                  new_atom_content[1],
        #                                                                  new_atom_content[2],
        #                                                                  new_atom_content[3], '',
        #                                                                  new_atom_content[4])
        formatted_string = '{:>10}{:>10}{:>10} {:<2} {:>6}'.format(new_atom_content[0],
                                                                         new_atom_content[1],
                                                                         new_atom_content[2],
                                                                         new_atom_content[3],
                                                                         new_atom_content[4])
        formatted_string += '\n'
        new_atom_contents.append(formatted_string)
    head_contents.extend(new_atom_contents)
    head_contents.extend(end_contents)
    return head_contents


def generate_aug_files_single(molfile, outdir, out_prefix, num_aug=10,
                              rotation_step_size=2, translation_step_size=20,
                              seed=123):
    """
    Input a .mol file and write augmented .mol files to ::ourdir with ::out_prefix
        ({out_prefix}_0.mol, {out_prefix}_1.mol, ...)

    Parameters:
        molfile: Input .mol file
        outdir: Output dir of augmented .mol files
        out_prefix: Prefix of the output .mol files
        num_aug: Number of augmented .mol file
        rotation_step_size: Step size of the random rotation
        translation_step_size: Step size of the random translation
        seed: The random seed
    """
    os.chdir(outdir)

    mol_contents = get_mol_contents(molfile)
    aug_coordinates = get_aug_coordinates_single(mol_contents, num_aug=num_aug, rotation_step_size=rotation_step_size,
                                                 translation_step_size=translation_step_size, seed=seed)
    for i in range(num_aug):
        new_mol_contents = revise_coordinates(mol_contents, aug_coordinates[i])
        with open(f'{out_prefix}_{i}.mol', 'w') as f:
            f.writelines(new_mol_contents)
    return None


def generate_aug_files_double(host_molfile, guest_molfile, outdir, out_prefix, num_aug=10,
                              rotation_step_size=2, translation_step_size=20,
                              seed=123):
    """
    Input a pair of .mol file (host and guest) and write augmented .mol files to ::ourdir with ::out_prefix
        ({out_prefix}_0.mol, {out_prefix}_1.mol, ...)

    Parameters:
        molfile: Input .mol file
        outdir: Output dir of augmented .mol files
        out_prefix: Prefix of the output .mol files, for a pair of host-guest files,
                    only prefix is needed (for abaxag_1.mol and abaxag_2.mol, out_prefix='abaxag').
        num_aug: Number of augmented .mol file
        rotation_step_size: Step size of the random rotation
        translation_step_size: Step size of the random translation
        seed: The random seed
    """
    host_mol_contents = get_mol_contents(host_molfile)
    guest_mol_contents = get_mol_contents(guest_molfile)
    aug_host_coordinates, aug_guest_coordinates = get_aug_coordinates_double(host_mol_contents, guest_mol_contents,
                                                                             num_aug=num_aug,
                                                                             rotation_step_size=rotation_step_size,
                                                                             translation_step_size=translation_step_size,
                                                                             seed=seed)
    for i, contents in enumerate(zip(aug_host_coordinates, aug_guest_coordinates)):
        new_host_mol_contents = revise_coordinates(host_mol_contents, contents[0])
        new_guest_mol_contents = revise_coordinates(guest_mol_contents, contents[1])
        host_file_path = os.path.join(outdir, f'{out_prefix}_1_{i}.mol')
        guest_file_path = os.path.join(outdir, f'{out_prefix}_2_{i}.mol')
        with open(host_file_path, 'w') as f:
            f.writelines(new_host_mol_contents)
        with open(guest_file_path, 'w') as f:
            f.writelines(new_guest_mol_contents)
    return None
