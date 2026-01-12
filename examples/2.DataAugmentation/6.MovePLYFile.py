"""
Move all the .ply files to a new folder and copy the corresponding guest .mol files

Running in Terminal with:
python 6.MovePLYFile.py
"""
import os
import shutil
from tqdm import tqdm
# Base Calculation Directory for xTB
plydir = '/path/to/xtb_out'

plyfiles = [i for i in os.listdir(plydir)]
plyfiles.sort()

guest_mol_dir = '/path/to/splited/guest'
target_host_dir = '/path/to/new_folder/host_ply'
target_guest_dir = '/path/to/new_folder/guest'

for name in tqdm(plyfiles):
    prefix = name.split('_')[0]
    aug_index = name.split('_')[-1]
    if os.path.exists(os.path.join(target_host_dir, f'{name}.ply')):
        os.remove(os.path.join(target_host_dir, f'{name}.ply'))
        shutil.copy(os.path.join(plydir, name, 'vtx.ply'), os.path.join(target_host_dir, f'{name}.ply'))
    else:
        shutil.copy(os.path.join(plydir, name, 'vtx.ply'), os.path.join(target_host_dir, f'{name}.ply'))
    if os.path.exists(os.path.join(target_guest_dir, f'{prefix}_2_{aug_index}.mol')):
        os.remove(os.path.join(target_guest_dir, f'{prefix}_2_{aug_index}.mol'))
        guest_path = os.path.join(guest_mol_dir, f'{prefix}_2_{aug_index}.mol')
        shutil.copy(guest_path, target_guest_dir)
    else:
        guest_path = os.path.join(guest_mol_dir, f'{prefix}_2_{aug_index}.mol')
        shutil.copy(guest_path, target_guest_dir)
