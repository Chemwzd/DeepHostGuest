"""
Convert the augmented mol files into xyz format and organize them
by placing the host and guest structures into their respective folders.

python 2.ConvertMolToXYZ_mp.py

Alternatively, you can execute the operation directly in the IDE.
"""

import multiprocessing
from DeepHostGuest.data_augmentation.run_multisim import *

run_multisim = RunMultisim(schrodinger=None,
                           obabel='/path/to/obabel')

if __name__ == "__main__":
    mol_dir = '/path/to/aug_mol_dir'
    xyz_dir = '/path/to/target_dir'
    os.makedirs(xyz_dir, exist_ok=True)

    mol_files = os.listdir(mol_dir)
    mol_prefixes = [i.split('.')[0] for i in mol_files]

    with multiprocessing.Pool(processes=24) as pool:
        for prefix in mol_prefixes:
            pool.apply_async(run_multisim.convert_pdb_to_mol,
                             args=(os.path.join(mol_dir, f'{prefix}.mol'),
                                   os.path.join(xyz_dir, f'{prefix}.xyz')))
        pool.close()
        pool.join()

    host_dir = '/path/to/splited/host'
    guest_dir = '/path/to/splited/guest'
    os.chdir(xyz_dir)
    for file in os.listdir(os.getcwd()):
        if file.split('_')[1] == '1':
            shutil.copy(file, host_dir)
        elif file.split('_')[1] == '2':
            guest_mol_file = os.path.join(mol_dir, f"{file.split('.')[0]}.mol")
            shutil.copy(guest_mol_file, guest_dir)
