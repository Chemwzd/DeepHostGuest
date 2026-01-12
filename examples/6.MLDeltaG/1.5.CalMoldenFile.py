"""
Calculate molden files from host, guest and host-guest complexes
to prepare input files for features extraction.

Running in Terminal with:
python 1.5.CalMoldenFile.py
"""
import os
import sys
from rdkit import Chem
import shutil
import pandas as pd

sys.path.append('/path/to/DeepHostGuest/Parent/Folder')

from DeepHostGuest.utils.utilities import get_disconnected_rdmols

info_excel = '/path/to/host_guest_features_SI.xlsx'
info_df = pd.read_excel(info_excel, sheet_name='Sheet2')

xtb = '/path/to/xTB'
obabel = '/path/to/obabel'

# Revise Related Folder Path
calculation_dir = '/path/to/CalculateMoldenFile'
opt_struc_dir = '/path/to/OptStructureDir'  # opt_struc_dir in 1.4
split_struc_dir = '/path/to/SplitHostGuest' # Empty folder for storing individual host and guest files.
ref_host_dir = '/path/to/HostMolDir' # host_mol_dir in 1.1

os.makedirs(calculation_dir, exist_ok=True)

errors = []

names = [i.split('_')[0] + '_' + i.split('_')[1] for i in os.listdir(opt_struc_dir)]
names = list(set(names))
names.sort()
print(len(names))

# Split Host and Guest Structures
for name in names:
    split_host_mol_file = os.path.join(split_struc_dir, f'{name}_host.mol')
    split_guest_mol_file = os.path.join(split_struc_dir, f'{name}_guest.mol')
    if os.path.exists(split_host_mol_file) and os.path.exists(split_guest_mol_file):
        continue

    concat_file = os.path.join(opt_struc_dir, f'{name}_gfn2.mol')
    complex_mol = Chem.MolFromMolFile(concat_file, removeHs=False, sanitize=False)
    disconnected_mols = get_disconnected_rdmols(complex_mol)
    ref_host_prefix = name.split('.')[0].split('_')[0]
    ref_host_mol = Chem.MolFromMolFile(os.path.join(ref_host_dir, f'{ref_host_prefix}.mol'), removeHs=False)

    if disconnected_mols[0].GetNumAtoms() == ref_host_mol.GetNumAtoms():
        host, guest = disconnected_mols[0], disconnected_mols[1]
    elif disconnected_mols[1].GetNumAtoms() == ref_host_mol.GetNumAtoms():
        host, guest = disconnected_mols[1], disconnected_mols[0]
    else:
        raise ValueError(f'No Host in complex {name}')

    Chem.MolToMolFile(host, split_host_mol_file)
    Chem.MolToMolFile(guest, split_guest_mol_file)

# Calculate molden.input
for mol_file in os.listdir(split_struc_dir):
    name = mol_file.split('.')[0]
    name_calculation_dir = os.path.join(calculation_dir, name)
    os.makedirs(name_calculation_dir, exist_ok=True)

    shutil.copy(os.path.join(split_struc_dir, mol_file), name_calculation_dir)
    os.chdir(name_calculation_dir)
    print(f'===========================xTB calculation for {name}===========================')
    os.system(f'{obabel} {mol_file} -O {name}.xyz')

    if os.path.exists('molden.input'):
        print(f'xTB calculation for {name} is already done.')
        continue

    try:
        os.system(f'{xtb} {name}.xyz --molden -v')
    except Exception as ex:
        print(ex)
        errors.append(name)

print(f'Errors: {errors}')
