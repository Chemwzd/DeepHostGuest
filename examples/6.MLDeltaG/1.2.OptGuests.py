"""
Optimize Guest Conformations through xTB.

Running in Terminal with:
python 1.2.OptGuests.py
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm

xtb = '/path/to/xtb'
info_excel = '/path/to/host_guest_features_SI.xlsx'
info_df = pd.read_excel(info_excel, sheet_name='Sheet2')

# output directory of 1.0.ConvertGuestMol.py
initial_guest_dir = '/path/to/1.0/output'
opt_guest_dir = '/path/to/OptGuests'
calculation_dir = '/path/to/calculation'
os.makedirs(opt_guest_dir, exist_ok=True)
os.makedirs(calculation_dir, exist_ok=True)

for guest_file in tqdm(os.listdir(initial_guest_dir)):
    print(f'Guest file: {guest_file}')
    prefix = guest_file.split('.')[0]
    charge_value = info_df.loc[info_df['ID'] == prefix, 'GuestCharge'].values[0]
    name_cal_dir = os.path.join(calculation_dir, f'{prefix}')

    target_opt_mol = os.path.join(opt_guest_dir, f'{prefix}_opt.mol')
    if os.path.exists(target_opt_mol):
        print(f'{prefix} has already been Optimized.')
        continue

    os.makedirs(name_cal_dir, exist_ok=True)
    shutil.copy(os.path.join(initial_guest_dir, guest_file), os.path.join(name_cal_dir, guest_file))
    os.chdir(name_cal_dir)
    os.system(f'{xtb} {guest_file} -o -c {charge_value} ')

    opt_mol_file = 'xtbopt.mol'
    shutil.copy(opt_mol_file, target_opt_mol)
