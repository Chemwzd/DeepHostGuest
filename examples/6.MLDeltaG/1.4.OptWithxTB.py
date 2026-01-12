"""
Post-optimize Complexes through xTB.

Running in Terminal with:
python 1.4.OptWithxTB.py
"""
import os
import sys
from rdkit import Chem
import shutil
import pandas as pd


info_excel = '/path/to/host_guest_features_SI.xlsx'
info_df = pd.read_excel(info_excel, sheet_name='Sheet2')

xtb = '/path/to/xTB'
obabel = '/path/to/obabel'

# 修改计算文件夹
pose_prediction_dir = '/path/to/PredictionDir'
calculation_dir = '/path/to/CalculationDir'
opt_struc_dir = '/path/to/OptStructureDir'

os.makedirs(opt_struc_dir, exist_ok=True)

errors = []

names = [i.split('_')[0] + '_' + i.split('_')[1] for i in os.listdir(pose_prediction_dir)]
names = list(set(names))
names.sort()
print(len(names))

for name in names:
    host_charge_value = info_df.loc[info_df['ID'] == name, 'HostCharge'].values[0]
    guest_charge_value = info_df.loc[info_df['ID'] == name, 'GuestCharge'].values[0]
    charge_value = host_charge_value + guest_charge_value

    concat_file = os.path.join(pose_prediction_dir, f'{name}_concat_pre.mol')
    name_calculation_dir = os.path.join(calculation_dir, name)
    os.makedirs(name_calculation_dir, exist_ok=True)

    shutil.copy(concat_file, name_calculation_dir)
    os.chdir(name_calculation_dir)
    print(f'===========================xTB calculation for {name}===========================')
    opt_mol_file = os.path.join(name_calculation_dir, 'xtbopt.mol')
    if os.path.exists(opt_mol_file) or os.path.exists(os.path.join(opt_struc_dir, f'{name}_gfn2.mol')):
        print(f'{name} xTB calculation for {name} is already done.')
        # shutil.copy(opt_mol_file, os.path.join(opt_struc_dir, f'{name}_pre_{method}_freez{freeze_host}.mol'))
        continue

    try:
        copied_concat_file = os.path.join(name_calculation_dir, f'{name}_concat_pre.mol')
        os.system(f'{obabel} {copied_concat_file} -O {copied_concat_file}')
        if charge_value != 0:
            os.system(f'{xtb} {copied_concat_file} -c {charge_value} --alpb water -v')
        else:
            os.system(f'{xtb} {copied_concat_file} -c {charge_value} -v')
        if os.path.exists(opt_mol_file):
            shutil.copy(opt_mol_file, os.path.join(opt_struc_dir, f'{name}_gfn2.mol'))
        else:
            # If Optimization Falied, Using GFN-FF Instead.
            if charge_value != 0:
                os.system(f'{xtb} {copied_concat_file} -c {charge_value} -P 32 --gfnff --alpb water -v')
            else:
                os.system(f'{xtb} {copied_concat_file} -c {charge_value} -P 32 --gfnff -v')
            shutil.copy(opt_mol_file, os.path.join(opt_struc_dir, f'{name}_gfn2.mol'))
    except Exception as ex:
        print(ex)
        errors.append(name)

print(f'Errors: {errors}')


