import os
from rdkit import Chem
from rdkit.Chem import rdmolops
import shutil
import time
import networkx as nx

xtb = '/path/to/your/xtb'
obabel = '/path/to/your/obabel'

# Concat guest mol with host mol to get the complex molecular files.
pre_guest_dir = '/path/to/your/PredictionDir'
host_file = '/path/to/your/Host.mol'
host_mol = Chem.MolFromMolFile(host_file, removeHs=False, sanitize=True)
concat_out_dir = '/path/to/your/ConcatStructures'

for guest_file in os.listdir(pre_guest_dir):
    if guest_file.endswith('_pre.mol'):
        guest_mol = Chem.MolFromMolFile(os.path.join(pre_guest_dir, guest_file), removeHs=False)
        concat_mol = rdmolops.CombineMols(host_mol, guest_mol)
        Chem.MolToMolFile(concat_mol, os.path.join(concat_out_dir, guest_file))
        os.system(f'{obabel} {os.path.join(concat_out_dir, guest_file)} -O {os.path.join(concat_out_dir, guest_file)}')

names = [i.split('.')[0] for i in os.listdir(concat_out_dir)]
names.sort()

# Optimization Parameters
methods_dict = {'gfn2': '--gfn 2', 'gfnff': '--gfnff'}
method = 'gfn2'
level = 'normal'

# Revise Calculation and Output Directories
calculation_dir = '/path/to/your/xTBCalculations'
opt_struc_dir = '/path/to/your/OptStructures'
os.makedirs(calculation_dir, exist_ok=True)
os.makedirs(opt_struc_dir, exist_ok=True)

errors = []

for name in names:
    name_calculation_dir = os.path.join(calculation_dir, name)
    os.makedirs(name_calculation_dir, exist_ok=True)

    print(f'===========================xTB calculation for {name}===========================')
    opt_mol_file = os.path.join(name_calculation_dir, 'xtbopt.mol')
    if os.path.exists(opt_mol_file):
        continue

    charge = 12
    os.chdir(name_calculation_dir)
    concat_file = os.path.join(concat_out_dir, f'{name}.mol')
    shutil.copy(concat_file, name_calculation_dir)
    try:
        os.system(f'{xtb} {name}.mol -c {charge} --opt {level} -P 32 {methods_dict[method]} --alpb water -v')
        shutil.copy(opt_mol_file, os.path.join(opt_struc_dir, f'{name}.mol'))
    except:
        errors.append(name)

print(f'Errors: {errors}')


