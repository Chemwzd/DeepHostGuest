"""
python 2.PosePrediction_CS.py

Alternatively, you can execute the operation directly in the IDE.
"""
import os
import sys
import numpy as np
import shutil
import json

sys.path.append(f'/path/to/your/DeepHostGuest/Parent/Directory')
from DeepHostGuest.DockingFunction_withPenalty import dock_compound
from DeepHostGuest.models import *
from DeepHostGuest.utils.data import *
from DeepHostGuest.utils.utilities import preprocessing

from rdkit import Chem

seed = 1000
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)


removeHs = False

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 'cpu' or 'cuda'. CUDA is available for loading models, but the DE algorithm will continue to run on the CPU.

# For testing purpose, popsize can be set to 5 or smaller with
# iter_times 100 or smaller
popsize = 30
iter_times = 2000

# default parameters
method = 'new_de'
revise_popsize = True
with_penalty = True
adapt_bound = True

# Change Directories
host_ply = '/path/to/your/vtx_down.ply'
host_mol_file = '/path/to/your/Host.mol'
guest_structure_dir = '/path/to/your/Guests'
checkpoint_path = '/path/to/DeepHostGuest/Checkpoint'
test_dir = f'/path/to/your/pose_prediction_dir'
json_file = '/path/to/your/results.json'
num_cpus = 32

# The coordination bonds in the metal cage need to be corrected;
# otherwise, RDKit may encounter errors.
host_mol = preprocessing(
    host_mol_file,
    metal={'Pd': 2}, from_atoms=(6, 7), sanitize=True)

model_threshold = 10.

if removeHs:
    ligand_model = LigandNet(13, edge_features=7, residual_layers=10, dropout_rate=0.10)
else:
    ligand_model = LigandNet(14, edge_features=7, residual_layers=10, dropout_rate=0.10)
target_model = TargetNet(1, edge_features=3, residual_layers=10, dropout_rate=0.10)
model = DeepDock(ligand_model, target_model, hidden_dim=64, n_gaussians=10, dropout_rate=0.10,
                 dist_threhold=float(model_threshold)).to(device)

checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])

if not os.path.exists(test_dir):
    shutil.copytree(guest_structure_dir, test_dir)

if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        opt_results = json.load(f)
else:
    opt_results = {}

names = os.listdir(guest_structure_dir)
names.sort()
total_num = len(names)
process_num = 1
names = [i for i in names if i.startswith('4c_')]
os.chdir(test_dir)
for name in names:
    print(f'Working with {name}')
    guest_mol = Chem.MolFromMolFile(name, removeHs=removeHs, sanitize=True)
    guest_save_file = f"{name.split('.')[0]}_2_pre.mol"
    if os.path.exists(guest_save_file):
        print(f'{name} has been calculated!')
        process_num += 1
        continue

    opt_mol, init_mol, result = dock_compound(guest_mol, host_ply, model, dist_threshold=6.,
                                              removeHs=removeHs, seed=seed, device=device,
                                              savepath=f'{name}_opt_process.txt',
                                              host_mol=host_mol, canonicalize_guest=False, popsize=popsize,
                                              maxiter=iter_times,
                                              mutation=(0.5, 1), recombination=0.8, workers=num_cpus,
                                              updating='deferred',
                                              disp=True, revise_popsize=revise_popsize
                                              )
    Chem.MolToMolFile(opt_mol, guest_save_file)
    opt_results[name] = {'fun': result['fun']}
    with open(f"{name}_result_{result['fun']}.txt", 'w') as f1:
        f1.write(f"{result['fun']}")
    with open(json_file, 'w') as f:
        json.dump(opt_results, f)
    process_num += 1

    print(f'{process_num} processed.')