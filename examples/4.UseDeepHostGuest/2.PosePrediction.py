import os
import sys
import numpy as np
import shutil
import json

sys.path.append(f'/path/to/your/DeepHostGuest/Parent/Directory')
from DeepHostGuest.DockingFunction_withPenalty import dock_compound, count_num_opt_parameters
from DeepHostGuest.models import *
from DeepHostGuest.utils.data import *

from rdkit import Chem


# Default Parameters
seed = 114514   # For Reproductivity
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

removeHs = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_threshold = 10.
method = 'new_de'
popsize = 20
iter_times = 1000
revise_popsize = True
with_penalty = True
num_cpus = 32

# Revise Path
job_name = 'vuqzal'
guest_mol = Chem.MolFromMolFile(f'{job_name}_2.mol', removeHs=removeHs, sanitize=True)
host_ply = f'{job_name}_1.ply'
host_mol = Chem.MolFromMolFile(f'{job_name}_1.mol', removeHs=removeHs, sanitize=True)
guest_save_file = f"{job_name}_2_pre.mol"
json_file = f'{job_name}.json'
checkpoint_path = '/path/to/DeepHostGuestCheckPoint'

if removeHs:
    ligand_model = LigandNet(13, edge_features=7, residual_layers=10, dropout_rate=0.10)
else:
    ligand_model = LigandNet(14, edge_features=7, residual_layers=10, dropout_rate=0.10)
target_model = TargetNet(1, edge_features=3, residual_layers=10, dropout_rate=0.10)
model = DeepDock(ligand_model, target_model, hidden_dim=64, n_gaussians=10, dropout_rate=0.10,
                 dist_threhold=model_threshold).to(device)

checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])

adapt_bound = True

if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        opt_results = json.load(f)
else:
    opt_results = {}

print(f'Working with {job_name}')
if os.path.exists(guest_save_file):
    print(f'{job_name} has been calculated!')

opt_mol, init_mol, result = dock_compound(guest_mol, host_ply, model, dist_threshold=6.,
                                          removeHs=removeHs, seed=seed, device=device,
                                          savepath=f'{job_name}_opt_process.txt',
                                          host_mol=host_mol, canonicalize_guest=True, popsize=popsize,
                                          maxiter=iter_times,
                                          mutation=(0.5, 1), recombination=0.8, workers=num_cpus, updating='deferred',
                                          disp=True, revise_popsize=revise_popsize
                                          )
Chem.MolToMolFile(opt_mol, guest_save_file)
opt_results[job_name] = {'fun': result['fun']}
with open(f"{job_name}_result_{result['fun']}.txt", 'w') as f1:
    f1.write(f"{result['fun']}")
with open(json_file, 'w') as f:
    json.dump(opt_results, f)
