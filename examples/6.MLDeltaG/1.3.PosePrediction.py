"""
Optimize Guest Conformations through xTB.

Running in Terminal with:
python 1.2.OptGuests.py

Alternatively, you can execute the operation directly in the IDE.
"""
import sys
import numpy as np
import shutil
import json
from tqdm import tqdm

sys.path.append('/path/to/DeepHostGuest/Parent/Folder')
from DeepHostGuest.DockingFunction_withPenalty import dock_compound
from DeepHostGuest.models import *
from DeepHostGuest.utils.data import *
from DeepHostGuest.utils.utilities import concat_molecules_withmol

from rdkit import Chem
import pandas as pd

info_excel = '/path/to/host_guest_features_SI.xlsx'
info_df = pd.read_excel(info_excel, sheet_name='Sheet2')

checkpoint_path = os.path.join('/path/to/DeepHostGuestCheckPoint')
guest_structure_dir = '/path/to/OptGuestsDir'
host_dir = '/path/to/HostCalculationDir'
json_file = '/path/to/JSON/ResultFiles'
test_dir = '/path/to/PredictionDir'
num_cpus = 32

avail_idx = info_df['ID'].tolist()
print(f'Total Available index: {len(avail_idx)}')

# Default Parameters
seed = 114514
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

removeHs = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# For Testing Purpose, the popsize and iter_times can be smaller, such as
# popsize = 1, iter_times = 100
method = 'new_de'
popsize = 20

iter_times = 2000
revise_popsize = True
with_penalty = True
adapt_bound = True

model_threshold = 10.

if removeHs:
    ligand_model = LigandNet(13, edge_features=7, residual_layers=10, dropout_rate=0.10)
else:
    ligand_model = LigandNet(14, edge_features=7, residual_layers=10, dropout_rate=0.10)
target_model = TargetNet(1, edge_features=3, residual_layers=10, dropout_rate=0.10)
model = DeepDock(ligand_model, target_model, hidden_dim=64, n_gaussians=10, dropout_rate=0.10,
                 dist_threhold=model_threshold).to(device)

checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])


if not os.path.exists(test_dir):
    os.makedirs(test_dir)

else:
    for idx in avail_idx:
        guest_file = f'{idx}_opt.mol'
        if guest_file not in os.listdir(test_dir):
            shutil.copy(os.path.join(guest_structure_dir, guest_file), test_dir)

if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        opt_results = json.load(f)
else:
    opt_results = {}

process_num = 1
os.chdir(test_dir)
undo_names = []
undo_names.sort()

for prefix in avail_idx:
    guest_save_file = f"{prefix}_2_pre.mol"
    if not os.path.exists(guest_save_file):
       undo_names.append(f'{prefix}_opt.mol')

print(f'{len(undo_names)} files have not been processed.')

for name in tqdm(undo_names):
    prefix = name.split('.')[0].rstrip('_opt')
    host_name = prefix.split('_')[0]

    host_ply = os.path.join(host_dir, host_name, 'vtx_down.ply')
    host_mol = Chem.MolFromMolFile(os.path.join(host_dir, host_name, f'{host_name}.mol'),
                                   removeHs=False)


    print(f'Working with {name}')
    guest_mol = Chem.MolFromMolFile(name, removeHs=removeHs, sanitize=True)

    guest_save_file = f"{prefix}_2_pre.mol"
    if os.path.exists(guest_save_file):
        print(f'{name} has been calculated!')
        process_num += 1
        continue

    opt_mol, init_mol, result = dock_compound(guest_mol, host_ply, model, dist_threshold=6.,
                                              removeHs=removeHs, seed=seed, device=device,
                                              savepath=f'{prefix}_opt_process.txt',
                                              host_mol=host_mol, canonicalize_guest=True, popsize=popsize,
                                              maxiter=iter_times,
                                              mutation=(0.5, 1), recombination=0.8, workers=num_cpus,
                                              updating='deferred',
                                              disp=True, revise_popsize=revise_popsize
                                              )
    Chem.MolToMolFile(opt_mol, guest_save_file)
    concat_mol = concat_molecules_withmol(host_mol, opt_mol)
    Chem.MolToMolFile(concat_mol, f"{prefix}_concat_pre.mol")

    opt_results[prefix] = {'fun': result['fun']}
    with open(f"{prefix}_result_{result['fun']}.txt", 'w') as f1:
        f1.write(f"{result['fun']}")
    with open(json_file, 'w') as f:
        json.dump(opt_results, f)
    process_num += 1

    print(f'{process_num} processed.')