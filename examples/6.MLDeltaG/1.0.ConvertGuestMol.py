"""
Convert the SMILES entries from the table into molecular structure files.

Running in Terminal with:
python 1.0.ConvertGuestMol.py
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_mol(
    smiles: str,
    n_confs: int = 10,
    max_iters: int = 1000,
    seed: int = 1000,
    mmff_variant: str = "MMFF94s",
    prune_rms: float = 0.5,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    mol = Chem.AddHs(mol)

    # Embed multiple conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if not conf_ids:
        raise RuntimeError("Conformer embedding failed")

    use_mmff = AllChem.MMFFHasAllMoleculeParams(mol)

    if use_mmff:
        AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            numThreads=0,
            maxIters=max_iters,
            mmffVariant=mmff_variant,
        )
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
    else:
        AllChem.UFFOptimizeMoleculeConfs(
            mol,
            numThreads=0,
            maxIters=max_iters,
        )

    best_cid, best_e = None, float("inf")
    for cid in conf_ids:
        if use_mmff:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        e = ff.CalcEnergy()
        if e < best_e:
            best_e, best_cid = e, cid

    if best_cid is None:
        raise RuntimeError("Failed to evaluate conformer energies")
    best_conf = Chem.Conformer(mol.GetConformer(best_cid))
    mol.RemoveAllConformers()
    mol.AddConformer(best_conf, assignId=True)

    return mol


info_excel = '/path/to/host_guest_features_SI.xlsx'
info_df = pd.read_excel(info_excel, sheet_name='Sheet2')

obabel = '/path/to/obabel'
struc_path = '/path/to/output'
os.chdir(struc_path)

prefixes = [i.split('.')[0] for i in os.listdir(struc_path)]
prefixes = set(prefixes)

for prefix in prefixes:
    mol_file = os.path.join(struc_path, prefix + '.mol')
    sdf_file = os.path.join(struc_path, prefix + '.sdf')
    if os.path.exists(mol_file):
        continue
    os.system(f'{obabel} {sdf_file} -O {mol_file}')

id_list = info_df['ID'].tolist()
from tqdm import tqdm
for index in tqdm(id_list):
    if f'{index}.mol' in os.listdir(struc_path):
        continue
    smiles = info_df.loc[info_df['ID'] == index, 'GuestSmiles'].values[0]
    try:
        rd_mol = smiles_to_mol(smiles)
        Chem.MolToMolFile(rd_mol, f'{index}.mol')
        os.system(f'{obabel} {index}.mol -O {index}.mol')
    except Exception as e:
        print(e)
        print(f'{index} failed')