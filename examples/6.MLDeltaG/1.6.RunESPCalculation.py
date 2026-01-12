"""
Calculate ESP files from host and guest molecules
to prepare input files for features extraction (ESP Fitness Calculation).

Running in Terminal with:
python 1.6.RunESPCalculation.py
"""
import sys

sys.path.append('/path/to/DeepHostGuest/Parent/Folder')

from DeepHostGuest.data_augmentation.generate_mesh import *
from DeepHostGuest.data_augmentation.from_vtx_to_mesh import *
from tqdm import tqdm
import json
import os
import multiprocessing

xtb_path = xtb = '/path/to/xTB'
multiwfn_path = '/path/to/Multiwfn'

xtb = ESP(xtb_path, multiwfn_path)

calculation_path = '/path/to/CalculateMoldenFile' # calculation_dir in 1.5

names = os.listdir(calculation_path)
outdir = calculation_path

os.chdir(calculation_path)


def process_file(name):
    try:
        out1, errors1 = xtb.run_molden_to_fch(
            molden_path=os.path.join(calculation_path, name, 'molden.input'),
            workdir=os.path.join(outdir, name))
        out2, errors2 = xtb.run_fch_to_esp(
            fchpath=os.path.join(outdir, name, 'molden.fch'),
            workdir=os.path.join(outdir, name), grid_points_spacing=1, rename=True)
        os.chdir(os.path.join(calculation_path, name))
        pdb_file = 'esp.pdb'
        ply_file = 'esp.ply'
        surf = get_polydata_connectivity_remove(pdb_file)
        esp = surf_to_mesh(surf, ply_file)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    with multiprocessing.Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, names), total=len(names)):
            pass
