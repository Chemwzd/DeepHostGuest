"""
Use Multiwfn and Cubegen to generate surface vertices file: vtx.pdb.

Running in Terminal with:
python 4.MoldenToVertices_mp.py
"""

from DeepHostGuest.data_augmentation.generate_mesh import *
from tqdm import tqdm
import os
import json
import multiprocessing

xtb = ESP('/path/to/xtb',
          '/path/to/Multiwfn')

# Base Calculation Directory for xTB
basedir = '/path/to/xtb_out'
names = os.listdir(basedir)
outdir = basedir

os.chdir(basedir)

def process_file(name):
    try:
        out1, errors1 = xtb.run_molden_to_fch(
            molden_path=os.path.join(basedir, name, 'molden.input'),
            workdir=os.path.join(outdir, name))
        out2, errors2 = xtb.run_fch_to_esp(
            fchpath=os.path.join(outdir, name, 'molden.fch'),
            workdir=os.path.join(outdir, name), grid_points_spacing=1)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    with multiprocessing.Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, names), total=len(names)):
            pass
        # pool.map(process_file, names)
