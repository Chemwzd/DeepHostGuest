"""
Use xtb to perform calculations for Hosts
and generate wavefunction file: molden.input

Running in Terminal with:
python 3.Runxtb.py
"""
import os
import shutil
import multiprocessing

from DeepHostGuest.data_augmentation.generate_mesh import *
from tqdm import tqdm

xtb = ESP('/path/to/xtb',
          '/path/to/Multiwfn')

host_path = '/path/to/splited/host'
host_files = os.listdir(host_path)
prefixes = [i.split('.')[0] for i in host_files]

# Base Calculation Directory for xTB
outdir = '/path/to/xtb_out'

for host_file in tqdm(host_files):
    prefix = host_file.split('.')[0]
    xtb_out_dir = os.path.join(outdir, prefix)
    os.makedirs(xtb_out_dir, exist_ok=True)
    shutil.copy(os.path.join(host_path, host_file), xtb_out_dir)

# Multi process
with multiprocessing.Pool(processes=12) as pool:
    for prefix in prefixes:
        pool.apply_async(xtb.run_xtb_single,
                         args=(os.path.join(host_path, f'{prefix}.xyz'),
                               os.path.join(outdir, prefix)))
    pool.close()
    pool.join()

# Single process
# for host_file in tqdm(host_files):
#     prefix = host_file.split('.')[0]
#     print(f"======Processing {host_file}======")
#     xtb_out_dir = os.path.join(outdir, prefix)
#     os.makedirs(xtb_out_dir, exist_ok=True)
#     shutil.copy(os.path.join(host_path, host_file), xtb_out_dir)
#     try:
#         host_xtb, _ = xtb.run_xtb_single(
#             os.path.join(host_path, host_file),
#             outpath=xtb_out_dir)
#     except Exception as e:
#         print(e)



