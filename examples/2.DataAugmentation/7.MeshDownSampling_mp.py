"""
Downsample the PLY file to facilitate efficient reading and training.

Running in Terminal with:
python 7.MeshDownSampling_mp.py
"""
import pyvista as pv
from DeepHostGuest.data_augmentation.from_vtx_to_mesh import *
import os
import multiprocessing
from tqdm import tqdm
import shutil

basedir = '/path/to/host_ply'
outdir = '/path/to/host_ply_ds'
os.makedirs(outdir, exist_ok=True)

# Copy vtx.ply to outdir
for plyfile in os.listdir(basedir):
    if plyfile.endswith('.ply'):
        shutil.copy(os.path.join(basedir, plyfile), outdir)

ply_files = os.listdir(outdir)
os.chdir(outdir)

if __name__ == "__main__":
    def process_file(file):
        esp = downsampling(file, file, target_reduction=0.9)
        return None

    with multiprocessing.Pool(processes=32) as pool:
        list(tqdm(pool.imap(process_file, ply_files), total=len(ply_files)))
