"""
Align ESP files of host and guest molecules
to prepare input files for features extraction (ESP Fitness Calculation).

Running in Terminal with:
python 1.7.AlignESP.py

Alternatively, you can execute the operation directly in the IDE.
"""

from MLPredictDeltaG import AlignPointCloud, ply_to_xyz_prop_txt
from sugar.molecule import HostMolecule
import os
import warnings

warnings.filterwarnings('ignore')
from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')

if __name__ == '__main__':
    calculation_dir = '/path/to/CalculateMoldenFile' # calculation_dir in 1.5
    names = [i.split('_')[0] + '_' + i.split('_')[1] for i in os.listdir(calculation_dir)]
    names = set(names)
    from tqdm import tqdm

    for name in tqdm(names):
        print(f'Processing {name}')

        host_mol = HostMolecule.init_from_mol_file(os.path.join(calculation_dir, f'{name}_host', f'{name}_host.mol'))
        host_centroid = host_mol.get_centroid_remove_h()

        aligner = AlignPointCloud(cube_center=host_centroid, num_points=64, step_size=0.4)

        ply_to_xyz_prop_txt(os.path.join(calculation_dir, f'{name}_host', 'esp.ply'),
                            os.path.join(calculation_dir, f'{name}_host', 'esp.txt'))
        ply_to_xyz_prop_txt(os.path.join(calculation_dir, f'{name}_guest', 'esp.ply'),
                            os.path.join(calculation_dir, f'{name}_guest', 'esp.txt'))

        host_aligned, host_esp = aligner.align_point_cloud(os.path.join(calculation_dir, f'{name}_host', 'esp.txt'),
                                                   os.path.join(calculation_dir, f'{name}_host', 'esp_aligned.ply'))
        guest_aligned, guest_esp = aligner.align_point_cloud(os.path.join(calculation_dir, f'{name}_guest', 'esp.txt'),
                                                     os.path.join(calculation_dir, f'{name}_guest', 'esp_aligned.ply'))

        ply_to_xyz_prop_txt(os.path.join(calculation_dir, f'{name}_host', 'esp_aligned.ply'),
                            os.path.join(calculation_dir, f'{name}_host', 'esp_aligned.txt'))
        ply_to_xyz_prop_txt(os.path.join(calculation_dir, f'{name}_guest', 'esp_aligned.ply'),
                            os.path.join(calculation_dir, f'{name}_guest', 'esp_aligned.txt'))
