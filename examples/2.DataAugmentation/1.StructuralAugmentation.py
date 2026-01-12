"""
python 1.StructuralAugmentation.py

Alternatively, you can execute the operation directly in the IDE.
"""
import os
from DeepHostGuest.data_augmentation.generate_aug_graph import generate_aug_files_double

structure_dir = '/path/to/your/in'
out_dir = '/path/to/your/out'


prefixes = [i.split('_')[0] for i in os.listdir(os.getcwd())]
prefixes = set(prefixes)
os.chdir(structure_dir)

for prefix in prefixes:
    host_molfile = f'{prefix}_1.mol'
    guest_molfile = f'{prefix}_2.mol'

    generate_aug_files_double(host_molfile, guest_molfile, outdir=out_dir, out_prefix=prefix, num_aug=20,
                              rotation_step_size=2, translation_step_size=20, seed=123)