---

# DeepHostGuest
A Geometric Deep Learning Method for Predicting Host-Guest Binding Conformations.

# Requirements
DeepHostGuest has been tested  on Ubuntu 22.04 with an Intel® Core™ i9-14900KF processor and RTX 4090 GPU alongside CUDA 12.1.

Make sure you have the following installed:
`CUDA 12.1`
`PyTorch-2.1.2` (DeepHostGuest can also be used on a purely CPU setup. Simply install the CPU version of PyTorch)
`torchvision-0.16.2`
`torchaudio-2.1.2`
`torch_cluster-1.6.3`
`torch_scatter-2.1.2`
`torch_sparse-0.6.18`
`torch_spline_conv-1.2.2`
`torch_geometric-2.5.3`

For Preprocessing, `xTB` and `Multiwfn` are needed.

---
# 1.Installation

```bash
git clone https://github.com/Chemwzd/DeepHostGuest.git
cd DeepHostGuest
conda create -n DeepHostGuest python=3.9
conda activate DeepHostGuest
pip install -r requirements.txt
pip install torch==2.1.2+cu121 torchvision==0.16.2 torchaudio==2.1.2 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-spline-conv==1.2.2 torch-cluster==1.6.3 torch-geometric==2.5.3 -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu117.html
```
Model checkpoints are stored in the ./ckpt folder.


# 2.How to Use DeepHostGuest for Testing
## Terminal
Note: Modify the relevant path information properly.
```bash
cd ./examples/4.UseDeepHostGuest
python 2.PosePrediction.py  # Alternatively, it can be run directly in an IDE.
```


# 3.How to Perform Predictions on Your Own Host-Guest Systems
1. Prepare your host molecule file (`.mol` format is recommended for RDKit parsing), and generate the `vtx_down.ply` file required for model input using `./examples/4.UseDeepHostGuest/1.GenerateHostInput.py`.
2. Prepare your guest molecule file (`.mol` format is recommended). Run DeepHostGuest for single molecule prediction with reference to the above steps. For batch processing, the above code can be nested in a loop to enable high-throughput prediction.


# 3.How to Training DeepHostGuest
1. Complete preprocessing pipeline (Structure data augmentation → Host xTB calculation → Host ESP calculation → ESP downsampling): see `./examples/2.DataAugmentation`
2. DeepHostGuest model training: see `./examples/3.ModelTraining`


# 4.Machine Learning and SHAP Analysis for Binding Free Energy
DeepHostGuest is incompatible with AutoGluon due to environment constraints. You will need to set up a separate environment to run AutoGluon. Refer to https://auto.gluon.ai/stable/install.html
## Jupyter Notebook
1. Read the feature file: `./examples/6.MLDeltaG/Data/host_guest_features.xlsx`
2. Run the notebook: `3.0.AutoGluon_and_SHAP.ipynb` for feature selection, 5-fold cross-validation model training and SHAP analysis (global & local SHAP analysis).


# 5.Data Sources
All datasets, including raw structural data, enhanced structural data, binding free energy data and structures, and crystalline sponge prediction inputs, are available at [Zenodo](https://zenodo.org/records/18222349).



# Authors
 - Zidi Wang (wangzd@shanghaitech.edu.cn)


# 6.Optional: Benchmark with General Machine-Learning Potentials
`DeepHostGuest/MLPotentialDocking.py` provides an ASE-calculator based docking path for comparing DeepHostGuest with general atomistic machine-learning potentials such as MACE-OFF or Meta FAIRChem UMA. The optimisation keeps the host molecule fixed, randomly initialises the guest, optimises the same `6 + n` guest variables (rotation, translation, and rotatable-bond torsions), and uses `scipy.optimize.differential_evolution` to minimise the ASE calculator energy. The translation bounds are centred on the host centroid and constrained by the host/guest size estimate so the guest centroid is sampled near the host centroid rather than far outside the host; pass `center_bound_radius` to override this radius explicitly.

Install optional dependencies in a separate environment if needed:

```bash
pip install ase mace-torch fairchem-core
```

Minimal MACE-OFF example:

```python
from rdkit import Chem
from DeepHostGuest.MLPotentialDocking import dock_compound_with_mace_off

host = Chem.MolFromMolFile("host.mol", removeHs=False)
guest = Chem.MolFromMolFile("guest.mol", removeHs=False)

complex_mol, guest_mol, init_guest_mol, result = dock_compound_with_mace_off(
    guest_mol=guest,
    host_mol=host,
    model="medium",
    device="cuda",              # use "cpu" when no CUDA device is available
    maxiter=100,
    output_complex_path="mace_off_complex.mol",
    output_guest_path="mace_off_guest.mol",
    savepath="mace_off_de_history.txt",
)
print(result["fun"], result["success"])
```

Minimal Meta FAIRChem UMA example:

```python
from rdkit import Chem
from DeepHostGuest.MLPotentialDocking import dock_compound_with_fairchem_uma

host = Chem.MolFromMolFile("host.mol", removeHs=False)
guest = Chem.MolFromMolFile("guest.mol", removeHs=False)

complex_mol, guest_mol, init_guest_mol, result = dock_compound_with_fairchem_uma(
    guest_mol=guest,
    host_mol=host,
    model="uma-s-1p1",
    task_name="omol",           # finite molecular host-guest complexes
    device="cuda",              # use "cpu" when no CUDA device is available
    charge=0,                    # total charge of the host-guest complex
    spin_multiplicity=1,         # total spin multiplicity; 1 = singlet
    maxiter=100,
    output_complex_path="uma_complex.mol",
    output_guest_path="uma_guest.mol",
    savepath="uma_de_history.txt",
)
print(result["fun"], result["success"])
```

Because general ML potentials require atom types and atom coordinates, this workflow uses the host `.mol` conformer rather than the host `.ply` surface mesh as the scoring input. For UMA, `task_name="omol"` is a good default for finite molecular complexes; choose another FAIRChem task such as `"omc"` only when it better represents your host-guest system. The `charge` and `spin_multiplicity` arguments are written to `ase.Atoms.info` as the total charge and spin multiplicity of the host-guest complex. If you also enable isolated-energy subtraction (`subtract_host_energy=True` or `subtract_guest_energy=True`), pass `host_charge`/`host_spin_multiplicity` and `guest_charge`/`guest_spin_multiplicity` for those isolated systems. The output files contain the predicted low-energy complex and the corresponding guest conformation.

Troubleshooting:
- If MACE raises `AttributeError: module 'torch.compiler' has no attribute 'is_compiling'`, update `mace-torch`/`torch` when possible. The MACE-OFF helper also installs a small compatibility shim automatically for older PyTorch 2.x environments.
- FAIRChem UMA models can require Hugging Face/model access and may download checkpoints on first use. If model loading fails, verify your `fairchem-core` installation, credentials, model name, and `device` setting.
- For UMA/OMOL calculations, explicitly set the total system `charge` and `spin_multiplicity` (for example, `1` for a singlet and `2` for a doublet) so the correct molecular state is scored.
- Be careful not to swap `guest_mol` and `host_mol`: only `guest_mol` is randomly initialised and optimised, while `host_mol` is kept fixed.
