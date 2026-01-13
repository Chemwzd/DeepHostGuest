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

