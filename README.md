---

# DeepHostGuest
A Geometric Deep Learning Method for Predicting Host-Guest Binding Conformations.

---
# 1.Installation
```bash
git clone https://github.com/Chemwzd/DeepHostGuest.git
cd DeepHostGuest
pip install -r requirements.txt
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
1.Prepare your host molecule file (`.mol` format is recommended for RDKit parsing), and generate the `vtx_down.ply` file required for model input using `./examples/4.UseDeepHostGuest/1.GenerateHostInput.py`.
2.Prepare your guest molecule file (`.mol` format is recommended). Run DeepHostGuest with reference to Step 2 for single-molecule prediction. For batch processing, the above codes can be nested in a loop for high-throughput execution.

# 3.How to Training DeepHostGuest
1. Complete preprocessing pipeline (Structure data augmentation → Host xTB calculation → Host ESP calculation → ESP downsampling): see `./examples/2.DataAugmentation`
2. DeepHostGuest model training: see `./examples/3.ModelTraining`

# 4.Machine Learning and SHAP Analysis for Binding Free Energy
## Jupyter Notebook
1. Read the feature file: `./examples/6.MLDeltaG/Data/host_guest_features.xlsx`
2. Run the notebook: `3.0.AutoGluon_and_SHAP.ipynb` for feature selection, 5-fold cross-validation model training and SHAP analysis (global & local SHAP analysis).

# Authors
## Maintainer
 - Zidi Wang (wangzd@shanghaitech.edu.cn)

