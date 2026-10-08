# DeepHostGuest

**Geometric deep learning for predicting supramolecular host–guest binding conformations.**

This repository contains the full code pipeline of the manuscript
*"Learning to Dock: Geometric Deep Learning for Predicting Supramolecular
Host-Guest Complexes"*. DeepHostGuest learns crystal-structure-informed
host–guest distance preferences from experimentally resolved supramolecular
complexes and predicts guest binding conformations inside a host cavity
(translation + rotation + rotatable-bond torsions), optionally followed by
GFN2-xTB geometry refinement.

- Trained checkpoint: [`ckpt/`](ckpt) (used for all results in the manuscript)
- Datasets (structural data, augmented data, binding free-energy data, prediction
  inputs): **https://zenodo.org/records/18222349**
- Licensed under the [MIT License](LICENSE); it adapts the
  [DeepDock](https://github.com/OptiMaL-PSE-Lab/DeepDock) framework (MIT).

> **Scope.** DeepHostGuest performs *crystal-structure-informed pose
> generation*. It reproduces crystal-reference binding arrangements; it does not
> predict solution-state conformational ensembles, and its docking score is an
> optimization target, **not** a binding-affinity label. See
> [Method notes: code ↔ manuscript](#6-method-notes-code--manuscript).

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Environment setup](#2-environment-setup)
3. [Quick start (bundled example)](#3-quick-start-bundled-example)
4. [Full pipeline, step by step](#4-full-pipeline-step-by-step)
5. [Figure ↔ script reproducibility map](#5-figure--script-reproducibility-map)
6. [Method notes: code ↔ manuscript](#6-method-notes-code--manuscript)
7. [Troubleshooting](#7-troubleshooting)
8. [Citation, license and acknowledgements](#8-citation-license-and-acknowledgements)
9. [Changelog](#9-changelog)

---

## 1. Repository layout

```
DeepHostGuest/                     # installable Python package
├── models.py                      # GNN + mixture-density network
├── DockingFunction_withPenalty.py # pose search: -Σp objective + steric penalty + DE optimizer
├── MLPotentialDocking.py          # optional: pose search driven by MACE-OFF / UMA potentials
├── data_augmentation/             # surface mesh generation, augmentation utilities
└── utils/                         # graph featurisation, distributions, utilities

ckpt/                              # trained model checkpoint (released with the paper)
examples/
├── 1.CollectCCDC/                 # dataset curation from the CSD (notebook)
├── 2.DataAugmentation/            # 7 scripts: augmentation → xTB → ESP mesh → downsampling
├── 3.ModelTraining/               # training notebook (PyTorch + PyG)
├── 4.UseDeepHostGuest/            # ▶ main inference: host input, pose prediction, xTB post-opt
├── 5.CrystallineSpongesPrediction/# crystalline-sponge workflow (3 scripts)
└── 6.MLDeltaG/                    # downstream DFT/ML analysis of binding strength
requirements.txt
```

---

## 2. Environment setup

**Tested configuration** (as reported in the manuscript): Ubuntu 22.04,
Intel i9-14900KF, single NVIDIA RTX 4090 (24 GB), CUDA 12.1.
CPU-only inference works but is slower.

```bash
git clone https://github.com/Chemwzd/DeepHostGuest.git
cd DeepHostGuest

conda create -n DeepHostGuest python=3.9
conda activate DeepHostGuest
pip install -r requirements.txt

# PyTorch + PyTorch Geometric (CUDA 12.1 wheels; adjust for your CUDA version)
pip install torch==2.1.2+cu121 torchvision==0.16.2 torchaudio==2.1.2 \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-spline-conv==1.2.2 \
  torch-cluster==1.6.3 torch-geometric==2.5.3 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

**External tools** (not pip-installable):

| Tool | Used for | Notes |
|---|---|---|
| [xTB](https://github.com/grimme-lab/xtb) | host wavefunction (step 2), post-optimization (steps 4–6) | tested with **6.4.1**; must be in `PATH` |
| [Multiwfn](http://sobereva.com/multiwfn/) | ESP evaluation on host surface (step 2) | tested with v3.8-dev |
| Open Babel | molecular format conversion | `conda install -c conda-forge openbabel` |
| CSD Python API | dataset construction (step 1) | requires a CSD licence |
| Gaussian 09 | DFT binding free energies (step 7) | protocol in SI §5.2 |
| Schrödinger / AutoDock / ORCA / stk / cgbind | benchmark comparators (Fig. 2) | protocols in SI §3.3; run externally |

---

## 3. Quick start (bundled example)

Two complete host–guest examples (`vuqzal`, `alegii`) are bundled, **including
pre-computed host surface meshes (`.ply`)**, so you can run pose prediction
immediately without installing xTB/Multiwfn:

```bash
cd examples/4.UseDeepHostGuest

# 1) edit the two path variables at the top of the script:
#    checkpoint_path -> <repo>/ckpt/dist10_data1400_False_300_16_0.001_0.001_mlr_minTestLoss.chk
#    (and keep the working directory = this folder; job_name = 'vuqzal')

# 2) predict a guest pose (~1 min on GPU)
python 2.PosePrediction.py
# -> vuqzal_2_pre.mol        predicted guest conformation
# -> vuqzal_opt_process.txt  optimizer trace
# -> vuqzal.json             final objective value

# 3) optional: GFN2-xTB refinement of the predicted complex
python 3.PostOptimize_xTB.py --host vuqzal_1.mol --guest vuqzal_2_pre.mol --charge 0
# -> postopt/vuqzal_1_vuqzal_2_pre_opt.mol
```

To predict on your **own** host: prepare `host.mol`, then generate the ESP
surface mesh with `1.GenerateHostInput.py` (requires xTB + Multiwfn + Open
Babel) → produces `vtx_down.ply`, then run step 2 above.

---

## 4. Full pipeline, step by step

### Step 1 — Dataset construction (`1.CollectCCDC/`)

Curates host–guest crystal complexes from the Cambridge Structural Database
(CSD). **Requires a CSD licence and the `ccdc` Python API.**

1. Open `1.ExtractCCDC.ipynb` and set `base_dir` to your working directory.
2. Run the notebook (it can take hours for the full CSD) — it performs:
   disorder resolution → component extraction → validation → host/guest
   classification (pywindow cavity diameter, threshold 2.0 Å) → host–guest
   pairing (centroid-distance criterion) → export of molecular files.
3. Manual curation of the candidates (three criteria in SI §1.1) yields the
   final **1,499 high-confidence host–guest complexes**.

The curated dataset is also available directly on
[Zenodo](https://zenodo.org/records/18222349).

### Step 2 — Data augmentation & host surface preparation (`2.DataAugmentation/`)

Run the numbered scripts **in order** (each script starts with placeholder
paths — edit them first):

| Script | Purpose |
|---|---|
| `1.StructuralAugmentation.py` | 10-fold random rotation/translation augmentation of the curated complexes |
| `2.ConvertMolToXYZ_mp.py` | convert host structures to `.xyz` |
| `3.Runxtb.py` | GFN2-xTB single-point calculation → Molden files |
| `4.MoldenToVertices_mp.py` | ESP evaluation (Multiwfn, 0.001 a.u. isosurface, 1 Å grid) |
| `5.VerticesToPLYMesh_mp.py` | build the host surface mesh |
| `6.MovePLYFile.py` | organise outputs |
| `7.MeshDownSampling_mp.py` | downsample meshes to ≈1,000 nodes (`vtx_down.ply`) |

Output: for every host, a `*.ply` surface-mesh file carrying per-node ESP
values — the host input representation of DeepHostGuest.

### Step 3 — Model training (`3.ModelTraining/`)

Open `1.TrainDeepDockStepLR.ipynb` and run all cells.
Data split (also stated in SI §1.3):

- 1,400 of the 1,499 structures × 10-fold augmentation = **14,000** samples;
- random **9 : 1** split → **12,600 training / 1,400 validation**;
  the validation set is used for checkpoint selection (`*_minTestLoss.chk`);
- the remaining **99 structures are held out** and are *never* used for
  training, model selection or parameter tuning — they form the independent
  benchmark of the manuscript.

Training settings: Adam, lr `1e-3`, StepLR (γ = 0.2 every 50 epochs),
300 epochs, batch size 16, auxiliary atom/bond losses with weight `1e-3`;
≈ 8 h and ≈ 18 GB GPU memory on one RTX 4090.

### Step 4 — Pose prediction (`4.UseDeepHostGuest/`)

The main inference entry point.

1. **Host input** — `1.GenerateHostInput.py`: host `.mol` → `vtx_down.ply`
   (xTB single point → Molden → Multiwfn ESP → mesh → downsampling).
   *Skip if you use the bundled `.ply` files.*
2. **Prediction** — `2.PosePrediction.py`: loads the checkpoint, builds the
   model, and searches the guest pose with differential evolution over
   6 + n_rotatable_bonds parameters. Output: `<name>_2_pre.mol`.
3. **Post-optimization** — `3.PostOptimize_xTB.py` (see Step 5).

Key settings in the released script (matching the manuscript):
`dist_threshold = 6.0 Å`, DE population size 20, max. 1,000 iterations,
mutation ∈ (0.5, 1.0), crossover 0.8, random seed 114514, host fixed.

### Step 5 — GFN2-xTB post-optimization

All reported structures were refined with GFN2-xTB (`--opt`; charged complexes
with the ALPB implicit water model):

```bash
python examples/4.UseDeepHostGuest/3.PostOptimize_xTB.py \
  --host host.mol --guest guest_pre.mol --charge -4 --solvent water --outdir ./postopt
```

- Use `--charge` = **total** charge of the complex (host + guest).
- Add `--gfnff-fallback` to fall back to GFN-FF optimization for difficult
  systems (used for a small number of cases in the manuscript).
- The script passes `--opt` internally — xTB performs a single-point
  calculation without it and never writes `xtbopt.mol`.

### Step 6 — Crystalline sponges (`5.CrystallineSpongesPrediction/`)

For the Pd₆L₄ crystalline-sponge systems (8 guests, SI §4):

1. `1.GenerateHostInput.py` — host surface mesh from the guest-free crystal
   (refcode TUPBOZ).
2. `2.PosePrediction_CS.py` — for the 10 lowest-energy guest conformations
   (from a MacroModel conformational search), predict poses in parallel;
   DE settings: population 30, max. 2,000 iterations, seed 1000.
   Edit the paths and the `names` list first.
3. `3.PostOpt_xTB.py` — GFN2-xTB optimization with `--opt` (ALPB water).

### Step 7 — Binding-strength analysis: DFT & ML (`6.MLDeltaG/`)

**DFT workflow (manuscript Figure 5):** predicted complexes are refined and
used as input for DFT binding free-energy calculations — protocol in
**SI §5.2** (Gaussian 09; B3LYP-D3(BJ)/6-31G(d); def2-TZVP single points;
Shermo thermochemistry; SMD solvation). DFT input preparation is a manual
Gaussian workflow and is not scripted in this repository.

**ML/SHAP workflow (manuscript Figure 6):** run in order

| Script | Purpose |
|---|---|
| `1.0.ConvertGuestMol.py`, `1.1.PrepareHostInput.py` | prepare guest/host inputs |
| `1.2.OptGuests.py` | guest conformer optimization (xTB `-o`) |
| `1.3.PosePrediction.py` | predict host–guest poses for the 876-complex dataset |
| `1.4.OptWithxTB.py` | GFN2-xTB post-optimization (`--opt`) |
| `1.5.CalMoldenFile.py` – `1.7.AlignESP.py` | ESP descriptors for host/guest |
| `2.0.ExtractFeatures.ipynb` | assemble the 19-descriptor feature table |
| `3.0.AutoGluon_and_SHAP.ipynb` | 5-fold AutoGluon CV + SHAP analysis |

`3.0` requires a separate environment with AutoGluon installed
(https://auto.gluon.ai/stable/install.html).

### Step 8 — Accuracy evaluation (RMSD)

The RMSD metric is a **manual two-step procedure** (SI §3.4), not scripted in
this repository:

1. **Host-anchored alignment** — superpose the predicted/optimized complex onto
   the experimental crystal structure by fitting *host atoms only*
   (e.g. "Superpose Structures" in BIOVIA Materials Studio).
2. **Guest RMSD** — in this aligned frame, compute the heavy-atom RMSD of the
   guest with a **symmetry-corrected** algorithm
   ([dockrmsd](https://github.com/Eric-W-Bell/DockRMSD); `pydockrmsd` is in
   `requirements.txt`; also available in Schrödinger).

Report categories used in the manuscript:
`RMSD ≤ 2.0 Å` accurate · `2.0 < RMSD ≤ 3.0 Å` intermediate ·
`RMSD > 3.0 Å` low-accuracy · *failed* = no valid pose could be generated.

---

## 5. Figure ↔ script reproducibility map

| Manuscript item | Where the numbers come from |
|---|---|
| **Fig. 1 / Scheme 1** (framework) | conceptual — see `DeepHostGuest/` package |
| **Fig. 2** (RMSD distribution, 99 benchmark) | poses from `examples/4/2.PosePrediction.py`; no-xTB variant = same script without Step 5; full workflow = + `3.PostOptimize_xTB.py`; RMSD per Step 8. Comparators follow SI §3.3 |
| **Fig. 3** (representative predictions) | outputs of `examples/4/2.PosePrediction.py` |
| **Fig. 4** (crystalline sponges) | `examples/5/` (3 scripts) |
| **Fig. 5** (DFT correlation, 68 systems) | poses via `examples/4` / `examples/6`; DFT per SI §5.2 (manual) |
| **Fig. 6 / Tables S3–S4** (ML, SHAP) | `examples/6/1.0`–`1.7`, `2.0`, `3.0` |
| **Figs. S2–S5** (dataset diversity) | `examples/1`, `examples/2` |
| **Fig. S10** (host-class benchmark) | same outputs as Fig. 2, grouped by host class |
| **Fig. S11** (runtime) | wall-clock times logged by `examples/4/2.PosePrediction.py` |
| **Figs. S13–S15** (sponge & extended set) | `examples/5`; extended-set inputs on Zenodo |

---

## 6. Method notes: code ↔ manuscript

Stated explicitly here to avoid any ambiguity between the code and the paper:

1. **Two different objectives.** The **training loss** is the negative
   log-likelihood (NLL) of the learned mixture density,
   `-log Σ_k π_k N(d | μ_k, σ_k)`, averaged over host–guest node pairs with
   `d ≤ 10 Å` (`DeepHostGuest/models.py::mdn_loss_fn`). The **docking/search
   objective** used at inference is the negative **sum of probability
   densities**, `-Σ p(d)`, over pairs with `d ≤ 6 Å`, plus a steric penalty
   (`DockingFunction_withPenalty.py`). The −Σp convention follows the released
   DeepDock implementation. Do not use the docking score as an affinity estimate.
2. **Distance cutoffs.** Training mask 10 Å; docking cutoff 6 Å (default since
   v1.0.1, matching every released pipeline); penalty cutoffs 3 Å (host–guest)
   and 1.5 Å (guest intramolecular, non-bonded pairs).
3. **Steric penalty.** A *pseudo* Lennard-Jones term with a single global
   distance scale (n = 3 Å) — a fast soft penalty that suppresses atomic
   clashes during optimization, **not** an element-resolved physical potential
   (no per-element van der Waals radii).
4. **Mixture density.** π is softmax-normalised (Σπ = 1), so `Σ_k π_k N(·)` is a
   proper density; the network floors σ ≥ 1.1 Å and μ ≥ 1 Å by construction.
5. **Optimization.** Differential evolution
   (`scipy.optimize.differential_evolution`) over 6 + n_rotatable guest
   parameters; population 20 (99-benchmark) / 30 (sponges), max.
   1,000 / 2,000 iterations, mutation ∈ (0.5, 1.0), crossover 0.8; guest
   initialised randomly; host fixed.
6. **xTB settings.** All reported post-optimizations used `--opt`; charged
   complexes additionally used `--alpb water`; GFN-FF fallback was used where
   GFN2 optimization failed (see `CHANGELOG.md`).
7. **RMSD** — see Step 8 (manual, host-anchored, symmetry-corrected).

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: torch_scatter` / PyG import errors | install the PyG wheel set matching your exact torch/CUDA combo (see §2) |
| `xtb: command not found` | install xTB and add it to `PATH`, or pass `--xtb /path/to/xtb` |
| `obabel: command not found` | `conda install -c conda-forge openbabel`; the post-opt script can also run without it |
| `xTB did not produce xtbopt.mol` | ensure `--opt` is passed; for difficult systems retry with `--gfnff-fallback` |
| CUDA out of memory | reduce batch size (training); large-host inference can be run on CPU |
| `Error Conformers Generation` | check that the guest `.mol` has valid bonds/valences (RDKit sanitisation) |
| Slow prediction for flexible guests | expected: runtime grows with rotatable-bond count (see Fig. S11); reduce `maxiter` for quick tests |
| No Materials Studio for RMSD alignment | any host-atom-only superposition works (RDKit `AlignMol` / PyMOL restricted to host atoms), followed by `pydockrmsd` |

---

## 8. Citation, license and acknowledgements

If you use this code, please cite the manuscript (see [`CITATION.cff`](CITATION.cff)):

> Wang, Z.; Zhang, T.; Yu, M.; Zhou, C.; Xu, Z.; Liu, H.; Wen, Y.; Chen, L.;
> Zheng, J.; Jiang, S. *Learning to Dock: Geometric Deep Learning for Predicting
> Supramolecular Host-Guest Complexes* (manuscript under review).

**License:** MIT (see [`LICENSE`](LICENSE)). This project adapts the
DeepDock framework (Méndez-Lucio et al., *Nat. Mach. Intell.* **2021**,
https://github.com/OptiMaL-PSE-Lab/DeepDock, MIT licence), whose notice is
retained in the LICENSE file.

**Acknowledgements:** we thank the developers of DeepDock, xTB, Multiwfn,
pywindow, ProLIF, DockRMSD, RDKit, PyTorch Geometric, AutoGluon and SHAP,
whose tools this pipeline builds upon.

---

## 9. Changelog

See [`CHANGELOG.md`](CHANGELOG.md).

- **v1.0.1** (2026-10-08): fixed the missing `--opt` in the MLΔG xTB script;
  unified the `dist_threshold` default to 6 Å; documented the training/docking
  objectives and all cutoffs; added the xTB post-optimization CLI, LICENSE,
  CITATION and this README.
- **v1.0.0**: initial release accompanying the manuscript submission.
