# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-10-08

### Fixed
- `examples/6.MLDeltaG/1.4.OptWithxTB.py`: both the primary GFN2-xTB command and
  the GFN-FF fallback were missing the `--opt` flag. Without it xTB performs a
  single-point calculation and never writes `xtbopt.mol`, so the released script
  did not reproduce the post-optimization protocol described in the manuscript.
  Both branches now run geometry optimization (`--opt`). Reported during peer
  review; see commit `7e8454d`.

### Changed
- `DeepHostGuest/DockingFunction_withPenalty.py`: the `dist_threshold` default is
  now **6.0 Å** (previously 5.0 Å), matching the value used by every released
  prediction pipeline (`examples/4`, `examples/5`, `examples/6`) and the
  manuscript. Behaviour is unchanged for scripts that pass the value explicitly.
- Documentation of the two distinct objectives: the **training loss** is the
  negative log-likelihood (`models.py::mdn_loss_fn`, pairs ≤ 10 Å), while the
  **docking/search objective** is the negative sum of probability densities
  (`DockingFunction_withPenalty.py`, pairs ≤ 6 Å) plus the steric penalty. See
  module docstrings and README §"Method notes: code ↔ manuscript".

### Added
- `examples/4.UseDeepHostGuest/3.PostOptimize_xTB.py`: command-line helper for
  GFN2-xTB post-optimization of a predicted complex (with ALPB water for charged
  systems and optional GFN-FF fallback).
- `LICENSE` (MIT, retaining the DeepDock notice), `CITATION.cff`, `.gitignore`.
- Rewritten step-by-step `README.md`, including a figure ↔ script reproducibility
  map and a troubleshooting section.

## [1.0.0] - 2026-08-05

### Added
- Initial public release accompanying the manuscript submission:
  dataset construction (`examples/1.CollectCCDC`), data augmentation
  (`examples/2.DataAugmentation`), model training (`examples/3.ModelTraining`),
  pose prediction (`examples/4.UseDeepHostGuest`), crystalline-sponge prediction
  (`examples/5.CrystallineSpongesPrediction`) and downstream ML/SHAP analysis
  (`examples/6.MLDeltaG`), together with the trained checkpoint in `ckpt/`.
