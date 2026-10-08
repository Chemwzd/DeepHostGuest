# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [1.0.2] - 2026-10-08

Code-review hardening pass (Alibaba Open Code Review, five-dimension rule group:
correctness, security, performance, maintainability, test coverage). No change to
the numerical results: the objective functions, cutoffs and optimization settings
are untouched.

### Fixed — correctness
- `DeepHostGuest/utils/data.py`: `HostGuest_dataset.process()` filtered and
  transformed `self.data_list`, an attribute that does not exist yet while the
  dataset is being built (only `pre_filter` / `pre_transform` users were
  affected). It now operates on the freshly built local list.
- `DeepHostGuest/utils/data.py`: `HostGuest_dataset.__init__` raised an opaque
  `FileNotFoundError` from `torch.load` when `processed/data.pt` was missing; it
  now explains how to build the dataset. Loading a list of `Data` objects also
  requires `weights_only=False` explicitly, otherwise PyTorch ≥ 2.6 refuses the
  file.
- `DeepHostGuest/utils/data.py`: `i.rstrip('.ply')` strips *characters*, not a
  suffix, so prefixes ending in `.`, `p`, `l` or `y` were truncated (e.g.
  `cage_ally.ply` → `cage_al`); replaced with `os.path.splitext`.
- `DeepHostGuest/utils/data.py`: `mol_to_graph` now reports which guest file
  RDKit failed to parse instead of dereferencing `None`.
- `DeepHostGuest/utils/data.py`: `Mol2MolSupplier` had a bare `except: pass` and
  used a possibly-unbound `name`, which raised `NameError` for mol2 files without
  `Name` entries.
- `DeepHostGuest/DockingFunction_withPenalty.py`: `score_compound()` masked the
  NumPy probability array with a `torch.where` index tensor. On CUDA that tensor
  cannot be converted to NumPy, so the function crashed whenever
  `device='cuda'`; the mask is now computed in NumPy (identical semantics).
- `DeepHostGuest/DockingFunction_withPenalty.py`: the non-cached penalty branch of
  `OptimizeConformation.score_conformation` called `calculate_penalty_all()`,
  which was never defined in the module (`NameError` whenever
  `OptimizeConformation` was used directly). The helper is now implemented as a
  thin wrapper around the cached `_PenaltyCalculator`, so both paths return the
  same numbers (verified on the bundled `vuqzal` example).
- `DeepHostGuest/DockingFunction_withPenalty.py`: conformer embedding no longer
  ignores an `EmbedMolecule` failure and raises a descriptive error instead of
  failing later inside the force-field code (`_ensure_conformer` helper).
- `DeepHostGuest/utils/utilities.py`: `get_xtb_free_energy()` tested
  `if 'sp.out' and 'g98.out' in os.listdir(...)`, which is truthy whenever
  `g98.out` exists (a non-empty string is truthy); both files are now checked
  explicitly.
- `DeepHostGuest/utils/run_multiwfn.py`: `run_fch_to_esp()` and `run_fch_to_ed()`
  called `os.chdir(workdir)` **before** `os.makedirs(workdir)`, raising
  `FileNotFoundError` for any new directory.
- `DeepHostGuest/utils/utilities.py`: `preprocessing()` now names the metal whose
  formal charge is missing instead of raising a bare `KeyError`.

### Fixed — side effects / maintainability
- All Multiwfn/xTB helpers (`run_multiwfn.Multiwfn.*`, `convert_fchk2xyz`,
  `get_xtb_free_energy`) restore the process working directory in a `finally`
  block, so they can be called in loops. `get_xtb_free_energy` also stopped
  copying `settings.ini` into a path that could re-nest after `chdir`.
- Removed the private `sugar` dependency from the shipped pipeline:
  `DeepHostGuest/utils/geometry.py` now provides `cal_rotation_matrix`,
  `rotation_around_axis`, `translation` and `norm_vector` (NumPy only, verified
  bit-for-bit identical to the original implementation over 200 randomised
  cases); `heavy_atom_centroid` replaces `HostMolecule.get_centroid_remove_h`;
  the unused `HostMolecule` import in `data_augmentation/run_multisim.py` and
  `examples/6.MLDeltaG/MLPredictDeltaG.py` was dropped. The remaining optional
  uses are guarded by imports that explain how to install the toolkit.
- `utils/utilities.py`: dropped the unused `pydockrmsd.hungarian` import and a
  duplicated `warnings` import.
- Documentation cleanup (comments/docstrings only, no behaviour change): removed 54
  leftover artefacts of an earlier global search-and-replace in which stage
  numbers had been expanded into identifiers — `1.preprocessing.generate_structural_data`,
  `2.train_deepdock`, `3.use_deepdock` (`# skip 3.use_deepdock-membered rings` now
  reads `# skip 3-membered rings`, `torch.randn(5, 2.train_deepdock)` reads
  `torch.randn(5, 2)`, …) — and corrected stale module paths in usage examples
  (`DeepDockHostGuest.preprocessing.*` → `DeepHostGuest.data_augmentation.*`,
  `/examples/2.DataAugmentation`). `files_prefix = [i.rstrip('_host.xyz') ...]`
  in the docstring examples was corrected to `os.path.splitext(i)[0]` for the same
  reason as the code fix above.

### Verification
- `DeepHostGuest/utils/geometry.py` vs. the original `sugar.utilities`
  implementation: max |Δ| = 0 over 200 random cases (rotation matrix,
  rotation, translation, normalisation).
- Structural augmentation end-to-end on the bundled `vuqzal` example: valid
  RDKit-readable output, host and guest receive the same rigid transform
  (max host–guest distance deviation 1.3 × 10⁻⁴ Å, i.e. output rounding), and
  results are byte-identical for a fixed seed.
- Dataset pipeline on the bundled example (torch 2.8.0 + PyG 2.6.1):
  `read_ply` reads 345 vertices / 689 faces; the processed dataset holds a host
  graph (345 nodes, 2068 edges, 1 node feature, 3 Cartesian edge features) and a
  guest graph (16 nodes, 30 edges, 14 node features) — matching
  `TargetNet(1, edge_features=3)` and `LigandNet(14, edge_features=7)` as used by
  the released prediction scripts; `torch.load(path)` without `weights_only` was
  reproduced to fail on torch 2.8.0 (`UnpicklingError`), confirming the need for
  the fix; `rstrip('.ply')` was reproduced to truncate `vuqzal_1_ally.ply` to
  `vuqzal_1_a`.
- Steric penalty: 0 for the provided non-clashing pose (min. contact 3.5 Å),
  2.0 × 10⁹ for a deliberately clashing pose (min. contact 0.51 Å);
  `calculate_penalty_all()` matches the cached path exactly.
- Distance masking: the NumPy rewrite reproduces the previous mask exactly
  (219 / 500 synthetic pairs beyond 6 Å).
- xTB/Multiwfn helpers (`generate_mesh.ESP`, stub executable): workdir auto-created
  for a non-existent directory (previously `FileNotFoundError`), working directory
  restored after every call, cache short-circuit still returns `(0, 0)`.
- `_ensure_conformer`: molecules that already carry a conformer are returned
  unchanged; embedding is reproducible for a fixed seed; a forced
  `EmbedMolecule` failure now raises a descriptive `ValueError`.
- All 39 Python files compile; every modified module was re-checked with
  `python -m py_compile`.

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
