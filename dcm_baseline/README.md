# dcm_baseline

DCM (Direct Collocation Method) baseline package, used by the Bézier project for
downstream comparisons. This is the `orbit_transfer` Python package plus its case
database, **extracted** from the formerly-vendored `orbit-transfer-analysis/` sibling
repo so the main project no longer reaches into that folder via `sys.path`.

## Contents
- `src/orbit_transfer/` — the package (astrodynamics, dynamics, collocation, optimizer,
  classification, database, types, constants, …). Two-pass pipeline: Hermite–Simpson →
  peak-detect → Multi-Phase LGL.
- `data/trajectories.duckdb` — parametric LEO-to-LEO case database read by the DCM tools.
- `pyproject.toml` — package metadata (deps: casadi, numpy, scipy, scikit-learn, duckdb, matplotlib).

## Install
```
pip install -e dcm_baseline
```
Then `import orbit_transfer` works with no path hacks. Consumed by
`tools/dcm_visualize_case.py` and `tools/dcm_downstream_experiment.py`.

## Status

The former vendored `orbit-transfer-analysis/` folder is **slated for removal** (deletion deferred
as a safety net while the extraction is validated). It still contains a duplicate copy of this
package, which goes away with it — `dcm_baseline/` is the canonical home. The folder's report/notes
material was archived to `~/code/orbit-transfer-archive/`.
