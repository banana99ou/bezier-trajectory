# `orbit-transfer-analysis` Pruning Audit

## Purpose

This document is a practical pruning guide for `orbit-transfer-analysis`.

It is written for a light cleanup policy:

- preserve the full project structure
- avoid deleting material that would force future users to rediscover the project from scratch
- delete only clearly regenerable clutter without further review

This subtree is not a single-purpose package. It contains:

- the main `orbit_transfer` collocation codebase
- DuckDB data and saved trajectory files
- manuscript and report assets
- an optional nested BLADE subproject in `blade-orbit/`
- a separate handoff-style bundle in `docs/report_comparison/student_package/`

Because of that, pruning must be based on reproducibility and role, not only on file size or whether a file is tracked by git.

## Read First

For a new user or agent, the fastest non-destructive orientation path is:

1. `README.md`
2. `traceability.md`
3. `pyproject.toml`
4. `environment.yml`
5. `src/orbit_transfer/`
6. `scripts/`

If the task includes BLADE or BLADE-to-collocation workflows, also read:

1. `blade-orbit/README.md`
2. `blade-orbit/pyproject.toml`
3. `src/orbit_transfer/optimizer/blade_collocation.py`

## Top-Level Folder Map


| Path                  | Role                             | Classification       | Notes                                                    |
| --------------------- | -------------------------------- | -------------------- | -------------------------------------------------------- |
| `README.md`           | project entry point              | absolutely necessary | primary orientation file                                 |
| `pyproject.toml`      | package metadata and deps        | absolutely necessary | needed for installation                                  |
| `environment.yml`     | conda environment                | absolutely necessary | environment recreation                                   |
| `traceability.md`     | figure/table/data map            | absolutely necessary | prevents starting from scratch                           |
| `src/orbit_transfer/` | main package                     | absolutely necessary | core code                                                |
| `scripts/`            | runnable entry points            | absolutely necessary | reproducibility and analysis workflows                   |
| `tests/`              | regression and sanity checks     | absolutely necessary | confidence and future maintenance                        |
| `data/`               | databases and saved trajectories | mixed                | some files are essential, some are optional variants     |
| `manuscript/`         | main paper assets                | absolutely necessary | active paper-facing material                             |
| `docs/`               | specifications and notes         | mixed                | keep core specs, review duplicated bundles               |
| `reports/`            | modular technical writeups       | absolutely necessary | project knowledge base                                   |
| `blade-orbit/`        | optional nested BLADE project    | nice to have         | required only for BLADE-related workflows                |
| `manuscript-ko/`      | Korean manuscript variant        | nice to have         | useful but not core runtime                              |
| `ko_part2.tex`        | extra manuscript fragment        | nice to have         | retain unless clearly obsolete                           |
| `sessions/`           | session continuity notes         | nice to have         | helps later agents                                       |
| `글감/`                 | research notes and drafts        | nice to have         | contextual material                                      |
| `results/`            | generated local outputs          | manual review        | useful references, but usually regenerable               |
| `.claude/`            | local agent/tooling context      | nice to have         | not core science, but low risk to keep                   |
| `.git`                | nested repo metadata             | absolutely necessary | do not delete unless intentionally removing repo history |


## Absolutely Necessary

Keep these if the goal is to use the folder without rebuilding understanding from zero.

### Core project identity

- `README.md`
- `pyproject.toml`
- `environment.yml`
- `traceability.md`

### Core code

- `src/orbit_transfer/`
- `scripts/`
- `tests/`

### Paper and technical understanding

- `manuscript/`
- `reports/`
- the non-duplicate specification files in `docs/`

### Data required for current reproducibility

- `data/trajectories.duckdb`
- `data/trajectories/`

These are singled out because `traceability.md` points major figures and tables to `data/trajectories.duckdb`, and the `.npz` trajectory store in `data/trajectories/` supports DB-linked trajectory recovery.

### Important caution

Do not equate `untracked` with `disposable`.

At the time of audit, the nested `orbit-transfer-analysis` repo contains untracked script source files and manuscript figure assets. Those may still be active project work and should not be auto-pruned just because they are not committed.

## Nice To Have

These are worth keeping under a light cleanup policy, but they are not strictly required for the default collocation pipeline.

### Optional but useful project material

- `manuscript-ko/`
- `ko_part2.tex`
- `sessions/`
- `글감/`
- `.claude/`

### Optional analysis and alternate deliverables

- `blade-orbit/`
- `docs/report_comparison/student_package/`

Rationale:

- `blade-orbit/` is needed for BLADE-related comparisons, but the main two-pass collocation stack does not require it.
- `docs/report_comparison/student_package/` looks redundant at first glance, but it is a separate handoff/report bundle, not just junk duplication.

### Optional data variants

These are useful for side analyses, alternative baselines, or archived experiments, but not required for the minimum reusable project state:

- `data/colloc_baseline.duckdb`
- `data/trajectories_new.duckdb`
- `data/trajectories_circular.duckdb`
- `data/trajectories_eccentric.duckdb`
- `data/trajectories_new/`
- `data/trajectories_circular/`
- `data/trajectories_eccentric/`

### Local result folders

- `results/blade_poc/`
- `results/mismatch_thrust/`

These are almost certainly generated, but they may still be useful as quick visual references or checkpoint artifacts. They should not be deleted automatically in a conservative cleanup.

## Irrelevant Or Regenerable

These are cleanup targets in principle because they are generated artifacts, caches, or packaging leftovers.

### Safe categories

- `.DS_Store`
- `*.egg-info/`
- `__pycache__/`
- `*.pyc`
- `*.duckdb.wal`
- LaTeX aux and build artifacts such as `*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out`, `*.toc`, `*.synctex.gz`
- ignored local result directories such as `blade-orbit/scripts/results/`
- BLADE GPU/vendor staging artifacts if present:
  - `blade-orbit/vendor/`
  - `blade-orbit/train.yaml`
  - `blade-orbit/.gpu-farm-last-job`

These are supported both by project structure and by `.gitignore` rules.

## Safe To Prune Now

The following items were present on disk during the audit and are safe to remove immediately:

- `.DS_Store`
- `data/.DS_Store`
- `blade-orbit/.DS_Store`
- `src/orbit_transfer.egg-info/`
- `data/trajectories_circular.duckdb.wal`
- `blade-orbit/scripts/results/`

Rationale:

- they are operating-system clutter, packaging metadata, a database WAL sidecar, or explicitly ignored generated script outputs
- removing them does not erase primary code, primary data, manuscript sources, or nested git history

## Cleanup Performed In This Pass

The following items were actually removed during this cleanup:

- `.DS_Store`
- `data/.DS_Store`
- `blade-orbit/.DS_Store`
- `src/orbit_transfer.egg-info/`
- `data/trajectories_circular.duckdb.wal`
- `blade-orbit/scripts/results/`

No source files, manuscript source files, tracked data snapshots, nested `.git` directories, or active-looking untracked scripts were deleted.

## Manual Review Before Deletion

These may look redundant or generated, but they should be reviewed by a human before pruning.

### Active-looking untracked source files

Examples observed during audit:

- `scripts/run_colloc_baseline.py`
- `scripts/run_sampling.py`
- `scripts/run_sampling_extended.py`
- multiple plotting and inspection scripts in `scripts/`

These are source files, not clutter.

### Extra manuscript figures not listed in `traceability.md`

Examples observed during audit:

- `manuscript/figures/fig3_blade_classification_da_di.pdf`
- `manuscript/figures/fig4_blade_classification_da_T.pdf`
- `manuscript/figures/fig5_blade_classification_di_T.pdf`
- `manuscript/figures/fig_blade_*.pdf`
- `manuscript/figures/fig_compare_*.pdf`

These are likely generated, but they may still be part of an in-progress paper revision.

### Local result folders

- `results/blade_poc/`
- `results/mismatch_thrust/`

### Optional subprojects and bundles

- `blade-orbit/`
- `docs/report_comparison/student_package/`

### Alternate data snapshots

- `data/colloc_baseline.duckdb`
- `data/trajectories_new.duckdb`
- `data/trajectories_circular.duckdb`
- `data/trajectories_eccentric.duckdb`

These are not clutter by default. They are optional only relative to the minimum reusable project.

## Special Note On `blade-orbit`

`blade-orbit/` is a nested git repository and should be treated as a separate subproject.

Status:

- not required for the default `orbit_transfer` two-pass collocation pipeline
- required for BLADE-specific workflows and `bezier_orbit` imports
- contains its own source tree, tests, docs, and repo history

Conclusion:

- keep `blade-orbit/` under light cleanup
- prune only its clearly generated local artifacts
- do not remove `blade-orbit/.git`

## Special Note On `docs/report_comparison/student_package`

This directory is not just casual duplication. It is a report-oriented bundle with its own README and user-facing structure.

Conclusion:

- keep it unless you explicitly decide to abandon that handoff/report deliverable
- do not auto-delete it during routine cleanup

## Suggested First-Run Path For A New Agent

If a later user or agent needs to work from this folder efficiently:

1. Read `README.md` for project scope.
2. Read `traceability.md` to see which data drives which paper outputs.
3. Check `pyproject.toml` and `environment.yml` before running anything.
4. Start in `src/orbit_transfer/types.py`, `src/orbit_transfer/optimizer/two_pass.py`, and `src/orbit_transfer/collocation/`.
5. Use `scripts/run_single_case.py` or `scripts/run_comparison.py` for quick execution.
6. Touch `blade-orbit/` only if the task explicitly involves BLADE or `bezier_orbit`.

## Cleanup Result Policy

If you continue pruning after this audit:

1. delete only items listed in `Safe To Prune Now`
2. document any additional deletion in this file before removing it
3. never prune source, data, or manuscript assets solely because they are untracked

