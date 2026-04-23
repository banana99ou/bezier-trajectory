## Bézier Trajectory – Orbital Rendezvous Optimizer

Bézier-curve based trajectory optimizer for a simplified orbital rendezvous scenario (Progress-to-ISS inspired). The optimizer minimizes a delta-v surrogate (control acceleration minus gravity, including J2) subject to an Earth-centric spherical Keep‑Out Zone (KOZ), with the KOZ nonconvex constraint convexified segment-by-segment via De Casteljau subdivision.

See `Project_Spec.md` for the underlying math (scaling, D/E/G matrices, objective, KOZ convexification) and `doc/` for the paper drafts, evidence packs, and experiment design notes that this code backs.

## Repository layout

- **`orbital_docking/`** — Python package: Bézier primitives, D/E/G matrices, De Casteljau subdivision, KOZ + boundary constraints, SCP loop (`optimization.py`), caching, visualization. Also hosts `downstream_collocation.py` and `dymos_t6.py` for comparisons against collocation.
- **`rust_optimizer/`** — Rust workspace with a core crate (`core/`) and PyO3 bindings (`pybind/`, Python module `bezier_opt`). Optional fast backend; Python is the reference implementation. See `rust_optimizer/QA_strategy.md` for cross-validation rules.
- **`Orbital_Docking_Optimizer.py`** — main entry point; wraps the `orbital_docking` package.
- **`sweep_koz_ablation.py`** — KOZ-altitude sweep for the subdivision-count ablation study (N=7).
- **`tools/`** — A/B benchmarks, diagnostics, Dymos comparison, J2 validation, DCM downstream experiment, figure/CSV builders.
- **`tests/`** — `unit/`, `integration/`, `regression/`, `property/` splits; conftest under `tests/conftest.py`.
- **`orbit-transfer-analysis/`** — sibling project absorbed into this repo for the DCM (direct collocation) pipeline used in downstream comparisons.
- **`doc/`** — paper drafts (Korean), evidence/execution tracking, experiment design notes.
- **`figures/`**, **`artifacts/`**, **`cache/`**, **`results/`** — generated outputs. Most are gitignored.
- **`archive/`** — legacy single-curve sphere-avoidance scripts kept for reference.

## Install

Python dependencies:

```bash
pip install -r requirements.txt
```

The scripts target Python 3.11+. The repo-local `.venv/` is a Python 3.11 environment that already has the Rust extension installed; system Python will work for the pure-Python path but needs `maturin develop` (below) if you want the Rust backend.

Optional Rust backend (requires a Rust toolchain and `maturin`):

```bash
cd rust_optimizer/pybind
maturin develop --release
```

This builds and installs the `bezier_opt` extension into the active venv.

## Run the main optimizer

```bash
python Orbital_Docking_Optimizer.py
```

Useful flags:

- `-N {6,7,8} [...]` — Bézier degrees to run (default `6 7 8`).
- `--objective {dv,energy}` — L1-style delta-v proxy (default) or legacy L2 control-energy.
- `--no-cache` — ignore existing pickled results under `cache/` and recompute.
- `--max-iter N`, `--tol T` — SCP outer-loop budget / convergence tolerance.
- `--scp-prox`, `--scp-trust-radius` — outer-loop stabilization (proximal weight + trust radius).
- `--enforce-prograde` — add prograde-motion constraint.
- `--n-jobs K` — parallelize across segment counts (`K=0` for auto).
- `--show` — display figures interactively (default: save to `figures/` only).
- `-v` / `-d` — verbose / debug output.

Outputs are written under `figures/` (plots) and `cache/` (pickled optimization results).

## KOZ-altitude subdivision ablation

```bash
python sweep_koz_ablation.py
```

Sweeps KOZ altitudes at N=7 across all segment counts to isolate regimes where subdivision count actually matters. Results feed `doc/subdivision_ablation_pack.md` and `doc/degree_ablation_pack.md`.

## Diagnostic, benchmark, and comparison tools

All live under `tools/`. Representative entry points:

- **A/B endpoint-feasibility fix** — `tools/ab_endpoint_fix_parallel.py`, CSV + Markdown reports into `artifacts/ab_tests/`. Example:
  ```bash
  python tools/ab_endpoint_fix_parallel.py --objective dv --max-iter 1000 --tol 1e-3 --workers 32
  python tools/ab_endpoint_fix_parallel.py --orders 2 3 4 --seg-counts 2 4 8 16 32 64
  ```
- **Convergence diagnostic** — `tools/convergence_diagnostic.py`.
- **J2 validation** — `tools/fetch_j2_reference_data.py`, `tools/verify_j2_logic.py`.
- **Dymos T6 comparison** — `tools/t6_dymos_compare.py`, `tools/t6_dymos_time_sweep.py`, evidence pack builder `tools/t6_evidence_pack.py`.
- **DCM downstream experiment** — `tools/dcm_downstream_experiment.py` (Bézier as Pass‑1 replacement for a Hermite–Simpson → LGL pipeline from `orbit-transfer-analysis/`). Design: `doc/dcm_downstream_experiment_design.md`.
- **Boundary-condition feasibility diagnostics** — `tools/diagnose_n3_bc_feasibility.py`, `tools/diagnose_boundary_velocity_cases.py`.
- **Paper figure/CSV builders** — `tools/build_csv.py`, `tools/build_f3.py` … `tools/build_f5.py`, `tools/build_t6_f6.py`, `tools/extract_pngs_to_artifacts.py`.

Each tool is self-contained with `--help`.

## Tests

```bash
pytest
```

Layout under `tests/`:
- `unit/` — Bézier primitives, D/E/G, De Casteljau, gravity + J2, constraints, downstream collocation.
- `integration/` — full SCP feasibility and KOZ satisfaction.
- `regression/` — golden-run and J2 baseline snapshots.
- `property/` — property-based checks.

Regression snapshots pin the current physics + numerics; a diff there means either an intentional change or a bug.

## Papers and evidence

- **`Project_Spec.md`** — problem statement, coordinate frame, Bézier derivative matrices, KOZ convexification, objective.
- **`doc/paper_draft_korean.md`**, **`doc/paper_draft_korean.pdf`** — current manuscript.
- **`doc/paper_evidence_map.md`**, **`doc/paper_execution_state.md`** — claim → evidence mapping and status.
- **`doc/method_artifact_pack.md`**, **`doc/demonstration_evidence_pack.md`**, **`doc/degree_ablation_pack.md`**, **`doc/subdivision_ablation_pack.md`**, **`doc/dcm_downstream_pack.md`** — artifact packs per figure/table.

## Scenario (summary)

- Chaser (Progress-like) at 245 km circularized parking orbit.
- Target (ISS-like) at 400 km circular orbit.
- Inclination 51.64°, RAAN 0°, 30° phase lag.
- KOZ: Earth-centric sphere at 100 km altitude (`‖r(τ)‖ ≥ R_E + 100 km`).
- Fixed transfer time `T` (variable-time extension is future work).

Full derivation of endpoints and matrices is in `Project_Spec.md`.
