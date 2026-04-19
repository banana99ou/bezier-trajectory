# DCM Experiment — Session Summary

Context for resuming work on the Bézier-vs-DCM downstream experiment in a fresh chat.

## The project

Two subprojects coexist in this repo:

- **`orbital_docking/`** — Bézier SCP optimizer for orbital trajectories (Rust backend via `bezier_opt` pybind). Originally built for Progress→ISS rendezvous (fixed 1500s transfer). Generalized during this session.
- **`orbit-transfer-analysis/`** — DCM (Direct Collocation Method) two-pass pipeline (Hermite-Simpson → peak detect → Multi-Phase LGL) for parametric LEO-to-LEO transfers. Builds a trajectory database at `data/trajectories.duckdb` (220 converged cases, 29 failed).

## The original question

Professor's request: compare `DB seed → DCM → output` (baseline) vs `DB seed → my optimizer → DCM → output` (proposed). Document in `doc/dcm_db_experiment_note.md`.

## What we built

### Files created / modified

| File | Purpose |
|------|---------|
| `CLAUDE.md` (repo root) | Permission rules: no commit/edit without explicit request |
| `doc/dcm_downstream_experiment_design.md` | Experiment design + revision history |
| `doc/dcm_experiment_findings.md` | Findings 1–4 (the scientific record) |
| `tools/dcm_downstream_experiment.py` | The experiment runner |
| `tools/dcm_visualize_case.py` | Bézier/DCM trajectory visualization (Plotly HTML) |
| `orbital_docking/optimization.py` | Added `transfer_time` parameter (was hardcoded 1500s) |
| `rust_optimizer/pybind/src/lib.rs` | Added `transfer_time` parameter in pyo3 signature |

### Key git commit

`cc4e9d8` (2026-04-17) — "feat: DCM downstream experiment with Bézier Pass 1 replacement" contains the Pass 1 replacement pipeline, plus Finding 1–3 docs. Finding 4 and CLAUDE.md are uncommitted as of the session end.

## The four findings

### Finding 1: Bézier fails for multi-revolution transfers

A degree-N Bézier is a single polynomial arc. With `T_normed > ~0.5`, endpoints span more than 180° of orbital arc, and the curve's convex hull includes Earth's interior. Even with control points on a valid orbit (6778 km), the polynomial curve dips through Earth. The SCP optimizer "solves" this by pushing middle control points to 13 million km (2× Earth-Moon distance) — feasible but physically meaningless.

**Cutoff**: Bézier is usable only for `T_normed < ~0.5` (single-arc, sub-orbital transfers).

### Finding 2: Direct Bézier warm-start does not help DCM

For short-transfer cases (`T_normed ≤ 0.5`), Bézier upstream is feasible. But using its sampled `(t, x, u)` as a warm-start for the full two-pass DCM:

- Both pipelines converge to the same cost (machine precision)
- Proposed pipeline is **8–33× slower** than baseline

Two reasons: (1) Bézier uses fixed `ν₀=0, νf=π` while DCM optimizes ν as free variables — warm-start trajectory starts at the wrong orbital position; (2) DCM's own initialization is already well-engineered.

### Finding 3: Bézier *can* replace Pass 1 (WORKS)

Pass 1 (Hermite-Simpson) exists only to find the thrust peak structure for Pass 2. Bézier produces a thrust profile in ~0.1s. Feed it through `detect_peaks()` → `determine_phase_structure()` → Pass 2 directly.

Results on 4 short-transfer cases:

| Case | T_norm | Baseline | Proposed | Speedup |
|------|--------|----------|----------|---------|
| 2    | 0.280  | 0.91s    | 0.42s    | **2.2×** |
| 4    | 0.280  | 2.36s    | 5.63s    | 0.4× (slower) |
| 6    | 0.280  | 1.17s    | 0.57s    | **2.1×** |
| 126  | 0.500  | 1.11s    | 0.84s    | **1.3×** |

3 of 4 cases faster, **identical final cost**. Case 4 slower because Bézier detected 2 peaks vs baseline's 1 — more phases = more Pass 2 work.

### Finding 4: Bézier does NOT rescue DCM-failed cases

Ran Pass 1 replacement on all 29 DB rows with `converged=FALSE`:

| Outcome | Count |
|---------|-------|
| Rescued (proposed wins where baseline fails) | **0** |
| Baseline now converges (solver drift since DB build) | 14 |
| Both still fail | 15 |
| Bézier upstream feasible | 4 of 29 |

25 of 29 failed cases have `T_normed ≥ 2` — beyond Bézier's reach (Finding 1). The remaining few are genuinely hard (`da = -500 km` hitting altitude floor, extreme `delta_a`). Even when Bézier upstream is feasible, Pass 2 can't converge.

## The 5 options considered (for context)

At one point we brainstormed 5 directions:

1. **Use DCM's optimal ν₀/νf as Bézier endpoints** — circular (needs DCM result to start)
2. **Grid search over (ν₀, νf) in Bézier** — cheap, addresses Finding 2 endpoint mismatch
3. **Target DCM-failed cases** — tried (Finding 4), dead end
4. **Bézier as Pass 1 replacement** — tried (Finding 3), works
5. **Monte Carlo perturbations** — not tried, tests robustness

## Current state

- Option 4 works but tested on only 4 cases — needs broader validation to characterize speedup distribution across the parameter space
- Option 3 is closed (Finding 4)
- Options 1, 2, 5 untried

## Suggested next directions

### Option A: Scale up option 4
Run Pass 1 replacement on ALL 220 converged cases. Characterize speedup distribution, find the regimes where it helps most / least. This is the cleanest path to a publishable result.

Command template:
```bash
.venv/bin/python tools/dcm_downstream_experiment.py \
    --converged converged --max-t-normed 0.5 \
    --outdir results/dcm_pass1_replace_full
```
(Remove `--max-t-normed` to include all cases and see Bézier failure pattern too.)

### Option B: Try option 2 (grid search over ν)
If Finding 2's endpoint mismatch was the real blocker, grid-searching ν₀/νf in the Bézier optimizer (64 runs × 0.1s = 6s per case) could produce warm-starts that actually align with DCM's preferred geometry. Would need a new `bezier_warm_start_grid()` function.

### Option C: Try option 5 (Monte Carlo robustness)
Frame the comparison as: across N random perturbations of the initial guess, which pipeline (baseline DCM with jittered guess vs Bézier-Pass2 with jittered Bézier init) converges more reliably? Tests robustness rather than average-case.

### Option D: Piecewise Bézier for multi-rev transfers
Finding 1 blocks 25 of 29 failed cases. A piecewise Bézier (K arcs, each covering < 180°) could handle multi-rev. But this is a significant optimizer rewrite — out of scope of the current "thin wrapper" constraint.

## Key file locations

### Code
- Experiment runner: `tools/dcm_downstream_experiment.py`
  - `bezier_warm_start()` — generates (t, x, u) from Bézier SCP, uses departure-orbit Kepler propagation for initial control points (not straight line)
  - `run_baseline()` — full two-pass DCM
  - `run_proposed()` — Bézier → peak detect → Pass 2
  - `load_cases()` — DB query with `converged_filter` arg: "converged" | "failed" | "all"
- Visualization: `tools/dcm_visualize_case.py` — produces 3D + radius + thrust plots per case

### Data
- DB: `orbit-transfer-analysis/data/trajectories.duckdb`
  - Column name is `T_normed` (NOT `T_max_normed` — but `TransferConfig` field is `T_max_normed`, map carefully)
  - 220 converged, 29 failed collocation cases
- Results: `results/dcm_pass1_replace/` (option 4 — 4 cases), `results/dcm_failed_rescue/` (option 3 — 29 cases)

### Key APIs reused
- `orbit_transfer.optimizer.two_pass.TwoPassOptimizer` — baseline DCM
- `orbit_transfer.classification.peak_detection.detect_peaks` — peak detection from `(t, ‖u‖, T)`
- `orbit_transfer.classification.classifier.determine_phase_structure` — phases from peaks
- `orbit_transfer.collocation.interpolation.interpolate_pass1_to_pass2` — resample to LGL nodes
- `orbit_transfer.collocation.multiphase_lgl.MultiPhaseLGLCollocation` — Pass 2 directly
- `orbital_docking.optimization.optimize_orbital_docking(P_init, transfer_time=T, ...)` — Bézier SCP
- `orbital_docking.bezier.BezierCurve.point()`, `.velocity()`, `.acceleration()` — sample curve (velocity is in parameter space, divide by T for physical units)

## Non-obvious gotchas

1. **`R_E` mismatch**: `orbital_docking.constants.EARTH_RADIUS_KM = 6371.0` (mean); `orbit_transfer.constants.R_E = 6378.137` (equatorial). Always use `orbit_transfer.R_E` since DCM is the downstream truth.

2. **Bézier velocity scaling**: `BezierCurve.velocity(τ)` returns the parameter-space derivative `dr/dτ`. For physical velocity: `v = dr/dτ / T`.

3. **Thrust reconstruction from Bézier**: `u = r''(τ)/T² - g(r(τ))` where `g = gravity_acceleration` (two-body). The Bézier optimizer uses 2-body + J2 internally, but the thrust computed here is 2-body only — acceptable since the warm-start is refined by Pass 2 anyway.

4. **Rust backend rebuild**: if you modify `rust_optimizer/pybind/src/lib.rs`, rebuild with `cd rust_optimizer/pybind && .venv/bin/python -m maturin develop --release` — the Python venv is at `/Volumes/Sandisk/code/bezier-trajectory/.venv/`.

5. **Initial control points for Bézier**: straight-line interpolation between `r0` and `rf` goes through Earth when `ν₀=0, νf=π`. Use departure-orbit Kepler propagation instead: `P_init[j] = kepler_propagate(r0, v0, j/N * T, MU_EARTH)`. Already implemented in `bezier_warm_start()`.

6. **DB column vs TransferConfig field**: DB has `T_normed`, `TransferConfig` has `T_max_normed`. Map in `load_cases()`:
   ```python
   rid, rh0, rda, rdi, rtmax, re0, ref = row
   config=TransferConfig(..., T_max_normed=float(rtmax), ...)
   ```

7. **Python environment**: use `.venv/bin/python` explicitly. System python lacks `duckdb`, `casadi`.

8. **Permission rule (from CLAUDE.md)**: do not commit or edit files without explicit instruction. "Do X" is not permission to commit.

## How to resume

In the new chat, open with:

> Read `doc/dcm_experiment_session_summary.md` for full context. We were exploring whether the Bézier SCP optimizer can improve DCM's downstream performance. Options 3 and 4 are explored. I want to [pick one: scale up option 4 / try option 2 / try option 5 / something else].
