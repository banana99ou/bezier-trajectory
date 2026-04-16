# DCM Downstream Experiment Design

Supersedes the preliminary notes in `dcm_db_experiment_note.md`.

## Goal

Test whether using the Bézier SCP optimizer as an upstream warm-starter improves the downstream DCM (two-pass collocation) performance, compared to DCM running with its own naive initial guess.

The claim is deliberately narrow: identify a subset of cases where the upstream optimizer helps, and report that subset honestly — including cases where it does not help.

## Pipelines

### Baseline

```
TransferConfig
  → linear_interpolation_guess()          # cosine-blended Kepler orbits
  → TwoPassOptimizer.solve()              # H-S → peak detect → LGL
  → result
```

### Proposed

```
TransferConfig
  → oe_to_rv() for departure/arrival      # ECI endpoints (r0, v0, rf, vf)
  → generate_initial_control_points()     # straight-line Bézier P_init
  → optimize_orbital_docking()            # Bézier SCP with KOZ
  → sample Bézier curve to (t, x, u)     # interface bridge
  → TwoPassOptimizer.solve(x_init, u_init, t_init)
  → result
```

Both pipelines start from the same `TransferConfig` (case parameters). No stored DB trajectories are needed as seeds — only the case parameters `(h0, delta_a, delta_i, T_max_normed, e0, ef)`.

## Code Mapping

| Experiment concept    | Code                                                                 |
|-----------------------|----------------------------------------------------------------------|
| Case parameters       | `orbit-transfer-analysis/data/trajectories.duckdb` (converged rows)  |
| TransferConfig        | `orbit-transfer-analysis/src/orbit_transfer/types.py`                |
| DCM naive guess       | `orbit-transfer-analysis/src/orbit_transfer/optimizer/initial_guess.py` → `linear_interpolation_guess()` |
| DCM solver            | `orbit-transfer-analysis/src/orbit_transfer/optimizer/two_pass.py` → `TwoPassOptimizer` |
| Upstream optimizer    | `orbital_docking/optimization.py` → `optimize_orbital_docking()`     |
| Bézier initialization | `orbital_docking/optimization.py` → `generate_initial_control_points()` |
| ECI conversion        | `orbit-transfer-analysis/src/orbit_transfer/astrodynamics/orbital_elements.py` → `oe_to_rv()` |

## Interface Bridge

The Bézier optimizer and DCM speak different representations. A thin conversion layer is needed.

### Bézier optimizer input (from TransferConfig)

1. Convert orbital elements to ECI:
   - `oe_to_rv((a0, e0, i0, 0, 0, nu0), MU)` → `r0, v0`
   - `oe_to_rv((af, ef, if_, 0, 0, nuf), MU)` → `rf, vf`
2. `P_init = generate_initial_control_points(degree, r0, rf)` — straight-line control points
3. `v0, v1` boundary conditions from the orbital velocities, scaled by `T`
4. `r_e = R_E + h_min` as the KOZ radius
5. Transfer time `T = config.T_max` (not the hardcoded 1500 s)

### Bézier optimizer output → DCM input

Given optimized control points `P_opt` (shape `(N+1, 3)`):

1. Sample at `M` uniformly spaced `τ_k ∈ [0, 1]`:
   - `r(τ_k)` — position from Bézier evaluation
   - `v(τ_k) = r'(τ_k) / T` — velocity from first derivative
   - `a(τ_k) = r''(τ_k) / T²` — geometric acceleration from second derivative
2. Compute thrust: `u(τ_k) = a(τ_k) - g(r(τ_k))` where `g` is gravity at that position
3. Assemble: `t_init = τ_k * T`, `x_init = [r; v]` (6×M), `u_init` (3×M)
4. Pass to `TwoPassOptimizer.solve(x_init=x_init, u_init=u_init, t_init=t_init)`

## Metrics

| Metric                  | Source                     |
|-------------------------|----------------------------|
| Converged               | `TrajectoryResult.converged` |
| Cost (L2)               | `TrajectoryResult.cost`    |
| Solve time              | wall-clock `perf_counter`  |
| Total ΔV                | integrated `‖u‖` over time |
| Min altitude margin     | `min(‖r‖) - (R_E + h_min)` |
| Boundary condition error | `‖x(0) - x_ref‖`, `‖x(T) - x_ref‖` |
| Peak count / profile class | from peak detection     |
| Iteration count         | IPOPT stats                |

Record the upstream Bézier optimizer metrics separately:
- Bézier converged, cost, elapsed time, feasibility, iterations

## Fairness Rules

- Downstream `TwoPassOptimizer` settings (IPOPT tolerances, segment counts, phase structure logic) must be identical for both pipelines.
- Both pipelines use the same `TransferConfig` for the same case.
- Failures (either upstream or downstream) are retained and reported, never filtered.
- Metrics are computed the same way for both.

## Case Selection

Source: converged rows from `trajectories.duckdb` where `method = 'collocation'`.

Start with simple cases first:
- Circular-to-circular (`e0 ≈ 0, ef ≈ 0`)
- Small inclination change (`delta_i < 5°`)
- Single altitude slice (e.g., `h0 = 400 km`)

Expand to harder cases once the bridge is validated.

## Open Questions

1. **True anomaly**: DCM treats `nu0, nuf` as free decision variables. The Bézier optimizer takes fixed endpoints. Options:
   - Fix `nu0 = 0, nuf = π` (same defaults as `linear_interpolation_guess`)
   - Search over a small grid of `(nu0, nuf)` pairs
   - Use the DB's stored optimal `nu0, nuf` as hints

2. **Eccentricity**: The Bézier optimizer demo uses circular orbits. For `e ∈ [0, 0.1]`, the position endpoints still work, but velocity BCs change. Should be fine — `oe_to_rv` handles it.

3. **Bézier degree and segments**: Which `(N, n_seg)` to use? Candidates: `N=6, n_seg=16` (well-tested in the docking problem).

4. **Gravity model mismatch**: Bézier optimizer uses 2-body + J2 linearization. DCM uses full 2-body dynamics via CasADi. The mismatch is acceptable — the Bézier output is only a warm-start, not the final answer.

## Design Decisions

- **BLADE is not part of this experiment.**
- **`scripts/run_dcm_db_experiment.py` is not reused** — build the experiment script fresh.
- **No rewrite of the Bézier optimizer** — only a thin wrapper to handle `TransferConfig → endpoints` and `control points → (t, x, u)`.
- The experiment script should live at repo root level (e.g., `tools/` or `scripts/`) since it bridges both subprojects.
