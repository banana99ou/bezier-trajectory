# Chat Record: N=3 Feasibility Diagnostics

Date: 2026-02-22

## User questions

1. Why does `N=3` not produce viable solutions?
2. Why does control acceleration not decrease with increasing `N_seg`?

## What was reviewed

- `README.md`
- `Project_Spec.md`
- Core implementation:
  - `orbital_docking/optimization.py`
  - `orbital_docking/constraints.py`
  - `Orbital_Docking_Optimizer.py`
  - `orbital_docking/visualization.py`
  - `orbital_docking/constants.py`

## Initial diagnosis summary

- `N=3` with both endpoint velocity constraints can become too restrictive.
- `N_seg` mainly refines KOZ linearization, not necessarily the control-acceleration metric monotonicity.

## Scripts created during this chat

### 1) `diagnose_boundary_velocity_cases.py`

Purpose:
- Construct and draw boundary-constrained Bézier curves (no optimizer) for:
  - `N=3 current v0+v1`
  - `N=3 TES v0+v1`
  - `N=3 TES v0 only`
  - `N=3 TES v1 only`
  - `N=4 TES v0+v1`

Output:
- Figure: `figures/boundary_velocity_diagnostic.png`

Observed:
- All cases were drawable.
- All were KOZ-violating at the constructed boundary-only stage.

### 2) `diagnose_n3_bc_feasibility.py`

Purpose:
- Run the actual optimizer for `N=3` under four BC modes:
  - `none`
  - `v0_only`
  - `v1_only`
  - `both`
- Compare feasibility over `n_seg = [2, 4, 8, 16, 32, 64]`.

Run used:
- velocity model: `current`
- cache: disabled
- max iter: 120
- tol: `1e-3`

Output:
- Figure: `figures/n3_bc_feasibility_current.png`

## Key result (confirmed)

For `N=3`:

- `none`: feasible across all tested `n_seg`
- `v0_only`: feasible across all tested `n_seg`
- `v1_only`: feasible across all tested `n_seg`
- `both (v0+v1)`: infeasible across all tested `n_seg`

This isolates the issue to boundary-condition tightness for the `N=3` case when both endpoint velocities are enforced.

## Practical conclusion from this chat

- The `N=3` failure is not due to inability to draw or represent a cubic curve.
- The optimizer becomes infeasible when both endpoint velocity BCs are active for this setup.
- Increasing `n_seg` improves KOZ margin but does not recover feasibility in that overconstrained mode.
