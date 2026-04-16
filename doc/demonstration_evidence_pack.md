# Demonstration Evidence Pack

This document is the merge-ready output of the demonstration-evidence workstream. It freezes the `T2` schema, records the current representative settings, and constrains `F3` so it cannot drift into an untraceable gallery.

## Scope

This pack owns:

- `T2. Demonstration outcome summary`
- `F3. Representative optimized trajectories for selected settings`

It does not own the subdivision ablation or the downstream comparison.

## Frozen reporting policy

The demonstration claim is evaluated under one fixed scenario:

- simplified orbital-transfer case from `Orbital_Docking_Optimizer.py`
- same start/end geometry used in the current cached orbital summary dataset
- current phase-lag setting: `120 deg`
- same KOZ radius and transfer time as recorded in `doc/paper_execution_state.md`

The demonstration table must report:

- `solve_success`
- `safety_margin_km`
- `dv_proxy_m_s`
- `runtime_s`
- `outer_iterations`

Definitions are inherited from `doc/paper_execution_state.md`.

Failure policy:

- if a selected setting returns valid metadata but is infeasible, keep the row and show `solve_success = false`
- if a selected setting has no valid metadata, report it explicitly rather than omitting it

## Representative settings selected for T2 and F3

The current representative settings are:

- `demo_N6_seg16`
- `demo_N7_seg16`
- `demo_N8_seg16`

Why these settings were selected:

- one representative feasible case per supported curve order in the active higher-order regime
- common subdivision count for clean cross-row comparison under velocity boundary conditions
- `n_seg = 16` is the current boundary-conditioned representative setting, not a claim of optimality

This is a curation choice, not a claim that `n_seg = 16` is uniquely best.

## T2 draft

Source: `artifacts/paper_artifacts/orbital_results_summary.csv`


| Setting        | Degree | `n_seg` | Solve success | Safety margin (km) | Objective-aligned effort `dv_proxy` (m/s) | Runtime (s) | Outer iterations |
| -------------- | ------ | ------- | ------------- | ------------------ | ----------------------------------------- | ----------- | ---------------- |
| `demo_N6_seg16` | 6      | 16      | True          | 14.452             | 6693.886                                  | 34.418      | 10000            |
| `demo_N7_seg16` | 7      | 16      | True          | 13.613             | 6411.942                                  | 49.868      | 10000            |
| `demo_N8_seg16` | 8      | 16      | True          | 13.035             | 6286.886                                  | 59.891      | 10000            |


## T2 interpretation boundary

Safe reading:

- the current framework can produce feasible trajectories for the demonstrated setting across the active `N = 6, 7, 8` range with velocity boundary conditions enforced under the refreshed Rust-backed run at 120-deg phase lag
- the current dataset supports quantitative reporting of feasibility, effort, runtime, and iteration count rather than relying on pictures alone
- safety margins of ~13–14.5 km confirm the KOZ constraint is active and respected in all three cases
- all three degrees reach the 10000-iteration cap; `dv_proxy` monotonically decreases with degree while runtime monotonically increases

Unsafe reading:

- broad validation across domains
- general continuous-safety proof beyond the spherical-KOZ assumptions
- downstream warm-start value inferred from trajectory appearance alone
- blanket superiority of higher degree
- the assumption that warm-start value follows automatically from these upstream trajectories

## F3 curation rule

Every `F3` panel must map directly to one `T2` row.

Current panel mapping:

- panel A -> `demo_N6_seg16`
- panel B -> `demo_N7_seg16`
- panel C -> `demo_N8_seg16`

### Preferred source assets

The current paper-facing draft asset is:

- `figures/f3_representative_settings.png`

It should be composed from the dedicated representative exports:

- `figures/demo_N6_seg16.png`
- `figures/demo_N7_seg16.png`
- `figures/demo_N8_seg16.png`

If the panel composition is regenerated later, it should either:

1. use the dedicated `demo_N*_seg16.png` exports, or
2. regenerate a dedicated three-panel figure from the cached solutions using the same settings.

### Caption draft

`Representative optimized trajectories for selected settings in the demonstration problem. Each panel shows the trajectory geometry relative to the spherical keep-out zone for one representative feasible setting, and the corresponding quantitative outcomes are reported in T2. The figure is illustrative support for the demonstration claim, not standalone evidence.`

## Open follow-up for the final paper figure

- review the current draft export `figures/f3_representative_settings.png` and replace it only if the crop quality is not publication-ready
- keep the panel labels aligned with the `T2` row IDs above

