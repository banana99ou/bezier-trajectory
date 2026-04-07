# Subdivision Ablation Pack

This document is the merge-ready output of the primary ablation workstream. It freezes the subdivision-count protocol for the current orbital dataset, defines the paper-safe reading of `reduced conservatism`, and records the current `T3` draft plus the `F4` panel specification.

## Scope

This pack owns:

- `T3. Subdivision-count ablation`
- `F4. Runtime and outcome trends versus subdivision count`

## Fixed protocol for the current T3 / F4 branch

Baseline setting:

- degree: `N = 6`
- objective: `dv`
- subdivision sweep: `n_seg = [2, 4, 8, 16, 32, 64]`
- same scenario, same endpoint setup, same solver settings, same reporting metrics across the full sweep

Why `N = 6` is the current baseline:

- it is the middle order in the active `N = 5, 6, 7` study
- it avoids tying the primary subdivision claim to the lowest-order or highest-order edge case

This is a branch baseline, not a claim that sixth degree is intrinsically privileged.

## Operational definition of reduced conservatism

For the present paper, `reduced conservatism` must be interpreted through reported metrics under fixed settings. It must not mean only that a path looks tighter.

The safe operational reading is:

- whether increasing `n_seg` preserves comparable certified clearance while reducing objective-aligned effort, or improves both, in a way consistent with a less conservative approximation
- while also changing computation burden

Current empirical reality from the refreshed full `N = 6` sweep:

- `safety_margin_km` stays effectively constant at about `145 km`
- `dv_proxy_m_s` changes only slightly across the full sweep
- `runtime_s` increases substantially with larger `n_seg`

Therefore the current dataset supports `runtime cost growth` much more clearly than a strong effective-conservatism improvement claim.

## T3 draft

Source: `artifacts/paper_artifacts/orbital_results_summary.csv`

| Degree | `n_seg` | Solve success | Safety margin (km) | Objective-aligned effort `dv_proxy` (m/s) | Runtime (s) | Outer iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 2 | True | 145.000 | 17730.286 | 3.123 | 500 |
| 6 | 4 | True | 145.000 | 17866.995 | 3.334 | 500 |
| 6 | 8 | True | 145.000 | 17898.397 | 3.892 | 500 |
| 6 | 16 | True | 145.000 | 17905.937 | 5.063 | 500 |
| 6 | 32 | True | 145.000 | 17905.284 | 7.075 | 500 |
| 6 | 64 | True | 145.000 | 17906.558 | 11.511 | 500 |

## T3 interpretation boundary

Safe reading:

- the current subdivision sweep shows a clear computation-burden increase as `n_seg` grows
- the current refreshed higher-order dataset does not show a strong monotone safety-margin change under the chosen metric
- the objective-aligned effort is nearly flat across the full sweep, so any claimed effective-conservatism benefit remains weak under the current orbital metric set
- any claim of reduced conservatism should therefore remain cautious and scenario-specific

Unsafe reading:

- higher subdivision is always better
- the current table proves a universal monotone trade-off
- visual path proximity alone is evidence of reduced conservatism

## F4 panel specification

Current draft asset:

- `figures/f4_subdivision_tradeoff_N6.png`

### Required panels

1. Runtime versus `n_seg`
2. Objective-aligned effort `dv_proxy_m_s` versus `n_seg`

Optional panel only if it adds real information:

3. Safety margin versus `n_seg`

Given the current data, the safety-margin panel is likely low value because the metric is effectively flat.

### Caption draft

`Runtime and outcome trends versus subdivision count for the sixth-degree baseline setting. Runtime increases clearly as the subdivision-based safety approximation is refined, while the current objective-aligned effort metric remains nearly flat and the sampled safety margin stays effectively constant. The present higher-order orbital dataset therefore supports a computation-burden trade-off more clearly than a strong effective-conservatism claim.`

## Immediate follow-up for the final paper figure

- review `figures/f4_subdivision_tradeoff_N6.png` and decide whether the paper should keep the safety-margin overlay or simplify to a strict two-panel runtime-plus-effort figure
- keep the interpretation text tied to the observed flat safety margin and nearly flat `dv_proxy` values
