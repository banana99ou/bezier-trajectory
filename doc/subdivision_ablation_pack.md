# Subdivision Ablation Pack

This document is the merge-ready output of the primary ablation workstream. It freezes the subdivision-count protocol for the current orbital dataset, defines the paper-safe reading of `reduced conservatism`, and records the current `T3` draft plus the `F4` panel specification.

## Scope

This pack owns:

- `T3. Subdivision-count ablation`
- `F4. Runtime and outcome trends versus subdivision count`

## Fixed protocol for the current T3 / F4 branch

Baseline setting:

- degree: `N = 7`
- objective: `dv`
- subdivision sweep: `n_seg = [2, 4, 8, 16, 32, 64]`
- same scenario, same endpoint setup, same solver settings, same reporting metrics across the full sweep

Why `N = 7` is the current baseline:

- it is the middle order in the active `N = 6, 7, 8` study
- it avoids tying the primary subdivision claim to the lowest-order or highest-order edge case

This is a branch baseline, not a claim that seventh degree is intrinsically privileged.

## Operational definition of reduced conservatism

For the present paper, `reduced conservatism` must be interpreted through reported metrics under fixed settings. It must not mean only that a path looks tighter.

The safe operational reading is:

- whether increasing `n_seg` preserves comparable certified clearance while reducing objective-aligned effort, or improves both, in a way consistent with a less conservative approximation
- while also changing computation burden

Current empirical reality from the refreshed full `N = 7` sweep under the `120 deg` phase-lag scenario:

- `n_seg = 2` is infeasible (negative safety margin); the remaining five counts (`n_seg = 4, 8, 16, 32, 64`) produce feasible solutions
- `safety_margin_km` drops monotonically from `144.1` (n_seg=4) to `0.9` (n_seg=64) — the trajectory approaches the KOZ boundary as subdivision refines the approximation
- `dv_proxy_m_s` decreases from `9,288` (n_seg=4) to `6,292` (n_seg=64), a ~1.5x improvement, with the steepest gains between n_seg=4 and n_seg=16
- `runtime_s` ranges from `34` (n_seg=8) to `145` (n_seg=64); all runs reach the 10000-iteration cap

Therefore the current dataset supports a clear conservatism-reduction claim for `n_seg >= 4`: finer subdivision produces a tighter (less conservative) KOZ approximation, which both allows the trajectory closer to the boundary and reduces objective-aligned effort, at the cost of increased computation time. The `n_seg = 2` infeasibility shows that too-coarse subdivision fails to produce a feasible trajectory in this geometry.

## T3 draft

Source: `artifacts/paper_artifacts/orbital_results_summary.csv`

| Degree | `n_seg` | Solve success | Safety margin (km) | Objective-aligned effort `dv_proxy` (m/s) | Runtime (s) | Outer iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 2 | False | −284.085 | 67357.586 | 22.391 | 10000 |
| 7 | 4 | True | 144.056 | 9288.199 | 54.521 | 10000 |
| 7 | 8 | True | 57.034 | 6831.697 | 33.621 | 10000 |
| 7 | 16 | True | 13.613 | 6411.942 | 49.868 | 10000 |
| 7 | 32 | True | 3.600 | 6315.435 | 78.790 | 10000 |
| 7 | 64 | True | 0.892 | 6291.595 | 144.607 | 10000 |

## T3 interpretation boundary

Safe reading:

- `n_seg = 2` is infeasible, demonstrating that too-coarse subdivision cannot guarantee a feasible trajectory in this geometry
- for the five feasible settings (`n_seg >= 4`), the subdivision sweep shows a clear monotone conservatism-reduction trend: safety margin decreases from ~144 km to ~1 km, indicating the trajectory approaches the KOZ boundary more closely
- simultaneously, the objective-aligned effort decreases ~1.5x from `n_seg = 4` to `n_seg = 64`, with the steepest gains between `n_seg = 4` and `n_seg = 16`
- all runs reach the 10000-iteration cap; runtime ranges from ~34 s to ~145 s across the feasible settings
- the trend is scenario-specific: it relies on the 120-deg phase-lag geometry that produces a trajectory approaching the KOZ

Unsafe reading:

- the current table proves a universal monotone trade-off across all scenarios
- the trend must persist for all orbital geometries or problem classes
- the coarsest n_seg settings are sufficient because they are feasible

## F4 panel specification

Current draft asset:

- `figures/f4_subdivision_tradeoff_N7.png`

### Required panels

1. Runtime versus `n_seg`
2. Objective-aligned effort `dv_proxy_m_s` versus `n_seg`

Optional panel only if it adds real information:

3. Safety margin versus `n_seg`

Given the current data, the safety-margin panel is high value: it shows a clear monotone decrease from ~145 km to ~1 km, directly visualizing the conservatism-reduction effect.

### Caption draft

`Runtime and outcome trends versus subdivision count for the seventh-degree baseline setting under the 120-deg phase-lag scenario. n_seg=2 is infeasible; for n_seg >= 4, the safety margin decreases steadily from ~144 km to ~1 km and the objective-aligned effort drops ~1.5x, indicating that finer subdivision produces a less conservative KOZ approximation that allows closer approach and lower control effort. Runtime ranges from ~34 s to ~145 s across the feasible sweep. All runs reach the 10000-iteration cap.`

## Immediate follow-up for the final paper figure

- regenerate `figures/f4_subdivision_tradeoff_N7.png` from the refreshed 120-deg cache — the safety-margin panel should be kept as it now shows a clear monotone trend
- keep the interpretation text tied to the monotone conservatism-reduction trend and the computation-cost trade-off
