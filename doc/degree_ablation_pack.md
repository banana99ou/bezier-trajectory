# Degree Ablation Pack

This document is the merge-ready output of the degree-ablation branch. It keeps `T4` on the main-paper path, freezes the matched comparison protocol, and records the current `F5` draft asset from the refreshed Rust-backed `N = 6, 7, 8` study.

## Scope

This pack owns:

- `T4. Degree ablation under matched settings`
- `F5. Multi-order performance trends`

## Fixed protocol

The active degree comparison is:

- degrees: `N = 6, 7, 8`
- representative matched table setting: `n_seg = 16`
- full trend figure sweep: `n_seg = [2, 4, 8, 16, 32, 64]`
- objective: `dv`
- solver backend: `rust`
- maximum outer iterations: `10000`
- tolerance: `1e-12`
- endpoint velocity constraints: enforced
- same scenario, same endpoint setup, and same reporting metrics across all compared runs

Interpretation rule:

- `T4` answers the fixed-setting degree question at `n_seg = 16`
- `F5` shows how the degree tradeoff interacts with subdivision count
- neither artifact may be used to claim unconditional superiority of higher degree

## T4 draft

Source: `artifacts/paper_artifacts/orbital_results_summary.csv`

Selected smoothness-adjacent reporting column:

- `mean_control_accel_ms2`
- This is a control-effort summary, not a formal smoothness proof. It is included only as an operationally useful secondary indicator.

| Setting | Degree | Control points | `n_seg` | Solve success | Safety margin (km) | Objective-aligned effort `dv_proxy` (m/s) | Runtime (s) | Outer iterations | Mean control accel (m/s^2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `demo_N6_seg16` | 6 | 7 | 16 | True | 14.452 | 6693.886 | 34.418 | 10000 | 4.485 |
| `demo_N7_seg16` | 7 | 8 | 16 | True | 13.613 | 6411.942 | 49.868 | 10000 | 4.330 |
| `demo_N8_seg16` | 8 | 9 | 16 | True | 13.035 | 6286.886 | 59.891 | 10000 | 4.251 |

## T4 interpretation boundary

Safe reading:

- all three degrees are feasible under the matched boundary-conditioned protocol with comparable safety margins (~13–14.5 km)
- `dv_proxy` monotonically decreases with degree: `N = 6` (6,694 m/s) > `N = 7` (6,412 m/s) > `N = 8` (6,287 m/s)
- mean control acceleration also monotonically decreases with degree: 4.485 > 4.330 > 4.251 m/s^2
- runtime monotonically increases with degree: 34.4 s < 49.9 s < 59.9 s
- all three degrees reach the 10000-iteration cap
- the effort differences across degrees are modest (~6% between best and worst), while the runtime difference is ~1.7x
- the current evidence supports a clean expressiveness-versus-computation tradeoff: higher degree improves objective but costs more compute

Unsafe reading:

- higher degree is unconditionally better (the monotone trend is observed in one scenario only)
- the small `dv_proxy` edge for `N = 8` automatically justifies the runtime cost
- the degree comparison alone proves stronger boundary-condition conclusions beyond the tested setup

## F5 draft asset

Current draft asset:

- `figures/f5_multi_order_tradeoff_N678.png`

### Required panels

1. Objective-aligned effort `dv_proxy_m_s` versus `n_seg`, degree-coded
2. Runtime versus `n_seg`, degree-coded

### Current reading

- the effort curves remain close across `N = 6, 7, 8` for moderate-to-high n_seg (n_seg >= 8)
- `N = 8` gives the lowest `dv_proxy` across the feasible sweep, while `N = 8` n_seg=4 is infeasible (the only infeasible cell in the full grid)
- runtime grows with degree across the sweep, with `N = 8` consistently the slowest
- the safety-margin curves are now informative: all three degrees show the same monotone decrease with n_seg, confirming the conservatism-reduction trend is degree-independent
- `F5` therefore supports a clean performance-tradeoff reading and is worth keeping as a figure

### Caption draft

`Multi-order performance trends across Bézier degree for the refreshed boundary-conditioned study. Degree-coded curves show how objective-aligned effort and runtime vary across the matched subdivision sweep for `N = 6, 7, 8`. The figure supports a tradeoff interpretation rather than a blanket higher-degree-is-better claim.`

## Immediate follow-up

- keep `T4` on the main-paper path
- review whether `F5` reads clearly enough to stay in the main text after captioning and manuscript integration
- keep the degree discussion metric-specific: `dv_proxy` monotonically decreases with degree, runtime monotonically increases, and the tradeoff is modest (~6% effort vs ~1.7x runtime)
