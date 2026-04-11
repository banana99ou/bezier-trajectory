# Degree Ablation Pack

This document is the merge-ready output of the degree-ablation branch. It keeps `T4` on the main-paper path, freezes the matched comparison protocol, and records the current `F5` draft asset from the refreshed Rust-backed `N = 5, 6, 7` study.

## Scope

This pack owns:

- `T4. Degree ablation under matched settings`
- `F5. Multi-order performance trends`

## Fixed protocol

The active degree comparison is:

- degrees: `N = 5, 6, 7`
- representative matched table setting: `n_seg = 16`
- full trend figure sweep: `n_seg = [2, 4, 8, 16, 32, 64]`
- objective: `dv`
- solver backend: `rust`
- maximum outer iterations: `500`
- tolerance: `1e-6`
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

| Setting | Degree | Control points | `n_seg` | Solve success | Safety margin (km) | Objective-aligned effort `dv_proxy` (m/s) | Runtime (s) | Mean control accel (m/s^2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `demo_N5_seg16` | 5 | 6 | 16 | True | 145.000 | 17814.989 | 5.211 | 11.966 |
| `demo_N6_seg16` | 6 | 7 | 16 | True | 145.000 | 17905.937 | 5.063 | 12.034 |
| `demo_N7_seg16` | 7 | 8 | 16 | True | 145.000 | 17752.065 | 17.768 | 11.938 |

## T4 interpretation boundary

Safe reading:

- all three degrees are feasible under the matched boundary-conditioned protocol
- `N = 7` gives the lowest reported `dv_proxy` at the representative `n_seg = 16` setting
- `N = 6` is the fastest at the representative setting
- the effort differences among `N = 5, 6, 7` are modest relative to the runtime penalty seen for `N = 7`
- the current evidence supports a tradeoff story, not a dominance story

Unsafe reading:

- higher degree is better in general
- the small `dv_proxy` edge for `N = 7` automatically justifies the runtime cost
- the degree comparison alone proves stronger boundary-condition conclusions beyond the tested setup

## F5 draft asset

Current draft asset:

- `figures/f5_multi_order_tradeoff_N567.png`

### Required panels

1. Objective-aligned effort `dv_proxy_m_s` versus `n_seg`, degree-coded
2. Runtime versus `n_seg`, degree-coded

### Current reading

- the effort curves remain close across `N = 5, 6, 7`
- `N = 7` is slightly better on `dv_proxy` in several settings, but not by a large margin
- the runtime penalty for `N = 7` is clear across the sweep
- `F5` therefore supports a clean performance-tradeoff reading and is worth keeping as a figure, at least in draft form

### Caption draft

`Multi-order performance trends across Bézier degree for the refreshed boundary-conditioned study. Degree-coded curves show how objective-aligned effort and runtime vary across the matched subdivision sweep for `N = 5, 6, 7`. The figure supports a tradeoff interpretation rather than a blanket higher-degree-is-better claim.`

## Immediate follow-up

- keep `T4` on the main-paper path
- review whether `F5` reads clearly enough to stay in the main text after captioning and manuscript integration
- keep the degree discussion metric-specific: `N = 7` is slightly better on effort, while `N = 6` is markedly faster at the representative setting
