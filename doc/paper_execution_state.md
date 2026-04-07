# Paper Execution State

This document is the mutable source of truth for the artifact-build phase. It exists to preserve shared context across parallel workstreams and to stop notation, metric, and scope drift.

Last updated: 2026-04-07

## Workflow source of truth

The canonical scientific control layer remains:

- `doc/paper_claim_scope_nonclaims.md`
- `doc/paper_evidence_map.md`
- `doc/figure_table_per_claim_plan.md`
- `doc/paper_technical_skeleton.md`

This file does not replace those documents. It records the current execution state used to implement them.

## Current implementation-grounded dataset

The current orbital summary dataset is generated from the existing optimizer and cached under:

- `artifacts/paper_artifacts/orbital_results_summary.json`
- `artifacts/paper_artifacts/orbital_results_summary.csv`

The configuration represented by that dataset is:

- objective: `dv`
- degrees: `N = 5, 6, 7`
- subdivision sweep target: `n_seg = [2, 4, 8, 16, 32, 64]`
- completed refreshed subdivision rows: `n_seg = [2, 4, 8, 16, 32, 64]` for the current `N=6` baseline
- max iterations for the refreshed higher-order paper pass: `500`
- tolerance: `1e-6`
- proximal weight: `1e-6`
- trust radius: `2000.0`
- prograde preservation: enabled
- endpoint velocity constraints: enforced
- transfer time: `1500.0 s`
- KOZ radius: `6471.0 km`
- solver backend: `rust`

Generated figure assets from the refreshed higher-order pass currently include:

- `figures/comparison_N5.png`
- `figures/comparison_N6.png`
- `figures/comparison_N7.png`
- `figures/demo_N5_seg16.png`
- `figures/demo_N6_seg16.png`
- `figures/demo_N7_seg16.png`
- `figures/f3_representative_settings.png`
- `figures/f4_subdivision_tradeoff_N6.png`
- `figures/performance_N5.png`
- `figures/performance_N6.png`
- `figures/performance_N7.png`
- `figures/time_vs_order.png`

## Frozen claim boundary

The active core artifact package is tied to:

- `C1`: control-point-space formulation
- `C2`: conservative spherical-KOZ handling
- `C3`: SCP as a sequence of convex QPs
- `C5`: feasible demonstration trajectories
- `C6`: subdivision-count trade-off

Secondary but still live:

- `C7`: degree trade-off

Conditional only:

- `C9`: warm-start usefulness for downstream direct collocation

Explicitly not in the current evidence package:

- planner superiority
- true delta-v optimality
- broad cross-domain validation
- dynamic-obstacle support in the present paper

## Frozen method notation lock

Use the following symbols consistently in Sections 3 and 4 and in all method-facing artifact captions:

- `P in R^{(N+1) x 3}`: control-point matrix
- `x in R^{3(N+1)}`: stacked control-point decision vector
- `D_N`: Bezier difference matrix
- `E_M`: degree-elevation matrix
- `L_{1,N} = E_{N-1} D_N`: degree-preserving velocity map
- `L_{2,N} = E_{N-1} D_N E_{N-1} D_N`: degree-preserving acceleration map
- `G_N`: Bernstein Gram matrix
- `tilde G_N = L_{2,N}^T G_N L_{2,N}`: quadratic operator for the legacy L2 acceleration-energy form
- `S^{(s)}`: equal-parameter subdivision matrix for sub-arc `s`
- `P^{(s)} = S^{(s)} P`: sub-arc control polygon
- `q_k^{(s)}`: `k`th control point of sub-arc `s`
- `c^{(s)}`: centroid of the sub-arc control polygon
- `n^{(s)}`: supporting-half-space normal for sub-arc `s`
- `H^{(s)}`: supporting half-space for sub-arc `s`
- `n_seg`: KOZ subdivision count
- `n_lin`: objective linearization segment count

Mandatory wording lock:

- say `sequence of convex QPs`
- do not say or imply exact one-shot convexification
- describe `dv` as an IRLS-weighted L1-style control-effort proxy
- do not describe the objective as true delta-v optimality

## Frozen safety boundary

The method-support proposition is limited to the spherical-KOZ case.

Assumptions that must remain explicit:

1. The KOZ is spherical.
2. Each sub-arc uses a fixed supporting half-space constructed from that sub-arc's centroid.
3. All control points of that subdivided sub-arc satisfy the corresponding half-space inequality.
4. The normal construction excludes the degenerate centroid-at-center case.

Safe consequence:

- the entire sub-arc lies in the supporting half-space and therefore outside the spherical KOZ

Unsafe overreach that must be avoided:

- arbitrary obstacle geometry
- unconditional continuous safety
- a global obstacle-avoidance theorem

## Frozen results metric lock

These definitions are now fixed for `T2`, `T3`, and any companion figures built from the current orbital dataset:

- `solve_success`: use the persisted feasibility outcome `info["feasible"]`
- `safety_margin_km`: `min_radius_km - KOZ_RADIUS`
- `objective_aligned_effort`: `dv_proxy_m_s`
- `runtime_s`: `info["elapsed_time"]`
- `outer_iterations`: `info["iterations"]`

Interpretation boundary:

- `solve_success` and `safety_margin_km` are related but not identical columns; the binary status reports whether a feasible final trajectory was obtained, while the margin keeps the underlying geometry visible.

Failure-reporting policy:

- if a planned run returns valid metadata but is infeasible, keep the row and set `solve_success = false`
- if a planned run does not produce valid metadata, report it explicitly as missing rather than dropping it silently

Operational definition of reduced conservatism:

- for the current paper, conservatism must be interpreted through reported metrics under fixed settings, not by visual path tightness
- the safest reading is whether increasing `n_seg` preserves comparable certified clearance while reducing objective-aligned effort, or improves both, under fixed settings
- if the certified clearance stays effectively unchanged and the objective-aligned effort also stays nearly flat, the paper should report that the current evidence shows computation-burden change more clearly than effective-conservatism improvement

## Current case registry

Representative demonstration settings selected for `T2` and `F3`:

- `demo_N5_seg16`: degree `5`, `n_seg = 16`
- `demo_N6_seg16`: degree `6`, `n_seg = 16`
- `demo_N7_seg16`: degree `7`, `n_seg = 16`

Rationale:

- these three cases share the same scenario and the same boundary-conditioned subdivision count
- they provide one representative feasible case per supported curve order in the active higher-order regime
- they are suitable for `C5` without conflating the demonstration claim with the subdivision ablation

Baseline setting for `T3` and `F4`:

- `ablation_N6_all_seg`: degree `6` with target `n_seg = [2, 4, 8, 16, 32, 64]`

Rationale:

- degree `6` is the middle order in the active `5, 6, 7` study
- this choice is a workflow baseline, not a scientific claim that sixth degree is privileged

## Artifact registry

| Artifact | Role | Current state | Primary input |
|---|---|---|---|
| `Formal safety statement` | method proof boundary | draft-ready | `doc/method_section_build.md` |
| `T1` | notation/operator table | draft-ready | implementation operators |
| `F1` | KOZ geometry figure | spec-ready | legacy KOZ illustration logic plus centroid-based assumptions |
| `F2` | SCP pipeline figure | spec-ready | current SCP loop in `orbital_docking/optimization.py` |
| `T2` | demonstration summary table | data-ready | `artifacts/paper_artifacts/orbital_results_summary.csv` |
| `F3` | representative trajectory panel | draft-ready | `figures/f3_representative_settings.png` |
| `T3` | subdivision ablation table | data-ready | `artifacts/paper_artifacts/orbital_results_summary.csv` |
| `F4` | subdivision trade-off figure | draft-ready | `figures/f4_subdivision_tradeoff_N6.png` |
| `T4` | degree ablation table | draft-ready | `doc/degree_ablation_pack.md` |
| `F5` | multi-order trend figure | draft-ready | `figures/f5_multi_order_tradeoff_N567.png` |
| `T6` | downstream comparison table | placeholder-only | placeholder text only |

## Resolved conditional decisions

### T6 decision

- Current status: placeholder-only for the current paper pass
- Reason: downstream direct-collocation work is still under active development and should not control the defended evidence package
- Required wording until that changes: warm-start use remains `intended use`, not demonstrated downstream value

### Degree-range decision

- Resolved range for `T4` and `F5`: `N = 5, 6, 7`
- Reason: this range is the active boundary-conditioned study regime and matches the refreshed optimizer CLI and results packaging

### C6 interpretation decision

- The refreshed higher-order dataset still does not show a meaningful change in `safety_margin_km`; it remains effectively constant at about `145 km`
- The refreshed `N=6` sweep also keeps `dv_proxy_m_s` nearly flat across `n_seg = 2 ... 64`
- Therefore the present evidence still supports runtime-growth claims much more clearly than strong effective-conservatism claims
- Any final C6 wording must say this plainly unless a different metric or a stronger dataset is added

## Remaining open work

- integrate the new `T4` / `F5` degree-ablation pack into manuscript prose
- decide after manuscript integration whether `F5` is clean enough to justify main-text space relative to the table
