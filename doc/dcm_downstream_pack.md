# DCM Downstream Pack

This document is the merge-ready output of the downstream-value workstream. It freezes the Pass-1-replacement protocol, fixes what T6 and F6 may and may not claim, and records the current draft of both artifacts.

## Scope

This pack owns:

- `T6. Pass-1-replacement comparison under matched Pass 2`
- `F6. Per-case speedup and runtime composition`

Supports: `C9` (downstream warm-start usefulness), under the revised Pass-1-replacement framing.

## Fixed protocol for the current T6 / F6 branch

Pipeline A — baseline (full two-pass DCM):

- Pass 1: Hermite-Simpson collocation on a uniform mesh — produces initial phase structure
- Peak detection on `‖u(t)‖` to determine phase boundaries
- Pass 2: Multi-phase Legendre-Gauss-Lobatto collocation — final trajectory

Pipeline B — proposed (Bézier-replaces-Pass-1):

- Bézier SCP optimizer on control-point-space (degree 6, `n_seg = 16`) — produces thrust profile in a single convex-QP loop
- Same peak detection on `‖u(τ)‖ = r''(τ)/T² − g(r(τ))` to determine phase boundaries
- Same Pass 2 (Multi-phase LGL) with identical dynamics, tolerances, constraint set, and stopping rules

What is matched across the two pipelines:

- identical downstream transcription (Multi-phase LGL, same number of nodes per phase, same dynamics model, same objective)
- identical solver (IPOPT via CasADi) with identical tolerances and maximum iteration budget
- identical boundary condition protocol (`ν₀`, `νf` free in Pass 2)
- identical problem definition (same `h0`, `delta_a`, `delta_i`, `T_max_normed` from DB)

What differs across the two pipelines:

- origin of the warm-start state/control trajectory and the phase-structure decision (Pass 1 H-S vs Bézier SCP)

This is a **pipeline-variant comparison within a matched downstream protocol**. It is not a naive-vs-warm-started comparison (that original framing is closed; see `doc/dcm_experiment_findings.md` Finding 2) and it is not a planner-class comparison against DCM as a method.

## Regime in which the comparison is defensible

The proposed pipeline requires the Bézier upstream to produce a feasible thrust profile and Pass 2 to converge from the resulting warm-start. Two empirical boundaries constrain the regime.

**Eccentricity boundary** (from `results/dcm_pass1_replace_full/`, the `max_ecc ≤ 0.1` run):

- all 4 near-circular cases (`e0 = ef = 0`) with `T_normed ≤ 0.5` yielded a feasible Bézier upstream and both pipelines converged
- all 6 mildly elliptic cases (`e0 = 0.050`, `ef = 0.079`) with the same `T_normed` range yielded an infeasible Bézier upstream — the curve interior dips below the KOZ even when all control points are placed on the departure orbit
- the boundary sweep (below) extends this: across 112 near-circular cases, 34 had a feasible Bézier upstream; zero eccentric cases in that run had a feasible upstream

**Transfer-time boundary** (from `results/dcm_pass1_replace_boundary/`, the full-range run):

- for `T_normed ≤ 0.5`: 4 of 4 both-converged — the proposed pipeline is reliable
- for `0.5 < T_normed ≤ 1.0`: 1 of 3 both-converged
- for `1.0 < T_normed ≤ 2.0`: 1 of 14 both-converged — Bézier upstream often still feasible, but Pass 2 typically does not converge from the resulting warm-start
- for `T_normed > 2.0`: 1 of 91 both-converged — the multi-revolution regime is hostile to the proposed pipeline, but not uniformly so

The T6 claim is therefore scoped to **circular-to-circular transfers in which both pipelines converge**. Seven cases in the boundary run satisfy this condition, spanning `T_normed` from `0.28` to `2.05`. The claim is reported over that seven-case set; the much larger set of cases in which Pass 2 fails from the Bézier warm-start is documented separately as the transfer-time boundary.

## T6 draft

Source: `artifacts/paper_artifacts/t6_downstream_comparison.csv` (112 rows from `results/dcm_pass1_replace_boundary/aggregate.csv`).

T6 is reported over the 7 cases in which both pipelines converged. The remaining 105 cases are reported in the boundary-run appendix (below) to document the regime boundary rather than to pad the speedup metric with zero-effect rows.

| Case | `T_normed` | `h0` (km) | `delta_a` (km) | `delta_i` (deg) | Baseline time (s) | Bézier (s) | Pass 2 (s) | Proposed total (s) | Speedup | `|cost_delta|` | Peaks base / prop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2   | 0.280 | 400 |   −2.49  | 13.85 | 0.76 | 0.04 | 0.28 | 0.31 | **2.47×** | 5.0e−12 | 1 / 1 |
| 4   | 0.280 | 400 |   −2.49  |  3.00 | 1.33 | 0.03 | 2.47 | 2.50 | 0.53×     | 1.8e−10 | 1 / 1 |
| 6   | 0.280 | 400 |   −2.49  | 13.85 | 0.47 | 0.03 | 0.27 | 0.30 | **1.56×** | 5.0e−12 | 1 / 1 |
| 20  | 1.920 | 400 |  −78.47  |  0.83 | 3.51 | 0.23 | 6.60 | 6.83 | 0.51×     | 6.0e−10 | 4 / 4 |
| 39  | 0.510 | 400 | 1262.69  |  9.39 | 0.85 | 0.03 | 0.54 | 0.57 | **1.50×** | 4.7e−11 | 2 / 2 |
| 114 | 2.052 | 400 | 1224.14  |  4.66 | 1.94 | 0.24 | 1.71 | 1.95 | 0.99×     | 1.6e−6  | 4 / 5 |
| 126 | 0.500 | 400 | 1137.93  |  4.14 | 0.79 | 0.03 | 0.62 | 0.65 | **1.22×** | 8.8e−11 | 2 / 2 |

Headline summary (over the 7 both-converged cases):

- median speedup `1.22×`, min `0.51×`, max `2.47×`, faster in `4 / 7` cases
- for 6 of 7 cases, peak counts match exactly and `|cost_delta|` stays below `1e−9` — the pipelines are converging to the same local optimum and the final cost is preserved to better than machine precision relative to the baseline cost scale
- for case 114, the Bézier upstream detects one more peak than Pass 1 (5 vs 4); Pass 2 consequently uses a different phase structure and converges to a nearby but distinct local optimum, with a `1.7%` cost delta — a material caveat that the Bézier phase-structure decision is not always identical to Pass 1's
- the cases span `T_normed` from `0.28` to `2.05`, so the speedup story is not confined to sub-orbital transfers

Upstream-feasibility table (eccentricity boundary at short `T_normed`, from `results/dcm_pass1_replace_full/`):

| Regime                             | Cases | Bézier upstream feasible | Both pipelines converge |
|---|---:|---:|---:|
| `T_normed ≤ 0.5`, `e0 = ef = 0`    | 4  | 4 / 4 | 4 / 4 |
| `T_normed ≤ 0.5`, `0 < e ≤ 0.1`    | 6  | 0 / 6 | 0 / 6 |

Boundary sweep (circular-ish converged set, `max_ecc ≤ 0.01` default), from `results/dcm_pass1_replace_boundary/`:

| `T_normed` bucket | Cases | Bézier upstream feasible | Both pipelines converge |
|---|---:|---:|---:|
| `T ≤ 0.5`          |  4 |  4 | 4 |
| `0.5 < T ≤ 1.0`    |  3 |  3 | 1 |
| `1.0 < T ≤ 2.0`    | 14 | 12 | 1 |
| `T > 2.0`          | 91 | 15 | 1 |
| **Total**          | 112 | 34 | 7 |

In the boundary-sweep run all 34 Bézier-feasible cases were circular — eccentricity is sharp at any `T_normed` in this dataset. The transfer-time axis is softer: Bézier upstream remains feasible in `~15%` of multi-revolution cases, but downstream Pass 2 rarely converges from the resulting warm-start at `T_normed > 1`.

## T6 interpretation boundary

Safe reading:

- within the scoped regime (circular transfers in which both pipelines converge), the Bézier upstream can replace Pass 1 inside the two-pass DCM pipeline; on 6 of 7 cases final cost is preserved to `|cost_delta| < 1e−9`
- median end-to-end speedup is `1.22×`, with `4 / 7` cases faster and `3 / 7` cases slower; the speedup distribution spans from `0.51×` to `2.47×`, dominated by how well the Bézier warm-start aligns with Pass 2's optimum rather than by any savings from skipping Pass 1
- on case 114, the Bézier phase-structure decision differs from Pass 1's by one peak, and Pass 2 converges to a distinct local optimum with a `1.7%` cost delta — the Bézier upstream is therefore not a drop-in replacement that always recovers the same answer, only one that does so on 6 of the 7 tested cases
- both regime boundaries (eccentricity and transfer time) are first-class parts of the evidence; the proposed pipeline is not applicable outside them
- the comparison supports the framework's stated intended use as a warm-start generator for downstream solvers, in the specific sense of replacing the Pass 1 initialization stage within an otherwise matched two-pass DCM

Unsafe reading:

- any form of method-class superiority over DCM or over direct collocation more broadly
- a general "Bézier is faster than DCM" claim — the proposed pipeline *is* DCM (Pass 2 is identical); only the initialization stage differs
- extrapolation of the speedup distribution to the population of cases in which Pass 2 fails from the Bézier warm-start — the `1 / 91` both-converged rate at `T_normed > 2.0` means the scoped regime is a minority of the DB
- any implication that the Bézier upstream replaces DCM end-to-end; Pass 2 is load-bearing in both pipelines
- any claim that the `1.22×` median speedup is representative of multi-revolution transfers in general; case 20 (`T_normed = 1.92`) is slower (`0.51×`), case 114 (`T_normed = 2.05`) is roughly neutral (`0.99×`), and both are outliers in Pass 2 runtime

## F6 panel specification

Current draft asset:

- `figures/f6_downstream_speedup.png`

Required panels:

1. Per-case end-to-end speedup (baseline_time / proposed_total_time), sorted, with a reference line at `1×`
2. Runtime composition: baseline stage vs Bézier+Pass 2 stack per case

Reported cases: the `7` both-converged cases only. The 105 cases in which one or both pipelines fail are represented in the regime tables, not in the figure, so the figure does not imply a misleading average.

Caption draft:

`F6. Per-case comparison between the full two-pass DCM pipeline (baseline) and the Bézier-replaces-Pass-1 pipeline (proposed), with identical Pass 2 solver, dynamics, and tolerances. Panel (a) shows end-to-end speedup sorted by case; the proposed pipeline is faster in 4 of 7 cases (median 1.22×, min 0.51×, max 2.47×). Panel (b) decomposes runtime into the stages incurred by each pipeline. Six of seven cases preserve the baseline final cost to |cost_delta| < 1e-9; case 114 is the exception (1.7% cost delta, five peaks detected versus the baseline's four). Results are scoped to circular transfers in which both pipelines converge, spanning T_normed from 0.28 to 2.05; the eccentricity and transfer-time boundaries that define this scope are reported in the regime tables.`

## Boundary-run appendix (for limitations section)

Source: `results/dcm_pass1_replace_boundary/aggregate.csv` (112 converged DB cases at default `max_ecc ≤ 0.01`).

Paragraph-ready statement:

The proposed pipeline has two independent regime boundaries. The first is eccentricity: every case in the 112-case boundary sweep with feasible Bézier upstream had `e0 = ef = 0`, and every elliptic case tested at short `T_normed` in the separate 10-case eccentricity run (`results/dcm_pass1_replace_full/`) had an infeasible Bézier upstream, so this boundary is sharp in the tested dataset. The second is transfer time: the Bézier upstream remains feasible on `15` of `91` cases at `T_normed > 2.0`, but Pass 2 converges on only `1` of those; at `1 < T_normed ≤ 2`, `12 / 14` Bézier upstreams are feasible but only `1 / 14` both-converge; at `T_normed ≤ 1`, the combined both-convergence rate is `5 / 7`. Multi-revolution transfers are therefore not uniformly outside the representational range of a degree-6 Bézier, but they are hostile to the downstream Pass 2 when initialized from this warm-start. The boundary sweep `aggregate.csv` preserves the per-case metadata for reviewers.

Counts by `T_normed` bucket are reproduced in the T6 boundary-sweep table above and in `tools/build_t6_boundary_summary.py` output. This table is paragraph-ready for the limitations section.

## Immediate follow-up for the final paper figure

- after 1b completes: populate the boundary appendix, regenerate summary counts
- update §6.4 of `doc/results_section_draft.md` using T6 headline numbers and the regime statement from this pack
- update `doc/paper_evidence_map.md` C9 row and `doc/conditional_branch_decisions.md` T6 section to reflect the Pass-1-replacement framing and data-ready status
- do not modify `tools/dcm_downstream_experiment.py` runner; the pack is built from its existing output schema
