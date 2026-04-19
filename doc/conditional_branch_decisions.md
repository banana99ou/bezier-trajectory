# Conditional Branch Decisions

This document records the current go/no-go decisions for the non-core branches so later work does not reopen the same scope questions by accident.

## T6 decision: Pass-1-replacement framing, data-ready (reversed 2026-04-17)

### Current decision

- `T6. Pass-1-replacement comparison under matched Pass 2` is data-ready and on the main paper path.
- The original naive-vs-warm-started framing is closed (see `doc/dcm_experiment_findings.md` Finding 2).
- T6 is owned by `doc/dcm_downstream_pack.md` and backed by `artifacts/paper_artifacts/t6_downstream_comparison.csv` and `figures/f6_downstream_speedup.png`.

### Why the decision was reversed

- the three conditions originally required for reversal (stable downstream protocol, completed matched comparison table, actual reported results) are now met
- the Bézier-replaces-Pass-1 pipeline uses identical Pass 2 solver, dynamics, tolerances, and boundary-condition protocol as the baseline — only the origin of the warm-start and phase structure differs
- on 6 of 7 both-converged cases the final cost is preserved to `|cost_delta| < 1e-9`; case 114 is the caveat (1.7% cost delta from a 5-vs-4 peak-count mismatch)
- median speedup is `1.22×` (min `0.51×`, max `2.47×`) with 4 of 7 faster end-to-end
- both regime boundaries (eccentricity at short `T_normed`; downstream Pass 2 divergence at long `T_normed`) are first-class evidence and are reported as part of the pack

### Required wording for C9 and §6.4

Use the scoped, pipeline-variant framing:

- the Bézier upstream can replace Pass 1 of the two-pass DCM pipeline within the scoped regime of circular transfers in which both pipelines converge, preserving final cost on 6 of 7 tested cases under matched Pass 2 and reducing end-to-end runtime on 4 of 7

Do not use any of the following:

- method-class superiority over DCM or direct collocation
- a general "Bézier is faster than DCM" claim
- any extension of the speedup claim outside the scoped regime

### Earlier placeholder policy (historical, superseded)

- before 2026-04-17 the section above required T6 to stay placeholder-only and C9 to stay at "intended use" wording
- superseded by the reversal recorded in this section; retained here as audit trail

## Degree branch decision

### Resolved degree range

- The active degree range is `N = 6, 7, 8`.

### Why this range is frozen

It is the only range that is simultaneously consistent with:

- `doc/figure_table_per_claim_plan.md`
- the updated intended study regime for the paper
- the boundary-condition motivation for moving away from the lower-order cases
- the current optimizer CLI after refocusing it on the higher-order study

The paper package is now being re-centered on `N = 6, 7, 8`.

### Placement decision

- `T4` remains on the main-paper path.
- `F5` now has a concrete draft asset and should be evaluated as a supporting main-paper figure rather than treated as a speculative extra.
- They are not on the core critical path.
- They should be built only after the method-core pack, the demonstration pack, and the subdivision-ablation pack are merged.

### Why they are not dropped

- `C7` remains a live, though secondary, claim in the technical skeleton
- the higher-order regime is the one actually motivated by the boundary-condition issue
- the remaining gap is disciplined aggregation and interpretation for `N = 6, 7, 8`

### Why they are not promoted to the critical path

- `C7` is secondary
- the paper can remain defensible without immediate `T4` / `F5` completion
- promoting them earlier would increase merge complexity without strengthening the core argument as much as `T1`-`F4`

## Current branch status summary

| Branch | Current status | Current action |
|---|---|---|
| `T6` | data-ready, on main-paper path | Pass-1-replacement framing; 7 both-converged cases (`T_normed` 0.28–2.05); `|cost_delta| < 1e-9` on 6 / 7; case 114 has 1.7% cost delta from phase-structure mismatch; median speedup `1.22×` |
| `F6` | data-ready, on main-paper path | per-case speedup + runtime composition from `figures/f6_downstream_speedup.png` |
| `T4` | data-ready, on main-paper path | table populated with refreshed 120-deg data; interpret metric-specific tradeoff (N=8 best effort, N=7 fastest convergence) |
| `F5` | needs regeneration | regenerate from updated 120-deg cache; the figure now has clear differentiation across degrees |
