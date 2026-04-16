# Conditional Branch Decisions

This document records the current go/no-go decisions for the non-core branches so later work does not reopen the same scope questions by accident.

## T6 decision: placeholder only for the current paper pass

### Decision

- `T6. Downstream direct-collocation initialization comparison` stays placeholder-only for the current paper pass.

### Why

- downstream direct-collocation work is still under active development
- the current paper pass should not let that moving target destabilize the core method and results package
- the requested workflow is to keep `T6` as a placeholder rather than to promote or reject the claim based on an interim spike

### Required wording until this changes

Use the narrow fallback:

- the framework is intended as a warm-start generator for downstream solvers

Do not use the stronger demonstrated-value wording unless `T6` is actually built and stabilized.

### What would be required to reverse this decision

1. a stable downstream protocol
2. a completed matched comparison table
3. actual reported results rather than placeholders

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
| `T6` | placeholder only | keep `C9` at intended-use wording |
| `T4` | data-ready, on main-paper path | table populated with refreshed 120-deg data; interpret metric-specific tradeoff (N=8 best effort, N=7 fastest convergence) |
| `F5` | needs regeneration | regenerate from updated 120-deg cache; the figure now has clear differentiation across degrees |
