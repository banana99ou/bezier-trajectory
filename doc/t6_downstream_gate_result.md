# T6 Downstream Gate Result

## Verdict

`T6` should remain deferred as paper evidence.

The first matched downstream instance is fairness-clean enough to be informative, but it does not support promotion of `C9` from intended use to demonstrated usefulness:

- both initializations solved successfully
- the Bézier warm start reached a slightly lower final objective
- the Bézier warm start took substantially more iterations and more wall-clock time

That is a mixed result, not a clean warm-start win.

## Matched Gate Instance

- protocol: [doc/t6_downstream_protocol.md](/Volumes/Sandisk/code/bezier-trajectory/doc/t6_downstream_protocol.md)
- runner: `python3 tools/downstream_dc_compare.py --output artifacts/paper_artifacts/t6_gate_run.json`
- upstream warm-start source:
  - degree `4`
  - `n_seg = 16`
  - objective `energy`
- downstream direct collocation:
  - `12` intervals
  - fixed transfer time `1500 s`
  - same solver/settings for both initializations

## Observed Comparison

### Naive Initialization

- solve success: `true`
- solve time: `0.0416 s`
- iteration count: `15`
- final objective: `0.1444069345`
- max equality violation: `6.82e-13`
- min KOZ node margin: `145.0 km`

### Bézier Warm Start

- solve success: `true`
- solve time: `0.1806 s`
- iteration count: `50`
- final objective: `0.1443395091`
- max equality violation: `8.46e-13`
- min KOZ node margin: `145.0 km`

## Interpretation Boundary

This result does not justify a positive warm-start claim in the current paper:

- reliability: no gain, because both runs solved
- convergence: worse for the warm start on this tested instance
- final quality: only a very small objective improvement

The current repo now has an explicit, reproducible `T6` spike, but the outcome is mixed enough that the scientifically honest decision is to keep `C9` in intended-use wording.

## Recommended Disposition

- keep `T6` off the paper critical path
- do not write demonstrated warm-start usefulness from this result
- if `T6` is revisited later, do it only with:
  - a stronger downstream formulation rationale
  - more than one matched instance
  - and a predeclared criterion for what counts as a meaningful warm-start benefit
