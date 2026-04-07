# T6 Downstream Direct-Collocation Protocol

## Status

This is a gated feasibility protocol for `C9` / `T6`, not a paper claim promotion.
Until a matched downstream comparison shows a materially useful result under the fairness rules below, `C9` remains intended use only.

## Scope Lock

- Supported claim only: the upstream Bézier method may be useful as a warm-start generator for downstream direct collocation on the tested setup.
- Explicitly out of scope:
  - planner replacement
  - "better than direct collocation"
  - any benchmark suite broader than naive-initialized DC versus Bézier-warm-started DC
- The downstream problem must be identical across both runs except for the initial guess.

## Tested Downstream Problem

The current spike uses one fixed-time direct-collocation problem on the same orbital-transfer geometry already used in the repo's demonstration setup:

- start orbit: Progress-like 245 km circular orbit
- target orbit: ISS-like 400 km circular orbit
- plane: shared 51.64 deg inclination, zero-RAAN simplification
- phase offset: 30 deg
- transfer time: `1500 s`
- keep-out zone: spherical radius `EARTH_RADIUS_KM + 100 km`

The downstream transcription is deliberately minimal:

- state at each node: position `r_k in R^3`, velocity `v_k in R^3`
- control at each node: commanded acceleration `u_k in R^3`
- mesh: one fixed direct-collocation grid with `n_intervals = 12` for the gate instance
- dynamics: trapezoidal defects using the same two-body + J2 gravity model already used in `orbital_docking/optimization.py`
- path constraint: spherical KOZ margin at every collocation node
- boundary constraints: exact initial and final position/velocity states
- objective: trapezoidal integral of squared control magnitude

This is not a full benchmark-quality downstream stack. It is the smallest scientifically readable direct-collocation spike that preserves matched-problem fairness.

## Fairness Rules

- Both runs use the same downstream decision variables, mesh, dynamics, objective, KOZ constraint definition, solver, tolerances, and stopping rules.
- Both runs populate the full state/control variable vector. There are no hidden collocation-only variables in this transcription.
- The only difference between runs is the initial guess.
- Failed solves are retained and reported.
- Constraint satisfaction is reported as:
  - maximum equality residual
  - minimum KOZ node margin
  - minimum line-segment margin diagnostic between adjacent node positions

If any of those conditions change between runs, the result is exploratory only and does not support `T6`.

## Initialization Definitions

### Naive Initialization

The naive initializer is a deterministic cubic Hermite state profile built from the exact downstream endpoint positions and velocities:

- position guess: endpoint-consistent cubic Hermite interpolation over the fixed transfer time
- velocity guess: analytic time derivative of that Hermite profile
- control guess: inferred from the Hermite acceleration minus gravity

This baseline is intentionally simple but not artificially weak. It uses the same boundary information available to the warm start and fills the full downstream variable vector.

### Bézier Warm Start

The warm-start initializer is built from the upstream Bézier control-point solution:

- sample the upstream Bézier curve at the downstream collocation nodes
- use sampled position for interior state nodes
- use sampled Bézier derivative divided by transfer time for interior velocity nodes
- use sampled Bézier acceleration divided by transfer-time squared, minus gravity, for interior control nodes
- overwrite the first and last state nodes with the exact downstream boundary states so both initializations use the same boundary information

This keeps the comparison about interior initialization quality rather than boundary-data mismatch.

## Gate Instance

The first gate run uses a single tested instance:

- upstream warm-start source: degree-4 Bézier solve with `n_seg = 16`
- upstream objective mode: `energy`
- downstream mesh: `12` intervals
- downstream solver: `scipy.optimize.minimize(..., method=\"trust-constr\")`

This one-instance run is enough for a go/no-go verdict. It is not enough by itself for a broad claim.

## Required Metrics For T6

The gate runner records the `T6` core metrics for both initializations:

- solve success
- solve time
- iteration count
- final objective
- final constraint satisfaction

Constraint satisfaction is stored as a structured object rather than a single scalar because the relevant failure modes are different.

## Promotion Rule

Promote `T6` toward paper evidence only if all of the following hold:

- the downstream comparison is fairness-clean
- the warm-started run is materially better on at least one core metric without obvious degradation elsewhere
- the result is not just a cosmetic transient difference

If the first matched instance is weak, mixed, or solver-sensitive, keep `T6` as exploratory and leave `C9` in intended-use wording.

## Kill Criteria

Stop or defer if:

- the downstream problem definition has to diverge materially from the stated orbital setup
- the solver path requires substantial new infrastructure or dependency churn
- the naive baseline cannot be defended as honest
- the warm-start export cannot populate the same variable blocks fairly
- the first matched result is not clearly informative
