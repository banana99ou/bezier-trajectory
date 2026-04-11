# DCM DB Experiment Note

## Status

This note records the current interpretation of the professor's request before the actual DB file and DCM code are inspected.

It is a planning note, not a validated description of the downstream stack.

## Reasonability Check

The request is reasonable.

- Logically, the proposed comparison is coherent.
- Scientifically, it is defensible if both pipelines are compared on matched cases with the same downstream solver and the same evaluation rules.
- The weak point is the phrase "good looking curve," which is too subjective to use as a final experiment definition. That will need a reproducible selection rule.

## Interpreted Request

Current reading of the request:

1. There exists a baseline pipeline:
   `DB initial guess -> DCM -> output`
2. Select some promising cases from the DB.
3. Extract the initial path from those cases.
4. Feed that same initial path into the upstream optimizer.
5. Feed the optimizer output into DCM.
6. Compare the resulting downstream output against the DB-based baseline.

Under this reading, the proposed pipeline is:

`same DB initial guess -> my optimizer -> DCM -> output`

and the baseline is:

`same DB initial guess -> DCM -> output`

## Scientific Goal

The goal is not to prove that `my optimizer + DCM` is always better than the DB-based route.

The intended goal is narrower and more defensible:

- identify a subset or region of cases where `my optimizer + DCM` performs better than the DB-based baseline
- report that region honestly, including cases where the proposed pipeline does not help

This is a valid experiment framing. It avoids the unsupported claim of broad superiority.

## Minimum Fairness Rules

For the comparison to be scientifically readable:

- the downstream DCM problem must be the same across both runs
- DCM settings, tolerances, stopping rules, and mesh must match
- the compared cases must come from the same DB entries
- failures must be retained and reported, not filtered out
- metrics must be computed the same way for both pipelines

If those conditions are not met, the result is exploratory only.

## Candidate Metrics

Metrics mentioned in discussion:

- final cost
- compute time
- safety margin

Metrics that should also be recorded if available:

- success or failure
- iteration count
- final constraint violation
- any feasibility flag produced by DCM

## Ambiguities To Resolve

The current interpretation still depends on several unknowns:

- what the DB actually stores:
  - raw initial guesses
  - multiple candidate curves
  - already-postprocessed outputs
- what exactly "compare with DB" means:
  - compare against `DB -> DCM` final outputs
  - compare against raw DB trajectories
  - compare against several DB seeds per case
- how the initial path is represented:
  - sampled states
  - control points
  - waypoints
  - some other format
- what DCM expects as input and what it outputs
- how "good looking curve" should be operationalized without selection bias

The most logical interpretation at present is:

- baseline comparison target is the existing `DB initial guess -> DCM` result

## Files Needed

The request to the colleague is appropriate. The minimum useful artifacts are:

- the DB file
- the DCM code
- any config files needed to run DCM
- one minimal runnable example for a single case
- any metric definitions or evaluation scripts

## Immediate Next Step

Once the files arrive:

1. inspect the DB schema and determine what one entry contains
2. inspect the DCM input-output contract
3. reproduce one baseline run
4. insert the upstream optimizer between DB and DCM on that same case
5. compare both runs under matched settings

## Safe Current Claim

Safe wording at this stage:

The intended experiment is to test whether an upstream optimizer can improve downstream DCM performance on some subset of DB-seeded cases, under a matched comparison against the existing DB-to-DCM pipeline.
