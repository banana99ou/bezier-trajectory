# T6 Feasibility Audit

Superseded for the current paper pass: the active workflow keeps `T6` placeholder-only while downstream direct-collocation development is still in flux. This note is retained only as a historical audit artifact.

This document records the current feasibility verdict for `T6. Downstream direct-collocation initialization comparison`.

## Verdict

Do not promote `T6` into the defended paper package yet.

Reason:

- a real downstream direct-collocation spike now exists
- the first gate run is fairness-clean enough to inspect
- but the first matched result does not show materially useful warm-start benefit

## Inputs inspected

- protocol definition: `doc/t6_downstream_protocol.md`
- downstream implementation: `orbital_docking/downstream_collocation.py`
- runner: `tools/downstream_dc_compare.py`
- unit checks: `tests/unit/test_downstream_collocation.py`
- gate output: `artifacts/paper_artifacts/t6_gate_run.json`

## Validation status

- unit test status: pass
- naive and warm-start initializers both populate the same downstream variable blocks
- both runs use the same downstream problem, mesh, dynamics, objective, KOZ constraint, solver family, and stopping-rule configuration
- both runs converge on the tested instance

## Gate instance configuration

- upstream warm-start source: degree `4`, `n_seg = 16`
- upstream objective: `energy`
- downstream transcription: fixed-time direct collocation with `12` intervals
- downstream solver: `scipy.optimize.minimize(method="trust-constr")`

## First matched result

| Initialization | Success | Solve time (s) | Iterations | Final objective | Max equality violation | Min node margin (km) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | True | 0.0454 | 15 | 0.1444069 | `6.82e-13` | 145.0 |
| Bézier warm start | True | 0.1757 | 50 | 0.1443395 | `8.46e-13` | 145.0 |

## Interpretation

Safe reading:

- the infrastructure now exists to run a narrow naive-versus-warm-start direct-collocation comparison on the demonstrated orbital setup
- on the first matched instance, the warm start achieves only a tiny objective improvement while taking more time and more iterations
- this is not evidence for useful downstream warm-start value

Unsafe reading:

- planner superiority
- demonstrated warm-start benefit
- any paper claim that the current downstream comparison supports `C9`

## Promotion decision

Current decision:

- keep `C9` at intended-use wording
- keep `T6` off the core critical path

`T6` should only be reconsidered if additional matched instances or a stronger protocol show a materially useful benefit without obvious regressions in time, iterations, or constraint satisfaction.
