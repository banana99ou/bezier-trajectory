# T6 Failure Audit

This document implements the failure-audit plan for the downstream direct-collocation spike.
It is an audit artifact, not a repair proposal.

## Audit Scope

Failure hypotheses:

- `F1`: the warm-start export is inconsistent with the downstream transcription
- `F2`: the comparison is not truly fair even if the downstream solver call is matched
- `F3`: the downstream problem is too weak to produce physically credible orbital-transfer trajectories
- `F4`: the trajectories only look wrong because of visualization or an arbitrary external reference

Primary evidence sources:

- `orbital_docking/downstream_collocation.py`
- `orbital_docking/optimization.py`
- `doc/t6_downstream_protocol.md`
- `artifacts/paper_artifacts/t6_gate_run.json`
- `artifacts/paper_artifacts/t6_evidence_pack.json`
- `artifacts/paper_artifacts/t6_velocity_audit.json`
- `artifacts/paper_artifacts/t6_failure_audit.json`
- `figures/t6_failure_audit.png`

## Hypothesis Status

### `F1` Warm-start export inconsistency

Status: `confirmed`

Why:

- the sampled upstream Bézier endpoint velocities differ from the downstream fixed endpoint velocities by:
  - start: `7.123 km/s`
  - end: `6.721 km/s`
- the raw sampled upstream curve is only mildly inconsistent with the downstream trapezoidal transcription:
  - max raw-sample position defect norm: `1.910`
  - max raw-sample velocity defect norm: `0.00246`
- after overwriting the endpoint states to match the downstream boundary conditions, the warm-start defects explode:
  - max overwritten warm-start position defect norm: `443.658`
  - max overwritten warm-start velocity defect norm: `7.124`

Interpretation:

- the dominant warm-start failure is not “the upstream curve is generally unusable”
- the dominant failure is that the exported warm start is made transcription-inconsistent by forcing downstream endpoint states onto an upstream interior trajectory generated under different boundary conditions

### `F2` Comparison fairness mismatch

Status: `confirmed`

Why:

- the downstream solver call is matched in the narrow sense: same problem object, solver, tolerances, objective, and constraints; only `x0` changes
- the upstream warm-start source does **not** solve the same problem as the downstream NLP:
  - upstream uses `v0=None`, `v1=None`
  - upstream uses `enforce_prograde=True`
  - upstream KOZ handling differs from downstream node-only spherical KOZ constraints

Interpretation:

- the comparison is fair at the downstream call level
- it is not fully fair at the scientific problem-definition level, because the warm-start source is produced under a different upstream boundary/constraint regime than the downstream target

### `F3` Downstream model too weak

Status: `confirmed`

Why:

- the downstream problem enforces:
  - exact endpoint position and velocity states
  - trapezoidal discrete dynamics with two-body + J2 gravity
  - node-only spherical KOZ constraints
  - quadratic control-effort objective
- the downstream problem does **not** enforce:
  - thrust bounds
  - continuous-time KOZ safety between nodes
  - prograde motion
  - radial-speed limits
  - orbital-credibility constraints beyond endpoint states and discrete defects
- both final curves remain pathological relative to a local prograde circular-orbit reference:
  - `naive_final`: minimum speed ratio `0.048`, retrograde nodes `3`, max angle to prograde `165.6 deg`
  - `warm_final`: minimum speed ratio `0.055`, retrograde nodes `3`, max angle to prograde `165.6 deg`

Interpretation:

- this is not just a warm-start problem
- the downstream NLP itself admits solutions that are numerically feasible under its own rules but physically non-credible as orbital-transfer motion

### `F4` Visualization-only or reference-only issue

Status: `rejected`

Why:

- the odd behavior is reproduced numerically in `artifacts/paper_artifacts/t6_velocity_audit.json`
- the key pathology is not only visual; it appears directly in stored `r,v` arrays
- the code confirms that nothing in the current downstream model forbids this behavior

Interpretation:

- the plots are not the root problem
- the plots merely exposed a real issue already present in the underlying trajectories

## Confirmed / Suspected / Unknown Evidence Map

| Category | Item | Evidence |
|---|---|---|
| Confirmed | Same downstream solver/settings/objective/constraints for naive and warm runs | `run_downstream_comparison()` in `orbital_docking/downstream_collocation.py` |
| Confirmed | Only `x0` changes inside the downstream run | same function |
| Confirmed | Warm-start initial max dynamics defect is about `404.6` vs about `3.0` for naive | `artifacts/paper_artifacts/t6_evidence_pack.json` |
| Confirmed | Upstream warm-start source does not enforce downstream endpoint velocity constraints | `build_demo_bezier_warm_start()` in `orbital_docking/downstream_collocation.py` |
| Confirmed | Upstream warm-start source uses prograde shaping absent downstream | same file plus `orbital_docking/optimization.py` |
| Confirmed | Raw sampled warm start is only mildly inconsistent before endpoint overwrite | `artifacts/paper_artifacts/t6_failure_audit.json` |
| Confirmed | Final naive and warm solutions satisfy downstream discrete constraints numerically | `artifacts/paper_artifacts/t6_gate_run.json` |
| Confirmed | Final naive and warm solutions remain strongly non-orbital relative to local prograde circular reference | `artifacts/paper_artifacts/t6_velocity_audit.json` |
| Suspected | Endpoint overwrite is the dominant immediate trigger of the warm-start defect spike | strongly indicated by raw-vs-overwritten defect comparison in `t6_failure_audit.json` |
| Suspected | Warm start is additionally disadvantaged because the upstream feasible set differs from the downstream one | code-level mismatch plus protocol |
| Unknown | Whether final trajectories violate continuous-time dynamics between nodes | not yet replayed densely |
| Unknown | Whether a denser downstream mesh would materially reduce the same qualitative pathology | not yet tested |
| Unknown | Whether the iteration gap is dominated by endpoint overwrite alone or by broader upstream/downstream mismatch | not yet isolated experimentally |

## Root-Cause Ranking

### 1. Dominant cause

Warm-start export is not transcription-consistent with the downstream direct-collocation defect equations.

Why this is ranked first:

- the raw sampled upstream curve is not catastrophically bad under the downstream transcription
- the catastrophic defect appears after endpoint overwrite
- the start/end velocity mismatch is on the order of full orbital speed, not a small perturbation

### 2. Secondary cause

The comparison is only matched at the downstream call level; the upstream warm-start source solves a materially different problem.

Why this is ranked second:

- upstream and downstream do not share the same endpoint-velocity, prograde, or KOZ regimes
- even a perfectly coded export would still be transferring from a different problem

### 3. Tertiary cause

The downstream direct-collocation problem is too weak to enforce physically credible orbital-transfer motion.

Why this matters:

- even after convergence, both final trajectories remain non-credible under the velocity audit
- so fixing warm-start export alone would not make the downstream formulation scientifically sound enough

### 4. Minor / rejected cause

Visualization ambiguity is secondary.

Why:

- the issue survives numerical audit
- the plots did not create the pathology; they exposed it

## Missing Diagnostics Before Any Fix

Still missing:

- dense between-node continuous-time replay of the final trajectories under the same control and gravity model
- per-iteration `trust-constr` history for both downstream runs
- mesh-refinement sensitivity for the same downstream problem
- tighter localization of how much of the warm-start failure is concentrated in the first/last intervals versus the interior after overwrite

## Decision Memo

The current audit supports the following causal diagnosis:

1. The immediate warm-start failure is primarily an export inconsistency problem caused by overwriting downstream endpoint states onto an upstream curve that was not generated with matching endpoint velocity constraints.
2. The broader comparison is scientifically weakened because the upstream warm-start source and the downstream target are not actually the same problem.
3. Even if the export were repaired, the downstream NLP as currently posed is too weak to support physically credible orbital-transfer behavior.

Therefore the next implementation step should **not** be a blind warm-start tweak or solver retune. The eventual fix work should first address the dominant export inconsistency, while keeping in view that the downstream problem definition itself is also underconstrained.
