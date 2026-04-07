# T6 Visual Evidence

This file is an audit index for the implemented downstream direct-collocation comparison.
It does not promote the claim. It points to the concrete evidence.

## Generated visual outputs

- `figures/t6_evidence_paths.png`
  - scenario geometry
  - naive initial guess path
  - warm-start initial guess path
  - final solved trajectory from naive run
  - final solved trajectory from warm-start run

- `figures/t6_solver_checks.png`
  - solve-time / iteration / objective / equality-residual comparison
  - matched-setup sanity-check list

## Generated raw evidence

- `artifacts/paper_artifacts/t6_gate_run.json`
  - final matched run results used for the gate decision

- `artifacts/paper_artifacts/t6_evidence_pack.json`
  - exact scenario values
  - exact naive initial guess states/controls
  - exact warm-start initial guess states/controls
  - final solved trajectories
  - solver outcomes
  - explicit sanity checks

## What the figures show

- Both runs use the same tested orbital-transfer setup:
  - transfer time `1500 s`
  - KOZ radius `6471 km`
  - `12` collocation intervals / `13` nodes

- Naive initialization:
  - cubic Hermite state guess from the exact endpoint positions and velocities
  - control inferred from Hermite acceleration minus gravity

- Warm-start initialization:
  - sampled upstream degree-4, `n_seg=16` Bézier solution
  - same boundary states enforced at the first and last nodes
  - control inferred from sampled Bézier acceleration minus gravity

## Outcome summary

- Naive:
  - success `true`
  - solve time `0.0416 s`
  - iterations `15`
  - final objective `0.1444069345`

- Warm-start:
  - success `true`
  - solve time `0.1806 s`
  - iterations `50`
  - final objective `0.1443395091`

## Why the naive run looks better

On this tested instance:

- both runs solved
- both satisfied essentially identical final constraints
- the warm start only improved the final objective slightly
- the naive run reached feasibility and termination faster and in fewer iterations

The evidence pack also shows why this likely happened under the downstream solver:

- naive initial guess:
  - initial objective `0.1439329124`
  - max dynamics defect `3.00248`
- warm-start initial guess:
  - initial objective `0.0019815032`
  - max dynamics defect `404.55806`

So the warm start begins with a lower downstream objective but a far worse downstream dynamics residual. That is consistent with a control-point-space warm start that is geometrically reasonable yet poorly matched to this direct-collocation transcription's defect equations. The solver then spends more work repairing dynamics consistency.

That is why this spike is scientifically mixed rather than supportive of a positive `C9` claim.

## Missing evidence

- A full per-iteration downstream convergence trace was not logged for this spike.
- The current visual evidence shows final outcomes and summary solver behavior, not a detailed iteration-by-iteration trust-constr history.
