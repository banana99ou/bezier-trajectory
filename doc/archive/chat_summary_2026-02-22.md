# Chat Summary - 2026-02-22

## Context

You observed a trend where higher `N_seg` trajectories climb to higher apoapsis and return, with higher cumulative control acceleration, while control acceleration near the apex appears small.  
Question: is the optimizer effectively minimizing peak control acceleration rather than cumulative effort?

## Hypotheses Reviewed

1. **H1: KOZ conservatism vs segmentation**  
   Increasing `n_seg` may make linearized KOZ constraints more conservative and push trajectories outward.

2. **H2: No trust-region drift (proxy test)**  
   Without trust-region/proximal regularization, SCP iterations may drift to odd local shapes; tested via initialization perturbations.

3. **H3: Boundary-condition scaling mismatch**  
   Endpoint velocity/acceleration constraints may be applied in `tau`-derivative units instead of physical-time units.

4. **H4: Objective surrogate mismatch**  
   The optimizer surrogate may diverge from dense integrated physical control effort `integral ||a_geom - a_grav||^2 d tau`.

## Main Code Changes Made

### 1) Added diagnostic runner

- New file: `diagnose_objective_vs_profile.py`
- Provides ablation-style tests for H1-H4.
- Reports:
  - `J_sur` (optimizer surrogate)
  - `J_dense` (dense integrated effort)
  - `J_plot` (profile-sampled effort)
  - endpoint velocity error checks
  - heuristic verdicts

### 2) Fixed boundary condition scaling

- Updated `orbital_docking/constraints.py`:
  - `build_boundary_constraints(..., T=1.0)` now applies physical-time scaling:
    - velocity constraints scaled by `1/T`
    - acceleration constraints scaled by `1/T^2`
- Updated `orbital_docking/optimization.py`:
  - passes `T = TRANSFER_TIME_S` into `build_boundary_constraints(...)`

### 3) Updated H3 verdict logic

- `diagnose_objective_vs_profile.py` H3 verdict now detects whichever branch (`as-is` vs `scaled-by-T`) has lower endpoint error, so it remains valid before and after the BC fix.

## Key Findings From Runs

- **H3 (BC scaling mismatch): confirmed and fixed in formulation.**
  - After fix, the physically correct branch is now `as-is` input (no manual `* T` preprocessing).
  - Manual `scaled-by-T` now correctly appears worse.

- **H4 (surrogate mismatch): strongly supported.**
  - Significant gap between surrogate and dense integrated effort remains.
  - This is now a primary modeling issue.

- **H1 (KOZ conservatism): weak/inconclusive as dominant cause.**
  - KOZ relaxation had little effect in tested runs relative to large apoapsis growth.

- **H2 (trust-region drift proxy): weak/inconclusive.**
  - Perturbation sensitivity was small in most tests.
  - Some H2 runs were contaminated by intentionally wrong H3 branch choices in diagnostics and should be interpreted carefully.

## Practical Takeaway

The endpoint BC unit mismatch was real and is now fixed.  
The remaining large discrepancy is mostly about objective fidelity (H4), not KOZ conservatism or initialization drift.

## Suggested Next Step

Update diagnostics so H2/H4 always evaluate the physically correct BC branch after the fix, then re-run full sweeps for final apples-to-apples conclusions.
