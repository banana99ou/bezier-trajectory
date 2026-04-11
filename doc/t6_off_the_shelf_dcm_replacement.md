# T6 Off-The-Shelf DCM Replacement

## Verdict

Recommended replacement for the bespoke downstream `DCM` path:

- `Dymos` on top of `OpenMDAO`

Fallback only if dependency friction blocks `Dymos`:

- `CasADi + IPOPT`

Not recommended as the primary replacement:

- `pycollo`

## Why A Replacement Is Needed

The current downstream stack for `T6` is implemented in:

- `orbital_docking/downstream_collocation.py`

That makes the experiment scientifically weak for the stated question.

The comparison was supposed to answer:

- does the Bézier trajectory help as a warm start for downstream direct collocation?

But once the downstream solver itself became a bespoke, underconstrained implementation, the comparison drifted into:

- how does this custom NLP behave?

That is the wrong target.

## Selection Criteria

The replacement needs to satisfy all of the following better than the current custom DCM:

1. truly off-the-shelf direct collocation or pseudospectral transcription support
2. exact endpoint constraints, path constraints, and fixed-duration support
3. easy injection of two initial guesses into the same downstream problem
4. a well-used, externally maintained library rather than another repo-local solver
5. Python interoperability with the current repo
6. enough trust that a `T6` result would primarily reflect warm-start quality rather than homemade NLP behavior

## Candidate Comparison

### 1. `Dymos` + `OpenMDAO`

Assessment:

- strongest candidate

Why:

- it is an actual off-the-shelf trajectory-optimization framework, not just a symbolic backend
- it already provides collocation / pseudospectral transcriptions, path constraints, boundary constraints, timeseries handling, and optimizer integration
- it is explicitly designed for optimal control problems in Python
- it supports fixed-time trajectories, state/control initialization, and path constraints in a way that matches the `T6` fairness requirement well
- the downstream comparison can be posed once, then solved twice with only the initial guess changed

Repo fit:

- the repo already has Python ODE-style dynamics pieces that can be wrapped as a Dymos phase ODE:
  - `r_dot = v`
  - `v_dot = u + g_J2(r)`
- the existing naive and Bézier warm-start arrays can be mapped into Dymos state/control timeseries
- the exact endpoint position/velocity constraints map naturally to initial/final boundary constraints
- KOZ radius can be posed as a path constraint on `||r|| - r_koz`

Main downside:

- heavier dependency stack than the current repo
- likely requires `openmdao`, `dymos`, and preferably a strong NLP backend

Why it is still the best choice:

- the added dependency cost buys a large reduction in scientific ambiguity
- unlike the current custom DCM, Dymos lets `T6` test a warm-start question on a trusted optimal-control framework instead of a homemade transcription

### 2. `CasADi + IPOPT`

Assessment:

- viable fallback, but not the best primary replacement

Why:

- `CasADi` is a mature optimal-control / nonlinear-optimization tool with strong derivative support and standard use in trajectory optimization
- however, it is still closer to a toolkit than a fully packaged trajectory layer
- you would still need to write a fair amount of direct-collocation formulation code yourself

Why it loses to `Dymos` here:

- it reduces homemade-NLP risk less than Dymos
- it still leaves substantial transcription responsibility in repo-local code
- that is better than the current `trust-constr` spike, but still not as clean as using a higher-level off-the-shelf trajectory framework

When to use it anyway:

- if `Dymos` dependency/setup overhead proves unacceptable
- if the team wants a lighter-weight Python path while accepting more custom formulation code

### 3. `pycollo`

Assessment:

- not recommended

Why:

- it is much less established in the ecosystem
- available signals indicate pre-alpha maturity
- it does not buy enough trust relative to the migration cost

## Recommendation

Use `Dymos` as the replacement downstream solver for `T6`.

More specifically:

1. retire `orbital_docking/downstream_collocation.py` from the paper-facing `T6` path
2. rebuild the downstream problem once in `Dymos`
3. run the exact same downstream problem twice:
   - naive initialization
   - Bézier warm-start initialization
4. keep the fairness rule strict: only the initial guess changes

## Minimal Mapping Into `Dymos`

The smallest credible downstream problem in `Dymos` would be:

- states:
  - `r_x, r_y, r_z`
  - `v_x, v_y, v_z`
- controls:
  - `u_x, u_y, u_z`
- duration:
  - fixed at `1500 s`
- dynamics:
  - two-body + J2 gravity using the same physics source already used upstream
- path constraint:
  - spherical KOZ margin
- boundary constraints:
  - exact initial and final position/velocity vectors
- objective:
  - same control-effort objective for both runs

Initial guess handling should be:

- naive: Hermite-derived state/control profiles
- warm start: sampled upstream Bézier state/control profiles

That gives a much cleaner `T6` than continuing to patch the bespoke DCM.

## Final Recommendation

If the goal is a scientifically defensible `T6`, the replacement should be:

- `Dymos` first choice
- `CasADi + IPOPT` only fallback

The current bespoke DCM should not be the basis for further paper-facing reruns.
