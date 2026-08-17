# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the only file loaded into every session automatically. Everything an agent needs in order to work on this branch is here, not in a file this one points at.

## Goal (until 2026-08-19)

**A 2-page conference paper, first draft due Wednesday 2026-08-19.** The central claim is the space-time lift. Performance is housekeeping — the numbers need to be reasonable, not impressive.

The goal is not "make the solver good." It is: **make the paper's claims true of the code.** Two failure modes it exists to exclude:

1. The paper claims a convex-hull guarantee the code does not implement. (True today — see G1/G2 below.)
2. A figure comes from a run where elastic slack hid a constraint violation. (Possible today — `wall` renders a trajectory that is 0.112 inside an obstacle.)

## Workstreams

A session can open with just an item id (`B1`, `A2`, `C1`).

**A. Paper** — no dependency on solver correctness except A1.
- A1 freeze the constraint formulation on paper (blocks B4)
- A2 §formulation — lifting, tube geometry, half-space in (x,y,t), monotonicity, finite-height tubes, why this is not time-slicing
- A3 related work, incl. how this differs from TEB
- A4 limitations — passing class comes from initialization; multi-start is not exhaustive
- A5 demo scenario definitions (pendulum, sliding door) as parameters
- A6 figure slots — what each figure must show, defined before any exist

**B. Solver**
- B0 repair the venv (nothing runs until this is done)
- B1 fix G1 — `n[dim-1] = -dot(n_xy, vel)` before normalizing, recompute `lb`
- B2 a KOZ test with a **moving** obstacle that fails before B1 and passes after
- B3 re-benchmark; sweep `cap_bulge_ratio` on `wall`
- B4 fix G2 — one plane per (segment, obstacle), side committed per passing class
- B5 port the SCvx machinery from `main` (only if B3 shows it is still needed)
- B6 procedural seeds (left/right/wait/hurry) + multi-start
- B7 feasibility gate: a run with `total_slack > 0` cannot produce a figure

**C. References** — external ground truth for solver logic and math. When C and an agent's assertion disagree, C wins, and the disagreement is recorded.
- C1 novelty positioning (**blocks A2**) — time-as-a-coordinate is old (Erdmann & Lozano-Pérez configuration-time space, space-time A*/SIPP, velocity obstacles). The contribution has to be narrower and true.
- C2 SCvx ground truth — trust region, exact penalty, what convergence actually requires
- C3 Bezier/B-spline safe-corridor formulations — validates the one-plane-per-segment claim
- C4 topology — H-signature, TEB

## Established facts — do not re-derive

Three separate agents have each spent ~50 tool calls rediscovering these.

- **The merge-base `e849ff7` contains no Rust.** Both lineages wrote `rust_optimizer/` from scratch after 2026-04-03, so `git merge main` is an add/add conflict on every file. Bringing solver work over from `main` is a **file-level port**, never a merge or a rebase.
- `bezier.rs` and `de_casteljau.rs` are **byte-identical** between `main` and this branch.
- **`0d130fc` (the `|pred|` stationarity fix) has no pre-image here.** `spacetime_optimizer.rs` has no merit function, no `pred`, and no ratio test; every step is accepted unconditionally (`spacetime_optimizer.rs:369-392`). "Converged" means `delta < tol && slack < 1e-10` — the step got small, which is not an optimality claim.
- **G1 — the body-case KOZ normal has a zero time coefficient.** `spacetime_constraints.rs:82-104`. The tube is `‖p_xy − p₀ − v·p_t‖ ≤ r`, so its true normal is `(n̂, −n̂·v)`; the code drops the time term. This regressed in `29a43c6` (which fixed an unrelated slider artifact and never mentioned the normal); `c4ec13c` had it right. Because obstacle time windows default to `±inf` (`geometry.py:194-195`), the cap branch is unreachable — so on `original` and `diverse`, **every KOZ row in every iteration is a zero-time row**, and the QP is told that moving a control point in time cannot change clearance.
- **G2 — one plane per (segment, control point, obstacle).** `spacetime_constraints.rs:191-193`. The convex-hull certificate needs **one** plane satisfied by **all** of a segment's control points; per-point planes prove only that each point individually is outside its own plane, which is exactly as strong as sampling the curve at finitely many points. Worse, opposing normals let a segment's hull straddle the tube, and a straddling hull *contains* it. Confirmed against six papers in `doc/notes/_shared/c3_safe_corridor_refs.md`; the only per-control-point method in the literature (EGO-Planner) is an explicit soft penalty claiming no guarantee.
  - **The dense subdivision matrix is NOT part of this defect.** An earlier version of this line implied it was. De Casteljau weights are non-negative and sum to one, so each sub-control-point is a convex combination of the parents and the certificate transfers intact. Sparsifying that matrix would *break* the guarantee.
- `wall` and `diverse` are infeasible. **Parameter tuning was already tried and failed** — `489eb85` shortened the wall's `t_end` 8.0 → 5.0 and added low-segment configs. (Superseded numbers: this line previously recorded −0.112 and −0.495; see the scenario table for measured values.)
- **The KOZ tests cannot fail on G1.** Every one uses `vel=[0,0]`, where a zero time coefficient is genuinely correct, and `tests/unit/test_spacetime_constraints.py:65` asserts `A[:, 2::3] == 0`, which enshrines the bug. Those tests exercise `spacetime_bezier/constraints.py` — the dead Python builder. The Rust builder has no tests at all.
- **Note 001 contradicts itself, so A1 is real work.** `doc/notes/001_problem_formulation/main.tex` §"Body 경우" derives `n = (d_xy/‖d_xy‖, 0)` and states the time component is "정확히 0" — presenting it as the correct formulation. §"잘못된 패턴" then forbids exactly that: evaluate the obstacle at `p₀ + v·t`, build the normal from spatial coordinates only, emit a zero time component. The body derivation *is* those three steps. Separately, §"선형화 지점" presents per-control-point linearization as an improvement over per-centroid ("더 조밀한 표본") — that is G2. **The note enshrines both defects as design. Do not seed the paper from it until A1 revises it.**

## Decided against — do not re-propose

- `git merge` or rebase of `main` (no shared Rust ancestor; see above)
- MIQP / big-M binary side variables — Clarabel has no integer support, and it kills the real-time story
- Purging `orbital_docking` before the deadline — zero paper value, and the SCvx machinery being ported lives in the shared `optimizer.rs`
- A profiling pass — 200-iteration runs are a G2 symptom, not a performance problem
- Porting `main`'s full 5-pillar verification harness — B7 buys the honesty that's needed

## Session protocol

Last act of every session: tick the workstream item and commit. Prefix commits `paper(A2):`, `solver(B1):`, `refs(C1):`. `git log` is the status board — the last five commit messages are surfaced automatically at session start.

## Architecture

- **Rust is the sole optimizer backend.** The Python optimizer has been removed.
- The Rust backend uses Clarabel (interior-point conic solver) for QP subproblems.
- Elastic relaxation (slack variables on KOZ constraints) handles infeasible subproblems.
- The debugger is a trace consumer over Rust execution, not a separate optimizer.

## Core Idea

Lift 2D (or 3D) moving-obstacle avoidance into space-time by adding time as an explicit Bezier coordinate. A Bezier curve in (x, y, t) simultaneously plans **path and timing**. Key properties:

- Constant-velocity obstacles become **straight tubes** in (x, y, t) -- static geometry
- The existing Bezier convex hull property, De Casteljau subdivision, and supporting half-space constraints apply **directly** in the higher-dimensional space
- The optimizer minimizes spatial acceleration energy while threading the curve between obstacle tubes
- Time-limited obstacles (e.g. a wall that disappears) become **finite-height tubes** -- the curve can "wait" then pass through

## What Must Be Reused

The baseline `orbital_docking/` package has dimension-agnostic building blocks. **Import them, don't rewrite them:**

- `bezier.py` -- `get_D_matrix(N)`, `get_E_matrix(N)`, `get_G_matrix(N)`, `BezierCurve` class
- `de_casteljau.py` -- `segment_matrices_equal_params(N, n_seg)`
- `constraints.py` -- reference for how half-space constraints are built (adapt for moving obstacles)

What is genuinely **new** in the spacetime extension:
1. KOZ constraint builder lifts moving obstacles into static world-tubes in space-time. For each segment local control point, a supporting half-space is computed against the capsule geometry in (x, y, alpha*t) scaled space.
2. One Bezier coordinate is the time axis. The control points live in ordinary higher-dimensional Bezier space, so the existing convex-hull and supporting-half-space logic is reused with the time coordinate included in the constraint geometry.
3. Time-limited obstacles become finite-height tubes by clipping with `t_start` and `t_end`.
4. Time monotonicity constraint: `P[i+1, t] - P[i, t] >= min_dt`
5. Objective penalizes only **spatial** acceleration, not time-acceleration

## Important Implementation Warning

The intended model is a static obstacle in space-time, not a 2D obstacle re-evaluated on each time slice.

Bad pattern to avoid:
- Compute `pos0 + vel * t_seg`
- Build a normal using only spatial coordinates
- Emit a half-space with zero coefficient on the time coordinate

Correct pattern:
- Treat the obstacle as one object in `(x, y, t)` or `(x, y, z, t)`
- Build supporting half-spaces in the full lifted space so the time coordinate can appear in the plane equation

> **The shipped code currently does the bad pattern.** `spacetime_constraints.rs:96` — *"Normal's t-component is 0 for the cylinder body"* — and because obstacle time windows default to `±inf`, that branch handles every row on `original` and `diverse`. This section describes the target, not the current state. Fixing it is **B1**.

## Commands

```bash
# Launch the interactive sandbox (live re-solve on every slider change) — main entrypoint
python3 -m spacetime_bezier

# Regenerate the pre-baked scenario JSON (only needed for file:// viewing)
python3 -m spacetime_bezier.io

# Open the static interactive demo (uses pre-baked JSON; no live re-solve)
open figures/spacetime_bezier_interactive.html

# Run live optimizer step debugger
python3 tools/spacetime_opt_debug.py

# Compare Rust output across scenarios
python3 tools/compare_backends.py --all

# Run tests (requires Rust extension built)
pytest

# Build Rust extension
cd rust_optimizer/pybind && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

## Demo Scenarios

Registry keys are in `spacetime_bezier/scenarios.py` (`SCENARIO_MAP`).

| Key | Purpose | Status |
|-----|---------|--------|
Measured 2026-08-17, degree 8, `max_iter=200`, trust radius 0.5, after the solver
was reduced to a single graded path. `certificate` is the violation of the
half-spaces the returned iterate's own control points generate — nonzero means the
convex-hull guarantee does **not** hold for that curve, whatever its clearance.

| Key | Purpose | Status |
|-----|---------|--------|
| `original` | 3 moving obstacles; basic proof of concept | Clearance **+0.317** at 8 segments, certificate clean, but **does not converge** (trust collapse). ⚠️ The previously recorded "Feasible, ~39 iterations / +0.8348" was **the straight-line initial guess handed back unchanged** — the best-iterate fallback was seeded with the input, so the solver's own output was discarded. Verified: initial-guess clearance is exactly +0.8348. |
| `diverse` | Varied sizes/speeds/directions; shows generality | **Infeasible** — clearance −0.710, certificate violated by 3.52, trust collapse at 28 iterations. |
| `wall` | Curve should "wait" until the wall vanishes — time as a real optimization dimension | **Works at 2 segments only**: clearance **+0.038**, converged on stationarity, certificate clean, zero rejected steps. Every higher segment count fails (−0.11 to −0.15) — one plane per *control point* fights itself as segments multiply. |

## Key Files

### Python package (`spacetime_bezier/`)
- `__main__.py` -- Entrypoint: `python3 -m spacetime_bezier` → launches the sandbox
- `sandbox.py` -- Interactive sandbox HTTP server (live re-solve on slider change)
- `optimize.py` -- Public API: `optimize_spacetime()`, `optimize_scenario()`, debug stepper factories
- `constraints.py` -- Python KOZ constraint builder (used by Python debug stepper only)
- `rust_debug_stepper.py` -- Steps through actual Rust optimizer execution via debug log
- `debug_stepper.py` -- Dead Python SCP stepper; no external imports. Kept pending cleanup, do not treat as a reference implementation.
- `debug_session.py` -- Stateful session for the step debugger UI
- `debug_trace.py` -- `DebugFrame` schema shared by both steppers
- `geometry.py` -- `MovingObstacle`, `bezier_curve()`, `obstacle_array_bundle()`
- `objective.py` -- Energy matrix and initial guess construction
- `scenarios.py` -- Scenario definitions and `SCENARIO_MAP` registry
- `io.py` -- CLI entrypoint, JSON I/O, interactive viewer launcher

### Rust optimizer (`rust_optimizer/`)
- `core/src/spacetime_optimizer.rs` -- SCP outer loop with elastic relaxation
- `core/src/spacetime_constraints.rs` -- KOZ capsule geometry, boundary, monotonicity, box constraints
- `core/src/optimizer.rs` -- Shared `solve_qp()` (Clarabel wrapper) and orbital docking optimizer
- `pybind/src/lib.rs` -- PyO3 bindings exposing `optimize_spacetime_bezier()`

### Tools
- `tools/spacetime_opt_debug.py` -- HTTP server for the live step debugger UI
- `tools/compare_backends.py` -- Runs scenarios and diffs results

### Figures
- `figures/spacetime_bezier_interactive.html` -- Interactive HTML demo with debug overlays
- `figures/spacetime_bezier_opt_debug.html` -- Live optimizer step debugger UI
- `figures/spacetime_scenarios.json` -- Optimized control points for all scenarios (generated)

## After the paper

The branch's longer-term direction, paused until 2026-08-19. It still holds.

A **personal research / debug sandbox** for the space-time Bezier idea: an interactive workbench for posing problems, surfacing weakpoints in the optimizer and the formulation, and prototyping against them. Drag obstacles in 3D, slide parameters live, get "why did it fail?" in one line, scrub the SCP iteration history, save and reload scenarios.

Rules that keep it honest:

- **One canonical execution model.** Same request + same configuration = same execution = same result = same traceable explanation. `scp_step` in the Rust core is both the batch iteration and the debug step; a debug session never simulates an alternative stepper.
- **One frontend, two tempos.** Diagnostic mode is a drawer over the same scene, sharing scenario state, camera, and server — never a separate page or a reload.
- **Rust is the sole engine.** It owns the SCP loop, production KOZ half-space generation, candidate acceptance, final iterate selection, and trace emission. Python is not a reference solver or an oracle.
- **Trace is observability over the real run,** not a parallel optimizer. Reconstruction from partial logs is an acceptable bridge; fake stage synthesis is not.
- **Geometry authenticity.** Anything drawn either came from the backend that ran, or is explicitly marked derived. No time-sliced 2D plane labeled as a 3D supporting surface.
- Trace frames must distinguish *current iterate / raw candidate / accepted iterate / best feasible iterate / final returned iterate*. These are different concepts; never conflate them.

Full history of this direction is in git — `VISION.md` and `ToDo.md` were removed in the docs consolidation and can be restored with `git checkout <commit> -- VISION.md ToDo.md`.

## Known Issues

- **G1** and **G2** above — the two KOZ constraint defects. These are the branch's most consequential open problems and are the reason `wall` and `diverse` fail.
- `wall` and `diverse` are infeasible; elastic relaxation keeps the solver running but the returned curve penetrates the obstacle.
- The Rust KOZ builder (`spacetime_constraints.rs`) has no tests. The Python tests cover a builder that production does not use.
- `debug_stepper.py` is dead code — nothing outside the file imports it. Safe to delete after 2026-08-19.
- The venv may be broken (built against a Homebrew Python that has since been upgraded). If `python3 -m spacetime_bezier` fails with a dyld error, rebuild it and re-run `maturin develop --release`. This is **B0**.
