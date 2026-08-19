# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is **direction and guardrails**: what we are doing, what is already settled, and what not
to re-propose. It is the only file loaded into every session automatically, so it is kept short
enough to actually read. Anything carrying a derivation, a measurement, or an evidence trail lives
in a note under `doc/notes/`, and **every such block leaves a one-line pointer here**. If you are
about to add more than five lines of reasoning to this file, it belongs in a note.

Where the reasoning went:
- [`doc/notes/005_formulation_freeze.md`](doc/notes/005_formulation_freeze.md) — the frozen formulation, the velocity/acceleration derivation, occluder geometry
- [`doc/notes/006_solver_record.md`](doc/notes/006_solver_record.md) — evidence behind every established fact, and the scenario measurement history
- [`doc/notes/007_sandbox_direction.md`](doc/notes/007_sandbox_direction.md) — where the sandbox goes after the paper
- [`doc/notes/008_schedule.md`](doc/notes/008_schedule.md) — 제출 마감일과 9월 4일 역산 마일스톤
- [`PAPER_1.md`](PAPER_1.md) / [`PAPER_2.md`](PAPER_2.md) — paper claims and scope boundary
- [`doc/refs/c1_novelty.md`](doc/refs/c1_novelty.md) — prior-art verification; **it outranks any agent's assertion about novelty**

## Goal (until 2026-09-04)

**한국항공우주학회 2026년도 추계학술대회 발표논문. 온라인 제출 마감 2026년 9월 4일(금).** 학회
템플릿이 2페이지 고정이므로, 이 파일이 줄곧 가정해 온 "2-page conference paper"가 곧 이
제출물이다. 역산 마일스톤은 [`doc/notes/008_schedule.md`](doc/notes/008_schedule.md).

**A 2-page conference paper, first draft due Wednesday 2026-08-19.** The central claim is the space-time lift. Performance is housekeeping — the numbers need to be reasonable, not impressive.

The goal is not "make the solver good." It is: **make the paper's claims true of the code.** Two failure modes it exists to exclude:

1. The paper claims a convex-hull guarantee the code does not implement. (True today — see G1/G2 below.)
2. A figure comes from a run where elastic slack hid a constraint violation. (Possible today — `wall` renders a trajectory that is 0.112 inside an obstacle.)

## Formulation decisions — frozen 2026-08-19 (item A1)

Decisions, not findings: they define the problem the solver must answer. An agent that finds one
inconvenient must raise it, not work around it. **Full text, rationale and derivation:
[`doc/notes/005_formulation_freeze.md`](doc/notes/005_formulation_freeze.md).**

1. The curve parameter is not time — **do not write "minimizes spatial acceleration"** until item B8 lands.
2. Velocity and acceleration are rational in the control points, so **no quadratic acceleration-energy matrix exists.** Do not try to derive one.
3. **Rename the objective, do not re-derive it**: it is a parameter-domain smoothness regularizer. Physics goes into constraints.
4. The speed cap is a **constraint, not a cost** — a slant limit per control-polygon leg, proven sufficient in note 005. The acceleration cap is bilinear, so linearize it per iteration.
5. Arrival time is freed with a **linear** time penalty; the subproblem stays a QP. Speed cap and time penalty must land **together**, or arrival collapses to an artifact.
6. The energy-versus-time weight is a **preference, not a threshold** — report it as a scenario parameter.
7. **Every number here is stale by construction** until B8–B10 land. Measurement is one pass at the end; the PI's draft carries none.
8. Linearization handles a non-convex **free side**, never a non-convex **forbidden set** — the latter has no supporting half-space, so the certificate has nothing to stand on.

## Workstreams

A session can open with just an item id (`B1`, `A2`, `C1`).

**A. Paper** — no dependency on solver correctness except A1.
- A1 freeze the constraint formulation. **Done 2026-08-19**, note 005 — what remains is revising note 001, which still enshrines both fixed defects as design
- A2 §formulation — lifting, tube geometry, half-space in (x,y,t), monotonicity, finite-height tubes, why this is not time-slicing
- A3 related work, incl. how this differs from TEB
- A4 limitations — passing class comes from initialization; multi-start is not exhaustive
- A5 demo scenario definitions as parameters. **Decided 2026-08-19:** the line-of-sight demo is a *moving, non-straight* occluder — see [`PAPER_1.md`](PAPER_1.md) §"which line-of-sight demo" and note 005 Part 3
- A6 figure slots — what each figure must show, defined before any exist

**B. Solver**
- ~~B0 repair the venv~~ **DONE**
- ~~B1 fix G1~~ **DONE** — tube is a capsule around the slanted centreline; the time component falls out of the slant
- ~~B2 a KOZ test with a **moving** obstacle that fails before B1 and passes after~~ **DONE** — `tests/unit/test_spacetime_koz_geometry.py`
- B3 re-benchmark, after the B8–B10 freeze. **Do not sweep `cap_bulge_ratio`** — it is dead code and the sweep can only produce a flat line (measured; note 006). Sweep `elastic_weight` instead.
- ~~B4 fix G2~~ **DONE** — one plane per (segment, obstacle) aimed at the segment centroid
- ~~B5 port the SCvx machinery from `main`~~ **DONE** — `848bf3b`, then reduced to one canonical iteration in `9b9c3d3`
- B6 procedural seeds (left/right/wait/hurry) + multi-start. **Demoted 2026-08-19** — it was justified by `wall` being infeasible, which was false. May still buy better local optima; blocks nothing.
- B7 feasibility gate: a run with `total_slack > 0` cannot produce a figure
- B8 rename the objective to what it measures, and add the velocity/acceleration rows as control-point operators (formulation decisions 1–3; the derivation is done, in note 005)
- B9 slant-limit speed cap as a hard constraint, not a cost (formulation decision 4). Needs the second-order cone plumbed into `solve_qp`, or the linear per-axis fallback
- B10 linear time penalty + freed arrival time. **Lands together with B9** (formulation decision 5)
- B11 one run at three spatial dimensions plus time. **Cheap** — the solver is already dimension-generic, see Established facts; this is a scenario definition and a run, not a solver change. Retires the paper's largest weakness
- B12 occlusion constraint builder, moving non-straight occluder as a chain of straight tubes. **Gated** on the per-piece space-time convexity check — do not write it before that check runs (note 005 Part 3)

**C. References** — external ground truth for solver logic and math. When C and an agent's assertion disagree, C wins, and the disagreement is recorded.
- C1 novelty positioning (**blocks A2**) — time-as-a-coordinate is old (Erdmann & Lozano-Pérez configuration-time space, space-time A*/SIPP, velocity obstacles). The contribution has to be narrower and true.
- C2 SCvx ground truth — trust region, exact penalty, what convergence actually requires
- C3 Bezier/B-spline safe-corridor formulations — validates the one-plane-per-segment claim
- C4 topology — H-signature, TEB

## Established facts — do not re-derive

Three separate agents have each spent ~50 tool calls rediscovering these. **Evidence, measurements
and the original diagnoses: [`doc/notes/006_solver_record.md`](doc/notes/006_solver_record.md).**

- **The merge-base `e849ff7` contains no Rust.** Both lineages wrote `rust_optimizer/` from scratch, so `git merge main` is an add/add conflict on every file. Bringing solver work over from `main` is a **file-level port**, never a merge or a rebase.
- `bezier.rs` and `de_casteljau.rs` are **byte-identical** between `main` and this branch.
- **The SCP loop has a real ratio test.** Since `9b9c3d3` there is a merit function, a predicted reduction, and a rho governing accept/reject and the trust update. What is still missing is any **dual or KKT residual**, so `converged` asserts feasibility plus no-further-progress — *not* stationarity of the original problem.
- **G1 — FIXED 2026-08-17.** The keep-out normal's time component now falls out of the tube's slanted centreline instead of being written as zero.
- **G2 — FIXED 2026-08-17.** One plane per (segment, obstacle), aimed at the segment centroid, plus a rotation term in the subproblem. The dense subdivision matrix was never part of this defect — sparsifying it would *break* the guarantee.
- **`wall` and `diverse` are FEASIBLE.** Corrected 2026-08-19. Both clear and certify once the elastic penalty weight exceeds the scenario's exact-penalty threshold; the weight was pinned at 100 inside the Rust binding and unreachable from Python. Below threshold, a penetrating curve is genuinely the cheaper answer — the solver was right about the wrong problem.
- **The certificate is evaluated at the RETURNED iterate.** Fixed 2026-08-19; it used to report the loop's final *reference* point, which is a different trajectory whenever the best-iterate fallback fires.
- **The KOZ tests in `test_spacetime_constraints.py` cannot fail on G1** — every one uses zero velocity, and line 65 asserts the time column is zero, which enshrines the old bug. They exercise `spacetime_bezier/constraints.py`, the dead Python builder. The Rust builder *is* covered, by `tests/unit/test_spacetime_koz_geometry.py`.
- **The solver is already dimension-generic.** `dim` comes from the array shape; the only guard is `dim >= 2` (`rust_optimizer/pybind/src/lib.rs:144`). **A run at three spatial dimensions plus time costs one scenario definition and one run, not a solver change** — item B11, and the cheapest way to retire the paper's largest weakness.
- **The core lift is NOT novel.** Osburn, Peterson & Salmon (arXiv:2508.10203, Aug 2025) published the lift, time as a Bezier coordinate, hull half-spaces in the lifted space, finite-height prisms and time monotonicity — on Clarabel. **What survives is decomposition-free.** Never phrase the hook as "time as a coordinate". See [`doc/refs/c1_novelty.md`](doc/refs/c1_novelty.md).
- **The paper documents did not drift; the containers moved.** Part A of `PAPER_1.md` is byte-identical to the committed `PAPER_CLAIM.md`; Part B is the former `PAPER_OUTLINE.md`. **Risk: `PAPER_OUTLINE.md`, `PAPER_2.md` and `doc/notes/004_probabilistic_koz/` were never committed** and exist only in the working tree.
- **Note 001 contradicts itself, so A1 is real work.** `doc/notes/001_problem_formulation/main.tex` derives the zero-time normal in one section and forbids it in another, and presents per-control-point linearization as an improvement. **The note enshrines both fixed defects as design. Do not seed the paper from it until A1 revises it.**

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

> ⚠️ **The list above is not a novelty claim.** Four of its five items were published by Osburn et
> al. in August 2025 — see [`doc/refs/c1_novelty.md`](doc/refs/c1_novelty.md). It describes what the
> code does, not what is new about it. What is new is *decomposition-free*.

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

The intended model is a static obstacle **in space-time**, not a 2D obstacle re-evaluated on each
time slice. The bad pattern — evaluate the obstacle at `pos0 + vel*t`, build the normal from spatial
coordinates only, emit a zero coefficient on the time coordinate — was defect G1 and is fixed.
`tube_geometry` now builds a capsule around the slanted centreline in the full lifted space.

`SPACETIME_AXIS_SCALE` is pinned at 1.0 as a declared modelling choice; the consequence is a
constant-time cross-section stretched by the obstacle speed, i.e. conservative — measured at 1.32x
on `original`. History and evidence: [`doc/notes/006_solver_record.md`](doc/notes/006_solver_record.md).

## Commands

```bash
# THE entrypoint. There is exactly one. Opens the viewer immediately and solves
# NOTHING up front: each scenario / degree / segment-count is solved on demand,
# ~0.5-0.7s. Exits 1 if the port is busy, naming the pid that holds it and
# warning when that process loaded the Rust extension before your last build.
python3 -m spacetime_bezier

# Refresh the static file:// fallback data. SLOW: 16 configurations, several of
# which run to the iteration cap (~15 min). Only needed for offline viewing.
python3 -m spacetime_bezier --bake

# `python3 -m spacetime_bezier.io` and `.sandbox` are NOT entrypoints as of
# 2026-08-18. They bound a second default port (8765 vs 8767), which is how two
# sandboxes ended up live at once -- the older four days stale on a different
# interpreter, answering with pre-G1/G2 geometry and no signal in the UI.

# Open the static demo (pre-baked data, no live re-solve)
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

**FIGURE-GRADE** means all three of: the sampled curve clears every obstacle, the loop converged for
a principled reason, and the control-point hull satisfies the half-spaces it generates. Clearance
alone means none of that.

> ⚠️ **Every number below is stale by construction** — formulation decision 7. The objective is
> about to change (B8–B10), so nothing measured before that freeze survives it. None of these may
> reach the paper. Re-measure in one pass after the freeze.

| Key | Best config | Clearance | Converged | Certificate | Elastic weight | Figure-grade |
|-----|-------------|-----------|-----------|-------------|----------------|--------------|
| `original` | N8_seg4 | **+0.620** (10 iters) | **yes**, stationary | **0.000** | 100 | **yes — all 5 configs** |
| `diverse` | N8_seg4 | **+0.1136** (9 iters) | **yes** | **0.000** | 800 | **yes** |
| `wall` | N10_seg16 | **+0.0751** (19 iters) | **yes** | **0.000** | 100000 | **yes** |

The elastic weight is part of the result and part of reproducing it: it is a **modelling decision,
not a tuning knob**, because above the exact-penalty threshold the penalized and constrained
problems share a solution and below it they do not. Measured thresholds and the full history — the
segment-count reversal, the `+0.8348` straight-line artifact, why `wall` looked unsolved for months
— are in [`doc/notes/006_solver_record.md`](doc/notes/006_solver_record.md).

## Key Files

The full tree is [`README.md`](README.md) § Repo map — not repeated here. Only the entries that
carry a warning or are easy to mistake:

- `spacetime_bezier/__main__.py` -- **the** entrypoint. `io.py` and `sandbox.py` are not, as of 2026-08-18
- `spacetime_bezier/optimize.py` -- public API, the elastic-weight ladder, `optimize_scenario`
- `rust_optimizer/core/src/spacetime_constraints.rs` -- KOZ capsule geometry; both fixed defects lived here
- `rust_optimizer/core/src/spacetime_optimizer.rs` -- SCP loop, ratio test, certificate at the returned iterate
- `rust_optimizer/core/src/optimizer.rs` -- `solve_qp`; **emits linear cones only**, see note 005 on the speed cap
- `spacetime_bezier/constraints.py`, `debug_stepper.py` -- **dead**, see Known Issues
- `doc/notes/001_problem_formulation/` -- **contradicts itself**; do not seed the paper from it
- `doc/notes/005_formulation_freeze.md`, `006_solver_record.md`, `007_sandbox_direction.md`, `doc/refs/c1_novelty.md` -- the reasoning moved out of this file

## After the paper

The branch's longer-term direction — a personal research and debug sandbox for the space-time Bezier
idea — is paused until the paper ships and still holds. Its five honesty rules (one canonical
execution model, one frontend, Rust as sole engine, trace as observability, geometry authenticity)
govern how code is written here even now:
[`doc/notes/007_sandbox_direction.md`](doc/notes/007_sandbox_direction.md).

## Known Issues

- **The viewer's inline `const SCENARIOS` blob is test mock data, and it is committed.** An
  integration test writing to a tmp dir used to overwrite the tracked `figures/` page; it now reads
  one control point at the origin and no `diverse` or `wall`. Fixed at the source 2026-08-18, but
  **the committed data is still wrong and only `--bake` regenerates it.** Serving over HTTP is
  unaffected — only `file://` shows the mock.
- **Baked and live runs use different solver parameters.** `--bake` defaults to `tol=1e-12,
  max_iter=10000`; the sandbox posts `tol=1e-6, max_iter=30`; the scenario table was made with
  neither, and there is no CLI flag for trust radius at all. A live violation of "one canonical
  execution model", left alone deliberately — changing a default silently changes every number.
- **`BENCHMARKS.md` is untracked and four months stale.** It cites the deleted `ToDo.md` and
  describes a solver that predates the SCvx port. **Its numbers must never be quoted.**
- `spacetime_bezier/constraints.py`, `debug_stepper.py` and `tests/unit/test_spacetime_constraints.py`
  are a dead triple — production uses none of them, and they pin a time scale (0.5) that disagrees
  with the Rust (1.0). Delete all three together.
- The venv may be broken (built against a Homebrew Python since upgraded). If `python3 -m
  spacetime_bezier` fails with a dyld error, rebuild it and re-run `maturin develop --release`.
