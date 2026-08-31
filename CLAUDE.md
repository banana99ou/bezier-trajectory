# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is **direction and guardrails**: what we are doing, what is already settled, and what not
to re-propose. It is the only file loaded into every session automatically, so it is kept short
enough to actually read. Anything carrying a derivation, a measurement, or an evidence trail lives
elsewhere. If you are about to add more than five lines of reasoning to this file, it belongs in
one of the documents below.

Where the reasoning went:
- [`PAPER_1.md`](PAPER_1.md) — paper 1: claim, the formulation derivation, occluder geometry, demo scenario, 제출 일정
- [`PAPER_2.md`](PAPER_2.md) — paper 2: risk field, prior art, scope boundary against paper 1
- [`doc/refs/novelty_positioning.md`](doc/refs/novelty_positioning.md) — prior-art verification; **it outranks any agent's assertion about novelty**
- `doc/notes/` — 연구노트, bound for the lab's shared repository. **Not part of this repository.** Each note is its own git repo on disk and this repo ignores the whole folder; read `doc/notes/README.md` before putting anything there
- **`tests/`** — every measured number lives beside the assertion that re-measures it. A number in a markdown file cannot fail, so it is not evidence

## Goal (until 2026-09-04)

**한국항공우주학회 2026년도 추계학술대회 발표논문. 온라인 제출 마감 2026년 9월 4일(금).** 학회
템플릿 파일 자체가 2페이지이므로 이 파일이 줄곧 가정해 온 "2-page conference paper"를 그대로
쓴다 — 다만 **학회가 정규 학술대회에 대해 분량 제한을 명시한 문서는 없다**(확인 2026-08-20,
[`paper/README.md`](paper/README.md)). 2페이지는 선택이지 인용할 규정이 아니다. 역산 마일스톤과
마감 재확인 절차는 [`PAPER_1.md`](PAPER_1.md) §제출 일정.

The central claim is decomposition-free, not the space-time lift. Performance is housekeeping — the numbers need to be reasonable, not impressive.

The goal is not "make the solver good." It is: **make the paper's claims true of the code.** Two failure modes it exists to exclude:

1. The paper claims a convex-hull guarantee the code does not implement. (Closed by the G1 and G2 fixes; it is what the certificate exists to keep closed.)
2. A figure comes from a run where elastic slack hid a constraint violation. (Closed 2026-08-20 — B7's `figure_grade` gate, and `tools/make_paper_figure.py` refuses to draw a run that fails it.)

## Formulation decisions — frozen 2026-08-19 (item A1)

Decisions, not findings: they define the problem the solver must answer. An agent that finds one
inconvenient must raise it, not work around it. **Full text, rationale and derivation:
[`PAPER_1.md`](PAPER_1.md) §"The derivation behind the decisions".**

1. The curve parameter is not time — **do not write "minimizes spatial acceleration"** until item B8 lands.
2. Velocity and acceleration are rational in the control points, so **no quadratic acceleration-energy matrix exists.** Do not try to derive one.
3. **Rename the objective, do not re-derive it**: it is a parameter-domain smoothness regularizer. Physics goes into constraints.
4. The speed cap is a **constraint, not a cost** — a slant limit per control-polygon leg, proven sufficient in PAPER_1. The acceleration cap is bilinear, so linearize it per iteration.
5. Arrival time is freed with a **linear** time penalty; the subproblem stays a QP. Speed cap and time penalty must land **together**, or arrival collapses to an artifact.
6. The energy-versus-time weight is a **preference, not a threshold** — report it as a scenario parameter.
7. **Every number here is stale by construction** until B8–B10 land. Measurement is one pass at the end; the PI's draft carries none.
8. Linearization handles a non-convex **free side**, never a non-convex **forbidden set** — the latter has no supporting half-space, so the certificate has nothing to stand on. **Corrected 2026-08-31, PAPER_1 §"The obstacle" / §"The wall":** the forbidden set — the lifted tube, and the shadow — IS non-convex, and that is the problem the construction solves. What has a supporting half-space is each *approach's piece* of the clipped keep-out volume, and the wall is the support of that piece. The decision stands as: never build a wall against the forbidden set as a whole, and never take the tube for a capsule.

## Workstreams

A session can open with just an item id (`B1`, `A2`, `C1`).

**A. Paper** — no dependency on solver correctness except A1.
- A1 freeze the constraint formulation. **Done 2026-08-19**, absorbed into PAPER_1 — what remains is revising note 001, which still enshrines both fixed defects as design
- A2 §formulation — lifting, tube geometry, half-space in (x,y,t), monotonicity, finite-height tubes, why this is not time-slicing
- A3 related work, incl. how this differs from TEB
- A4 limitations — passing class comes from initialization; multi-start is not exhaustive
- A5 demo scenario definitions as parameters. **Decided 2026-08-19** — see [`PAPER_1.md`](PAPER_1.md) §"The demo scenario and its figure"
- A6 figure slots — what each figure must show, defined before any exist. The occlusion figure's slot is defined in [`PAPER_1.md`](PAPER_1.md) §"The demo scenario and its figure"

**B. Solver**
- ~~B0 repair the venv~~ **DONE**
- ~~B1 fix G1~~ **DONE** — tube is a capsule around the slanted centreline; the time component falls out of the slant. (Capsule was the constant-velocity case B1 fixed; the tube has since been generalised to a curved one, and to the shadow's center surface — PAPER_1 §"The obstacle", §"The shadow in the lifted space")
- ~~B2 a KOZ test with a **moving** obstacle that fails before B1 and passes after~~ **DONE** — `tests/unit/test_spacetime_koz_geometry.py`
- ~~B3 re-benchmark after the freeze~~ **DONE 2026-08-20** — one pass, 28 runs (README corrected its own earlier "22"), table in README §Measurements. Swept `elastic_weight` via the ladder; `cap_bulge_ratio` untouched (dead code, measured flat)
- ~~B4 fix G2~~ **DONE** — one plane per (segment, obstacle) aimed at the segment centroid
- ~~B5 port the SCvx machinery from `main`~~ **DONE** — `848bf3b`, then reduced to one canonical iteration in `9b9c3d3`
- B6 procedural seeds (left/right/wait/hurry) + multi-start. **Demoted 2026-08-19** — it was justified by `wall` being infeasible, which was false. May still buy better local optima; blocks nothing.
- ~~B7 feasibility gate~~ **DONE 2026-08-20** — `b8728b0`, slack tolerance corrected in `5982d4c`: `figure_grade` requires converged AND certificate ≤ 1e-6 AND clearance > 0 AND **slack ≤ 1e-6** (placed inside an eight-order gap between the populations — 117× above the worst good residue, six orders below the smallest bad one — not on its edge). Each condition tested to sink the gate alone. `tools/make_paper_figure.py` refuses to draw a run that fails it
- ~~B8 rename the objective; derivative operators~~ **DONE 2026-08-20** — `efb0a70`: `build_smoothness_regularizer` (alias warns), Rust comments corrected, difference/derivative operators built from `get_D_matrix`. The regularizer provably scores a cruise and a wait-then-dash identically
- ~~B9 slant-limit speed cap~~ **DONE 2026-08-20** — `c1866ea`, **second-order cone, not the fallback**. The cone carries no slack: the elastic penalty may relax keep-out rows, never the physics. Off by default (`v_max=None`)
- ~~B10 time penalty + freed arrival~~ **DONE 2026-08-20** — `c1866ea`, same commit as B9. `free_arrival_time` is an explicit flag; `time_weight>0` with no speed cap raises `UncappedTimePenaltyError` — the trap was measured first (arrival collapses to exactly min_dt×gaps and ignores the weight). Golden config unchanged. **The B8–B10 freeze has landed: the one-pass re-measurement (B3) is now unblocked**
- ~~B11 one run at three spatial dimensions plus time~~ **DONE 2026-08-19** — `fence3d` (renamed from `wall3d` 2026-08-24, definition unchanged), committed `c25503c`: all four configs converge and certify on the first ladder rung, the curve climbs the fence rather than going around, and deleting the third spatial coordinate makes every config penetrate. Its timing profile is an objective artifact (control-point times on the min_dt floor) — geometry is evidence, timing is not, until B8–B10 land
- ~~B12 line-of-sight occlusion~~ **DONE 2026-08-20, REBUILT 2026-08-30** — `ed8c50a`+`dd51d3e`+`99a6714`, then `46a15f7`. **There is no occlusion builder, and there are not two keep-out zones.** An obstacle's zone and the shadow it casts are one set at two stretch factors, so the centreline generalizes to a **center surface**: the shadow of the centreline with the ground station as a point light source. `u = 1` is the body — the obstacle's radius never changes, the umbra widens, because a point source at finite distance casts constant *angular* thickness. For a fixed `u` the stretch is affine, so the surface is again a Bezier curve and subdivision, hodographs and the De Casteljau support bounds carry over unchanged. **No station means `u = 1` identically and every formula reduces to the plain lifted tube** — that reduction is the regression the rewrite stands on, witnessed by `tests/unit/test_spacetime_koz_geometry.py`. Derivation and the falsification test live in the `rust_optimizer/core/src/spacetime_generator.rs` module docstring and `tests/unit/test_center_surface_geometry.py`
  - **API.** `spacetime_occlusion_rows_exact` is gone. `spacetime_koz_rows_exact` takes `stations` and returns both wall kinds in one row set, split on a per-row station index (`-1` is the obstacle's own zone). Rows stay elastic-relaxable, `occlusion_violation_reference` at the returned iterate still feeds `figure_grade`, and `compute_los_margin` in geometry.py is still the independent pure-Python check
  - **RETIRED, do not reintroduce:** the per-piece conservative containing ball and the single flat wall against it; and clipping the BODY by proximity, then casting the cone from the shrunken body — the shadow depends on the whole occluder, so that destroys containment. **A zero time coefficient is now a REPORTABLE DEFECT on every row, shadow rows included** — the old "zero by design, a prism with time-parallel walls, not-G1" note is dead, because the center surface carries the obstacle's own time coordinate
  - `station_fence`'s measured numbers (baseline loses the link 7.79 s of 10, the climb to z=2.12 over a 0.90 fence) are **stale** — retired builder. Its 2D-is-genuinely-infeasible finding is geometry, not builder, and stands

**C. References** — external ground truth for solver logic and math. When C and an agent's assertion disagree, C wins, and the disagreement is recorded.
- C1 novelty positioning (**blocks A2**) — time-as-a-coordinate is old (Erdmann & Lozano-Pérez configuration-time space, space-time A*/SIPP, velocity obstacles). The contribution has to be narrower and true.
- C2 SCvx ground truth — trust region, exact penalty, what convergence actually requires
- C3 Bezier/B-spline safe-corridor formulations — validates the one-plane-per-segment claim
- C4 topology — H-signature, TEB

## Established facts — do not re-derive

Three separate agents have each spent ~50 tool calls rediscovering these. **Every number below is
carried by a test that fails when it goes stale** — `tests/unit/test_spacetime_koz_geometry.py`
and `tests/integration/test_scvx_invariants.py` are where the evidence lives.

- **The merge-base `e849ff7` contains no Rust.** Both lineages wrote `rust_optimizer/` from scratch, so `git merge main` is an add/add conflict on every file. Bringing solver work over from `main` is a **file-level port**, never a merge or a rebase.
- `bezier.rs` and `de_casteljau.rs` are **byte-identical** between `main` and this branch.
- **The SCP loop has a real ratio test.** Since `9b9c3d3` there is a merit function, a predicted reduction, and a rho governing accept/reject and the trust update. What is still missing is any **dual or KKT residual**, so `converged` asserts feasibility plus no-further-progress — *not* stationarity of the original problem.
- **G1 — FIXED 2026-08-17.** The keep-out normal's time component now falls out of the tube's slanted centreline instead of being written as zero.
- **G2 — FIXED 2026-08-17, and the key widened 2026-08-26 (twice).** One plane per **(segment, obstacle, local approach)** — the band cut at every interior local maximum of the centreline's distance to the segment centroid — aimed at the segment centroid, plus a rotation term in the subproblem. It was one plane per (segment, obstacle) until the wall moved onto the clipped volume, then one per connected component until the same day's second correction: a bend that WRAPS the centroid is one component, and its single wall is non-separating by theorem (panel B2 measured it at margin −1.154 on a clear segment; the cut gives two walls at +0.529/−0.162). The wire name `component_idx` survives and now indexes approaches. Grouping rows without the index folds two genuinely different walls together. The dense subdivision matrix was never part of this defect — sparsifying it would *break* the guarantee.
- **The keep-out wall is the SUPPORT of the clipped KOZ volume, not a hull projection.** Since `0a8bc9e`+ the plane is built against `K_m ∩ B(c, r_clip)` itself: normal from the component's own nearest centreline point, offset a rigorous De Casteljau ceiling on that component's support. **The retired route — select a parameter interval, hull the subdivided centreline, project the centroid, push out by `r_m` — is an OUTER approximation**, measured 1.13–2.23 past the clip radius on all six `curve` pairs, and it can never yield more than one wall. Do not reintroduce it. `r_clip = clamp(d, r_m, r_clip_max)`; the floor binds exactly when the centroid is inside the keep-out zone.
- **`wall` and `diverse` are FEASIBLE.** Corrected 2026-08-19. Both clear and certify once the elastic penalty weight exceeds the scenario's exact-penalty threshold; the weight was pinned at 100 inside the Rust binding and unreachable from Python. Below threshold, a penetrating curve is genuinely the cheaper answer — the solver was right about the wrong problem.
- **The certificate is evaluated at the RETURNED iterate.** Fixed 2026-08-19; it used to report the loop's final *reference* point, which is a different trajectory whenever the best-iterate fallback fires.
- **`obstacle_pos0` / `obstacle_vel` are not parameters of anything.** `0918df5` moved the Rust API to lifted control points and did not update the tests. 75 of 338 tests were red at that commit and stayed red; 13 of them were `tests/unit/test_spacetime_koz_geometry.py`, which CLAUDE.md lists as the Rust builder's coverage — so **the Rust builder had no live coverage at all** between `0918df5` and 2026-08-26. Repaired for that file; **52 are still red at `46a15f7` (52 failed / 298 passed / 1 skipped, measured 2026-08-30)**, the same drift in `test_objective_matches_rust.py`, `test_speed_cap_and_time_penalty.py`, `test_station_fence_scenario.py` and others. `test_occlusion_geometry.py` is no longer among them because `46a15f7` deleted it — every one of its assertions was about the retired builder's objects. Convert `pos0`/`vel` obstacles with `spacetime_bezier.geometry.obstacle_array_bundle`; a legacy obstacle with no window needs one, because the active window is now intrinsic to the control points.
- **The dead Python triple is gone** — `327a28e` deleted `spacetime_bezier/constraints.py`, `debug_stepper.py`, `tests/unit/test_spacetime_constraints.py` and `BENCHMARKS.md` on 2026-08-20. The lesson they carried is worth keeping: those KOZ tests could not fail on G1, because every one used zero velocity and one asserted the time column was zero, enshrining the bug. **The Rust builder's coverage is `tests/unit/test_spacetime_koz_geometry.py`** (and `tests/unit/test_center_surface_geometry.py` since `46a15f7`); nothing else covers it.
- **The solver is already dimension-generic.** `dim` comes from the array shape; the only guard is `dim >= 2` (`rust_optimizer/pybind/src/lib.rs:144`). **A run at three spatial dimensions plus time costs one scenario definition and one run, not a solver change** — item B11, and the cheapest way to retire the paper's largest weakness.
- **The core lift is NOT novel.** Osburn, Peterson & Salmon (arXiv:2508.10203, Aug 2025) published the lift, time as a Bezier coordinate, hull half-spaces in the lifted space, finite-height prisms and time monotonicity — on Clarabel. **What survives is decomposition-free.** Never phrase the hook as "time as a coordinate". See [`doc/refs/novelty_positioning.md`](doc/refs/novelty_positioning.md).
- **Note 001 contradicts itself, so A1 is real work.** In `doc/notes/001_problem_formulation/main.tex` (a separate repo on disk, not tracked here): §"Body 경우" derives the zero-time normal and calls it "정확히 0"; §"잘못된 패턴" then forbids exactly those three steps; §"선형화 지점" presents per-control-point linearization as an improvement ("더 조밀한 표본"), which is G2. **The note enshrines both fixed defects as design. Do not seed the paper from it until A1 revises it.**

## Decided against — do not re-propose

- `git merge` or rebase of `main` (no shared Rust ancestor; see above)
- MIQP / big-M binary side variables — Clarabel has no integer support, and it kills the real-time story
- Purging `orbital_docking` before the deadline — zero paper value, and the SCvx machinery being ported lives in the shared `optimizer.rs`
- A profiling pass — 200-iteration runs are a G2 symptom, not a performance problem
- Porting `main`'s full 5-pillar verification harness — B7 buys the honesty that's needed

## Session protocol

Last act of every session: tick the workstream item and commit. Prefix commits `paper(A2):`, `solver(B1):`, `refs(C1):`. `git log` is the status board — the last five commit messages are surfaced automatically at session start.

## Architecture

Rust is the sole optimizer backend; the Python optimizer has been removed. Clarabel (interior-point
conic) solves the QP subproblems, and elastic relaxation — slack variables on the KOZ rows —
handles infeasible ones.

Five rules keep it honest. They were written for the post-paper sandbox and they govern how code is
written here now:

- **One canonical execution model.** Same request + same configuration = same execution = same
  result = same traceable explanation. `scp_step` in the Rust core is both the batch iteration and
  the debug step; a debug session never simulates an alternative stepper.
- **One frontend, two tempos.** Diagnostic mode is a drawer over the same scene, sharing scenario
  state, camera, and server — never a separate page or a reload.
- **Rust is the sole engine.** It owns the SCP loop, production KOZ half-space generation,
  candidate acceptance, final iterate selection, and trace emission. Python is not a reference
  solver and not an oracle.
- **Trace is observability over the real run,** not a parallel optimizer. Reconstruction from
  partial logs is an acceptable bridge; fake stage synthesis is not.
- **Geometry authenticity.** Anything drawn either came from the backend that ran, or is explicitly
  marked derived. No time-sliced 2D plane labeled as a 3D supporting surface.

Trace frames must distinguish *current iterate / raw candidate / accepted iterate / best feasible
iterate / final returned iterate*. These are different concepts; never conflate them.

## Core Idea

Lift 2D (or 3D) moving-obstacle avoidance into space-time by adding time as an explicit Bezier
coordinate, so a curve in (x, y, t) plans **path and timing at once**. An obstacle with arbitrary
polynomial motion becomes a **curved tube** around its lifted centreline (constant velocity is the
straight special case and is never the setup); a time-limited one is a finite tube the curve can
wait out; and with a ground station **the keep-out zone is the obstacle and its shadow as one set**
— the centreline generalises to a **center surface**, the shadow of the centreline with the station
as a point light, the obstacle's radius fixed and the umbra widening as a cone. The existing
convex-hull / De Casteljau / supporting-half-space machinery applies directly in the lifted space,
to the tube and to the shadow alike.

> ⚠️ **That paragraph is a description, not a novelty claim.** Osburn et al. published all of it in
> August 2025 — [`doc/refs/novelty_positioning.md`](doc/refs/novelty_positioning.md). What is new
> here is *decomposition-free*. Never open with "time as a coordinate."

**The bad pattern, which was defect G1 and is fixed:** evaluate the obstacle at `pos0 + vel*t`,
build the normal from spatial coordinates only, emit a zero coefficient on the time coordinate.
That is a 2D obstacle re-evaluated per time slice, not a static obstacle in space-time. The zone is
now generated in the full lifted space by `Generator` in
`rust_optimizer/core/src/spacetime_generator.rs`, and **a zero time coefficient is a reportable
defect on every row, the shadow's included.** `SPACETIME_AXIS_SCALE` stays pinned at 1.0 as a
declared modelling choice — the consequence, and the measured conservatism it buys, are documented
at `rust_optimizer/core/src/spacetime_constraints.rs:97`.

**Reuse, do not rewrite.** `orbital_docking/` supplies the dimension-agnostic building blocks the
spacetime code imports: `bezier.py` (D/E/G matrices, `BezierCurve`) and `de_casteljau.py`
(`segment_matrices_equal_params`).

## Commands

Full quickstart — install, build, run, test — is [`README.md`](README.md) §Quickstart. Only the
two things that are guardrails rather than instructions:

```bash
python3 -m spacetime_bezier          # THE frontend — config panel, solve, one page on 8767
python3 -m spacetime_bezier.viewer   # superseded sanity viewer, still on disk
./tools/watch_paper.sh               # manuscript .docx -> .md sidecar + .pdf, on every save
```

- **Writing the paper is its own workflow** — the tools, the venue's rules, and what not to assert
  about them are in [`paper/README.md`](paper/README.md). Read it before touching a manuscript.

- **Two ways in, never two servers.** Both bind 8767, and both exit 1 when it is held, naming the
  pid and warning when that process loaded the Rust extension before your last build. The shared
  port is the mutual exclusion — do not work around it by changing the port.
- `python3 -m spacetime_bezier.io` and `.sandbox` are **not** entrypoints, as of 2026-08-18. They
  bound a second default port (8765 against 8767), which is how two sandboxes ended up live at
  once — the older four days stale on a different interpreter, answering with pre-G1/G2 geometry
  and no signal in the UI.

## Demo Scenarios

Registry keys are in `spacetime_bezier/scenarios.py` (`SCENARIO_MAP`). What each one shows, the
last measured pass, and the elastic weight each result needs: [`README.md`](README.md)
§Measurements.

The B8–B10 freeze landed and the one-pass re-measurement ran **2026-08-20** (item B3); README §Measurements is current at defaults. Runs enabling `v_max` / `time_weight` / `free_arrival_time` are different problems — re-measure per scenario.

**`loiter` is the paper's demo scenario and its elastic weight is MEASURED.** Corrected
2026-08-31 — this paragraph previously said the weight was unmeasured, that `loiter` was absent from
README §Measurements, and that the registry held 1e6. All three were wrong. The registry holds
**1e5** (`SCENARIO_ELASTIC_WEIGHT` in `spacetime_bezier/scenarios.py`), the measurement was taken
2026-08-30 across six runs — defaults and priced, each with and without `sound_clip` — and README
§Measurements carries it in both the main table and its own dated block. Every rung certifies and
returns the same trajectory; the rungs differ only in iteration count, which is why 1e5 is
registered rather than the first rung that certifies.

**Re-measured 2026-08-31 on the current build: all six runs reproduce exactly** — iterations
3/3/3/3/10/12, weights 100 ×4 then 1e5 ×2, clearances 35.3204 / 35.3767 / 36.8417 / 36.8459 /
24.3026 / 30.5243, arrival 80.000 pinned ×4 then 51.502 and 55.747, hull and occlusion certificates
0.000 throughout, slack ≤ 3e-12, figure-grade on all six. The full record is the comment above
`"loiter"` in `scenarios.py`; do not quote a rung from here.

## Key Files

The full tree is [`README.md`](README.md) § Repo map — not repeated here. Only the entries that
carry a warning or are easy to mistake:

- `spacetime_bezier/__main__.py` -- serves `frontend.py` as of 2026-08-21, THE entrypoint. `io.py` and `sandbox.py` are not entrypoints (2026-08-18), and the sandbox is no longer what `-m spacetime_bezier` starts
- `spacetime_bezier/frontend.py` + `static/frontend.html` -- the one frontend (`fe5c510`): config panel, solve endpoint, axis picker with server-supplied banners (t always vertical when shown), opt-in layers all default-off, diagnostics drawer. All verdicts server-side via `figure_grade_failures`; replay reuses the trace viewer's child script by import so parameters cannot drift. **Covered by `tests/integration/test_frontend.py` (38 tests)** -- request/response path, port mutual exclusion, plane patches proven against their own rows, result/replay cache, per-frame replay planes, cancel kills a running solve
- `spacetime_bezier/viewer.py` -- superseded by `frontend.py` 2026-08-21, no longer an entrypoint. Nothing stored, the solve
  path is just the solve, and the client only draws -- verdict fields are computed server-side from the
  solver's own numbers. Serves `static/viewer.html`; shares port 8767 so it cannot run beside the sandbox
- `spacetime_bezier/optimize.py` -- public API, the elastic-weight ladder, `optimize_scenario`
- `rust_optimizer/core/src/spacetime_generator.rs` -- **the set itself**: `Generator`, the centreline read through a station as a point light source. `station: None` is the plain lifted tube. The module docstring carries the derivation and why the widening radius is required rather than cosmetic
- `rust_optimizer/core/src/spacetime_obstacle.rs` -- the wall built on that set: `clip_band`, `component_support`, `clip_geometry`, `rotation_correction`
- `rust_optimizer/core/src/spacetime_constraints.rs` -- row assembly and bookkeeping, `SPACETIME_AXIS_SCALE`; both fixed defects lived here, and the retired occlusion builder did too until `46a15f7`
- `rust_optimizer/core/src/spacetime_optimizer.rs` -- SCP loop, ratio test, certificate at the returned iterate
- `rust_optimizer/core/src/optimizer.rs` -- `solve_qp` + `solve_qp_with_socs`; second-order cones landed with B9 (the speed-cap cone carries no elastic slack, by design)
- `paper/` -- one directory per venue, holding the manuscript in the society's own template. [`paper/README.md`](paper/README.md) is the handbook: the render/watch pipeline, the verified venue rules, the invariants that break silently (one template paragraph carries the section break that keeps the title block out of the two-column body), and the state of the draft. `tools/render_paper.py`, `watch_paper.sh`, `make_manuscript_skeleton.py`, `docx_edit.py` are its machinery
- `doc/notes/001_problem_formulation/` -- a separate repo, ignored by this one. **Contradicts itself**; do not seed the paper from it

## Known Issues

- **ONE TEST IS RED, deliberately, and the decision is the user's.** `wall` was densified
  2026-08-24 at the user's request (spacing 0.8 → 0.5, 13 → 21 circles). The forbidden slab barely
  moves, but 8 more per-segment planes make the relaxation more conservative, and the consequence
  is measured: `wall` N8_seg2 went from +0.102 (clears while standing on 1.35 of slack) to −0.129
  (penetrates outright). That row was the ONLY real specimen of "clears by sampling yet fails the
  certificate", so `tests/integration/test_figure_grade_gate.py::test_a_clearing_run_can_still_be_standing_on_slack`
  now fails. 19 configurations across `wall`, `diverse` and `original` were probed; no replacement
  specimen exists. **Do not "fix" this by weakening the test, re-anchoring it on a synthetic row,
  or reverting the density** — the user has the evidence and is choosing between keeping the
  density, restoring it, and dropping the test. Every `wall` row in README §Measurements is stale
  for the same reason and is marked so.
- **`spacetime_bezier/viewer.py` is UNVERIFIED.** Tracked 2026-08-20 because CLAUDE.md already
  documented it as live, not because it was checked. No test exercises it -- nothing in `tests/`
  imports it, and no test opens a browser, so the server, the static page, and the verdict fields
  it computes have never been asserted against anything. It compiles and it starts; that is the
  whole of the evidence. Treat any picture it draws as unconfirmed until a test covers the
  request/response path.
- **A superseded viewer stack is still on disk and still documented as live.**
  `spacetime_bezier/viewer.py` replaced it, but `spacetime_bezier/sandbox.py` (478 lines),
  `figures/spacetime_bezier_interactive.html` (3,868 lines), the `--bake` +
  `sync_interactive_viewer` path in `io.py`, and `figures/spacetime_scenarios.json` were never
  deleted — that was left as a separate, explicit decision. Meanwhile the interactive page's inline
  `const SCENARIOS` blob is committed **test mock data** (one control point at the origin, no
  `diverse`, no `wall`); the source was fixed 2026-08-18 but the committed data was not, and only
  `--bake` regenerates it. Serving over HTTP is unaffected — only `file://` shows the mock.
- **Baked and live runs use different solver parameters.** `--bake` defaults to `tol=1e-12,
  max_iter=10000`; the sandbox posts `tol=1e-6, max_iter=30`; the scenario table was made with
  neither, and there is no CLI flag for trust radius at all. A live violation of "one canonical
  execution model", left alone deliberately — changing a default silently changes every number.
- The venv may be broken (built against a Homebrew Python since upgraded). If `python3 -m
  spacetime_bezier` fails with a dyld error, rebuild it and re-run `maturin develop --release`.
