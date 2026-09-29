# WORKSTREAM.md — what is open

A finished item is struck through

A session can open with just an id — `ksas.submit`, `solver.test-drift`.

Each item is: **what is open**, **what closes it**, and **where its reasoning lives**. No
derivations here; no status anywhere else.

---

## Now — 한국항공우주학회 2026 추계, **제출 완료 2026-09-03**. 남은 것은 포스터.

Full venue rules, schedule and manuscript state:
[`paper/ksas_2026_fall/README.md`](paper/ksas_2026_fall/README.md).

### ~~`ksas.submit` — 제출과 결제~~
**제출 완료 2026-09-03**, 마감 하루 전. 원고와 400자 초록을 제출 웹페이지에 넣었다. 회비 납부와
사전등록 결제 완료는 제출의 전제 조건이다 — **결제가 아직 안 끝났다면 이 항목은 닫힌 것이 아니다.**

### ~~`ksas.abstract-count` — 초록 글자수~~
제출 폼이 `abstract_400.md`를 받았으므로 세는 방식과 무관하게 들어갔다. 400자 정확히라는 여유
없음은 그대로이니, 저널판에서 초록을 다시 쓸 때 이 사실을 되풀이하지 말 것.

### `ksas.poster` — 포스터
**초안 존재, 2026-09-03.** `paper/ksas_2026_fall/poster/` — A0 세로, 한글, 블록 11개.
`python3 tools/render_poster.py`로 빌드하고, 결과 숫자는 전부 `occlusion_figure.json`에서 주입된다.
원고가 잘라낸 세 논증(지지 반공간 구성, 매 반복 재구성, 볼록성)과 임무 동기가 모두 들어갔다.
학술대회는 11/10(화)~13(금), 하이원리조트(강원 정선).
인쇄 규격은 A0 세로로 확정 — 2026 추계 안내는 아직 없지만 2025 추계·2026 춘계가 같은 문장으로
"보드판 95cm×238cm, A0 사이즈 부착가능"을 명시한다(`paper/ksas_2026_fall/template/PROVENANCE.md`
§포스터 규격). 학회 포스터 템플릿은 없다.
*Closes when:* **(a)** 발표자가 내용을 읽고 승인한다, **(b)** 10월 중순 프로그램 안내가 뜨면 세션
일시·장소를 확인한다.
→ [`paper/ksas_2026_fall/poster/README.md`](paper/ksas_2026_fall/poster/README.md)

---

## Blocking honesty, not the deadline

### `repo.red-test` — two tests are red by choice, and each choice is yours
Each lost its specimen to a solver improvement; none is a regression.
1. `tests/integration/test_figure_grade_gate.py::test_a_clearing_run_can_still_be_standing_on_slack`
   — `wall` was densified 2026-08-24 at your request, which turned its only specimen from +0.102
   into −0.129. 19 configurations were probed; no replacement specimen exists.
2. `tests/unit/test_spacetime_koz_geometry.py::test_wall_penetrates_below_its_penalty_threshold`
   — since `2ea964d` (2026-09-20), `wall` held at weight 100 clears instead of penetrating.
A third, `test_zero_slack_leaves_a_real_margin[wall-10-16]`, went red with the cap reach and green
again the same day, 2026-09-29, when `wall` started by waiting: from its own start the scene now
ends with zero slack. Tests that need a FAILING `wall` run now pin the straight start by name.
*Closes when:* you pick, per test, one of — drop it, restore what removed its specimen, or accept a
replacement specimen. **Do not close any of them by weakening the test.**
→ [`SOLVER.md`](SOLVER.md) §Known Issues

---

## Next — idea 1 저널

Runs here in [`paper/journal_1/`](paper/journal_1/). The `bezier-trajectory-journal` worktree was
retired 2026-09-04; branch `paper/journal-1` holds no commit that this branch does not.

### `journal.scope` — what the six pages carry that two could not
The arguments explicitly deferred to the journal: supporting half-space construction, per-iteration
reconstruction, and the convexity discussion.
*Closes when:* the journal document states its own claim, its own figure slots, and what it adds
over the conference version — as an artifact doc beside its manuscript, the way
`paper/ksas_2026_fall/README.md` does. → [`idea/spacetime.md`](idea/spacetime.md) §Artifacts

### `journal.measure` — one measurement pass, at the end
Nothing may be quoted before it. The list of what must be measured is
[`idea/spacetime.md`](idea/spacetime.md) §What must be measured — including the number the
limitations section needs: how often the clip's hole actually fires.

---

## Solver — blocks nothing

### `solver.multistart` — procedural seeds (left / right / wait / hurry) + multi-start
Demoted 2026-08-19: it was justified by `wall` being infeasible, which was false. It may still buy
better local optima, and the passing homotopy class still comes from the initialization — a stated
limitation of every artifact rendering idea 1. A multi-start over the penalty weight alone — cold
starts at 1e3 / 1e4 / 1e5 on every config the default run failed, keep the best — was tried
2026-09-07 and removed 2026-09-08: it changed no verdict (23 of 28 either way) and it reported the
retried run under the requested start weight. If multi-start returns, it returns over seeds, and
the row names the start it came from. → [`SOLVER.md`](SOLVER.md) §Decided against

### `solver.init-blend` — where does `wall` stop working as the initial guess moves? *(pinned 2026-09-29)*
Your note, in your words: *"wall is not getting done right. test different init curves. find out
at what conf does current straight init works. then take that working resulting traj and blend it
with straight init to make many multiple of different init traj. then check where things stop
working."* A blend of two control polygons of the same degree is always a valid initial guess:
the endpoints are shared and the time column stays monotone. The solver already takes one
(`optimize_spacetime_from_control_points`), so the probe needs no solver change.
What it buys over the three-seed test of [`SPEC.md`](paper/journal_1/SPEC.md) §4.7:
- **An invariant that cannot fail if the solver is right.** A blend weight of one starts the
  solver on a certified solution. It must come back certified. If it does not, that is a
  construction defect, and no seed argument can explain it away.
- **A basin width instead of a yes or no.** The blend weight where solving stops is a number for
  the limitations section: how close the initial guess has to be. It is one slice of the basin,
  along one line, not the whole basin.
- **A split between timing and path.** Blending the time column and the spatial columns
  separately says which one the failure is about. The 2026-09-22 retries that recovered `wall`
  changed timing only.
**Corrected 2026-09-29: no re-fit is needed across segment counts.** The segment count does not
change the trajectory — every degree-8 `wall` configuration is one degree-8 curve, and the segments
only set how finely the certificate splits it. So the cheapest first probe solves nothing: take the
certified N8_seg16 trajectory and evaluate its certificate at 2, 3 and 4 segments. If it passes, a
safe answer exists there and the straight start is the problem; if it fails, the coarse split is
too conservative to hold that path. Only a change of degree needs a re-fit.
**Answered 2026-09-29, the user's call: a better start for this scene only, no front end.** `wall`
now starts by waiting and every setting certifies; a certified start comes back certified (both
tests in `tests/integration/test_timing_starts.py`). The segment count does not change the
trajectory, so no re-fit is needed across settings. The basin-width sweep is not scheduled.
*Closes when:* you drop it, or ask for the sweep.

### ~~`solver.curve-seg16` — the floor's own cost is a limit cycle, and it is not this session's~~
**Closed 2026-09-29.** E1 was refused and is gone; `2ea964d` ended the cycle, and under the cap
reach the configuration still certifies (2026-09-29 policy sweep). The freeze pass re-measures it.

`curve` N8_seg16 certifies with `sound_clip` off and, with it on, escalates to the weight cap and
runs to the iteration cap inside an exact limit cycle no iterate of which is feasible. **Owned by
the other (solver) session, not the journal one**, and as of 2026-09-09 an uncommitted `E1` change
to `clip_band` (`rust_optimizer/core/src/spacetime_obstacle.rs`) sits on disk against it. **This
blocks nothing in the journal workstream** — it is one configuration of one scenario, named so it
is not mistaken for an unknown.
*Closes when:* that session lands or abandons the change and the configuration is re-measured.
→ [`SOLVER.md`](SOLVER.md) §Measurements

### `solver.kkt` — duals are extracted, but `converged` tests no KKT residual
Since 2026-08-31 Clarabel's dual vector is read (`solve_qp_with_socs_duals`); it drives the
exactness margin (`max_koz_dual` vs the weight, tested both ways in
`tests/integration/test_scvx_invariants.py::TestElasticComplementarity`). `converged` still asserts
feasibility plus no-further-progress, not stationarity of the original problem. Nothing is
scheduled; the item exists so the gap is not mistaken for a guarantee.

---

## References — unread external ground truth

**When a reference and an agent's assertion disagree, the reference wins, and the disagreement gets
recorded.** Safe-corridor ground truth is read:
[`doc/refs/safe_corridor_references.md`](doc/refs/safe_corridor_references.md).

### `refs.scvx` — SCvx ground truth
Trust region, exact penalty, what convergence actually requires. The loop in
`spacetime_optimizer.rs` has never been checked against it.

### `refs.topology` — H-signature, TEB
Which passing class a plan lands in. Bears on `solver.multistart` and on how idea 1 is positioned
against TEB in every related-work section.

---

## Housekeeping

### `note.001` — 연구노트 001 still enshrines two fixed defects as design
`doc/notes/001_problem_formulation/main.tex` (its own repo on disk, not tracked here) derives the
zero-time normal and calls it "정확히 0", then forbids exactly those steps, then presents
per-control-point linearization as an improvement — which is the G2 defect.
*Closes when:* the note is revised, or explicitly retired. Until then, do not seed a manuscript
from it.

### `repo.superseded-viewer` — a dead viewer stack is still on disk
`spacetime_bezier/sandbox.py` (478 lines), `figures/spacetime_bezier_interactive.html` (3,868
lines), the `--bake` path in `io.py`, and `figures/spacetime_scenarios.json`. The interactive
page's committed `SCENARIOS` blob is test mock data. Deleting them was left as its own decision.
*Closes when:* you decide delete or keep. → [`SOLVER.md`](SOLVER.md) §Known Issues

### `repo.param-drift` — baked and live runs use different solver parameters
`--bake` defaults to `tol=1e-12, max_iter=10000`; the sandbox posts `tol=1e-6, max_iter=30`; the
scenario table was made with neither, and there is no flag for trust radius at all. A live
violation of "one canonical execution model", left alone deliberately because changing a default
silently changes every number.

---

## Idea 1 저널 — parallel tracks, opened 2026-09-18

**These five items belong to the journal in [`paper/journal_1/`](paper/journal_1/) and to nothing
else.** They are written as *parallel* tracks: column "files" is a boundary, not a hint. Two
sessions may run at the same time iff their file sets are disjoint. One session takes one track,
updates its owning document, and commits under `journal:`.

The specification every track is written against is
[`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) — claim, formulation as numbered definitions
and lemmas, the algorithm as one procedure, the experiments as things that can fail. **SPEC.md
follows the code**; a solver change that alters behaviour either edits SPEC.md §2–3 in the same
commit or says "SPEC impact: none" in its message.

| track | files it owns | may measure? |
|---|---|---|
| `journal.draft` | `paper/journal_1/main.tex` and its `.bib` | no — quotes nothing |
| `journal.baseline` | a new top-level directory of its own | its own solver only |
| `journal.bench` | `tools/bench.py`, `tools/make_tables.py`, `spacetime_bezier/families.py` + its tests | **only after `journal.freeze`** |
| `journal.demos` | `spacetime_bezier/scenarios.py`, figure scripts under `tools/` | **only after `journal.freeze`** |
| `journal.freeze` | `rust_optimizer/`, `SOLVER.md` | it *is* the freeze |
| `journal.release` | a public repository, at the end | no — it ships what `journal.bench` measured |

**Two collisions, and they are the only two.**

1. `journal.bench` and `journal.demos` both write `tools/`. **Not parallel** — demos waits for the
   bench runner to exist.
2. `journal.freeze` rebuilds the extension in place, so anything measuring while it is open gets a
   mixed build. That happened on 2026-09-09 and cost a whole table. **While `journal.freeze` is
   open, every other track may write code and none may quote a number.**

So: `journal.draft` + `journal.baseline` + `journal.bench` run concurrently; `journal.freeze`
whenever; `journal.demos` last.

### ~~`journal.freeze` — one solver, named, with a clean tree behind it~~
**Closed 2026-09-29 on `b506f34`**: walls at the trust region's cap, suite red only on the three
of `repo.red-test`, the 28-configuration table re-taken on that commit in
[`SOLVER.md`](SOLVER.md) §Measurements. `journal.bench` may now measure.
**E1 is refused — 2026-09-18, the user's decision: a hail mary, not a fix.** The committed clip
rule of [`SPEC.md`](paper/journal_1/SPEC.md) §2.4 stands.
**E1 is already gone from the tree (verified 2026-09-21)**, and the limit cycle it was aimed at was
repaired a different way by `2ea964d` (2026-09-20) — the wall reach frozen at the initial trust
radius, `curve` N8_seg16 going from the iteration cap to a stationary stop.
**The reach moved to the trust region's cap, 2026-09-29, the user's decision** — frozen at the start
it covered no step past the starting radius. Why, what was compared, and the guarantee that does
not depend on it: [`SPEC.md`](paper/journal_1/SPEC.md) §2.7.
*Closes when:* the working tree is committed, the suite runs with only the three reds of
`repo.red-test`, and the 28-configuration table is re-taken on that commit and named in
[`SOLVER.md`](SOLVER.md) §Measurements.
→ [`SOLVER.md`](SOLVER.md) §Measurements, [`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) §2.4

### `journal.draft` — the manuscript, venue-neutral, started from the spec
§2 formulation and §3 algorithm of SPEC.md carry no measured number, so they are writable before
anything is measured. Results and figures are left as slots that fail the build when empty.
**First deliverable is a skeleton draft, not a manuscript** — introduction, contributions, figure
list, experiment table, roughly five pages, no results — because the advisor gives no venue advice
until he has read a draft (SPEC.md §6). Three things this track owns that were decided in chat on
2026-09-20 and recorded nowhere else until now:
- **Figure slot 2 — per-iterate reconstruction — is sketched first, with fake data, before any
  prose.** It is the only picture in which the claim is visible and nobody has attempted it; if it
  cannot be drawn legibly that has to be known in the first week, not in December.
- **The prior-publication paragraph**: what the journal adds over the KSAS two-pager, written as
  the cover-letter paragraph an editor reads, not as a section heading. SPEC.md §1 and the README's
  "what the journal adds" are its source.
- **The November poster is the reviewer panel.** Every question asked at the poster is written
  down verbatim, in the asker's words, unanswered; that list is the introduction's list of
  objections to pre-empt. Better than any simulated review, because the askers do not already
  believe the framing.
*Closes when:* a venue-neutral LaTeX draft exists whose methods and algorithm sections are
complete and whose every number arrives from a figure-grade sidecar.
→ [`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) §1, §2, §3, §5

### `journal.release` — the code ships with the paper
The audience decided 2026-09-20 expects a public implementation, and a release is the cheapest
citation multiplier the paper has. Nothing is scheduled; the item exists so it is not forgotten
until the week before submission. Scope is the solver, the scenarios the paper uses, and the bench
that regenerates its tables — not the sandbox or the retired viewer (`repo.superseded-viewer`).
*Closes when:* a tagged public repository reproduces every table and figure in the manuscript
from one command.
→ [`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) §6, `journal.bench`

### `journal.baseline` — Osburn re-implemented, dimension-agnostic from the first line
Stage 1 is 2D + time: IRIS-sampled sets over the lifted space, a graph of convex sets with Bézier
control points, time-monotonicity and velocity-cone rows, Clarabel — the same solver as ours,
which is what makes the comparison fair. Write the spatial part as `x[:-1]` and the time part as
`x[-1]` throughout; a hard-coded dimension is the one thing that makes Stage 2 a rewrite instead of
a configuration. Stage 2 (3D + time) is a separate go/no-go after Stage 1's cost is known.
**No baseline has ever been run. This is the experiment that decides whether the paper has
evidence or only an argument.**
*Closes when:* Stage 1 runs both methods on Osburn's moving-obstacle case and his cluttered case,
graded by our own independent certificate on both sides, with the IRIS sample count swept.
→ [`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) §4.3

### `journal.bench` — numbers regenerable by one command, or they are not numbers
`tools/bench.py` and `tools/make_tables.py` per [`demo_specs/C_traffic_and_backbone.md`](paper/journal_1/demo_specs/C_traffic_and_backbone.md)
Parts 2 and 3, plus committing `spacetime_bezier/families.py` and its two test files, which sit
untracked. The bench records both mtimes — the compiled extension's and the package's — and
refuses a dirty tree, so "which build produced this number" stops being a question.
**Since 2026-09-29 the bench also runs the reliability pass of [`SPEC.md`](paper/journal_1/SPEC.md)
§4.7**: both traffic arms (`traffic_curved` landed in `families.py` that day, with tests), every run
binned, and the seed test on every unsolved run. The wait and hurry seeds do not exist in the
package yet.
*Closes when:* one command regenerates every table the manuscript uses, and a dirty tree or a
non-figure-grade sidecar makes it fail rather than print.
→ [`paper/journal_1/demo_specs/C_traffic_and_backbone.md`](paper/journal_1/demo_specs/C_traffic_and_backbone.md)

### `journal.demos` — the pairs, each with a baseline that can fail
Demo B's journal figure and its honest baseline arm (the audit found the figure and the tested
pair running different configurations); demo A's production scenario; demo WAKE in or out.
Each demo is a pair differing in exactly one named thing — if the baseline also succeeds, the
constraint was slack and the figure proves nothing.
*Closes when:* every figure slot of [`SPEC.md`](paper/journal_1/SPEC.md) §5 has a figure-grade
pair behind it.
→ [`paper/journal_1/demo_specs/`](paper/journal_1/demo_specs/), [`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) §4.2
