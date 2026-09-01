# WORKSTREAM.md — what is open

A finished item is struck through

A session can open with just an id — `ksas.submit`, `solver.test-drift`.

Each item is: **what is open**, **what closes it**, and **where its reasoning lives**. No
derivations here; no status anywhere else.

---

## Now — 한국항공우주학회 2026 추계, 제출 마감 **2026-09-04(금)**

Full venue rules, schedule and manuscript state:
[`paper/ksas_2026_fall/README.md`](paper/ksas_2026_fall/README.md).

### `ksas.submit` — 제출과 결제
The manuscript is written and the 400자 초록 is drafted; neither has been submitted. **Submitting
requires paid membership and completed 사전등록 결제, and that payment is due before the paper
deadline, not before the 사전등록 deadline.** A late oral request is demoted to poster, so
submitting early is what buys the oral slot.
*Closes when:* the paper is submitted on the web page, the 400자 초록 is typed into the form, and
the registration payment has cleared. Needs the author, not an agent.

### `ksas.abstract-count` — 초록 글자수
`abstract_400.md` is 400 characters plus its newline. That is exactly on the limit, so how the
submission form counts — spaces, punctuation, the trailing newline — decides whether it fits.
*Closes when:* the count rule is confirmed against the form and the text fits with margin.

### `ksas.poster` — 포스터
Nothing drafted. 학술대회는 11/10(화)~13(금), 하이원리조트(강원 정선). The poster is where the
argument the two-page manuscript cut can come back — supporting half-space construction,
per-iteration reconstruction, the convexity discussion.
*Closes when:* a poster exists in `paper/ksas_2026_fall/` and states the mission motivation, which
the advisor's review requires in the talk as well as the manuscript.

---

## Blocking honesty, not the deadline

### `repo.red-test` — one test is red by choice, and the choice is yours
`tests/integration/test_figure_grade_gate.py::test_a_clearing_run_can_still_be_standing_on_slack`
fails because `wall` was densified 2026-08-24 at your request, which turned its only specimen from
+0.102 into −0.129. 19 configurations were probed; no replacement specimen exists.
*Closes when:* you pick one of — keep the density and drop the test, restore the density, or accept
a synthetic specimen. **Do not close it by weakening the test.** → [`SOLVER.md`](SOLVER.md) §Known Issues

---

## Next — idea 1 저널

Runs in parallel in the `bezier-trajectory-journal` worktree, branch `paper/journal-1`.

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
limitation of every artifact rendering idea 1.

### `solver.kkt` — no dual or KKT residual
`converged` asserts feasibility plus no-further-progress, not stationarity of the original problem.
Nothing is scheduled; the item exists so the gap is not mistaken for a guarantee.

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
