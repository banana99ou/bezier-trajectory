# Idea 1 — English journal manuscript

**Idea:** [`idea/spacetime.md`](../../idea/spacetime.md) — the claim, the formulation, the geometry, the novelty.
**Numbers:** [`SOLVER.md`](../../SOLVER.md) §Measurements. A number that is not in a figure-grade sidecar does not go in the paper.
**Conference version:** [`paper/ksas_2026_fall/`](../ksas_2026_fall/) — 2 pages, Korean, submitted 2026-09-03.

This file is the artifact document. It states what the manuscript claims, what figures it needs,
and what it adds over the conference version. It owns no measurement and quotes no number.

---

## Venue — undecided, and not mine to decide

The user decides with the PI. Recorded so the decision is made once and not re-argued: the page
budget, how hard a reviewer will push on a comparison baseline, and who reviews it all follow from
this. **Graduation is 2027-02**, so submission has to land well before it; review runs after.

Until a venue is chosen, write venue-neutral: LaTeX, no template, no page target.

---

## The claim — one claim, three consequences, one guarantee

Written this shape on purpose. A numbered list of five co-equal contributions invites a reviewer to
pick the weakest and attack it; one claim with consequences does not.

### The claim

A **decomposition-free** convexification of space-time collision avoidance, rebuilt **per iterate**
around the current trajectory. Its size is linear in (segments × obstacles) and it carries no
discrete structure — no integer variables, no region graph, no mode enumeration. Conservatism is
bounded by the trust region, not by the geometric complexity of the free space.

**Never open with "time as a coordinate."** That lift was published in August 2025. What is new is
decomposition-free. → [`idea/spacetime.md`](../../idea/spacetime.md) §Novelty and prior art.

### Three consequences — evidence the claim buys something, not separate claims

1. **Moving obstacles are structurally free.** An obstacle is a lifted Bézier curve of arbitrary
   degree with a radius, so a moving, curving, accelerating body produces the same rows as a static
   one. A space-only decomposition either restricts to static environments or re-decomposes per
   time step.
2. **Timing is a decision inside the same convex subproblem.** Arrival time, the speed cap and the
   time penalty are variables, not a pre-committed grid. There is no path-then-time-allocation
   stage, and therefore no two stages fighting when the timing changes which regions would have
   been needed.
3. **Constraints that depend on timing are ordinary rows** — line of sight through a moving
   shadow being the one the demos exercise. The structural point is *ordering*, not expressibility:
   a free-space decomposition must be fixed before the timing that decides which cells are free, so
   it commits to the timing it was supposed to optimize. Do not write "a decomposition cannot
   express visibility"; that sentence is false as stated (demo-B fairness audit, 2026-09-09).

### The guarantee — what was *not* given up

The section exists to answer the reviewer's first question: removing the decomposition made the
problem smaller, so what was traded away?

- **Continuous-time feasibility by hull containment.** A Bézier segment lies in the convex hull of
  its control points, so a half-space satisfied at every control point is satisfied for all times in
  the segment — not at collocation nodes. **Cite this as standard machinery**, because it is; Bézier
  and B-spline planners have used the hull property for years. Claiming it as novel invites a
  correct rejection.
- **The coverage condition, which is ours.** The half-spaces provably contain the clipped keep-out
  volume, with the reach floor as a sufficient condition, and the residual hole named and measured.
  Nobody in the corridor / convex-optimization family reports coverage at all — they report that the
  solver converged. → [`idea/spacetime.md`](../../idea/spacetime.md) §"Reachability, and exactly
  when the certificate covers K_m" and §"The hole, stated as a hole".

### Limitations — stated plainly, in their own section

- Local optimum; the homotopy class comes from the initialization. → `solver.multistart`
- `converged` asserts feasibility plus no-further-progress, **not stationarity** of the original
  problem. There is no dual and no KKT residual. → `solver.kkt`
- The clip's hole, with the measured rate at which it fires. → `journal.measure`

---

## What the journal adds over the two-page conference version

The conference manuscript deferred these explicitly; they are the reason a journal version exists.

- **The supporting half-space construction**, in full.
- **Per-iteration reconstruction** — the sharp form of the claim, and currently unwritten anywhere.
- **The convexity discussion**, including why free space cannot be convexified.
- **Proofs as numbered lemmas.** [`idea/spacetime.md`](../../idea/spacetime.md) §"Formulation —
  rigorous statement" has them as prose: support-half-space containment, the De Casteljau ceiling,
  the reachability condition, the shadow lemma.
- **A comparison baseline that was actually run**, and a scaling experiment. Neither exists.
- **Demos with a mission**, each built as a pair — the constrained run against a baseline that
  visibly fails at the same thing.

---

## Figure slots — what each must prove, not what it looks like

A slot is filled only by a figure-grade run. Slots are named here so the measurement pass knows
what to produce; the scenarios behind them are still being chosen.

1. **Formulation schematic.** The clipped keep-out volume, the clipping ball, the supporting
   half-space, and the trust region — one picture that makes the reach floor obvious.
2. **Per-iterate reconstruction.** The same segment across iterations, showing the half-space
   re-aiming as the trajectory moves. This is the figure the central claim lives in.
3. **Line-of-sight demo, as a pair.** Baseline loses the link; the constrained run holds it. The
   sight margin over time beneath the geometry.
4. **Timing demo, as a pair.** The agent waits for a gap that opens and closes. Baseline commits
   early and is caught.
5. **Scaling curve.** Solve time against obstacle count, with the linear-size claim overlaid.
6. **Coverage.** How often the hole fires, and what enforcing the reach floor costs in clearance.

Rule inherited from the conference pipeline: **numbers are injected from figure-grade sidecars,
never typed.** A build that cannot find a sidecar, or finds one that is not figure-grade, must fail.

---

## Status

Manuscript: **not started.** Nothing in this directory but this file.
Open items are [`WORKSTREAM.md`](../../WORKSTREAM.md) `journal.scope` and `journal.measure`.
**`journal.measure` has not run, so no number may be quoted anywhere yet** — including here.
