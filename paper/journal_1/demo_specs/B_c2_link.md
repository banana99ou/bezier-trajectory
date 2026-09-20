# Demo B — C2-link continuity for UAM

**Slot:** [`paper/journal_1/README.md`](../README.md) figure slot 3, "Line-of-sight demo, as a pair".
**Carries:** consequence 3 — constraints a decomposition cannot express are ordinary rows.
**Scene:** `loiter`, already in metres and seconds, already a verified pair.
**Status:** the scene, the pair and the tests exist. Missing: the journal figure, the mission
paragraph, three sidecar fields the caption needs, and one honest baseline arm.

This is a build specification. It owns no measurement and quotes a number only where the number is
already in a committed sidecar or a committed test, and then with its source beside it.

---

## 1. Mission statement for the journal

Written to close the `Open` item in [`idea/spacetime.md:1271`](../../../idea/spacetime.md) for this
demo: the line-of-sight constraint is currently justified only as something a convex decomposition
cannot express, which is a methodological differentiator and not a reason anyone would need it.
This is the reason. Argued from first principles; it cites nothing internal.

> An uncrewed or remotely supervised aircraft operating beyond the visual line of sight of its
> operator is flown entirely through its command-and-control link, so the link is not an avionics
> convenience but the control path itself, and its availability is a flight-critical property in
> the same sense that thrust is. Certification of such operations therefore does not ask whether a
> link exists; it asks for a stated continuity — a bound on how long the aircraft may be without
> commands, and a defined response when that bound is exceeded. When the bound is exceeded the
> aircraft has no commanded response available, only a preprogrammed contingency: hold, return to
> the departure point, or terminate. Each of those is a loss of the mission, and in a dense urban
> corridor each is also an unfiled trajectory that the surrounding traffic did not plan around, so
> a link outage propagates into a separation problem. In low-altitude urban airspace the dominant
> outage mechanism is not transmit power or range — the ground station is a few hundred metres away
> — but geometry: another airframe, a structure, or a terrain feature passing between the aircraft
> and the station blocks the path the signal has to take. An occlusion of that kind caused by
> another participant in the same managed airspace is not a random fade, because that participant's
> own flight plan is filed and known, which means the outage is predictable before it happens and
> can be planned against rather than reacted to. Planning against it is worth doing precisely
> because the cheapest remedy is usually timing: an aircraft that is a few seconds late crosses
> behind the obstruction instead of through it, converting an unplanned contingency into a
> scheduled delay of known size — and that trade, delay against link continuity, is exactly what a
> planner has to be able to price.

Constraints on how the implementer may use this:

- It appears once, in the demo's introduction, and is **not** repeated in the abstract.
- It must not claim a specific numeric continuity requirement, a specific regulation, or a specific
  aircraft class. Naming a document is a literature decision — see §7, open question 5.
- The scenario is a **model** of that mission, not the mission: the occluder is a single sphere on a
  known circular lap, the station is a point, and occlusion is binary. Say so, in the same
  paragraph, next to the sentence that introduces the scene. `idea/spacetime.md` §"Decided against"
  already refuses the radar equation and detection probability; the journal must inherit that
  refusal visibly rather than let a reviewer discover it.

---

## 2. What exists and is reusable

### The scene

| Thing | Where | Note |
|---|---|---|
| `scenario_loiter()` | [`spacetime_bezier/scenarios.py:363-524`](../../../spacetime_bezier/scenarios.py) | Metres and seconds. Docstring records *why* each choice exists — do not restate it in the paper, cite the design reasons and re-derive the two that are geometric. |
| Tangent corridor | `scenarios.py:472` | `corridor_y = orbit_r * corridor_z / body_z` = 37.5 m. Derived, not tuned. This is the sentence the journal must carry: the spot arrives moving *along* the corridor, not across it. |
| Corridor width ±5 m | `scenarios.py:468`, bounds at `scenarios.py:513-517` | Spot radius at the corridor altitude is `body_r * corridor_z / body_z` = 7.5 m > 5 m, so **no lateral offset the corridor allows can clear the spot**. This is the load-bearing geometric fact of the whole demo. |
| Phase 51° | `scenarios.py:477` | Scanned, not derived. Journal must say scanned and say against what three conditions (run certifies, baseline fails, graft fails). |
| Altitude band floor 60 m above occluder top 56 m | `scenarios.py:516`, `occluder_top` in the sidecar | Keep-out rows present and never binding — so what forces the timing is the line-of-sight rows and nothing else. Say it. |
| `trust_radius` 5.0 as a scenario key | `scenarios.py:522` | A length. The figure run overrides it to 1.0 (see §5/§7). |
| Elastic weight 1e5 with its ladder | `scenarios.py:287-314` | The ladder (1e4/1e5/1e6 → same trajectory, different iteration counts) is journal-quality evidence that the weight is not a tuned knob. Reuse it in the limitations or reproducibility section. |

### The pair

`solve_pair` — [`tools/make_paper_figure.py:51-92`](../../../tools/make_paper_figure.py). The two
arms share one `common` dict and differ by `stations=sc["stations"]` versus `stations=None`
(lines 85 and 90). Nothing else differs. That single-call-site construction is what makes the
"differ by one block" claim checkable, and the integration test imports this function rather than
reimplementing it ([`tests/integration/test_loiter_scenario.py:65-79`](../../../tests/integration/test_loiter_scenario.py)) — keep that coupling.

### The four tests, and what each would fail if

All in [`tests/integration/test_loiter_scenario.py`](../../../tests/integration/test_loiter_scenario.py),
run at `N=8, n_seg=8, time_weight=10, v_max=5, sound_clip=True` (lines 84-87).

1. **`test_the_pair_differs_by_the_occlusion_rows_and_nothing_else`** (line 170).
   FAILS IF: either arm stops converging or starts penetrating; the baseline ever begins building
   shadow rows; the baseline's true margin stops being negative (which would make the false-zero
   warning undemonstrable); the scene produces zero shadow rows; the **constrained** curve violates
   the recomputed shadow rows by more than 1e-6 (two baselines in the pair); or the **baseline**
   curve satisfies them to better than 1.0 (two constrained runs in the pair). The last two are the
   guard a peer added after falsifying an earlier version by making both halves baselines — the
   scoring helper `_shadow_violation_at` (line 116) recomputes the rows from scratch via
   `bezier_opt.spacetime_koz_rows_exact` and grades both curves on one scale.
2. **`test_the_baseline_loses_line_of_sight_for_a_measurable_interval`** (line 229).
   FAILS IF: the baseline keeps the link anywhere-to-everywhere (`margin < 0` never true); the
   blocked interval is ≤ 1.0 s (a graze, not an interval); or the deepest block is shallower than
   −1.0 m. Asserted on `compute_los_margin`, never on the baseline's own certificate, because that
   certificate is 0.0 *by construction* for a trajectory that loses the link.
3. **`test_the_constrained_run_holds_the_link_and_both_computations_agree`** (line 259).
   FAILS IF: the occlusion certificate leaves zero; any occlusion plane was dropped (uncertifiable
   is not certified); `koz_unsound_clips` is nonzero (the certificate would cover only the clipped
   pieces); the independently sampled margin goes negative while the certificate stays at zero —
   which would mean one of two independent computations of one physical fact is wrong, and *that*
   would be the finding; the margin falls below +5.0 m or the clearance below 10.0 m; or
   `figure_grade_failures` returns any reason.
4. **`test_the_retiming_is_what_saves_the_run_not_the_path`** (line 303).
   FAILS IF: the two arrival times are within 5.0 s (the graft could not then separate path from
   timing); the constrained run arrives *earlier* than the baseline; the graft produces any NaN;
   fewer than 99% of graft samples have an active obstacle (`+inf` is defined and expected at a
   window boundary, an all-`+inf` graft is no measurement at all); the constrained path deviates
   ≥ 1.0 m laterally (a sidestep would make "timing did it" unsupported even if the graft failed);
   or **either direction of the graft reverses** — constrained path on baseline schedule must lose
   the link, baseline path on constrained schedule must keep it.

### Gates that already refuse

- `figure_grade_failures` — [`spacetime_bezier/optimize.py:164`](../../../spacetime_bezier/optimize.py).
  Written `if not (x <= tol)` so NaN fails; an earlier inline rewrite inverted that polarity and
  drew figures for missing evidence.
- `figure_grade_or_die` — `make_paper_figure.py:156-185`, maps info keys to row keys with NaN
  defaults and calls the real predicate.
- Gate 1b, unsound clips — `make_paper_figure.py:222-226`.
- Gate 2, the baseline must actually fail — `make_paper_figure.py:234-238`.
- Sidecar-driven build refusal — [`tools/render_poster.py:41-80`](../../../tools/render_poster.py),
  eleven conditions including the graft direction. Its refusals are themselves tested against eight
  doctored sidecars (`tests/unit/test_poster_gate.py`).

---

## 3. The journal figure

One figure, three panels, plus one optional fourth. Panels A and B already exist as code; panel C
exists as an unused poster figure. Nothing here needs new geometry.

### Panel A — the scene at the critical instant

Reuse `make_paper_figure.py:310-424` verbatim in structure: 3-D view at `t*` = the instant the
**baseline's** sight line is most blocked (`i_star = argmin(m_base)`, line 283). Station, the body
on its lap with direction-of-travel arrows, the shadow cone from the station through the body, the
spot where the cone crosses the corridor altitude, the corridor slab, and both vehicles at `t*`
with their sight lines — the blocked one dashed with a cross where it meets the body.

Journal-specific changes:
- The panel must be authored at the journal's column width, not KSAS's 3.15 in. The existing
  `rcParams` block (line 269) is sized for a 3.4 in figure; re-author, do not scale.
- Add the spot radius and the corridor half-width as an annotated pair (7.5 m against ±5 m). That
  single annotation is what makes "no sidestep is available" visible rather than asserted, and it
  is the panel's only job that the conference version does not already do.
- Axes are already `x [m]`, `y [m]`, `z [m]`. Keep the units; do not drop them for space.

### Panel B — line-of-sight margin against time, both runs

Reuse `make_paper_figure.py:426-439`. Baseline dashed grey, constrained blue, zero line, the
blocked interval shaded, `t*` marked as a dotted vertical shared with panel A.

Journal-specific changes:
- The y-axis label must define the quantity, not just name it: the margin is
  `distance(segment[station, vehicle(t)], obstacle centre(t)) − r`, minimised over obstacles active
  at `t` ([`spacetime_bezier/geometry.py:395-457`](../../../spacetime_bezier/geometry.py)). Put the
  definition in the caption; label the axis `sight-line clearance [m]`.
- State in the caption that this curve is **sampled**, so a negative value proves the link was lost
  and a positive value does not prove it was kept — the certificate is the guarantee. That
  limitation is written into `compute_los_margin`'s own docstring and must not be lost in transit.

### Panel C — the schedule comparison

**Already drawn and currently unused:** `fig_runs` in
[`paper/ksas_2026_fall/poster/make_poster_figures.py:499-531`](../../../paper/ksas_2026_fall/poster/make_poster_figures.py)
("drawn but does not fit on the poster", line 19). It plots the two returned curves in `(x, t)` with
the occluded band on the corridor axis shaded, the baseline's blocked stretch in red, both arrival
times labelled, and the axis reachability bound as a dotted line. It reads control points and every
number from the sidecar — no re-solve.

This is the panel that carries the sentence "the constrained run arrives later on an almost
unchanged path", because in `(x, t)` the two curves lie on top of each other in `x` and separate in
`t`. Port it into the journal figure; the only changes needed are the width, the label sizes
(authored for a poster at 15-17 pt), and the wording in §4 below about "unchanged".

### Panel D — the graft, both directions (recommended, not required)

Two extra margin-versus-time traces: constrained path on the baseline schedule (must dip below
zero) and baseline path on the constrained schedule (must stay above). The forward direction is the
poster's central claim and is already gated; the reverse direction currently exists **only** inside
the test (`_graft`, line 150) and reaches no sidecar and no figure. Drawing both is the difference
between "retiming was necessary" and "retiming was necessary and sufficient", and it costs one
extra call to `los_margin_at`.

### Caption claims and the exact sidecar field each number comes from

Sidecar convention is `tools/make_paper_figure.py:451-511`, written to
`figures/paper1/occlusion_figure.json`. Fields marked **NEW** do not exist yet.

| Caption claim | Field |
|---|---|
| baseline arrival | `baseline.arrival_time` |
| constrained arrival | `constrained.arrival_time` |
| the delay | computed in the injector as `constrained.arrival_time − baseline.arrival_time` (this is how `NUMdelay` is already produced, `render_poster.py:127`) — never typed |
| baseline deepest block | `baseline.min_los_margin` |
| baseline blocked interval | `baseline.los_loss_interval` (both endpoints) |
| constrained minimum margin | `constrained.min_los_margin` |
| constrained clearance from the body | `constrained.min_clearance` |
| lateral deviation, both arms | `constrained.lateral_deviation`, `baseline.lateral_deviation` |
| corridor half-width it is compared against | **NEW** `scenario.corridor_half` |
| shadow spot radius at the corridor altitude | **NEW** `scenario.spot_radius_at_corridor` |
| forward graft | `constrained.graft_min_los_margin` |
| reverse graft | **NEW** `constrained.reverse_graft_min_los_margin` |
| "the baseline's own occlusion certificate reads 0.0 for a trajectory that loses the link" | **NEW** `baseline.occlusion_certificate` (must be present *and* 0.0 for the sentence to be sayable) |
| "and it violates the shadow rows by X when scored against them" | **NEW** `baseline.shadow_violation_scored`, **NEW** `constrained.shadow_violation_scored`, **NEW** `n_shadow_rows` — the one-scale comparison `_shadow_violation_at` already computes in the test |
| "no plane was dropped" | **NEW** `constrained.occlusion_planes_dropped` (read by the gate at `make_paper_figure.py:180-182`, never written to the sidecar) |
| "unsound clips = 0" | `constrained.sound_clip`/top-level `sound_clip` and `constrained.unsound_clips`; **NEW** `baseline.unsound_clips` |
| iterations, wall clock, machine | `constrained.iterations`, `constrained.solve_seconds`, `machine` |
| axis reachability bound and the conservatism gap | `axis_arrival_bound`, `constrained.arrival_gap_to_axis_bound` |
| the snapshot instant in panel A | `snapshot_time` |
| configuration (N, segments, weights, trust radius, v_max) | `N`, `n_seg`, `elastic_weight`, `time_weight`, `v_max`, `trust_radius`, `free_arrival` |
| provenance | `generated`, `git`, `machine` |
| path length / excess over the straight line | **NEW** `constrained.path_length`, `constrained.path_length_excess_fraction` — only if the caption makes the claim; the docstring's "+0.00 percent" is at a different configuration and is not in any sidecar |

Rule inherited from the conference pipeline and restated in `paper/journal_1/README.md:107`:
**numbers are injected from a figure-grade sidecar, never typed.** The journal needs its own
injector on the model of `render_poster.py`, with its own gate, and the gate must refuse on the new
fields too — a `NEW` field that is absent must fail the build, not default to zero.

---

## 4. Gaps to close

1. **The figure and the graded pair are not the same run.** The committed sidecar
   (`figures/paper1/occlusion_figure.json`) records `n_seg: 48, trust_radius: 1.0,
   time_weight: 40.0`; the integration test grades `N=8, n_seg=8, time_weight=10.0` with the
   scene's own `trust_radius` 5.0. The two produce materially different results — the sidecar's
   arrival gap over the baseline is 6.6 s with 0.55 m of lateral deviation; the test's docstrings
   record 15.7 s and 0.078 m at the other configuration. **The journal cannot claim the pair is
   verified while drawing a different run.** Pick one configuration (§7, question 1), make the test
   constants and the figure invocation read from one place, and add the test in §6 that fails when
   they diverge.
2. **The committed sidecar is from a dirty tree** (`"git": "9f05cd5+dirty"`). The tool already
   appends `+dirty` (`make_paper_figure.py:448-450`) — the marker works; nothing refuses on it. For
   a journal, a `+dirty` sidecar is not reproducible provenance. The journal gate must refuse it.
3. **The reverse graft never leaves the test.** See panel D and the sidecar table.
4. **The baseline's false zero is not in the sidecar.** The most rhetorically valuable number in the
   whole demo — a converged run reporting `occlusion_violation_reference = 0.0` while blocked for
   1.56 s — is asserted in a test and invisible to the figure pipeline.
5. **The one-scale shadow-row score is not in the sidecar.** `_shadow_violation_at` is the
   measurement that turns "the halves differ by the occlusion rows" from a statement about a call
   site into a statement about the two returned curves. It belongs in the sidecar and in the
   caption.
6. **`occlusion_planes_dropped` is gated on but never recorded.** A downstream build cannot re-check
   the condition the gate exists for.
7. **The graft experiment has no figure panel.** It is currently prose plus one gate condition. It
   is the demo's mechanism claim and should be visible.
8. **Table 1 of the conference paper was typed by hand** from the scenario source. The journal's own
   rule forbids that. Add a `scenario` block to the sidecar carrying `orbit_r`, `body_z`, `body_r`,
   `corridor_z`, `corridor_half`, `corridor_y`, `spot_radius_at_corridor`, `T`, `coord_bounds`,
   `min_dt`, `max_iter`, `tol`, `init_curve`, and `dim`, and inject the parameter table from it.
   The tool's own docstring already claims "the figure is reproducible from the sidecar alone"
   (`make_paper_figure.py:29-30`); today it is not, because the scene is not in it.
9. **"Unchanged path" needs a number, not an adjective.** At the sidecar's configuration the
   constrained run moves 0.55 m laterally. Write the claim as the comparison that matters: the
   deviation is a small fraction of the ±5 m corridor and an order of magnitude below the 7.5 m
   spot radius, so it cannot be what cleared the shadow — and the graft measures that directly.
   Do not write "identical path".
10. **Units on every axis** — already satisfied in all three panels (`m`, `s`). Preserve them
    through the re-authoring at journal column width; the KSAS pass shrank to 5 pt labels once.

---

## 5. Fairness audit of the pair

### What the current baseline is

Our own solver, same scene, same band, same weights, same arrival freedom, with `stations=None`
(`make_paper_figure.py:90`). It is an **ablation of our own method**, not another method.

### Is "no occlusion rows" structural?

The structural argument the journal wants to make is: *a free-space convex decomposition has no
place to put a visibility constraint.* State it precisely, because the strong form is false and a
reviewer will know it.

- **True and defensible.** A decomposition method commits to a partition of the collision-free set
  *before* the solve, and then chooses cells. To carry visibility, the blocked set must be part of
  what was decomposed. Here the blocked set is the union over time of the shadow cast by a moving
  sphere from a point station: in the lifted space that is a curved, twisting tube whose
  cross-section both translates and rotates. Its complement is not a fixed convex partition; a
  decomposition would have to cover the corridor with cells that are valid only over time slices,
  and the number of slices needed is set by how fast the shadow sweeps — a cost that grows with the
  occluder's motion and that has to be paid before the optimizer knows when the vehicle arrives.
  That last clause is the real bite: **the decomposition must be fixed before the timing is decided,
  and here the timing is what determines which cells are free.**
- **False if overstated.** "A decomposition *cannot* express it" is too strong. A space-time GCS
  formulation can encode any region it is willing to decompose, shadow included. The honest claim
  is about **cost and ordering**, not expressiveness: a decomposition can express it only by
  decomposing a rotating non-convex swept volume in advance, and the resulting cell count and
  region graph are what our per-iterate construction avoids.

### Would a reviewer accept the current baseline?

**Partly — and not for the claim it is being asked to support.** Two separate questions:

- *Is the constraint active in this scene?* Yes, and the ablation is the right instrument. The
  baseline loses the link for a measured interval, the corridor geometry makes a sidestep
  impossible, and both graft directions are checked. A reviewer will accept this.
- *Does a decomposition method fail here?* The ablation says nothing about it. Deleting a constraint
  and observing that it is then violated is a tautology. Presented as a comparison baseline, this is
  a straw man and a reviewer is right to say so.

**Verdict: the ablation is sound as an activity check and inadequate as a comparison baseline. Label
it an ablation in the paper, and add one cheap arm that is structural.**

### The cheap structural arm to add

**Arm 3, "fixed time allocation".** Solve the same scene with the occlusion rows **on** and the
arrival time **pinned** (`free_arrival_time=False`, or pinned to the baseline's arrival), so the
solver may move the path but not the schedule. This is the path-then-time-allocation structure that
every decomposition-plus-time-allocation pipeline has, reproduced inside our own solver so nothing
else differs.

- Predicted outcome, from the scene's own geometry: infeasible or standing on slack. The spot radius
  at the corridor altitude is 7.5 m, the corridor allows ±5 m, and at tangency the spot is centred
  on the corridor axis — so no admissible lateral offset clears it at the blocked instant. If the
  arm comes back certified with a positive true margin, the demo's central claim is false and that
  is the finding.
- Cost: **one extra solve and one flag.** No new Rust, no new geometry, no new scenario. Roughly
  half a day including the sidecar field, the gate condition and the test.
- What it buys: the paper can then say "with the timing fixed, no path in this corridor keeps the
  link", which is a statement about a *class of methods* rather than about a deleted constraint —
  and it is measured, not argued.
- Interpretation hazard: an infeasible return has to be reported as "the solver returned infeasible
  under these settings", not as "no solution exists". Distinguishing those needs the corridor-width
  argument above stated as a geometric lemma, which the scenario docstring already contains.

### What a full baseline would cost, for the record

Running Osburn's space-time GCS or an IRIS-seeded decomposition on this scene remains
**blocked on the PI** (`paper/journal_1/README.md` build list, item 6 in the decisions record). It is
weeks, not days: obtaining or reimplementing IRIS in the lifted space, decomposing the swept shadow,
and defending the decomposition's own parameters as fairly chosen. **Do not start it on the strength
of this spec.** The recommendation here is arm 3 now, the full baseline only if the PI directs it.

---

## 6. Tests to add

Few, because the scenario is already covered. Each must be able to fail today.

1. **`test_the_figure_configuration_is_the_configuration_the_pair_is_graded_at`**
   FAILS IF: `figures/paper1/occlusion_figure.json` disagrees with the integration test's constants
   on any of `N`, `n_seg`, `time_weight`, `v_max`, `trust_radius`, `elastic_weight`, `sound_clip`.
   **This test fails right now** (48 vs 8 segments, 1.0 vs 5.0 trust radius, 40 vs 10 time weight) —
   which is the point: it is the guard for gap 1, and it must be written before the configuration is
   chosen, not after.
2. **`test_the_journal_sidecar_is_from_a_clean_tree`**
   FAILS IF: `sidecar["git"]` contains `+dirty`, or the recorded sha is not an ancestor of HEAD.
   **Also fails right now** on the committed sidecar.
3. **`test_the_sidecar_carries_every_field_the_caption_quotes`**
   FAILS IF: any field in §3's table is missing, or is NaN, or the caption template references a
   macro the injector did not define. Absent evidence must refuse — this is the same NaN polarity
   the figure-grade gate is written with.
4. **`test_the_reverse_graft_is_recorded_and_holds`**
   FAILS IF: `constrained.reverse_graft_min_los_margin` is absent, or ≤ 0 (the baseline path on the
   constrained schedule losing the link would mean the schedule is not sufficient either, and the
   mechanism is something the paper does not name).
5. **`test_fixed_time_allocation_cannot_keep_the_link`** — only if arm 3 is adopted.
   FAILS IF: the pinned-arrival, occlusion-constrained solve returns figure-grade **and** its
   independently sampled margin stays positive. That outcome would mean a path-only solution exists
   in this corridor, and the demo's structural claim would be false.
6. **Journal gate tamper test**, on the model of `tests/unit/test_poster_gate.py`.
   FAILS IF: any one of the journal injector's refusal conditions, flipped in a doctored sidecar,
   still builds. One doctored case per condition, including the new fields.

Not needed: anything re-testing the scene geometry, the graft direction, the false zero, or the
certificate — items 1-4 of §2 already cover those and cover them adversarially.

---

## 7. Open questions the implementer must not decide alone

1. **Which configuration is the journal's run** — the sidecar's `n_seg=48, trust_radius=1.0,
   time_weight=40` (what the conference paper and poster were built from and what Table 2 of a
   *submitted* paper reports) or the test's `n_seg=8, trust_radius=5.0, time_weight=10` (what the
   verified pair is graded at). They differ in the headline delay and in the lateral deviation.
   Changing to the second means the journal's numbers do not match the published conference table,
   which is defensible but must be a deliberate decision, stated. Ask the user.
2. **Whether arm 3 (fixed time allocation) is added at all**, and if it is, whether an infeasible
   return is acceptable as a published result or must be replaced by a bounded-suboptimality
   statement. This changes what the paper claims about a class of methods.
3. **Whether the comparison baseline stays an ablation** pending the PI, and how the paper labels it
   in the meantime. The decisions record marks the real baseline blocked on the PI; the journal
   cannot silently promote an ablation to a comparison.
4. **Whether "0.55 m" is allowed to be described as an unchanged path** in the journal's prose, or
   whether the claim must be restated as the ratio comparison in §4 item 9.
5. **Whether the mission paragraph may cite a named regulatory framework.** Writing it from first
   principles is deliberate; citing a specific rule is a literature decision with a venue
   dependency, and the venue is undecided.
6. **Whether panel D (the graft) is a panel or stays prose**, which is a figure-real-estate decision
   and depends on the venue's column width.

---

## Provenance of every number quoted in this file

`figures/paper1/occlusion_figure.json` (generated 2026-08-31, git `9f05cd5+dirty`, Apple M2) for the
figure configuration and its results; `tests/integration/test_loiter_scenario.py` docstrings for the
graded configuration's results; `spacetime_bezier/scenarios.py:363-524` for scene geometry, which is
source, not measurement. Nothing here is a new measurement, and no solver was run to write it.
