# Demo A — approach to a rotating target

**Build specification.** No production code exists for this scenario yet. This file says what to
build, what each number has to be, and what each test must fail on. It owns one measurement pass —
the cluster-cost spike in §8 — and nothing else; every other number here is a *target*, marked as
such, and becomes a fact only when a test re-measures it.

Carries **consequence 1** (moving obstacles are structurally free) and **consequence 2** (timing is
a variable inside the same convex subproblem) in one picture.
→ [`paper/journal_1/README.md`](../README.md) figure slot 4.

---

## 1. Mission and scope

A chaser approaches a docking/capture port on a target whose appendages — solar arrays on a drive
assembly, or a boom — rotate about the target's own axis. The corridor to the port is swept by the
appendages, so it opens and closes with the rotation phase. The chaser's decision is **when** to
cross, not **where**.

**Claimed:** that arrival timing against a moving, periodic obstacle field is decided inside one
convex subproblem, with no path-then-time-allocation stage; and that a cluster of ball obstacles on
Bézier circular paths costs the same kind of row as a static ball, at a price measured in §8.

**Not claimed, and the spec must not let the figure imply it:**

- **No gravity, no orbital mechanics, no Δv.** The scene is a short-horizon relative-motion
  (target-fixed) frame with kinematic constraints only: a speed cap and a keep-out set. This follows
  from the frozen decision that with time a free coordinate there is no quadratic Δv surrogate
  (`idea/spacetime.md` §Frozen decisions; DECISIONS.md). Do **not** import
  `orbital_docking/gravity.py` or `constants.py`. The vocabulary of the origin problem — keep-out
  zone radius, Δv surrogate — belongs to `orbital_docking`'s two-body transfer, not here.
- **Not a tumbling target.** A genuinely tumbling body carries the port around with it, which makes
  the terminal constraint a *moving point*. `optimize_spacetime` pins `p_end` to a fixed
  space-time point, so a moving terminal condition is not expressible without new machinery. Demo A
  is the **rotating-appendage** case: a stabilised bus with a fixed port, and appendages that
  rotate. Say "rotating", never "tumbling", in the caption and the manuscript. → §9 risk R1.
- **Not a rendezvous GNC result.** No navigation, no actuator model, no plume constraint.

---

## 2. Scene geometry

### Units — nondimensional, argued

The scene is built in the **nondimensional 10-unit box**, not in metres. Two measured reasons and
one editorial one:

1. Every solver knob in this repository was tuned in that box; `loiter` is the one scenario in
   metres and it needed a scenario-level `trust_radius` of 5.0 and its own elastic weight before it
   would certify at all (`spacetime_bezier/scenarios.py`, `scenario_loiter` docstring §"Two solver
   knobs are lengths"). A new scene in mission units re-opens both at once.
2. The trust radius enters the clip radius as `reach = seg_radius + trust * sqrt(dim)`, so it is a
   length and does not travel with a rescale. Holding the scene in the tuned box keeps the one knob
   whose default is meaningful.
3. A nondimensional scene cannot be mistaken for a claim about a particular spacecraft. Redress to
   metres, if at all, in the caption only ("≈" scale note), never in the scenario dictionary.

The spike (§8) is in these units. If the PI wants mission units, that is a re-measurement, not a
relabelling. → §9 risk R7.

### Frame and layout

Target-fixed frame, origin at the target's centre of mass, three spatial coordinates plus time
(`dim = 4`, the `fence3d` / `door3d` shape). The appendages rotate in the `z = 0` plane about `+z`.
The approach axis is `+x`.

| element | rule | spike value |
|---|---|---|
| target bus | static ball at the origin, present over the whole obstacle horizon | `HUB_R = 0.6` |
| docking standoff (`p_end` spatial) | on the approach axis, **outside** the appendage's swept band and clear of the bus | `[1.2, 0, 0]` |
| chaser start (`p_start`) | on the approach axis, **outside** the swept band | `[7.0, 0, 0]` |
| appendage ball radius | fixed for every cluster size — see §3 | `BALL_R = 0.4` |
| appendage ball centres | fill a fixed radial interval `[RHO_IN, RHO_OUT]` | `[2.0, 4.0]` |
| swept radial band | derived: `[RHO_IN - BALL_R, RHO_OUT + BALL_R]` | `[1.6, 4.4]` |
| appendage count | 2, opposed (see below — 1 is measurably vacuous) | `n_app = 2` |
| rotation | one revolution per `LAP_TIME` | `LAP_TIME = 20` |
| obstacle horizon | `T_OBS >= 1.5 * T_NOM` — see the trap below | `T_OBS = 30` |
| nominal arrival (`p_end` time) | scanned — see below | `T_NOM = 15` |
| approach corridor | hard box rows: `x in [X_INNER, 10]`, `\|y\| <= CORRIDOR_HALF`, `\|z\| <= CORRIDOR_HALF` | `X_INNER = 1.0`, `CORRIDOR_HALF = 0.25` |
| phase | scanned — see below | `PHASE = 2.199 rad` |

**Construction rules, not numbers:**

- **The port is outside the swept band, the corridor crosses it.** `port_x < RHO_IN - BALL_R`. The
  chaser cannot dock where a panel sweeps; what it must do is *cross* the band. This is what makes
  the space-only baseline structurally infeasible (§5) without the port itself sitting inside an
  obstacle, which would read as rigged.
- **`CORRIDOR_HALF < BALL_R`.** No lateral offset the corridor allows can dodge a ball on the axis,
  so a sidestep is not an escape. This is `loiter`'s "the corridor has a WIDTH" argument in its
  narrow form, and it is what makes timing the only escape. Both arms of the pair get the same box.
- **`X_INNER` just below `port_x`.** Do not fly past the port into the bus. Without it the solver
  was measured diving to `x = 1.37` and hiding between appendage radii (spike, discarded variant).
- **`T_OBS >= 1.5 * T_NOM`.** A freed arrival is clamped at `time_ub_scale * T_NOM = 1.5 * T_NOM`
  (`optimize_spacetime`), and an obstacle's active window is *intrinsic to its control-point times*
  (`spacetime_bezier/geometry.py`, `obstacle_array_bundle` docstring). An appendage chain that stops
  at `T_NOM` therefore **ceases to exist** exactly where the freed arrival is allowed to go, and the
  run "solves" the timing problem by arriving after the obstacles have vanished. This is a silent
  vacuity, not an error. The scenario builder must raise if `T_OBS < 1.5 * T_NOM`, and a test must
  assert it (§6, T7).
- **The phase is SCANNED, not derived** — the precedent is `scenario_loiter`, whose 51° was scanned
  against three simultaneous conditions. Here the scan criterion is: an appendage crosses the
  approach axis **while the straight nominal plan is inside the swept band**. The blocked instants
  are `t_k = (k*pi - phase) / omega` with `omega = 2*pi / LAP_TIME` (opposed appendages block every
  half turn); the nominal plan is inside the band over
  `t in [T_NOM*(start_x - band_out)/(start_x - port_x), T_NOM*(start_x - band_in)/(start_x - port_x)]`.
  At the spike values: blocked at `t = 3, 13, 23`; nominal plan in the band over `[6.7, 14.0]`;
  `t = 13` lands inside it. Measured seed clearance **−0.3736**, deepest against `a0b0_q3` at
  `t = 12.99`.
- **`T_NOM` is the second scanned knob and it is what makes the scenario non-vacuous.** Measured in
  the spike: at `T_NOM = 11.2` the straight plan is already collision-free (the demo proves
  nothing); at 14 / 15 / 16 the straight plan collides and the run is figure-grade; at 18 the run
  reaches the iteration cap. Scan it, record the window, register one value.
- **Two appendages, not one.** Measured: with `n_app = 1` the run escapes by arriving at `t = 2.9`
  before the first crossing, and the exact row builder finds **9 rows / 1 wall** at the returned
  iterate — the appendages do not constrain the answer at all, and the figure would show a
  constraint that is not binding. With `n_app = 2` the same scan gives 387 rows / 43 walls and the
  binding clearance is set by an appendage ball (`a1b7_q1`). One appendage is a vacuous demo.

---

## 3. Obstacle construction

**One appendage = a cluster of balls, each on its own circular path, each path a chain of cubic
quarter-arcs.** This is `scenario_loiter`'s construction reused verbatim
(`spacetime_bezier/scenarios.py`, `scenario_loiter`, the `lap` list): a cubic cannot close a circle,
so a lap is four cubic quarter-arcs on contiguous time windows, with the quarter-circle constant
`k = 0.5522847498` (radius overshoot 0.03 %, far below the ball radius).

Per ball: `n_quarters = ceil(T_OBS / (LAP_TIME/4))` obstacles — at the spike values, `30 / 5 = 6`
quarter-arcs (1.5 laps). Ball `j` of appendage `a` starts at angle
`phase + 2*pi*a/n_app` and radius `rho_j`, and the whole chain is the base lap rotated by that
angle, exactly as `loiter` rotates its lap by its scanned phase.

**Lifted obstacle count** = `n_app * n_balls * n_quarters + 1` (the bus). At the demo size
(`n_app = 2`, `n_balls = 8`, `n_quarters = 6`): **97 obstacles**.

**Cluster refinement rule — this is what makes the cost sweep mean anything.** `BALL_R` and the
radial interval `[RHO_IN, RHO_OUT]` the centres fill are **fixed**; raising `n_balls` only refines
the discretisation of *the same physical appendage*, so the swept volume, the blocked angular
window and the difficulty are invariant and obstacle count is the only thing that moves. Two
conditions travel with it:

- **Sealing.** `(RHO_OUT - RHO_IN) / (n_balls - 1) <= 2 * BALL_R` or adjacent balls leave a pore.
  `door3d`'s docstring records what a pore costs: two stacked rows of `r = 0.9` spheres left a
  diagonal pore at centre distance 1.84 against a radii sum of 1.8, and the solver threaded it at
  +0.007 clearance. **Assert the sealing condition in the builder**, do not leave it to review. At
  `[2.0, 4.0]` with `BALL_R = 0.4` the smallest sealed cluster is `n_balls = 4`; 2 and 3 have pores
  and are not admissible sizes.
- **The hub is degree 1 and the arcs are degree 3, in one array.** Written as `{pos0, vel, r,
  t_start, t_end}`, the bus normalises to a two-control-point lifted obstacle
  (`spacetime_bezier/geometry.py:105`, `normalize_obstacle`); the arcs are four-control-point cubics.
  `obstacle_array_bundle` (`spacetime_bezier/geometry.py:513`) degree-elevates to the highest degree
  present via `elevate_to_degree` (`:158`), which is **exact** — "this is what lets obstacles of
  mixed degree share one rectangular array without any of them being resampled or approximated". The
  manuscript's consequence 1 should cite exactly these two functions; the scene exercises the claim
  rather than asserting it.

---

## 4. Solver knobs that must travel with the scenario

`scenario_loiter`'s docstring is the precedent for every line here: "Two solver knobs are lengths, so
they do NOT come along for free when the scene is scaled", and "The arrival time is meant to be
FREE. That is a solve flag, not a scenario key."

| knob | value in the spike | why it is not a default, and who owns it |
|---|---|---|
| `coord_bounds` | `[[1.0, 10], [-0.25, 0.25], [-0.25, 0.25]]` | **Scenario key.** The approach corridor. Hard box rows outside the elastic slack range — the penalty cannot buy through them, which is the whole mechanism (`scenario_loiter`, "enforced as HARD box rows"). A band that excludes an endpoint is refused loudly, so `X_INNER <= port_x` is checked for you. |
| `trust_radius` | `0.25` | **Scenario key.** A length. The scene is ~10 units across; `loiter`'s ratio is 5/200. Callers must forward it the way they forward `coord_bounds` (`optimize.py:744`). Swept in the spike over 0.5 / 0.25 / 0.125 on a discarded variant: it moved nothing there, so 0.25 is a scale argument, not a measured optimum. |
| `free_arrival_time` | `True` | **Solve flag.** Consequence 2 is not demonstrable with a pinned arrival. |
| `time_weight` | `10.0` | **Solve flag.** `free_arrival_time=True` with `time_weight=0` **raises** `DegenerateFreeArrivalError` by design (`optimize.py:379`, `check_free_arrival_is_costed`): a freed arrival nothing prices is an interior-point tie-break, measured to swing 10.08 → 12.24 with the trust radius alone. 10.0 is `loiter`'s figure value. |
| `v_max` | `2.0` | **Solve flag, and required.** `time_weight > 0` with no `v_max` raises `UncappedTimePenaltyError`; the trap was measured — arrival collapses to `min_dt × gaps`. The cap is a second-order cone carrying no elastic slack: the penalty may relax keep-out rows, never the physics. It is also a scene parameter: it sets how long crossing the band takes, hence how wide a gap the run needs. |
| `sound_clip` | `True` | Default, and named anyway. The reach floor makes the certificate speak for the whole keep-out zone. Measured 0 unsound clips on every spike run — but that is *forced* by the floor and is therefore not evidence about the geometry; the honest reading is that the gate condition is satisfiable here. |
| `elastic_weight` | **unmeasured** | The default start of 100 with in-loop escalation was enough for every figure-grade spike run (0 raises). That is not a registry entry. Before any number from this scenario is quoted, measure the weight the way `loiter`'s was measured — every rung, with the iteration counts recorded beside it — and register it in `SCENARIO_ELASTIC_WEIGHT`. → §9 risk R5. |
| `min_dt` | `0.1` | Default. At a 15–30 s horizon the floor never binds — the same exemption `loiter` records. Assert `arrival_on_min_dt_floor == 0` anyway. |
| degree / segments | `N = 8`, `n_seg = 8` | `loiter`'s figure configuration. A registered config list must be scanned before the scenario is registered; the spike fixed it on purpose so the cluster axis is the only thing moving. |

---

## 5. The pair

**Constrained arm.** The scene as built in §2–3: the appendages are moving obstacles, each ball a
chain of lifted cubic quarter-arcs.

**Baseline arm.** The same appendages replaced by their **swept, time-invariant volume**: for each
appendage radius `rho_j`, a sealed static ring of balls of radius `BALL_R` at that radius, present
over `[0, T_OBS]`. Ring spacing `<= 2*BALL_R` for the same sealing reason as §3 — a pore in the ring
would let the baseline through for a discretisation reason and destroy the comparison.

**The one knob that differs** is whether the appendage carries its motion in the time coordinate or
is replaced by the union of its positions over the horizon. Start, end, corridor box, trust radius,
degree, segments, weights, `free_arrival_time`, `time_weight`, `v_max`, `sound_clip` and the seed
are identical. The baseline necessarily has more *obstacles* (405 against 97 at the demo size) —
that is intrinsic to the abstraction, and it makes the baseline **tighter**, not looser: a finer
ring is a closer over-approximation of the swept set. State that in the caption; a reviewer will ask.

**Why the failure is structural and not a straw man.**

- The swept volume of a rotating appendage over a full turn is an **annular shell with no door**.
  Any decomposition of the *time-invariant* free space — IRIS regions, a corridor graph, a
  visibility graph, a safe-flight-corridor chain — is built on that set, and the port is on the
  far side of it. The failure is not a tuning artefact of our solver; it is a property of the set a
  space-only method has to plan in.
- The baseline is not asked to do something impossible in principle: the *same* solver, the *same*
  rows, the *same* seed. Only the obstacle set changed, and it changed in the direction a
  space-only method is forced to take.
- **The corridor box is load-bearing and must be declared, not hidden.** The appendages sweep a
  plane; without the `|z| <= 0.25` bound the baseline can climb over the annulus and succeed. The
  bound is a real operational constraint (a rendezvous approach corridor), it is applied identically
  to both arms, and its load-bearing role is asserted by a test (§6, T6b) rather than left for a
  reviewer to discover. Publishing the pair without that sentence would be the straw man.

**Measured on the spike scene** (§8, E1 build): the baseline returns penetrating by **−0.3897**,
holding **22.4** of elastic slack at the weight cap after 3 raises, stopping by trust collapse,
certificate 22.4. It is a failed solve in every field the gate reads. Whether the manuscript may
call that "infeasible" — the solver returns a penetrating iterate, it does not prove infeasibility —
is §9 risk R6.

---

## 6. Tests — each as "FAILS IF …"

Structured after `tests/integration/test_loiter_scenario.py`: one module fixture solving the pair
through the figure tool's own `solve_pair`, so the run that is graded is the run that is drawn, and
every assertion checked against **both** halves. A claim that holds for the baseline too is not
evidence about the constraint.

**T1 — non-vacuity: the nominal plan collides.**
Assert the straight seed's `compute_min_clearance` against the true obstacle trajectories is
negative by a margin (target: `< -0.2`; spike measured **−0.3736**), and that the deepest violation
is against an *appendage* obstacle, not the bus.
*FAILS IF* the straight plan is already collision-free — then `T_NOM` or the phase has drifted out
of the scanned window and the demo proves nothing. (This is the test the discarded `T_NOM = 11.2`
variant would have failed at +0.6000.)

**T2 — the pair differs in the intended way and only in it.**
Both halves solved from the same seed with the same knobs; assert the constrained half's obstacle
set is all-moving (every appendage obstacle's first and last control-point times differ and its
spatial control points are not all equal) and the baseline's appendage obstacles are all static
(degree-1, zero spatial extent in time) and span `[0, T_OBS]`; assert both halves see the same
corridor box, trust radius, weights and seed.
*FAILS IF* the baseline starts carrying motion, or the two halves stop sharing a knob — in which
case every other test compares two different problems. (This is the falsification `loiter`'s peer
found: the earlier version of that test was satisfied by *two baselines*.)

**T3 — the baseline fails, structurally.**
Assert the baseline is **not** figure-grade and name why: `min_clearance < 0` **and**
`total_slack > 1.0` **and** `converged is False`. Assert the penetration is against a ring obstacle.
Assert it separately for two cluster sizes so the failure is not one ring's accident.
*FAILS IF* the baseline converges, clears, or fails only on `koz_unsound_clips` — a baseline that
fails on the clip gate has not failed at the *constraint*, and the pair would prove nothing about
decomposition. Spike: −0.3897 clearance, 22.4 slack, trust collapse.

**T4 — the constrained run certifies, and two computations agree.**
Assert `figure_grade_failures(row) == []`: converged, hull certificate `<= 1e-6`, clearance `> 0`,
slack `<= 1e-6`, `koz_unsound_clips == 0`, speed-cap violation `<= 1e-6`, a finite resolved weight.
Then assert the *independent* sampled clearance agrees in sign and magnitude with the certificate's
verdict. Spike: certificate 2.55e-07, slack 5.45e-10, clearance +0.2981, 0 unsound clips, stop
`merit_streak`, 36 iterations.
*FAILS IF* either computation contradicts the other — and that disagreement is the finding, not a
tolerance to widen.

**T5 — timing, not path: the schedule graft, both directions.**
Take the constrained run's **spatial path** and fly it on the **nominal plan's schedule**
(uniform in the curve parameter, arrival `T_NOM`); re-measure clearance. Then take the nominal
plan's spatial path (the straight line) and fly it on the **constrained run's schedule**.
Assert the first is negative and the second is positive.
Guards, all three, or the graft stops being a measurement: (a) the two arrival times must differ by
a real margin (spike: 4.967 against 15.0); (b) the graft must produce no NaN and must still have an
active obstacle at essentially every sample; (c) the straight-line-on-the-new-schedule result is
itself the "the path is not doing the work" guard, and it is stronger than a lateral-deviation
bound — the spatial path *is* the seed's, exactly.
Spike: constrained path on the nominal schedule **−0.3530**; straight nominal path on the
constrained schedule **+0.2702**; constrained run's own **+0.2981**. Maximum lateral excursion of
the constrained run 0.163 of the 0.25 the corridor allows.
*FAILS IF* either direction reverses — if the constrained path still clears on the old schedule, the
retiming was decoration; if the straight path still collides on the new schedule, the mechanism is
something this test does not name.

**T6a — the cluster is what creates the problem (attribution).**
Solve the identical scene with the appendages removed (bus only). Assert the straight seed is
already collision-free and the run converges figure-grade with no retiming.
*FAILS IF* the no-cluster arm still collides or still retimes — then the corridor, the bus or the
time penalty is producing the effect the figure attributes to the appendages. Spike: seed +0.6000,
solution +0.6000, arrival 2.900, converged, 0.02 s.

**T6b — the corridor is load-bearing, and we say so.**
Solve the **baseline** with the out-of-plane bound widened past the appendage plane. Assert it then
succeeds.
*FAILS IF* the baseline still fails without the corridor — which would mean the corridor is not the
reason, and §5's honesty paragraph is wrong and must be rewritten rather than kept.

**T7 — the obstacles outlive the freed arrival.**
Assert `T_OBS >= time_ub_scale * T_NOM`, and assert that at the returned arrival time at least one
appendage obstacle is still active.
*FAILS IF* the run arrives after the appendages' windows close — the timing "solution" would be an
artefact of obstacles ceasing to exist, and no other test in this file would catch it.

**T8 — cluster size is a refinement, not a different scene.**
Across the admissible cluster sizes, assert the swept radial band and the blocked instants are
identical to tolerance, and the sealing condition holds at every size.
*FAILS IF* raising `n_balls` changes the geometry — then the cost sweep in §8 is measuring two
things at once and its scaling statement is void.

---

## 7. Figure spec

**Panel (a) — one frame at the arrival instant**, viewed down `+z` onto the appendage plane.
Draws: the bus ball; both appendages as their chains of balls at `t = t_arrival`; the swept annulus
as a light ring behind them (this is the baseline's obstacle, drawn once, so the reader sees the two
arms of the pair in one picture); the corridor box in outline; the chaser at the port. The gap the
run threaded is the wedge between the two appendages, and it should be visibly aligned with the
approach axis.

**Panel (b) — a phase-vs-time strip** beneath (a), sharing the time axis. Draws: shaded bands where
the approach axis is blocked (computed from the scene's own parameters, never from solver output);
the nominal plan's band transit as one bar; the constrained run's band transit as a second bar,
visibly inside a gap; both arrival times marked. Optionally a third bar for the graft (constrained
path on the nominal schedule) with its collision instant marked, since T5 is the claim the figure
exists to carry.

**What the caption claims** — and nothing more: *the same spatial path, flown on two schedules; the
schedule chosen by the solver inside one convex subproblem; the space-only baseline, given the
tightest time-invariant over-approximation of the same appendages, cannot cross at all.* The caption
must also carry the corridor sentence from §5 and the "rotating, not tumbling" scope from §1.

**Sidecar fields injected** (numbers are never typed — the rule inherited from the conference
pipeline): `arrival_time` for both arms; `min_clearance` for both; `certificate_violation`;
`total_slack`; `koz_unsound_clips`; `iterations`; `elastic_weight` and `weight_raises`;
`n_obstacles`; the graft's two minima; the seed's clearance; `stop_label`; and the provenance
triple (git sha, extension path, **extension mtime**). A build that cannot find a sidecar, or finds
one that is not figure-grade, must fail — `tools/make_paper_figure.py` already refuses.

---

## 8. Spike results — what a cluster costs

### The question, and the stop rule stated before the answer

*Is a rotating cluster affordable at the size demo A needs, and how does cost scale with cluster
size?*

**The result that would have made me recommend dropping demo A**, fixed before the runs:
(i) the demo-size cluster taking more than about a minute per solve, or
(ii) no cluster size reaching figure grade at the figure's configuration, or
(iii) per-iteration cost growing faster than linearly in obstacle count — which would contradict the
paper's own linear-size claim and turn the demo into evidence against it.
None of the three happened. **Verdict: affordable, with a named ceiling.**

### Provenance — read this before quoting any number below

| | |
|---|---|
| git sha | `5b3ef7882c8e62dedcc9f54a7d07cacf253aec86`, worktree dirty (8 files) |
| extension | `.venv/lib/python3.14/site-packages/bezier_opt/bezier_opt.cpython-314-darwin.so` |
| **extension mtime** | **2026-09-09T14:42:43** — the **E1** build |
| Python | 3.14.6, arm64 Darwin |
| configuration | `N = 8`, `n_seg = 8`, `max_iter = 200`, `tol = 1e-6`, `trust_radius = 0.25`, `min_dt = 0.1`, `free_arrival_time=True`, `time_weight = 10`, `v_max = 2`, `sound_clip=True`, elastic weight default start 100 with in-loop escalation, straight seed |
| scene | `T_OBS = 30`, `LAP_TIME = 20`, `T_NOM = 15`, `HUB_R = 0.6`, `BALL_R = 0.4`, `[RHO_IN, RHO_OUT] = [2.0, 4.0]`, `PHASE = 2.199`, start `[7,0,0]`, port `[1.2,0,0]`, corridor `x∈[1,10]`, `\|y\|,\|z\| <= 0.25` |
| script | `spike_cluster.py` in the session scratchpad (throwaway, not committed) |

**Concurrent-edit caveat.** The worktree was dirty during the sweep and other sessions were editing
`spacetime_bezier/optimize.py`, `objective.py` and `scenarios.py` while it ran. Those modules
marshal the call; the solving is in the `.so`, whose mtime was verified unchanged on both sides of
every solve. Still: this is a dirty-tree measurement, and the re-measurement that fills §8 for the
paper must be taken on a clean tree with the sha recorded.

**E1 caveat.** This build carries an uncommitted change to `clip_band` in
`rust_optimizer/core/src/spacetime_obstacle.rs` ("E1 (2026-09-09)"): with `sound_clip` on, the
clipping radius is the minimal sound radius rather than a clamp on `r_nearest`. **These numbers are
not comparable with SOLVER.md §Measurements**, which was taken on the pre-E1 build (mtime
2026-09-08 20:22). Every run below recorded the `.so` mtime immediately before and immediately after
its solve; all eleven read `2026-09-09T14:42:43` on both sides, so the sweep sits on one build.
Nothing was rebuilt.

### The sweep — cluster size is the only thing that moves

Ball radius and the radial interval the centres fill are fixed, so every row is the **same physical
appendage** at a different discretisation (§3). "rows"/"walls" are counted at the **returned
iterate** by `bezier_opt.spacetime_koz_rows_exact`, the exact row builder the certificate uses.

| run | appendages × balls | obstacles | rows | walls | wall-clock | iters | s/iter | clearance | certificate | slack | unsound | arrival | weight (raises) | stop | figure-grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `rot_app1_b4` | 1 × 4 | 25 | 9 | 1 | 0.22 s | 19 | 0.012 | +0.6000 | 0.00e+00 | 9.2e-11 | 0 | 2.900 | 100 (0) | stationary | yes |
| `rot_app1_b8` | 1 × 8 | 49 | 9 | 1 | 0.46 s | 20 | 0.023 | +0.6000 | 0.00e+00 | 9.2e-11 | 0 | 2.900 | 100 (0) | stationary | yes |
| `rot_app1_b16` | 1 × 16 | 97 | 9 | 1 | 1.41 s | 25 | 0.057 | +0.6000 | 0.00e+00 | 9.2e-11 | 0 | 2.900 | 100 (0) | stationary | yes |
| `rot_app2_b4` | 2 × 4 | 49 | 216 | 24 | 0.86 s | 36 | 0.024 | +0.2981 | 2.72e-07 | 5.6e-09 | 0 | 4.967 | 100 (0) | merit_streak | **yes** |
| **`rot_app2_b8`** | **2 × 8** | **97** | **387** | **43** | **1.74 s** | **36** | **0.048** | **+0.2981** | **2.55e-07** | **5.5e-10** | **0** | **4.967** | **100 (0)** | **merit_streak** | **yes** |
| `rot_app2_b16` | 2 × 16 | 193 | 756 | 84 | 19.20 s | 200 | 0.096 | +0.3287 | 0.00e+00 | −4.4e-14 | 0 | 4.995 | 1e5 (3) | iteration_cap | no |
| `rot_app2_b16` (`max_iter=400`) | 2 × 16 | 193 | 765 | 85 | 36.85 s | 400 | 0.092 | +0.5831 | 0.00e+00 | −3.1e-12 | 0 | 5.243 | 1e5 (3) | iteration_cap | no |

Pair and attribution arms, same build and configuration:

| run | obstacles | rows | walls | wall-clock | iters | s/iter | clearance | certificate | slack | arrival | stop | figure-grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `swept_app2_b4` (baseline) | 205 | 1467 | 163 | 9.76 s | 65 | 0.150 | **−0.2434** | 8.19e+00 | 8.19 | 5.645 | trust_collapse | no |
| `swept_app2_b8` (baseline) | 405 | 4194 | 466 | 35.67 s | 76 | 0.469 | **−0.3897** | 2.24e+01 | 22.4 | 9.662 | trust_collapse | no |
| `nocluster` (attribution) | 1 | 9 | 1 | 0.02 s | 17 | 0.001 | +0.6000 | 0.00e+00 | 9.2e-11 | 2.900 | stationary | yes |

Eleven solves, all under 40 s, total sweep well inside the budget; none exceeded the 2-minute
abandon threshold.

### What the numbers say

1. **Per-iteration cost is linear in the number of keep-out rows, at one constant.**
   `s/iter ÷ rows` across the constrained arm: 1.10e-4, 1.24e-4, 1.27e-4 (216 / 387 / 765 rows) and
   across the baseline arm: 1.02e-4, 1.12e-4 (1467 / 4194 rows). Same constant, a 19× span of rows,
   two different obstacle kinds. **≈1.1e-4 s per row per iteration on this machine and build.** Rows
   are themselves linear in obstacle count at fixed segments, so the cost model is
   `time ≈ 1.1e-4 × rows × iterations`. This is the paper's linear-size claim showing up as
   wall-clock, and it is the one number from this spike worth quoting — with its build.
2. **A cluster obstacle is not special.** 97 rotating cubic-arc obstacles cost the same per row as
   405 static ones. Consequence 1 is not just structurally true, it is true in wall-clock.
3. **The demo size is comfortable.** 2 × 8 (97 obstacles, 387 rows) solves figure-grade in **1.74 s**
   in 36 iterations at the *default* start weight with zero raises.
4. **The ceiling is iterations, not rows.** At 2 × 16 the run is feasible (+0.33 / +0.58) and
   certified (0.00e+00) with essentially zero slack, and still fails the gate on `converged` alone
   at both 200 and 400 iterations. That is the same signature as `curve` N8_seg16 in SOLVER.md
   §Measurements: a run that holds a certificate but never satisfies no-further-progress. **Do not
   raise `max_iter` to make this "pass"** — 16 balls per appendage buys nothing the scene needs (§3:
   the swept volume is identical), so the answer is to cap the registered cluster at 8.
5. **The baseline's cost is not a defect of the baseline.** It is expensive (35.67 s) because the
   returned trajectory sits *inside* the ring, so almost every obstacle generates rows — 4194 of
   them. That is the honest cost of the abstraction, and it is worth one sentence in the manuscript.

### Verdict

**Affordable.** Build demo A at 2 appendages × 8 balls (97 lifted obstacles), degree 8, 8 segments.
Register 4 and 8 as the cluster sizes; do not register 16.

---

## 9. Risks and open questions — the implementer must not decide these alone

- **R1 — "tumbling" is not what this demo does.** A tumbling target carries the port with it, which
  needs a *moving* terminal constraint; `p_end` is a fixed space-time point. Either the paper says
  "rotating appendages" everywhere, or someone scopes new machinery. Do not quietly widen the
  caption. *(PI / user.)*
- **R2 — the approach corridor is load-bearing.** Without the out-of-plane bound the baseline flies
  over the appendage plane and succeeds. The spec's position is: declare it, apply it to both arms,
  and test it (T6b). If the PI judges that too weak, the alternative is a genuinely
  three-dimensional swept set (appendages on non-coplanar circles), which is more obstacles and a
  new phase scan. *(PI.)*
- **R3 — cluster refinement above 8 stops converging while remaining certified.** Same signature as
  `curve` N8_seg16. Recorded, not fixed; owned by `solver.multistart` / the convergence criterion,
  not by this demo.
- **R4 — the scene is fragile in `T_NOM`.** 14/15/16 figure-grade, 18 hits the iteration cap, 11.2
  is vacuous. The scan must be recorded in the scenario docstring the way `loiter` records its phase
  scan, including the values that failed. A single registered number with no scan behind it is the
  thing `loiter`'s docstring exists to prevent.
- **R5 — the elastic weight is unmeasured.** Default 100 with zero raises sufficed for every
  figure-grade run, but that is a solve outcome, not a registry measurement. Measure the ladder and
  register it before quoting anything. *(Owner: whoever runs `journal.measure`.)*
- **R6 — what "the baseline fails" is allowed to mean.** The solver returns a penetrating iterate
  holding 22.4 of slack; it does not *prove* infeasibility. The manuscript may say "the space-only
  baseline does not produce a collision-free trajectory, holding N of violation at the weight cap";
  whether it may say "infeasible" is a claim about a solver's output, and the PI should rule. *(PI.)*
- **R7 — units.** Nondimensional here, argued in §2. If the paper wants metres for the orbital
  dressing, that is a re-tune of `trust_radius` and the elastic weight and a re-measurement of every
  number in §8 — `loiter` is the precedent for how much that costs. *(User / PI.)*
- **R8 — E1.** Every number in §8 is on the E1 build (extension mtime 2026-09-09T14:42:43) and is
  **not** comparable with SOLVER.md §Measurements. If E1 is reverted or lands differently, §8 is
  void and must be re-run, not adjusted.
- **R9 — one seed, one basin.** The straight seed works at `T_NOM = 15`. It was measured **not** to
  work on a faster-rotating variant (one lap per 10 s), where every start weight, trust radius and
  time weight tried returned a penetrating iterate by trust collapse while a hand-built
  hold-then-dash control polygon was feasible and certified at 0.00e+00. Demo A therefore sits
  inside the paper's own declared seed-dependence limitation. If the scene is ever re-tuned faster,
  a designed initial guess (a new `init_curve` mode — only `straight` and `quadratic_bow` exist,
  `spacetime_bezier/objective.py:171`) becomes new code and a new decision. *(PI, and it is also the
  `solver.multistart` item.)*
