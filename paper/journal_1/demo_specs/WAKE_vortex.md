# Demo WAKE — "Cross behind a leader once its wake has decayed"

**Status: build specification. No production code was written for it.** Every affordance below was
read in the source and is cited `file:line`. Every number is measured and carries its configuration,
git sha and extension build in §8; nothing here is an estimate presented as a measurement.

**Owner of the claim.** This demo carries **consequence 2** (timing is a variable inside the same
convex subproblem) and touches **consequence 1** (the keep-out moves). It does **not** carry the
non-convex-tube part of consequence 1 — every wake body's lifted tube is degree 1, so a supporting
half-space exists for each; the curved-tube evidence is `curve` and `loiter` and must stay theirs
(`spacetime_bezier/scenarios.py:323`, `:363`).

---

## 1. Mission and scope

A leader aircraft crosses the follower's route. Its wake vortex pair trails behind it, drifts with a
known crosswind, descends, and **decays**: after a known lifetime the hazard ceases to exist. The
follower is in an assigned airway and flight-level block and must cross the leader's track. Crossing
early means flying through live wake. The right plan holds until the wake **at the crossing point**
has decayed and then crosses, not a second later than necessary.

**What is modelled.** A wake as a chain of ball keep-out zones laid along the leader's past track,
one per shed instant, each with its own birth time (when the leader shed it) and death time (birth +
lifetime), each translating at the wind velocity with a constant sink rate. The decay is a hard
on/off in time, because that is what the formulation represents exactly: an obstacle's existence
window **is** the time coordinate of its first and last lifted control point
(`spacetime_bezier/geometry.py:14-27`, `:44-50`; `:520-522`).

**What is not modelled, and must never be claimed.** No vortex dynamics: no circulation, no Crow
instability, no ground effect, no stratification, no linking or bursting; no gradual weakening (the
hazard is present at full radius and then absent); no induced rolling moment on the follower and
therefore no encounter-severity metric; no atmospheric turbulence and no uncertainty of any kind —
the wind, sink rate and lifetime are exact and known. The ball radius is a hazard-plus-buffer
envelope chosen for the demo, not a computed vortex-hazard boundary. **The demo's subject is the
solver's ability to price an obstacle lifetime, not wake physics.** Any paper sentence that reads as
a wake-encounter result rather than a trajectory-optimization result is out of scope.

---

## 2. Scene geometry

### 2.1 Units — nondimensional, with a stated scale map

**Decision: author the scene in the nondimensional 10-unit box, and publish a single scale map in
the caption.** The argument, both halves:

* Every solver knob in this repository was tuned in the 10-unit box. `DEFAULT_TRUST_RADIUS = 0.5`
  (`spacetime_bezier/optimize.py:19`) and `DEFAULT_INITIAL_ELASTIC_WEIGHT = 100.0` (`:61`) are the
  values the 28-configuration table was taken at. `loiter` is the recorded cost of leaving that box:
  it needed a scenario-level `trust_radius` of 5.0 because the clip reach is built from it, and the
  stock 0.5 against a 200 m scene left the constrained run losing the link with its arrival pinned
  against the horizon; and it needed its elastic weight re-measured
  (`spacetime_bezier/scenarios.py:363` docstring, "Two solver knobs are lengths"). WAKE in the box
  needs neither override — measured: trust radius 0.5, elastic weight resolving to 100 with **zero**
  raises (§8).
* A reviewer needs metres. Give them the map instead of the metres, and the numbers are checkable
  both ways.

**Scale map (publish this, do not bury it):** 1 length unit = **1600 m**, 1 time unit = **20 s**,
hence 1 speed unit = **80 m/s**. Under that map every scene constant lands in a physically sane
range, which is the point of choosing it:

| scene quantity | nondimensional | in the scale map | sanity |
|---|---|---|---|
| follower speed cap `v_max` | 1.5 u/s | 120 m/s | light transport |
| leader speed `V_L` | 2.0 u/s | 160 m/s | leader faster than follower |
| wake body radius `r` | 0.5 u | 800 m | hazard + buffer envelope |
| wake lifetime `tau` | 5.0 u_t | 100 s | 60–180 s is the usual range |
| sink rate `w_z` | 0.015 u/s | 1.2 m/s | initial descent 1.5–2 m/s |
| crosswind `w_x` | 0.10 u/s | 8 m/s | ≈16 kt |
| route length | 9.0 u | 14.4 km | |
| lateral airway block | ±1.0 u | ±1600 m | |
| flight-level block | ±0.15 u | ±240 m | ≈ ±800 ft, tighter than RVSM |
| modelled wake length | 5.5 u | 8.8 km behind the leader | truncated, see §3.4 |
| authoring horizon `T` | 15.0 u_t | 300 s | |

The sink-to-along-track ratio is the one number that has to be right for the picture not to lie:
0.075 u of sink over 10 u of leader travel during one lifetime = **0.0075**, against a real wake
descending ~150 m while its generator flies ~20 km = 0.0075. That agreement is why the sink rate is
0.015 and not something rounder.

### 2.2 The scene

Frame: `x` along the follower's route, `y` across it, `z` altitude. Time is the fourth coordinate,
so `dim = 4` and the viewer must filter this scenario out exactly as it filters `loiter`
(`spacetime_bezier/scenarios.py:526` block comment).

```
T (authoring / clamp horizon) = 15.0        # NOT the arrival; see §2.3
start = [0.5, 5.0, 5.0,  0.0]
end   = [9.5, 5.0, 5.0, 10.0]               # 10.0 is the SEED arrival, freed at solve time
coord_bounds = [[0.0, 10.0], [4.0, 6.0], [4.85, 5.15]]

x_w   = 5.0     # leader track / crossing plane
y0    = 2.0     # leader y at t = 0
V_L   = 2.0     # leader ground speed along +y
z_lvl = 5.0     # leader and follower share the level (the realistic encounter)
r     = 0.5     # wake body radius
ds    = 0.25    # shed interval
s_max = 2.75    # last shed instant modelled
tau   = 5.0     # wake lifetime
wind  = [0.10, 0.0, -0.015]                 # drift, no along-track component, sink
```

Leader: `pos0 = [5.0, 2.0, 5.0]`, `vel = [0.0, 2.0, 0.0]`, `r = 0.5`, window `[0.0, T]`, name
`LEAD`. **Identical in both arms of the pair** — the one knob touches wake bodies only. Measured: it
never binds (clearance +3.5177 on the constrained run, §8).

### 2.3 `T` is load-bearing, and getting it wrong silently destroys the pair

`T` here is **not** the arrival time. It is (a) the clamp `normalize_obstacle` applies to every
authored window, `t_start = max(0, min(t_start, T))`, `t_end = max(t_start, min(t_end, T))`
(`spacetime_bezier/geometry.py:137-139`), and (b) it must cover the whole interval a freed arrival
can reach. The freed arrival's upper bound is `P_init[-1, -1] * time_ub_scale` with
`time_ub_scale = 1.5` by default and no way to override it through `optimize_scenario`
(`spacetime_bezier/optimize.py:428`, `:494`; `optimize_scenario` at `:709` does not take the
argument, and `tools/make_paper_figure.py:51` `solve_pair` does not pass it either). With a seed
arrival of 10.0 that bound is **15.0**, confirmed in every run's `time_ub_used` (§8).

So `T = 1.5 x seed arrival = 15.0`. **Measured consequence of setting `T = 10.0` instead:** every
baseline window is clamped to `[0.0, 10.0]`, the follower simply waits past 10.0 where nothing
exists any more, and the baseline comes back **converged, figure-grade, clearance +0.4558, arrival
13.0837** — the pair collapses into two successes and proves nothing. With `T = 15.0` the same
baseline penetrates by 0.1834 and does not converge. Both numbers in §8.

### 2.4 The construction rule

Two conditions, and both are checked by a test rather than by this arithmetic (§6.1, §6.2):

1. **The straight plan must enter live wake.** The straight seed crosses `x = 5.0` at `t = 5.0`,
   inside the seal. Measured on the seed polygon: minimum clearance **−0.4409**.
2. **A waiting plan must exist inside the horizon.** The chain seals the corridor cross-section
   through `t = 6.0` and first opens a hole at `t = 6.2`; the follower needs
   `(9.5 − 5.5)/1.5 ≈ 2.7` more units of time after crossing, so the earliest feasible arrival is
   around 9.4 — comfortably below the 15.0 bound. Measured arrival **9.7437**, with
   `arrival_on_time_ub = 0`: the answer is set by the geometry, not by the bound.

The gap between them is the demo. The wake-free vehicle would arrive at **6.0000** (= 9.0 / 1.5, the
speed cap binding); the wake costs **3.7437** units of time, 74.9 s under the scale map.

---

## 3. Obstacle construction

### 3.1 `make_wall` cannot be used, and the reason is the demo's whole point

`make_wall` takes a **single** `t_start` / `t_end` and stamps it on every circle in the row
(`spacetime_bezier/scenarios.py:75-78`). A wake is exactly the object that row cannot express: each
body has its own birth time. Recommend adding a sibling helper beside it — same authoring style,
same legacy dicts, so the existing `@_canonical` decorator (`:24-45`) does the lifting unchanged:

```python
def make_wake(shed_path, t_shed, lifetime, wind, radius,
              color="#e67e22", name_prefix="K") -> list[dict]:
    """A row of balls along a leader's past track, each with its OWN window.

    `shed_path(s)` returns the leader's position at shed time `s`; `t_shed` is
    the list of shed instants. Body i is authored as the legacy straight
    obstacle whose position law is `shed_path(s_i) + wind * (t - s_i)`, i.e.
    `pos0 = shed_path(s_i) - wind * s_i`, `vel = wind`, alive on
    `[s_i, s_i + lifetime]`.

    This is `make_wall` with the window moved from the ROW to the BODY, which is
    the only structural difference and the entire content of demo WAKE.
    """
```

### 3.2 The lift, and the test that pins it

Each authored body is the legacy `{pos0, vel, r, t_start, t_end}` form.
`normalize_obstacle` (`spacetime_bezier/geometry.py:105-151`) takes the legacy branch at `:124` and
builds exactly two lifted control points (`:140-146`):

```
cps = [[pos0 + vel * t_start, t_start],
       [pos0 + vel * t_end,   t_end  ]]
```

so the body's **existence window is intrinsic to its first and last control-point time** — there is
no `t_start`/`t_end` pair downstream at all (`geometry.py:520-522`), and `MovingObstacle.t_start` /
`.t_end` are properties read off row 0 and row −1 (`geometry.py:44-50`).
`bezier_obstacle_from_moving` (`geometry.py:490-498`) is the same function with a list-valued return
for the JSON boundary.

**Cite this test, not this paragraph:**
`tests/unit/test_bezier_obstacle_roundtrip.py::test_finite_time_window_is_carried_by_the_control_points`
— a `{pos0:[2.5,5.0], vel:[0,0], r:0.5, t_start:0.0, t_end:5.0}` body with `T = 10.0` must lift to
control points spanning `[0, 5]`, not `[0, T]`. *It would fail if* the reader ever widened a short
window back to the scenario duration, which is precisely the modelling error the WAKE baseline
deliberately commits. Its neighbour
`test_a_straight_legacy_obstacle_lifts_to_its_own_endpoints` *would fail if* the lift stopped
evaluating `pos0 + vel*t` at the window ends, which is what makes the drift and sink faithful.

### 3.3 The bodies

For `k = 0 .. 11`, shed instant `s_k = 0.25 k` (so `s` runs 0 to 2.75):

```
shed position  L(s_k) = [5.0, 2.0 + 2.0*s_k, 5.0]          # = [5.0, 2.0 + 0.5k, 5.0]
pos0           = L(s_k) - wind*s_k = [5.0 - 0.10*s_k, 2.0 + 0.5k, 5.0 + 0.015*s_k]
vel            = [0.10, 0.0, -0.015]
r              = 0.5
t_start, t_end = s_k, s_k + 5.0
name           = f"K{k}"
```

Chain: 12 bodies from `y = 2.0` to `y = 7.5`, centre spacing `d = 0.5` along `y`.
**Plus the leader = 13 lifted obstacles.**

### 3.4 The spacing argument

Two failure modes bound the spacing from opposite sides, and both are measured history in this
repository, not intuition.

**Too sparse → a false corridor.** `door3d` records the escape route in its own docstring
(`spacetime_bezier/scenarios.py:229-241`): two stacked rows of `r = 0.9` spheres at centre distance
1.84 against a radii sum of 1.80 left a diagonal pore that the solver threaded at +0.007 clearance;
a sealed row that did not span far enough in `y` was flown around at its end for +0.084. A chain
whose union has a hole does not make the follower wait — it makes it thread, and every timing claim
in the demo becomes a claim about a hole.

**Too dense → conservatism, at a cost that is also measured.** `wall` was densified 2026-08-24 from
spacing 0.8 to 0.5, 13 circles to 21. The forbidden slab barely moved, but eight more per-segment
planes made the relaxation more conservative and `wall` N8_seg2 went from +0.102 to **−0.129** —
penetrating outright — which is why one test in this repository is red on purpose
(`SOLVER.md:527-540`; `tests/integration/test_figure_grade_gate.py::test_a_clearing_run_can_still_be_standing_on_slack`).
**Do not touch `wall`'s density for this or any other reason.** Density is not free, and its price
is paid at low segment counts.

**The convention this repository already follows** is spacing ≤ radius:

| scenario | `r` | authored spacing | realised centre distance `d` | `d/r` | pinch half-thickness `sqrt(r² − (d/2)²)` |
|---|---|---|---|---|---|
| `wall` before 2026-08-24 | 0.5 | 0.8 | 0.833 | 1.67 | 0.276 |
| `wall` now (`scenarios.py:135`) | 0.5 | 0.5 | 0.500 | 1.00 | 0.433 |
| `fence3d` (`:189`) | 0.9 | 0.9 | 0.900 | 1.00 | 0.779 |
| `door3d` (`:252`) | 1.6 | 0.8 | 0.824 | 0.51 | 1.546 |
| **WAKE (recommended)** | **0.5** | **0.5** | **0.500** | **1.00** | **0.433** |

WAKE sits at `d/r = 1.0`, the loose end of the established band, alongside `wall` and `fence3d`. It
gets away with the loose end because the chain is only 5.5 long, so it costs **12** bodies against
`wall`'s 21 and `door3d`'s 18.

**Spacing is set by the leader, not chosen freely.** `d = V_L * ds`, so with `V_L = 2.0` the shed
interval `ds = 0.25` is what buys `d = 0.5`. Halving `ds` doubles the body count for the same
geometry; the general rule for a re-scaled leader is `ds <= r / V_L`.

**Why the flight-level block is ±0.15 and not wider.** At the pinch between two neighbours the union
is only as thick as the pinch half-thickness, 0.433, and the oldest live body has sunk
`0.015 * 5.0 = 0.075`, so the sealed band about `z = 5.0` is `[4.567 − 0.075, 5.433 − 0.075]`.
`[4.85, 5.15]` sits inside that with 0.083 to spare at the top. **This margin was bought with a
failed run, not with arithmetic:** the first scene tried `w_z = −0.03` with a `±0.20` block, and the
baseline — whose widened window lets the same bodies keep sinking for the full 15.0 instead of 5.0,
reaching `z = 4.55` — flew **over** the sunk wake and came back converged and figure-grade at
clearance +0.00029, arrival 11.99. The pair was destroyed by the sink rate. Numbers in §8, discarded
arm.

### 3.5 What the solver actually builds

Rows are one plane per (segment, obstacle, component) — a **wall** — binding every control point of
its segment, so `keep-out rows = walls x (N+1)`. Verified at the returned iterate via
`bezier_opt.spacetime_koz_rows_exact` (`rust_optimizer/pybind/src/lib.rs:399-490`; a wall is a
unique `(segment_idx, obstacle_idx, component_idx)` among rows whose `station_idx` is −1):

| arm | obstacles | `n_seg` | walls built | keep-out rows | rows / walls |
|---|---|---|---|---|---|
| constrained | 13 | 16 | 29 | 261 | 9 = N+1 |
| baseline | 13 | 16 | 34 | 306 | 9 |
| no-wake (leader only) | 1 | 16 | 4 | 36 | 9 |

29 of a possible `13 x 16 = 208`: a wall is built only where the obstacle is within reach of the
segment, which is the linearity claim doing visible work.

---

## 4. Solver knobs that must travel with the scenario

| knob | value | why it must travel |
|---|---|---|
| `free_arrival_time` | `True` | The demo IS the arrival time. Refused unless priced: `check_free_arrival_is_costed` raises `DegenerateFreeArrivalError` when `time_weight == 0`, because the reported arrival is then an interior-point tie-break measured to swing 10.08 → 12.24 with the trust radius alone (`optimize.py:379-393`). |
| `time_weight` | `10.0` | Must be > 0 for the above. Measured insensitive here: 1.0 → arrival 9.8245, 10.0 → 9.7437, 100.0 → 9.5447, all figure-grade. Two decades of weight move the arrival by 2.9%, so the arrival is a property of the geometry. |
| `v_max` | `1.5` | Must be > 0 with a time penalty or `check_time_penalty_is_capped` raises `UncappedTimePenaltyError` (`optimize.py:274-296`) — an uncapped time penalty collapses the arrival to `min_dt x gaps` for every scenario alike. It must also **bind**: the non-binding threshold on this scene is `chord/(min_dt*N) = 9.0/0.8 = 11.25`, and 1.5 is far below it. Confirmed by `arrival_on_min_dt_floor = 0` and by the no-wake arm returning exactly `9.0/1.5 = 6.0000`. Well inside `MAX_SPEED_CAP = 1e6` (`optimize.py:350`) and below the ~1e5 measured degradation. |
| `min_dt` | `0.1` | The shipped default; the floor never binds here (`arrival_on_min_dt_floor = 0` on every arm). |
| `scp_trust_radius` | `0.5` (the default — **do not** set a scenario key) | A length. The scene is 10 units, i.e. the box every knob was tuned in, so unlike `loiter` (`scenarios.py:363`) no override is needed. Measured to work; setting a key would silently change the clip reach `seg_radius + trust*sqrt(dim)`. |
| `elastic_weight` | start at the default 100, escalate in-loop | Measured: the constrained run at `time_weight = 10` resolves to **100 with zero raises**. **Do not add `wake` to `SCENARIO_ELASTIC_WEIGHT`** (`scenarios.py:287`) — an unnecessary registry entry hides the fact that this scene is easy, and `wall`'s 1e5 there is a measurement of `wall`. |
| `coord_bounds` | `[[0,10],[4.0,6.0],[4.85,5.15]]` | The airway and the flight-level block. Hard rows outside the elastic slack range, so the penalty cannot buy through them — the `loiter` recipe (`scenarios.py:363` docstring, "The corridor as a TUBE"). They are what make going *around* and going *over* unavailable, so that timing is the escape. Endpoints must lie inside or the solver refuses loudly (`optimize.py:471-489`). `optimize_scenario` reads the key off the scenario (`:743`); `solve_pair` forwards it by hand (`tools/make_paper_figure.py:74`). |
| `sound_clip` | `True` | The reach floor, so the certificate speaks for the whole keep-out zone. Measured `koz_unsound_clips = 0` on every arm. |
| `N`, `n_seg` | `8`, `16` | **16 segments is required, and this is measured, not stylistic:** at `n_seg = 8` the constrained run trust-collapses, penetrating by 0.2843 with 2.425 of slack at weight 1e5. The same pattern as `wall`, which certifies only at 16 segments (`SOLVER.md:314-316`). |

Register in `SCENARIO_MAP` (`scenarios.py:526`) as `"wake": (scenario_wake, [(8, 16), (10, 16)])`,
with `(8, 8)` recorded in the docstring as a **measured** failure rather than omitted silently.

---

## 5. The pair

### 5.1 The baseline, one knob

`scenario_wake(wake_lifetime=tau)` builds the constrained scene. The baseline is the **same
constructor with every wake body's window widened to the full authoring horizon**:

```
constrained:  t_start, t_end = s_k, s_k + tau        # tau = 5.0
baseline:     t_start, t_end = 0.0, T                # T   = 15.0
```

Nothing else changes: same `pos0`, same `vel`, same `r`, same leader, same `coord_bounds`, same
`N`, `n_seg`, `v_max`, `time_weight`, `min_dt`, trust radius, start weight, `sound_clip`, seed.
Because `normalize_obstacle` lifts a legacy body to the two endpoints of its own window
(`geometry.py:140-146`), the baseline body's lifted segment is the **same line in space-time**,
extended at both ends — the constrained body's segment is a sub-segment of it. That is what makes
this one knob and not two, and §6.2 asserts it on the arrays rather than on the call site.

### 5.2 Why the failure is structural

A decomposition-based planner builds free-space regions (IRIS/GCS-style) once, or once per fixed
time slice, and the region graph carries no notion of a region that exists only on an interval. It
must therefore treat a keep-out zone as present wherever and whenever it can be — the conservative
closure over the horizon. Widening the window to `[0, T]` **is** that closure expressed in this
solver's own obstacle format, which is why it is a fair one-knob comparison rather than a detuned
straw man: the baseline solves the same convex subproblems with the same weights and the same
geometry, and is handed strictly more obstacle than the truth.

The measured result is that the closure has no feasible crossing at all. The corridor cross-section
is sealed for the **entire** horizon (§6.1 lattice: zero unblocked points at `t = 6.0`, `10.0` and
`14.9`), so the baseline pushes its arrival to the upper bound and penetrates anyway:
**arrival 15.0000 with `arrival_on_time_ub = 1`, clearance −0.1834, hull certificate 1.5077, slack
1.5077, stop reason `trust_collapse`, not converged, not figure-grade.** The pinned arrival is the
signature worth putting in the caption: *the baseline tries to wait, and there is nothing to wait
for.*

### 5.3 A third arm, and why it is needed

The pair alone cannot carry "timing, not path", because the baseline fails by arriving **later**
than the constrained run, not earlier. Measured: the constrained spatial path flown on the
**baseline's** schedule scores **+3.3959** — it clears, because the baseline's schedule crosses
`x = 5` at `t = 11.94` when the wake is long dead. Grafting onto a schedule that is later than the
one under test measures nothing.

So the demo carries a **third, unconstrained-in-time reference: the wake-free arm** — the same scene
with the wake bodies removed and the leader kept. Its schedule is the earliest the vehicle can fly
(arrival exactly `9.0 / 1.5 = 6.0000`, the speed cap binding), and it is the correct graft partner.
It is not a baseline and must never be presented as one; it is the zero of the delay axis.

---

## 6. Tests

One file, `tests/integration/test_wake_scenario.py`, module-scoped fixture solving all three arms
once, on the pattern of `tests/integration/test_loiter_scenario.py` — and, like that file, it must
import the figure tool's own `solve_pair`-equivalent rather than reimplementing the arms, so a drift
in the figure's parameters fails here instead of producing a figure nobody re-checked
(`test_loiter_scenario.py:65-80`).

`compute_min_clearance` (`geometry.py:356`) gives a scalar; the grafts need a per-sample clearance.
Add a test-local `clearance_at(positions, times, obstacles)` mirroring
`los_margin_at` (`geometry.py:440-487`): for each normalized obstacle, mask on
`t_start <= t <= t_end`, take `||position − obstacle_positions_at(cps, t)|| − r`, minimise over the
active ones, `+inf` where none is active. If it is wanted in production it belongs next to
`los_margin_at`, for the reason recorded there — one arithmetic, not two copies.

### 6.1 Non-vacuity — the chain has no hole, and the gate does open

Sweep a lattice of the corridor cross-section (`y` in `[4.0, 6.0]` x 161, `z` in `[4.85, 5.15]` x
41) and, for each point, sweep `x` from 3.0 to 7.0 x 241, asking whether **any** live body contains
the sample. Assert **zero** unblocked `(y, z)` points at `t = 3.0, 5.0, 6.0`, and **more than zero**
at `t = 6.5`.

**FAILS IF** a `(y, z)` line through the chain is clear at a blocking time — the chain has a false
corridor and every waiting claim in this file is a claim about a hole; or if no hole ever appears —
the corridor never opens and the constrained run's success is impossible, so the fixture is wrong.
Measured: 0 / 0 / 0 unblocked at 3.0 / 5.0 / 6.0; 73 of 6601 at 6.2; 1712 at 6.5; 4991 at 6.8.

### 6.2 Non-vacuity — the seed genuinely conflicts

Evaluate `clearance_at` on the straight seed polygon from `start` to `end`. Assert `< −0.2`.
**FAILS IF** the initial guess already clears the wake, in which case the solver was never asked the
question. Measured **−0.4409**.

### 6.3 The pair differs only in the windows

For every wake body, assert (a) the baseline's lifted window is exactly `(0.0, T)`; (b) the
constrained window is `(s_k, s_k + tau)` and is strictly inside it; (c) the two lie on the **same
line**: `obstacle_positions_at(constrained_cps, t)` equals
`obstacle_positions_at(baseline_cps, t)` to 1e-12 at a dozen times inside the constrained window;
(d) radii, names and the leader's obstacle are identical arrays; (e) the two solve calls received
identical `N`, `n_seg`, `v_max`, `time_weight`, `min_dt`, trust radius, start weight, `sound_clip`,
`coord_bounds`, seed.

**FAILS IF** any second knob has crept in — in which case every comparison in the file is between
two different problems, which is the doctoring
`test_loiter_scenario.py:198-221` was written to catch after a peer turned that pair into two
baselines.

### 6.4 The baseline fails, and fails the way the claim says

Assert on the baseline arm: `min_clearance < −0.1`; `converged is False`;
`figure_grade_failures(row) != []`; `arrival_on_time_ub == 1.0`.

**FAILS IF** the baseline crosses cleanly (the lifetime was decoration and the figure proves
nothing); or if it fails while its arrival is *not* on the bound, which would mean it failed for
some reason other than "there was nothing to wait for" and the caption's explanation is wrong.
Measured: −0.1834, not converged (`trust_collapse`), arrival 15.0000 on the bound, certificate
1.5077.

**Variant to hold in reserve, not to ship as well.** Removing `coord_bounds`' lateral pair turns the
baseline's failure from infeasibility into a detour around the chain's `y = 2.0` end — a lateral
excursion of at least `5.0 − (2.0 − 0.433) = 3.43`, 38% of the 9.0 route. It is a weaker statement
(a detour is a cost, infeasibility is a proof) and it costs a second scene. If the reviewers ask for
it, the bound to assert is `max |y − 5| > 3.0` on the baseline; ship one, not both.

### 6.5 The constrained run certifies, with nothing uncovered

Assert: `converged`; `koz_violation_reference <= 1e-6`; `koz_unsound_clips == 0.0`;
`total_koz_slack_returned <= 1e-6`; `speed_cap_violation <= 1e-6`;
`compute_min_clearance > 0.1`; `figure_grade_failures(row) == []`. Assert the two independent
computations agree — a certificate of 0.0 beside a negative sampled clearance is the finding, not a
pass (`test_loiter_scenario.py:259-282`).

**FAILS IF** the hull certificate leaves zero; if a clipped volume escapes the wall built against it
(then the certificate speaks for less than the keep-out zone, `optimize.py:184-224`); or if the
sampled clearance goes negative while the certificate stays at zero.
Measured: converged `stationary` in 13 iterations, certificate 0.0, slack 2.09e-09, unsound clips 0,
clearance +0.2065, arrival 9.7437, figure-grade.

### 6.6 Timing, not path — the graft

Take the constrained run's **spatial path**, fly it on the **wake-free arm's schedule** (same curve
parameter, the other run's time column — `test_loiter_scenario.py:150-162`), and re-measure with
`clearance_at`. Assert `< 0`. Reverse: the wake-free path on the constrained schedule, assert `> 0`.

Three guards, on the pattern of `test_loiter_scenario.py:317-368`:

1. the two schedules differ by more than 2.0 in arrival, and the constrained one is **later** — or
   both grafts are the run itself;
2. no NaN, and more than 99% of graft samples have an active obstacle — an all-`+inf` graft is no
   measurement;
3. the constrained lateral deviation `max |y − 5|` is below 0.05 — if the run sidestepped, "the
   retiming saved it" does not follow even when the graft happens to fail.

**FAILS IF** either graft direction reverses, or any guard trips.
Measured: constrained path on the wake-free schedule **−0.3822**; wake-free path on the constrained
schedule **+0.1595**; arrivals 9.7437 against 6.0000; lateral deviation **7.19e-05**.

**Record, do not assert, the baseline graft: +3.3959.** It is in the sidecar with the one-line
reason (§5.3) so that a later reader does not "restore" the loiter-shaped test that cannot work
here.

**And the honest caveat that must be in the spec and the paper.** The constrained run does use
0.0934 of its ±0.15 vertical block — 62% of it — so the path is not literally unchanged in `z`. That
is why §6.7 exists.

### 6.7 The geometric freedom is not what does the work

Re-solve the constrained arm with `coord_bounds` pinched to `[[0,10],[4.99,5.01],[4.99,5.01]]` —
the same wake, the same knobs, a corridor collapsed onto the axis. Assert it still converges
figure-grade, and that its arrival exceeds the full-block arrival by **less than 0.2**.

**FAILS IF** collapsing the corridor changes the arrival materially — then the lateral and vertical
freedom, not the retiming, is buying the crossing, and §6.6's graft is being carried by the 0.0934
of climb.
Measured: pinched arrival **9.7688** against **9.7437**. The freedom buys **0.0251**, which is
**0.7%** of the 3.7437 delay. (A one-solve test; if the file's runtime budget is tight, keep it —
this is the assertion that answers the obvious reviewer objection.)

### 6.8 Decay actually matters

Re-solve the constrained arm with `wake_lifetime = 15.0` — every window becomes `[s_k, T]` after the
clamp, the geometry, drift and sink untouched. Assert it reproduces the baseline's failure:
`min_clearance < −0.1`, `converged is False`.

**FAILS IF** a wake that never decays still lets the follower cross — which would mean something
other than the lifetime opens the gate, the sink being the obvious candidate. This is the test that
keeps the descent honest.
Measured: clearance **−0.18335848** against the baseline's **−0.18335847**; certificate 1.50767298
against 1.50767295. The two agree to seven significant figures — as they must, since a wake alive
from its birth to the horizon and a wake alive from time zero to the horizon differ only where the
follower never is.

### 6.9 The `T` clamp is not a free parameter

Assert `scenario_wake()["T"] == 1.5 * scenario_wake()["end"][-1]`, with the reason in the message.

**FAILS IF** someone "tidies" `T` to match the arrival, which silently truncates the baseline's
windows and lets it wait past the horizon. Measured consequence of `T = 10.0`: the baseline comes
back converged, figure-grade, clearance +0.4558, arrival 13.0837 — a pair of two successes.

### 6.10 The leader never binds

Assert the leader's own per-obstacle minimum clearance on the constrained run exceeds 2.0.

**FAILS IF** the leader body becomes the binding constraint, in which case the demo is a
"don't hit the other aeroplane" demo and the wake is decoration. Measured **+3.5177**; the binding
bodies are K7 (+0.2072), K6 (+0.3451), K8 (+0.5194) — the three straddling the corridor axis at the
moment of crossing.

---

## 7. Figure specification

**Layout: three space panels in a row above one margin-versus-time panel**, on the pattern of
`tools/make_paper_figure.py`.

* **Panels A/B/C — `(x, y)` plan view at three times: `t = 5.0` (before), `t = 6.73` (at the
  crossing, the constrained run's own crossing instant), `t = 8.5` (after).** In each: the airway
  block as a light band, the flight-level block noted in the panel label, the leader as a filled
  marker with its track, the wake chain as circles, and the two trajectories' positions at that
  instant as markers with their travelled paths behind them.
  **Live versus decayed bodies must be visually distinguished and the distinction must be the
  figure's subject**: a live body is a filled circle at full opacity; a decayed body is drawn as a
  dashed outline at low opacity **in the position it would have had** — so the reader sees the chain
  emptying from the bottom, which is the mechanism. A legend entry for each. Never omit the decayed
  bodies: a chain that just gets shorter reads as an obstacle leaving, not as one expiring.
* **Panel D — clearance versus time**, both arms plus the wake-free arm, zero line marked, the
  constrained arm's crossing instant marked, and the baseline's arrival marked at 15.0 against the
  freed-arrival bound. This is the panel that proves the claim; the plan views are what make it
  legible.
* Optional inset, if the space-time picture earns it: the `(x, t)` lift at `y = 5.0, z = 5.0`,
  showing each wake body's tube as a finite segment and the constrained trajectory's steep,
  near-vertical stretch at the crossing plane — the visual signature `door3d` describes
  (`scenarios.py:219-223`). Only if it does not crowd the four panels.

**Caption claim (one sentence, and it must be the sentence the tests assert):** *the follower holds
short and crosses 3.74 units of time (74.9 s) later than it could in wake-free air, at the moment
the wake at the crossing point expires; the identical problem with the wake's lifetimes removed —
one knob, the obstacle windows — has no crossing at all, pushes its arrival to the horizon bound and
penetrates by 0.18.* Do not write "faster", "optimal", or "guaranteed globally"; `converged` asserts
feasibility plus no-further-progress, never stationarity.

**Sidecar (`wake_figure.json`) fields.** Everything in `tools/make_paper_figure.py:450-510` that
applies, minus the station fields, plus these:

* provenance: `generated`, `git` (with `+dirty`), **`extension_path` and `extension_mtime` as an ISO
  local timestamp** (a rebuilt extension is a different solver — see §8), `python`, `machine`;
* configuration: `N`, `n_seg`, `free_arrival`, `time_weight`, `v_max`, `min_dt`, `trust_radius`
  (resolved, not the key), `sound_clip`, `initial_elastic_weight`, `coord_bounds`, `T`,
  `time_ub_used`;
* scene: `leader_speed`, `wind`, `wake_lifetime`, `shed_interval`, `body_radius`, `n_wake_bodies`,
  `n_obstacles`, `chain_y_extent`, `scale_map` (the three factors in §2.1);
* per arm (`constrained`, `baseline`, `no_wake`, and `long_lifetime` if run): `arrival_time`,
  `arrival_on_time_ub`, `arrival_on_min_dt_floor`, `min_clearance`, `koz_certificate`,
  `total_slack`, `unsound_clips`, `elastic_weight` (resolved) and `weight_raises`,
  `speed_cap_violation`, `iterations`, `stop_label`, `converged`, `figure_grade`,
  `figure_grade_reasons`, `solve_seconds`, `walls_at_returned_iterate`, `keepout_rows`,
  `lateral_deviation`, `vertical_deviation`, `crossing_time_at_x_w`;
* the claim's own numbers: `delay_vs_wake_free` (= constrained arrival − no-wake arrival),
  `graft_constrained_path_on_no_wake_schedule`, `graft_no_wake_path_on_constrained_schedule`,
  `graft_constrained_path_on_baseline_schedule` (recorded, with `graft_baseline_note`),
  `pinched_corridor_arrival` and `freedom_buys` (§6.7), `seal_lattice` (the time → unblocked-count
  map from §6.1), `binding_obstacles` (the three smallest per-obstacle clearances by name);
* both returned control polygons, so the figure can be redrawn from the run that was measured.

---

## 8. Experiment results

**These are a feasibility probe for the spec, not a measurement pass for the paper.** They were
taken on a **dirty working tree** and on the **E1 solver**, and they are **not comparable to
`SOLVER.md` §Measurements**, whose 2026-09-08 pass is pre-E1.

**Provenance.**

* `git rev-parse HEAD` = `5b3ef7882c8e62dedcc9f54a7d07cacf253aec86`, working tree **dirty**:
  modified `paper/ksas_2026_fall/poster/{README.md,make_poster_figures.py,poster.tex}` and
  **`rust_optimizer/core/src/spacetime_obstacle.rs`**; untracked `paper/journal_1/`,
  `spacetime_bezier/families.py`, `tests/integration/test_families_rows.py`,
  `tests/unit/test_families.py`.
* Extension: `/Users/hyeon-yongjeong/code/bezier-trajectory-merge/.venv/lib/python3.14/site-packages/bezier_opt/bezier_opt.cpython-314-darwin.so`,
  `os.path.getmtime` = `1788932563.3962753` = **2026-09-09T14:42:43 local**. **This is the E1
  build.** A build stamped 2026-09-08 20:22 is pre-E1; every number below is post-E1.
* **E1**, uncommitted in `rust_optimizer/core/src/spacetime_obstacle.rs:393-406`: with
  `sound_clip` on, the clipping radius is now the minimal sound radius `floor.min(r_max)` rather
  than `r_nearest.clamp(floor.min(r_max), r_max)`. The `sound_clip = False` arm keeps the old rule.
  Every run here used `sound_clip = True`, so every run is on the new clip rule.
* Python 3.14.6, arm64 Darwin. Not rebuilt; nothing in the repository was edited except this file.
* Throwaway scripts: `wake_probe.py` … `wake_probe5.py` in the session scratchpad. They are not
  repository files and are not a deliverable.

**Common to every solve:** `N = 8`, `dim = 4`, `p_start`/`p_end` and `coord_bounds` of §2.2,
`free_arrival_time = True`, `min_dt = 0.1`, `v_max = 1.5`, `sound_clip = True`,
`scp_trust_radius = 0.5`, starting `elastic_weight = 100` with in-loop escalation,
`max_iter = 200`, `tol = 1e-6`, `init_curve = {"mode": "straight"}`, `stations = None`.
`time_ub_used = 15.0` on every arm. `koz_unsound_clips = 0.0` on every arm.

### 8.1 The recommended scene (`w_z = −0.015`, block `[4.85, 5.15]`)

| arm | `n_seg` | `tw` | iters | stop | arrival | on ub | clearance | certificate | slack | weight/raises | walls | rows | s | figure-grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **constrained** | 16 | 10 | 13 | stationary | **9.7437** | 0 | **+0.2065** | 0.0 | 2.09e-09 | 100 / 0 | 29 | 261 | 1.28 | **yes** |
| **baseline (windows → [0,15])** | 16 | 10 | 61 | trust_collapse | **15.0000** | **1** | **−0.1834** | 1.5077 | 1.5077 | 1e5 / 3 | 34 | 306 | 4.30 | no |
| long lifetime `tau = 15` | 16 | 10 | 90 | trust_collapse | 15.0000 | 1 | −0.18336 | 1.50767 | 1.50767 | 1e5 / 3 | 34 | 306 | 4.49 | no |
| wake-free (leader only) | 16 | 10 | 7 | stationary | **6.0000** | 0 | +1.3000 | 0.0 | 1.55e-10 | 100 / 0 | 4 | 36 | 0.03 | yes |
| constrained, `n_seg = 8` | 8 | 10 | 43 | trust_collapse | 8.7806 | 0 | −0.2843 | 2.4250 | 2.4250 | 1e5 / 3 | 30 | 270 | 0.40 | no |
| constrained, `tw = 1` | 16 | 1 | 12 | stationary | 9.8245 | 0 | +0.2748 | 0.0 | 8.37e-11 | 100 / 0 | 29 | 261 | 1.16 | yes |
| constrained, `tw = 100` | 16 | 100 | 27 | stationary | 9.5447 | 0 | +0.1579 | 0.0 | 1.76e-09 | 1000 / 1 | 29 | 261 | 1.93 | yes |
| constrained, corridor pinched to axis | 16 | 10 | 27 | stationary | 9.7688 | 0 | +0.2180 | 0.0 | 2.84e-08 | 1000 / 1 | 29 | 261 | 2.10 | yes |
| **baseline with `T = 10` (the trap)** | 16 | 10 | 14 | stationary | 13.0837 | 0 | **+0.4558** | 0.0 | 2.83e-09 | 100 / 0 | 29 | 261 | 1.55 | yes — **pair collapsed** |

`speed_cap_violation` was 0.0 on every arm except the baseline's 2.03e-11.

### 8.2 Derived, from the returned control polygons (8001-point sampling, no further solves)

| quantity | value |
|---|---|
| constrained crossing of `x = 5.0` | `t = 6.7291`, `y = 5.00000`, `z = 5.0934` |
| wake-free crossing of `x = 5.0` | `t = 3.0000` |
| baseline crossing of `x = 5.0` | `t = 11.9386` |
| delay attributable to the wake | **3.7437** (74.9 s) |
| constrained lateral deviation `max|y−5|` | **7.19e-05** |
| constrained vertical deviation `max|z−5|` | 0.0934 (62% of the ±0.15 block) |
| graft: constrained path on wake-free schedule | **−0.3822** |
| graft: wake-free path on constrained schedule | **+0.1595** |
| graft: constrained path on baseline schedule | +3.3959 (records why the baseline is the wrong graft partner) |
| straight seed on its own schedule | **−0.4409** |
| binding bodies on the constrained run | K7 +0.2072, K6 +0.3451, K8 +0.5194 |
| leader on the constrained run | +3.5177 (never binds) |
| corridor freedom buys (pinched vs full block) | **0.0251**, 0.7% of the delay |
| seal lattice, constrained chain, unblocked of 6601 | t=3.0: 0 · t=5.0: 0 · t=6.0: 0 · t=6.2: 73 · t=6.5: 1712 · t=6.8: 4991 |
| seal lattice, baseline chain | t=6.0: 0 · t=10.0: 0 · t=14.9: 0 |

### 8.3 The discarded scene, recorded because it is the reason for two constants

First attempt: `w_z = −0.03`, block `[4.80, 5.20]`, otherwise identical.

| arm | `n_seg` | iters | stop | arrival | clearance | certificate | figure-grade |
|---|---|---|---|---|---|---|---|
| constrained | 8 | 47 | trust_collapse | 9.3078 | −0.1643 | 1.3355 | no |
| **baseline** | 8 | 19 | **stationary** | 11.9939 | **+0.00029** | 0.0 | **yes** |

The baseline's widened window lets its bodies keep sinking for 15.0 instead of 5.0, reaching
`z = 4.55`; it waited and flew **over** the sunk wake, threading at +0.00029. The pair was destroyed
by the sink rate, not by the lifetime. That is why `w_z = −0.015` and the block is `[4.85, 5.15]`
(§3.4), and why §6.8 exists.

### 8.4 What this section does and does not establish

It establishes that the scene solves under near-default knobs, that the one-knob pair separates
cleanly, and that the graft and ablation instruments read the way the claim needs. It establishes
**nothing** for the paper: one seed, one machine, a dirty tree, no repeat runs, no comparison to any
external planner, and a solver that differs from the one every recorded measurement in `SOLVER.md`
was taken on. Every number here must be re-measured by the bench runner
(`DECISIONS.md` build list item 2) on a clean tree with a recorded extension build before it appears
anywhere a reader can see it.

---

## 9. Risks and open questions the implementer must not decide alone

1. **Units (`DECISIONS.md` §Forks, still OPEN).** This spec recommends the nondimensional box plus a
   published scale map, and shows the map produces sane metres (§2.1). If the PI or the venue wants
   mission units in the scenario itself, the trust radius and elastic weight must be re-measured the
   way `loiter`'s were — that is a measurement pass, not an edit.
2. **The corridor is what makes timing the only escape, and it is a modelling choice.** A reviewer
   can say the airway and level block were drawn to force the answer. The honest replies are §6.7
   (collapsing the corridor to the axis changes the arrival by 0.7%) and §6.4's reserve variant (the
   detour bound without the lateral block). **Which of these ships is a paper-argument decision, not
   an implementation one.**
3. **The vertical block is tight, ±0.15, and it is derived from the pinch geometry and the sink
   rate.** Change `r`, `ds`, `tau` or `w_z` and it must be re-derived and §6.1 re-run. Three
   constants are coupled; none may be tuned alone.
4. **The 3D-ness is not load-bearing and must not be claimed as such.** The descent is real and
   physically scaled, but §6.8 exists precisely because the descent must *not* be what opens the
   gate. A 2D cut of this scene would wait in the same way. The load-bearing-`z` evidence is
   `fence3d`'s (`scenarios.py:161`), and WAKE must not borrow it.
5. **The wake is truncated at `s_max = 2.75`** — 8.8 km of an 16 km wake. It is defensible (the
   omitted part is far outside the corridor and cannot bind) but it is unverified. **Recommended
   guard:** one run with `s_max` doubled, asserting the arrival is unchanged to 1e-6. Cheap, and it
   converts an assumption into a measurement.
6. **`n_seg = 16` is required and was found, not derived.** The 8-segment failure (§8.1) may be a
   basin problem rather than a resolution problem — the same open question as `curve` N8_seg4
   (`SOLVER.md`, `WORKSTREAM.md` `solver.multistart`). Do not present 16 as a property of the
   formulation.
7. **The E1 clip change is uncommitted and unmeasured against the table.** Everything in §8 is on
   it. If E1 is reverted or lands, §8 is void and must be re-run.
8. **Where WAKE sits in the demo package.** `DECISIONS.md` sequences B → C → A with a stop rule on
   A and calls WAKE "a cheap fourth demo". It is cheap — 13 obstacles, 1.3 s per solve — and it is
   the only demo whose baseline is *infeasible* rather than *worse*, which is the strongest form of
   the structural-failure argument the package needs. **Whether that promotes it ahead of A is the
   student's and the PI's call, not the implementer's.**
9. **Naming.** This spec calls the scenario `wake`. If it is registered, `SCENARIO_MAP`,
   `viewer.py`'s four-coordinate filter and the figure tool all need the key added; the viewer would
   otherwise plot `z` as if it were time (`scenarios.py:526` block comment).
