# Demo C — the `traffic` family, the bench runner, and the table generator

**Build specification.** Written to be implemented without asking a question, and to be reviewed
against. It contains no production code and authorises none of it: the implementer writes
`spacetime_bezier/families.py`, `tools/bench.py`, `tools/make_tables.py` and their tests, and
nothing else.

**Status of the inputs.** `spacetime_bezier/families.py`, `tests/unit/test_families.py` and
`tests/integration/test_families_rows.py` exist in the working tree, uncommitted and untested at the
time this spec was written. §1.1 says, part by part, what is kept, amended or replaced.

**Provenance of every number quoted below as *measured* — read this before using any of them.**

All measurements in this document were taken 2026-09-09 between roughly 14:37 and 14:41 local, in
this worktree at git `5b3ef78` (dirty: `paper/ksas_2026_fall/poster/*` modified, `paper/journal_1/`,
`spacetime_bezier/families.py`, `tests/unit/test_families.py`,
`tests/integration/test_families_rows.py` untracked), Python 3.14, arm64 Darwin 25.4.0, with
throwaway scripts in the scratchpad. Each is reproducible from the configuration stated beside it.

**They were taken against a solver that is no longer installed.** Every probe printed the extension
mtime on every run and it read `2026-09-08T20:22:09` each time. At **2026-09-09T14:42:43** — after
the last of them — another session rebuilt the extension in place, carrying an uncommitted change
labelled `E1 (2026-09-09)` in `rust_optimizer/core/src/spacetime_obstacle.rs:396-407`: with
`sound_clip` on, the clipping radius is now the **minimal sound radius** `floor.min(r_max)` instead
of `r_nearest` clamped into `[floor.min(r_max), r_max]`. The `sound_clip=False` arm keeps the old
rule.

So there are **three different solvers** in play and no two of their numbers may be compared:

| build | extension mtime | what it produced |
|---|---|---|
| the `SOLVER.md` §Measurements pass | 2026-09-08T19:18:23, branch `solver/dual-elastic-weight` | the 28-configuration table |
| pre-E1 | 2026-09-08T20:22:09 | **every number in this document** |
| post-E1 (installed now) | 2026-09-09T14:42:43 | nothing yet |

Consequences, and they are not cosmetic:

- **The wall and row counts in §1.3 are `clip_band` outputs and E1 changes `clip_band` directly.**
  They will move. They are quoted here to establish the *shape* of the axis — linear growth, well
  inside the `segments × obstacles` bound, `rows == walls × (N + 1)` — not the values.
- **The timings in §1.4, §1.7.3 and §2.6 are pre-E1** and are not comparable to the 28-configuration
  table either, which is a third build on another branch. They are quoted to size the bench (a solve
  is seconds, not minutes) and to establish that repeats measure the machine, not the solver.
- **§2.8's backbone spot-check invariant is expected to be RED on the installed build.** That is the
  test doing its job: it pins three configurations against constants lifted from a pre-E1 table, and
  a solver change is exactly what it exists to catch. Implement it anyway; §2.8 says what to do when
  it goes red.
- **`frontend.extension_build_time()` reads `bezier_opt.__file__`, which is the package's
  `__init__.py`, not the compiled `.so`** (`frontend.py:178-188`). Those are two files and only one
  of them is the solver. §2.4 therefore requires the bench to record **both** mtimes.

**Nothing here is checked in as a repository measurement**, and no number in this document may be
quoted in the manuscript. The whole point of Parts 2 and 3 is that the repository stops needing
hand-carried numbers like these — and this rebuild, landing mid-spec and invalidating a table
nobody can regenerate in one command, is the argument for building them.

---

## 0. What this demo has to carry, and what it must not quietly become

From `DECISIONS.md`: demo C carries **the CLAIM itself** — a convexification whose problem size is
linear in (segments × obstacles), with no discrete structure — and it doubles as the scaling
experiment. Its axis is Osburn's Table I axis (obstacle count) with the half a space-only
decomposition cannot do: every obstacle moves.

Three ways this demo can stop being that demo, and the guard against each:

1. **A vacuous count axis.** An obstacle the trajectory never approaches produces no wall (the clip
   ball drops it), so "20 aircraft" would cost the same as "4 aircraft plus 16 decorations".
   Guarded by §1.3's placement rule and the `FAILS IF` tests in §1.8.
2. **A confounded axis.** The horizon knob moving the geometry, or the 3+t member solving a
   different conflict schedule from its 2+t twin. Guarded by §1.5, §1.6 and their tests.
3. **A curve with no time on it.** The paper currently has **no wall-clock number at all**
   (`DECISIONS.md` "DO NOT HAVE"; `idea/spacetime.md` §"Benchmark comparison" — "Our side of this
   table is deliberately empty"). Part 2 exists to produce it, and §2.6's timing protocol is the
   part a reviewer will attack.

---

## 1. The four axes

### 1.1 Verdict on the existing draft, part by part

| Part of `families.py` (draft) | Verdict | Why |
|---|---|---|
| `SCENE`/`CORRIDOR_Y`/`X_START`/`X_END`/`DEFAULT_T` constants | **Keep** | The nondimensional box is the right unit choice — §1.2. |
| `_endpoints`, `_nominal` | **Keep** | Correct and minimal. |
| `traffic_crossings` crossing-point rule (stratified `s`, crossing point and time pinned to the nominal plan, `pos0 = x_cross − vel·t_cross`) | **Keep** | This is what makes the count axis non-vacuous, and it is measured to work — §1.3. |
| `traffic_crossings` T-independence (`pos0` invariant in `T`, `vel ∝ 1/T`) | **Keep** | This is what makes the horizon axis clean; verified algebraically in §1.5. |
| Draw order (`theta`, `phi`, `speed`, `radius` in a fixed order whatever the dimension) | **Keep** | This is what makes the dimension axis a lift rather than a re-roll — §1.6. |
| `family_motion`'s three kinds and the four-quarter circle chain | **Keep** | Degree 1 / cubic / chain-of-cubics is exactly the three tube shapes the certificate should be measured against. |
| `@_canonical` on the generators | **Keep** | It is the `SCENARIO_MAP` contract (`scenarios.py:24`). |
| Scenario `name` fields (`traffic_n08`, `dim_3d`, `horizon_T20`) | **Amend** | They do not carry the seed, so two seeds of one grid point collide in the bench's output. §1.7 fixes the naming rule. |
| `cross_s` carried on the obstacle dict | **Amend** | `_canonical` → `normalize_obstacle` keeps only `control_points`, `radius`, `name`, `color` (`geometry.py:118-122`), so `cross_s` is **silently dropped** from the returned scenario. Move the schedule into the `family` key — §1.7. |
| `FAMILY_MAP` as `name -> (generator, values, config)` and `family_members(name)` yielding `(value, scenario)` | **Amend** | No place for seeds, arms or per-family solve knobs, and `optimize_scenario` reads NONE of the solve knobs off the scenario dict (§1.7.2). Replaced by `FamilySpec` — §1.7.3. |
| `DEFAULT_CONFIG = (8, 8)` as a module constant | **Amend** | Keep the value; move it into `FamilySpec.configs` so a robustness arm is a flag, not an edit. |
| A baseline arm | **Missing — add** | `DECISIONS.md`: "Every demo is a PAIR." The draft has no baseline at all. §1.9 specifies one. |

Nothing in the draft is replaced wholesale. Its geometry is sound; what it lacks is provenance
plumbing and a pair.

### 1.2 FORK 1 — units. **Recommendation: nondimensional 10-unit box.**

Keep the draft's scene: a 10×10 (or 10×10×10) box, corridor on `y = 5` (`z = 5` in 3+t), start
`x = 0.5`, end `x = 9.5`, horizon `T = 10`.

The argument is `scenario_loiter`'s docstring, which is the repository's own record of what leaving
this box cost (`scenarios.py:363-463`):

- **The trust radius is a length.** `clip_band` builds the clip radius as
  `reach = seg_radius + trust·sqrt(dim)`, so the stock `0.5` against a 200 m scene is degenerate;
  `loiter` had to introduce a `trust_radius` scenario key (`scenarios.py:522`) and every caller had
  to learn to forward it (`optimize.py:747-752`, `make_paper_figure.py:66-69`).
- **The elastic weight is problem-scaled.** `loiter` needed a `SCENARIO_ELASTIC_WEIGHT` entry at
  `1e5` with a measured ladder recorded beside it (`scenarios.py:296-309`) before it certified.
- **`min_dt` is the one that survives a rescale** (`scenarios.py:451-452`) — and it survives because
  it is a time, not a length.

So a mission-units family would open three tuning questions per grid point, on the axis whose whole
purpose is that only ONE thing changes. **Measured against that (pre-E1, extension mtime
2026-09-08T20:22:09):** the draft family, in the nondimensional box, at the stock knobs (trust `0.5`, `min_dt 0.1`, elastic start `100` escalating,
degree 8 / 8 segments, arrival pinned), is figure-grade at **6 of the 7 count grid points**
(n = 1, 2, 4, 8, 12, 20 pass; n = 16 at seed 0 does not — §1.3). Zero scenario-specific knobs were
needed.

**Mission units are a caption, not a scene.** The family carries a `scale` record —
`{"length_unit": "m", "length_per_unit": 200.0, "time_unit": "s", "time_per_unit": 1.0}` — inside
`sc["family"]`, defaulting to `1.0`/`1.0`. Figures and table captions multiply by it; **the solver
never sees it**, and §3 requires the axis label to be built from that record rather than typed.

- **FAILS IF** any family member sets a `trust_radius` or `coord_bounds` scenario key, or appears in
  `SCENARIO_ELASTIC_WEIGHT`. Any of those means the nondimensional claim of this section has quietly
  stopped holding and the family has become a second `loiter`.

### 1.3 FORK 3 — guaranteed conflict. **Recommendation: guaranteed conflict is the default mode, and `realistic` exists beside it as a named second mode.**

`traffic_crossings` keeps its rule: every aircraft is placed so that its centre is **at the nominal
plan's position at the nominal plan's time**. The crossing fraction, angle, speed ratio and radius
are drawn; the crossing point and time are not.

**Why, measured — PRE-E1 (extension mtime 2026-09-08T20:22:09); E1 changes `clip_band`, so these
counts will move and must be re-taken. The shape is the claim here, not the values.** The wall count
at the straight seed (degree 8, 8 segments, `trust_radius=0.5`, `sound_clip=True`,
`spacetime_koz_rows_exact`, one wall per distinct `(segment, obstacle, component)`):

| aircraft `n` | 1 | 2 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|
| walls | 5 | 7 | 17 | 30 | 47 | 60 | 71 |
| bound `segments × obstacles` | 8 | 16 | 32 | 64 | 96 | 128 | 160 |
| exact rows | 45 | 63 | 153 | 270 | 423 | 540 | 639 |

Two things to read off it. First, the growth is linear and the bound holds with room to spare —
about `3.5n + 3` walls against a bound of `8n` — because the clip ball reaches only obstacles near
the segment. Second, and this is the fork: **that gap is exactly what a `realistic` placement would
widen without bound.** With obstacles placed at random the realized wall count would measure how
many aircraft happened to fall near the straight seed, which is a property of the seed, not of `n`.
The claim under test is "size is linear in (segments × obstacles)"; the only honest way to measure
it is to make every obstacle eligible to contribute and then report the realized count beside the
nominal bound, which is what the table in §3.3 does.

`rows == walls × (N + 1)` holds exactly in the table above (9 control points per segment at degree
8). That is the linear-size claim in its most literal form and §1.8 pins it.

**`mode="realistic"` is specified, not deferred.** Same signature, same draws, one difference: the
crossing fraction `s` is drawn in `(-0.4, 1.4)` instead of being stratified over `(0.15, 0.85)`, so
an aircraft whose `s` falls outside `(0, 1)` crosses the corridor axis before the plan gets there or
after it has gone, and never conflicts. It exists so the paper can answer "is this an artificial
worst case?" with one supporting panel and one sentence, and the bench treats it as an arm
(§2.2). It is **not** on the scaling curve.

**What the paper's scaling figure looks like under Forks 1–3 (all three together):** one panel.
Horizontal axis, aircraft count `n ∈ {1, 2, 4, 8, 12, 16, 20}`, linear. Left vertical axis, median
solve wall-clock in seconds over 5 seeds, with a min–max band. Right vertical axis, median realized
wall count, with the `segments × obstacles` bound as a dashed line above it — the two curves
together are the claim ("time grows like the row count, and the row count is linear and below its
bound"). A grid point that is not figure-grade at every seed is drawn with open markers and counted
in the caption; nothing is dropped. Osburn's three Table I points (0.29 s / 0.52 s / 4.12 s) appear
as isolated markers with a hatched legend entry reading *different machine, different problem,*
never as a connected series. The free-arrival arm (§1.4) appears as a second, thinner series at
`n ∈ {4, 8, 20}` only.

### 1.4 FORK 2 — arrival mode. **Recommendation: the scaling axis is PINNED arrival; free arrival is a separate arm on a three-point subgrid.**

**What a freed arrival requires, in code.** `optimize.py:377-390`,
`check_free_arrival_is_costed`, raises `DegenerateFreeArrivalError` when `free_arrival_time=True`
and `time_weight == 0.0`: the smoothness regularizer is blind to the time coordinate, so with no
time penalty the arrival appears in no term of the objective and what comes back is an
interior-point tie-break on a flat face (the docstring at `optimize.py:302-322` records the
measured 10.08 → 12.24 swing driven by the trust radius alone). `optimize.py:260-275`,
`check_time_penalty_is_capped`, then raises `UncappedTimePenaltyError` when `time_weight > 0` with
no `v_max`. Both are called, in that order, at `optimize.py:453-455`. So **freeing the arrival on
this family adds two more knobs — `time_weight` and `v_max` — to the axis whose entire purpose is
that one thing moves.**

**And it is measured to be fragile here** (pre-E1, extension mtime 2026-09-08T20:22:09). Same
family, same configuration, the only difference being
`free_arrival_time=True, time_weight=10.0, v_max=1.5`:

| arm | n = 4 | n = 8 | n = 20 |
|---|---|---|---|
| pinned | conv, 20 it, clearance +0.1309, **figure-grade** | conv, 26 it, +0.1219, **figure-grade** | conv, 41 it, +0.0410, **figure-grade** |
| free | conv, 9 it, +0.1008, arrival 7.845, **figure-grade** | conv, 44 it, +0.1100, arrival 6.282, **figure-grade** | **trust_collapse**, 99 it, clearance −0.4307, certificate 7.24, arrival **13.619** |

The n = 20 free run put its arrival at 13.6 s, past the scene's horizon `T = 10`. That matters
structurally, not just numerically: `normalize_obstacle` clamps a legacy `{pos0, vel}` obstacle's
window to `[0, T]` (`geometry.py:137-139`), and `optimize_spacetime`'s upper clamp on a freed
arrival is `time_ub_scale·T = 15` (`optimize.py:507`). **So a freed arrival on this family opens a
degenerate escape — outwait the entire traffic picture, after which the scene is empty.** Pinning
the arrival closes it. (This is the reason the free arm's subgrid is small and its records are read
with `arrival_on_time_ub` and `arrival_time > T` in view; §2.4 records both.)

**And the backbone table says so directly.** `SOLVER.md` §Measurements: *"Defaults otherwise: speed
cap off, arrival pinned… A run that enables `v_max` / `time_weight` / `free_arrival_time` is a
different problem and must be re-measured."* The pinned arm is the arm that regenerates the existing
28-configuration table; a free-arrival scaling curve would share no row with it.

**This is not giving up consequence 2.** Timing-as-a-decision is carried by demo B (`loiter`, whose
paper figure already runs `time_weight=10, v_max=5, free_arrival_time=True` and whose retiming is
falsification-tested in `tests/integration/test_loiter_scenario.py:303-379`) and by demo A. Demo C
carries the CLAIM — linear size — and the pinned arm measures exactly that.

**Under this recommendation the scaling figure** is the panel described at the end of §1.3: a
pinned-arrival series across the full count grid, plus a thin free-arrival series at
`n ∈ {4, 8, 20}` whose n = 20 point is an open marker, captioned with what it costs and why — a
freed arrival on a scene whose obstacles expire at `T` can buy feasibility by waiting the scene out,
so the constraint that makes the timing meaningful is missing here and present in demos A and B.

### 1.5 Axis: obstacle count — `family_traffic`

**Signature.**

```python
def traffic_crossings(n_obs: int, dim_spatial: int = 2, T: float = DEFAULT_T,
                      seed: int = 0, mode: str = "guaranteed") -> list[dict]
```
```python
@_canonical
def family_traffic(n_obs: int, dim_spatial: int = 2, T: float = DEFAULT_T,
                   seed: int = 0, mode: str = "guaranteed") -> dict
```

`n_obs < 1` raises `ValueError`. `dim_spatial not in (2, 3)` raises `ValueError`. `mode` not in
`("guaranteed", "realistic")` raises `ValueError` naming both.

**Scene geometry.** Exactly the draft's.

- 2+t: `start = [0.5, 5.0, 0.0]`, `end = [9.5, 5.0, T]`.
- 3+t: `start = [0.5, 5.0, 5.0, 0.0]`, `end = [9.5, 5.0, 5.0, T]`.
- `init_curve = {"mode": "straight"}` — the nominal plan **is** the initial guess, which is what
  lets the placement rule guarantee a conflict at the seed.
- No `coord_bounds`, no `stations`, no `trust_radius` key. The uniform default box
  (`coord_lb=-20, coord_ub=20`, `optimize.py:428-429`) contains the scene with margin.

**Obstacle placement rule**, per aircraft `i` in `range(n_obs)`, with
`rng = numpy.random.default_rng(seed)` created once per call:

1. Crossing fraction. `guaranteed`: `s = 0.15 + 0.70·(i + rng.uniform(0.2, 0.8)) / n_obs`.
   `realistic`: `s = rng.uniform(-0.4, 1.4)`.
2. `t_cross = s·T`; `x_cross = (1 − s)·start_spatial + s·end_spatial`.
3. `theta = deg2rad(rng.uniform(30, 150)) · (+1 if i even else −1)` — the alternating sign puts
   consecutive conflicts on alternating sides of the corridor.
4. `phi = deg2rad(rng.uniform(-30, 30))` — **drawn in 2+t as well and then unused**, so the draw
   order does not depend on the dimension. §1.6 depends on this.
5. `speed = v_nom · rng.uniform(0.6, 1.4)` with `v_nom = ‖end − start‖_spatial / T`.
6. `radius = rng.uniform(0.35, 0.6)`.
7. Direction, unit-normalised: 2+t `cos θ·â + sin θ·p̂` with `p̂ = (−â_y, â_x)`; 3+t
   `(cos θ·â + sin θ·p̂)·cos φ + ẑ·sin φ` with `p̂ = (−â_y, â_x, 0)`.
8. `vel = speed · direction`; `pos0 = x_cross − vel·t_cross`.

Steps 3–6 must be drawn in that order and unconditionally, `phi` included.

**Randomness and seeding.** One `default_rng(seed)`, created inside the generator, consumed in the
order above. No global RNG anywhere in the module. Two calls with the same arguments return equal
dicts; two calls differing only in `seed` return different obstacles.

**Solver knobs that travel with it.** None as scenario keys (§1.2), and this is deliberate: the
scenario dict is the wrong carrier, because `optimize_scenario` reads only `obstacles`, `start`,
`end`, `init_curve`, `stations`, `coord_bounds`, `trust_radius`, `name`, `title`, `T`
(`optimize.py:731-752`, `optimize.py:915-922`) and would **silently ignore** an `elastic_weight` or
`v_max` key. Solve knobs live in `FamilySpec.solve` (§1.7.3), where the bench reads them.

### 1.6 Axis: spatial dimension — `family_dimension`

```python
def family_dimension(dim_spatial: int, n_obs: int = 8, seed: int = 0) -> dict
```

Delegates to `family_traffic(n_obs, dim_spatial=dim_spatial, T=DEFAULT_T, seed=seed)`, then
overwrites `name`, `title` and `family`. **Not decorated with `@_canonical`** — `family_traffic`
already applied it and the obstacles come back canonical; a second application would re-normalise
already-canonical control points for nothing. (It would in fact be harmless —
`normalize_obstacle` is idempotent on the canonical form, `geometry.py:112-122` — but a decorator
that does nothing is a decorator someone will later move.)

The axis is clean because of step 4 of §1.5: the same seed produces the same crossing fractions,
angles, speeds and radii in both dimensions, and the third dimension adds only the elevation the
`phi` draw was already holding.

### 1.7 Axis: horizon — `family_horizon`; axis: motion — `family_motion`; and the record contract

```python
def family_horizon(T: float, n_obs: int = 8, seed: int = 0) -> dict
```

Delegates to `family_traffic(n_obs, dim_spatial=2, T=T, seed=seed)`. The spatial picture is
`T`-invariant by construction and this is algebra, not luck: `speed ∝ v_nom ∝ 1/T` and
`t_cross = s·T`, so `vel·t_cross = speed·direction·s·T` has `speed·T = ‖axis‖·u` free of `T`, hence
`pos0` is `T`-invariant and `vel ∝ 1/T`. Grid: `T ∈ {2, 5, 10, 20}`.

```python
@_canonical
def family_motion(kind: str, T: float = DEFAULT_T) -> dict
```

One obstacle, radius `0.9`, crossing at mid-transit. `kind="straight"` is a degree-1 lifted tube
(convex — a plane touching it anywhere supports the whole tube, `scenarios.py:325-330`);
`kind="arc"` is a single cubic that goes out across the corridor and back (non-convex tube — the
case the construction exists for); `kind="circle"` is a chain of four cubic quarter-arcs on
contiguous time windows (`k = 0.5522847498`), so the tube RETURNS and one clip ball can cut the
centreline twice. No seed — this axis is deterministic. Unknown `kind` raises `ValueError` whose
message contains `unknown motion kind`.

#### 1.7.1 The scenario dict every member returns

Every key, and it must satisfy the `SCENARIO_MAP` contract as `optimize_scenario` reads it:

| key | type | required by | value |
|---|---|---|---|
| `name` | `str` | `optimize.py:915`, and it is the bench's join key | see §1.7.2 |
| `title` | `str` | `optimize.py:916` | human-readable, e.g. `"Scheduled Traffic, 8 Aircraft"` |
| `init_curve` | `dict` | `optimize.py:734`, `optimize.py:681` | `{"mode": "straight"}` on every member |
| `obstacles` | `list[dict]` | `optimize.py:731` | canonical: `control_points` (list of lists, last column time, non-decreasing), `radius > 0`, plus optional `name`, `color`. Produced by `@_canonical`. |
| `start`, `end` | `list[float]`, length `dim` | `optimize.py:732-733` | `start[-1] == 0.0`, `end[-1] == T` |
| `T` | `float` | `optimize.py:921` | horizon |
| `family` | `dict` | nothing downstream — provenance for §2 | §1.7.2 |

Absent on purpose: `stations`, `coord_bounds`, `trust_radius` (§1.2, §1.5).

#### 1.7.2 The `name` rule and the `family` record

**`name` must be unique across the entire grid of every family, including seeds**, because it is
what the bench writes into each record and what a human greps for. The rule:

- traffic: `f"traffic_{mode[:4]}_n{n_obs:02d}_d{dim_spatial}_T{T:g}_s{seed}"`
- dimension: `f"dim_d{dim_spatial}_n{n_obs:02d}_s{seed}"`
- horizon: `f"horizon_T{T:g}_n{n_obs:02d}_s{seed}"`
- motion: `f"motion_{kind}_T{T:g}"`

`family` is the provenance record and **carries the schedule that `@_canonical` drops**:

```python
{"axis": "obstacle_count",        # or "dimension" | "horizon" | "motion"
 "value": 8,                      # the knob value, the figure's x
 "n_obs": 8, "dim_spatial": 2, "T": 10.0, "seed": 0, "mode": "guaranteed",
 "schedule": [{"name": "AC01", "cross_s": 0.183, "radius": 0.42,
               "speed": 0.79, "theta_deg": 61.3, "phi_deg": -12.4}, ...],
 "scale": {"length_unit": "m", "length_per_unit": 1.0,
           "time_unit": "s", "time_per_unit": 1.0}}
```

`schedule` is empty for `motion`. `value` is the number the scaling figure plots on its x axis and
it is never re-derived downstream.

#### 1.7.3 `FamilySpec` and the registry

```python
@dataclass(frozen=True)
class FamilySpec:
    generator: Callable[..., dict]     # takes (value, seed=...) for seeded families, (value,) otherwise
    values: tuple                      # the knob grid
    seeds: tuple[int, ...]             # () for deterministic families
    configs: tuple[tuple[int, int], ...]   # (degree, segments) pairs
    solve: dict                        # knobs for optimize_spacetime, NOT scenario keys

FAMILY_MAP: dict[str, FamilySpec] = {
    "traffic":   FamilySpec(family_traffic,   (1, 2, 4, 8, 12, 16, 20), (0, 1, 2, 3, 4), ((8, 8),), BASE_SOLVE),
    "dimension": FamilySpec(family_dimension, (2, 3),                   (0, 1, 2, 3, 4), ((8, 8),), BASE_SOLVE),
    "horizon":   FamilySpec(family_horizon,   (2.0, 5.0, 10.0, 20.0),   (0, 1, 2, 3, 4), ((8, 8),), BASE_SOLVE),
    "motion":    FamilySpec(family_motion,    ("straight", "arc", "circle"), (),         ((8, 8),), BASE_SOLVE),
}

BASE_SOLVE = {"max_iter": 200, "tol": 1e-6, "scp_prox_weight": 0.3,
              "scp_trust_radius": 0.5, "elastic_weight": 100.0,
              "escalate_elastic_weight": True, "min_dt": 0.1, "sound_clip": True,
              "v_max": None, "time_weight": 0.0, "free_arrival_time": False}

def family_members(name: str) -> Iterator[tuple[Any, int, dict]]:
    """Yield (knob value, seed, scenario) over the family's grid × seeds.
    Deterministic families yield seed = -1."""
```

`BASE_SOLVE` mirrors `optimize_scenario`'s own defaults (`optimize.py:709-721`) and
`frontend.SOLVE_DEFAULTS` (`frontend.py:100-113`) so a family run and a registry run are the same
problem under the same knobs. **Five seeds, not one, and this is measured, not a preference** (pre-E1, extension mtime
2026-09-08T20:22:09; the specific grid point that fails may move under E1, the seed spread will
not): at
`n = 16`, degree 8 / 8 segments, pinned arrival, seed 0 does not converge (trust collapse at 65
iterations, escalated to `1e5`, clearance −0.1918, not figure-grade, 2.86 s) while seed 1 and seed 2
converge in 18 and 16 iterations at weight 100, clearance +0.0934 and +0.0747, figure-grade, in
0.85 s and 0.75 s. **A one-seed scaling curve would have reported a factor-of-four wall-clock
outlier and a hole at n = 16 as properties of the count axis.** They are properties of a seed.

### 1.8 Invariants a unit test must pin

`tests/unit/test_families.py` (no extension needed) and `tests/integration/test_families_rows.py`
(needs `bezier_opt`). The draft file already contains most of these; the list below is the
complete required set, with the four additions marked **NEW**.

- **FAILS IF** a member is missing any key of §1.7.1, or `len(end) != len(start)`, or
  `start[-1] != 0.0`, or `end[-1] != T`, or `T <= 0`, or any obstacle's `control_points` has a
  column count different from `len(start)`, or `radius <= 0`, or its first control point's time
  exceeds its last — i.e. the member could not be handed to `optimize_scenario` unchanged.
- **FAILS IF** any aircraft in any `guaranteed` traffic member has a closest approach to the
  straight uniform-speed nominal plan (sampled at ≥ 801 instants, obstacle position by bisection on
  the monotone time coordinate) that is **not** strictly less than its radius. Such an aircraft adds
  no wall, and the count axis would be counting decorations.
- **FAILS IF** two crossing fractions in `traffic_crossings(20)` land within 0.02 of each other, or
  any falls outside `[0.15, 0.85]` — twenty aircraft must be twenty encounters, not a clump.
- **NEW — FAILS IF** a `realistic` member's aircraft are *all* conflicts, or *none* are. That mode
  exists to be a mixture; if it degenerates in either direction it is no longer the comparison
  §1.3 claims it is. (Assert over `traffic_crossings(20, mode="realistic", seed=s)` for
  `s in range(5)` pooled, not per seed.)
- **FAILS IF** `family_traffic(8, seed=3) != family_traffic(8, seed=3)`, or
  `family_traffic(8, seed=3)["obstacles"] == family_traffic(8, seed=4)["obstacles"]` — the seed must
  be the only randomness and it must actually do something.
- **FAILS IF** for any `T ∈ {2, 5, 20}` some aircraft's `pos0` differs from its `T = 10` twin
  (atol 1e-12), or its `vel·T` differs from the twin's `vel·10` (atol 1e-12), or its radius or
  crossing fraction moved — the horizon knob must move the clock and nothing else.
- **FAILS IF** the 2+t and 3+t members of one seed differ in any crossing fraction, radius, or speed
  magnitude, or if the 3+t member's `start`/`end`/obstacle control points are not 4-dimensional —
  the dimension axis would be comparing two different conflict schedules.
- **FAILS IF** `motion` collapses: `straight` must have exactly one obstacle with 2 control points,
  `arc` exactly one with 4, `circle` exactly four with 4 each whose consecutive endpoints agree to
  1e-12 and whose last spatial endpoint returns to the first. Also **FAILS IF**
  `family_motion("helix")` does not raise `ValueError`.
- **FAILS IF** any `motion` kind's obstacle never overlaps the nominal plan at the same instant.
- **NEW — FAILS IF** any member's `name` collides with another member's across the whole
  `FAMILY_MAP` × seeds grid, or if `sc["family"]["schedule"]` is empty for a traffic member
  (`@_canonical` dropping the schedule again).
- **NEW — FAILS IF** any member carries a `trust_radius`, `coord_bounds` or `stations` key, or
  appears in `scenarios.SCENARIO_ELASTIC_WEIGHT` — the nondimensional claim of §1.2 has silently
  lapsed.
- **FAILS IF** (integration) a straight-tube traffic member's wall count at the straight seed
  exceeds `segments × obstacles`, or is zero, or does not grow monotonically with `n`, or the
  20-aircraft count is not more than twice the 1-aircraft count. Walls are the distinct
  `(segment_idx, obstacle_idx, component_idx)` triples returned by
  `bezier_opt.spacetime_koz_rows_exact(p=P0, obstacle_ctrl=…, obstacle_r=…, n_seg=…,
  trust_radius=0.5, sound_clip=True, stations=None)` — indices 2, 4, 5 of the returned tuple.
- **NEW — FAILS IF** the exact row count is not `walls × (N + 1)` for any traffic member. That
  identity IS the linear-size claim in its most literal form: one plane per wall, shared by every
  control point of its segment (`spacetime_koz_rows_exact.__doc__`), and a change to it is a change
  to the claim.
- **FAILS IF** (integration) a horizon or dimension member breaks the `segments × obstacles` bound
  at any `T` or any dimension.

Each of these can fail: every one of them was checked to be a real assertion against the built
extension while writing this spec, and the wall/row table in §1.3 is the data they are asserting
about.

### 1.9 The pair — demo C's baseline

`DECISIONS.md`: *every demo is a PAIR, and the baseline's failure must be STRUCTURAL.*

**Arm `snapshot`.** Same scenario, one difference: every obstacle is replaced by a **static** ball
at its position at `t = 0`, spanning the whole horizon. Concretely, for each canonical obstacle,
replace its control points by `[[p(0)…, 0.0], [p(0)…, T]]` with the same radius, where `p(0)` is
`obstacle_positions_at(cps, [0.0])[0]` (`geometry.py:172-183`). Solve; then **score the returned
trajectory against the TRUE moving obstacles** with `compute_min_clearance`.

Why this is structural and not a straw man: a space-only decomposition of free space exists only if
free space does not move. The standard remedy in that family is to decompose a snapshot. This arm
is that remedy, executed with our own solver so nothing else differs — which is `DECISIONS.md`
option (c), ablation against ourselves, and §5 records that whether it is allowed to stand as *the*
paper's baseline is not this spec's call.

Non-vacuity guards, copied in discipline from `tests/integration/test_loiter_scenario.py`:

- **FAILS IF** either half does not converge — then the comparison is about a broken run, not about
  the constraint.
- **FAILS IF** the snapshot half's clearance **against its own snapshot obstacles** is not > 0 —
  then it did not even solve the problem it was given, and its failure against the true obstacles is
  not attributable to the snapshot.
- **FAILS IF** the space-time half's clearance against the true obstacles is not > 0.
- **FAILS IF** the snapshot half's clearance against the TRUE obstacles is ≥ 0, or is negative by
  less than 0.05 in scene units — a graze is not a demonstration, and a snapshot that happens to
  work means this grid point proves nothing.
- **FAILS IF** the two halves return the same control points to 1e-9 — then the arms are not two
  arms.

---

## 2. `tools/bench.py`

One JSONL record per run. **The runner never prints a table and never rounds** — Part 3 does that.

### 2.1 CLI

```
tools/bench.py --out PATH.jsonl
               (--family NAME [--values V,V,...] [--seeds S,S,...]
                | --scenarios NAME,NAME,... [--all-configs])
               [--configs 8x8,8x16]        default: the family's / scenario's own list
               [--arms ours,clipfloor_off,snapshot,free_arrival,realistic]   default: ours
               [--repeats N]               default: 3
               [--timeout SECONDS]         default: none
               [--resume]                  skip identity keys already in --out
               [--dry-run]                 print the plan (one line per planned run) and exit 0
               [--limit N]                 stop after N runs; for smoke tests
```

`--family` and `--scenarios` are mutually exclusive and one is required. In `--scenarios` mode the
generator and configs come from `scenarios.SCENARIO_MAP` and the starting elastic weight from
`scenarios.scenario_elastic_weight(name)` — that is what makes §2.7's backbone invariant possible.
In `--family` mode they come from `FamilySpec`. `--dry-run` must enumerate the exact same run list
the real pass would, so a plan can be reviewed before hours are spent.

Exit status: `0` if every planned run produced a record (figure-grade or not); `1` if any run
produced no record at all. **A run that fails is a record, not a missing row** (§2.5).

### 2.2 The arms

| arm | what changes | why it exists |
|---|---|---|
| `ours` | nothing; `BASE_SOLVE` / the scenario's registered weight | the measurement |
| `clipfloor_off` | `sound_clip=False` | regenerates `SOLVER.md` §Measurements' "Clearance OFF" / "Unsound walls" / "Grade OFF" columns — the existing ablation |
| `snapshot` | obstacles frozen at `t = 0` (§1.9); clearance re-scored against the true obstacles | demo C's pair |
| `free_arrival` | `free_arrival_time=True, time_weight=10.0, v_max=1.5` | Fork 2's second series; refused by `optimize.py:377-390` / `:260-275` if either knob is dropped, which is the intended behaviour |
| `realistic` | `mode="realistic"` in the generator | Fork 3's supporting panel |

An arm is a field on the record, never a separate file.

### 2.3 How one run is executed

The bench **builds the result row itself** rather than calling `optimize_scenario`, and this is a
deliberate copy with a stated reason — the same trade `frontend.py:549-566` documents for
`_run_ladder`. Two reasons: `optimize_scenario` runs `compute_min_clearance` inside the region a
naive timer would enclose (`optimize.py:806-808`), which is Python post-processing and not solver
time; and it does not expose `escalate_elastic_weight`, which the ablation arms need
(`optimize_spacetime` does — `optimize.py:657`). The copy is paid for by the parity test in §2.7.

Order, exactly:

1. Build the scenario from the generator (or `SCENARIO_MAP`).
2. Warm-up: on the first run of the process only, solve that configuration once and discard the
   result (§2.6).
3. For `r` in `range(repeats)`: `t0 = perf_counter(); c0 = process_time()` →
   `optimize_spacetime(...)` → record `perf_counter() - t0` and `process_time() - c0`. Keep the
   `(P_opt, info)` of the **last** repeat; the solve is deterministic (§2.6) so which one is kept
   is arbitrary and the record says it is the last.
4. `min_clearance_3000 = compute_min_clearance(P_opt, obstacles, dim, n_eval=3000)` — matches
   `optimize_scenario` (`optimize.py:807`).
5. `min_clearance = compute_min_clearance(P_opt, obstacles, dim, n_eval=20001)` — matches the paper
   figure and the frontend (`make_paper_figure.py:196`, `frontend.py:121`). **This is the value the
   gate row is fed.**
6. Walls and rows at the **returned** iterate, via `spacetime_koz_rows_exact` at the run's own
   `trust_radius` and `sound_clip` (§2.4, `walls_returned` / `rows_returned`), and at the **initial
   guess** (`walls_seed` / `rows_seed`).
7. Build the gate row with the key mapping of `optimize.py:846-889` — never a reimplementation —
   and call `figure_grade_failures(row)`.
8. Emit one JSON object, one line, `json.dumps(..., allow_nan=False)` with non-finite floats
   written as `null` and a sibling `*_nonfinite` string (`"nan"` / `"inf"` / `"-inf"`), because JSON
   has no NaN and a silently-dropped NaN inverts the gate's polarity everywhere downstream.
9. `flush()` and `os.fsync()` after every record, so a killed pass keeps everything it finished.

### 2.4 The record

Every field, and where it comes from. `info` is `optimize_spacetime`'s second return value.

**Identity and provenance**

| field | source |
|---|---|
| `record_version` | literal `1`; bumped when this table changes |
| `identity` | sha256 hex of the canonical JSON of §2.8's key |
| `timestamp` | `datetime.now(timezone.utc).isoformat()` at record write |
| `git_commit` | `frontend.git_provenance()["git_commit"]` (`frontend.py:210`) |
| `git_dirty` | `frontend.git_provenance()["git_dirty"]` (`frontend.py:211`) |
| `extension_path` | `frontend.provenance_payload()["extension_path"]` (`frontend.py:225`) |
| `extension_build_time` | `frontend.extension_build_time()` — mtime of `bezier_opt.__file__` (`frontend.py:170-188`) |
| `extension_so_path`, `extension_so_build_time` | path and mtime of the **compiled** module: the first `*.so` / `*.pyd` in `Path(bezier_opt.__file__).parent`. `frontend.extension_build_time()` reads the package's `__init__.py`, which is not the solver; a rebuild that touches only the compiled artifact would leave that field stale. Both are recorded and §3.4 refuses on a disagreement across a table. |
| `newest_rust_source_time` | `frontend.newest_rust_source_time()` (`frontend.py:191-197`) |
| `extension_stale` | `frontend.provenance_payload()["extension_stale"]` (`frontend.py:230-232`); `true` means the record describes a build older than the sources and §3 must say so |
| `python_version` | `platform.python_version()` |
| `machine` | `platform.platform()` + `sysctl -n machdep.cpu.brand_string` on Darwin, else `platform.processor()`; the field is the joined string, `null` if unavailable |
| `cpu_count` | `os.cpu_count()` |
| `load_avg_before`, `load_avg_after` | `os.getloadavg()[0]` before and after the repeats (§2.6) |
| `hostname` | `socket.gethostname()` |

**What was run**

| field | source |
|---|---|
| `mode` | `"family"` or `"scenario"` |
| `family`, `axis`, `axis_value`, `seed`, `n_obs`, `dim_spatial`, `T`, `placement_mode`, `schedule`, `scale` | `sc["family"]` (§1.7.2); `null` in scenario mode |
| `scenario_name` | `sc["name"]` |
| `arm` | the arm (§2.2) |
| `N`, `n_seg` | the configuration |
| `dim` | `len(sc["start"])` |
| `n_obstacles` | `len(sc["obstacles"])` |
| `initial_elastic_weight`, `max_iter`, `tol`, `scp_prox_weight`, `trust_radius`, `min_dt`, `sound_clip`, `escalate_elastic_weight`, `v_max`, `time_weight`, `free_arrival_time`, `time_ub_scale`, `coord_lb`, `coord_ub` | the **resolved** arguments actually passed to `optimize_spacetime`, never the defaults they came from |
| `has_stations`, `has_coord_bounds` | `sc.get("stations") is not None`, `sc.get("coord_bounds") is not None` |

**Timing**

| field | source |
|---|---|
| `repeats` | `--repeats` |
| `solve_seconds` | **median** of the per-repeat `perf_counter` deltas |
| `solve_seconds_min`, `solve_seconds_max`, `solve_seconds_all` | the same deltas |
| `solve_cpu_seconds` | median of the per-repeat `process_time` deltas |
| `clearance_seconds` | `perf_counter` around step 5 alone; a Python cost, reported separately so it can never be mistaken for solver time |
| `rows_seconds` | `perf_counter` around step 6 |
| `warmup_seconds` | the discarded warm-up solve, or `null` |

**Solver result** — every field from `info`, named by its `info` key

| field | `info` key |
|---|---|
| `converged` | `converged` |
| `stop_reason`, `stop_label` | `stop_reason`, via `optimize._STOP_REASONS` |
| `iterations` | `iterations` |
| `accept_count`, `reject_count`, `null_step_count`, `bootstrap_count` | same names |
| `certificate_violation` | `koz_violation_reference` |
| `occlusion_violation` | `occlusion_violation_reference` |
| `occlusion_planes_dropped` | `occlusion_planes_dropped` |
| `koz_unsound_clips` | `koz_unsound_clips` |
| `sound_clip_reported` | `sound_clip` |
| `total_slack` | `total_koz_slack_returned` |
| `total_slack_last_subproblem` | `total_koz_slack` |
| `elastic_weight` | `final_elastic_weight` — **absent ⇒ NaN, never the start weight** (`optimize.py:800-805`) |
| `weight_raises` | `weight_raises`, `−1` if absent (`optimize.py:869-871`) |
| `max_koz_dual` | `max_koz_dual` |
| `speed_cap_violation` | `speed_cap_violation` |
| `arrival_time` | `arrival_time` |
| `arrival_on_min_dt_floor` | `arrival_on_min_dt_floor` |
| `time_ub_used`, `arrival_on_time_ub` | same names |
| `final_delta_norm`, `final_trust`, `rho_mean`, `rho_min`, `rho_max`, `rho_samples` | same names |
| `returned_best_iterate` | `returned_best_iterate` |
| `backend` | `backend` |
| `solver_min_clearance` | `min_clearance` — the solver's own, kept only so it can be compared with the independent one |

**Independent oracles** (Python, against the true geometry — never the solver's hulls)

| field | oracle |
|---|---|
| `min_clearance` | `geometry.compute_min_clearance(P_opt, obstacles, dim, n_eval=20001)` |
| `min_clearance_3000` | the same at `n_eval=3000` |
| `clearance_sampling_delta` | `min_clearance_3000 − min_clearance` |
| `min_clearance_vs_true` | for arm `snapshot` only: the same at 20001 against the **true** obstacles; `null` otherwise |
| `walls_seed`, `rows_seed` | distinct `(seg, obs, comp)` triples / row count from `spacetime_koz_rows_exact` at the straight initial guess |
| `walls_returned`, `rows_returned` | the same at the returned control points |
| `walls_bound` | `n_seg × n_obstacles` — the claim's nominal bound |
| `control_points` | `P_opt.tolist()` — so a figure can be redrawn from the run that was measured, the way `make_paper_figure.py:502-506` does |

**Verdict**

| field | source |
|---|---|
| `figure_grade` | `not figure_grade_failures(gate_row)` |
| `figure_grade_reasons` | `figure_grade_failures(gate_row)` verbatim |
| `gate_row` | the dict passed to the gate, echoed, so a reviewer can re-run the predicate on the record alone |

**Failure**

| field | source |
|---|---|
| `error_type` | exception class name, or `"timeout"`; `null` on success |
| `error_message` | `str(exc)`, truncated to 2000 chars |
| `error_traceback` | last 4000 chars of `traceback.format_exc()` |

### 2.5 Failure handling

The solve, the clearance oracles and the row oracle are wrapped together. On `Exception` (never
`BaseException` — `KeyboardInterrupt` must still stop the pass): every measured field is `null` with
its `*_nonfinite` sibling absent, `figure_grade` is `false`, `figure_grade_reasons` is
`["run raised <ErrorType>: <first line of message>"]`, `error_*` are filled, and **the record is
written**. The runner continues to the next run.

With `--timeout S`, each run executes in a `multiprocessing` child started with the `spawn` method,
because the Rust loop holds the GIL and cannot be interrupted in-process — the constraint
`frontend.py` names at its `solve_via_subprocess` (`frontend.py:1178-1183`). On expiry the child is
killed and a record is written with `error_type: "timeout"` and `timeout_seconds: S`. Without
`--timeout` the run is in-process and bounded by `max_iter`; every solve measured for this spec
finished in under 5 s, so the default is no timeout.

### 2.6 Timing protocol

**This is the number the paper does not have at all**, so the protocol is stated in full and its
weaknesses are named rather than smoothed.

- **What is timed.** `optimize_spacetime` and nothing else. The clearance oracles (`n_eval=20001` is
  a Python sampling with per-obstacle injected samples, `geometry.py:209-247`) and the row oracle are
  timed separately into `clearance_seconds` and `rows_seconds` and are never added to
  `solve_seconds`. Scenario construction, JSON writing and provenance collection are outside every
  timer.
- **Clock.** `time.perf_counter()` for wall, `time.process_time()` for CPU. Both are recorded so a
  machine under load is visible in the record itself: wall ≫ CPU means the number is about the
  machine.
- **Warm-up.** One discarded solve of the first planned configuration, per process. **Measured to be
  insurance rather than a correction** (pre-E1, extension mtime 2026-09-08T20:22:09): six
  consecutive solves of the traffic family at `n = 8`,
  degree 8 / 8 segments, pinned arrival, gave 0.6810, 0.6754, 0.6724, 0.6762, 0.6755, 0.6810 s — the
  first is not an outlier and the whole spread is 1.3 % of the minimum. `warmup_seconds` is recorded
  so this claim stays checkable rather than becoming folklore.
- **Repeats and the statistic. `solve_seconds` is the MEDIAN**, with `min`, `max` and the full list
  beside it. The reason to prefer median over min: the solve is **deterministic** — in the six-repeat
  run above every repeat returned 26 iterations and clearance 0.121926, so repeats measure the
  machine, not the solver; the minimum is the machine's luckiest moment and the median with the
  observed spread beside it is what survives a reviewer asking how it was taken. Three repeats is
  the default because the observed spread is ~1 %; if a future pass sees `solve_seconds_max /
  solve_seconds_min > 1.2` on any record, §3 flags that record and the pass should be re-taken.
- **Idle machine.** `load_avg_before` / `load_avg_after` are recorded, and the runner prints a
  banner at start naming the load and telling the operator to close other work. This is a **recorded
  fact, not a gate** — the runner never refuses on load, because a refusal on a number the operator
  cannot control produces a habit of `--force`. §3.4 renders a warning marker for any record whose
  load exceeded 1.0.
- **Comparability to Osburn.** `idea/spacetime.md` §"Benchmark comparison" fixes the terms: Osburn
  ran Python + CVXPY + Clarabel on an AMD Ryzen 7 9700X; we run Rust + Clarabel on Apple silicon.
  Same convex solver, different language and machine. Every table and figure that puts the two side
  by side must carry both machines in the caption, drawn from the `machine` field and from the text
  in `idea/spacetime.md` — never as a connected series (§1.3).

### 2.7 Resumability

The **identity key** is the canonical JSON (sorted keys, no whitespace) of:

```
{mode, family, scenario_name, axis, axis_value, seed, arm, N, n_seg,
 initial_elastic_weight, max_iter, tol, scp_prox_weight, trust_radius, min_dt,
 sound_clip, escalate_elastic_weight, v_max, time_weight, free_arrival_time,
 repeats, record_version, git_commit, extension_build_time}
```

`identity` is its sha256. With `--resume` the runner reads `--out`, collects every parseable
record's `identity`, and skips a planned run whose key is present. Unparseable lines are counted and
reported, never silently skipped.

**`git_commit` and `extension_build_time` are inside the key on purpose.** A resume after a rebuild
or a commit therefore re-runs everything, which is the correct polarity: a record produced by a
different build is not a record of this build, and a resume that reused it would produce a table
mixing two solvers. Reviewers should check this specific inclusion — dropping it is the obvious
"optimisation" and it is the one that silently corrupts a pass.

### 2.8 Invariants a test must pin

`tests/integration/test_bench.py`.

- **FAILS IF** the bench's row disagrees with `optimize_scenario`'s row for the same scenario and
  configuration. Concretely: run `optimize_scenario(scenario_original(), [(8, 4)], verbose=False)`
  and `bench.run_one(...)` on the same scenario/config/knobs, and assert equality of
  `control_points` (atol 0), `iterations`, `converged`, `stop_reason`, `certificate_violation`,
  `total_slack`, `elastic_weight`, `weight_raises`, `koz_unsound_clips`, and of the bench's
  `min_clearance_3000` against `optimize_scenario`'s `min_clearance` (atol 0). This is the price of
  §2.3's copy. *What would make it fail:* the bench forwarding a different `scp_prox_weight`,
  `trust_radius` or `time_ub_scale`, or reading `final_elastic_weight` with a non-NaN default. It
  cannot pass vacuously: it compares fifteen fields including a full control polygon bit-for-bit.
- **FAILS IF** `min_clearance` (20001 samples) exceeds `min_clearance_3000` by more than 1e-6 on any
  record of the pass. Denser sampling of a sufficient-condition detector must not report a
  materially *larger* minimum; if it does, the two samplings disagree about one curve, and that
  disagreement is the finding, not a tolerance to widen. (Stated as ≤ 1e-6 and not as an exact
  inequality because the two grids are `linspace(0,1,3000)` and `linspace(0,1,20001)` and neither is
  a subset of the other, plus per-obstacle injected samples — `geometry.py:209-247`.)
- **FAILS IF** the runner regenerates `SOLVER.md` §Measurements' spot-checked rows to anything other
  than the printed digit. The spot-check set, run in `--scenarios` mode with `--arms ours` at the
  registered configuration and the registered starting weight: **`original` N8_seg4**,
  **`fence3d` N8_seg2**, **`door3d` N8_seg4**. The three were chosen because all three converge in
  ≤ 13 iterations with zero weight raises, so the check costs a few seconds. The expected values are
  checked into the test file as constants **lifted verbatim from the table**, with the table's own
  provenance paragraph quoted beside them. Comparison: `min_clearance` to atol 5e-5 (the table
  prints four decimals), `iterations` / `weight_raises` / `elastic_weight` / `stop_label` /
  `figure_grade` **exactly**. *What would make it fail:* any solver change that moves those three
  configurations; the bench forwarding a knob the table's pass did not use (a leaked `time_weight`,
  a `sound_clip=False`); a rebuild of the extension that changes the answers. **This test is expected to be RED on the
  build installed as of 2026-09-09T14:42:43**, because the `E1` change to `clip_band`
  (`spacetime_obstacle.rs:396-407`) is exactly the kind of change it exists to catch, and the
  constants are lifted from a pre-E1 table. **When it goes red the required action is to re-run the
  full pass and update the table and these constants in one commit** — never to loosen the
  tolerance. The test's docstring must say that, and must say that
  the constants are a regression pin against a specific build, not a law.
- **FAILS IF** a record with `figure_grade == false` has an empty `figure_grade_reasons`, or a
  record with `figure_grade == true` has a non-empty one. Assert over every record the test pass
  produces, including a **deliberately failing run** injected for the purpose: solve
  `family_traffic(16, seed=0)` at `(8, 8)` pinned, which is measured to end in trust collapse with
  clearance −0.1918, certificate 1.589 and slack 1.589 (§1.7.3), and assert its
  `figure_grade_reasons` names *not converged*, *hull certificate violated*, *penetrates* and
  *elastic slack*. Without that injected run the assertion could hold vacuously on an all-green
  pass, which is the exact failure mode this bullet exists to prevent.
- **FAILS IF** `--resume` re-runs a record whose identity is already in the file, or skips one whose
  `git_commit` or `extension_build_time` differs. Both directions, on a synthetic file.
- **FAILS IF** a run that raises produces no record. Assert by monkeypatching `optimize_spacetime`
  to raise, then checking the output file has exactly one line, with `error_type` set,
  `figure_grade == false` and a non-empty `figure_grade_reasons`.
- **FAILS IF** any record contains a JSON `NaN`, `Infinity` or `-Infinity` token — the writer must
  use `allow_nan=False` and the `*_nonfinite` sibling convention, because a downstream `json.load`
  that quietly accepts `NaN` and a `float(row.get(k, 0.0))` default together invert the gate's
  polarity (`optimize.py:100-107`).

---

## 3. `tools/make_tables.py`

JSONL in; markdown for `SOLVER.md` and LaTeX for the manuscript out. **The generator computes no
solver quantity.** It reads, filters, aggregates and formats.

### 3.1 CLI

```
tools/make_tables.py --in bench/*.jsonl
                     --table backbone|scaling|axes|coverage
                     --format md|tex
                     --out PATH
                     [--arm ours]            which arm is the table's primary column
                     [--expect-grid PATH]    JSON of the grid points the table must contain
```

Multiple `--in` files are concatenated; a duplicate `identity` is an error naming the file and line
of both copies (two records of one run means one of them is stale).

### 3.2 Tables and their columns

**`backbone`** — regenerates `SOLVER.md` §Measurements, same column order, with one new column.
Arms `ours` and `clipfloor_off` joined on (scenario, N, n_seg).

| column | unit | source |
|---|---|---|
| Scenario | — | `scenario_name` |
| Config | — | `f"N{N}_seg{n_seg}"` |
| Unsound walls (floor OFF) | count of walls | `koz_unsound_clips` of the `clipfloor_off` record |
| Clearance ON / OFF | scene length units (dimensionless for families; metres for `loiter`, via `scale`) | `min_clearance` of each arm, `%+.4f` |
| Cert ON | scene length units | `certificate_violation`, `%.2e` |
| Iters ON | iterations | `iterations` |
| Raises | count of ×10 steps | `weight_raises` |
| Weight | dimensionless penalty | `elastic_weight`, `%g` |
| Stop | — | `stop_label`, first token only |
| **Solve (s)** — new | seconds, wall-clock, median of `repeats` | `solve_seconds`, `%.3f` |
| Grade ON / OFF | — | `figure_grade` → `**yes**` / `no` |

**`scaling`** — demo C's curve, arm `ours`, aggregated over seeds.

| column | unit | source |
|---|---|---|
| Aircraft | count | `axis_value` |
| Seeds | k/N | figure-grade record count over total record count |
| Walls (seed) / Walls (returned) | count | median `walls_seed` / `walls_returned` over figure-grade records |
| Bound | count | `walls_bound` (identical across seeds; if not, error) |
| Rows | count | median `rows_returned` |
| Iters | iterations | median `iterations` |
| Solve | seconds, median [min, max] over seeds of `solve_seconds` | `%.3f` |
| Clearance | scene length units | median `min_clearance`, `%+.4f` |
| Grade | k/N | as Seeds |

**`axes`** — one block per axis (dimension, horizon, motion), same columns as `scaling` with
`Aircraft` replaced by the axis's own knob and unit (`Spatial dims` — count; `Horizon` — seconds;
`Motion` — kind).

**`coverage`** — figure slot 6. Per configuration: `koz_unsound_clips` with the floor off, the
clearance cost of turning the floor on (`min_clearance` ON − OFF), and the rate at which the hole
fires, defined as *records whose `certificate_violation ≤ 1e-6` and `min_clearance ≤ 0`* over all
records — the quantity `idea/spacetime.md` §"What must be measured" asks for by name.

**Units are printed in the header row, always**, and the length unit comes from the records' `scale`
field, never from a literal in the generator.

### 3.3 Missing and non-figure-grade records — never silently dropped

Four cases, four renderings:

1. **Present, figure-grade.** Normal.
2. **Present, not figure-grade.** Values render normally with a trailing `†`, `Grade` reads `no`,
   and a numbered footnote under the table carries the record's `figure_grade_reasons` **verbatim**
   and its `scenario_name`/`seed`. In an aggregate cell the record is excluded from the median and
   the `k/N` counter shows it.
3. **Present, errored** (`error_type` non-null). Every measured cell renders as `—`, `Grade` reads
   `error: <error_type>`, and the footnote carries `error_message`.
4. **Absent.** If `--expect-grid` is given (and for `backbone` and `scaling` it is **required**), a
   grid point with no record renders a full row of `MISSING` and **the generator exits non-zero**
   with a message naming every missing point. A silently short table is the failure this rule
   exists for.

An aggregate over zero figure-grade records renders `n/a (0/N figure-grade)`, never a blank and
never a median over non-figure-grade runs.

### 3.4 The sidecar rule

Mirrors `tools/render_poster.py` (`render_poster.py:41-80`) and the discipline
`paper/journal_1/README.md` §Figure slots states: *numbers are injected from figure-grade sidecars,
never typed; a build that cannot find a sidecar, or finds one that is not figure-grade, must fail.*

`make_tables.py` exposes `check(records, table)` which raises `SystemExit` listing every failure:

- any record selected as a **primary** cell of the table (not a footnoted one) whose `figure_grade`
  is false and which the table does not mark as such;
- records within one table disagreeing on `git_commit`, on `extension_build_time`, or on
  `extension_so_build_time` — a table assembled from two builds is not a table, and the compiled
  module's mtime is the one that actually identifies the solver;
- any record with `extension_stale == true` — it describes a build older than the Rust sources;
- any record with `git_dirty == true`, **unless** `--allow-dirty` is passed, in which case every
  affected row is marked `‡` and the caption says so;
- a `--expect-grid` point with no record (§3.3 case 4);
- any record whose `solve_seconds_max / solve_seconds_min > 1.2` or whose `load_avg_before > 1.0` —
  marked `‡` with a footnote, and a `SystemExit` only under `--strict-timing`.

Alongside every `.tex` table the generator writes `<out>.numbers.tex`: a block of
`\newcommand{\TBLxxx}{...}` macros for every number the manuscript prose quotes (largest aircraft
count solved figure-grade, its solve time, the walls-to-bound ratio, the hole's firing rate). **The
manuscript never types a result value**, exactly as `poster.tex` never does, and the corresponding
mechanical check is §3.5's second bullet. The generated files carry a
`% GENERATED by tools/make_tables.py -- do not edit.` header naming the input files and their
`git_commit`.

### 3.5 Invariants a test must pin

`tests/unit/test_make_tables.py`, on synthetic records — no solver needed.

- **FAILS IF** a record that is not figure-grade is rendered as a plain number with no `†`, no
  footnote and no `no` in its Grade column; **or** a `--expect-grid` point with no record produces a
  table and a zero exit status. Both halves are asserted by mutation, the way
  `tests/unit/test_poster_gate.py:54-79` does it: take a synthetic all-green record set, flip one
  record's `figure_grade` to `false` with a non-empty `figure_grade_reasons`, and assert the
  rendered output contains the `†`, contains each reason string verbatim, and that the row's Grade
  cell is `no`; then delete one record entirely and assert `SystemExit` naming the missing grid
  point. *Why it cannot pass vacuously:* the unmutated set must render with **no** `†` and exit `0`,
  which is asserted first.
- **FAILS IF** `<out>.numbers.tex` is not byte-identical to what the generator produces from the
  input records — the `test_numbers_tex_is_exactly_what_the_sidecar_generates` pattern
  (`tests/unit/test_poster_gate.py:81-84`), which is how a hand-edited generated file is caught
  before it reaches a PDF.
- **FAILS IF** the manuscript `.tex` contains, as a bare literal, any value that a `\TBL` macro
  provides — the `test_poster_tex_quotes_no_result_value_as_a_literal` pattern
  (`tests/unit/test_poster_gate.py:87-104`), with the same layout-number exclusion.
- **FAILS IF** `check()` accepts a record set whose members disagree on `git_commit` or
  `extension_build_time`, or one containing a record with `extension_stale == true`.

---

## 4. Order of work

1. `families.py` amendments (§1.1) + `tests/unit/test_families.py` + `tests/integration/test_families_rows.py`. No solver time; §1.8's list is the definition of done.
2. `tools/bench.py` + `tests/integration/test_bench.py`. The parity test (§2.8, bullet 1) before the backbone spot-check.
3. A first real pass: `--scenarios` over the whole registry, arms `ours,clipfloor_off`, `--repeats 3`. This is what makes `SOLVER.md` §Measurements regenerable and gives it its missing Solve column. **On the currently installed post-E1 build this pass will not reproduce the committed table**, and the diff between them is the first thing the `E1` change owes a measurement — see §5 item 4.
4. `tools/make_tables.py` + `tests/unit/test_make_tables.py`, then regenerate the backbone table and diff it against the committed one. **A diff beyond the new column is a finding, not a formatting problem.**
5. The traffic pass: `--family traffic --arms ours,snapshot`, 7 values × 5 seeds × 2 arms = 70 runs. At the measured per-solve costs (0.02 s at n = 1 to ~4 s at n = 20, ×3 repeats) that is well under an hour.
6. The `dimension`, `horizon`, `motion` passes; then `free_arrival` and `realistic` on their subgrids.

---

## 5. Open questions the implementer must NOT decide alone

1. **Whether the `snapshot` arm (§1.9) may stand as demo C's published baseline.** `DECISIONS.md`
   lists the baseline as BLOCKED on the PI, with three options; `snapshot` is a form of option (c),
   ablation against ourselves, which that document calls the weakest. Build the arm — it is cheap,
   it is a genuine structural pair, and it is needed either way as the fallback. **Do not write it
   into the manuscript as *the* baseline** without the PI's answer.
2. **Whether `n = 16`'s seed-0 non-convergence is reported or repaired.** Measured (§1.7.3): seed 0
   collapses, seeds 1 and 2 converge cheaply. Reporting it costs one open marker and one honest
   caption sentence; repairing it means multistart, which is `WORKSTREAM.md` `solver.multistart` and
   a different piece of work. The spec's default is **report it**; changing that default is a
   scope decision.
3. **Whether the `E1` clip_band change is in or out before the measurement passes run.** It is
   uncommitted, it is another session's work, and it changes the clipping radius under
   `sound_clip=True` — which is the default and the arm every headline number comes from. A pass
   taken across a decision that is still being made produces a table that has to be thrown away.
   Get a yes or a no on `E1` first, then take the pass; and either way the `E1` owner owes a
   before/after on the 28 configurations, which §2 makes a one-command job.
4. **The scaling figure's Osburn overlay.** Whether the three Table I points appear on the panel at
   all, given they are a different machine and a different problem, is a positioning call the PI and
   the venue choice settle. The bench and the table generator are indifferent; both produce the
   numbers either way.
