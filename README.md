# Space-Time Bezier Trajectory Sandbox

A research / debug sandbox for **space-time Bezier trajectory optimization**. The premise being
probed: adding time as an explicit Bezier coordinate lifts moving obstacles into static tubes, so
the existing convex-hull / supporting-half-space machinery handles moving-obstacle avoidance
directly. The sandbox exists to stress that premise — pose problems, find where the optimizer
breaks, prototype fixes.

Drag obstacles, tweak parameters, watch the optimizer re-solve.

## Start here

| you want | read |
|---|---|
| direction, workstreams, architecture rules | [`CLAUDE.md`](CLAUDE.md) — single source of direction for this branch |
| **paper 1** — offline, known obstacle motion | [`PAPER_1.md`](PAPER_1.md) — claim, method, scenarios, risks, 논문 뼈대 |
| **paper 2** — online, uncertain hazards (future) | [`PAPER_2.md`](PAPER_2.md) — risk field, prior art, open question |
| prior-art verification record | [`doc/refs/novelty_positioning.md`](doc/refs/novelty_positioning.md) |
| safe-corridor ground truth | [`doc/refs/safe_corridor_references.md`](doc/refs/safe_corridor_references.md) |
| 연구노트 — separate repos, ignored here | `doc/notes/README.md` (on disk, not tracked by this repo) |

The two papers have a hard scope boundary. Paper 1 assumes obstacle motion is known and
deterministic and solves offline. Paper 2 relaxes exactly that assumption — limited sensing,
uncertain hazards, receding horizon. Do not mix their claims.

## Repo map

```
├── CLAUDE.md                  direction and guardrails — read first; reasoning lives in PAPER_1.md / PAPER_2.md
├── PAPER_1.md                 paper 1 central doc (offline, known motion)
├── PAPER_2.md                 paper 2 central doc (online, uncertain hazards)
├── BENCHMARKS.md              STALE — predates commit 848bf3b; see "Measurements" below
│
├── spacetime_bezier/          Python package: API, scenarios, sandbox server
│   ├── __main__.py              entrypoint — `python3 -m spacetime_bezier`
│   ├── optimize.py              public API: optimize_spacetime(), optimize_scenario()
│   ├── scenarios.py             scenario definitions + SCENARIO_MAP registry
│   ├── geometry.py              MovingObstacle, bezier_curve(), obstacle bundling
│   ├── objective.py             initial guess; energy matrix (Rust builds the one used)
│   ├── sandbox.py               interactive HTTP server, live re-solve on drag
│   ├── io.py                    JSON I/O, scenario baking, viewer launcher
│   ├── viewer.py                sanity-check viewer server — `python3 -m spacetime_bezier.viewer`;
│   │                            binds the same 8767 as the sandbox and refuses to share it
│   ├── static/                  viewer.html + plotly.min.js, served by viewer.py
│   ├── rust_debug_stepper.py    steps the real Rust run via its emitted trace
│   ├── debug_session.py         stateful session for the step-debugger UI
│   ├── debug_trace.py           DebugFrame schema
│   ├── constraints.py           dead Python KOZ builder (pins a stale time scale); delete with debug_stepper.py
│   └── debug_stepper.py         dead Python SCP stepper; not a reference implementation
│
├── rust_optimizer/            the only optimizer backend
│   ├── core/src/
│   │   ├── spacetime_optimizer.rs   SCP outer loop, SCvx ratio test, elastic relaxation
│   │   ├── spacetime_generator.rs   the keep-out generator: centreline, or its shadow (center
│   │   │                            surface) through a station — one zone, two stretch factors
│   │   ├── spacetime_obstacle.rs    clip ball, nearest point, approach cut, support ceiling
│   │   ├── spacetime_constraints.rs one row builder over the generators; boundary, time
│   │   │                            monotonicity, trust box
│   │   ├── optimizer.rs             shared solve_qp() (Clarabel) + orbital docking optimizer
│   │   ├── bezier.rs                D/E/G matrices, byte-identical to main branch
│   │   ├── de_casteljau.rs          subdivision matrices
│   │   ├── constraints.rs           orbital-docking constraints (legacy path)
│   │   └── gravity.rs               legacy
│   └── pybind/src/lib.rs        PyO3 bindings: optimize_spacetime_bezier()
│
├── orbital_docking/           LEGACY. Earlier orbital-rendezvous work. Still supplies the
│                              dimension-agnostic building blocks (bezier.py, de_casteljau.py)
│                              that the spacetime code imports. Not the headline.
│
├── paper/                     manuscripts in the societies' own templates, one dir per venue
│   ├── README.md                HANDBOOK — pipeline, venue rules, invariants, draft state
│   └── ksas_2026_fall/          KSAS 2026 추계: manuscript.docx + .md sidecar, template/
│
├── tools/                     spacetime_opt_debug.py (debug UI server),
│                              compare_backends.py (scenario diff), plus legacy diagnostics
│   ├── render_paper.py          manuscript .docx -> .md sidecar + .pdf
│   ├── watch_paper.sh           re-render on every save (fswatch, hash-guarded)
│   ├── make_manuscript_skeleton.py  build the empty manuscript from the template
│   └── docx_edit.py             surgical paragraph edits, superscript citations
│
├── tests/                     pytest: unit / integration / property / regression
│                              tests/data/golden_run.json is the regression baseline
│
├── figures/
│   ├── spacetime_bezier_interactive.html   3D demo with debug overlays
│   ├── spacetime_bezier_opt_debug.html     live SCP step debugger UI
│   ├── spacetime_scenarios.json            baked control points (generated)
│   ├── paper1/                             the paper's figures + their sidecars
│   └── legacy/                             older renderings, kept for provenance
│
└── doc/
    ├── refs/novelty_positioning.md     prior-art map, per-source verification status
    ├── refs/papers/           archived source PDFs (not committed)
    ├── refs/safe_corridor_references.md  C3 ground truth for one plane per (segment, obstacle)
    ├── notes/                 NOT PART OF THIS REPO — gitignored. 연구노트 are a standalone
    │                          project; each note directory is its own git repository on disk,
    │                          with its own history, bound for the lab's shared note repo.
    │                          Rules for what qualifies: doc/notes/README.md
    └── archive/               old chat-derived notes
```

## Quickstart

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Build the Rust optimizer (one-time, requires a Rust toolchain + maturin)
cd rust_optimizer/pybind && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd ../..

# 3. Launch THE frontend (config panel, solve button, in-place scene update)
python3 -m spacetime_bezier
# opens http://127.0.0.1:8767/ immediately and solves NOTHING up front;
# each scenario / degree / segment count you pick is one solve.
# If the port is busy it prints who holds it and exits 1 rather than
# silently moving to another port or reusing a server built from old code.

# 4. (Optional) Regenerate pre-baked scenario JSON for file:// viewing
python3 -m spacetime_bezier --bake     # SLOW: many configurations
# then: open figures/spacetime_bezier_interactive.html

# 5. (Optional) Live step-through debugger
python3 tools/spacetime_opt_debug.py
# then open figures/spacetime_bezier_opt_debug.html

# 6. Tests
pytest
```

## Interacting with the sandbox

Every obstacle, plus the start (green) and goal (red) markers, is a grab-handle in the 3D view:

- **L-click + drag** — slide the handle across (x, y) at its current t.
- **Shift + L-click + drag** — slide along t at fixed (x, y). On an obstacle's end control point
  this shrinks/extends its active time window; on the goal marker it re-scales scenario T live.
- **Escape** while dragging — cancel and revert to the pre-drag position.
- **Right-click drag** — pan. **L-click drag on empty space** — rotate. **Scroll** — zoom.

Re-solve fires on release (debounced ~150 ms). The trajectory dims while a drag is in progress or
a solve is in flight, so a stale curve is visibly stale.

## Demo scenarios

Registered in `SCENARIO_MAP` in `spacetime_bezier/scenarios.py`. Each is 2D + time: moving
obstacles, fixed endpoints, and the optimizer plans path *and* timing.

| Scenario | What it shows |
|---|---|
| `original` | 3 moving obstacles — curve threads between constant-velocity tubes in (x, y, t) |
| `wall` | a wall that vanishes at a known time; the curve should wait, then pass — time as a real optimization dimension |
| `diverse` | varied sizes / speeds / directions; stress case |
| `fence3d` | three spatial dimensions + time; the curve climbs a wide low **moving** fence instead of going around — the dimension is load-bearing, and the motion is what makes waiting futile. Renamed from `wall3d` 2026-08-24; the problem definition is unchanged |
| `door3d` | the complement of `fence3d`: a **static** tall wall in three spatial dimensions that vanishes at t=5, so the cheap answer is to wait rather than climb. Nothing here makes the third dimension load-bearing — a 2D cut would wait identically |
| `station_fence` | the first line-of-sight scenario — keep line of sight to a fixed station past a moving, non-straight fence (chain of time-windowed pieces). Occlusion rows on: climbs and holds the link; off: loses it for 7.8 s of 10. The fence is both occluder and keep-out body, so the occlusion constraint subsumes collision. **Its numbers below were measured under the retired occlusion builder and are stale** |
| `loiter` | **the paper's demo** — metres and seconds: a body loitering 50 m above a ground control station on a 30 m circle, and a 200 m corridor at 62.5 m altitude whose axis lies ON the shadow ring, so the shadow spot sweeps ALONG the corridor instead of across it. Occlusion rows off: the link is lost over [14.96, 16.52] s at −2.169. On: the run arrives 15.7 s later and holds +10.642. **Retiming is the escape and it is measured, not asserted** — graft the baseline's schedule onto the constrained path and line of sight falls to −2.109, while the path itself deviates 0.08 m laterally over 200 m |

The paper's figure comes from `loiter` via `tools/make_paper_figure.py --free-arrival --time-weight
10 --v-max 5 --sound-clip`, which refuses to draw unless the constrained run is figure-grade, the
baseline measurably fails, **and** `koz_unsound_clips` is zero — see the `loiter` block under
§Measurements for why that third gate exists.

## Measurements

Last full pass **2026-08-26, after the keep-out wall moved onto the clipped KOZ volume** — one
run of every registered configuration at defaults: speed cap off, arrival time pinned,
elastic-weight ladder on. **28 runs**, not the 22 this section used to claim — `original` 5,
`curve` 4, `diverse` 5, `wall` 6, `fence3d` 4, `door3d` 2, `station_fence` 2. The old count
predated three of those scenarios and was never corrected. A run that enables `v_max` /
`time_weight` / `free_arrival_time` is a different problem and must be re-measured.
**FIGURE-GRADE** is the B7 gate, checked per run: converged AND certificate ≤ 1e-6 at the returned
iterate AND clearance > 0 against the true obstacle trajectories AND total slack ≤ 1e-6 — plus the
occlusion certificate where a station exists.

| Key | Best config | Clearance | Iters | Elastic weight | Figure-grade |
|-----|-------------|-----------|-------|----------------|--------------|
| `original` | N8_seg4 | **+0.6204** | 9 | 100 | **yes — all 5 configs** |
| `curve` | N8_seg16 | **+0.1000** | 14 | 300 | **3 of 4.** N8_seg8 +0.0919 @800, N10_seg8 +0.1000 @1e5. **N8_seg4 does not converge** (certificate 0.44) — it did not converge before this change either (certificate 0.35 @1e4), so it is a standing failure, not a regression |
| `diverse` | N8_seg4 | **+0.1136** | 9 | 800 (3000–10000 at 8–16 seg; N10_seg16 certifies at 3000) | **yes — all 5 configs** |
| `wall` | N10_seg16 | **+0.2058** | 9 | 3000 | **3 of 6.** N8_seg16 +0.2017 @3000, N10_seg24 +0.1484 @800. **N8_seg2/3/4 penetrate** — that is the 2026-08-24 densification (spacing 0.8 → 0.5, 13 → 21 circles), not this change; see Known Issues in CLAUDE.md |
| `fence3d` | N8_seg2 | **+0.1623** | 9 | 100 | **yes — all 4 configs**, first ladder rung. Measured under the name `wall3d`; the rename changed no parameter |
| `door3d` | N8_seg4 | +0.9827 | 9 | 100 | **yes — both configs**, first ladder rung. Clearance is not the point: the evidence is that the curve **waits** |
| `station_fence` | N8_seg8 | **+1.0770** | 17 | 10000 | **1 of 2 — re-measured 2026-08-31 under the center-surface builder, and the two configs traded places.** Clearance is slack by construction (occlusion subsumes keep-out); the binding number is the occlusion certificate **0.000**, with slack 1.4e-12. **N8_seg16 now fails**: 37 iterations, ladder stuck at 100, trust collapse without certifying, occlusion certificate **1.067e-02** and slack 1.06e-02 — it holds the larger clearance (+1.1429) while having lost the link. Under the retired builder this was reversed: N8_seg16 certified at 1e5 in 56 iterations (+1.5454) and N8_seg8 did not converge. **`best` is picked by clearance, not by figure-grade, so this scenario's `best` key still names the failing config** |
| `loiter` | N8_seg16 | **+36.8417** | 3 | 100 | **yes — both configs**, first ladder rung. **Measured 2026-08-30, not part of the pass above.** At defaults the arrival is pinned, so the scenario's own point does not show here; the demo is the priced run in the block below, and so is the finding |

### `loiter`, measured 2026-08-30 — and the figure-grade gate does not see an unsound clip

`loiter` was not in the pass above; these six runs are its own, all on one build of the extension
(checked: same mtime before and after). `sound_clip` floors the clip radius at the reach, PAPER_1
statement 8. `koz_unsound_clips` is statement 7 counted at the returned iterate: **walls** — one count per
emitted plane, per generator, so a two-approach clip counts twice — whose clip ball did not reach
the segment radius plus the trust-box reach, so the wall was built against a clipped piece **the
next iterate could leave**.

| Run | Iters | Weight | Clearance | Min sight margin | Arrival | Unsound clips | `figure_grade` |
|---|---|---|---|---|---|---|---|
| N8_seg8 defaults | 3 | 100 | +35.3204 | +12.067 | 80.00 pinned | **2** | yes |
| N8_seg8 defaults + `sound_clip` | 3 | 100 | +35.3767 | +12.090 | 80.00 pinned | 0 | yes |
| N8_seg16 defaults | 3 | 100 | +36.8417 | +12.646 | 80.00 pinned | 0 | yes |
| N8_seg16 defaults + `sound_clip` | 3 | 100 | +36.8459 | +12.647 | 80.00 pinned | 0 | yes |
| N8_seg8 priced | 10 | 1e5 | +24.3026 | +7.365 | 51.502 | **24** | yes |
| N8_seg8 priced + `sound_clip` — **the figure** | 12 | 1e5 | +30.5243 | +10.642 | 55.747 | 0 | yes |

Priced is free arrival, `time_weight` 10, `v_max` 5, weight pinned at the registered 1e5 — the
registry value at measurement time, `c32e88b` plus the center-surface working tree. Both
certificates are 0.000, total slack ≤ 3e-12 and the speed-cap violation is 0.0 on all six. Every
row was taken twice, once through the elastic ladder and once pinned at the rung it settled on, and
the two clearances agree exactly; the second pass exists only because the result row does not carry
`koz_unsound_clips`.

**Measured twice, down two different call paths.** The table above came through
`optimize_scenario` and its ladder; the same six configurations were run independently through
`optimize_spacetime` via `make_paper_figure`'s own `solve_pair`, and the counts agree — 2 at
N8_seg8 defaults, 0 with the floor, 0 at N8_seg16 either way, 24 priced at the default clip, 0
priced with the floor. Two paths that share only the Rust core had to agree, and do.

**The last two columns together are the finding.** `figure_grade` passes the priced default-clip
run while 24 of its clipped volumes escape the ball their wall was built against.
`figure_grade_failures` in `optimize.py` never reads `koz_unsound_clips` — the key is not even
propagated into the result row — so **on this condition the B7 gate cannot fail.** Only
`tools/make_paper_figure.py` refuses, and only for the figure. A run can be reported figure-grade
in this table and still rest on a certificate that covers less than the obstacle.

**The ladder settles at 100 at defaults, not at the registered 1e5.** The registered weight belongs
to the priced problem; a defaults row quoting it would be quoting a rung this scenario never needs
with the arrival pinned.

**What the clipped-volume construction changed, measured against the same configurations built
with the retired hull-of-band plane.** Where the obstacle is straight its tube is convex and the
two constructions agree on the plane, so `original`, `fence3d` and `door3d` move only in the last
digits. Where it curves they differ, and the difference is iterations rather than answers: `curve`
N8_seg8 went from 11 iterations to 39 and from +0.0894 to +0.0919 of clearance, and `curve`
N10_seg8 now needs the ladder's top rung (1e5) where it used to certify at 800. That is the
conservatism of a rigorous support ceiling being paid in step size, which is what sequential convex
programming pays it in. Nothing that certified before stopped certifying.

**The clip-radius floor fires on the real scenario, not only in the unit tests.** On `curve` at the
straight seed, segment 1 against the arc has its centroid 0.598 from the centreline of a tube of
radius 0.9 — inside the keep-out zone — so the clip radius is floored to 0.900 instead of shrinking
to 0.598. Sweeping the clip radius over every segment and both obstacles at four trust radii, the
clipped volume has **exactly one connected component everywhere** and nothing is dropped: time
monotonicity keeps the lifted tube self-avoiding, so no ball can cut a piece out of its middle.

**Walls group by local approach, not by connected component — changed 2026-08-26.** The band is cut
at every interior local maximum of the centreline's distance to the segment centroid, one wall per
dip of the profile. The reason is the wrapping bend of panel B2: one connected lump curling around
the centroid got one non-separating wall (margin −1.154 on a segment whose centroid is 0.205
OUTSIDE the keep-out zone, a row needing trust ≈1.04 against the default 0.5); the cut gives the
same geometry two walls at +0.529/−0.162 with nothing else changed. At the straight seed the cut
emits **exactly the same row count as the component grouping on all seven registered scenarios** —
every obstacle here approaches each segment once — so the table above did not move. The multi-wall
path is exercised by the panel tests (`test_panel_b1_...`, `test_panel_b2_...`) and by the Rust
in-crate tests in `spacetime_obstacle.rs`, but by no scenario in this repository.

The elastic weight is part of the result, not a tuning knob: above the exact-penalty threshold the
penalized and constrained problems share a solution, below it they do not, and the threshold
depends on the optimal multipliers so it differs per scenario and per segment count.
`optimize_scenario` escalates through `ELASTIC_WEIGHT_LADDER` and records the weight that
certified. Segment count reversed direction when one-plane-per-segment landed — `original` reaches
+0.620 at 4 segments against +0.323 at 8 — so old config lists that start at 8 segments miss the
best result entirely.

Timing profiles at these defaults are objective artifacts (the smoothness term is blind to the
time coordinate; control-point times can sit on the `min_dt` floor). Geometry is evidence; timing
is not, until a run sets the speed cap and time penalty.

## Architecture in one paragraph

Rust is the sole optimizer backend. One `scp_step` in the Rust core is both the batch iteration and
the debug step — debugger sessions observe the real run through an emitted trace, they do not
implement a second optimizer. Python handles request shaping, scenario definitions, JSON I/O, and
the debug UI server. See `CLAUDE.md` §Architecture for the five rules this enforces.

## Extending

- **Add a scenario** — edit `spacetime_bezier/scenarios.py`, register in `SCENARIO_MAP`.
- **Change the optimizer** — `rust_optimizer/core/src/spacetime_optimizer.rs` (outer loop) or
  `spacetime_constraints.rs` (KOZ geometry), then rebuild with `maturin develop --release`.
- **Change the UI** — `figures/spacetime_bezier_interactive.html` or
  `figures/spacetime_bezier_opt_debug.html`.

Run `pytest` before opening a PR.

## Where this goes, after the paper

Paused until the paper ships; the direction still holds and the rules in `CLAUDE.md` §Architecture
already govern how the code is written. The target is a personal research and debug workbench for
the space-time Bezier idea — pose a problem, find where the optimizer breaks, prototype against it.
Drag obstacles in 3D, slide parameters live, get "why did it fail?" in one line, scrub the SCP
iteration history, save and reload scenarios.
