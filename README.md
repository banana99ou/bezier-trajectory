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
│   ├── constraints.py           Python KOZ builder — debug stepper only, NOT production
│   └── debug_stepper.py         dead Python SCP stepper; not a reference implementation
│
├── rust_optimizer/            the only optimizer backend
│   ├── core/src/
│   │   ├── spacetime_optimizer.rs   SCP outer loop, SCvx ratio test, elastic relaxation
│   │   ├── spacetime_constraints.rs KOZ tube geometry, boundary, time monotonicity, box
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
├── tools/                     11 legacy diagnostics were deleted 2026-08-30 (`paper/journal-1`):
│                              7 imported only `orbital_docking`, 2 were branch-migration
│                              artifact diffing, 2 were matplotlib concept demos the frontend
│                              replaced. What is left is what the paper or the solver uses.
│   ├── sweep_scenarios.py       one pass over the registry at defaults, gate fields printed
│   ├── diagnose_clip.py         per (segment, obstacle): row, out of reach, or no half-space
│   ├── compare_backends.py      run scenarios through the Rust optimizer, dump traces
│   ├── trace_viewer.py          replay child script, imported by frontend.py
│   ├── spacetime_opt_debug.py   debug UI server
│   ├── make_paper_figure.py     the paper figure, refused unless figure-grade
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
| `curve` | one obstacle on a genuinely **curved** lifted centreline, so its tube is non-convex and has no supporting half-space. The only scenario where the clip-and-support construction is doing real work rather than reproducing an easier answer |
| `loiter` | **the paper's demo** — a body orbiting a ground station, its shadow sweeping *along* a corridor the vehicle must cross. Occlusion rows on: waits and holds the link; off: loses it. Metres and seconds. The corridor sits above the body's reach, so keep-out never binds and the **timing** is set by line of sight alone |

The paper's figure comes from `loiter` via `tools/make_paper_figure.py`, which refuses to
draw unless the constrained run is figure-grade AND the baseline measurably fails.

`station_fence` was **removed 2026-08-30**. It was the occlusion demo before `loiter`, every one of
its numbers was measured under the occlusion builder that `46a15f7` retired, and its own test file
asserted the retired design — that a curved occluder must be a chain of *straight* pieces, because
a capsule around a curved centreline is not convex. The center-surface builder handles the curved
centreline directly, so that assertion had become a test of a decision the code no longer makes.
Its live coverage moved to `loiter` rather than being deleted: the frontend occlusion-layer tests,
the plane-drop refusal test, and the obstacle-format tests all now run against `loiter`.

## Measurements

Last full pass **2026-08-31, after the figure-grade gate started reading what the certificate
COVERS** — every registered configuration run twice, once at the default clip and once with the
reach floor (`sound_clip=True`, PAPER_1 statement 8). 28 configurations, 56 runs: `original` 5,
`curve` 4, `loiter` 2, `diverse` 5, `wall` 6, `fence3d` 4, `door3d` 2. Defaults otherwise: speed cap
off, arrival pinned, elastic-weight ladder on. A run that enables `v_max` / `time_weight` /
`free_arrival_time` is a different problem and must be re-measured.

**FIGURE-GRADE** is the B7 gate: converged AND certificate ≤ 1e-6 at the returned iterate AND
clearance > 0 against the true obstacle trajectories AND total slack ≤ 1e-6 AND the occlusion
certificate where a station exists AND — **new 2026-08-31** — **zero uncovered clipped volumes.**

### The finding: at the default clip, 26 of 28 configurations were graded on a certificate that covers less than the obstacle

The rows certify against `K_m ∩ B(c, ρ)`, not against `K_m`. The gap closes only where PAPER_1
statement (7) holds, counted at the returned iterate as `koz_unsound_clips`. The Rust core has
counted it since the clip landed; `optimize_scenario` never put it in the result row and
`figure_grade_failures` never read it, so **on this condition the gate could not fail.** Only
`tools/make_paper_figure.py` refused, and only for the figure. Closed 2026-08-31.

Measured across the registry at defaults, only **`curve` N10_seg8 and `loiter` N8_seg16** have zero
uncovered pairs. Every other configuration has between 1 and **197** (`wall` N10_seg24). With the
reach floor on, **every configuration goes to zero**, and figure-grade goes from **2 of 28 to 23 of
28**. Of the five that still fail with the floor on, four were already failing for reasons that
have nothing to do with the clip; the fifth is a regression the floor causes and it is named below.

**The floor is free on most of the registry and not on all of it.** Clearance, default clip → floor:

| Key | Configs | Uncovered at default | Cost of the floor | Figure-grade with floor |
|---|---|---|---|---|
| `original` | 5 | 4–10 | **none** — clearance identical to 4 dp on all five (best N8_seg4 **+0.6204** @100, 9 iters) | **5 of 5** |
| `diverse` | 5 | 25–54 | **none** — identical on all five (best N8_seg4 **+0.1136** @800, 9 iters) | **5 of 5** |
| `fence3d` | 4 | 22–49 | **none** — identical on all four (best N8_seg2 **+0.1623** @100, 9 iters) | **4 of 4** |
| `door3d` | 2 | 36, 60 | **none**, slightly better (N8_seg4 +0.9827 → +0.9828, N8_seg8 +1.0009 → +1.0044) | **2 of 2** |
| `loiter` | 2 | 2, 0 | **none**, slightly better (N8_seg8 +35.3204 → +35.3767, N8_seg16 +36.8417 → +36.8459) | **2 of 2** |
| `wall` | 6 | 13–197 | **real.** N8_seg16 +0.2017 → **+0.0556**, N10_seg24 +0.1484 → **+0.0673**; N10_seg16 unchanged at **+0.2058**. N8_seg2/3/4 penetrate either way — the 2026-08-24 densification, see Known Issues in CLAUDE.md | **3 of 6** |
| `curve` | 4 | 0–10 | **mixed, and one regression.** N8_seg8 +0.1000 → +0.0947 but certifies at 800 instead of 1e4; N10_seg8 +0.1000 → +0.0976 at 3000 instead of 1e5. **N8_seg16 loses its certificate**: 2.6e-11 @300 at the default clip, 6.35e-06 @1e5 with the floor, hitting the 200-iteration cap. N8_seg4 does not converge either way | **2 of 4** |

The `curve` N8_seg16 row is the honest cost and is recorded rather than dropped: a rigorous support
ceiling is paid in step size, and on the one scenario whose tube is genuinely non-convex it is
sometimes paid in convergence. `curve` N8_seg4 was a standing failure before this change.

**The reach floor is now the DEFAULT — decided 2026-08-31, and it is not a mode.** Every number in
this repository's history before that date was produced with it off, which means produced with a
certificate that covers only the clipped piece wherever the segment came close to an obstacle. The
flag survives so this table stays reproducible; `sound_clip=False` is a measurement setting, not a
way to run the solver.

**One interaction, measured, that the flip exposed.** The reach is the segment's own radius plus one
trust step, so the floor GROWS WITH THE TRUST RADIUS. On `original` (a 10-unit scene) with a speed
cap and a priced free arrival: trust 2.0 takes 21 iterations without the floor and 22 with it, trust
4.0 takes 13 either way, and **trust 8.0 goes from converging in 13 iterations to hitting the
300-iteration cap** — its returned iterate is still certified at 2.36e-07, so that is a convergence
declaration rather than a correctness failure, but the clip has stopped localizing anything: the
floor is about 14 on a scene 10 across. A trust radius large relative to the scene and a sound clip
are in tension by construction, which is exactly what the reachability lemma says and is worth
stating in the paper rather than discovering twice.

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
