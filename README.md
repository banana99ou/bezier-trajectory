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
├── tools/                     spacetime_opt_debug.py (debug UI server),
│                              compare_backends.py (scenario diff), plus legacy diagnostics
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

# 3. Launch the interactive sandbox (live re-solve on drag)
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
| `wall3d` | three spatial dimensions + time; the curve climbs a wide low fence instead of going around — the dimension is load-bearing |
| `station_fence` | **the paper's demo** — keep line of sight to a fixed station past a moving, non-straight fence (chain of time-windowed pieces). Occlusion rows on: climbs and holds the link; off: loses it for 7.8 s of 10. The fence is both occluder and keep-out body, so the occlusion constraint subsumes collision |

The paper's figure comes from `station_fence` via `tools/make_paper_figure.py`, which refuses to
draw unless the constrained run is figure-grade AND the baseline measurably fails.

## Measurements

Last full pass **2026-08-20, after the B8–B10 formulation freeze** — one run of every registered
configuration (22 total) at defaults: speed cap off, arrival time pinned, elastic-weight ladder on.
A run that enables `v_max` / `time_weight` / `free_arrival_time` is a different problem and must be
re-measured. **FIGURE-GRADE** is the B7 gate, checked per run: converged AND certificate ≤ 1e-6 at
the returned iterate AND clearance > 0 against the true obstacle trajectories AND total slack
≤ 1e-8 — plus the occlusion certificate where a station exists.

| Key | Best config | Clearance | Iters | Elastic weight | Figure-grade |
|-----|-------------|-----------|-------|----------------|--------------|
| `original` | N8_seg4 | **+0.620** | 9 | 100 | **yes — all 5 configs** |
| `diverse` | N8_seg4 | **+0.1136** | 9 | 800 (3000–10000 at 8–16 seg; N10_seg16 certifies at 3000) | **yes — all 5 configs** |
| `wall` | N10_seg16 | **+0.0751** | 19 | 100000 | **only this config** — five others fail, incl. N8_seg2 which *clears* (+0.102) but is uncertified: the exact case the gate exists to catch |
| `wall3d` | N8_seg2 | **+0.1623** | 9 | 100 | **yes — all 4 configs**, first ladder rung |
| `station_fence` | N8_seg8 | +1.163 | 124 | 100000 | **yes — both configs.** Clearance is slack by construction (occlusion subsumes keep-out); the binding numbers are the independent min line-of-sight margin **+0.338** and the occlusion certificate **0.000** |

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
