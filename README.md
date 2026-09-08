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
| the current goal, and the traps | [`CLAUDE.md`](CLAUDE.md) — 36 lines, loaded into every session |
| **what is open right now** | [`WORKSTREAM.md`](WORKSTREAM.md) — open items only; a finished item is deleted |
| **idea 1** — offline, known obstacle motion | [`idea/spacetime.md`](idea/spacetime.md) — claim, formulation, geometry, novelty, demo scenario |
| **idea 2** — online, uncertain hazards (future) | [`idea/risk_field.md`](idea/risk_field.md) — risk field, prior art, open question |
| **the code** — architecture, facts, measurements, known issues | [`SOLVER.md`](SOLVER.md) |
| **the KSAS 발표논문 and its poster** | [`paper/ksas_2026_fall/README.md`](paper/ksas_2026_fall/README.md) — 일정, venue rules, manuscript state |
| how to write a manuscript here | [`paper/README.md`](paper/README.md) — render/watch pipeline, silent-breakage invariants |
| safe-corridor ground truth | [`doc/refs/safe_corridor_references.md`](doc/refs/safe_corridor_references.md) |
| 연구노트 — separate repos, ignored here | `doc/notes/README.md` (on disk, not tracked by this repo) |

**One idea, many artifacts.** An idea document owns the truth — the claim, the derivation, why it
is new. An artifact document owns one rendering of it: which venue, which figures, which deadline,
what state the draft is in. Numbers live in [`SOLVER.md`](SOLVER.md) beside the tests that
re-measure them, and an artifact quotes a measurement pass rather than owning one.

The two ideas have a hard scope boundary. Idea 1 assumes obstacle motion is known and deterministic
and solves offline. Idea 2 relaxes exactly that assumption — limited sensing, uncertain hazards,
receding horizon. Do not mix their claims.

## Repo map

```
├── CLAUDE.md                  the current goal and the traps — read first, 36 lines
├── WORKSTREAM.md              what is open — items only, ids you can open a session with
├── SOLVER.md                  the code: architecture, established facts, measurements, known issues
├── idea/
│   ├── spacetime.md             idea 1 (offline, known motion) — claim, formulation, novelty, demo scenario
│   └── risk_field.md            idea 2 (online, uncertain hazards) — future work
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
│   │   ├── bezier.rs                D/E/G matrices (allocation-free kernels since 8dc5a4b; no
│   │   │                            longer byte-identical to main, results bit-identical)
│   │   ├── de_casteljau.rs          subdivision matrices
│   │   ├── constraints.rs           orbital-docking constraints (legacy path)
│   │   └── gravity.rs               legacy
│   └── pybind/src/lib.rs        PyO3 bindings: optimize_spacetime_bezier()
│
├── orbital_docking/           LEGACY. Earlier orbital-rendezvous work. Still supplies the
│                              dimension-agnostic building blocks (bezier.py, de_casteljau.py)
│                              that the spacetime code imports. Not the headline.
│
├── paper/                     one dir per venue: the manuscript + that artifact's own README
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
# PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 suppresses PyO3 0.24.2's "maximum supported
# version 3.13" check and builds against the stable ABI. Without it the build fails.
#
# PIN THE INTERPRETER. There are several worktrees of this repo on disk, and maturin
# resolves PYO3_PYTHON from the environment -- it will happily build against ANOTHER
# worktree's venv, report success, and install an extension you did not build. That
# failure is silent and it is the stale-extension class this repo has been burned by.
J=$(git rev-parse --show-toplevel)
cd "$J/rust_optimizer/pybind" && env VIRTUAL_ENV="$J/.venv" \
    PYO3_PYTHON="$J/.venv/bin/python3" \
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
    PATH="$J/.venv/bin:$PATH" \
    "$J/.venv/bin/maturin" develop --release
cd "$J"

# 2b. VERIFY the build landed in THIS worktree. This is the check, not the build --
# if the path is not under your worktree, step 2 lied and you are running stale code.
python3 -c "import bezier_opt; print(bezier_opt.__file__)"

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
| `loiter` | **the paper's demo** — metres and seconds: a body loitering 50 m above a ground control station on a 30 m circle, and a 200 m corridor at 62.5 m altitude whose axis lies ON the shadow ring, so the shadow spot sweeps ALONG the corridor instead of across it. Occlusion rows off: the link is lost over [14.96, 16.52] s at −2.169. On: the run arrives 15.7 s later and holds +10.642. **Retiming is the escape and it is measured, not asserted** — graft the baseline's schedule onto the constrained path and line of sight falls to −2.109, while the path itself deviates 0.08 m laterally over 200 m |

The paper's figure comes from `loiter` via `tools/make_paper_figure.py --free-arrival --time-weight
10 --v-max 5 --sound-clip`, which refuses to draw unless the constrained run is figure-grade, the
baseline measurably fails, **and** `koz_unsound_clips` is zero — see the `loiter` block under
§Measurements for why that third gate exists.

## Measurements

Moved. Every measured number, the dated pass it came from, and the elastic weight each result needs
now live in [`SOLVER.md`](SOLVER.md) §Measurements — beside the architecture facts and known issues
they qualify, and beside the tests in [`tests/`](tests/) that re-measure them.

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
