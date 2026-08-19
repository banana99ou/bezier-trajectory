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
| prior-art verification record | [`doc/refs/c1_novelty.md`](doc/refs/c1_novelty.md) |
| frozen formulation, derivations, occluder geometry | [`doc/notes/005_formulation_freeze.md`](doc/notes/005_formulation_freeze.md) |
| evidence behind every established fact; measurement history | [`doc/notes/006_solver_record.md`](doc/notes/006_solver_record.md) |
| where the sandbox goes after the paper | [`doc/notes/007_sandbox_direction.md`](doc/notes/007_sandbox_direction.md) |
| safe-corridor ground truth | [`doc/notes/_shared/c3_safe_corridor_refs.md`](doc/notes/_shared/c3_safe_corridor_refs.md) |

The two papers have a hard scope boundary. Paper 1 assumes obstacle motion is known and
deterministic and solves offline. Paper 2 relaxes exactly that assumption — limited sensing,
uncertain hazards, receding horizon. Do not mix their claims.

## Repo map

```
├── CLAUDE.md                  direction and guardrails — read first; reasoning lives in doc/notes/
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
│   └── risk_field/                         paper 2 figures + the scripts that make them
│
└── doc/
    ├── refs/c1_novelty.md     prior-art map, per-source verification status
    ├── refs/papers/           archived source PDFs (not committed)
    ├── notes/_shared/         cross-workstream reference notes
    ├── notes/001_.../         LaTeX formulation note — contradicts itself, see CLAUDE.md
    ├── notes/005_formulation_freeze.md  frozen formulation, velocity/acceleration
    │                                    derivation, occluder geometry (item A1)
    ├── notes/006_solver_record.md       evidence for every established fact;
    │                                    scenario measurement history
    ├── notes/007_sandbox_direction.md   post-paper direction + honesty rules
    ├── notes/004_probabilistic_koz/  paper 2 draft (Korean LaTeX): lobe geometry, convexity,
    │                                 saddle merging. See PAPER_2.md for positioning.
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

Paper 1 replaces these with a **door** scenario (the `wall` behaviour, cleaned up) and a **stealth**
scenario (hide in a moving occluder's shadow). See `PAPER_1.md` §4. The stealth scenario has no
constraint builder in the Rust core yet.

## Measurements

**There is currently no citable performance number.** `BENCHMARKS.md` predates commit `848bf3b`
(SCvx ratio test) and the constraint-geometry rewrite in `0f37794` (B1/B2/B4 — correct tube
geometry, one plane per segment). Its numbers describe a solver that no longer exists.

Before quoting any number, read `PAPER_1.md` §6: the obstacle hull is rebuilt per iteration from
the current iterate and is only valid inside the time span it was built for. If a step slides a
segment outside that span, the constraint asserts nothing, reported slack is zero, and a
feasibility gate will pass a penetrating trajectory. Clearance must be re-verified against the
true obstacle trajectory, not against the hulls the solver used.

## Architecture in one paragraph

Rust is the sole optimizer backend. One `scp_step` in the Rust core is both the batch iteration and
the debug step — debugger sessions observe the real run through an emitted trace, they do not
implement a second optimizer. Python handles request shaping, scenario definitions, JSON I/O, and
the debug UI server. See `CLAUDE.md` §After the paper for the rules this enforces (backend honesty,
geometry authenticity, one execution model).

## Extending

- **Add a scenario** — edit `spacetime_bezier/scenarios.py`, register in `SCENARIO_MAP`.
- **Change the optimizer** — `rust_optimizer/core/src/spacetime_optimizer.rs` (outer loop) or
  `spacetime_constraints.rs` (KOZ geometry), then rebuild with `maturin develop --release`.
- **Change the UI** — `figures/spacetime_bezier_interactive.html` or
  `figures/spacetime_bezier_opt_debug.html`.

Run `pytest` before opening a PR.
