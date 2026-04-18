# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch Goal: `integrate/rust-into-spacetime`

Build a **research sandbox / paper-pitch demo** for **space-time Bezier trajectory optimization for moving obstacle avoidance**. An interactive workbench where a user drags obstacles, tweaks parameters, and watches the optimizer re-solve.

For north-star direction (target UX, foundational rules, non-goals, near-term capabilities), see [`VISION.md`](VISION.md).

## Architecture

- **Rust is the sole optimizer backend.** The Python optimizer has been removed.
- The Rust backend uses Clarabel (interior-point conic solver) for QP subproblems.
- Elastic relaxation (slack variables on KOZ constraints) handles infeasible subproblems.
- The debugger is a trace consumer over Rust execution, not a separate optimizer.

## Core Idea

Lift 2D (or 3D) moving-obstacle avoidance into space-time by adding time as an explicit Bezier coordinate. A Bezier curve in (x, y, t) simultaneously plans **path and timing**. Key properties:

- Constant-velocity obstacles become **straight tubes** in (x, y, t) -- static geometry
- The existing Bezier convex hull property, De Casteljau subdivision, and supporting half-space constraints apply **directly** in the higher-dimensional space
- The optimizer minimizes spatial acceleration energy while threading the curve between obstacle tubes
- Time-limited obstacles (e.g. a wall that disappears) become **finite-height tubes** -- the curve can "wait" then pass through

## What Must Be Reused

The baseline `orbital_docking/` package has dimension-agnostic building blocks. **Import them, don't rewrite them:**

- `bezier.py` -- `get_D_matrix(N)`, `get_E_matrix(N)`, `get_G_matrix(N)`, `BezierCurve` class
- `de_casteljau.py` -- `segment_matrices_equal_params(N, n_seg)`
- `constraints.py` -- reference for how half-space constraints are built (adapt for moving obstacles)

What is genuinely **new** in the spacetime extension:
1. KOZ constraint builder lifts moving obstacles into static world-tubes in space-time. For each segment local control point, a supporting half-space is computed against the capsule geometry in (x, y, alpha*t) scaled space.
2. One Bezier coordinate is the time axis. The control points live in ordinary higher-dimensional Bezier space, so the existing convex-hull and supporting-half-space logic is reused with the time coordinate included in the constraint geometry.
3. Time-limited obstacles become finite-height tubes by clipping with `t_start` and `t_end`.
4. Time monotonicity constraint: `P[i+1, t] - P[i, t] >= min_dt`
5. Objective penalizes only **spatial** acceleration, not time-acceleration

## Important Implementation Warning

The intended model is a static obstacle in space-time, not a 2D obstacle re-evaluated on each time slice.

Bad pattern to avoid:
- Compute `pos0 + vel * t_seg`
- Build a normal using only spatial coordinates
- Emit a half-space with zero coefficient on the time coordinate

Correct pattern:
- Treat the obstacle as one object in `(x, y, t)` or `(x, y, z, t)`
- Build supporting half-spaces in the full lifted space so the time coordinate can appear in the plane equation

## Commands

```bash
# Launch the interactive sandbox (live re-solve on every slider change) — main entrypoint
python3 -m spacetime_bezier

# Regenerate the pre-baked scenario JSON (only needed for file:// viewing)
python3 -m spacetime_bezier.io

# Open the static interactive demo (uses pre-baked JSON; no live re-solve)
open figures/spacetime_bezier_interactive.html

# Run live optimizer step debugger
python3 tools/spacetime_opt_debug.py

# Compare Rust output across scenarios
python3 tools/compare_backends.py --all

# Run tests (requires Rust extension built)
pytest

# Build Rust extension
cd rust_optimizer/pybind && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

## Demo Scenarios

| Scenario | Purpose |
|----------|---------|
| Original (3 moving obstacles) | Basic proof of concept |
| Diverse (varied sizes/speeds/directions) | Shows generality (currently infeasible -- hard problem) |
| Disappearing wall | Curve "waits" until wall vanishes -- demonstrates time as a real optimization dimension |

## Key Files

### Python package (`spacetime_bezier/`)
- `__main__.py` -- Entrypoint: `python3 -m spacetime_bezier` → launches the sandbox
- `sandbox.py` -- Interactive sandbox HTTP server (live re-solve on slider change)
- `optimize.py` -- Public API: `optimize_spacetime()`, `optimize_scenario()`, debug stepper factories
- `constraints.py` -- Python KOZ constraint builder (used by Python debug stepper only)
- `rust_debug_stepper.py` -- Steps through actual Rust optimizer execution via debug log
- `debug_stepper.py` -- Python SCP stepper (retained for `clip_trust_region` utility)
- `debug_session.py` -- Stateful session for the step debugger UI
- `debug_trace.py` -- `DebugFrame` schema shared by both steppers
- `geometry.py` -- `MovingObstacle`, `bezier_curve()`, `obstacle_array_bundle()`
- `objective.py` -- Energy matrix and initial guess construction
- `scenarios.py` -- Scenario definitions and `SCENARIO_MAP` registry
- `io.py` -- CLI entrypoint, JSON I/O, interactive viewer launcher

### Rust optimizer (`rust_optimizer/`)
- `core/src/spacetime_optimizer.rs` -- SCP outer loop with elastic relaxation
- `core/src/spacetime_constraints.rs` -- KOZ capsule geometry, boundary, monotonicity, box constraints
- `core/src/optimizer.rs` -- Shared `solve_qp()` (Clarabel wrapper) and orbital docking optimizer
- `pybind/src/lib.rs` -- PyO3 bindings exposing `optimize_spacetime_bezier()`

### Tools
- `tools/spacetime_opt_debug.py` -- HTTP server for the live step debugger UI
- `tools/compare_backends.py` -- Runs scenarios and diffs results

### Figures
- `figures/spacetime_bezier_interactive.html` -- Interactive HTML demo with debug overlays
- `figures/spacetime_bezier_opt_debug.html` -- Live optimizer step debugger UI
- `figures/spacetime_scenarios.json` -- Optimized control points for all scenarios (generated)

## Known Issues

- Diverse scenario is infeasible across all configs (elastic relaxation keeps solver running but can't find a feasible path)
- `debug_stepper.py` (Python SCP stepper) is retained but no longer used as an optimizer backend; only `clip_trust_region` is imported from it
- Rust debug log path is hardcoded to a machine-specific location in `spacetime_optimizer.rs` and `spacetime_constraints.rs`
- Two pre-existing test failures in `orbital_docking/` tests: golden value drift and a removed warning check
