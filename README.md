# Space-Time Bezier Trajectory Sandbox

A personal research / debug sandbox for **space-time Bezier trajectory optimization**. The premise being probed: lifting moving obstacles into space-time (adding time as an explicit Bezier coordinate) turns them into static tubes, so the existing convex-hull / supporting-half-space machinery should handle moving-obstacle avoidance directly. The sandbox exists to stress that premise — pose problems, find where the optimizer breaks, prototype fixes.

Drag obstacles, tweak parameters, watch the optimizer re-solve.

**Current focus (until 2026-08-19): a conference paper on the space-time formulation.** Sandbox feature work is paused. For the goal, workstreams, known solver defects, and architecture, see [`CLAUDE.md`](CLAUDE.md) — it is the single source of direction for this branch.

## Interacting with the sandbox

Every obstacle, plus the start (green) and goal (red) markers, is a grab-handle in the 3D view:

- **L-click + drag** — slide the handle across (x, y) at its current t.
- **Shift + L-click + drag** — slide along t at fixed (x, y). On an obstacle's end control point this shrinks/extends its active time window; on the goal marker it re-scales scenario T live.
- **Escape** while dragging — cancel and revert to the pre-drag position.
- **Right-click drag** — pan the camera. **L-click drag on empty space** — rotate. **Shift + L-click drag on empty space** — pan. **Scroll** — zoom.

The re-solve fires automatically on release (debounced ~150 ms). The trajectory dims while a drag is in progress or a solve is in flight so you can tell the current curve is stale.

## Quickstart

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Build the Rust optimizer (one-time, requires a Rust toolchain + maturin)
cd rust_optimizer/pybind && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd ../..

# 3. Launch the interactive sandbox (live re-solve on every slider change)
python3 -m spacetime_bezier
# opens http://127.0.0.1:8767/ automatically

# 4. (Optional) Regenerate the pre-baked scenario JSON for file:// viewing
python3 -m spacetime_bezier.io
# then: open figures/spacetime_bezier_interactive.html

# 5. (Optional) Launch the live step-through debugger
python3 tools/spacetime_opt_debug.py
# then open figures/spacetime_bezier_opt_debug.html

# 6. Run tests
pytest
```

## Demo scenarios

Defined in `spacetime_bezier/scenarios.py`. Each is a 2D + time problem: moving obstacles, fixed endpoints, the optimizer plans both path *and* timing.

| Scenario | What it shows | Status |
|----------|---------------|--------|
| `original` (3 moving obstacles) | Basic proof of concept — curve threads between constant-velocity tubes in (x, y, t). | Feasible |
| `wall` | A wall that vanishes at a known time. The curve *should* "wait" in space-time then pass through, demonstrating time as a real optimization dimension. | **Infeasible** — the returned curve penetrates the wall by 0.112. The behavior this scenario exists to show does not currently work. |
| `diverse` (varied sizes / speeds / directions) | Stress case. | **Infeasible** — elastic relaxation keeps the solver running but no feasible path is found. |

Both failures trace to the two KOZ constraint defects documented in [`CLAUDE.md`](CLAUDE.md) under Established facts.

Scenarios are registered in `SCENARIO_MAP`; add your own by appending to `scenarios.py`.

## What's in the repo

- `spacetime_bezier/` — Python package. Public API, constraint builders, geometry, scenarios, debug stepper, JSON I/O.
- `rust_optimizer/` — Rust SCP optimizer (Clarabel QP + elastic relaxation). `core/` is the engine, `pybind/` is the PyO3 binding.
- `tools/` — `spacetime_opt_debug.py` (debug UI HTTP server), `compare_backends.py` (scenario diff tool).
- `figures/` — Interactive HTML demos and generated JSON outputs.
- `orbital_docking/` — **Legacy module.** Earlier orbital-rendezvous work (Bezier + spherical Earth KOZ). Still imports dimension-agnostic building blocks (`bezier.py`, `de_casteljau.py`) that the spacetime code reuses. Not the current headline.
- `tests/` — `pytest` suite for both modules.

## Architecture in one paragraph

Rust is the sole optimizer backend. A single SCP `scp_step` function in the Rust core is both the batch iteration and the debug step — debugger sessions observe the real run via an emitted trace, they don't implement a second optimizer. Python handles request shaping, scenario definitions, JSON I/O, and the debug UI server. See [`CLAUDE.md`](CLAUDE.md) §After the paper for the foundational rules (backend honesty, geometry authenticity, one execution model) that these choices enforce.

## Contributing / extending

- **Add a scenario**: edit `spacetime_bezier/scenarios.py`, register in `SCENARIO_MAP`.
- **Change the optimizer**: edit `rust_optimizer/core/src/spacetime_optimizer.rs` (outer loop) or `spacetime_constraints.rs` (KOZ geometry), then rebuild with `maturin develop --release`.
- **Change the UI**: edit `figures/spacetime_bezier_interactive.html` or `figures/spacetime_bezier_opt_debug.html`.

Before opening a PR, run `pytest`.
