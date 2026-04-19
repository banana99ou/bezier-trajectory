# ToDo

Action items for moving toward the VISION north star. Ordering follows the phase plan — each phase gates the next.

## Phase 0 — Foundation hardening

- [x] Replace hardcoded Rust debug log path in `rust_optimizer/core/src/optimizer.rs` (`debug_log_solver`) with env-var-driven path (`BEZIER_DEBUG_LOG`); no-op when unset.
- [x] Audit `spacetime_bezier/optimize.py` for layer bleed: debugger factories moved to `rust_debug_stepper.py`; clearance utility moved to `geometry.py`; `optimize.py` is now public-API + batch orchestration only.
- [x] Verify sandbox and debugger share one execution path (both drive `spacetime_optimizer::scp_step`; added `test_batch_and_stepper_produce_identical_final_iterate` in `tests/integration/test_spacetime_optimize.py`).
- [x] Measure baseline solve latency for the default scenario — recorded in `BENCHMARKS.md`.
- [x] Set solve-latency budget (≤200 ms for default scenario); default at 177 ms median, Phase 0 perf work not needed — Phase 1 unblocked.
- [x] Fix the two pre-existing `orbital_docking/` test failures (regenerated golden; dropped stale `pytest.warns` on removed warning).

## Phase 1 — Parameter interactivity

- [x] Add sliders to `figures/spacetime_bezier_interactive.html` for: `N`, `n_seg`, proximal weight, trust-region radius, time scaling α.
- [x] Wire sliders to a debounced re-solve (debounce window: ~150 ms). — Implemented via new `spacetime_bezier/sandbox.py` HTTP server (`python3 -m spacetime_bezier`) + client-side `scheduleSolve()` in the HTML.
- [x] Surface solver status (solved / infeasible / iteration count) in the UI. — `#solverStatus` panel.
- [x] Handle infeasible re-solves gracefully — keep last feasible trajectory visible, mark current as infeasible rather than blanking the view. — `lastFeasibleResponse` retention in `applySolveResponse()`.
- [ ] Confirm re-solve latency meets the Phase 0 budget under slider scrubbing. — Needs in-browser measurement with the server running.

## Phase 2 — Direct manipulation

Landed as a 5-commit series on 2026-04-20 (`e9d3311` → `c5d4f8b`). Obstacles travel over the wire as BezierObstacle (`{control_points: [[x,y,t], ...], radius, ...}`); internal APIs still see the legacy `{pos0, vel, r, t_start, t_end}` dict via a converter in `spacetime_bezier/geometry.py`. Degree-1 is a pure schema change (byte-identical parity verified); degree ≥2 raises `NotImplementedError` until the Rust KOZ builder is extended (Phase 3).

- [x] Add 3D picking for obstacles in the interactive viewer — per-frame screen-space pick targets built from the obstacle control points in BezierObstacle shape.
- [x] Implement drag for obstacle control points: L-click slides (x, y) at fixed t; Shift+L slides along t at fixed (x, y). Camera is locked during drag; Escape cancels; re-solve fires on release.
- [x] Add drag handles for start and goal endpoints: L-click slides xy; Shift+L on end edits scenario T live (start is pinned at t = 0). A unified pick pass chooses the nearest handle when obstacle CPs and endpoints overlap.
- [x] Visual feedback during drag: the trajectory is dimmed while a drag or solve is in flight; the existing status pill doubles as the solving spinner.
- [ ] Obstacle velocity handles — deferred. Velocity is now fully editable via the two control-point t-coordinates, so a separate velocity handle isn't load-bearing for the demo.

## Phase 3 — Diagnostic bridge

- [ ] Define a one-line infeasibility summary format: obstacle id, segment index, iteration, dominant violated row.
- [ ] Emit that summary from the Rust backend through the trace layer (extend `DebugFrame` if needed).
- [ ] Render the summary in a diagnostic panel in `spacetime_bezier_interactive.html`.
- [ ] Add an "Open in debug mode" button that deep-links to `figures/spacetime_bezier_opt_debug.html` with the failing run's trace preloaded.
- [ ] Verify the debugger UI can consume a trace produced by the sandbox (forces trace schema to be the real shared contract).

## Phase 4 — Persistence & comparison

- [ ] Design scenario JSON schema: obstacles (position, velocity, radius, `t_start`, `t_end`), endpoints, solver params, `N`, `n_seg`.
- [ ] Add "Save scenario" and "Load scenario" buttons in the sandbox UI (client-side file I/O is enough).
- [ ] Ensure a reopened scenario lands on identical solver state (same input → same output, per VISION §One canonical execution model).
- [ ] Add a side-by-side comparison mode: two sandbox panes, independent parameters, synchronized camera and time scrub.
- [ ] Highlight diffs between the two solves (trajectory deltas, KOZ activity differences).

## Phase 5 — Streaming observer (deferred)

- [ ] Define a streaming observer interface in Rust that emits trace frames live without buffering the full run.
- [ ] Stream frames to the UI via WebSocket or SSE from `tools/spacetime_opt_debug.py` (or a successor).
- [ ] Enable live inspection during a long solve without altering solver control flow.

## Cross-cutting / housekeeping

- [ ] Keep `VISION.md`, `README.md`, and `CLAUDE.md` in sync as phases land — move near-term-capability items to "done" as they complete.
- [ ] When removing deferred items, delete rather than leaving `// removed` stubs (per repo style).
- [ ] Update `tools/compare_backends.py` if the request schema changes in Phase 0.
