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

- [ ] Add 3D picking for obstacles in the interactive viewer (raycast against tube geometry).
- [ ] Implement drag-on-ground-plane (or drag-in-view-plane) for obstacle position; re-solve on release.
- [ ] Extend drag to obstacle velocity (e.g. drag a velocity handle) so timing geometry can be edited.
- [ ] Add drag handles for start and goal endpoints; re-solve on release.
- [ ] Visual feedback during drag: ghost trajectory, or "solving…" spinner.

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
