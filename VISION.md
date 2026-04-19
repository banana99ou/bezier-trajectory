# VISION

North-star direction for the `integrate/rust-into-spacetime` branch. This is what the tool should *feel like* when it works, and the foundational rules that make that feeling trustworthy.

For "what exists today / how to run it", see `README.md` and `CLAUDE.md`.

## North Star

**A research sandbox for the space-time Bezier idea.** An interactive workbench where a professor (or anyone pitching the paper) can drag moving obstacles around, tweak parameters, and watch the optimizer re-solve in real time. The tool *is* the demo — not a separate thing you run to generate figures for a demo.

The core research claim being exhibited: lifting moving obstacles into space-time turns them into static tubes, so the existing Bezier convex-hull / supporting-half-space machinery applies directly in the higher-dimensional space. The sandbox should make that claim feel obvious by letting the user poke the problem and see the optimizer respond.

## Target user experience

What a sandbox session should look like when fully built out:

- **Drag to pose the problem.** Grab an obstacle in 3D, drop it somewhere new, release — the optimizer re-solves and the trajectory updates. Same for endpoints.
- **Slide parameters live.** `N` (curve order), `n_seg` (segments), proximal weight, trust-region radius, time scaling α — all exposed, all re-solve on change.
- **"Why did it fail?" in one line.** When infeasible, a plain-English diagnosis: which obstacle, which segment, which iteration introduced the violation. Not a stack trace. Not raw solver output.
- **Pause and scrub time.** Step through the SCP iteration history, scrub the time axis of the solved trajectory, toggle obstacle tubes on and off.
- **Side-by-side comparison.** Two solves, different scenarios or different parameters, rendered together so the difference is the point of the view.
- **Scenario save/load.** A session can be serialized to JSON and reopened later — so an interesting failure case is reproducible, not lost.

The debugger (the step-through-each-SCP-stage view) is one mode within this sandbox, not the product itself. It's where you go when "why did it fail?" isn't enough.

## Foundations

These rules existed before the sandbox framing and remain load-bearing. They are what keep the sandbox honest — without them, "live re-solve" would drift into "live render of plausible-looking state".

### One canonical execution model

```text
same request + same backend + same configuration
= same optimizer execution
= same result
= same traceable explanation
```

Batch solves and debugger sessions are two views of the same execution, not two different execution models. Concretely: `scp_step` in the Rust core is both the batch iteration and the debug step — a debug session never simulates an alternative stepper.

### Rust is the sole engine

Rust (Clarabel QP + elastic relaxation for infeasible KOZ subproblems) owns:

- SCP outer loop
- KOZ supporting-half-space generation used in production
- candidate generation, acceptance, trust-region clipping
- final iterate selection
- authoritative trace emission

Python is no longer a backend. The Python SCP stepper (`spacetime_bezier/debug_stepper.py`) is retained only for the `clip_trust_region` utility; it is not a reference solver or an oracle. If the sandbox ever shows a result, that result came from Rust.

### Trace is observability over the real run

The debugger is a trace consumer over Rust execution. It is not a parallel optimizer.

- A null observer → normal batch mode.
- A collecting observer → debugger playback.
- A streaming observer (future) → live inspection with no change to solver logic.

Reconstruction from partial logs is acceptable as a temporary bridge. Fake stage synthesis from another backend is not.

### Geometry authenticity

Every geometry object shown in the UI must either come from the backend that actually ran (supporting planes, tubes at the α-scaled time axis, KOZ rows, closest points) or be explicitly marked as derived visualization (sampled curves, segment highlights, tube meshes rendered from authentic obstacle parameters).

No time-sliced-2D plane should be labeled as a 3D supporting surface.

### Layer separation

- **Public API**: request normalization, backend dispatch.
- **Optimizer engine**: SCP execution (Rust).
- **Trace layer**: observer interfaces, event definitions.
- **Debugger session**: navigation over collected traces.
- **UI**: visualization only.

The public optimizer module should not simultaneously be the production implementation, the debugger factory, and the fallback-policy owner. Today it mostly isn't — keep it that way.

## Diagnostic Mode

The sandbox's "step through the optimizer" view. When "why did it fail?" needs more than one line, the user drops into this mode.

### Core principles

- **Backend-authentic.** Every stage shown must come from the Rust run that actually happened, or be explicitly labeled as derived visualization. No invented state.
- **Geometry, not text.** Each stage has a visual: control polygons, obstacle tubes, supporting planes, active-segment highlights. Text-only stage dumps are a last resort, not the default.
- **Row-level KOZ inspection.** For any active obstacle-segment pair, the user can see: which obstacle, which segment, which local row, the exact plane equation including the time coefficient, `lhs - lb` for current / candidate / accepted iterates, and whether the row is violated.
- **Diffable layers.** Each stage shows what changed from the previous one.
- **Missing data is explicit.** If a backend can't expose a stage's internals yet, the stage is shown with "not available" — never filled in from somewhere else.

### Conceptual pipeline stages

The stages a trace should conceptually cover (detailed taxonomy lives in code — `rust_debug_stepper.py` stage names — not in this doc, so it doesn't drift):

init guess → segment subdivision → obstacle geometry → supporting-surface generation → constraint assembly → objective assembly → solver call → candidate filter → post-eval → finalize.

### Frame schema

Every frame a viewer consumes should carry enough to render the stage and diagnose the row. Minimum: backend label, stage name, iteration, current/candidate/accepted control points, per-segment control polygons and centroids, per-obstacle geometry, per-active-pair KOZ rows (obstacle, segment, row indices, normal including time coefficient, lower bound, margins), solver status (raw + interpreted), trust-region state, diagnostics distinguishing authentic from derived.

The schema must distinguish *current iterate / raw candidate / accepted iterate / best feasible iterate / final returned iterate*. These are different concepts; never conflate them.

## Non-goals

- A stage-by-stage formal debugger spec — the old `DEBUGGER_SPEC.md` 10-stage taxonomy now lives in code, not in prose.
- Polished production UI. This is a workbench; ugly-but-honest beats pretty-but-fake.
- Shareable static reports / exported PDFs. The live tool is the artifact.
- Python as a permanent reference backend. Already removed; should stay removed.
- Interactive pausing inside the Rust solver. Trace replay is enough.
- Identical internal algorithms across backends (there is only one backend now).

## Near-term capabilities needed

The sandbox is not all there yet. Concretely missing:

- **Higher-degree Bezier obstacles** — the wire format already carries a list of control points per obstacle; degree ≥2 needs the Rust KOZ builder to de Casteljau-subdivide obstacles the same way trajectory segments already are.
- **Diagnostic panel** — one-line infeasibility summary, with a "open in debug mode" affordance.
- **Scenario save/load** — serialize the current sandbox state, reopen later.

Landed already (keeping for context until this doc is rotated):

- Free-form `N` / `n_seg` inputs and parameter sliders (prox, trust, α, cap-roundness) with debounced re-solve.
- Live obstacle editing via draggable control-point handles (L-click xy at fixed t; Shift+L along t). Start/end handles drag the same way, with Shift+L on end re-scaling scenario T live.

Other items (comparison mode, Rust-internal stage exposure beyond what the debug log currently captures) are desirable but secondary.

## Acceptance for the north star

The sandbox is working when a first-time user — say a professor — can:

1. Open the tool in a browser, see a default scenario solving.
2. Grab an obstacle, move it, watch the curve re-solve.
3. Make the problem infeasible on purpose, read the one-line diagnosis, and step into the debug view to see which KOZ row blew up.
4. Save the scenario, close the tab, reopen it, and land on the same state.

Everything else is in service of that.

## Practical direction for design decisions

When choosing between options, prefer the one that moves toward:

- Rust as the canonical engine.
- One execution path with optional tracing.
- One shared event schema.
- Explicit backend policy at the boundary.
- Debugger replay from authentic trace.
- Row-level geometry inspectability for KOZ constraints.
- Interactivity over batch re-runs.

Reject changes that deepen:

- Silent backend substitution (Python shouldn't come back as a hidden semantic authority).
- Debugger-only orchestration that diverges from the batch path.
- Visualizations that can't be tied back to actual solver geometry.
- UI features that require snapshotting state and rendering it offline rather than re-solving live.
