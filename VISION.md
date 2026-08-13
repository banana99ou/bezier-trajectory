# VISION

North-star direction for the `integrate/rust-into-spacetime` branch. This is what the tool should *feel like* when it works, and the foundational rules that make that feeling trustworthy.

For "what exists today / how to run it", see `README.md` and `CLAUDE.md`.

## North Star

**A personal research / debug sandbox for the space-time Bezier idea.** An interactive workbench the author uses to pose problems, surface weakpoints in the optimizer and the space-time formulation, and prototype new ideas against them. Not a tech demo, not a pitch artifact — the intended user is the person maintaining this branch.

The research claim under test: lifting moving obstacles into space-time turns them into static tubes, so the existing Bezier convex-hull / supporting-half-space machinery applies directly in the higher-dimensional space. The sandbox exists to *stress* that claim — find the cases where it bends or breaks, and iterate.

## Target user experience

What a working sandbox session looks like for the author:

- **Drag to pose the problem.** Grab an obstacle in 3D, drop it somewhere new, release — the optimizer re-solves and the trajectory updates. Same for endpoints.
- **Slide parameters live.** `N` (curve order), `n_seg` (segments), proximal weight, trust-region radius, time scaling α — all exposed, all re-solve on change.
- **"Why did it fail?" in one line.** When infeasible, a plain-English diagnosis: which obstacle, which segment, which iteration introduced the violation. Not a stack trace. Not raw solver output.
- **Pause and scrub time.** Step through the SCP iteration history, scrub the time axis of the solved trajectory, toggle obstacle tubes on and off.
- **Side-by-side comparison.** Two solves, different scenarios or different parameters, rendered together so the difference is the point of the view.
- **Scenario save/load.** A session can be serialized to JSON and reopened later — so an interesting failure case is reproducible, not lost.

The debugger (the step-through-each-SCP-stage view) is one mode within this sandbox — a drawer that slides in over the same scene, not a separate page. It's where you go when "why did it fail?" isn't enough.

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

### One frontend, two tempos

Batch mode and diagnostic mode are views on the same page, not separate apps. They share scenario state, 3D scene, camera, and server. Diagnostic mode is a side drawer (stage list, row-level KOZ inspector, iteration scrubber) that slides in — never a separate URL, never a page reload, never a re-imported scenario.

*Why it matters:* the core research loop (pose → observe → diagnose → iterate) breaks if "drop into debug" costs a reload or a scenario round-trip. Context loss — lost camera, lost drag history, lost "the thing I was just looking at" — kills the loop.

*Why drawer and not inlined:* batch tempo (slider → fast resolve → look at curve) and debug tempo (step, inspect row 42 of obstacle 3's plane equation, diff iterate N vs N+1) have genuinely different chrome needs. One giant view serving both clutters the batch case (most sessions); a drawer keeps the batch view clean and reveals debug tools on demand.

Today the repo has two HTML pages and two servers — this foundation is a target, not current state. Rotate toward it, not away from it.

### Rust is the sole engine

Rust (Clarabel QP + elastic relaxation for infeasible KOZ subproblems) owns:

- SCP outer loop
- KOZ supporting-half-space generation used in production
- candidate generation, acceptance, trust-region clipping
- final iterate selection
- authoritative trace emission

Python is no longer a backend. The Python SCP stepper (`spacetime_bezier/debug_stepper.py`) is dead code pending cleanup — nothing outside the file imports it, and it is not a reference solver or an oracle. If the sandbox ever shows a result, that result came from Rust.

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
- A separate debugger app / page / URL. Diagnostic mode is a drawer inside the sandbox, not its own tool.
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

The sandbox is working when the author, mid-investigation, can:

1. Open the tool, land on a scenario that solves, and start poking immediately — no boilerplate between thought and experiment.
2. Grab an obstacle, move it into a configuration suspected of breaking the solver, and watch the curve re-solve (or fail).
3. When something breaks, read the one-line diagnosis, drop into the debug view, and trace the failure down to the specific KOZ row / iteration that blew up — without leaving the browser.
4. Save the failing scenario, come back days later, and land on the exact same state to continue probing.

Everything else is in service of that loop: pose → observe → diagnose → iterate.

## Practical direction for design decisions

When choosing between options, prefer the one that moves toward:

- Rust as the canonical engine.
- One execution path with optional tracing.
- One shared event schema.
- Explicit backend policy at the boundary.
- Debugger replay from authentic trace.
- Row-level geometry inspectability for KOZ constraints.
- Interactivity over batch re-runs.
- One frontend with a diagnostic drawer over separate debug/batch pages.

Reject changes that deepen:

- Silent backend substitution (Python shouldn't come back as a hidden semantic authority).
- Debugger-only orchestration that diverges from the batch path.
- Visualizations that can't be tied back to actual solver geometry.
- UI features that require snapshotting state and rendering it offline rather than re-solving live.
- Separate debugger frontends that lose scenario state, camera, or viewport when switching modes.
