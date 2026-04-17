# Project Spec: Space-Time Bezier Optimizer North Star

## Purpose
This document defines the architectural north star for the `feature/spacetime-bezier-proposal` branch.

It is not a bug list and not a temporary cleanup memo. It is the target shape the project should converge toward so that optimizer behavior, debugger behavior, and backend labeling remain trustworthy.

This spec is project-level. `DEBUGGER_SPEC.md` remains the debugger-focused contract, but it must conform to the execution and backend rules defined here.

## Problem Statement
The current branch has drifted into an inconsistent state where:

- backend intent and actual execution do not always match
- debugger orchestration is not the same as batch optimizer orchestration
- Python still shapes semantics even though the branch intent is Rust-first
- partial Rust observability makes some debugger views technically honest but practically misleading

The core problem is not one solver defect. It is a trust failure caused by architectural inconsistency.

## North Star
The project should have:

**one canonical optimizer execution model, one explicit backend policy layer, and one common trace schema**

More concretely:

- Rust is the canonical production optimizer for the space-time branch
- batch optimization and debugger playback are two views of the same execution, not two different execution models
- the debugger is a trace consumer, not a second optimizer
- explicit backend selection must never silently substitute another backend
- every displayed geometry object must come from the backend that actually ran, or be explicitly marked as derived visualization

## Primary Architectural Principle
The system must be designed so that:

```text
same request + same backend + same configuration
= same optimizer execution
= same result
= same traceable explanation
```

Debugging is not a separate algorithm. It is observability over the real algorithm.

## Canonical Execution Model
The canonical model is:

1. Normalize a problem request.
2. Select a backend through an explicit backend policy.
3. Run one optimizer engine.
4. Optionally attach an observer to collect an execution trace.
5. Return both the optimization result and backend-authentic metadata.

In other words:

```text
optimize(request, observer=None)
optimize(request, observer=trace_sink)
```

The presence of a debugger must not change solver orchestration, candidate acceptance logic, trust-region behavior, or finalization behavior.

## Backend Policy
Backend policy must be decided at the API boundary, not hidden inside the optimizer loop.

Required semantics:

- `backend="rust"` means Rust must run or the call fails
- `backend="python"` means Python must run or the call fails
- `backend="auto"` may fall back according to explicit policy
- any fallback must be surfaced in result metadata and in any emitted trace
- no debugger path may silently substitute Python for Rust

The meaning of a backend choice must be stable across:

- batch optimization
- debugger sessions
- tests
- CLI tools

## Backend Roles
### Rust
Rust is the canonical execution engine for the space-time optimizer branch.

Rust owns:

- the SCP outer loop
- KOZ supporting-surface generation used in production
- candidate generation and acceptance semantics
- trust-region clipping or acceptance policy
- final iterate selection policy
- authoritative trace emission for Rust runs

### Python
Python is allowed only as a temporary reference backend.

Python may be kept for:

- regression comparison
- geometry inspection during migration
- oracle-style tests

Python must not remain the hidden source of production semantics for the branch.

If Python and Rust disagree, that disagreement must appear as an explicit backend difference, not as silent behavior borrowing.

## Trace and Debug Model
The debugger must operate on an execution trace emitted by the real optimizer run.

Replay is acceptable.
Reconstruction from partial logs is acceptable only as a temporary bridge.
Fake stage synthesis from another backend is not acceptable.

The preferred model is an observer interface:

```text
OptimizerEngine.run(request, observer)
```

Where:

- a null observer yields normal batch mode
- a collecting observer yields debugger playback
- a future streaming observer may support live inspection without changing solver logic

## Required Execution Stages
The canonical trace model must preserve these conceptual stages:

- `init-guess`
- `segment-subdivision`
- `obstacle-geometry`
- `supporting-surface-generation`
- `constraint-assembly`
- `objective-assembly`
- `solver-call`
- `candidate-filter`
- `post-eval`
- `finalize`

Backends may differ in how much detail they expose at each stage, but not in whether a shown stage is authentic.

If a backend cannot expose full internals for a stage yet, the stage must still be represented with an explicit "not available for backend" style diagnostic.

## Common Trace Schema
All backends must emit a common frame/event schema sufficient for shared UI rendering and backend comparison.

Minimum event properties:

- backend
- stage
- iteration
- control points relevant to that stage
- obstacle and segment payloads
- solver payload
- candidate or acceptance payload
- diagnostics describing what is real versus derived

The schema must distinguish:

- current iterate
- raw candidate
- accepted iterate
- best feasible iterate
- final returned iterate

These are different concepts and must never be conflated.

## Solver Semantics Rule
Solver outcome and optimizer acceptance are separate layers.

The architecture must explicitly represent:

- raw solver status
- whether a candidate vector was returned
- whether the optimizer accepted that candidate
- why a candidate was rejected
- what iterate was actually advanced

This rule exists to prevent Python and Rust solver differences from being hidden inside debugger presentation.

## Geometry Authenticity Rule
For obstacle avoidance debugging, the system must preserve the actual geometry used to generate constraints.

For every active obstacle-segment interaction that produces KOZ rows, the architecture should make it possible to recover or emit:

- obstacle identity
- segment identity
- geometric object type
- closest point or closest world-tube point
- support point
- supporting normal including time coefficient
- lower bound
- row indices or row mapping
- current, candidate, and accepted margins where available

Showing only a worst-case summary is not sufficient as the long-term project target.

## Debugger Role
The debugger is a viewer over authentic execution artifacts.

It may add derived visualization such as:

- sampled curves
- segment highlighting
- world-tube meshes generated from authentic obstacle parameters
- plane rendering from authentic half-space coefficients

But it must not invent optimization state that the backend did not produce.

## Module Responsibility Target
The project should converge toward the following separation of concerns:

- public API layer: request normalization and backend policy only
- optimizer engine layer: actual SCP execution only
- backend adapters: Rust and Python implementations behind a common request/result contract
- trace layer: observer interfaces and event definitions
- debugger session layer: navigation over collected traces
- UI layer: visualization only

The public optimizer module should not simultaneously act as:

- backend policy switchboard
- production optimizer implementation
- debugger factory
- fallback semantics owner

## Testing Philosophy
Tests should reflect the intended architecture, not preserve accidental drift.

The important invariants to test are:

- explicit backend calls do not silently fall back
- batch and debug modes produce the same final result for the same backend
- traces identify the real backend used
- stage payloads are authentic for the backend that emitted them
- Python and Rust comparison tests are labeled as comparison tests, not hidden semantic coupling

## Non-Goals
This spec does not require:

- interactive pausing inside the Rust solver
- identical internal solver algorithms across Python and Rust
- keeping Python as a permanent production backend
- perfect trace completeness on day one

What it does require is honesty, stability, and one canonical execution story.

## Acceptance Criteria
The architecture is correct when the following statements are true:

- Running the optimizer normally or under the debugger does not change backend behavior.
- `backend="rust"` means Rust actually ran.
- Any fallback is explicit and inspectable.
- The debugger can explain a result by replaying authentic backend events.
- Displayed supporting surfaces correspond to geometry actually used by the backend.
- Missing backend detail is explicit rather than replaced with fake detail.
- Python, if still present, is visibly a reference backend rather than the hidden semantic authority.

## Practical Project Direction
When making design decisions, prefer the option that moves the codebase toward:

- Rust as the canonical engine
- one execution path with optional tracing
- one shared event schema
- explicit backend policy at the boundary
- debugger replay from authentic trace
- row-level geometry inspectability for KOZ constraints

Reject changes that deepen any of the following:

- silent backend substitution
- debugger-only orchestration
- Python-defined production semantics for Rust runs
- backend-specific frame formats with incompatible meanings
- visualizations that cannot be tied back to actual solver geometry
