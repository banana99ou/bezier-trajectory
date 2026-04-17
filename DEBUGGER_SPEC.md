# Optimizer Debugger Spec

## Purpose
The optimizer debugger exists to let us inspect, visually and step by step, how a space-time optimization problem is formed and solved.

It should make it possible to answer questions like:
- Is segmentation working as intended?
- Are obstacle constraints built from true lifted 3D supporting surfaces, or from time-sliced 2D approximations?
- Which pipeline layer is causing failure or distortion?
- How much does each layer change the trajectory?
- Where do Rust and Python differ, if both are supported?

## Primary Goal
Make every meaningful layer of the optimization pipeline observable, with enough geometry and metadata to diagnose formulation bugs.

## Non-Goals
- Not just "show final trajectory"
- Not just "show solver iterations"
- Not a backend-specific toy
- Not a fake stepper that claims Rust but actually runs Python

## Core Principles
- Truthful backend semantics: if backend is `rust`, every stage shown must come from Rust data or be explicitly marked as derived visualization.
- Stable frame schema: the viewer consumes one common frame format across backends.
- Stage fidelity over convenience: missing stage data is acceptable; fake stage data is not.
- Visual-first debugging: every stage should have geometry, not just text.
- Diffable layers: each stage should show what changed from the previous stage.

## Supported Backends
- `rust`
- `python` only as a temporary reference/debug oracle, if still present

The UI must always display the active backend prominently.

### Backend Rule
If a backend cannot expose a stage yet:
- show the stage name
- show `not available for backend`
- do not silently substitute another backend

## Required Pipeline Stages
These are conceptual stages. Backend implementations may expose them differently, but the viewer contract should preserve them.

### 1. `init-guess`
- Initial control polygon
- Initial sampled curve
- Initial clearance
- Problem dimensions and config

### 2. `segment-subdivision`
- Equal-parameter subdivision result
- Per-segment control polygons
- Segment centroids
- Segment sampled curves
- Segment index selection UI

### 3. `obstacle-geometry`
- Raw obstacle representation used by the backend
- For moving obstacles: world-tube / capsule / swept geometry in space-time
- Time clipping by `t_start`, `t_end`
- Explicit distinction between:
  - intended 3D geometry
  - approximated geometry actually used

### 4. `supporting-surface-generation`
- Supporting half-surface / half-space generation
- For each active obstacle-segment interaction:
  - closest point / support point
  - surface normal
  - time coefficient
  - lower bound
  - which geometric object generated it
- Must visually show plane orientation in space-time
- Must make it obvious if the backend is using time-sliced planes

### 5. `constraint-assembly`
- Boundary constraints
- Time monotonicity constraints
- KOZ constraints
- Optional slack constraints if present
- Counts and shapes
- Mapping from obstacle/segment to rows

### 6. `objective-assembly`
- Objective type
- Proximal term
- Trust-region term or clip policy
- Which coordinates are penalized
- Explicitly show whether time acceleration is penalized

### 7. `solver-call`
- Solver backend (`Clarabel`, `trust-constr`, etc.)
- Solver status
- Returned candidate availability
- Whether result is exact, approximate, failed, infeasible, or fallback

### 8. `candidate-filter`
- Trust region clipping or acceptance logic
- Slack restoration or feasibility filtering
- Candidate vs accepted trajectory

### 9. `post-eval`
- Delta norm
- Clearance
- Feasible/infeasible
- Best feasible iterate tracking
- What changed from previous iterate

### 10. `finalize`
- Final returned trajectory
- Last iterate
- Best feasible iterate
- Why that final one was selected

## Required Visual Layers
Each frame should support toggling these layers:
- Current control polygon
- Previous control polygon
- Candidate control polygon
- Accepted control polygon
- Best feasible control polygon
- Sampled trajectory
- Per-segment sampled curves
- Segment local control polygons
- Obstacle centerlines / capsules / world-tubes
- Supporting planes / half-spaces
- Closest point / support point markers
- Violated constraints highlighted
- Active obstacle-segment pairs only

## Segmentation Debug Requirements
For each segment:
- Show `A_seg @ P`
- Show local segment control points
- Show centroid
- Show sampled local curve
- Show which obstacles are active for that segment
- Show which generated half-spaces come from that segment
- Show whether local segment points lie on the expected side of the generated planes

### Segmentation Sanity Checks
The debugger should make these obvious:
- segment count matches requested `n_seg`
- segments cover the full curve continuously
- local control polygons align with sampled global curve
- centroids move monotonically in time
- increasing `n_seg` changes local geometry in understandable ways

## Constraint and Geometry Diagnostics
For each generated KOZ row, the debugger should be able to answer:
- Which obstacle created this row?
- Which segment created this row?
- Which local control point / row index does it apply to?
- What is the exact plane equation?
- What is `lhs - lb` for:
  - current iterate
  - candidate iterate
  - accepted iterate
- Is this row violated?
- If violated, by how much?
- Is this violation local or systematic?

This is the part that should expose a time-sliced-vs-3D-supporting-surface bug immediately.

## Frame Payload Contract
Minimum common payload fields:
- `backend`
- `stage`
- `iteration`
- `control_points`
- `segments`
- `obstacles`
- `koz`
- `metrics`
- `solver`
- `trust_region`
- `diagnostics`

### `segments`
Each segment should include:
- `segment_index`
- `control_points`
- `centroid`
- `centroid_time`
- `sampled_curve`

### `koz`
Each active obstacle interaction should include:
- `obstacle_index`
- `obstacle_name`
- `geometry_type`
  - `time_slice_disk`
  - `capsule`
  - `world_tube`
- `center` or closest-point data
- `support_point`
- `normal`
- `time_coefficient`
- `lower_bound`
- `row_indices`
- `lhs_current`
- `lhs_candidate` if available
- `margin_current`
- `margin_candidate` if available

### `solver`
Must include:
- solver name
- raw status
- interpreted status
- candidate available?
- accepted?
- reason for rejection if rejected

## Backend Semantics Spec
### Rust Backend
Must expose actual Rust behavior.
If Rust only exposes one outer iteration at a time, that is acceptable.
If Rust cannot expose internal half-space generation yet, show:
- `stage present but internal geometry unavailable`

Do not fake Python geometry.

### Python Backend
Allowed only as:
- reference backend
- regression oracle
- temporary comparison tool

If Python remains, it should be visually labeled as reference/debug-only.

## Comparison Mode
The debugger should eventually support:
- `rust`
- `python`
- `rust vs python`

Comparison mode should diff:
- segment centroids
- active obstacle sets
- generated half-space normals
- lower bounds
- constraint counts
- solver status
- accepted iterate
- clearance progression

## UX Requirements
- `Next stage`
- `Next iteration`
- `Prev`
- `Reset`
- Segment selector
- Obstacle selector
- Toggle raw data
- Toggle planes / tubes / curves / candidate / accepted / best
- Stage summary in plain English

### Summary Text Should Answer
- What stage am I looking at?
- What geometry was created here?
- What changed relative to the previous stage?
- Did anything become infeasible here?
- Which backend generated this frame?

## Correctness Requirements
The debugger is correct only if:
- backend label matches actual backend execution
- no silent fallback to another backend
- stage payloads are backend-authentic
- missing data is explicit
- shown geometry is the geometry actually used by the solver

## Immediate Gaps In Current State
Current implementation problems relative to this spec:
- debugger/backend honesty was broken
- Rust live stepping does not yet expose full internal stages
- Python and Rust stage fidelity are asymmetric
- current Rust visualization still lacks proper per-stage internal geometry exposure
- deprecated Python path still shapes too much of the debugger design

## Suggested Build Order
1. Fix backend honesty.
2. Define a stable frame schema.
3. Make Rust expose true internal stages:
   - segment subdivision
   - obstacle geometry
   - supporting surface generation
   - constraint assembly
   - solver result
4. Add row-level KOZ margin inspection.
5. Add comparison mode.
6. Remove Python as the primary debugger engine.

## Acceptance Criteria
The debugger is good enough when it can visually prove:
- whether KOZ uses time-sliced 2D planes or true lifted 3D supporting surfaces
- whether segmentation is behaving correctly
- which stage introduces infeasibility
- which backend-specific difference causes behavior divergence
- how much each stage changes the trajectory
