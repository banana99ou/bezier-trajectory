# Paper claim, method, and demo scenarios

Workstream C output. Rename freely — the name is not load-bearing.

Supporting evidence lives in `doc/refs/c1_novelty.md` (prior-art map, comparison against Osburn,
source verification status) and `doc/notes/_shared/c3_safe_corridor_refs.md` (ground truth on
using one plane per segment and obstacle). This file is the short version: what we claim, why it
is different, how it works, and what could make it false.

---

## 1. The claim

> A decomposition-free space-time formulation in which each Bezier segment's time extent is
> itself a decision variable, so the obstacle's convex over-approximation is rebuilt from the
> current iterate on every successive-convexification iteration rather than being fixed before
> the solve begins.

That single property is what permits timing to be optimised — including stopping and waiting —
and what keeps arbitrary known non-straight obstacle motion affordable.

Do not claim "time as a coordinate." That was published in August 2025 by Osburn, Peterson and
Salmon, along with the space-time lift, finite-height tubes, the Bezier convex-hull argument in
the lifted space, and time monotonicity on control points. Four of the five items listed in the
project instruction file under "What is genuinely new" appear in that paper.

Do not claim speed either. Osburn solves a single-moving-obstacle case in about half a second
using the same solver we use. Any advantage on small problems is a small factor, not an order of
magnitude. The argument is how cost scales with dimension, not wall-clock time on a
two-dimensional toy problem.

## 2. What is different

Everyone else does one of two things that we do not.

| approach | free space | obstacle over-approximation | time allocation |
|---|---|---|---|
| Osburn, and the graph-of-convex-sets family generally | decomposed in advance into convex cells, plus a combinatorial layer to choose among them | not used | chosen, but only inside static cells |
| MADER, the semantic-corridor work, and driving-domain corridor methods | none | convex hull over a fixed time interval | fixed by the spline knots |
| ours | none | convex hull over an adaptive time interval | a decision variable |

Two consequences worth stating in the paper:

- MADER structurally cannot wait. Its time allocation is baked into the knot vector before the
  solve. Waiting is a behaviour that exists only when the time extent is free.
- The decomposition is a recurring cost, not a setup cost. Osburn's own sweep shows a seventeen
  per cent better trajectory costing seventeen times the compute, with free-space coverage still
  incomplete at the top of the sweep. Both halves of that machine — seeding convex regions by
  random sampling, then searching the resulting graph — degrade as dimensions are added, and we
  are adding one.

## 3. Method

1. Lift into the four-dimensional space of three spatial coordinates and time, with time as an
   explicit Bezier coordinate. The convex-hull property then holds verbatim in four dimensions
   with no side conditions, unlike corridor methods that write position as a function of time,
   which need an extra condition because the spatial axes and the temporal axis are not
   equivalent.
2. On each successive-convexification iteration, and for each pairing of one segment with one
   obstacle: take a convex over-approximation of the obstacle over that segment's current time
   span, emit exactly one supporting half-space, and apply it to every control point of that
   segment. One plane per segment-and-obstacle pair, never one per control point. In the lifted
   space the plane's normal is the spatial unit normal paired with a time component equal to
   minus the dot product of that spatial normal with the obstacle's velocity. The time component
   is not optional; it is what makes the plane support the tube at all.
3. For non-straight obstacle motion, adopt MADER's construction: sample the obstacle's known
   positions across the span, inflate each sample, and take the convex hull of the result. The
   whole tube need not be convex — only the slice overlapping one segment's span.
4. Enforce time monotonicity on the control points' time components, with a minimum separation.
5. The same construction covers all three constraint types: straight keep-out tubes, bent tubes
   from zigzagging obstacles, and shadow cones as described in section five.

Design rule: the segment count must be high enough that each segment's span sees roughly monotone
obstacle motion. A span covering a direction reversal produces a fat hull. This is the same
monotone knob as everywhere else — more segments give tighter hulls and a strictly larger
feasible set.

## 4. Demo scenarios

These replace the three scenarios listed in the project instruction file.

### Scenario one — wait for the door

The curve must stop and wait for a time-gated obstacle to clear. This proves the formulation:
time is a real optimisation dimension, and here is a behaviour that requires it.

Who else can wait, verified:

| method | waits? | output |
|---|---|---|
| Safe Interval Path Planning, Phillips and Likhachev, 2011 | yes — its edges are combined wait-then-move actions that jump through multiple time steps to a time when an obstacle has passed | discrete grid path |
| Space-Time Rapidly-exploring Random Tree, 2022 | yes | sampled path |
| MADER | no — the knots fix the timing | smooth trajectory |
| Osburn, and graph-of-convex-sets methods generally | not demonstrated | smooth trajectory |
| the temporal-logic graph-of-convex-sets paper | no — it does not describe robots waiting in place, and achieves timing compliance through path selection instead; its "doors" are logical key-precedence, not time gates | smooth trajectory |

Safe Interval Path Planning assumes that dynamic obstacles occupy locations during known time
intervals, which is our exact assumption. Same assumption, different output class. The sentence
for the paper: waiting exists in discrete and sampling-based planners under exactly our
assumptions, and does not exist in the continuous-optimisation family, because that family fixes
timing before solving.

### Scenario two — sneak past a radar

Hide inside the shadow cast by zigzagging moving obstacles. The mirrored variant is to avoid
those shadows in order to maintain line of sight to a ground controller. This proves the method
choice: a nonconvex, time-varying mission constraint that a decomposition-based method would have
to re-carve around.

Observer model: binary line-of-sight occlusion from a point source. No radar equation, no
detection probability. The claim is extensibility to real radar avoidance, not accurate radar
avoidance.

Prior art on the stealth half is games and graphics, and discrete — stealthy path planning against
dynamic observers, corridor-map stealth planning, covert planning against imperfect observers. All
of them use static terrain as the occluder. No continuous trajectory optimisation hiding behind a
moving occluder was found.

## 5. The shadow lemma

Statement. Let the observer be a single point, and let the obstacle be any convex body that does
not contain the observer. Define the shadow as the set of all positions whose straight line of
sight back to the observer is interrupted by the obstacle. Then the shadow is convex.

Proof, in words. Measure everything from the observer. A position lies in the shadow exactly when
you can reach it by starting at the observer, travelling to some point of the obstacle, and then
continuing outward along that same ray without turning — that is, the position is some obstacle
point stretched away from the observer by a factor of at least one. So the shadow, measured from
the observer, is the set of all obstacle points stretched by any factor of one or more.

Now take two shadow positions and form any weighted average of them, with non-negative weights
summing to one. Each of the two is an obstacle point multiplied by its own stretch factor. Let
the combined stretch factor be the same weighted average of the two stretch factors; since each
of them is at least one, so is that average. Divide the weighted average of the two positions by
the combined stretch factor. What remains is a weighted average of the two original obstacle
points, again with non-negative weights summing to one, so it is itself a point of the obstacle
because the obstacle is convex. The weighted average of the two shadow positions is therefore an
obstacle point multiplied by a stretch factor of at least one — which is exactly the description
of a shadow position. So the average lies in the shadow, and the shadow is convex.

This holds for any convex obstacle in any number of dimensions, so it covers spheres in three
spatial dimensions unchanged. It was derived here, not taken from a source. It is elementary and
independently checkable, but if it appears in the paper it should be stated as our result rather
than cited to someone else.

Consequence — hiding is cheaper than avoiding:

| constraint | region | convex? | machinery |
|---|---|---|---|
| stay out of the keep-out zone | outside a tube | no | supporting half-space plus a committed side |
| stay out of the shadow, to keep the controller link | outside a cone | no | supporting half-space plus a committed side — the same code |
| stay inside the shadow, to hide from the radar | inside a cone | yes | plain linear rows, with no side to commit and no passing class |

Space-time caveat. At each instant the shadow is convex, but the cone swings as the obstacle
moves, so the shadow volume in the lifted four-dimensional space is not convex. The remedy is the
same as for the bent tube: over a segment's span, require the position to lie in the intersection
of the shadows across that span. An intersection of convex sets is convex, and it is a sound inner
approximation, because being inside all of them implies being inside the one at the actual time.
Conservative, and it tightens with segment count.

Two honest failure modes, both physically correct. The intersection goes empty if the obstacle
moves far within one span — you cannot hide behind something that has left. And the choice of
which obstacle to hide behind, in what order, is a discrete choice with the same combinatorial
structure as the passing class. That is where solver item B6, procedural seeds and multi-start,
earns its place.

Suggested demo form: a hard constraint requiring the vehicle to stay hidden for the whole
traverse, rather than minimising exposure. It is convex, it is a one-line claim, and the figure
reads at a glance.

## 6. The risk this contribution creates

Read this before implementing solver item B7, the feasibility gate.

An adaptively-built hull is valid only inside the time span it was built to cover. If the hull is
constructed for a span taken from the current iterate, and the next iterate's segment slides
outside that span, the constraint becomes silently meaningless — the curve occupies a time the
hull never accounted for. The solver then reports total slack as zero on a constraint that
asserts nothing, and the feasibility gate passes a trajectory that penetrates an obstacle.

Two conditions, neither of them automatic:

1. the segment's control-point time components must stay within the span its hull covers, either
   by explicit constraint or by a trust region tight enough to guarantee it; and
2. the feasibility gate must re-verify clearance against the true obstacle trajectory, not
   against the hulls the solver used.

This makes the trust region from solver item B5 — porting the successive-convexification
machinery from the main branch, landed in commit 848bf3b — correctness-critical rather than a
convergence aid. Without it, nothing keeps the iterate inside the region where its own
constraints are valid.

Also for the solver workstream: the door scenario needs the finite-height tube, meaning the
branch of the keep-out-zone builder that clips against an obstacle's start and end times. The
project instruction file records that obstacle time windows default to plus and minus infinity,
which makes that branch unreachable. The scenario's headline behaviour may run through code that
never executes.

## 7. Decided against

- Convexifying free space by a coordinate transform, such as inversion, a conformal map, or a
  Nyquist-style transform. The obstruction is topological, not geometric: a convex set is simply
  connected, free space around an obstacle is not, and continuous invertible maps preserve that
  distinction. Inversion about the obstacle moves the hole to the origin; it does not remove it.
  The hole is the fact that passing left and passing right are genuinely distinct plans. The
  strongest published version of this idea is Rimon and Koditschek's navigation functions, whose
  titles concede the point — geometrically complicated but topologically simple spaces — and
  whose convergence guarantee holds only from almost all initial conditions, the excluded set
  being the saddle points that topology forces to exist.
- Receding-horizon replanning. Obstacle motion is deterministic and known, so there is nothing to
  re-estimate.
- The radar equation, detection probability, and target fluctuation models. Binary occlusion only.
- Competing on supporting more constraint types. That is Osburn's own stated contribution, so
  competing there head-on loses. Compete instead on the kind of constraint — nonconvex and
  time-varying — that a convex-cell framework cannot absorb.

## 8. What must be measured

No current performance number exists. The benchmarks file predates commit 848bf3b and the
uncommitted solver work on top of it. Its numbers describe a solver that no longer exists and
must not be quoted.

To be measured once the two constraint defects are fixed, meaning solver items B1 and B4:

- the door scenario demonstrably waiting;
- the stealth scenario hidden from start to finish;
- minimum clearance verified against the true obstacle trajectory, not against the solver's hulls;
- solve time against Osburn's half-second single-obstacle case and roughly four-second cluttered
  case, on the same solver and a comparable class of hardware.

## 9. Weakest links

Ranked by how much damage each would do if it turned out to be wrong.

1. MADER's fixed-knot time allocation is the pivot of section two, and it reached us through a
   fetch summariser rather than the source document. The solver workstream read MADER in full for
   references item C3 and its quoted equation is consistent, but that specific sentence is
   unconfirmed. Confirm it before paper item A2, the formulation section, relies on it.
2. The shadow lemma is ours, and nobody else has reviewed it.
3. "No paper does this" is weaker than a proof. No decomposition-free space-time formulation was
   found, but absence of evidence is not proof of absence.
4. Erdmann and Lozano-Perez, 1987, is unverified — the archived copy is a scan with no text
   layer. It is a lineage citation only, and nothing load-bearing rests on it.
