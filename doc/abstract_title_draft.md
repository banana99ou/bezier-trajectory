# Abstract and Title Draft

## Title

**Bézier-Based Continuous-Safety Trajectory Initialization via Segment-Wise Convexification**

Subtitle (optional, if venue allows): Demonstration on Orbital Transfer

## Abstract

This paper presents a trajectory-initialization framework based on Bézier curves for constrained motion planning with continuous keep-out-zone avoidance. The method operates entirely in control-point space: the decision variables are the Bézier control points, and derivative operators, subdivision matrices, boundary conditions, and keep-out constraints are all expressed as linear maps on those control points. Continuous avoidance of a spherical keep-out zone is enforced conservatively by partitioning the curve into sub-arcs via De Casteljau subdivision and imposing supporting-half-space constraints on each sub-arc's control polygon. The convex-hull property of Bézier curves guarantees that the entire sub-arc lies outside the keep-out zone whenever its control points satisfy the half-space inequality. The resulting optimization is solved as a sequence of convex quadratic subproblems within a successive-convexification loop.

The framework is demonstrated on a simplified orbital-transfer problem. Ablation studies over subdivision count and Bézier degree characterize the computation-versus-approximation trade-off: finer subdivision increases runtime without strongly affecting the safety margin or effort metric under the tested scenario, while degree changes produce modest performance differences with a clear runtime penalty at higher order. [PLACEHOLDER: If T6 is completed, add one sentence on the downstream warm-start result.] The formulation is geometric and application-agnostic in construction, though the present evidence comes from a single domain. The method does not claim superiority over direct collocation, global optimality, or true fuel optimality. It is intended to generate smooth, continuously safe warm starts for downstream trajectory-optimization solvers.
