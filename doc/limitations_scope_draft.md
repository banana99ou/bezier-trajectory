# Limitations and Scope Draft

## 7. Limitations and Scope

This section closes the paper's overclaim loopholes after the evidence has been presented. Its purpose is to separate what has been demonstrated from what has been suggested, and to make the paper's boundary conditions explicit rather than leaving them for reviewers to discover.

### Spherical keep-out zone scope

The continuous-exclusion argument presented in Section 4.1 is tied to a spherical KOZ geometry together with the specific subdivision-and-supporting-half-space construction. The resulting safety statement is conservative and assumption-dependent: it requires that the KOZ is spherical, that each sub-arc uses a fixed supporting half-space constructed from that sub-arc's centroid, that all control points of the subdivided sub-arc satisfy the corresponding half-space inequality, and that the centroid does not coincide with the KOZ center. The paper does not claim that this construction generalizes to arbitrary obstacle geometry, non-spherical exclusion zones, or time-varying obstacles without additional formulation work. Extension to more general obstacle classes is possible in principle but has not been demonstrated.

### Surrogate objective interpretation

The paper-level objective is an IRLS-weighted L1-style proxy for control effort in the presence of affinely linearized gravity. It is suitable for generating smooth, feasibility-respecting trajectories, but it does not constitute a true delta-v-optimal objective. The method does not optimize fuel consumption directly, and the surrogate value should not be compared against true fuel-optimal benchmarks without explicit qualification. The affine gravity treatment is local: it linearizes the gravitational field at representative sub-arc positions, so the objective model is approximate even within each convex subproblem.

### Warm-start value versus demonstrated downstream improvement

The framework is designed to produce trajectories intended as warm starts for downstream solvers. [PLACEHOLDER: If T6 is completed, replace this paragraph with the demonstrated-value version.] The current paper does not include a completed downstream direct-collocation comparison. Therefore, warm-start usefulness remains an intended use rather than a demonstrated result. The distinction matters: a visually smooth trajectory is not the same as a measured improvement in downstream solver convergence, speed, or final solution quality.

### Single-domain evidence base

The formulation is geometric and dimension-agnostic in construction: the control-point-space operators, subdivision logic, and supporting-half-space constraints do not depend on orbital semantics. However, the empirical evidence in this paper comes from a single simplified orbital-transfer demonstration. The paper does not claim validated portability across domains. Any statement about application generality should be read as a property of the mathematical construction, not as a demonstrated fact supported by cross-domain experiments.

### No global optimality claim

The method solves a sequence of convex quadratic subproblems within a successive-convexification loop. Each subproblem is convex, but the outer loop does not guarantee convergence to the global optimum of the original nonconvex problem. The results should be interpreted as locally refined feasible trajectories, not as globally optimal solutions.

### Fixed transfer time

The present formulation uses a fixed transfer time. It does not optimize free final time, timing allocation, or waiting behavior. All derivative and objective quantities are scaled by the fixed time mapping $t = T\tau$. Problems requiring time-optimal or free-final-time solutions would need formulation extensions not addressed in this paper.

### Solver and convergence behavior

All reported runs use a maximum of 500 outer SCP iterations. Runs that reach this limit without meeting the convergence tolerance are reported with their terminal iterate rather than excluded. The convergence behavior of the outer SCP loop is empirically adequate for the tested scenarios but is not supported by a formal convergence proof.
