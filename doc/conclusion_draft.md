# Conclusion Draft

## 8. Conclusion

This paper presented a Bézier-based trajectory-initialization framework for continuous spherical-keep-out-zone avoidance. The method operates entirely in control-point space, using structured derivative operators and De Casteljau subdivision to construct conservative supporting-half-space constraints on Bézier sub-arcs. The resulting optimization is solved as a sequence of convex quadratic subproblems within a successive-convexification loop.

The framework was demonstrated on a simplified orbital-transfer problem. Ablation studies over subdivision count showed a clear computation-burden increase with finer subdivision, while the safety margin and objective-aligned effort remained effectively flat under the current metric set and scenario. Ablation studies over Bézier degree showed that all tested orders produced feasible trajectories under the boundary-conditioned protocol, with modest differences in effort and a clear runtime penalty at higher degree. The evidence supports a trade-off interpretation rather than a blanket claim that more subdivision or higher degree is universally better.

[PLACEHOLDER: If T6 is completed, add a paragraph summarizing the downstream warm-start comparison result and its interpretation boundary.]

The formulation is geometric and application-agnostic in construction, but the present evidence comes from a single domain demonstration. The paper does not claim superiority over direct collocation, true delta-v optimality, or validated cross-domain portability. The trajectories produced by this framework are intended as smooth, safety-respecting warm starts for downstream solvers. The method's value lies in the combination of continuous geometric safety enforcement, low-dimensional control-point-space parameterization, and structured convex subproblem assembly, rather than in any single one of these properties in isolation.
