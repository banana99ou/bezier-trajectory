# DCM Experiment Findings

## Finding 1: Bézier representation fails for multi-revolution transfers

**Date**: 2026-04-16

### Observation

When applying the Bézier SCP optimizer to orbit-transfer cases with `T_normed > ~0.5` (more than half an orbital period), the initial Bézier curve passes through Earth even though all control points lie on valid orbits above the KOZ.

Tested on case 20 from `trajectories.duckdb`:
- `h0=400`, `delta_a=-78.5 km`, `delta_i=0.82°`, `T_normed=1.92` (1.92 orbits)
- 7 control points (degree 6) sampled along the departure orbit at 6778 km radius
- All control points correctly above KOZ (6528 km)
- The resulting Bézier curve dips to 160 km radius — deep inside Earth
- The curve also makes a retrograde loop

### Root cause

A degree-N Bézier curve has N+1 control points. The curve lies within the **convex hull** of these control points, not along the path connecting them.

For multi-revolution transfers, control points sampled ~100° apart around an orbit produce a convex hull that spans the full orbit diameter and contains the Earth's center. The polynomial curve is free to pass through the interior of this hull.

The KOZ constraint linearization (supporting half-spaces at segment centroids) cannot recover from this — the centroids are near the origin, so the normals point in unhelpful directions.

After optimization, the optimizer "solves" this by pushing middle control points to extreme distances (13 million km — 2× Earth-Moon distance) to force the polynomial to curve around Earth. This produces a mathematically feasible but physically meaningless trajectory.

### Implication for the experiment

The Bézier SCP optimizer is designed for sub-orbital maneuvers (the original docking scenario uses T=1500s, a fraction of one orbital period). For multi-revolution transfers, the single-arc polynomial representation is fundamentally unsuitable.

The DCM downstream experiment should restrict to cases where `T_normed` is small enough that the trajectory is a sub-orbital arc. Empirically, `T_normed < 0.5` is a reasonable cutoff — the trajectory covers less than 180° of orbital arc, so the convex hull of control points stays on one side of Earth.

### Possible extensions (not currently planned)

- Piecewise Bézier (multiple arcs, each covering < 180°)
- Higher degree curves with more control points
- Different parameterization (e.g., orbital-element-space Bézier)

### Evidence

- Visualization: `results/dcm_visualize/case_020_init_guess_3d.html`
- Radius vs τ plot: `results/dcm_visualize/case_020_radius_vs_tau.html`
- Experiment JSON: `results/dcm_test3/case_000020.json`

---

## Finding 2: Bézier warm-start does not improve DCM for short-transfer cases

**Date**: 2026-04-16

### Observation

Ran 4 short-transfer cases (`T_normed ≤ 0.5`) where the Bézier optimizer produces feasible trajectories. In all cases:

- Both pipelines converge to the same cost (identical to machine precision)
- The Bézier warm-start makes DCM **slower**, not faster (8–33× overhead)
- The Bézier optimizer itself is fast (~0.1s) but the overall pipeline is slower

| Case | T_normed | Baseline time | Proposed time | Cost delta |
|------|----------|--------------|---------------|------------|
| 2    | 0.280    | 0.62s        | 8.59s         | ~0         |
| 4    | 0.280    | 1.31s        | 33.20s        | ~0         |
| 6    | 0.280    | 0.46s        | 8.87s         | ~0         |
| 126  | 0.500    | 0.84s        | 12.04s        | ~0         |

### Root cause

Two independent issues:

1. **True anomaly mismatch.** The Bézier optimizer uses fixed endpoints at ν₀=0, νf=π. DCM treats ν₀ and νf as free decision variables and finds optimal departure/arrival points at completely different orbital positions (e.g., ν₀=281° for case 2). The warm-start trajectory begins at the wrong location on the orbit, forcing IPOPT to rework the entire solution.

2. **DCM's own initialization is already good.** The two-pass pipeline (cosine-blended Kepler guess + free ν₀/νf search + H-S → LGL refinement) is well-engineered for this problem class. The Bézier warm-start is solving a more constrained subproblem (fixed endpoints, position-only polynomial) and cannot compete with DCM's richer search space.

### Conclusion

For cases where DCM already converges reliably, a direct Bézier warm-start adds cost without benefit. The experiment as originally designed — "does the Bézier warm-start improve the final DCM answer?" — produces a negative result.

### Evidence

- Experiment results: `results/dcm_short_transfer/`
- Visualizations: `results/dcm_visualize/case_002_*.html`, `case_004_*.html`, `case_126_*.html`

---

## Finding 3: Revised experiment direction — Bézier as Pass 1 replacement

**Date**: 2026-04-16

### Motivation

Findings 1–2 show that the Bézier optimizer cannot beat DCM as a trajectory refiner. However, DCM's Pass 1 (Hermite-Simpson collocation) exists solely to discover the thrust peak structure so Pass 2 (Multi-Phase LGL) can set up the correct phase boundaries. This is expensive.

The Bézier optimizer produces a thrust profile in ~0.1s. If we can detect peaks from that profile and derive the phase structure directly, we can skip Pass 1 entirely.

### Revised pipelines

```
Baseline:   TransferConfig → Pass 1 (H-S) → peak detect → Pass 2 (LGL) → result
Proposed:   TransferConfig → Bézier SCP → peak detect → Pass 2 (LGL) → result
```

The comparison is no longer "does Bézier find a better answer?" but "can Bézier replace Pass 1 and save wall-clock time while preserving solution quality?"

### Why this is more promising

- Pass 1 is the most expensive stage (bulk of solve time in all tested cases)
- The Bézier optimizer runs in ~0.1s vs Pass 1's 0.3–5s
- The final answer comes from Pass 2 (LGL) either way — only the warm-start and phase structure differ
- Even if the final cost is identical, a speedup is a concrete, publishable result

### Implementation notes

- Extract thrust profile from the Bézier curve: `u(τ) = r''(τ)/T² - g(r(τ))`
- Run existing `detect_peaks()` on `‖u(τ)‖` to get peak count and locations
- Build phase structure via existing `determine_phase_structure()`
- Feed Bézier-sampled `(t, x, u)` directly to `MultiPhaseLGLCollocation` as warm-start
- True anomaly: use ν₀, νf from the Bézier boundary conditions (or search a small grid)

### Open question

The ν₀/νf mismatch from Finding 2 still applies. Options:
- Grid search over (ν₀, νf) in the Bézier optimizer (~64 runs × 0.1s = ~6s, still cheaper than Pass 1 for hard cases)
- Accept the suboptimal ν and let Pass 2's free-ν search correct it
