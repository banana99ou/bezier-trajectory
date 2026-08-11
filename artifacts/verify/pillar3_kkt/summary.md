# Pillar 3 -- KKT / feasibility at x* (phase70, phase120, phase135, phase170, planechange)

**PASS gate = primal feasibility**: dense-grid min‖r‖ ≥ r_e and velocity-BC residuals < 1e-6 (endpoints fixed by construction). Optimality is certified separately by **Pillar 1** (an independent COLD-start NLP, Rust-blind, reaches the same optimum; Rust sits above it by the conservatism gap → 0 as n_seg grows). `ratio_*` are the true- and surrogate-objective gradients projected onto the equality nullspace — **diagnostics only, expected nonzero and NOT gated**: the equality set omits the active convex-hull KOZ half-spaces (which hold the sampled curve ~11–60 km clear), so the projection intentionally excludes part of the true active set.

| scenario | n_seg | min_r | clearance | bc_v0 | bc_v1 | ratio_surrogate | ratio_true | feasible |
|---|---|---|---|---|---|---|---|---|
| phase70 | 8 | 6616.000 | 145.000 | 1.14e-16 | 5.79e-16 | 1.604e-07 | 3.060e-06 | True |
| phase70 | 16 | 6616.000 | 145.000 | 1.14e-16 | 5.79e-16 | 1.604e-07 | 3.060e-06 | True |
| phase70 | 32 | 6616.000 | 145.000 | 1.14e-16 | 5.79e-16 | 1.604e-07 | 3.060e-06 | True |
| phase120 | 8 | 6534.216 | 63.216 | 1.31e-16 | 6.87e-16 | 2.609e-01 | 2.609e-01 | True |
| phase120 | 16 | 6486.614 | 15.614 | 1.31e-16 | 6.47e-16 | 2.389e-01 | 2.389e-01 | True |
| phase120 | 32 | 6474.884 | 3.884 | 1.31e-16 | 6.47e-16 | 2.329e-01 | 2.329e-01 | True |
| phase135 | 8 | 6549.831 | 78.831 | 2.37e-16 | 6.47e-16 | 3.336e-01 | 3.336e-01 | True |
| phase135 | 16 | 6490.935 | 19.935 | 2.37e-16 | 6.47e-16 | 3.126e-01 | 3.126e-01 | True |
| phase135 | 32 | 6476.048 | 5.048 | 2.37e-16 | 6.47e-16 | 3.088e-01 | 3.088e-01 | True |
| phase170 | 8 | 6589.538 | 118.538 | 3.80e-16 | 5.79e-16 | 4.233e-01 | 4.233e-01 | True |
| phase170 | 16 | 6502.894 | 31.894 | 4.00e-16 | 2.89e-16 | 4.059e-01 | 4.059e-01 | True |
| phase170 | 32 | 6479.381 | 8.381 | 3.80e-16 | 2.89e-16 | 4.032e-01 | 4.032e-01 | True |
| planechange | 8 | 6540.166 | 69.166 | 2.57e-16 | 4.18e-16 | 2.672e-01 | 2.672e-01 | True |
| planechange | 16 | 6489.076 | 18.076 | 1.31e-16 | 3.67e-16 | 2.477e-01 | 2.477e-01 | True |
| planechange | 32 | 6475.539 | 4.539 | 2.57e-16 | 4.18e-16 | 2.436e-01 | 2.436e-01 | True |

- primal feasibility on all 15 cells (5 scenarios x 3 meshes): **True**

## VERDICT: PASS
<!-- provenance: commit=db97dcd1e4bca14279a909f9ea4da29fe3a8d583 dirty=0 ext=825716be634e -->

---

_produced by commit `db97dcd1e4bca14279a909f9ea4da29fe3a8d583`, Rust extension `825716be634e`; working tree clean under tools/verify, orbital_docking, rust_optimizer._
