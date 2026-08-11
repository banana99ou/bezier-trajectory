# Pillar 1 -- Independent NLP cross-check (5 geometries)

SciPy trust-constr solves the same transfer as a direct NLP on the true
nonconvex objective with the true dense-grid KOZ, from a COLD start that
carries no Rust information. The Rust convex-hull KOZ is a subset of the
true feasible set, so the gap must be non-negative and must shrink with
n_seg -- a NEGATIVE gap would mean Rust cuts below the true optimum, i.e.
violates the true KOZ, and is gated as such.

A geometry where the independent solve does not itself converge yields
NO REFERENCE: there is nothing to compare against, and calling that a
solver failure would blame the code under test for the oracle's limits.
It is not a pass either -- the coverage line below says how many
geometries this pillar actually certifies.

| scenario | verdict |
|---|---|
| phase70 | **PASS** |
| phase120 | **PASS** |
| phase135 | **NO REFERENCE** |
| phase170 | **PASS** |
| planechange | **PASS** |

- independent reference obtained on 4/5 geometries: phase70, phase120, phase170, planechange (no reference: phase135)
- every referenced geometry agrees: **True**

## VERDICT: PASS

---

## phase70

- **Independent reference (projected start, Rust-blind)**: converged=True (optimality=9.65e-09 vs 1e-06, constraint violation=1.33e-15 vs 1e-06 km; exit status=1, 543 iterations of 1500), J_true=3.347149e-05, min_r=6616.000 km, feasible=True
- Warm start (from n_seg=64 Rust): status=1, J_true=3.347149e-05, min_r=6616.000 km; cold≈warm agree=True (|dJ|/J=0.000%); warm move off Rust=0.0000
- gap%(n_seg=16)=0.000  gap%(64)=0.000  (Rust conservative above the true optimum; shrinks)

| n_seg | iters | J_true_rust | J_true_scipy | gap% | min_r_rust | min_r_scipy |
|---|---|---|---|---|---|---|
| 8 | 6 | 3.347149e-05 | 3.347149e-05 | 0.000 | 6616.000 | 6616.000 |
| 16 | 6 | 3.347149e-05 | 3.347149e-05 | 0.000 | 6616.000 | 6616.000 |
| 32 | 6 | 3.347149e-05 | 3.347149e-05 | 0.000 | 6616.000 | 6616.000 |
| 64 | 6 | 3.347149e-05 | 3.347149e-05 | 0.000 | 6616.000 | 6616.000 |

**checks**: rust_feasible=True, cold_feasible=True, cold_converged=True, cold≈warm=True, close(0≤gap≤10%)=True, gap_shrinks=True → **PASS**

## phase120

- **Independent reference (projected start, Rust-blind)**: converged=True (optimality=9.69e-09 vs 1e-06, constraint violation=2.22e-15 vs 1e-06 km; exit status=1, 310 iterations of 1500), J_true=2.199909e-05, min_r=6471.000 km, feasible=True
- Warm start (from n_seg=64 Rust): status=1, J_true=2.199909e-05, min_r=6471.000 km; cold≈warm agree=True (|dJ|/J=0.000%); warm move off Rust=0.0001
- gap%(n_seg=16)=2.882  gap%(64)=0.175  (Rust conservative above the true optimum; shrinks)

| n_seg | iters | J_true_rust | J_true_scipy | gap% | min_r_rust | min_r_scipy |
|---|---|---|---|---|---|---|
| 8 | 8 | 2.476447e-05 | 2.199909e-05 | 12.570 | 6534.216 | 6471.000 |
| 16 | 8 | 2.263317e-05 | 2.199909e-05 | 2.882 | 6486.614 | 6471.000 |
| 32 | 8 | 2.215674e-05 | 2.199909e-05 | 0.717 | 6474.884 | 6471.000 |
| 64 | 8 | 2.203768e-05 | 2.199909e-05 | 0.175 | 6471.969 | 6471.000 |

**checks**: rust_feasible=True, cold_feasible=True, cold_converged=True, cold≈warm=True, close(0≤gap≤10%)=True, gap_shrinks=True → **PASS**

## phase135

- **Independent reference (straight-line start, Rust-blind)**: converged=False (optimality=9.45e-04 vs 1e-06, constraint violation=3.65e-07 vs 1e-06 km; exit status=0, 1500 iterations of 1500), J_true=8.307425e-05, min_r=6471.988 km, feasible=True
- Warm start (from n_seg=64 Rust): status=2, J_true=8.258906e-05, min_r=6471.000 km; cold≈warm agree=True (|dJ|/J=0.584%); warm move off Rust=0.0003
- gap%(n_seg=16)=2.201  gap%(64)=-0.413  (Rust conservative above the true optimum; shrinks)

| n_seg | iters | J_true_rust | J_true_scipy | gap% | min_r_rust | min_r_scipy |
|---|---|---|---|---|---|---|
| 8 | 9 | 9.262479e-05 | 8.307425e-05 | 11.496 | 6549.831 | 6471.988 |
| 16 | 9 | 8.490308e-05 | 8.307425e-05 | 2.201 | 6490.935 | 6471.988 |
| 32 | 9 | 8.316189e-05 | 8.307425e-05 | 0.105 | 6476.048 | 6471.988 |
| 64 | 9 | 8.273132e-05 | 8.307425e-05 | -0.413 | 6472.251 | 6471.988 |

**checks**: rust_feasible=True, cold_feasible=True, cold_converged=False, cold≈warm=True, close(0≤gap≤10%)=True, gap_shrinks=True → **NO REFERENCE**

  The independent solve did not reach a first-order point on this geometry, so there is no reference to compare the Rust result against. Every number above is reported; none of them is a verdict on the solver.

## phase170

- **Independent reference (projected start, Rust-blind)**: converged=True (optimality=9.73e-09 vs 1e-06, constraint violation=2.22e-15 vs 1e-06 km; exit status=1, 597 iterations of 1500), J_true=4.493266e-04, min_r=6471.000 km, feasible=True
- Warm start (from n_seg=64 Rust): status=1, J_true=4.493266e-04, min_r=6471.000 km; cold≈warm agree=True (|dJ|/J=0.000%); warm move off Rust=0.0006
- gap%(n_seg=16)=2.959  gap%(64)=0.181  (Rust conservative above the true optimum; shrinks)

| n_seg | iters | J_true_rust | J_true_scipy | gap% | min_r_rust | min_r_scipy |
|---|---|---|---|---|---|---|
| 8 | 168 | 5.071281e-04 | 4.493266e-04 | 12.864 | 6589.538 | 6471.000 |
| 16 | 12 | 4.626229e-04 | 4.493266e-04 | 2.959 | 6502.894 | 6471.000 |
| 32 | 12 | 4.525899e-04 | 4.493266e-04 | 0.726 | 6479.381 | 6471.000 |
| 64 | 12 | 4.501418e-04 | 4.493266e-04 | 0.181 | 6473.077 | 6471.000 |

**checks**: rust_feasible=True, cold_feasible=True, cold_converged=True, cold≈warm=True, close(0≤gap≤10%)=True, gap_shrinks=True → **PASS**

## planechange

- **Independent reference (projected start, Rust-blind)**: converged=True (optimality=8.65e-09 vs 1e-06, constraint violation=1.78e-15 vs 1e-06 km; exit status=1, 459 iterations of 1500), J_true=6.015869e-05, min_r=6471.000 km, feasible=True
- Warm start (from n_seg=64 Rust): status=1, J_true=6.015869e-05, min_r=6471.000 km; cold≈warm agree=True (|dJ|/J=0.000%); warm move off Rust=0.0003
- gap%(n_seg=16)=2.205  gap%(64)=0.134  (Rust conservative above the true optimum; shrinks)

| n_seg | iters | J_true_rust | J_true_scipy | gap% | min_r_rust | min_r_scipy |
|---|---|---|---|---|---|---|
| 8 | 10 | 6.583734e-05 | 6.015869e-05 | 9.439 | 6540.166 | 6471.000 |
| 16 | 10 | 6.148523e-05 | 6.015869e-05 | 2.205 | 6489.076 | 6471.000 |
| 32 | 10 | 6.048383e-05 | 6.015869e-05 | 0.540 | 6475.539 | 6471.000 |
| 64 | 10 | 6.023951e-05 | 6.015869e-05 | 0.134 | 6472.139 | 6471.000 |

**checks**: rust_feasible=True, cold_feasible=True, cold_converged=True, cold≈warm=True, close(0≤gap≤10%)=True, gap_shrinks=True → **PASS**

<!-- provenance: commit=db97dcd1e4bca14279a909f9ea4da29fe3a8d583 dirty=0 ext=825716be634e -->

---

_produced by commit `db97dcd1e4bca14279a909f9ea4da29fe3a8d583`, Rust extension `825716be634e`; working tree clean under tools/verify, orbital_docking, rust_optimizer._
