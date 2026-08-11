# Pillar 5 -- Optimality (external KKT, n_seg=16)

Independent of the solver: gradient by finite differences on the dense
true objective, active set and constraint normals rebuilt in NumPy.

| scenario | stop | iters | KKT rel resid | active rows | max KOZ dual | descent hits/tested | verdict |
|---|---|---|---|---|---|---|---|
| phase70 | 4 | 6 | 3.060e-06 | 0 | 0.000e+00 | 0/1600 | PASS |
| phase120 | 4 | 8 | 1.131e-05 | 2 | 2.655e-08 | 0/644 | PASS |
| phase135 | 4 | 9 | 5.585e-06 | 3 | 6.403e-08 | 0/496 | PASS |
| phase170 | 4 | 12 | 4.408e-06 | 4 | 1.895e-07 | 0/333 | PASS |
| planechange | 4 | 10 | 4.866e-06 | 3 | 4.481e-08 | 0/552 | PASS |

- KKT stationarity residual below 0.001 on every scenario: **True**
- no feasible descent direction found (400 directions x 4 step sizes per scenario, 3625 certified trials total): **True**
- exact-penalty rule w_s >= ||lambda_KOZ||_inf (0.01 >= 1.895e-07): **True** (design_freeze section 5 recorded this as 'estimated ~1e-6'; it is now MEASURED)

## VERDICT: PASS
<!-- provenance: commit=3e73f70f1f6fdfe50e7731a74919c39056677c2e dirty=0 ext=825716be634e -->

---

_produced by commit `3e73f70f1f6fdfe50e7731a74919c39056677c2e`, Rust extension `825716be634e`; working tree clean under tools/verify, orbital_docking, rust_optimizer._
