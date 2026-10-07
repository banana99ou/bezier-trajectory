# Ablation: the normal-change term in the obstacle rows

Two builds of the same source, differing in one line of `rust_optimizer/core/src/optimizer.rs`: the "without" arm replaces `constraints::build_koz_constraints_linearized(` with `constraints::build_koz_constraints(`, so the subproblem's obstacle rows keep the normal fixed at the reference. The merit, ratio test, mu, Delta_0 and stopping rules are identical. Produced by `tools/build_normal_term_ablation.py`; each arm's full Monte Carlo output is in `with/` and `without/`. The `without/` provenance line reports the repo source as clean because the patch was applied to a temporary copy; the extension hash is what identifies each build (`builds.json`, check A).

## Near-set Monte Carlo (1024 instances per n_seg)

Converged = certified by the paper's definition (h <= 1e-6 km and a principled stop). Cells: median [5th, 95th percentile]; rates with 95% Wilson interval. The cost change pairs each instance across the arms, over instances whose final iterate satisfies the avoidance condition in both.

| n_seg | converged, with (%) | converged, without (%) | iterations, with | iterations, without | without: stops | without, not converged but h <= 1e-6 | J without vs with (%) |
|---:|:--|:--|:--|:--|:--|---:|:--|
| 4 | 100.0 [99.6, 100.0] | 0.0 [0.0, 0.4] | 12 [10, 15] | 32 [23, 43] | 반복 한도 4, 신뢰영역 붕괴 1020 | 3/1024 | 202 [152, 410] (n=3) |
| 8 | 100.0 [99.6, 100.0] | 0.2 [0.1, 0.7] | 8 [7, 9] | 31 [27, 39] | 신뢰영역 붕괴 1022, 모형 정지점 2 | 993/1022 | 1.16 [0.221, 5.31] (n=995) |
| 16 | 100.0 [99.6, 100.0] | 9.9 [8.2, 11.8] | 8 [7, 9] | 29 [10, 35] | 신뢰영역 붕괴 923, 모형 정지점 101 | 923/923 | 0.175 [0.0192, 0.829] (n=1024) |
| 32 | 100.0 [99.6, 100.0] | 11.4 [9.6, 13.5] | 8 [7, 9] | 26 [10, 31] | 신뢰영역 붕괴 907, 모형 정지점 117 | 907/907 | 0.0352 [0.00243, 0.0845] (n=1024) |
| 64 | 100.0 [99.6, 100.0] | 23.2 [20.8, 25.9] | 8 [7, 9] | 24 [9, 27] | 신뢰영역 붕괴 786, 모형 정지점 238 | 786/786 | 0.00976 [0.000888, 0.0172] (n=1024) |

## Table-2 scenarios

A phantom rejection: the candidate satisfies the rows it was solved against and keeps every subdivided control point outside the sphere, but violates the rows rebuilt at itself with the normal re-aimed (tolerance 1e-6 km throughout).

| scenario | n_seg | with: stop, iterations, rejects | without: stop, iterations, rejects (phantom) | J without vs with (%) |
|:--|---:|:--|:--|---:|
| phase70 | 4 | 모형 정지점, 6, 0 | 모형 정지점, 6, 0 (0) | -0.000 |
| phase70 | 8 | 모형 정지점, 6, 0 | 모형 정지점, 6, 0 (0) | +0.000 |
| phase70 | 16 | 모형 정지점, 6, 0 | 모형 정지점, 6, 0 (0) | -0.000 |
| phase70 | 32 | 모형 정지점, 6, 0 | 모형 정지점, 6, 0 (0) | -0.000 |
| phase70 | 64 | 모형 정지점, 6, 0 | 모형 정지점, 6, 0 (0) | +0.000 |
| phase120 | 4 | 모형 정지점, 12, 0 | 신뢰영역 붕괴, 38, 25 (0) | — (h = 44.4 km) |
| phase120 | 8 | 모형 정지점, 8, 0 | 신뢰영역 붕괴, 35, 26 (26) | +2.604 |
| phase120 | 16 | 모형 정지점, 8, 0 | 신뢰영역 붕괴, 49, 33 (33) | +0.603 |
| phase120 | 32 | 모형 정지점, 8, 0 | 신뢰영역 붕괴, 26, 21 (21) | +0.064 |
| phase120 | 64 | 모형 정지점, 8, 0 | 신뢰영역 붕괴, 25, 20 (20) | +0.010 |
| phase135 | 4 | 모형 정지점, 14, 0 | 신뢰영역 붕괴, 31, 19 (0) | — (h = 88.4 km) |
| phase135 | 8 | 모형 정지점, 11, 0 | 신뢰영역 붕괴, 32, 24 (24) | +0.313 |
| phase135 | 16 | 모형 정지점, 11, 0 | 신뢰영역 붕괴, 29, 22 (22) | +0.066 |
| phase135 | 32 | 모형 정지점, 10, 0 | 신뢰영역 붕괴, 26, 20 (20) | +0.006 |
| phase135 | 64 | 모형 정지점, 10, 0 | 신뢰영역 붕괴, 26, 20 (20) | +0.007 |
| phase170 | 4 | 모형 정지점, 43, 19 | 신뢰영역 붕괴, 38, 23 (0) | — (h = 276 km) |
| phase170 | 8 | merit 감소 연속, 168, 22 | 신뢰영역 붕괴, 30, 24 (24) | +4.956 |
| phase170 | 16 | 모형 정지점, 12, 0 | 신뢰영역 붕괴, 34, 26 (26) | +0.842 |
| phase170 | 32 | 모형 정지점, 12, 0 | 신뢰영역 붕괴, 36, 27 (27) | +0.364 |
| phase170 | 64 | 모형 정지점, 12, 0 | 신뢰영역 붕괴, 28, 22 (22) | +0.007 |
| planechange | 4 | 모형 정지점, 17, 2 | 신뢰영역 붕괴, 37, 24 (0) | — (h = 78.7 km) |
| planechange | 8 | 모형 정지점, 10, 1 | 신뢰영역 붕괴, 30, 23 (23) | +0.246 |
| planechange | 16 | 모형 정지점, 10, 1 | 신뢰영역 붕괴, 30, 23 (23) | +0.084 |
| planechange | 32 | 모형 정지점, 10, 0 | 신뢰영역 붕괴, 28, 22 (22) | +0.026 |
| planechange | 64 | 모형 정지점, 10, 1 | 신뢰영역 붕괴, 25, 20 (20) | +0.005 |

Rejected steps over the grid, total (phantom): with 46 (19); without 464 (373).

## Check A -- each arm ran its own build: PASS

- `with`: built `b944059ee741`, Monte Carlo imported `b944059ee741`
- `without`: built `a464fb1315e9`, Monte Carlo imported `a464fb1315e9`

## Check B -- the "with" arm reproduces the paper's tables: PASS

| scenario | n_seg | vs paper_tables.csv | J rel. diff | iterations |
|:--|---:|:--|---:|:--|
| phase70 | 16 | OK | 1.1e-08 | 6 vs 6 |
| phase120 | 16 | OK | 1.6e-07 | 8 vs 8 |
| phase135 | 16 | OK | 5.0e-08 | 11 vs 11 |
| phase170 | 16 | OK | 4.8e-08 | 12 vs 12 |
| planechange | 16 | OK | 3.1e-08 | 10 vs 10 |
| phase120 | 4 | OK | 5.1e-08 | 12 vs 12 |
| phase120 | 8 | OK | 1.9e-07 | 8 vs 8 |
| phase120 | 16 | OK | 1.6e-07 | 8 vs 8 |
| phase120 | 32 | OK | 1.2e-07 | 8 vs 8 |
| phase120 | 64 | OK | 8.5e-08 | 8 vs 8 |
| phase120 | 16 | OK | 1.6e-07 | 8 vs 8 |

## Check C -- where the obstacle constraint is inactive, the arms agree: PASS

| scenario | n_seg | min half-space slack (km) | arms agree | J rel. diff |
|:--|---:|---:|:--|---:|
| phase70 | 4 | 16.1 | OK | 6.2e-14 |
| phase70 | 8 | 108.8 | OK | 7.1e-14 |
| phase70 | 16 | 135.5 | OK | 9.7e-14 |
| phase70 | 32 | 142.6 | OK | 9.7e-14 |
| phase70 | 64 | 144.4 | OK | 1.4e-13 |

Monte Carlo checks, `with` arm:
- Check 3 -- re-solving 32 instances gives identical outputs: PASS
- Check 1 -- baseline reproduces Table 3: PASS
- Check 2 -- rotation about Earth's axis leaves the solution unchanged: PASS

Monte Carlo checks, `without` arm:
- Check 3 -- re-solving 32 instances gives identical outputs: PASS
- Check 1 -- baseline reproduces Table 3: FAIL
- Check 2 -- rotation about Earth's axis leaves the solution unchanged: FAIL

---

_both arms built from commit `b6c184e2ea74e8579b20720686b4499df943a45a`, source clean; this script sha256 `b8a67a17670d`; extensions: with `b944059ee741`, without `a464fb1315e9`._
