# Monte Carlo over perturbed baseline instances

- range set `wide`, sampler `lhs`, 4096 of 4096 instances present, seed 20260930, degree 7
- Delta_0: 2000 km doubled until it exceeds the iteration-1 repair (2-norm); instances by Delta_0: 2000 km: 2374, 4000 km: 1722
- ranges: lag 100..170, progress_alt 195..295, iss_alt 350..450, dinc -20..20, draan -15..15, iss_u 0..360
- cells: median [5th, 95th percentile]; rates with 95% Wilson interval

- constraint active: some half-space row within 0.001 km of equality; interior: closest approach strictly between the endpoints; closest sub-arc pressed: the sub-arc holding the closest approach is itself within that tolerance. The clearance ratio uses certified, active, interior solves whose closest sub-arc is pressed -- where the section 3.1 estimate applies. Cost is the objective J; its change uses certified solves where the constraint is active.

| n_seg | solves | errors | certified % [95% CI] | constraint active % | interior % | closest sub-arc pressed % | measured/estimated clearance | J vs n_seg=64 (%) | iterations |
|---:|---:|---:|:--|---:|---:|---:|:--|:--|:--|
| 4 | 4096 | 0 | 98.6 [98.2, 98.9] | 98.6 | 2.9 | 43.3 | 0.967 [0.951, 0.98] max 0.984 | 171 [49.2, 671] | 15 [8, 53] max 1000, >100: 149 |
| 8 | 4096 | 0 | 100.0 [99.9, 100.0] | 92.0 | 96.8 | 60.0 | 0.967 [0.91, 0.985] max 0.996 | 11.6 [5.02, 12.6] | 10 [6, 16] max 931, >100: 36 |
| 16 | 4096 | 0 | 100.0 [99.9, 100.0] | 90.3 | 97.0 | 75.9 | 0.978 [0.916, 0.994] max 0.999 | 2.56 [1.27, 2.77] | 9 [6, 14] max 81, >100: 0 |
| 32 | 4096 | 0 | 100.0 [99.9, 100.0] | 89.8 | 97.0 | 80.6 | 0.976 [0.917, 0.997] max 0.999 | 0.502 [0.25, 0.544] | 9 [6, 14] max 24, >100: 0 |
| 64 | 4096 | 0 | 100.0 [99.9, 100.0] | 89.7 | 97.0 | 83.8 | 0.975 [0.918, 0.998] max 1.002 | 0 [0, 0] | 9 [6, 13] max 22, >100: 0 |

## Clearance ratio by whether the closest sub-arc is pressed

| n_seg | pressed: n, ratio | not pressed: n, ratio |
|---:|:--|:--|
| 4 | 45, 0.967 [0.951, 0.98] | 59, 1.03 [1, 1.07] |
| 8 | 2252, 0.967 [0.91, 0.985] | 1504, 1.11 [0.976, 1.5] |
| 16 | 2808, 0.978 [0.916, 0.994] | 892, 1.01 [0.965, 1.06] |
| 32 | 2965, 0.976 [0.917, 0.997] | 715, 0.998 [0.946, 1.02] |
| 64 | 3082, 0.975 [0.918, 0.998] | 594, 0.997 [0.967, 1.01] |

Clearance scaling exponent over n_seg [8, 16, 32, 64] (per instance, 3665 instances): -1.99 [-2.01, -1.93]; the arc estimate predicts -2.

## Certified / solved, by geometric central angle

| central angle (deg) | n_seg 4 | n_seg 8 | n_seg 16 | n_seg 32 | n_seg 64 |
|:--|---:|---:|---:|---:|---:|
| 100–110 | 472/479 | 479/479 | 479/479 | 479/479 | 479/479 |
| 110–120 | 578/580 | 580/580 | 580/580 | 580/580 | 580/580 |
| 120–130 | 597/599 | 599/599 | 599/599 | 599/599 | 599/599 |
| 130–140 | 609/617 | 617/617 | 617/617 | 617/617 | 617/617 |
| 140–150 | 585/596 | 596/596 | 596/596 | 596/596 | 596/596 |
| 150–160 | 629/630 | 630/630 | 630/630 | 630/630 | 630/630 |
| 160–170 | 401/415 | 415/415 | 415/415 | 415/415 | 415/415 |
| 170–180 | 39/49 | 49/49 | 49/49 | 49/49 | 49/49 |

## Every non-certified solve, by stop reason

| n_seg | stop reason | count |
|---:|:--|---:|
| 4 | 반복 한도 | 38 |
| 4 | 신뢰영역 붕괴 | 20 |

Certificate vs dense-probe disagreements: 58

## Check 3 -- re-solving 32 instances gives identical outputs: PASS

160 repeated solves, 0 differing in any of certified, koz_binds, margin_km, ctrl_cost_ms2, objective, iters, stop, hull_violation_km.

## Check 1 -- baseline reproduces Table 3: PASS

| n_seg | vs Table 3 | max rel. diff (margin, cost, J) | iterations |
|---:|:--|---:|:--|
| 4 | OK | 5.1e-08 | 12 vs 12 |
| 8 | OK | 1.9e-07 | 8 vs 8 |
| 16 | OK | 2.8e-07 | 8 vs 8 |
| 32 | OK | 1.2e-07 | 8 vs 8 |
| 64 | OK | 8.5e-08 | 8 vs 8 |

## Check 2 -- rotation about Earth's axis leaves the solution unchanged: PASS

| n_seg | rotated vs baseline | max rel. diff (margin, cost, J) | iterations |
|---:|:--|---:|:--|
| 4 | OK | 4.6e-12 | 12 vs 12 |
| 8 | OK | 1.4e-09 | 8 vs 9 |
| 16 | OK | 1.2e-10 | 8 vs 9 |
| 32 | OK | 4.0e-10 | 8 vs 9 |
| 64 | OK | 3.3e-10 | 8 vs 9 |

This run used 12 worker(s). No runtime is summarized: a solve's cost per iteration is set by n_seg and N, not by the geometry, so the runtime-versus-n_seg claim rests on the baseline timing (Table 3, min of 15 on the paper's machine) and this study reports iterations. runs.csv keeps the solver's wall clock `runtime_s` and the process CPU time `cpu_s` as diagnostics.

<!-- provenance: commit=8860782ec22e4600da7531e354ec1264f466d8e3 dirty=0 ext=2c67baf508e0 -->

---

_produced by commit `8860782ec22e4600da7531e354ec1264f466d8e3`, Rust extension `2c67baf508e0`; working tree clean under tools/verify, orbital_docking, rust_optimizer._
