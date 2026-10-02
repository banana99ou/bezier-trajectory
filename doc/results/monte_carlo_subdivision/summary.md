# Monte Carlo over perturbed baseline instances

- range set `near`, sampler `lhs`, 1024 of 1024 instances present, seed 20260930, degree 7
- Delta_0: 2000 km doubled until it exceeds the iteration-1 repair (2-norm); instances by Delta_0: 2000 km: 1024
- ranges: lag 110..130, progress_alt 225..265, iss_alt 380..420, dinc -5..5, draan -5..5, iss_u 0..360
- cells: median [5th, 95th percentile]; rates with 95% Wilson interval

- constraint active: some half-space row within 0.001 km of equality; interior: closest approach strictly between the endpoints; closest sub-arc pressed: the sub-arc holding the closest approach is itself within that tolerance. The clearance ratio uses certified, active, interior solves whose closest sub-arc is pressed -- where the section 3.1 estimate applies. Cost is the objective J; its change uses certified solves where the constraint is active.

| n_seg | solves | errors | certified % [95% CI] | constraint active % | interior % | closest sub-arc pressed % | measured/estimated clearance | J vs n_seg=64 (%) | iterations |
|---:|---:|---:|:--|---:|---:|---:|:--|:--|:--|
| 4 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 0.0 | 0.0 | — max nan | 214 [127, 490] | 12 [10, 15] max 758, >100: 9 |
| 8 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 100.0 | 69.9 | 0.967 [0.919, 0.99] max 0.993 | 12 [10.9, 13] | 8 [7, 9] max 279, >100: 5 |
| 16 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 100.0 | 76.4 | 0.97 [0.923, 0.997] max 0.999 | 2.63 [2.34, 2.87] | 8 [7, 9] max 20, >100: 0 |
| 32 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 100.0 | 80.4 | 0.968 [0.905, 0.998] max 0.999 | 0.515 [0.42, 0.552] | 8 [7, 9] max 11, >100: 0 |
| 64 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 100.0 | 82.2 | 0.971 [0.907, 0.999] max 1.002 | 0 [0, 0] | 8 [7, 9] max 10, >100: 0 |

## Clearance ratio by whether the closest sub-arc is pressed

| n_seg | pressed: n, ratio | not pressed: n, ratio |
|---:|:--|:--|
| 4 | 0, — | 0, — |
| 8 | 716, 0.967 [0.919, 0.99] | 308, 0.991 [0.969, 1.02] |
| 16 | 782, 0.97 [0.923, 0.997] | 242, 1 [0.982, 1.01] |
| 32 | 823, 0.968 [0.905, 0.998] | 201, 0.997 [0.937, 1] |
| 64 | 842, 0.971 [0.907, 0.999] | 182, 0.998 [0.951, 1] |

Clearance scaling exponent over n_seg [8, 16, 32, 64] (per instance, 1024 instances): -1.99 [-2.02, -1.97]; the arc estimate predicts -2.

## Certified / solved, by geometric central angle

| central angle (deg) | n_seg 4 | n_seg 8 | n_seg 16 | n_seg 32 | n_seg 64 |
|:--|---:|---:|---:|---:|---:|
| 100–110 | 36/36 | 36/36 | 36/36 | 36/36 | 36/36 |
| 110–120 | 487/487 | 487/487 | 487/487 | 487/487 | 487/487 |
| 120–130 | 470/470 | 470/470 | 470/470 | 470/470 | 470/470 |
| 130–140 | 31/31 | 31/31 | 31/31 | 31/31 | 31/31 |

## Every non-certified solve, by stop reason

| n_seg | stop reason | count |
|---:|:--|---:|

Certificate vs dense-probe disagreements: 0

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
