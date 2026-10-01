# Monte Carlo over perturbed baseline instances

- range set `near`, sampler `lhs`, 1024 of 1024 instances present, seed 20260930, degree 7
- Delta_0: 2000 km doubled until it exceeds the iteration-1 repair (2-norm); instances by Delta_0: 2000 km: 1024
- ranges: lag 110..130, progress_alt 225..265, iss_alt 380..420, dinc -5..5, draan -5..5, iss_u 0..360
- cells: median [5th, 95th percentile]; rates with 95% Wilson interval

- constraint active: some half-space row within 0.001 km of equality; interior: closest approach strictly between the endpoints. The clearance ratio uses certified solves that are both; the cost change uses certified solves where the constraint is active.

| n_seg | solves | errors | certified % [95% CI] | constraint active % | interior % | measured/estimated clearance | cost vs n_seg=64 (%) | iterations |
|---:|---:|---:|:--|---:|---:|:--|:--|:--|
| 4 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 0.0 | — | 76.7 [51.1, 140] | 12 [10, 15] max 758, >100: 9 |
| 8 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 100.0 | 0.971 [0.923, 1.01] | 6.21 [5.74, 7.6] | 8 [7, 9] max 279, >100: 5 |
| 16 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 100.0 | 0.982 [0.926, 1.01] | 1.4 [1.29, 1.79] | 8 [7, 9] max 20, >100: 0 |
| 32 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 100.0 | 0.977 [0.906, 1] | 0.281 [0.242, 0.35] | 8 [7, 9] max 11, >100: 0 |
| 64 | 1024 | 0 | 100.0 [99.6, 100.0] | 100.0 | 100.0 | 0.979 [0.91, 1] | 0 [0, 0] | 8 [7, 9] max 10, >100: 0 |

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
