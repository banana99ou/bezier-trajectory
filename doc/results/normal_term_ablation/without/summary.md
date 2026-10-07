# Monte Carlo over perturbed baseline instances

- range set `near`, sampler `lhs`, 1024 of 1024 instances present, seed 20260930, degree 7
- Delta_0: 2000 km doubled until it exceeds the iteration-1 repair (2-norm); instances by Delta_0: 2000 km: 1024
- ranges: lag 110..130, progress_alt 225..265, iss_alt 380..420, dinc -5..5, draan -5..5, iss_u 0..360
- cells: median [5th, 95th percentile]; rates with 95% Wilson interval

- constraint active: some half-space row within 0.001 km of equality; interior: closest approach strictly between the endpoints; closest sub-arc pressed: the sub-arc holding the closest approach is itself within that tolerance. The clearance ratio uses certified, active, interior solves whose closest sub-arc is pressed -- where the section 3.1 estimate applies. Cost is the objective J; its change uses certified solves where the constraint is active.

| n_seg | solves | errors | certified % [95% CI] | constraint active % | interior % | closest sub-arc pressed % | measured/estimated clearance | J vs n_seg=64 (%) | iterations |
|---:|---:|---:|:--|---:|---:|---:|:--|:--|:--|
| 4 | 1024 | 0 | 0.0 [0.0, 0.4] | 0.0 | 0.0 | 0.0 | — max nan | — | — max 1000, >100: 22 |
| 8 | 1024 | 0 | 0.2 [0.1, 0.7] | 100.0 | 100.0 | 100.0 | 0.938 [0.934, 0.943] max 0.944 | — | 10 [10, 10] max 61, >100: 0 |
| 16 | 1024 | 0 | 9.9 [8.2, 11.8] | 100.0 | 100.0 | 69.3 | 0.982 [0.953, 0.998] max 0.998 | 2.89 [1.55, 3.11] | 10 [9, 18] max 57, >100: 0 |
| 32 | 1024 | 0 | 11.4 [9.6, 13.5] | 100.0 | 100.0 | 75.2 | 0.989 [0.928, 0.999] max 0.999 | 0.551 [0.336, 0.591] | 10 [9, 11] max 41, >100: 0 |
| 64 | 1024 | 0 | 23.2 [20.8, 25.9] | 100.0 | 100.0 | 82.8 | 0.978 [0.919, 0.999] max 1.000 | 0 [0, 0] | 10 [9, 11] max 45, >100: 0 |

## Clearance ratio by whether the closest sub-arc is pressed

| n_seg | pressed: n, ratio | not pressed: n, ratio |
|---:|:--|:--|
| 4 | 0, — | 0, — |
| 8 | 2, 0.938 [0.934, 0.943] | 0, — |
| 16 | 70, 0.982 [0.953, 0.998] | 31, 1 [0.964, 1] |
| 32 | 88, 0.989 [0.928, 0.999] | 29, 0.998 [0.961, 1] |
| 64 | 197, 0.978 [0.919, 0.999] | 41, 0.994 [0.947, 1] |

Clearance scaling exponent over n_seg [8, 16, 32, 64] (per instance, 0 instances): —; the arc estimate predicts -2.

## Certified / solved, by geometric central angle

| central angle (deg) | n_seg 4 | n_seg 8 | n_seg 16 | n_seg 32 | n_seg 64 |
|:--|---:|---:|---:|---:|---:|
| 100–110 | 0/36 | 1/36 | 9/36 | 9/36 | 12/36 |
| 110–120 | 0/487 | 1/487 | 92/487 | 45/487 | 125/487 |
| 120–130 | 0/470 | 0/470 | 0/470 | 61/470 | 96/470 |
| 130–140 | 0/31 | 0/31 | 0/31 | 2/31 | 5/31 |

## Every non-certified solve, by stop reason

| n_seg | stop reason | count |
|---:|:--|---:|
| 4 | 반복 한도 | 4 |
| 4 | 신뢰영역 붕괴 | 1020 |
| 8 | 신뢰영역 붕괴 | 1022 |
| 16 | 신뢰영역 붕괴 | 923 |
| 32 | 신뢰영역 붕괴 | 907 |
| 64 | 신뢰영역 붕괴 | 786 |

Certificate vs dense-probe disagreements: 3641

## Check 3 -- re-solving 32 instances gives identical outputs: PASS

160 repeated solves, 0 differing in any of certified, koz_binds, margin_km, ctrl_cost_ms2, objective, iters, stop, hull_violation_km.

## Check 1 -- baseline reproduces Table 3: FAIL

| n_seg | vs Table 3 | max rel. diff (margin, cost, J) | iterations |
|---:|:--|---:|:--|
| 4 | MISMATCH | 3.3e-01 | 38 vs 12 |
| 8 | MISMATCH | 6.3e-02 | 35 vs 8 |
| 16 | MISMATCH | 1.6e-02 | 49 vs 8 |
| 32 | MISMATCH | 4.3e-02 | 26 vs 8 |
| 64 | MISMATCH | 1.5e-03 | 25 vs 8 |

## Check 2 -- rotation about Earth's axis leaves the solution unchanged: FAIL

| n_seg | rotated vs baseline | max rel. diff (margin, cost, J) | iterations |
|---:|:--|---:|:--|
| 4 | DIFFERENT | 1.5e-01 | 38 vs 32 |
| 8 | DIFFERENT | 7.5e-02 | 35 vs 31 |
| 16 | DIFFERENT | 4.2e-02 | 49 vs 29 |
| 32 | DIFFERENT | 1.4e-03 | 26 vs 26 |
| 64 | DIFFERENT | 1.1e-02 | 25 vs 24 |

This run used 12 worker(s). No runtime is summarized: a solve's cost per iteration is set by n_seg and N, not by the geometry, so the runtime-versus-n_seg claim rests on the baseline timing (Table 3, min of 15 on the paper's machine) and this study reports iterations. runs.csv keeps the solver's wall clock `runtime_s` and the process CPU time `cpu_s` as diagnostics.

<!-- provenance: commit=b6c184e2ea74e8579b20720686b4499df943a45a dirty=0 ext=a464fb1315e9 -->

---

_produced by commit `b6c184e2ea74e8579b20720686b4499df943a45a`, Rust extension `a464fb1315e9`; working tree clean under tools/verify, orbital_docking, rust_optimizer._
