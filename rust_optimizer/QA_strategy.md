# Rust Migration QA Strategy

The core risk is: **does the Rust optimizer produce physically correct trajectories**, not just "feasible by its own metrics"? The Python and Rust solvers use different QP backends (trust-constr vs Clarabel), so we shouldn't expect identical outputs — but we need to verify the solutions are valid under the **same physics model**.

## 1. Cross-validate Rust solutions through Python's physics

Feed Rust's `P_opt` back into Python's evaluation/validation code — use Python as the ground truth checker:

- Recompute `min_radius` from Rust's P_opt using Python's `BezierCurve.point()`
- Recompute `cost_true_energy` from Rust's P_opt using Python's `_build_ctrl_accel_quadratic()`
- Check KOZ constraint satisfaction using Python's `build_koz_constraints()`
- Check boundary conditions (endpoint preservation, velocity/accel BCs if any)

## 2. Per-component unit parity tests

The building blocks should match exactly (same math, no solver involved):

- Gravity at ~20 random positions
- D/E/G matrices for degrees 2–6
- Segment matrices for various N/n_seg combos
- Bernstein basis + derivative weights at several tau values
- `_build_ctrl_accel_quadratic()` H/f/c output for same P_ref

## 3. Solution quality sweep

Run both solvers across a grid of (degree, n_seg, max_iter) and compare:

- Does Rust always produce feasible solutions when Python does?
- Is Rust's cost within a reasonable factor of Python's?
- Does Rust maintain endpoint constraints?

## 4. Edge cases

- N=2 (minimal degree for acceleration)
- n_seg=1 (no subdivision)
- With velocity/acceleration BCs
- With prograde enforcement
