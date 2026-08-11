# Pillar 4a -- Diagnostics (n_seg=16, energy, 5 geometries)

A correct SCvx run shows rho ~ 1 on optimality steps, slack -> 0, a stated
stop reason, and -- the gate the others cannot provide -- a returned iterate
that is the best one visited.

| scenario | steps | stop reason | merit drift | final slack | verdict |
|---|---|---|---|---|---|
| phase70 | 4 | model stationarity (4) | +0.00e+00 | 1.29e-19 | **PASS** |
| phase120 | 7 | model stationarity (4) | +0.00e+00 | 3.66e-20 | **PASS** |
| phase135 | 8 | model stationarity (4) | +0.00e+00 | 2.61e-19 | **PASS** |
| phase170 | 11 | model stationarity (4) | +0.00e+00 | 3.12e-19 | **PASS** |
| planechange | 8 | model stationarity (4) | +0.00e+00 | 5.48e-21 | **PASS** |

## VERDICT: PASS

---

## phase70 (n_seg=16, r0=2000 km)

- accepted steps: 4 (iterations=6)
- rho on optimality steps in (0.5,2): **True**  (values: 0.993, 0.980, 1.362)
- merit monotone within each phase: **True** (near-tautological; consistency check only)
- returned iterate is the BEST visited (drift +0.000e+00): **True** — the gate rho and monotonicity cannot provide; catches a run that descends then wanders off its own optimum
- slack -> 0 (final=1.29e-19): **True**
- stopped for a stated reason (not the cap, not trust collapse): **True** (stop_reason=4 [model stationarity], iterations=6, final trust=4000 km, floor=0.01)
- feasible + Prop-1 certificate at the final iterate (final_cp_violation_km=0.00e+00): **True**
- every QP hit the requested tolerances (qp_almost_solved=0): **True**
- no degenerate KOZ normals skipped (koz_degenerate_segments=0): **True**

See `iter_trace_phase70.png` (orange band = feasibility-restoration).

## phase120 (n_seg=16, r0=2000 km)

- accepted steps: 7 (iterations=8)
- rho on optimality steps in (0.5,2): **True**  (values: 1.000, 0.998, 1.000, 1.000, 1.092, 0.929)
- merit monotone within each phase: **True** (near-tautological; consistency check only)
- returned iterate is the BEST visited (drift +0.000e+00): **True** — the gate rho and monotonicity cannot provide; catches a run that descends then wanders off its own optimum
- slack -> 0 (final=3.66e-20): **True**
- stopped for a stated reason (not the cap, not trust collapse): **True** (stop_reason=4 [model stationarity], iterations=8, final trust=8000 km, floor=0.01)
- feasible + Prop-1 certificate at the final iterate (final_cp_violation_km=0.00e+00): **True**
- every QP hit the requested tolerances (qp_almost_solved=0): **True**
- no degenerate KOZ normals skipped (koz_degenerate_segments=0): **True**

See `iter_trace_phase120.png` (orange band = feasibility-restoration).

## phase135 (n_seg=16, r0=4000 km)

- accepted steps: 8 (iterations=9)
- rho on optimality steps in (0.5,2): **True**  (values: 1.000, 1.003, 0.998, 0.998, 0.999, 1.462, inf)
- merit monotone within each phase: **True** (near-tautological; consistency check only)
- returned iterate is the BEST visited (drift +0.000e+00): **True** — the gate rho and monotonicity cannot provide; catches a run that descends then wanders off its own optimum
- slack -> 0 (final=2.61e-19): **True**
- stopped for a stated reason (not the cap, not trust collapse): **True** (stop_reason=4 [model stationarity], iterations=9, final trust=1.6e+04 km, floor=0.01)
- feasible + Prop-1 certificate at the final iterate (final_cp_violation_km=0.00e+00): **True**
- every QP hit the requested tolerances (qp_almost_solved=0): **True**
- no degenerate KOZ normals skipped (koz_degenerate_segments=0): **True**

See `iter_trace_phase135.png` (orange band = feasibility-restoration).

## phase170 (n_seg=16, r0=4000 km)

- accepted steps: 11 (iterations=12)
- rho on optimality steps in (0.5,2): **True**  (values: 1.000, 1.002, 0.996, 0.973, 0.966, 0.965, 0.965, 0.968, 0.986, 1.002)
- merit monotone within each phase: **True** (near-tautological; consistency check only)
- returned iterate is the BEST visited (drift +0.000e+00): **True** — the gate rho and monotonicity cannot provide; catches a run that descends then wanders off its own optimum
- slack -> 0 (final=3.12e-19): **True**
- stopped for a stated reason (not the cap, not trust collapse): **True** (stop_reason=4 [model stationarity], iterations=12, final trust=1.6e+04 km, floor=0.01)
- feasible + Prop-1 certificate at the final iterate (final_cp_violation_km=0.00e+00): **True**
- every QP hit the requested tolerances (qp_almost_solved=0): **True**
- no degenerate KOZ normals skipped (koz_degenerate_segments=0): **True**

See `iter_trace_phase170.png` (orange band = feasibility-restoration).

## planechange (n_seg=16, r0=2000 km)

- accepted steps: 8 (iterations=10)
- rho on optimality steps in (0.5,2): **True**  (values: 1.000, 0.996, 0.998, 0.996, 0.995, 0.984, 0.880)
- merit monotone within each phase: **True** (near-tautological; consistency check only)
- returned iterate is the BEST visited (drift +0.000e+00): **True** — the gate rho and monotonicity cannot provide; catches a run that descends then wanders off its own optimum
- slack -> 0 (final=5.48e-21): **True**
- stopped for a stated reason (not the cap, not trust collapse): **True** (stop_reason=4 [model stationarity], iterations=10, final trust=4000 km, floor=0.01)
- feasible + Prop-1 certificate at the final iterate (final_cp_violation_km=0.00e+00): **True**
- every QP hit the requested tolerances (qp_almost_solved=0): **True**
- no degenerate KOZ normals skipped (koz_degenerate_segments=0): **True**

See `iter_trace_planechange.png` (orange band = feasibility-restoration).

<!-- provenance: commit=db97dcd1e4bca14279a909f9ea4da29fe3a8d583 dirty=0 ext=825716be634e -->

---

_produced by commit `db97dcd1e4bca14279a909f9ea4da29fe3a8d583`, Rust extension `825716be634e`; working tree clean under tools/verify, orbital_docking, rust_optimizer._
