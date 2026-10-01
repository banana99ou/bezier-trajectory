# KOZ investigation: repair the certificate model, separate feasibility from convergence

The radius-freezing experiments did not isolate one cause. There are three distinct issues:

1. The current row Jacobian omits parts of the geometry derivative. Endpoint classification and nearest-query roundoff are additional implementation defects.
2. Even with corrected derivatives and trust-independent merit, the selected wall changes branch at tube endpoints. At the diagnosed wall stall this is a **continuous kink**, not a discontinuous jump. A single-branch model misses an available descent direction. Backtracking and sampled bundles improve violation but did not produce a usable output in the targeted experiments.
3. A centroid-selected plane is only one sufficient certificate. Its failure does not mean that the same trajectory lacks a separating certificate. Independently checked counterexamples demonstrate this at unchanged degree, subdivision, and trajectory.

## Structural repair implemented experimentally

[`support_plane_solver.py`](../tools/experiments/support_plane_solver.py) optimizes the separating normals together with the trajectory controls. It does not differentiate a nearest-point projection or a moving clipping construction.

For trajectory subsegment control points \(Q_{ki}(P)\), obstacle control points \(C_{mj}\), and obstacle radius \(r_m\), its constraints are

\[
 n_{km}^{T}(Q_{ki}(P)-C_{mj}) \ge r_m + \delta - s_{km}
 \quad\text{for every }i,j,\qquad
 \|n_{km}\|_2^2=1,\quad s_{km}\ge0.
\]

Every trajectory point is a convex combination of its subsegment controls. Every obstacle centerline point is a convex combination of its obstacle controls. Consequently, with zero slack these finite inequalities separate the **entire trajectory subarc** from the **entire obstacle control hull plus its radius ball**. For a line obstacle that enclosure is exactly the capsule; for a curved obstacle it is conservative. No sampled clearance is needed for this implication.

All constraints in this finite NLP are smooth: the trajectory/normal products are bilinear and normalization is quadratic. They remain nonconvex. Introducing normal variables removes the artificial requirement that the normal be the centroid's nearest-point direction; it does not guarantee a global solution.

The implementation has two phases:

- Restore feasibility by minimizing explicit nonnegative separation slack.
- Once a full certificate and the hard constraints verify, fix slack to zero and minimize the existing spatial parameter-domain smoothness objective. A fixed positive objective scale improves numerical conditioning without changing that objective mathematically.

Optimizer status is not the success criterion. The returned candidate is checked again after normalizing all normals. It must meet the requested positive margin, endpoints, coordinate/time bounds and minimum time-control increments. An invalid refinement cannot replace a certified first-phase incumbent. The output says `certified_feasible`, with `stationarity_verified: false`; it does not claim an optimal trajectory.

An explicit optional portfolio retries failed local solves using degree-preserving waiting/hurry initializations. Every initial control polygon and attempt is recorded. These are separate starts, not hidden extra iterations. A finite portfolio can still fail and cannot establish infeasibility.

## Controlled evidence

Experiments and raw outputs are under `artifacts/koz-principled-20260922/`. The reference source is `5b3ef7882c8e62dedcc9f54a7d07cacf253aec86`; frozen example inputs below preserve its obstacle geometry.

- The original baseline's gate passes 23/28 configurations. The same baseline with post-hoc timing-seed retries recovers all three coarse wall failures and curve N8/seg4, giving a **post-hoc 27/28 portfolio**. Curve N8/seg16 still fails the unchanged 200-iteration gate. This is not a new single-run benchmark result.
- Fixed geometry plus full numerical row derivatives passes 21/28. Adding a fixed 0.5 geometric reach passes 24/28. Neither establishes a general fix.
- Whole-capsule certificates validate degree-8 wall trajectories with 2, 3, and 4 subdivisions that the centroid oracle rejects at those exact same trajectories. All 23 obstacles, including the two moving tubes, are checked.
- The experimental free-plane solver finds certified outputs for the five original failure configurations (wall N8/seg2,3,4 and curve N8/seg4,16), with timing retries where necessary. These are **targeted feasibility results**, not 28/28 validation of a replacement backend. Case outputs, initializations, inequalities, objective values, and an independent audit are retained in the artifact directory.
- Objective quality remains a limitation. The reusable curve N8/seg4 run has objective about 465, whereas an earlier isolated free-plane run reached about 7.31. Both certify. This sensitivity is a reason not to promote the prototype as a production optimizer.

Changing the seed alone already recovers several cases. Therefore planner recovery cannot be attributed solely to the free normals. The same-trajectory certificate counterexamples establish the separate representational defect without that confounding factor.

The [final independent audit](../artifacts/koz-principled-20260922/final_independent_audit/REPORT.md) reimplements subdivision and objective integration, checks every obstacle control vertex against normalized returned normals, and records source/input/output hashes. All five outputs meet a support margin of `1e-4` within `1e-8`; maximum hard-constraint residual is below `4e-15`.

| Case | Selected initialization | Smoothness objective | Dense clearance (diagnostic only) |
|---|---|---:|---:|
| Wall N8/seg2 | Wait power 2 | 57.806394 | 0.496109 |
| Wall N8/seg3 | Wait power 3 | 8.539424 | 0.371607 |
| Wall N8/seg4 | Wait power 2 | 0.214570 | 0.366283 |
| Curve N8/seg4 | Requested straight seed | 465.371488 | 0.100000 |
| Curve N8/seg16 | Wait power 2 | 59.575576 | 0.075494 |

These runs allow up to 1000 iterations **per local phase**, and retries consume additional solves. They are not a matched 200-iteration speed or convergence comparison. Twenty-four focused tests pass. No stationarity or optimality claim follows from the feasibility certificates.

## Reproduce

From the repository root:

```sh
.venv/bin/python -m tools.experiments.support_plane_solver \
  --timing-retries tools/experiments/support_plane_cases/wall_8_4.json /tmp/wall_planes.json
.venv/bin/python -m tools.experiments.support_plane_solver \
  tools/experiments/support_plane_cases/curve_8_4.json /tmp/curve_planes.json
.venv/bin/python -m pytest -q tests/unit/test_support_plane_solver.py
```

The tests exercise continuous-geometry counterexamples, all control vertices of curved obstacles, analytic Jacobians, normal scaling, hard constraints, requested clearance, misleading optimizer success, incumbent retention, and retry provenance.

## Boundaries before adoption

The default solver is unchanged. This prototype handles body obstacles only; it does not implement station shadows, speed-cap cones, or free-arrival/time-weight options. A whole control hull can exclude useful routes through a curved obstacle's concavity. Extending the method to local support domains requires an explicit coverage argument and domain-validity constraints; simply freezing the old clipping data is insufficient.

These are numerical finite-inequality certificates with a positive margin, not interval-arithmetic proofs. General stationarity and global search reliability are unproved. The larger regression suite, all supported physical constraints, and solution quality must be checked before any default-backend change.

## Research basis

GPT-5.5 reviewed primary sources. [Curtis and Overton's SQP-GS](https://cs.nyu.edu/~overton/papers/pdffiles/sqpgs.pdf) supports sampled-gradient treatment of locally Lipschitz nonsmooth constraints, under assumptions that have not been established for every branch of the current clipping oracle. Its stationarity result is not a global-feasibility guarantee. [Zhang, Liniger and Borrelli's collision-avoidance formulation](https://arxiv.org/abs/1711.03449) motivates explicit separation variables for convex sets. The implementation here uses the direct finite support inequalities above; it is not a claimed implementation of either paper's full algorithm.
