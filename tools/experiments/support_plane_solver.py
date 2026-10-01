"""Experimental joint trajectory/support-plane solver for body obstacles.

This intentionally does not replace the SCP backend. Each complete Bezier
obstacle is enclosed by its control hull plus its radius ball. There is no
centroid-selected normal, moving clipping domain, or differentiated projection.
This enclosure is conservative for curved obstacles; shadows are unsupported.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from orbital_docking.de_casteljau import segment_matrices_equal_params
from spacetime_bezier.objective import build_smoothness_regularizer


def verify(p, normals, obstacles, n_seg, start, end, *, min_dt=0.1,
           coord_bounds=None, time_ub=None):
    """Verify finite support inequalities, independently of optimizer status.

    For unit n, min_i n.Q_i - max_j n.C_j - radius >= margin
    proves separation of the entire trajectory subarc from the entire tube.
    Floating-point checks with a positive margin are not interval arithmetic.
    """
    p = np.asarray(p, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if p.ndim != 2 or len(p) < 2 or n_seg < 1:
        raise ValueError("Invalid trajectory or subdivision count")
    dim = p.shape[1]
    expected = (n_seg, len(obstacles), dim)
    if normals.shape != expected or not np.isfinite(normals).all():
        raise ValueError(f"Expected finite normals of shape {expected}")
    lengths = np.linalg.norm(normals, axis=-1)
    if np.any(lengths < 1e-12) or not np.isfinite(p).all():
        raise ValueError("Nonfinite controls or degenerate certificate normal")
    normals = normals / lengths[..., None]
    q = np.asarray(segment_matrices_equal_params(len(p)-1, n_seg)) @ p
    margins = np.empty((n_seg, len(obstacles)))
    for o, obstacle in enumerate(obstacles):
        controls = np.asarray(obstacle["control_points"], dtype=float)
        radius = float(obstacle["radius"])
        if controls.ndim != 2 or controls.shape[1] != dim or not len(controls) \
                or not np.isfinite(controls).all() or not np.isfinite(radius) or radius < 0:
            raise ValueError("Invalid obstacle controls/radius")
        margins[:, o] = (np.einsum("sd,skd->sk", normals[:, o], q).min(axis=1)
                         - (normals[:, o] @ controls.T).max(axis=1)
                         - radius)
    bounds = np.asarray(coord_bounds if coord_bounds is not None
                        else [(-20., 20.)] * (dim-1), dtype=float)
    upper = float(time_ub if time_ub is not None else 1.5 * end[-1])
    hard = max(float(np.max(np.abs(p[0]-start))),
               float(np.max(np.abs(p[-1]-end))),
               float(np.max(bounds[:, 0]-p[:, :-1])),
               float(np.max(p[:, :-1]-bounds[:, 1])),
               float(-p[:, -1].min()), float(p[:, -1].max()-upper),
               float(min_dt-np.diff(p[:, -1]).min()), 0.)
    margin = float(margins.min()) if margins.size else float("inf")
    return {"minimum_margin": margin, "hard_violation": hard,
            "certified": bool(margin > 0. and hard <= 1e-8),
            "pair_margins": margins.tolist(), "pair_count": margins.size}


def solve(p_init, obstacles, n_seg, *, clearance=1e-4, min_dt=0.1,
          coord_bounds=None, time_ub=None, max_iter=400):
    """Two local NLP phases: restore separation, then minimize smoothness.

    Normalization excludes the zero-plane/slack degeneracy. The bilinear
    normal/control products remain nonconvex: failure is not infeasibility.
    A verified phase-I incumbent is retained if objective refinement fails.
    """
    p_init = np.asarray(p_init, dtype=float)
    if p_init.ndim != 2 or len(p_init) < 3 or not np.isfinite(p_init).all():
        raise ValueError("Need finite Bezier controls of degree at least two")
    if n_seg < 1 or clearance <= 0 or min_dt < 0 or not obstacles:
        raise ValueError("Need segments, obstacles, positive clearance and nonnegative min_dt")
    degree, dim = len(p_init)-1, p_init.shape[1]
    controls = [np.asarray(o["control_points"], dtype=float) for o in obstacles]
    radii = np.asarray([o["radius"] for o in obstacles], dtype=float)
    if any(c.ndim != 2 or c.shape[1] != dim or len(c) == 0
           or not np.isfinite(c).all() for c in controls) or np.any(radii < 0) \
            or not np.isfinite(radii).all():
        raise ValueError("Invalid obstacle controls/radii")
    count, nv = len(controls), p_init.size
    ng, vertices = n_seg*count, max(map(len, controls))
    obstacle_vertices = np.asarray([np.concatenate([c, np.repeat(c[-1:], vertices-len(c), axis=0)])
                                    for c in controls])
    matrices = np.asarray(segment_matrices_equal_params(degree, n_seg))
    centroid = (matrices @ p_init).mean(axis=1)
    a, b = obstacle_vertices[:, 0], obstacle_vertices[:, -1]
    ab = b-a
    denom = np.maximum(np.sum(ab*ab, axis=1), 1e-30)
    u = np.clip(np.einsum("sod,od->so", centroid[:, None]-a, ab)/denom, 0., 1.)
    normals = centroid[:, None] - a - u[..., None]*ab
    lengths = np.linalg.norm(normals, axis=-1)
    normals[lengths < 1e-12] = np.eye(dim)[0]
    normals /= np.linalg.norm(normals, axis=-1)[..., None]
    normal_end = nv + dim*ng

    def unpack(x):
        return (x[:nv].reshape(p_init.shape),
                x[nv:normal_end].reshape(n_seg, count, dim),
                x[normal_end:].reshape(n_seg, count))

    def separation(x):
        p, n, slack = unpack(x)
        return (np.einsum("sod,skd->sok", n, matrices@p)[..., None]
                - np.einsum("sod,ovd->sov", n, obstacle_vertices)[:, :, None]
                - radii[None, :, None, None] - clearance
                + slack[:, :, None, None]).ravel()

    def separation_jac(x):
        p, n, _ = unpack(x)
        q = matrices @ p
        shape = (n_seg, count, degree+1, vertices)
        jac = np.zeros((*shape, len(x)))
        cp_jac = np.einsum("skl,sod->sokld", matrices, n)
        jac[..., :nv] = np.broadcast_to(cp_jac[:, :, :, None],
                                       (*shape, degree+1, dim)).reshape(*shape, nv)
        for seg in range(n_seg):
            for obs in range(count):
                group = seg*count+obs
                jac[seg, obs, :, :, nv+group*dim:nv+(group+1)*dim] = (
                    q[seg, :, None]-obstacle_vertices[obs, None])
                jac[seg, obs, :, :, normal_end+group] = 1.
        return jac.reshape(-1, len(x))

    def unit(x):
        return (np.sum(unpack(x)[1]**2, axis=-1)-1.).ravel()

    def unit_jac(x):
        jac = np.zeros((ng, len(x)))
        for group, normal in enumerate(unpack(x)[1].reshape(ng, dim)):
            jac[group, nv+group*dim:nv+(group+1)*dim] = 2*normal
        return jac

    x = np.r_[p_init.ravel(), normals.ravel(), np.zeros(ng)]
    x[normal_end:] = np.maximum(-separation(x).reshape(n_seg, count, degree+1, vertices)
                               .min(axis=(2, 3)), 0.).ravel()
    spatial = coord_bounds if coord_bounds is not None else [(-20., 20.)]*(dim-1)
    upper = float(time_ub if time_ub is not None else 1.5*p_init[-1, -1])
    bounds = [(float(v), float(v)) if i in (0, degree)
              else ((0., upper) if j == dim-1 else tuple(spatial[j]))
              for i, point in enumerate(p_init) for j, v in enumerate(point)]
    bounds += [(-1., 1.)]*(dim*ng) + [(0., None)]*ng
    time_rows = np.zeros((degree, len(x)))
    for i in range(degree):
        time_rows[i, (i+1)*dim+dim-1] = 1.
        time_rows[i, i*dim+dim-1] = -1.
    constraints = [{"type": "ineq", "fun": separation, "jac": separation_jac},
                   {"type": "eq", "fun": unit, "jac": unit_jac},
                   LinearConstraint(time_rows, min_dt, np.inf)]
    f = np.r_[np.zeros(normal_end), np.ones(ng)]
    first = minimize(lambda z: f@z, x, jac=lambda z: f, method="SLSQP",
                     bounds=bounds, constraints=constraints,
                     options={"maxiter": max_iter, "ftol": 1e-10})
    h = build_smoothness_regularizer(degree, dim)

    def audit(z):
        try:
            p, n, _ = unpack(z)
            checked = verify(p, n, obstacles, n_seg, p_init[0], p_init[-1], min_dt=min_dt,
                             coord_bounds=spatial, time_ub=upper)
            checked["geometrically_separated"] = checked["certified"]
            checked["requested_clearance"] = clearance
            checked["clearance_shortfall"] = max(0., clearance-checked["minimum_margin"])
            checked["certified"] = (checked["certified"] and
                                    checked["clearance_shortfall"] <= min(1e-8, .01*clearance))
            return checked
        except ValueError as error:
            return {"certified": False, "reason": str(error),
                    "minimum_margin": None, "hard_violation": None}

    chosen, chosen_audit = first.x, audit(first.x)
    second = None
    second_audit = None
    chosen_phase = 1
    objective_scale = max(1., abs(float(.5*first.x[:nv]@h@first.x[:nv])))
    if chosen_audit["certified"]:
        bounds[-ng:] = [(0., 0.)]*ng
        # A fixed positive scale leaves the objective unchanged mathematically,
        # while keeping its initial magnitude comparable to the unit-normal rows.
        second = minimize(lambda z: .5*z[:nv]@h@z[:nv]/objective_scale, first.x,
                          jac=lambda z: np.r_[h@z[:nv]/objective_scale, np.zeros(len(z)-nv)],
                          method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": max_iter, "ftol": 1e-9})
        second_audit = audit(second.x)
        if second_audit["certified"] and second.x[:nv]@h@second.x[:nv] <= chosen[:nv]@h@chosen[:nv]:
            chosen, chosen_audit = second.x, second_audit
            chosen_phase = 2
    p, n, slack = unpack(chosen)
    return {"control_points": p.tolist(), "normals": n.tolist(),
            "certificate": chosen_audit, "objective": float(.5*p.ravel()@h@p.ravel()),
            "phase1_slack": float(f@first.x), "phase1_iterations": first.nit,
            "phase1_status": str(first.message),
            "phase2_iterations": second.nit if second is not None else 0,
            "phase2_status": str(second.message) if second is not None else "not run",
            "phase2_certificate": second_audit, "returned_phase": chosen_phase,
            "objective_scale": objective_scale,
            "stationarity_verified": False,
            "status": "certified_feasible" if chosen_audit["certified"] else "uncertified_candidate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON: p_init, obstacles, n_seg; optional solver settings")
    parser.add_argument("output", type=Path)
    parser.add_argument("--timing-retries", action="store_true",
                        help="After an uncertified solve, try a documented finite timing-seed portfolio")
    args = parser.parse_args()
    problem = json.loads(args.input.read_text())
    result = solve_with_timing_restarts(**problem) if args.timing_retries else solve(**problem)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(result["status"], result["certificate"]["minimum_margin"], result["objective"])


def solve_with_timing_restarts(p_init, obstacles, n_seg, **settings):
    """Retry uncertified local solves; record each seed and never infer infeasibility.

    Waiting seeds use exact degree-elevated s^2 and s^3 spatial progress
    along the endpoint chord, with linear time. A final hurry seed uses
    monotone sqrt(i/N) Bernstein coefficients (not exactly sqrt(s)).
    These are new initializations, not extra iterations from the failed point.
    The portfolio is incomplete and is not a global planning guarantee.
    """
    p_init = np.asarray(p_init, dtype=float)
    degree = len(p_init)-1
    seeds = [("requested", p_init)]
    for power in (2, 3):
        if power > degree:
            continue
        progress = np.array([math.comb(i, power)/math.comb(degree, power)
                             if i >= power else 0. for i in range(degree+1)])
        seeds.append((f"wait_power_{power}", progress))
    seeds.append(("hurry_sqrt_coefficients", np.sqrt(np.linspace(0., 1., degree+1))))
    attempts = []
    result = None
    for label, seed in seeds:
        if seed.ndim == 1:
            q = np.empty_like(p_init)
            q[:, :-1] = p_init[0, :-1] + seed[:, None]*(p_init[-1, :-1]-p_init[0, :-1])
            q[:, -1] = np.linspace(p_init[0, -1], p_init[-1, -1], degree+1)
            seed = q
        result = solve(seed, obstacles, n_seg, **settings)
        attempts.append({"seed": label, "initial_controls": seed.tolist(),
                         "status": result["status"], "certificate": result["certificate"],
                         "phase1_iterations": result["phase1_iterations"],
                         "phase2_iterations": result["phase2_iterations"]})
        if result["certificate"]["certified"]:
            break
    result["attempts"] = attempts
    result["selected_seed"] = attempts[-1]["seed"]
    return result


if __name__ == "__main__":
    main()
