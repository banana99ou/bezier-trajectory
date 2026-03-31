"""
Space-Time Bézier Trajectory Optimizer

Optimizes a Bézier curve in (x, y, t) space to avoid constant-velocity
moving obstacles, using the same SCP + De Casteljau + supporting half-space
approach from the orbital docking framework.

Reuses baseline modules:
  - orbital_docking.bezier          (D, E, G matrices)
  - orbital_docking.de_casteljau    (segment subdivision matrices)

What's new here:
  1. KOZ constraints adapted for moving obstacles (pos0 + vel*t_seg)
  2. Time-limited obstacles (t_start / t_end)
  3. Time monotonicity constraint
  4. Objective penalizes only spatial acceleration

Outputs optimized control points as JSON for the interactive HTML demo.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

from orbital_docking.bezier import get_D_matrix, get_E_matrix, get_G_matrix, BezierCurve
from orbital_docking.de_casteljau import segment_matrices_equal_params


# ══════════════════════════════════════════════════════════════════
# Space-time KOZ constraints for moving obstacles (vectorized)
# ══════════════════════════════════════════════════════════════════

def build_spacetime_koz_constraints(A_list, P, obstacles, dim=3):
    """
    Build supporting half-space constraints for moving obstacles.

    Same idea as baseline build_koz_constraints, but the KOZ center
    moves: obstacle position = pos0 + vel * t_seg, where t_seg is the
    time coordinate of the segment centroid.

    Vectorized: builds constraint rows per (segment, obstacle) batch
    instead of element-by-element Python loops.
    """
    n_cp = P.shape[0]
    spatial_dim = dim - 1
    n_cp_seg = A_list[0].shape[0]  # N+1

    # Precompute obstacle arrays
    n_obs = len(obstacles)
    obs_pos0 = np.array([o['pos0'] for o in obstacles])     # (n_obs, spatial_dim)
    obs_vel  = np.array([o['vel']  for o in obstacles])      # (n_obs, spatial_dim)
    obs_r    = np.array([o['r']    for o in obstacles])      # (n_obs,)
    obs_t0   = np.array([o.get('t_start', -np.inf) for o in obstacles])
    obs_t1   = np.array([o.get('t_end',    np.inf) for o in obstacles])

    blocks_A = []
    blocks_lb = []

    for A_seg in A_list:
        Q = A_seg @ P                    # (n_cp_seg, dim)
        centroid = Q.mean(axis=0)
        t_seg = centroid[-1]
        c_spatial = centroid[:spatial_dim]

        # Which obstacles are active at this segment's centroid time?
        active = (t_seg >= obs_t0) & (t_seg <= obs_t1)
        if not active.any():
            continue

        # Obstacle positions at t_seg
        o_positions = obs_pos0 + obs_vel * t_seg   # (n_obs, spatial_dim)

        # Normals: centroid - obstacle center (spatial only)
        diffs = c_spatial[None, :] - o_positions     # (n_obs, spatial_dim)
        dists = np.linalg.norm(diffs, axis=1)        # (n_obs,)
        valid = active & (dists > 1e-10)
        if not valid.any():
            continue

        idx = np.where(valid)[0]
        n_hats = diffs[idx] / dists[idx, None]       # (n_valid, spatial_dim)
        o_pos_valid = o_positions[idx]                # (n_valid, spatial_dim)
        r_valid = obs_r[idx]                          # (n_valid,)
        n_valid = len(idx)

        # Build constraint matrix block.
        # For obstacle o and segment CP i, the constraint is:
        #   sum_k  n_hat[o,k] * sum_j A_seg[i,j] * P[j,k]  >=  n_hat[o] . o_pos[o] + r[o]
        #
        # In flat layout P = [P[0,0], P[0,1], ..., P[0,dim-1], P[1,0], ...],
        # coefficient for P[j, k] sits at column j*dim + k.
        # So for each spatial dim k, we place n_hat[o,k] * A_seg[i,j] at col j*dim+k.

        n_rows = n_valid * n_cp_seg
        A_block = np.zeros((n_rows, n_cp * dim))

        for k in range(spatial_dim):
            # n_hats[:, k] shaped (n_valid,) ; A_seg shaped (n_cp_seg, n_cp)
            # Product: (n_valid, n_cp_seg, n_cp)
            coeffs = n_hats[:, k:k+1, None] * A_seg[None, :, :]
            # Reshape to (n_rows, n_cp) and scatter into columns k, k+dim, k+2*dim, ...
            A_block[:, k::dim] = coeffs.reshape(n_rows, n_cp)

        # Lower bounds: n_hat . o_pos + r, repeated for each CP in segment
        lb_per_obs = np.einsum('ok,ok->o', n_hats, o_pos_valid) + r_valid
        lb_block = np.repeat(lb_per_obs, n_cp_seg)

        blocks_A.append(A_block)
        blocks_lb.append(lb_block)

    if not blocks_A:
        return None

    return LinearConstraint(np.vstack(blocks_A),
                            np.concatenate(blocks_lb),
                            np.full(sum(len(b) for b in blocks_lb), np.inf))


# ══════════════════════════════════════════════════════════════════
# Boundary constraints (fix start/end position)
# ══════════════════════════════════════════════════════════════════

def build_boundary_constraints(n_cp, dim, p_start, p_end):
    """Fix first and last control points via equality constraints."""
    rows = []
    vals = []
    for k in range(dim):
        row0 = np.zeros(n_cp * dim); row0[k] = 1.0
        rows.append(row0); vals.append(p_start[k])
        rowN = np.zeros(n_cp * dim); rowN[(n_cp - 1) * dim + k] = 1.0
        rows.append(rowN); vals.append(p_end[k])

    A = np.array(rows); v = np.array(vals)
    return LinearConstraint(A, v, v)


# ══════════════════════════════════════════════════════════════════
# Time monotonicity constraint
# ══════════════════════════════════════════════════════════════════

def build_time_monotonicity(n_cp, dim, min_dt=0.3):
    """Enforce P[i+1, t] - P[i, t] >= min_dt (time coordinate = last dim)."""
    t_idx = dim - 1
    rows = []
    for i in range(n_cp - 1):
        row = np.zeros(n_cp * dim)
        row[(i + 1) * dim + t_idx] = 1.0
        row[i * dim + t_idx] = -1.0
        rows.append(row)
    A = np.array(rows)
    return LinearConstraint(A, np.full(len(rows), min_dt), np.full(len(rows), np.inf))


# ══════════════════════════════════════════════════════════════════
# Objective: L2 spatial acceleration energy (uses baseline G_tilde)
# ══════════════════════════════════════════════════════════════════

def build_energy_objective(N, dim):
    """
    Build H matrix for 0.5 x^T H x objective that penalizes spatial
    acceleration energy only (not time-acceleration).

    Uses baseline BezierCurve.G_tilde = (EDED)^T G (EDED).
    """
    # G_tilde is (N+1, N+1) — the quadratic form in control-point space
    bc = BezierCurve(np.zeros((N + 1, dim)))
    G_tilde = bc.G_tilde  # from baseline

    spatial_dim = dim - 1
    n_cp = N + 1
    H = np.zeros((n_cp * dim, n_cp * dim))

    # Only penalize spatial dimensions, not the time coordinate
    for k in range(spatial_dim):
        for i in range(n_cp):
            for j in range(n_cp):
                H[i * dim + k, j * dim + k] += G_tilde[i, j]

    return 0.5 * (H + H.T)


def build_initial_guess(p_start, p_end, n_cp, init_curve=None):
    """
    Build the initial control polygon for SCP.

    The default is a straight line in space-time. For 2D spatial demos we can
    optionally add a quadratic-looking lateral bow so the initial curve swings
    toward a corner instead of passing through the middle of the workspace.
    """
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    s_vals = np.linspace(0.0, 1.0, n_cp)
    P = np.array([(1.0 - s) * p_start + s * p_end for s in s_vals])

    if not init_curve or init_curve.get('mode', 'straight') == 'straight':
        return P

    if init_curve.get('mode') != 'quadratic_bow':
        raise ValueError(f"Unknown init_curve mode: {init_curve.get('mode')}")

    spatial_dim = p_start.size - 1
    if spatial_dim < 2:
        return P

    bow = float(init_curve.get('bow', 0.0))
    if bow <= 0.0:
        return P

    chord = p_end[:2] - p_start[:2]
    chord_norm = np.linalg.norm(chord)
    if chord_norm < 1e-12:
        return P

    normal = np.array([-chord[1], chord[0]], dtype=float) / chord_norm
    workspace_center = init_curve.get('workspace_center')
    if workspace_center is not None:
        workspace_center = np.asarray(workspace_center, dtype=float)
        midpoint = 0.5 * (p_start[:2] + p_end[:2])
        if np.dot(workspace_center - midpoint, normal) > 0.0:
            normal *= -1.0
    normal *= float(init_curve.get('side', 1.0))

    bump = bow * (4.0 * s_vals * (1.0 - s_vals))
    P[:, :2] += bump[:, None] * normal
    return P


# ══════════════════════════════════════════════════════════════════
# SCP loop
# ══════════════════════════════════════════════════════════════════

def optimize_spacetime(N, dim, p_start, p_end, obstacles, n_seg=8,
                       max_iter=30, tol=1e-6, scp_prox_weight=0.5,
                       verbose=True, init_curve=None):
    """Successive convexification optimizer for space-time Bézier."""
    n_cp = N + 1
    p_start, p_end = np.array(p_start), np.array(p_end)

    # Initialize with either a straight line or a bowed demo curve.
    P = build_initial_guess(p_start, p_end, n_cp, init_curve=init_curve)

    # Precompute (using baseline De Casteljau)
    A_list = segment_matrices_equal_params(N, n_seg)
    H_energy = build_energy_objective(N, dim)
    bc_con = build_boundary_constraints(n_cp, dim, p_start, p_end)
    mono_con = build_time_monotonicity(n_cp, dim, min_dt=0.1)

    # Bounds
    lb = np.full(n_cp * dim, -20.0)
    ub = np.full(n_cp * dim, 20.0)
    for k in range(dim):
        lb[k] = ub[k] = p_start[k]
        lb[(n_cp-1)*dim + k] = ub[(n_cp-1)*dim + k] = p_end[k]
    for i in range(n_cp):
        lb[i*dim + dim-1] = 0.0
        ub[i*dim + dim-1] = p_end[-1] * 1.5
    bounds = Bounds(lb, ub)

    if verbose:
        print(f"SCP: N={N}, dim={dim}, n_seg={n_seg}, n_cp={n_cp}, "
              f"n_obs={len(obstacles)}")

    best_P = P.copy()
    for it in range(max_iter):
        x_ref = P.flatten()

        koz_con = build_spacetime_koz_constraints(A_list, P, obstacles, dim)

        H = H_energy + scp_prox_weight * np.eye(n_cp * dim)
        f = -scp_prox_weight * x_ref

        constraints = [bc_con, mono_con]
        if koz_con is not None:
            constraints.append(koz_con)

        res = minimize(
            lambda x: 0.5 * x @ H @ x + f @ x,
            x_ref, jac=lambda x: H @ x + f,
            hess=lambda x: H,
            method='trust-constr',
            constraints=constraints, bounds=bounds,
            options={'maxiter': 80, 'gtol': 1e-9, 'verbose': 0},
        )

        P_new = res.x.reshape(n_cp, dim)
        delta = np.linalg.norm(P_new - P)
        clearance = compute_min_clearance(P_new, obstacles, dim)

        if verbose:
            print(f"  iter {it+1}: delta={delta:.6f}, cost={res.fun:.4f}, "
                  f"clearance={clearance:.4f}")

        P = P_new
        if clearance > 0:
            best_P = P.copy()
        if delta < tol:
            if verbose: print(f"  Converged at iter {it+1}")
            break

    final = compute_min_clearance(P, obstacles, dim)
    if final < 0 and compute_min_clearance(best_P, obstacles, dim) > 0:
        P = best_P
        if verbose: print("  Using best feasible iterate")

    return P


# ══════════════════════════════════════════════════════════════════
# Clearance check (vectorized)
# ══════════════════════════════════════════════════════════════════

def compute_min_clearance(P, obstacles, dim, n_eval=1500):
    """Evaluate min clearance of Bézier curve to all obstacles."""
    bc = BezierCurve(P)
    taus = np.linspace(0, 1, n_eval)
    pts = np.array([bc.point(t) for t in taus])  # (n_eval, dim)

    spatial_dim = dim - 1
    worst = np.inf

    for obs in obstacles:
        pos0 = np.array(obs['pos0'])
        vel = np.array(obs['vel'])
        r = obs['r']
        t0 = obs.get('t_start', -np.inf)
        t1 = obs.get('t_end', np.inf)

        t_vals = pts[:, -1]
        active = (t_vals >= t0) & (t_vals <= t1)
        if not active.any():
            continue

        o_positions = pos0[None, :] + vel[None, :] * t_vals[active, None]
        dists = np.linalg.norm(pts[active, :spatial_dim] - o_positions, axis=1) - r
        worst = min(worst, dists.min())

    return worst


# ══════════════════════════════════════════════════════════════════
# Helper: make a wall from a line of overlapping circles
# ══════════════════════════════════════════════════════════════════

def make_wall(p1, p2, thickness=0.5, spacing=0.7, color='#e67e22',
              name_prefix='W', vel=None, t_start=None, t_end=None):
    """Create a row of overlapping circles between p1 and p2."""
    p1, p2 = np.array(p1), np.array(p2)
    length = np.linalg.norm(p2 - p1)
    n_circles = max(2, int(length / spacing) + 1)
    obstacles = []
    for i in range(n_circles):
        alpha = i / (n_circles - 1)
        pos = ((1 - alpha) * p1 + alpha * p2).tolist()
        obs = {'pos0': pos, 'vel': vel or [0.0, 0.0], 'r': thickness,
               'color': color, 'name': f'{name_prefix}{i}'}
        if t_start is not None: obs['t_start'] = t_start
        if t_end is not None: obs['t_end'] = t_end
        obstacles.append(obs)
    return obstacles


# ══════════════════════════════════════════════════════════════════
# Scenarios
# ══════════════════════════════════════════════════════════════════

def scenario_original():
    return {
        'name': 'original',
        'title': '3 Moving Obstacles',
        'init_curve': {
            'mode': 'quadratic_bow',
            'bow': 2.3,
            'side': 1.0,
            'workspace_center': [5.0, 5.0],
        },
        'obstacles': [
            {'pos0': [2.0, 8.0], 'vel': [0.5, -0.7], 'r': 0.8, 'color': '#e74c3c', 'name': 'A'},
            {'pos0': [6.0, 2.0], 'vel': [-0.3, 0.5], 'r': 0.7, 'color': '#2980b9', 'name': 'B'},
            {'pos0': [4.5, 5.5], 'vel': [0.1, -0.3], 'r': 0.6, 'color': '#27ae60', 'name': 'C'},
        ],
        'start': [0.5, 1.0, 0.0],
        'end': [8.5, 8.5, 10.0],
        'T': 10.0,
    }


def scenario_diverse():
    return {
        'name': 'diverse',
        'title': 'Diverse Moving Obstacles',
        'init_curve': {
            'mode': 'quadratic_bow',
            'bow': 3.2,
            'side': 1.0,
            'workspace_center': [5.0, 5.0],
        },
        'obstacles': [
            {'pos0': [1.0, 6.0], 'vel': [0.8, -0.1], 'r': 0.6, 'color': '#e74c3c', 'name': 'A'},
            {'pos0': [5.0, 5.0], 'vel': [0.05, -0.15], 'r': 1.0, 'color': '#2980b9', 'name': 'B'},
            {'pos0': [8.0, 1.0], 'vel': [-0.6, 0.4], 'r': 0.5, 'color': '#27ae60', 'name': 'C'},
            {'pos0': [3.0, 1.5], 'vel': [0.3, 0.6], 'r': 0.4, 'color': '#9b59b6', 'name': 'D'},
            {'pos0': [4.0, 8.0], 'vel': [0.1, -0.5], 'r': 0.9, 'color': '#e67e22', 'name': 'E'},
            {'pos0': [7.0, 7.0], 'vel': [-0.4, -0.3], 'r': 0.35, 'color': '#1abc9c', 'name': 'F'},
            {'pos0': [2.5, 4.0], 'vel': [0.0, 0.0], 'r': 0.7, 'color': '#f39c12', 'name': 'G'},
        ],
        'start': [0.5, 0.5, 0.0],
        'end': [9.0, 9.0, 10.0],
        'T': 10.0,
    }


def scenario_wall():
    """Wall that disappears at t=5. Curve should wait then pass through."""
    wall_obs = make_wall(
        p1=[2.5, 0.0], p2=[2.5, 10.0],
        thickness=0.5, spacing=0.8,
        color='#e67e22', name_prefix='W',
        t_start=0.0, t_end=5.0,
    )
    other_obs = [
        {'pos0': [7.0, 3.0], 'vel': [0.0, 0.3], 'r': 0.6, 'color': '#2980b9', 'name': 'M1'},
        {'pos0': [6.5, 8.0], 'vel': [0.2, -0.2], 'r': 0.5, 'color': '#27ae60', 'name': 'M2'},
    ]
    return {
        'name': 'wall',
        'title': 'Disappearing Wall (t<5)',
        'init_curve': {'mode': 'straight'},
        'obstacles': wall_obs + other_obs,
        'start': [1.0, 5.0, 0.0],
        'end': [8.0, 5.0, 10.0],
        'T': 10.0,
    }


# ══════════════════════════════════════════════════════════════════
# Run optimization for a scenario
# ══════════════════════════════════════════════════════════════════

def optimize_scenario(scenario, configs):
    obstacles = scenario['obstacles']
    p_start = scenario['start']
    p_end = scenario['end']
    init_curve = scenario.get('init_curve')

    results = {}
    for N, n_seg in configs:
        print(f"\n{'='*60}")
        print(f"[{scenario['name']}] degree={N}, segments={n_seg}")
        print(f"{'='*60}")

        P_opt = optimize_spacetime(
            N=N, dim=3, p_start=p_start, p_end=p_end,
            obstacles=obstacles, n_seg=n_seg,
            max_iter=200, scp_prox_weight=0.3,
            init_curve=init_curve,
        )

        clearance = compute_min_clearance(P_opt, obstacles, dim=3, n_eval=3000)
        print(f"  Final clearance: {clearance:.4f}")

        key = f"N{N}_seg{n_seg}"
        results[key] = {
            'N': N, 'n_seg': n_seg,
            'control_points': P_opt.tolist(),
            'min_clearance': float(clearance),
            'feasible': bool(clearance > 0),
        }

    feasible = {k: v for k, v in results.items() if v['feasible']}
    best_key = (max(feasible, key=lambda k: feasible[k]['min_clearance'])
                if feasible else
                max(results, key=lambda k: results[k]['min_clearance']))

    print(f"[{scenario['name']}] Best: {best_key} "
          f"(clearance={results[best_key]['min_clearance']:.4f})")

    return {
        'name': scenario['name'], 'title': scenario['title'],
        'best': best_key, 'obstacles': obstacles,
        'start': p_start, 'end': p_end, 'T': scenario['T'],
        'init_curve': init_curve,
        'results': results,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

SCENARIO_MAP = {
    'original': (scenario_original,  [(4, 8), (6, 8), (8, 8)]),
    'diverse':  (scenario_diverse,   [(8, 8), (8, 16), (10, 16)]),
    'wall':     (scenario_wall,      [(8, 16), (10, 16), (10, 24)]),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Space-time Bézier optimizer')
    parser.add_argument('scenarios', nargs='*', default=list(SCENARIO_MAP.keys()),
                        choices=list(SCENARIO_MAP.keys()),
                        help='Which scenarios to run (default: all)')
    args = parser.parse_args()

    # Load existing results if running a subset (so we merge, not overwrite)
    out_path = 'figures/spacetime_scenarios.json'
    all_outputs = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_outputs = json.load(f)

    for name in args.scenarios:
        sc_fn, configs = SCENARIO_MAP[name]
        all_outputs[name] = optimize_scenario(sc_fn(), configs)

    os.makedirs('figures', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_outputs, f, indent=2)
    print(f"\nSaved: {out_path}")

    for name in args.scenarios:
        data = all_outputs[name]
        best = data['results'][data['best']]
        print(f"\n{data['title']}: best={data['best']}, "
              f"clearance={best['min_clearance']:.4f}")
        for row in best['control_points']:
            print(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],")


if __name__ == '__main__':
    main()
