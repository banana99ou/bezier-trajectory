#!/usr/bin/env python3
"""
Quadratic 3D Bézier orbital transfer with Earth keep-out, using:
- Adaptive detection of violating subintervals (triangle–sphere test)
- Linear half-space constraints per violating interval (centroid outward normal)
- Single global QP over original control points (fixed end points)
- Fixed-point iterations until feasible / converged
- Visualization and split-count comparison

Author: (you)
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  Bézier (Quadratic) utilities
# ─────────────────────────────────────────────────────────────────────────────

class BezierQuad3D:
    """Quadratic 3D Bézier with control points P0,P1,P2 (shape (3,3))."""

    def __init__(self, control_points: np.ndarray):
        P = np.asarray(control_points, dtype=float)
        if P.shape != (3, 3):
            raise ValueError("Quadratic Bézier requires control_points shape (3,3) for 3D.")
        self.P = P  # rows: P0,P1,P2; cols: x,y,z

    def eval(self, t: float) -> np.ndarray:
        """Evaluate B(t) for t in [0,1]."""
        P0, P1, P2 = self.P
        u = 1.0 - t
        return (u*u) * P0 + 2*u*t * P1 + (t*t) * P2

    def split(self, tau: float) -> Tuple["BezierQuad3D", "BezierQuad3D"]:
        """De Casteljau split at tau ∈ [0,1]. Return left and right segments."""
        P0, P1, P2 = self.P
        P01 = (1-tau)*P0 + tau*P1
        P12 = (1-tau)*P1 + tau*P2
        P0112 = (1-tau)*P01 + tau*P12
        left = np.vstack([P0, P01, P0112])
        right = np.vstack([P0112, P12, P2])
        return BezierQuad3D(left), BezierQuad3D(right)

    def split_N(self, N: int) -> List["BezierQuad3D"]:
        """Split into N equal-parameter segments."""
        if N <= 0:
            raise ValueError("N must be a positive integer")
        segs = []
        a = 0.0
        for k in range(N):
            b = (k+1)/N
            segs.append(self.subcurve(a, b))
            a = b
        return segs

    def subcurve(self, a: float, b: float) -> "BezierQuad3D":
        """Return sub-curve restricted to t ∈ [a,b]."""
        if not (0.0 <= a < b <= 1.0):
            raise ValueError("Require 0 <= a < b <= 1")
        # Split at a, take right; then split that at s=(b-a)/(1-a), take left
        R = self.split(a)[1]
        s = (b - a) / (1.0 - a)
        L_of_R, _ = R.split(s)
        return L_of_R

    def subcurve_weights(self, a: float, b: float) -> np.ndarray:
        """
        Return W (3x3) such that: sub_control_points(a,b) = W @ P, where P is (3x3) with rows P0,P1,P2.
        Each row of W gives the convex weights over [P0,P1,P2] for that subcurve control point.
        """
        # Use identity basis trick to extract weights
        I = np.eye(3)
        basis_curve = BezierQuad3D(I)  # treat "points" as 3D basis vectors
        sub_basis = basis_curve.subcurve(a, b).P  # shape (3,3), rows are weights over original P rows
        # sub_basis[j] is the weight vector (w0,w1,w2) for sub-control-point j
        return sub_basis  # (3,3)

    def second_derivative_vector(self) -> np.ndarray:
        """
        B''(t) is constant for quadratic:
        B''(t) = 2*(P2 - 2P1 + P0)
        """
        P0, P1, P2 = self.P
        return 2.0*(P2 - 2.0*P1 + P0)

    def centroid(self) -> np.ndarray:
        return self.P.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry: triangle–sphere intersection (for quadratic Bézier control triangle)
# ─────────────────────────────────────────────────────────────────────────────

def _closest_point_on_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-15)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab

def closest_point_on_triangle(p, a, b, c):
    """
    Return closest point on triangle Δ(a,b,c) to point p. Robust barycentric clamp.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a

    bp = p - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b

    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3 + 1e-15)
        return a + v * ab

    cp = p - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c

    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6 + 1e-15)
        return a + w * ac

    va = d3*d6 - d5*d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-15)
        return b + w * (c - b)

    # Inside face region
    denom = 1.0 / (np.dot(ab, ab)*np.dot(ac, ac) - np.dot(ab, ac)**2 + 1e-15)
    v = (np.dot(ac, ac)*np.dot(ab, ap) - np.dot(ab, ac)*np.dot(ac, ap)) * denom
    w = (np.dot(ab, ab)*np.dot(ac, ap) - np.dot(ab, ac)*np.dot(ab, ap)) * denom
    return a + v*ab + w*ac

def triangle_sphere_margin(tri: np.ndarray, center: np.ndarray, R: float) -> float:
    """
    Positive margin => triangle safely outside sphere by that distance.
    Negative margin => intersects/inside by |margin|.
    tri: (3,3) triangle vertices
    """
    q = closest_point_on_triangle(center, tri[0], tri[1], tri[2])
    return np.linalg.norm(q - center) - R


# ─────────────────────────────────────────────────────────────────────────────
#  Detect violating intervals by adaptive subdivision
# ─────────────────────────────────────────────────────────────────────────────

def find_violating_intervals(curve: BezierQuad3D,
                             center: np.ndarray,
                             R: float,
                             max_depth: int = 12,
                             tol_len: float = 1e-3) -> List[Tuple[float, float]]:
    """
    Return list of parameter intervals [ta,tb] whose control triangle may
    intersect the sphere. Conservative (may over-mark small near-tangent cases).
    """
    out: List[Tuple[float, float]] = []
    stack = [(0.0, 1.0, curve, 0)]
    while stack:
        ta, tb, seg, depth = stack.pop()
        tri = seg.P
        margin = triangle_sphere_margin(tri, center, R)
        if margin > 0.0:
            # definitely outside: safe
            continue

        # If all three ctrl pts are well inside, definitely violating
        dists = np.linalg.norm(tri - center, axis=1)
        if np.all(dists < R - 1e-6):
            out.append((ta, tb))
            continue

        if depth >= max_depth or (tb - ta) < tol_len:
            # uncertain or thin: mark as violating to be safe
            out.append((ta, tb))
            continue

        L, Rseg = seg.split(0.5)
        tm = 0.5*(ta + tb)
        stack.append((tm, tb, Rseg, depth+1))
        stack.append((ta, tm, L,    depth+1))

    # Merge adjacent intervals
    out.sort()
    merged: List[Tuple[float, float]] = []
    for a, b in out:
        if not merged or a > merged[-1][1] + 1e-9:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(a, b) for a, b in merged]


# ─────────────────────────────────────────────────────────────────────────────
#  Build and solve the single global QP (one curve), fixed-point iterations
# ─────────────────────────────────────────────────────────────────────────────

def global_repair_QP(P_init: np.ndarray,
                     center: np.ndarray,
                     R: float,
                     max_iters: int = 10,
                     lambda_reg: float = 1.0,
                     lambda_acc: float = 1e-2,
                     eps_stop: float = 1e-5,
                     max_depth: int = 12) -> Tuple[np.ndarray, List[dict]]:
    """
    Solve one-curve global optimization with half-space constraints from violating intervals.
    Endpoints (P0,P2) are fixed to initial endpoints.
    Minimize: lambda_reg * ||P - P_prev||^2 + lambda_acc * ||P0 - 2P1 + P2||^2
    subject to: n_i^T (W_ij @ P) >= R, for each violating interval i and sub-ctrl j=0,1,2.
    Returns (P_opt, history)
    """
    P_prev = P_init.copy()
    history = []

    for k in range(max_iters):
        curve_prev = BezierQuad3D(P_prev)
        viol = find_violating_intervals(curve_prev, center, R, max_depth=max_depth)

        # Compute normals (constants at iteration k) and weights
        normals = []
        weights = []  # list of (W_i: 3x3) per interval
        for (a, b) in viol:
            sub = curve_prev.subcurve(a, b)
            c_i = sub.centroid()
            n_hat = c_i - center
            n_norm = np.linalg.norm(n_hat)
            if n_norm < 1e-12:
                # degenerate: pick some outward based on midpoint on curve
                mid = sub.eval(0.5)
                n_hat = mid - center
                n_norm = np.linalg.norm(n_hat) + 1e-12
            n_hat /= n_norm
            normals.append(n_hat)
            weights.append(curve_prev.subcurve_weights(a, b))  # 3x3

        # Decision variable: P (3x3)
        P = cp.Variable((3, 3))

        # Equality constraints: lock endpoints
        cons = [
            P[0, :] == P_init[0, :],  # start fixed
            P[2, :] == P_init[2, :]   # goal fixed
        ]

        # Half-space constraints for all violating intervals
        for n_hat, W in zip(normals, weights):
            # For each sub-control point j (0..2): n^T (W_j @ P) >= R
            for j in range(3):
                wj = W[j, :]  # shape (3,)
                p_sub_j = wj @ P          # (3,) affine in P
                cons.append(n_hat @ p_sub_j >= R)

        # Objective: minimal change + curvature (acceleration proxy)
        acc_vec = P[0, :] - 2.0*P[1, :] + P[2, :]
        obj = (lambda_reg * cp.sum_squares(P - P_prev)
               + lambda_acc * cp.sum_squares(acc_vec))

        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)

        if P.value is None:
            raise RuntimeError("QP infeasible or solver failed. Try loosening constraints or increasing max_depth.")

        P_new = P.value
        # Record
        history.append({
            "iter": k,
            "num_intervals": len(viol),
            "obj": prob.value,
        })

        # Stopping conditions
        delta = np.linalg.norm(P_new - P_prev)
        if len(viol) == 0 or delta < eps_stop:
            return P_new, history

        P_prev = P_new

    return P_prev, history


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_sphere(ax, R: float, center=(0,0,0), color='gray', alpha=0.25, res=40):
    cx, cy, cz = center
    u = np.linspace(0, 2*np.pi, res)
    v = np.linspace(0, np.pi, res)
    uu, vv = np.meshgrid(u, v)
    x = cx + R*np.cos(uu)*np.sin(vv)
    y = cy + R*np.sin(uu)*np.sin(vv)
    z = cz + R*np.cos(vv)
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, alpha=alpha)

def plot_curve(ax, curve: BezierQuad3D, label=None, color='C0', lw=2.0, n=200):
    ts = np.linspace(0, 1, n)
    P = np.array([curve.eval(t) for t in ts])
    ax.plot(P[:,0], P[:,1], P[:,2], color=color, lw=lw, label=label)

def plot_ctrl_polygon(ax, curve: BezierQuad3D, color='k', ls='--', marker='o'):
    P = curve.P
    ax.plot(P[:,0], P[:,1], P[:,2], ls=ls, marker=marker, color=color, alpha=0.8)


# ─────────────────────────────────────────────────────────────────────────────
#  Demo / Paper Figure Generator
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    np.random.seed(0)
    # Earth at origin
    center = np.array([0.0, 0.0, 0.0])
    R = 6.0  # exclusion radius (scaled units)

    # Start / goal (outside Earth)
    P0 = np.array([-12.0, -10.0,  2.0])
    P2 = np.array([ 12.0,  10.0, -1.0])

    # Pick a middle control to deliberately dip into Earth
    P1 = np.array([  0.0,  0.0, -8.0])  # this causes a deep incursion
    P_init = np.vstack([P0, P1, P2])

    curve0 = BezierQuad3D(P_init)

    # Run global fixed-point QP repair
    P_opt, hist = global_repair_QP(P_init, center, R,
                                   max_iters=15,
                                   lambda_reg=1.0,
                                   lambda_acc=5e-2,
                                   eps_stop=1e-6,
                                   max_depth=12)
    curve_opt = BezierQuad3D(P_opt)

    # Figure 1: before vs after
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot_sphere(ax, R, center, color='gray', alpha=0.3)
    plot_curve(ax, curve0, label='Initial curve', color='C3', lw=2.0)
    plot_ctrl_polygon(ax, curve0, color='C3', ls='--')
    plot_curve(ax, curve_opt, label='Optimized (single curve)', color='C0', lw=2.5)
    plot_ctrl_polygon(ax, curve_opt, color='C0', ls='--')
    ax.set_title("Earth keep-out: single-curve global optimization")
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()

    # Figure 2: show that increasing subdivision constraint density helps
    split_list = [2, 4, 8, 16]
    colors = ['C1','C2','C4','C6']
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    plot_sphere(ax2, R, center, color='gray', alpha=0.3)
    plot_curve(ax2, curve0, label='Initial', color='0.5', lw=1.5)
    plot_ctrl_polygon(ax2, curve0, color='0.5', ls='--')

    # For comparison, we "force" constraint generation by limiting max_depth
    for sp, col in zip(split_list, colors):
        P_prev = P_init.copy()
        P_tmp, _ = global_repair_QP(P_prev, center, R,
                                    max_iters=15,
                                    lambda_reg=1.0,
                                    lambda_acc=5e-2,
                                    eps_stop=1e-6,
                                    max_depth=int(np.log2(sp))+1)  # rough tie to split density
        c = BezierQuad3D(P_tmp)
        plot_curve(ax2, c, label=f'Optimized (constraints~{sp})', color=col, lw=2.0)

    ax2.set_title("More violating subsegments ⇒ more linear walls ⇒ easier feasibility")
    ax2.legend()
    ax2.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

    # Print iteration log
    for rec in hist:
        print(f"Iter {rec['iter']:2d} | violating intervals: {rec['num_intervals']:2d} | obj={rec['obj']:.4e}")


if __name__ == "__main__":
    demo()
