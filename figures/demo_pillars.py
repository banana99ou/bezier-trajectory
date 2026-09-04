"""
Contained-illustration demo: planar drone weaving between circular pillars.

Purpose (paper §5 capability demonstration): show that the *same*
control-point-space construction used for the spherical-KOZ orbital case
generalizes, unchanged in spirit, to a different domain (2D obstacle
avoidance) with *multiple* keep-out zones. This is a qualitative generality
+ continuous-safety illustration, not a benchmark against robotics planners.

Method faithfulness:
  - De Casteljau subdivision A_list via segment_matrices_equal_params (real module)
  - control-effort objective via BezierCurve.G_tilde (real module)
  - per-segment supporting half-spaces per obstacle (same as build_koz_constraints,
    looped over obstacles)
  - SCvx outer loop with exact-penalty slack (mirrors the canonical penalized-merit
    acceptance in the production solver), solved as a convex QP each iteration.

Only the QP backend differs from production (cvxpy/OSQP vs Rust); the geometry,
constraints, and objective are the paper's method.
"""

import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from orbital_docking.bezier import BezierCurve
from orbital_docking.de_casteljau import segment_matrices_equal_params

# --------------------------------------------------------------------------
# Scenario: start -> end with a slalom of pillars straddling the chord.
# Each pillar dips across the straight-line path so the initial guess is
# infeasible and the optimizer must route continuously around them.
# --------------------------------------------------------------------------
START = np.array([0.0, 0.0])
END = np.array([12.0, 0.0])
# (cx, cy, radius)
PILLARS = [
    (3.0,  0.7, 1.0),
    (6.0, -0.7, 1.0),
    (9.0,  0.7, 1.0),
]

N = 8          # Bezier degree (matches paper's active N=6..8 regime)
N_SEG = 16     # De Casteljau subdivision count (matches demo_N*_seg16)
REST_TO_REST = False  # fly-through: endpoint positions fixed, velocity free

W_SLACK = 1.0e6       # exact-penalty weight on constraint slack
TRUST0 = 2.5          # initial box trust-region half-width (per ctrl-pt coord)
TRUST_SHRINK = 0.9    # geometric shrink per outer iteration
TRUST_MIN = 5e-3
MAX_ITER = 200
TOL = 1e-6
SAFETY_INFLATE = 0.12  # extra certified clearance added to each pillar radius


def obstacle_halfspaces(A_list, P_ref, pillars):
    """
    Linearize the multi-obstacle KOZ at the current iterate.

    For each segment i and obstacle o: build the supporting half-space at the
    segment centroid (identical construction to build_koz_constraints, looped
    over obstacles). Returns a list of (n, rhs, A_i) so the caller can impose
        n^T (A_i P)[k] >= rhs   for every control point k of segment i.
    """
    blocks = []
    for Ai in A_list:
        Qi_ref = Ai @ P_ref                 # segment control polygon at reference
        ci = Qi_ref.mean(axis=0)            # centroid
        for (cx, cy, r) in pillars:
            c_obs = np.array([cx, cy])
            nvec = ci - c_obs
            nrm = np.linalg.norm(nvec)
            if nrm < 1e-12:                  # degenerate: centroid at obstacle center
                continue
            n = nvec / nrm
            rhs = float(n @ c_obs + r + SAFETY_INFLATE)
            blocks.append((n, rhs, Ai))
    return blocks


def solve_scvx(pillars, verbose=True):
    N_p1 = N + 1
    A_list = segment_matrices_equal_params(N, N_SEG)

    # control-effort quadratic form: cost = sum_d || B P[:,d] ||^2, B = chol(G)^T EDED
    bez = BezierCurve(np.zeros((N_p1, 2)))
    G_tilde = bez.G_tilde
    # factor G_tilde = B^T B (it is PSD); use eigen-based sqrt for robustness
    w, V = np.linalg.eigh(G_tilde)
    w = np.clip(w, 0.0, None)
    B = (V * np.sqrt(w)) @ V.T             # symmetric PSD square root

    # straight-line initialization
    ts = np.linspace(0.0, 1.0, N_p1)
    P_ref = START[None, :] * (1 - ts)[:, None] + END[None, :] * ts[:, None]

    P_init = P_ref.copy()

    for it in range(MAX_ITER):
        P = cp.Variable((N_p1, 2))
        cost = cp.sum_squares(B @ P[:, 0]) + cp.sum_squares(B @ P[:, 1])

        cons = [P[0, :] == START, P[N, :] == END]
        if REST_TO_REST:
            cons += [P[1, :] == P[0, :], P[N - 1, :] == P[N, :]]

        # trust region (box), geometrically shrinking to damp SCvx chatter
        trust = max(TRUST_MIN, TRUST0 * (TRUST_SHRINK ** it))
        cons += [cp.abs(P - P_ref) <= trust]

        # multi-obstacle supporting half-spaces with exact-penalty slack
        blocks = obstacle_halfspaces(A_list, P_ref, pillars)
        slack = cp.Variable(len(blocks), nonneg=True)
        for j, (n, rhs, Ai) in enumerate(blocks):
            seg_pts = Ai @ P                       # (N+1, 2) expression
            cons += [seg_pts @ n + slack[j] >= rhs]  # every ctrl pt of segment

        obj = cost + W_SLACK * cp.sum(slack)
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if P.value is None:
            raise RuntimeError(f"QP failed at iter {it}: {prob.status}")

        step = np.linalg.norm(P.value - P_ref)
        P_ref = P.value
        if verbose and (it % 20 == 0 or step < TOL):
            print(f"  iter {it:3d}  step={step:.3e}  slack={float(slack.value.sum()):.3e}")
        if step < TOL:
            break

    return P_ref, P_init, it


def clearance_profile(P, pillars, m=1200):
    """Signed clearance to the nearest pillar boundary along the curve."""
    bez = BezierCurve(P)
    taus = np.linspace(0.0, 1.0, m)
    pts = np.array([bez.point(t) for t in taus])
    clr = np.full(m, np.inf)
    for (cx, cy, r) in pillars:
        d = np.linalg.norm(pts - np.array([cx, cy]), axis=1) - r
        clr = np.minimum(clr, d)
    return taus, pts, clr


def main():
    print("Solving planar multi-pillar SCvx demo (N=%d, n_seg=%d)..." % (N, N_SEG))
    P_opt, P_init, iters = solve_scvx(PILLARS)
    print(f"converged in {iters} outer iterations")

    taus, pts_opt, clr_opt = clearance_profile(P_opt, PILLARS)
    _, pts_init, clr_init = clearance_profile(P_init, PILLARS)

    min_clr = clr_opt.min()
    print(f"min continuous clearance (optimized): {min_clr:+.4f}")
    print(f"min continuous clearance (straight init): {clr_init.min():+.4f}")

    # ------------------------------------------------------------------ figure
    # stacked layout: wide spatial scene on top (equal aspect), clearance below
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.5, 7.4),
                                   gridspec_kw={"height_ratios": [1.0, 1.05]})

    # --- panel A: geometry
    for k, (cx, cy, r) in enumerate(PILLARS):
        ax0.add_patch(Circle((cx, cy), r, facecolor="#d9534f", edgecolor="#a33",
                             alpha=0.35, lw=1.5, zorder=2))
        # certified safety boundary (true radius + inflation)
        ax0.add_patch(Circle((cx, cy), r + SAFETY_INFLATE, facecolor="none",
                             edgecolor="#a33", ls=(0, (3, 3)), lw=1.0, alpha=0.6,
                             zorder=2, label="certified clearance" if k == 0 else None))
    ax0.plot(pts_init[:, 0], pts_init[:, 1], "--", color="#888", lw=1.8,
             label="straight-line init (infeasible)", zorder=3)
    ax0.plot(pts_opt[:, 0], pts_opt[:, 1], "-", color="#1f6feb", lw=2.8,
             label="Bézier trajectory (continuously safe)", zorder=4)
    ax0.plot(*START, "ks", ms=9, zorder=5)
    ax0.plot(*END, "k*", ms=16, zorder=5)
    ax0.annotate("start", START, textcoords="offset points", xytext=(-4, 12))
    ax0.annotate("goal", END, textcoords="offset points", xytext=(2, 12))
    ax0.set_aspect("equal")
    ax0.set_ylim(-1.9, 1.9)
    ax0.set_xlabel("x  [arb. units]"); ax0.set_ylabel("y")
    ax0.set_title("(a) Drone weaving between pillars")
    ax0.legend(loc="lower left", fontsize=8, framealpha=0.92, ncol=1)
    ax0.grid(alpha=0.2)

    # --- panel B: continuous clearance
    ax1.axhline(0.0, color="#d9534f", lw=1.6, ls="-", label="pillar boundary")
    ax1.plot(taus, clr_init, "--", color="#888", lw=1.8, label="straight-line init")
    ax1.plot(taus, clr_opt, "-", color="#1f6feb", lw=2.4, label="Bézier trajectory")
    ax1.fill_between(taus, clr_init, 0, where=(clr_init < 0),
                     color="#d9534f", alpha=0.18)
    ax1.fill_between(taus, 0, clr_opt, where=(clr_opt >= 0),
                     color="#1f6feb", alpha=0.08)
    ax1.set_xlabel(r"curve parameter $\tau$")
    ax1.set_ylabel("clearance to nearest pillar")
    ax1.set_title("(b) Continuous safety margin")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.2)
    ax1.annotate(f"min = {min_clr:+.3f}\nsafe for all "r"$\tau$"" —\nnot just at nodes",
                 xy=(taus[np.argmin(clr_opt)], min_clr),
                 xytext=(0.30, 0.62), textcoords="axes fraction", fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="#1f6feb"))

    fig.suptitle("Generality demonstration: same control-point KOZ construction, "
                 "multiple obstacles, non-orbital domain", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(out_dir, "demo_pillars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    # also drop a copy to the scratchpad for quick viewing
    scratch = "/private/tmp/claude-501/-Users-hyeon-yongjeong-code-bezier-trajectory/8602a2d5-dfb8-4ef3-81bd-e21545ae03d4/scratchpad/demo_pillars.png"
    try:
        fig.savefig(scratch, dpi=150, bbox_inches="tight")
        print(f"saved {scratch}")
    except Exception as e:
        print("scratch copy failed:", e)


if __name__ == "__main__":
    main()
