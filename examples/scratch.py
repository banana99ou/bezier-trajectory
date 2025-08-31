import numpy as np
from scipy.special import comb
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ─────────────────────────────────────────────────────────────────────────────
# Bézier curve core
# ─────────────────────────────────────────────────────────────────────────────
class BezierCurve:
    def __init__(self, control_points: np.ndarray):
        """control_points: shape (N+1, dim). Degree = N."""
        P = np.array(control_points, dtype=float)
        if P.ndim != 2:
            raise ValueError("control_points must be (N+1, dim)")
        self.control_points = P
        self.degree = P.shape[0] - 1
        self.dimension = P.shape[1]

    def point(self, tau: float) -> np.ndarray:
        """Evaluate Bézier point p(tau)."""
        N, d = self.degree, self.dimension
        out = np.zeros(d, dtype=float)
        for i in range(N + 1):
            b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            out += b * self.control_points[i]
        return out

    def sample(self, num=200):
        ts = np.linspace(0.0, 1.0, num)
        P = np.array([self.point(t) for t in ts])
        return ts, P

# ─────────────────────────────────────────────────────────────────────────────
# De Casteljau split as matrices  (linear maps on original control points)
# ─────────────────────────────────────────────────────────────────────────────
def de_casteljau_split_1d(N, tau, basis_index):
    """Split the 1D control vector e_j at tau → return (left, right) coeffs."""
    # Start from scalar control points that are one-hot at basis_index
    w = np.zeros(N+1, dtype=float); w[basis_index] = 1.0
    left = [w[0]]
    right = [w[-1]]
    W = w.copy()
    for _ in range(1, N+1):
        W = (1 - tau) * W[:-1] + tau * W[1:]
        left.append(W[0])
        right.append(W[-1])
    L = np.array(left)      # (N+1,)
    R = np.array(right[::-1])
    return L, R

def de_casteljau_split_matrices(N, tau):
    """Return (S_left, S_right) s.t. L = S_left @ P, R = S_right @ P."""
    S_left  = np.zeros((N+1, N+1), dtype=float)
    S_right = np.zeros((N+1, N+1), dtype=float)
    for j in range(N+1):
        L, R = de_casteljau_split_1d(N, tau, j)
        S_left[:,  j] = L
        S_right[:, j] = R
    return S_left, S_right

def segment_matrices_equal_params(N, n_seg):
    """
    Return list of Ai matrices, one per segment, so that
    segment i control points Qi = Ai @ P (P is (N+1, dim)).
    Uses standard repeated split at tau = 1/k (k = n_seg, n_seg-1, ..., 2).
    """
    if n_seg < 1:
        raise ValueError("n_seg must be >= 1")
    if n_seg == 1:
        return [np.eye(N+1)]
    mats = []
    remainder = np.eye(N+1)
    for k in range(n_seg, 1, -1):
        tau = 1.0 / k
        S_L, S_R = de_casteljau_split_matrices(N, tau)
        mats.append(S_L @ remainder)
        remainder = S_R @ remainder
    mats.append(remainder)  # last segment
    return mats  # list of (N+1,N+1)

# ─────────────────────────────────────────────────────────────────────────────
# Half-space linearization: n_i^T (A_i @ P)_k >= r_e
# Iterative LP/QP on original control points with endpoints fixed
# ─────────────────────────────────────────────────────────────────────────────
def build_linear_constraint_matrix(A_list, n_list, dim, r_e):
    """
    Build LinearConstraint(lb<=A@x<=ub) for all segments and their control points.
    x = vec(P) with shape ((N+1)*dim,), row-major by control point.
    """
    Np1 = A_list[0].shape[1]
    rows = []
    lbs  = []
    for Ai, n in zip(A_list, n_list):
        # For each control point of this segment
        for k in range(Np1):
            row = np.zeros(Np1 * dim)
            # Contribution from every original control point j
            for j in range(Np1):
                coeff = Ai[k, j]  # scalar
                # Insert coeff * n into x-block for P_j
                start = j * dim
                row[start:start+dim] += coeff * n
            rows.append(row)
            lbs.append(r_e)
    A = np.vstack(rows)
    lb = np.array(lbs)
    ub = np.full_like(lb, np.inf, dtype=float)
    return LinearConstraint(A, lb, ub)

def curvature_H(Np1, dim, lam=0.0):
    """Return H = 2*lam*(I_dim ⊗ D^T D) for second-difference smoothing."""
    if lam <= 0: 
        return np.zeros((Np1*dim, Np1*dim))
    D = np.zeros((Np1-2, Np1))
    for r in range(Np1-2):
        D[r, r:r+3] = [1, -2, 1]
    DtD = D.T @ D
    H = np.kron(np.eye(dim), 2*lam*DtD)
    return H

def optimize_bezier_outside_sphere(P_init, n_seg=8, r_e=6.0, 
                                   max_iter=20, tol=1e-6, lam_smooth=0.0, 
                                   verbose=False, keep_last_normal=True):
    """
    Fixed-point iteration:
      1) from current P, compute segment centroids and normals
      2) solve QP (SLSQP) with linear constraints n_i^T (A_i P)_k >= r_e
      3) repeat until convergence
    Endpoints P0, PN are fixed.
    Returns optimized control points and diagnostic dict.
    """
    P = P_init.copy()
    Np1, dim = P.shape
    N = Np1 - 1
    A_list = segment_matrices_equal_params(N, n_seg)

    Hs = curvature_H(Np1, dim, lam=lam_smooth)
    x0 = P.reshape(-1)

    # Bounds: fix endpoints to start/goal; interior free
    lb = x0.copy(); ub = x0.copy()
    # unlock all first
    lb[:] = -np.inf; ub[:] =  np.inf
    # lock P0 and PN
    lb[:dim] = ub[:dim] = x0[:dim]
    lb[-dim:] = ub[-dim:] = x0[-dim:]
    bounds = Bounds(lb, ub)

    # Cache normals across iterations if needed
    normals_prev = None
    info = {"iter": 0, "feasible": False, "min_radius": None}

    def obj(x):
        # 0.5||x - x0||^2 + 0.5 x^T H x  (H comes from curvature)
        dx = x - x0
        return 0.5 * dx @ dx + 0.5 * (x @ (Hs @ x))

    def grad(x):
        return (x - x0) + (Hs @ x)

    for it in range(1, max_iter+1):
        # Build normals from current P
        P_now = P
        normals = []
        eps = 1e-12
        for Ai in A_list:
            Qi = Ai @ P_now  # (N+1, dim), control points of this segment
            ci = Qi.mean(axis=0)  # centroid   (see FMCL note) 
            n = ci / (np.linalg.norm(ci) + eps) if np.linalg.norm(ci) > 1e-9 else None
            if n is None and normals_prev is not None and keep_last_normal:
                n = normals_prev[len(normals)]
            if n is None:
                # fall back to outward normal from start->goal plane
                v = P_now[-1] - P_now[0]
                n = v / (np.linalg.norm(v) + eps)
            normals.append(n)

        lin_con = build_linear_constraint_matrix(A_list, normals, dim, r_e)

        res = minimize(obj, P_now.reshape(-1), method="SLSQP",
                       jac=grad, constraints=[lin_con], bounds=bounds,
                       options=dict(maxiter=200, ftol=1e-12, disp=False))
        x_new = res.x
        P_new = x_new.reshape(Np1, dim)

        delta = np.linalg.norm(P_new - P)
        P = P_new
        normals_prev = normals
        if verbose:
            print(f"[iter {it}] delta={delta:.3e}, success={res.success}, status={res.status}")

        if delta < tol:
            break

    # Final feasibility check by dense sampling
    curve = BezierCurve(P)
    ts = np.linspace(0, 1, 1000)
    pts = np.array([curve.point(t) for t in ts])
    radii = np.linalg.norm(pts, axis=1)
    min_r = float(np.min(radii))
    info.update({"iter": it, "feasible": bool(min_r >= r_e - 1e-6), "min_radius": min_r})
    return P, info

# ─────────────────────────────────────────────────────────────────────────────
# Utility: plotting & checking
# ─────────────────────────────────────────────────────────────────────────────
def add_wire_sphere(ax, radius=3.0, center=(0.0, 0.0, 0.0), color='gray', alpha=0.3, resolution=40):
    cx, cy, cz = center
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    uu, vv = np.meshgrid(u, v)
    x = cx + radius * np.cos(uu) * np.sin(vv)
    y = cy + radius * np.sin(uu) * np.sin(vv)
    z = cz + radius * np.cos(vv)
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, alpha=alpha)

def set_axes_equal_around(ax, center=(0,0,0), radius=1.0, pad=0.05):
    cx, cy, cz = center
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    x0 = min(x0, cx - radius); x1 = max(x1, cx + radius)
    y0 = min(y0, cy - radius); y1 = max(y1, cy + radius)
    z0 = min(z0, cz - radius); z1 = max(z1, cz + radius)
    max_range = max(x1 - x0, y1 - y0, z1 - z0)
    half = 0.5 * max_range * (1 + pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect((1, 1, 1))

def plot_segments(ax, P_opt, n_seg, color_cycle=('r','g','b','c','m','y','k')):
    curve = BezierCurve(P_opt)
    N = curve.degree
    A_list = segment_matrices_equal_params(N, n_seg)

    # draw control polygon of the whole curve
    ax.plot(P_opt[:,0], P_opt[:,1], P_opt[:,2], 'ko--', lw=1.5, label=f'control (N={N})')

    # draw segments with distinct colors
    ts = np.linspace(0, 1, 200)
    lines = []
    cols  = []
    for i, Ai in enumerate(A_list):
        Qi = Ai @ P_opt
        seg = BezierCurve(Qi)
        Pseg = np.array([seg.point(t) for t in ts])
        lines.append(np.column_stack((Pseg[:,0], Pseg[:,1], Pseg[:,2])))
        cols.append(color_cycle[i % len(color_cycle)])
        # also plot segment control polygons
        ax.plot(Qi[:,0], Qi[:,1], Qi[:,2], f'{cols[-1]}o--', alpha=0.6)
    lc = Line3DCollection(lines, colors=cols, linewidths=2.0)
    ax.add_collection3d(lc)

# ─────────────────────────────────────────────────────────────────────────────
# Demo / experiment per your PI’s checklist
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    exclusion_zone = 6.0  # sphere radius r_e

    # ① Initialize quadratic Bézier in 3D (degree 2) connecting arbitrary start/goal
    P0 = np.array([ 10.0, -10.0,  0.0])
    P2 = np.array([ -10.0,  10.0,  0.0])
    # Choose a middle control point INSIDE the sphere to start with (violates constraint)
    P1 = np.array([  0.0,  0.0,  1.0])
    P_init = np.vstack([P0, P1, P2])  # (3,3) degree=2

    # ② Optimize for different subdivision counts to show feasibility emerges
    split_list = [2, 4, 8, 16, 32, 64]
    results = []

    for n_seg in split_list:
        P_opt, info = optimize_bezier_outside_sphere(P_init, n_seg=n_seg, r_e=exclusion_zone, max_iter=100, tol=1e-8, lam_smooth=1e-6, verbose=False)
        results.append((n_seg, P_opt, info))
        print(f"n_seg={n_seg:>2d} | iter={info['iter']:>2d} | min_radius={info['min_radius']:.3f} | feasible={info['feasible']}")

    # ─────────────────────────────────────────────────────────────────────────────
# Pretty paper figure: 3 columns + isometric (orthographic) view
# ─────────────────────────────────────────────────────────────────────────────
def set_isometric(ax, elev=35.264, azim=45.0, ortho=True):
    ax.view_init(elev=elev, azim=azim)
    try:
        if ortho:
            ax.set_proj_type('ortho')  # mpl ≥3.2
    except Exception:
        pass

def beautify_3d_axes(ax, ticks=False):
    """Clean, paper-friendly 3D axes. Safe across Matplotlib versions."""
    # turn off grid
    ax.grid(False)

    # Pane face/edge colors (newer mpl: axis.pane; older: set_pane_color)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        # Face
        try:
            axis.pane.set_facecolor((1, 1, 1, 1))
        except Exception:
            try:
                axis.set_pane_color((1, 1, 1, 1))
            except Exception:
                pass
        # Edge
        try:
            axis.pane.set_edgecolor("0.85")
        except Exception:
            # No simple fallback on very old versions; safe to ignore.
            pass

    # Optional ticks
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    else:
        ax.tick_params(axis='both', which='major', labelsize=8, pad=2)

    # Hide box spines if available (not always on 3D)
    try:
        for sp in ax.spines.values():
            sp.set_visible(False)
    except Exception:
        pass

# ③ Visualization & comparison (3-column isometric, tight layout)
COLS = 3
rows = int(np.ceil(len(results) / COLS))

# ~2.2–2.4 in per panel: journal-friendly
COL_W = 2.3
ROW_H = 2.3
fig, axes = plt.subplots(
    rows, COLS, subplot_kw={'projection': '3d'},
    figsize=(COLS * COL_W, rows * ROW_H)
)

# Disable any auto layout engines that trigger 3D tightbbox calls
for _setter in ("set_layout_engine", "set_constrained_layout"):
    try:
        getattr(fig, _setter)(False if _setter == "set_constrained_layout" else None)
    except Exception:
        pass

# Manual compact spacing (safe with 3D)
fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04, wspace=0.02, hspace=0.02)

axes = np.atleast_1d(axes).ravel()

for i, (n_seg, P_opt, info) in enumerate(results):
    ax = axes[i]
    add_wire_sphere(ax, radius=exclusion_zone, color='0.7', alpha=0.22, resolution=28)
    plot_segments(ax, P_opt, n_seg)
    set_axes_equal_around(ax, center=(0,0,0), radius=exclusion_zone*1.25, pad=0.02)
    set_isometric(ax, elev=35.264, azim=45.0, ortho=True)
    beautify_3d_axes(ax, ticks=False)
    ax.set_title(f"{n_seg} segs  |  min |p|={info['min_radius']:.2f}", fontsize=8, pad=2)
    ax.set_xlabel('X', labelpad=1, fontsize=7)
    ax.set_ylabel('Y', labelpad=1, fontsize=7)
    ax.set_zlabel('Z', labelpad=1, fontsize=7)

# Hide any unused panels
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Save a crisp figure for the paper (no layout engine = stable)
# plt.savefig("bezier_segments_isometric.png", dpi=400, bbox_inches="tight", pad_inches=0.02)

plt.show()
