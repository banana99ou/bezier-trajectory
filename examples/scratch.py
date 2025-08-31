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
    Uses repeated split at tau = 1/k (k = n_seg, n_seg-1, ..., 2).
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
# ─────────────────────────────────────────────────────────────────────────────
def build_linear_constraint_matrix(A_list, n_list, dim, r_e):
    """
    Build LinearConstraint(lb<=A@x<=ub) for all segments and their control points.
    x = vec(P) with shape ((N+1)*dim,), row-major by control point.
    """
    Np1 = A_list[0].shape[1]
    rows, lbs = [], []
    for Ai, n in zip(A_list, n_list):
        for k in range(Np1):
            row = np.zeros(Np1 * dim)
            for j in range(Np1):
                coeff = Ai[k, j]  # scalar
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
    Fixed-point iteration with SLSQP inner solve.
    Endpoints P0, PN are fixed. Returns (optimized control points, info).
    """
    P = P_init.copy()
    Np1, dim = P.shape
    N = Np1 - 1
    A_list = segment_matrices_equal_params(N, n_seg)

    Hs = curvature_H(Np1, dim, lam=lam_smooth)
    x0 = P.reshape(-1)

    # Bounds: fix endpoints to start/goal; interior free
    lb = x0.copy(); ub = x0.copy()
    lb[:] = -np.inf; ub[:] =  np.inf
    lb[:dim] = ub[:dim] = x0[:dim]       # lock P0
    lb[-dim:] = ub[-dim:] = x0[-dim:]    # lock PN
    bounds = Bounds(lb, ub)

    normals_prev = None
    info = {"iter": 0, "feasible": False, "min_radius": None}

    def obj(x):
        dx = x - x0
        return 0.5 * dx @ dx + 0.5 * (x @ (Hs @ x))

    def grad(x):
        return (x - x0) + (Hs @ x)

    for it in range(1, max_iter+1):
        P_now = P
        normals = []
        eps = 1e-12
        for Ai in A_list:
            Qi = Ai @ P_now
            ci = Qi.mean(axis=0)
            n = ci / (np.linalg.norm(ci) + eps) if np.linalg.norm(ci) > 1e-9 else None
            if n is None and normals_prev is not None and keep_last_normal:
                n = normals_prev[len(normals)]
            if n is None:
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
            print(f"[iter {it}] Δ={delta:.3e}  success={res.success}  status={res.status}")

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
def add_wire_sphere(ax, radius=3.0, center=(0.0, 0.0, 0.0), color='gray', alpha=0.25, resolution=40):
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

def set_isometric(ax, elev=35.264, azim=45.0, ortho=True):
    ax.view_init(elev=elev, azim=azim)
    try:
        if ortho:
            ax.set_proj_type('ortho')  # mpl ≥3.2
    except Exception:
        pass

def beautify_3d_axes(ax, show_ticks=True, show_grid=True):
    """Paper-friendly 3D axes with ticks + optional grid."""
    ax.grid(show_grid)
    if show_ticks:
        ax.tick_params(axis='both', which='major', labelsize=8, pad=2)
    else:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # pane styling
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1, 1, 1, 1))
            axis.pane.set_edgecolor("0.85")
        except Exception:
            pass

def plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', show_seg_ctrl_if=8, lw=2.0):
    """Plot whole-curve control polygon + each segment with a smooth color gradient."""
    curve = BezierCurve(P_opt)
    N = curve.degree
    A_list = segment_matrices_equal_params(N, n_seg)

    # Whole-curve control polygon (black)
    ax.plot(P_opt[:,0], P_opt[:,1], P_opt[:,2], 'k.-', lw=1.2, ms=4)

    # Prepare colors
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(0.0)] if len(A_list) == 1 else [cmap(i/(len(A_list)-1)) for i in range(len(A_list))]

    # Draw segments
    ts = np.linspace(0, 1, 180)
    lines, cols = [], []
    for i, Ai in enumerate(A_list):
        Qi = Ai @ P_opt
        seg = BezierCurve(Qi)
        Pseg = np.array([seg.point(t) for t in ts])
        lines.append(np.column_stack((Pseg[:,0], Pseg[:,1], Pseg[:,2])))
        cols.append(colors[i])

        # Only show segment control polygons when n_seg is small
        if n_seg <= show_seg_ctrl_if:
            ax.plot(Qi[:,0], Qi[:,1], Qi[:,2], '-', color=colors[i], alpha=0.55, lw=1.0, marker='o', ms=3)

    lc = Line3DCollection(lines, colors=cols, linewidths=lw, alpha=0.95)
    ax.add_collection3d(lc)

def plot_initial_guess(ax, P_init, linestyle=':', color='0.5', lw=1.6):
    """Grey dotted initial curve overlay."""
    init_curve = BezierCurve(P_init)
    _, Ppts = init_curve.sample(300)
    ax.plot(Ppts[:,0], Ppts[:,1], Ppts[:,2], linestyle=linestyle, color=color, lw=lw)

# ─────────────────────────────────────────────────────────────────────────────
# Demo / experiment (4 panels, ticks + grid, initial overlay on top-left)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    r_e = 6.0  # sphere radius

    # Quadratic Bézier in 3D (degree 2) connecting arbitrary start/goal
    P0 = np.array([ 4.0, -7.0,  0.0])
    P2 = np.array([ -5.0,  7.0,  3.0])
    # Middle CP slightly away from origin (your fix)
    P1 = np.array([  -1.0,   -1.0,  2.0])  # tweak as needed
    P_init = np.vstack([P0, P1, P2])  # (3,3) degree=2

    # Choose exactly 4 cases for the figure (top-left is 2 segs)
    split_list = [2, 4, 8, 16, 32, 64]
    results = []
    for n_seg in split_list:
        P_opt, info = optimize_bezier_outside_sphere(
            P_init, n_seg=n_seg, r_e=r_e, max_iter=120, tol=1e-8,
            lam_smooth=1e-6, verbose=False
        )
        results.append((n_seg, P_opt, info))
        print(f"n_seg={n_seg:>2d} | iter={info['iter']:>3d} | min_radius={info['min_radius']:.3f} | feasible={info['feasible']}")

    # Figure: 2×2 panels with ticks + grid; top-left overlays initial guess
    fig, axes = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(7.0, 6.4))
    axes = axes.ravel()

    for i, (n_seg, P_opt, info) in enumerate(results):
        ax = axes[i]
        add_wire_sphere(ax, radius=r_e, color='#f38d0e', alpha=0.22, resolution=28)
        plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', show_seg_ctrl_if=8, lw=2.1)

        # Overlay initial guess ONLY on top-left (i==0) with 2 segs
        if i == 0 and n_seg == 2:
            plot_initial_guess(ax, P_init, linestyle=':', color='0.5', lw=1.8)

        set_axes_equal_around(ax, center=(0,0,0), radius=r_e*1.25, pad=0.04)
        set_isometric(ax, elev=35.264, azim=45.0, ortho=True)
        beautify_3d_axes(ax, show_ticks=True, show_grid=True)

        ax.set_title(f"{n_seg} segs   |   min |p| = {info['min_radius']:.2f}", fontsize=9, pad=3)
        ax.set_xlabel('X', labelpad=4, fontsize=9)
        ax.set_ylabel('Y', labelpad=4, fontsize=9)
        ax.set_zlabel('Z', labelpad=4, fontsize=9)

    # Tight-ish layout for papers
    fig.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.04, wspace=0.02, hspace=0.02)

    # Save if you want camera-ready output
    # plt.savefig("bezier_outside_sphere_2x2.png", dpi=450, bbox_inches="tight", pad_inches=0.02)

    plt.show()
