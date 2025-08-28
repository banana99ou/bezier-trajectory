import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from typing import Tuple, List
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class BezierCurve:
    def __init__(self, control_points: np.ndarray):
        """
        control_points: (n_ctrl, dim)
        """
        self.control_points = np.array(control_points, dtype=float)
        self.degree = self.control_points.shape[0] - 1
        self.dimension = self.control_points.shape[1]

    def Get_Bezier_Point(self, tau: float) -> np.ndarray:
        N = self.degree
        result = np.zeros(self.dimension, dtype=float)
        for i in range(N + 1):
            Berstein = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            result += Berstein * self.control_points[i]
        return result

    def De_Casteljau(self, control_points: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a control polygon at parameter tau into left/right control polygons.
        Returns (L, R), both shape (n_ctrl, dim).
        """
        P = np.asarray(control_points, dtype=float)
        if P.ndim != 2:
            raise ValueError("control_points must be 2D array (n_ctrl, dim)")
        if not (0.0 <= tau <= 1.0):
            raise ValueError("tau must be in [0, 1]")

        left = [P[0]]
        right = [P[-1]]

        W = P.copy()
        for _ in range(1, len(P)):
            W = (1.0 - tau) * W[:-1] + tau * W[1:]
            left.append(W[0])
            right.append(W[-1])

        L = np.vstack(left)           # left boundary
        R = np.vstack(right[::-1])    # right boundary (reversed)
        return L, R

    def Split_to_N_Seg(self, N: int) -> List["BezierCurve"]:
        """
        Split the curve into N equal-parameter sub-curves using repeated
        De Casteljau splits at tau = 1/k (k = N, N-1, ..., 2).
        Returns a list of N BezierCurve instances.
        """
        if N <= 0:
            raise ValueError("N must be a positive integer")
        if N == 1:
            return [BezierCurve(self.control_points.copy())]

        segments: List[BezierCurve] = []
        remainder = self.control_points.copy()
        for k in range(N, 1, -1):
            tau = 1.0 / k
            L, remainder = self.De_Casteljau(remainder, tau)
            segments.append(BezierCurve(L))
        segments.append(BezierCurve(remainder))
        return segments

exclusion_zone = 6

# ---- Control points & subdivision ----
control_points = np.array([[-10, -3, 0],
                           [-7, 6, 0],
                           [ 9,-6, 0],
                           [ 10, 3, 0]], dtype=float)
curve = BezierCurve(control_points)
segs = curve.Split_to_N_Seg(10)

# ---- 3D plotting ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Control polygon
xs = control_points[:, 0]
ys = control_points[:, 1]
zs = np.zeros(len(control_points)) if control_points.shape[1] == 2 else control_points[:, 2]
ax.plot(xs, ys, zs, 'bo--', label='control polygon')

# Segments as one Line3DCollection (fast)
ts = np.linspace(0.0, 1.0, 200)
seg_lines = []
seg_cols = []
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for idx, seg_curve in enumerate(segs):
    P = np.array([seg_curve.Get_Bezier_Point(t) for t in ts])  # (M, dim)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    seg_lines.append(np.column_stack((P[:, 0], P[:, 1], P[:, 2])))
    seg_cols.append(colors[idx % len(colors)])

lc = Line3DCollection(seg_lines, colors=seg_cols, linewidths=1.5)
ax.add_collection3d(lc)

for idx, seg_curve in enumerate(segs):
    xs = seg_curve.control_points[:, 0]
    ys = seg_curve.control_points[:, 1]
    zs = np.zeros(len(seg_curve.control_points)) if seg_curve.control_points.shape[1] == 2 else seg_curve.control_points[:, 2]
    ax.plot(xs, ys, zs, f'{colors[idx % len(colors)]}o--')

# Optional: wireframe sphere (Earth surrogate) at origin
def add_wire_sphere(ax, radius=3.0, center=(0.0, 0.0, 0.0), color='gray', alpha=0.3, resolution=40):
    cx, cy, cz = center
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    uu, vv = np.meshgrid(u, v)
    x = cx + radius * np.cos(uu) * np.sin(vv)
    y = cy + radius * np.sin(uu) * np.sin(vv)
    z = cz + radius * np.cos(vv)
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, alpha=alpha)

add_wire_sphere(ax, radius=exclusion_zone, color='gray', alpha=0.3)

# Equal aspect & labels
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

set_axes_equal_around(ax, center=(0,0,0), radius=3.0)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.show()
