import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
# (No change to your math / class)

class BezierCurve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points, dtype=float)
        self.degree = self.control_points.shape[0] - 1
        self.dimension = self.control_points.shape[1]

    def Get_Bezier_Point(self, tau):
        N = self.degree
        result = np.zeros(self.dimension)
        for i in range(N + 1):
            Berstein = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            result += Berstein * self.control_points[i]
        return result
    
    def De_Casteljau(self, tau):
        length = len(self.control_points)
        new_control_points = np.empty((length, self.dimension), dtype=float)
        new_control_points[0] = self.control_points[0]

        N = 2
        for i in range(N + 1):
            Berstein = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            result += Berstein * self.control_points[i]
        new_control_points[i] = 

        new_control_points[length-1] = self.Get_Bezier_Point(tau)

        return new_control_points[self.degree]

# ---- Your control points (2D or 3D both work) ----
control_points = np.array([[0,0,0],
                           [2,1,4],
                           [4,-1,1],
                           [6,3,2],
                           [5,8,-3]])
curve = BezierCurve(control_points)

# ---- 3D plotting (only plt-related changes) ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw curve as red dots
for tau in np.linspace(0, 1, 100):
    p = curve.Get_Bezier_Point(tau)
    if len(p) == 2:
        x, y = p
        z = 0.0
    else:
        x, y, z = p
    ax.plot([x], [y], [z], 'r.')

# Draw control polygon in 3D (z=0 if 2D input)
xs = control_points[:, 0]
ys = control_points[:, 1]
zs = np.zeros(len(control_points)) if control_points.shape[1] == 2 else control_points[:, 2]
ax.plot(xs, ys, zs, 'go--', label='control polygon')

x, y, z = curve.De_Casteljau(0.25)
ax.plot(x, y, z, 'bo')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Make axes roughly equal scale in 3D
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range]) / 2.0
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])

set_axes_equal(ax)
ax.legend()
plt.show()
