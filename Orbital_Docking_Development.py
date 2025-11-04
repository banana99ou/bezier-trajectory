# Imports and Constants
import numpy as np
from scipy.special import comb
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Orbital parameters (from Project_Spec.md)
ISS_ALTITUDE_KM = 423  # ISS orbit altitude in km AMSL
CHASER_ALTITUDE_KM = 300  # Chaser altitude in km AMSL
KOZ_ALTITUDE_KM = 100  # Keep Out Zone altitude in km AMSL

EARTH_RADIUS_KM = 6371.0  # Earth radius in km
KOZ_RADIUS = EARTH_RADIUS_KM + KOZ_ALTITUDE_KM  # KOZ radius from Earth center
ISS_RADIUS = EARTH_RADIUS_KM + ISS_ALTITUDE_KM
CHASER_RADIUS = EARTH_RADIUS_KM + CHASER_ALTITUDE_KM

# Gravitational parameters
EARTH_MU = 3.986004418e14  # m³/s²

# Scaling: use km as base unit
SCALE_FACTOR = 1e3  # 1 unit = 1 km
EARTH_MU_SCALED = EARTH_MU / (SCALE_FACTOR**3)  # Scaled for km

def get_D_matrix(N):
    """
    Compute derivative matrix D for Bézier curve of degree N.
    
    From Project_Spec.md:
    [D]_i,j = N × { -1 if j=i, 1 if j=i+1, 0 otherwise }
    
    Args:
        N: Degree of Bézier curve
    
    Returns:
        D: (N, N+1) matrix
    """
    D = np.zeros((N, N+1))
    for i in range(N):
        D[i, i] = -N
        D[i, i+1] = N
    return D

def get_E_matrix(N):
    """
    Compute elevation matrix E for Bézier curve of degree N.
    Elevates degree from N to N+1.
    
    From Project_Spec.md equation:
    E_{N→N+1} with specific structure
    
    Args:
        N: Original degree
    
    Returns:
        E: (N+2, N+1) matrix
    """
    E = np.zeros((N+2, N+1))
    
    # First row: i=1, j=1 → 1
    E[0, 0] = 1.0
    
    # Last row: i=N+2, j=N+1 → 1
    E[N+1, N] = 1.0
    
    # Middle rows: 2 ≤ i ≤ N+1 (1-indexed) → rows 1 to N (0-indexed)
    for row_idx in range(1, N+1):  # 0-indexed row: 1 to N
        i_1idx = row_idx + 1  # Convert to 1-indexed for formula
        # j = i-1 (1-indexed) → col = row_idx - 1 (0-indexed)
        E[row_idx, row_idx - 1] = (N + 2 - i_1idx) / (N + 1)
        # j = i (1-indexed) → col = row_idx (0-indexed)
        E[row_idx, row_idx] = (i_1idx - 1) / (N + 1)
    
    return E

# # Test matrices for degree 2 (quadratic) and degree 3
# print("="*60)
# print("Testing D and E matrices")
# print("="*60)

# # Test D matrix for N=2 (quadratic: 3 control points → 2 control points)
# print("\n1. D matrix for N=2 (degree 2, 3→2 control points):")
# D2 = get_D_matrix(2)
# print(f"D2 (2×3):\n{D2}")
# print("Expected pattern: [-N, N, 0] on each row")
# print(f"Values: -2 and 2 (correct for N=2)")

# # Test D matrix for N=3 (cubic: 4 control points → 3 control points)
# print("\n2. D matrix for N=3 (degree 3, 4→3 control points):")
# D3 = get_D_matrix(3)
# print(f"D3 (3×4):\n{D3}")
# print("Expected from spec D3→2 (unscaled pattern):")
# print("[[-1, 1, 0, 0],")
# print(" [0, -1, 1, 0],")
# print(" [0, 0, -1, 1]]")
# print(f"Scaled by N=3: multiply by 3")
# D3_expected = np.array([[-3, 3, 0, 0], [0, -3, 3, 0], [0, 0, -3, 3]])
# print(f"Match: {np.allclose(D3, D3_expected)}")

# # Test E matrix for N=2 (elevates from degree 2 to 3)
# print("\n3. E matrix for N=2 (elevates degree 2→3, 3→4 control points):")
# E2 = get_E_matrix(2)
# print(f"E2 (4×3):\n{E2}")
# print("\nExpected E2→3 from spec:")
# E2_expected = np.array([[1, 0, 0], [2/3, 1/3, 0], [0, 1/3, 2/3], [0, 0, 1]])
# print(E2_expected)
# print(f"\n✓ Match: {np.allclose(E2, E2_expected)}")

# # Test E matrix for N=3 (elevates from degree 3 to 4)
# print("\n4. E matrix for N=3 (elevates degree 3→4, 4→5 control points):")
# E3 = get_E_matrix(3)
# print(f"E3 (5×4):\n{E3}")
# print("\nExpected pattern for N=3:")
# print("Row 0: [1,   0,   0,   0]")
# print("Row 1: [3/4, 1/4, 0,   0]")
# print("Row 2: [0,   2/4, 2/4, 0]")
# print("Row 3: [0,   0,   1/4, 3/4]")
# print("Row 4: [0,   0,   0,   1]")
# E3_expected = np.array([
#     [1, 0, 0, 0],
#     [3/4, 1/4, 0, 0],
#     [0, 2/4, 2/4, 0],
#     [0, 0, 1/4, 3/4],
#     [0, 0, 0, 1]
# ])
# print(f"✓ Match: {np.allclose(E3, E3_expected)}")

class BezierCurve:
    """
    Bézier curve implementation using D/E matrices for derivatives.
    """
    
    def __init__(self, control_points):
        P = np.array(control_points, dtype=float)
        if P.ndim != 2: # cus np.array makes control_points a 2D matrix
            raise ValueError("control_points must be (N+1, dim)")
        self.control_points = P
        self.degree = P.shape[0] - 1 # = N
        self.dimension = P.shape[1] # P.shape = (N+1, dim)
        
        # Precompute D and E matrices
        self.D = get_D_matrix(self.degree)
        if self.degree > 0:
            self.E = get_E_matrix(self.degree - 1)  # E elevates from N-1 to N
    
    def point(self, tau):
        """Evaluate curve at parameter tau using Bernstein basis."""
        N, d = self.degree, self.dimension
        out = np.zeros(d)
        for i in range(N + 1):
            b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            out += b * self.control_points[i]
        return out
    
    def velocity_control_points(self):
        """
        Compute velocity control points using V = EDP.
        Returns control points for velocity curve (degree N, N+1 control points).
        """
        if self.degree == 0:
            return np.zeros((1, self.dimension))
        
        # V = E @ D @ P
        # P is (N+1, dim), D is (N, N+1), E is (N+1, N)
        # Result: (N+1, dim)
        V_ctrl = self.E @ self.D @ self.control_points
        return V_ctrl
    
    def acceleration_control_points(self):
        """
        Compute acceleration control points using A = EDEDP.
        Returns control points for acceleration curve.
        """
        # Acceleration control points using EDEDP as instructed.
        if self.degree < 2:
            return np.zeros(((self.degree+1), self.dimension))
        # Use EDEDP directly per the analytic/formal instructions.
        A_ctrl = self.E @ self.D @ self.E @ self.D @ self.control_points
        # (Resulting shape: (N+1, d);
        return A_ctrl
    
    def velocity(self, tau):
        """Evaluate velocity at parameter tau."""
        if self.degree == 0:
            return np.zeros(self.dimension)
        V_ctrl = self.velocity_control_points()
        # Velocity curve has degree N (same as original)
        N = self.degree
        out = np.zeros(self.dimension)
        for i in range(N + 1):
            b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            out += b * V_ctrl[i]
        return out

    def acceleration(self, tau):
        """Evaluate acceleration at parameter tau."""
        if self.degree < 2:
            return np.zeros(((self.degree+1), self.dimension))
        A_ctrl = self.acceleration_control_points()
        # Acceleration curve has degree N due to the E matrix
        N_accel = self.degree
        out = np.zeros(self.dimension)
        for i in range(N_accel + 1):
            b = comb(N_accel, i) * (tau ** i) * ((1 - tau) ** (N_accel - i))
            out += b * A_ctrl[i]
        return out

# # Test
# P_test = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]])  # Quadratic
# curve = BezierCurve(P_test)
# print(f"Test curve degree: {curve.degree}")
# print(f"Velocity at τ=0: {curve.velocity(0)}")
# print(f"Acceleration at τ=0: {curve.acceleration(0)}")
def de_casteljau_split_1d(N, tau, basis_index):
    """
    Compute De Casteljau subdivision coefficients for a single basis vector.
    """
    w = np.zeros(N+1)
    w[basis_index] = 1.0
    left = [w[0]]
    right = [w[-1]]
    W = w.copy()
    
    for _ in range(1, N+1):
        W = (1 - tau) * W[:-1] + tau * W[1:]
        left.append(W[0])
        right.append(W[-1])
    
    L = np.array(left)
    R = np.array(right[::-1])
    return L, R

def de_casteljau_split_matrices(N, tau):
    """Compute subdivision matrices S_left and S_right."""
    S_left = np.zeros((N+1, N+1))
    S_right = np.zeros((N+1, N+1))
    
    for j in range(N+1):
        L, R = de_casteljau_split_1d(N, tau, j)
        S_left[:, j] = L
        S_right[:, j] = R
    return S_left, S_right

def segment_matrices_equal_params(N, n_seg):
    """
    Generate segment matrices for equal-parameter splitting.
    Returns list of (N+1, N+1) matrices, one per segment.
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
    mats.append(remainder)
    return mats

# # Test
# print("Testing segmentation for N=2, n_seg=4:")
# mats = segment_matrices_equal_params(2, 4)
# print(f"Number of segments: {len(mats)}")
# for i, A in enumerate(mats):
#     print(f"Segment {i+1} matrix shape: {A.shape}")
def build_koz_constraints(A_list, P, r_e, dim=3, c_KOZ=None):
    """
    Build KOZ (Keep Out Zone) linear constraints for all segments.

    For each segment j:
    1. Compute CG (centroid) of control points: Qi = Ai @ P
    2. Generate unit vector nj from KOZ center (c_KOZ) to CG
    3. Create half-space constraint: nj^T @ Qi >= r_e

    Args:
        A_list: List of segment transformation matrices
        P: Control points (N+1, dim)
        r_e: KOZ radius
        dim: Spatial dimension
        c_KOZ: KOZ center, shape (dim,) or None (if None, uses origin and warns)

    Returns:
        LinearConstraint object
    """
    Np1 = A_list[0].shape[1]
    rows, lbs = [], []

    if c_KOZ is None:
        import warnings
        warnings.warn("c_KOZ (KOZ center) not specified; defaulting to origin.")
        c_KOZ = np.zeros(dim)

    for Ai in A_list:
        Qi = Ai @ P  # Control points of segment i
        ci = Qi.mean(axis=0)  # Centroid (CG)

        # Unit vector from KOZ center to CG
        Nj = ci - c_KOZ
        Nj_norm = np.linalg.norm(Nj)
        if Nj_norm < 1e-12:
            # If CG coincides with KOZ center, skip this segment (degenerate case)
            continue

        nj = Nj / Nj_norm  # Unit normal vector
        
        # Create constraint for each control point in segment
        for k in range(Np1):
            row = np.zeros(Np1 * dim)
            for j in range(Np1):
                coeff = Ai[k, j]
                start = j * dim
                row[start:start+dim] += coeff * nj
            rows.append(row)
            lbs.append(r_e)
    
    if len(rows) == 0:
        # No constraints generated
        A_const = np.zeros((1, Np1 * dim))
        lb_const = np.array([-np.inf])
        ub_const = np.array([np.inf])
    else:
        A_const = np.vstack(rows)
        lb_const = np.array(lbs)
        ub_const = np.full_like(lb_const, np.inf)
    
    return LinearConstraint(A_const, lb_const, ub_const)

# print("KOZ constraint building function ready.")

def build_boundary_constraints(P_init, v0=None, v1=None, a0=None, a1=None, dim=3):
    """
    Build boundary condition equality constraints.
    
    Args:
        P_init: Initial control points (N+1, dim)
        v0: Initial velocity (optional, shape (dim,))
        v1: Final velocity (optional, shape (dim,))
        a0: Initial acceleration (optional, shape (dim,))
        a1: Final acceleration (optional, shape (dim,))
        dim: Spatial dimension
    
    Returns:
        list of LinearConstraint objects
    """
    Np1 = P_init.shape[0]
    N = Np1 - 1
    
    constraints = []
    
    # Position constraints: p(0) = P0, p(1) = PN
    # These are handled via bounds, not equality constraints
    # (we'll set bounds separately)
    
    # Velocity constraints
    if v0 is not None:
        # v(0) = N(P1 - P0)
        # N*P1 - N*P0 = v0
        row = np.zeros(Np1 * dim)
        row[0:dim] = -N  # -N * P0
        row[dim:2*dim] = N  # N * P1
        constraints.append(LinearConstraint(row.reshape(1, -1), v0, v0))
    
    if v1 is not None:
        # v(1) = N(PN - PN-1)
        # N*PN - N*PN-1 = v1
        row = np.zeros(Np1 * dim)
        row[-2*dim:-dim] = -N  # -N * P_{N-1}
        row[-dim:] = N  # N * PN
        constraints.append(LinearConstraint(row.reshape(1, -1), v1, v1))
    
    # Acceleration constraints
    if a0 is not None and N >= 2:
        # a(0) = N(N-1)(P2 - 2P1 + P0)
        row = np.zeros(Np1 * dim)
        row[0:dim] = N*(N-1)  # N(N-1) * P0
        row[dim:2*dim] = -2*N*(N-1)  # -2N(N-1) * P1
        row[2*dim:3*dim] = N*(N-1)  # N(N-1) * P2
        constraints.append(LinearConstraint(row.reshape(1, -1), a0, a0))
    
    if a1 is not None and N >= 2:
        # a(1) = N(N-1)(PN - 2PN-1 + PN-2)
        row = np.zeros(Np1 * dim)
        row[-3*dim:-2*dim] = N*(N-1)  # N(N-1) * P_{N-2}
        row[-2*dim:-dim] = -2*N*(N-1)  # -2N(N-1) * P_{N-1}
        row[-dim:] = N*(N-1)  # N(N-1) * PN
        constraints.append(LinearConstraint(row.reshape(1, -1), a1, a1))
    
    return constraints

# def compute_cost_function(P, n_samples=100):
#     """
#     Compute cost function: ∫ ||a_geometric(τ) - a_gravitational(τ)||² dτ
    
#     From Project_Spec.md: "geometric acc of bezier - gravitational acc on each point"
#     We interpret this as minimizing the difference (fuel usage excluding gravity).
    
#     Args:
#         P: Control points (N+1, dim)
#         n_samples: Number of samples for numerical integration
    
#     Returns:
#         Cost value (scalar)
#     """
#     curve = BezierCurve(P)
#     ts = np.linspace(0, 1, n_samples)
    
#     cost = 0.0
#     dtau = 1.0 / (n_samples - 1)
    
#     for tau in ts:
#         # Geometric acceleration
#         a_geom = curve.acceleration(tau)
        
#         # Gravitational acceleration at this point
#         pos = curve.point(tau)
#         r = np.linalg.norm(pos)
#         if r < 1e-6:
#             a_grav_mag = 0.0
#         else:
#             a_grav_mag = EARTH_MU_SCALED / r**2
        
#         # Direction toward origin (negative of position unit vector)
#         if r > 1e-6:
#             a_grav = -a_grav_mag * (pos / r)
#         else:
#             a_grav = np.zeros_like(pos)
        
#         # Cost: ||a_geom - a_grav||²
#         diff = a_geom - a_grav
#         cost += np.linalg.norm(diff)**2 * dtau
    
#     return cost

def _compute_cost_only(P_flat, Np1, dim, n_samples=50):
    """Helper function to compute cost only (no recursion)."""
    P = P_flat.reshape(Np1, dim)
    curve = BezierCurve(P)
    ts = np.linspace(0, 1, n_samples)
    
    cost = 0.0
    accel = 0.0
    dtau = 1.0 / (n_samples - 1)
    
    for tau in ts:
        # Geometric acceleration at each point on the curve
        a_geom = curve.acceleration(tau)

        # Gravitational acceleration at each point on the curve
        pos = curve.point(tau)
        r = np.linalg.norm(pos)
        
        # Direction toward origin (negative of position unit vector)
        if r > 1e-6:
            a_grav = -EARTH_MU_SCALED / r**2 * (pos / r)
        else:
            a_grav = np.zeros_like(pos)
        
        # Cost: ||a_geom - a_grav||²
        # print("_coompute_cost_only DEBUG")
        # print("a_geom:", a_geom)
        # print("a_grav:", a_grav)
        diff = a_geom - a_grav
        norm_diff = np.linalg.norm(diff)
        cost += norm_diff**2 * dtau
        accel += norm_diff * dtau
    
    return cost, accel

def cost_function_gradient_hessian(P_flat, Np1, dim, n_samples=50, compute_grad=True):
    """
    Compute cost function, gradient, and Hessian approximation.
    Returns: (cost, gradient, hessian)
    
    Args:
        compute_grad: If False, only compute cost (for efficiency when gradient not needed)
    """
    cost, _ = _compute_cost_only(P_flat, Np1, dim, n_samples)
    
    if not compute_grad:
        return cost, None, None
    
    # Compute gradient using finite differences
    eps = 1e-6
    grad = np.zeros(Np1 * dim)
    for i in range(Np1 * dim):
        P_pert = P_flat.copy()
        P_pert[i] += eps
        cost_pert, _ = _compute_cost_only(P_pert, Np1, dim, n_samples)
        grad[i] = (cost_pert - cost) / eps
    
    # Simple diagonal Hessian approximation (regularization)
    hess = np.eye(Np1 * dim) * 1e-6
    
    return cost, grad, hess

# print("Cost function ready (simplified version).")
# print("Note: Full analytical gradient/hessian would be better for production.")

def optimize_orbital_docking(
    P_init, 
    n_seg=8, 
    r_e=KOZ_RADIUS, 
    max_iter=20,
    tol=1e-6,
    v0=None,
    v1=None,
    a0=None,
    a1=None,
    sample_count=100,
    verbose=True
):
    """
    Optimize Bézier curve for orbital docking.
    
    Args:
        P_init: Initial control points (N+1, dim)
        n_seg: Number of segments for KOZ linearization
        r_e: KOZ radius
        max_iter: Max optimization iterations
        tol: Convergence tolerance
        v0, v1: Velocity boundary conditions
        a0, a1: Acceleration boundary conditions
        verbose: Print progress
    
    Returns:
        (P_opt, info_dict)
    """
    P = P_init.copy()
    Np1, dim = P.shape
    N = Np1 - 1
    
    # Get segment matrices
    A_list = segment_matrices_equal_params(N, n_seg)
    
    # Set up bounds: fix endpoints for position constraints
    x0 = P.reshape(-1)
    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    lb[:dim] = ub[:dim] = x0[:dim]  # Lock P0
    lb[-dim:] = ub[-dim:] = x0[-dim:]  # Lock PN
    bounds = Bounds(lb, ub)
    
    # Build boundary constraints
    boundary_constraints = build_boundary_constraints(P, v0, v1, a0, a1, dim)
    
    # Store results
    info = {"iterations": 0, "feasible": False, "cost": np.inf}
    
    # Iterative optimization: update KOZ constraints based on current solution
    for it in range(1, max_iter + 1):
        # Build KOZ constraints based on current control points
        koz_constraint = build_koz_constraints(A_list, P, r_e, dim)
        
        # Combine all constraints
        all_constraints = [koz_constraint] + boundary_constraints
        
        # Objective function
        def objective(x):
            cost, _ = _compute_cost_only(x, Np1, dim, n_samples=sample_count)
            return cost
        
        # Gradient - compute gradient separately
        def gradient(x):
            _, grad, _ = cost_function_gradient_hessian(x, Np1, dim, compute_grad=True, n_samples=sample_count)
            return grad
        
        # Solve
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = minimize(
                objective,
                P.reshape(-1),
                method='trust-constr',
                jac=gradient,
                constraints=all_constraints,
                bounds=bounds,
                options={'maxiter': 50, 'gtol': 1e-8, 'disp': False}
            )
        
        P_new = res.x.reshape(Np1, dim)
        delta = np.linalg.norm(P_new - P)
        P = P_new
        
        if verbose:
            print(f"Iter {it}: cost={res.fun:.6e}, delta={delta:.6e}")
        
        if delta < tol:
            break
    
    # Final feasibility check
    curve = BezierCurve(P)
    ts_check = np.linspace(0, 1, 1000)
    pts = np.array([curve.point(t) for t in ts_check])
    radii = np.linalg.norm(pts, axis=1)
    min_radius = float(np.min(radii))
    _, accel = _compute_cost_only(P.reshape(-1), Np1, dim, n_samples=sample_count)
    
    info.update({
        "iterations": it,
        "feasible": min_radius >= r_e - 1e-6,
        "min_radius": min_radius,
        "cost": res.fun,
        "accel": accel
    })
    
    return P, info


# Uncomment to run:
# P_opt, info = optimize_orbital_docking(
#     P_init, 
#     n_seg=8, 
#     r_e=KOZ_RADIUS,
#     max_iter=120,
#     verbose=True
# )
# print(f"\n✅ Optimization complete!")
# print(f"   Iterations: {info['iterations']}")
# print(f"   Feasible: {info['feasible']}")
# print(f"   Min radius: {info['min_radius']:.1f} km")
# print(f"   Cost: {info['cost']:.6e}")

# def plot_trajectory_3d(P_init, P_opt, r_e, title="Orbital Docking Trajectory"):
#     """Plot 3D trajectory visualization."""
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Earth (smaller and more transparent)
#     u = np.linspace(0, 2*np.pi, 20)
#     v = np.linspace(0, np.pi, 20)
#     earth_radius = EARTH_RADIUS_KM * 0.7  # Make Earth 70% of original size
#     x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
#     y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
#     z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
#     ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.15, color='blue')
    
#     # KOZ sphere
#     x_koz = r_e * np.outer(np.cos(u), np.sin(v))
#     y_koz = r_e * np.outer(np.sin(u), np.sin(v))
#     z_koz = r_e * np.outer(np.ones(np.size(u)), np.cos(v))
#     ax.plot_wireframe(x_koz, y_koz, z_koz, alpha=0.5, color='red', linewidth=0.5)
    
#     # Trajectories
#     curve_init = BezierCurve(P_init)
#     curve_opt = BezierCurve(P_opt)
#     ts = np.linspace(0, 1, 200)
    
#     pts_init = np.array([curve_init.point(t) for t in ts])
#     pts_opt = np.array([curve_opt.point(t) for t in ts])
    
#     ax.plot(pts_init[:, 0], pts_init[:, 1], pts_init[:, 2], 
#            '--', color='gray', linewidth=2, label='Initial')
#     ax.plot(pts_opt[:, 0], pts_opt[:, 1], pts_opt[:, 2], 
#            '-', color='green', linewidth=3, label='Optimized')
    
#     # Control points
#     ax.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2], 
#               c='red', s=50, marker='o', label='Control Points')
#     ax.scatter(P_opt[:, 0], P_opt[:, 1], P_opt[:, 2], 
#               c='blue', s=50, marker='s')
    
#     ax.set_xlabel('X (km)')
#     ax.set_ylabel('Y (km)')
#     ax.set_zlabel('Z (km)')
#     ax.legend()
#     ax.set_title(title)
    
#     plt.tight_layout()
#     return fig

def plot_profiles(P_opt, title="Trajectory Profiles"):
    """Plot position, velocity, acceleration profiles."""
    curve = BezierCurve(P_opt)
    ts = np.linspace(0, 1, 300)
    
    positions = np.array([curve.point(t) for t in ts])
    velocities = np.array([curve.velocity(t) for t in ts])
    accelerations = np.array([curve.acceleration(t) for t in ts])
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Position
    axes[0].plot(ts, positions[:, 0], 'r-', label='X')
    axes[0].plot(ts, positions[:, 1], 'g-', label='Y')
    axes[0].plot(ts, positions[:, 2], 'b-', label='Z')
    axes[0].plot(ts, np.linalg.norm(positions, axis=1), 'k--', label='||Position||')
    axes[0].set_ylabel('Position (km)')
    axes[0].set_title('Position vs τ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Velocity
    axes[1].plot(ts, velocities[:, 0], 'r-', label='Vx')
    axes[1].plot(ts, velocities[:, 1], 'g-', label='Vy')
    axes[1].plot(ts, velocities[:, 2], 'b-', label='Vz')
    axes[1].plot(ts, np.linalg.norm(velocities, axis=1), 'k--', label='||Velocity||')
    axes[1].set_ylabel('Velocity (km/s)')
    axes[1].set_title('Velocity vs τ')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Acceleration
    axes[2].plot(ts, accelerations[:, 0], 'r-', label='Ax')
    axes[2].plot(ts, accelerations[:, 1], 'g-', label='Ay')
    axes[2].plot(ts, accelerations[:, 2], 'b-', label='Az')
    axes[2].plot(ts, np.linalg.norm(accelerations, axis=1), 'k--', label='||Acceleration||')
    axes[2].set_xlabel('Parameter τ')
    axes[2].set_ylabel('Acceleration (km/s²)')
    axes[2].set_title('Acceleration vs τ')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig
# Professional visualization helpers (adapted from Orbital_Docking_Optimizer.py)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def add_wire_sphere(ax, radius=3.0, center=(0.0, 0.0, 0.0), color='gray', alpha=0.25, resolution=40):
    """
    Add a wireframe sphere to a 3D plot.
    
    Args:
        ax: 3D matplotlib axes
        radius: Sphere radius (in km)
        center: Sphere center coordinates (in km)
        color: Sphere color
        alpha: Transparency (0-1)
        resolution: Number of grid points for sphere generation
    """
    cx, cy, cz = center
    # Generate spherical coordinates
    u = np.linspace(0, 2*np.pi, resolution)  # Azimuthal angle
    v = np.linspace(0, np.pi, resolution)    # Polar angle
    uu, vv = np.meshgrid(u, v)
    
    # Convert to Cartesian coordinates
    x = cx + radius * np.cos(uu) * np.sin(vv)
    y = cy + radius * np.sin(uu) * np.sin(vv)
    z = cz + radius * np.cos(vv)
    
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, alpha=alpha)

def add_earth_sphere(ax, radius=EARTH_RADIUS_KM, center=(0.0, 0.0, 0.0), color='blue', alpha=0.3):
    """
    Add Earth as a wireframe sphere for orbital context.
    
    Args:
        ax: 3D matplotlib axes
        radius: Earth radius in km
        center: Earth center coordinates
        color: Earth color
        alpha: Transparency
    """
    add_wire_sphere(ax, radius=radius, center=center, color=color, alpha=alpha, resolution=20)

def set_axes_equal_around(ax, center=(0,0,0), radius=1.0, pad=0.05):
    """
    Set 3D axes to equal aspect ratio around a specified center and radius.
    
    Args:
        ax: 3D matplotlib axes
        center: Center point for the view (in km)
        radius: Radius around center to include (in km)
        pad: Additional padding factor
    """
    cx, cy, cz = center
    
    # Get current axis limits
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    
    # Expand limits to include the specified sphere
    x0 = min(x0, cx - radius); x1 = max(x1, cx + radius)
    y0 = min(y0, cy - radius); y1 = max(y1, cy + radius)
    z0 = min(z0, cz - radius); z1 = max(z1, cz + radius)
    
    # Set equal aspect ratio
    max_range = max(x1 - x0, y1 - y0, z1 - z0)
    half = 0.5 * max_range * (1 + pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect((1, 1, 1))  # Equal aspect ratio

def set_isometric(ax, elev=35.264, azim=45.0, ortho=True):
    """
    Set isometric view for 3D plot.
    
    Args:
        ax: 3D matplotlib axes
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        ortho: Whether to use orthogonal projection
    """
    ax.view_init(elev=elev, azim=azim)
    try:
        if ortho:
            ax.set_proj_type('ortho')  # Requires matplotlib ≥3.2
    except Exception:
        pass  # Fallback for older matplotlib versions

def beautify_3d_axes(ax, show_ticks=True, show_grid=True):
    """
    Apply paper-friendly styling to 3D axes.
    
    Args:
        ax: 3D matplotlib axes
        show_ticks: Whether to show axis ticks and labels
        show_grid: Whether to show grid lines
    """
    ax.grid(show_grid)
    if show_ticks:
        ax.tick_params(axis='both', which='major', labelsize=8, pad=2)
    else:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
    # Style the axis panes (background planes)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1, 1, 1, 1))  # White background
            axis.pane.set_edgecolor("0.85")        # Light gray edges
        except Exception:
            pass  # Fallback for older matplotlib versions

def plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', show_seg_ctrl_if=8, lw=2.0):
    """
    Plot Bézier curve with color-coded segments and control polygon.
    
    Args:
        ax: 3D matplotlib axes
        P_opt: Optimized control points
        n_seg: Number of segments
        cmap_name: Colormap name for segment colors
        show_seg_ctrl_if: Show segment control polygons if n_seg <= this value
        lw: Line width for the curve
    """
    curve = BezierCurve(P_opt)
    N = curve.degree
    A_list = segment_matrices_equal_params(N, n_seg)

    # Plot whole-curve control polygon in black
    ax.plot(P_opt[:,0], P_opt[:,1], P_opt[:,2], 'k.-', lw=1.2, ms=4)

    # Generate colors for segments using colormap
    # base_colors = ['#FF4444', '#4444FF', '#44FF44'] # Red, Blue, Green - high contrast
    base_colors = ['#E74C3C', '#3498DB', '#F39C12'] # (red, blue, orange)
    # base_colors = ['#C0392B', '#2874A6', '#27AE60'] # (darker red, blue, green)
    colors = [base_colors[i % 3] for i in range(len(A_list))]
    # cmap = plt.get_cmap(cmap_name)
    # colors = [cmap(0.0)] if len(A_list) == 1 else [cmap(i/(len(A_list)-1)) for i in range(len(A_list))]

    # Draw each segment with its color
    ts = np.linspace(0, 1, 180)  # Parameter values for smooth curves
    lines, cols = [], []
    for i, Ai in enumerate(A_list):
        Qi = Ai @ P_opt  # Control points for segment i
        seg = BezierCurve(Qi)
        Pseg = np.array([seg.point(t) for t in ts])
        lines.append(np.column_stack((Pseg[:,0], Pseg[:,1], Pseg[:,2])))
        cols.append(colors[i])

        # Show segment control polygons only for small numbers of segments
        if n_seg <= show_seg_ctrl_if:
            ax.plot(Qi[:,0], Qi[:,1], Qi[:,2], '-', color=colors[i], alpha=0.55, lw=1.0, marker='o', ms=5)
        if n_seg > show_seg_ctrl_if:
            ax.plot(Qi[:,0], Qi[:,1], Qi[:,2], '-', color=colors[i], alpha=0.55, lw=1.0, marker='o', ms=3)

    # Create 3D line collection for efficient rendering
    lc = Line3DCollection(lines, colors=cols, linewidths=lw, alpha=0.95)
    ax.add_collection3d(lc)

def optimize_all_segment_counts(P_init, r_e=KOZ_RADIUS, segment_counts=[2, 4, 8, 16, 32, 64], 
                                max_iter=120, tol=1e-8, verbose=True):
    """
    Run optimization for multiple segment counts and return results.
    
    Args:
        P_init: Initial control points (N+1, dim)
        r_e: KOZ radius
        segment_counts: List of segment counts to test
        max_iter: Maximum iterations per optimization
        tol: Convergence tolerance
        verbose: Print progress
        
    Returns:
        List of tuples: [(n_seg, P_opt, info), ...] where info includes acceleration
    """
    results = []
    
    print(f"Optimizing trajectories for segment counts: {segment_counts}")
    print("=" * 60)
    
    for n_seg in segment_counts:
        if verbose:
            print(f"\nRunning optimization for n_seg={n_seg}...")
        
        # Run optimization
        P_opt, info = optimize_orbital_docking(
            P_init, 
            n_seg=n_seg, 
            r_e=r_e, 
            max_iter=max_iter, 
            tol=tol,
            sample_count=100,
            verbose=False
        )
        
        
        results.append((n_seg, P_opt, info))
        
        if verbose:
            print(f"  ✓ n_seg={n_seg:>2d} | iter={info['iterations']:>3d} | "
                  f"acceleration={info['accel']:.3f} | feasible={info['feasible']}")
    
    print("\n" + "=" * 60)
    print("All optimizations complete!")
    
    return results
    
def create_trajectory_comparison_figure(P_init, r_e, results):
    """
    Create 2×3 layout showing trajectories with different segment counts.
    Uses same aesthetics as Orbital_Docking_Optimizer.py.
    
    Args:
        P_init: Initial control points
        r_e: KOZ radius
        results: List of (n_seg, P_opt, info) tuples
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    
    # Create 2×3 subplot layout
    axes = []
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        axes.append(ax)
    
    # Plot trajectories for different segment counts
    segment_counts = [2, 4, 8, 16, 32, 64]
    
    for i, (n_seg, P_opt, info) in enumerate(results):
        ax = axes[i]
        
        # Add Earth (smaller blue sphere, more transparent)
        add_earth_sphere(ax, radius=EARTH_RADIUS_KM * 0.7, color='blue', alpha=1)
        
        # Add safety zone (KOZ - smaller red sphere)
        add_wire_sphere(ax, radius=r_e, color='red', alpha=0.2, resolution=15)
        
        # Plot optimized trajectory (thicker lines with color-coded segments)
        plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', lw=3.0)
        
        # Add start and end markers (larger markers)
        ax.scatter(P_init[0,0], P_init[0,1], P_init[0,2], 
                  color='green', s=120, label='Chaser', zorder=10)
        ax.scatter(P_init[-1,0], P_init[-1,1], P_init[-1,2], 
                  color='orange', s=120, label='Target', zorder=10)
        ax.legend(fontsize=8)
        
        # Professional styling (adjusted radius for proper zoom)
        # Use larger radius to include Earth and trajectories properly
        view_radius = max(EARTH_RADIUS_KM * 1.2, np.linalg.norm(P_init, axis=1).max() * 1.1)
        set_axes_equal_around(ax, center=(0,0,0), radius=view_radius, pad=0.1)
        set_isometric(ax, elev=20, azim=45)
        beautify_3d_axes(ax, show_ticks=True, show_grid=True)
        
        # Title with acceleration (convert to m/s² from km/s²)
        accel_ms2 = info['accel'] * 1e3  # Convert km/s² to m/s²
        ax.set_title(f'{n_seg} Segments\nAccel: {accel_ms2:.1f} m/s²', fontsize=10, pad=10)
    
    return fig

def create_performance_figure(results):
    """
    Create performance figure showing acceleration vs segment count.
    
    Args:
        results: List of (n_seg, P_opt, info) tuples
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Extract data
    segment_counts = [n_seg for n_seg, _, _ in results]
    accelerations = [info['accel'] * 1e-3 for _, _, info in results]  # Convert km/s² to m/s²
    
    # Create acceleration performance graph
    ax.plot(segment_counts, accelerations, 'bo-', linewidth=3, markersize=10)
    ax.set_xlabel('Number of Segments', fontsize=14)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=14)
    ax.set_title('Performance Improvement with More Segments', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)  # Log scale for segment counts
    
    # Add data point labels
    for i, (seg, accel) in enumerate(zip(segment_counts, accelerations)):
        ax.annotate(f'{accel:.1f}', (seg, accel), textcoords="offset points", xytext=(0,10), ha='center',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    return fig

# print("Optimization function ready.")
# print("⚠️  Note: This uses simplified cost/gradient - optimize in later cells!")

# Create initial guess: simple quadratic curve from chaser to ISS
# Positions in km, scaled appropriately

# Start: chaser position (at 300km altitude)
theta_start = -np.pi / 4  # 45 degrees
P0 = np.array([
    CHASER_RADIUS * np.cos(theta_start),
    CHASER_RADIUS * np.sin(theta_start),
    0.0
])

# End: ISS position (at 423km altitude)  
theta_end = np.pi / 4  # -45 degrees
P2 = np.array([
    ISS_RADIUS * np.cos(theta_end),
    ISS_RADIUS * np.sin(theta_end),
    0.0
])

# Middle control point (straight line approximation)
P1 = (P0 + P2) / 2

P_init = np.vstack([P0, P1, P2])

print(f"Initial control points (km):")
print(f"  P0 (chaser): {P0}")
print(f"  P1 (mid):    {P1}")
print(f"  P2 (ISS):    {P2}")
print(f"\nKOZ radius: {KOZ_RADIUS:.1f} km")

# OPTIMIZATION - This is the expensive operation
# Once run, results stay in kernel memory for visualization
print("\n⚠️  Starting optimization (this may take a while)...")
print("    You can interrupt and use cached results for visualization\n")

# Run optimization for all segment counts
# This may take a while depending on max_iter and convergence
segment_counts = [2, 4, 8, 16, 32, 64]

# Use the same P_init from Cell 17 (if already computed) or redefine here
# If P_init is not available, uncomment the following:
# theta_start = np.pi / 4
# theta_end = -np.pi / 4
# P0 = np.array([
#     CHASER_RADIUS * np.cos(theta_start),
#     CHASER_RADIUS * np.sin(theta_start),
#     0.0
# ])
# P2 = np.array([
#     ISS_RADIUS * np.cos(theta_end),
#     ISS_RADIUS * np.sin(theta_end),
#     0.0
# ])
# P1 = (P0 + P2) / 2
# P_init = np.vstack([P0, P1, P2])

# Run optimizations for all segment counts
results_all = optimize_all_segment_counts(
    P_init, 
    r_e=KOZ_RADIUS,
    segment_counts=segment_counts,
    max_iter=60,
    tol=1e-3,
    verbose=True
)

# Create and display the 2×3 comparison figure
fig_comparison = create_trajectory_comparison_figure(P_init, KOZ_RADIUS, results_all)

# Create and display the performance figure
fig_performance = create_performance_figure(results_all)

plt.show()

print("\n✅ Trajectory comparison complete!")
print(f"Displayed {len(results_all)} trajectories with segment counts: {segment_counts}")
print("\n✅ Performance figure complete!")
