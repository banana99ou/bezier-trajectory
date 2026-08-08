"""
Shared library for the SCvx fix-verification harness (see plan
fancy-coalescing-forest / doc/tmp.md pillars 1-4).

Everything here is objective-agnostic reference physics used to CHECK the Rust
SCvx optimizer, independent of what the Rust solver internally minimizes.

Key idea (do not violate): the Rust `cost_true_energy` is a *linearized
surrogate*. The single source of truth for cross-solver comparison is
`J_true(P)` below, which evaluates the true nonconvex control-effort functional
    J = mean_k || a_geom(tau_k)/T^2  -  a_grav_total(r(tau_k)) ||^2
on a dense tau grid with the TRUE (non-linearized) two-body + J2 gravity.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.special import comb

# Make `orbital_docking` importable regardless of cwd (repo root = parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orbital_docking import constants  # noqa: E402
from orbital_docking.bezier import BezierCurve, get_D_matrix, get_E_matrix  # noqa: E402
from orbital_docking.optimization import (  # noqa: E402
    _accel_total,
    generate_initial_control_points,
    optimize_orbital_docking,
)

MU = constants.EARTH_MU_SCALED
R_E_EARTH = constants.EARTH_RADIUS_KM
J2 = constants.EARTH_J2


# ----------------------------------------------------------------------------
# Scenario construction (matches tools/probe_frozen_jacobian.py)
# ----------------------------------------------------------------------------

def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def eci_from_circular(radius_km, inc_deg, raan_deg, u_deg):
    """Position + circular velocity of a point on a circular orbit (ECI, km / km/s)."""
    inc, raan, u = np.deg2rad([inc_deg, raan_deg, u_deg])
    r_pqw = np.array([radius_km * np.cos(u), radius_km * np.sin(u), 0.0])
    v_circ = np.sqrt(MU / radius_km)
    v_pqw = v_circ * np.array([-np.sin(u), np.cos(u), 0.0])
    q = _rotz(raan) @ _rotx(inc)
    return q @ r_pqw, q @ v_pqw


_SCENARIOS = {
    # The trust-matters regime studied throughout the fix (120 deg phase lag).
    "phase120": dict(N=7, progress_alt=245.0, iss_alt=400.0, inc=51.64,
                     raan=0.0, iss_u=45.0, lag=120.0, r_e=6471.0, T=1500.0),
    # A milder geometry (KOZ less aggressively active).
    "phase70": dict(N=7, progress_alt=245.0, iss_alt=400.0, inc=51.64,
                    raan=0.0, iss_u=45.0, lag=70.0, r_e=6471.0, T=1500.0),
}


def make_scenario(name, N=None):
    """Return a scenario dict: P_start, P_end, v0, v1, r_e, T, N, P_init."""
    s = _SCENARIOS[name]
    deg = int(N if N is not None else s["N"])
    progress_r = R_E_EARTH + s["progress_alt"]
    iss_r = R_E_EARTH + s["iss_alt"]
    P_start, v0 = eci_from_circular(progress_r, s["inc"], s["raan"], s["iss_u"] - s["lag"])
    P_end, v1 = eci_from_circular(iss_r, s["inc"], s["raan"], s["iss_u"])
    P_init = generate_initial_control_points(deg, P_start, P_end)
    return dict(name=name, N=deg, P_start=P_start, P_end=P_end, v0=v0, v1=v1,
                r_e=float(s["r_e"]), T=float(s["T"]), P_init=np.asarray(P_init, float))


# ----------------------------------------------------------------------------
# Vectorized Bezier evaluation (fast; the NLP calls these thousands of times)
# ----------------------------------------------------------------------------

def bernstein_matrix(N, taus):
    """(len(taus), N+1) Bernstein basis matrix for degree N."""
    taus = np.asarray(taus, float)
    i = np.arange(N + 1)
    coef = comb(N, i)
    return coef[None, :] * (taus[:, None] ** i[None, :]) * ((1.0 - taus)[:, None] ** (N - i)[None, :])


def _accel_ctrl_points(P):
    """Acceleration control points A = EDED P (degree N, tau^-2 units)."""
    N = P.shape[0] - 1
    if N < 2:
        return np.zeros_like(P)
    D = get_D_matrix(N)
    E = get_E_matrix(N - 1)
    return E @ D @ E @ D @ P


def positions(P, taus):
    """r(tau_k), shape (len(taus), dim)."""
    return bernstein_matrix(P.shape[0] - 1, taus) @ P


def accels_tau(P, taus):
    """d^2r/dtau^2 at tau_k (NOT divided by T^2), shape (len(taus), dim)."""
    return bernstein_matrix(P.shape[0] - 1, taus) @ _accel_ctrl_points(P)


def grav_total(rs):
    """True two-body + J2 acceleration for a stack of positions (n,3) -> (n,3)."""
    rs = np.atleast_2d(rs)
    return np.array([_accel_total(r, MU, R_E_EARTH, J2) for r in rs])


def J_true(P, T, n_dense=1200):
    """
    Canonical true nonconvex objective:
        mean_k || a_geom(tau_k)/T^2 - a_grav_total(r(tau_k)) ||^2   [ (km/s^2)^2 ]
    The 1/n_dense mean-normalization mirrors the Rust w_seg = 1/n_lin_seg scaling
    so magnitudes are comparable to `cost_true_energy` (though NOT identical:
    Rust uses linearized gravity + segment centroids).
    """
    P = np.asarray(P, float)
    taus = np.linspace(0.0, 1.0, int(n_dense))
    r = positions(P, taus)
    a_geom = accels_tau(P, taus) / (T * T)
    u = a_geom - grav_total(r)
    return float(np.mean(np.sum(u * u, axis=1)))


def _grav_jacobians(rs, h=1e-3):
    """Numeric gravity Jacobians dg/dr at each position, shape (n,3,3)."""
    from orbital_docking.optimization import _jacobian_numeric
    return np.array([_jacobian_numeric(
        lambda r: _accel_total(r, MU, R_E_EARTH, J2), rr, h) for rr in rs])


def J_true_and_grad(P, T, n_dense=1200):
    """
    J_true and its analytic gradient wrt the flattened control points.
    Returns (J, grad_flat) with grad_flat shape ((N+1)*dim,).
    """
    P = np.asarray(P, float)
    N = P.shape[0] - 1
    taus = np.linspace(0.0, 1.0, int(n_dense))
    Bp = bernstein_matrix(N, taus)                       # (n, N+1)  -> r = Bp @ P
    EDED = get_E_matrix(N - 1) @ get_D_matrix(N) @ get_E_matrix(N - 1) @ get_D_matrix(N)
    Ca = Bp @ EDED                                       # (n, N+1)  -> a_tau = Ca @ P
    r = Bp @ P
    a = (Ca @ P) / (T * T)
    g = grav_total(r)
    Jg = _grav_jacobians(r)                              # (n,3,3)
    u = a - g                                            # (n,3)
    J = float(np.mean(np.sum(u * u, axis=1)))
    # w_k[d'] = sum_d u_k[d] * Jg_k[d,d']
    w = np.einsum("kd,kde->ke", u, Jg)                   # (n,3)
    grad = (2.0 / n_dense) * ((Ca.T @ u) / (T * T) - Bp.T @ w)  # (N+1, 3)
    return J, grad.reshape(-1)


def min_radius(P, n=1000):
    """Dense min ||r(tau)|| (km) -- the TRUE KOZ metric (mirror optimization.py)."""
    taus = np.linspace(0.0, 1.0, int(n))
    return float(np.min(np.linalg.norm(positions(np.asarray(P, float), taus), axis=1)))


def velocity_endpoints(P, T):
    """Physical (v0, v1) implied by the control points: v = (N/T)(P_{i+1}-P_i)."""
    N = P.shape[0] - 1
    v0 = (N / T) * (P[1] - P[0])
    v1 = (N / T) * (P[-1] - P[-2])
    return v0, v1


# ----------------------------------------------------------------------------
# Rust solver wrapper (always cache-off to avoid stale-binary hits)
# ----------------------------------------------------------------------------

def run_rust(scenario, n_seg=16, max_iter=1000,
             tol=1e-8, scp_trust_radius=2000.0, scp_prox_weight=0.0,
             sample_count=100, enforce_prograde=False, **overrides):
    """Run the Rust SCvx solver on a scenario. Returns (P_opt, info)."""
    kwargs = dict(
        n_seg=n_seg, r_e=scenario["r_e"], max_iter=max_iter, tol=tol,
        v0=scenario["v0"], v1=scenario["v1"], sample_count=sample_count,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius, transfer_time=scenario["T"],
        enforce_prograde=enforce_prograde, verbose=False,
        use_cache=False, ignore_existing_cache=True,
    )
    kwargs.update(overrides)
    P_opt, info = optimize_orbital_docking(scenario["P_init"], **kwargs)
    return np.asarray(P_opt, float), dict(info)


# ----------------------------------------------------------------------------
# Tiny output helpers (CSV + markdown), matching the tools/ house style
# ----------------------------------------------------------------------------

def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


ARTIFACT_ROOT = _REPO_ROOT / "artifacts" / "verify"
