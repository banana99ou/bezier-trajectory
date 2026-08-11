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
# Scenario construction — authoritative. (Formerly mirrored from a probe script;
# those were deleted with the scvx_freeze investigation they belonged to.)
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


# Scenario grid. `inc`/`raan` set the departure orbit plane; `inc1`/`raan1`
# default to them (coplanar) and may be given separately for a plane change,
# which makes the two velocity boundary conditions point in materially
# different directions. Larger `lag` = the chord passes deeper through the KOZ,
# so the half-space rows bind harder and the normals re-aim further per step —
# which is exactly the regime that exposes solver defects (§9).
_SCENARIOS = {
    # The trust-matters regime studied throughout the fix (120 deg phase lag).
    "phase120": dict(N=7, progress_alt=245.0, iss_alt=400.0, inc=51.64,
                     raan=0.0, iss_u=45.0, lag=120.0, r_e=6471.0, T=1500.0, r0=2000.0),
    # A milder geometry (KOZ less aggressively active).
    "phase70": dict(N=7, progress_alt=245.0, iss_alt=400.0, inc=51.64,
                    raan=0.0, iss_u=45.0, lag=70.0, r_e=6471.0, T=1500.0, r0=2000.0),
    # Harder phasings: the KOZ binds over a longer arc than phase120.
    "phase135": dict(N=7, progress_alt=245.0, iss_alt=400.0, inc=51.64,
                     raan=0.0, iss_u=45.0, lag=135.0, r_e=6471.0, T=1500.0, r0=2000.0),
    # 170 deg is nearly antipodal: the straight-line initial guess passes 5420 km
    # INSIDE the keep-out sphere, so r0 must be large enough to repair that at
    # iteration 1 (design_freeze section 5). r0=2000 fails with stop_reason=3;
    # r0>=4000 converges, and to the same answer for every r0 in 4000..12000.
    "phase170": dict(N=7, progress_alt=245.0, iss_alt=400.0, inc=51.64,
                     raan=0.0, iss_u=45.0, lag=170.0, r_e=6471.0, T=1500.0, r0=4000.0),
    # Plane change: same 120 deg phasing, but the target orbit is inclined 20 deg
    # further and 15 deg off in RAAN, so v0 and v1 differ in DIRECTION as well as
    # magnitude (~25 deg apart). The transfer is genuinely three-dimensional, and
    # the segment centroids leave the departure plane — the case where a single
    # supporting half-space per segment is least representative of the sphere.
    "planechange": dict(N=7, progress_alt=245.0, iss_alt=400.0, inc=51.64,
                        raan=0.0, inc1=71.64, raan1=15.0, iss_u=45.0, lag=120.0,
                        r_e=6471.0, T=1500.0, r0=2000.0),
}


def make_scenario(name, N=None):
    """Return a scenario dict: P_start, P_end, v0, v1, r_e, T, N, P_init."""
    s = _SCENARIOS[name]
    deg = int(N if N is not None else s["N"])
    progress_r = R_E_EARTH + s["progress_alt"]
    iss_r = R_E_EARTH + s["iss_alt"]
    P_start, v0 = eci_from_circular(progress_r, s["inc"], s["raan"], s["iss_u"] - s["lag"])
    P_end, v1 = eci_from_circular(iss_r, s.get("inc1", s["inc"]),
                                  s.get("raan1", s["raan"]), s["iss_u"])
    P_init = generate_initial_control_points(deg, P_start, P_end)
    return dict(name=name, N=deg, P_start=P_start, P_end=P_end, v0=v0, v1=v1,
                r_e=float(s["r_e"]), T=float(s["T"]), r0=float(s.get("r0", 2000.0)),
                P_init=np.asarray(P_init, float))


def bc_angle_deg(sc):
    """Angle between the two velocity boundary conditions (deg) — how much the
    departure and arrival headings differ. 0 would mean parallel."""
    a, b = sc["v0"], sc["v1"]
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


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


def J_true(P, T, n_nodes=192):
    """
    Canonical true nonconvex objective, as an INTEGRAL:
        J = int_0^1 || a_geom(tau)/T^2 - a_grav_total(r(tau)) ||^2 dtau
    in (km/s^2)^2, matching the normalization of the Rust exact-Gram objective
    (though NOT identical to it: Rust linearizes gravity per segment).

    Gauss-Legendre, NOT a uniform mean. The former `mean over linspace` form was
    a Riemann sum with full-weight endpoints and converged only as O(1/n): at its
    n_dense=1200 default it carried ~1.6e-3 relative error, which is LARGER than
    differences this harness is routinely asked to resolve (the A/B objective gap
    is 1.7e-3). It also manufactured phantom descent directions in
    `tools/verify/optimality.py` -- a "1.4e-5 improvement" that reversed sign to
    +5.0e-5 once the quadrature converged. Measured convergence of this form:
    n_nodes 48 -> 96 -> 192 changes J by <1e-12 relative, i.e. it is exact to
    f64 for this integrand.
    """
    P = np.asarray(P, float)
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    taus = 0.5 * (x + 1.0)          # map [-1,1] -> [0,1]
    wts = 0.5 * w
    r = positions(P, taus)
    a_geom = accels_tau(P, taus) / (T * T)
    u = a_geom - grav_total(r)
    return float(np.sum(wts * np.sum(u * u, axis=1)))


def _grav_jacobians(rs, h=1e-3):
    """Numeric gravity Jacobians dg/dr at each position, shape (n,3,3)."""
    from orbital_docking.optimization import _jacobian_numeric
    return np.array([_jacobian_numeric(
        lambda r: _accel_total(r, MU, R_E_EARTH, J2), rr, h) for rr in rs])


def J_true_and_grad(P, T, n_nodes=192):
    """
    J_true and its analytic gradient wrt the flattened control points.
    Returns (J, grad_flat) with grad_flat shape ((N+1)*dim,).

    Gauss-Legendre, on the SAME nodes as J_true, so this function's J agrees
    with J_true(P, T) to floating-point round-off (~1e-15 relative; not
    bit-identical, since the two differ in summation order).
    `tests/regression/test_harness_oracle.py` asserts both that agreement and
    the gradient against central differences.

    History: until 2026-08-11 this was a uniform mean over `linspace` -- the
    O(1/n) Riemann sum that was removed from J_true on 2026-08-09 but never from
    its gradient twin. Measured on phase120, that form carried 0.196%-0.207%
    relative error at the n_dense=600 its caller used. Pillar 1 optimizes with
    this objective and gradient, then scores the result with the corrected
    J_true, and reports a gap that at n_seg=64 was 0.165% -- SMALLER than the
    oracle's own error. That is the same defect shape as the 2026-08-09 A/B bug:
    an oracle asked to resolve a difference finer than itself.
    """
    P = np.asarray(P, float)
    N = P.shape[0] - 1
    x, w_gl = np.polynomial.legendre.leggauss(int(n_nodes))
    taus = 0.5 * (x + 1.0)                               # map [-1,1] -> [0,1]
    wts = 0.5 * w_gl
    Bp = bernstein_matrix(N, taus)                       # (n, N+1)  -> r = Bp @ P
    EDED = get_E_matrix(N - 1) @ get_D_matrix(N) @ get_E_matrix(N - 1) @ get_D_matrix(N)
    Ca = Bp @ EDED                                       # (n, N+1)  -> a_tau = Ca @ P
    r = Bp @ P
    a = (Ca @ P) / (T * T)
    g = grav_total(r)
    Jg = _grav_jacobians(r)                              # (n,3,3)
    u = a - g                                            # (n,3)
    J = float(np.sum(wts * np.sum(u * u, axis=1)))
    # w_k[d'] = sum_d u_k[d] * Jg_k[d,d']
    w = np.einsum("kd,kde->ke", u, Jg)                   # (n,3)
    uw = wts[:, None] * u
    ww = wts[:, None] * w
    grad = 2.0 * ((Ca.T @ uw) / (T * T) - Bp.T @ ww)     # (N+1, 3)
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
             tol=1e-8, scp_trust_radius=None, scp_prox_weight=0.0,
             sample_count=100, enforce_prograde=False, **overrides):
    """Run the Rust SCvx solver on a scenario. Returns (P_opt, info).

    scp_trust_radius defaults to the scenario's own r0: it must exceed the
    iteration-1 boundary-condition repair distance, which is a property of the
    geometry, not a global constant (design_freeze section 5).
    """
    if scp_trust_radius is None:
        scp_trust_radius = float(scenario.get("r0", 2000.0))
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
