"""The independent oracle.

Design reference: `doc/dcm_downstream_experiment_design.md` section 5, and
principle P5 -- a solver's own `converged` flag is a claim, not evidence.

Nothing in this module calls a solver. It takes a returned control history,
rebuilds the control the way the transcription defines it, integrates the
baseline's *own* equations of motion forward with an adaptive Runge-Kutta
scheme, and measures the result against thresholds frozen in the case file.

An oracle that cannot fail is worth nothing, so this module also carries the
checks that can void its own verdicts:

  * `check_resolution`     -- doubling the integration accuracy must not move
                              the measured quantities.
  * `check_cost_reconstruction` -- the objective the grader recomputes from its
                              own control interpolant must agree with the value
                              the solver reports. Disagreement means the grader
                              is not reading the control the solver produced.
  * `check_corruption`     -- a deliberately broken solution must be rejected.
  * `dynamics_residual`    -- the bridge's `u = r''/T^2 - g(r)` construction,
                              fed back through these same equations of motion,
                              must give zero to machine precision.

`self_test()` runs the ones that need no solver output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from orbit_transfer.astrodynamics.orbital_elements import oe_to_rv
from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.dynamics.eom import spacecraft_eom_numpy

from .cases import Case, Thresholds


# --------------------------------------------------------------------------
# Control reconstruction
# --------------------------------------------------------------------------

def barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    """Barycentric interpolation weights for an arbitrary node set.

    Computed in a variable affinely rescaled to [-1, 1]. The barycentric formula
    is invariant under that rescaling -- the Jacobian appears as a common factor
    in numerator and denominator and cancels -- but the products that form the
    weights are not, and on a physical time axis of thousands of seconds they
    overflow well before the node count reaches anything unusual.
    """
    x = np.asarray(nodes, dtype=float)
    lo, hi = x[0], x[-1]
    span = hi - lo
    s = 2.0 * (x - lo) / span - 1.0 if span > 0 else x - lo
    n = len(s)
    w = np.ones(n)
    for j in range(n):
        diff = s[j] - s
        diff[j] = 1.0
        w[j] = 1.0 / np.prod(diff)
    return w


class PhasewiseControl:
    """The control as the multi-phase LGL transcription defines it.

    Within each phase the control is the degree-(N-1) polynomial through that
    phase's N LGL nodes; across a phase boundary it is discontinuous, because
    the transcription gives the two phases independent control variables there.

    Reconstruction caveat, stated because it bounds what this class can claim:
    the baseline assembles its result into a single strictly-increasing time
    array and drops the duplicate node at each linkage, keeping the later
    phase's value. The earlier phase's terminal control value is therefore not
    recoverable from the returned arrays, and this class substitutes the later
    phase's initial value in its place. `check_cost_reconstruction` is what
    decides whether that substitution mattered for a given solution.
    """

    def __init__(self, t: np.ndarray, u: np.ndarray, phases: list[dict]):
        self.t0 = float(t[0])
        self.t1 = float(t[-1])
        self.segments = []
        for phase in phases:
            a, b = float(phase["t_start"]), float(phase["t_end"])
            # Nodes of this phase, taken from the assembled array. The tolerance
            # absorbs the epsilon nudge the baseline applies to enforce strict
            # monotonicity.
            span = max(b - a, 1e-12)
            sel = np.where((t >= a - 1e-9 * span) & (t <= b + 1e-9 * span))[0]
            if len(sel) < 2:
                continue
            nodes = t[sel]
            lo, hi = nodes[0], nodes[-1]
            width = hi - lo
            self.segments.append({
                "t_start": a,
                "t_end": b,
                "lo": lo,
                "width": width if width > 0 else 1.0,
                "nodes_scaled": (2.0 * (nodes - lo) / width - 1.0
                                 if width > 0 else nodes - lo),
                "values": u[:, sel],
                "weights": barycentric_weights(nodes),
            })
        if not self.segments:
            raise ValueError("no usable phase segment in the returned solution")

    def _segment_for(self, tq: float) -> dict:
        for seg in self.segments:
            if tq <= seg["t_end"]:
                return seg
        return self.segments[-1]

    def __call__(self, tq: float) -> np.ndarray:
        tq = float(np.clip(tq, self.t0, self.t1))
        seg = self._segment_for(tq)
        # Evaluate in the same rescaled variable the weights were built in.
        s = 2.0 * (tq - seg["lo"]) / seg["width"] - 1.0
        w, f = seg["weights"], seg["values"]
        d = s - seg["nodes_scaled"]
        hit = np.where(np.abs(d) < 1e-13)[0]
        if len(hit):
            return f[:, hit[0]].copy()
        c = w / d
        return (f @ c) / c.sum()

    def sample(self, times: np.ndarray) -> np.ndarray:
        return np.column_stack([self(float(tq)) for tq in times])


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

@dataclass
class GradeResult:
    """What the oracle measured, and whether it accepts. Every field is a
    measurement; `verdict` is the only derived quantity."""

    verdict: bool = False
    reasons: list[str] = field(default_factory=list)

    pos_err_km: float = float("nan")
    vel_err_km_s: float = float("nan")
    min_altitude_km: float = float("nan")
    altitude_floor_km: float = float("nan")
    max_thrust_km_s2: float = float("nan")
    thrust_cap_km_s2: float = float("nan")
    objective_grader: float = float("nan")
    objective_solver: float = float("nan")
    cost_recon_rel_err: float = float("nan")

    propagation_ok: bool = False
    n_eval: int = 0


def _propagate(x0, control, T, rtol, atol, n_dense):
    """Integrate the baseline's own equations of motion under the given control."""

    def rhs(tq, xq):
        return spacecraft_eom_numpy(xq, control(tq), mu=MU_EARTH, include_j2=True)

    sol = solve_ivp(
        rhs, (0.0, T), np.asarray(x0, dtype=float),
        method="DOP853", rtol=rtol, atol=atol, dense_output=True,
    )
    t_dense = np.linspace(0.0, T, n_dense)
    return sol, sol.sol(t_dense) if sol.success else None, t_dense


def _objective(control, T, n_gauss=200):
    """The exact integral of the squared control over the reconstructed
    interpolant, by composite Gauss-Legendre. Independent of the quadrature the
    solver used, which is the point: two independent computations must agree."""
    xg, wg = np.polynomial.legendre.leggauss(8)
    edges = np.linspace(0.0, T, n_gauss + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        for xi, wi in zip(xg, wg):
            u = control(mid + half * xi)
            total += wi * half * float(u @ u)
    return total


def grade(
    case: Case,
    thresholds: Thresholds,
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    phases: list[dict],
    nuf: float,
    objective_solver: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    n_dense: int = 4000,
) -> GradeResult:
    """Judge one returned solution. Never consults the solver's own flags."""
    res = GradeResult(objective_solver=float(objective_solver))
    res.thrust_cap_km_s2 = case.u_max
    res.altitude_floor_km = case.h_min

    try:
        control = PhasewiseControl(np.asarray(t), np.asarray(u), phases)
    except ValueError as exc:
        res.reasons.append(f"control reconstruction failed: {exc}")
        return res

    T = float(t[-1] - t[0])
    sol, dense, t_dense = _propagate(
        np.asarray(x)[:, 0], control, T, rtol, atol, n_dense
    )
    res.n_eval = int(sol.nfev)
    if dense is None:
        res.reasons.append(f"propagation failed: {sol.message}")
        return res
    res.propagation_ok = True

    # Terminal boundary condition, against the target orbit at the returned nuf.
    cfg = case.to_transfer_config()
    rf, vf = oe_to_rv((cfg.af, cfg.ef, cfg.if_, 0.0, 0.0, float(nuf)), MU_EARTH)
    res.pos_err_km = float(np.linalg.norm(dense[:3, -1] - rf))
    res.vel_err_km_s = float(np.linalg.norm(dense[3:, -1] - vf))

    # Altitude and thrust, over the continuous path -- not only at the nodes,
    # which is the only place the transcription enforces them.
    res.min_altitude_km = float(np.min(np.linalg.norm(dense[:3], axis=0)) - R_E)
    u_dense = control.sample(t_dense)
    res.max_thrust_km_s2 = float(np.max(np.linalg.norm(u_dense, axis=0)))

    res.objective_grader = _objective(control, T)
    denom = max(abs(res.objective_solver), 1e-30)
    res.cost_recon_rel_err = abs(res.objective_grader - res.objective_solver) / denom

    if res.pos_err_km > thresholds.pos_err_km:
        res.reasons.append(
            f"terminal position error {res.pos_err_km:.3e} km "
            f"> {thresholds.pos_err_km:.3e}")
    if res.vel_err_km_s > thresholds.vel_err_km_s:
        res.reasons.append(
            f"terminal velocity error {res.vel_err_km_s:.3e} km/s "
            f"> {thresholds.vel_err_km_s:.3e}")
    if res.min_altitude_km < case.h_min - thresholds.altitude_slack_km:
        res.reasons.append(
            f"minimum altitude {res.min_altitude_km:.3f} km "
            f"< floor {case.h_min:.1f} km")
    cap = case.u_max * (1.0 + thresholds.thrust_rel_slack)
    if res.max_thrust_km_s2 > cap:
        res.reasons.append(
            f"maximum thrust {res.max_thrust_km_s2:.6e} km/s^2 > cap {cap:.6e}")
    if res.cost_recon_rel_err > thresholds.cost_recon_rel_tol:
        res.reasons.append(
            f"grader could not reproduce the solver objective "
            f"(relative error {res.cost_recon_rel_err:.3e}); verdict void")

    res.verdict = not res.reasons
    return res


# --------------------------------------------------------------------------
# Checks on the oracle itself
# --------------------------------------------------------------------------

def check_resolution(
    case: Case, thresholds: Thresholds, t, x, u, phases, nuf, objective_solver,
    coarse_rtol: float = 1e-10, coarse_atol: float = 1e-12,
    coarse_dense: int = 4000,
) -> tuple[bool, dict]:
    """Re-grade at tightened tolerance and doubled sampling. If the measured
    quantities move enough to put a verdict in doubt, the grader is not
    converged and its verdict is void.

    Each quantity is compared against a scale that includes the threshold it
    will actually be judged by. A purely relative comparison is wrong here: the
    terminal errors approach zero on the *best* solutions, so their relative
    reproducibility becomes dominated by the integrator's own noise floor, and
    the check would fire hardest exactly where the solution is most accurate --
    backwards. What matters is whether the movement is large compared with the
    decision threshold, because only then could it change a verdict.

    Fails when: a measured quantity shifts by more than
    `thresholds.resolution_rel_tol` of its own decision threshold, or when the
    two resolutions disagree about the verdict.
    """
    base = grade(case, thresholds, t, x, u, phases, nuf, objective_solver,
                 rtol=coarse_rtol, atol=coarse_atol, n_dense=coarse_dense)
    fine = grade(case, thresholds, t, x, u, phases, nuf, objective_solver,
                 rtol=1e-12, atol=1e-14, n_dense=8000)
    if not (base.propagation_ok and fine.propagation_ok):
        return False, {"error": "propagation failed in the resolution check"}

    # The scale each quantity is judged on, so "converged enough" means
    # "converged enough to decide", not "converged to full relative precision".
    scales = {
        "pos_err_km": thresholds.pos_err_km,
        "vel_err_km_s": thresholds.vel_err_km_s,
        "min_altitude_km": thresholds.altitude_slack_km,
        "max_thrust_km_s2": case.u_max * thresholds.thrust_rel_slack,
        "objective_grader": abs(objective_solver),
    }
    deltas = {}
    for name, scale in scales.items():
        a, b = getattr(base, name), getattr(fine, name)
        deltas[name] = abs(a - b) / max(abs(a), abs(b), scale, 1e-30)
    ok = max(deltas.values()) <= thresholds.resolution_rel_tol
    deltas["verdict_stable"] = base.verdict == fine.verdict
    return bool(ok and deltas["verdict_stable"]), deltas


def check_cost_reconstruction(result: GradeResult, thresholds: Thresholds) -> bool:
    """The grader's own objective must match the solver's.

    Fails when: the two independent computations of the same integral differ by
    more than `thresholds.cost_recon_rel_tol`.
    """
    return bool(result.cost_recon_rel_err <= thresholds.cost_recon_rel_tol)


def check_corruption(
    case: Case, thresholds: Thresholds, t, x, u, phases, nuf, objective_solver,
    shift_km: float = 50.0,
) -> tuple[bool, dict]:
    """Feed the grader a solution broken by a known amount and require rejection.

    The initial position is displaced by `shift_km`, which is far beyond the
    terminal position threshold. A grader that still accepts is not a gate.

    Fails when: the corrupted solution is accepted.
    """
    x_bad = np.array(x, dtype=float, copy=True)
    x_bad[0, 0] += shift_km
    bad = grade(case, thresholds, t, x_bad, u, phases, nuf, objective_solver)
    return (not bad.verdict), {
        "shift_km": shift_km,
        "corrupted_pos_err_km": bad.pos_err_km,
        "corrupted_verdict": bad.verdict,
        "reasons": bad.reasons,
    }


def bridge_residual(
    x: np.ndarray, u: np.ndarray, accel_geom: np.ndarray
) -> float:
    """The zero-residual invariant on the Bezier bridge (design section 5).

    The bridge builds its control as `u = a_geom - g(r)`, where `a_geom` is the
    curve's second derivative divided by the squared transfer time and `g` is
    the baseline's gravity. Adding gravity back through *these* equations of
    motion must return `a_geom` exactly -- the same two terms, subtracted then
    added. So this quantity is at machine precision if and only if the bridge
    and the grader use the identical gravity model.

    Deliberately not a finite-difference check: a finite difference reports its
    own truncation error, which is orders of magnitude larger than the model
    disagreement it would be trying to detect, so it could not fail for the
    reason that matters.

    Fails when: the returned relative residual is not at machine precision,
    which means the bridge and the grader disagree about the model and every
    measurement downstream of the bridge is suspect.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    accel_geom = np.asarray(accel_geom, dtype=float)
    worst = 0.0
    for k in range(x.shape[1]):
        xdot = spacecraft_eom_numpy(x[:, k], u[:, k], mu=MU_EARTH,
                                    include_j2=True)
        scale = max(np.linalg.norm(accel_geom[:, k]), 1e-30)
        worst = max(worst,
                    float(np.linalg.norm(xdot[3:] - accel_geom[:, k]) / scale))
    return worst


def self_test(verbose: bool = True) -> bool:
    """Checks that need no solver output: an analytically known trajectory must
    be accepted, and the same trajectory broken by a known amount must be
    rejected.

    The accepted case is a pure Keplerian arc with zero control, integrated by
    the same routine, so the grader is being asked to confirm something that
    cannot be false if it is correct. The rejected case is that same arc with a
    displaced start.
    """
    case = Case(case_id="SELFTEST", stratum="X", h0=400.0, delta_a=0.0,
                delta_i=0.0, T_normed=0.5, e0=0.0, ef=0.0)
    cfg = case.to_transfer_config()
    T = cfg.T_max

    # A coasting arc: zero control, so the true motion is the model's own
    # free-drift solution and the grader must reproduce it. Node count matches a
    # real coast phase; a grader fed a hundreds-of-nodes single phase would be
    # asked to do something the transcription never produces.
    #
    # The arc is built backwards from the target, not forwards to it. The
    # baseline pins the ascending node and the argument of perigee to zero, so a
    # state obtained by forward propagation is generally *not* expressible in
    # that parametrization and a round-trip through `rv_to_oe` would silently
    # lose the argument of perigee. Starting from a target the parametrization
    # can express removes that failure mode from the test.
    nuf = 1.0
    rf, vf = oe_to_rv((cfg.af, cfg.ef, cfg.if_, 0.0, 0.0, nuf), MU_EARTH)

    back = solve_ivp(
        lambda tq, xq: spacecraft_eom_numpy(xq, np.zeros(3), mu=MU_EARTH,
                                            include_j2=True),
        (T, 0.0), np.concatenate([rf, vf]),
        method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True,
    )
    if not back.success:
        if verbose:
            print("FAIL: self-test backward propagation did not complete")
        return False

    t_nodes = np.linspace(0.0, T, 12)
    dense = back.sol(t_nodes)

    phases = [{"t_start": 0.0, "t_end": T, "n_nodes": len(t_nodes),
               "type": "coast"}]
    u_zero = np.zeros((3, len(t_nodes)))
    th = Thresholds()
    good = grade(case, th, t_nodes, dense, u_zero, phases, nuf, 0.0)

    # A zero objective makes the relative cost check degenerate; it is exercised
    # on real solutions instead.
    good_ok = good.propagation_ok and good.pos_err_km < th.pos_err_km

    bad_ok, bad_info = check_corruption(
        case, th, t_nodes, dense, u_zero, phases, nuf, 0.0)

    # The invariant, on a case where it must hold exactly: with zero control the
    # geometric acceleration is gravity itself.
    accel_geom = np.column_stack([
        spacecraft_eom_numpy(dense[:, k], np.zeros(3), mu=MU_EARTH,
                             include_j2=True)[3:]
        for k in range(dense.shape[1])
    ])
    resid = bridge_residual(dense, u_zero, accel_geom)

    # The resolution check must pass at the resolution the experiment uses...
    res_ok, res_info = check_resolution(
        case, th, t_nodes, dense, u_zero, phases, nuf, 0.0)
    # ...and must FAIL when the coarse side is deliberately crippled. Without
    # this second half the check is untested: a function that always returns
    # True would pass the first half.
    res_fires, fire_info = check_resolution(
        case, th, t_nodes, dense, u_zero, phases, nuf, 0.0,
        coarse_rtol=1e-2, coarse_atol=1e-2, coarse_dense=40)
    res_fires = not res_fires

    if verbose:
        print(f"  exact coast accepted : {good_ok} "
              f"(position error {good.pos_err_km:.3e} km)")
        print(f"  corrupted rejected   : {bad_ok} "
              f"(position error {bad_info['corrupted_pos_err_km']:.3e} km)")
        print(f"  bridge residual      : {resid:.3e} (relative)")
        print(f"  resolution check ok  : {res_ok} "
              f"(worst shift {max(v for v in res_info.values() if isinstance(v, float)):.3e} "
              f"of threshold)")
        print(f"  ...and it can fail   : {res_fires} "
              f"(crippled integrator shift "
              f"{max(v for v in fire_info.values() if isinstance(v, float)):.3e})")
    return bool(good_ok and bad_ok and res_ok and res_fires and resid < 1e-12)
