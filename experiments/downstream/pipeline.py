"""The four arms, driven stage by stage.

Design reference: `doc/dcm_downstream_experiment_design.md` sections 4.2-4.5.

Why this module does not call `TwoPassOptimizer`
------------------------------------------------
`dcm_baseline/` is not modified by this experiment. But design section 4.4
requires the primary runs to have no true-anomaly retries, no tolerance
relaxation, and no Pass-1-as-result fallback, and all three of those live inside
`TwoPassOptimizer.solve`. Driving `HermiteSimpsonCollocation` and
`MultiPhaseLGLCollocation` directly gives exact protocol control while leaving
the baseline source untouched. The stage sequence here is otherwise identical to
`two_pass.py`: solve, detect peaks, determine phases, interpolate, solve again.

The recovery ladder is available through `RunConfig.recovery`, which is what the
secondary "as shipped" run of design section 4.4 uses -- symmetrically, for
every arm.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import comb

from orbit_transfer.astrodynamics.orbital_elements import oe_to_rv
from orbit_transfer.classification.classifier import (
    classify_profile, determine_phase_structure,
)
from orbit_transfer.classification.peak_detection import detect_peaks
from orbit_transfer.collocation.hermite_simpson import HermiteSimpsonCollocation
from orbit_transfer.collocation.interpolation import interpolate_pass1_to_pass2
from orbit_transfer.collocation.multiphase_lgl import MultiPhaseLGLCollocation
from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.dynamics.two_body import gravity_acceleration
from orbit_transfer.dynamics.j2_perturbation import j2_acceleration
from orbit_transfer.optimizer.initial_guess import linear_interpolation_guess

from .cases import Case, Thresholds
from . import grader as G

ARMS = ("BASE", "PROP", "MIX-S", "MIX-W")

# Display-only solver options, merged into the baseline's own IPOPT settings.
# `print_time` suppresses CasADi's timing table and `sb` IPOPT's banner; neither
# touches the numerics. They are passed rather than redirecting file descriptors
# because IPOPT writes from C and flushes after a Python-level restore.
QUIET_SOLVER_OPTIONS = {"print_time": False, "ipopt.sb": "yes"}

# Outcome taxonomy (design section 4.5). Exactly one applies to every run.
STAGE1_FAIL = "STAGE1_FAIL"
STAGE2_FAIL = "STAGE2_FAIL"
SUCCESS_UNVERIFIED = "SUCCESS_UNVERIFIED"
SUCCESS_VERIFIED = "SUCCESS_VERIFIED"


@dataclass(frozen=True)
class BezierConfig:
    """The upstream stage under test, as frozen in `doc/design_freeze.md` section 5.

    Not tuned for this experiment. Recorded per run so any later change is
    visible in the artifact rather than inferred.
    """

    degree: int = 7
    n_seg: int = 16
    max_iter: int = 40
    tol: float = 1e-8
    elastic_weight: float = 1e-2
    scp_trust_radius: float = 2000.0
    sample_count: int = 100


@dataclass(frozen=True)
class RunConfig:
    """Protocol switches. The primary comparison runs with the defaults; every
    switch applies to all four arms or to none (design P4)."""

    recovery: bool = False       # true-anomaly retries + tolerance relaxation
    n_profile_points: int = 61   # Pass-1 grid, so both stage-1 profiles are
                                 # peak-detected on identical sampling
    grade: bool = True
    self_checks: bool = False    # per-case resolution and corruption checks


# --------------------------------------------------------------------------
# Gravity, in the baseline's model
# --------------------------------------------------------------------------

def dcm_gravity(r: np.ndarray) -> np.ndarray:
    """Gravitational acceleration in the *baseline's* model, which is the model
    the downstream problem is posed in.

    The Bezier optimizer's internal gravity uses a different Earth radius
    (6371.0 km, mean) from the baseline's (6378.137 km, equatorial), so the two
    J2 terms differ by about 0.2 percent of J2. The bridge deliberately uses
    this function rather than the Bezier package's, because the control it emits
    is about to be handed to the baseline. `bezier_stage` measures and reports
    the resulting discrepancy instead of hiding it.
    """
    return gravity_acceleration(r, MU_EARTH) + j2_acceleration(r, mu=MU_EARTH)


# --------------------------------------------------------------------------
# Stage 1a: the baseline's own Hermite-Simpson pass
# --------------------------------------------------------------------------

def run_pass1(cfg, recovery: bool = False):
    """Pass 1, with the recovery ladder off by default.

    Mirrors `two_pass.py` lines 57-122 exactly, except that the retry and
    relaxation blocks are conditional instead of unconditional.
    """
    from orbit_transfer.config import MAX_NU_RETRIES, TOL_RELAXATION_FACTOR

    hs = HermiteSimpsonCollocation(cfg)
    t_g, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(cfg, hs.N_points)
    result = hs.solve(x_guess=x_g, u_guess=u_g,
                      nu0_guess=nu0_g, nuf_guess=nuf_g,
                      ipopt_options=dict(QUIET_SOLVER_OPTIONS))
    attempts = 1

    if recovery and not result.converged:
        rng = np.random.default_rng(42)
        for _ in range(MAX_NU_RETRIES):
            nu0_r, nuf_r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
            t_g, x_g, u_g, _, _ = linear_interpolation_guess(cfg, hs.N_points)
            result = hs.solve(x_guess=x_g, u_guess=u_g,
                              nu0_guess=nu0_r, nuf_guess=nuf_r,
                              ipopt_options=dict(QUIET_SOLVER_OPTIONS))
            attempts += 1
            if result.converged:
                break
        if not result.converged:
            t_g, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(
                cfg, hs.N_points)
            result = hs.solve(
                x_guess=x_g, u_guess=u_g, nu0_guess=nu0_g, nuf_guess=nuf_g,
                ipopt_options={
                    **QUIET_SOLVER_OPTIONS,
                    'ipopt.tol': 1e-4 * TOL_RELAXATION_FACTOR,
                    'ipopt.constr_viol_tol': 1e-4 * TOL_RELAXATION_FACTOR,
                },
            )
            attempts += 1

    return result, attempts


# --------------------------------------------------------------------------
# Stage 1b: the Bezier stage, and the bridge into the baseline's model
# --------------------------------------------------------------------------

def _bernstein_matrix(tau: np.ndarray, degree: int) -> np.ndarray:
    i = np.arange(degree + 1)
    return (comb(degree, i)[None, :]
            * tau[:, None] ** i[None, :]
            * (1.0 - tau)[:, None] ** (degree - i)[None, :])


def _fit_control_points(r_ref: np.ndarray, tau: np.ndarray, degree: int):
    """Least-squares Bezier control points through a reference position history.

    Why not the package's straight-line initialiser: the baseline pins the
    departure and arrival true anomalies to 0 and pi, so the two endpoints are
    antipodal and the straight line between them passes through the centre of
    the Earth -- through the middle of the keep-out sphere, whose supporting
    half-spaces are degenerate exactly there. Seeding instead from the Keplerian
    blend that the baseline's own Pass 1 is seeded with is both non-degenerate
    and symmetric: the two stage-1 solvers start from the same trajectory.
    """
    B = _bernstein_matrix(tau, degree)
    P, *_ = np.linalg.lstsq(B, r_ref.T, rcond=None)
    P[0] = r_ref[:, 0]
    P[-1] = r_ref[:, -1]
    return P


def bezier_stage(case: Case, bez: BezierConfig, n_points: int = 61):
    """Solve the Bezier SCP problem and translate it into the baseline's terms.

    Returns `(profile, info)` where `profile` carries `t`, `x`, `u` on the
    Pass-1 grid, or `(None, info)` if the stage produced nothing usable.
    """
    from orbital_docking.optimization import optimize_orbital_docking
    from orbital_docking.bezier import BezierCurve

    cfg = case.to_transfer_config()
    T = cfg.T_max

    # Endpoints in the baseline's own convention: nu0 = 0, nuf = pi, ascending
    # node and argument of perigee pinned to zero. Pass 2 frees both anomalies
    # again for every arm, so fixing them here is a stage-1 property, not an
    # advantage or a handicap (design 4.3).
    r0, v0 = oe_to_rv((cfg.a0, cfg.e0, cfg.i0, 0.0, 0.0, 0.0), MU_EARTH)
    rf, vf = oe_to_rv((cfg.af, cfg.ef, cfg.if_, 0.0, 0.0, np.pi), MU_EARTH)

    t_ref, x_ref, _, _, _ = linear_interpolation_guess(cfg, n_points)
    tau = t_ref / T
    P_init = _fit_control_points(x_ref[:3], tau, bez.degree)

    info: dict = {"bezier_config": asdict(bez)}
    t_start = time.perf_counter()
    try:
        P_opt, bez_info = optimize_orbital_docking(
            P_init,
            n_seg=bez.n_seg,
            r_e=R_E + case.h_min,      # the keep-out sphere IS the altitude floor
            max_iter=bez.max_iter,
            tol=bez.tol,
            v0=v0, v1=vf,
            sample_count=bez.sample_count,
            scp_trust_radius=bez.scp_trust_radius,
            elastic_weight=bez.elastic_weight,
            transfer_time=T,
            verbose=False,
            use_cache=False,           # design 7: caching off for every timed run
        )
    except Exception as exc:                                  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["wall_s"] = time.perf_counter() - t_start
        return None, info
    info["wall_s"] = time.perf_counter() - t_start
    info.update({k: bez_info.get(k) for k in (
        "feasible", "curve_clears_koz", "solve_slack_vanished", "iterations",
        "termination_reason", "min_radius", "cost", "bc_v0_rel_err",
        "bc_v1_rel_err",
    )})

    if not bez_info.get("feasible", False):
        info["stage1_reject"] = (
            f"Bezier stage not feasible: {bez_info.get('termination_reason')}")
        return None, info

    # --- the bridge ---
    curve = BezierCurve(P_opt)
    r = np.column_stack([curve.point(s) for s in tau])
    v = np.column_stack([curve.velocity(s) for s in tau]) / T
    a_geom = np.column_stack([curve.acceleration(s) for s in tau]) / T ** 2

    g_dcm = np.column_stack([dcm_gravity(r[:, k]) for k in range(r.shape[1])])
    u = a_geom - g_dcm

    # The zero-residual invariant (design 5, second self-check). Adding the
    # baseline's gravity back to this control must return a_geom exactly.
    x = np.vstack([r, v])
    info["bridge_residual"] = G.bridge_residual(x, u, a_geom)

    # The model gap the bridge cannot remove: the Bezier optimizer minimised
    # against its own gravity constants. Measured, not assumed negligible.
    from orbital_docking.constants import (
        EARTH_J2, EARTH_MU_SCALED, EARTH_RADIUS_KM,
    )
    from orbital_docking.optimization import _accel_total
    g_bez = np.column_stack([
        _accel_total(r[:, k], EARTH_MU_SCALED, EARTH_RADIUS_KM, EARTH_J2)
        for k in range(r.shape[1])
    ])
    scale = max(float(np.max(np.linalg.norm(u, axis=0))), 1e-30)
    info["gravity_model_gap_rel"] = float(
        np.max(np.linalg.norm(g_dcm - g_bez, axis=0)) / scale)

    return {"t": t_ref, "x": x, "u": u, "T_f": T, "nu0": 0.0, "nuf": np.pi}, info


# --------------------------------------------------------------------------
# Structure and Pass 2
# --------------------------------------------------------------------------

def structure_from(t, u, T):
    """Peak detection and phase structure, identical for both stage-1 sources."""
    u_mag = np.linalg.norm(u, axis=0)
    n_peaks, peak_times, peak_widths = detect_peaks(t, u_mag, T)
    phases = determine_phase_structure(peak_times, peak_widths, T)
    return n_peaks, classify_profile(n_peaks), phases


def run_pass2(cfg, phases, T_fixed, warm_t, warm_x, warm_u, nu0, nuf):
    """Pass 2, called directly so no fallback can substitute a Pass-1 result."""
    t_ph, x_ph, u_ph = interpolate_pass1_to_pass2(warm_t, warm_x, warm_u, phases)
    lgl = MultiPhaseLGLCollocation(cfg, phases, T_fixed=T_fixed)
    return lgl.solve(x_phases=x_ph, u_phases=u_ph,
                     nu0_guess=nu0, nuf_guess=nuf,
                     ipopt_options=dict(QUIET_SOLVER_OPTIONS))


# --------------------------------------------------------------------------
# The arms
# --------------------------------------------------------------------------

@dataclass
class ArmResult:
    case_id: str = ""
    stratum: str = ""
    arm: str = ""
    outcome: str = STAGE1_FAIL
    note: str = ""

    # structure
    n_peaks_stage1: int = -1
    profile_class: int = -1
    n_phases: int = -1
    phase_boundaries: list = field(default_factory=list)
    node_allocation: list = field(default_factory=list)

    # horizon
    horizon_s: float = float("nan")
    pass1_T_f: float = float("nan")
    horizon_left_bound: bool = False

    # objective
    objective_solver: float = float("nan")
    pass1_cost: float = float("nan")
    return_status: str = ""
    solve_succeeded: bool = False
    converged_flag: bool = False

    # grader
    grade: dict = field(default_factory=dict)
    resolution_check: dict = field(default_factory=dict)
    corruption_check: dict = field(default_factory=dict)

    # timing and provenance
    t_stage1_s: float = float("nan")
    t_structure_s: float = float("nan")
    t_pass2_s: float = float("nan")
    t_total_s: float = float("nan")
    stage1_attempts: int = 0
    bezier_info: dict = field(default_factory=dict)


def _stage1(source: str, case: Case, cfg, bez: BezierConfig, run: RunConfig):
    """Run one stage-1 source. Returns `(profile_or_None, meta)`."""
    t0 = time.perf_counter()
    if source == "pass1":
        result, attempts = run_pass1(cfg, recovery=run.recovery)
        meta = {"wall_s": time.perf_counter() - t0, "attempts": attempts,
                "T_f": float(result.T_f), "cost": float(result.cost),
                "bezier_info": {}}
        if not result.converged:
            meta["reject"] = "Pass 1 did not converge"
            return None, meta
        return ({"t": result.t, "x": result.x, "u": result.u,
                 "T_f": float(result.T_f), "nu0": result.nu0,
                 "nuf": result.nuf}, meta)

    profile, info = bezier_stage(case, bez, run.n_profile_points)
    meta = {"wall_s": time.perf_counter() - t0, "attempts": 1,
            "T_f": cfg.T_max, "cost": float(info.get("cost") or float("nan")),
            "bezier_info": info}
    if profile is None:
        meta["reject"] = info.get("stage1_reject") or info.get(
            "error", "Bezier stage produced no usable profile")
    return profile, meta


def run_arm(
    arm: str,
    case: Case,
    thresholds: Thresholds,
    bez: BezierConfig | None = None,
    run: RunConfig | None = None,
) -> ArmResult:
    """One (case, arm) run, start to graded verdict."""
    bez = bez or BezierConfig()
    run = run or RunConfig()
    cfg = case.to_transfer_config()
    res = ArmResult(case_id=case.case_id, stratum=case.stratum, arm=arm)
    t_begin = time.perf_counter()

    structure_src = {"BASE": "pass1", "PROP": "bezier",
                     "MIX-S": "bezier", "MIX-W": "pass1"}[arm]
    warm_src = {"BASE": "pass1", "PROP": "bezier",
                "MIX-S": "pass1", "MIX-W": "bezier"}[arm]

    profiles: dict = {}
    stage1_wall = 0.0
    for src in dict.fromkeys((structure_src, warm_src)):
        prof, meta = _stage1(src, case, cfg, bez, run)
        stage1_wall += meta["wall_s"]
        res.stage1_attempts += meta["attempts"]
        if src == "pass1":
            res.pass1_T_f = meta["T_f"]
            res.pass1_cost = meta["cost"]
        else:
            res.bezier_info = meta["bezier_info"]
        if prof is None:
            res.outcome = STAGE1_FAIL
            res.note = f"{src}: {meta['reject']}"
            res.t_stage1_s = stage1_wall
            res.t_total_s = time.perf_counter() - t_begin
            return res
        profiles[src] = prof
    res.t_stage1_s = stage1_wall

    # Matched horizon: both arms run Pass 2 on T_max. Whether the free horizon
    # actually sat at its bound is recorded per case rather than assumed
    # (design 4.3).
    horizon = cfg.T_max
    if "pass1" in profiles:
        rel = abs(res.pass1_T_f - cfg.T_max) / max(cfg.T_max, 1e-12)
        res.horizon_left_bound = bool(rel > 1e-3)
    res.horizon_s = horizon

    # Structure from one source, warm start from the other. Both are rescaled
    # onto the matched horizon before use.
    t1 = time.perf_counter()
    sp = profiles[structure_src]
    t_struct = sp["t"] * (horizon / max(sp["T_f"], 1e-12))
    res.n_peaks_stage1, res.profile_class, phases = structure_from(
        t_struct, sp["u"], horizon)
    res.t_structure_s = time.perf_counter() - t1
    res.n_phases = len(phases)
    res.phase_boundaries = [[float(p["t_start"]), float(p["t_end"])]
                            for p in phases]
    res.node_allocation = [int(p["n_nodes"]) for p in phases]

    wp = profiles[warm_src]
    t_warm = wp["t"] * (horizon / max(wp["T_f"], 1e-12))

    t2 = time.perf_counter()
    try:
        r2 = run_pass2(cfg, phases, horizon, t_warm, wp["x"], wp["u"],
                       wp["nu0"], wp["nuf"])
    except Exception as exc:                                  # noqa: BLE001
        res.outcome = STAGE2_FAIL
        res.note = f"Pass 2 raised {type(exc).__name__}: {exc}"
        res.t_pass2_s = time.perf_counter() - t2
        res.t_total_s = time.perf_counter() - t_begin
        return res
    res.t_pass2_s = time.perf_counter() - t2

    stats = r2.solver_stats or {}
    res.return_status = str(stats.get("return_status", "unknown"))
    res.solve_succeeded = bool(stats.get("solve_succeeded", False))
    res.converged_flag = bool(r2.converged)
    res.objective_solver = float(r2.cost)

    if not r2.converged:
        res.outcome = STAGE2_FAIL
        res.note = f"Pass 2 did not converge ({res.return_status})"
        res.t_total_s = time.perf_counter() - t_begin
        return res

    if not run.grade:
        res.outcome = SUCCESS_UNVERIFIED
        res.note = "grading disabled"
        res.t_total_s = time.perf_counter() - t_begin
        return res

    g = G.grade(case, thresholds, r2.t, r2.x, r2.u, phases, r2.nuf, r2.cost)
    res.grade = {k: v for k, v in asdict(g).items()}
    res.outcome = SUCCESS_VERIFIED if g.verdict else SUCCESS_UNVERIFIED
    if not g.verdict:
        res.note = "; ".join(g.reasons)

    if run.self_checks:
        ok, deltas = G.check_resolution(case, thresholds, r2.t, r2.x, r2.u,
                                        phases, r2.nuf, r2.cost)
        res.resolution_check = {"passed": bool(ok), **deltas}
        if not ok:
            res.outcome = SUCCESS_UNVERIFIED
            res.note = ("; ".join(filter(None, [
                res.note, "grader not resolution-converged; verdict void"])))
        ok_c, info_c = G.check_corruption(case, thresholds, r2.t, r2.x, r2.u,
                                          phases, r2.nuf, r2.cost)
        res.corruption_check = {"passed": bool(ok_c), **info_c}
        if not ok_c:
            res.outcome = SUCCESS_UNVERIFIED
            res.note = "; ".join(filter(None, [
                res.note, "grader accepted a corrupted solution; verdict void"]))

    res.t_total_s = time.perf_counter() - t_begin
    return res
