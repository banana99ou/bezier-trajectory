"""
Optimization entrypoints for space-time Bezier trajectories.

Public API only: request normalization, backend dispatch, batch orchestration.
Debugger factories live in ``rust_debug_stepper``; clearance computation and
obstacle geometry live in ``geometry``.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import compute_min_clearance, obstacle_array_bundle
from .objective import build_initial_guess

# Mirrors DEFAULT_TRUST_RADIUS in rust_optimizer/core/src/spacetime_optimizer.rs.
DEFAULT_TRUST_RADIUS = 0.5

# Weight on the elastic (virtual-control) slack in the QP subproblem. Mirrors the
# pybind default in rust_optimizer/pybind/src/lib.rs.
#
# This is an EXACT-PENALTY weight: above a problem-dependent threshold the
# penalized solution solves the original constrained problem, below it violating
# a KOZ is simply cheaper than obeying it, and the solver returns a penetrating
# curve that looks converged. It was hard-wired at 100.0 and reachable only by
# calling the Rust binding directly, so no scenario could be tuned through this
# module -- which is how `wall` and `diverse` came to be recorded as infeasible.
# They are not: both clear and certify at a higher weight. Per-scenario values
# live in scenarios.py; see SCENARIO_MAP.
DEFAULT_ELASTIC_WEIGHT = 100.0

# Penalty continuation ladder. `optimize_scenario` walks this in order and stops
# at the first weight whose run is converged AND certified AND clearing.
#
# Escalating an exact penalty until the constraint violation vanishes is the
# standard remedy when the threshold is unknown, and the threshold here IS
# unknown -- it depends on the optimal multipliers, which differ per scenario and
# per segment count. Fixing one weight instead is what made `wall` and `diverse`
# look infeasible: at 100 a penetrating curve is simply cheaper than a clear one,
# so the solver correctly returned a penetrating curve for the problem it was
# actually given. Measured thresholds: `diverse` certifies from 800 (N8_seg4,
# 4 segments) to 10000 (8 and 16 segments); `original` certifies at 100.
#
# The weight that succeeded is recorded per config as `elastic_weight`, so a
# number in the table can always be traced to the penalty that produced it.
ELASTIC_WEIGHT_LADDER = (100.0, 300.0, 800.0, 3000.0, 1e4, 1e5)

try:
    import bezier_opt as _bezier_opt_rs
except ImportError:  # pragma: no cover - exercised when the native extension is unavailable.
    _bezier_opt_rs = None


# ---------------------------------------------------------------------------
# Feasibility gate (item B7)
#
# "A run with total_slack > 0 cannot produce a figure." The four conditions are
# independent and none may stand in for another:
#
#   converged     -- the loop stopped for a principled reason, not the cap.
#   certificate   -- the control-point hull satisfies the half-spaces it
#                    generates, rebuilt at the RETURNED iterate. This is the
#                    property the paper claims; clearance is not.
#   clearance     -- the sampled curve misses the TRUE obstacle trajectories.
#                    Re-verified in Python against the obstacles themselves, not
#                    against the hulls the solver used, which is the condition
#                    PAPER_1 sec. 6 requires because an adaptively-built hull is
#                    valid only inside the time span it was built for.
#   total_slack   -- the elastic relaxation bought nothing. Every subproblem is
#                    solved elastically, so a converged, certified, clearing run
#                    can still have been standing on slack; this is the condition
#                    that catches it.
#
# NaN fails every comparison, so a run that never accepted a step (slack unknown)
# is not figure-grade. That is the intended answer for "no evidence".
# ---------------------------------------------------------------------------

FIGURE_GRADE_CERTIFICATE_TOL = 1e-6

# Measured 2026-08-20, not chosen. Clarabel is an interior-point method, so the
# slack variables approach zero from above and never reach it, and `total_slack`
# is a SUM over rows -- 108 rows on `original`, 4400 on `wall` N10_seg16 -- so
# the residue also grows with problem size. At the recorded configurations:
#
#     wall   N10_seg16   2.4e-14        original N8_seg4   1.3e-10
#     wall3d N8_seg2     3.8e-11        diverse  N8_seg4   1.4e-09
#
# All four are converged and carry a hull certificate of 0.0 to 2.2e-10 at the
# returned iterate, so none of them is standing on relaxation; the numbers are
# convergence noise. A threshold at 1e-10 would reject `original` and `diverse`
# -- the two runs the gate exists to pass -- so it would be measuring the solver
# and not the trajectory.
#
# 1e-8 sits two orders above the worst observed residue and six or more below any
# relaxation that could hide a violation: a genuine one is of the order of the
# clearance itself, 0.01 to 1. Anything in between separates them, and the
# guarantee is carried by the certificate above, which is evaluated against the
# EXACT rows rebuilt at the returned iterate.
FIGURE_GRADE_SLACK_TOL = 1e-8


def figure_grade_failures(row: dict) -> list[str]:
    """Every reason ``row`` is not figure-grade. Empty list means it is."""
    reasons = []
    if not bool(row.get("converged", False)):
        reasons.append(f"not converged ({row.get('stop_label', 'unknown')})")
    certificate = float(row.get("certificate_violation", float("nan")))
    if not certificate <= FIGURE_GRADE_CERTIFICATE_TOL:
        reasons.append(f"hull certificate violated by {certificate:.3e}")
    # Occlusion (item B12) is a separate guarantee and gets a separate gate
    # condition: the keep-out certificate says nothing about line of sight, so it
    # cannot stand in for this. Defaults to 0.0, which is what a run with no
    # station genuinely has -- zero occlusion rows cannot be violated.
    occlusion = float(row.get("occlusion_violation", 0.0))
    if not occlusion <= FIGURE_GRADE_CERTIFICATE_TOL:
        reasons.append(f"line of sight lost, occlusion certificate {occlusion:.3e}")
    # A dropped occlusion plane is NOT a satisfied one. When the station sits
    # inside a piece's inflated body no supporting half-space exists, so the
    # builder emits no row -- and a certificate summed over the rows that do
    # exist then reads 0.0 for a trajectory whose true line of sight is gone.
    # Measured: a straight flight past a fast occluder came back converged,
    # certified 0.0 and figure-grade with a true sight margin of -0.3999.
    # Defaults to 0.0 for the same reason `occlusion_violation` does: a pre-B12
    # row has no stations, so it has nothing to drop.
    dropped = float(row.get("occlusion_planes_dropped", 0.0))
    if not dropped <= 0.0:
        reasons.append(
            f"{dropped:.0f} occlusion plane(s) could not be built; line of sight "
            "is uncertifiable here, not certified"
        )
    clearance = float(row.get("min_clearance", float("nan")))
    if not clearance > 0.0:
        reasons.append(f"penetrates by {-clearance:.3e}")
    slack = float(row.get("total_slack", float("nan")))
    if not slack <= FIGURE_GRADE_SLACK_TOL:
        reasons.append(f"elastic slack {slack:.3e} > {FIGURE_GRADE_SLACK_TOL:g}")
    return reasons


def is_figure_grade(row: dict) -> bool:
    """True only when every gate condition holds. See ``figure_grade_failures``."""
    return not figure_grade_failures(row)


class UncappedTimePenaltyError(ValueError):
    """A time penalty was requested with no speed cap to hold it back."""


def check_time_penalty_is_capped(v_max, time_weight) -> None:
    """Refuse the one configuration that can only produce an artifact.

    Formulation decision 5 and PAPER_1 sec. "Arrival time is linear": a linear
    penalty on the arrival time with **no** speed cap has nothing opposing it.
    The smoothness regularizer is blind to timing (item B8), and the only
    remaining floor on arrival is the time-monotonicity minimum separation, so
    the optimum collapses to ``t_start + min_dt * (number of control-point
    gaps)`` -- roughly 1.0 s at the shipped settings -- *for every scenario*,
    independent of geometry. A run reporting that has measured the constraint
    set, not the problem.

    That is why item B9 and item B10 land together. Raising here is deliberate:
    a warning is something a batch sweep discards, and the resulting numbers are
    indistinguishable from real ones once they reach a table.
    """
    if float(time_weight) > 0.0 and (v_max is None or not float(v_max) > 0.0):
        raise UncappedTimePenaltyError(
            "time_weight > 0 with no speed cap (v_max) is an artifact generator: "
            "nothing opposes the time penalty, so the arrival time collapses to "
            "min_dt times the number of control-point gaps regardless of the "
            "scenario. Pass v_max, or set time_weight=0. "
            "(Formulation decision 5 -- items B9 and B10 land together.)"
        )


# Mirrors `stop_reason` in rust_optimizer/core/src/spacetime_optimizer.rs.
# Only `merit_streak` and `stationary` are success claims; the rest mean the loop
# gave up, and the labels say so rather than leaving it to the reader.
_STOP_REASONS = {
    0: "iteration_cap (gave up)",
    1: "merit_streak",
    2: "trust_collapse (gave up unless certified)",
    3: "qp_failure (gave up)",
    4: "stationary",
    -1: "still running / not reported",
}


def _optimize_spacetime_rust(
    P_init: np.ndarray,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
    elastic_weight: float = DEFAULT_ELASTIC_WEIGHT,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    cap_bulge_ratio: float = 2.0,
    v_max: float | None = None,
    time_weight: float = 0.0,
    free_arrival_time: bool = False,
    stations=None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Call the native Rust backend for the space-time optimizer."""
    if _bezier_opt_rs is None or not hasattr(_bezier_opt_rs, "optimize_spacetime_bezier"):
        raise RuntimeError("Rust space-time optimizer is not available in bezier_opt.")

    check_time_penalty_is_capped(v_max, time_weight)

    P_init = np.asarray(P_init, dtype=float)
    n_cp, dim = P_init.shape
    spatial_dim = dim - 1
    pos0, vel, radius, t_start, t_end = obstacle_array_bundle(obstacles, spatial_dim)
    time_upper = float(P_init[-1, -1]) * float(time_ub_scale)
    # No station means no occlusion rows at all (item B12), which is the default
    # and reproduces every pre-B12 run bit for bit. `None` is passed through
    # rather than an empty array so the Rust side has one code path for "off".
    station_arr = None
    if stations is not None:
        station_arr = np.asarray(stations, dtype=float).reshape(-1, spatial_dim)
        if station_arr.shape[0] == 0:
            station_arr = None

    P_opt, info = _bezier_opt_rs.optimize_spacetime_bezier(
        p_init=P_init,
        obstacle_pos0=pos0,
        obstacle_vel=vel,
        obstacle_r=radius,
        obstacle_t_start=t_start,
        obstacle_t_end=t_end,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        elastic_weight=elastic_weight,
        min_dt=min_dt,
        coord_lb=coord_lb,
        coord_ub=coord_ub,
        time_lb=time_lb,
        time_ub=time_upper,
        cap_bulge_ratio=cap_bulge_ratio,
        v_max=v_max,
        time_weight=float(time_weight),
        free_arrival_time=bool(free_arrival_time),
        stations=station_arr,
    )
    P_opt = np.asarray(P_opt, dtype=float)
    info = dict(info)
    info["backend"] = "rust"

    if verbose:
        iterations = int(info.get("iterations", -1))
        clearance = float(info.get("min_clearance", math.nan))
        feasible = bool(info.get("feasible", 0.0))
        delta = float(info.get("final_delta_norm", math.nan))
        total_slack = float(info.get("total_koz_slack", 0.0))
        print(
            f"SCP: N={n_cp - 1}, dim={dim}, n_seg={n_seg}, n_cp={n_cp}, n_obs={len(obstacles)}, backend=rust"
        )
        # Termination label comes from the solver, which is the only thing that knows
        # why it stopped. The previous version re-derived it here as
        # "delta < tol -> converged", which reports a small step as an optimality
        # claim -- and on the legacy path every step is accepted unconditionally, so a
        # small step can mean the iterate stopped moving for any reason at all.
        reason = _STOP_REASONS.get(int(info.get("stop_reason", -1)), "unknown")
        converged = bool(info.get("converged", 0.0))
        print(f"  rust result: iterations={iterations}/{max_iter}, clearance={clearance:.4f}")
        print(
            f"  termination: {reason}, converged={converged}, feasible={feasible}, "
            f"delta={delta:.2e}, koz_slack={total_slack:.2e}"
        )
        if info.get("returned_best_iterate", 0.0):
            print(
                "  NOTE: the final iterate penetrated an obstacle; returning the best "
                "feasible iterate seen instead. It is NOT the point the loop stopped on."
            )
        print(
            f"  steps: accept={int(info.get('accept_count', 0))} "
            f"reject={int(info.get('reject_count', 0))} "
            f"null={int(info.get('null_step_count', 0))} "
            f"bootstrap={int(info.get('bootstrap_count', 0))}"
        )
        print(
            f"  rho: mean={info.get('rho_mean', math.nan):.4f} "
            f"min={info.get('rho_min', math.nan):.4f} "
            f"max={info.get('rho_max', math.nan):.4f} "
            f"n={int(info.get('rho_samples', 0))}, "
            f"final_trust={info.get('final_trust', math.nan):.3e}"
        )
        print(
            f"  hull certificate violation at returned iterate: "
            f"{info.get('koz_violation_reference', math.nan):.3e}"
        )

    return P_opt, info


def optimize_spacetime_from_control_points(
    P_init,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
    elastic_weight: float = DEFAULT_ELASTIC_WEIGHT,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    cap_bulge_ratio: float = 2.0,
    v_max: float | None = None,
    time_weight: float = 0.0,
    free_arrival_time: bool = False,
    stations=None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Optimize a space-time Bezier curve from an initial control polygon.

    Returns (control_points, info) where info always contains 'backend'.
    """
    return _optimize_spacetime_rust(
        np.asarray(P_init, dtype=float),
        obstacles,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        elastic_weight=elastic_weight,
        min_dt=min_dt,
        coord_lb=coord_lb,
        coord_ub=coord_ub,
        time_lb=time_lb,
        time_ub_scale=time_ub_scale,
        cap_bulge_ratio=cap_bulge_ratio,
        v_max=v_max,
        time_weight=time_weight,
        free_arrival_time=free_arrival_time,
        stations=stations,
        verbose=verbose,
    )


def optimize_spacetime(
    N: int,
    dim: int,
    p_start,
    p_end,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
    elastic_weight: float = DEFAULT_ELASTIC_WEIGHT,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    cap_bulge_ratio: float = 2.0,
    v_max: float | None = None,
    time_weight: float = 0.0,
    free_arrival_time: bool = False,
    stations=None,
    verbose: bool = True,
    init_curve: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Public optimizer entrypoint.

    Returns (control_points, info) where info always contains 'backend'.
    """
    n_cp = int(N) + 1
    P_init = build_initial_guess(p_start, p_end, n_cp, init_curve=init_curve)
    if dim != P_init.shape[1]:
        raise ValueError(f"Expected dim={dim}, got initial guess with dim={P_init.shape[1]}")
    return optimize_spacetime_from_control_points(
        P_init,
        obstacles,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        elastic_weight=elastic_weight,
        min_dt=min_dt,
        coord_lb=coord_lb,
        coord_ub=coord_ub,
        time_lb=time_lb,
        time_ub_scale=time_ub_scale,
        cap_bulge_ratio=cap_bulge_ratio,
        v_max=v_max,
        time_weight=time_weight,
        free_arrival_time=free_arrival_time,
        stations=stations,
        verbose=verbose,
    )


def optimize_scenario(
    scenario: dict,
    configs: list[tuple[int, int]],
    max_iter: int = 200,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.3,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
    elastic_weight: float | None = None,
    min_dt: float = 0.1,
    v_max: float | None = None,
    time_weight: float = 0.0,
    free_arrival_time: bool = False,
    verbose: bool = True,
) -> dict:
    """Run optimization for all requested degree/segment-count pairs.

    ``elastic_weight=None`` (the default) walks ``ELASTIC_WEIGHT_LADDER`` and
    keeps the first run that is converged, certified and clearing. Passing a
    float pins the weight and disables continuation, which is what the
    reproducibility tests want.
    """
    ladder = (
        tuple(ELASTIC_WEIGHT_LADDER) if elastic_weight is None else (float(elastic_weight),)
    )
    obstacles = scenario["obstacles"]
    p_start = scenario["start"]
    p_end = scenario["end"]
    init_curve = scenario.get("init_curve")
    # Absent key means no occlusion rows (item B12). Every scenario that predates
    # B12 therefore solves exactly the problem it always did.
    stations = scenario.get("stations")

    results = {}
    for N, n_seg in configs:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[{scenario['name']}] degree={N}, segments={n_seg}")
            print(f"{'=' * 60}")

        # Keep the best run seen, not merely the last one tried. Escalating past
        # a weight that already cleared can make things worse -- `wall` N8_seg2
        # clears at +0.1035 with w=800 and penetrates at -0.1474 with w=1e5 --
        # so a ladder that returned its final rung would report the worse answer
        # for every config that never certifies.
        best_rank = None
        P_opt = opt_info = clearance = None
        used_weight = ladder[0]
        for candidate_weight in ladder:
            P_try, info_try = optimize_spacetime(
                N=N,
                dim=len(p_start),
                p_start=p_start,
                p_end=p_end,
                obstacles=obstacles,
                n_seg=n_seg,
                max_iter=max_iter,
                tol=tol,
                scp_prox_weight=scp_prox_weight,
                scp_trust_radius=scp_trust_radius,
                elastic_weight=candidate_weight,
                min_dt=min_dt,
                v_max=v_max,
                time_weight=time_weight,
                free_arrival_time=free_arrival_time,
                stations=stations,
                verbose=verbose,
                init_curve=init_curve,
            )
            clearance_try = compute_min_clearance(
                P_try, obstacles, dim=len(p_start), n_eval=3000
            )
            cert_try = float(info_try.get("koz_violation_reference", float("nan")))
            # The ladder must not stop on a run that has lost line of sight, so
            # the occlusion certificate joins the keep-out one in the stopping
            # test. Zero when there is no station.
            occ_try = float(info_try.get("occlusion_violation_reference", 0.0))
            cleared_try = (
                bool(info_try.get("converged", 0.0))
                and cert_try <= 1e-6
                and occ_try <= 1e-6
                and clearance_try > 0.0
            )
            # Same ordering the scenario-level `_rank` uses to pick `best`.
            rank_try = (
                cleared_try,
                clearance_try > 0.0,
                clearance_try,
            )
            if best_rank is None or rank_try > best_rank:
                best_rank = rank_try
                P_opt, opt_info, clearance, used_weight = (
                    P_try,
                    info_try,
                    clearance_try,
                    candidate_weight,
                )
            if verbose:
                print(f"  w={candidate_weight:g}: clearance={clearance_try:.4f}, "
                      f"certificate={cert_try:.4g}, occlusion={occ_try:.4g}")
            if cleared_try:
                break

        backend_used = opt_info["backend"]
        if verbose:
            print(f"  Final clearance: {clearance:.4f} (elastic_weight={used_weight:g})")

        key = f"N{N}_seg{n_seg}"
        # `feasible` says the sampled curve misses the obstacles. `certified`
        # says the control-point hull satisfies the half-spaces it generates,
        # which is the property the paper claims. They are different, and a run
        # can pass the first while failing the second -- so both are recorded and
        # neither is allowed to stand in for the other.
        certificate = float(opt_info.get("koz_violation_reference", float("nan")))
        occlusion = float(opt_info.get("occlusion_violation_reference", 0.0))
        total_slack = float(opt_info.get("total_koz_slack_returned", float("nan")))
        results[key] = {
            "N": int(N),
            "n_seg": int(n_seg),
            "control_points": np.asarray(P_opt, dtype=float).tolist(),
            "min_clearance": float(clearance),
            "feasible": bool(clearance > 0.0),
            "backend": backend_used,
            "converged": bool(opt_info.get("converged", 0.0)),
            "stop_reason": int(opt_info.get("stop_reason", -1)),
            "stop_label": _STOP_REASONS.get(int(opt_info.get("stop_reason", -1)), "unknown"),
            "iterations": int(opt_info.get("iterations", -1)),
            "certificate_violation": certificate,
            "certified": bool(certificate <= 1e-6),
            "occlusion_violation": occlusion,
            "occlusion_certified": bool(occlusion <= 1e-6),
            "occlusion_planes_dropped": float(
                opt_info.get("occlusion_planes_dropped", 0.0)
            ),
            "total_slack": total_slack,
            "accept_count": int(opt_info.get("accept_count", 0)),
            "reject_count": int(opt_info.get("reject_count", 0)),
            "returned_best_iterate": bool(opt_info.get("returned_best_iterate", 0.0)),
            "trust_radius": float(scp_trust_radius),
            "elastic_weight": float(used_weight),
            "speed_cap_violation": float(opt_info.get("speed_cap_violation", 0.0)),
            "arrival_time": float(opt_info.get("arrival_time", float("nan"))),
        }
        results[key]["figure_grade"] = is_figure_grade(results[key])
        results[key]["figure_grade_reasons"] = figure_grade_failures(results[key])
        if verbose and not results[key]["figure_grade"]:
            print(
                "  NOT FIGURE-GRADE: "
                + "; ".join(results[key]["figure_grade_reasons"])
            )

    def _rank(item):
        _, v = item
        return (
            bool(v["feasible"]) and bool(v.get("certified", False)),
            bool(v["feasible"]),
            float(v["min_clearance"]),
        )

    best_key = max(results.items(), key=_rank)[0]

    if verbose:
        print(
            f"[{scenario['name']}] Best: {best_key} (clearance={results[best_key]['min_clearance']:.4f})"
        )

    return {
        "name": scenario["name"],
        "title": scenario["title"],
        "best": best_key,
        "obstacles": obstacles,
        "start": p_start,
        "end": p_end,
        "T": scenario["T"],
        "init_curve": init_curve,
        "results": results,
    }


def optimize_scenarios(
    scenario_names: list[str],
    scenario_map: dict,
    existing_outputs: dict | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.3,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
    elastic_weight: float | None = None,
    min_dt: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Optimize the selected scenarios and merge them with any existing outputs."""
    all_outputs = dict(existing_outputs or {})
    for name in scenario_names:
        scenario_fn, configs = scenario_map[name]
        all_outputs[name] = optimize_scenario(
            scenario_fn(),
            configs,
            max_iter=max_iter,
            tol=tol,
            scp_prox_weight=scp_prox_weight,
            scp_trust_radius=scp_trust_radius,
            elastic_weight=elastic_weight,
            min_dt=min_dt,
            verbose=verbose,
        )
    return all_outputs
