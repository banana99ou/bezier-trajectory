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
# "A run with total_slack > 0 cannot produce a figure." The conditions are
# independent and none may stand in for another. There are SIX, and the count is
# written out here because the previous version said "four" while the predicate
# checked five:
#
#   converged        -- the loop stopped for a principled reason, not the cap.
#   certificate      -- the control-point hull satisfies the keep-out half-spaces
#                       it generates, rebuilt at the RETURNED iterate. This is
#                       the property the paper claims; clearance is not.
#   occlusion        -- the same statement for the line-of-sight half-spaces
#                       (item B12). A separate guarantee: the keep-out
#                       certificate says nothing about whether the station is
#                       still visible.
#   occlusion drops  -- every line-of-sight plane that was in range could
#                       actually be built. A window whose station lies inside the
#                       inflated occluder has no supporting half-space at all, so
#                       it is UNCERTIFIABLE, not certified.
#   clearance        -- the sampled curve misses the TRUE obstacle trajectories.
#                       Re-verified in Python against the obstacles themselves,
#                       not against the hulls the solver used, which is the
#                       condition PAPER_1 sec. 6 requires because an
#                       adaptively-built hull is valid only inside the time span
#                       it was built for.
#   total_slack      -- the elastic relaxation bought nothing. Every subproblem
#                       is solved elastically, so a converged, certified,
#                       clearing run can still have been standing on slack; this
#                       is the condition that catches it.
#   speed cap        -- the returned curve satisfies the slant-limit cones. The
#                       cap is hard (formulation decision 4), so a violation is a
#                       broken constraint, not a priced relaxation.
#
# NaN fails every comparison, so a run that never accepted a step (slack unknown)
# is not figure-grade. That is the intended answer for "no evidence". The three
# conditions that are OPTIONAL features -- occlusion, occlusion drops, speed cap
# -- default to 0.0 instead, because a run without stations or without a cap
# genuinely has nothing to violate, and defaulting them to NaN would sink every
# scenario in the table.
# ---------------------------------------------------------------------------

FIGURE_GRADE_CERTIFICATE_TOL = 1e-6

# Measured, not chosen -- and the previous version of this comment was wrong in
# every clause, so what it now says is only what was measured.
#
# Clarabel is an interior-point method, so the slack variables approach zero
# without reaching it. THE RESIDUE IS DRIVEN BY THE PENALTY WEIGHT, NOT BY THE
# ROW COUNT. Measured on `original` N8_seg4, where the row count is fixed at 108
# and only the weight moves:
#
#     w=30    2.57e-01 (not converged)     w=3000   3.00e-11
#     w=100   1.27e-10                     w=1e4    7.15e-13
#     w=300   2.84e-10                     w=1e5    1.23e-13
#     w=800   2.74e-12                     w=1e6   -1.65e-14
#
# Four orders of weight buy four orders of residue at constant row count. The
# claim that the residue "grows with problem size" does not survive that table.
#
# THE RESIDUE CAN BE NEGATIVE. -1.65e-14 above, which is why the comparison below
# is on the ABSOLUTE value: `slack <= tol` would have passed an arbitrarily large
# negative number, and a negative total is a solver artifact rather than a
# credit against a violation.
#
# The quoted row counts were wrong too: `wall` N10_seg16 has 2640 KOZ rows, not
# 4400. Counted with `spacetime_koz_rows_exact` at the straight seed;
# `original` N8_seg4 has 108, `diverse` N8_seg4 252, `wall3d` N8_seg2 234.
#
# THE TWO POPULATIONS, measured across a sweep of `diverse` and `wall3d` at
# degrees 8-12, 8-32 segments and weights 800 to 1e5:
#
#   GOOD  (converged, certificate <= 1e-6, clearance > 0): worst residue
#         8.51e-9, at `diverse` N10_seg16 with the ladder's w=3000. That is 85%
#         of the OLD 1e-8 gate -- the gate was within a factor of 1.18 of
#         rejecting a good run for the solver's arithmetic.
#   BAD   (a genuine relaxation): the smallest observed is 1.35, at `wall`
#         N8_seg2 on the ladder's w=800 rung. That run is worth naming, because
#         it is the exact case this condition exists for: its SAMPLED clearance
#         is +0.102, so it passes the clearance condition, and it is standing on
#         1.35 of slack to do it. The rest of the population runs 1.37 to 1.98
#         across `wall`, 2.29 (`wall` N8_seg2 pinned at w=100, the historical
#         penetrating run), 6.52 (the blob scenario), and 7.2 to 9.7 (`diverse`
#         below its penalty threshold). A genuine relaxation is of the order of
#         the clearance it bought, which is why the two populations do not
#         overlap.
#
# 1e-6 sits 117x above the worst good residue and six orders below the smallest
# bad one. The gap between the populations is about eight orders wide and this
# threshold is placed inside it rather than on its edge. The guarantee is carried
# by the certificate above, which is evaluated against the EXACT rows rebuilt at
# the returned iterate; this condition only catches a run standing on relaxation.
FIGURE_GRADE_SLACK_TOL = 1e-6


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
    # ABSOLUTE value: the interior-point residue can come back negative
    # (-1.65e-14 measured), and `slack <= tol` would have passed any negative
    # number however large. NaN still fails, which is the point of the idiom.
    slack = abs(float(row.get("total_slack", float("nan"))))
    if not slack <= FIGURE_GRADE_SLACK_TOL:
        reasons.append(f"elastic slack {slack:.3e} > {FIGURE_GRADE_SLACK_TOL:g}")
    # The speed cap is a HARD constraint (formulation decision 4), so a nonzero
    # violation is not slack the run paid for -- it is a constraint the returned
    # curve does not satisfy. It was recorded in every row and read by nothing,
    # which made it decorative. Defaults to 0.0, which is the true value for a
    # capless run: there are no cones, so there is nothing to violate.
    speed_cap = float(row.get("speed_cap_violation", 0.0))
    if not speed_cap <= FIGURE_GRADE_CERTIFICATE_TOL:
        reasons.append(f"speed cap violated by {speed_cap:.3e}")
    return reasons


def is_figure_grade(row: dict) -> bool:
    """True only when every gate condition holds. See ``figure_grade_failures``."""
    return not figure_grade_failures(row)


class UncappedTimePenaltyError(ValueError):
    """A time penalty was requested with no speed cap to hold it back.

    **This guard is necessary and not sufficient, and the difference is
    measured.** What produces the artifact is a cap that does not BIND, and "no
    cap at all" is only the extreme case of that. On the obstacle-free problem
    with ``min_dt=0.1`` and ``N=8``, any ``v_max`` above
    ``chord / (min_dt * N) ~= 13.7`` leaves the cones slack, and the arrival time
    returns exactly ``min_dt * gaps = 0.800000`` -- the same collapse, with this
    guard passing.

    The detector for the case the guard cannot see is the info flag
    ``arrival_on_min_dt_floor``: it compares the RETURNED arrival against
    ``t_start + min_dt * (control-point gaps)`` and is set when they agree to
    1e-6. It is a flag on the result, not a refusal, because a non-binding cap is
    a legitimate configuration -- the number it produces just is not a
    measurement of the scenario.
    """


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


class DegenerateFreeArrivalError(ValueError):
    """The arrival time was freed and then left out of the objective.

    The smoothness regularizer is blind to the time coordinate -- measured, not
    assumed: the Rust cost oracle returns bit-identical values for a cruise and a
    wait-then-dash with the same spatial control points. With ``time_weight=0``
    the linear term is zero as well, so with ``free_arrival_time=True`` the
    arrival time appears in NO term of the objective. It is a free variable of a
    function that does not depend on it.

    What comes back is therefore whichever point on the flat optimal face the
    interior-point solver happened to centre on. Measured on the obstacle-free
    problem, `converged=1`, `stop_reason=stationary`, `cost` ~1e-12 at every row:

        trust radius   0.10   0.25   0.50   1.00   2.00
        arrival       10.08  10.26  10.52  11.04  12.24

    A 2.2 s spread driven by a knob that is not part of the problem, reported as
    a converged measurement. A speed cap does not repair it: the cap bounds the
    arrival from below and the objective is still flat above that bound.

    Pass a positive ``time_weight`` to make the arrival mean something, or leave
    ``free_arrival_time=False`` (the default) to pin it.
    """


class SpeedCapOutOfRangeError(ValueError):
    """``v_max`` is not a number the cone formulation can carry."""


# A RANGE CHECK, not a numerical guarantee, and the difference is measured.
#
# The cone is `|| dx || <= v_max * dt`. As `v_max` grows the two sides separate
# by orders of magnitude, Clarabel's scaling degrades, and the run reports
# `qp_failure` or `first_qp_infeasible` with `speed_cap_violation` 0.00 -- a
# failure that reads as a satisfied constraint. On `original` N8_seg4 with a
# pinned arrival that degradation is already visible well BELOW this bound:
#
#     v_max   1e1 .. 3e4   stationary, converged, clearance +0.6204 (== no cap)
#     v_max   1e5          qp_failure at iteration 8,  violation 0.00
#     v_max   3e5          qp_failure at iteration 2,  violation 0.00
#     v_max   1e6          first_qp_infeasible at iteration 1, violation 0.00
#
# So 1e6 does not separate "solves" from "does not solve" -- nothing scalar
# does, because the threshold moves with the scenario, the segment count and the
# trust radius. What this bound excludes is input that is nonsense on its face
# (negative, zero, infinite, absurd), and what surfaces the rest is the
# `first_qp_infeasible` stop label plus the gate condition on
# `speed_cap_violation`. Documented rather than tuned: a bound presented as a
# safety threshold that is not one is worse than no bound.
MAX_SPEED_CAP = 1e6


def check_speed_cap_is_representable(v_max) -> None:
    """Refuse a ``v_max`` that is outside the representable range.

    ``None`` is how "no cap" is spelled and is always accepted. Everything else
    must be a positive finite number no larger than ``MAX_SPEED_CAP``. Zero and
    negatives are refused rather than silently reinterpreted as "off": the Rust
    builder does treat them as off, and an unnoticed sign error that turns a cap
    into no cap is exactly the failure this project keeps finding.

    **Passing this check does not mean the cone will solve.** See
    ``MAX_SPEED_CAP`` for the measured degradation, which begins around 1e5 on
    `original` -- an order of magnitude below the bound.
    """
    if v_max is None:
        return
    value = float(v_max)
    if not (value > 0.0) or not math.isfinite(value) or value > MAX_SPEED_CAP:
        raise SpeedCapOutOfRangeError(
            f"v_max={v_max!r} is outside the representable range "
            f"(0, {MAX_SPEED_CAP:g}]. Pass None for no cap. Above the upper "
            "bound the second-order cone degenerates numerically and the run "
            "reports qp_failure with a speed_cap_violation of 0.00, which reads "
            "as a satisfied constraint."
        )


def check_free_arrival_is_costed(free_arrival_time, time_weight) -> None:
    """Refuse a freed arrival time that nothing in the objective can price.

    Same reasoning as ``check_time_penalty_is_capped`` and the same remedy: raise
    rather than warn, because the returned number is indistinguishable from a
    real one once it reaches a table.
    """
    if bool(free_arrival_time) and float(time_weight) == 0.0:
        raise DegenerateFreeArrivalError(
            "free_arrival_time=True with time_weight=0 leaves the arrival time "
            "in no term of the objective, so the reported value is an "
            "interior-point tie-break on a flat optimal face -- measured to "
            "swing 10.08 -> 12.24 with the trust radius alone while reporting "
            "converged and stationary. Pass time_weight > 0 (with a speed cap), "
            "or leave free_arrival_time=False."
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
    5: (
        "first_qp_infeasible -- trust radius may be too small to reach the speed "
        "cap from the initial guess (gave up)"
    ),
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
    # PAPER_1 statement (8): clamp the clip radius from below by
    # E + Delta*sqrt(d+1) so statement (7) holds unconditionally and the
    # construction is sound by construction, at the cost of conservatism where
    # the row binds. Off by default -- PAPER_1 calls the choice between the two
    # an OPEN EXPERIMENTAL QUESTION, so both are reachable and measurable.
    sound_clip: bool = False,
    v_max: float | None = None,
    time_weight: float = 0.0,
    free_arrival_time: bool = False,
    stations=None,
    coord_bounds=None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Call the native Rust backend for the space-time optimizer.

    ``coord_bounds`` -- optional per-spatial-coordinate box, a sequence of
    ``(lo, hi)`` pairs, one per spatial coordinate. ``None`` keeps the uniform
    ``[coord_lb, coord_ub]`` box every run has always had. The rows are HARD --
    they sit outside the elastic slack range, so the penalty can never buy its
    way through a workspace wall. An altitude band is
    ``coord_bounds=[(-12, 12), (-12, 12), (0, 6)]``.
    """
    if _bezier_opt_rs is None or not hasattr(_bezier_opt_rs, "optimize_spacetime_bezier"):
        raise RuntimeError("Rust space-time optimizer is not available in bezier_opt.")

    # Order matters: `v_max=0.0` with a time penalty is the uncapped-artifact
    # configuration first and an out-of-range cap second, and the first is the
    # more informative refusal.
    check_time_penalty_is_capped(v_max, time_weight)
    check_free_arrival_is_costed(free_arrival_time, time_weight)
    check_speed_cap_is_representable(v_max)

    P_init = np.asarray(P_init, dtype=float)
    n_cp, dim = P_init.shape
    spatial_dim = dim - 1
    coord_lb_by_axis = coord_ub_by_axis = None
    if coord_bounds is not None:
        bounds = np.asarray(coord_bounds, dtype=float)
        if bounds.shape != (spatial_dim, 2):
            raise ValueError(
                f"coord_bounds must be {spatial_dim} (lo, hi) pairs -- one per "
                f"spatial coordinate -- got shape {bounds.shape}"
            )
        if not np.all(bounds[:, 0] < bounds[:, 1]):
            raise ValueError(f"coord_bounds has an empty or reversed interval: {bounds.tolist()}")
        # The box rows exempt the pinned endpoints, so bounds that exclude an
        # endpoint would not make the QP infeasible -- they would make the curve
        # leap from the pinned point into the band and back, which is almost
        # certainly a scenario bug. Refuse it loudly instead.
        for label, point in (("start", P_init[0, :spatial_dim]), ("end", P_init[-1, :spatial_dim])):
            if np.any(point < bounds[:, 0]) or np.any(point > bounds[:, 1]):
                raise ValueError(
                    f"coord_bounds {bounds.tolist()} excludes the {label} point "
                    f"{point.tolist()}; endpoints are pinned, so the bound cannot "
                    "move them -- move the endpoint or widen the band"
                )
        coord_lb_by_axis = bounds[:, 0].tolist()
        coord_ub_by_axis = bounds[:, 1].tolist()
    obstacle_ctrl, obstacle_radii = obstacle_array_bundle(obstacles, spatial_dim)
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
        obstacle_ctrl=obstacle_ctrl,
        obstacle_r=obstacle_radii,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        elastic_weight=elastic_weight,
        min_dt=min_dt,
        coord_lb=coord_lb,
        coord_ub=coord_ub,
        coord_lb_by_axis=coord_lb_by_axis,
        coord_ub_by_axis=coord_ub_by_axis,
        time_lb=time_lb,
        time_ub=time_upper,
        sound_clip=sound_clip,
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
        # Two different numbers describing two different trajectories. The gate
        # reads the RETURNED one, so that is the one printed first and the one
        # labelled as such; the last subproblem's slack was what this line used
        # to show, which on a rejected or fallback step belongs to a point
        # nobody received. Both are kept because their disagreement is
        # informative -- measured 2.24 vs 1.62 on `diverse` N8_seg4.
        returned_slack = float(info.get("total_koz_slack_returned", math.nan))
        last_slack = float(info.get("total_koz_slack", math.nan))
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
            f"delta={delta:.2e}, slack(returned)={returned_slack:.2e}, "
            f"slack(last subproblem)={last_slack:.2e}"
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
    # PAPER_1 statement (8): clamp the clip radius from below by
    # E + Delta*sqrt(d+1) so statement (7) holds unconditionally and the
    # construction is sound by construction, at the cost of conservatism where
    # the row binds. Off by default -- PAPER_1 calls the choice between the two
    # an OPEN EXPERIMENTAL QUESTION, so both are reachable and measurable.
    sound_clip: bool = False,
    v_max: float | None = None,
    time_weight: float = 0.0,
    free_arrival_time: bool = False,
    stations=None,
    coord_bounds=None,
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
        sound_clip=sound_clip,
        v_max=v_max,
        time_weight=time_weight,
        free_arrival_time=free_arrival_time,
        stations=stations,
        coord_bounds=coord_bounds,
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
    # PAPER_1 statement (8): clamp the clip radius from below by
    # E + Delta*sqrt(d+1) so statement (7) holds unconditionally and the
    # construction is sound by construction, at the cost of conservatism where
    # the row binds. Off by default -- PAPER_1 calls the choice between the two
    # an OPEN EXPERIMENTAL QUESTION, so both are reachable and measurable.
    sound_clip: bool = False,
    v_max: float | None = None,
    time_weight: float = 0.0,
    free_arrival_time: bool = False,
    stations=None,
    coord_bounds=None,
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
        sound_clip=sound_clip,
        v_max=v_max,
        time_weight=time_weight,
        free_arrival_time=free_arrival_time,
        stations=stations,
        coord_bounds=coord_bounds,
        verbose=verbose,
    )


def optimize_scenario(
    scenario: dict,
    configs: list[tuple[int, int]],
    max_iter: int = 200,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.3,
    scp_trust_radius: float | None = None,
    elastic_weight: float | None = None,
    min_dt: float = 0.1,
    v_max: float | None = None,
    time_weight: float = 0.0,
    free_arrival_time: bool = False,
    sound_clip: bool = False,
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
    # Absent key means the uniform default box, so every scenario without one
    # solves exactly the problem it always did.
    coord_bounds = scenario.get("coord_bounds")
    # A trust radius is a LENGTH, so it belongs to the scene as much as the box
    # does: 0.5 against a 200 m scene is degenerate (see `scenario_loiter`). An
    # explicit argument still wins; absent both, the module default, so every
    # scenario without the key solves exactly the problem it always did.
    trust_radius = (
        float(scp_trust_radius)
        if scp_trust_radius is not None
        else float(scenario.get("trust_radius", DEFAULT_TRUST_RADIUS))
    )

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
                scp_trust_radius=trust_radius,
                elastic_weight=candidate_weight,
                min_dt=min_dt,
                v_max=v_max,
                time_weight=time_weight,
                free_arrival_time=free_arrival_time,
                sound_clip=sound_clip,
                stations=stations,
                coord_bounds=coord_bounds,
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
            # The RESOLVED radius, not the argument: the scene may have
            # supplied it, and this row is what the verdicts and every
            # recorded measurement downstream read.
            "trust_radius": float(trust_radius),
            "elastic_weight": float(used_weight),
            "speed_cap_violation": float(opt_info.get("speed_cap_violation", 0.0)),
            "arrival_time": float(opt_info.get("arrival_time", float("nan"))),
            # Not a gate condition -- see the Rust export. It flags an arrival
            # time that is a property of `min_dt` rather than of the scenario,
            # which is a modelling artifact and not a constraint violation.
            "arrival_on_min_dt_floor": float(
                opt_info.get("arrival_on_min_dt_floor", 0.0)
            ),
            # The upper clamp on a freed arrival, and whether the answer is
            # sitting on it. Also not a gate condition: an arrival at the bound
            # is a legitimate answer to the problem as posed, it just was not
            # posed by the geometry.
            "time_ub_used": float(opt_info.get("time_ub_used", float("nan"))),
            "arrival_on_time_ub": float(opt_info.get("arrival_on_time_ub", 0.0)),
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
    scp_trust_radius: float | None = None,
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
