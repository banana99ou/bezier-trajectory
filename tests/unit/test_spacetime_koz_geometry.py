"""Geometry tests for the Rust space-time obstacle-avoidance builder.

Item B2. These are the first tests that touch the Rust builder at all; the
existing `test_spacetime_constraints.py` covers a Python builder production does
not use, and one of its assertions (`A[:, 2::3] == 0`) actively enshrines the
defect this file exists to catch.

Every test below FAILS on the pre-2026-08-17 geometry. The two properties are:

  1. A MOVING obstacle's half-space must have a nonzero time coefficient. The old
     builder set it to zero, which told the solver that arriving earlier or later
     could not change clearance -- deleting the entire point of the space-time
     lift. Every pre-existing obstacle test used a stationary obstacle, where
     zero is genuinely correct, so none of them could catch it.

  2. All control points of one segment must share ONE half-space per obstacle.
     That single fact is what upgrades finitely many checks into a
     continuous-time guarantee via the convex hull property.
"""

import numpy as np
import pytest

import bezier_opt

from spacetime_bezier.optimize import optimize_spacetime

DIM = 3  # x, y, t


def _exact_rows(P, obstacles, n_seg):
    """The EXACT half-spaces -- the rows the certificate is evaluated with.

    Not the rows handed to the solver. Those carry a rotation term that makes
    them self-consistent under the step, and they are documented as NOT
    conservative; the two properties below hold only for the exact rows.
    """
    P = np.asarray(P, dtype=float)
    pos0 = np.array([o["pos0"] for o in obstacles], dtype=float)
    vel = np.array([o["vel"] for o in obstacles], dtype=float)
    r = np.array([o["r"] for o in obstacles], dtype=float)
    t0 = np.array([o.get("t_start", -1e18) for o in obstacles], dtype=float)
    t1 = np.array([o.get("t_end", 1e18) for o in obstacles], dtype=float)
    normals, lbs, seg, cp, obs = bezier_opt.spacetime_koz_rows_exact(
        p=P, obstacle_pos0=pos0, obstacle_vel=vel, obstacle_r=r,
        obstacle_t_start=t0, obstacle_t_end=t1, n_seg=n_seg,
    )
    return {
        "normals": np.asarray(normals, dtype=float).reshape(-1, DIM),
        "lbs": np.asarray(lbs, dtype=float),
        "seg": np.asarray(seg),
        "cp": np.asarray(cp),
        "obs": np.asarray(obs),
    }


def _koz_rows(P, obstacles, n_seg):
    """Rows as the SOLVER sees them, via the debug context."""
    P = np.asarray(P, dtype=float)
    np1 = P.shape[0]
    pos0 = np.array([o["pos0"] for o in obstacles], dtype=float)
    vel = np.array([o["vel"] for o in obstacles], dtype=float)
    r = np.array([o["r"] for o in obstacles], dtype=float)
    t0 = np.array([o.get("t_start", -1e18) for o in obstacles], dtype=float)
    t1 = np.array([o.get("t_end", 1e18) for o in obstacles], dtype=float)

    ctx = bezier_opt.SpacetimeScpContext(
        p_init=P,
        obstacle_pos0=pos0,
        obstacle_vel=vel,
        obstacle_r=r,
        obstacle_t_start=t0,
        obstacle_t_end=t1,
        n_seg=n_seg,
        min_dt=0.05,
        coord_lb=-50.0,
        coord_ub=50.0,
        time_lb=0.0,
        time_ub=50.0,
        scp_trust_radius=0.5,
        elastic_weight=100.0,
        tol=1e-6,
    )
    out = ctx.step()
    (_p, _info, seg, cp, obs, _it, normals, supports, centers, lbs, margins, _slack) = out
    return {
        "seg": np.asarray(seg),
        "cp": np.asarray(cp),
        "obs": np.asarray(obs),
        "normals": np.asarray(normals, dtype=float).reshape(-1, DIM),
        "lbs": np.asarray(lbs, dtype=float),
        "supports": np.asarray(supports, dtype=float).reshape(-1, DIM),
    }


def _straight_guess(start, end, n_cp):
    return np.linspace(np.asarray(start, float), np.asarray(end, float), n_cp)


MOVING = [{"pos0": [5.0, 0.0], "vel": [0.0, 1.2], "r": 1.0}]
STATIONARY = [{"pos0": [5.0, 0.0], "vel": [0.0, 0.0], "r": 1.0}]


def test_moving_obstacle_normal_has_nonzero_time_component():
    """A moving obstacle's tube leans in space-time, so its outward normal must
    have a time component.

    FAILS IF: the time coefficient is zero -- which is exactly what the previous
    builder emitted, and what `test_spacetime_constraints.py:65` still asserts
    for the dead Python builder.
    """
    P = _straight_guess([0.0, 0.0, 0.0], [10.0, 0.0, 8.0], 9)
    rows = _koz_rows(P, MOVING, n_seg=4)

    assert len(rows["normals"]) > 0, "no obstacle rows were generated at all"
    time_parts = np.abs(rows["normals"][:, DIM - 1])
    assert time_parts.max() > 1e-6, (
        "every half-space has a zero time coefficient: the solver is being told "
        "that moving a control point in time cannot change clearance"
    )


def test_stationary_obstacle_normal_has_zero_time_component():
    """The converse, which pins the sign convention rather than just the
    magnitude: a stationary obstacle's tube is vertical, so zero IS correct.

    FAILS IF: the new geometry sprays a spurious time component everywhere, which
    would make the previous test pass for the wrong reason.
    """
    P = _straight_guess([0.0, 0.0, 0.0], [10.0, 0.0, 8.0], 9)
    rows = _koz_rows(P, STATIONARY, n_seg=4)

    assert len(rows["normals"]) > 0
    time_parts = np.abs(rows["normals"][:, DIM - 1])
    assert time_parts.max() < 1e-9, (
        f"stationary obstacle produced a time component of {time_parts.max():.3e}"
    )


def test_time_component_tracks_obstacle_velocity():
    """Faster obstacle, more lean, larger time coefficient.

    FAILS IF: the time component is present but constant, i.e. bolted on rather
    than derived from the tube's actual slant.
    """
    P = _straight_guess([0.0, 0.0, 0.0], [10.0, 0.0, 8.0], 9)
    magnitudes = []
    for speed in (0.3, 1.0, 3.0):
        obs = [{"pos0": [5.0, 0.0], "vel": [0.0, speed], "r": 1.0}]
        rows = _koz_rows(P, obs, n_seg=4)
        magnitudes.append(float(np.abs(rows["normals"][:, DIM - 1]).max()))

    assert magnitudes[0] < magnitudes[1] < magnitudes[2], (
        f"time component does not grow with obstacle speed: {magnitudes}"
    )


def test_one_half_space_per_segment_and_obstacle():
    """All control points of a segment must share one plane per obstacle.

    This is the convex-hull certificate: one half-space satisfied by every
    control point bounds the whole curve segment. Per-control-point planes prove
    nothing about the segment between them.

    FAILS IF: control points within a segment carry different normals or
    different bounds for the same obstacle -- the previous builder's behaviour.
    """
    P = _straight_guess([0.0, 0.0, 0.0], [10.0, 0.0, 8.0], 9)
    rows = _exact_rows(P, MOVING, n_seg=4)

    groups = {}
    for i in range(len(rows["lbs"])):
        key = (int(rows["seg"][i]), int(rows["obs"][i]))
        groups.setdefault(key, []).append(i)

    assert groups, "no rows to group"
    multi = [k for k, v in groups.items() if len(v) > 1]
    assert multi, "each segment/obstacle produced only one row; nothing to compare"

    for key, idxs in groups.items():
        normals = rows["normals"][idxs]
        bounds = rows["lbs"][idxs]
        assert np.allclose(normals, normals[0], atol=1e-12), (
            f"segment/obstacle {key} has differing normals across its control points"
        )
        assert np.allclose(bounds, bounds[0], atol=1e-12), (
            f"segment/obstacle {key} has differing bounds across its control points"
        )


def test_half_space_actually_supports_the_tube():
    """The plane must have the WHOLE tube on the far side, not just touch it.

    Samples the true obstacle surface densely in time and checks every sample is
    on the forbidden side of the plane.

    FAILS IF: the plane cuts through the obstacle, which is what a normal with a
    dropped time component does once the obstacle moves -- it would permit
    trajectories that pass straight through.
    """
    P = _straight_guess([0.0, 0.0, 0.0], [10.0, 0.0, 8.0], 9)
    obs = MOVING[0]
    rows = _exact_rows(P, MOVING, n_seg=4)

    worst = -np.inf
    for i in range(len(rows["lbs"])):
        n = rows["normals"][i]
        lb = rows["lbs"][i]
        for t in np.linspace(0.0, 8.0, 60):
            cx = obs["pos0"][0] + obs["vel"][0] * t
            cy = obs["pos0"][1] + obs["vel"][1] * t
            for ang in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
                pt = np.array([
                    cx + obs["r"] * np.cos(ang),
                    cy + obs["r"] * np.sin(ang),
                    t,
                ])
                # Forbidden side means n.pt <= lb. Positive excess is a breach.
                worst = max(worst, float(n @ pt - lb))

    assert worst <= 1e-9, (
        f"a half-space cuts through the obstacle by {worst:.4e}; it does not "
        "support the tube, so satisfying it does not imply avoiding the obstacle"
    )


@pytest.mark.parametrize("n_seg", [2, 4, 8])
def test_more_segments_cannot_shrink_the_feasible_set(n_seg):
    """Subdivision produces tighter, nested hulls, so raising the segment count
    can only relax the constraint set -- never tighten it.

    Recorded as a property the geometry must satisfy. Checked here in the weak
    form that a solve at each segment count still produces rows and a finite
    result; the strong form belongs with B3's benchmark sweep.

    FAILS IF: a segment count produces no obstacle rows at all, which would mean
    the constraint silently vanished.
    """
    P = _straight_guess([0.0, 0.0, 0.0], [10.0, 0.0, 8.0], 9)
    rows = _koz_rows(P, MOVING, n_seg=n_seg)
    assert len(rows["lbs"]) > 0
    assert np.all(np.isfinite(rows["lbs"]))
    assert np.all(np.isfinite(rows["normals"]))


def test_moving_obstacle_scenario_converges_certified():
    """End-to-end on moving obstacles: converge, clear every obstacle, and carry
    the hull certificate.

    `original` has three obstacles moving at up to 0.86 units/sec, so a builder
    with a zero time coefficient cannot represent their geometry at all.

    FAILS IF: the run does not converge, penetrates, or converges while its
    control points violate the half-spaces they generate.
    """
    from spacetime_bezier.scenarios import SCENARIO_MAP

    fn, _ = SCENARIO_MAP["original"]
    sc = fn()
    _, info = optimize_spacetime(
        N=8, dim=3, p_start=sc["start"], p_end=sc["end"],
        obstacles=sc["obstacles"], n_seg=4, max_iter=200, tol=1e-6,
        scp_trust_radius=0.5, min_dt=0.1, verbose=False,
        init_curve=sc.get("init_curve"),
    )
    assert info["converged"] == 1.0, f"did not converge (stop={info['stop_reason']})"
    assert info["min_clearance"] > 0.0, f"penetrates by {-info['min_clearance']:.4f}"
    assert info["koz_violation_reference"] <= 1e-6, (
        f"converged with certificate violation {info['koz_violation_reference']:.3e}"
    )


@pytest.mark.xfail(
    reason="`wall` is unsolved: one plane per segment cannot pass a segment's "
           "control points on opposite sides of the same obstacle, and the "
           "literature's fix is more segments, which plateaus here at -0.09. "
           "This records the gap rather than hiding it -- item B6 (procedural "
           "seeds / multi-start) is the next lever.",
    strict=True,
)
def test_wall_scenario_waits_for_the_wall():
    """The wall vanishes at t = 5 and the curve should wait for it.

    Kept as a strict expected-failure so that if a later change makes it pass,
    the suite says so instead of staying silent.
    """
    from spacetime_bezier.scenarios import SCENARIO_MAP

    fn, _ = SCENARIO_MAP["wall"]
    sc = fn()
    best = max(
        optimize_spacetime(
            N=8, dim=3, p_start=sc["start"], p_end=sc["end"],
            obstacles=sc["obstacles"], n_seg=ns, max_iter=200, tol=1e-6,
            scp_trust_radius=0.5, min_dt=0.1, verbose=False,
            init_curve=sc.get("init_curve"),
        )[1]["min_clearance"]
        for ns in (2, 8, 12, 24)
    )
    assert best > 0.0, f"best configuration still penetrates by {-best:.4f}"
