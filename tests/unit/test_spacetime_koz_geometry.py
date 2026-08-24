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

from spacetime_bezier.geometry import compute_min_clearance
from spacetime_bezier.optimize import optimize_spacetime

DIM = 3  # x, y, t


def _exact_rows(P, obstacles, n_seg):
    """The EXACT half-spaces -- the rows the certificate is evaluated with.

    Not the rows handed to the solver. Those carry a rotation term that makes
    them self-consistent under the step, and they are documented as NOT
    conservative; the two properties below hold only for the exact rows.

    The row width is read off `P`, not off the module-level `DIM`: item B11 runs
    the same builder at four coordinates, and a helper that hardcodes three would
    reshape those rows into nonsense instead of failing.
    """
    P = np.asarray(P, dtype=float)
    dim = P.shape[1]
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
        "normals": np.asarray(normals, dtype=float).reshape(-1, dim),
        "lbs": np.asarray(lbs, dtype=float),
        "seg": np.asarray(seg),
        "cp": np.asarray(cp),
        "obs": np.asarray(obs),
    }


def _exact_certificate(P, obstacles, n_seg):
    """Hull-certificate violation recomputed from control points alone.

    Sum of positive violations of the EXACT half-spaces over every subdivided
    control point: `max(0, lb - n.q)` where `q = (A_seg @ P)[cp]`. This is the
    same quantity the Rust layer reports as `koz_violation_reference`, derived
    here independently so a test can contradict the solver instead of echoing it.

    Two independent computations must agree, and that is what caught the defect
    this function was written for. Until 2026-08-19 the Rust layer exported the
    violation of the loop's final *reference* point, not of the control points it
    actually returned -- a different trajectory whenever the best-feasible
    fallback fires. Measured on `diverse` N8_seg4: reported 1.61634 against a true
    2.26425 at the returned points, a 40% understatement of the quantity backing
    the paper's guarantee. This recomputation agrees with the fixed export to six
    digits on both a certified run (0.000) and an uncertified one (2.264249); the
    pre-fix value survives as `koz_violation_last_reference` for provenance.
    """
    from orbital_docking.de_casteljau import segment_matrices_equal_params

    P = np.asarray(P, dtype=float)
    rows = _exact_rows(P, obstacles, n_seg)
    a_list = segment_matrices_equal_params(P.shape[0] - 1, n_seg)

    violation = 0.0
    for normal, lb, seg_idx, cp_idx in zip(
        rows["normals"], rows["lbs"], rows["seg"], rows["cp"]
    ):
        q = (np.asarray(a_list[int(seg_idx)], dtype=float) @ P)[int(cp_idx)]
        violation += max(0.0, float(lb) - float(normal @ q))
    return violation


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


# `MOVING` starts ON the trajectory line and recedes from it, so every row's
# spatial normal has a NEGATIVE dot product with the obstacle velocity.
# `APPROACHING` starts to one side and closes in, which puts that dot product
# positive on some rows. The distinction is load-bearing -- see the test below.
APPROACHING = [{"pos0": [5.0, -5.0], "vel": [0.0, 1.2], "r": 1.0}]


@pytest.mark.parametrize("obstacles", [MOVING, APPROACHING], ids=["receding", "approaching"])
def test_half_space_actually_supports_the_tube(obstacles):
    """The plane must have the WHOLE tube on the far side, not just touch it.

    Samples the true obstacle surface densely in time and checks every sample is
    on the forbidden side of the plane.

    FAILS IF: the plane cuts through the obstacle, which is what a normal with a
    dropped time component does once the obstacle moves -- it would permit
    trajectories that pass straight through.

    The `approaching` case exists because the `receding` one CANNOT fail for that
    reason, which was measured on 2026-08-19 rather than assumed. Along the
    centreline the emitted normal satisfies n_t = -n_s.vel exactly, so the plane's
    value on the tube is constant in time; zeroing n_t makes it vary as
    (n_s.vel) t instead. When n_s.vel is negative on every row -- which is what a
    receding obstacle gives -- that variation only ever lowers the value, the
    maximum stays at t=0, and the worst surface excess is unchanged: -0.020345
    both as built and with the time column zeroed. On the approaching obstacle
    the same mutation gives -0.0647 as built against +3.471 zeroed. Only the
    second version is evidence about the time coefficient.
    """
    P = _straight_guess([0.0, 0.0, 0.0], [10.0, 0.0, 8.0], 9)
    obs = obstacles[0]
    rows = _exact_rows(P, obstacles, n_seg=4)

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
    P_opt, info = optimize_spacetime(
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

    # Recompute both verdicts from the RETURNED control points instead of
    # trusting `info`. Asserting only on reported fields is a check that cannot
    # fail: a solver handing its input straight back, with converged=1 and a
    # certificate of 0, passed the three assertions above -- and reported the
    # +0.8348 of the initial guess, the exact signature of the `9b9c3d3`
    # best-iterate defect.
    independent_clearance = compute_min_clearance(
        P_opt, sc["obstacles"], dim=3, n_eval=20_001
    )
    assert independent_clearance > 0.0, (
        f"reported clearance {info['min_clearance']:.4f} but the returned curve "
        f"penetrates by {-independent_clearance:.4f}"
    )
    assert independent_clearance == pytest.approx(info["min_clearance"], abs=5e-3), (
        f"reported clearance {info['min_clearance']:.4f} disagrees with the "
        f"returned curve's {independent_clearance:.4f}"
    )
    assert _exact_certificate(P_opt, sc["obstacles"], n_seg=4) <= 1e-6, (
        "the returned control points violate the half-spaces they generate"
    )


def test_wall_scenario_waits_for_the_wall():
    """The wall vanishes at t = 5, and the curve waits for it.

    This was a strict xfail until 2026-08-19, recording `wall` as unsolved and
    naming multi-start (B6) as the next lever. That was wrong. `wall` is
    solvable; what blocked it was the elastic penalty weight, which was pinned
    at 100 inside the Rust binding and unreachable from Python. Below this
    scenario's exact-penalty threshold a penetrating curve is genuinely the
    cheaper answer, so the solver returned one and every sweep over degree and
    segment count plateaued near -0.09.

    The old xfail swept n_seg but could not vary the weight, so it could never
    have discovered this -- it asserted the very limitation it was caused by.

    FAILS IF: the run does not converge, the returned curve penetrates, or its
    control points violate the half-spaces they generate.
    """
    from spacetime_bezier.scenarios import SCENARIO_MAP, scenario_elastic_weight

    fn, _ = SCENARIO_MAP["wall"]
    sc = fn()
    n_seg = 16
    P_opt, info = optimize_spacetime(
        N=10, dim=3, p_start=sc["start"], p_end=sc["end"],
        obstacles=sc["obstacles"], n_seg=n_seg, max_iter=200, tol=1e-6,
        scp_trust_radius=0.5, min_dt=0.1, verbose=False,
        elastic_weight=scenario_elastic_weight("wall"),
        init_curve=sc.get("init_curve"),
    )
    assert info["converged"] == 1.0, f"did not converge (stop={info['stop_reason']})"
    assert info["min_clearance"] > 0.0, f"penetrates by {-info['min_clearance']:.4f}"

    independent_clearance = compute_min_clearance(
        P_opt, sc["obstacles"], dim=3, n_eval=20_001
    )
    assert independent_clearance > 0.0, (
        f"reported {info['min_clearance']:.4f} but the returned curve penetrates "
        f"by {-independent_clearance:.4f}"
    )
    assert _exact_certificate(P_opt, sc["obstacles"], n_seg=n_seg) <= 1e-6, (
        "the returned control points violate the half-spaces they generate"
    )


def test_wall_penetrates_below_its_penalty_threshold():
    """The companion fact: at the old fixed weight of 100, `wall` really does
    penetrate.

    This is why the scenario was misread as infeasible for months, and it is
    what makes the weight a modelling decision rather than a tuning knob.

    FAILS IF: weight 100 already solves `wall`, which would mean the threshold
    story above is wrong and the fix is something else.
    """
    from spacetime_bezier.scenarios import SCENARIO_MAP

    fn, _ = SCENARIO_MAP["wall"]
    sc = fn()
    _, info = optimize_spacetime(
        N=10, dim=3, p_start=sc["start"], p_end=sc["end"],
        obstacles=sc["obstacles"], n_seg=16, max_iter=200, tol=1e-6,
        scp_trust_radius=0.5, min_dt=0.1, verbose=False,
        elastic_weight=100.0,
        init_curve=sc.get("init_curve"),
    )
    assert info["min_clearance"] < 0.0, (
        "weight 100 now clears `wall`; the penalty-threshold explanation for the "
        "old infeasibility record needs revisiting"
    )


# --- Item B11: three spatial coordinates plus time -------------------------
#
# Same builder, same solver, one more column. Nothing in Rust changed to make
# these pass; `dim` has always come from the array shape. What these tests
# exclude is the opposite claim -- that the code only *looks* dimension-generic
# and quietly drops the third spatial coordinate or the time component once the
# row width grows.

# The obstacle APPROACHES the trajectory line. That is not cosmetic: the
# support check below is only sensitive to a dropped time coefficient when some
# row's spatial normal has a positive dot product with the obstacle velocity.
# Measured on this geometry -- worst surface excess as built -0.081, with the
# time column zeroed +3.599, with the z column zeroed +0.219. Measured on a
# RECEDING obstacle (pos0 [5,0,0], vel [0,0.8,0.3]) -- as built -0.018, time
# column zeroed -0.018, i.e. no signal at all. See the note on
# `test_half_space_actually_supports_the_tube`.
MOVING_3D = [{"pos0": [5.0, -5.0, -2.0], "vel": [0.0, 1.2, 0.5], "r": 1.0}]


def test_half_space_supports_the_tube_in_three_spatial_dimensions():
    """The four-coordinate rows must still support the true obstacle tube.

    Samples the moving sphere's surface densely in time and in both angles, and
    checks every sample is on the forbidden side of every emitted plane. This is
    the three-dimensional twin of
    `test_half_space_actually_supports_the_tube`, and it is the check that a
    dropped coordinate cannot survive: if the builder ignored z, or emitted a
    three-wide normal that the caller reshaped, the plane would slice the sphere.

    FAILS IF: any surface sample lands on the free side of a plane by more than
    solver noise, or the rows come back with a width other than four.
    """
    P = _straight_guess([0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 8.0], 9)
    rows = _exact_rows(P, MOVING_3D, n_seg=4)
    obs = MOVING_3D[0]

    assert rows["normals"].shape[1] == 4, (
        f"rows are {rows['normals'].shape[1]} wide at four coordinates"
    )
    assert len(rows["lbs"]) > 0, "no obstacle rows were generated at all"
    assert np.abs(rows["normals"][:, -1]).max() > 1e-6, (
        "every four-coordinate half-space has a zero time coefficient"
    )

    worst = -np.inf
    thetas = np.linspace(0.0, np.pi, 12)
    phis = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    for normal, lb in zip(rows["normals"], rows["lbs"]):
        for t in np.linspace(0.0, 8.0, 40):
            centre = np.asarray(obs["pos0"], float) + np.asarray(obs["vel"], float) * t
            for th in thetas:
                for ph in phis:
                    offset = obs["r"] * np.array([
                        np.sin(th) * np.cos(ph),
                        np.sin(th) * np.sin(ph),
                        np.cos(th),
                    ])
                    pt = np.append(centre + offset, t)
                    # Forbidden side means n.pt <= lb; positive excess is a breach.
                    worst = max(worst, float(normal @ pt - float(lb)))

    assert worst <= 1e-9, (
        f"a half-space cuts through the sphere by {worst:.4e} at four "
        "coordinates; satisfying it does not imply avoiding the obstacle"
    )


def test_fence3d_climbs_and_is_certified():
    """Item B11 end to end: three spatial coordinates plus time, converged,
    clearing, certified, and cross-checked against the true obstacle motion.

    The second half is the part that makes the extra dimension evidence rather
    than decoration. The fence spans the full y extent of the corridor, so the
    curve can only get past it by leaving that span sideways or by leaving the
    fence's z extent. This asserts it does the latter *while still inside the y
    span* -- the avoidance happens in the third spatial coordinate, and a
    two-coordinate run of the same geometry could not reproduce it. (Measured:
    the identical scenario without z penetrates by 0.47 to 0.78 and converges at
    no config.)

    FAILS IF: the run does not converge; the returned curve penetrates; its
    control points violate the half-spaces they generate; `compute_min_clearance`
    -- computed here from the returned control points and the obstacles' true
    motion, not from anything the solver reported -- disagrees with the reported
    clearance; or the curve crosses the fence by going around it in y, which
    would mean the third coordinate carried nothing.
    """
    from spacetime_bezier.scenarios import SCENARIO_MAP
    from spacetime_bezier.geometry import bezier_curve

    fn, _ = SCENARIO_MAP["fence3d"]
    sc = fn()
    assert len(sc["start"]) == 4, "fence3d must have three spatial coordinates plus time"
    n_seg = 2

    P_opt, info = optimize_spacetime(
        N=8, dim=4, p_start=sc["start"], p_end=sc["end"],
        obstacles=sc["obstacles"], n_seg=n_seg, max_iter=200, tol=1e-6,
        scp_prox_weight=0.3, scp_trust_radius=0.5, min_dt=0.1, verbose=False,
        init_curve=sc.get("init_curve"),
    )

    assert P_opt.shape[1] == 4, f"solver returned {P_opt.shape[1]} coordinates"
    assert info["converged"] == 1.0, f"did not converge (stop={info['stop_reason']})"
    assert info["min_clearance"] > 0.0, f"penetrates by {-info['min_clearance']:.4f}"

    independent_clearance = compute_min_clearance(
        P_opt, sc["obstacles"], dim=4, n_eval=20_001
    )
    assert independent_clearance > 0.0, (
        f"reported {info['min_clearance']:.4f} but the returned curve penetrates "
        f"by {-independent_clearance:.4f}"
    )
    assert independent_clearance == pytest.approx(info["min_clearance"], abs=5e-3), (
        f"reported clearance {info['min_clearance']:.4f} disagrees with the "
        f"returned curve's {independent_clearance:.4f}"
    )
    assert _exact_certificate(P_opt, sc["obstacles"], n_seg=n_seg) <= 1e-6, (
        "the returned control points violate the half-spaces they generate"
    )

    # Where the curve crosses the advancing fence plane, and how it got past.
    fence = [o for o in sc["obstacles"] if o["name"].startswith("F")]
    y_lo = min(o["pos0"][1] for o in fence)
    y_hi = max(o["pos0"][1] for o in fence)
    fence_r = max(o["r"] for o in fence)
    fence_z = fence[0]["pos0"][2]
    fence_x0 = fence[0]["pos0"][0]
    fence_vx = fence[0]["vel"][0]

    pts = bezier_curve(P_opt, 20_001)
    gap = pts[:, 0] - (fence_x0 + fence_vx * pts[:, 3])
    assert gap[0] < 0.0 < gap[-1], (
        "the curve does not start behind the fence and end ahead of it, so "
        "there is no crossing to inspect"
    )
    crossing = pts[int(np.argmin(np.abs(gap)))]

    assert y_lo <= crossing[1] <= y_hi, (
        f"the curve crossed at y={crossing[1]:.3f}, outside the fence's span "
        f"[{y_lo}, {y_hi}] -- it went around, so the third spatial coordinate "
        "is not what got it past"
    )
    assert abs(crossing[2] - fence_z) > fence_r, (
        f"the curve crossed at z={crossing[2]:.3f}, inside the fence's z extent "
        f"{fence_z} +/- {fence_r}"
    )
