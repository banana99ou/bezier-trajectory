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

from spacetime_bezier.geometry import compute_min_clearance, obstacle_array_bundle
from spacetime_bezier.optimize import optimize_spacetime

DIM = 3  # x, y, t


def _bundle(P, obstacles):
    """Lifted obstacle control points for the Rust call.

    A legacy ``pos0``/``vel`` obstacle with no window used to mean "always"; the
    active window is now intrinsic to the control points, so "always" has to be
    made explicit. The trajectory's own time span is the faithful translation:
    outside it the obstacle could not constrain anything anyway.
    """
    P = np.asarray(P, dtype=float)
    t_lo, t_hi = float(P[:, -1].min()), float(P[:, -1].max())
    filled = []
    for o in obstacles:
        if "control_points" in o:
            filled.append(o)
            continue
        o = dict(o)
        o.setdefault("t_start", min(0.0, t_lo))
        o.setdefault("t_end", t_hi)
        filled.append(o)
    return obstacle_array_bundle(filled, P.shape[1] - 1)


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
    ctrl, radii = _bundle(P, obstacles)
    # Unpacked by name so a future widening of the tuple fails loudly here rather
    # than silently mis-assigning a column.
    (
        normals, lbs, seg, cp, obs, comp, sta, _rho, _sound,
        _dropped, _dropped_shadow, _unsound,
    ) = bezier_opt.spacetime_koz_rows_exact(
        p=P, obstacle_ctrl=ctrl, obstacle_r=radii, n_seg=n_seg,
    )
    # No station was passed, so every row must be a body row. A shadow row here
    # would mean the generator list grew one nobody asked for.
    assert np.all(np.asarray(sta) == -1)
    return {
        "normals": np.asarray(normals, dtype=float).reshape(-1, dim),
        "lbs": np.asarray(lbs, dtype=float),
        "seg": np.asarray(seg),
        "cp": np.asarray(cp),
        "obs": np.asarray(obs),
        "comp": np.asarray(comp),
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
    ctrl, radii = _bundle(P, obstacles)

    ctx = bezier_opt.SpacetimeScpContext(
        p_init=P,
        obstacle_ctrl=ctrl,
        obstacle_r=radii,
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
    (_p, info, seg, cp, obs, _it, normals, supports, centers, lbs, margins, _slack) = out
    return {
        "seg": np.asarray(seg),
        "cp": np.asarray(cp),
        "obs": np.asarray(obs),
        "comp": np.asarray(info["koz_component"]),
        "normals": np.asarray(normals, dtype=float).reshape(-1, DIM),
        "lbs": np.asarray(lbs, dtype=float),
        "supports": np.asarray(supports, dtype=float).reshape(-1, DIM),
    }


def _straight_guess(start, end, n_cp):
    return np.linspace(np.asarray(start, float), np.asarray(end, float), n_cp)


MOVING = [{"pos0": [5.0, 0.0], "vel": [0.0, 1.2], "r": 1.0}]
STATIONARY = [{"pos0": [5.0, 0.0], "vel": [0.0, 0.0], "r": 1.0}]
# `MOVING` recedes fast enough that at 8 segments every segment centroid is
# further from its centreline than `r_m + E + trust*sqrt(dim)` -- the obstacle is
# genuinely OUT OF REACH and emitting no row is the correct answer, not a
# vanished constraint. Measured: closest centroid 2.68 against a reach of 2.67.
# A test about subdivision needs an obstacle that stays in reach at every count.
NEARBY = [{"pos0": [5.0, 0.9], "vel": [0.0, 0.15], "r": 1.0}]


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


def test_one_half_space_per_segment_obstacle_and_component():
    """All control points of a segment must share one plane per clipped component.

    This is the convex-hull certificate: one half-space satisfied by every
    control point bounds the whole curve segment. Per-control-point planes prove
    nothing about the segment between them.

    **The component index joins the key.** One obstacle can approach one segment
    more than once -- the band is cut at every interior local maximum of the
    centreline's distance to the segment centroid, one wall per approach -- and
    grouping on (segment, obstacle) alone would compare two genuinely different
    planes and call the difference a defect. What must hold, and does, is that
    within one approach every control point sees the same normal and the same
    bound. (The index kept the name `component` from the connected-component
    grouping it replaced on 2026-08-26.)

    FAILS IF: control points within a segment carry different normals or
    different bounds for the same obstacle and component -- the pre-G2 builder's
    behaviour.
    """
    P = _straight_guess([0.0, 0.0, 0.0], [10.0, 0.0, 8.0], 9)
    rows = _exact_rows(P, MOVING, n_seg=4)

    groups = {}
    for i in range(len(rows["lbs"])):
        key = (int(rows["seg"][i]), int(rows["obs"][i]), int(rows["comp"][i]))
        groups.setdefault(key, []).append(i)

    assert groups, "no rows to group"
    multi = [k for k, v in groups.items() if len(v) > 1]
    assert multi, "each segment/obstacle produced only one row; nothing to compare"

    for key, idxs in groups.items():
        normals = rows["normals"][idxs]
        bounds = rows["lbs"][idxs]
        assert np.allclose(normals, normals[0], atol=1e-12), (
            f"segment/obstacle/component {key} has differing normals across its "
            "control points"
        )
        assert np.allclose(bounds, bounds[0], atol=1e-12), (
            f"segment/obstacle/component {key} has differing bounds across its "
            "control points"
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
    rows = _koz_rows(P, NEARBY, n_seg=n_seg)
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
    # Read off the lifted control points: the active window is intrinsic to them
    # and a `pos0`/`vel` pair is no longer carried.
    fence = [o for o in sc["obstacles"] if o["name"].startswith("F")]
    cps = [np.asarray(o["control_points"], dtype=float) for o in fence]
    y_lo = min(float(c[0][1]) for c in cps)
    y_hi = max(float(c[0][1]) for c in cps)
    fence_r = max(float(o["radius"]) for o in fence)
    fence_z = float(cps[0][0][2])
    fence_t0 = float(cps[0][0][3])
    fence_x0 = float(cps[0][0][0])
    fence_vx = (float(cps[0][-1][0]) - fence_x0) / (float(cps[0][-1][3]) - fence_t0)

    pts = bezier_curve(P_opt, 20_001)
    gap = pts[:, 0] - (fence_x0 + fence_vx * (pts[:, 3] - fence_t0))
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


# ---------------------------------------------------------------------------
# The clipped-KOZ-volume construction, on four cases whose answers are known by
# hand. Planar geometry (one spatial coordinate plus time) so the picture can be
# drawn and checked by eye; the builder is dimension-generic and the tests above
# exercise three and four coordinates.
#
# The hairpin's time coordinate is affine in the parameter (2, 4, 6, 8), which is
# what makes it a legal lifted obstacle rather than a decorative curve, and it is
# symmetric about t = 5. That symmetry is what several assertions below stand on.
# ---------------------------------------------------------------------------

PANEL_STRAIGHT = np.array([[[0.0, 0.0], [8.0, 0.0]]])
PANEL_HAIRPIN = np.array([[[7.0, 2.0], [-2.0, 4.0], [-2.0, 6.0], [7.0, 8.0]]])
PANEL_AXIS = 5.0  # the hairpin's axis of symmetry, in the time coordinate


def _panel_seg(cx, cy, half):
    """Four control points on a short segment centred at ``(cx, cy)``."""
    return np.array([
        [cx - half, cy], [cx - half / 3, cy], [cx + half / 3, cy], [cx + half, cy]
    ])


def _panel_walls(Q, ctrl, r_m, trust):
    """The exact walls for one segment against one obstacle, grouped by component.

    The lower bound IS the offset ``b``: with one segment the De Casteljau matrix
    is the identity, so the row's ``grad . p`` is ``n . Q_k`` and
    ``lb = n . Q_k - g_k = b`` exactly.
    """
    Q = np.asarray(Q, dtype=float)
    (
        normals, lbs, _seg, _cp, _obs, comp, _sta, rho, sound,
        dropped, _dropped_shadow, unsound,
    ) = bezier_opt.spacetime_koz_rows_exact(
        p=Q, obstacle_ctrl=ctrl, obstacle_r=np.array([r_m]), n_seg=1,
        trust_radius=trust,
        # PINNED, not inherited. These panels demonstrate the CONSTRUCTION's own
        # floor -- the clip radius at the obstacle radius, PAPER_1 statement (7)
        # -- and the reach floor (`sound_clip`, statement (8)) is a larger,
        # separate one that hides it: with the reach floor on, panel B3's radius
        # is 0.907107 instead of 0.900000 and the obstacle-radius floor never
        # binds. `sound_clip` became the DEFAULT 2026-08-31, so this has to be
        # written down rather than inherited; a panel whose subject is decided by
        # a default is a panel that silently changes subject when the default
        # moves, which is the whole lesson of that day.
        sound_clip=False,
    )
    normals = np.asarray(normals, dtype=float).reshape(-1, 2)
    lbs = np.asarray(lbs, dtype=float)
    comp = np.asarray(comp, dtype=int)
    rho = np.asarray(rho, dtype=float)
    c = Q.mean(axis=0)
    walls = []
    for j in sorted(set(comp.tolist())):
        m = comp == j
        n = normals[m][0]
        b = float(lbs[m][0])
        walls.append({
            "component": j,
            "n": n,
            "b": b,
            "rho": float(rho[m][0]),
            "sound": bool(np.asarray(sound)[m][0]),
            "centroid_margin": float(n @ c - b),
            "margins": normals[m] @ Q.T,
        })
    return walls, int(dropped), int(unsound), c


def _bez(ctrl, t):
    P = np.repeat(np.asarray(ctrl, dtype=float)[None], len(t), axis=0)
    while P.shape[1] > 1:
        s = t[:, None, None]
        P = (1.0 - s) * P[:, :-1] + s * P[:, 1:]
    return P[:, 0]


def _clipped_volume_points(ctrl, r_m, c, rho, n_tau=1400, n_dir=360):
    """Dense sample of ``KOZ ∩ Ball(c, rho)`` -- computed here, not by the solver."""
    A = _bez(ctrl, np.linspace(0.0, 1.0, n_tau))
    th = np.linspace(0.0, 2.0 * np.pi, n_dir, endpoint=False)
    ring = np.stack([np.cos(th), np.sin(th)], axis=1)
    pts = []
    for rad in np.linspace(0.0, r_m, 11):
        pts.append((A[:, None, :] + rad * ring[None, :, :]).reshape(-1, 2))
    Z = np.concatenate(pts)
    return Z[np.linalg.norm(Z - c, axis=1) <= rho]


def _centreline_distance(ctrl, c, n_tau=200_001):
    A = _bez(ctrl, np.linspace(0.0, 1.0, n_tau))
    i = int(np.argmin(np.linalg.norm(A - c, axis=1)))
    return float(np.linalg.norm(A[i] - c)), A[i]


def test_panel_a_straight_obstacle_is_exact():
    """A straight tube's wall is its own surface, to the last decimal.

    The offset is a rigorous De Casteljau ceiling on the component's support, and
    on a straight tube with the normal perpendicular to it the ceiling is not
    merely tight but EXACT: every subdivided control point has the same `n . g`.
    So this panel pins the construction against arithmetic, not against a
    tolerance.

    FAILS IF: the offset is not the tube's own radius, or the centroid margin is
    not the true clearance -- which is what happens if the wall is built against
    anything other than the clipped volume.
    """
    Q = np.array([[2.0, 2.0], [2.6, 2.0], [3.2, 2.0], [3.8, 2.0]])
    walls, dropped, _unsound, c = _panel_walls(Q, PANEL_STRAIGHT, r_m=0.6, trust=0.5)

    assert dropped == 0
    assert len(walls) == 1, f"a straight tube cannot split; got {len(walls)} walls"
    w = walls[0]
    assert w["n"][1] == pytest.approx(1.0, abs=1e-12), f"normal {w['n']} is not (0, 1)"
    assert w["n"][0] == pytest.approx(0.0, abs=1e-12), f"normal {w['n']} is not (0, 1)"
    assert w["b"] == pytest.approx(0.6, abs=1e-9), (
        f"offset {w['b']:.9f} is not the tube's own surface at y = r_m = 0.6"
    )
    assert w["centroid_margin"] == pytest.approx(1.4, abs=1e-9), (
        f"centroid margin {w['centroid_margin']:.9f} is not d - r_m = 2.0 - 0.6"
    )


def test_panel_b1_the_ball_severs_the_bend_into_two_walls():
    """Two lumps, two walls, and they must be exact mirrors.

    The segment centroid sits on the hairpin's axis of symmetry, so the clip ball
    catches the two arms as mirror-image components. Reflecting through the axis
    maps one component onto the other, so it must map one wall onto the other:
    same normal in the spatial coordinate, opposite in time, and
    ``b_upper = b_lower - 2 * axis * n_time(lower)``. That identity cannot hold by
    accident, which is what makes it worth asserting instead of the raw numbers.

    FAILS IF: the parameter intervals are fused back into one -- the pre-2026-08-26
    behaviour, where `clip_band` took the min and max over qualifying samples and
    a bend could never produce more than one wall.
    """
    Q = _panel_seg(3.6, PANEL_AXIS, 0.6)
    walls, dropped, _unsound, c = _panel_walls(Q, PANEL_HAIRPIN, r_m=0.9, trust=0.5)

    assert dropped == 0
    assert len(walls) == 2, (
        f"the ball should sever the bend into two lumps; got {len(walls)} wall(s)"
    )
    # 1e-7, not machine epsilon. The geometry is exactly symmetric; the SOLVE for
    # the nearest parameter is not. Newton stops on `next - s` and on the distance
    # failing to improve, and on a quadratic minimum that last comparison is
    # rounding-limited, so the two arms can stop one step apart -- about
    # sqrt(eps) in the parameter, which is what shows up here. It is the
    # parameter solve's floor, not the construction's.
    lower, upper = sorted(walls, key=lambda w: w["n"][1])
    assert lower["n"][0] == pytest.approx(upper["n"][0], abs=1e-7), (
        "the two normals disagree in the spatial coordinate, so they are not mirrors"
    )
    assert lower["n"][1] == pytest.approx(-upper["n"][1], abs=1e-7), (
        "the two normals do not have opposite time components"
    )
    assert upper["b"] == pytest.approx(
        lower["b"] - 2.0 * PANEL_AXIS * lower["n"][1], abs=1e-5
    ), "the two offsets are not reflections of each other through the axis"
    assert lower["centroid_margin"] == pytest.approx(
        upper["centroid_margin"], abs=1e-6
    ), "the segment sits on the axis, so the two margins must be equal"
    assert lower["centroid_margin"] > 0.0, (
        "the segment sits in the corridor between the arms, so both margins are positive"
    )


def test_panel_b2_a_wrapping_bend_gets_one_wall_per_approach():
    """The bend wraps the centroid; cutting at the crest gives two walls that fit.

    The clipped volume here is ONE connected lump curled around the centroid, and
    the centroid sits inside its convex hull -- so a single wall against the lump
    is non-separating BY THEOREM, whatever its normal. Until 2026-08-26 that is
    what this segment got: one wall, centroid margin -1.1543, on a segment whose
    centroid is OUTSIDE the keep-out zone. A clear segment reported as 1.15 deep,
    a row unsatisfiable inside the default trust radius (it needed ~1.04 against
    0.5), and elastic slack absorbing a violation that did not exist.

    The band is now cut at the interior local maximum of |gamma(s) - c| -- the
    crest between the obstacle's two approaches -- and each approach gets its own
    wall. Same clip radius, same clipped volume, same offsets rule; only the
    grouping changed.

    FAILS IF: the builder returns to one wall for the whole lump (walls != 2), or
    either wall stops clearing by more than the ceiling's conservatism -- the
    fused wall's margin was -1.15, an order of magnitude below the bound asserted
    here, so a regression cannot hide inside the tolerance.
    """
    Q = _panel_seg(2.2, 5.35, 0.55)
    walls, dropped, _unsound, c = _panel_walls(Q, PANEL_HAIRPIN, r_m=0.9, trust=0.5)

    assert dropped == 0
    d, _f = _centreline_distance(PANEL_HAIRPIN[0], c)
    assert d > 0.9, (
        f"this panel needs the centroid OUTSIDE the keep-out zone; d = {d:.4f}"
    )
    assert len(walls) == 2, (
        f"one wall per approach: the hairpin approaches this segment twice, got "
        f"{len(walls)} wall(s)"
    )
    # The two walls face opposite sides of the mouth: one normal tilts toward
    # +t, the other toward -t.
    assert walls[0]["n"][1] * walls[1]["n"][1] < 0.0
    # Both margins sit far above the fused wall's -1.1543. The lower one may dip
    # slightly negative: the offset is a rigorous De Casteljau CEILING on the
    # support, and its conservatism (~0.16 here) lands in the margin. That is a
    # step-size cost, not a correctness cost.
    for w in walls:
        assert w["centroid_margin"] > -0.3, (
            f"wall {w['component']} margin {w['centroid_margin']:+.4f} is back in "
            "fused-wall territory (-1.15); the cut has regressed"
        )
    assert max(w["centroid_margin"] for w in walls) > 0.0, (
        "at least one approach's wall should clear the centroid outright"
    )

def test_panel_b3_the_clip_radius_is_floored_at_the_obstacle_radius():
    """Centroid inside the keep-out zone: the floor fires and the direction exists.

    Three things at once, and each of them was a defect before 2026-08-26.

    The clip radius is floored at the obstacle radius, so the ball is not smaller
    than the obstacle it is clipping. Unfloored it would be ``d = 0.35`` against a
    tube of radius ``0.9``: the ball lies wholly inside the tube, the wall lands
    on the ball, and a 0.55-deep penetration reports as a margin of exactly zero.

    The normal exists. ``y*`` collapses onto ``c`` here, so ``(c - y*)/|c - y*|``
    is ``0/0`` -- but ``y*`` always sat on the ray from the nearest centreline
    point out to the centroid, so the direction is that ray and it does not
    vanish. By symmetry it must be exactly ``(1, 0)``.

    The margin is negative and the wall is emitted anyway.

    FAILS IF: the radius is ``min(d, r_clip_max)``; or the builder falls back to
    projecting onto the un-inflated centreline hull; or it drops the component.
    """
    Q = _panel_seg(0.6, PANEL_AXIS, 0.20)
    walls, dropped, _unsound, c = _panel_walls(Q, PANEL_HAIRPIN, r_m=0.9, trust=0.5)

    assert dropped == 0, "the direction exists here; nothing should be dropped"
    assert len(walls) == 1
    w = walls[0]

    d, f = _centreline_distance(PANEL_HAIRPIN[0], c)
    assert d == pytest.approx(0.35, abs=1e-6), f"panel premise moved: d = {d:.6f}"
    assert w["rho"] == pytest.approx(0.9, abs=1e-12), (
        f"clip radius {w['rho']:.6f} is not floored at the obstacle radius 0.9; "
        f"unfloored it would be d = {d:.4f}"
    )
    assert w["n"][0] == pytest.approx(1.0, abs=1e-9), (
        f"normal {w['n']} is not (1, 0); the centroid and the hairpin's tip both "
        "lie on the axis of symmetry"
    )
    assert w["centroid_margin"] < 0.0
    # Bracket, not an equality: the offset is a rigorous ceiling on the support,
    # so it sits between the tube surface along n and the clip ball's own support.
    assert float(f[0]) + 0.9 <= w["b"] <= float(c[0]) + w["rho"] + 1e-9, (
        f"offset {w['b']:.6f} is outside [{float(f[0]) + 0.9:.6f}, "
        f"{float(c[0]) + w['rho']:.6f}]"
    )


@pytest.mark.parametrize(
    "label,cx,cy,half,ctrl,r_m",
    [
        ("A", 2.9, 2.0, 0.9, PANEL_STRAIGHT, 0.6),
        ("B1", 3.6, PANEL_AXIS, 0.6, PANEL_HAIRPIN, 0.9),
        ("B2", 2.2, 5.35, 0.55, PANEL_HAIRPIN, 0.9),
        ("B3", 0.6, PANEL_AXIS, 0.20, PANEL_HAIRPIN, 0.9),
    ],
)
def test_every_point_of_the_clipped_volume_is_behind_some_wall(
    label, cx, cy, half, ctrl, r_m
):
    """The invariant that cannot be true if an offset is under-estimated.

    Each wall contains its own component, so their union -- the whole clipped
    keep-out volume -- must have no point strictly on the free side of EVERY wall.
    The volume is sampled here, from the obstacle's control points and the clip
    radius the builder reported; the solver contributes only the normals and the
    offsets.

    This is the one place the construction could be silently unsound: the offset
    is a maximum over the component, and a maximum that steps over a narrow peak
    is too small, which puts a sliver of the keep-out zone on the ALLOWED side.
    The builder therefore emits a rigorous De Casteljau ceiling rather than a
    sampled maximum, and this is the check that the ceiling really is one.

    FAILS IF: any offset is below its component's true support. Verified by
    injecting `offset -= 0.05` into the Rust builder: panels A, B2 and B3 go red
    with up to 2.96e-2 of volume on the free side, four orders above the sampling
    resolution.

    **B1 does NOT go red under that mutation, and the reason is worth stating.**
    This is a union test -- a point only has to be behind SOME wall -- so where a
    component is severed into two, each shrunken half-space is still covered by
    the other one's. The check is therefore strong for a single wall and weaker
    for several; panel B1's guarantee comes from its mirror identity instead,
    which no offset error can preserve.
    """
    Q = (
        np.array([[2.0, 2.0], [2.6, 2.0], [3.2, 2.0], [3.8, 2.0]])
        if label == "A"
        else _panel_seg(cx, cy, half)
    )
    walls, _dropped, _unsound, c = _panel_walls(Q, ctrl, r_m=r_m, trust=0.5)
    assert walls, "no wall to check"

    Z = _clipped_volume_points(ctrl[0], r_m, c, walls[0]["rho"])
    assert len(Z) > 5_000, f"only {len(Z)} sample points; the check would be weak"

    free_side = np.min(
        np.stack([Z @ w["n"] - w["b"] for w in walls], axis=0), axis=0
    )
    worst = float(free_side.max())
    assert worst <= 1e-6, (
        f"{label}: {int((free_side > 1e-6).sum())} of {len(Z)} sampled points of "
        f"the clipped keep-out volume are on the free side of every wall, worst by "
        f"{worst:.3e} -- an offset is below its component's support"
    )
