"""Occlusion constraint geometry (item B12).

The guarantee the paper claims rests on one property of this builder and nothing
else: **the convex outer approximation must CONTAIN the true space-time shadow**.
If it does, then a control-point hull outside the approximation is outside the
true shadow, and line of sight holds for the whole segment. If it does not, every
row it emits certifies nothing, and the certificate reported at the returned
iterate is a number about a set the trajectory does not have to avoid.

These tests attack that property directly, sampling the TRUE shadow of the TRUE
moving occluder and asking whether each sampled point is where the approximation
says it is. They do not re-derive the builder; they contradict it.
"""

from __future__ import annotations

import numpy as np
import pytest

bezier_opt = pytest.importorskip("bezier_opt")


DIM = 4  # three spatial coordinates plus time
STATION = np.array([[5.0, -2.0, 0.3]])

# One moving occluder piece with a real time window. Motion is mostly TANGENTIAL
# to the sight line, which PAPER_1 records as the mechanism that violates
# space-time convexity an order of magnitude worse than radial motion -- so this
# is the case the inflation exists for, not a case it happens to survive.
MOVING_PIECE = [
    {"pos0": [5.0, 3.6, 0.1], "vel": [0.6, -0.32, 0.0], "r": 0.8,
     "t_start": 0.0, "t_end": 6.0, "name": "P0"},
]

# Control polygon: a climb over the shadow, so some planes come from the clear
# branch and some from the rolled one.
P_CLIMB = np.array([
    [0.5, 5.0, 0.5, 0.0],
    [2.0, 5.0, 1.2, 2.0],
    [3.5, 5.0, 2.0, 4.0],
    [5.0, 5.0, 2.2, 5.0],
    [6.5, 5.0, 2.0, 6.0],
    [8.0, 5.0, 1.2, 8.0],
    [9.5, 5.0, 0.5, 10.0],
])


def _obstacle_arrays(obstacles):
    return (
        np.array([o["pos0"] for o in obstacles], dtype=float),
        np.array([o["vel"] for o in obstacles], dtype=float),
        np.array([o["r"] for o in obstacles], dtype=float),
        np.array([o.get("t_start", -1e18) for o in obstacles], dtype=float),
        np.array([o.get("t_end", 1e18) for o in obstacles], dtype=float),
    )


def _occlusion_rows(P, obstacles, stations, n_seg):
    """The EXACT occlusion rows -- the ones the certificate is evaluated with.

    Not the rows the QP is handed. Those carry a rotation term and are documented
    as not conservative; every property below holds only for the exact rows.
    """
    P = np.asarray(P, dtype=float)
    spatial_dim = P.shape[1] - 1
    pos0, vel, r, t0, t1 = _obstacle_arrays(obstacles)
    out = bezier_opt.spacetime_occlusion_rows_exact(
        p=P, obstacle_pos0=pos0, obstacle_vel=vel, obstacle_r=r,
        stations=np.asarray(stations, dtype=float),
        obstacle_t_start=t0, obstacle_t_end=t1, n_seg=n_seg,
    )
    normals, lbs, seg, cp, obs, st, centers, radii, t_lo, t_hi, margins = out
    return {
        "normals": np.asarray(normals, dtype=float).reshape(-1, spatial_dim),
        "lbs": np.asarray(lbs, dtype=float),
        "seg": np.asarray(seg),
        "cp": np.asarray(cp),
        "obs": np.asarray(obs),
        "station": np.asarray(st),
        "centers": np.asarray(centers, dtype=float).reshape(-1, spatial_dim),
        "radii": np.asarray(radii, dtype=float),
        "t_lo": np.asarray(t_lo, dtype=float),
        "t_hi": np.asarray(t_hi, dtype=float),
        "margins": np.asarray(margins, dtype=float),
    }


def _planes(rows):
    """One entry per (segment, occluder piece, station) -- the plane itself.

    The rows repeat each plane once per control point of its segment; this
    collapses them back to the distinct planes, and asserts on the way that the
    repetition really is a repetition.
    """
    planes = {}
    for i in range(len(rows["lbs"])):
        key = (int(rows["seg"][i]), int(rows["obs"][i]), int(rows["station"][i]))
        entry = {
            "normal": rows["normals"][i],
            "lb": float(rows["lbs"][i]),
            "center": rows["centers"][i],
            "radius": float(rows["radii"][i]),
            "t_lo": float(rows["t_lo"][i]),
            "t_hi": float(rows["t_hi"][i]),
        }
        if key in planes:
            prev = planes[key]
            assert np.allclose(prev["normal"], entry["normal"]), (
                "control points of one segment must share ONE plane -- per-control-point "
                "planes weaken the convex-hull guarantee to a finite set of point conditions"
            )
            assert prev["lb"] == pytest.approx(entry["lb"])
        else:
            planes[key] = entry
    return planes


def _sample_unit_sphere(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _true_shadow_points(station, center_t, radius, n_dir=60, lambdas=(1.0, 1.4, 2.5, 6.0), seed=0):
    """Points of the TRUE shadow the ball at one instant casts from ``station``.

    The shadow is every body point pushed away from the station by a factor of at
    least one (PAPER_1 Part A sec. 5). Sampling the body's SURFACE and pushing it
    out reaches the boundary, which is where a containment claim is decided.
    """
    body = center_t[None, :] + radius * _sample_unit_sphere(n_dir, seed=seed)
    pts = []
    for lam in lambdas:
        pts.append(station[None, :] + lam * (body - station[None, :]))
    return np.concatenate(pts, axis=0)


def _segment_point_distance(station, points, center):
    """Distance from ``center`` to each segment [station, point]."""
    seg = points - station[None, :]
    seg_sq = np.einsum("ij,ij->i", seg, seg)
    tau = np.clip(np.einsum("ij,ij->i", center[None, :] - station[None, :], seg) / seg_sq, 0.0, 1.0)
    foot = station[None, :] + tau[:, None] * seg
    return np.linalg.norm(foot - center, axis=1)


# ---------------------------------------------------------------------------
# The conservative direction
# ---------------------------------------------------------------------------


def test_piece_body_contains_the_occluder_across_its_whole_window():
    """FAILS IF the outer approximation is not inflated for the occluder's motion.

    The approximation is one ball per (segment, piece) standing in for a ball that
    MOVES across the window. Containment is the whole argument: drop the
    inflation term and this fails for every moving occluder, which is exactly the
    edit that would silently turn every occlusion row into a row about nothing.
    """
    rows = _occlusion_rows(P_CLIMB, MOVING_PIECE, STATION, n_seg=4)
    assert len(rows["lbs"]) > 0
    vel = np.array(MOVING_PIECE[0]["vel"], dtype=float)
    pos0 = np.array(MOVING_PIECE[0]["pos0"], dtype=float)
    r = MOVING_PIECE[0]["r"]

    checked = 0
    for plane in _planes(rows).values():
        assert plane["t_hi"] >= plane["t_lo"]
        for t in np.linspace(plane["t_lo"], plane["t_hi"], 25):
            true_center = pos0 + vel * t
            reach = np.linalg.norm(true_center - plane["center"]) + r
            assert reach <= plane["radius"] + 1e-9, (
                f"true occluder at t={t:.3f} reaches {reach:.6f}, outside the "
                f"approximation's {plane['radius']:.6f}"
            )
            checked += 1
    assert checked > 0

    # The check can fail: with no inflation at all the same comparison is false.
    naive = max(
        np.linalg.norm((pos0 + vel * t) - plane["center"]) + r - r
        for plane in _planes(rows).values()
        for t in (plane["t_lo"], plane["t_hi"])
    )
    assert naive > 1e-6, "the occluder must actually move, or this test proves nothing"


def test_true_shadow_lies_inside_the_approximated_shadow():
    """FAILS IF the approximation does not contain the true space-time shadow.

    Per piece, per instant in the piece's window: sample the shadow the real
    moving ball casts and require every sample to be shadowed by the
    approximation too. The approximation may forbid more than the truth -- that
    is conservatism in the safe direction -- but it may never forbid less.
    """
    rows = _occlusion_rows(P_CLIMB, MOVING_PIECE, STATION, n_seg=4)
    station = STATION[0]
    vel = np.array(MOVING_PIECE[0]["vel"], dtype=float)
    pos0 = np.array(MOVING_PIECE[0]["pos0"], dtype=float)
    r = MOVING_PIECE[0]["r"]

    worst = -np.inf
    worst_uninflated = -np.inf
    n_checked = 0
    for plane in _planes(rows).values():
        for t in np.linspace(plane["t_lo"], plane["t_hi"], 9):
            pts = _true_shadow_points(station, pos0 + vel * t, r)
            dist = _segment_point_distance(station, pts, plane["center"])
            # Inside the approximation's shadow means the sight segment comes
            # within the approximated radius of the approximated centre.
            slack = plane["radius"] - dist
            worst = max(worst, float(np.max(dist - plane["radius"])))
            worst_uninflated = max(worst_uninflated, float(np.max(dist - r)))
            assert np.all(slack >= -1e-9), (
                f"a true shadow point at t={t:.3f} escapes the approximation by "
                f"{float(np.max(-slack)):.6e}"
            )
            n_checked += pts.shape[0]
    assert n_checked > 1000
    assert worst < 0.0
    # The counterfactual that makes the assertion above evidence: an
    # approximation centred at the same place but NOT inflated fails it.
    assert worst_uninflated > 0.0


def test_half_space_contains_the_true_shadow():
    """FAILS IF a row's forbidden side does not cover the shadow it stands for.

    This is the property the certificate rests on. A control point satisfying
    ``normal . q >= lower_bound`` is outside the half-space; if the half-space
    contains the true shadow, that control point is outside the shadow. Sampling
    the true shadow and requiring every sample on the FORBIDDEN side is the
    contrapositive, checked directly.

    Also asserts the station itself is on the free side -- the validity condition
    for the whole construction. A plane with the observer on the forbidden side
    describes no shadow at all.
    """
    rows = _occlusion_rows(P_CLIMB, MOVING_PIECE, STATION, n_seg=4)
    station = STATION[0]
    vel = np.array(MOVING_PIECE[0]["vel"], dtype=float)
    pos0 = np.array(MOVING_PIECE[0]["pos0"], dtype=float)
    r = MOVING_PIECE[0]["r"]

    uninflated_would_fail = False
    for plane in _planes(rows).values():
        n, lb = plane["normal"], plane["lb"]
        assert float(n @ station) >= lb - 1e-9, (
            "the station must lie on the FREE side, else the half-space is not a "
            "supporting half-space of the shadow"
        )
        for t in np.linspace(plane["t_lo"], plane["t_hi"], 9):
            pts = _true_shadow_points(station, pos0 + vel * t, r, seed=7)
            vals = pts @ n
            assert np.all(vals <= lb + 1e-9), (
                f"a true shadow point at t={t:.3f} sits on the FREE side by "
                f"{float(np.max(vals - lb)):.6e}"
            )
            # Without the inflation the offset would be `n . centre + r`, and
            # the same samples would break it. Recorded so the tolerance above
            # is known to be separating something.
            lb_uninflated = float(n @ plane["center"]) + r
            if float(np.max(vals)) > lb_uninflated + 1e-9:
                uninflated_would_fail = True
    assert uninflated_would_fail, (
        "the inflated offset must be strictly larger than the uninflated one "
        "somewhere, or this test cannot distinguish them"
    )


# ---------------------------------------------------------------------------
# Row structure and the off-by-default guarantee
# ---------------------------------------------------------------------------


def test_one_plane_per_segment_piece_station_applied_to_every_control_point():
    """The claim's cost model, checked as a row count.

    'One linearized supporting-half-space row per (segment, occluder-piece,
    station)' -- so the number of distinct planes is exactly that product, and
    each is asserted at every control point of its segment, which is what carries
    it from the control polygon to the curve.
    """
    n_seg = 4
    # Both pieces span the whole horizon here. A piece whose window misses a
    # segment emits nothing for it -- covered separately below -- so the exact
    # product only holds when every pairing is live.
    pieces = [
        dict(MOVING_PIECE[0], t_start=0.0, t_end=10.0),
        {"pos0": [3.0, 3.0, 0.1], "vel": [0.0, -0.2, 0.0], "r": 0.6,
         "t_start": 0.0, "t_end": 10.0, "name": "P1"},
    ]
    two_stations = np.array([[5.0, -2.0, 0.3], [4.0, -3.0, 0.5]])
    rows = _occlusion_rows(P_CLIMB, pieces, two_stations, n_seg=n_seg)
    planes = _planes(rows)

    np1 = P_CLIMB.shape[0]
    assert len(planes) == n_seg * len(pieces) * len(two_stations)
    assert len(rows["lbs"]) == len(planes) * np1
    for key in planes:
        cps = sorted(
            int(rows["cp"][i])
            for i in range(len(rows["lbs"]))
            if (int(rows["seg"][i]), int(rows["obs"][i]), int(rows["station"][i])) == key
        )
        assert cps == list(range(np1))


def test_no_station_means_no_occlusion_rows():
    """Off by default. An empty station set must produce nothing at all."""
    rows = _occlusion_rows(P_CLIMB, MOVING_PIECE, np.zeros((0, 3)), n_seg=4)
    assert len(rows["lbs"]) == 0
    assert rows["normals"].shape == (0, 3)


def test_piece_outside_the_segment_time_window_emits_nothing():
    """A piece whose window misses the plan emits no row.

    The chain construction depends on this: pieces on adjacent windows must
    constrain only the segments that overlap them, otherwise a two-piece chain
    would apply both pieces everywhere and forbid the union for all time.
    """
    far_future = [dict(MOVING_PIECE[0], t_start=50.0, t_end=60.0)]
    rows = _occlusion_rows(P_CLIMB, far_future, STATION, n_seg=4)
    assert len(rows["lbs"]) == 0

    # The same piece with a window that DOES overlap emits rows -- so the empty
    # result above is the window doing its job, not the builder being inert.
    overlapping = [dict(MOVING_PIECE[0], t_start=0.0, t_end=10.0)]
    assert len(_occlusion_rows(P_CLIMB, overlapping, STATION, n_seg=4)["lbs"]) > 0


def test_row_margin_is_reproduced_by_the_row_itself():
    """The exported margin must be what the row actually says at ``P``.

    ``margin = normal . q - lower_bound`` where ``q = (A_seg @ P)[cp]``. Recomputed
    from the control points alone, so a builder that reported a margin from one
    geometry while emitting a row from another would be caught.
    """
    from orbital_docking.de_casteljau import segment_matrices_equal_params

    n_seg = 4
    rows = _occlusion_rows(P_CLIMB, MOVING_PIECE, STATION, n_seg=n_seg)
    a_list = segment_matrices_equal_params(P_CLIMB.shape[0] - 1, n_seg)
    spatial_dim = P_CLIMB.shape[1] - 1

    for i in range(len(rows["lbs"])):
        q = (np.asarray(a_list[int(rows["seg"][i])], dtype=float) @ P_CLIMB)[int(rows["cp"][i])]
        recomputed = float(rows["normals"][i] @ q[:spatial_dim]) - float(rows["lbs"][i])
        assert recomputed == pytest.approx(float(rows["margins"][i]), abs=1e-9)
