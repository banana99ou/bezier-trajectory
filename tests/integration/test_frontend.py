"""The frontend's request/response path, over real HTTP, without a browser.

`spacetime_bezier/viewer.py` shipped untested and CLAUDE.md had to record it as
UNVERIFIED: "It compiles and it starts; that is the whole of the evidence."
This file exists so the same sentence cannot be written about
`spacetime_bezier/frontend.py`.

Every test below carries a FAILS IF line naming the outcome that would sink it.
A check that cannot fail is not evidence, and "the page loaded" is not a claim
about anything the page says.

The server is bound on an ephemeral port and driven with `handle_request` in a
worker thread. Not `serve_forever`: `tests/conftest.py` makes that raise, because
a test once left it holding port 8765 for ten hours. The production default stays
hard-wired at 8767 with no CLI flag to move it, and `test_port_guard_*` below is
what pins that.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

bezier_opt = pytest.importorskip("bezier_opt")

from spacetime_bezier import frontend  # noqa: E402
from spacetime_bezier.optimize import optimize_scenario  # noqa: E402
from spacetime_bezier.scenarios import SCENARIO_MAP, scenario_elastic_weight  # noqa: E402

HOST = "127.0.0.1"

# `original` N8_seg4 -- the configuration README sec. Measurements records at
# +0.620, measured here to full precision so a geometry change cannot hide inside
# a rounded number. This is the anchor: if the solver's answer moves, this file
# fails before any picture is drawn from it.
ORIGINAL_CLEARANCE = 0.6204425675058312
CLEARANCE_TOL = 1e-6


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------



def assert_koz_planes_are_one_per_pair(planes, P, obstacles, n_seg, trust_radius):
    """At most one plane per (segment, obstacle), and every MISSING one justified.

    The G2 fix made this one plane per (segment, obstacle) rather than one per
    control point, and the count used to be asserted as exactly n_seg * n_obs.
    Since the 2026-08-24 construction change that is no longer right: a clip ball
    of radius `min(R, R_max)` with `R_max = r + E + Delta*sqrt(d+1)` emits NO row
    for an obstacle nothing in the trust region can reach, which is PAPER_1
    statement (7)'s first regime.

    Relaxing the count to "<= 12" would turn this into a check that cannot fail --
    zero planes would pass. So the omissions are verified instead: for every pair
    with no plane, the distance from that segment's centroid to the obstacle's
    lifted centreline must genuinely exceed `R_max`. A plane dropped for any other
    reason fails here.
    """
    import numpy as np
    from spacetime_bezier.geometry import normalize_obstacles
    from orbital_docking.de_casteljau import segment_matrices_equal_params

    P = np.asarray(P, dtype=float)
    dim = P.shape[1]
    norm = normalize_obstacles(obstacles)
    n_obs = len(norm)

    pairs = [(int(pl["seg"]), int(pl["obs"])) for pl in planes]
    assert len(pairs) == len(set(pairs)), "a (segment, obstacle) pair got two planes"
    assert len(pairs) <= n_seg * n_obs

    a_list = [np.asarray(a, dtype=float) for a in segment_matrices_equal_params(P.shape[0] - 1, n_seg)]
    missing = []
    for s_i in range(n_seg):
        Q = a_list[s_i] @ P
        c = Q.mean(axis=0)
        E = float(np.max(np.linalg.norm(Q - c, axis=1)))
        for o_i in range(n_obs):
            if (s_i, o_i) in pairs:
                continue
            cps = np.asarray(norm[o_i]["control_points"], dtype=float)
            r_m = float(norm[o_i]["radius"])
            # Distance from the centroid to the lifted centreline, sampled finely.
            ss = np.linspace(0.0, 1.0, 2001)
            from spacetime_bezier.geometry import _eval_at

            R = float(np.min(np.linalg.norm(_eval_at(cps, ss) - c, axis=1)))
            r_max = r_m + E + trust_radius * np.sqrt(dim)
            missing.append((s_i, o_i, R, r_max))
            assert R > r_max, (
                f"segment {s_i} obstacle {o_i} got no plane but is within reach: "
                f"R={R:.4f} <= R_max={r_max:.4f}"
            )
    return missing


@pytest.fixture(scope="module")
def server():
    """A real HTTP server on an ephemeral port, driven one request at a time."""
    srv = frontend.make_server(HOST, 0)
    srv.timeout = 0.2
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            srv.handle_request()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    try:
        yield f"http://{HOST}:{srv.server_address[1]}"
    finally:
        stop.set()
        # One throwaway connection so the blocking accept returns promptly.
        try:
            with socket.create_connection(srv.server_address, timeout=0.5):
                pass
        except OSError:
            pass
        thread.join(timeout=3.0)
        srv.server_close()


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return response.status, response.read(), response.headers.get("Content-Type")


def post(base: str, path: str, payload, timeout: float = 900.0):
    """POST JSON; return ``(status, decoded body)`` for success AND for errors."""
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(
        base + path, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture(scope="module")
def original(server):
    status, data = post(server, "/api/solve", {"scenario": "original", "N": 8, "n_seg": 4})
    assert status == 200, data
    return data


@pytest.fixture(scope="module")
def station_fence(server):
    # The weight is pinned at the one measured to certify. Left blank the ladder
    # would walk six rungs of a run that takes seconds each, and the rung that
    # wins is already known; the ladder itself is exercised by `original`.
    status, data = post(
        server,
        "/api/solve",
        {
            "scenario": "station_fence",
            "N": 8,
            "n_seg": 8,
            "elastic_weight": scenario_elastic_weight("station_fence"),
        },
    )
    assert status == 200, data
    return data


# ---------------------------------------------------------------------------
# (a) the page
# ---------------------------------------------------------------------------


def test_index_serves_the_page(server):
    """GET / returns the frontend page, wired to the local plotly bundle.

    FAILS IF: the route is missing, the file is not found, the page is served
    with the wrong content type, or it links a CDN instead of `/static/plotly`
    -- which is the difference between a page that works offline and one that
    silently renders nothing.
    """
    status, body, content_type = get(server, "/")
    assert status == 200
    assert "text/html" in content_type
    text = body.decode("utf-8")
    assert '<script src="/static/plotly.min.js">' in text
    # No external asset of any kind. The page has to draw the same picture on a
    # machine with no network, because that is where it will be read.
    assert "http://" not in text and "https://" not in text
    assert "cdn" not in text.lower()

    status, body, content_type = get(server, "/static/plotly.min.js")
    assert status == 200 and len(body) > 1_000_000


def test_health_names_this_app_and_its_extension(server):
    """The health probe identifies the process holding the port.

    FAILS IF: the app id, pid or extension build time is missing -- those three
    are what `describe_port_conflict` prints when the port is held, so a health
    payload without them turns a precise refusal into "something is on 8767".
    """
    status, body, _ = get(server, "/api/health")
    assert status == 200
    payload = json.loads(body)
    assert payload["app"] == "spacetime-bezier-frontend"
    assert isinstance(payload["pid"], int)
    assert "extension_build_time" in payload
    assert "git_commit" in payload


def test_catalog_offers_every_scenario_with_the_agreed_views(server):
    """The catalog is the whole registry, with the agreed view roster.

    The roster is a pair of user decisions: 2026-08-24 removed the
    dropped-coordinate lift projections ("I just need xy(z)t"), and 2026-08-26
    brought exactly ONE of them back -- (x,y,t), asked for while designing
    `loiter`, whose temporal-slot behaviour a spatial view cannot show.
    (x,z,t) and (y,z,t) stay gone.

    FAILS IF: a registered scenario is missing (`viewer.py` dropped the
    4-column scenarios rather than project them, which is what this page exists
    to fix), a 2D scenario loses the lift or the flat top-down view, or the
    4-column roster drifts from those two decisions in either direction.
    """
    status, body, _ = get(server, "/api/scenarios")
    assert status == 200
    catalog = json.loads(body)["scenarios"]
    assert set(catalog) == set(SCENARIO_MAP)
    assert [v["id"] for v in catalog["original"]["views"]] == ["xyt", "xy"]
    assert catalog["original"]["default_view"] == "xyt"
    assert [v["id"] for v in catalog["fence3d"]["views"]] == ["xyz", "xyz_all", "xyt"]
    assert catalog["station_fence"]["default_view"] == "xyz"
    assert catalog["station_fence"]["stations"] == [[5.0, -2.0, 0.3]]


def test_axis_banners_say_how_time_is_carried(server):
    """Each view carries the sentence that keeps its picture honest.

    FAILS IF: a view stops saying how time is carried (slider, colour, or the
    vertical axis), a 4-column view stops declaring that patches and hulls are
    projections from (x,y,z,t), or an undrawable column count grows views.
    """
    views = {v["id"]: v for v in frontend.axis_views(4)}
    assert set(views) == {"xyz", "xyz_all", "xyt"}
    assert views["xyz"]["kind"] == "spatial" and "slider" in views["xyz"]["banner"]
    assert views["xyz_all"]["kind"] == "spatial_all"
    assert views["xyz_all"]["banner"] == "all three axes are space; TIME is the color"
    # The 2026-08-26 readmission: the (x,y,t) lift keeps time vertical, and it
    # must say BOTH that z is gone and that patches are projections.
    assert views["xyt"]["kind"] == "lift"
    assert views["xyt"]["cols"] == [0, 1, 3], "time must be the vertical axis"
    assert views["xyt"]["banner"] == "the VERTICAL axis is TIME"
    assert "z is dropped" in views["xyt"]["dropped_note"]
    for view in views.values():
        assert "projections from (x,y,z,t)" in view["dropped_note"]
    assert frontend.axis_views(5) == [], "5 columns has no honest projection"
    assert frontend.axis_views(2) == []
    # 2D scenarios: the lift keeps time vertical; the flat view never claims to
    # be the lift -- two columns, kind "flat", time carried by the cursor.
    lift3, flat3 = frontend.axis_views(3)
    assert lift3["kind"] == "lift"
    assert lift3["banner"] == "the VERTICAL axis is TIME"
    assert lift3["cols"][-1] == 2, "time must be the vertical axis of the lift"
    assert flat3["kind"] == "flat" and flat3["cols"] == [0, 1]
    assert "cursor" in flat3["banner"]


# ---------------------------------------------------------------------------
# (b) the solve, for `original`
# ---------------------------------------------------------------------------


def test_solve_original_reports_the_measured_clearance(original):
    """The number on the page is the number the batch path measures.

    FAILS IF: the independent clearance moves off the recorded value by more than
    1e-6 (a geometry or solver change), or the solver's own reported clearance
    and the independent recomputation disagree -- the pair exists to catch a
    server that reports one number twice.
    """
    verdict = original["verdict"]
    assert verdict["clearance_independent"] == pytest.approx(
        ORIGINAL_CLEARANCE, abs=CLEARANCE_TOL
    )
    assert verdict["clearance_reported"] == pytest.approx(
        verdict["clearance_independent"], abs=CLEARANCE_TOL
    )
    assert verdict["clearance_delta"] < CLEARANCE_TOL
    assert verdict["certificate_reported"] == pytest.approx(0.0, abs=1e-9)
    assert verdict["certificate_recomputed"] == pytest.approx(0.0, abs=1e-9)


def test_solve_original_is_figure_grade_with_no_reasons(original):
    """The gate is computed server-side and answers for this configuration.

    FAILS IF: `figure_grade` disagrees with its own reason list (a chip that says
    PASS beside a list of failures), or this configuration stops passing the gate
    -- README records it as certified at weight 100.
    """
    verdict = original["verdict"]
    assert verdict["figure_grade_reasons"] == []
    assert verdict["figure_grade"] is True
    assert verdict["converged"] is True
    assert verdict["stop_label"] == "stationary"
    assert original["resolved"]["elastic_weight"] == 100.0
    assert original["resolved"]["ladder_walked"] is True
    assert original["resolved"]["rungs"][0]["cleared"] is True


def test_gate_verdict_is_the_shared_predicate_not_a_copy(original):
    """`figure_grade` comes from `figure_grade_failures`, NaN polarity intact.

    An earlier reimplementation of this mapping in `tools/make_paper_figure.py`
    rewrote the conditions as ``if x > tol``, which inverts the NaN polarity and
    drew figures for runs with no evidence at all.

    FAILS IF: the server's verdict disagrees with the shared predicate on the
    run's own row, or a row with missing evidence (NaN certificate, NaN slack) is
    graded as passing.
    """
    from spacetime_bezier.optimize import figure_grade_failures

    verdict = original["verdict"]
    row = {
        "converged": verdict["converged"],
        "stop_label": verdict["stop_label"],
        "certificate_violation": verdict["certificate_reported"],
        "total_slack": verdict["total_slack"],
        "min_clearance": verdict["clearance_independent"],
        "speed_cap_violation": verdict["speed_cap_violation"],
    }
    assert figure_grade_failures(row) == verdict["figure_grade_reasons"]

    blind = dict(row, certificate_violation=float("nan"), total_slack=float("nan"))
    reasons = figure_grade_failures(blind)
    assert reasons, "absent evidence must refuse, not pass"
    assert any("certificate" in r for r in reasons)


def test_solve_returns_every_layer_block(original):
    """Each default layer has its data, and the shapes agree with each other.

    FAILS IF: a block the page draws is absent or empty, the curve is not the
    lifted dimension wide, the speed array is not sampled at the same points as
    the curve, or the segment hulls do not cover the requested segment count --
    each of those is a layer that silently draws nothing.
    """
    solution, segments = original["solution"], original["segments"]
    assert np.asarray(solution["curve"]).shape == (frontend.CURVE_SAMPLES, 3)
    assert len(solution["speed"]) == frontend.CURVE_SAMPLES
    assert np.asarray(solution["control_points"]).shape == (9, 3)
    assert np.asarray(solution["init_control_points"]).shape == (9, 3)
    assert segments["n_seg"] == 4 and len(segments["hulls"]) == 4
    assert np.asarray(segments["hulls"][0]).shape == (9, 3)

    # At most one plane per (segment, obstacle). That IS the G2 fix -- a count of
    # 108 here would mean one plane per control point again -- and every pair that
    # got NO plane is checked to be genuinely out of reach, so the weaker count
    # cannot hide a dropped constraint.
    assert len(original["planes"]["koz"]) <= 12
    assert_koz_planes_are_one_per_pair(
        original["planes"]["koz"],
        original["solution"]["control_points"],
        original["scenario"]["obstacles"],
        n_seg=4,
        trust_radius=float(original["request"]["trust_radius"]),
    )
    assert original["planes"]["occlusion"] == []
    # Every plane is asserted at every control point, so the ledger is exactly
    # (planes x control points). Derived rather than hardcoded at 108: the plane
    # count now depends on how many obstacles are in reach, and a magic number
    # would fail for a reason that has nothing to do with the ledger.
    assert original["ledger"]["total"] == len(original["planes"]["koz"]) * 9
    assert original["ledger"]["violated"] == 0
    assert original["sight"] is None and original["los"] is None
    assert original["shadow_balls"] == []
    assert original["scenario"]["views"][0]["id"] == "xyt"
    assert original["provenance"]["git_commit"]
    assert "extension_stale" in original["provenance"]


def test_plane_patches_are_patches_of_the_solver_s_own_half_spaces(original):
    """Every drawn patch lies on its row's plane and points at the free side.

    This is the impossibility check on the one piece of geometry the server
    builds rather than reads: a patch is a bounded piece of the half-space the
    Rust builder emitted, so its four corners must satisfy the row with equality,
    the arrow's far end must satisfy it strictly, and the patch's reported margin
    must equal the tightest slack among the control points it supports.

    FAILS IF: any of those three is violated -- which is what a plane invented in
    the viewer, or attached to the wrong segment, would look like.
    """
    hulls = [np.asarray(h, dtype=float) for h in original["segments"]["hulls"]]
    planes = original["planes"]["koz"]
    assert planes
    for plane in planes:
        n = np.asarray(plane["normal"], dtype=float)
        lb = plane["lb"]
        for corner in plane["corners"]:
            assert abs(float(n @ np.asarray(corner, dtype=float)) - lb) < 1e-9
        tip = np.asarray(plane["free_arrow"][1], dtype=float)
        assert float(n @ tip) > lb + 1e-12, "the arrow must point OUT of the obstacle"
        hull = hulls[plane["seg"]]
        tightest = min(float(n @ q) - lb for q in hull)
        assert tightest == pytest.approx(plane["margin"], abs=1e-9)


def test_speed_is_the_curve_s_own_derivative(original):
    """The speed colouring is |d(spatial)/dt|, not a polyline difference.

    FAILS IF: the server's speed disagrees with a finite difference of the drawn
    curve by more than the discretisation error, i.e. if it is measuring some
    other curve.
    """
    curve = np.asarray(original["solution"]["curve"], dtype=float)
    speed = np.asarray(original["solution"]["speed"], dtype=float)
    steps = np.diff(curve, axis=0)
    fd = np.linalg.norm(steps[:, :-1], axis=1) / steps[:, -1]
    mid = 0.5 * (speed[:-1] + speed[1:])
    assert np.allclose(fd, mid, rtol=2e-3, atol=2e-3)
    assert np.all(speed > 0.0)


def test_solve_matches_the_batch_path_exactly(original):
    """The page's solve and `optimize_scenario` return the same trajectory.

    The frontend runs its own copy of the ladder loop so it can keep the solver's
    raw `info` -- which `optimize_scenario` drops -- and that copy is the one
    thing here that could drift from the batch path. Two independent routes to
    the same configuration must land on the same control points, bit for bit.

    FAILS IF: any default, any ladder rung, or the ranking rule diverges between
    the two -- the page would then be reporting numbers the scenario table does
    not.
    """
    scenario_fn, _ = SCENARIO_MAP["original"]
    batch = optimize_scenario(scenario_fn(), [(8, 4)], verbose=False)["results"]["N8_seg4"]
    assert np.allclose(
        np.asarray(batch["control_points"], dtype=float),
        np.asarray(original["solution"]["control_points"], dtype=float),
        atol=0.0,
        rtol=0.0,
    )
    assert batch["elastic_weight"] == original["resolved"]["elastic_weight"]
    assert batch["figure_grade"] is original["verdict"]["figure_grade"]


# ---------------------------------------------------------------------------
# (c) the solve, for `station_fence`
# ---------------------------------------------------------------------------


def test_station_fence_returns_the_occlusion_layers(station_fence):
    """Station, sight fan, line-of-sight margin, shadow balls, occlusion rows.

    FAILS IF: any of the five is missing or empty -- each is a layer the page
    draws for this scenario and nothing else would signal its absence -- or the
    sight fan is not sampled from the independent margin array.
    """
    assert station_fence["scenario"]["stations"] == [[5.0, -2.0, 0.3]]
    sight = station_fence["sight"]
    assert sight is not None
    assert sight["station"] == [5.0, -2.0, 0.3]
    assert len(sight["lines"]) == frontend.SIGHT_LINES
    for line in sight["lines"]:
        assert len(line["v"]) == 3
        assert line["lost"] is (line["m"] is not None and line["m"] < 0.0)

    los = station_fence["los"]
    assert los is not None and len(los["t"]) == len(los["m"]) >= frontend.LOS_SAMPLES
    assert los["min"] == pytest.approx(station_fence["verdict"]["los_min_margin"])

    assert station_fence["shadow_balls"], "the solver exported no occlusion bodies"
    for ball in station_fence["shadow_balls"]:
        assert len(ball["c"]) == 3 and ball["R"] > 0.0 and ball["t1"] > ball["t0"]
    assert station_fence["planes"]["occlusion"]
    assert station_fence["ledger"]["occlusion_total"] > 0


def test_station_fence_keeps_line_of_sight_and_certifies_it(station_fence):
    """The constrained run holds the link, and both certificates say so.

    FAILS IF: the occlusion certificate leaves zero, a plane could not be built
    (uncertifiable is not certified), the independent margin goes negative while
    the certificate stays at zero -- the disagreement that would mean one of the
    two is wrong -- or the run stops being figure-grade.
    """
    verdict = station_fence["verdict"]
    assert verdict["occlusion_reported"] == pytest.approx(0.0, abs=1e-6)
    assert verdict["occlusion_recomputed"] == pytest.approx(0.0, abs=1e-6)
    assert verdict["occlusion_planes_dropped"] == 0.0
    assert verdict["los_lost"] is False
    assert verdict["los_min_margin"] > 0.0
    assert verdict["figure_grade_reasons"] == []
    assert verdict["figure_grade"] is True


def test_station_fence_occlusion_normals_carry_no_time_component(station_fence):
    """Occlusion rows are spatial by design; keep-out rows are not.

    A zero time coefficient on a keep-out row beside a moving obstacle is the G1
    defect signature. On an occlusion row it is the formulation: a shadow
    half-space is a prism with time-parallel walls. Both facts are asserted here
    so neither can be quietly turned into the other.

    FAILS IF: an occlusion patch grows a nonzero time coefficient, or the
    keep-out rows of this moving-obstacle scenario lose theirs.
    """
    for plane in station_fence["planes"]["occlusion"]:
        assert plane["normal"][-1] == 0.0
        assert any(abs(c) > 0.0 for c in plane["normal"][:-1])
    moving = [
        plane for plane in station_fence["planes"]["koz"] if abs(plane["normal"][-1]) > 1e-12
    ]
    assert moving, "every obstacle here moves; a keep-out plane with no time term is G1"


def test_station_fence_offers_the_agreed_views(station_fence):
    """The 4-column scenario is reachable, with the agreed view roster.

    The roster: the two spatial views (2026-08-24 decision) plus the (x,y,t)
    lift the same user brought back 2026-08-26 for `loiter`'s temporal slot.

    FAILS IF: the scenario comes back undrawable (the `viewer.py` behaviour this
    page replaces), the roster drifts from those two decisions in either
    direction, the default is not the cursor view, or a view stops declaring
    its patches projections from the full lifted space.
    """
    views = station_fence["scenario"]["views"]
    assert [v["id"] for v in views] == ["xyz", "xyz_all", "xyt"]
    assert station_fence["scenario"]["default_view"] == "xyz"
    for view in views:
        assert view["cols"] == ([0, 1, 3] if view["id"] == "xyt" else [0, 1, 2])
        assert "projections" in view["dropped_note"]


# ---------------------------------------------------------------------------
# (d) bad requests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, fragment",
    [
        ({"scenario": "nope"}, "unknown scenario"),
        ({"scenario": "original", "N": 0}, "must be >= 1"),
        ({"scenario": "original", "N": "eight"}, "must be an integer"),
        ({"scenario": "original", "tol": "tight"}, "must be a number"),
        ({"scenario": "original", "trust_radius": -1}, "trust_radius must be > 0"),
        # The uncapped time penalty guard. It is an artifact generator, refused in
        # `optimize.py`; what is asserted here is that the refusal reaches the
        # client as a message rather than as a 500.
        ({"scenario": "original", "time_weight": 1.0}, "artifact generator"),
    ],
)
def test_bad_requests_answer_400_with_a_message(server, payload, fragment):
    """A rejected configuration is a 400 with an explanation, never a traceback.

    FAILS IF: any of these returns 200 (the configuration was accepted), or
    returns 500 with a traceback -- which in the UI reads as "the solver broke"
    rather than "that configuration is not allowed", a materially different and
    much more alarming claim.
    """
    status, data = post(server, "/api/solve", payload)
    assert status == 400, data
    assert "traceback" not in data
    assert fragment in data["error"]


def test_malformed_body_is_a_400_not_a_crash(server):
    """A body that is not JSON is refused cleanly.

    FAILS IF: the server answers 500, or accepts the garbage and solves the
    default configuration -- silently answering a question nobody asked.
    """
    status, data = post(server, "/api/solve", b"{not json")
    assert status == 400
    assert "not valid JSON" in data["error"]


def test_unknown_paths_are_404(server):
    """FAILS IF: an unrouted path returns anything but 404 -- in particular a 200
    with the page, which would make every typo look like a working endpoint."""
    status, data = post(server, "/api/solv", {"scenario": "original"})
    assert status == 404
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(server, "/api/nothing")
    assert excinfo.value.code == 404


# ---------------------------------------------------------------------------
# (e) replay
# ---------------------------------------------------------------------------


def test_replay_lands_where_the_solve_landed(server, original):
    """The replay is the run: the stepping context ends on the returned iterate.

    The drift number is `tools/trace_viewer.py`'s own check, computed by the child
    script this endpoint reuses rather than re-implements. It is the thing that
    catches a stepping context built with parameters that do not match
    `optimize_spacetime`'s defaults -- a replay that looks like the solve and is
    not.

    FAILS IF: drift exceeds 1e-9 without the best-iterate fallback having fired,
    no frames come back, or the frames stop carrying the accept/reject decision.
    """
    status, data = post(
        server,
        "/api/replay",
        {
            "scenario": "original",
            "N": 8,
            "n_seg": 4,
            "elastic_weight": original["resolved"]["elastic_weight"],
        },
    )
    assert status == 200, data
    assert data["returned_best_iterate"] is False
    assert data["drift"] <= 1e-9
    assert data["frames"]
    assert data["frames"][-1]["it"] == original["verdict"]["iterations"]
    for frame in data["frames"]:
        assert isinstance(frame["advanced"], bool)
        assert np.asarray(frame["P"]).shape == (9, 3)
        assert np.asarray(frame["cand"]).shape == (9, 3)
    # The replay's last accepted iterate IS the solve's answer, to the drift above.
    assert np.allclose(
        np.asarray(data["frames"][-1]["P"], dtype=float),
        np.asarray(original["solution"]["control_points"], dtype=float),
        atol=1e-9,
    )


def test_replay_carries_the_solver_s_own_trust_radius(server, original):
    """Trust boxes are sized from STTRACE, not from a viewer-side guess.

    FAILS IF: the trace comes back empty or with columns the parser does not
    recognise (the header check refuses rather than mislabel), or a frame gets no
    trust radius -- the box would then be drawn at some default half-width and
    read as the solver's trust region.
    """
    status, data = post(
        server,
        "/api/replay",
        {
            "scenario": "original",
            "N": 8,
            "n_seg": 4,
            "elastic_weight": original["resolved"]["elastic_weight"],
        },
    )
    assert status == 200
    assert data["warnings"] == []
    assert data["trace"], "no STTRACE lines -- the timeline and the boxes are empty"
    assert data["trace_header"][0] == "it"
    assert "trust_after" in data["trace_header"]
    assert all(frame["trust_after"] is not None for frame in data["frames"])
    assert all(frame["trust_after"] > 0.0 for frame in data["frames"])


@pytest.mark.parametrize(
    "extra, fragment",
    [
        ({"v_max": 5.0}, "does not carry the speed cap"),
        ({"v_max": 5.0, "time_weight": 1.0}, "speed cap"),
        ({"free_arrival_time": True}, "free_arrival_time"),
    ],
)
def test_replay_refuses_a_run_it_cannot_reproduce(server, extra, fragment):
    """A replay that would be a different problem is refused, not approximated.

    The reused child script builds the stepping context without the speed cap,
    the time penalty or a freed arrival. Replaying such a run anyway would
    produce frames of some other optimisation, and the drift number alone would
    not say why.

    FAILS IF: the endpoint answers 200 for any of these -- frames of a run nobody
    asked for, presented as the run.
    """
    payload = {"scenario": "original", "N": 8, "n_seg": 4, "elastic_weight": 100.0}
    payload.update(extra)
    status, data = post(server, "/api/replay", payload)
    assert status == 400, data
    assert fragment in data["error"]


def test_replay_frames_carry_their_own_half_spaces(server, original):
    """Every frame carries planes rebuilt at ITS reference by the exact builder.

    FAILS IF: a frame has no planes block (the plane toggles would silently draw
    nothing during replay -- the exact complaint this feature answers), the count
    is not one plane per (segment, obstacle), a patch corner leaves its own
    plane, or the first and last frames carry identical planes -- the signature
    of the returned iterate's planes being stamped onto every earlier iterate.
    """
    status, data = post(
        server,
        "/api/replay",
        {"scenario": "original", "N": 8, "n_seg": 4,
         "elastic_weight": original["resolved"]["elastic_weight"]},
    )
    assert status == 200, data
    frames = data["frames"]
    assert frames and all("planes" in frame for frame in frames)
    mid = frames[len(frames) // 2]
    assert len(mid["planes"]["koz"]) <= 12, "one plane per (segment, obstacle)"
    assert mid["planes"]["occlusion"] == []
    for plane in mid["planes"]["koz"]:
        n = np.asarray(plane["normal"], dtype=float)
        for corner in plane["corners"]:
            assert abs(float(n @ np.asarray(corner, dtype=float)) - plane["lb"]) < 1e-9
    if len(frames) > 1:
        assert frames[0]["planes"]["koz"] != frames[-1]["planes"]["koz"]


# ---------------------------------------------------------------------------
# (f) the port guard
# ---------------------------------------------------------------------------


def test_port_guard_reports_a_held_port(server):
    """A held port is described, and a free one is not.

    FAILS IF: `describe_port_conflict` returns None for a port that is genuinely
    held -- the launch would then proceed and two servers would answer on
    different ports, which is exactly the failure the shared port exists to
    prevent -- or if it invents a conflict on a free port.
    """
    held = socket.socket()
    held.bind((HOST, 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        message = frontend.describe_port_conflict(HOST, port)
        assert message is not None
        assert f"Port {port}" in message
        assert "Stop it with:" in message
    finally:
        held.close()

    free = socket.socket()
    free.bind((HOST, 0))
    free_port = free.getsockname()[1]
    free.close()
    assert frontend.describe_port_conflict(HOST, free_port) is None


def test_port_guard_names_the_other_server_and_its_extension(server):
    """When the holder is one of ours, the message names it and its build.

    FAILS IF: the message does not identify the app or the pid -- the reason for
    printing it at all is that only the human can confirm the process is theirs
    to kill.
    """
    port = int(server.rsplit(":", 1)[1])
    message = frontend.describe_port_conflict(HOST, port)
    assert message is not None
    assert "spacetime-bezier-frontend" in message
    assert f"kill {os.getpid()}" in message


def test_main_exits_1_rather_than_moving_to_a_free_port(monkeypatch, capsys, server):
    """The launcher refuses; it does not hop.

    FAILS IF: `main` returns 0 while another server holds the port, or binds
    anything at all -- a second live server answering on a second port is how a
    four-days-stale sandbox once kept serving pre-fix geometry invisibly.
    """
    port = int(server.rsplit(":", 1)[1])
    monkeypatch.setattr(frontend, "DEFAULT_PORT", port)

    def refuse(*args, **kwargs):
        raise AssertionError("main bound a server despite the port being held")

    monkeypatch.setattr(frontend, "make_server", refuse)
    assert frontend.main(["--host", HOST, "--no-open"]) == 1
    assert f"Port {port}" in capsys.readouterr().err


def test_the_cli_has_no_port_flag(server):
    """8767 is not configurable from the command line.

    The shared port IS the mutual exclusion between this page, the sandbox and
    the sanity viewer. A `--port` flag is a supported way to run two of them.

    FAILS IF: `--port` is accepted, or the default constant moves off 8767.
    """
    assert frontend.DEFAULT_PORT == 8767
    with pytest.raises(SystemExit):
        frontend.main(["--port", "9999", "--no-open"])


def test_cancel_kills_a_running_solve(server):
    """/api/cancel terminates the solve in flight, and only a solve in flight.

    FAILS IF: cancel claims to have cancelled when nothing runs, or a genuine
    in-flight solve survives it -- the pending request must come back as a 400
    naming the cancellation rather than run to completion.
    """
    status, data = post(server, "/api/cancel", {})
    assert status == 200 and data["cancelled"] is False

    result = {}

    def run():
        result["resp"] = post(
            server, "/api/solve",
            {"scenario": "wall", "N": 10, "n_seg": 24, "max_iter": 200},
        )

    worker = threading.Thread(target=run)
    worker.start()
    for _ in range(100):                     # wait for the child to register
        time.sleep(0.1)
        if frontend._ACTIVE_SOLVE["proc"] is not None:
            break
    status, data = post(server, "/api/cancel", {})
    assert status == 200 and data["cancelled"] is True
    worker.join(timeout=60)
    assert not worker.is_alive()
    solve_status, body = result["resp"]
    assert solve_status == 400
    assert "cancelled" in body["error"]


# ---------------------------------------------------------------------------
# (g) the result cache
# ---------------------------------------------------------------------------


def test_solve_cache_answers_with_the_stored_run_and_says_so(server, original):
    """An identical request is answered from the cache, marked as such.

    FAILS IF: the second answer re-solves (`cached` stays False), returns
    different control points than the stored run, or rewrites the stored
    provenance -- the cached copy must be the run's own response with only the
    `cached` flag flipped, because its solved_at and solve_ms describe THAT run.
    """
    status, again = post(server, "/api/solve", {"scenario": "original", "N": 8, "n_seg": 4})
    assert status == 200, again
    assert original["provenance"]["cached"] is False
    assert again["provenance"]["cached"] is True
    assert again["solution"]["control_points"] == original["solution"]["control_points"]
    assert again["verdict"] == original["verdict"]
    first = {k: v for k, v in original["provenance"].items() if k != "cached"}
    second = {k: v for k, v in again["provenance"].items() if k != "cached"}
    assert first == second


def test_solve_cache_key_distinguishes_parameters(server, original):
    """A changed parameter misses the cache and runs fresh.

    FAILS IF: a request differing only in `tol` is answered from the cache --
    a stored run presented as a run with parameters it did not have.
    """
    status, data = post(
        server, "/api/solve", {"scenario": "original", "N": 8, "n_seg": 4, "tol": 2e-6}
    )
    assert status == 200, data
    assert data["provenance"]["cached"] is False


def test_replay_cache_returns_the_same_frames(server, original):
    """The second identical replay request does not re-solve.

    FAILS IF: the repeat answer is not marked cached, or its frames differ from
    the first answer's -- either would mean the cache is returning some other
    run's trace.
    """
    payload = {
        "scenario": "original",
        "N": 8,
        "n_seg": 4,
        "elastic_weight": original["resolved"]["elastic_weight"],
    }
    status, first = post(server, "/api/replay", payload)
    assert status == 200, first
    status, second = post(server, "/api/replay", payload)
    assert status == 200, second
    assert second["cached"] is True
    assert second["frames"] == first["frames"]
    assert second["drift"] == first["drift"]


def test_module_entrypoint_serves_this_frontend():
    """`python3 -m spacetime_bezier` starts this server and not the old one.

    FAILS IF: `__main__` still points at `io.main` (the sandbox) or at
    `viewer.main` -- the entrypoint would launch a page none of the tests above
    cover.
    """
    import importlib

    entry = importlib.import_module("spacetime_bezier.__main__")
    assert entry.main is frontend.main
