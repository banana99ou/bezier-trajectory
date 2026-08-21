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


def test_catalog_offers_every_scenario_including_the_four_column_ones(server):
    """The catalog is the whole registry, with an axis picker for 4-column runs.

    FAILS IF: a registered scenario is missing (`viewer.py` dropped `wall3d` and
    `station_fence` rather than project them, which is what this page exists to
    fix), or a 4-column scenario comes back with fewer than the four axis triples
    the page needs to draw it honestly.
    """
    status, body, _ = get(server, "/api/scenarios")
    assert status == 200
    catalog = json.loads(body)["scenarios"]
    assert set(catalog) == set(SCENARIO_MAP)
    assert [v["id"] for v in catalog["original"]["views"]] == ["xyt"]
    assert [v["id"] for v in catalog["wall3d"]["views"]] == ["xyz", "xyt", "xzt", "yzt"]
    assert catalog["station_fence"]["default_view"] == "xyt"
    assert catalog["station_fence"]["stations"] == [[5.0, -2.0, 0.3]]


def test_axis_banners_say_what_the_vertical_axis_is(server):
    """Each view carries the sentence that keeps its picture honest.

    FAILS IF: a lift view stops announcing that the vertical axis is time, the
    spatial view stops announcing that time is the colour, or a 4-column
    projection stops naming the coordinate it drops -- the case where two curves
    appear to touch and are metres apart in the axis nobody mentioned.
    """
    views = {v["id"]: v for v in frontend.axis_views(4)}
    assert views["xyz"]["banner"] == "all three axes are space; TIME is the color"
    assert views["xyz"]["dropped_note"] is None
    for vid, dropped in (("xyt", "z"), ("xzt", "y"), ("yzt", "x")):
        assert views[vid]["banner"] == "the VERTICAL axis is TIME"
        assert f"coordinate {dropped} is dropped" in views[vid]["dropped_note"]
        assert views[vid]["cols"][-1] == 3, "time must be the vertical axis"
    assert frontend.axis_views(5) == [], "5 columns has no honest projection"
    assert frontend.axis_views(2) == []


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

    # One plane per (segment, obstacle): 4 segments x 3 obstacles. That IS the
    # G2 fix -- a count of 108 here would mean one plane per control point again.
    assert len(original["planes"]["koz"]) == 12
    assert original["planes"]["occlusion"] == []
    assert original["ledger"]["total"] == 108
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


def test_station_fence_draws_in_four_axis_triples(station_fence):
    """The 4-column scenario is reachable, with t vertical wherever it is shown.

    FAILS IF: the scenario comes back undrawable (the `viewer.py` behaviour this
    page replaces), the default view is not the lift, or any lift triple puts
    time somewhere other than the vertical axis.
    """
    views = station_fence["scenario"]["views"]
    assert [v["id"] for v in views] == ["xyz", "xyt", "xzt", "yzt"]
    assert station_fence["scenario"]["default_view"] == "xyt"
    for view in views:
        if view["kind"] == "lift":
            assert view["labels"][2] == "t"
            assert view["cols"][2] == 3


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


def test_module_entrypoint_serves_this_frontend():
    """`python3 -m spacetime_bezier` starts this server and not the old one.

    FAILS IF: `__main__` still points at `io.main` (the sandbox) or at
    `viewer.main` -- the entrypoint would launch a page none of the tests above
    cover.
    """
    import importlib

    entry = importlib.import_module("spacetime_bezier.__main__")
    assert entry.main is frontend.main
