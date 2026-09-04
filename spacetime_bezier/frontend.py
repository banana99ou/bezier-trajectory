"""The frontend server: one page, one port, two tempos.

This module replaces ``sandbox.py`` and ``viewer.py`` as the thing
``python3 -m spacetime_bezier`` starts. Those files stay on disk (their deletion
is a separate recorded decision); this one takes the entrypoint.

Four rules from CLAUDE.md sec. Architecture govern every function below, and each
one is a constraint on what may be computed WHERE:

* **One canonical execution model.** ``/api/solve`` runs the same ladder over the
  same ``optimize_spacetime`` call with the same defaults ``optimize_scenario``
  uses, so a number on the page is a number the batch path would report.
  ``/api/replay`` drives ``bezier_opt.SpacetimeScpContext`` through
  ``tools/trace_viewer.py``'s child script -- imported, not copied, so the
  parameter list cannot drift -- and returns that script's own drift check.
* **Rust is the sole engine.** Constraint rows come from
  ``spacetime_koz_rows_exact``, the builder the certificate itself uses -- ONE
  call, returning the obstacle's own walls and its shadow's walls together, split
  here on the row's station index. Nothing here re-derives a half-space.
* **Trace is observability over the real run.** The replay frames are the
  stepping context's own state; there is no viewer-side stepper.
* **Geometry authenticity.** Everything the page draws is either a solver output,
  a scenario parameter, or a bounded patch of a solver-emitted half-space
  computed here from that row's own normal. The lift view's shadow cone is gone
  with the conservative containing ball it was tessellated from: since the shadow
  moved onto the center surface there is no such ball, and drawing a cone the
  solver never built would be a picture of a constraint that is not there. The
  shadow's walls are exported like any other wall.

The verdicts are computed here and not in the browser. ``figure_grade`` comes
from ``figure_grade_failures``, reached through the same info-key mapping
``tools/make_paper_figure.py`` uses (NaN defaults, so absent evidence refuses);
clearance is recomputed independently by ``compute_min_clearance``; line of sight
by ``compute_los_margin``. The client picks columns and assembles plotly traces.

Run with:  python3 -m spacetime_bezier
"""

from __future__ import annotations

import argparse
import importlib.util
import importlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from orbital_docking.bezier import BezierCurve
from orbital_docking.de_casteljau import segment_matrices_equal_params

from .geometry import (
    bezier_curve,
    compute_los_margin,
    compute_min_clearance,
    obstacle_array_bundle,
)
from .objective import build_initial_guess
from .optimize import (
    DEFAULT_TRUST_RADIUS,
    DEFAULT_INITIAL_ELASTIC_WEIGHT,
    _STOP_REASONS,
    figure_grade_failures,
    optimize_spacetime,
)
from .scenarios import SCENARIO_MAP, scenario_elastic_weight

DEFAULT_HOST = "127.0.0.1"

# Hard-wired, and there is no CLI flag to move it. The shared port IS the mutual
# exclusion between this server, the sandbox and the sanity viewer: two live
# servers on two default ports is how a four-days-stale sandbox once kept
# answering with pre-G1/G2 geometry, invisibly. `make_server` takes a port so a
# test can bind an ephemeral one; nothing else may pass it.
DEFAULT_PORT = 8767

HEALTH_PATH = "/api/health"
HEALTH_APP_ID = "spacetime-bezier-frontend"

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[1]

# `optimize_scenario`'s defaults, which produced every number in README
# sec. Measurements. Not "reasonable values" -- the same values, so that a solve
# requested from the page with the panel untouched is the batch run.
SOLVE_DEFAULTS = {
    "max_iter": 200,
    "tol": 1e-6,
    "scp_prox_weight": 0.3,
    "trust_radius": DEFAULT_TRUST_RADIUS,
    "min_dt": 0.1,
    # PAPER_1 statement (8), off by default because that is what produced every
    # number in README sec. Measurements. It is exposed on the panel because the
    # gate now READS what the certificate covers: at the default clip most
    # configurations return pairs the certificate does not cover, so a solve with
    # this off is honestly reported as not figure-grade rather than passing on a
    # condition nothing checked.
    "sound_clip": True,
}

# Dense samples of the returned curve. The same evaluator the clearance check
# uses, so the curve drawn is the curve that was checked.
CURVE_SAMPLES = 400

# Independent clearance sample count, matching `tools/make_paper_figure.py` --
# the number the gate is fed there is the number the gate is fed here.
CLEARANCE_SAMPLES = 20001
LOS_SAMPLES = 2001
SIGHT_LINES = 16

# Rows sent to the constraint ledger, tightest first. `wall` N10_seg16 has 2640
# exact rows; the certificate below is summed over ALL of them, and only the
# table is truncated. Both counts travel with it so the truncation is visible.
LEDGER_ROWS = 400

_STARTED_AT = datetime.now().isoformat(timespec="seconds")
_SOLVE_LOCK = threading.Lock()

# In-process result cache, keyed on the RESOLVED request parameters (never the
# raw body, whose key order and blank-vs-absent fields vary). A hit returns the
# stored response byte-for-byte with `provenance.cached` flipped to True -- the
# run's own solved_at and solve_ms stay, because they describe the run.
#
# Why in-process caching cannot serve stale geometry: the Rust extension loads
# once per process, so changing the solver requires a server restart, and the
# restart empties the cache with it. There is no key for the build because the
# process IS the build.
_CACHE_LOCK = threading.Lock()
_SOLVE_CACHE: OrderedDict[str, dict] = OrderedDict()
_REPLAY_CACHE: OrderedDict[str, dict] = OrderedDict()
_SOLVE_CACHE_MAX = 24
_REPLAY_CACHE_MAX = 8


def _cache_get(cache: OrderedDict, key: str) -> dict | None:
    with _CACHE_LOCK:
        entry = cache.get(key)
        if entry is None:
            return None
        cache.move_to_end(key)
        return entry


def _cache_put(cache: OrderedDict, key: str, value: dict, limit: int) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        while len(cache) > limit:
            cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Provenance: which code answered, and is it the code you last built
# ---------------------------------------------------------------------------


def extension_build_time() -> float | None:
    """Modification time of the compiled Rust extension this process imports.

    The one number that separates "a server is running" from "a server is running
    the solver you just built": the extension loads once per process, so a server
    started before a `maturin develop` answers with old geometry for its whole
    life unless this is surfaced.
    """
    try:
        import bezier_opt
    except ImportError:
        return None
    path = getattr(bezier_opt, "__file__", None)
    if not path:
        return None
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def newest_rust_source_time() -> float | None:
    """Modification time of the most recently edited ``.rs`` file in the tree."""
    try:
        times = [p.stat().st_mtime for p in (REPO_ROOT / "rust_optimizer").rglob("*.rs")]
    except OSError:
        return None
    return max(times) if times else None


def git_provenance() -> dict:
    def _run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = _run("rev-parse", "--short", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
    }


def provenance_payload() -> dict:
    """Everything the header strip needs to say which code produced the picture."""
    built = extension_build_time()
    newest = newest_rust_source_time()
    try:
        import bezier_opt

        ext_path = getattr(bezier_opt, "__file__", None)
    except ImportError:
        ext_path = None
    # "Stale" is a claim about the build, so it is only made when both numbers
    # exist. Unknown is reported as unknown, never as fresh.
    stale = None
    if built is not None and newest is not None:
        stale = bool(newest > built)
    return {
        **git_provenance(),
        "extension_path": ext_path,
        "extension_build_time": built,
        "newest_rust_source_time": newest,
        "extension_stale": stale,
    }


def health_payload() -> dict:
    return {
        "app": HEALTH_APP_ID,
        "pid": os.getpid(),
        "started": _STARTED_AT,
        "executable": sys.executable,
        **provenance_payload(),
    }


# ---------------------------------------------------------------------------
# Port guard -- the pattern from sandbox.py / viewer.py, unchanged in behaviour
# ---------------------------------------------------------------------------


def port_is_listening(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_server(host: str, port: int, timeout: float = 0.5) -> dict | None:
    """Health payload of whatever answers on ``(host, port)``, else None."""
    url = f"http://{host}:{port}{HEALTH_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or "app" not in payload:
        return None
    return payload


def describe_port_conflict(host: str, port: int) -> str | None:
    """Explain who holds ``port``, or None when it is free.

    Launching fails rather than hopping to a free port, and that is the whole
    point of the shared port. The stale server has to be visible, and only the
    human can confirm the process is theirs to kill.
    """
    if not port_is_listening(host, port):
        return None

    lines = [f"Port {port} on {host} is already in use."]
    other = probe_server(host, port)
    if other is None:
        lines.append("Whatever holds it is not a space-time Bezier server.")
        stop_hint = f"lsof -nP -iTCP:{port} -sTCP:LISTEN     # then kill the pid"
    else:
        lines.append(
            f"It is `{other.get('app')}`: pid {other.get('pid')}, up since {other.get('started')}."
        )
        lines.append(f"  interpreter: {other.get('executable')}")
        theirs = other.get("extension_build_time")
        mine = extension_build_time()
        if theirs is None:
            lines.append("  WARNING: it has no Rust extension loaded, so every solve there fails.")
        elif mine is not None and mine > theirs + 1.0:
            built = datetime.fromtimestamp(theirs).isoformat(timespec="seconds")
            local = datetime.fromtimestamp(mine).isoformat(timespec="seconds")
            lines.append(
                f"  WARNING: it loaded the Rust extension built {built}, but yours is "
                f"from {local}. It is solving with old code."
            )
        stop_hint = f"kill {other.get('pid')}"
    lines.append(f"Stop it with:  {stop_hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def _json_safe(value):
    """Coerce numpy types and map non-finite floats to None (JSON null).

    ``compute_los_margin`` reports ``+inf`` where no occluder is active, so nulls
    reach the client legitimately. Every verdict the client would otherwise have
    to derive from one -- "was the link lost here" -- is computed on this side and
    sent as a boolean.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


class RequestError(ValueError):
    """A bad request. Answered 400 with a message, never a 500 traceback."""


def _positive_int(payload: dict, key: str, default: int) -> int:
    raw = payload.get(key, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise RequestError(f"{key} must be an integer, got {raw!r}") from None
    if value < 1:
        raise RequestError(f"{key} must be >= 1, got {value}")
    return value


def _float_or_none(payload: dict, key: str):
    raw = payload.get(key)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise RequestError(f"{key} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise RequestError(f"{key} must be finite, got {raw!r}")
    return value


def _float(payload: dict, key: str, default: float) -> float:
    value = _float_or_none(payload, key)
    return float(default) if value is None else value


# ---------------------------------------------------------------------------
# Axis views
#
# Which coordinate triples this page will draw, and the sentence that has to sit
# above each one. Computed here rather than in the browser because these strings
# are the honesty of the picture: a projection that silently drops a coordinate
# is exactly the "time-sliced 2D plane labelled as a 3D surface" the geometry
# authenticity rule forbids. Time is the LAST column always, and it is always the
# vertical axis when it is shown.
# ---------------------------------------------------------------------------

LIFT_BANNER = "the VERTICAL axis is TIME"
SPATIAL_BANNER = "all three axes are space; TIME is the color"


def _dropped_note(name: str) -> str:
    return (
        f"coordinate {name} is dropped — apparent contact may be separation in {name}"
    )


def axis_views(dim: int) -> list[dict]:
    """The coordinate triples that can be drawn honestly for ``dim`` columns.

    ``dim`` counts spatial coordinates PLUS the time column. Three columns is the
    lifted plane and has exactly one view, so the page shows no picker. Four
    columns get the four triples the spec fixes. Anything else returns an empty
    list, and the page refuses to draw rather than guess which column is which.
    """
    names = ["x", "y", "z"][: dim - 1] + ["t"]
    if dim == 3:
        return [{
            "id": "xyt",
            "button": "(x,y,t) lift",
            "cols": [0, 1, 2],
            "labels": ["x", "y", "t"],
            "space_idx": [0, 1],
            "kind": "lift",
            "dropped": None,
            "banner": LIFT_BANNER,
            "dropped_note": None,
        }, {
            # Top-down instant view: both spatial coordinates, no third axis.
            # `kind: "flat"` tells the client to draw a 2D-style scene driven by
            # the time cursor -- the one place a time scrubber is the honest
            # representation, because nothing here claims to be the lift.
            "id": "xy",
            "button": "(x,y) top-down",
            "cols": [0, 1],
            "labels": ["x", "y"],
            "space_idx": [0, 1],
            "kind": "flat",
            "dropped": None,
            "banner": "top-down (x, y); TIME is the color and the cursor",
            "dropped_note": None,
        }]
    if dim == 4:
        # Two spatial views (user decision 2026-08-24: "I just need xy(z)t"),
        # plus the (x,y,t) lift brought back 2026-08-26 at the same user's
        # request while designing `loiter` -- the temporal-slot behaviour that
        # scenario exists for is invisible in a spatial view, because the slot
        # IS the time axis. (x,z,t) and (y,z,t) stay gone. Time is carried by
        # the slider in one spatial view and by color in the other. Every view
        # projects the half-space patches and hulls, which live in (x,y,z,t) --
        # named in the note so the picture never claims to be more than it is.
        proj_note = "half-space patches and hulls are projections from (x,y,z,t)"
        return [{
            "id": "xyz",
            "button": "(x,y,z) @ t",
            "cols": [0, 1, 2],
            "labels": ["x", "y", "z"],
            "space_idx": [0, 1, 2],
            "kind": "spatial",
            "dropped": "t",
            "banner": "all three axes are space; the slider sets the instant",
            "dropped_note": proj_note,
        }, {
            "id": "xyz_all",
            "button": "(x,y,z) all t",
            "cols": [0, 1, 2],
            "labels": ["x", "y", "z"],
            "space_idx": [0, 1, 2],
            "kind": "spatial_all",
            "dropped": "t",
            "banner": SPATIAL_BANNER,
            "dropped_note": proj_note,
        }, {
            "id": "xyt",
            "button": "(x,y,t) lift",
            "cols": [0, 1, 3],
            "labels": ["x", "y", "t"],
            "space_idx": [0, 1],
            "kind": "lift",
            "dropped": "z",
            "banner": LIFT_BANNER,
            "dropped_note": _dropped_note("z") + "; " + proj_note,
        }]
    return []


DEFAULT_VIEW = {3: "xyt", 4: "xyz"}


# ---------------------------------------------------------------------------
# Scenario catalog
# ---------------------------------------------------------------------------


def scenario_catalog() -> dict:
    """Every registered scenario, with the configs it is registered with.

    Unlike ``viewer.py``, nothing is filtered out: a four-column scenario gets an
    axis picker instead of being withheld, so `fence3d` and `loiter` are
    reachable from the page.
    """
    catalog = {}
    for name, (scenario_fn, configs) in SCENARIO_MAP.items():
        scenario = scenario_fn()
        dim = len(scenario["start"])
        catalog[name] = {
            "name": scenario["name"],
            "title": scenario["title"],
            "dim": dim,
            "obstacles": scenario["obstacles"],
            "start": list(scenario["start"]),
            "end": list(scenario["end"]),
            "T": float(scenario["T"]),
            "stations": scenario.get("stations"),
            "coord_bounds": scenario.get("coord_bounds"),
            "trust_radius": float(scenario.get("trust_radius", DEFAULT_TRUST_RADIUS)),
            "configs": [[int(N), int(n_seg)] for N, n_seg in configs],
            "registered_elastic_weight": scenario_elastic_weight(name),
            "views": axis_views(dim),
            "default_view": DEFAULT_VIEW.get(dim),
            "drawable": bool(axis_views(dim)),
        }
    return {
        "scenarios": catalog,
        "solve_defaults": dict(SOLVE_DEFAULTS),
        "initial_elastic_weight": float(DEFAULT_INITIAL_ELASTIC_WEIGHT),
    }


def reload_scenarios() -> dict:
    """Re-read ``scenarios.py`` from disk and drop every cached result.

    Exists for the scenario-tuning loop: the solve child re-imports the module
    fresh on every run anyway, but the parent's catalog and the result cache do
    not -- so an edited scenario used to require a full server restart, and
    worse, the cache key carries only the scenario NAME, so a re-solve after an
    edit could silently answer with the pre-edit result. One button instead of
    restart-retab-reselect. The compiled Rust extension is NOT reloaded --
    rebuilding that still requires a restart, and the staleness banner says so.
    """
    from . import scenarios as _scenarios_module

    importlib.reload(_scenarios_module)
    global SCENARIO_MAP, scenario_elastic_weight
    SCENARIO_MAP = _scenarios_module.SCENARIO_MAP
    scenario_elastic_weight = _scenarios_module.scenario_elastic_weight
    with _CACHE_LOCK:
        _SOLVE_CACHE.clear()
        _REPLAY_CACHE.clear()
    return scenario_catalog()


# ---------------------------------------------------------------------------
# The canonical solve
# ---------------------------------------------------------------------------


def _run_ladder(scenario: dict, N: int, n_seg: int, params: dict):
    """``optimize_scenario``'s solve for one configuration, keeping the raw info.

    Deliberately the same shape as ``optimize.optimize_scenario``: ONE solve,
    with the elastic weight escalated in-loop by the solver (SNOPT elastic
    mode; see ``optimize.DEFAULT_INITIAL_ELASTIC_WEIGHT``). The name is kept so
    every caller and cache key stays put; the "rungs" return is now a single
    entry describing the run's weight trajectory, which is what the header
    table renders.

    It is written out here instead of calling ``optimize_scenario`` for one
    reason: that function returns a summary row and drops ``info``, and the
    solver's OWN reported clearance is one leg of a pair the header shows against
    an independent recomputation. Two Python samplings of the same curve would
    not be that pair. ``tests/integration/test_frontend.py`` pins the two paths
    together on `original` -- identical control points, or this drifted.
    """
    initial_weight = (
        DEFAULT_INITIAL_ELASTIC_WEIGHT
        if params["elastic_weight"] is None
        else float(params["elastic_weight"])
    )
    obstacles = scenario["obstacles"]
    dim = len(scenario["start"])
    stations = scenario.get("stations")

    P_opt, info_opt = optimize_spacetime(
        N=N,
        dim=dim,
        p_start=scenario["start"],
        p_end=scenario["end"],
        obstacles=obstacles,
        n_seg=n_seg,
        max_iter=params["max_iter"],
        tol=params["tol"],
        scp_prox_weight=params["scp_prox_weight"],
        scp_trust_radius=params["trust_radius"],
        elastic_weight=initial_weight,
        min_dt=params["min_dt"],
        sound_clip=params["sound_clip"],
        v_max=params["v_max"],
        time_weight=params["time_weight"],
        free_arrival_time=params["free_arrival_time"],
        stations=stations,
        coord_bounds=scenario.get("coord_bounds"),
        verbose=False,
        init_curve=scenario.get("init_curve"),
    )
    clearance = compute_min_clearance(P_opt, obstacles, dim=dim, n_eval=3000)
    cert = float(info_opt.get("koz_violation_reference", float("nan")))
    occ = float(info_opt.get("occlusion_violation_reference", 0.0))
    cleared = (
        bool(info_opt.get("converged", 0.0))
        and cert <= 1e-6
        and occ <= 1e-6
        and clearance > 0.0
    )
    used_weight = float(info_opt.get("final_elastic_weight", initial_weight))
    rungs = [{
        "elastic_weight": used_weight,
        "initial_elastic_weight": float(initial_weight),
        "weight_raises": int(info_opt.get("weight_raises", 0)),
        "max_koz_dual": float(info_opt.get("max_koz_dual", float("nan"))),
        "clearance": float(clearance),
        "certificate": cert,
        "occlusion": occ,
        "cleared": cleared,
    }]
    return P_opt, info_opt, used_weight, float(clearance), rungs


def _gate_row(info: dict, clearance: float, has_stations: bool) -> dict:
    """The info-key mapping ``tools/make_paper_figure.py:figure_grade_or_die`` uses.

    Copied rather than reinvented, and the copy carries the reason. An earlier
    version of that function rewrote the gate conditions inline as ``if x > tol``,
    which INVERTS the NaN polarity: the real gate is written ``if not (x <= tol)``
    precisely so that a missing or NaN input -- stale extension, no accepted step
    -- fails. The rewrite drew in exactly those cases. So the keys are mapped with
    NaN defaults, absent evidence refuses, and ``figure_grade_failures`` is called
    rather than re-expressed; any condition the gate grows later arrives here for
    free as long as its row key is forwarded.

    The one generalisation over that script: it hard-codes a NaN default for
    ``occlusion_violation`` because `loiter` always has a station, so a
    missing key means a stale extension. Here the scenario may genuinely have no
    station, and a run with no occlusion rows has nothing to violate -- so the key
    is only forced to NaN when stations are present, and otherwise left for
    ``figure_grade_failures`` to default to 0.0.
    """
    row = {
        "converged": bool(info.get("converged", 0.0)),
        "stop_label": _STOP_REASONS.get(int(info.get("stop_reason", -1)), "unknown"),
        "certificate_violation": float(info.get("koz_violation_reference", np.nan)),
        "total_slack": float(info.get("total_koz_slack_returned", np.nan)),
        "min_clearance": float(clearance),
    }
    if has_stations:
        row["occlusion_violation"] = float(
            info.get("occlusion_violation_reference", np.nan)
        )
    for passthrough in (
        "speed_cap_violation",
        "occlusion_planes_dropped",
        "koz_unsound_clips",
    ):
        if passthrough in info:
            row[passthrough] = float(info[passthrough])
    return row


# ---------------------------------------------------------------------------
# Half-space patches
#
# A supporting half-space is unbounded, and an unbounded plane in a 3D scene
# tells the reader nothing about WHICH control points it separates. Every patch
# below is therefore cut to the extent of the segment hull it supports: the same
# hull the row was built for, so the patch is exactly as wide as the claim.
# ---------------------------------------------------------------------------

# Floor on a patch half-width, so a degenerate (near-collinear) segment hull still
# produces something visible. Purely a drawing size; the row it comes from is
# unchanged, and the margin travels with it.
_MIN_PATCH_HALF_WIDTH = 0.15


def _tangent_basis(normal: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal directions inside the plane ``normal . q = const``.

    The first follows the hull's own leading direction, so a patch lines up with
    the geometry it supports rather than with the coordinate axes. The second is
    completed from the standard basis when the hull is nearly one-dimensional and
    the SVD's second right singular vector would be an arbitrary direction of the
    ambient space rather than of the plane.
    """
    n_unit = normal / np.linalg.norm(normal)
    if spread.shape[0] >= 2:
        _u, sing, vt = np.linalg.svd(spread, full_matrices=False)
    else:
        sing, vt = np.zeros(1), np.zeros((1, normal.shape[0]))

    def _in_plane(vec):
        vec = vec - (vec @ n_unit) * n_unit
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else None

    u1 = None
    for k in range(vt.shape[0]):
        if sing[k] > 1e-9:
            u1 = _in_plane(vt[k])
            if u1 is not None:
                break
    if u1 is None:
        for axis in np.eye(normal.shape[0]):
            u1 = _in_plane(axis)
            if u1 is not None:
                break
    u2 = None
    for k in range(vt.shape[0]):
        if sing[k] > 1e-9:
            cand = _in_plane(vt[k])
            if cand is not None and np.linalg.norm(cand - (cand @ u1) * u1) > 1e-6:
                u2 = cand - (cand @ u1) * u1
                u2 = u2 / np.linalg.norm(u2)
                break
    if u2 is None:
        for axis in np.eye(normal.shape[0]):
            cand = axis - (axis @ n_unit) * n_unit
            cand = cand - (cand @ u1) * u1
            if np.linalg.norm(cand) > 1e-6:
                u2 = cand / np.linalg.norm(cand)
                break
    return u1, u2


def _plane_patch(normal, lower_bound: float, hull: np.ndarray) -> dict | None:
    """A bounded quadrilateral of the half-space ``normal . q >= lower_bound``.

    Returned in the FULL lifted space (all ``dim`` columns). The client selects
    the displayed columns, which is a linear projection, so the projected corners
    still bound a flat quadrilateral -- the shadow of a real patch of a real
    supporting hyperplane, never a plane synthesised in the view.

    ``free_arrow`` is two points, not a direction: a normal does not survive
    projection the way a point does, so the free side is indicated by the
    projection of an actual segment that starts on the patch and ends on the
    permitted side of it.
    """
    n = np.asarray(normal, dtype=float)
    nn = float(n @ n)
    if not math.isfinite(nn) or nn <= 1e-24:
        return None
    hull = np.asarray(hull, dtype=float)
    centroid = hull.mean(axis=0)
    # Foot of the perpendicular from the hull centroid: the patch sits on the
    # plane, directly "under" the points it constrains.
    center = centroid + (lower_bound - float(n @ centroid)) / nn * n
    spread = hull - center
    spread = spread - np.outer(spread @ n, n) / nn
    u1, u2 = _tangent_basis(n, spread)
    if u1 is None or u2 is None:
        return None
    e1 = max(float(np.max(np.abs(spread @ u1))), _MIN_PATCH_HALF_WIDTH)
    e2 = max(float(np.max(np.abs(spread @ u2))), _MIN_PATCH_HALF_WIDTH)
    corners = [
        center - e1 * u1 - e2 * u2,
        center + e1 * u1 - e2 * u2,
        center + e1 * u1 + e2 * u2,
        center - e1 * u1 + e2 * u2,
    ]
    arrow_len = 0.35 * max(e1, e2)
    n_unit = n / math.sqrt(nn)
    return {
        "corners": [c.tolist() for c in corners],
        "free_arrow": [center.tolist(), (center + arrow_len * n_unit).tolist()],
    }


def _koz_planes(
    P: np.ndarray, obstacles: list[dict], n_seg: int, a_list, dim: int, stations=None
):
    """One patch per (segment, obstacle, component), plus the full exact-row ledger.

    Both come from ``spacetime_koz_rows_exact`` -- the builder the certificate
    itself uses -- evaluated at the RETURNED control points. Grouping is not a
    summarisation: there IS one plane per group, shared by every control point of
    that segment, so the group has one normal and one bound. Its margin is the
    tightest of its control points, which is the number that decides whether the
    plane is active.

    **The component index is part of the key.** One obstacle can present two
    separated lumps of tube to one segment -- the clip ball cuts the centreline
    twice -- and each lump gets its own wall. Grouping on (segment, obstacle)
    alone would fold two genuinely different planes into one and draw whichever
    happened to have the tighter margin, which is a picture of a constraint the
    solver never had.
    """
    import bezier_opt

    spatial_dim = dim - 1
    obstacle_ctrl, obstacle_radii = obstacle_array_bundle(obstacles, spatial_dim)
    # The builder also reports the clip radius per row and the two hole counts
    # (idea/spacetime.md statements 7 and 8). The drawing does not use them yet, but they
    # are unpacked by name so a future widening of the tuple fails loudly here
    # rather than silently mis-assigning a column.
    (
        normals, lbs, seg, cp, obs, comp, sta, _rho, _sound,
        _dropped, _dropped_shadow, _unsound,
    ) = bezier_opt.spacetime_koz_rows_exact(
        p=P,
        obstacle_ctrl=obstacle_ctrl,
        obstacle_r=obstacle_radii,
        n_seg=n_seg,
        stations=(
            None if stations is None
            else np.asarray(stations, dtype=float).reshape(-1, spatial_dim)
        ),
    )
    normals = np.asarray(normals, dtype=float).reshape(-1, dim)
    lbs = np.asarray(lbs, dtype=float)
    seg = np.asarray(seg, dtype=int)
    cp = np.asarray(cp, dtype=int)
    obs = np.asarray(obs, dtype=int)
    comp = np.asarray(comp, dtype=int)
    sta = np.asarray(sta, dtype=int)

    hulls = [np.asarray(a, dtype=float) @ P for a in a_list]
    # ONE row set, split by which generator built each wall. A row with a station
    # index is a wall on that station's shadow; -1 is the obstacle's own zone.
    # The split is bookkeeping, not a second geometry -- they come from the same
    # call, at the same iterate, through the same builder.
    ledger: list[dict] = []
    occ_ledger: list[dict] = []
    groups: dict[tuple[int, int, int, int], dict] = {}
    for k in range(normals.shape[0]):
        s_i, c_i, o_i, j_i = int(seg[k]), int(cp[k]), int(obs[k]), int(comp[k])
        st_i = int(sta[k])
        q = hulls[s_i][c_i]
        slack = float(normals[k] @ q) - float(lbs[k])
        row = {
            "seg": s_i,
            "cp": c_i,
            "obs": o_i,
            "component": j_i,
            "station": st_i,
            "n_spatial": normals[k, :-1].tolist(),
            "n_time": float(normals[k, -1]),
            "slack": slack,
        }
        (occ_ledger if st_i >= 0 else ledger).append(row)
        key = (s_i, o_i, j_i, st_i)
        entry = groups.get(key)
        if entry is None or slack < entry["margin"]:
            groups[key] = {
                "seg": s_i,
                "obs": o_i,
                "component": j_i,
                "station": st_i,
                "normal": normals[k].tolist(),
                "lb": float(lbs[k]),
                "margin": slack,
            }

    planes, occ_planes = [], []
    for key in sorted(groups):
        entry = groups[key]
        patch = _plane_patch(entry["normal"], entry["lb"], hulls[entry["seg"]])
        if patch is None:
            continue
        if entry["station"] >= 0:
            occ_planes.append({**entry, **patch, "kind": "occlusion"})
        else:
            planes.append({**entry, **patch, "kind": "koz"})

    def _summary(rows):
        cert = sum(max(0.0, -row["slack"]) for row in rows)
        return cert, sum(1 for row in rows if row["slack"] < 0.0)

    certificate, violated = _summary(ledger)
    occ_certificate, occ_violated = _summary(occ_ledger)
    # A zero time coefficient is now a REPORTABLE DEFECT on every row, shadow rows
    # included. They used to be time-parallel prisms by construction; the center
    # surface carries the obstacle's own time coordinate, so a shadow wall is a
    # space-time wall like any other and a zero here means the same thing G1 meant.
    zero_time = sum(
        1 for row in ledger + occ_ledger if abs(row["n_time"]) < 1e-12
    )
    ledger.sort(key=lambda row: row["slack"])
    occ_ledger.sort(key=lambda row: row["slack"])
    return {
        "planes": planes,
        "ledger": ledger[:LEDGER_ROWS],
        "ledger_total": len(ledger),
        "ledger_violated": violated,
        "ledger_zero_time": zero_time,
        "ledger_truncated": len(ledger) > LEDGER_ROWS,
        "certificate_recomputed": float(certificate),
        "occlusion": {
            "planes": occ_planes,
            "rows": occ_ledger[:LEDGER_ROWS],
            "rows_total": len(occ_ledger),
            "rows_violated": occ_violated,
            "certificate_recomputed": float(occ_certificate),
            # The conservative containing ball this used to draw a cone from no
            # longer exists: the shadow is generated by the center surface and its
            # walls ARE the geometry, drawn above. Emitting a cone the solver
            # never built would be a picture of a constraint that is not there.
            "shadow_balls": [],
        },
    }


# ---------------------------------------------------------------------------
# Solve endpoint
# ---------------------------------------------------------------------------


def _physical_speed(P: np.ndarray, num_pts: int) -> list[float]:
    """|d(spatial)/dt| at the same parameters the curve is drawn at.

    Computed from the velocity control points -- the ``E D P`` operators in
    ``orbital_docking.bezier`` -- so it is the curve's own derivative and not a
    finite difference of the drawn polyline. Both numerator and denominator are
    derivatives with respect to the curve PARAMETER, and their ratio is the
    physical speed because the parameter cancels. Non-positive ``dt`` cannot occur
    for a solver output (the monotonicity rows forbid it) and is reported as a
    null rather than as a number.
    """
    curve = BezierCurve(np.asarray(P, dtype=float))
    v_ctrl = np.asarray(curve.velocity_control_points(), dtype=float)
    v_samples = bezier_curve(v_ctrl, num_pts=num_pts)
    dt = v_samples[:, -1]
    space = np.linalg.norm(v_samples[:, :-1], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 1e-12, space / dt, np.nan)
    return [float(s) for s in speed]


def _solve_params(payload: dict):
    """Validate one solve request; raises ``RequestError`` on anything off.

    Shared by the in-process solve body and the subprocess wrapper, so a bad
    request is refused in the parent, fast, before a child is ever spawned.
    """
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    name = payload.get("scenario")
    if name not in SCENARIO_MAP:
        raise RequestError(
            f"unknown scenario {name!r}; known: {', '.join(sorted(SCENARIO_MAP))}"
        )
    scenario_fn, configs = SCENARIO_MAP[name]
    dim = len(scenario_fn()["start"])
    if not axis_views(dim):
        raise RequestError(
            f"scenario {name!r} has {dim} coordinates; this page draws 3 or 4 and "
            "refuses to guess which column is time"
        )
    N = _positive_int(payload, "N", configs[0][0])
    n_seg = _positive_int(payload, "n_seg", configs[0][1])
    params = {
        "max_iter": _positive_int(payload, "max_iter", SOLVE_DEFAULTS["max_iter"]),
        "tol": _float(payload, "tol", SOLVE_DEFAULTS["tol"]),
        "scp_prox_weight": SOLVE_DEFAULTS["scp_prox_weight"],
        "trust_radius": _float(payload, "trust_radius", SOLVE_DEFAULTS["trust_radius"]),
        "min_dt": _float(payload, "min_dt", SOLVE_DEFAULTS["min_dt"]),
        "elastic_weight": _float_or_none(payload, "elastic_weight"),
        "v_max": _float_or_none(payload, "v_max"),
        "time_weight": _float(payload, "time_weight", 0.0),
        "free_arrival_time": bool(payload.get("free_arrival_time", False)),
        "sound_clip": bool(payload.get("sound_clip", SOLVE_DEFAULTS["sound_clip"])),
    }
    if params["trust_radius"] <= 0.0:
        raise RequestError(f"trust_radius must be > 0, got {params['trust_radius']}")
    if params["tol"] <= 0.0:
        raise RequestError(f"tol must be > 0, got {params['tol']}")
    if params["min_dt"] <= 0.0:
        raise RequestError(f"min_dt must be > 0, got {params['min_dt']}")
    return name, N, n_seg, params


def solve_from_payload(payload: dict) -> dict:
    """Run one configuration and return everything the page draws.

    Every verdict in the response was computed here, from the solver's own
    numbers or from an independent recomputation against the true geometry. The
    client selects columns and builds plotly traces; it decides nothing.

    This is the synchronous, in-process solve -- the code path the parity test
    pins against ``optimize_scenario``. The HTTP endpoint reaches it through
    ``solve_via_subprocess`` so a running solve can be cancelled.
    """
    name, N, n_seg, params = _solve_params(payload)
    scenario_fn, configs = SCENARIO_MAP[name]
    scenario = scenario_fn()
    dim = len(scenario["start"])
    views = axis_views(dim)

    stations = scenario.get("stations")
    P_init = build_initial_guess(
        scenario["start"], scenario["end"], N + 1, init_curve=scenario.get("init_curve")
    )
    started = time.perf_counter()
    # One solve at a time. Two concurrent runs would interleave nothing in the
    # Rust, but they would double the wall clock reported on the page.
    with _SOLVE_LOCK:
        P_opt, info, used_weight, clearance_3000, rungs = _run_ladder(
            scenario, N, n_seg, params
        )
    solve_ms = (time.perf_counter() - started) * 1000.0

    P_opt = np.asarray(P_opt, dtype=float)
    a_list = [np.asarray(a, dtype=float) for a in segment_matrices_equal_params(N, n_seg)]

    # The independent leg. Separate implementation, separate language, and the
    # sample count the paper figure's gate is fed.
    clearance = float(
        compute_min_clearance(P_opt, scenario["obstacles"], dim=dim, n_eval=CLEARANCE_SAMPLES)
    )
    koz = _koz_planes(
        P_opt, scenario["obstacles"], n_seg, a_list, dim, stations=stations or None
    )

    occlusion = None
    los = None
    sight = None
    if stations:
        occlusion = koz["occlusion"]
        t_vals, margins = compute_los_margin(
            P_opt, stations[0], scenario["obstacles"], dim=dim, n_eval=LOS_SAMPLES
        )
        finite = margins[np.isfinite(margins)]
        los = {
            "t": t_vals.tolist(),
            "m": margins.tolist(),
            "min": float(finite.min()) if finite.size else float("inf"),
            "lost": bool(np.any(margins < 0.0)),
        }

    curve = bezier_curve(P_opt, num_pts=CURVE_SAMPLES)
    speed = _physical_speed(P_opt, CURVE_SAMPLES)

    if stations and los is not None:
        # Sight lines are sampled from the INDEPENDENT margin array, so a red line
        # is a line the independent check called lost -- not one the solver's rows
        # called lost. `lost` is decided here; the client only picks the colour.
        t_arr = np.asarray(los["t"], dtype=float)
        m_arr = np.asarray(los["m"], dtype=float)
        step = max(1, curve.shape[0] // SIGHT_LINES)
        lines = []
        for k in range(0, curve.shape[0], step):
            t_q = float(curve[k, -1])
            j = int(np.argmin(np.abs(t_arr - t_q)))
            margin = float(m_arr[j])
            lines.append({
                "v": curve[k, : dim - 1].tolist(),
                "t": t_q,
                "m": margin,
                "lost": bool(margin < 0.0),
            })
        sight = {"station": [float(v) for v in stations[0]], "lines": lines}

    gate_row = _gate_row(info, clearance, bool(stations))
    reasons = figure_grade_failures(gate_row)
    certificate_reported = float(info.get("koz_violation_reference", float("nan")))
    clearance_reported = float(info.get("min_clearance", float("nan")))

    verdict = {
        "figure_grade": not reasons,
        "figure_grade_reasons": reasons,
        "converged": bool(info.get("converged", 0.0)),
        "stop_reason": int(info.get("stop_reason", -1)),
        "stop_label": _STOP_REASONS.get(int(info.get("stop_reason", -1)), "unknown"),
        "iterations": int(info.get("iterations", -1)),
        "accept_count": int(info.get("accept_count", 0)),
        "reject_count": int(info.get("reject_count", 0)),
        "returned_best_iterate": bool(info.get("returned_best_iterate", 0.0)),
        "elastic_weight": float(used_weight),
        # The pairs. Reported and recomputed always travel together; neither is
        # ever shown standing in for the other.
        "certificate_reported": certificate_reported,
        "certificate_recomputed": koz["certificate_recomputed"],
        "certificate_delta": abs(certificate_reported - koz["certificate_recomputed"]),
        "clearance_reported": clearance_reported,
        "clearance_independent": clearance,
        "clearance_delta": abs(clearance_reported - clearance),
        "clearance_ladder": float(clearance_3000),
        "total_slack": float(info.get("total_koz_slack_returned", float("nan"))),
        "occlusion_reported": (
            float(info.get("occlusion_violation_reference", float("nan")))
            if stations
            else None
        ),
        "occlusion_recomputed": (
            occlusion["certificate_recomputed"] if occlusion is not None else None
        ),
        "occlusion_planes_dropped": float(info.get("occlusion_planes_dropped", 0.0)),
        # WHAT THE CERTIFICATE COVERS -- a gate condition since 2026-08-31, and
        # shown beside the certificate because the two answer different
        # questions. `sound_clip` says whether the zero was forced or happened.
        "koz_unsound_clips": float(info.get("koz_unsound_clips", float("nan"))),
        "sound_clip": bool(info.get("sound_clip", 0.0)),
        "los_min_margin": (los["min"] if los is not None else None),
        "los_lost": (los["lost"] if los is not None else None),
        "speed_cap_violation": float(info.get("speed_cap_violation", 0.0)),
        "arrival_time": float(info.get("arrival_time", float("nan"))),
        "arrival_on_min_dt_floor": float(info.get("arrival_on_min_dt_floor", 0.0)),
    }

    response = {
        "request": {
            "scenario": name,
            "N": N,
            "n_seg": n_seg,
            "elastic_weight": params["elastic_weight"],
            "sound_clip": params["sound_clip"],
            "v_max": params["v_max"],
            "time_weight": params["time_weight"],
            "free_arrival_time": params["free_arrival_time"],
            "trust_radius": params["trust_radius"],
            "max_iter": params["max_iter"],
            "tol": params["tol"],
            "min_dt": params["min_dt"],
            "scp_prox_weight": params["scp_prox_weight"],
        },
        "resolved": {
            "elastic_weight": float(used_weight),
            "initial_elastic_weight": float(rungs[0]["initial_elastic_weight"]),
            "weight_raises": int(rungs[0]["weight_raises"]),
            "rungs": rungs,
        },
        "scenario": {
            "name": name,
            "title": scenario["title"],
            "dim": dim,
            "start": list(scenario["start"]),
            "end": list(scenario["end"]),
            "T": float(scenario["T"]),
            "obstacles": scenario["obstacles"],
            "stations": stations,
            "views": views,
            "default_view": DEFAULT_VIEW.get(dim),
        },
        "solution": {
            "control_points": P_opt.tolist(),
            "init_control_points": np.asarray(P_init, dtype=float).tolist(),
            "curve": curve.tolist(),
            "speed": speed,
            "t_lo": float(P_opt[0, -1]),
            "t_hi": float(P_opt[-1, -1]),
        },
        "segments": {
            "n_seg": n_seg,
            "hulls": [(a @ P_opt).tolist() for a in a_list],
        },
        "planes": {
            "koz": koz["planes"],
            "occlusion": (occlusion["planes"] if occlusion is not None else []),
        },
        "ledger": {
            "rows": koz["ledger"],
            "total": koz["ledger_total"],
            "violated": koz["ledger_violated"],
            "zero_time": koz["ledger_zero_time"],
            "truncated": koz["ledger_truncated"],
            "occlusion_rows": (occlusion["rows"] if occlusion is not None else []),
            "occlusion_total": (occlusion["rows_total"] if occlusion is not None else 0),
        },
        "shadow_balls": (occlusion["shadow_balls"] if occlusion is not None else []),
        "sight": sight,
        "los": los,
        "verdict": verdict,
        "provenance": {
            **provenance_payload(),
            "solved_at": datetime.now().isoformat(timespec="seconds"),
            "solve_ms": round(solve_ms, 1),
            "cached": False,
        },
    }
    return _json_safe(response)


# ---------------------------------------------------------------------------
# Cancellable solve: the Rust loop holds the GIL and cannot be interrupted
# in-process, so the HTTP endpoint runs the SAME `solve_from_payload` in a
# child process, which /api/cancel can kill. The cache wraps the subprocess:
# a hit never spawns one.
# ---------------------------------------------------------------------------

_ACTIVE_SOLVE: dict = {"proc": None}

SOLVE_CHILD = (
    "import json, sys\n"
    "from spacetime_bezier.frontend import solve_from_payload, RequestError\n"
    "try:\n"
    "    sys.stdout.write(json.dumps(solve_from_payload(json.loads(sys.argv[1]))))\n"
    "except (RequestError, ValueError) as exc:\n"
    "    sys.stdout.write(json.dumps({'__request_error__': str(exc)}))\n"
)


def solve_via_subprocess(payload: dict) -> dict:
    """Cache lookup, then ``solve_from_payload`` in a killable child process.

    Validation runs here first so a bad request is a fast 400 with no child.
    The child re-runs it, which is redundancy, not a second opinion. A child
    killed by /api/cancel surfaces as a 400 saying so; a child that raised a
    refusal (e.g. the uncapped-time-penalty trap) hands its message back to be
    answered as the 400 it would have been in-process.
    """
    name, N, n_seg, params = _solve_params(payload)
    cache_key = json.dumps(
        {"scenario": name, "N": N, "n_seg": n_seg, **params}, sort_keys=True
    )
    cached = _cache_get(_SOLVE_CACHE, cache_key)
    if cached is not None:
        hit = dict(cached)
        hit["provenance"] = {**cached["provenance"], "cached": True}
        return hit

    with _SOLVE_LOCK:
        proc = subprocess.Popen(
            [sys.executable, "-c", SOLVE_CHILD, json.dumps(payload)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=REPO_ROOT,
        )
        _ACTIVE_SOLVE["proc"] = proc
        try:
            out, err = proc.communicate(timeout=1800)
        finally:
            _ACTIVE_SOLVE["proc"] = None
    if proc.returncode < 0:
        raise RequestError("solve cancelled")
    if proc.returncode != 0:
        raise RequestError(_explain_child_failure(err))
    data = json.loads(out)
    if isinstance(data, dict) and "__request_error__" in data:
        raise RequestError(data["__request_error__"])
    _cache_put(_SOLVE_CACHE, cache_key, data, _SOLVE_CACHE_MAX)
    return data


def _explain_child_failure(stderr: str) -> str:
    """Turn a child traceback into a diagnosis when the cause is recognisable.

    The recognisable case: the compiled extension and this worktree's Python
    disagree on the binding signature. That happens when ANOTHER checkout runs
    `maturin develop` into the shared venv -- the extension then answers for
    different solver code than the code on disk here. A raw traceback in the UI
    reads as "the solver broke"; this is a different and more actionable claim:
    two sessions are building into one venv, and whoever builds last wins.
    """
    tail = stderr[-2000:]
    if "unexpected keyword argument" in tail or "missing required argument" in tail:
        built = extension_build_time()
        when = (
            datetime.fromtimestamp(built).isoformat(timespec="seconds")
            if built
            else "unknown"
        )
        return (
            "the compiled Rust extension no longer matches this worktree's Python "
            f"solver code (extension built {when}). Another session has likely run "
            "`maturin develop` from a different checkout into the shared venv with "
            "a changed binding signature. Solving from this worktree needs either "
            "that solver change ported here, or the extension rebuilt from THIS "
            "worktree's sources -- coordinate first: a rebuild clobbers the other "
            f"session's build the same way.\n\nOriginal error:\n{tail[-600:]}"
        )
    return f"solve failed:\n{tail}"


def cancel_active_solve() -> dict:
    """Kill the child of the solve in flight, if there is one."""
    proc = _ACTIVE_SOLVE.get("proc")
    if proc is None or proc.poll() is not None:
        return {"cancelled": False, "note": "no active solve"}
    proc.kill()
    return {"cancelled": True}


# ---------------------------------------------------------------------------
# Replay endpoint
#
# The child script is IMPORTED from tools/trace_viewer.py rather than copied.
# That file's `_capture_replay` builds a `SpacetimeScpContext` with a parameter
# list that must match `optimize_spacetime`'s defaults exactly, or the replay is
# a different run -- and it carries the drift check that catches a mismatch. A
# second copy of that parameter list here is a second thing to keep in sync, and
# the failure mode of getting it wrong is a replay that looks like the solve and
# is not.
# ---------------------------------------------------------------------------

_TRACE_VIEWER_PATH = REPO_ROOT / "tools" / "trace_viewer.py"
_trace_viewer_cache: dict = {}


def trace_viewer_module():
    """``tools/trace_viewer.py`` loaded as a module, for its child script.

    Loaded by path because ``tools/`` is not a package. Cached, because the load
    executes the module body.
    """
    module = _trace_viewer_cache.get("module")
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        "_spacetime_frontend_trace_viewer", _TRACE_VIEWER_PATH
    )
    if spec is None or spec.loader is None:
        raise RequestError(f"cannot load {_TRACE_VIEWER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _trace_viewer_cache["module"] = module
    return module


def replay_from_payload(payload: dict) -> dict:
    """Drive the canonical SCP iteration once per frame, and return the trace.

    A subprocess, not an in-process call, for the reason `tools/trace_viewer.py`
    gives: the Rust trace is ``eprintln`` straight to fd 2, which a Python-level
    stderr redirect does not capture. The frames come back from the same child,
    so the STTRACE row for iteration *k* and the frame for iteration *k* are the
    same iteration of the same run.

    The elastic weight must be the one the SOLVE resolved to, not ``None``: the
    child pins a single weight and the solve may have walked a ladder, so
    replaying with the request's blank weight would replay the ladder's first
    rung and call it the run.
    """
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    name = payload.get("scenario")
    if name not in SCENARIO_MAP:
        raise RequestError(
            f"unknown scenario {name!r}; known: {', '.join(sorted(SCENARIO_MAP))}"
        )
    scenario_fn, configs = SCENARIO_MAP[name]
    N = _positive_int(payload, "N", configs[0][0])
    n_seg = _positive_int(payload, "n_seg", configs[0][1])

    # The child script takes keep-out parameters only. A run with a speed cap, a
    # time penalty or a freed arrival is a DIFFERENT problem, and replaying it
    # without them would produce frames that diverge from the solve for a reason
    # the drift number alone would not explain. Refuse instead of drawing frames
    # of a run nobody asked for.
    if _float_or_none(payload, "v_max") is not None:
        raise RequestError(
            "replay does not carry the speed cap: tools/trace_viewer.py's child "
            "script builds the stepping context without v_max, so a replay of a "
            "capped run would be a different problem. Solve without v_max to replay."
        )
    if _float(payload, "time_weight", 0.0) != 0.0:
        raise RequestError(
            "replay does not carry the time penalty (time_weight > 0): the child "
            "script builds the stepping context without it, so the replay would be "
            "a different problem."
        )
    if bool(payload.get("free_arrival_time", False)):
        raise RequestError(
            "replay does not carry free_arrival_time: the child script builds the "
            "stepping context with the arrival pinned, so the replay would be a "
            "different problem."
        )

    module = trace_viewer_module()
    request = {
        "scenario": name,
        "N": N,
        "seg": n_seg,
        "elastic_weight": _float_or_none(payload, "elastic_weight"),
        "max_iter": _positive_int(payload, "max_iter", SOLVE_DEFAULTS["max_iter"]),
        "tol": _float(payload, "tol", SOLVE_DEFAULTS["tol"]),
        "trust_radius": _float(payload, "trust_radius", SOLVE_DEFAULTS["trust_radius"]),
        "min_dt": _float(payload, "min_dt", SOLVE_DEFAULTS["min_dt"]),
    }
    cache_key = json.dumps(request, sort_keys=True)
    cached = _cache_get(_REPLAY_CACHE, cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    env = dict(os.environ, SPACETIME_SCVX_TRACE="1")
    proc = subprocess.run(
        [sys.executable, "-c", module.CHILD_SCRIPT, json.dumps(request)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RequestError(_explain_child_failure(proc.stderr))
    child = json.loads(proc.stdout)

    columns = module.TRACE_HEADER.split(",")
    rows, warnings, header_seen = [], [], None
    for line in proc.stderr.splitlines():
        if not line.startswith("STTRACE"):
            continue
        if line.startswith("STTRACE,it,"):
            header_seen = line[len("STTRACE,") :].replace("\\", "")
            continue
        parts = line.split(",")[1:]
        if len(parts) != len(columns):
            warnings.append(f"malformed trace line skipped: {line[:80]}")
            continue
        row = dict(zip(columns, parts))
        for key in columns:
            if key != "outcome":
                row[key] = float(row[key])
        rows.append(row)
    if header_seen is not None and header_seen != module.TRACE_HEADER:
        # Same refusal `tools/trace_viewer.py` makes: a silently reordered column
        # is exactly the lie the header check exists to prevent.
        raise RequestError(
            "trace header mismatch -- the Rust emits different columns than "
            f"tools/trace_viewer.py expects.\n  emitted:  {header_seen}\n"
            f"  expected: {module.TRACE_HEADER}\nRefusing to render rather than "
            "mislabel columns."
        )
    if not rows:
        warnings.append(
            "NO TRACE LINES CAPTURED -- the loaded extension predates the STTRACE "
            "emission or the env var did not reach it. The timeline is empty for "
            "that reason, not because the loop did nothing."
        )
    if request["elastic_weight"] is None:
        warnings.append(
            "no elastic_weight was sent, so the replay pinned the scenario's "
            f"registered weight ({scenario_elastic_weight(name):g}). If the solve "
            "walked the ladder and stopped on a different rung, this replay is a "
            "different run -- send the weight the solve resolved to."
        )

    # Trust radius per frame comes from the trace's own `trust_after` column, so
    # the boxes are the solver's trust region and not a viewer-side guess. Joined
    # by iteration index; a frame with no trace row gets None and draws nothing.
    trust_by_it = {int(row["it"]): float(row["trust_after"]) for row in rows}
    frames = child.get("replay") or []
    for frame in frames:
        frame["trust_after"] = trust_by_it.get(int(frame["it"]))

    # Half-space walls per frame, rebuilt at each frame's reference by the SAME
    # exact Rust builders the certificate uses -- the rows the next subproblem
    # linearizes at that reference. Real geometry at a real iterate, never the
    # returned iterate's planes redrawn onto an earlier one.
    scenario = scenario_fn()
    dim = len(scenario["start"])
    stations = scenario.get("stations")
    a_list = [np.asarray(a, dtype=float) for a in segment_matrices_equal_params(N, n_seg)]
    for frame in frames:
        P_f = np.asarray(frame["P"], dtype=float)
        built = _koz_planes(
            P_f, scenario["obstacles"], n_seg, a_list, dim, stations=stations or None
        )
        frame["planes"] = {
            "koz": built["planes"],
            "occlusion": built["occlusion"]["planes"] if stations else [],
        }

    drift = float(child.get("replay_drift", float("nan")))
    fell_back = bool(child.get("info", {}).get("returned_best_iterate", 0.0))
    response = _json_safe({
        "request": request,
        "cached": False,
        "frames": frames,
        "drift": drift,
        "returned_best_iterate": fell_back,
        # The sentence that says what the drift number means, written where the
        # number is produced rather than left to the reader.
        "drift_note": (
            f"replay endpoint vs returned control points: max |delta| = {drift:.2e}"
            + (
                " — the best-iterate fallback fired, so the returned point is an "
                "EARLIER iterate than the replay's last frame (expected mismatch)."
                if fell_back
                else ". A large value here without the fallback means the replay is "
                "not the run — do not trust either."
            )
        ),
        "trace_header": columns,
        "trace": rows,
        "warnings": warnings,
        "rejected": sum(1 for frame in frames if not frame.get("advanced")),
    })
    _cache_put(_REPLAY_CACHE, cache_key, response, _REPLAY_CACHE_MAX)
    return response


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class FrontendHandler(BaseHTTPRequestHandler):
    server_version = "SpacetimeFrontend/1.0"

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "frontend.html", "text/html; charset=utf-8")
            elif self.path == "/static/plotly.min.js":
                self._send_file(STATIC_DIR / "plotly.min.js", "application/javascript")
            elif self.path == HEALTH_PATH:
                self._send_json(200, _json_safe(health_payload()))
            elif self.path == "/api/scenarios":
                self._send_json(200, _json_safe(scenario_catalog()))
            else:
                self._send_json(404, {"error": f"Unknown path: {self.path}"})
        except BrokenPipeError:
            pass
        except Exception:
            self._send_json(500, {"error": "internal", "traceback": traceback.format_exc()})

    def do_POST(self):
        try:
            if self.path == "/api/solve":
                self._send_json(200, solve_via_subprocess(self._read_json()))
            elif self.path == "/api/replay":
                self._send_json(200, replay_from_payload(self._read_json()))
            elif self.path == "/api/cancel":
                self._send_json(200, cancel_active_solve())
            elif self.path == "/api/reload":
                self._send_json(200, _json_safe(reload_scenarios()))
            else:
                self._send_json(404, {"error": f"Unknown path: {self.path}"})
        except BrokenPipeError:
            pass
        except (RequestError, ValueError) as exc:
            # A bad request is answered as a bad request. A traceback in the UI
            # reads as a solver failure, which is a different and much more
            # alarming claim than "that configuration is not accepted".
            self._send_json(400, {"error": str(exc)})
        except Exception:
            self._send_json(500, {"error": "internal", "traceback": traceback.format_exc()})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise RequestError(f"request body is not valid JSON: {exc}") from None

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # quieter than the default one-line-per-asset
        if "/api/" in str(args[0] if args else ""):
            super().log_message(fmt, *args)


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bind the server.

    ``port`` exists so a test can bind an ephemeral one and exercise the real
    request/response path. It is NOT a way to run a second frontend: ``main`` does
    not expose it, there is no CLI flag, and the guard in ``main`` refuses to
    start when 8767 is held rather than moving.
    """
    return ThreadingHTTPServer((host, port), FrontendHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Space-time Bezier frontend — one page, port 8767"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser")
    args = parser.parse_args(argv)

    conflict = describe_port_conflict(args.host, DEFAULT_PORT)
    if conflict:
        print(conflict, file=sys.stderr)
        return 1
    if extension_build_time() is None:
        print(
            "WARNING: the Rust extension `bezier_opt` is not importable; every solve "
            "will fail.\nBuild it with: cd rust_optimizer/pybind && "
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release",
            file=sys.stderr,
        )

    server = make_server(args.host, DEFAULT_PORT)
    url = f"http://{args.host}:{DEFAULT_PORT}/"
    prov = provenance_payload()
    print(
        f"spacetime frontend: {url}  (pid {os.getpid()}, commit "
        f"{prov.get('git_commit')}{' +dirty' if prov.get('git_dirty') else ''})"
    )
    if prov.get("extension_stale"):
        print(
            "WARNING: the compiled extension is OLDER than the Rust sources. "
            "Rebuild before trusting anything this page draws.",
            file=sys.stderr,
        )
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
