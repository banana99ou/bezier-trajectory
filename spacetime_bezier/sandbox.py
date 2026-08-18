"""
HTTP entrypoint for the interactive space-time Bezier sandbox.

Serves ``figures/spacetime_bezier_interactive.html`` and exposes:

- ``POST /api/solve`` — runs one SCP solve via ``RustOptimizerStepper`` and
  returns final control points plus a ``trace_id`` that resolves to the full
  per-stage trace and an optional one-line ``diagnosis`` for infeasible runs.
- ``GET /api/trace/<id>`` — returns the frames from the most recent solve
  keyed by ``trace_id``.
- ``GET /api/scenarios`` — scenario catalog for first paint.

Batch and diagnostic mode share this code path (VISION §"One canonical
execution model"): every solve produces a trace, whether or not the drawer
is open to consume it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .geometry import bezier_obstacle_from_moving, moving_obstacle_from_bezier
from .objective import build_initial_guess
from .rust_debug_stepper import create_spacetime_debug_stepper_from_control_points
from .scenarios import SCENARIO_MAP

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
SANDBOX_HTML = "spacetime_bezier_interactive.html"
FIGURES_DIR = REPO_ROOT / "figures"
TRACE_PATH_PREFIX = "/api/trace/"

# Identifies a running sandbox to a would-be second one. See `describe_port_conflict`.
HEALTH_APP_ID = "spacetime-bezier-sandbox"
HEALTH_PATH = "/api/health"
_STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")

# Single-slot trace cache. A new solve overwrites the previous entry; clients
# that still hold an old trace_id get a 404. ThreadingHTTPServer can race
# concurrent solves, so every access goes through _TRACE_LOCK.
_TRACE_CACHE: dict[str, list[dict]] = {}
_TRACE_LOCK = threading.Lock()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive space-time Bezier sandbox server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open the browser")
    return parser


def extension_build_time() -> float | None:
    """Modification time of the compiled Rust extension this process imports.

    This is the one number that distinguishes "a sandbox is running" from "a
    sandbox is running the solver you just built". The extension is loaded once
    per process, so a server started before a `maturin develop` keeps answering
    with the old geometry for its whole life and nothing in the UI says so.
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


def health_payload() -> dict:
    """What ``GET /api/health`` returns: enough to decide whether to trust this server."""
    return {
        "app": HEALTH_APP_ID,
        "pid": os.getpid(),
        "started": _STARTED_AT,
        "executable": sys.executable,
        "extension_build_time": extension_build_time(),
    }


def port_is_listening(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_sandbox(host: str, port: int, timeout: float = 0.5) -> dict | None:
    """Health payload of a sandbox already on ``(host, port)``, else None."""
    url = f"http://{host}:{port}{HEALTH_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("app") != HEALTH_APP_ID:
        return None
    return payload


def describe_port_conflict(host: str, port: int) -> str | None:
    """Explain who holds ``port``, or None when it is free.

    Launching fails rather than hopping to a free port or silently reusing the
    running server. Both alternatives were tried implicitly and both went wrong
    on this machine: two default ports (8765 and 8767) produced two live
    sandboxes, the older one four days stale on a different interpreter, and a
    browser pointed at it would have solved with pre-fix KOZ geometry with no
    signal that anything was different. The stale server has to be visible, and
    only the human can confirm the process is theirs to kill.
    """
    if not port_is_listening(host, port):
        return None

    lines = [f"Port {port} on {host} is already in use."]
    other = probe_sandbox(host, port)
    if other is None:
        lines.append("Whatever holds it is not a space-time Bezier sandbox.")
        stop_hint = f"lsof -nP -iTCP:{port} -sTCP:LISTEN     # then kill the pid"
    else:
        lines.append(
            f"It is a sandbox: pid {other.get('pid')}, up since {other.get('started')}."
        )
        lines.append(f"  interpreter: {other.get('executable')}")
        theirs = other.get("extension_build_time")
        mine = extension_build_time()
        if theirs is None:
            lines.append(
                "  WARNING: it has no Rust extension loaded, so every solve there fails."
            )
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


def _scenario_preset_bezier(name: str) -> dict:
    """Load a named preset and return it with obstacles in BezierObstacle shape."""
    if name not in SCENARIO_MAP:
        raise ValueError(f"Unknown scenario: {name}")
    scenario_fn, configs = SCENARIO_MAP[name]
    scenario = scenario_fn()
    T = float(scenario.get("T", 10.0))
    default_N, default_n_seg = configs[0]
    return {
        "name": scenario["name"],
        "title": scenario["title"],
        "obstacles": [bezier_obstacle_from_moving(o, T) for o in scenario["obstacles"]],
        "start": list(scenario["start"]),
        "end": list(scenario["end"]),
        "T": T,
        "init_curve": scenario.get("init_curve"),
        "default_N": int(default_N),
        "default_n_seg": int(default_n_seg),
    }


def scenario_catalog() -> dict:
    """Build the scenario metadata served to the UI on first paint."""
    return {name: _scenario_preset_bezier(name) for name in SCENARIO_MAP}


def _diagnose(frames: list[dict], info: dict) -> str | None:
    """Build a one-line infeasibility summary from collected frames.

    Returns None when the solve is feasible. Otherwise scans the last
    ``supporting-surface-generation`` frame (which carries per-row KOZ data)
    for the row with the smallest ``margin_current`` — i.e. the most-violated
    constraint on the final iterate — and formats it as a single sentence.
    """
    if info.get("feasible"):
        return None

    worst_row = None
    worst_iter = 0
    for frame in reversed(frames):
        if frame.get("stage") != "supporting-surface-generation":
            continue
        all_rows = frame.get("payload", {}).get("koz", {}).get("all_rows") or []
        if not all_rows:
            continue
        frame_iter = frame.get("iteration")
        if frame_iter is None:
            frame_iter = 0
        # Prefer rows with positive slack when elastic was engaged; otherwise
        # fall back to the most-negative margin.
        slacked = [r for r in all_rows if float(r.get("slack", 0.0)) > 1e-9]
        candidate = (
            max(slacked, key=lambda r: float(r.get("slack", 0.0)))
            if slacked
            else min(all_rows, key=lambda r: float(r.get("margin_current", 0.0)))
        )
        worst_row = candidate
        # `or` would misreport iteration 0 as the frame's; only None falls back.
        row_iter = candidate.get("iteration")
        worst_iter = int(row_iter) if row_iter is not None else int(frame_iter) + 1
        break

    if worst_row is None:
        return f"infeasible (min_clearance={float(info.get('min_clearance', 0.0)):.3f}); no KOZ row data"

    return (
        f"infeasible: obstacle '{worst_row.get('obstacle_name', '?')}', "
        f"segment {int(worst_row.get('segment_index', -1))}, "
        f"iteration {worst_iter}, "
        f"cp={int(worst_row.get('cp_idx', -1))}, "
        f"margin={float(worst_row.get('margin_current', 0.0)):.3f}, "
        f"slack={float(worst_row.get('slack', 0.0)):.3f}"
    )


def _store_trace(frames: list[dict]) -> str:
    """Insert frames into the single-slot trace cache and return a new trace_id."""
    trace_id = uuid.uuid4().hex
    with _TRACE_LOCK:
        _TRACE_CACHE.clear()
        _TRACE_CACHE[trace_id] = frames
    return trace_id


def get_trace(trace_id: str) -> list[dict] | None:
    """Look up a cached trace by ID. Returns None if the ID is stale or unknown."""
    with _TRACE_LOCK:
        frames = _TRACE_CACHE.get(trace_id)
    return frames


def solve_from_payload(payload: dict) -> dict:
    """Run one optimizer call for the sandbox UI.

    The payload is expected to carry the full problem state in BezierObstacle
    wire format: ``obstacles`` (list), ``start``, ``end``, ``T``, plus solver
    params. ``scenario_name`` is informational (used as a label and as the
    fallback preset when the UI requests a fresh scenario).

    Every solve goes through ``RustOptimizerStepper`` and emits a full per-stage
    trace; the response carries a ``trace_id`` that resolves via
    ``GET /api/trace/<id>`` to the collected frames.
    """
    scenario_name = str(payload.get("scenario_name", ""))
    # Backfill missing problem state from the named preset (first paint path).
    preset = _scenario_preset_bezier(scenario_name) if scenario_name in SCENARIO_MAP else None

    bezier_obstacles = payload.get("obstacles")
    if bezier_obstacles is None:
        if preset is None:
            raise ValueError("Payload missing 'obstacles' and no valid 'scenario_name' preset")
        bezier_obstacles = preset["obstacles"]
    p_start = list(payload.get("start") or (preset["start"] if preset else []))
    p_end = list(payload.get("end") or (preset["end"] if preset else []))
    T = float(payload.get("T", preset["T"] if preset else 10.0))
    init_curve = payload.get("init_curve", preset["init_curve"] if preset else None)

    N = int(payload["N"])
    n_seg = int(payload["n_seg"])
    scp_prox_weight = float(payload.get("scp_prox_weight", 0.5))
    scp_trust_radius = float(payload.get("scp_trust_radius", 0.0))
    time_ub_scale = float(payload.get("time_ub_scale", 1.5))
    max_iter = int(payload.get("max_iter", 30))
    tol = float(payload.get("tol", 1e-6))
    min_dt = float(payload.get("min_dt", 0.1))
    cap_bulge_ratio = float(payload.get("cap_bulge_ratio", 2.0))

    legacy_obstacles = [moving_obstacle_from_bezier(bo) for bo in bezier_obstacles]

    P_init = build_initial_guess(p_start, p_end, int(N) + 1, init_curve=init_curve)

    t0 = time.perf_counter()
    stepper = create_spacetime_debug_stepper_from_control_points(
        p_init=P_init,
        obstacles=legacy_obstacles,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        min_dt=min_dt,
        time_ub_scale=time_ub_scale,
        cap_bulge_ratio=cap_bulge_ratio,
    )
    P_opt, info = stepper.run_to_completion()
    frames = stepper.frames_as_dicts()
    solve_ms = (time.perf_counter() - t0) * 1000.0

    info_out = {}
    for k, value in dict(info).items():
        if isinstance(value, (np.floating,)):
            info_out[k] = float(value)
        elif isinstance(value, (np.integer,)):
            info_out[k] = int(value)
        else:
            info_out[k] = value
    info_out["feasible"] = bool(info_out.get("feasible", 0.0))
    info_out["iterations"] = int(info_out.get("iterations", -1))

    trace_id = _store_trace(frames)
    diagnosis = _diagnose(frames, info_out)

    return {
        "scenario_name": scenario_name,
        "N": N,
        "n_seg": n_seg,
        "scp_prox_weight": scp_prox_weight,
        "scp_trust_radius": scp_trust_radius,
        "time_ub_scale": time_ub_scale,
        "max_iter": max_iter,
        "tol": tol,
        "min_dt": min_dt,
        "cap_bulge_ratio": cap_bulge_ratio,
        "control_points": np.asarray(P_opt, dtype=float).tolist(),
        "init_control_points": np.asarray(P_init, dtype=float).tolist(),
        "obstacles": bezier_obstacles,
        "start": list(p_start),
        "end": list(p_end),
        "T": T,
        "info": info_out,
        "solve_ms": solve_ms,
        "trace_id": trace_id,
        "trace_frame_count": len(frames),
        "diagnosis": diagnosis,
    }


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    body = handler.rfile.read(content_length)
    return json.loads(body.decode("utf-8"))


def _json_safe(obj):
    """Replace non-finite floats with null so the payload is valid JSON.

    `json.dumps` emits the bare tokens NaN / Infinity / -Infinity, which are not
    JSON; every browser's JSON.parse rejects the whole document. `wall` returns
    ``best_clearance = -inf`` whenever no iterate was ever feasible, so the
    entire scenario failed in the UI with "solve failed: bad JSON" -- while every
    curl check passed, because Python's json.loads accepts those tokens by
    default. Serializing through here and with allow_nan=False means a
    non-finite value can no longer leave this process disguised as JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    return obj


def _write_json(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    encoded = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(encoded)


def make_handler():
    class SandboxRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(FIGURES_DIR), **kwargs)

        def log_message(self, format, *args):  # pragma: no cover
            return

        def do_GET(self):
            if self.path == "/":
                self.path = f"/{SANDBOX_HTML}"
            if self.path == HEALTH_PATH:
                return _write_json(self, health_payload())
            if self.path == "/api/scenarios":
                return _write_json(self, scenario_catalog())
            if self.path.startswith(TRACE_PATH_PREFIX):
                trace_id = self.path[len(TRACE_PATH_PREFIX):]
                frames = get_trace(trace_id)
                if frames is None:
                    return _write_json(self, {"error": "Unknown or stale trace_id"}, status=404)
                return _write_json(self, {"trace_id": trace_id, "frames": frames})
            return super().do_GET()

        def do_POST(self):
            try:
                payload = _read_json_body(self)
            except json.JSONDecodeError as exc:
                return _write_json(self, {"error": f"Invalid JSON: {exc}"}, status=400)

            if self.path != "/api/solve":
                return _write_json(self, {"error": f"Unknown endpoint: {self.path}"}, status=404)

            try:
                response = solve_from_payload(payload)
                return _write_json(self, response)
            except Exception as exc:
                return _write_json(
                    self,
                    {"error": str(exc), "traceback": traceback.format_exc()},
                    status=500,
                )

    return SandboxRequestHandler


def main(argv: list[str] | None = None) -> int:
    """Serve the sandbox until interrupted. Returns a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    conflict = describe_port_conflict(args.host, args.port)
    if conflict is not None:
        print(conflict, file=sys.stderr)
        return 1

    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler())
    except OSError as exc:
        # Lost a race with something that grabbed the port between the probe and
        # the bind, or the address is not ours to bind at all.
        print(f"Could not bind {args.host}:{args.port} -- {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    build = extension_build_time()
    built = (
        datetime.fromtimestamp(build).isoformat(timespec="seconds")
        if build is not None
        else "MISSING -- solves will fail; run maturin develop --release"
    )
    print(f"Serving space-time Bezier sandbox at {url}")
    print(f"  pid {os.getpid()}, Rust extension built {built}")
    print("  nothing is pre-solved; each selection is one solve, ~0.6s")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0
