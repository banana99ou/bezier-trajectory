"""
Minimal sanity-check viewer server.

Three rules govern this file:
nothing stored (every picture comes from a solve that ran in this process),
the solve path is just the solve (no debug stepper, no trace), and the client
only draws (verdict fields are computed here, from the solver's own numbers,
the same way ``optimize_scenario`` records them).

Run with:  python3 -m spacetime_bezier.viewer
"""

from __future__ import annotations

import argparse
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
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .geometry import compute_min_clearance
from .objective import build_initial_guess
from .optimize import _STOP_REASONS, DEFAULT_TRUST_RADIUS, optimize_spacetime_from_control_points
from .scenarios import SCENARIO_MAP, scenario_elastic_weight

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
HEALTH_PATH = "/api/health"
HEALTH_APP_ID = "spacetime-bezier-viewer"

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Defaults matching optimize_scenario -- the batch path that produced every
# number in the scenario table. Same request + same configuration = same result.
SOLVE_DEFAULTS = {
    "max_iter": 200,
    "tol": 1e-6,
    "scp_prox_weight": 0.3,
    "scp_trust_radius": DEFAULT_TRUST_RADIUS,
    "min_dt": 0.1,
}

_STARTED_AT = datetime.now().isoformat(timespec="seconds")
_SOLVE_LOCK = threading.Lock()
_CACHE: dict[tuple, dict] = {}
_CACHE_LOCK = threading.Lock()


def extension_build_time() -> float | None:
    """Modification time of the compiled Rust extension this process imports.

    The one number that distinguishes "a viewer is running" from "a viewer is
    running the solver you just built": the extension loads once per process, so
    a server started before a `maturin develop` answers with old geometry for
    its whole life unless this is surfaced.
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


def health_payload() -> dict:
    return {
        "app": HEALTH_APP_ID,
        "pid": os.getpid(),
        "started": _STARTED_AT,
        "executable": sys.executable,
        "extension_build_time": extension_build_time(),
        **git_provenance(),
    }


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

    Launching fails rather than hopping to a free port: two live servers on two
    default ports is exactly how a four-days-stale sandbox once kept answering
    with pre-fix geometry. The stale server has to be visible, and only the
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


def _json_safe(value):
    """Coerce numpy types and map non-finite floats to None (JSON null)."""
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


# The client plots columns 0, 1, 2 of every control point as (x, y, t) and
# recomputes clearance the same way. A scenario with three spatial coordinates
# has four columns, so the viewer would draw its z as time and cross-check
# against the wrong geometry -- a picture that looks fine and is false. Until the
# client learns to project, such scenarios are withheld rather than misdrawn.
VIEWER_COLUMNS = 3


def scenario_catalog() -> dict:
    """Scenario definitions in the one wire format: {pos0, vel, r, t_start?, t_end?}."""
    catalog = {}
    for name, (scenario_fn, configs) in SCENARIO_MAP.items():
        scenario = scenario_fn()
        if len(scenario["start"]) != VIEWER_COLUMNS:
            continue
        catalog[name] = {
            "name": scenario["name"],
            "title": scenario["title"],
            "obstacles": scenario["obstacles"],
            "start": list(scenario["start"]),
            "end": list(scenario["end"]),
            "T": float(scenario["T"]),
            "init_curve": scenario.get("init_curve"),
            "configs": [[int(N), int(n_seg)] for N, n_seg in configs],
            "default_elastic_weight": scenario_elastic_weight(name),
            "solve_defaults": dict(SOLVE_DEFAULTS),
        }
    return catalog


def solve_from_payload(payload: dict) -> dict:
    name = payload.get("scenario")
    if name not in SCENARIO_MAP:
        raise ValueError(f"Unknown scenario: {name!r}")
    scenario_fn, configs = SCENARIO_MAP[name]
    scenario = scenario_fn()
    if len(scenario["start"]) != VIEWER_COLUMNS:
        raise ValueError(
            f"Scenario {name!r} has {len(scenario['start'])} coordinates; this viewer "
            f"draws {VIEWER_COLUMNS} and would plot a spatial axis as time"
        )

    N = int(payload.get("N", configs[0][0]))
    n_seg = int(payload.get("n_seg", configs[0][1]))
    if N < 1 or n_seg < 1:
        raise ValueError(f"N and n_seg must be positive, got N={N}, n_seg={n_seg}")
    params = {key: float(payload.get(key, default)) for key, default in SOLVE_DEFAULTS.items()}
    params["max_iter"] = int(params["max_iter"])
    elastic_weight = float(payload.get("elastic_weight", scenario_elastic_weight(name)))

    cache_key = (name, N, n_seg, elastic_weight, *sorted(params.items()))
    with _CACHE_LOCK:
        hit = _CACHE.get(cache_key)
    if hit is not None:
        stale = dict(hit)
        stale["provenance"] = dict(hit["provenance"], cached=True)
        return stale

    P_init = build_initial_guess(
        scenario["start"], scenario["end"], N + 1, init_curve=scenario.get("init_curve")
    )
    started = time.perf_counter()
    with _SOLVE_LOCK:
        P_opt, info = optimize_spacetime_from_control_points(
            P_init,
            scenario["obstacles"],
            n_seg=n_seg,
            max_iter=params["max_iter"],
            tol=params["tol"],
            scp_prox_weight=params["scp_prox_weight"],
            scp_trust_radius=params["scp_trust_radius"],
            elastic_weight=elastic_weight,
            min_dt=params["min_dt"],
            verbose=False,
        )
    solve_ms = (time.perf_counter() - started) * 1000.0

    # Same function and sample count optimize_scenario records, so the number
    # here is the number the scenario table would show for this configuration.
    clearance = compute_min_clearance(P_opt, scenario["obstacles"], dim=3, n_eval=3000)
    certificate = float(info.get("koz_violation_reference", float("nan")))
    stop_reason = int(info.get("stop_reason", -1))
    verdict = {
        "feasible": bool(clearance > 0.0),
        "min_clearance": float(clearance),
        "converged": bool(info.get("converged", 0.0)),
        "stop_reason": stop_reason,
        "stop_label": _STOP_REASONS.get(stop_reason, "unknown"),
        "certified": bool(certificate <= 1e-6),
        "certificate_violation": certificate,
        "iterations": int(info.get("iterations", -1)),
        "elastic_weight": elastic_weight,
        "returned_best_iterate": bool(info.get("returned_best_iterate", 0.0)),
    }
    response = _json_safe(
        {
            "request": {"scenario": name, "N": N, "n_seg": n_seg, "elastic_weight": elastic_weight, **params},
            "title": scenario["title"],
            "obstacles": scenario["obstacles"],
            "start": list(scenario["start"]),
            "end": list(scenario["end"]),
            "T": float(scenario["T"]),
            "control_points": np.asarray(P_opt, dtype=float).tolist(),
            "init_control_points": np.asarray(P_init, dtype=float).tolist(),
            "verdict": verdict,
            "info": info,
            "provenance": {
                **git_provenance(),
                "extension_build_time": extension_build_time(),
                "solved_at": datetime.now().isoformat(timespec="seconds"),
                "solve_ms": round(solve_ms, 1),
                "cached": False,
            },
        }
    )
    with _CACHE_LOCK:
        _CACHE[cache_key] = response
    return response


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "SpacetimeSanityViewer/1.0"

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "viewer.html", "text/html; charset=utf-8")
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
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._send_json(200, solve_from_payload(payload))
            else:
                self._send_json(404, {"error": f"Unknown path: {self.path}"})
        except BrokenPipeError:
            pass
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            self._send_json(500, {"error": "internal", "traceback": traceback.format_exc()})

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Space-time Bezier sanity-check viewer")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser")
    args = parser.parse_args(argv)

    conflict = describe_port_conflict(args.host, args.port)
    if conflict:
        print(conflict, file=sys.stderr)
        return 1
    if extension_build_time() is None:
        print(
            "WARNING: the Rust extension `bezier_opt` is not importable; every solve will fail.\n"
            "Build it with: cd rust_optimizer/pybind && "
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release",
            file=sys.stderr,
        )

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}/"
    prov = git_provenance()
    print(f"sanity viewer: {url}  (pid {os.getpid()}, commit {prov.get('git_commit')}"
          f"{' +dirty' if prov.get('git_dirty') else ''})")
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
