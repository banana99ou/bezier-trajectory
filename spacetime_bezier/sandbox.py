"""
HTTP entrypoint for the interactive space-time Bezier sandbox.

Serves ``figures/spacetime_bezier_interactive.html`` and exposes a single
``POST /api/solve`` endpoint that re-runs ``optimize_spacetime`` on every
parameter change. This is the same code path used by batch mode, so results
from the sandbox are byte-identical to ``python3 -m spacetime_bezier.io``
for the same inputs (VISION §"One canonical execution model").
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .objective import build_initial_guess
from .optimize import optimize_spacetime
from .scenarios import SCENARIO_MAP

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
SANDBOX_HTML = "spacetime_bezier_interactive.html"
FIGURES_DIR = REPO_ROOT / "figures"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive space-time Bezier sandbox server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open the browser")
    return parser


def scenario_catalog() -> dict:
    """Build the scenario metadata served to the UI on first paint."""
    catalog = {}
    for name, (scenario_fn, configs) in SCENARIO_MAP.items():
        scenario = scenario_fn()
        default_N, default_n_seg = configs[0]
        catalog[name] = {
            "name": scenario["name"],
            "title": scenario["title"],
            "obstacles": scenario["obstacles"],
            "start": list(scenario["start"]),
            "end": list(scenario["end"]),
            "T": float(scenario.get("T", 10.0)),
            "init_curve": scenario.get("init_curve"),
            "default_N": int(default_N),
            "default_n_seg": int(default_n_seg),
        }
    return catalog


def solve_from_payload(payload: dict) -> dict:
    """Run one optimizer call for the sandbox UI. Returns the response dict."""
    scenario_name = str(payload["scenario_name"])
    if scenario_name not in SCENARIO_MAP:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    scenario_fn, _configs = SCENARIO_MAP[scenario_name]
    scenario = scenario_fn()

    N = int(payload["N"])
    n_seg = int(payload["n_seg"])
    scp_prox_weight = float(payload.get("scp_prox_weight", 0.5))
    scp_trust_radius = float(payload.get("scp_trust_radius", 0.0))
    time_ub_scale = float(payload.get("time_ub_scale", 1.5))
    max_iter = int(payload.get("max_iter", 30))
    tol = float(payload.get("tol", 1e-6))
    min_dt = float(payload.get("min_dt", 0.1))
    capsule_time_scale = float(payload.get("capsule_time_scale", 0.5))

    p_start = scenario["start"]
    p_end = scenario["end"]
    P_init = build_initial_guess(p_start, p_end, int(N) + 1, init_curve=scenario.get("init_curve"))

    t0 = time.perf_counter()
    P_opt, info = optimize_spacetime(
        N=N,
        dim=len(p_start),
        p_start=p_start,
        p_end=p_end,
        obstacles=scenario["obstacles"],
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        min_dt=min_dt,
        time_ub_scale=time_ub_scale,
        capsule_time_scale=capsule_time_scale,
        verbose=False,
        init_curve=scenario.get("init_curve"),
    )
    solve_ms = (time.perf_counter() - t0) * 1000.0

    info_out = {}
    for key, value in dict(info).items():
        if isinstance(value, (np.floating,)):
            info_out[key] = float(value)
        elif isinstance(value, (np.integer,)):
            info_out[key] = int(value)
        else:
            info_out[key] = value
    info_out["feasible"] = bool(info_out.get("feasible", 0.0))
    info_out["iterations"] = int(info_out.get("iterations", -1))

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
        "capsule_time_scale": capsule_time_scale,
        "control_points": np.asarray(P_opt, dtype=float).tolist(),
        "init_control_points": np.asarray(P_init, dtype=float).tolist(),
        "obstacles": scenario["obstacles"],
        "start": list(p_start),
        "end": list(p_end),
        "T": float(scenario.get("T", 10.0)),
        "info": info_out,
        "solve_ms": solve_ms,
    }


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    body = handler.rfile.read(content_length)
    return json.loads(body.decode("utf-8"))


def _write_json(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    encoded = json.dumps(payload).encode("utf-8")
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
            if self.path == "/api/scenarios":
                return _write_json(self, scenario_catalog())
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


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), make_handler())
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving space-time Bezier sandbox at {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
