"""
HTTP entrypoint for the live optimizer step debugger.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spacetime_bezier.debug_session import OptimizerDebugSession, SessionConfig
from spacetime_bezier.scenarios import SCENARIO_MAP

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEBUG_HTML = "spacetime_bezier_opt_debug.html"
FIGURES_DIR = REPO_ROOT / "figures"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live space-time optimizer step debugger")
    parser.add_argument("scenario", nargs="?", choices=list(SCENARIO_MAP.keys()), help="Scenario to preload")
    parser.add_argument("-N", type=int, default=None, help="Bezier degree to preload")
    parser.add_argument("--n-seg", type=int, default=None, help="Segment count to preload")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum SCP iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--scp-prox-weight", type=float, default=0.3, help="SCP proximal weight")
    parser.add_argument("--scp-trust-radius", type=float, default=0.0, help="Trust-region radius")
    parser.add_argument("--min-dt", type=float, default=0.1, help="Minimum time increment")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open the browser")
    return parser


def _resolve_initial_config(args) -> SessionConfig:
    scenario_name = args.scenario or next(iter(SCENARIO_MAP))
    _scenario_fn, configs = SCENARIO_MAP[scenario_name]
    if args.N is None and args.n_seg is None:
        N, n_seg = configs[0]
    else:
        N = int(args.N) if args.N is not None else int(configs[0][0])
        if args.n_seg is not None:
            n_seg = int(args.n_seg)
        else:
            same_degree = [int(cfg_n_seg) for cfg_degree, cfg_n_seg in configs if int(cfg_degree) == N]
            n_seg = same_degree[0] if same_degree else int(configs[0][1])
    return SessionConfig(
        scenario_name=scenario_name,
        N=int(N),
        n_seg=int(n_seg),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        scp_prox_weight=float(args.scp_prox_weight),
        scp_trust_radius=float(args.scp_trust_radius),
        min_dt=float(args.min_dt),
        backend="python",
    )


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
    handler.end_headers()
    handler.wfile.write(encoded)


def make_handler(session: OptimizerDebugSession):
    class DebugRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(FIGURES_DIR), **kwargs)

        def log_message(self, format, *args):  # pragma: no cover
            return

        def do_GET(self):
            if self.path == "/":
                self.path = f"/{DEBUG_HTML}"
            if self.path == "/api/state":
                return _write_json(self, session.get_state())
            if self.path == "/api/configs":
                return _write_json(self, session.available_scenarios())
            return super().do_GET()

        def do_POST(self):
            try:
                payload = _read_json_body(self)
                if self.path == "/api/start":
                    config = SessionConfig(
                        scenario_name=str(payload["scenario_name"]),
                        N=int(payload["N"]),
                        n_seg=int(payload["n_seg"]),
                        max_iter=int(payload.get("max_iter", 200)),
                        tol=float(payload.get("tol", 1e-6)),
                        scp_prox_weight=float(payload.get("scp_prox_weight", 0.3)),
                        scp_trust_radius=float(payload.get("scp_trust_radius", 0.0)),
                        min_dt=float(payload.get("min_dt", 0.1)),
                        backend="python",
                    )
                    return _write_json(self, session.start(config))
                if self.path == "/api/action":
                    action = str(payload.get("action", "")).strip().lower()
                    if action == "next":
                        return _write_json(self, session.next())
                    if action == "prev":
                        return _write_json(self, session.prev())
                    if action == "next-iteration":
                        return _write_json(self, session.next_iteration())
                    if action == "reset":
                        return _write_json(self, session.reset())
                    return _write_json(self, {"error": f"Unknown action: {action}"}, status=400)
                return _write_json(self, {"error": f"Unknown endpoint: {self.path}"}, status=404)
            except Exception as exc:  # pragma: no cover
                return _write_json(self, {"error": str(exc)}, status=500)

    return DebugRequestHandler


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    session = OptimizerDebugSession()
    session.start(_resolve_initial_config(args))

    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    url = f"http://{args.host}:{args.port}/{DEBUG_HTML}"
    print(f"Serving live optimizer debugger at {url}")
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
