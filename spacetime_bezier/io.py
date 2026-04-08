"""
CLI and JSON I/O helpers for the space-time Bezier package.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from .optimize import optimize_scenarios
from .scenarios import SCENARIO_MAP

DEFAULT_OUTPUT_PATH = Path("figures/spacetime_scenarios.json")
DEFAULT_VIEWER_HOST = "127.0.0.1"
DEFAULT_VIEWER_PORT = 8765
VIEWER_HTML = "spacetime_bezier_interactive.html"


def load_outputs(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def save_outputs(outputs: dict, path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(outputs, handle, indent=2)
    return path


def _port_is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def open_interactive_viewer(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    host: str = DEFAULT_VIEWER_HOST,
    port: int = DEFAULT_VIEWER_PORT,
) -> str:
    """
    Serve the figures directory and open the interactive viewer in a browser.

    Using a local HTTP server is required because the viewer fetches
    `spacetime_scenarios.json`, which is unreliable from a `file://` URL.
    """
    output_path = Path(output_path)
    viewer_dir = output_path.parent.resolve()
    html_path = viewer_dir / VIEWER_HTML
    if not html_path.exists():
        raise FileNotFoundError(f"Missing interactive viewer HTML: {html_path}")

    if not _port_is_listening(host, port):
        subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", host],
            cwd=str(viewer_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(20):
            if _port_is_listening(host, port):
                break
            time.sleep(0.1)

    url = f"http://{host}:{port}/{VIEWER_HTML}"
    webbrowser.open(url)
    return url


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Space-time Bezier optimizer")
    parser.add_argument(
        "scenarios",
        nargs="*",
        default=list(SCENARIO_MAP.keys()),
        choices=list(SCENARIO_MAP.keys()),
        help="Which scenarios to run (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "python", "rust"],
        default="rust",
        help="Optimization backend to use.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Maximum number of iterations for optimization.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the scenario JSON consumed by the interactive demo.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the interactive HTML viewer.",
    )
    parser.add_argument(
        "--viewer-host",
        default=DEFAULT_VIEWER_HOST,
        help="Host for the local viewer HTTP server.",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=DEFAULT_VIEWER_PORT,
        help="Port for the local viewer HTTP server.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    existing_outputs = load_outputs(output_path)
    all_outputs = optimize_scenarios(
        scenario_names=args.scenarios,
        scenario_map=SCENARIO_MAP,
        existing_outputs=existing_outputs,
        backend=args.backend,
        max_iter=args.max_iter,
    )
    saved_path = save_outputs(all_outputs, output_path)
    print(f"\nSaved: {saved_path}")

    for name in args.scenarios:
        data = all_outputs[name]
        best = data["results"][data["best"]]
        print(f"\n{data['title']}: best={data['best']}, clearance={best['min_clearance']:.4f}")
        for row in best["control_points"]:
            print(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],")

    if not args.no_open:
        viewer_url = open_interactive_viewer(
            output_path=saved_path,
            host=args.viewer_host,
            port=args.viewer_port,
        )
        print(f"\nOpened interactive viewer: {viewer_url}")
