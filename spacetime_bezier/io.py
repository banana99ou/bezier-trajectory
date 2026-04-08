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
DEFAULT_N_SEG_SWEEP = (2, 8, 32, 64)


def _positive_int(value: str) -> int:
    degree = int(value)
    if degree <= 0:
        raise argparse.ArgumentTypeError("N must be a positive integer")
    return degree


def _normalize_degree_args(argv: list[str]) -> list[str]:
    normalized = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token != "-N":
            normalized.append(token)
            idx += 1
            continue

        idx += 1
        degree_tokens = []
        while idx < len(argv):
            candidate = argv[idx]
            try:
                _positive_int(candidate)
            except (ValueError, argparse.ArgumentTypeError):
                break
            degree_tokens.append(candidate)
            idx += 1

        if not degree_tokens:
            raise argparse.ArgumentTypeError("-N requires at least one positive integer")

        for degree in degree_tokens:
            normalized.extend(["-N", degree])

    return normalized


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
        "-N",
        action="append",
        type=_positive_int,
        default=None,
        help="Override with one or more positive Bezier degrees and run n_seg in {2, 8, 32, 64}.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for optimization.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Maximum number of iterations for optimization.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "python", "rust"],
        default="rust",
        help="Optimization backend to use.",
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


def _resolve_scenario_map(base_map: dict, degree_overrides: list[int] | None) -> dict:
    if degree_overrides is None:
        return base_map
    degrees = list(dict.fromkeys(int(degree) for degree in degree_overrides))
    return {
        name: (scenario_fn, [(degree, n_seg) for degree in degrees for n_seg in DEFAULT_N_SEG_SWEEP])
        for name, (scenario_fn, _configs) in base_map.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        normalized_argv = _normalize_degree_args(raw_argv)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    args = parser.parse_args(normalized_argv)

    output_path = Path(args.output)
    existing_outputs = load_outputs(output_path)
    scenario_map = _resolve_scenario_map(SCENARIO_MAP, args.N)
    all_outputs = optimize_scenarios(
        scenario_names=args.scenarios,
        scenario_map=scenario_map,
        existing_outputs=existing_outputs,
        backend=args.backend,
        max_iter=args.max_iter,
        tol=args.tol
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
