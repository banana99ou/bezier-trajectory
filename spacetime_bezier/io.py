"""
Command line surface and JSON I/O for the space-time Bezier package.

There is exactly one entrypoint, ``python3 -m spacetime_bezier``. It launches
the sandbox; ``--bake`` re-solves everything instead. Nothing here is runnable
as ``__main__``: this module is imported by ``__init__``, so running it that way
would execute the file twice under two names, and a test monkeypatching one copy
would leave the shipped command running the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .optimize import optimize_scenarios
from .sandbox import DEFAULT_HOST, DEFAULT_PORT, SANDBOX_HTML
from .sandbox import main as sandbox_main
from .scenarios import SCENARIO_MAP

DEFAULT_OUTPUT_PATH = Path("figures/spacetime_scenarios.json")
DEFAULT_N_SEG_SWEEP = (2, 8, 32, 64)
DEFAULT_MAX_ITER = 10000
DEFAULT_TOL = 1e-12


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



DEFAULT_VIEWER_PATH = Path(__file__).resolve().parents[1] / "figures" / SANDBOX_HTML

VIEWER_BLOB_PREFIX = "const SCENARIOS = "


def viewer_path_for(output_path: str | Path) -> Path:
    """The viewer that belongs to a given scenario JSON: its sibling.

    Deriving this rather than hardcoding the repo's copy is a safety property,
    not tidiness. `sync_interactive_viewer` used to write the tracked
    `figures/` page no matter where the JSON went, so the integration test --
    which redirects output to a tmp directory and feeds in one fake control
    point at the origin -- overwrote the real page's offline data anyway, and
    that mock was committed. A test must not be able to reach a tracked file.
    """
    return Path(output_path).parent / SANDBOX_HTML


def sync_interactive_viewer(
    outputs: dict,
    viewer_path: str | Path = DEFAULT_VIEWER_PATH,
) -> Path:
    """Rewrite the viewer's inline scenario blob from `outputs`.

    The interactive page carries its own baked copy of the scenario data so it
    works over `file://` with no server. Nothing used to keep that copy in step
    with `figures/spacetime_scenarios.json`, so the two silently diverged -- the
    page shipped trajectories from a solver three months and two defect fixes out
    of date. Regenerating one without the other is what made that possible, so
    the pipeline now writes both or neither.
    """
    viewer_path = Path(viewer_path)
    text = viewer_path.read_text()

    start = text.index(VIEWER_BLOB_PREFIX)
    brace = text.index("{", start)
    depth = 0
    end = None
    for idx in range(brace, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise ValueError(f"Unbalanced SCENARIOS blob in {viewer_path}")

    blob = json.dumps(outputs, indent=2)
    viewer_path.write_text(text[:start] + VIEWER_BLOB_PREFIX + blob + text[end:])
    return viewer_path


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


# `open_interactive_viewer` lived here until 2026-08-18. It spawned a detached
# `python -m http.server` with start_new_session=True: no terminate, no atexit,
# no recorded pid, so Ctrl-C never reached it and it outlived every parent. It
# used port 8765, which was also this module's sandbox default, so one --bake
# run permanently squatted the port the plain command needed -- the Errno 48
# that started this rewrite. It also served the page WITHOUT /api/solve, so the
# browser it opened answered every slider move with 501. Its stated reason (the
# viewer "needs" a server because file:// cannot fetch the JSON) was obsolete:
# the page falls back to the inline SCENARIOS blob, which --bake rewrites in the
# same run.


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Space-time Bezier viewer. By default this LAUNCHES THE SANDBOX and "
            "solves nothing: the page opens immediately and solves only the "
            "scenario / degree / segment count you select. Use --bake to "
            "pre-solve every configuration and refresh the static data."
        )
    )
    parser.add_argument(
        "--bake",
        action="store_true",
        help=(
            "Solve every scenario at every configuration and rewrite "
            "figures/spacetime_scenarios.json and the viewer's inline copy. "
            "Slow -- it is 16 configurations, several of which run to the "
            "iteration cap. Only needed to refresh the file:// fallback data."
        ),
    )
    # `default` is deliberately not set alongside `choices` here. argparse before
    # 3.13 validates the default list against `choices` as a single value, so a
    # bare `python3 -m spacetime_bezier` died on 3.11 and 3.12 with
    # "invalid choice: ['original', 'diverse', 'wall']". Resolved in `main`.
    parser.add_argument(
        "scenarios",
        nargs="*",
        choices=list(SCENARIO_MAP.keys()),
        help="[--bake only] Which scenarios to re-solve (default: all).",
    )
    parser.add_argument(
        "-N",
        action="append",
        type=_positive_int,
        default=None,
        help="[--bake only] One or more Bezier degrees; runs n_seg in {2, 8, 32, 64}.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help=f"[--bake only] Optimizer tolerance (default: {DEFAULT_TOL:g}).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help=f"[--bake only] Iteration cap (default: {DEFAULT_MAX_ITER}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"[--bake only] Scenario JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host for the sandbox server (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for the sandbox server (default: {DEFAULT_PORT}).",
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


#: Flags that only mean something under --bake. Passing one without it used to
#: be silently ignored: `python3 -m spacetime_bezier wall -N 12 --max-iter 50`
#: launched an unrelated sandbox and dropped every one of those words.
_BAKE_ONLY = {
    "scenarios": "scenario names",
    "N": "-N",
    "tol": "--tol",
    "max_iter": "--max-iter",
    "output": "--output",
}


def main(argv: list[str] | None = None) -> int:
    """Launch the sandbox, or with --bake re-solve everything instead.

    Returns a process exit code. The default path serves until interrupted, so
    this function blocks -- which is why nothing imports it as a helper and why
    `tests/conftest.py` makes `serve_forever` raise under pytest. A test that
    called this without --bake once left a listening socket behind for ten
    hours and broke the next launch with Errno 48.
    """
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        normalized_argv = _normalize_degree_args(raw_argv)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    args = parser.parse_args(normalized_argv)

    if not args.bake:
        ignored = [label for attr, label in _BAKE_ONLY.items() if getattr(args, attr)]
        if ignored:
            parser.error(
                f"{', '.join(ignored)}: only meaningful with --bake. "
                "Without --bake this launches the sandbox and solves nothing up front."
            )
        # Open first, solve on demand: one selection is one POST /api/solve.
        sandbox_argv = ["--host", args.host, "--port", str(args.port)]
        if args.no_open:
            sandbox_argv.append("--no-open")
        return sandbox_main(sandbox_argv)

    scenarios = args.scenarios or list(SCENARIO_MAP.keys())
    output_path = Path(args.output or DEFAULT_OUTPUT_PATH)
    max_iter = DEFAULT_MAX_ITER if args.max_iter is None else args.max_iter
    tol = DEFAULT_TOL if args.tol is None else args.tol
    existing_outputs = load_outputs(output_path)
    scenario_map = _resolve_scenario_map(SCENARIO_MAP, args.N)
    all_outputs = optimize_scenarios(
        scenario_names=scenarios,
        scenario_map=scenario_map,
        existing_outputs=existing_outputs,
        max_iter=max_iter,
        tol=tol,
    )
    saved_path = save_outputs(all_outputs, output_path)
    print(f"\nSaved: {saved_path}")
    viewer_target = viewer_path_for(saved_path)
    if viewer_target.exists():
        try:
            viewer = sync_interactive_viewer(all_outputs, viewer_target)
            print(f"Synced viewer: {viewer}")
        except (OSError, ValueError) as exc:  # pragma: no cover - viewer is optional
            print(f"WARNING: could not sync the interactive viewer: {exc}")
    else:
        print(f"No viewer beside {saved_path}; left the page's inline data alone.")

    for name in scenarios:
        data = all_outputs[name]
        best = data["results"][data["best"]]
        verdict = "CONVERGED" if best.get("converged") else "did NOT converge"
        cert = best.get("certificate_violation", float("nan"))
        cert_txt = "certified" if best.get("certified") else f"certificate VIOLATED by {cert:.3e}"
        print(
            f"\n{data['title']}: best={data['best']}, "
            f"clearance={best['min_clearance']:.4f}, {verdict} "
            f"({best.get('stop_label', '?')}), {cert_txt}"
        )
        if not (best.get("converged") and best.get("certified") and best.get("feasible")):
            print("  NOT SUITABLE AS A FIGURE: this run does not support the hull claim.")
        for row in best["control_points"]:
            print(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],")

    if not args.no_open and viewer_target.exists():
        print(f"\nOpen it with:  open {viewer_target}")
        print(f"Or live-solve:  python3 -m spacetime_bezier")
    return 0
