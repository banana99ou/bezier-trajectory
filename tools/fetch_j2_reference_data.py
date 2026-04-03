#!/usr/bin/env python3
"""
Fetch and normalize external reference data for the J2 validation suite.

Default behavior:
- download the EGM-2008 normalized coefficient table from Vallado's public repo
- extract the fully normalized C20 term
- generate a deterministic J2 validation dataset in JSON format
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbital_docking import constants
from orbital_docking.j2_validation import (
    DEFAULT_EGM2008_C20_URL,
    build_reference_dataset,
    parse_c20_from_egm_text,
    save_reference_dataset,
)

DEFAULT_OUTPUT_PATH = REPO_ROOT / "tests" / "data" / "j2_reference" / "egm2008_degree2_samples.json"
DEFAULT_RAW_PATH = REPO_ROOT / "artifacts" / "j2_reference" / "EGM-08norm100.txt"
DEFAULT_SOURCE_LABEL = "CelesTrak/Vallado EGM-08 normalized coefficients"


def _download_text(url: str, timeout_s: float) -> str:
    """
    Download text from a URL.

    Python's certificate store is not always configured on local machines, so fall
    back to curl when urllib fails.
    """
    req = urllib.request.Request(
        str(url),
        headers={"User-Agent": "bezier-trajectory-j2-validator/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        res = subprocess.run(
            ["curl", "-LfsS", str(url)],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and normalize J2 reference data.")
    parser.add_argument("--url", default=DEFAULT_EGM2008_C20_URL, help="Coefficient table URL.")
    parser.add_argument("--source-label", default=DEFAULT_SOURCE_LABEL, help="Human-readable source label.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Normalized JSON output path.")
    parser.add_argument(
        "--raw-output",
        default=str(DEFAULT_RAW_PATH),
        help="Optional raw downloaded coefficient file output path. Use empty string to skip.",
    )
    parser.add_argument(
        "--gradient-step-km",
        type=float,
        default=1e-3,
        help="Central-difference step used to derive the reference acceleration from the scalar potential.",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Download timeout in seconds.")
    args = parser.parse_args()

    raw_text = _download_text(args.url, args.timeout_s)
    c20 = parse_c20_from_egm_text(raw_text)
    dataset = build_reference_dataset(
        mu_km3_s2=constants.EARTH_MU_SCALED,
        r_e_km=constants.EARTH_RADIUS_KM,
        c20_normalized=c20,
        source_url=args.url,
        source_label=args.source_label,
        gradient_step_km=args.gradient_step_km,
    )

    output_path = Path(args.output)
    save_reference_dataset(output_path, dataset)

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_text)

    print("=" * 80)
    print("J2 reference dataset written")
    print(f"output: {output_path}")
    if args.raw_output:
        print(f"raw:    {Path(args.raw_output)}")
    print(f"url:    {args.url}")
    print(f"c20:    {c20:.15e}")
    print(f"j2:     {dataset['constants']['j2']:.15e}")
    print(f"samples:{len(dataset['samples'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
