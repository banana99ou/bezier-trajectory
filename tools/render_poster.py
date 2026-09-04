#!/usr/bin/env python3
"""Build the KSAS poster: inject measured numbers, then run xelatex.

The poster quotes no number that was typed by hand. Every result on it comes
from ``figures/paper1/occlusion_figure.json`` -- the sidecar that
``tools/make_paper_figure.py`` writes beside the figure it drew -- and this
script writes them into ``numbers.tex`` as macros the poster expands. It then
draws the poster's own figures (``poster/make_poster_figures.py``, which reads
the same sidecar for every number it prints) and runs xelatex twice.

The gate below is not decoration. It refuses to build, and each condition names
the way the demo could stop being a demo:

  * a certificate above 1e-6, or slack above 1e-6, or a nonzero unsound clip
    count -- the run no longer certifies, so its numbers are not figure-grade;
  * clearance at or below zero -- the returned trajectory penetrates;
  * a baseline that does NOT lose the link -- there is nothing to show;
  * a constrained run that does not keep it -- the method did not work;
  * a graft that keeps the link -- the retiming is not what saved the run, and
    the poster's central claim would be false.

Run it and see: flip any one of those in the JSON and the build stops.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIDECAR = ROOT / "figures" / "paper1" / "occlusion_figure.json"
POSTER_DIR = ROOT / "paper" / "ksas_2026_fall" / "poster"

TOL = 1e-6


def check(sidecar: dict) -> None:
    """Raise unless the sidecar is figure-grade AND still shows what it claims."""
    con, base = sidecar["constrained"], sidecar["baseline"]
    failures = []

    def bad(cond: str, ok: bool) -> None:
        if not ok:
            failures.append(cond)

    bad(f"constrained koz_certificate {con['koz_certificate']:.3e} > {TOL}",
        con["koz_certificate"] <= TOL)
    bad(f"constrained occlusion_certificate {con['occlusion_certificate']:.3e} > {TOL}",
        con["occlusion_certificate"] <= TOL)
    bad(f"baseline koz_certificate {base['koz_certificate']:.3e} > {TOL}",
        base["koz_certificate"] <= TOL)
    bad(f"constrained total_slack {con['total_slack']:.3e} > {TOL}",
        con["total_slack"] <= TOL)
    bad(f"baseline total_slack {base['total_slack']:.3e} > {TOL}",
        base["total_slack"] <= TOL)
    bad(f"constrained unsound_clips {con['unsound_clips']}", con["unsound_clips"] == 0)
    bad(f"constrained min_clearance {con['min_clearance']:.4f} <= 0",
        con["min_clearance"] > 0.0)
    bad(f"baseline min_los_margin {base['min_los_margin']:+.4f} >= 0 -- the "
        f"baseline does not lose the link, so there is nothing to show",
        base["min_los_margin"] < 0.0)
    bad(f"constrained min_los_margin {con['min_los_margin']:+.4f} <= 0 -- the "
        f"constrained run does not keep the link",
        con["min_los_margin"] > 0.0)
    bad(f"graft_min_los_margin {con['graft_min_los_margin']:+.4f} >= 0 -- the "
        f"constrained path flown on the baseline schedule KEEPS the link, so "
        f"retiming is not what saved the run",
        con["graft_min_los_margin"] < 0.0)

    if failures:
        raise SystemExit(
            "REFUSING to build the poster -- the sidecar is not figure-grade, "
            "or no longer shows what the poster claims:\n  - "
            + "\n  - ".join(failures)
            + f"\n\nsidecar: {SIDECAR}\nRe-run tools/make_paper_figure.py, or fix the poster's claim."
        )


def _sci(value: float) -> str:
    """LaTeX math for a weight, e.g. 1e5 -> ``10^{5}``, 3000 -> ``3\\times 10^{3}``."""
    mantissa, exponent = f"{value:.0e}".split("e")
    exp = int(exponent)
    if mantissa == "1":
        return f"10^{{{exp}}}"
    return f"{mantissa}\\times 10^{{{exp}}}"


def macros(sidecar: dict) -> str:
    con, base = sidecar["constrained"], sidecar["baseline"]
    lo, hi = base["los_loss_interval"]
    out = {
        # provenance
        "NUMgenerated": sidecar["generated"],
        "NUMgit": sidecar["git"],
        "NUMmachine": sidecar["machine"],
        # solver configuration, as run
        "NUMdegree": f"{sidecar['N']}",
        "NUMnseg": f"{sidecar['n_seg']}",
        "NUMvmax": f"{sidecar['v_max']:g}",
        "NUMtimeweight": f"{sidecar['time_weight']:g}",
        "NUMelastic": _sci(sidecar["elastic_weight"]),
        "NUMtrust": f"{sidecar['trust_radius']:g}",
        # baseline -- occlusion rows off
        "NUMbaseArrival": f"{base['arrival_time']:.1f}",
        "NUMbaseLos": f"{base['min_los_margin']:+.2f}",
        "NUMbaseLossLo": f"{lo:.2f}",
        "NUMbaseLossHi": f"{hi:.2f}",
        "NUMbaseLateral": f"{base['lateral_deviation']:.2f}",
        "NUMbaseIters": f"{base['iterations']}",
        "NUMbaseSolve": f"{base['solve_seconds']:.2f}",
        # constrained -- occlusion rows on
        "NUMconArrival": f"{con['arrival_time']:.1f}",
        "NUMconLos": f"{con['min_los_margin']:+.2f}",
        "NUMconLateral": f"{con['lateral_deviation']:.2f}",
        "NUMconIters": f"{con['iterations']}",
        "NUMconSolve": f"{con['solve_seconds']:.2f}",
        "NUMconClearance": f"{con['min_clearance']:.1f}",
        "NUMgraft": f"{con['graft_min_los_margin']:+.2f}",
        # the conservatism the lifted KOZ costs
        "NUMaxisBound": f"{sidecar['axis_arrival_bound']:.1f}",
        "NUMarrivalGap": f"{con['arrival_gap_to_axis_bound']:.1f}",
        "NUMsnapshot": f"{sidecar['snapshot_time']:.1f}",
        "NUMdelay": f"{con['arrival_time'] - base['arrival_time']:.1f}",
    }
    lines = [
        "% GENERATED by tools/render_poster.py -- do not edit.",
        f"% source: figures/paper1/occlusion_figure.json ({sidecar['generated']})",
        "",
    ]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numbers-only", action="store_true",
                    help="write numbers.tex and stop; do not run xelatex")
    args = ap.parse_args()

    sidecar = json.loads(SIDECAR.read_text())
    check(sidecar)

    numbers = POSTER_DIR / "numbers.tex"
    numbers.write_text(macros(sidecar), encoding="utf-8")
    print(f"wrote {numbers.relative_to(ROOT)} from {SIDECAR.relative_to(ROOT)}")
    if args.numbers_only:
        return 0

    if shutil.which("xelatex") is None:
        raise SystemExit("xelatex not on PATH -- install TeX Live, or pass --numbers-only")

    # The poster's own figures, from the same sidecar the numbers come from.
    proc = subprocess.run(
        [sys.executable, str(POSTER_DIR / "make_poster_figures.py"), "--sidecar", str(SIDECAR)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        raise SystemExit("make_poster_figures.py failed")
    sys.stdout.write(proc.stdout)

    for pass_no in (1, 2):  # twice, so any cross-reference settles
        proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "poster.tex"],
            cwd=POSTER_DIR, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.stdout.write(proc.stdout[-4000:])
            raise SystemExit(f"xelatex failed on pass {pass_no}")

    pdf = POSTER_DIR / "poster.pdf"
    print(f"wrote {pdf.relative_to(ROOT)} ({pdf.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
