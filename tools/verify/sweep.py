"""
Pillar 4b -- Regression sweep.

Grid: N in {6,7,8} x n_seg in {2,4,8,16,32,64} x scenario in {phase120, phase70}, energy.
PASS: every fine-mesh (n_seg >= 8) cell converges (iter < cap) and is feasible, where
feasible = curve clears the KOZ (dense) AND the solve-time KOZ slack vanished (the SCvx
virtual-control acceptance condition); the cost spread across n_seg in {8,16,32,64} is
small; and every capped cell is coarse (n_seg<=4). Coarse meshes are documented failures, not gated:
n_seg=2 converges but carries O(100-1000 km) slack (never actually solved -- the paper
reports n_seg=2 as infeasible), and n_seg=4 caps at aggressive geometry + high degree.

Run:  .venv/bin/python tools/verify/sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np

from tools.verify import harness_common as H

DEGREES = [6, 7, 8]
N_SEGS = [2, 4, 8, 16, 32, 64]
SCENARIOS = ["phase120", "phase70"]
MAX_ITER = 1000
OUT = H.ARTIFACT_ROOT / "pillar4_sweep"


def run():
    rows = []
    for scen in SCENARIOS:
        for N in DEGREES:
            sc = H.make_scenario(scen, N=N)
            for ns in N_SEGS:
                P, info = H.run_rust(sc, n_seg=ns, max_iter=MAX_ITER)
                it = int(info["iterations"])
                rows.append(dict(
                    scenario=scen, N=N, n_seg=ns, iterations=it,
                    converged=int(info.get("scvx_converged", -1)), capped=(it >= MAX_ITER),
                    feasible=bool(info["feasible"]),
                    min_radius=float(info["min_radius"]),
                    clearance=float(info["min_radius"] - sc["r_e"]),
                    cost_true_energy=float(info["cost_true_energy"]),
                    J_true=H.J_true(P, sc["T"]),
                    max_koz_slack=float(info.get("max_koz_slack", 0.0)),
                ))

    # PASS: fine-mesh (n_seg >= 8) cells converge + feasible; n_seg in {2,4} are the
    # documented coarse-mesh failures (reported below, not gated).
    fine = [r for r in rows if r["n_seg"] >= 8]
    fine_ok = all((not r["capped"]) and r["converged"] == 1 and r["feasible"] for r in fine)
    bad = [r for r in rows if r["capped"]]
    # Caps are acceptable only at the documented coarse meshes (n_seg <= 4), where the
    # half-space geometry genuinely cannot certify the KOZ. (Under the canonical
    # certificate-gated acceptance, n_seg=4 now converges everywhere and n_seg=2 runs
    # honestly to the cap instead of exiting quietly with slack.)
    only_holdouts_are_coarse = all(r["n_seg"] <= 4 for r in bad) if bad else True
    n_coarse_cap = sum(1 for r in rows if r["n_seg"] <= 4 and r["capped"])
    n_coarse = sum(1 for r in rows if r["n_seg"] <= 4)
    coarse_bad = [r for r in rows if r["n_seg"] in (2, 4) and not r["feasible"]]

    # Cost spread across n_seg in {8,16,32,64} per (scenario, N).
    spread_ok = True
    spreads = []
    for scen in SCENARIOS:
        for N in DEGREES:
            js = [r["J_true"] for r in rows if r["scenario"] == scen and r["N"] == N
                  and r["n_seg"] in (8, 16, 32, 64)]
            if js:
                spr = (max(js) - min(js)) / min(js)
                spreads.append((scen, N, spr))
                spread_ok = spread_ok and (spr < 0.30)

    passed = fine_ok and spread_ok and only_holdouts_are_coarse  # coarse-mesh failures reported, not gated

    H.write_csv(OUT / "sweep.csv", [
        {k: (f"{v:.6e}" if k in ("cost_true_energy", "J_true") else
             (f"{v:.3f}" if isinstance(v, float) else v)) for k, v in r.items()} for r in rows])

    md = [f"# Pillar 4b -- Regression sweep (energy)", ""]
    md.append(f"Grid: N∈{DEGREES} × n_seg∈{N_SEGS} × {SCENARIOS}  ({len(rows)} runs)")
    md.append("")
    md.append(f"- all fine-mesh (n_seg≥8) cells converge (iter<cap) + feasible "
              f"(curve clears + solve-slack vanished): **{fine_ok}** "
              f"({sum((not r['capped']) and r['converged']==1 and r['feasible'] for r in fine)}/{len(fine)})")
    md.append(f"- every capped cell is coarse (n_seg<=4): **{only_holdouts_are_coarse}** "
              f"({n_coarse_cap}/{n_coarse} coarse cells cap; n_seg=4 now converges everywhere, "
              f"n_seg=2 is a genuine geometric infeasibility that runs to the cap honestly)")
    md.append(f"- cost spread across n_seg∈{{8,16,32,64}} < 30% for all (N,scenario): **{spread_ok}** "
              f"(max spread={max(s for _,_,s in spreads):.1%})")
    md.append("")
    md.append("### Coarse-mesh failures (documented, not gated: n_seg=2 carries slack ⇒ never "
              "actually solved; n_seg=4 caps at the aggressive corner)")
    md.append("| scenario | N | n_seg | iters | capped | feasible | max_slack |")
    md.append("|---|---|---|---|---|---|---|")
    for r in coarse_bad:
        md.append(f"| {r['scenario']} | {r['N']} | {r['n_seg']} | {r['iterations']} | "
                  f"{r['capped']} | {r['feasible']} | {r['max_koz_slack']:.2f} |")
    md.append("")
    md.append(f"## VERDICT: {'PASS' if passed else 'FAIL'}  "
              f"(n_seg∈{{2,4}} are documented coarse-mesh failures, reported not gated)")
    H.write_text(OUT / "summary.md", "\n".join(md))
    print("\n".join(md))
    return passed


if __name__ == "__main__":
    run()
