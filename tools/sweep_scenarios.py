"""One-pass sweep over the scenario registry, at defaults.

Prints the same fields the feasibility gate reads, so a row that says
``grade=True`` is a row `tools/make_paper_figure.py` would agree to draw.
Not a benchmark: no timing, no tuning, one configuration per scenario unless
asked for more.

    python3 tools/sweep_scenarios.py [scenario ...] [--all-configs]
"""

from __future__ import annotations

import sys

from spacetime_bezier.optimize import optimize_scenario
from spacetime_bezier.scenarios import SCENARIO_MAP


def main(argv: list[str]) -> int:
    all_configs = "--all-configs" in argv
    sound_clip = "--sound-clip" in argv
    names = [a for a in argv if not a.startswith("--")] or list(SCENARIO_MAP)

    header = (
        f"{'scenario':<15}{'config':<10}{'conv':<6}{'iters':<7}{'acc/rej':<9}"
        f"{'clearance':<12}{'koz cert':<11}{'occ cert':<11}{'slack':<11}"
        f"{'weight':<10}{'grade':<6}"
    )
    print(f"clip: {'sound (statement 8, clamped)' if sound_clip else 'tangent (statement 8 off)'}")
    print(header)
    print("-" * len(header))

    worst = 0
    for name in names:
        fn, configs = SCENARIO_MAP[name]
        use = configs if all_configs else configs[:1]
        out = optimize_scenario(fn(), use, verbose=False, sound_clip=sound_clip)
        for key, r in out["results"].items():
            grade = bool(r.get("figure_grade"))
            acc_rej = f"{r.get('accept_count')}/{r.get('reject_count')}"
            if not grade:
                worst = 1
            print(
                f"{name:<15}{key:<10}{str(r.get('converged')):<6}"
                f"{str(r.get('iterations')):<7}"
                f"{acc_rej:<9}"
                f"{r.get('min_clearance', float('nan')):<12.5f}"
                f"{r.get('certificate_violation', float('nan')):<11.3e}"
                f"{r.get('occlusion_violation', float('nan')):<11.3e}"
                f"{r.get('total_slack', float('nan')):<11.3e}"
                f"{r.get('elastic_weight', float('nan')):<10.0f}"
                f"{str(grade):<6}"
            )
            if not grade:
                print(f"{'':<15}  reasons: {r.get('figure_grade_reasons')}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
