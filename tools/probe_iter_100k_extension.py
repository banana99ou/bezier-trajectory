"""
Extension to itercost sweep: add max_iter=100000 data point for both unfrozen
and warm-start K=5, to extend the dv(max_iter) curve another decade.

Appends to itercost_runs.json and updates itercost_summary.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from orbital_docking.optimization import (
    optimize_orbital_docking,
    generate_initial_control_points,
)
from tools.probe_frozen_jacobian import (
    eci_from_circular,
    EARTH_RADIUS_KM,
    PROGRESS_ALT,
    ISS_ALT,
    INC,
    RAAN,
    ISS_U,
    KOZ_RADIUS,
    TRANSFER_TIME,
    N_DEG,
)


N_SEG = 16
TOL = 1e-12
PHASE_LAG = 120.0
EXTRA_CAPS = [100000]
OUT_DIR = Path("artifacts/probe_frozen_jacobian")


def make_p_init():
    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT
    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - PHASE_LAG)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N_DEG, P_start, P_end)
    return P_init, v0, v1


def run(*, P_init, v0, v1, freeze, fai, max_iter):
    t0 = time.time()
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=N_SEG, r_e=KOZ_RADIUS, max_iter=max_iter, tol=TOL,
        v0=v0, v1=v1, sample_count=100,
        objective_mode="dv",
        scp_prox_weight=1e-6, scp_trust_radius=2000.0,
        transfer_time=TRANSFER_TIME,
        freeze_gravity_jacobian=freeze, freeze_after_iter=fai,
        verbose=False, use_cache=False, ignore_existing_cache=True,
    )
    return P_opt, info, time.time() - t0


def main():
    P_init, v0, v1 = make_p_init()
    runs_path = OUT_DIR / "itercost_runs.json"
    if runs_path.exists():
        existing = json.loads(runs_path.read_text())
    else:
        existing = []

    print(f"\n=== Extending itercost sweep with max_iter={EXTRA_CAPS} ===")
    print(
        f"{'max_iter':>10} {'mode':>10} {'iters':>6} {'dv (m/s)':>12} "
        f"{'final_Δ':>12} {'wall (s)':>10} {'wall/iter (ms)':>15}"
    )
    print("-" * 90)
    new_runs = []
    for mi in EXTRA_CAPS:
        for mode_label, freeze, fai in [("unfrozen", False, 1), ("K=5", True, 5)]:
            P_opt, info, wall = run(
                P_init=P_init, v0=v0, v1=v1,
                freeze=freeze, fai=fai, max_iter=mi,
            )
            iters = int(info["iterations"])
            wall_per = wall / max(iters, 1) * 1000
            print(
                f"{mi:>10d} {mode_label:>10s} {iters:>6d} "
                f"{info['dv_proxy_m_s']:>12.3f} "
                f"{info.get('final_delta_norm', 0.0):>12.3e} "
                f"{wall:>10.2f} {wall_per:>15.3f}"
            )
            new_runs.append({
                "max_iter": mi,
                "mode": mode_label,
                "freeze": freeze,
                "freeze_after_iter": fai,
                "iters": iters,
                "dv_proxy_m_s": float(info["dv_proxy_m_s"]),
                "final_delta_norm": float(info.get("final_delta_norm", 0.0)),
                "wall_time_s": float(wall),
                "wall_per_iter_ms": float(wall_per),
                "min_radius_km": float(info["min_radius"]),
                "termination_reason": str(info.get("termination_reason", "")),
            })

    all_runs = existing + new_runs
    runs_path.write_text(json.dumps(all_runs, indent=2))
    print(f"\nAppended {len(new_runs)} runs to {runs_path}")

    # Regenerate plot + summary
    import plotly.graph_objects as go
    fig = go.Figure()
    for mode_label, color in [("unfrozen", "royalblue"), ("K=5", "crimson")]:
        sub = [r for r in all_runs if r["mode"] == mode_label]
        sub.sort(key=lambda r: r["max_iter"])
        x = [r["max_iter"] for r in sub]
        y = [r["dv_proxy_m_s"] for r in sub]
        deltas = [r["final_delta_norm"] for r in sub]
        walls = [r["wall_time_s"] for r in sub]
        text = [
            f"max_iter={mi}<br>dv={dv:.2f}<br>final_Δ={d:.2e}<br>wall={w:.1f}s"
            for mi, dv, d, w in zip(x, y, deltas, walls)
        ]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            marker=dict(size=10, color=color),
            line=dict(width=2, color=color),
            name=mode_label, text=text, hoverinfo="text",
        ))
    fig.update_layout(
        title=(
            "Convergence rate: Δv vs max_iter (K=5 vs unfrozen, extended to 100k)<br>"
            "<sub>n_seg=16, dv objective, 120° lag.</sub>"
        ),
        xaxis=dict(title="max_iter cap", type="log"),
        yaxis=dict(title="dv_proxy (m/s)"),
        height=550,
    )
    fig.write_html(str(OUT_DIR / "itercost_dv_vs_iter.html"), include_plotlyjs="cdn")
    print(f"Updated: {OUT_DIR / 'itercost_dv_vs_iter.html'}")

    # Update summary with diminishing-returns table
    lines_extra = []
    lines_extra.append("\n## Diminishing returns: cost vs benefit at higher max_iter\n")
    lines_extra.append("| max_iter | mode | dv (m/s) | wall (s) | Δdv vs cap=1000 | Δwall vs cap=1000 |")
    lines_extra.append("|----------|------|----------|----------|-----------------|--------------------|")

    for mode_label in ["unfrozen", "K=5"]:
        sub = sorted([r for r in all_runs if r["mode"] == mode_label],
                     key=lambda r: r["max_iter"])
        base = next((r for r in sub if r["max_iter"] == 1000), None)
        if base is None:
            continue
        for r in sub:
            if r["max_iter"] not in (1000, 10000, 100000):
                continue
            ddv = r["dv_proxy_m_s"] - base["dv_proxy_m_s"]
            dwall = r["wall_time_s"] - base["wall_time_s"]
            lines_extra.append(
                f"| {r['max_iter']} | {mode_label} | {r['dv_proxy_m_s']:.3f} | "
                f"{r['wall_time_s']:.2f} | {ddv:+.3f} | {dwall:+.2f} |"
            )

    summary_path = OUT_DIR / "itercost_summary.md"
    if summary_path.exists():
        existing_md = summary_path.read_text()
        if "Diminishing returns" not in existing_md:
            summary_path.write_text(existing_md + "\n".join(lines_extra) + "\n")
            print(f"Appended diminishing-returns section to {summary_path}")


if __name__ == "__main__":
    main()
