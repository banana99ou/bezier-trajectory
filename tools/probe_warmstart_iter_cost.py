"""
Iteration-cost probe: does K=5 warm-start freeze converge in fewer iterations,
or just at lower per-iter cost?

Sweep max_iter ∈ {100, 500, 1000, 3000, 10000, 30000} for both unfrozen and
warm-start K=5. Compare dv, final_delta_norm, and wall time at each cap. Two
useful views:

  - dv(max_iter): does K=5 reach final dv earlier (in iter count)?
  - wall(max_iter): per-iter wall time difference, scales linearly with iters

Outputs:
  - itercost_runs.json
  - itercost_summary.md
  - itercost_dv_vs_iter.html  — Plotly: dv vs max_iter for both modes
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
MAX_ITER_GRID = [100, 500, 1000, 3000, 10000, 30000]
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    P_init, v0, v1 = make_p_init()

    print(f"\n=== Iteration-cost sweep: K=5 vs unfrozen, max_iter sweep ===")
    print(
        f"{'max_iter':>10} {'mode':>10} {'iters':>6} {'dv (m/s)':>12} "
        f"{'final_Δ':>12} {'wall (s)':>10} {'wall/iter (ms)':>15}"
    )
    print("-" * 90)

    runs = []
    for mi in MAX_ITER_GRID:
        for mode_label, freeze, fai in [("unfrozen", False, 1), ("K=5", True, 5)]:
            P_opt, info, wall = run(
                P_init=P_init, v0=v0, v1=v1,
                freeze=freeze, fai=fai, max_iter=mi,
            )
            iters = int(info["iterations"])
            wall_per = wall / max(iters, 1) * 1000  # ms/iter
            print(
                f"{mi:>10d} {mode_label:>10s} {iters:>6d} "
                f"{info['dv_proxy_m_s']:>12.3f} "
                f"{info.get('final_delta_norm', 0.0):>12.3e} "
                f"{wall:>10.2f} {wall_per:>15.3f}"
            )
            runs.append({
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

    (OUT_DIR / "itercost_runs.json").write_text(json.dumps(runs, indent=2))

    # ---- Plot: dv vs max_iter ----
    import plotly.graph_objects as go
    fig = go.Figure()

    for mode_label, color in [("unfrozen", "royalblue"), ("K=5", "crimson")]:
        sub = [r for r in runs if r["mode"] == mode_label]
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
            name=mode_label,
            text=text, hoverinfo="text",
        ))

    fig.update_layout(
        title=(
            "Convergence rate: Δv vs max_iter (warm-start K=5 vs unfrozen)<br>"
            "<sub>n_seg=16, dv objective, 120° lag. If K=5 converges earlier, "
            "its curve plateaus sooner.</sub>"
        ),
        xaxis=dict(title="max_iter cap", type="log"),
        yaxis=dict(title="dv_proxy (m/s)"),
        height=550,
    )
    out_html = OUT_DIR / "itercost_dv_vs_iter.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\nWrote: {out_html}")

    # ---- Summary markdown ----
    lines = []
    lines.append("# Iteration-cost probe — K=5 vs unfrozen across max_iter")
    lines.append("")
    lines.append(
        f"Setup: n_seg={N_SEG}, dv objective, {PHASE_LAG}° lag, tol={TOL}, "
        "scp_prox=1e-6, trust=2000 km, cache ignored."
    )
    lines.append("")
    lines.append(
        "| max_iter | mode | iters | dv (m/s) | final_Δ | wall (s) | wall/iter (ms) |"
    )
    lines.append(
        "|----------|------|-------|----------|---------|----------|----------------|"
    )
    for r in runs:
        lines.append(
            f"| {r['max_iter']} | {r['mode']} | {r['iters']} | "
            f"{r['dv_proxy_m_s']:.3f} | {r['final_delta_norm']:.2e} | "
            f"{r['wall_time_s']:.2f} | {r['wall_per_iter_ms']:.3f} |"
        )
    lines.append("")

    # Per-cap deltas
    lines.append("## Per-cap dv comparison (K=5 − unfrozen)")
    lines.append("")
    lines.append("| max_iter | dv unfrozen | dv K=5 | Δdv | Δdv % |")
    lines.append("|----------|-------------|--------|-----|-------|")
    for mi in MAX_ITER_GRID:
        u = next(r for r in runs if r["max_iter"] == mi and r["mode"] == "unfrozen")
        f5 = next(r for r in runs if r["max_iter"] == mi and r["mode"] == "K=5")
        ddv = f5["dv_proxy_m_s"] - u["dv_proxy_m_s"]
        pct = ddv / max(abs(u["dv_proxy_m_s"]), 1e-12) * 100
        lines.append(
            f"| {mi} | {u['dv_proxy_m_s']:.3f} | {f5['dv_proxy_m_s']:.3f} | "
            f"{ddv:+.3f} | {pct:+.4f}% |"
        )

    lines.append("")
    lines.append("## Wall/iter comparison")
    lines.append("")
    walls_u = [r["wall_per_iter_ms"] for r in runs if r["mode"] == "unfrozen"]
    walls_5 = [r["wall_per_iter_ms"] for r in runs if r["mode"] == "K=5"]
    if walls_u and walls_5:
        u_med = float(np.median(walls_u))
        f5_med = float(np.median(walls_5))
        savings = (u_med - f5_med) / u_med * 100
        lines.append(
            f"Median wall/iter: unfrozen {u_med:.3f} ms, K=5 {f5_med:.3f} ms — "
            f"K=5 saves {savings:.1f}% per iter."
        )
        lines.append("")
        lines.append(
            "This per-iter saving is the cost of `compute_segment_lin` (16 "
            "numerical-Jacobian builds @ n_seg=16). The QP solve dominates "
            "the rest, so the saving is modest."
        )

    (OUT_DIR / "itercost_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote: {OUT_DIR/'itercost_summary.md'}")


if __name__ == "__main__":
    main()
