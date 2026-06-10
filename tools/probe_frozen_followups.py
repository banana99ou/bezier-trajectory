"""
Follow-up probes after the iter-1 freeze refutation:
  Test 1 — phase-lag sweep (does the refutation generalize across scenarios?)
  Test 2 — energy objective (does refutation generalize across cost functions?)
  Test 3 — warm-start freeze (does freezing AFTER a few iters recover unfrozen quality?)

n_seg=16, max_iter=10000, tol=1e-12 (rely on max_iter, not tol).

Outputs to artifacts/probe_frozen_jacobian/:
  - followups_runs.json     — raw records
  - followups_summary.md    — verdict per test + tables
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from orbital_docking import constants
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
MAX_ITER = 10000
TOL = 1e-12
OUT_DIR = Path("artifacts/probe_frozen_jacobian")


def make_p_init(progress_lag_deg: float):
    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT
    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - progress_lag_deg)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N_DEG, P_start, P_end)
    return P_init, v0, v1


def run(
    *,
    P_init,
    v0,
    v1,
    n_seg=N_SEG,
    objective_mode="dv",
    freeze_gravity_jacobian=False,
    freeze_after_iter=1,
    max_iter=MAX_ITER,
):
    t0 = time.time()
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=n_seg,
        r_e=KOZ_RADIUS,
        max_iter=max_iter,
        tol=TOL,
        v0=v0,
        v1=v1,
        sample_count=100,
        objective_mode=objective_mode,
        scp_prox_weight=1e-6,
        scp_trust_radius=2000.0,
        transfer_time=TRANSFER_TIME,
        freeze_gravity_jacobian=freeze_gravity_jacobian,
        freeze_after_iter=freeze_after_iter,
        verbose=False,
        use_cache=False,
        ignore_existing_cache=True,
    )
    return P_opt, info, time.time() - t0


def summarize_record(P_opt, info, wall) -> dict:
    return {
        "iterations": int(info["iterations"]),
        "wall_time_s": float(wall),
        "feasible": bool(info["feasible"]),
        "min_radius_km": float(info["min_radius"]),
        "dv_proxy_m_s": float(info["dv_proxy_m_s"]),
        "cost_true_energy": float(info["cost_true_energy"]),
        "max_control_accel_ms2": float(info["max_control_accel_ms2"]),
        "final_delta_norm": float(info.get("final_delta_norm", float("nan"))),
        "max_koz_slack": float(info.get("max_koz_slack", 0.0)),
        "termination_reason": str(info.get("termination_reason", "")),
    }


# ---- Test 1: phase-lag sweep ----

def test_phase_lag_sweep():
    lags = [60.0, 120.0, 180.0, 240.0]
    results = []
    print("\n=== Test 1: phase-lag sweep (dv objective, iter-1 freeze) ===")
    print(
        f"{'lag (deg)':>10} {'mode':>10} {'iters':>6} {'dv (m/s)':>12} "
        f"{'min_r (km)':>12} {'wall (s)':>10} {'final_delta':>12} {'feasible':>10}"
    )
    print("-" * 100)
    for lag in lags:
        P_init, v0, v1 = make_p_init(lag)
        for freeze in (False, True):
            _, info, wall = run(
                P_init=P_init, v0=v0, v1=v1,
                objective_mode="dv",
                freeze_gravity_jacobian=freeze,
                freeze_after_iter=1,
            )
            r = summarize_record(None, info, wall)
            r.update({"test": "phase_lag", "phase_lag_deg": lag,
                      "freeze": freeze, "freeze_after_iter": 1,
                      "objective_mode": "dv"})
            results.append(r)
            mode = "frozen" if freeze else "unfrozen"
            print(
                f"{lag:>10.1f} {mode:>10s} {r['iterations']:>6d} "
                f"{r['dv_proxy_m_s']:>12.3f} {r['min_radius_km']:>12.3f} "
                f"{wall:>10.2f} {r['final_delta_norm']:>12.3e} "
                f"{str(r['feasible']):>10s}"
            )
    return results


# ---- Test 2: energy objective ----

def test_energy_objective():
    results = []
    print("\n=== Test 2: energy objective (120 deg lag, iter-1 freeze) ===")
    print(
        f"{'mode':>10} {'iters':>6} {'cost_true':>12} {'dv (m/s)':>12} "
        f"{'min_r (km)':>12} {'wall (s)':>10} {'final_delta':>12} {'feasible':>10}"
    )
    print("-" * 100)
    P_init, v0, v1 = make_p_init(120.0)
    for freeze in (False, True):
        _, info, wall = run(
            P_init=P_init, v0=v0, v1=v1,
            objective_mode="energy",
            freeze_gravity_jacobian=freeze,
            freeze_after_iter=1,
        )
        r = summarize_record(None, info, wall)
        r.update({"test": "energy_objective", "phase_lag_deg": 120.0,
                  "freeze": freeze, "freeze_after_iter": 1,
                  "objective_mode": "energy"})
        results.append(r)
        mode = "frozen" if freeze else "unfrozen"
        print(
            f"{mode:>10s} {r['iterations']:>6d} {r['cost_true_energy']:>12.4e} "
            f"{r['dv_proxy_m_s']:>12.3f} {r['min_radius_km']:>12.3f} "
            f"{wall:>10.2f} {r['final_delta_norm']:>12.3e} "
            f"{str(r['feasible']):>10s}"
        )
    return results


# ---- Test 3: warm-start freeze ----

def test_warm_start_freeze():
    results = []
    print("\n=== Test 3: warm-start freeze (120 deg lag, dv, n_seg=16) ===")
    print(
        f"{'freeze_after':>13} {'iters':>6} {'dv (m/s)':>12} {'Δ vs unfrozen (%)':>18} "
        f"{'min_r (km)':>12} {'wall (s)':>10} {'||P_f-P_u||':>14}"
    )
    print("-" * 100)
    P_init, v0, v1 = make_p_init(120.0)

    # Baseline unfrozen
    P_u, info_u, wall_u = run(
        P_init=P_init, v0=v0, v1=v1,
        objective_mode="dv",
        freeze_gravity_jacobian=False,
        freeze_after_iter=1,
    )
    r_u = summarize_record(P_u, info_u, wall_u)
    r_u.update({"test": "warm_start", "phase_lag_deg": 120.0,
                "freeze": False, "freeze_after_iter": 0,
                "objective_mode": "dv"})
    results.append(r_u)
    print(
        f"{'never (unfrz)':>13s} {r_u['iterations']:>6d} "
        f"{r_u['dv_proxy_m_s']:>12.3f} {'(baseline)':>18s} "
        f"{r_u['min_radius_km']:>12.3f} {wall_u:>10.2f} {'0.000':>14s}"
    )

    # Warm-start freeze sweep
    for fai in (1, 5, 20, 100, 500, 2000):
        P_f, info_f, wall_f = run(
            P_init=P_init, v0=v0, v1=v1,
            objective_mode="dv",
            freeze_gravity_jacobian=True,
            freeze_after_iter=fai,
        )
        r = summarize_record(P_f, info_f, wall_f)
        delta_p = float(np.linalg.norm(P_f - P_u))
        dv_pct = (r["dv_proxy_m_s"] - r_u["dv_proxy_m_s"]) / max(
            abs(r_u["dv_proxy_m_s"]), 1e-12
        ) * 100.0
        r.update({"test": "warm_start", "phase_lag_deg": 120.0,
                  "freeze": True, "freeze_after_iter": fai,
                  "objective_mode": "dv",
                  "p_dist_to_unfrozen": delta_p,
                  "dv_pct_vs_unfrozen": dv_pct})
        results.append(r)
        print(
            f"{fai:>13d} {r['iterations']:>6d} "
            f"{r['dv_proxy_m_s']:>12.3f} {dv_pct:>+17.3f}% "
            f"{r['min_radius_km']:>12.3f} {wall_f:>10.2f} {delta_p:>14.3f}"
        )

    return results


def write_summary_md(all_results, path: Path):
    by_test = {}
    for r in all_results:
        by_test.setdefault(r["test"], []).append(r)

    lines = []
    lines.append("# Frozen-Jacobian probe — follow-up tests")
    lines.append("")
    lines.append(
        f"Setup: n_seg={N_SEG}, max_iter={MAX_ITER}, tol={TOL}, dv objective "
        "unless noted, scp_prox=1e-6, trust=2000 km, cache ignored."
    )
    lines.append("")

    # --- Test 1 ---
    lines.append("## Test 1 — Phase-lag sweep")
    lines.append("")
    lines.append(
        "Question: does the iter-1-freeze refutation hold across different "
        "transfer geometries, or is it specific to the 120° case?"
    )
    lines.append("")
    lines.append(
        "| lag (°) | mode | iters | dv (m/s) | min_r (km) | feasible | wall (s) | Δdv vs unfrozen |"
    )
    lines.append(
        "|---------|------|-------|----------|------------|----------|----------|-----------------|"
    )
    pl_records = sorted(by_test.get("phase_lag", []), key=lambda r: (r["phase_lag_deg"], r["freeze"]))
    pl_by_lag = {}
    for r in pl_records:
        pl_by_lag.setdefault(r["phase_lag_deg"], {})[r["freeze"]] = r
    for lag in sorted(pl_by_lag.keys()):
        u = pl_by_lag[lag].get(False)
        f_ = pl_by_lag[lag].get(True)
        if u:
            lines.append(
                f"| {lag:.0f} | unfrozen | {u['iterations']} | "
                f"{u['dv_proxy_m_s']:.3f} | {u['min_radius_km']:.3f} | "
                f"{u['feasible']} | {u['wall_time_s']:.2f} | (baseline) |"
            )
        if f_:
            dv_pct = (f_["dv_proxy_m_s"] - u["dv_proxy_m_s"]) / max(abs(u["dv_proxy_m_s"]), 1e-12) * 100 if u else float("nan")
            lines.append(
                f"| {lag:.0f} | frozen | {f_['iterations']} | "
                f"{f_['dv_proxy_m_s']:.3f} | {f_['min_radius_km']:.3f} | "
                f"{f_['feasible']} | {f_['wall_time_s']:.2f} | "
                f"{dv_pct:+.2f}% |"
            )
    lines.append("")

    # --- Test 2 ---
    lines.append("## Test 2 — Energy objective")
    lines.append("")
    lines.append(
        "Question: does the refutation hold for the energy objective, which "
        "lacks IRLS reweighting and may behave differently?"
    )
    lines.append("")
    lines.append(
        "| mode | iters | cost (energy) | dv (m/s) | min_r (km) | feasible | wall (s) |"
    )
    lines.append(
        "|------|-------|---------------|----------|------------|----------|----------|"
    )
    eo_records = sorted(by_test.get("energy_objective", []), key=lambda r: r["freeze"])
    eo_by_freeze = {r["freeze"]: r for r in eo_records}
    for freeze in (False, True):
        r = eo_by_freeze.get(freeze)
        if r:
            mode = "frozen" if freeze else "unfrozen"
            lines.append(
                f"| {mode} | {r['iterations']} | {r['cost_true_energy']:.4e} | "
                f"{r['dv_proxy_m_s']:.3f} | {r['min_radius_km']:.3f} | "
                f"{r['feasible']} | {r['wall_time_s']:.2f} |"
            )
    if False in eo_by_freeze and True in eo_by_freeze:
        u = eo_by_freeze[False]
        f_ = eo_by_freeze[True]
        cost_pct = (f_["cost_true_energy"] - u["cost_true_energy"]) / max(abs(u["cost_true_energy"]), 1e-12) * 100
        dv_pct = (f_["dv_proxy_m_s"] - u["dv_proxy_m_s"]) / max(abs(u["dv_proxy_m_s"]), 1e-12) * 100
        lines.append("")
        lines.append(f"Energy cost gap: {cost_pct:+.2f}%; dv side-effect: {dv_pct:+.2f}%.")
    lines.append("")

    # --- Test 3 ---
    lines.append("## Test 3 — Warm-start freeze")
    lines.append("")
    lines.append(
        "Question: does freezing AFTER a few iterations (instead of at iter-1) "
        "recover unfrozen quality? If yes, the prof's intuition holds with a "
        "warm-up period."
    )
    lines.append("")
    lines.append(
        "| freeze_after | iters | dv (m/s) | Δdv % | min_r (km) | wall (s) | "
        "‖P_frozen − P_unfrozen‖ (km) |"
    )
    lines.append(
        "|--------------|-------|----------|-------|------------|----------|------------------------------|"
    )
    ws_records = by_test.get("warm_start", [])
    # Sort: unfrozen baseline first, then by freeze_after_iter
    ws_records = sorted(ws_records, key=lambda r: (r["freeze"], r.get("freeze_after_iter", 0)))
    for r in ws_records:
        if not r["freeze"]:
            lines.append(
                f"| _never (unfrozen)_ | {r['iterations']} | "
                f"{r['dv_proxy_m_s']:.3f} | (baseline) | "
                f"{r['min_radius_km']:.3f} | {r['wall_time_s']:.2f} | 0.0 |"
            )
        else:
            lines.append(
                f"| {r['freeze_after_iter']} | {r['iterations']} | "
                f"{r['dv_proxy_m_s']:.3f} | {r.get('dv_pct_vs_unfrozen', float('nan')):+.2f}% | "
                f"{r['min_radius_km']:.3f} | {r['wall_time_s']:.2f} | "
                f"{r.get('p_dist_to_unfrozen', 0.0):.2f} |"
            )
    lines.append("")

    lines.append("## Synthesis")
    lines.append("")
    lines.append(
        "*This section is filled in manually after inspecting the tables — "
        "see end-of-run console output for the headline.*"
    )

    path.write_text("\n".join(lines) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_results.extend(test_phase_lag_sweep())
    all_results.extend(test_energy_objective())
    all_results.extend(test_warm_start_freeze())

    (OUT_DIR / "followups_runs.json").write_text(json.dumps(all_results, indent=2))
    write_summary_md(all_results, OUT_DIR / "followups_summary.md")

    print("\nWrote:")
    print(f"  {OUT_DIR/'followups_runs.json'}")
    print(f"  {OUT_DIR/'followups_summary.md'}")


if __name__ == "__main__":
    main()
