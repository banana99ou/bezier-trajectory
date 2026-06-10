"""
Probe: does freezing the gravity Jacobian at iter-1 P_ref change anything?

Background: prof's verbal feedback was "선형화 한 번 하고 수렴할 때까지 상수 유지"
— linearize once and keep constant until convergence. The action list (line 33)
scopes this probe to the gravity Jacobian only. KOZ linearization continues
to update each iteration as before.

Sweep: n_seg in {2, 4, 8, 16, 32} x freeze in {False, True}.
Scenario matches tools/convergence_diagnostic.py (N=7, 120 deg phase lag,
T=1500 s, r_e=6471 km, dv objective, prox 1e-6, trust 2000 km).

Tolerance is set to 1e-6 (cache evidence already settles tol=1e-12). Cache is
ignored for both reads and writes to make these probe runs reproducible without
contaminating the paper-run cache.

Outputs (artifacts/probe_frozen_jacobian/):
  - runs.csv          — one row per (n_seg, freeze) with all metrics
  - summary.md        — side-by-side table + headline finding
  - jacobian_drift.png — log-scale Frobenius drift vs SCP iter, faceted by n_seg
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from orbital_docking import constants
from orbital_docking.optimization import (
    optimize_orbital_docking,
    generate_initial_control_points,
)


EARTH_RADIUS_KM = constants.EARTH_RADIUS_KM


def _rotz(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def eci_from_circular(radius_km, inc_deg, raan_deg, u_deg):
    mu = constants.EARTH_MU_SCALED
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    u = np.deg2rad(u_deg)
    r_pqw = np.array([radius_km * np.cos(u), radius_km * np.sin(u), 0.0])
    v_circ = np.sqrt(mu / radius_km)
    v_pqw = v_circ * np.array([-np.sin(u), np.cos(u), 0.0])
    q = _rotz(raan) @ _rotx(inc)
    return q @ r_pqw, q @ v_pqw


# --- Scenario (matches convergence_diagnostic.py) ---

N_DEG = 7
PROGRESS_ALT = 245.0
ISS_ALT = 400.0
INC = 51.64
RAAN = 0.0
ISS_U = 45.0
PROGRESS_LAG = 120.0
KOZ_RADIUS = 6471.0
TRANSFER_TIME = 1500.0

# --- Sweep ---

SEGMENT_COUNTS = [2, 4, 8, 16, 32]
TOL = 1e-6
MAX_ITER = 1000

OUT_DIR = Path("artifacts/probe_frozen_jacobian")


def make_p_init():
    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT
    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - PROGRESS_LAG)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N_DEG, P_start, P_end)
    return P_init, v0, v1


def run_one(n_seg: int, freeze: bool, P_init, v0, v1) -> dict:
    t0 = time.time()
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=n_seg,
        r_e=KOZ_RADIUS,
        max_iter=MAX_ITER,
        tol=TOL,
        v0=v0,
        v1=v1,
        sample_count=100,
        objective_mode="dv",
        scp_prox_weight=1e-6,
        scp_trust_radius=2000.0,
        transfer_time=TRANSFER_TIME,
        freeze_gravity_jacobian=freeze,
        verbose=False,
        use_cache=False,
        ignore_existing_cache=True,
    )
    wall = time.time() - t0

    drift_history = info.get("jacobian_drift_history") or []
    # Convert PyList-of-PyList to numpy 2D array if non-empty.
    drift_array = (
        np.array(drift_history, dtype=float) if drift_history else np.empty((0, 0))
    )

    record = {
        "n_seg": int(n_seg),
        "freeze_gravity_jacobian": bool(freeze),
        "iterations": int(info["iterations"]),
        "max_iter": MAX_ITER,
        "tol": TOL,
        "wall_time_s": float(wall),
        "elapsed_time_s": float(info.get("elapsed_time", wall)),
        "feasible": bool(info["feasible"]),
        "min_radius_km": float(info["min_radius"]),
        "safety_margin_km": float(info["min_radius"] - KOZ_RADIUS),
        "cost_true_energy": float(info["cost_true_energy"]),
        "dv_proxy_m_s": float(info["dv_proxy_m_s"]),
        "max_control_accel_ms2": float(info["max_control_accel_ms2"]),
        "mean_control_accel_ms2": float(info["mean_control_accel_ms2"]),
        "final_delta_norm": float(info.get("final_delta_norm", float("nan"))),
        "max_koz_slack": float(info.get("max_koz_slack", 0.0)),
        "total_koz_slack": float(info.get("total_koz_slack", 0.0)),
        "koz_linear_max_violation": float(info.get("koz_linear_max_violation", 0.0)),
        "termination_reason": str(info.get("termination_reason", "")),
        # Drift summary (NaN when frozen=True since no per-iter recompute).
        "jacobian_drift_max": float(info.get("jacobian_drift_max", float("nan"))),
        "jacobian_drift_last_iter_max": float(
            info.get("jacobian_drift_last_iter_max", float("nan"))
        ),
        # Median over (iters, segs) when available.
        "jacobian_drift_median": (
            float(np.median(drift_array)) if drift_array.size > 0 else float("nan")
        ),
        # Where the drift trace lives — shape (iters, segs).
        "_drift_array": drift_array,
    }
    return record


def write_runs_csv(records, path: Path):
    fieldnames = [
        k for k in records[0].keys() if not k.startswith("_")
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fieldnames})


def write_summary_md(records, path: Path):
    rows_by_nseg = {}
    for r in records:
        rows_by_nseg.setdefault(int(r["n_seg"]), {})[bool(r["freeze_gravity_jacobian"])] = r

    lines = []
    lines.append("# Frozen-gravity-Jacobian probe — summary")
    lines.append("")
    lines.append(
        f"Scenario: N={N_DEG}, 120° phase lag, T={TRANSFER_TIME} s, "
        f"r_e={KOZ_RADIUS} km, objective=dv, prox=1e-6, trust=2000 km. "
        f"tol={TOL}, max_iter={MAX_ITER}, cache ignored."
    )
    lines.append("")
    lines.append("## Side-by-side: frozen vs unfrozen")
    lines.append("")
    lines.append(
        "| n_seg | mode | iters | dv (m/s) | min_r (km) | wall (s) | "
        "feasible | drift_max | drift_last_iter_max |"
    )
    lines.append(
        "|-------|------|-------|----------|------------|----------|----------|-----------|---------------------|"
    )

    def fmt_drift(v):
        if v != v:  # NaN
            return "—"
        return f"{v:.3e}"

    for n_seg in sorted(rows_by_nseg.keys()):
        for freeze in (False, True):
            r = rows_by_nseg[n_seg].get(freeze)
            if r is None:
                continue
            mode = "frozen" if freeze else "unfrozen"
            lines.append(
                f"| {n_seg} | {mode} | {r['iterations']:>4d} | "
                f"{r['dv_proxy_m_s']:.3f} | {r['min_radius_km']:.3f} | "
                f"{r['wall_time_s']:.2f} | {r['feasible']} | "
                f"{fmt_drift(r['jacobian_drift_max'])} | "
                f"{fmt_drift(r['jacobian_drift_last_iter_max'])} |"
            )

    lines.append("")
    lines.append("## Deltas (frozen − unfrozen, % of unfrozen)")
    lines.append("")
    lines.append("| n_seg | Δdv (%) | Δmin_r (km) | Δiter | Δwall (s) |")
    lines.append("|-------|---------|-------------|-------|-----------|")
    for n_seg in sorted(rows_by_nseg.keys()):
        u = rows_by_nseg[n_seg].get(False)
        f_ = rows_by_nseg[n_seg].get(True)
        if u is None or f_ is None:
            continue
        dv_pct = (f_["dv_proxy_m_s"] - u["dv_proxy_m_s"]) / max(
            abs(u["dv_proxy_m_s"]), 1e-12
        ) * 100.0
        d_min_r = f_["min_radius_km"] - u["min_radius_km"]
        d_iter = f_["iterations"] - u["iterations"]
        d_wall = f_["wall_time_s"] - u["wall_time_s"]
        lines.append(
            f"| {n_seg} | {dv_pct:+.4f} | {d_min_r:+.3f} | {d_iter:+d} | {d_wall:+.2f} |"
        )

    lines.append("")
    lines.append("## Notes on interpretation")
    lines.append("")
    lines.append(
        "- `drift_max` is the maximum over (iters × segments) of "
        "‖J_i_t − J_i_1‖_F for the unfrozen run. Compares against the gravity "
        "Jacobian Frobenius norm itself, which for two-body+J2 at LEO altitude "
        "is on the order of √6·μ/r³ ≈ 3.3e-6 (km/s²/km)."
    )
    lines.append(
        "- `drift_last_iter_max` is the drift at the final iteration — closest "
        "to what the freeze approximation gives up at convergence."
    )
    lines.append(
        "- Frozen mode is identical-by-construction at iter 1, so the drift "
        "trace is empty and reported as `—`."
    )

    path.write_text("\n".join(lines) + "\n")


def write_drift_plot(records, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unfrozen = [r for r in records if not r["freeze_gravity_jacobian"]]
    unfrozen.sort(key=lambda r: r["n_seg"])

    n_panels = len(unfrozen)
    cols = min(3, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)

    for i, r in enumerate(unfrozen):
        ax = axes[i // cols][i % cols]
        drift = r["_drift_array"]
        if drift.size == 0:
            ax.set_title(f"n_seg={r['n_seg']} (no data)")
            ax.set_xlabel("SCP iter")
            ax.set_ylabel("‖J_i_t − J_i_1‖_F")
            continue

        n_iters, n_segs = drift.shape
        x = np.arange(1, n_iters + 1)
        for s in range(n_segs):
            ax.plot(x, np.maximum(drift[:, s], 1e-30), alpha=0.6, linewidth=0.8)
        ax.plot(
            x,
            np.maximum(drift.max(axis=1), 1e-30),
            color="black",
            linewidth=1.5,
            label="max across segs",
        )
        ax.set_yscale("log")
        ax.set_xlabel("SCP iter")
        ax.set_ylabel("‖J_i_t − J_i_1‖_F")
        ax.set_title(f"n_seg={r['n_seg']} (iters={r['iterations']})")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Hide unused panels
    for j in range(n_panels, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(
        "Per-segment gravity-Jacobian Frobenius drift (unfrozen runs)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    P_init, v0, v1 = make_p_init()

    records = []
    print(
        f"{'n_seg':>6} {'mode':>10} {'iters':>6} {'dv (m/s)':>12} "
        f"{'min_r (km)':>12} {'wall (s)':>10} {'drift_max':>12} {'feasible':>10}"
    )
    print("-" * 92)
    for n_seg in SEGMENT_COUNTS:
        for freeze in (False, True):
            r = run_one(n_seg, freeze, P_init, v0, v1)
            records.append(r)
            mode = "frozen" if freeze else "unfrozen"
            drift_str = (
                f"{r['jacobian_drift_max']:.3e}"
                if r["jacobian_drift_max"] == r["jacobian_drift_max"]
                else "—"
            )
            print(
                f"{n_seg:>6d} {mode:>10s} {r['iterations']:>6d} "
                f"{r['dv_proxy_m_s']:>12.3f} {r['min_radius_km']:>12.3f} "
                f"{r['wall_time_s']:>10.2f} {drift_str:>12s} {str(r['feasible']):>10s}"
            )

    write_runs_csv(records, OUT_DIR / "runs.csv")
    write_summary_md(records, OUT_DIR / "summary.md")
    write_drift_plot(records, OUT_DIR / "jacobian_drift.png")

    # Also drop a JSON for downstream tooling (drift array converted to nested list).
    json_records = []
    for r in records:
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec["drift_history"] = r["_drift_array"].tolist()
        json_records.append(rec)
    (OUT_DIR / "runs.json").write_text(json.dumps(json_records, indent=2))

    print()
    print(f"Wrote: {OUT_DIR/'runs.csv'}")
    print(f"Wrote: {OUT_DIR/'summary.md'}")
    print(f"Wrote: {OUT_DIR/'jacobian_drift.png'}")
    print(f"Wrote: {OUT_DIR/'runs.json'}")


if __name__ == "__main__":
    main()
