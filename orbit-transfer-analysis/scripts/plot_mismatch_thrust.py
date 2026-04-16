"""불일치 케이스 추력 프로파일 비교.

comparison_runs에서 BLADE/Colloc 분류가 다른 케이스를 골라
두 솔버의 추력 프로파일을 나란히 시각화한다.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb

from orbit_transfer.types import TransferConfig
from orbit_transfer.benchmark import TransferBenchmark

DB_PATH = os.path.expanduser(
    "~/fmcl-database-student/student-kit/downloads/EXP-20260403-001/merged.duckdb"
)
OUT_DIR = "results/mismatch_thrust"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 불일치 케이스 선택 ────────────────────────────────────────────
CASES = {
    "blade_bi_colloc_uni":  dict(blade=1, colloc=0),   # BLADE=bimodal, Colloc=unimodal
    "blade_uni_colloc_bi":  dict(blade=0, colloc=1),   # BLADE=unimodal, Colloc=bimodal
    "blade_bi_colloc_multi": dict(blade=1, colloc=2),  # BLADE=bimodal, Colloc=multimodal
}

def fetch_case(con, blade_class, colloc_class):
    row = con.execute("""
        SELECT h0, delta_a, delta_i, T_max_normed, e0, ef,
               blade_profile_class, colloc_profile_class,
               blade_cost, colloc_cost, blade_n_peaks, colloc_n_peaks
        FROM comparison_runs
        WHERE blade_converged=true AND colloc_converged=true
          AND blade_profile_class=? AND colloc_profile_class=?
        LIMIT 1
    """, [blade_class, colloc_class]).df()
    if row.empty:
        return None
    return row.iloc[0]

def run_and_plot(cfg_row, label):
    config = TransferConfig(
        h0=cfg_row["h0"],
        delta_a=cfg_row["delta_a"],
        delta_i=cfg_row["delta_i"],
        T_max_normed=cfg_row["T_max_normed"],
        e0=cfg_row["e0"],
        ef=cfg_row["ef"],
    )
    print(f"\n[{label}]")
    print(f"  h0={config.h0:.0f}km, Δa={config.delta_a:.0f}km, "
          f"Δi={config.delta_i:.2f}°, T={config.T_max_normed:.3f}")
    print(f"  BLADE class={cfg_row['blade_profile_class']}(peaks={cfg_row['blade_n_peaks']}), "
          f"Colloc class={cfg_row['colloc_profile_class']}(peaks={cfg_row['colloc_n_peaks']})")
    print(f"  cost: BLADE={cfg_row['blade_cost']:.6f}, Colloc={cfg_row['colloc_cost']:.6f}")

    bench = TransferBenchmark(config)

    print("  Collocation 실행...", flush=True)
    bench.run_collocation()

    print("  BLADE 실행...", flush=True)
    bench.run_blade(K=12, n=2, max_iter=50, tol_bc=1e-3,
                    relax_alpha=0.3, trust_region=5.0,
                    n_dense=30, ta_free=True)

    # ── 플롯 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    CLASS_NAMES = {0: "Unimodal", 1: "Bimodal", 2: "Multimodal"}

    for ax, method in zip(axes, ["collocation", "blade"]):
        res = bench.results.get(method)
        if res is None:
            ax.set_title(f"{method} — no result")
            continue

        t_min = res.t / 60.0
        u_mag = np.linalg.norm(res.u, axis=0)
        ax.plot(t_min, u_mag * 1e3, "k-", lw=1.2)
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("||u|| [mm/s²]")
        ax.grid(True, alpha=0.3)

        n_peaks = res.metrics.get("n_peaks", "?")
        cls = res.metrics.get("profile_class", "?")
        cls_name = CLASS_NAMES.get(cls, str(cls))
        cost = res.metrics.get("cost_l1", float("nan"))
        ax.set_title(f"{method.upper()}\n{cls_name} (peaks={n_peaks}, cost={cost:.5f})")

    fig.suptitle(
        f"h0={config.h0:.0f}km  Δa={config.delta_a:.0f}km  "
        f"Δi={config.delta_i:.2f}°  T={config.T_max_normed:.3f}",
        fontsize=11
    )
    fig.tight_layout()
    out_path = f"{OUT_DIR}/{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {out_path}")

# ── main ─────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
for label, classes in CASES.items():
    row = fetch_case(con, classes["blade"], classes["colloc"])
    if row is None:
        print(f"[{label}] 케이스 없음, 건너뜀")
        continue
    run_and_plot(row, label)
con.close()

print("\n완료!")
