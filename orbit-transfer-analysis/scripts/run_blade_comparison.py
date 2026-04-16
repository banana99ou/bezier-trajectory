"""BLADE vs 기존 솔버 비교 (Phase A PoC).

단일 케이스에서 Hohmann, Lambert, Collocation, BLADE, BLADE-Collocation
5가지 솔버를 동일 조건으로 비교한다.

Usage:
    python scripts/run_blade_comparison.py
"""

from __future__ import annotations
import sys
import os
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbit_transfer.types import TransferConfig
from orbit_transfer.benchmark import TransferBenchmark


def main():
    # ── 테스트 케이스 ──
    config = TransferConfig(
        h0=400.0,
        delta_a=500.0,
        delta_i=0.0,
        T_max_normed=0.7,
        e0=0.0,
        ef=0.0,
    )

    print("=" * 70)
    print("BLADE vs 기존 솔버 비교 (Phase A PoC)")
    print("=" * 70)
    print(f"Config: h0={config.h0}km, Δa={config.delta_a}km, "
          f"Δi={config.delta_i}°, T/T₀={config.T_max_normed}")
    print(f"  a0={config.a0:.1f}km → af={config.af:.1f}km")
    print(f"  T_max={config.T_max:.1f}s, u_max={config.u_max} km/s²")
    print("-" * 70)

    bench = TransferBenchmark(config)

    # ── 1) 기존 솔버 ──
    print("\n[1/5] Hohmann...", flush=True)
    bench.run_hohmann()
    print("  Done.")

    print("[2/5] Lambert...", flush=True)
    bench.run_lambert()
    print("  Done.")

    print("[3/5] Collocation (Two-Pass)...", flush=True)
    res_col = bench.run_collocation()
    col_tf = res_col.extra.get("T_f")
    print(f"  Done. T_f={col_tf:.1f}s, converged={res_col.converged}")

    # ── 2) BLADE — 콜로케이션 T_f에 맞춰 공정 비교 ──
    print("[4/5] BLADE-SCP (ta_free + coupling)...", flush=True)
    blade_kwargs = dict(
        K=12, n=2, max_iter=50, tol_bc=1e-3,
        relax_alpha=0.3, trust_region=5.0,
        n_dense=30,
        ta_free=True,  # 도착 ta를 QP 변수로 편입 (phase_search 대체)
    )
    if col_tf is not None:
        blade_kwargs["t_f_override"] = col_tf
        print(f"  Using collocation T_f = {col_tf:.1f}s")
    res_blade = bench.run_blade(**blade_kwargs)
    print(f"  Done. bc_viol={res_blade.extra['blade_bc_viol']:.4f}, "
          f"status={res_blade.extra['blade_status']}")

    # ── 2b) BLADE-Collocation 순차 최적화 ──
    print("[5/5] BLADE-Collocation (sequential)...", flush=True)
    res_bc = bench.run_blade_collocation(
        blade_K=12, blade_n=2,
        blade_l1_lambda=0.0,
        coast_threshold=0.05,
    )
    bc_stats = res_bc.extra
    print(f"  Done. converged={res_bc.converged}, "
          f"blade_n_peaks={bc_stats.get('blade_n_peaks', '?')}, "
          f"topo_n_peaks={bc_stats.get('topo_n_peaks', '?')}, "
          f"match={bc_stats.get('classification_match', '?')}")
    if res_bc.converged:
        print(f"  dynamics_verified={bc_stats.get('dynamics_verified', '?')}, "
              f"max_residual={bc_stats.get('dynamics_max_residual', float('nan')):.2e}, "
              f"n_refinements={bc_stats.get('n_refinements', 0)}")
        print(f"  total_time={bc_stats.get('total_time', float('nan')):.2f}s")

    # ── 3) 결과 출력 ──
    print("\n")
    bench.print_summary()

    # BLADE 고유 정보
    print("\n[BLADE 상세]")
    print(f"  BC violation: {res_blade.extra['blade_bc_viol']:.6f}")
    print(f"  SCP iterations: {res_blade.extra['blade_n_iter']}")
    print(f"  Coasting segments: {res_blade.extra['blade_n_coasting']}/{res_blade.extra['blade_K']}")
    print(f"  Solve time: {res_blade.extra['blade_solve_time']:.2f}s")
    print(f"  Segment norms: {[f'{nm:.3f}' for nm in res_blade.extra['blade_seg_norms']]}")
    if res_blade.extra.get('blade_ta_opt') is not None:
        print(f"  Optimal arrival ta: {res_blade.extra['blade_ta_opt']:.1f}°")

    # ── 4) CSV 저장 ──
    outdir = "results/blade_poc"
    os.makedirs(outdir, exist_ok=True)
    bench.export_csv(f"{outdir}/comparison.csv")
    bench.export_trajectory_csv(outdir)
    print(f"\n결과 저장: {outdir}/")

    # ── 5) 추력 프로파일 비교 그림 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_methods = len(bench.results)
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        axes = np.array(axes).ravel()

        for i, (method, res) in enumerate(bench.results.items()):
            ax = axes[i]
            t_min = res.t / 60.0  # [min]
            u_mag = np.linalg.norm(res.u, axis=0)
            ax.plot(t_min, u_mag * 1e3, "k-", lw=1.2)  # [m/s²]
            ax.set_title(f"{method.upper()}", fontsize=11)
            ax.set_ylabel("||u|| [mm/s²]")
            ax.set_xlabel("Time [min]")
            ax.grid(True, alpha=0.3)

            # 메트릭 표시
            m = res.metrics
            info = f"Δv={m.get('dv_total', 0):.4f} km/s"
            if not res.is_impulsive:
                info += f"\npeaks={m.get('n_peaks', '?')}"
            ax.text(0.98, 0.95, info, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

        # 빈 subplot 숨기기
        for j in range(n_methods, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Thrust Profiles: h0={config.h0}km, Δa={config.delta_a}km, "
                     f"Δi={config.delta_i}°", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = f"{outdir}/thrust_profiles_comparison.png"
        fig.savefig(fig_path, dpi=150)
        print(f"[그림] {fig_path}")
        plt.close(fig)
    except Exception as e:
        print(f"[그림 생성 실패: {e}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
