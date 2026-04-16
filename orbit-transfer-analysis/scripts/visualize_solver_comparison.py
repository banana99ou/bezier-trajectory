"""Collocation vs BLADE-SCP: 추력 프로파일 + 피크 탐지 + 분류 시각화.

BLADE를 먼저 실행(L1 희소 유도 포함)한 후, 그 결과를
Collocation 초기치로 전달하여 같은 로컬 미니마 근방에서 비교한다.

Usage:
    cd orbit-transfer-analysis
    PYTHONPATH=src python scripts/visualize_solver_comparison.py
"""

import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbit_transfer.types import TransferConfig
from orbit_transfer.benchmark import TransferBenchmark
from orbit_transfer.classification.peak_detection import detect_peaks
from orbit_transfer.classification.classifier import classify_profile

# ── 테스트 케이스 ──
CASES = [
    {
        "label": "Case A — Unimodal",
        "config": TransferConfig(h0=400, delta_a=200, delta_i=2.0, T_max_normed=0.5),
        "desc": r"$\Delta a$=200 km, $\Delta i$=2°, $T/T_0$=0.5",
        "l1_lambda": 0.05,
    },
    {
        "label": "Case B — Bimodal",
        "config": TransferConfig(h0=400, delta_a=500, delta_i=5.0, T_max_normed=0.8),
        "desc": r"$\Delta a$=500 km, $\Delta i$=5°, $T/T_0$=0.8",
        "l1_lambda": 0.03,
    },
    {
        "label": "Case C — Low altitude raise",
        "config": TransferConfig(h0=300, delta_a=350, delta_i=3.0, T_max_normed=0.6),
        "desc": r"h0=300 km, $\Delta a$=350 km, $\Delta i$=3°, $T/T_0$=0.6",
        "l1_lambda": 0.03,
    },
    {
        "label": "Case D — High altitude",
        "config": TransferConfig(h0=800, delta_a=800, delta_i=3.0, T_max_normed=0.7),
        "desc": r"h0=800 km, $\Delta a$=800 km, $\Delta i$=3°, $T/T_0$=0.7",
        "l1_lambda": 0.05,
    },
]

CLASS_NAMES = {0: "Unimodal (class 0)", 1: "Bimodal (class 1)", 2: "Multimodal (class 2)"}
CLASS_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#E91E63"}

METHOD_COLORS = {
    "collocation": "#E53935",
    "blade":       "#7B1FA2",
}


def run_solvers(case):
    """BLADE 먼저 → Collocation (BLADE 결과를 초기치로) 실행."""
    config = case["config"]
    bench = TransferBenchmark(config)
    results = {}
    l1_lambda = case.get("l1_lambda", 0.0)

    # 1) BLADE 먼저 (L1 희소 유도 적용)
    blade_r = None
    try:
        t0 = time.time()
        blade_r = bench.run_blade(l1_lambda=l1_lambda)
        elapsed = time.time() - t0
        results["blade"] = blade_r
        n_coast = blade_r.extra.get("blade_n_coasting", 0)
        print(f"  [blade]       conv={blade_r.converged}, "
              f"cost_l2={blade_r.metrics.get('cost_l2', 0):.6f}, "
              f"l1={l1_lambda}, coast_segs={n_coast}, {elapsed:.2f}s")
    except Exception as exc:
        print(f"  [blade] FAILED: {exc}")

    # 2) Collocation (BLADE 결과를 초기치 + 동일 L1)
    try:
        t0 = time.time()
        if blade_r is not None and blade_r.converged:
            r = bench.run_collocation(
                x_init=blade_r.x,
                u_init=blade_r.u,
                t_init=blade_r.t,
                l1_lambda=l1_lambda,
            )
            init_src = "BLADE warm-start"
        else:
            r = bench.run_collocation(l1_lambda=l1_lambda)
            init_src = "default"
        elapsed = time.time() - t0
        results["collocation"] = r
        print(f"  [collocation] conv={r.converged}, "
              f"cost_l2={r.metrics.get('cost_l2', 0):.6f}, "
              f"init={init_src}, {elapsed:.2f}s")
    except Exception as exc:
        print(f"  [collocation] FAILED: {exc}")

    return results


def plot_thrust_with_peaks(ax, t, u, method, color):
    """추력 프로파일 + 피크 탐지 마커를 하나의 축에 그린다."""
    u_mag = np.linalg.norm(u, axis=0)
    T_sec = t[-1] - t[0]
    t_hr = t / 3600.0

    # 피크 탐지
    n_peaks, peak_times, peak_widths = detect_peaks(t, u_mag, T_sec)
    prof_class = classify_profile(n_peaks)
    cls_label = CLASS_NAMES.get(prof_class, f"class {prof_class}")

    # 추력 프로파일 곡선
    scale = 1e3  # km/s² → ×10³
    ax.plot(t_hr, u_mag * scale, color=color, linewidth=1.8,
            label=f"{method}")
    ax.fill_between(t_hr, 0, u_mag * scale, color=color, alpha=0.07)

    # 피크 마커 (▼)
    if n_peaks > 0:
        peak_hr = peak_times / 3600.0
        peak_vals = np.interp(peak_times, t, u_mag) * scale
        ax.scatter(peak_hr, peak_vals, color=color, s=90, zorder=5,
                   edgecolors="white", linewidth=1.5, marker="v")

        # FWHM 범위 밴드
        for pt, pw in zip(peak_times, peak_widths):
            hw = pw / 2.0
            lo = max((pt - hw) / 3600, t_hr[0])
            hi = min((pt + hw) / 3600, t_hr[-1])
            ax.axvspan(lo, hi, color=color, alpha=0.05)

    return n_peaks, prof_class, cls_label


def main():
    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 2, figsize=(16, 4.0 * n_cases),
                             gridspec_kw={"width_ratios": [3, 1]})
    if n_cases == 1:
        axes = axes.reshape(1, -1)

    for i, case in enumerate(CASES):
        print(f"\n{'='*55}")
        print(f" {case['label']}")
        print(f"{'='*55}")

        results = run_solvers(case)

        ax_thrust = axes[i, 0]
        ax_info = axes[i, 1]

        if not results:
            ax_thrust.text(0.5, 0.5, "No results", ha="center", va="center",
                           transform=ax_thrust.transAxes, fontsize=14)
            ax_info.axis("off")
            continue

        # 각 솔버의 추력 프로파일 + 피크 마커
        info_lines = []
        for method, r in results.items():
            color = METHOD_COLORS.get(method, "gray")
            n_pk, p_cls, cls_lbl = plot_thrust_with_peaks(
                ax_thrust, r.t, r.u, method.capitalize(), color)
            cost_l2 = r.metrics.get("cost_l2", float("nan"))
            tof_hr = (r.t[-1] - r.t[0]) / 3600.0
            info_lines.append({
                "method": method.capitalize(),
                "n_peaks": n_pk,
                "class": p_cls,
                "cls_label": cls_lbl,
                "cost_l2": cost_l2,
                "tof_hr": tof_hr,
                "converged": r.converged,
                "color": color,
            })

        # 추력 축 꾸미기
        ax_thrust.set_xlabel("Time [hr]", fontsize=10)
        ax_thrust.set_ylabel(r"$\|\mathbf{u}\| \times 10^3$ [km/s²]", fontsize=10)
        ax_thrust.set_title(f"{case['label']}", fontsize=12, fontweight="bold",
                            loc="left")
        ax_thrust.text(0.99, 0.97, case["desc"], transform=ax_thrust.transAxes,
                       fontsize=8, va="top", ha="right",
                       bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                                 ec="gray", alpha=0.8))
        ax_thrust.legend(fontsize=9, loc="upper left")
        ax_thrust.grid(True, alpha=0.3)
        ax_thrust.set_ylim(bottom=0)

        # 정보 패널 (오른쪽)
        ax_info.axis("off")
        y_pos = 0.92
        ax_info.text(0.05, y_pos, "Detection Summary", fontsize=11,
                     fontweight="bold", transform=ax_info.transAxes, va="top")
        y_pos -= 0.08

        for info in info_lines:
            block = (
                f'{info["method"]}\n'
                f'  Peaks: {info["n_peaks"]}\n'
                f'  {info["cls_label"]}\n'
                f'  Cost L2: {info["cost_l2"]:.6f}\n'
                f'  TOF: {info["tof_hr"]:.3f} hr\n'
                f'  Converged: {"✓" if info["converged"] else "✗"}'
            )
            ax_info.text(0.05, y_pos, block, fontsize=8.5,
                         transform=ax_info.transAxes, va="top",
                         family="monospace",
                         bbox=dict(boxstyle="round,pad=0.4", fc=info["color"],
                                   alpha=0.10, ec=info["color"]))
            y_pos -= 0.45

    fig.suptitle(
        "Thrust Profile Peak Detection & Classification\n"
        "BLADE-SCP (ℓ₁ sparsity) → Collocation (warm-started)  |  ▼ = peak, shaded = FWHM",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = "scripts/solver_peak_comparison.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
