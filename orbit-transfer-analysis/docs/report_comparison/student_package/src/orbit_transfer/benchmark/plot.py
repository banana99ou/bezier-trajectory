"""비교 그림 생성 모듈."""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import BenchmarkResult
    from ..types import TransferConfig

# 기법별 기본 색상
METHOD_COLORS = {
    "hohmann": "#1976D2",       # 파랑
    "lambert": "#388E3C",       # 초록
    "collocation": "#F44336",   # 빨강
}
_DEFAULT_COLOR = "#9C27B0"  # 보라 (커스텀 기법)


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, _DEFAULT_COLOR)


def _label(method: str) -> str:
    return {"hohmann": "Hohmann", "lambert": "Lambert",
            "collocation": "Collocation"}.get(method, method)


# ---------------------------------------------------------------------------
# 추력 프로파일 비교
# ---------------------------------------------------------------------------

def plot_thrust_profiles(
    results: dict[str, "BenchmarkResult"],
    config: "TransferConfig | None" = None,
    outpath: str | None = None,
    n_impulse_pts: int = 300,
) -> plt.Figure:
    """추력 크기 프로파일을 나란히 비교한다.

    임펄스 기법은 Δv를 수직 화살표로 표시한다.
    연속 추력 기법은 ||u(t)|| 곡선으로 표시한다.

    Parameters
    ----------
    results : dict  { method_name: BenchmarkResult }
    config : TransferConfig, optional  (시간 정규화에 사용)
    outpath : str, optional  저장 경로 (.pdf, .png 등)
    n_impulse_pts : int  임펄스 가시화용 격자 점 수

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    t_all = [r.tof for r in results.values() if r.tof > 0]
    t_ref = max(t_all) if t_all else 1.0

    for method, res in results.items():
        color = _color(method)
        lbl = _label(method)
        tof = res.tof

        if res.is_impulsive:
            # 코스팅 구간: u=0 선
            t_norm = np.array([0.0, 1.0]) * (tof / t_ref)
            ax.plot(t_norm, [0.0, 0.0], color=color, linewidth=1.5,
                    linestyle="--", alpha=0.6)
            # 임펄스: 화살표
            for imp in res.impulses:
                t_imp = imp["t"] / t_ref
                dv = imp["dv"]
                ax.annotate(
                    "", xy=(t_imp, dv * 1e3), xytext=(t_imp, 0.0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
                )
                ax.text(t_imp, dv * 1e3 * 1.05, f"{dv * 1e3:.2f}",
                        ha="center", va="bottom", fontsize=7, color=color)
        else:
            t_arr = res.t
            u_mag = np.linalg.norm(res.u, axis=0) * 1e3  # → 10⁻³ km/s²
            t_norm = (t_arr - t_arr[0]) / t_ref
            ax.plot(t_norm, u_mag, color=color, linewidth=1.5, label=lbl)
            ax.fill_between(t_norm, 0, u_mag, alpha=0.15, color=color)

    # 범례 (임펄스 기법도 포함)
    handles = []
    for method, res in results.items():
        color = _color(method)
        lbl = _label(method)
        if res.is_impulsive:
            p = mpatches.Patch(color=color, label=f"{lbl} (impulsive)", alpha=0.7)
        else:
            p = mpatches.Patch(color=color, label=lbl)
        handles.append(p)
    ax.legend(handles=handles, fontsize=9)

    # 축 레이블
    if config is not None:
        ax.set_xlabel(f"Time [s]  (T_ref = {t_ref:.0f} s)")
    else:
        ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|\mathbf{u}\|$ [$\times 10^{-3}$ km/s²]  /  $\Delta v$ [×10⁻³ km/s]")
    ax.set_title("Thrust Profile Comparison")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 3D 궤적 비교
# ---------------------------------------------------------------------------

def plot_trajectories(
    results: dict[str, "BenchmarkResult"],
    config: "TransferConfig | None" = None,
    outpath: str | None = None,
) -> plt.Figure:
    """3D ECI 궤적을 겹쳐 그린다.

    Parameters
    ----------
    results : dict  { method_name: BenchmarkResult }
    config : TransferConfig, optional
    outpath : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from ..constants import R_E

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for method, res in results.items():
        if not res.converged or res.x.shape[1] < 2:
            continue
        color = _color(method)
        lbl = _label(method)
        x, y, z = res.x[0], res.x[1], res.x[2]
        ax.plot(x, y, z, color=color, linewidth=1.5, label=lbl)
        ax.scatter([x[0]], [y[0]], [z[0]], color=color, s=40, marker="o", zorder=5)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], color=color, s=40, marker="^", zorder=5)

    # 지구 구
    u_s = np.linspace(0, 2 * np.pi, 30)
    v_s = np.linspace(0, np.pi, 20)
    xs = R_E * np.outer(np.cos(u_s), np.sin(v_s))
    ys = R_E * np.outer(np.sin(u_s), np.sin(v_s))
    zs = R_E * np.outer(np.ones_like(u_s), np.cos(v_s))
    ax.plot_surface(xs, ys, zs, alpha=0.15, color="#2196F3")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("3D ECI Trajectory Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 지표 막대 그래프
# ---------------------------------------------------------------------------

def plot_metrics_bar(
    results: dict[str, "BenchmarkResult"],
    metrics_to_plot: list[str] | None = None,
    outpath: str | None = None,
) -> plt.Figure:
    """주요 비교 지표를 막대 그래프로 나란히 표시한다.

    Parameters
    ----------
    results : dict  { method_name: BenchmarkResult }
    metrics_to_plot : list of str, optional
        None이면 ['dv_total', 'cost_l1', 'tof_norm'] 사용
    outpath : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["dv_total", "cost_l1", "tof_norm"]

    metric_labels = {
        "dv_total": "Total Δv [km/s]",
        "cost_l1": "L1 cost ∫||u||dt [km/s]",
        "cost_l2": "L2 cost ∫||u||²dt [km²/s³]",
        "tof": "ToF [s]",
        "tof_norm": "ToF / T₀",
        "dv1": "Δv₁ [km/s]",
        "dv2": "Δv₂ [km/s]",
    }

    n_metrics = len(metrics_to_plot)
    n_methods = len(results)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    methods = list(results.keys())
    colors = [_color(m) for m in methods]
    x_pos = np.arange(n_methods)

    for ax, key in zip(axes, metrics_to_plot):
        vals = []
        for method in methods:
            v = results[method].metrics.get(key, np.nan)
            vals.append(float(v) if v is not None else np.nan)

        bars = ax.bar(x_pos, vals, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([_label(m) for m in methods], rotation=15, ha="right")
        ax.set_ylabel(metric_labels.get(key, key))
        ax.set_title(metric_labels.get(key, key))
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Method Comparison", fontsize=12, y=1.01)
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, bbox_inches="tight")

    return fig
