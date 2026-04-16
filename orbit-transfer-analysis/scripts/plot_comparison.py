#!/usr/bin/env python
"""비교 시각화 스크립트 (논문 스타일).

세 종류의 비교 그림을 생성한다:
  Fig. A  — 시간 제약별 추력 프로파일 비교 (T_max_normed 스윕)
  Fig. B  — Hohmann·Lambert·Collocation 비용 비교 막대 그래프
  Fig. C  — 추력 프로파일 + 궤적 (단일 케이스 4-패널)

사용 예:
    python scripts/plot_comparison.py --outdir results/paper_comparison --figdir manuscript/figures/comparison
    python scripts/plot_comparison.py --case paper_main --figdir results/figs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── 전역 스타일 ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.85,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "lines.linewidth":    1.4,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── 색상 팔레트 (논문과 동일) ───────────────────────────────────────────────
C_HOHMANN    = "#2196F3"   # blue
C_LAMBERT    = "#FF9800"   # orange
C_COLLOC     = "#4CAF50"   # green
C_INFEASIBLE = "#E53935"   # red (Hohmann 불가 표시)
LW = 1.4

METHOD_STYLE = {
    "hohmann":    {"color": C_HOHMANN,  "ls": "-",  "lw": LW,   "label": "Hohmann"},
    "lambert":    {"color": C_LAMBERT,  "ls": "--", "lw": LW,   "label": "Lambert"},
    "collocation":{"color": C_COLLOC,   "ls": "-",  "lw": LW+0.4, "label": "Collocation"},
}


# ===========================================================================
# 데이터 로딩 헬퍼
# ===========================================================================

def _load_results(case_dir: str) -> dict:
    """metrics_summary.csv + *_trajectory.csv 를 읽어 결과 dict 반환."""
    import csv

    results = {}

    summary_path = os.path.join(case_dir, "metrics_summary.csv")
    if not os.path.exists(summary_path):
        return results

    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            results[method] = {
                "converged": row["converged"].lower() == "true",
                "is_impulsive": row["is_impulsive"].lower() == "true",
                "metrics": {k: _safe_float(v) for k, v in row.items()
                            if k not in ("method", "converged", "is_impulsive")},
            }

    # 궤적 데이터
    for method in list(results.keys()):
        traj_path = os.path.join(case_dir, f"{method}_trajectory.csv")
        if not os.path.exists(traj_path):
            continue
        with open(traj_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        keys = list(rows[0].keys())
        data = {k: np.array([float(r[k]) for r in rows]) for k in keys}
        results[method]["traj"] = data

    return results


def _safe_float(v: str) -> float | str:
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


# ===========================================================================
# Fig. A : 시간 제약 스윕 — 추력 프로파일 비교
# ===========================================================================

def fig_time_sweep(
    time_outdir: str,
    figpath: str,
    T_list: list[float] | None = None,
) -> None:
    """T_max_normed 별 콜로케이션 추력 프로파일을 1행으로 비교.

    T_list 기본값: [0.3, 0.4, 0.5, 0.7, 1.1]
    """
    if T_list is None:
        T_list = [0.3, 0.4, 0.5, 0.7, 1.1]

    # T=0.3~1.1 케이스 이름 매핑
    T_name = {T: f"T_{int(T*10):02d}" for T in T_list}

    n = len(T_list)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0), sharey=True)
    if n == 1:
        axes = [axes]

    HOHMANN_TOF_NORM = 0.5  # 순수 경사각 전이 Hohmann TOF_norm

    for ax, T in zip(axes, T_list):
        case_dir = os.path.join(time_outdir, T_name[T])
        res = _load_results(case_dir)

        # Collocation 추력 프로파일
        if "collocation" in res and res["collocation"]["converged"]:
            traj = res["collocation"].get("traj", {})
            if traj:
                t = traj.get("t[s]", np.array([]))
                u_cols = ["ux[km/s2]", "uy[km/s2]", "uz[km/s2]"]
                u_arr = np.column_stack([traj[c] for c in u_cols if c in traj])
                u_mag = np.linalg.norm(u_arr, axis=1) if u_arr.ndim == 2 else np.zeros_like(t)
                tof = t[-1] if len(t) > 0 else 1.0
                t_norm = t / tof if tof > 0 else t
                ax.plot(t_norm, u_mag * 1e3, color=C_COLLOC, lw=LW, zorder=3,
                        label="Collocation")
                ax.fill_between(t_norm, 0, u_mag * 1e3, color=C_COLLOC, alpha=0.15)

        # Hohmann 불가 표시
        if T < HOHMANN_TOF_NORM:
            ax.axvspan(0, 1, alpha=0.06, color=C_INFEASIBLE, zorder=0)
            ax.text(0.97, 0.97, "Hohmann\ninfeasible", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color=C_INFEASIBLE,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_INFEASIBLE,
                               alpha=0.8))
        else:
            # Hohmann 임펄스 위치 표시 (t_norm=0.5에 수직선)
            ax.axvline(HOHMANN_TOF_NORM, color=C_HOHMANN, ls=":", lw=1.0,
                       label="Hohmann burn", zorder=2)

        ax.set_title(f"$T_{{max}}/T_0 = {T:.1f}$", pad=3)
        ax.set_xlabel(r"Normalized time $t/T_f$")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        if ax is axes[0]:
            ax.set_ylabel(r"$\|\mathbf{u}\|$ [$\times 10^{-3}$ km/s$^2$]")

    # 공통 범례
    handles = [
        Line2D([0], [0], color=C_COLLOC, lw=LW, label="Collocation"),
        Line2D([0], [0], color=C_HOHMANN, ls=":", lw=1.0, label="Hohmann burn (norm.)"),
        matplotlib.patches.Patch(facecolor=C_INFEASIBLE, alpha=0.3, label="Hohmann infeasible region"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle(
        r"Thrust profiles: time-constrained transfer ($\Delta i = 22.7°$, $h_0 = 561$ km)",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)
    print(f"[plot] Fig. A saved: {figpath}")


# ===========================================================================
# Fig. B : 비용 비교 막대 그래프
# ===========================================================================

def fig_cost_bar(
    case_dirs: dict[str, str],
    figpath: str,
    metric: str = "cost_l1",
) -> None:
    """여러 케이스의 Hohmann Δv, Lambert Δv, Collocation L1 을 막대 그래프로 비교.

    Parameters
    ----------
    case_dirs : dict  케이스 라벨 → 케이스 디렉토리
    figpath : str
    metric : str  collocation 비교 지표 ('cost_l1' 또는 'cost_l2')
    """
    labels = list(case_dirs.keys())
    n = len(labels)

    hoh_vals   = []
    lamb_vals  = []
    coll_vals  = []
    feasible   = []   # Hohmann 가능 여부

    for lbl in labels:
        res = _load_results(case_dirs[lbl])
        hv  = res.get("hohmann",    {}).get("metrics", {}).get("dv_total", np.nan)
        lv  = res.get("lambert",    {}).get("metrics", {}).get("dv_total", np.nan)
        cv  = res.get("collocation",{}).get("metrics", {}).get(metric, np.nan)
        hof = res.get("hohmann",    {}).get("metrics", {}).get("tof_norm", 0.5)
        hoh_vals.append(float(hv) if hv == hv else np.nan)  # nan check
        lamb_vals.append(float(lv) if lv == lv else np.nan)
        coll_vals.append(float(cv) if cv == cv else np.nan)
        # T_max_normed 를 디렉토리 이름에서 추출 (T_03 → 0.3)
        feasible.append(True)  # 별도 플래그

    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 1.5*n), 4))

    bars_h = ax.bar(x - w, hoh_vals, w, color=C_HOHMANN, alpha=0.85,
                    label=r"Hohmann $\Delta v$", zorder=3)
    bars_l = ax.bar(x,     lamb_vals, w, color=C_LAMBERT, alpha=0.85,
                    label=r"Lambert $\Delta v$", zorder=3)
    bars_c = ax.bar(x + w, coll_vals, w, color=C_COLLOC,  alpha=0.85,
                    label=f"Collocation $L^1$ cost",     zorder=3)

    # 수치 레이블
    for bars in (bars_h, bars_l, bars_c):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"$\Delta v$ or $L^1$ cost [km/s]")
    ax.legend()
    ax.set_title("Comparison: Hohmann / Lambert / Collocation")

    fig.tight_layout()
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)
    print(f"[plot] Fig. B saved: {figpath}")


# ===========================================================================
# Fig. C : 단일 케이스 4-패널 (추력 프로파일 3종 + 3D 궤적)
# ===========================================================================

def _load_impulses(case_dir: str, method: str) -> list[dict]:
    """임펄스 CSV 로드. 없으면 빈 리스트 반환."""
    import csv
    path = os.path.join(case_dir, f"{method}_impulses.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def fig_case_panel(
    case_dir: str,
    figpath: str,
    case_title: str = "",
) -> None:
    """상단: 추력/임펄스 프로파일 (Hohmann·Lambert·Collocation), 하단: 3D 궤적.

    임펄스 기법은 Δv 화살표(stem)로, 연속 추력은 ||u(t)|| 곡선으로 표시.
    Layout:
      [thrust_hohmann] [thrust_lambert] [thrust_collocation]
      [              3D 궤적 비교 (span 3, 더 큰 높이)              ]
    """
    res = _load_results(case_dir)
    if not res:
        print(f"[경고] 결과 없음: {case_dir}")
        return

    fig = plt.figure(figsize=(14, 9.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.9],
                           hspace=0.42, wspace=0.35)

    methods = ["hohmann", "lambert", "collocation"]
    titles  = ["(a) Hohmann", "(b) Lambert", "(c) Collocation"]
    colors  = [C_HOHMANN, C_LAMBERT, C_COLLOC]

    # ── 상단: 추력 프로파일 ─────────────────────────────────────────────────
    for col, (method, title, color) in enumerate(zip(methods, titles, colors)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, pad=3)

        if method not in res:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            continue

        r    = res[method]
        traj = r.get("traj", {})
        m    = r.get("metrics", {})
        tof  = m.get("tof", 1.0)
        if not isinstance(tof, float) or not np.isfinite(tof) or tof <= 0:
            tof = 1.0

        if r.get("is_impulsive", False):
            # ── 임펄스 기법: Δv 벡터를 수직 화살표로 표시 ──────────────────
            impulses = _load_impulses(case_dir, method)
            if impulses:
                t_norm_imp = [imp["t[s]"] / tof for imp in impulses]
                dv_vals    = [imp["dv[km/s]"] for imp in impulses]
                # markerline, stemlines, baseline
                markerline, stemlines, baseline = ax.stem(
                    t_norm_imp, dv_vals,
                    linefmt=f"-",
                    markerfmt="o",
                    basefmt=" ",
                )
                plt.setp(stemlines,  color=color, linewidth=2.0)
                plt.setp(markerline, color=color, markersize=7)
                # 각 임펄스에 Δv 수치 레이블
                for tn, dv in zip(t_norm_imp, dv_vals):
                    ax.annotate(f"  $\\Delta v$={dv:.3f}", xy=(tn, dv),
                                fontsize=7.5, va="bottom", color=color)
                ax.set_ylabel(r"$\Delta v$ [km/s]")
            else:
                ax.text(0.5, 0.5, "Impulse data\nnot found",
                        transform=ax.transAxes, ha="center", fontsize=8)
                ax.set_ylabel(r"$\Delta v$ [km/s]")
        else:
            # ── 연속 추력: ||u(t)|| 곡선 ─────────────────────────────────────
            if traj:
                t    = traj.get("t[s]", np.array([]))
                u_arr = np.column_stack(
                    [traj[c] for c in ["ux[km/s2]","uy[km/s2]","uz[km/s2]"] if c in traj]
                )
                u_mag  = np.linalg.norm(u_arr, axis=1) if u_arr.ndim == 2 else np.zeros_like(t)
                t_norm = t / t[-1] if len(t) > 0 and t[-1] > 0 else t
                ax.plot(t_norm, u_mag * 1e3, color=color, lw=LW)
                ax.fill_between(t_norm, 0, u_mag * 1e3, color=color, alpha=0.18)
            ax.set_ylabel(r"$\|\mathbf{u}\|$ [$\times 10^{-3}$ km/s$^2$]")

        # 비용 정보 텍스트 박스
        dv   = m.get("dv_total", np.nan)
        l1   = m.get("cost_l1",  np.nan)
        tofn = m.get("tof_norm", np.nan)
        cost_str = f"$\\Delta v$={dv:.3f}" if isinstance(dv, float) and np.isfinite(dv) and dv > 0 else ""
        l1_str   = f"$L^1$={l1:.3f}"      if isinstance(l1, float) and np.isfinite(l1) and l1 > 0 else ""
        info = ", ".join(filter(bool, [cost_str, l1_str]))
        if isinstance(tofn, float) and np.isfinite(tofn):
            info += f"\n$T_f/T_0$={tofn:.3f}"
        if info:
            ax.text(0.98, 0.97, info, transform=ax.transAxes,
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

        ax.set_xlabel(r"Normalized time $t/T_f$")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(bottom=0)

    # ── 하단: 3D 궤적 ──────────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[1, :], projection="3d")

    for method, color in zip(methods, colors):
        if method not in res:
            continue
        traj = res[method].get("traj", {})
        if not traj:
            continue
        x_ = traj.get("x[km]", np.array([]))
        y_ = traj.get("y[km]", np.array([]))
        z_ = traj.get("z[km]", np.array([]))
        if len(x_) == 0:
            continue
        lbl = METHOD_STYLE[method]["label"]
        ax3d.plot(x_, y_, z_, color=color, lw=1.4, alpha=0.9, label=lbl)
        ax3d.scatter(x_[0],  y_[0],  z_[0],  color=color, s=40,
                     marker="o", zorder=5, label=f"{lbl} start")
        ax3d.scatter(x_[-1], y_[-1], z_[-1], color=color, s=60,
                     marker="*", zorder=5, label=f"{lbl} end")

    _draw_earth_sphere(ax3d)

    ax3d.set_xlabel("X [km]", labelpad=4)
    ax3d.set_ylabel("Y [km]", labelpad=4)
    ax3d.set_zlabel("Z [km]", labelpad=4)
    ax3d.set_title("(d) Transfer Trajectories (ECI frame)", pad=6)
    # start/end 범례 생략, 기법 색상만 표시
    handles = [Line2D([0],[0], color=c, lw=2, label=METHOD_STYLE[m]["label"])
               for m, c in zip(methods, colors) if m in res]
    ax3d.legend(handles=handles, loc="upper left", fontsize=8)
    ax3d.tick_params(labelsize=7.5)

    # ── 제목 ────────────────────────────────────────────────────────────────
    if case_title:
        fig.suptitle(case_title, fontsize=11, y=1.01)

    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)
    print(f"[plot] Fig. C saved: {figpath}")


def _draw_earth_sphere(ax, R_E: float = 6378.0, alpha: float = 0.15,
                       n: int = 30) -> None:
    """3D 축에 반투명 지구 구체 그리기."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = R_E * np.outer(np.cos(u), np.sin(v))
    ys = R_E * np.outer(np.sin(u), np.sin(v))
    zs = R_E * np.outer(np.ones(n), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#1565C0", alpha=alpha, linewidth=0,
                    antialiased=False)


# ===========================================================================
# Fig. D : T_max_normed 스윕 비용 꺾은선 그래프
# ===========================================================================

def fig_cost_vs_T(
    time_outdir: str,
    figpath: str,
    T_list: list[float] | None = None,
) -> None:
    """T_max_normed 변화에 따른 비용 변화 꺾은선."""
    if T_list is None:
        T_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]
    T_name = {T: f"T_{int(T*10):02d}" for T in T_list}

    hoh_dv  = []
    coll_l1 = []
    coll_l2 = []
    conv    = []

    for T in T_list:
        case_dir = os.path.join(time_outdir, T_name[T])
        res = _load_results(case_dir)
        hv  = res.get("hohmann",    {}).get("metrics", {}).get("dv_total", np.nan)
        cl1 = res.get("collocation",{}).get("metrics", {}).get("cost_l1",  np.nan)
        cl2 = res.get("collocation",{}).get("metrics", {}).get("cost_l2",  np.nan)
        ok  = res.get("collocation",{}).get("converged", False)
        hoh_dv.append(float(hv)  if isinstance(hv, float) and np.isfinite(hv) else np.nan)
        coll_l1.append(float(cl1) if isinstance(cl1, float) and np.isfinite(cl1) else np.nan)
        coll_l2.append(float(cl2) if isinstance(cl2, float) and np.isfinite(cl2) else np.nan)
        conv.append(ok)

    T_arr = np.array(T_list)
    HOHMANN_TOF_NORM = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # 왼쪽: L1 비용 비교
    ax = axes[0]
    ax.axvline(HOHMANN_TOF_NORM, color=C_HOHMANN, ls=":", lw=1.2,
               label=r"Hohmann TOF ($T_h/T_0 = 0.5$)", zorder=2)
    ax.axvspan(T_arr.min(), HOHMANN_TOF_NORM, alpha=0.07, color=C_INFEASIBLE,
               label="Hohmann infeasible")

    ax.plot(T_arr, hoh_dv, "o-", color=C_HOHMANN, lw=LW, ms=5,
            label=r"Hohmann $\Delta v$")
    ax.plot(T_arr, coll_l1, "s-", color=C_COLLOC, lw=LW, ms=5,
            label=r"Collocation $L^1$")

    # 수렴 실패 표시
    for i, (T, ok) in enumerate(zip(T_list, conv)):
        if not ok:
            ax.scatter(T, 0, marker="x", color="red", s=60, zorder=5)

    ax.set_xlabel(r"$T_{max}/T_0$")
    ax.set_ylabel(r"$\Delta v$ or $L^1$ cost [km/s]")
    ax.set_title(r"$L^1$ fuel cost vs. time budget")
    ax.legend(fontsize=7)

    # 오른쪽: L2 비용 (콜로케이션 목적함수)
    ax2 = axes[1]
    ax2.axvline(HOHMANN_TOF_NORM, color=C_HOHMANN, ls=":", lw=1.2)
    ax2.axvspan(T_arr.min(), HOHMANN_TOF_NORM, alpha=0.07, color=C_INFEASIBLE)

    ax2.plot(T_arr, np.array(coll_l2) * 1e3, "s-", color=C_COLLOC, lw=LW, ms=5,
             label=r"Collocation $L^2$ ($\times 10^{-3}$)")

    for i, (T, ok) in enumerate(zip(T_list, conv)):
        if not ok:
            ax2.scatter(T, 0, marker="x", color="red", s=60, zorder=5)

    ax2.set_xlabel(r"$T_{max}/T_0$")
    ax2.set_ylabel(r"$L^2$ cost $[\times 10^{-3}$ km$^2$/s$^3]$")
    ax2.set_title(r"$L^2$ energy cost vs. time budget")
    ax2.legend(fontsize=7)

    fig.suptitle(
        r"Effect of time budget on transfer cost ($\Delta i = 22.7°$, $h_0 = 561$ km)",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)
    print(f"[plot] Fig. D saved: {figpath}")


# ===========================================================================
# Fig. E : 복합 케이스 Hohmann vs Collocation 히트맵
# ===========================================================================

def fig_combined_heatmap(
    combined_outdir: str,
    figpath: str,
) -> None:
    """Δa-Δi 격자에서 Collocation L1 / Hohmann Δv 비율 히트맵."""
    import csv

    agg_path = os.path.join(combined_outdir, "aggregate_results.csv")
    if not os.path.exists(agg_path):
        print(f"[경고] {agg_path} 없음. fig_combined_heatmap 건너뜀")
        return

    da_vals = sorted({200, 500, 1000})
    di_vals = sorted({5.0, 10.0, 15.0})

    hoh_grid   = np.full((len(da_vals), len(di_vals)), np.nan)
    coll_grid  = np.full((len(da_vals), len(di_vals)), np.nan)
    ratio_grid = np.full((len(da_vals), len(di_vals)), np.nan)
    conv_grid  = np.zeros((len(da_vals), len(di_vals)), dtype=bool)

    with open(agg_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            da = float(row["delta_a_km"])
            di = float(row["delta_i_deg"])
            hv = row.get("hohmann_dv_total", "") or ""
            cl1 = row.get("collocation_cost_l1", "") or ""
            cc  = row.get("collocation_converged", "False")
            if da not in da_vals or di not in di_vals:
                continue
            i = da_vals.index(int(da))
            j = di_vals.index(di)
            if hv:
                hoh_grid[i, j] = float(hv)
            if cl1:
                coll_grid[i, j] = float(cl1)
                if float(cl1) > 0 and hv:
                    ratio_grid[i, j] = float(hv) / float(cl1)
            conv_grid[i, j] = cc.lower() == "true"

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    da_labels = [f"Δa={da} km" for da in da_vals]
    di_labels = [f"Δi={di:.0f}°" for di in di_vals]

    datasets = [
        (hoh_grid,   r"Hohmann $\Delta v$ [km/s]",  "Blues",   "%.3f"),
        (coll_grid,  r"Collocation $L^1$ [km/s]",   "Greens",  "%.3f"),
        (ratio_grid, r"$\Delta v_{Hoh}$ / $L^1_{Coll}$", "RdYlGn", "%.3f"),
    ]

    for ax, (data, title, cmap, fmt) in zip(axes, datasets):
        im = ax.imshow(data, cmap=cmap, aspect="auto",
                       vmin=np.nanmin(data), vmax=np.nanmax(data))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(di_vals)))
        ax.set_xticklabels(di_labels, fontsize=8)
        ax.set_yticks(range(len(da_vals)))
        ax.set_yticklabels(da_labels, fontsize=8)
        ax.set_title(title, fontsize=9)

        # 격자 수치 표시
        for i in range(len(da_vals)):
            for j in range(len(di_vals)):
                val = data[i, j]
                if np.isfinite(val):
                    color = "white" if val > np.nanmean(data) * 1.2 else "black"
                    ax.text(j, i, fmt % val, ha="center", va="center",
                            fontsize=8, color=color)
                    # 수렴 실패 표시
                    if not conv_grid[i, j]:
                        ax.text(j, i+0.35, "FAIL", ha="center", va="center",
                                fontsize=6.5, color="red", fontweight="bold")

    fig.suptitle(
        r"Combined $\Delta a$–$\Delta i$ transfer: Hohmann vs Collocation",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)
    print(f"[plot] Fig. E saved: {figpath}")


# ===========================================================================
# Fig. F : Hohmann 불가 vs 가능 — 핵심 비교 (리뷰어 대응용)
# ===========================================================================

def fig_feasibility_comparison(
    time_outdir: str,
    figpath: str,
) -> None:
    """T_max_normed=0.3(불가)와 0.7(가능)의 추력 프로파일·비용 병렬 비교.

    리뷰어에게 제안 기법의 time-constrained 우위를 시각적으로 전달하는 그림.
    """
    cases = [
        ("T_03", 0.3, "Infeasible for Hohmann\n$(T_{max}/T_0 = 0.3 < 0.5)$"),
        ("T_05", 0.5, "Boundary case\n$(T_{max}/T_0 = 0.5)$"),
        ("T_07", 0.7, "Feasible for Hohmann\n$(T_{max}/T_0 = 0.7)$"),
    ]

    fig = plt.figure(figsize=(13, 5.5))
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    HOHMANN_TOF_NORM = 0.5

    for col, (case_name, T, subtitle) in enumerate(cases):
        case_dir = os.path.join(time_outdir, case_name)
        res = _load_results(case_dir)

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col], hspace=0.45, height_ratios=[2, 1]
        )
        ax_thrust = fig.add_subplot(inner[0])
        ax_bar    = fig.add_subplot(inner[1])

        # ─ 추력 프로파일 ───────────────────────────────────────────────
        # 연속 추력(collocation)은 ||u(t)|| 곡선, 임펄스는 Δv stem으로 표시
        # y축 범위를 임펄스/연속 추력에 맞게 두 개의 y축으로 분리
        ax_imp = ax_thrust                          # 임펄스용 (왼쪽 y)
        ax_cont = ax_thrust.twinx()                 # 연속 추력용 (오른쪽 y)

        imp_plotted = False
        for method, color in [("hohmann", C_HOHMANN), ("lambert", C_LAMBERT)]:
            if method not in res:
                continue
            r       = res[method]
            m       = r.get("metrics", {})
            tof_val = m.get("tof", 1.0)
            if not isinstance(tof_val, float) or not np.isfinite(tof_val) or tof_val <= 0:
                tof_val = 1.0
            impulses = _load_impulses(case_dir, method)
            if impulses:
                t_norm_imp = [imp["t[s]"] / tof_val for imp in impulses]
                dv_vals    = [imp["dv[km/s]"] for imp in impulses]
                ml, sl, bl = ax_imp.stem(t_norm_imp, dv_vals,
                                         linefmt="-", markerfmt="o", basefmt=" ")
                plt.setp(sl, color=color, linewidth=2.0)
                plt.setp(ml, color=color, markersize=6)
                for tn, dv in zip(t_norm_imp, dv_vals):
                    ax_imp.annotate(f" {dv:.3f}", xy=(tn, dv),
                                    fontsize=6.5, va="bottom", color=color)
                imp_plotted = True

        if imp_plotted:
            ax_imp.set_ylabel(r"$\Delta v$ [km/s]", fontsize=7.5, color="black")
            ax_imp.tick_params(axis="y", labelsize=7)
        else:
            ax_imp.set_yticks([])

        # collocation 연속 추력
        if "collocation" in res:
            traj = res["collocation"].get("traj", {})
            if traj:
                t = traj.get("t[s]", np.array([]))
                u_arr = np.column_stack(
                    [traj[c] for c in ["ux[km/s2]", "uy[km/s2]", "uz[km/s2]"]
                     if c in traj]
                )
                u_mag = np.linalg.norm(u_arr, axis=1) if u_arr.ndim == 2 else np.zeros_like(t)
                t_n   = t / t[-1] if len(t) > 0 and t[-1] > 0 else t
                ax_cont.plot(t_n, u_mag * 1e3, color=C_COLLOC, lw=LW,
                             label="Collocation")
                ax_cont.fill_between(t_n, 0, u_mag * 1e3, color=C_COLLOC, alpha=0.15)
        ax_cont.set_ylabel(r"$\|\mathbf{u}\|$ [$\times 10^{-3}$ km/s²]",
                           fontsize=7.5, color=C_COLLOC)
        ax_cont.tick_params(axis="y", labelsize=7, labelcolor=C_COLLOC)
        ax_cont.set_ylim(bottom=0)

        # 범례 통합
        from matplotlib.lines import Line2D as _L2D
        legend_handles = [
            _L2D([0],[0], color=C_HOHMANN, lw=2, label="Hohmann"),
            _L2D([0],[0], color=C_LAMBERT, lw=2, label="Lambert"),
            _L2D([0],[0], color=C_COLLOC,  lw=2, label="Collocation"),
        ]
        ax_imp.legend(handles=legend_handles, fontsize=6.5, loc="upper left")

        # Hohmann 불가 표시
        if T < HOHMANN_TOF_NORM:
            ax_imp.axvspan(0, 1, alpha=0.07, color=C_INFEASIBLE, zorder=0)
            ax_imp.text(0.5, 0.97, "Hohmann infeasible",
                        transform=ax_imp.transAxes,
                        ha="center", va="top", fontsize=7.5, color=C_INFEASIBLE,
                        fontweight="bold")
        else:
            ax_imp.axvline(HOHMANN_TOF_NORM, color=C_HOHMANN,
                           ls=":", lw=1.0, alpha=0.8, zorder=2)

        ax_thrust.set_title(f"$T_{{max}}/T_0 = {T:.1f}$\n{subtitle}",
                            fontsize=8.5, pad=3)
        ax_imp.set_xlabel(r"$t/T_f$", fontsize=8)
        ax_imp.set_xlim(-0.05, 1.05)
        ax_imp.set_ylim(bottom=0)

        # ─ 비용 막대 ────────────────────────────────────────────────────
        methods_cost = []
        vals_cost    = []
        colors_cost  = []

        for method, color, key in [
            ("hohmann",    C_HOHMANN, "dv_total"),
            ("lambert",    C_LAMBERT, "dv_total"),
            ("collocation",C_COLLOC,  "cost_l1"),
        ]:
            if method not in res:
                methods_cost.append(method.capitalize())
                vals_cost.append(np.nan)
            else:
                v = res[method].get("metrics", {}).get(key, np.nan)
                methods_cost.append(method.capitalize())
                vals_cost.append(float(v) if isinstance(v, float) and np.isfinite(v) else np.nan)
            colors_cost.append(color)

        x_b = np.arange(len(methods_cost))
        bars = ax_bar.bar(x_b, vals_cost, color=colors_cost, alpha=0.85, width=0.6)
        ax_bar.set_xticks(x_b)
        ax_bar.set_xticklabels(methods_cost, fontsize=7.5)
        ax_bar.set_ylabel("[km/s]", fontsize=7.5)
        ax_bar.set_title("Cost comparison", fontsize=7.5, pad=2)

        for bar, v in zip(bars, vals_cost):
            if np.isfinite(v):
                ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6)

        # Hohmann 불가 케이스의 bar를 빗금 표시
        if T < HOHMANN_TOF_NORM and np.isfinite(vals_cost[0]):
            bars[0].set_hatch("///")
            bars[0].set_edgecolor("red")

    fig.suptitle(
        r"Continuous vs. impulsive transfer: feasibility under time constraint"
        "\n"
        r"($\Delta i = 22.7°$, $h_0 = 561$ km)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)
    print(f"[plot] Fig. F saved: {figpath}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="논문 스타일 비교 시각화 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--outdir",  default="results",
                        help="비교 실행 결과 루트 (기본: results)")
    parser.add_argument("--figdir", default="results/figures",
                        help="그림 저장 디렉토리 (기본: results/figures)")
    parser.add_argument("--fmt",    default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="그림 저장 형식 (기본: pdf)")
    parser.add_argument("--case",   default=None,
                        help="단일 케이스 패널 (예: paper_main)")
    parser.add_argument("--figs",   nargs="+",
                        default=["A", "B", "C", "D", "E", "F"],
                        help="생성할 그림 선택 (A-F)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    fmt = args.fmt

    # 경로 설정
    time_outdir     = os.path.join(args.outdir, "time_full")
    combined_outdir = os.path.join(args.outdir, "combined_full")
    paper_case_dir  = os.path.join(args.outdir, "paper_comparison", "paper_main")

    def fp(name):
        return os.path.join(args.figdir, f"{name}.{fmt}")

    # ── Fig. A: 시간 스윕 추력 프로파일 ──────────────────────────────────
    if "A" in args.figs and os.path.isdir(time_outdir):
        fig_time_sweep(time_outdir, fp("figA_time_sweep_thrust"),
                       T_list=[0.3, 0.4, 0.5, 0.7, 1.1])

    # ── Fig. B: 비용 막대 그래프 (paper_comparison 케이스) ───────────────
    if "B" in args.figs and os.path.isdir(paper_case_dir):
        fig_cost_bar(
            {"paper_main": paper_case_dir},
            fp("figB_cost_bar"),
        )

    # ── Fig. C: 단일 케이스 4-패널 ────────────────────────────────────────
    if "C" in args.figs:
        case_dir = paper_case_dir if args.case is None else \
                   os.path.join(args.outdir, args.case)
        if os.path.isdir(case_dir):
            case_name = os.path.basename(case_dir)
            title = f"Case: {case_name}  " \
                    r"($h_0 = 561$ km, $\Delta i = 22.7°$, $T_{max}/T_0 = 0.7$)"
            fig_case_panel(case_dir, fp("figC_case_panel"), case_title=title)

    # ── Fig. D: 비용 vs T_max 꺾은선 ─────────────────────────────────────
    if "D" in args.figs and os.path.isdir(time_outdir):
        fig_cost_vs_T(time_outdir, fp("figD_cost_vs_T"),
                      T_list=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])

    # ── Fig. E: 복합 케이스 히트맵 ────────────────────────────────────────
    if "E" in args.figs and os.path.isdir(combined_outdir):
        fig_combined_heatmap(combined_outdir, fp("figE_combined_heatmap"))

    # ── Fig. F: 가용성 비교 (리뷰어 대응) ────────────────────────────────
    if "F" in args.figs and os.path.isdir(time_outdir):
        fig_feasibility_comparison(time_outdir, fp("figF_feasibility"))

    print(f"\n완료. 그림 저장 위치: {os.path.abspath(args.figdir)}")


if __name__ == "__main__":
    main()
