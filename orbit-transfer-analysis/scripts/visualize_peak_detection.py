"""피크 탐지 결과 시각화 스크립트.

합성 데이터(8종) + 실제 궤적 데이터(npz)에 대해 피크 탐지 결과를 시각화한다.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbit_transfer.classification.peak_detection import detect_peaks
from orbit_transfer.classification.classifier import classify_profile


CLASS_NAMES = {0: "Unimodal", 1: "Bimodal", 2: "Multimodal"}


def make_gaussian(t, centers, widths, amplitudes):
    u = np.zeros_like(t)
    for c, w, a in zip(centers, widths, amplitudes):
        u += a * np.exp(-0.5 * ((t - c) / w) ** 2)
    return u


def plot_peak_result(ax, t, u_mag, T, title="", label_extra=""):
    """단일 축에 피크 탐지 결과를 플롯."""
    n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u_mag, T)
    profile_class = classify_profile(n_peaks)
    cls_name = CLASS_NAMES.get(profile_class, f"class {profile_class}")

    t_min, t_max = t.min(), t.max()

    ax.plot(t, u_mag, "b-", lw=1.2, alpha=0.7, label="u_mag (raw)")
    ax.set_title(f"{title}\n{n_peaks} peaks → {cls_name}{label_extra}", fontsize=10)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("||u|| [km/s²]")

    yhi = np.max(u_mag) * 1.05 if np.max(u_mag) > 0 else 1.0

    for i, (pt, pw) in enumerate(zip(peak_times, peak_widths_fwhm)):
        ax.axvline(pt, color="r", ls="--", lw=1.0, alpha=0.8)
        ax.annotate(
            f"P{i+1}\nt={pt:.1f}\nFWHM={pw:.1f}",
            xy=(pt, yhi),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=7,
            color="r",
            va="top",
        )
        # FWHM 범위 표시 (시간 범위 내로 클리핑)
        fwhm_lo = max(pt - pw / 2, t_min)
        fwhm_hi = min(pt + pw / 2, t_max)
        ax.axvspan(fwhm_lo, fwhm_hi, alpha=0.1, color="red")

    ax.set_xlim(t_min - (t_max - t_min) * 0.02, t_max + (t_max - t_min) * 0.02)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    # ─── Part 1: 합성 데이터 8종 ───
    fig1, axes1 = plt.subplots(2, 4, figsize=(20, 9))
    fig1.suptitle("Synthetic Examples — Peak Detection Results", fontsize=13, fontweight="bold")

    T = 1000.0
    t = np.linspace(0, T, 500)

    # (1) Unimodal — 중앙 피크
    u = make_gaussian(t, [T / 2], [T / 8], [1.0])
    plot_peak_result(axes1[0, 0], t, u, T, "① Unimodal (center)")

    # (2) Bimodal — 양쪽 피크
    u = make_gaussian(t, [T / 4, 3 * T / 4], [T / 10, T / 10], [1.0, 0.8])
    plot_peak_result(axes1[0, 1], t, u, T, "② Bimodal")

    # (3) Multimodal — 3봉
    u = make_gaussian(t, [T / 6, T / 2, 5 * T / 6], [T / 12, T / 12, T / 12], [0.8, 1.0, 0.6])
    plot_peak_result(axes1[0, 2], t, u, T, "③ Multimodal (3 peaks)")

    # (4) 경계 피크 — 시작점에서 감소
    u = np.exp(-3.0 * t / T)
    plot_peak_result(axes1[0, 3], t, u, T, "④ Boundary (start decay)")

    # (5) 경계 피크 — 양쪽 + 중앙
    u_c = make_gaussian(t, [T / 2], [T / 10], [1.0])
    u = u_c + 0.8 * np.exp(-5.0 * t / T) + 0.8 * np.exp(-5.0 * (T - t) / T)
    plot_peak_result(axes1[1, 0], t, u, T, "⑤ Both boundaries + center")

    # (6) 끝점 직전 살짝 감소 (near-boundary)
    u = make_gaussian(t, [T / 3], [T / 10], [1.0])
    u += make_gaussian(t, [0.95 * T], [T / 15], [0.8])
    plot_peak_result(axes1[1, 1], t, u, T, "⑥ Near-end peak (slight drop)")

    # (7) 시작점 직후 살짝 증가 후 감소
    u = make_gaussian(t, [0.05 * T], [T / 15], [0.8])
    u += make_gaussian(t, [2 * T / 3], [T / 10], [1.0])
    plot_peak_result(axes1[1, 2], t, u, T, "⑦ Near-start peak (slight rise)")

    # (8) 저해상도 (N=30) 3봉 → 보간
    t_coarse = np.linspace(0, T, 30)
    u_coarse = make_gaussian(
        t_coarse, [T / 6, T / 2, 5 * T / 6], [T / 12, T / 12, T / 12], [0.8, 1.0, 0.6]
    )
    plot_peak_result(axes1[1, 3], t_coarse, u_coarse, T, "⑧ Low-res (N=30)", "\n(interpolated)")

    fig1.tight_layout()
    out1 = "scripts/peak_detection_synthetic.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")

    # ─── Part 2: 실제 궤적 npz 데이터 ───
    npz_dir = Path("data/trajectories")
    npz_files = sorted(npz_dir.glob("traj_*.npz"))
    if not npz_files:
        print("No npz files found, skipping real data plots.")
        return

    # 최대 12개 선택 (균등 간격)
    n_show = min(12, len(npz_files))
    indices = np.linspace(0, len(npz_files) - 1, n_show, dtype=int)
    selected = [npz_files[i] for i in indices]

    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(18, 4.0 * n_rows))
    fig2.suptitle("Real Trajectory Data — Peak Detection Results", fontsize=13, fontweight="bold")

    if n_rows == 1:
        axes2 = axes2.reshape(1, -1)
    axes_flat = axes2.flatten()

    for idx, npz_path in enumerate(selected):
        d = np.load(npz_path)
        t_r = d["t"]
        u_r = d["u"]  # (3, N)
        u_mag_r = np.linalg.norm(u_r, axis=0)
        T_r = t_r[-1] - t_r[0]
        if T_r <= 0:
            T_r = 1.0

        mono = "✓" if np.all(np.diff(t_r) >= 0) else "✗"
        plot_peak_result(
            axes_flat[idx],
            t_r,
            u_mag_r,
            T_r,
            f"{npz_path.name}  (N={len(t_r)}, mono={mono})",
        )

    for idx in range(n_show, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig2.tight_layout()
    out2 = "scripts/peak_detection_real.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")

    # ─── Part 3: 고도별 DB npz ───
    h_dirs = sorted(Path("data").glob("h*/trajectories"))
    if h_dirs:
        all_h_npz = []
        for hd in h_dirs:
            files = sorted(hd.glob("traj_*.npz"))
            if files:
                step = max(1, len(files) // 3)
                for f in files[::step][:3]:
                    all_h_npz.append((hd.parent.name, f))

        if all_h_npz:
            n_h = min(12, len(all_h_npz))
            n_cols_h = 4
            n_rows_h = (n_h + n_cols_h - 1) // n_cols_h
            fig3, axes3 = plt.subplots(n_rows_h, n_cols_h, figsize=(18, 4.0 * n_rows_h))
            fig3.suptitle(
                "Altitude-Specific Trajectories — Peak Detection",
                fontsize=13,
                fontweight="bold",
            )
            if n_rows_h == 1:
                axes3 = axes3.reshape(1, -1)
            axes_h = axes3.flatten()

            for idx, (h_name, npz_path) in enumerate(all_h_npz[:n_h]):
                d = np.load(npz_path)
                t_r = d["t"]
                u_r = d["u"]
                u_mag_r = np.linalg.norm(u_r, axis=0)
                T_r = t_r[-1] - t_r[0]
                if T_r <= 0:
                    T_r = 1.0
                mono = "✓" if np.all(np.diff(t_r) >= 0) else "✗"
                plot_peak_result(
                    axes_h[idx],
                    t_r,
                    u_mag_r,
                    T_r,
                    f"[{h_name}] {npz_path.name} (N={len(t_r)}, mono={mono})",
                )

            for idx in range(n_h, len(axes_h)):
                axes_h[idx].set_visible(False)

            fig3.tight_layout()
            out3 = "scripts/peak_detection_altitude.png"
            fig3.savefig(out3, dpi=150, bbox_inches="tight")
            print(f"Saved: {out3}")

    print("Done. Open the PNG files to inspect.")


if __name__ == "__main__":
    main()
