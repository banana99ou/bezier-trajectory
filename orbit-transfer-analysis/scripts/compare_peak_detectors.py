"""피크 탐지 방법론 비교 벤치마크.

4가지 방법을 동일 추력 프로파일에 적용하여 비교:
  1. find_peaks (현재 구현, baseline)
  2. Topological Persistence
  3. Savitzky-Golay 미분 영교차
  4. CWT (find_peaks_cwt)
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter
from scipy.ndimage import uniform_filter1d

from orbit_transfer.types import TransferConfig
from orbit_transfer.optimizer.two_pass import TwoPassOptimizer
from orbit_transfer.config import (
    PEAK_PROMINENCE_RATIO, PEAK_MIN_DISTANCE_RATIO, PEAK_INTERP_POINTS,
)


# ================================================================
# Method 1: Current find_peaks (baseline)
# ================================================================
def method_current(t, u_mag, T):
    """현재 구현 (find_peaks + CubicSpline + smoothing)."""
    t = np.asarray(t, dtype=float)
    u_mag = np.asarray(u_mag, dtype=float)
    if len(t) < 2:
        return 0, np.array([])

    sort_idx = np.argsort(t)
    t_s, u_s = t[sort_idx], u_mag[sort_idx]
    _, uniq = np.unique(t_s, return_index=True)
    t_u, u_u = t_s[uniq], u_s[uniq]

    if len(t_u) >= 4 and len(t_u) < PEAK_INTERP_POINTS:
        cs = CubicSpline(t_u, u_u)
        t_w = np.linspace(t_u[0], t_u[-1], PEAK_INTERP_POINTS)
        u_w = np.maximum(cs(t_w), 0.0)
    else:
        t_w, u_w = t_u, u_u

    t_span = t_w[-1] - t_w[0]
    min_dist = max(int(len(t_w) * PEAK_MIN_DISTANCE_RATIO * (T / t_span)), 1) if t_span > 0 else 1
    smooth_win = max(2 * (min_dist // 4) + 1, 5)
    u_smooth = uniform_filter1d(u_w, size=smooth_win)

    u_max = np.max(u_smooth)
    if u_max == 0:
        return 0, np.array([])
    prom_thr = PEAK_PROMINENCE_RATIO * u_max

    peaks, _ = find_peaks(u_smooth, prominence=prom_thr, distance=min_dist)

    # 경계 피크 (간략 버전)
    N = len(u_smooth)
    margin = max(min(min_dist // 2, N // 10), 3)
    start_idx = int(np.argmax(u_smooth[:margin]))
    if u_smooth[start_idx] > prom_thr and (len(peaks) == 0 or peaks[0] - start_idx > min_dist):
        if u_smooth[min(start_idx + margin, N - 1)] < u_smooth[start_idx]:
            peaks = np.concatenate([[start_idx], peaks])
    end_region = u_smooth[N - margin:]
    end_local = int(np.argmax(end_region))
    end_idx = N - margin + end_local
    if u_smooth[end_idx] > prom_thr and (len(peaks) == 0 or end_idx - peaks[-1] > min_dist):
        if u_smooth[max(end_idx - margin, 0)] < u_smooth[end_idx]:
            peaks = np.append(peaks, end_idx)

    return len(peaks), t_w[peaks]


# ================================================================
# Method 2: Topological Persistence
# ================================================================
class _Peak:
    """위상적 지속성 피크."""
    def __init__(self, born_idx):
        self.born = self.left = self.right = born_idx
        self.died = None

    def persistence(self, seq):
        if self.died is None:
            return float('inf')
        return seq[self.born] - seq[self.died]


def _persistent_homology(seq):
    """0차 위상적 지속성 계산. O(n log n)."""
    n = len(seq)
    peaks = []
    idx_to_peak = [None] * n
    # 값이 큰 순서대로 처리
    indices = sorted(range(n), key=lambda i: seq[i], reverse=True)

    for idx in indices:
        lft = idx_to_peak[idx - 1] if idx > 0 and idx_to_peak[idx - 1] is not None else None
        rgt = idx_to_peak[idx + 1] if idx < n - 1 and idx_to_peak[idx + 1] is not None else None

        if lft is None and rgt is None:
            peaks.append(_Peak(idx))
            idx_to_peak[idx] = len(peaks) - 1
        elif lft is not None and rgt is None:
            peaks[lft].right += 1
            idx_to_peak[idx] = lft
        elif lft is None and rgt is not None:
            peaks[rgt].left -= 1
            idx_to_peak[idx] = rgt
        else:  # both neighbors have peaks
            if seq[peaks[lft].born] > seq[peaks[rgt].born]:
                peaks[rgt].died = idx
                peaks[lft].right = peaks[rgt].right
                idx_to_peak[peaks[lft].right] = idx_to_peak[idx] = lft
            else:
                peaks[lft].died = idx
                peaks[rgt].left = peaks[lft].left
                idx_to_peak[peaks[rgt].left] = idx_to_peak[idx] = rgt

    return sorted(peaks, key=lambda p: p.persistence(seq), reverse=True)


def method_persistence(t, u_mag, T, persistence_ratio=0.10, interp_points=200):
    """위상적 지속성 기반 피크 탐지."""
    t = np.asarray(t, dtype=float)
    u_mag = np.asarray(u_mag, dtype=float)
    if len(t) < 2:
        return 0, np.array([])

    sort_idx = np.argsort(t)
    t_s, u_s = t[sort_idx], u_mag[sort_idx]
    _, uniq = np.unique(t_s, return_index=True)
    t_u, u_u = t_s[uniq], u_s[uniq]

    # 보간
    if len(t_u) >= 4 and len(t_u) < interp_points:
        cs = CubicSpline(t_u, u_u)
        t_w = np.linspace(t_u[0], t_u[-1], interp_points)
        u_w = np.maximum(cs(t_w), 0.0)
    else:
        t_w, u_w = t_u.copy(), u_u.copy()

    if np.max(u_w) == 0:
        return 0, np.array([])

    peaks = _persistent_homology(u_w)
    threshold = persistence_ratio * np.max(u_w)

    significant = [p for p in peaks if p.persistence(u_w) >= threshold]
    n_peaks = len(significant)
    peak_times = np.array([t_w[p.born] for p in significant])
    sort_t = np.argsort(peak_times)

    return n_peaks, peak_times[sort_t]


# ================================================================
# Method 3: Savitzky-Golay 미분 영교차
# ================================================================
def method_savgol(t, u_mag, T, interp_points=200, prominence_ratio=0.10):
    """S-G 1차 미분의 양→음 영교차로 피크 탐지."""
    t = np.asarray(t, dtype=float)
    u_mag = np.asarray(u_mag, dtype=float)
    if len(t) < 2:
        return 0, np.array([])

    sort_idx = np.argsort(t)
    t_s, u_s = t[sort_idx], u_mag[sort_idx]
    _, uniq = np.unique(t_s, return_index=True)
    t_u, u_u = t_s[uniq], u_s[uniq]

    # 보간
    if len(t_u) >= 4 and len(t_u) < interp_points:
        cs = CubicSpline(t_u, u_u)
        t_w = np.linspace(t_u[0], t_u[-1], interp_points)
        u_w = np.maximum(cs(t_w), 0.0)
    else:
        t_w, u_w = t_u.copy(), u_u.copy()

    if len(u_w) < 7:
        return 0, np.array([])

    # S-G 1차 미분 (window=11 또는 노드의 1/10, polyorder=3)
    win = min(max(len(u_w) // 10 | 1, 7), len(u_w))
    if win % 2 == 0:
        win += 1
    deriv = savgol_filter(u_w, window_length=win, polyorder=3, deriv=1)

    # 양 → 음 영교차점 = 피크
    zero_crossings = []
    for i in range(len(deriv) - 1):
        if deriv[i] >= 0 and deriv[i + 1] < 0:
            zero_crossings.append(i)

    if not zero_crossings:
        return 0, np.array([])

    # prominence 필터
    u_max = np.max(u_w)
    threshold = prominence_ratio * u_max
    peak_indices = [zc for zc in zero_crossings if u_w[zc] >= threshold]

    # 최소 거리 필터
    t_span = t_w[-1] - t_w[0]
    min_dist_idx = max(int(len(t_w) * 0.05), 3)
    filtered = []
    for idx in peak_indices:
        if not filtered or idx - filtered[-1] >= min_dist_idx:
            filtered.append(idx)

    return len(filtered), t_w[np.array(filtered)] if filtered else np.array([])


# ================================================================
# Method 4: CWT (find_peaks_cwt)
# ================================================================
def method_cwt(t, u_mag, T, interp_points=200, prominence_ratio=0.10):
    """연속 웨이블릿 변환 기반 피크 탐지."""
    t = np.asarray(t, dtype=float)
    u_mag = np.asarray(u_mag, dtype=float)
    if len(t) < 2:
        return 0, np.array([])

    sort_idx = np.argsort(t)
    t_s, u_s = t[sort_idx], u_mag[sort_idx]
    _, uniq = np.unique(t_s, return_index=True)
    t_u, u_u = t_s[uniq], u_s[uniq]

    # 보간
    if len(t_u) >= 4 and len(t_u) < interp_points:
        cs = CubicSpline(t_u, u_u)
        t_w = np.linspace(t_u[0], t_u[-1], interp_points)
        u_w = np.maximum(cs(t_w), 0.0)
    else:
        t_w, u_w = t_u.copy(), u_u.copy()

    if np.max(u_w) == 0 or len(u_w) < 10:
        return 0, np.array([])

    # 탐색 폭 범위: 데이터 길이의 5%~30%
    widths = np.arange(max(len(u_w) // 20, 2), len(u_w) // 3)
    if len(widths) < 2:
        return 0, np.array([])

    try:
        peaks = find_peaks_cwt(u_w, widths)
    except Exception:
        return 0, np.array([])

    # prominence 필터
    threshold = prominence_ratio * np.max(u_w)
    peaks = np.array([p for p in peaks if u_w[p] >= threshold])

    return len(peaks), t_w[peaks] if len(peaks) > 0 else np.array([])


# ================================================================
# 비교 벤치마크
# ================================================================
def main():
    configs = [
        ("A: da=400,di=8,T/T0=0.7",   TransferConfig(h0=400, delta_a=400, delta_i=8, T_max_normed=0.7)),
        ("B: da=500,di=10,T/T0=1.0",  TransferConfig(h0=400, delta_a=500, delta_i=10, T_max_normed=1.0)),
        ("C: da=50,di=10,T/T0=0.3",   TransferConfig(h0=400, delta_a=50, delta_i=10, T_max_normed=0.3)),
        ("D: da=300,di=2,T/T0=0.5",   TransferConfig(h0=400, delta_a=300, delta_i=2, T_max_normed=0.5)),
        ("E: da=100,di=5,T/T0=0.3",   TransferConfig(h0=400, delta_a=100, delta_i=5, T_max_normed=0.3)),
        ("F: da=1000,di=15,T/T0=1.2", TransferConfig(h0=400, delta_a=1000, delta_i=15, T_max_normed=1.2)),
        ("G: da=200,di=3,T/T0=0.4",   TransferConfig(h0=400, delta_a=200, delta_i=3, T_max_normed=0.4)),
        ("H: da=80,di=8,T/T0=0.4",    TransferConfig(h0=400, delta_a=80, delta_i=8, T_max_normed=0.4)),
    ]

    methods = {
        'Current\n(find_peaks)': method_current,
        'Topological\nPersistence': method_persistence,
        'Savitzky-Golay\nZero-Cross': method_savgol,
        'CWT': method_cwt,
    }

    n_cases = len(configs)
    n_methods = len(methods)

    fig, axes = plt.subplots(n_cases, n_methods + 1, figsize=(4 * (n_methods + 1), 3 * n_cases))

    results_table = []

    for row, (label, cfg) in enumerate(configs):
        print(f"[{row}] Solving {label}...", flush=True)
        opt = TwoPassOptimizer(cfg)
        res = opt.solve()
        t, u = res.t, np.linalg.norm(res.u, axis=0)
        T = res.T_f

        # Column 0: Raw data
        ax = axes[row, 0]
        ax.plot(t / 3600, u, 'b.-', markersize=3, linewidth=0.7)
        ax.set_title(f'{label}\n({len(t)} pts)', fontsize=8)
        if row == 0:
            ax.set_title(f'Raw Data\n{label}\n({len(t)} pts)', fontsize=8)
        ax.set_ylabel(r'$\|u\|$', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        row_result = {'case': label, 'n_pts': len(t)}

        for col, (mname, mfunc) in enumerate(methods.items(), start=1):
            n_pk, pk_times = mfunc(t, u, T)
            row_result[mname.replace('\n', ' ')] = n_pk

            # 보간하여 시각화
            sort_idx = np.argsort(t)
            t_s, u_s = t[sort_idx], u[sort_idx]
            _, uniq = np.unique(t_s, return_index=True)
            t_u, u_u = t_s[uniq], u_s[uniq]
            if len(t_u) >= 4:
                cs = CubicSpline(t_u, u_u)
                t_viz = np.linspace(t_u[0], t_u[-1], 200)
                u_viz = np.maximum(cs(t_viz), 0.0)
            else:
                t_viz, u_viz = t_u, u_u

            ax = axes[row, col]
            ax.plot(t_viz / 3600, u_viz, 'k-', linewidth=0.5, alpha=0.5)
            ax.plot(t / 3600, u, 'b.', markersize=2, alpha=0.5)
            if len(pk_times) > 0:
                ax.plot(pk_times / 3600, np.interp(pk_times, t_u, u_u), 'rv',
                        markersize=8, label=f'{n_pk} peaks')
                ax.legend(fontsize=7, loc='upper right')
            if row == 0:
                ax.set_title(f'{mname}\n{n_pk} peaks', fontsize=8)
            else:
                ax.set_title(f'{n_pk} peaks', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

        if row == n_cases - 1:
            for c in range(n_methods + 1):
                axes[row, c].set_xlabel('Time [hr]', fontsize=8)

        results_table.append(row_result)

    fig.suptitle('Peak Detection Method Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = 'scripts/peak_detection_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")

    # 결과 테이블 출력
    print("\n" + "="*80)
    print("PEAK COUNT COMPARISON")
    print("="*80)
    header = f"{'Case':<35} {'pts':>4}"
    for mname in methods:
        header += f" {mname.replace(chr(10),' '):>15}"
    print(header)
    print("-" * 80)
    for r in results_table:
        line = f"{r['case']:<35} {r['n_pts']:>4}"
        for mname in methods:
            key = mname.replace('\n', ' ')
            line += f" {r[key]:>15}"
        print(line)

    # 일치도 분석
    print("\n" + "="*80)
    print("AGREEMENT ANALYSIS")
    print("="*80)
    method_names = [m.replace('\n', ' ') for m in methods]
    total = len(results_table)
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            if j <= i:
                continue
            agree = sum(1 for r in results_table if r[m1] == r[m2])
            print(f"  {m1} vs {m2}: {agree}/{total} ({100*agree/total:.0f}%)")


if __name__ == '__main__':
    main()
