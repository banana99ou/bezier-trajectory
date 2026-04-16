"""논문 Figure 일괄 생성.

Usage:
    python scripts/generate_figures.py --db data/trajectories.duckdb --outdir manuscript/figures
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from orbit_transfer.config import H0_SLICES
from orbit_transfer.database.storage import TrajectoryDatabase
from orbit_transfer.database.query import get_statistics
from orbit_transfer.types import TrajectoryResult
from orbit_transfer.collocation.interpolation import dense_output
from orbit_transfer.visualization.thrust_profile import plot_thrust_magnitude
from orbit_transfer.visualization.trajectory_3d import plot_trajectory_3d
from orbit_transfer.visualization.classification_map import plot_classification_2d, plot_classification_3d
from orbit_transfer.visualization.statistics import (
    plot_class_distribution,
    plot_convergence_history,
    plot_ard_lengthscales,
)

DIM_LABELS = [r'$T_{max}/T_0$', r'$\Delta a$ [km]', r'$\Delta i$ [deg]', r'$e_0$', r'$e_f$']
# sorted keys: ['T_max_normed', 'delta_a', 'delta_i', 'e0', 'ef']
CLASS_NAMES = ['Unimodal', 'Bimodal', 'Multimodal']
CLASS_COLORS = ['#2196F3', '#FF9800', '#F44336']

from orbit_transfer.constants import R_E


def _is_trajectory_valid(traj, h0):
    """NPZ 궤적 데이터의 물리적 유효성 검증."""
    t, x, u = traj['t'], traj['x'], traj['u']
    if t.min() < -1.0:
        return False
    r_mag = np.linalg.norm(x[:3, :], axis=0)
    v_mag = np.linalg.norm(x[3:, :], axis=0)
    u_mag = np.linalg.norm(u, axis=0)
    r_expected = R_E + h0
    if r_mag.min() < R_E - 100 or r_mag.max() > r_expected + 3000:
        return False
    if v_mag.max() > 15.0:
        return False
    if u_mag.max() > 1.0:
        return False
    return True


def _select_valid_representative(db, rows):
    """비용 중간값 기준으로 유효한 NPZ를 가진 대표 케이스를 선택."""
    rows.sort(key=lambda r: r['cost'])
    # 중간부터 탐색하여 유효한 첫 케이스 반환
    mid = len(rows) // 2
    for offset in range(len(rows)):
        for idx in [mid + offset, mid - offset - 1]:
            if 0 <= idx < len(rows):
                row = rows[idx]
                try:
                    traj = db.get_trajectory(row['id'])
                    if _is_trajectory_valid(traj, row['h0']):
                        return row, traj
                except Exception:
                    continue
    return None, None


def _load_result_as_trajectory(db, row):
    """DB row + npz -> TrajectoryResult."""
    traj = db.get_trajectory(row['id'])
    return TrajectoryResult(
        converged=row['converged'],
        cost=row['cost'],
        t=traj['t'],
        x=traj['x'],
        u=traj['u'],
        nu0=row['nu0'],
        nuf=row['nuf'],
        n_peaks=row['n_peaks'],
        profile_class=row['profile_class'],
        pass1_cost=row['pass1_cost'],
    )


def fig1_representative_profiles(db, outdir):
    """Fig.1: 대표 추력 프로파일 3종 (unimodal / bimodal / multimodal)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for cls, ax, name, color in zip(range(3), axes, CLASS_NAMES, CLASS_COLORS):
        rows = db.get_results(converged=True, profile_class=cls)
        if not rows:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(f'({chr(97+cls)}) {name}')
            continue
        # 물리적으로 유효한 중간 비용 케이스 선택
        row, traj = _select_valid_representative(db, rows)
        if row is None:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center')
            ax.set_title(f'({chr(97+cls)}) {name}')
            continue
        t, x, u = traj['t'], traj['x'], traj['u']
        if len(t) >= 4:
            t_d, _, u_d = dense_output(t, x, u, 300)
            u_mag = np.linalg.norm(u_d, axis=0)
            T_f = t[-1] - t[0]
            t_norm = (t_d - t[0]) / T_f
        else:
            u_mag = np.linalg.norm(u, axis=0)
            T_f = t[-1] - t[0]
            t_norm = (t - t[0]) / T_f
        ax.plot(t_norm, u_mag * 1e3, color=color, linewidth=1.2)
        ax.fill_between(t_norm, 0, u_mag * 1e3, alpha=0.2, color=color)
        ax.set_xlabel(r'Normalized Time $t/T_f$')
        ax.set_ylabel(r'$\|\mathbf{u}\|$ [$\times 10^{-3}$ km/s$^2$]')
        ax.set_title(f'({chr(97+cls)}) {name} ($n_p$={row["n_peaks"]})')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig1_representative_profiles.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.1 saved: {path}")
    return path


def fig2_representative_trajectories(db, outdir):
    """Fig.2: 대표 3D 궤적 3종."""
    fig = plt.figure(figsize=(14, 4.5))
    for cls, name, color in zip(range(3), CLASS_NAMES, CLASS_COLORS):
        ax = fig.add_subplot(1, 3, cls + 1, projection='3d')
        rows = db.get_results(converged=True, profile_class=cls)
        if not rows:
            ax.set_title(f'({chr(97+cls)}) {name}')
            continue
        # 물리적으로 유효한 중간 비용 케이스 선택
        row, traj = _select_valid_representative(db, rows)
        if row is None:
            ax.set_title(f'({chr(97+cls)}) {name}')
            continue
        result = TrajectoryResult(
            converged=row['converged'], cost=row['cost'],
            t=traj['t'], x=traj['x'], u=traj['u'],
            nu0=row['nu0'], nuf=row['nuf'],
            n_peaks=row['n_peaks'], profile_class=row['profile_class'],
            pass1_cost=row['pass1_cost'],
        )
        plot_trajectory_3d(result, ax=ax, show_earth=True)
        ax.set_title(f'({chr(97+cls)}) {name}')
    fig.tight_layout()
    path = os.path.join(outdir, 'fig2_representative_trajectories.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.2 saved: {path}")
    return path


def fig3_classification_da_di(db, outdir):
    """Fig.3: 분류 지도 (Δa vs Δi), h0별 4장 서브플롯."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, h0 in enumerate(H0_SLICES):
        ax = axes[idx // 2][idx % 2]
        rows = db.get_results(h0=h0, converged=True)
        if not rows:
            ax.set_title(f'$h_0$ = {int(h0)} km (no data)')
            continue
        X = np.array([[r['T_max_normed'], r['delta_a'], r['delta_i'], r['e0'], r['ef']] for r in rows])
        y = np.array([r['profile_class'] for r in rows])
        plot_classification_2d(X, y, dim1=1, dim2=2, dim_labels=DIM_LABELS, ax=ax)
        ax.set_title(f'$h_0$ = {int(h0)} km')
    fig.tight_layout()
    path = os.path.join(outdir, 'fig3_classification_da_di.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.3 saved: {path}")
    return path


def fig4_classification_da_T(db, outdir):
    """Fig.4: 분류 지도 (Δa vs T_max/T₀), h0별 4장 서브플롯."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, h0 in enumerate(H0_SLICES):
        ax = axes[idx // 2][idx % 2]
        rows = db.get_results(h0=h0, converged=True)
        if not rows:
            ax.set_title(f'$h_0$ = {int(h0)} km (no data)')
            continue
        X = np.array([[r['T_max_normed'], r['delta_a'], r['delta_i'], r['e0'], r['ef']] for r in rows])
        y = np.array([r['profile_class'] for r in rows])
        plot_classification_2d(X, y, dim1=1, dim2=0, dim_labels=DIM_LABELS, ax=ax)
        ax.set_title(f'$h_0$ = {int(h0)} km')
    fig.tight_layout()
    path = os.path.join(outdir, 'fig4_classification_da_T.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.4 saved: {path}")
    return path


def fig5_classification_di_T(db, outdir):
    """Fig.5: 분류 지도 (Δi vs T_max/T₀), h0별 4장 서브플롯."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, h0 in enumerate(H0_SLICES):
        ax = axes[idx // 2][idx % 2]
        rows = db.get_results(h0=h0, converged=True)
        if not rows:
            ax.set_title(f'$h_0$ = {int(h0)} km (no data)')
            continue
        X = np.array([[r['T_max_normed'], r['delta_a'], r['delta_i'], r['e0'], r['ef']] for r in rows])
        y = np.array([r['profile_class'] for r in rows])
        plot_classification_2d(X, y, dim1=2, dim2=0, dim_labels=DIM_LABELS, ax=ax)
        ax.set_title(f'$h_0$ = {int(h0)} km')
    fig.tight_layout()
    path = os.path.join(outdir, 'fig5_classification_di_T.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.5 saved: {path}")
    return path


def fig6_classification_3d(db, outdir):
    """Fig.6: 3D 분류 지도 (대표 h0 1개)."""
    # 데이터가 가장 많은 슬라이스 사용
    best_h0, best_count = H0_SLICES[0], 0
    for h0 in H0_SLICES:
        rows = db.get_results(h0=h0, converged=True)
        if len(rows) > best_count:
            best_h0, best_count = h0, len(rows)

    rows = db.get_results(h0=best_h0, converged=True)
    X = np.array([[r['T_max_normed'], r['delta_a'], r['delta_i'], r['e0'], r['ef']] for r in rows])
    y = np.array([r['profile_class'] for r in rows])
    fig, ax = plot_classification_3d(X, y, dim_labels=DIM_LABELS)
    ax.set_title(f'Classification Map ($h_0$ = {int(best_h0)} km)')
    path = os.path.join(outdir, 'fig6_classification_3d.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.6 saved: {path}")
    return path


def fig7_convergence_history(db, outdir):
    """Fig.7: 적응적 샘플링 수렴 이력.

    DB에서 반복별 누적 샘플 수 → max entropy를 역산하여 근사.
    실제 entropy 기록이 없으므로, 고도별 누적 샘플 수 곡선으로 대체.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for h0 in H0_SLICES:
        rows = db.get_results(h0=h0, converged=True)
        if not rows:
            continue
        # 시간순 누적
        n_samples = len(rows)
        ax.plot(range(n_samples), range(n_samples),
                label=f'$h_0$={int(h0)} km', alpha=0.8)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cumulative Samples')
    ax.set_title('Adaptive Sampling Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, 'fig7_sampling_progress.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.7 saved: {path}")
    return path


def fig8_ard_lengthscales(db, outdir):
    """Fig.8: ARD 길이 척도 (h0별 비교, 5D)."""
    from orbit_transfer.sampling.gp_classifier import GPClassifierWrapper
    from orbit_transfer.sampling.lhs import normalize_params

    # sorted keys: ['T_max_normed', 'delta_a', 'delta_i', 'e0', 'ef']
    param_names = [r'$T_{max}/T_0$', r'$\Delta a$', r'$\Delta i$', r'$e_0$', r'$e_f$']
    n_dims = 5
    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.18
    x_pos = np.arange(n_dims)

    for idx, h0 in enumerate(H0_SLICES):
        rows = db.get_results(h0=h0, converged=True)
        if len(rows) < 20:
            continue
        X_phys = np.array([
            [r['T_max_normed'], r['delta_a'], r['delta_i'], r['e0'], r['ef']]
            for r in rows
        ])
        y = np.array([r['profile_class'] for r in rows])

        X_norm = normalize_params(X_phys)

        if len(np.unique(y)) < 2:
            continue

        gpc = GPClassifierWrapper(n_dims=n_dims)
        gpc.fit(X_norm, y)
        ls = gpc.get_length_scales()
        ax.bar(x_pos + idx * bar_width, ls, bar_width,
               label=f'$h_0$={int(h0)} km', alpha=0.8)

    ax.set_xticks(x_pos + bar_width * 1.5)
    ax.set_xticklabels(param_names)
    ax.set_ylabel('ARD Length Scale')
    ax.set_yscale('log')
    ax.set_title('ARD Length Scales by Altitude')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(outdir, 'fig8_ard_lengthscales.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.8 saved: {path}")
    return path


def fig9_cost_vs_params(db, outdir):
    """Fig.9: 비용 vs 파라미터 (클래스별 색상)."""
    rows = db.get_results(converged=True)
    if not rows:
        return None

    X = np.array([[r['T_max_normed'], r['delta_a'], r['delta_i'], r['e0'], r['ef']] for r in rows])
    y = np.array([r['profile_class'] for r in rows])
    cost = np.array([r['cost'] for r in rows])

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
    cmap = ListedColormap(CLASS_COLORS)

    for dim, ax, label in zip(range(5), axes, DIM_LABELS):
        for cls in range(3):
            mask = y == cls
            if np.any(mask):
                ax.scatter(X[mask, dim], cost[mask], c=[cmap(cls)],
                           label=CLASS_NAMES[cls], alpha=0.5, s=10)
        ax.set_xlabel(label)
        if dim == 0:
            ax.set_ylabel(r'Cost $J$')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(outdir, 'fig9_cost_vs_params.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.9 saved: {path}")
    return path


def fig10_peak_timing(db, outdir):
    """Fig.10: 피크 타이밍 분포 히스토그램."""
    from orbit_transfer.classification.peak_detection import detect_peaks

    rows = db.get_results(converged=True)
    if not rows:
        return None

    peak_fractions = {0: [], 1: [], 2: []}

    for row in rows:
        try:
            traj = db.get_trajectory(row['id'])
            if not _is_trajectory_valid(traj, row['h0']):
                continue
            u_mag = np.linalg.norm(traj['u'], axis=0)
            t = traj['t']
            T = t[-1] - t[0]
            if T <= 0:
                continue
            n_pk, peak_times, _ = detect_peaks(t, u_mag, T)
            cls = row['profile_class']
            for pt in peak_times:
                frac = (pt - t[0]) / T
                peak_fractions[cls].append(frac)
        except Exception:
            continue

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)
    for cls, ax, name, color in zip(range(3), axes, CLASS_NAMES, CLASS_COLORS):
        if peak_fractions[cls]:
            ax.hist(peak_fractions[cls], bins=30, color=color, alpha=0.7,
                    edgecolor='white', density=True)
        ax.set_xlabel(r'Normalized Time $t/T$')
        if cls == 0:
            ax.set_ylabel('Density')
        ax.set_title(f'({chr(97+cls)}) {name}')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Peak Timing Distribution', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, 'fig10_peak_timing.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.10 saved: {path}")
    return path


def fig11_hohmann_correspondence(db, outdir):
    """Fig.11: 호만전이 Δv vs L2 비용 상관."""
    from orbit_transfer.constants import MU_EARTH, R_E

    rows = db.get_results(converged=True)
    if not rows:
        return None

    dv_hohmann = []
    l2_cost = []
    classes = []

    for row in rows:
        h0 = row['h0']
        da = row['delta_a']
        di = row['delta_i']
        e0 = row['e0']
        ef = row['ef']
        a0 = R_E + h0
        af = a0 + da

        if af <= 0 or a0 <= 0:
            continue

        # Generalized two-impulse Δv for elliptical orbits (vis-viva)
        # Departure at periapsis of (a0, e0), arrival at apoapsis of (af, ef)
        # or vice versa, choosing the lower total Δv
        r_p0 = a0 * (1.0 - e0)
        r_a0 = a0 * (1.0 + e0)
        r_pf = af * (1.0 - ef)
        r_af = af * (1.0 + ef)

        def _vis_viva(r, a):
            return np.sqrt(max(MU_EARTH * (2.0 / r - 1.0 / a), 0.0))

        def _transfer_dv(r1, r2, a_dep, a_arr):
            """Two-impulse Δv: depart at r1, arrive at r2."""
            a_t = (r1 + r2) / 2.0
            if a_t <= 0 or r1 <= 0 or r2 <= 0:
                return np.inf
            v_dep = _vis_viva(r1, a_dep)
            v_t1 = _vis_viva(r1, a_t)
            v_t2 = _vis_viva(r2, a_t)
            v_arr = _vis_viva(r2, a_arr)
            return abs(v_t1 - v_dep) + abs(v_arr - v_t2)

        # Try four combinations, pick minimum
        candidates = [
            _transfer_dv(r_p0, r_af, a0, af),
            _transfer_dv(r_p0, r_pf, a0, af),
            _transfer_dv(r_a0, r_af, a0, af),
            _transfer_dv(r_a0, r_pf, a0, af),
        ]
        dv_orbit = min(candidates)

        # Plane change Δv (combined with first impulse)
        di_rad = np.radians(di)
        if di > 0:
            v_dep = _vis_viva(r_p0, a0)
            dv_plane = 2.0 * v_dep * np.sin(di_rad / 2.0)
            dv_total = np.sqrt(dv_orbit**2 + dv_plane**2)
        else:
            dv_total = dv_orbit

        dv_hohmann.append(dv_total)
        l2_cost.append(row['cost'])
        classes.append(row['profile_class'])

    dv_hohmann = np.array(dv_hohmann)
    l2_cost = np.array(l2_cost)
    classes = np.array(classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(CLASS_COLORS)
    for cls in range(3):
        mask = classes == cls
        if np.any(mask):
            ax.scatter(dv_hohmann[mask], l2_cost[mask], c=[cmap(cls)],
                       label=CLASS_NAMES[cls], alpha=0.5, s=15)

    ax.set_xlabel(r'Hohmann $\Delta v$ [km/s]')
    ax.set_ylabel(r'$L^2$ Cost $J$')
    ax.set_yscale('log')
    ax.set_title(r'Hohmann $\Delta v$ vs Continuous Thrust $L^2$ Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, 'fig11_hohmann_correspondence.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig.11 saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--db', type=str, default='data/trajectories_all.duckdb')
    parser.add_argument('--outdir', type=str, default='manuscript/figures')
    parser.add_argument('--only', type=str, default=None,
                        help='Generate specific figure (e.g., fig1, fig3)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    db = TrajectoryDatabase(db_path=args.db)
    results = db.get_results(converged=True)

    if not results:
        print("No results in database. Run run_database.py first.")
        db.close()
        return

    print(f"Database: {len(results)} converged results")

    generators = {
        'fig1': fig1_representative_profiles,
        'fig2': fig2_representative_trajectories,
        'fig3': fig3_classification_da_di,
        'fig4': fig4_classification_da_T,
        'fig5': fig5_classification_di_T,
        'fig6': fig6_classification_3d,
        'fig7': fig7_convergence_history,
        'fig8': fig8_ard_lengthscales,
        'fig9': fig9_cost_vs_params,
        'fig10': fig10_peak_timing,
        'fig11': fig11_hohmann_correspondence,
    }

    if args.only:
        keys = [k.strip() for k in args.only.split(',')]
    else:
        keys = generators.keys()

    for key in keys:
        if key in generators:
            try:
                generators[key](db, args.outdir)
            except Exception as e:
                print(f"  {key}: FAILED - {e}")
        else:
            print(f"  Unknown figure: {key}")

    db.close()
    print("Done.")


if __name__ == '__main__':
    main()
