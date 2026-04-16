"""논문 Table 데이터 생성.

Usage:
    python scripts/generate_tables.py --db data/trajectories.duckdb
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, 'src')

from orbit_transfer.config import H0_SLICES
from orbit_transfer.database.storage import TrajectoryDatabase
from orbit_transfer.database.query import get_statistics
from orbit_transfer.sampling.gp_classifier import GPClassifierWrapper
from orbit_transfer.sampling.lhs import normalize_params

CLASS_NAMES = ['Unimodal', 'Bimodal', 'Multimodal']


def table2_db_summary(db):
    """Table 2: DB 통계 요약 (고도별 총 샘플/수렴율/클래스 분포)."""
    print("\n=== Table 2: Database Summary ===")
    print(f"{'h0 [km]':>8}  {'Total':>6}  {'Conv.':>6}  {'Rate':>6}  "
          f"{'Uni':>5}  {'Bi':>5}  {'Multi':>5}  {'Mean Cost':>12}")
    print("-" * 72)

    totals = {'total': 0, 'converged': 0, 'uni': 0, 'bi': 0, 'multi': 0, 'costs': []}

    for h0 in H0_SLICES:
        stats = get_statistics(db, h0=h0)
        total = stats['total']
        conv = stats['converged']
        rate = stats['convergence_rate']
        cd = stats['class_distribution']
        uni = cd.get(0, 0)
        bi = cd.get(1, 0)
        multi = cd.get(2, 0)
        mc = stats['mean_cost']

        totals['total'] += total
        totals['converged'] += conv
        totals['uni'] += uni
        totals['bi'] += bi
        totals['multi'] += multi

        mc_str = f"{mc:.4e}" if mc is not None else "N/A"
        print(f"{int(h0):>8}  {total:>6}  {conv:>6}  {rate:>6.1%}  "
              f"{uni:>5}  {bi:>5}  {multi:>5}  {mc_str:>12}")

        # 비용 수집
        rows = db.get_results(h0=h0, converged=True)
        totals['costs'].extend([r['cost'] for r in rows])

    # 합계
    t = totals
    overall_rate = t['converged'] / t['total'] if t['total'] > 0 else 0
    overall_mc = np.mean(t['costs']) if t['costs'] else 0
    print("-" * 72)
    print(f"{'Total':>8}  {t['total']:>6}  {t['converged']:>6}  {overall_rate:>6.1%}  "
          f"{t['uni']:>5}  {t['bi']:>5}  {t['multi']:>5}  {overall_mc:>12.4e}")

    # LaTeX 형식
    print("\n% LaTeX Table 2")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Summary of the trajectory database.}")
    print(r"\label{tab:db_summary}")
    print(r"\begin{tabular}{rrrrccc}")
    print(r"\hline")
    print(r"$h_0$ [km] & Total & Converged & Rate & Unimodal & Bimodal & Multimodal \\")
    print(r"\hline")
    for h0 in H0_SLICES:
        stats = get_statistics(db, h0=h0)
        cd = stats['class_distribution']
        print(f"{int(h0)} & {stats['total']} & {stats['converged']} & "
              f"{stats['convergence_rate']:.1%} & "
              f"{cd.get(0,0)} & {cd.get(1,0)} & {cd.get(2,0)} \\\\")
    print(r"\hline")
    print(f"Total & {t['total']} & {t['converged']} & {overall_rate:.1%} & "
          f"{t['uni']} & {t['bi']} & {t['multi']} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def table3_class_cost(db):
    """Table 3: 클래스별 평균 비용."""
    print("\n=== Table 3: Cost Statistics by Class ===")
    print(f"{'Class':>12}  {'Count':>6}  {'Mean':>12}  {'Std':>12}  {'Min':>12}  {'Max':>12}")
    print("-" * 72)

    for cls, name in enumerate(CLASS_NAMES):
        rows = db.get_results(converged=True, profile_class=cls)
        if not rows:
            print(f"{name:>12}  {0:>6}  {'N/A':>12}  {'N/A':>12}  {'N/A':>12}  {'N/A':>12}")
            continue
        costs = np.array([r['cost'] for r in rows])
        print(f"{name:>12}  {len(costs):>6}  {np.mean(costs):>12.4e}  "
              f"{np.std(costs):>12.4e}  {np.min(costs):>12.4e}  {np.max(costs):>12.4e}")

    # LaTeX
    print("\n% LaTeX Table 3")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Cost statistics by profile class.}")
    print(r"\label{tab:class_cost}")
    print(r"\begin{tabular}{lrcccc}")
    print(r"\hline")
    print(r"Class & Count & Mean & Std & Min & Max \\")
    print(r"\hline")
    for cls, name in enumerate(CLASS_NAMES):
        rows = db.get_results(converged=True, profile_class=cls)
        if not rows:
            continue
        costs = np.array([r['cost'] for r in rows])
        print(f"{name} & {len(costs)} & {np.mean(costs):.2e} & "
              f"{np.std(costs):.2e} & {np.min(costs):.2e} & {np.max(costs):.2e} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def table4_ard_lengthscales(db):
    """Table 4: ARD 길이 척도 (고도별, 5D)."""
    print("\n=== Table 4: ARD Length Scales ===")
    # sorted keys: ['T_max_normed', 'delta_a', 'delta_i', 'e0', 'ef']
    param_names = ['T_max/T0', 'da', 'di', 'e0', 'ef']
    print(f"{'h0 [km]':>8}  {'T_max/T0':>10}  {'da':>10}  {'di':>10}  {'e0':>10}  {'ef':>10}")
    print("-" * 66)

    n_dims = 5
    all_ls = {}
    for h0 in H0_SLICES:
        rows = db.get_results(h0=h0, converged=True)
        if len(rows) < 20:
            print(f"{int(h0):>8}  " + "  ".join([f"{'N/A':>10}"] * n_dims))
            continue
        X_phys = np.array([
            [r['T_max_normed'], r['delta_a'], r['delta_i'], r['e0'], r['ef']]
            for r in rows
        ])
        y = np.array([r['profile_class'] for r in rows])

        if len(np.unique(y)) < 2:
            continue

        X_norm = normalize_params(X_phys)
        gpc = GPClassifierWrapper(n_dims=n_dims)
        gpc.fit(X_norm, y)
        ls = gpc.get_length_scales()
        all_ls[h0] = ls
        print(f"{int(h0):>8}  " + "  ".join([f"{ls[i]:>10.4f}" for i in range(n_dims)]))

    # LaTeX
    print("\n% LaTeX Table 4")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{ARD length scales by initial altitude.}")
    print(r"\label{tab:ard_lengthscales}")
    print(r"\begin{tabular}{rccccc}")
    print(r"\hline")
    print(r"$h_0$ [km] & $\ell_{T_{max}/T_0}$ & $\ell_{\Delta a}$ & $\ell_{\Delta i}$ & $\ell_{e_0}$ & $\ell_{e_f}$ \\")
    print(r"\hline")
    for h0, ls in all_ls.items():
        print(f"{int(h0)} & " + " & ".join([f"{ls[i]:.4f}" for i in range(n_dims)]) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return all_ls


def table5_solver_performance(db):
    """Table 5: 수렴 성능 통계."""
    print("\n=== Table 5: Solver Performance ===")
    print(f"{'h0 [km]':>8}  {'Mean Time':>10}  {'Std Time':>10}  {'Min':>8}  {'Max':>8}")
    print("-" * 52)

    for h0 in H0_SLICES:
        rows = db.get_results(h0=h0, converged=True)
        if not rows:
            continue
        times = np.array([r['solve_time'] for r in rows])
        print(f"{int(h0):>8}  {np.mean(times):>10.2f}  {np.std(times):>10.2f}  "
              f"{np.min(times):>8.2f}  {np.max(times):>8.2f}")

    # 전체
    rows = db.get_results(converged=True)
    if rows:
        times = np.array([r['solve_time'] for r in rows])
        print("-" * 52)
        print(f"{'Total':>8}  {np.mean(times):>10.2f}  {np.std(times):>10.2f}  "
              f"{np.min(times):>8.2f}  {np.max(times):>8.2f}")

    # LaTeX
    print("\n% LaTeX Table 5")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Solver performance statistics (converged cases).}")
    print(r"\label{tab:solver_performance}")
    print(r"\begin{tabular}{rcccc}")
    print(r"\hline")
    print(r"$h_0$ [km] & Mean [s] & Std [s] & Min [s] & Max [s] \\")
    print(r"\hline")
    for h0 in H0_SLICES:
        rows_h = db.get_results(h0=h0, converged=True)
        if not rows_h:
            continue
        times = np.array([r['solve_time'] for r in rows_h])
        print(f"{int(h0)} & {np.mean(times):.2f} & {np.std(times):.2f} & "
              f"{np.min(times):.2f} & {np.max(times):.2f} \\\\")
    print(r"\hline")
    if rows:
        times = np.array([r['solve_time'] for r in rows])
        print(f"All & {np.mean(times):.2f} & {np.std(times):.2f} & "
              f"{np.min(times):.2f} & {np.max(times):.2f} \\\\")
        print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(description='Generate paper tables')
    parser.add_argument('--db', type=str, default='data/trajectories.duckdb')
    args = parser.parse_args()

    db = TrajectoryDatabase(db_path=args.db)
    results = db.get_results(converged=True)

    if not results:
        print("No results in database.")
        db.close()
        return

    print(f"Database: {len(results)} converged results\n")

    table2_db_summary(db)
    table3_class_cost(db)
    table4_ard_lengthscales(db)
    table5_solver_performance(db)

    db.close()


if __name__ == '__main__':
    main()
