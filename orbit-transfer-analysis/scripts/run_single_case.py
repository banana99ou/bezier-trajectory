"""단일 궤도전이 케이스 실행.

Usage:
    python scripts/run_single_case.py --h0 400 --da 200 --di 0 --T_max 0.8
    python scripts/run_single_case.py --h0 400 --da 200 --di 3 --T_max 1.0 --e0 0.05 --ef 0.05 --plot
"""
import argparse
import sys
import time
import numpy as np

sys.path.insert(0, 'src')

from orbit_transfer.types import TransferConfig
from orbit_transfer.optimizer.two_pass import TwoPassOptimizer


def main():
    parser = argparse.ArgumentParser(description='Run single orbit transfer case')
    parser.add_argument('--h0', type=float, required=True, help='Initial altitude [km]')
    parser.add_argument('--da', type=float, required=True, help='Semi-major axis change [km]')
    parser.add_argument('--di', type=float, required=True, help='Inclination change [deg]')
    parser.add_argument('--T_max', type=float, required=True, help='Max transfer time / T0')
    parser.add_argument('--e0', type=float, default=0.0, help='Departure eccentricity')
    parser.add_argument('--ef', type=float, default=0.0, help='Arrival eccentricity')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    args = parser.parse_args()

    config = TransferConfig(
        h0=args.h0, delta_a=args.da, delta_i=args.di,
        T_max_normed=args.T_max, e0=args.e0, ef=args.ef,
    )

    print(f"Config: h0={args.h0}km, da={args.da}km, di={args.di}deg, "
          f"T_max/T0={args.T_max}, e0={args.e0}, ef={args.ef}")
    print(f"  a0={config.a0:.1f}km, af={config.af:.1f}km, "
          f"T_max={config.T_max:.1f}s, T_min={config.T_min:.1f}s")

    t0 = time.time()
    opt = TwoPassOptimizer(config)
    result = opt.solve()
    elapsed = time.time() - t0

    print(f"\nResult:")
    print(f"  Converged: {result.converged}")
    print(f"  Cost: {result.cost:.6e}")
    print(f"  Pass1 Cost: {result.pass1_cost}")
    print(f"  T_f: {result.T_f:.1f}s ({result.T_f/config.T0:.4f} T0)")
    print(f"  Peaks: {result.n_peaks}")
    print(f"  Class: {result.profile_class} ({'unimodal' if result.profile_class==0 else 'bimodal' if result.profile_class==1 else 'multimodal'})")
    print(f"  nu0: {np.degrees(result.nu0):.2f} deg")
    print(f"  nuf: {np.degrees(result.nuf):.2f} deg")
    print(f"  Time: {elapsed:.2f}s")

    if args.plot and result.converged:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from orbit_transfer.visualization.thrust_profile import plot_thrust_magnitude
        from orbit_transfer.visualization.trajectory_3d import plot_trajectory_3d

        fig1, _ = plot_thrust_magnitude(result)
        fig2, _ = plot_trajectory_3d(result)
        plt.show()


if __name__ == '__main__':
    main()
