"""TransferBenchmark: 궤도전이 기법 통합 비교 클래스."""

from __future__ import annotations
import os
import csv
import numpy as np
from typing import Any

from ..types import TransferConfig
from .result import BenchmarkResult
from .solvers import HohmannSolver, LambertSolver, CollocationSolver
from .metrics import compute_metrics


class TransferBenchmark:
    """여러 궤도전이 기법을 실행하고 결과를 비교·내보내기하는 클래스.

    Examples
    --------
    기본 사용:

    >>> from orbit_transfer.benchmark import TransferBenchmark
    >>> from orbit_transfer.types import TransferConfig
    >>> config = TransferConfig(h0=400, delta_a=500, delta_i=5.0, T_max_normed=0.8)
    >>> bench = TransferBenchmark(config)
    >>> bench.run_all()
    >>> bench.export_csv("results/comparison.csv")
    >>> bench.export_figures("results/")

    개별 기법만 실행:

    >>> bench.run_hohmann()
    >>> bench.run_collocation()
    """

    def __init__(self, config: TransferConfig):
        self.config = config
        self._results: dict[str, BenchmarkResult] = {}

    # -----------------------------------------------------------------------
    # 실행 메서드
    # -----------------------------------------------------------------------

    def run_hohmann(self, n_points: int = 300) -> BenchmarkResult:
        """호만 전이 실행."""
        solver = HohmannSolver(n_points=n_points)
        result = solver.solve(self.config)
        self._results["hohmann"] = result
        return result

    def run_lambert(
        self,
        tof: float | None = None,
        n_points: int = 300,
        prograde: bool = True,
        **lambert_kwargs,
    ) -> BenchmarkResult:
        """Lambert 전이 실행.

        Parameters
        ----------
        tof : float, optional  비행시간 [s]. None이면 Hohmann TOF 사용
        n_points : int
        prograde : bool
        **lambert_kwargs
            LambertSolver 추가 파라미터.
            예: i0_deg=37.5, Omega_i_deg=0, Omega_f_deg=56
        """
        solver = LambertSolver(tof=tof, n_points=n_points, prograde=prograde,
                               **lambert_kwargs)
        result = solver.solve(self.config)
        self._results["lambert"] = result
        return result

    def run_collocation(self) -> BenchmarkResult:
        """Two-Pass Collocation 최적화 실행."""
        solver = CollocationSolver()
        result = solver.solve(self.config)
        self._results["collocation"] = result
        return result

    def run_all(
        self,
        include_hohmann: bool = True,
        include_lambert: bool = True,
        include_collocation: bool = True,
        lambert_tof: float | None = None,
    ) -> dict[str, BenchmarkResult]:
        """모든 기법을 순서대로 실행한다.

        Parameters
        ----------
        include_* : bool  각 기법 포함 여부
        lambert_tof : float, optional  Lambert 비행시간 지정

        Returns
        -------
        dict { method: BenchmarkResult }
        """
        if include_hohmann:
            self.run_hohmann()
        if include_lambert:
            self.run_lambert(tof=lambert_tof)
        if include_collocation:
            self.run_collocation()
        return dict(self._results)

    @property
    def results(self) -> dict[str, BenchmarkResult]:
        """현재까지 실행된 결과 dict."""
        return dict(self._results)

    # -----------------------------------------------------------------------
    # CSV 내보내기
    # -----------------------------------------------------------------------

    def export_csv(
        self,
        path: str,
        results: dict[str, BenchmarkResult] | None = None,
    ) -> None:
        """비교 지표 요약 CSV를 내보낸다.

        각 기법이 한 행으로 저장된다.

        Parameters
        ----------
        path : str          저장 경로 (예: 'results/comparison.csv')
        results : dict, optional  None이면 self.results 사용
        """
        res = results if results is not None else self._results
        if not res:
            raise RuntimeError("먼저 run_*() 메서드로 실행 결과를 생성해 주세요.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # 모든 지표 키 수집 (순서 보존)
        all_keys: list[str] = []
        for r in res.values():
            for k in r.metrics:
                if k not in all_keys:
                    all_keys.append(k)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["method", "converged", "is_impulsive"] + all_keys
            writer.writerow(header)
            for method, r in res.items():
                row = [method, r.converged, r.is_impulsive]
                for k in all_keys:
                    v = r.metrics.get(k, "")
                    if v is None:
                        v = ""
                    elif isinstance(v, float) and np.isnan(v):
                        v = ""
                    row.append(v)
                writer.writerow(row)

        print(f"[benchmark] 요약 CSV 저장: {path}")

    def export_trajectory_csv(
        self,
        outdir: str,
        results: dict[str, BenchmarkResult] | None = None,
    ) -> None:
        """각 기법의 전체 궤적을 개별 CSV 파일로 내보낸다.

        파일명: <outdir>/<method>_trajectory.csv
        컬럼: t[s], x[km], y[km], z[km], vx[km/s], vy[km/s], vz[km/s],
               ux[km/s²], uy[km/s²], uz[km/s²], u_mag[km/s²]

        임펄스 기법은 impulse 정보를 <method>_impulses.csv에 추가 저장한다.

        Parameters
        ----------
        outdir : str          저장 디렉토리
        results : dict, optional
        """
        res = results if results is not None else self._results
        os.makedirs(outdir, exist_ok=True)

        for method, r in res.items():
            # 궤적 CSV
            traj_path = os.path.join(outdir, f"{method}_trajectory.csv")
            header = ["t[s]", "x[km]", "y[km]", "z[km]",
                      "vx[km/s]", "vy[km/s]", "vz[km/s]",
                      "ux[km/s2]", "uy[km/s2]", "uz[km/s2]", "u_mag[km/s2]"]
            u_mag = np.linalg.norm(r.u, axis=0)
            with open(traj_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for k in range(len(r.t)):
                    row = [
                        r.t[k],
                        r.x[0, k], r.x[1, k], r.x[2, k],
                        r.x[3, k], r.x[4, k], r.x[5, k],
                        r.u[0, k], r.u[1, k], r.u[2, k],
                        u_mag[k],
                    ]
                    writer.writerow(row)
            print(f"[benchmark] 궤적 CSV: {traj_path}")

            # 임펄스 CSV
            if r.is_impulsive and r.impulses:
                imp_path = os.path.join(outdir, f"{method}_impulses.csv")
                with open(imp_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["t[s]", "dvx[km/s]", "dvy[km/s]", "dvz[km/s]", "dv[km/s]"])
                    for imp in r.impulses:
                        dv_v = imp["dv_vec"]
                        writer.writerow([imp["t"], dv_v[0], dv_v[1], dv_v[2], imp["dv"]])
                print(f"[benchmark] 임펄스 CSV: {imp_path}")

    # -----------------------------------------------------------------------
    # 그림 내보내기
    # -----------------------------------------------------------------------

    def export_figures(
        self,
        outdir: str,
        results: dict[str, BenchmarkResult] | None = None,
        fmt: str = "pdf",
        metrics_to_plot: list[str] | None = None,
    ) -> None:
        """비교 그림을 outdir에 저장한다.

        생성 파일:
        - thrust_profiles.<fmt>    추력 프로파일 비교
        - trajectories.<fmt>       3D 궤적 비교
        - metrics_bar.<fmt>        지표 막대 그래프

        Parameters
        ----------
        outdir : str
        results : dict, optional
        fmt : str   저장 형식 ('pdf', 'png', 'svg')
        metrics_to_plot : list[str], optional   막대 그래프에 표시할 지표
        """
        from .plot import plot_thrust_profiles, plot_trajectories, plot_metrics_bar

        res = results if results is not None else self._results
        if not res:
            raise RuntimeError("먼저 run_*() 메서드로 실행 결과를 생성해 주세요.")

        os.makedirs(outdir, exist_ok=True)

        fig1 = plot_thrust_profiles(res, config=self.config)
        p1 = os.path.join(outdir, f"thrust_profiles.{fmt}")
        fig1.savefig(p1, bbox_inches="tight")
        plt.close(fig1)
        print(f"[benchmark] 그림 저장: {p1}")

        fig2 = plot_trajectories(res, config=self.config)
        p2 = os.path.join(outdir, f"trajectories.{fmt}")
        fig2.savefig(p2, bbox_inches="tight")
        plt.close(fig2)
        print(f"[benchmark] 그림 저장: {p2}")

        fig3 = plot_metrics_bar(res, metrics_to_plot=metrics_to_plot)
        p3 = os.path.join(outdir, f"metrics_bar.{fmt}")
        fig3.savefig(p3, bbox_inches="tight")
        plt.close(fig3)
        print(f"[benchmark] 그림 저장: {p3}")

    # -----------------------------------------------------------------------
    # 콘솔 출력
    # -----------------------------------------------------------------------

    def print_summary(
        self,
        results: dict[str, BenchmarkResult] | None = None,
    ) -> None:
        """비교 지표를 콘솔 표로 출력한다."""
        res = results if results is not None else self._results
        if not res:
            print("결과 없음. run_*() 실행 후 호출하세요.")
            return

        keys = ["dv_total", "dv1", "dv2", "dv_total_opt", "cost_l1", "cost_l2",
                "tof", "tof_norm", "n_peaks", "profile_class"]

        col_w = 16
        header = f"{'method':<14}" + "".join(f"{k:>{col_w}}" for k in keys)
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for method, r in res.items():
            row = f"{method:<14}"
            for k in keys:
                v = r.metrics.get(k, "")
                if v is None:
                    v = "-"
                elif isinstance(v, float):
                    v = f"{v:.5f}" if not np.isnan(v) else "-"
                row += f"{str(v):>{col_w}}"
            print(row)
        print("-" * len(header))


# matplotlib import (lazy)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore
