"""기법별 솔버 래퍼.

각 솔버는 TransferConfig를 받아 BenchmarkResult를 반환한다.
"""

from __future__ import annotations
import numpy as np

from ..types import TransferConfig
from .result import BenchmarkResult
from .metrics import compute_metrics


# ===========================================================================
# Hohmann 솔버
# ===========================================================================

class HohmannSolver:
    """호만 전이 (2-임펄스) 솔버.

    원궤도 가정 (e0 ≈ 0, ef ≈ 0).
    경사각 변화는 도착 번(burn 2)에서 전부 수행한다.
    최적 분리 Δv는 메트릭으로 별도 제공한다.
    """

    def __init__(self, n_points: int = 300):
        self.n_points = n_points

    def solve(self, config: TransferConfig) -> BenchmarkResult:
        """호만 전이 궤적 계산.

        Parameters
        ----------
        config : TransferConfig

        Returns
        -------
        BenchmarkResult  (is_impulsive=True)
        """
        from ..astrodynamics.hohmann import hohmann_trajectory

        t, x, u, impulses, hm = hohmann_trajectory(config, n_points=self.n_points)

        result = BenchmarkResult(
            method="hohmann",
            converged=True,
            t=t,
            x=x,
            u=u,
            is_impulsive=True,
            impulses=impulses,
            extra={"dv_total_opt": hm["dv_total_opt"],
                   "beta_opt_deg": hm["beta_opt_deg"]},
        )
        result.metrics = compute_metrics(result, config)
        # hohmann_trajectory에서 계산한 값으로 덮어쓰기
        result.metrics["tof"] = hm["tof"]
        result.metrics["dv_total_opt"] = hm["dv_total_opt"]
        result.metrics["beta_opt_deg"] = hm["beta_opt_deg"]
        return result


# ===========================================================================
# Lambert 솔버
# ===========================================================================

class LambertSolver:
    """Lambert 전이 (2-임펄스) 솔버.

    Curtis Algorithm 5.2로 두 위치벡터 간의 전이 속도를 계산한다.

    출발/도착 위치 기본값:
    - r1: 초기 원궤도 ν₀=0 위치 (i0, Omega_i 적용)
    - r2: 최종 원궤도에서 tof 후 위치 (if, Omega_f 적용)
    - tof: hohmann_tof (기본값) 또는 사용자 지정

    RAAN 차이가 있는 경우 (예: Seoul→Helsinki):
        LambertSolver(Omega_i_deg=0, Omega_f_deg=56)

    180° 전이 특이점 자동 회피:
        출발 anomaly를 π/6 오프셋하여 비특이 기하 사용
    """

    def __init__(
        self,
        tof: float | None = None,
        n_points: int = 300,
        prograde: bool = True,
        i0_deg: float = 0.0,
        Omega_i_deg: float = 0.0,
        Omega_f_deg: float = 0.0,
        nu0_deg: float = 0.0,
    ):
        """
        Parameters
        ----------
        tof : float, optional  비행시간 [s]. None이면 Hohmann TOF 사용
        n_points : int
        prograde : bool
        i0_deg : float   초기 궤도 경사각 [deg] (기본 0 = 적도)
        Omega_i_deg : float  초기 궤도 RAAN [deg]
        Omega_f_deg : float  최종 궤도 RAAN [deg]
        nu0_deg : float  출발 true anomaly [deg] (기본 0)
        """
        self._tof = tof
        self.n_points = n_points
        self.prograde = prograde
        self.i0_deg = i0_deg
        self.Omega_i_deg = Omega_i_deg
        self.Omega_f_deg = Omega_f_deg
        self.nu0_deg = nu0_deg

    def solve(self, config: TransferConfig) -> BenchmarkResult:
        """Lambert 전이 궤적 계산.

        Parameters
        ----------
        config : TransferConfig

        Returns
        -------
        BenchmarkResult  (is_impulsive=True)
        """
        from ..constants import MU_EARTH
        from ..astrodynamics.lambert import lambert
        from ..astrodynamics.orbital_elements import oe_to_rv
        from ..astrodynamics.hohmann import hohmann_tof
        from ..astrodynamics.kepler import kepler_propagate

        mu = MU_EARTH
        a0 = config.a0
        af = config.af
        i0 = float(np.radians(self.i0_deg))
        if_ = i0 + float(np.radians(config.delta_i))
        e0 = config.e0
        ef = config.ef
        Omega_i = float(np.radians(self.Omega_i_deg))
        Omega_f = float(np.radians(self.Omega_f_deg))
        nu0 = float(np.radians(self.nu0_deg))

        # --- 비행시간 ---
        # 기본: T_max_normed * T0 사용 (Hohmann TOF 대신)
        # → Δa=0일 때 Hohmann TOF = T0/2이면 타겟이 반드시 π 위치(180° 특이점)에
        #   도달하므로 대신 실제 허용 비행시간 상한을 사용한다
        if self._tof is not None:
            tof = self._tof
        elif abs(config.delta_a) < 1.0:
            # 동고도: T_max_normed * T0로 특이점 회피
            tof = config.T_max
        else:
            tof = hohmann_tof(a0, af, mu)

        # --- 출발 위치 ---
        r1, v1_circ = oe_to_rv((a0, e0, i0, Omega_i, 0.0, nu0), mu)

        # --- 도착 위치: 최종 원궤도에서 tof 후 위치 ---
        n_f = np.sqrt(mu / af ** 3)
        nu0_target = nu0  # 목표 위성도 같은 anomaly에서 출발 가정
        nu_f = (nu0_target + n_f * tof) % (2.0 * np.pi)
        r2, v2_circ = oe_to_rv((af, ef, if_, Omega_f, 0.0, nu_f), mu)

        # --- 180° 특이점 자동 회피 ---
        # r1과 r2가 반평행이면 출발 anomaly를 π/4 오프셋해서 재시도
        cos_dnu = float(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
        if abs(cos_dnu + 1.0) < 0.02:  # cos_dnu ≈ -1 (180° 근방)
            nu0_shifted = nu0 + np.pi / 4.0  # +45° 오프셋
            r1, v1_circ = oe_to_rv((a0, e0, i0, Omega_i, 0.0, nu0_shifted), mu)
            nu_f_shifted = (nu0_shifted + n_f * tof) % (2.0 * np.pi)
            r2, v2_circ = oe_to_rv((af, ef, if_, Omega_f, 0.0, nu_f_shifted), mu)
            nu0 = nu0_shifted
            nu_f = nu_f_shifted
            singularity_avoided = True
        else:
            singularity_avoided = False

        # --- Lambert 풀이 ---
        try:
            v1_lamb, v2_lamb = lambert(r1, r2, tof, mu, prograde=self.prograde)
        except (ValueError, RuntimeError) as exc:
            result = BenchmarkResult(
                method="lambert",
                converged=False,
                t=np.array([0.0, tof]),
                x=np.zeros((6, 2)),
                u=np.zeros((3, 2)),
                is_impulsive=True,
                impulses=[],
                extra={"error": str(exc)},
            )
            result.metrics = compute_metrics(result, config)
            return result

        # NaN/inf 결과 체크
        if not np.all(np.isfinite(v1_lamb)) or not np.all(np.isfinite(v2_lamb)):
            result = BenchmarkResult(
                method="lambert",
                converged=False,
                t=np.array([0.0, tof]),
                x=np.zeros((6, 2)),
                u=np.zeros((3, 2)),
                is_impulsive=True,
                impulses=[],
                extra={"error": "수치 발산 (NaN/inf)"},
            )
            result.metrics = compute_metrics(result, config)
            return result

        # --- 기동 벡터 ---
        dv1_vec = v1_lamb - v1_circ
        dv2_vec = v2_circ - v2_lamb
        dv1 = float(np.linalg.norm(dv1_vec))
        dv2 = float(np.linalg.norm(dv2_vec))

        # --- 궤적 생성 (전이 궤도 케플러 전파) ---
        ts = np.linspace(0.0, tof, self.n_points)
        x_arr = np.zeros((6, self.n_points))
        u_arr = np.zeros((3, self.n_points))
        for k, tk in enumerate(ts):
            rk, vk = kepler_propagate(r1, v1_lamb, tk, mu)
            x_arr[:3, k] = rk
            x_arr[3:, k] = vk

        impulses = [
            {"t": 0.0, "dv_vec": dv1_vec.copy(), "dv": dv1},
            {"t": tof, "dv_vec": dv2_vec.copy(), "dv": dv2},
        ]

        result = BenchmarkResult(
            method="lambert",
            converged=True,
            t=ts,
            x=x_arr,
            u=u_arr,
            is_impulsive=True,
            impulses=impulses,
            extra={
                "nu0_deg": float(np.degrees(nu0)),
                "nu_f_deg": float(np.degrees(nu_f)),
                "tof_used": tof,
                "singularity_avoided": singularity_avoided,
            },
        )
        result.metrics = compute_metrics(result, config)
        return result


# ===========================================================================
# Collocation 솔버
# ===========================================================================

class CollocationSolver:
    """Two-Pass Collocation 솔버 (TwoPassOptimizer 래퍼)."""

    def solve(self, config: TransferConfig) -> BenchmarkResult:
        """콜로케이션 최적 궤적 계산.

        Parameters
        ----------
        config : TransferConfig

        Returns
        -------
        BenchmarkResult  (is_impulsive=False)
        """
        from ..optimizer.two_pass import TwoPassOptimizer
        from ..collocation.interpolation import dense_output

        opt = TwoPassOptimizer(config)
        res = opt.solve()

        # dense output으로 균일 그리드 생성
        pb = None
        if res.solver_stats and "phase_boundaries" in res.solver_stats:
            pb = res.solver_stats["phase_boundaries"]
        t_d, x_d, u_d = dense_output(res.t, res.x, res.u, n_points=300,
                                      phase_boundaries=pb)

        result = BenchmarkResult(
            method="collocation",
            converged=res.converged,
            t=t_d,
            x=x_d,
            u=u_d,
            is_impulsive=False,
            impulses=[],
            extra={
                "n_peaks": res.n_peaks,
                "profile_class": res.profile_class,
                "pass1_cost": res.pass1_cost,
                "T_f": res.T_f,
                "nu0": res.nu0,
                "nuf": res.nuf,
            },
        )
        result.metrics = compute_metrics(result, config)
        # collocation 원본 비용 (최적화 목적함수 값)
        result.metrics["cost_l2_solver"] = float(res.cost) if res.converged else np.nan
        return result
