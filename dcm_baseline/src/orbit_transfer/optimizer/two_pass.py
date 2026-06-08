"""Two-Pass Hybrid Collocation 파이프라인."""

import numpy as np

from ..types import TransferConfig, TrajectoryResult
from ..config import MAX_NU_RETRIES, TOL_RELAXATION_FACTOR
from ..collocation.hermite_simpson import HermiteSimpsonCollocation
from ..collocation.multiphase_lgl import MultiPhaseLGLCollocation
from ..collocation.interpolation import interpolate_pass1_to_pass2
from ..optimizer.initial_guess import linear_interpolation_guess
from ..classification.peak_detection import detect_peaks
from ..classification.classifier import classify_profile, determine_phase_structure


class TwoPassOptimizer:
    """Two-Pass Hybrid Collocation 파이프라인.

    Pass 1: H-S collocation (피크 위치 탐지)
    -> 피크 탐지 -> Phase 구조 결정
    -> Pass 2: Multi-Phase LGL (정밀 최적화)

    수렴 실패 시 복구:
    1. 기본 초기값 -> 실패 시
    2. true anomaly 랜덤 변경 (MAX_NU_RETRIES회)
    3. 허용 오차 완화 (tol * TOL_RELAXATION_FACTOR)
    4. converged=False로 반환
    """

    def __init__(self, config: TransferConfig, l1_lambda: float = 0.0):
        self.config = config
        self.l1_lambda = l1_lambda

    def solve(
        self,
        x_init: np.ndarray | None = None,
        u_init: np.ndarray | None = None,
        t_init: np.ndarray | None = None,
    ) -> TrajectoryResult:
        """Two-Pass 최적화 실행.

        Parameters
        ----------
        x_init : ndarray, optional  외부 초기치 상태 (6, M). None이면 선형 보간 사용
        u_init : ndarray, optional  외부 초기치 제어 (3, M). None이면 zero
        t_init : ndarray, optional  외부 초기치 시간 (M,). 리샘플링에 사용

        Returns
        -------
        TrajectoryResult
        """
        # === Pass 1 ===
        hs = HermiteSimpsonCollocation(self.config, l1_lambda=self.l1_lambda)

        if x_init is not None:
            # 외부 초기치를 Pass 1 그리드에 리샘플링
            from scipy.interpolate import interp1d
            N1 = hs.N_points
            if t_init is None:
                t_init = np.linspace(0, 1, x_init.shape[1])
            t_target = np.linspace(t_init[0], t_init[-1], N1)
            x_g = interp1d(t_init, x_init, kind='linear',
                           fill_value='extrapolate')(t_target)
            if u_init is not None:
                u_g = interp1d(t_init, u_init, kind='linear',
                               fill_value='extrapolate')(t_target)
            else:
                u_g = np.zeros((3, N1))
            nu0_g = 0.0
            nuf_g = np.pi
        else:
            t_g, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(
                self.config, hs.N_points
            )

        result1 = hs.solve(
            x_guess=x_g, u_guess=u_g,
            nu0_guess=nu0_g, nuf_guess=nuf_g,
        )

        if not result1.converged:
            # true anomaly 랜덤 변경 시도
            rng = np.random.default_rng(42)
            for _ in range(MAX_NU_RETRIES):
                nu0_r = rng.uniform(0, 2 * np.pi)
                nuf_r = rng.uniform(0, 2 * np.pi)
                t_g, x_g, u_g, _, _ = linear_interpolation_guess(
                    self.config, hs.N_points
                )
                result1 = hs.solve(
                    x_guess=x_g, u_guess=u_g,
                    nu0_guess=nu0_r, nuf_guess=nuf_r,
                )
                if result1.converged:
                    break

            if not result1.converged:
                # 허용 오차 완화
                relaxed_opts = {
                    'ipopt.tol': 1e-4 * TOL_RELAXATION_FACTOR,
                    'ipopt.constr_viol_tol': 1e-4 * TOL_RELAXATION_FACTOR,
                }
                t_g, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(
                    self.config, hs.N_points
                )
                result1 = hs.solve(
                    x_guess=x_g, u_guess=u_g,
                    nu0_guess=nu0_g, nuf_guess=nuf_g,
                    ipopt_options=relaxed_opts,
                )

            if not result1.converged:
                result1.pass1_cost = None
                return result1

        pass1_cost = result1.cost

        # === 피크 탐지 + Phase 구조 ===
        u_mag = np.linalg.norm(result1.u, axis=0)
        n_peaks, peak_times, peak_widths = detect_peaks(
            result1.t, u_mag, result1.T_f
        )
        profile_class = classify_profile(n_peaks)
        phases = determine_phase_structure(
            peak_times, peak_widths, result1.T_f
        )

        # === 보간 (warm start) ===
        t_phases, x_phases, u_phases = interpolate_pass1_to_pass2(
            result1.t, result1.x, result1.u, phases
        )

        # === Pass 2 ===
        lgl = MultiPhaseLGLCollocation(self.config, phases, T_fixed=result1.T_f,
                                       l1_lambda=self.l1_lambda)
        result2 = lgl.solve(
            x_phases=x_phases, u_phases=u_phases,
            nu0_guess=result1.nu0, nuf_guess=result1.nuf,
        )
        result2.pass1_cost = pass1_cost

        if not result2.converged:
            # Pass 2 실패 시 Pass 1 결과로 대체
            result1.pass1_cost = pass1_cost
            return result1

        return result2
