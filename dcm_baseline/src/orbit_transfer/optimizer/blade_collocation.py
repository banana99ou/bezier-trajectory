"""BLADE → LGL 콜로케이션 순차 최적화 파이프라인.

BLADE-SCP로 고품질 초기해를 생성하고, 구조적 피크 분류 후
Multi-Phase LGL 콜로케이션에 warm-start하여 고정밀 해를 확보한다.
기존 2-pass 파이프라인(H-S → LGL)에서 Pass 1을 BLADE로 대체한다.
"""

from __future__ import annotations

import time as _time

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from ..types import TransferConfig, TrajectoryResult
from ..classification.blade_classifier import (
    blade_classify_segments,
    blade_phase_structure,
)
from ..classification.classifier import classify_profile
from ..classification.peak_detection import detect_peaks
from ..collocation.blade_warmstart import (
    interpolate_blade_to_lgl,
    blade_to_dense_trajectory,
)
from ..collocation.multiphase_lgl import MultiPhaseLGLCollocation
from ..collocation.interpolation import dense_output


class BladeCollocationOptimizer:
    """BLADE → LGL 콜로케이션 순차 최적화.

    파이프라인:
        1. BLADE-SCP 실행 → 초기해 + 구조적 피크 분류
        2. Phase 구조 결정 (BLADE 세그먼트 기반)
        3. BLADE 해를 LGL 노드에 warm-start
        4. Multi-Phase LGL 콜로케이션 (Pass 2 직접 진입)
        5. 동역학 잔차 검증
        6. (선택) 격자 조밀화 후 재실행

    Parameters
    ----------
    config : TransferConfig
    blade_K : int
        BLADE 세그먼트 수.
    blade_n : int
        BLADE 세그먼트 차수.
    blade_max_iter : int
        BLADE SCP 최대 반복.
    blade_tol_bc : float
        BLADE 경계조건 허용치.
    blade_l1_lambda : float
        ℓ₁ 정규화 강도 (코스팅 유도).
    lgl_nodes_peak : int
        Peak phase LGL 노드 수.
    lgl_nodes_coast : int
        Coast phase LGL 노드 수.
    coast_threshold : float
        Coast 판정 임계값 (최대 norm 대비).
    dynamics_tol : float
        동역학 잔차 허용치.
    max_refinements : int
        최대 격자 조밀화 횟수.
    blade_relax_alpha : float
        BLADE SCP 스텝 감쇠.
    blade_trust_region : float
        BLADE 신뢰영역 반경.
    n_dense : int
        BLADE 세그먼트당 궤적 출력 점 수.
    """

    def __init__(
        self,
        config: TransferConfig,
        blade_K: int = 12,
        blade_n: int = 2,
        blade_max_iter: int = 50,
        blade_tol_bc: float = 1e-3,
        blade_l1_lambda: float = 0.0,
        lgl_nodes_peak: int = 15,
        lgl_nodes_coast: int = 8,
        coast_threshold: float = 0.05,
        dynamics_tol: float = 1e-6,
        max_refinements: int = 2,
        blade_relax_alpha: float = 0.5,
        blade_trust_region: float = 5.0,
        n_dense: int = 30,
    ):
        self.config = config
        self.blade_K = blade_K
        self.blade_n = blade_n
        self.blade_max_iter = blade_max_iter
        self.blade_tol_bc = blade_tol_bc
        self.blade_l1_lambda = blade_l1_lambda
        self.lgl_nodes_peak = lgl_nodes_peak
        self.lgl_nodes_coast = lgl_nodes_coast
        self.coast_threshold = coast_threshold
        self.dynamics_tol = dynamics_tol
        self.max_refinements = max_refinements
        self.blade_relax_alpha = blade_relax_alpha
        self.blade_trust_region = blade_trust_region
        self.n_dense = n_dense

    def solve(self) -> TrajectoryResult:
        """순차 최적화 실행.

        Returns
        -------
        TrajectoryResult
            solver_stats에 BLADE/LGL 상세 정보 포함.
        """
        from bezier_orbit.normalize import from_orbit
        from bezier_orbit.blade.orbit import (
            OrbitBC, BLADEOrbitProblem, solve_blade_scp,
        )

        config = self.config
        t0_total = _time.perf_counter()

        # ── 1. BLADE-SCP ──
        cu = from_orbit(config.a0)
        dep = OrbitBC(
            a=config.a0, e=config.e0,
            inc=0.0, raan=0.0, aop=0.0, ta=0.0,
        )
        arr = OrbitBC(
            a=config.af, e=config.ef,
            inc=float(np.radians(config.delta_i)),
            raan=0.0, aop=0.0, ta=0.0,
        )
        t_f_phys = config.T_max
        t_f_norm = t_f_phys / cu.TU
        u_max_norm = config.u_max / cu.AU

        prob = BLADEOrbitProblem(
            dep=dep, arr=arr,
            t_f=t_f_norm,
            K=self.blade_K, n=self.blade_n,
            u_max=u_max_norm,
            l1_lambda=self.blade_l1_lambda,
            canonical_units=cu,
            max_iter=self.blade_max_iter,
            tol_bc=self.blade_tol_bc,
            relax_alpha=self.blade_relax_alpha,
            trust_region=self.blade_trust_region,
            n_steps_per_seg=self.n_dense,
            coupling_order=1,
            coupling_from_start=True,
            ta_free=True,
            algebraic_drift=True,
        )

        t0_blade = _time.perf_counter()
        blade_result = solve_blade_scp(prob)
        blade_time = _time.perf_counter() - t0_blade

        if not blade_result.converged:
            # BLADE 수렴 실패 — 빈 결과 반환
            return TrajectoryResult(
                converged=False,
                cost=float("inf"),
                t=np.array([0.0, t_f_phys]),
                x=np.zeros((6, 2)),
                u=np.zeros((3, 2)),
                nu0=0.0, nuf=0.0,
                n_peaks=0, profile_class=0,
                T_f=t_f_phys,
                solver_stats={"blade_status": blade_result.status},
            )

        # ── 2. 구조적 피크 분류 ──
        seg_types, blade_n_peaks = blade_classify_segments(
            blade_result.p_segments,
            threshold=self.coast_threshold,
        )
        blade_profile_class = classify_profile(blade_n_peaks)

        # ── 3. Phase 구조 결정 ──
        phases = blade_phase_structure(
            seg_types,
            K=self.blade_K,
            t_f=t_f_phys,
            n_nodes_peak=self.lgl_nodes_peak,
            n_nodes_coast=self.lgl_nodes_coast,
        )

        # ── 4. BLADE dense 궤적 재구성 ──
        r0_phys, v0_phys = dep.at_time(0.0)
        x0_norm = np.concatenate([cu.nondim_pos(r0_phys), cu.nondim_vel(v0_phys)])

        t_dense, x_dense, u_dense = blade_to_dense_trajectory(
            blade_result.p_segments, self.blade_n, self.blade_K,
            t_f_phys, x0_norm, cu, self.n_dense,
        )

        # ── 5. Warm-start: BLADE → LGL 노드 보간 ──
        _, x_phases, u_phases = interpolate_blade_to_lgl(
            t_dense, x_dense, u_dense, phases,
        )

        # 초기 nu 추출 (BLADE 결과에서)
        ta_opt = blade_result.ta_opt if blade_result.ta_opt is not None else 0.0
        nu0_guess = 0.0
        nuf_guess = ta_opt

        # ── 6. Multi-Phase LGL 콜로케이션 (Pass 2) ──
        n_refinements = 0
        current_phases = phases

        for attempt in range(1 + self.max_refinements):
            lgl = MultiPhaseLGLCollocation(
                config, current_phases, T_fixed=t_f_phys,
            )
            result = lgl.solve(
                x_phases=x_phases, u_phases=u_phases,
                nu0_guess=nu0_guess, nuf_guess=nuf_guess,
            )

            if not result.converged:
                break

            # ── 7. 동역학 잔차 검증 ──
            max_residual, residuals = verify_dynamics(
                result.t, result.x, result.u, config,
            )

            if max_residual < self.dynamics_tol:
                break

            # 격자 조밀화
            if attempt < self.max_refinements:
                n_refinements += 1
                current_phases = _refine_phases(current_phases)
                # 기존 해를 새 격자에 재보간
                pb = [(p["t_start"], p["t_end"]) for p in current_phases]
                _, x_phases, u_phases = interpolate_blade_to_lgl(
                    result.t, result.x, result.u, current_phases,
                )
                nu0_guess = result.nu0
                nuf_guess = result.nuf

        # ── 8. Topological persistence 피크 탐지 (교차 검증) ──
        if result.converged:
            pb = [(p["t_start"], p["t_end"]) for p in current_phases]
            t_d, x_d, u_d = dense_output(
                result.t, result.x, result.u,
                n_points=300, phase_boundaries=pb,
            )
            u_mag_d = np.linalg.norm(u_d, axis=0)
            try:
                topo_n_peaks, _, _ = detect_peaks(t_d, u_mag_d, t_f_phys)
            except Exception:
                topo_n_peaks = -1
        else:
            t_d, x_d, u_d = result.t, result.x, result.u
            topo_n_peaks = -1

        total_time = _time.perf_counter() - t0_total

        # solver_stats 조립
        stats = {
            "method": "blade_collocation",
            "phase_boundaries": [(p["t_start"], p["t_end"]) for p in current_phases],
            # BLADE 상세
            "blade_bc_viol": blade_result.bc_violation,
            "blade_n_iter": blade_result.n_iter,
            "blade_K": self.blade_K,
            "blade_n": self.blade_n,
            "blade_l1_lambda": self.blade_l1_lambda,
            "blade_status": blade_result.status,
            "blade_cost": float(blade_result.cost),
            "blade_time": blade_time,
            "blade_seg_types": seg_types,
            # 분류 교차 검증
            "blade_n_peaks": blade_n_peaks,
            "blade_profile_class": blade_profile_class,
            "topo_n_peaks": topo_n_peaks,
            "classification_match": blade_n_peaks == topo_n_peaks,
            # 동역학 검증
            "dynamics_max_residual": max_residual if result.converged else float("inf"),
            "dynamics_verified": (
                max_residual < self.dynamics_tol if result.converged else False
            ),
            "n_refinements": n_refinements,
            # 시간
            "total_time": total_time,
        }

        return TrajectoryResult(
            converged=result.converged,
            cost=result.cost if result.converged else float("inf"),
            t=t_d if result.converged else result.t,
            x=x_d if result.converged else result.x,
            u=u_d if result.converged else result.u,
            nu0=result.nu0,
            nuf=result.nuf,
            n_peaks=blade_n_peaks,
            profile_class=blade_profile_class,
            T_f=t_f_phys,
            solver_stats=stats,
        )


def verify_dynamics(
    t: NDArray,
    x: NDArray,
    u: NDArray,
    config: TransferConfig,
    n_check_per_interval: int = 5,
) -> tuple[float, NDArray]:
    """콜로케이션 노드 간 중간점에서 동역학 잔차 계산.

    Parameters
    ----------
    t : (N,) — 시간 노드 [s]
    x : (6, N) — 상태
    u : (3, N) — 제어
    config : TransferConfig
    n_check_per_interval : int
        인접 노드 간 검증 점 수.

    Returns
    -------
    max_residual : float
        최대 동역학 잔차 (위치 norm, [km]).
    residuals : (M,) array
        각 검증점의 잔차.
    """
    from ..dynamics.eom import spacecraft_eom_numpy

    N = len(t)
    if N < 2:
        return 0.0, np.array([])

    x_interp = CubicSpline(t, x, axis=1)
    u_interp = CubicSpline(t, u, axis=1)

    residuals = []

    for i in range(N - 1):
        dt = t[i + 1] - t[i]
        if dt < 1e-12:
            continue

        for j in range(1, n_check_per_interval + 1):
            frac = j / (n_check_per_interval + 1)
            t_mid = t[i] + frac * dt

            # 보간으로 얻은 상태/제어
            x_mid_interp = x_interp(t_mid)  # (6,)

            # RK4 단일 스텝 전파: x(t_i) → x(t_mid)
            h = frac * dt
            x_start = x[:, i].copy()
            x_rk4 = _rk4_step(x_start, t[i], h, u_interp, spacecraft_eom_numpy)

            # 위치 잔차
            dr = np.linalg.norm(x_mid_interp[:3] - x_rk4[:3])
            residuals.append(dr)

    residuals = np.array(residuals) if residuals else np.array([0.0])
    return float(np.max(residuals)), residuals


def _rk4_step(
    x: NDArray, t0: float, h: float,
    u_interp, eom_func,
) -> NDArray:
    """RK4 단일 스텝.

    Parameters
    ----------
    x : (6,) — 초기 상태 [km, km/s]
    t0 : float — 시작 시각 [s]
    h : float — 시간 스텝 [s]
    u_interp : callable — u(t) → (3,)
    eom_func : callable — f(x, u) → (6,)
    """
    def f(t_val, x_val):
        u_val = u_interp(t_val)
        return eom_func(x_val, u_val)

    k1 = f(t0, x)
    k2 = f(t0 + h / 2, x + h / 2 * k1)
    k3 = f(t0 + h / 2, x + h / 2 * k2)
    k4 = f(t0 + h, x + h * k3)
    return x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def _refine_phases(phases: list[dict]) -> list[dict]:
    """격자 조밀화: 각 phase의 노드 수를 증가.

    peak: ×1.67 (15→25→42)
    coast: ×1.5 (8→12→18)
    """
    refined = []
    for p in phases:
        new = dict(p)
        if p["type"] == "peak":
            new["n_nodes"] = int(p["n_nodes"] * 5 / 3)
        else:
            new["n_nodes"] = int(p["n_nodes"] * 3 / 2)
        refined.append(new)
    return refined
