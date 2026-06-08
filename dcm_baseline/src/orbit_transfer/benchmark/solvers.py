"""기법별 솔버 래퍼.

각 솔버는 TransferConfig를 받아 BenchmarkResult를 반환한다.
"""

from __future__ import annotations
import time as _time
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
    """Two-Pass Collocation 솔버 (TwoPassOptimizer 래퍼).

    Parameters
    ----------
    x_init : ndarray, optional  외부 초기치 상태 (6, M)
    u_init : ndarray, optional  외부 초기치 제어 (3, M)
    t_init : ndarray, optional  외부 초기치 시간 (M,)
    l1_lambda : float  ℓ₁ 정규화 강도 (기본 0, 코스팅 유도)
    """

    def __init__(self, x_init=None, u_init=None, t_init=None, l1_lambda=0.0):
        self.x_init = x_init
        self.u_init = u_init
        self.t_init = t_init
        self.l1_lambda = l1_lambda

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

        opt = TwoPassOptimizer(config, l1_lambda=self.l1_lambda)
        res = opt.solve(
            x_init=self.x_init,
            u_init=self.u_init,
            t_init=self.t_init,
        )

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


# ===========================================================================
# BLADE 솔버
# ===========================================================================

class BladeSolver:
    """BLADE-SCP (Bernstein Local Adaptive Degree Elements) 솔버.

    bezier-orbit-transfer-scp 패키지의 BLADE 궤도전이 SCP를
    benchmark 인터페이스에 맞춰 래핑한다.

    Parameters
    ----------
    K : int          세그먼트 수 (기본 12)
    n : int          세그먼트 차수 (기본 2)
    max_iter : int   SCP 최대 반복 (기본 50)
    tol_bc : float   경계조건 허용치 (기본 1e-3)
    l1_lambda : float  ℓ₁ 정규화 강도 (기본 0, 코스팅 유도)
    t_f_override : float | None  비행시간 [s] 고정값. None이면 T_max 사용
    n_dense : int    궤적 출력 밀도 — 세그먼트당 점 수 (기본 30)
    relax_alpha : float  SCP 스텝 감쇠 (기본 0.3)
    trust_region : float  신뢰영역 반경 (기본 5.0)
    """

    def __init__(
        self,
        K: int = 12,
        n: int = 2,
        max_iter: int = 50,
        tol_bc: float = 1e-3,
        l1_lambda: float = 0.0,
        t_f_override: float | None = None,
        n_dense: int = 30,
        relax_alpha: float = 0.5,
        trust_region: float = 5.0,
        coupling_order: int = 1,
        ta_free: bool = True,
        phase_search: bool = False,
        phase_n_grid: int = 18,
        algebraic_drift: bool = True,
    ):
        self.K = K
        self.n = n
        self.max_iter = max_iter
        self.tol_bc = tol_bc
        self.l1_lambda = l1_lambda
        self.t_f_override = t_f_override
        self.n_dense = n_dense
        self.relax_alpha = relax_alpha
        self.trust_region = trust_region
        self.coupling_order = coupling_order
        self.ta_free = ta_free
        self.phase_search = phase_search
        self.phase_n_grid = phase_n_grid
        self.algebraic_drift = algebraic_drift
        self.validate = False

    def solve(self, config: TransferConfig) -> BenchmarkResult:
        """BLADE-SCP 궤도전이 궤적 계산.

        Parameters
        ----------
        config : TransferConfig

        Returns
        -------
        BenchmarkResult  (is_impulsive=False)
        """
        from bezier_orbit.normalize import from_orbit
        from bezier_orbit.blade.orbit import (
            OrbitBC, BLADEOrbitProblem, solve_blade_scp,
            blade_phase_search, _propagate_blade_reference,
        )
        from bezier_orbit.bezier.basis import bernstein
        from ..classification.peak_detection import detect_peaks
        from ..classification.classifier import classify_profile

        # ── 1. 정규화 단위 ──
        cu = from_orbit(config.a0)

        # ── 2. 경계 궤도 정의 ──
        dep = OrbitBC(
            a=config.a0, e=config.e0,
            inc=0.0, raan=0.0, aop=0.0, ta=0.0,
        )
        arr = OrbitBC(
            a=config.af, e=config.ef,
            inc=float(np.radians(config.delta_i)),
            raan=0.0, aop=0.0, ta=0.0,
        )

        # ── 3. 비행시간 / 추력 제한 변환 ──
        if self.t_f_override is not None:
            t_f_phys = self.t_f_override  # [s]
        else:
            t_f_phys = config.T_max  # [s]
        t_f_norm = t_f_phys / cu.TU

        u_max_norm = config.u_max / cu.AU

        # ── 4. BLADE 문제 생성 + 풀기 ──
        prob = BLADEOrbitProblem(
            dep=dep, arr=arr,
            t_f=t_f_norm,
            K=self.K, n=self.n,
            u_max=u_max_norm,
            l1_lambda=self.l1_lambda,
            canonical_units=cu,
            max_iter=self.max_iter,
            tol_bc=self.tol_bc,
            relax_alpha=self.relax_alpha,
            trust_region=self.trust_region,
            n_steps_per_seg=self.n_dense,
            coupling_order=self.coupling_order,
            coupling_from_start=self.ta_free,  # ta_free 시 coupling 필수
            ta_free=self.ta_free,
            algebraic_drift=self.algebraic_drift,
            validate=self.validate,
        )

        t0 = _time.perf_counter()
        if self.phase_search and not self.ta_free:
            blade_result, ta_opt = blade_phase_search(
                prob, n_grid=self.phase_n_grid, refine=True,
            )
        else:
            blade_result = solve_blade_scp(prob)
            ta_opt = blade_result.ta_opt if blade_result.ta_opt is not None else 0.0
        solve_time = _time.perf_counter() - t0

        # ── 5. 궤적 재구성 (정규화 좌표) ──
        K, n = prob.K, prob.n
        deltas = np.full(K, 1.0 / K)

        r0_phys, v0_phys = dep.at_time(0.0)
        x0_norm = np.concatenate([cu.nondim_pos(r0_phys), cu.nondim_vel(v0_phys)])

        seg_trajs = _propagate_blade_reference(
            blade_result.p_segments, n, K, deltas, t_f_norm,
            x0_norm, cu.R_earth_star, self.n_dense,
        )

        # 세그먼트 연결 (마지막 세그먼트 제외하고 끝점 제거)
        x_norm = np.vstack([seg[:-1] for seg in seg_trajs[:-1]] + [seg_trajs[-1]])
        N = x_norm.shape[0]

        # ── 6. 추력 프로파일 재구성 ──
        n_per = seg_trajs[0].shape[0]
        u_norm = np.zeros((N, 3))
        idx = 0
        for k in range(K):
            pk = blade_result.p_segments[k]
            pts = n_per - 1 if k < K - 1 else n_per
            for j in range(pts):
                tau_local = j / (n_per - 1)
                B = bernstein(n, tau_local)
                u_norm[idx] = B @ pk
                idx += 1

        # ── 7. 물리 단위 변환 ──
        x_phys = np.zeros_like(x_norm)
        x_phys[:, :3] = x_norm[:, :3] * cu.DU   # [km]
        x_phys[:, 3:] = x_norm[:, 3:] * cu.VU   # [km/s]
        u_phys = u_norm * cu.AU                   # [km/s²]

        t_arr = np.linspace(0.0, t_f_phys, N)    # [s]

        # (N, 6) → (6, N),  (N, 3) → (3, N)
        x_out = x_phys.T
        u_out = u_phys.T

        # ── 8. 피크 분류 ──
        u_mag = np.linalg.norm(u_out, axis=0)
        try:
            n_peaks, peak_times, peak_widths = detect_peaks(t_arr, u_mag, t_f_phys)
        except Exception:
            n_peaks = 0
            peak_times = np.array([])
            peak_widths = np.array([])
        profile_class = classify_profile(n_peaks)

        # 코스팅 세그먼트 수
        seg_norms = [float(np.linalg.norm(pk)) for pk in blade_result.p_segments]
        n_coasting = sum(1 for nm in seg_norms if nm < 0.01)

        # ── 9. BenchmarkResult 패킹 ──
        result = BenchmarkResult(
            method="blade",
            converged=blade_result.converged,
            t=t_arr,
            x=x_out,
            u=u_out,
            is_impulsive=False,
            impulses=[],
            extra={
                "n_peaks": n_peaks,
                "profile_class": profile_class,
                "T_f": t_f_phys,
                "blade_bc_viol": blade_result.bc_violation,
                "blade_n_iter": blade_result.n_iter,
                "blade_K": K,
                "blade_n": n,
                "blade_l1_lambda": self.l1_lambda,
                "blade_status": blade_result.status,
                "blade_n_coasting": n_coasting,
                "blade_seg_norms": seg_norms,
                "blade_solve_time": solve_time,
                "blade_ta_opt": float(np.degrees(ta_opt)) if ta_opt else None,
                "validation_passed": (
                    blade_result.validation.passed
                    if blade_result.validation is not None else None
                ),
            },
        )
        result.metrics = compute_metrics(result, config)
        result.metrics["cost_l2_solver"] = float(blade_result.cost)
        result.metrics["blade_bc_viol"] = blade_result.bc_violation
        return result


# ===========================================================================
# BLADE-Collocation 순차 솔버
# ===========================================================================

class BladeCollocationSolver:
    """BLADE → LGL 콜로케이션 순차 최적화 (BladeCollocationOptimizer 래퍼).

    Parameters
    ----------
    blade_K : int
        BLADE 세그먼트 수 (기본 12).
    blade_n : int
        BLADE 세그먼트 차수 (기본 2).
    blade_l1_lambda : float
        ℓ₁ 정규화 강도 (기본 0.0).
    lgl_nodes_peak : int
        Peak phase LGL 노드 수 (기본 15).
    lgl_nodes_coast : int
        Coast phase LGL 노드 수 (기본 8).
    coast_threshold : float
        Coast 판정 임계값 (기본 0.05).
    dynamics_tol : float
        동역학 잔차 허용치 (기본 1e-6).
    max_refinements : int
        최대 격자 조밀화 횟수 (기본 2).
    """

    def __init__(
        self,
        blade_K: int = 12,
        blade_n: int = 2,
        blade_l1_lambda: float = 0.0,
        lgl_nodes_peak: int = 15,
        lgl_nodes_coast: int = 8,
        coast_threshold: float = 0.05,
        dynamics_tol: float = 1e-6,
        max_refinements: int = 2,
        **kwargs,
    ):
        self.blade_K = blade_K
        self.blade_n = blade_n
        self.blade_l1_lambda = blade_l1_lambda
        self.lgl_nodes_peak = lgl_nodes_peak
        self.lgl_nodes_coast = lgl_nodes_coast
        self.coast_threshold = coast_threshold
        self.dynamics_tol = dynamics_tol
        self.max_refinements = max_refinements
        self._extra_kwargs = kwargs

    def solve(self, config: TransferConfig) -> BenchmarkResult:
        """BLADE-Collocation 순차 최적화 실행.

        Parameters
        ----------
        config : TransferConfig

        Returns
        -------
        BenchmarkResult  (is_impulsive=False)
        """
        from ..optimizer.blade_collocation import BladeCollocationOptimizer
        from ..classification.peak_detection import detect_peaks
        from ..classification.classifier import classify_profile

        opt = BladeCollocationOptimizer(
            config,
            blade_K=self.blade_K,
            blade_n=self.blade_n,
            blade_l1_lambda=self.blade_l1_lambda,
            lgl_nodes_peak=self.lgl_nodes_peak,
            lgl_nodes_coast=self.lgl_nodes_coast,
            coast_threshold=self.coast_threshold,
            dynamics_tol=self.dynamics_tol,
            max_refinements=self.max_refinements,
            **self._extra_kwargs,
        )

        res = opt.solve()

        stats = res.solver_stats or {}

        result = BenchmarkResult(
            method="blade_collocation",
            converged=res.converged,
            t=res.t,
            x=res.x,
            u=res.u,
            is_impulsive=False,
            impulses=[],
            extra={
                "n_peaks": res.n_peaks,
                "profile_class": res.profile_class,
                "T_f": res.T_f,
                # BLADE 상세
                "blade_bc_viol": stats.get("blade_bc_viol", float("nan")),
                "blade_n_iter": stats.get("blade_n_iter", -1),
                "blade_K": self.blade_K,
                "blade_n": self.blade_n,
                "blade_status": stats.get("blade_status", "unknown"),
                "blade_time": stats.get("blade_time", float("nan")),
                # 분류 교차 검증
                "blade_n_peaks": stats.get("blade_n_peaks", -1),
                "topo_n_peaks": stats.get("topo_n_peaks", -1),
                "classification_match": stats.get("classification_match", False),
                # 동역학 검증
                "dynamics_max_residual": stats.get("dynamics_max_residual", float("nan")),
                "dynamics_verified": stats.get("dynamics_verified", False),
                "n_refinements": stats.get("n_refinements", 0),
                # 시간
                "total_time": stats.get("total_time", float("nan")),
            },
        )
        result.metrics = compute_metrics(result, config)
        if res.converged:
            result.metrics["cost_l2_solver"] = float(res.cost)
        return result
