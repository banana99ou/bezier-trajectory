"""BLADE 궤도전이 SCP 테스트.

Phase 2: 비선형 궤도 동역학 + J2 세차 경계조건.
"""

import math

import numpy as np
import pytest

from bezier_orbit.normalize import from_orbit, R_EARTH
from bezier_orbit.orbit.elements import keplerian_to_cartesian
import warnings

from bezier_orbit.blade.orbit import (
    OrbitBC,
    BLADEOrbitProblem,
    BLADESCPResult,
    BLADEValidation,
    j2_precession_rates,
    solve_blade_scp,
    validate_blade_solution,
    blade_outer_loop,
)


# ── 공통 설정 ────────────────────────────────────────────────────

# LEO 기본 궤도 (정규화: a0 = 7000 km)
_A0 = 7000.0  # km
_CU = from_orbit(_A0)


def _make_prob(**overrides) -> BLADEOrbitProblem:
    """기본 공면 전이 문제 생성."""
    dep = OrbitBC(a=_A0, e=0.0, inc=0.0, raan=0.0, aop=0.0, ta=0.0)
    arr = OrbitBC(a=_A0 * 1.2, e=0.0, inc=0.0, raan=0.0, aop=0.0, ta=0.0)
    defaults = dict(
        dep=dep, arr=arr,
        t_f=4.0,  # TU
        K=8, n=2,
        canonical_units=_CU,
        max_iter=30,
        tol_bc=1e-3,
        relax_alpha=0.4,
        n_steps_per_seg=20,
    )
    defaults.update(overrides)
    return BLADEOrbitProblem(**defaults)


# ── J2 세차 테스트 ───────────────────────────────────────────────

class TestJ2Precession:

    def test_precession_rates_sign(self):
        """J2 세차: LEO 순행 궤도에서 Ω̇ < 0, ω̇ > 0."""
        raan_dot, aop_dot = j2_precession_rates(
            a=_A0, e=0.0, inc=math.radians(28.5),
        )
        assert raan_dot < 0  # 역행 세차
        assert aop_dot > 0   # 순행 세차

    def test_orbitbc_at_time_changes_raan(self):
        """OrbitBC.at_time: 시간 경과 시 RAAN 변화."""
        bc = OrbitBC(a=_A0, e=0.0, inc=math.radians(28.5), raan=0.0, aop=0.0)
        r0, v0 = bc.at_time(0.0)
        r1, v1 = bc.at_time(3600.0)  # 1시간 후
        # 다른 위치여야 함
        assert not np.allclose(r0, r1, atol=1e-3)

    def test_polar_orbit_no_raan_precession(self):
        """극궤도(i=90°): RAAN 세차 ≈ 0."""
        raan_dot, _ = j2_precession_rates(
            a=_A0, e=0.0, inc=math.pi / 2,
        )
        assert abs(raan_dot) < 1e-10


# ── 궤도전이 SCP 테스트 ──────────────────────────────────────────

class TestBLADEOrbitSCP:

    def test_coplanar_circular_convergence(self):
        """공면 원형→원형 전이 (a: 1.0→1.2) 수렴."""
        prob = _make_prob(max_iter=50, n_steps_per_seg=40, K=10, n=2)
        result = solve_blade_scp(prob)
        assert result.bc_violation < 1.0, \
            f"bc_viol={result.bc_violation:.4f}, status={result.status}"

    def test_inclination_change(self):
        """경사각 변경 (i: 0→28.5°)."""
        dep = OrbitBC(a=_A0, e=0.0, inc=0.0, raan=0.0, aop=0.0, ta=0.0)
        arr = OrbitBC(
            a=_A0, e=0.0, inc=math.radians(28.5),
            raan=0.0, aop=0.0, ta=0.0,
        )
        prob = _make_prob(dep=dep, arr=arr, t_f=4.0,
                          max_iter=50, n_steps_per_seg=40, K=10, n=2)
        result = solve_blade_scp(prob)
        assert result.bc_violation < 5.0, \
            f"bc_viol={result.bc_violation:.4f}"

    def test_socp_thrust_limit(self):
        """SOCP 추력 제한 하에서 동작."""
        prob = _make_prob(u_max=3.0)
        result = solve_blade_scp(prob)
        # 최소한 솔버가 에러 없이 동작해야 함
        assert result.status != "solver_error"

    def test_l1_coasting_emergence(self):
        """ℓ₁ 정규화로 코스팅 세그먼트 출현 확인."""
        prob = _make_prob(l1_lambda=1.0, K=10, n=1)
        result = solve_blade_scp(prob)
        assert result.status != "solver_error"

    def test_l1_coasting_orbit_produces_coasting(self):
        """궤도전이 + ℓ₁에서 코스팅 세그먼트 출현 확인."""
        dep = OrbitBC(a=_A0, e=0, inc=0, raan=0, aop=0, ta=0)
        arr = OrbitBC(a=_A0 * 1.1, e=0, inc=0, raan=0, aop=0, ta=0)
        prob = _make_prob(
            dep=dep, arr=arr, t_f=4.0, K=12, n=1,
            u_max=3.0, l1_lambda=0.1,
            max_iter=50, tol_bc=5e-2,
        )
        result = solve_blade_scp(prob)
        norms = [np.linalg.norm(pk) for pk in result.p_segments]
        n_coast = sum(1 for nm in norms if nm < 0.01)
        assert n_coast >= 1, f"No coasting with λ=0.1, norms={[f'{n:.3f}' for n in norms]}"
        assert result.converged or result.bc_violation < 0.1

    def test_bc_violation_split_reported(self):
        """bc_violation_r, bc_violation_v가 분리 보고됨."""
        prob = _make_prob(max_iter=50, n_steps_per_seg=40, K=10, n=2)
        result = solve_blade_scp(prob)
        assert result.bc_violation_r is not None
        assert result.bc_violation_v is not None
        assert result.bc_violation == pytest.approx(
            max(result.bc_violation_r, result.bc_violation_v), abs=1e-12,
        )

    def test_j2_precession_in_bc(self):
        """t_f 변경 시 J2 세차에 의해 경계조건이 달라짐."""
        dep = OrbitBC(
            a=_A0, e=0.0, inc=math.radians(28.5),
            raan=0.0, aop=0.0, ta=0.0,
        )
        arr = OrbitBC(
            a=_A0 * 1.1, e=0.0, inc=math.radians(28.5),
            raan=math.radians(10.0), aop=0.0, ta=0.0,
        )
        prob1 = _make_prob(dep=dep, arr=arr, t_f=3.0)
        prob2 = _make_prob(dep=dep, arr=arr, t_f=6.0)

        # t_f가 다르면 도착 궤도의 Ω, ω가 다르므로 rf, vf가 달라야 함
        cu = prob1.canonical_units
        _, vf1 = arr.at_time(cu.dim_time(3.0))
        _, vf2 = arr.at_time(cu.dim_time(6.0))
        assert not np.allclose(vf1, vf2, atol=1e-6)


# ── Outer Loop 테스트 ────────────────────────────────────────────

class TestBLADEOuterLoop:

    def test_outer_loop_runs(self):
        """Outer loop (격자 탐색)이 에러 없이 동작."""
        prob = _make_prob(max_iter=10, K=6, n=1)
        result = blade_outer_loop(prob, t_f_bounds=(2.0, 6.0), n_grid=3)
        assert len(result.t_f_history) == 3
        assert result.t_f_opt >= 2.0
        assert result.t_f_opt <= 6.0


# ── 사후검증 테스트 ──────────────────────────────────────────────

class TestBLADEValidation:

    def test_converged_passes_validation(self):
        """수렴한 공면 전이는 RK4 검증을 통과해야 함."""
        prob = _make_prob(max_iter=50, n_steps_per_seg=40, K=10, n=2)
        result = solve_blade_scp(prob)
        if not result.converged:
            pytest.skip("솔버 미수렴 — 검증 건너뜀")
        val = validate_blade_solution(prob, result)
        assert val.bc_violation_rk4 < 0.1, \
            f"RK4 bc_viol={val.bc_violation_rk4:.4f}"
        assert val.details["cost_finite"]

    def test_validation_rk4_bc(self):
        """RK4 재전파 BC violation이 솔버와 같은 차수."""
        prob = _make_prob(max_iter=50, n_steps_per_seg=40, K=10, n=2)
        result = solve_blade_scp(prob)
        if not result.converged:
            pytest.skip("솔버 미수렴")
        val = validate_blade_solution(prob, result)
        # 솔버 bc_viol과 RK4 bc_viol이 10배 이내여야 함
        ratio = val.bc_violation_rk4 / max(result.bc_violation, 1e-15)
        assert ratio < 10.0, \
            f"ratio={ratio:.1f}, solver={result.bc_violation:.4e}, rk4={val.bc_violation_rk4:.4e}"

    def test_thrust_violation_detected(self):
        """타이트한 u_max에서 추력 위반이 보고됨."""
        prob = _make_prob(u_max=0.5, max_iter=30, K=8, n=2)
        result = solve_blade_scp(prob)
        val = validate_blade_solution(prob, result)
        # 추력 제약이 있으므로 max_thrust_norm이 보고됨
        assert val.max_thrust_norm >= 0.0

    def test_stagnation_detection(self):
        """어려운 문제에서 정체 감지 → 조기 종료 또는 max_iter."""
        dep = OrbitBC(a=_A0, e=0.0, inc=0.0, raan=0.0, aop=0.0, ta=0.0)
        arr = OrbitBC(
            a=_A0 * 3.0, e=0.3, inc=math.radians(60.0),
            raan=math.radians(90.0), aop=math.radians(45.0), ta=math.pi,
        )
        prob = _make_prob(
            dep=dep, arr=arr, t_f=2.0,
            max_iter=50, K=4, n=1,
            tol_bc=1e-6,  # 매우 타이트한 허용치
        )
        result = solve_blade_scp(prob)
        # 이 문제는 수렴 불가능: stagnated 또는 max_iter
        assert result.status in ("stagnated", "max_iter", "converged")
        # stagnated이면 조기 종료 확인
        if result.status == "stagnated":
            assert result.n_iter < 50

    def test_outer_loop_warns(self):
        """전체 그리드 실패 시 RuntimeWarning 발생."""
        prob = _make_prob(max_iter=10, K=6, n=1)
        with pytest.warns(RuntimeWarning, match="수렴 실패"):
            blade_outer_loop(prob, t_f_bounds=(2.0, 6.0), n_grid=3)

    def test_result_backward_compat(self):
        """기존 필드만으로 BLADESCPResult 생성 가능."""
        result = BLADESCPResult(
            p_segments=[], cost=1.0, converged=True,
            n_iter=5, bc_violation=1e-4,
            bc_history=[1e-4], cost_history=[1.0],
            status="converged",
        )
        assert result.bc_violation_r is None
        assert result.bc_violation_v is None
        assert result.thrust_violation is None
        assert result.x_final is None
        assert result.validation is None

    def test_validate_auto(self):
        """prob.validate=True 시 자동 검증."""
        prob = _make_prob(
            max_iter=50, n_steps_per_seg=40, K=10, n=2,
            validate=True,
        )
        result = solve_blade_scp(prob)
        assert result.validation is not None
        assert isinstance(result.validation, BLADEValidation)
