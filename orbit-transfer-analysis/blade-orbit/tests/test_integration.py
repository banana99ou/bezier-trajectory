"""통합 테스트: 전체 파이프라인."""

import numpy as np
import pytest

from bezier_orbit.normalize import from_orbit
from bezier_orbit.orbit.elements import keplerian_to_cartesian, cartesian_to_keplerian
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop
from bezier_orbit.scp.outer_loop import grid_search
from bezier_orbit.bezier.basis import bernstein_eval
from bezier_orbit.bezier.constraints import check_thrust_feasibility
from bezier_orbit.db.store import SimulationStore


class TestFullPipeline:
    """정규화 → 문제 정의 → SCP → 결과 저장 전체 흐름."""

    def test_coplanar_transfer_qp(self):
        """동일 평면 원형→타원 전이 (QP, 추력 무제약)."""
        # 1. 정규화
        a0_km = 6778.0  # LEO
        cu = from_orbit(a0_km)

        # 2. 경계조건 (정규화 단위)
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.1, 0.05, 0.0, 0.0, 0.0, np.pi, mu=1.0)

        # 3. 문제 정의
        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=8, u_max=None,
            perturbation_level=0,
            max_iter=10, tol_ctrl=1e-4, tol_bc=1e-2,
        )

        # 4. SCP 풀기
        result = solve_inner_loop(prob)

        # 기본 검증
        assert result.cost > 0.0
        assert result.cost < float("inf")
        assert result.P_u_opt.shape == (9, 3)

        # 5. 물리 단위 복원
        P_u_phys = cu.dim_accel(result.P_u_opt)
        # 추력 가속도가 합리적 범위 (O(10^{-4}~10^{-2}) km/s^2)
        max_accel = np.max(np.abs(P_u_phys))
        assert max_accel < 1.0  # km/s^2

    def test_coplanar_transfer_socp(self):
        """동일 평면 전이 (SOCP, 추력 제약)."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.1, 0.05, 0.0, 0.0, 0.0, np.pi, mu=1.0)

        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=8, u_max=5.0,
            max_iter=10, tol_ctrl=1e-4, tol_bc=1e-2,
        )

        result = solve_inner_loop(prob)
        assert result.cost > 0.0
        assert result.cost < float("inf")

    def test_lifting_equivalence_numerical(self):
        """Z = t_f · P_u 수치 동등성 확인 (보고서 007 section 4)."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.1, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)

        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=8, u_max=None,
            max_iter=5,
        )

        result = solve_inner_loop(prob)

        # Z = t_f · P_u 관계 검증
        Z_from_Pu = prob.t_f * result.P_u_opt
        np.testing.assert_allclose(result.Z_opt, Z_from_Pu, atol=1e-10)

    def test_save_load_roundtrip(self):
        """전체 파이프라인 결과를 DuckDB에 저장/로드."""
        cu = from_orbit(6778.0)
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.1, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)

        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=8, u_max=None,
            max_iter=5,
        )
        result = solve_inner_loop(prob)

        with SimulationStore(":memory:") as store:
            sim_id = store.save_simulation(prob, result, cu)

            # 로드 후 비용 확인
            loaded = store.load_simulation(sim_id)
            assert loaded["cost"] == pytest.approx(result.cost)
            assert loaded["bezier_N"] == 8

            # SCP 이력 확인
            history = store.load_scp_history(sim_id)
            assert len(history) == len(result.cost_history)

    def test_grid_search_integration(self):
        """격자 탐색 전체 흐름."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.1, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)

        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=3.0, N=8, u_max=None,
            max_iter=3, tol_ctrl=1e-3, tol_bc=1e-2,
        )

        t_f_grid = np.array([3.0, 4.0, 5.0])
        result = grid_search(prob, t_f_grid)

        assert len(result.cost_history) == 3
        # 비용이 유한한 t_f가 하나는 있어야 함
        finite_costs = [c for c in result.cost_history if c < float("inf")]
        assert len(finite_costs) > 0
