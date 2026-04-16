"""DuckDB 저장/로드 테스트."""

import numpy as np
import pytest

from bezier_orbit.normalize import from_orbit
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import SCPResult
from bezier_orbit.db.store import SimulationStore


@pytest.fixture
def store():
    """인메모리 DuckDB 저장소."""
    with SimulationStore(":memory:") as s:
        yield s


@pytest.fixture
def sample_data():
    """테스트용 샘플 문제/결과."""
    cu = from_orbit(6778.0)
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)

    prob = SCPProblem(
        r0=r0, v0=v0, rf=rf, vf=vf,
        t_f=4.0, N=8, u_max=None,
        perturbation_level=0, max_iter=10,
    )

    N = 8
    result = SCPResult(
        Z_opt=np.random.randn(N + 1, 3),
        P_u_opt=np.random.randn(N + 1, 3),
        cost=0.42,
        n_iter=5,
        converged=True,
        bc_violation=1e-7,
        ctrl_change_history=[1.0, 0.5, 0.1, 0.01, 0.001],
        cost_history=[1.0, 0.8, 0.5, 0.43, 0.42],
        bc_violation_history=[0.5, 0.2, 0.05, 0.001, 1e-7],
        trust_radius_history=[10.0, 10.0, 15.0, 22.5, 33.75],
        convergence_reason="ctrl",
        solve_time_s=1.23,
        solver_used="SCS",
    )
    return cu, prob, result


class TestSimulationStore:
    """저장/로드 왕복."""

    def test_save_and_load(self, store, sample_data):
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        loaded = store.load_simulation(sim_id)
        assert loaded["sim_id"] == sim_id
        assert loaded["a0_km"] == pytest.approx(cu.a0)
        assert loaded["DU_km"] == pytest.approx(cu.DU)
        assert loaded["TU_s"] == pytest.approx(cu.TU)
        assert loaded["cost"] == pytest.approx(0.42)
        assert loaded["converged"] is True
        assert loaded["n_iter"] == 5

    def test_extended_columns(self, store, sample_data):
        """확장된 컬럼 저장/로드."""
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        loaded = store.load_simulation(sim_id)
        assert loaded["drift_method"] == "bernstein"
        assert loaded["use_coupling"] is True
        assert loaded["solver_used"] == "SCS"
        assert loaded["convergence_reason"] == "ctrl"
        assert loaded["solve_time_s"] == pytest.approx(1.23)
        assert loaded["tol_ctrl"] == pytest.approx(prob.tol_ctrl)
        assert loaded["tol_bc"] == pytest.approx(prob.tol_bc)
        assert loaded["trust_region"] == pytest.approx(prob.trust_region)
        assert loaded["relax_alpha"] == pytest.approx(prob.relax_alpha)
        assert loaded["max_jn_degree"] == prob.max_jn_degree
        assert loaded["Cd_A_over_m"] == pytest.approx(prob.Cd_A_over_m)

    def test_roundtrip_boundary(self, store, sample_data):
        """경계조건 왕복."""
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        loaded = store.load_simulation(sim_id)
        np.testing.assert_allclose(
            [loaded["r0_x"], loaded["r0_y"], loaded["r0_z"]],
            prob.r0, atol=1e-12,
        )
        np.testing.assert_allclose(
            [loaded["vf_x"], loaded["vf_y"], loaded["vf_z"]],
            prob.vf, atol=1e-12,
        )

    def test_scp_history(self, store, sample_data):
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        history = store.load_scp_history(sim_id)
        assert len(history) == 5
        assert history[0]["iteration"] == 1
        assert history[-1]["cost"] == pytest.approx(0.42)
        assert history[-1]["ctrl_change"] == pytest.approx(0.001)

    def test_scp_history_extended(self, store, sample_data):
        """확장된 SCP 이력 (bc_violation, trust_radius)."""
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        history = store.load_scp_history(sim_id)
        assert history[0]["bc_violation"] == pytest.approx(0.5)
        assert history[-1]["bc_violation"] == pytest.approx(1e-7)
        assert history[0]["trust_radius"] == pytest.approx(10.0)
        assert history[-1]["trust_radius"] == pytest.approx(33.75)

    def test_list_simulations(self, store, sample_data):
        cu, prob, result = sample_data
        store.save_simulation(prob, result, cu)
        store.save_simulation(prob, result, cu)

        sims = store.list_simulations()
        assert len(sims) == 2
        assert "drift_method" in sims[0]
        assert "solve_time_s" in sims[0]

    def test_save_trajectory(self, store, sample_data):
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        n_pts = 100
        tau = np.linspace(0, 1, n_pts)
        states = np.random.randn(n_pts, 6)
        u_arr = np.random.randn(n_pts, 3)

        traj_id = store.save_trajectory(
            sim_id, tau, states, u_arr,
            Z=result.Z_opt, P_u=result.P_u_opt,
        )

        loaded = store.load_trajectory(sim_id)
        assert loaded is not None
        assert loaded["traj_id"] == traj_id
        np.testing.assert_allclose(loaded["tau"], tau, atol=1e-12)
        np.testing.assert_allclose(loaded["rx"], states[:, 0], atol=1e-12)
        np.testing.assert_allclose(loaded["vz"], states[:, 5], atol=1e-12)
        np.testing.assert_allclose(loaded["ux"], u_arr[:, 0], atol=1e-12)

    def test_trajectory_control_points(self, store, sample_data):
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        tau = np.linspace(0, 1, 10)
        states = np.random.randn(10, 6)
        store.save_trajectory(sim_id, tau, states, Z=result.Z_opt)

        loaded = store.load_trajectory(sim_id)
        Z_loaded = loaded["Z_flat"].reshape(result.Z_opt.shape)
        np.testing.assert_allclose(Z_loaded, result.Z_opt, atol=1e-12)

    def test_missing_simulation(self, store):
        with pytest.raises(ValueError):
            store.load_simulation(9999)

    def test_missing_trajectory(self, store):
        result = store.load_trajectory(9999)
        assert result is None


class TestOuterLoopStore:
    """Outer loop 저장/로드."""

    def test_save_and_load_outer_loop(self, store, sample_data):
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        # 가짜 OuterLoopResult 생성
        from types import SimpleNamespace
        outer = SimpleNamespace(
            t_f_opt=4.0,
            inner_result=result,
            t_f_history=[3.0, 4.0, 5.0],
            cost_history=[0.5, 0.42, 0.45],
            all_results=[result, result, result],
        )

        sweep_id = store.save_outer_loop(sim_id, "grid", (3.0, 5.0), outer)
        loaded = store.load_outer_loop(sim_id)
        assert loaded is not None
        assert loaded["sweep_id"] == sweep_id
        assert loaded["search_method"] == "grid"
        assert loaded["t_f_min"] == pytest.approx(3.0)
        assert loaded["t_f_max"] == pytest.approx(5.0)
        assert loaded["n_evaluations"] == 3
        assert len(loaded["t_f_values"]) == 3
        assert len(loaded["costs"]) == 3

    def test_missing_outer_loop(self, store):
        result = store.load_outer_loop(9999)
        assert result is None


class TestParamSweepStore:
    """파라미터 스윕 저장."""

    def test_save_param_sweep(self, store, sample_data):
        cu, prob, result = sample_data
        sim_id = store.save_simulation(prob, result, cu)

        sweep_id = store.save_param_sweep(sim_id, "N", 12.0, result)
        assert sweep_id >= 1

        rows = store.con.execute(
            "SELECT * FROM param_sweep WHERE sim_id = ?", [sim_id]
        ).fetchall()
        assert len(rows) == 1


class TestBLADEStore:
    """BLADE 저장/로드."""

    def test_save_and_load_blade(self, store):
        from bezier_orbit.blade.orbit import (
            BLADEOrbitProblem, BLADESCPResult, BLADEValidation, OrbitBC,
        )
        from bezier_orbit.normalize import from_orbit

        cu = from_orbit(6778.0)
        prob = BLADEOrbitProblem(
            dep=OrbitBC(a=6778.0, e=0.0, inc=0.0, raan=0.0, aop=0.0),
            arr=OrbitBC(a=7178.0, e=0.0, inc=0.1, raan=0.0, aop=0.0),
            t_f=5.0, K=10, n=2, u_max=0.01,
            canonical_units=cu,
        )

        result = BLADESCPResult(
            p_segments=[np.zeros((3, 3)) for _ in range(10)],
            cost=0.123,
            converged=True,
            n_iter=8,
            bc_violation=1e-5,
            bc_history=[0.1, 0.05, 0.01, 0.005, 0.001, 5e-4, 1e-4, 1e-5],
            cost_history=[0.5, 0.3, 0.2, 0.15, 0.13, 0.125, 0.124, 0.123],
            status="optimal",
            bc_violation_r=5e-6,
            bc_violation_v=5e-6,
            thrust_violation=0.0,
            validation=BLADEValidation(
                bc_violation_rk4=2e-5,
                bc_violation_r=1e-5,
                bc_violation_v=1e-5,
                max_thrust_norm=0.009,
                thrust_violation=0.0,
                energy_error=1e-8,
                passed=True,
                details={},
            ),
        )

        blade_id = store.save_blade_simulation(prob, result, cu)

        loaded = store.load_blade_simulation(blade_id)
        assert loaded["blade_id"] == blade_id
        assert loaded["cost"] == pytest.approx(0.123)
        assert loaded["converged"] is True
        assert loaded["K"] == 10
        assert loaded["n"] == 2
        assert loaded["dep_a"] == pytest.approx(6778.0)
        assert loaded["arr_a"] == pytest.approx(7178.0)
        assert loaded["DU_km"] == pytest.approx(cu.DU)

        val = store.load_blade_validation(blade_id)
        assert val is not None
        assert val["passed"] is True
        assert val["bc_violation_rk4"] == pytest.approx(2e-5)
        assert val["energy_error"] == pytest.approx(1e-8)

    def test_list_blade_simulations(self, store):
        from bezier_orbit.blade.orbit import (
            BLADEOrbitProblem, BLADESCPResult, OrbitBC,
        )

        prob = BLADEOrbitProblem(
            dep=OrbitBC(a=6778.0, e=0.0, inc=0.0, raan=0.0, aop=0.0),
            arr=OrbitBC(a=7178.0, e=0.0, inc=0.0, raan=0.0, aop=0.0),
            t_f=5.0, K=8, n=2,
        )
        result = BLADESCPResult(
            p_segments=[], cost=0.5, converged=True, n_iter=5,
            bc_violation=1e-4, bc_history=[], cost_history=[],
            status="optimal",
        )

        store.save_blade_simulation(prob, result)
        store.save_blade_simulation(prob, result)

        sims = store.list_blade_simulations()
        assert len(sims) == 2

    def test_missing_blade_validation(self, store):
        result = store.load_blade_validation(9999)
        assert result is None
