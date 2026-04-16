"""역학 모델 테스트.

테스트 항목:
1. 원궤도 1주기 전파 (u=0, RK4): 초기 상태 복귀 오차 확인
2. 에너지 보존 (u=0, J2 없이): 10주기 후 에너지 변동 < 1e-10
3. CasADi/NumPy 일관성: 동일 입력에서 오차 < 1e-14
4. J2 RAAN drift: 해석식과 0.5% 이내 일치 (i=45deg, 10주기)
"""

import numpy as np
import casadi as ca
import pytest

from orbit_transfer.constants import MU_EARTH, R_E, J2
from orbit_transfer.dynamics.two_body import gravity_acceleration
from orbit_transfer.dynamics.j2_perturbation import j2_acceleration
from orbit_transfer.dynamics.eom import spacecraft_eom_numpy, create_dynamics_function


# ============================================================
# 유틸리티: RK4 적분기
# ============================================================

def rk4_step(f, x, u, dt):
    """단일 RK4 스텝."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_propagate(f, x0, u, t_span, n_steps):
    """RK4 전파.

    Args:
        f: 운동방정식 함수 f(x, u) -> xdot
        x0: 초기 상태, shape (6,)
        u: 제어 입력, shape (3,)
        t_span: 전파 시간 [s]
        n_steps: 적분 스텝 수

    Returns:
        x_final: 최종 상태, shape (6,)
    """
    dt = t_span / n_steps
    x = x0.copy()
    for _ in range(n_steps):
        x = rk4_step(f, x, u, dt)
    return x


def rk4_propagate_history(f, x0, u, t_span, n_steps):
    """RK4 전파 (전체 이력 반환).

    Returns:
        t_arr: 시간 배열, shape (n_steps+1,)
        x_arr: 상태 이력, shape (n_steps+1, 6)
    """
    dt = t_span / n_steps
    t_arr = np.linspace(0.0, t_span, n_steps + 1)
    x_arr = np.zeros((n_steps + 1, len(x0)))
    x_arr[0] = x0.copy()
    for i in range(n_steps):
        x_arr[i + 1] = rk4_step(f, x_arr[i], u, dt)
    return t_arr, x_arr


def orbital_energy(x, mu=MU_EARTH):
    """비에너지(specific orbital energy) 계산."""
    r = x[:3]
    v = x[3:]
    return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)


# ============================================================
# 테스트용 초기조건
# ============================================================

def equatorial_circular_ic(h=400.0):
    """적도 원궤도 초기조건."""
    a = R_E + h
    v_circ = np.sqrt(MU_EARTH / a)
    return np.array([a, 0.0, 0.0, 0.0, v_circ, 0.0])


def inclined_circular_ic(h=400.0, inc_deg=45.0):
    """경사 원궤도 초기조건 (RAAN=0, omega=0, f=0).

    위치: x축 (ascending node)
    속도: 궤도면 내 y-z 평면 방향
    """
    a = R_E + h
    v_circ = np.sqrt(MU_EARTH / a)
    inc = np.radians(inc_deg)
    x0 = np.array([
        a, 0.0, 0.0,
        0.0, v_circ * np.cos(inc), v_circ * np.sin(inc),
    ])
    return x0


# ============================================================
# 테스트 1: 원궤도 1주기 전파 (J2 포함)
# ============================================================

class TestCircularOrbitPropagation:
    """원궤도 전파 테스트."""

    def test_one_period_return_with_j2(self):
        """J2 포함 적도 원궤도 1주기 전파 후 궤도 반경 보존 확인.

        J2는 RAAN/omega secular drift를 유발하므로 Cartesian 위치의 완전 복귀는
        기대할 수 없다. 대신 궤도 반경(||r||)이 보존되는지 검증한다.
        """
        x0 = equatorial_circular_ic(h=400.0)
        a = R_E + 400.0
        T_kep = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
        u = np.zeros(3)

        def eom(x, u):
            return spacecraft_eom_numpy(x, u, include_j2=True, include_drag=False)

        n_steps = 10000
        x_final = rk4_propagate(eom, x0, u, T_kep, n_steps)

        # 궤도 반경 보존: J2 하에서도 원궤도 반경은 거의 일정
        r0 = np.linalg.norm(x0[:3])
        rf = np.linalg.norm(x_final[:3])
        assert abs(rf - r0) < 0.1, f"궤도 반경 변화 {abs(rf-r0):.4f} km > 0.1 km"
        # 속력 보존
        v0 = np.linalg.norm(x0[3:])
        vf = np.linalg.norm(x_final[3:])
        assert abs(vf - v0) < 1e-4, f"속력 변화 {abs(vf-v0):.2e} km/s"

    def test_one_period_return_no_j2(self):
        """J2 미포함 원궤도 1주기 전파: 완전 복귀."""
        x0 = equatorial_circular_ic(h=400.0)
        a = R_E + 400.0
        T_period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
        u = np.zeros(3)

        def eom(x, u):
            return spacecraft_eom_numpy(x, u, include_j2=False, include_drag=False)

        n_steps = 10000
        x_final = rk4_propagate(eom, x0, u, T_period, n_steps)

        pos_err = np.linalg.norm(x_final[:3] - x0[:3])
        vel_err = np.linalg.norm(x_final[3:] - x0[3:])
        assert pos_err < 1e-8, f"위치 오차 {pos_err:.2e} km"
        assert vel_err < 1e-10, f"속도 오차 {vel_err:.2e} km/s"


# ============================================================
# 테스트 2: 에너지 보존 (J2 없이, u=0)
# ============================================================

class TestEnergyConservation:
    """에너지 보존 테스트."""

    def test_energy_conservation_10_periods(self):
        """10주기 후 에너지 변동 < 1e-10."""
        x0 = equatorial_circular_ic(h=400.0)
        a = R_E + 400.0
        T_period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
        u = np.zeros(3)

        def eom(x, u):
            return spacecraft_eom_numpy(x, u, include_j2=False, include_drag=False)

        E0 = orbital_energy(x0)

        n_steps_per_period = 5000
        total_steps = n_steps_per_period * 10
        t_total = T_period * 10.0

        _, x_history = rk4_propagate_history(eom, x0, u, t_total, total_steps)

        # 매 주기 끝에서 에너지 확인
        for period_idx in range(1, 11):
            step_idx = period_idx * n_steps_per_period
            E = orbital_energy(x_history[step_idx])
            dE = abs(E - E0)
            assert dE < 1e-10, (
                f"주기 {period_idx}에서 에너지 변동 {dE:.2e} > 1e-10"
            )


# ============================================================
# 테스트 3: CasADi/NumPy 일관성
# ============================================================

class TestCasADiNumPyConsistency:
    """CasADi와 NumPy 결과의 일관성 테스트."""

    def test_consistency_no_j2(self):
        """J2 미포함: CasADi와 NumPy 결과 차이 < 1e-14."""
        x0 = equatorial_circular_ic(h=400.0)
        u0 = np.array([1e-6, 2e-6, -1e-6])

        # NumPy
        xdot_np = spacecraft_eom_numpy(x0, u0, include_j2=False, include_drag=False)

        # CasADi
        eom_ca = create_dynamics_function(include_j2=False)
        xdot_ca = np.array(eom_ca(x0, u0)).flatten()

        err = np.max(np.abs(xdot_np - xdot_ca))
        assert err < 1e-14, f"CasADi/NumPy 불일치: {err:.2e}"

    def test_consistency_with_j2(self):
        """J2 포함: CasADi와 NumPy 결과 차이 < 1e-14."""
        # 경사 궤도에서 J2의 z-component도 확인
        x0 = inclined_circular_ic(h=400.0, inc_deg=45.0)
        u0 = np.array([0.0, 0.0, 0.0])

        # NumPy
        xdot_np = spacecraft_eom_numpy(x0, u0, include_j2=True, include_drag=False)

        # CasADi
        eom_ca = create_dynamics_function(include_j2=True)
        xdot_ca = np.array(eom_ca(x0, u0)).flatten()

        err = np.max(np.abs(xdot_np - xdot_ca))
        assert err < 1e-14, f"CasADi/NumPy 불일치 (J2): {err:.2e}"

    def test_consistency_with_thrust(self):
        """추력 포함: CasADi와 NumPy 결과 차이 < 1e-14."""
        x0 = inclined_circular_ic(h=500.0, inc_deg=30.0)
        u0 = np.array([1e-7, -5e-8, 3e-8])

        xdot_np = spacecraft_eom_numpy(x0, u0, include_j2=True, include_drag=False)

        eom_ca = create_dynamics_function(include_j2=True)
        xdot_ca = np.array(eom_ca(x0, u0)).flatten()

        err = np.max(np.abs(xdot_np - xdot_ca))
        assert err < 1e-14, f"CasADi/NumPy 불일치 (추력): {err:.2e}"


# ============================================================
# 테스트 4: J2 RAAN drift
# ============================================================

class TestJ2RAANDrift:
    """J2에 의한 RAAN drift 테스트."""

    def test_raan_drift_rate(self):
        """RAAN drift rate가 해석식과 0.5% 이내 일치 (i=45deg, 10주기).

        해석식: dOmega/dt = -3/2 * n * J2 * (R_E/p)^2 * cos(i)
        여기서 n = sqrt(mu/a^3), p = a (원궤도)
        """
        inc_deg = 45.0
        h = 400.0
        a = R_E + h
        inc = np.radians(inc_deg)

        x0 = inclined_circular_ic(h=h, inc_deg=inc_deg)
        T_period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
        n_mean = np.sqrt(MU_EARTH / a**3)
        u = np.zeros(3)

        def eom(x, u):
            return spacecraft_eom_numpy(x, u, include_j2=True, include_drag=False)

        n_periods = 10
        n_steps = 10000 * n_periods
        t_total = T_period * n_periods

        _, x_history = rk4_propagate_history(eom, x0, u, t_total, n_steps)

        # RAAN 추출: h = r x v, RAAN = atan2(h_x, -h_y)
        # 각운동량 벡터 h = r x v
        # RAAN = atan2(h[0], -h[1]) (Omega)
        def compute_raan(state):
            r = state[:3]
            v = state[3:]
            h_vec = np.cross(r, v)
            # h = [hx, hy, hz]
            # 궤도면 법선: h_vec
            # RAAN: Omega = atan2(hx, -hy)
            return np.arctan2(h_vec[0], -h_vec[1])

        raan_0 = compute_raan(x_history[0])
        raan_f = compute_raan(x_history[-1])

        # RAAN 변화량 (래핑 처리)
        d_raan = raan_f - raan_0
        # 작은 값이므로 래핑 불필요 예상

        # 해석식에 의한 예측
        p = a  # 원궤도: p = a
        raan_rate_analytical = -1.5 * n_mean * J2 * (R_E / p) ** 2 * np.cos(inc)
        d_raan_analytical = raan_rate_analytical * t_total

        # 상대 오차 0.5% 이내
        rel_err = abs((d_raan - d_raan_analytical) / d_raan_analytical)
        assert rel_err < 0.005, (
            f"RAAN drift 상대 오차 {rel_err*100:.4f}% > 0.5%\n"
            f"  수치: {np.degrees(d_raan):.6f} deg\n"
            f"  해석: {np.degrees(d_raan_analytical):.6f} deg"
        )
