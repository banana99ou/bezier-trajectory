"""궤도 요소 변환, 호만 전이, 케플러 전파 테스트."""

import numpy as np
import casadi as ca
import pytest

from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.astrodynamics.orbital_elements import oe_to_rv, rv_to_oe, oe_to_rv_casadi
from orbit_transfer.astrodynamics.hohmann import hohmann_dv, hohmann_tof
from orbit_transfer.astrodynamics.kepler import kepler_propagate


# ============================================================
# Round-trip 변환 테스트 (oe → rv → oe)
# ============================================================

class TestRoundTrip:
    """oe_to_rv → rv_to_oe round-trip 일관성 검증."""

    @pytest.mark.parametrize("oe", [
        # (a, e, i, Omega, omega, nu)
        (R_E + 400.0, 0.001, np.radians(51.6), np.radians(30.0),
         np.radians(45.0), np.radians(60.0)),
        # 높은 이심률
        (R_E + 500.0, 0.3, np.radians(28.5), np.radians(120.0),
         np.radians(270.0), np.radians(90.0)),
        # GEO-like
        (42164.0, 0.01, np.radians(0.05), np.radians(0.0),
         np.radians(0.0), np.radians(180.0)),
        # Molniya-like
        (26600.0, 0.74, np.radians(63.4), np.radians(80.0),
         np.radians(270.0), np.radians(30.0)),
    ], ids=["LEO-ISS", "HighEcc", "GEO", "Molniya"])
    def test_roundtrip(self, oe):
        """Round-trip 변환 오차 < 1e-12."""
        r, v = oe_to_rv(oe, MU_EARTH)
        oe2 = rv_to_oe(r, v, MU_EARTH)

        a1, e1, i1, O1, w1, nu1 = oe
        a2, e2, i2, O2, w2, nu2 = oe2

        assert abs(a2 - a1) < 1e-8, f"a mismatch: {a2} vs {a1}"
        assert abs(e2 - e1) < 1e-12, f"e mismatch: {e2} vs {e1}"
        assert abs(i2 - i1) < 1e-12, f"i mismatch: {i2} vs {i1}"

        # Angle wrapping을 고려한 비교
        def angle_diff(x, y):
            d = (x - y) % (2.0 * np.pi)
            if d > np.pi:
                d -= 2.0 * np.pi
            return abs(d)

        assert angle_diff(O2, O1) < 1e-12, f"Omega mismatch: {O2} vs {O1}"
        assert angle_diff(w2, w1) < 1e-12, f"omega mismatch: {w2} vs {w1}"
        assert angle_diff(nu2, nu1) < 1e-12, f"nu mismatch: {nu2} vs {nu1}"


# ============================================================
# 원궤도 (e=0) 특이 케이스
# ============================================================

class TestCircularOrbit:
    """e=0 원궤도 처리 테스트."""

    def test_circular_inclined(self):
        """원궤도(e=0) + 경사궤도: 정상 동작 확인."""
        a = R_E + 400.0
        e = 0.0
        i = np.radians(51.6)
        Omega = np.radians(30.0)
        omega = 0.0  # e=0이므로 무의미
        nu = np.radians(45.0)  # argument of latitude

        oe = (a, e, i, Omega, omega, nu)
        r, v = oe_to_rv(oe, MU_EARTH)

        # 원궤도 속성 검증
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        v_circ = np.sqrt(MU_EARTH / a)

        assert abs(r_mag - a) < 1e-8, f"r_mag={r_mag}, expected {a}"
        assert abs(v_mag - v_circ) < 1e-8, f"v_mag={v_mag}, expected {v_circ}"

        # 역변환
        oe2 = rv_to_oe(r, v, MU_EARTH)
        a2, e2, i2, O2, w2, nu2 = oe2

        assert abs(a2 - a) < 1e-8
        assert e2 < 1e-10  # 원궤도 확인
        assert abs(i2 - i) < 1e-12

    def test_circular_equatorial(self):
        """원궤도(e=0) + 적도궤도(i=0): 정상 동작 확인."""
        a = R_E + 400.0
        e = 0.0
        i = 0.0
        Omega = 0.0
        omega = 0.0
        nu = np.radians(90.0)  # true longitude

        oe = (a, e, i, Omega, omega, nu)
        r, v = oe_to_rv(oe, MU_EARTH)

        # 원궤도 속성 검증
        r_mag = np.linalg.norm(r)
        v_circ = np.sqrt(MU_EARTH / a)
        assert abs(r_mag - a) < 1e-8
        assert abs(np.linalg.norm(v) - v_circ) < 1e-8

        # 역변환
        oe2 = rv_to_oe(r, v, MU_EARTH)
        a2, e2, i2, _, _, nu2 = oe2

        assert abs(a2 - a) < 1e-8
        assert e2 < 1e-10
        assert i2 < 1e-10


# ============================================================
# i=0 적도궤도 특이 케이스
# ============================================================

class TestEquatorialOrbit:
    """i=0 적도궤도 처리 테스트."""

    def test_equatorial_elliptic(self):
        """적도 타원궤도(i=0, e>0): 정상 동작 확인."""
        a = R_E + 1000.0
        e = 0.1
        i = 0.0
        Omega = 0.0
        omega = np.radians(45.0)
        nu = np.radians(30.0)

        oe = (a, e, i, Omega, omega, nu)
        r, v = oe_to_rv(oe, MU_EARTH)

        # z 성분은 0이어야 함
        assert abs(r[2]) < 1e-10, f"r_z = {r[2]}"
        assert abs(v[2]) < 1e-10, f"v_z = {v[2]}"

        # 역변환
        oe2 = rv_to_oe(r, v, MU_EARTH)
        a2, e2, i2, O2, w2, nu2 = oe2

        assert abs(a2 - a) < 1e-8
        assert abs(e2 - e) < 1e-12
        assert i2 < 1e-10

        # omega+nu 합이 보존되어야 함
        def angle_diff(x, y):
            d = (x - y) % (2.0 * np.pi)
            if d > np.pi:
                d -= 2.0 * np.pi
            return abs(d)

        assert angle_diff(w2 + nu2, omega + nu) < 1e-10


# ============================================================
# 호만 전이 테스트
# ============================================================

class TestHohmann:
    """호만 전이 delta-v 검증."""

    def test_leo_to_geo(self):
        """LEO(h=200km) → GEO(h=35786km) 호만 전이: dv_total ≈ 3.935 km/s."""
        a1 = R_E + 200.0    # LEO
        a2 = R_E + 35786.0  # GEO

        dv1, dv2, dv_total = hohmann_dv(a1, a2, MU_EARTH)

        assert abs(dv_total - 3.935) < 0.01, \
            f"dv_total = {dv_total:.4f} km/s, expected ≈ 3.935 km/s"

        # 개별 delta-v 양수 확인
        assert dv1 > 0.0
        assert dv2 > 0.0

    def test_hohmann_tof(self):
        """호만 전이 비행시간 양수 확인."""
        a1 = R_E + 200.0
        a2 = R_E + 35786.0

        tof = hohmann_tof(a1, a2, MU_EARTH)
        assert tof > 0.0

        # 대략적 값 확인 (~5.25 hours)
        tof_hours = tof / 3600.0
        assert 5.0 < tof_hours < 5.5, f"tof = {tof_hours:.2f} hours"


# ============================================================
# CasADi vs NumPy 일관성 테스트
# ============================================================

class TestCasADiConsistency:
    """CasADi symbolic 버전과 NumPy 버전의 일관성 검증."""

    @pytest.mark.parametrize("oe", [
        (R_E + 400.0, 0.001, np.radians(51.6), np.radians(30.0),
         np.radians(45.0), np.radians(60.0)),
        (R_E + 500.0, 0.3, np.radians(28.5), np.radians(120.0),
         np.radians(270.0), np.radians(90.0)),
        (42164.0, 0.0, np.radians(0.0), 0.0, 0.0, np.radians(180.0)),
    ], ids=["LEO-ISS", "HighEcc", "GEO-circular"])
    def test_casadi_vs_numpy(self, oe):
        """CasADi와 NumPy 결과 오차 < 1e-14."""
        # NumPy
        r_np, v_np = oe_to_rv(oe, MU_EARTH)

        # CasADi: 수치 평가
        oe_mx = ca.MX.sym("oe", 6)
        r_mx, v_mx = oe_to_rv_casadi(oe_mx, MU_EARTH)

        f = ca.Function("f", [oe_mx], [r_mx, v_mx])
        oe_val = np.array(oe)
        r_ca, v_ca = f(oe_val)
        r_ca = np.array(r_ca).flatten()
        v_ca = np.array(v_ca).flatten()

        assert np.max(np.abs(r_ca - r_np)) < 1e-14, \
            f"r diff: {np.abs(r_ca - r_np)}"
        assert np.max(np.abs(v_ca - v_np)) < 1e-14, \
            f"v diff: {np.abs(v_ca - v_np)}"


# ============================================================
# 케플러 전파 테스트
# ============================================================

class TestKeplerPropagate:
    """케플러 전파 기본 검증."""

    def test_circular_orbit_period(self):
        """원궤도 1주기 전파 후 원래 상태로 복귀."""
        a = R_E + 400.0
        oe = (a, 0.0, np.radians(51.6), 0.0, 0.0, 0.0)
        r0, v0 = oe_to_rv(oe, MU_EARTH)

        # 1 주기
        T = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
        r1, v1 = kepler_propagate(r0, v0, T, MU_EARTH)

        assert np.linalg.norm(r1 - r0) < 1e-6, \
            f"Position error: {np.linalg.norm(r1 - r0)}"
        assert np.linalg.norm(v1 - v0) < 1e-8, \
            f"Velocity error: {np.linalg.norm(v1 - v0)}"

    def test_elliptic_orbit_period(self):
        """타원궤도 1주기 전파 후 원래 상태로 복귀."""
        a = R_E + 1000.0
        oe = (a, 0.3, np.radians(28.5), np.radians(45.0),
              np.radians(60.0), np.radians(0.0))
        r0, v0 = oe_to_rv(oe, MU_EARTH)

        T = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
        r1, v1 = kepler_propagate(r0, v0, T, MU_EARTH)

        assert np.linalg.norm(r1 - r0) < 1e-6, \
            f"Position error: {np.linalg.norm(r1 - r0)}"
        assert np.linalg.norm(v1 - v0) < 1e-8, \
            f"Velocity error: {np.linalg.norm(v1 - v0)}"

    def test_energy_conservation(self):
        """케플러 전파에서 에너지 보존 확인."""
        a = R_E + 600.0
        oe = (a, 0.1, np.radians(45.0), np.radians(90.0),
              np.radians(30.0), np.radians(120.0))
        r0, v0 = oe_to_rv(oe, MU_EARTH)

        energy0 = 0.5 * np.dot(v0, v0) - MU_EARTH / np.linalg.norm(r0)

        # 임의 시간 전파
        dt = 3600.0  # 1시간
        r1, v1 = kepler_propagate(r0, v0, dt, MU_EARTH)

        energy1 = 0.5 * np.dot(v1, v1) - MU_EARTH / np.linalg.norm(r1)

        assert abs(energy1 - energy0) < 1e-10, \
            f"Energy error: {abs(energy1 - energy0)}"
