"""orbit/elements.py 단위 테스트."""

import math
import numpy as np
import pytest

from bezier_orbit.orbit.elements import (
    keplerian_to_cartesian,
    cartesian_to_keplerian,
    state_to_vector,
    vector_to_state,
)


class TestKeplerianCartesianRoundTrip:
    """Keplerian ↔ Cartesian 왕복 변환 정확성."""

    @pytest.mark.parametrize(
        "a, e, inc, raan, aop, ta",
        [
            # LEO 원형 경사궤도
            (1.0, 0.001, math.radians(51.6), math.radians(30), math.radians(45), math.radians(60)),
            # 타원 궤도
            (1.5, 0.3, math.radians(28.5), math.radians(120), math.radians(270), math.radians(180)),
            # GTO-like
            (3.5, 0.7, math.radians(7.0), math.radians(0), math.radians(180), math.radians(90)),
            # 적도 원형
            (1.0, 0.0001, math.radians(0.01), 0.0, 0.0, math.radians(45)),
            # 극궤도
            (1.0, 0.01, math.radians(90.0), math.radians(60), math.radians(90), math.radians(0)),
        ],
    )
    def test_roundtrip(self, a, e, inc, raan, aop, ta):
        """Kep → Cart → Kep 왕복 변환."""
        r, v = keplerian_to_cartesian(a, e, inc, raan, aop, ta, mu=1.0)
        a2, e2, inc2, raan2, aop2, ta2 = cartesian_to_keplerian(r, v, mu=1.0)

        assert a2 == pytest.approx(a, rel=1e-10)
        assert e2 == pytest.approx(e, rel=1e-8)
        assert inc2 == pytest.approx(inc, abs=1e-10)

        # 각도는 2π 주기 → 차이의 sin/cos로 비교
        assert math.cos(raan2) == pytest.approx(math.cos(raan), abs=1e-10)
        assert math.sin(raan2) == pytest.approx(math.sin(raan), abs=1e-10)
        assert math.cos(aop2) == pytest.approx(math.cos(aop), abs=1e-8)
        assert math.sin(aop2) == pytest.approx(math.sin(aop), abs=1e-8)
        assert math.cos(ta2) == pytest.approx(math.cos(ta), abs=1e-8)
        assert math.sin(ta2) == pytest.approx(math.sin(ta), abs=1e-8)


class TestEnergyMomentum:
    """에너지 및 각운동량 보존 확인."""

    def test_specific_energy(self):
        """비역학 에너지 ε = v²/2 - μ/r = -μ/(2a)."""
        a, e = 2.0, 0.5
        r, v = keplerian_to_cartesian(a, e, 0.5, 1.0, 2.0, 1.0, mu=1.0)
        energy = np.linalg.norm(v) ** 2 / 2.0 - 1.0 / np.linalg.norm(r)
        expected = -1.0 / (2.0 * a)
        assert energy == pytest.approx(expected, rel=1e-12)

    def test_angular_momentum(self):
        """h = √(μ·a·(1-e²))."""
        a, e = 1.5, 0.3
        r, v = keplerian_to_cartesian(a, e, 0.8, 0.5, 1.2, 2.5, mu=1.0)
        h = np.linalg.norm(np.cross(r, v))
        expected = math.sqrt(a * (1.0 - e**2))
        assert h == pytest.approx(expected, rel=1e-12)


class TestStateVector:

    def test_roundtrip(self):
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        x = state_to_vector(r, v)
        r2, v2 = vector_to_state(x)
        np.testing.assert_allclose(r2, r)
        np.testing.assert_allclose(v2, v)
