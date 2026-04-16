"""공유 테스트 fixtures 및 레퍼런스 케이스 정의."""

import pytest
import numpy as np

from orbit_transfer.types import TransferConfig


# ============================================================
# 레퍼런스 케이스 (7개, sub-revolution + 타원궤도)
# ============================================================
REFERENCE_CASES = {
    "R1": {
        "config": TransferConfig(h0=400.0, delta_a=200.0, delta_i=0.0, T_max_normed=0.8),
        "expected_class": 0,  # unimodal
        "description": "Coplanar altitude raise, circular",
    },
    "R2": {
        "config": TransferConfig(h0=400.0, delta_a=100.0, delta_i=5.0, T_max_normed=1.2),
        "expected_class": 1,  # bimodal
        "description": "Plane-change dominant, circular",
    },
    "R3": {
        "config": TransferConfig(h0=400.0, delta_a=500.0, delta_i=3.0, T_max_normed=1.2),
        "expected_class": 1,  # bimodal
        "description": "Combined altitude + plane change, circular",
    },
    "R4": {
        "config": TransferConfig(h0=400.0, delta_a=-200.0, delta_i=0.0, T_max_normed=0.5),
        "expected_class": 0,  # unimodal
        "description": "Coplanar orbit lowering, circular",
    },
    "R5": {
        "config": TransferConfig(h0=400.0, delta_a=200.0, delta_i=0.0, T_max_normed=0.8, e0=0.05),
        "expected_class": 0,  # unimodal
        "description": "Coplanar altitude raise, eccentric departure",
    },
    "R6": {
        "config": TransferConfig(h0=400.0, delta_a=200.0, delta_i=3.0, T_max_normed=1.0, e0=0.05, ef=0.05),
        "expected_class": 0,  # unimodal
        "description": "Combined transfer, both eccentric",
    },
    "R7": {
        "config": TransferConfig(h0=400.0, delta_a=500.0, delta_i=0.0, T_max_normed=1.2, e0=0.1, ef=0.1),
        "expected_class": 2,  # multimodal
        "description": "High eccentricity, both orbits",
    },
}


@pytest.fixture(params=list(REFERENCE_CASES.keys()))
def reference_case(request):
    """모든 레퍼런스 케이스를 순회하는 fixture."""
    case_id = request.param
    case = REFERENCE_CASES[case_id]
    return case_id, case["config"], case["expected_class"]


@pytest.fixture
def coplanar_config():
    """Coplanar 전이 (R1)."""
    return REFERENCE_CASES["R1"]["config"]


@pytest.fixture
def plane_change_config():
    """Pure plane change (R2)."""
    return REFERENCE_CASES["R2"]["config"]


@pytest.fixture
def combined_config():
    """Combined transfer (R3)."""
    return REFERENCE_CASES["R3"]["config"]


@pytest.fixture
def eccentric_config():
    """Eccentric orbit transfer (R6)."""
    return REFERENCE_CASES["R6"]["config"]


@pytest.fixture
def mu():
    """지구 중력 상수."""
    from orbit_transfer.constants import MU_EARTH
    return MU_EARTH


@pytest.fixture
def leo_state():
    """LEO 원궤도 상태벡터 (h=400km, i=0)."""
    from orbit_transfer.constants import MU_EARTH, R_E
    a = R_E + 400.0
    v = np.sqrt(MU_EARTH / a)
    r = np.array([a, 0.0, 0.0])
    v_vec = np.array([0.0, v, 0.0])
    return r, v_vec
