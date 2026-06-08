"""비교 케이스 정의 모듈.

논문 시나리오 및 파생 케이스를 미리 정의한다.
각 케이스는 ComparisonCase 객체이며 TransferConfig와 메타데이터를 포함한다.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from ..types import TransferConfig


@dataclass
class ComparisonCase:
    """단일 비교 케이스.

    Attributes
    ----------
    name : str          케이스 고유 이름 (파일명에 사용)
    description : str   케이스 설명
    config : TransferConfig
    supports_collocation : bool
        False이면 CollocationSolver를 건너뜀
        (현재 콜로케이션은 Ω=0 고정 → RAAN 변화 미지원)
    notes : str         결과 해석 시 주의사항
    extra : dict        추가 파라미터 (RAAN, 위도 등 메타 정보)
    """
    name: str
    description: str
    config: TransferConfig
    supports_collocation: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 논문 기반 케이스 — Seoul(37.5°) → Helsinki(60.2°), h₀=561 km, Δa=0
# 참고: 논문에서 ΔRAAN=56°도 포함하나, 현재 콜로케이션은 Ω=0 고정
# 단순화: 경사각 변화만 반영, RAAN 변화는 Lambert/Hohmann 메모에만 기록
# ---------------------------------------------------------------------------

PAPER_MAIN = ComparisonCase(
    name="paper_main",
    description="Seoul→Helsinki 단순화 (Δi=22.7°, ΔRAAN 제외, h=561km)",
    config=TransferConfig(
        h0=561.0,
        delta_a=0.0,
        delta_i=22.7,
        T_max_normed=0.7,    # ~절반 궤도 주기 내 전이
        u_max=0.01,
        h_min=200.0,
    ),
    notes="논문 기준 케이스. RAAN 변화(56°)는 현재 콜로케이션 미지원으로 제외.",
    extra={
        "delta_RAAN_deg": 56.0,
        "lat_dep_deg": 37.5,
        "lat_arr_deg": 60.2,
        # LambertSolver 추가 파라미터 (RAAN 포함 버전)
        "lambert_kwargs": {"Omega_f_deg": 56.0},
    },
)

# Seoul→Helsinki 전체 케이스 (RAAN 포함) — Hohmann/Lambert만 유효
PAPER_MAIN_FULL = ComparisonCase(
    name="paper_main_full",
    description="Seoul→Helsinki 전체 (Δi=22.7°, ΔRAAN=56°, i0=37.5°, h=561km)",
    config=TransferConfig(
        h0=561.0,
        delta_a=0.0,
        delta_i=22.7,
        T_max_normed=0.7,
        u_max=0.01,
        h_min=200.0,
    ),
    supports_collocation=False,  # RAAN 변화 미지원
    notes="RAAN 변화 포함 전체 시나리오. Hohmann/Lambert만 실행.",
    extra={
        "delta_RAAN_deg": 56.0,
        "lat_dep_deg": 37.5,
        "lat_arr_deg": 60.2,
        "lambert_kwargs": {
            "i0_deg": 37.5,
            "Omega_i_deg": 0.0,
            "Omega_f_deg": 56.0,
        },
    },
)

# 추력 제약 포함 버전 (Reviewer #2-4 대응)
PAPER_MAIN_THRUST_CONSTRAINED = ComparisonCase(
    name="paper_main_constrained",
    description="Seoul→Helsinki (Δi=22.7°) + 추력 제약 u_max=2e-3 km/s²",
    config=TransferConfig(
        h0=561.0,
        delta_a=0.0,
        delta_i=22.7,
        T_max_normed=0.7,
        u_max=2e-3,          # 현실적 소형위성 수준
        h_min=200.0,
    ),
    notes="추력 제약 활성화 케이스. Reviewer #2-4 대응.",
    extra={"delta_RAAN_deg": 56.0},
)

# ---------------------------------------------------------------------------
# 경사각 변화 시리즈 — 논문의 Fig. 경사각 민감도 분석과 대응
# ---------------------------------------------------------------------------

INCLINATION_SERIES: list[ComparisonCase] = []
for _di in [5.0, 10.0, 15.0, 20.0, 22.7, 30.0, 40.0, 50.0]:
    INCLINATION_SERIES.append(ComparisonCase(
        name=f"di_{int(_di):02d}deg",
        description=f"순수 경사각 변화 Δi={_di}° (h=561km, Δa=0)",
        config=TransferConfig(
            h0=561.0,
            delta_a=0.0,
            delta_i=_di,
            T_max_normed=0.7,
            u_max=0.01,
            h_min=200.0,
        ),
        notes="논문 경사각 민감도 시리즈와 대응.",
    ))

# ---------------------------------------------------------------------------
# 전이시간 시리즈 — T_max_normed 변화 (논문 시간 제약 특성 강조)
# ---------------------------------------------------------------------------

TIME_SERIES: list[ComparisonCase] = []
for _T in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
    TIME_SERIES.append(ComparisonCase(
        name=f"T_{int(_T*10):02d}",
        description=f"Seoul→Helsinki 단순화, T_max_normed={_T:.1f}",
        config=TransferConfig(
            h0=561.0,
            delta_a=0.0,
            delta_i=22.7,
            T_max_normed=_T,
            u_max=0.01,
            h_min=200.0,
        ),
        notes="전이시간 변화에 따른 비용함수 민감도 분석.",
    ))

# ---------------------------------------------------------------------------
# 고도 + 경사각 복합 케이스 — Hohmann 대비 연속 추력 우위를 드러내는 영역
# ---------------------------------------------------------------------------

COMBINED_SERIES: list[ComparisonCase] = []
for _da, _di in [
    (200, 5.0), (200, 10.0), (200, 15.0),
    (500, 5.0), (500, 10.0), (500, 15.0),
    (1000, 5.0), (1000, 10.0), (1000, 15.0),
]:
    COMBINED_SERIES.append(ComparisonCase(
        name=f"da{int(_da)}_di{int(_di):02d}",
        description=f"복합 전이 Δa={_da} km, Δi={_di}° (h=400km)",
        config=TransferConfig(
            h0=400.0,
            delta_a=float(_da),
            delta_i=float(_di),
            T_max_normed=0.7,
            u_max=0.01,
            h_min=200.0,
        ),
    ))

# ---------------------------------------------------------------------------
# 단일 케이스 빠른 테스트용
# ---------------------------------------------------------------------------

QUICK_TEST = ComparisonCase(
    name="quick_test",
    description="빠른 기능 테스트 (Δi=5°, h=400km)",
    config=TransferConfig(
        h0=400.0,
        delta_a=0.0,
        delta_i=5.0,
        T_max_normed=0.6,
        u_max=0.01,
        h_min=200.0,
    ),
)

# ---------------------------------------------------------------------------
# 케이스 레지스트리
# ---------------------------------------------------------------------------

CASE_REGISTRY: dict[str, ComparisonCase] = {c.name: c for c in [
    PAPER_MAIN,
    PAPER_MAIN_FULL,
    PAPER_MAIN_THRUST_CONSTRAINED,
    QUICK_TEST,
    *INCLINATION_SERIES,
    *TIME_SERIES,
    *COMBINED_SERIES,
]}

# 그룹 이름 → 케이스 목록
CASE_GROUPS: dict[str, list[ComparisonCase]] = {
    "paper": [PAPER_MAIN, PAPER_MAIN_FULL, PAPER_MAIN_THRUST_CONSTRAINED],
    "inclination_series": INCLINATION_SERIES,
    "time_series": TIME_SERIES,
    "combined_series": COMBINED_SERIES,
    "all": list(CASE_REGISTRY.values()),
}


def get_case(name: str) -> ComparisonCase:
    """케이스 이름으로 ComparisonCase를 반환한다."""
    if name not in CASE_REGISTRY:
        available = list(CASE_REGISTRY.keys())
        raise KeyError(f"케이스 '{name}' 없음. 사용 가능: {available}")
    return CASE_REGISTRY[name]


def get_group(group_name: str) -> list[ComparisonCase]:
    """그룹 이름으로 케이스 목록을 반환한다."""
    if group_name not in CASE_GROUPS:
        available = list(CASE_GROUPS.keys())
        raise KeyError(f"그룹 '{group_name}' 없음. 사용 가능: {available}")
    return CASE_GROUPS[group_name]


def list_cases() -> None:
    """등록된 모든 케이스를 출력한다."""
    print(f"{'이름':<35} {'설명'}")
    print("-" * 80)
    for name, case in CASE_REGISTRY.items():
        print(f"{name:<35} {case.description}")
