# Bezier Curve-Based Orbit Transfer via Sequential Convex Programming

베지어 곡선 기반 궤적 성형과 순차 볼록 계획법(SCP)을 결합하여
3차원 연속추력 궤도전이 궤적최적화 문제를 푸는 연구 프로젝트.

---

## 핵심 방법론

- **베지어 궤적 성형**: 추진 가속도 벡터 `u(τ)`를 Bernstein 기저 함수로 전개하여
  목적함수를 제어점에 대한 볼록 이차형식(QP)으로 정식화
- **SCP (Sequential Convex Programming)**: 비선형 동역학을 참조 궤도 주변에서
  순차 선형화하여 매 반복마다 QP/SOCP를 풀어 수렴
- **보조변수 승강 (Auxiliary Variable Lifting)**: 비행시간 `t_f`를 설계변수로 포함할 때
  발생하는 이중선형 항을 `Z = t_f · P_u` 치환으로 아핀화하여 볼록성 유지
- **적응형 정규화**: 초기 궤도 반장축 `a₀`를 기준 거리 단위로 사용하여
  모든 궤도 영역에서 수치 안정성 확보

---

## 이론 보고서 (`docs/reports/`)

이론 기초를 연구노트 형식으로 정리. 보고서 간 상호 참조(`\cite{reportNNN}`)로 연결되며,
추후 학술 논문으로 엮을 수 있도록 표기법과 구조를 통일하여 작성.

PDF 파일명 형식: `ReUSV41_NNN_제목.pdf`

| No. | 제목 | PDF |
|-----|------|-----|
| 001 | 궤도역학 표준 정규화 (Canonical Units) | `ReUSV41_001_nondimensionalization.pdf` |
| 002 | 3차원 궤도 동역학 및 섭동 모델 | `ReUSV41_002_orbital_dynamics.pdf` |
| 003 | 베지어 성형 대상 선택 및 정식화 전략 | `ReUSV41_003_bezier_shaping_strategy.pdf` |
| 004 | 베지어 기저 및 적분 행렬 | `ReUSV41_004_bezier_basis.pdf` |
| 005 | 베지어 제약조건과 컨벡스화 | `ReUSV41_005_bezier_constraints.pdf` |
| 006 | SCP 알고리즘 정식화 | `ReUSV41_006_scp_formulation.pdf` |
| 007 | 자유 종말시간 — 보조변수 승강 | `ReUSV41_007_free_time_lifting.pdf` |
| 008 | 수치 검증 및 연구 시나리오 | `ReUSV41_008_numerical_verification.pdf` |
| 009 | 아핀 드리프트 근사 — 비선형 합성의 구간별 선형화 | `ReUSV41_009_affine_drift.pdf` |
| 010 | 베른슈타인 대수를 통한 중력 합성 | `ReUSV41_010_bernstein_algebra.pdf` |
| 011 | 드리프트 적분의 솔버 통합과 성능 최적화 | `ReUSV41_011_performance_analysis.pdf` |
| 012 | 대수적 커플링 행렬 — Bernstein 체인 룰 드리프트 감도 | `ReUSV41_012_algebraic_coupling.pdf` |
| 013 | 경로 제약의 볼록 처리 — 드 카스텔조 세분화와 볼록껍질 | `ReUSV41_013_path_constraints.pdf` |
| 014 | 수치 검증 업데이트 — 비공면 궤도 전이와 경로 제약 | `ReUSV41_014_noncoplanar_verification.pdf` |
| 015 | 베지어–GCF 직합 공간의 궤도전이 적용 — 코스팅 구간 통합 | `ReUSV41_015_direct_sum_coasting.pdf` |
| 016 | BLADE 수치 검증: 이중 적분기에서 궤도전이까지 | `ReUSV41_016_blade_verification.pdf` |
| 017 | BLADE-SCP 수치 병목 개선 방법론 | `ReUSV41_017_numerical_bottleneck.pdf` |

---

## 코드 구조 (`src/`)

```
src/bezier_orbit/
├── normalize.py          # 적응형 정규화 (보고서 001)
├── orbit/
│   ├── elements.py       # Keplerian ↔ Cartesian 변환
│   ├── dynamics.py       # 운동방정식, Jacobian (보고서 002)
│   └── perturbations.py  # J2~J4, 대기항력, SRP, 3체 섭동
├── bezier/
│   ├── basis.py          # B_N, D_N, I_N, Ī_N, G_N (보고서 004)
│   └── constraints.py    # 볼록껍질, 극값 정리 제약 (보고서 005)
├── scp/
│   ├── problem.py        # 문제 정의, 경계조건 선형 제약 구성
│   ├── inner_loop.py     # SCP 반복 QP/SOCP (보고서 006)
│   └── outer_loop.py     # t_f 탐색, 보조변수 승강 (보고서 007)
└── db/
    ├── schema.py         # DuckDB 테이블 정의
    └── store.py          # 저장 / 불러오기
```

---

## 데이터 관리

시뮬레이션 결과는 DuckDB로 체계 관리.

| 테이블 | 내용 |
|--------|------|
| `simulations` | 실행 메타데이터 (궤도 파라미터, solver 설정, 정규화 기준량) |
| `trajectories` | 궤적 데이터 (τ, 위치/속도, 베지어 제어점) |
| `scp_iterations` | SCP 수렴 이력 (비용, 제약 위반량) |
| `param_sweep` | 파라미터 스윕 결과 |

---

## 참고 저장소

**GA-IATCG** — 베지어 기반 컨벡스 최적화 핵심 방법론 연구
- 원격: `git@gitlab:research-base/ga-iatcg.git`
- 로컬: `~/gitlab/GA-IATCG/`
- 주요 참고: `014_free_time_phase2/` (보조변수 승강),
  `007_extremum_constraint/` (무손실 컨벡스화),
  `003_convexification/` (SCP 핵심 정식화)
