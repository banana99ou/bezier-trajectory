# CLAUDE.md — bezier-orbit-transfer-scp

## 프로젝트 개요

추진 가속도 벡터 `u(τ)`를 Bernstein 기저 함수로 전개(베지어 성형)하고,
비선형 궤도 동역학을 순차 선형화하여 매 반복마다 QP/SOCP를 푸는
**SCP(Sequential Convex Programming)** 기반 3차원 궤도전이 궤적최적화 연구.

---

## 핵심 설계 결정

### 1. 베지어 성형 대상 — 추진 가속도 `u(τ)`

추진 가속도(thrust acceleration) `u(τ)`를 Bernstein 기저로 전개한다.
(위치·속도 성형 방식은 채택하지 않음)

```
u(τ) = B_N(τ)^T · P_u,   τ ∈ [0, 1]
```

- 목적함수가 제어점 `P_u`에 대한 볼록 이차형식(Gram 행렬 G_N)이 됨
- `J = t_f · Σ p_k^T G_N p_k`
- 이론: `docs/reports/003_bezier_shaping_strategy/`

### 2. 자유 종말시간 처리 — 보조변수 승강

`t_f`를 설계변수로 포함하면 이중선형(bilinear) 항 `t_f · P_u`가 발생한다.
이를 다음 치환으로 해결한다.

```
Z ≜ t_f · P_u   (보조변수 승강)
```

- Inner loop: `Z`에 대한 SCP (볼록 QP/SOCP)
- Outer loop: `t_f` 업데이트 (격자 탐색 → 황금분할 탐색)
- 복원: `P_u* = Z* / t_f`
- 이론: `docs/reports/007_free_time_lifting/`

### 3. 목적함수

```
J = t_f · ∫₀¹ ‖u(τ)‖² dτ = (1/t_f) · Σ z_k^T G_N z_k
```

정규화 변수 기준. `t_f`를 outer loop에서 고정하면 `Z`에 대한 볼록 이차형식.

### 4. 추력 크기 제약

```python
thrust_limit = True   # ‖u(τ_j)‖ ≤ u_max → SOCP
thrust_limit = False  # 제약 없음 → unconstrained QP
```

SOCP 변환 시 lossless convexification (극값 정리) 적용. 이론: `docs/reports/005_bezier_constraints/`

### 5. 경계조건

- Keplerian elements 또는 Cartesian 상태벡터 모두 입력 가능
- 내부에서 Cartesian으로 변환 후 선형 등식 제약으로 처리
- 이론: `docs/reports/006_scp_formulation/`

### 6. 섭동 모델 (계층적 설계)

| 레벨 | 섭동 | 처리 |
|------|------|------|
| Level 0 (기본, 항상 포함) | J2 | 동역학에 직접 포함, 선형화 |
| Level 1 (선택) | J3–J6, 대기항력 | SCP 참조 궤도 기준 선형화 |
| Level 2 (고급, 선택) | 태양복사압(SRP), 3체 섭동 | Outer loop 보정 |

플래그로 on/off: `perturbation_level=0|1|2`
이론: `docs/reports/002_orbital_dynamics/`

### 7. 베지어 차수

N = 8 ~ 20 범위 지원. 기본값 N = 12.
수치 안정성: 정규화된 변수에서 계산, `G_N` 명시적 구성.

---

## 수치 정규화

**모든 내부 계산은 정규화된 변수로 수행한다.**

```python
DU = a0               # km  (초기 궤도 반장축)
TU = sqrt(a0**3 / mu) # s
VU = sqrt(mu / a0)    # km/s
AU = VU / TU          # km/s²  (= mu / a0²)
```

- 보조변수 `Z* = O(1)`, `P_u* = O(1)`, `t_f* = O(1)` 보장
- 입/출력 경계에서만 물리 단위 ↔ 정규화 단위 변환
- DuckDB 저장 시 `DU, TU, VU, a0` 함께 기록
- 이론: `docs/reports/001_nondimensionalization/`

---

## 코드 구현 전략

### 패키지 구조

```
src/bezier_orbit/
├── normalize.py          # 적응형 정규화 (DU/TU/VU 계산, 변환)
├── orbit/
│   ├── elements.py       # Keplerian ↔ Cartesian 변환
│   ├── dynamics.py       # ECI 운동방정식, Jacobian
│   └── perturbations.py  # J2~J6, 대기항력, SRP, 3체
├── bezier/
│   ├── basis.py          # B_N, D_N, I_N, Ī_N, G_N 행렬
│   └── constraints.py    # 볼록껍질 제약, 극값 정리 제약
├── scp/
│   ├── problem.py        # 문제 정의, 경계조건 선형 제약
│   ├── inner_loop.py     # SCP 반복 (QP/SOCP via CVXPY)
│   └── outer_loop.py     # t_f 탐색, 보조변수 승강
└── db/
    ├── schema.py         # DuckDB 테이블 정의
    └── store.py          # 저장 / 불러오기
```

### 구현 단계

**Phase 1** — 수학 기초 모듈
- `normalize.py`: DU/TU/VU 계산, 단위 변환 함수
- `bezier/basis.py`: N차 Bernstein 기저, D_N, I_N, Ī_N, G_N
- `orbit/elements.py`: Keplerian ↔ Cartesian (양방향)
- `orbit/dynamics.py`: ECI 2체 + J2 운동방정식, Jacobian

**Phase 2** — SCP 핵심 루프
- `orbit/perturbations.py`: J3–J6, 대기항력, SRP, 3체
- `bezier/constraints.py`: 볼록껍질, 극값 정리, SOCP 제약 생성
- `scp/inner_loop.py`: CVXPY로 QP/SOCP 반복, 수렴 판정
- `scp/outer_loop.py`: t_f 격자/황금분할 탐색, Z 복원

**Phase 3** — 데이터 관리 및 검증
- `db/schema.py`, `db/store.py`: DuckDB 스키마, 저장/로드
- 단위 테스트: Bernstein 적분 항등식, 정규화 역변환, Keplerian 변환
- 수치 검증: Z = t_f · P_u 수치 동등성

### 개발 환경

- Python 패키지 관리: `uv` + `pyproject.toml`
- 최적화 솔버: CVXPY (기본 SCS, 고정밀 Clarabel)
- 데이터: DuckDB
- 테스트: pytest

---

## 데이터 관리 (DuckDB)

| 테이블 | 내용 |
|--------|------|
| `simulations` | 메타데이터 (궤도 파라미터, solver 설정, 섭동 플래그, 정규화 기준량 DU/TU/VU) |
| `trajectories` | 궤적 데이터 (τ, 위치/속도 정규화값, 베지어 제어점 Z, P_u) |
| `scp_iterations` | SCP 수렴 이력 (반복 횟수, 비용, 제약 위반량) |
| `param_sweep` | 파라미터 스윕 결과 |

---

## 보고서 작성 규칙

- **양식**: `ReUSV41-numbered` 템플릿 (`~/Library/Mobile Documents/com~apple~CloudDocs/Texifier/0. 양식/ReUSV41-numbered/`)
- **위치**: `docs/reports/NNN_주제명/ReUSV41_NNN_주제명.tex`
- **TeX/PDF 파일명**: `ReUSV41_NNN_제목_영문` (예: `ReUSV41_007_free_time_lifting.tex/.pdf`)
- **심볼릭 링크**: 각 디렉토리에 `main.tex → ReUSV41_NNN_*.tex` 심볼릭 링크 유지 (서버 업로드 스크립트 호환)
- **저자**: Suwon Lee, `suwon.lee@kookmin.ac.kr`
- **소속**: Future Mobility Control Lab, Kookmin University, Seoul, Korea
- **날짜**: March 12, 2026 (공통)
- **상호 참조**: `\cite{reportNNN}` — 모든 보고서의 `references.bib`에 001–009 항목 포함
- **조판 워크플로**:
  1. `pdflatex -interaction=nonstopmode ReUSV41_NNN_제목.tex`
  2. `bibtex ReUSV41_NNN_제목`
  3. `pdflatex` × 2 (상호 참조 해결)
  4. Overfull hbox 경고 확인 후 수정 재조판
  5. 임시 파일 삭제: `rm -f ReUSV41_NNN_제목.{aux,log,bbl,blg,out,toc,lof,lot,fls,fdb_latexmk,synctex.gz}`
  6. `open ReUSV41_NNN_제목.pdf`
- 보고서 추가 시 `readme.md` 표 업데이트
- **서버 업로드**: `python ~/gitlab/nocodb-management/research-notes/scripts/upload.py sync` (새 노트만 업로드, 중복 건너뜀)

### 완성된 보고서 목록

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
| 015 | 베지어--GCF 직합 공간의 궤도전이 적용 — 코스팅 구간 통합 | `ReUSV41_015_direct_sum_coasting.pdf` |
| 016 | BLADE 수치 검증: 이중 적분기에서 궤도전이까지 | `ReUSV41_016_blade_verification.pdf` |
| 017 | BLADE-SCP 수치 병목 개선 방법론 | `ReUSV41_017_numerical_bottleneck.pdf` |

---

## 참고 저장소

**GA-IATCG** (`git@gitlab:research-base/ga-iatcg.git`, 로컬: `~/gitlab/GA-IATCG/`)

| 디렉토리 | 내용 | 본 프로젝트 관련성 |
|----------|------|-------------------|
| `003_convexification/` | SCP 핵심 정식화, 적분 행렬 I_N | 보고서 004, 006 기초 |
| `007_extremum_constraint/` | 부등식의 lossless convexification | 보고서 005 기초 |
| `014_free_time_phase2/` | **보조변수 승강** (Z = t_f · P_u) | 보고서 007, outer_loop.py |
