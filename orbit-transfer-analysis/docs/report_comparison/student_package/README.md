# LEO 궤도전이 비교 분석 패키지

**보고서**: `report/report.pdf`
**개발**: Future Mobility Control Lab (FMCL), Kookmin University
**연락처**: suwon.lee@kookmin.ac.kr

---

## 목차

1. [이 패키지가 하는 일](#1-이-패키지가-하는-일)
2. [전체 업무 파이프라인 개관](#2-전체-업무-파이프라인-개관)
3. [환경 설정](#3-환경-설정)
4. [디렉토리 구조와 각 파일의 역할](#4-디렉토리-구조와-각-파일의-역할)
5. [핵심 개념: 세 가지 궤도전이 기법](#5-핵심-개념-세-가지-궤도전이-기법)
6. [최적화 문제 정식화 (Collocation)](#6-최적화-문제-정식화-collocation)
7. [단계별 실습 가이드](#7-단계별-실습-가이드)
8. [결과 파일 해석 방법](#8-결과-파일-해석-방법)
9. [최적화 문제 수정 실험](#9-최적화-문제-수정-실험)
10. [파라미터 레퍼런스](#10-파라미터-레퍼런스)
11. [자주 겪는 문제와 해결책](#11-자주-겪는-문제와-해결책)

---

## 1. 이 패키지가 하는 일

저지구궤도(LEO, Low Earth Orbit) 위성이 궤도를 바꿀 때 어떤 방식으로 기동하는 것이 효율적인지를 세 가지 기법으로 비교합니다.

```
출발 궤도 (h0, i0)  ──[전이 기동]──►  도착 궤도 (h0+Δa, i0+Δi)
                         ↑
                3가지 방법으로 각각 계산:
                (1) Hohmann    : 2번의 순간 기동 (이상적 해석해)
                (2) Lambert    : 2번의 순간 기동 (수치해)
                (3) Collocation: 연속적인 미세 추력 (최적 제어 수치해)
```

**비교하고자 하는 핵심 질문:**
- 연속 추력 기법이 임펄스 기법보다 얼마나 효율적(또는 비효율적)인가?
- 시간 제약이 있을 때 임펄스 기법이 실현 불가능해지는 조건은 무엇인가?
- $\Delta a$ 와 $\Delta i$ 의 조합에 따라 각 기법의 상대적 성능이 어떻게 달라지는가?

---

## 2. 전체 업무 파이프라인 개관

이 패키지의 모든 작업은 아래 세 단계 파이프라인으로 구성됩니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: 임무 파라미터 정의                                           │
│                                                                     │
│  TransferConfig(h0, delta_a, delta_i, T_max_normed, u_max, h_min)  │
│      └─ src/orbit_transfer/types.py 에 정의된 데이터 클래스           │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: 각 기법으로 궤적 계산 (솔버 실행)                             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Hohmann     │  │  Lambert     │  │  Direct Collocation      │  │
│  │  (해석해)    │  │  (수치해)    │  │  (NLP 최적화, IPOPT)     │  │
│  │  ~0.1초      │  │  ~0.1초      │  │  ~5~30초                 │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         └─────────────────┴──────────────────────┘                  │
│                           │                                         │
│              BenchmarkResult (궤적, 비용 지표)                        │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: 결과 저장 및 시각화                                          │
│                                                                     │
│  results/<casename>/                                                │
│  ├── metrics_summary.csv      ← 비용·비행시간 수치 비교               │
│  ├── *_trajectory.csv         ← 궤적 상태벡터 시계열                  │
│  ├── *_impulses.csv           ← 임펄스 기법의 기동 정보               │
│  └── *.pdf                    ← 시각화 그림                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 파이프라인을 실행하는 방법 세 가지

| 방법 | 진입점 | 적합한 상황 |
|------|--------|-----------|
| **A) 사전 정의 케이스** | `scripts/run_comparison.py` | 보고서 결과 재현, 파라미터 스윕 |
| **B) Python 코드 직접** | `from orbit_transfer.benchmark import TransferBenchmark` | 나만의 조건 빠른 테스트 |
| **C) 최적화 직접 수정** | `scripts/custom_optimization.py` | 비용함수·제약조건 실험 |

---

## 3. 환경 설정

### 방법 1: conda (권장)

```bash
# 패키지 디렉토리로 이동
cd student_package

# conda 환경 생성 (Python 3.11 + 의존성 자동 설치)
conda env create -f environment.yml

# 환경 활성화
conda activate orbit-benchmark

# 패키지를 개발 모드로 설치 (src/orbit_transfer 를 python이 인식하도록)
pip install -e .
```

### 방법 2: pip만 사용

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .
```

### 설치 확인

```bash
python -c "from orbit_transfer.benchmark import TransferBenchmark; print('OK')"
# → OK
```

> **casadi 설치 오류 시**: `pip install casadi` 대신 `conda install -c conda-forge casadi` 를 시도하세요.
> casadi는 NLP 솔버 IPOPT를 호출하는 핵심 라이브러리입니다.

---

## 4. 디렉토리 구조와 각 파일의 역할

```
student_package/
│
├── README.md                      ← 이 문서
├── environment.yml                ← conda 환경 정의 (의존성 명세)
├── pyproject.toml                 ← pip 패키지 메타정보 (pip install -e . 용)
│
├── report/
│   └── report.pdf                 ← 분석 보고서 (결과 수치와 해석 포함)
│
├── figures/                       ← 보고서에 수록된 그림 원본 (PDF)
│   ├── figA_time_sweep_thrust.pdf
│   ├── figB_cost_bar.pdf
│   ├── figC_case_panel.pdf
│   ├── figD_cost_vs_T.pdf
│   ├── figE_combined_heatmap.pdf
│   └── figF_feasibility.pdf
│
├── scripts/                       ← 실행 스크립트 (여기서부터 시작)
│   ├── run_comparison.py          ← ① 비교 파이프라인 실행 (케이스 지정)
│   ├── plot_comparison.py         ← ② 결과 시각화 그림 생성
│   └── custom_optimization.py     ← ③ 최적화 정식화 직접 수정 실험
│
├── src/
│   └── orbit_transfer/            ← Python 패키지 루트
│       │
│       ├── types.py               ★ TransferConfig, TrajectoryResult 정의
│       ├── config.py              ★ 솔버 설정, 파라미터 범위
│       ├── constants.py           ★ MU_EARTH, R_E 등 물리 상수
│       │
│       ├── benchmark/             ← 비교 프레임워크 (학생이 주로 쓰는 인터페이스)
│       │   ├── benchmark.py       ★ TransferBenchmark 클래스 (3기법 통합 실행)
│       │   ├── cases.py           ★ 사전 정의 케이스 레지스트리
│       │   ├── solvers.py            HohmannSolver, LambertSolver, CollocationSolver
│       │   ├── result.py             BenchmarkResult 데이터 클래스
│       │   ├── metrics.py            비용 지표 계산 함수
│       │   └── plot.py               기본 시각화 함수
│       │
│       ├── astrodynamics/         ← 궤도역학 기반 알고리즘
│       │   ├── hohmann.py         ★ Hohmann 전이 해석해
│       │   ├── lambert.py         ★ Lambert 문제 수치 풀이
│       │   ├── orbital_elements.py   궤도요소 ↔ ECI 상태벡터 변환
│       │   └── kepler.py             Kepler 전파 (궤적 생성용)
│       │
│       ├── collocation/           ← 콜로케이션 이산화 구현
│       │   ├── hermite_simpson.py ★ Pass 1: Hermite-Simpson NLP
│       │   ├── multiphase_lgl.py  ★ Pass 2: Multi-Phase LGL NLP
│       │   ├── lgl_nodes.py          LGL 노드·가중치 계산
│       │   └── interpolation.py      Dense output 보간
│       │
│       ├── optimizer/             ← NLP 최적화기 상위 레이어
│       │   ├── two_pass.py        ★ Two-Pass 파이프라인 오케스트레이터
│       │   ├── solver.py             IPOPT 옵션 관리
│       │   └── initial_guess.py      초기값 생성 (선형 보간)
│       │
│       └── dynamics/              ← 운동방정식
│           ├── eom.py             ★ 우주선 운동방정식 (CasADi + NumPy)
│           ├── two_body.py           2체 중력 가속도
│           ├── j2_perturbation.py    $J_2$ 섭동 가속도
│           └── drag.py               지수 대기 항력
│
└── results/                       ← 시뮬레이션 실행 시 자동 생성됨
    └── <case_name>/
        ├── metrics_summary.csv
        ├── *_trajectory.csv
        └── ...
```

★ 표시 파일이 핵심 파일입니다. 나머지는 이들을 지원하는 보조 모듈입니다.

---

## 5. 핵심 개념: 세 가지 궤도전이 기법

### 공통 전제: 무엇을 전이하는가?

$$
\text{초기 원궤도:} \quad h_0,\; i_0,\; e_0 \approx 0
$$

$$
\text{목표 원궤도:} \quad h_0 + \Delta a,\; i_0 + \Delta i,\; e_f \approx 0
$$

세 기법 모두 동일한 출발 조건과 도착 조건을 갖지만, **어떤 궤적으로 이동하느냐**가 다릅니다.

---

### 기법 1: Hohmann 전이 (해석해)

```
출발 원궤도  ──[burn 1: Δv₁]──►  전이 타원궤도  ──[burn 2: Δv₂]──►  도착 원궤도
               t=0                                                   t=T_Hoh
```

- **가정**: 출발·도착 모두 원궤도 ($e = 0$)
- **비행시간**: 전이 타원의 반주기

$$
T_\mathrm{Hoh} = \pi \sqrt{\frac{a_\mathrm{transfer}^3}{\mu}}
$$

  $\Delta a = 0$ 인 경우 $T_\mathrm{Hoh} = T_0/2$ (초기 궤도 주기의 정확히 절반)

- **경사각 변화**: 두 번째 기동(burn 2)에서 $\Delta v_2$ 와 평면 변경을 동시에 수행
- **특징**: 해석적 공식으로 계산 → 거의 순간 완료, 최적성이 이론적으로 증명됨
- **한계**: 비행시간이 $T_\mathrm{Hoh}$ 로 고정 → $T_\mathrm{max} < T_\mathrm{Hoh}$ 이면 수행 불가

**관련 코드**: `src/orbit_transfer/astrodynamics/hohmann.py`

---

### 기법 2: Lambert 전이 (수치해)

```
출발 위치 r₁(t=0) ──[Δv₁]──►  전이 궤도 (임의의 원뿔곡선)  ──[Δv₂]──►  도착 위치 r₂(t=tof)
```

- **원리**: "주어진 두 위치벡터 $\mathbf{r}_1$, $\mathbf{r}_2$ 와 비행시간 $t_f$ 로 연결하는 궤도를 구하라" — Lambert 문제
- **비행시간**: 사용자가 지정 (기본값: Hohmann $T_\mathrm{Hoh}$ 사용)
- **특징**: Hohmann보다 일반적 (타원·쌍곡선 모두 가능), 임의의 $\mathbf{r}_1$, $\mathbf{r}_2$ 연결 가능
- **한계**: $180°$ 전이 특이점 존재 ($\mathbf{r}_1$ 과 $\mathbf{r}_2$ 가 반평행이면 해 없음)

**관련 코드**: `src/orbit_transfer/astrodynamics/lambert.py`

---

### 기법 3: Direct Collocation (수치 최적 제어)

```
출발 상태 x(0)  ──[연속 추력 u(t), t∈[0,T_f]]──►  도착 상태 x(T_f)
                      ↑
               추력 프로파일 u(t) 를 최적화
               동역학 방정식 ẋ = f(x,u) 를 만족하면서
               비용함수 J = ∫‖u‖² dt 를 최소화
```

- **원리**: 연속 시간 최적 제어 문제를 유한 차원 NLP(비선형 계획법)로 변환
  - 시간 구간을 $N$ 개 구간으로 나누어 각 점에서의 상태·제어를 결정변수로 설정
  - 구간 간 연속성 제약 = 동역학 방정식을 만족하도록 강제
- **비행시간 $T_f$**: 결정변수 (자유변수) → 범위 $[T_\mathrm{min},\, T_\mathrm{max}]$ 내에서 최적값 탐색
- **출발 이각 $\nu_0$**: 결정변수 → 위성이 궤도의 어느 지점에서 출발할지 자동 결정
- **특징**: 가장 일반적이고 강력하지만, 계산 시간이 상대적으로 오래 걸림 (5~30초)

**관련 코드**: `src/orbit_transfer/collocation/hermite_simpson.py`

---

### 세 기법 비교 요약

| 항목 | Hohmann | Lambert | Collocation |
|------|---------|---------|-------------|
| 추력 방식 | 임펄스 (순간) | 임펄스 (순간) | 연속 (유한 크기) |
| 비행시간 | 고정 (해석식) | 고정 (사전 지정) | 자유변수 (최적화됨) |
| 출발 이각 $\nu_0$ | 고정 (근지점) | 고정 ($\nu_0=0$) | 자유변수 (최적화됨) |
| $\Delta i$ 지원 | O | O | O |
| 계산 시간 | $< 0.1\,\mathrm{s}$ | $< 0.1\,\mathrm{s}$ | $5 \sim 30\,\mathrm{s}$ |
| 물리 모델 | 2체 문제 | 2체 문제 | 2체 $+$ $J_2$ 섭동 |
| 최적성 | 원궤도 전이에서 최적 | 준최적 | $L^2$ 비용 기준 수치 최적 |

> **중요**: Hohmann/Lambert는 $\nu_0$ 와 $T_f$ 가 고정된 **좁은 해 공간**을 탐색하는 반면,
> Collocation은 이 둘을 함께 최적화하는 **넓은 해 공간**을 탐색합니다.
> 따라서 두 방식의 비용 직접 비교는 "서로 다른 정식화"임을 이해하고 해석해야 합니다.

---

## 6. 최적화 문제 정식화 (Collocation)

Collocation이 내부적으로 무엇을 푸는지 이해하면 실험 설계에 도움이 됩니다.

### 최적 제어 문제 원형

$$
\min_{\mathbf{u}(t),\, T_f,\, \nu_0,\, \nu_f} \quad
J = \int_0^{T_f} \|\mathbf{u}(t)\|^2 \, dt
$$

$$
\text{s.t.} \quad
\dot{\mathbf{x}}(t) = \mathbf{f}\!\left(\mathbf{x}(t),\, \mathbf{u}(t)\right)
\quad \forall t \in [0, T_f]
$$

$$
\mathbf{x}(0) = \mathbf{r}(\mathrm{oe}_0,\, \nu_0), \qquad
\mathbf{x}(T_f) = \mathbf{r}(\mathrm{oe}_f,\, \nu_f)
$$

$$
\|\mathbf{u}(t)\| \leq u_\mathrm{max}, \quad
\|\mathbf{r}(t)\| \geq R_E + h_\mathrm{min}, \quad
T_\mathrm{min} \leq T_f \leq T_\mathrm{max}
$$

경계조건의 $\nu_0$, $\nu_f$ 는 자유변수로, 출발·도착 위치를 궤도 위에서 자유롭게 선택합니다.

### Hermite-Simpson 이산화

연속 문제를 컴퓨터로 풀기 위해 시간 구간을 $M = 30$ 개 소구간으로 나눕니다.

$$
t_k = \frac{k}{2M} T_f, \quad k = 0, 1, \ldots, 2M
$$

- 홀수 인덱스 $k = 2j+1$: Hermite midpoint (구간 중점)
- 짝수 인덱스 $k = 2j$: 주 노드

이산화 후 총 결정변수 수: $(6+3) \times 61 + 2 + 1 = 552$ 개 ($M=30$ 기준)

각 구간 $k \to k{+}1$ 에 대해 두 가지 제약이 부가됩니다.

**Simpson continuity** ($\mathbf{f}_k \equiv \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)$):

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{h}{6}\!\left(\mathbf{f}_k + 4\mathbf{f}_{k+1/2} + \mathbf{f}_{k+1}\right)
$$

**Hermite midpoint**:

$$
\mathbf{x}_{k+1/2} = \tfrac{1}{2}\!\left(\mathbf{x}_k + \mathbf{x}_{k+1}\right) + \frac{h}{8}\!\left(\mathbf{f}_k - \mathbf{f}_{k+1}\right)
$$

여기서 $h = T_f / M$ 은 심볼릭 구간 폭입니다.

### 운동방정식 $\mathbf{f}(\mathbf{x}, \mathbf{u})$

상태벡터 $\mathbf{x} = [r_x,\, r_y,\, r_z,\, v_x,\, v_y,\, v_z]^\top$ (ECI 좌표계, 단위: km, km/s)

$$
\dot{\mathbf{r}} = \mathbf{v}
$$

$$
\dot{\mathbf{v}} = -\frac{\mu}{\|\mathbf{r}\|^3}\mathbf{r}
              + \mathbf{a}_{J_2}(\mathbf{r})
              + \mathbf{u}
$$

$J_2$ 항은 지구 편평도에 의한 섭동 가속도로, `INCLUDE_J2 = True` 시 포함됩니다.

**관련 코드**: `src/orbit_transfer/dynamics/eom.py`

---

## 7. 단계별 실습 가이드

### STEP 0: 설치 확인 및 케이스 목록 조회

```bash
# 설치 확인
python -c "from orbit_transfer.benchmark import TransferBenchmark; print('설치 OK')"

# 사용 가능한 케이스 목록 확인
python scripts/run_comparison.py --list
```

출력 예:
```
이름                                설명
--------------------------------------------------------------------------------
paper_main                          Seoul→Helsinki 단순화 (Δi=22.7°, h=561km)
di_05deg                            순수 경사각 변화 Δi=5°
di_10deg                            순수 경사각 변화 Δi=10°
T_03                                T_max_normed=0.3 (Hohmann 불가 케이스)
T_07                                T_max_normed=0.7 (Hohmann 가능 케이스)
da200_di05                          복합 전이 Δa=200 km, Δi=5°
...
```

---

### STEP 1: 단일 케이스 실행 (Hohmann $+$ Lambert만, 빠름)

콜로케이션 없이 임펄스 기법만 먼저 확인합니다. 거의 즉시 완료됩니다.

```bash
python scripts/run_comparison.py \
    --case paper_main \
    --no-collocation \
    --outdir results/quick
```

출력:
```
케이스: paper_main  (Seoul→Helsinki 단순화, Δi=22.7°, h=561km)
  [Hohmann] OK | Δv=2.983 km/s | ToF=2876 s
  [Lambert] OK | Δv=6.345 km/s | ToF=4027 s
→ 결과 저장: results/quick/paper_main/
```

---

### STEP 2: 콜로케이션 포함 전체 실행 ($5 \sim 30$ 초 소요)

```bash
python scripts/run_comparison.py \
    --case paper_main \
    --outdir results/full
```

출력:
```
  [Hohmann    ] OK | Δv=2.983 km/s | ToF=2876 s | t=0.1s
  [Lambert    ] OK | Δv=6.345 km/s | ToF=4027 s | t=0.0s
  [Collocation] OK | L1=3.598 km/s | ToF=3545 s | t=8.3s
```

---

### STEP 3: 결과 파일 확인

```bash
ls results/full/paper_main/
```

```
metrics_summary.csv          ← 비용·비행시간 수치 비교표 (핵심)
hohmann_trajectory.csv       ← Hohmann 궤적 (시계열)
lambert_trajectory.csv       ← Lambert 궤적
collocation_trajectory.csv   ← Collocation 궤적 + 추력
hohmann_impulses.csv         ← Hohmann 기동 시각 및 Δv
lambert_impulses.csv         ← Lambert 기동 시각 및 Δv
solve_times.csv              ← 계산 소요 시간
case_meta.json               ← 케이스 파라미터
```

`metrics_summary.csv` 내용 예시:

```csv
method,converged,is_impulsive,dv_total,cost_l1,cost_l2,tof,tof_norm
hohmann,True,True,2.98315,0.0,nan,2876.3,0.5000
lambert,True,True,6.34486,0.0,nan,4026.9,0.7000
collocation,True,False,nan,3.59798,0.00433,3545.3,0.6163
```

> **열 설명**:
> `dv_total` = 임펄스 기법의 총 $\Delta v$ [km/s],
> `cost_l1` = $\int \|\mathbf{u}\| \, dt$ [km/s],
> `cost_l2` = $\int \|\mathbf{u}\|^2 \, dt$ (목적함수 값),
> `tof_norm` = $T_f / T_0$

---

### STEP 4: 시각화 그림 생성

```bash
# Fig. C: 추력 프로파일 + 3D 궤적 (4-패널)
python scripts/plot_comparison.py \
    --outdir results/full \
    --figdir results/full/figures \
    --figs C

# 모든 그림 한 번에
python scripts/plot_comparison.py \
    --outdir results/full \
    --figdir results/full/figures
```

생성 파일:
```
results/full/figures/
├── figB_cost_bar.pdf        ← 비용 막대 그래프
└── figC_case_panel.pdf      ← 추력 프로파일 + 3D 궤적
```

---

### STEP 5: 보고서 핵심 결과 전체 재현

보고서의 그림 A~F를 모두 재현하려면 세 그룹 시뮬레이션이 필요합니다.
**총 소요 시간: 30~90분 (콜로케이션 포함 시)**

```bash
# ① 보고서 기준 케이스 (Fig. B, C)
python scripts/run_comparison.py \
    --case paper_main \
    --outdir results/paper_comparison/paper_comparison

# ② 시간 시리즈 T_max/T₀ = 0.3 ~ 1.1 (Fig. A, D, F)
#    콜로케이션 포함 시 9개 케이스 × ~10초 = 약 90초
python scripts/run_comparison.py \
    --group time_series \
    --outdir results/time_full

# ③ 고도+경사각 복합 케이스 (Fig. E)
#    콜로케이션 포함 시 9개 케이스 × ~15초 = 약 135초
python scripts/run_comparison.py \
    --group combined_series \
    --outdir results/combined_full

# 모든 그림 생성
python scripts/plot_comparison.py \
    --outdir results/paper_comparison \
    --figdir results/all_figures \
    --figs B C

python scripts/plot_comparison.py \
    --outdir results/time_full \
    --figdir results/all_figures \
    --figs A D F

python scripts/plot_comparison.py \
    --outdir results/combined_full \
    --figdir results/all_figures \
    --figs E
```

> **시간 절약 팁**: 콜로케이션은 계산 비용이 크므로,
> 먼저 `--no-collocation` 으로 Hohmann/Lambert만 확인한 뒤
> 관심 있는 케이스만 콜로케이션 포함으로 다시 실행하세요.

---

### STEP 6: Python 코드에서 직접 사용

스크립트 없이 Python 코드로 직접 결과를 얻을 수 있습니다.

```python
from orbit_transfer.benchmark import TransferBenchmark
from orbit_transfer.types import TransferConfig

# 임무 조건 정의
config = TransferConfig(
    h0          = 400.0,   # 초기 고도 [km]
    delta_a     = 0.0,     # 장반경 변화 [km]  (0: 고도 유지)
    delta_i     = 10.0,    # 경사각 변화 [deg]
    T_max_normed= 0.7,     # T_max / T_0
    u_max       = 0.01,    # 최대 추력 가속도 [km/s^2]
    h_min       = 200.0,   # 최소 허용 고도 [km]
)

# 벤치마크 객체 생성
bm = TransferBenchmark(config)

# 각 기법 실행 (원하는 것만 선택 가능)
bm.run_hohmann()
bm.run_lambert()
bm.run_collocation()   # 가장 오래 걸림

# 콘솔 요약 출력
bm.print_summary()

# 결과 접근
hoh = bm.results["hohmann"]
col = bm.results["collocation"]

print(f"Hohmann Delta v:        {hoh.metrics['dv_total']:.4f} km/s")  # → 2.9832
print(f"Collocation L1 cost:    {col.metrics['cost_l1']:.4f} km/s")  # → 3.5980
print(f"Collocation T_f/T_0:    {col.metrics['tof_norm']:.4f}")       # → 0.6163

# CSV로 저장
bm.export_csv(outdir="my_results/test_case")
```

---

## 8. 결과 파일 해석 방법

### metrics_summary.csv 핵심 열

| 열 이름 | 단위 | 의미 |
|---------|------|------|
| `dv_total` | km/s | 임펄스 기법의 총 속도 증분 $\sum \|\Delta\mathbf{v}_i\|$ (Hohmann/Lambert) |
| `cost_l1` | km/s | $L^1$ 비용: $\int_0^{T_f} \|\mathbf{u}(t)\| \, dt$ |
| `cost_l2` | km$^2$/s$^3$ | $L^2$ 비용: $\int_0^{T_f} \|\mathbf{u}(t)\|^2 \, dt$ (NLP 목적함수) |
| `tof` | s | 비행시간 $T_f$ (절대값) |
| `tof_norm` | — | $T_f / T_0$ |
| `converged` | True/False | 수렴 성공 여부 |

### 비용 비교 시 주의사항

- **Hohmann `dv_total`** vs **Collocation `cost_l1`**: 단위는 모두 km/s이지만 물리적 의미가 다릅니다.
  - `dv_total`: 순간 기동의 속도 변화량 합 $\Delta v_1 + \Delta v_2$ → 연료 소모에 직접 대응
  - `cost_l1`: 연속 추력의 시간 적분 $\int \|\mathbf{u}\| \, dt$ → 유사 연료 소모 지표
- **Collocation `cost_l2`**: 실제 NLP 목적함수 → 수렴 품질 확인에 사용

### `*_trajectory.csv` 열 구성

```
t[s],  x[km],  y[km],  z[km],  vx[km/s],  vy[km/s],  vz[km/s],
ux[km/s2],  uy[km/s2],  uz[km/s2]   ← Hohmann/Lambert는 모두 0.0
```

### `*_impulses.csv` 열 구성

```
t[s],    dv[km/s],  dvx[km/s],  dvy[km/s],  dvz[km/s]
   0.0,    0.668,     0.123,      -0.655,      0.000     ← 1번 기동 (Δv₁)
3378.2,    0.669,    -0.124,       0.654,      0.000     ← 2번 기동 (Δv₂)
```

---

## 9. 최적화 문제 수정 실험

`scripts/custom_optimization.py` 파일을 편집기로 열고,
파일 상단의 `[A]~[E]` 구역을 수정한 뒤 실행합니다.

```bash
python scripts/custom_optimization.py --compare
# --compare: Hohmann 기준값도 함께 출력
```

---

### [A] 임무 파라미터 변경

```python
# 파일 상단의 MISSION 객체를 수정
MISSION = TransferConfig(
    h0          = 600.0,   # ← 고도를 400 → 600 km 으로 변경
    delta_a     = 200.0,   # ← 200 km 고도 상승 추가
    delta_i     = 5.0,     # ← 경사각 변화를 10° → 5° 로 줄이기
    T_max_normed= 0.4,     # ← 0.4 로 줄이면 Hohmann 불가 조건
    u_max       = 0.01,
    h_min       = 200.0,
)
```

> **실험 제안**: `T_max_normed` $= 0.3,\, 0.4,\, 0.5,\, 0.6,\, 0.7$ 로 순차적으로 바꾸면서
> 실행하면 보고서 Fig. D와 동일한 결과를 재현할 수 있습니다.

---

### [B] 비용함수 변경

```python
COST_TYPE = "L2"        # 기본: 에너지 최소화  J = ∫‖u‖² dt
COST_TYPE = "L1_approx" # 연료 최소화 근사     J = ∫√(‖u‖²+ε²) dt
COST_TYPE = "L1_time"   # 혼합: T_f + w·∫‖u‖² dt
COST_TYPE = "custom"    # 직접 구현 (build_objective() 함수 수정)
```

**`L2` (기본)**: 목적함수

$$J = \int_0^{T_f} \|\mathbf{u}(t)\|^2 \, dt$$

수렴이 가장 안정적입니다. 결과 궤적은 부드럽고 추력이 분산되는 경향이 있습니다.

**`L1_approx`**: 목적함수

$$J = \int_0^{T_f} \sqrt{\|\mathbf{u}(t)\|^2 + \varepsilon^2} \, dt, \quad \varepsilon = \texttt{L1\_EPS}$$

$\varepsilon$ 이 작을수록 실제 $L^1$ 에 가까워지면서 추력이 bang-bang에 가까운 형태(켜짐/꺼짐)로 집중됩니다.
단, $\varepsilon$ 이 너무 작으면 수렴 실패율이 높아집니다.

**`L1_time`**: 목적함수

$$J = T_f + w \int_0^{T_f} \|\mathbf{u}(t)\|^2 \, dt, \quad w = \texttt{W\_TIME\_ENERGY}$$

$w$ 를 크게 할수록 에너지 중시, 작게 할수록 비행시간 단축을 우선합니다.

---

### [C] 물리 모델 변경

```python
INCLUDE_J2 = True    # 기본: J2 섭동 포함 (현실적)
INCLUDE_J2 = False   # J2 제거 (순수 2체 문제, 빠름)
```

$J_2$ 섭동 가속도는 다음과 같습니다.

$$
\mathbf{a}_{J_2} = \frac{3\mu J_2 R_E^2}{2\|\mathbf{r}\|^5}
\begin{bmatrix}
x\!\left(5z^2/\|\mathbf{r}\|^2 - 1\right) \\
y\!\left(5z^2/\|\mathbf{r}\|^2 - 1\right) \\
z\!\left(5z^2/\|\mathbf{r}\|^2 - 3\right)
\end{bmatrix}
$$

고도가 낮을수록 ($h < 600\,\mathrm{km}$) $J_2$ 효과가 크게 나타납니다.

> **실험 제안**: `INCLUDE_J2 = True/False` 를 바꾸면서 비용함수와 궤적이
> 얼마나 달라지는지 비교해 보세요.

---

### [D] 이산화 해상도 변경

```python
N_SEGMENTS = 30    # 기본값 (속도와 정확도의 균형)
N_SEGMENTS = 10    # 빠른 탐색용 (정확도 낮음, ~1초)
N_SEGMENTS = 60    # 고해상도 (정확도 높음, ~30초)
```

$M$ 을 늘리면 NLP 결정변수 수가 $9 \times (2M+1) + 3$ 개로 선형 증가합니다.
$M = 10$ 이면 약 192개, $M = 60$ 이면 약 1,086개입니다.

> **주의**: $M$ 이 너무 크면 IPOPT 수렴에 더 오래 걸리거나 수렴 실패율이 높아질 수 있습니다.

---

### [E] 추가 제약조건 추가

`add_extra_constraints()` 함수 안에 CasADi 문법으로 제약조건을 추가합니다.

```python
def add_extra_constraints(opti, X, U, T_f, N):
    # 예 1: 최대 속도 제한  ‖v‖ ≤ V_max
    V_MAX = 9.0  # km/s
    for k in range(N):
        opti.subject_to(ca.dot(X[3:, k], X[3:, k]) <= V_MAX**2)

    # 예 2: 비행시간 하한  T_f ≥ 1000 s
    opti.subject_to(T_f >= 1000.0)

    # 예 3: z방향 추력 금지  u_z = 0
    for k in range(N):
        opti.subject_to(U[2, k] == 0.0)
```

### 결과 출력 및 그림

```bash
python scripts/custom_optimization.py --compare --outdir results/my_exp
```

```
results/my_exp/
├── custom_result.pdf      ← 5-패널 결과 그림
├── custom_trajectory.csv  ← 궤적 데이터
└── custom_metrics.csv     ← 비용 지표
```

---

## 10. 파라미터 레퍼런스

### TransferConfig 파라미터

| 파라미터 | 단위 | 기본 범위 | 설명 |
|----------|------|----------|------|
| `h0` | km | $300 \sim 1200$ | 초기 원궤도 고도 |
| `delta_a` | km | $-500 \sim 2000$ | 장반경 변화 ($> 0$: 고도 증가) |
| `delta_i` | deg | $0 \sim 50$ | 궤도 경사각 변화 |
| `T_max_normed` | — | $0.15 \sim 1.2$ | $T_\mathrm{max} / T_0$ |
| `u_max` | km/s² | $0.001 \sim 0.1$ | 최대 추력 가속도 |
| `h_min` | km | $150 \sim 300$ | 최소 허용 고도 |
| `e0` | — | $0 \sim 0.1$ | 초기 궤도 이심률 ($0$: 원궤도) |
| `ef` | — | $0 \sim 0.1$ | 목표 궤도 이심률 |

**`T_max_normed` 임계값** ($\Delta a = 0$ 인 경우):

| 값 | 의미 |
|----|------|
| $< 0.5$ | Hohmann $T_\mathrm{Hoh} = 0.5\,T_0$ 초과 → **Hohmann 불가**, 연속 추력만 성공 |
| $= 0.5$ | Hohmann TOF와 정확히 같음 (경계 케이스) |
| $> 0.5$ | Hohmann 가능, 세 기법 모두 실행 가능 |

> $\Delta a \neq 0$ 인 경우 $T_\mathrm{Hoh} \neq T_0/2$ 이므로 임계값이 달라집니다.

### 물리 상수 (`constants.py`)

| 상수 | 값 | 단위 | 의미 |
|------|----|------|------|
| `MU_EARTH` | $3.986004418 \times 10^5$ | km$^3$/s$^2$ | 지구 중력 상수 $\mu$ |
| `R_E` | $6378.137$ | km | 지구 적도 반경 |
| `J2` | $1.08263 \times 10^{-3}$ | — | $J_2$ 계수 |

### 솔버 설정 (`config.py`)

| 설정 | 기본값 | 의미 |
|------|--------|------|
| `HS_NUM_SEGMENTS` | $30$ | Hermite-Simpson 구간 수 $M$ |
| `IPOPT_TOL` (Pass 1) | $10^{-4}$ | IPOPT 허용 오차 |
| `IPOPT_MAX_ITER` | $500$ | 최대 반복 횟수 |
| `MAX_NU_RETRIES` | $3$ | 수렴 실패 시 $\nu_0$ 재시도 횟수 |

---

## 11. 자주 겪는 문제와 해결책

### Q1. 콜로케이션이 "수렴 실패"로 표시됩니다

**원인**: IPOPT가 제약조건을 만족하는 실행가능해를 찾지 못한 경우.

**해결책**:
1. `T_max_normed` 를 조금 늘려보세요 (시간 여유 확보)
2. `u_max` 를 늘려보세요 (더 강한 추력 허용)
3. `h_min` 을 낮춰보세요 (고도 제약 완화)
4. `custom_optimization.py` 에서 `IPOPT_TOL = 1e-3` 으로 완화해 보세요
5. `N_SEGMENTS` 를 줄여보세요 (NLP 규모 축소)

### Q2. Lambert가 매우 큰 $\Delta v$ 값을 반환합니다

**원인 1 — 비행시간 선택**: `paper_main` 케이스에서 Lambert는 $T_\mathrm{max} = 0.7\,T_0$ 을 비행시간으로 사용합니다. 이 경우 도착 위치가 출발 위치와 큰 각도를 이루어 $\Delta v = 6.345\,\mathrm{km/s}$ 가 나옵니다 (보고서 Table 2 참조). 이는 예상된 결과입니다.

**원인 2 — $180°$ 특이점**: $\Delta a \neq 0$ 인 케이스에서 Hohmann TOF를 사용하면 도착 위치가 $\approx 180°$ 에 가까워져 수치적 발산이 발생할 수 있습니다.

**해결책**: `--lambert-tof` 옵션으로 비행시간을 직접 지정하거나, `da1000` 과 같이 $\Delta a$ 가 더 큰 케이스를 사용하세요.

### Q3. `casadi` 또는 `ipopt` 관련 ImportError가 발생합니다

```bash
# conda-forge에서 재설치
conda install -c conda-forge casadi
```

### Q4. 그림이 생성되지 않습니다

`plot_comparison.py` 는 `results/` 디렉토리에 시뮬레이션 결과가 있어야 실행됩니다.
먼저 `run_comparison.py` 를 실행해 결과 파일을 만들어야 합니다.

### Q5. `pip install -e .` 이후에도 import가 안 됩니다

```bash
# PYTHONPATH를 명시적으로 설정
PYTHONPATH=src python scripts/run_comparison.py --list

# 또는 가상환경이 활성화되어 있는지 확인
which python  # conda/venv 경로인지 확인
```

---

## 보고서 결과와 그림 대응

| 보고서 그림 | 내용 | 재현 명령 (2단계) |
|------------|------|-----------------|
| Fig. A | $T_\mathrm{max}$ 스윕 추력 프로파일 | `run_comparison.py --group time_series` → `plot_comparison.py --figs A` |
| Fig. B | 비용 막대 그래프 | `run_comparison.py --case paper_main` → `plot_comparison.py --figs B` |
| Fig. C | 케이스 4-패널 | `run_comparison.py --case paper_main` → `plot_comparison.py --figs C` |
| Fig. D | 비용 vs $T_\mathrm{max}$ | `run_comparison.py --group time_series` → `plot_comparison.py --figs D` |
| Fig. E | $\Delta a$–$\Delta i$ 히트맵 | `run_comparison.py --group combined_series` → `plot_comparison.py --figs E` |
| Fig. F | 시간제약 실현가능성 | `run_comparison.py --group time_series` → `plot_comparison.py --figs F` |

---

## 문의

- **담당 교수**: Suwon Lee (suwon.lee@kookmin.ac.kr)
- **소속**: Future Mobility Control Lab (FMCL), Kookmin University
