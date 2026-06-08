# C1. NLP 정식화 및 Interior-Point 솔버

> 대응 코드: `src/orbit_transfer/optimizer/solver.py`, `src/orbit_transfer/optimizer/two_pass.py`

## 1. 목적

연속 추력 궤도전이 문제는 본래 무한 차원 함수 공간에서 정의된 최적 제어 문제(Optimal Control Problem, OCP)이다. 상태 $\mathbf{x}(t) \in \mathbb{R}^6$과 제어 $\mathbf{u}(t) \in \mathbb{R}^3$은 시간의 연속 함수이며, 동역학은 상미분방정식(ODE) $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$로 기술된다. 이러한 OCP를 수치적으로 풀기 위해서는 유한 차원의 비선형 프로그래밍(Nonlinear Programming, NLP) 문제로 변환하는 이산화(transcription) 과정이 필요하다.

본 문서는 다음 세 가지 주제를 다룬다.

1. **OCP에서 NLP로의 변환**: Direct collocation에 의해 연속 시간 문제를 유한 차원 NLP 표준형으로 매핑하는 과정
2. **Interior-point 솔버(IPOPT)의 수학적 원리**: 장벽 함수, KKT 조건, 프라이멀-듀얼 Newton 시스템, 적응적 장벽 파라미터 전략
3. **Two-Pass 전략과 수렴 실패 복구**: 허용 오차 연속체(continuation), 다중 시작점 전략, 실행가능성 복원

구현은 CasADi의 Opti stack을 통해 NLP를 심볼릭으로 구성하고, IPOPT 솔버에 전달하는 구조이다. `solver.py`는 IPOPT 옵션 관리를, `two_pass.py`는 Two-Pass 최적화 파이프라인의 전체 흐름과 수렴 실패 복구 로직을 담당한다.

---

## 2. 수학적 배경

### 2.1 최적 제어 문제의 정식화

시간 구간 $[0, T]$에서 다음 볼차(Bolza) 형태의 최적 제어 문제를 고려한다.

$$
\min_{\mathbf{u}(t),\, \nu_0,\, \nu_f} \quad J = \int_0^T \|\mathbf{u}(t)\|^2 \, dt
$$

subject to:

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t),\, \mathbf{u}(t)), \quad t \in [0, T]
$$

$$
\mathbf{x}(0) = \mathbf{g}_0(\nu_0), \quad \mathbf{x}(T) = \mathbf{g}_f(\nu_f)
$$

$$
\|\mathbf{u}(t)\| \leq u_{\max}, \quad \|\mathbf{r}(t)\| \geq R_E + h_{\min}
$$

여기서 $\mathbf{x} = [\mathbf{r}^\top,\, \mathbf{v}^\top]^\top \in \mathbb{R}^6$은 ECI 좌표계의 위치-속도 상태벡터, $\mathbf{u} \in \mathbb{R}^3$은 추력 가속도 벡터이다. $\mathbf{f}$는 J2 섭동을 포함한 운동방정식이며, $\mathbf{g}_0(\nu_0)$과 $\mathbf{g}_f(\nu_f)$는 궤도요소에서 ECI 상태벡터로의 변환 함수이다. $\nu_0$와 $\nu_f$는 출발/도착 궤도상의 위치를 나타내는 자유변수로, 솔버가 최적의 출발-도착 위치를 자동으로 결정한다.

### 2.2 NLP 표준형

일반적인 NLP 문제는 다음과 같이 정의된다 (Nocedal and Wright, 2006).

$$
\min_{\mathbf{z} \in \mathbb{R}^n} \quad f(\mathbf{z})
$$

$$
\text{s.t.} \quad \mathbf{c}(\mathbf{z}) = \mathbf{0} \quad (\mathbf{c} : \mathbb{R}^n \to \mathbb{R}^{m_e})
$$

$$
\quad\quad\;\; \mathbf{d}(\mathbf{z}) \geq \mathbf{0} \quad (\mathbf{d} : \mathbb{R}^n \to \mathbb{R}^{m_i})
$$

여기서 $\mathbf{z}$는 결정변수 벡터, $f$는 목적함수, $\mathbf{c}$는 등식 제약, $\mathbf{d}$는 부등식 제약이다. $n$은 결정변수의 수, $m_e$는 등식 제약의 수, $m_i$는 부등식 제약의 수이다.

### 2.3 OCP에서 NLP로의 변환

Direct collocation 이산화를 적용하면, 연속 시간 OCP가 유한 차원 NLP로 변환된다. 결정변수 벡터는 다음과 같이 구성된다.

$$
\mathbf{z} = \begin{bmatrix} \text{vec}(\mathbf{X}) \\ \text{vec}(\mathbf{U}) \\ \nu_0 \\ \nu_f \end{bmatrix} \in \mathbb{R}^n
$$

여기서 $\mathbf{X} \in \mathbb{R}^{6 \times N}$은 $N$개 배치점에서의 상태, $\mathbf{U} \in \mathbb{R}^{3 \times N}$은 제어 배열이다. $\text{vec}(\cdot)$은 행렬을 열 벡터로 펼치는 연산이다.

**목적함수.** 연속 시간 비용함수 $J = \int_0^T \|\mathbf{u}\|^2\, dt$는 수치 적분 규칙에 의해 이산 비용함수 $f(\mathbf{z})$로 근사된다. Hermite-Simpson collocation에서는 Simpson 구적법을, Multi-Phase LGL collocation에서는 LGL 가중치 적분을 사용한다.

**등식 제약 $\mathbf{c}(\mathbf{z}) = \mathbf{0}$.** 등식 제약은 두 부류로 구성된다.

1. **Collocation defect 제약**: 각 부분 구간에서 이산화된 동역학 방정식의 잔차가 0이 되어야 한다는 조건이다. Hermite-Simpson의 경우 Simpson 연속성 조건과 Hermite 중간점 조건이, LGL의 경우 미분 행렬에 의한 collocation 조건과 phase간 linkage 조건이 이에 해당한다.

2. **경계조건**: 출발/도착 궤도의 ECI 상태벡터가 궤도요소로부터 결정된 값과 일치해야 한다.

$$
\mathbf{X}_{:,0} - \mathbf{g}_0(\nu_0) = \mathbf{0}, \quad \mathbf{X}_{:,-1} - \mathbf{g}_f(\nu_f) = \mathbf{0}
$$

**부등식 제약 $\mathbf{d}(\mathbf{z}) \geq \mathbf{0}$.** 부등식 제약은 각 배치점에서 부과된다.

$$
u_{\max}^2 - \mathbf{u}_j^\top \mathbf{u}_j \geq 0, \quad j = 0, 1, \ldots, N-1 \quad \text{(추력 상한)}
$$

$$
\mathbf{r}_j^\top \mathbf{r}_j - (R_E + h_{\min})^2 \geq 0, \quad j = 0, 1, \ldots, N-1 \quad \text{(최소 고도)}
$$

### 2.4 NLP의 규모

Pass 1(Hermite-Simpson, $M = 30$)에서의 NLP 규모는 다음과 같다.

| 구성요소 | 크기 |
|----------|------|
| 결정변수 $n$ | $6 \times 61 + 3 \times 61 + 2 = 551$ |
| 등식 제약 $m_e$ | $6 \times 30 + 6 \times 30 + 6 + 6 = 372$ |
| 부등식 제약 $m_i$ | $61 + 61 = 122$ |
| 제약 합계 | 494 |

Pass 2(Multi-Phase LGL)에서의 NLP 규모는 phase 구조에 따라 달라진다. 예를 들어 bimodal 프로파일에서 3개 phase(피크 15노드 + coasting 8노드 + 피크 15노드)를 사용하는 경우, 결정변수는 $6 \times 38 + 3 \times 38 + 2 = 344$개이다. Phase 경계에서의 linkage 제약이 추가되는 반면, 전체 노드 수가 줄어들어 collocation 제약 수는 감소한다.

### 2.5 KKT 최적성 조건

NLP 표준형의 라그랑지안(Lagrangian)을 다음과 같이 정의한다.

$$
\mathcal{L}(\mathbf{z}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{z}) - \boldsymbol{\lambda}^\top \mathbf{c}(\mathbf{z}) - \boldsymbol{\mu}^\top \mathbf{d}(\mathbf{z})
$$

여기서 $\boldsymbol{\lambda} \in \mathbb{R}^{m_e}$는 등식 제약의 라그랑주 승수, $\boldsymbol{\mu} \in \mathbb{R}^{m_i}$는 부등식 제약의 라그랑주 승수이다.

Karush-Kuhn-Tucker(KKT) 조건은 정칙성 가정(constraint qualification) 하에서 국소 최적해의 필요조건이다 (Nocedal and Wright, 2006).

**(1) 정상성 조건 (Stationarity):**

$$
\nabla_{\mathbf{z}} \mathcal{L} = \nabla f(\mathbf{z}) - \nabla \mathbf{c}(\mathbf{z})^\top \boldsymbol{\lambda} - \nabla \mathbf{d}(\mathbf{z})^\top \boldsymbol{\mu} = \mathbf{0}
$$

**(2) 프라이멀 실행가능성 (Primal feasibility):**

$$
\mathbf{c}(\mathbf{z}) = \mathbf{0}, \quad \mathbf{d}(\mathbf{z}) \geq \mathbf{0}
$$

**(3) 듀얼 실행가능성 (Dual feasibility):**

$$
\boldsymbol{\mu} \geq \mathbf{0}
$$

**(4) 상보성 조건 (Complementarity):**

$$
\mu_j \, d_j(\mathbf{z}) = 0, \quad j = 1, \ldots, m_i
$$

상보성 조건은 각 부등식 제약에 대해 "승수가 0이거나 제약이 등호로 활성(active)이다"라는 것을 의미한다. 이 조건들을 모두 벡터로 모으면 KKT 시스템을 얻는다. Interior-point 방법은 이 KKT 시스템의 완화된(perturbed) 버전을 Newton 방법으로 풀어나가는 전략이다.

### 2.6 Interior-Point (장벽) 방법

Interior-point 방법은 부등식 제약 $\mathbf{d}(\mathbf{z}) \geq \mathbf{0}$을 로그 장벽 함수(logarithmic barrier function)로 대체하여, 부등식 제약이 없는 등식 제약 문제의 수열로 변환하는 방법이다 (Fiacco and McCormick, 1968; Wachter and Biegler, 2006).

**장벽 부문제(Barrier subproblem).** 장벽 파라미터 $\mu > 0$에 대해 다음 문제를 정의한다.

$$
\min_{\mathbf{z}} \quad \varphi_\mu(\mathbf{z}) = f(\mathbf{z}) - \mu \sum_{j=1}^{m_i} \ln\bigl(d_j(\mathbf{z})\bigr)
$$

$$
\text{s.t.} \quad \mathbf{c}(\mathbf{z}) = \mathbf{0}
$$

로그 장벽 항 $-\mu \ln(d_j)$는 $d_j \to 0^+$일 때 $+\infty$로 발산하여, 해가 실행가능 영역의 내부(interior)에 머물도록 강제한다. $\mu \to 0^+$으로 감소시키면 장벽 부문제의 해가 원래 NLP의 해에 수렴한다.

**완화된 KKT 시스템.** 장벽 부문제의 KKT 조건은 원래 KKT 조건에서 상보성 조건만 변경된다.

$$
\nabla f(\mathbf{z}) - \nabla \mathbf{c}(\mathbf{z})^\top \boldsymbol{\lambda} - \nabla \mathbf{d}(\mathbf{z})^\top \boldsymbol{\mu} = \mathbf{0}
$$

$$
\mathbf{c}(\mathbf{z}) = \mathbf{0}
$$

$$
\mu_j \, d_j(\mathbf{z}) = \mu, \quad j = 1, \ldots, m_i \quad \text{(완화된 상보성)}
$$

$$
\boldsymbol{\mu} > \mathbf{0}, \quad \mathbf{d}(\mathbf{z}) > \mathbf{0}
$$

원래 상보성 조건 $\mu_j d_j = 0$이 $\mu_j d_j = \mu$로 완화되었다. $\mu \to 0$이면 원래의 상보성 조건이 복원된다.

### 2.7 프라이멀-듀얼 Newton 시스템

완화된 KKT 시스템을 Newton 방법으로 선형화하면, 각 반복에서 다음 선형 시스템을 풀어야 한다. 슬랙 변수 $\mathbf{s} = \mathbf{d}(\mathbf{z})$를 도입하면, 프라이멀-듀얼 Newton 시스템은 다음과 같다 (Wachter and Biegler, 2006).

$$
\begin{bmatrix}
\mathbf{W} & -\nabla \mathbf{c}^\top & -\nabla \mathbf{d}^\top & \mathbf{0} \\
\nabla \mathbf{c} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\nabla \mathbf{d} & \mathbf{0} & \mathbf{0} & -\mathbf{I} \\
\mathbf{0} & \mathbf{0} & \mathbf{S} & \mathbf{M}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{z} \\
\Delta \boldsymbol{\lambda} \\
\Delta \boldsymbol{\mu} \\
\Delta \mathbf{s}
\end{bmatrix}
= -
\begin{bmatrix}
\nabla_\mathbf{z} \mathcal{L} \\
\mathbf{c} \\
\mathbf{d} - \mathbf{s} \\
\mathbf{S} \mathbf{M} \mathbf{e} - \mu \mathbf{e}
\end{bmatrix}
$$

여기서

- $\mathbf{W} = \nabla^2_{zz} \mathcal{L}$: 라그랑지안의 헤시안
- $\mathbf{S} = \text{diag}(s_1, \ldots, s_{m_i})$, $\mathbf{M} = \text{diag}(\mu_1, \ldots, \mu_{m_i})$
- $\mathbf{e} = [1, \ldots, 1]^\top$

$\Delta \mathbf{s}$를 소거하면, 축소된(condensed) 시스템을 얻는다.

$$
\begin{bmatrix}
\mathbf{W} + \nabla \mathbf{d}^\top \mathbf{S}^{-1} \mathbf{M} \nabla \mathbf{d} & -\nabla \mathbf{c}^\top \\
\nabla \mathbf{c} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{z} \\
\Delta \boldsymbol{\lambda}
\end{bmatrix}
= -
\begin{bmatrix}
\nabla_\mathbf{z} \mathcal{L} + \nabla \mathbf{d}^\top (\boldsymbol{\mu} - \mu \mathbf{S}^{-1} \mathbf{e}) \\
\mathbf{c}
\end{bmatrix}
$$

이 축소 시스템은 대칭이지만 부정치(indefinite)인 행렬을 갖는다. IPOPT는 이 시스템을 직접 솔버(예: MUMPS)로 인수분해하여 풀며, 관성 보정(inertia correction)을 통해 수치적 안정성을 유지한다.

### 2.8 IPOPT의 적응적 장벽 파라미터 전략

장벽 파라미터 $\mu$의 감소 전략은 interior-point 방법의 수렴 속도를 결정짓는 핵심 요소이다. IPOPT는 두 가지 전략을 제공한다 (Wachter and Biegler, 2006).

**(1) 단조 감소(monotone) 전략.** 고전적인 접근법으로, 장벽 부문제가 충분히 수렴하면 $\mu$를 고정 비율로 감소시킨다.

$$
\mu_{k+1} = \kappa_\mu \, \mu_k, \quad 0 < \kappa_\mu < 1
$$

**(2) 적응적(adaptive) 전략.** 현재 상보성 잔차에 기반하여 $\mu$를 적응적으로 설정한다.

$$
\mu_{k+1} = \max\left(\frac{\epsilon_{\text{tol}}}{10},\; \min\left(\kappa_\mu \, \mu_k,\; \mu_k^{\theta_\mu}\right)\right)
$$

여기서 $\theta_\mu > 1$은 초선형 수렴을 유도하는 지수이다. 적응적 전략은 $\mu$가 빠르게 감소하지만 실행가능 영역을 벗어나지 않는 균형점을 추구한다.

본 구현에서는 `mu_strategy = "adaptive"`를 사용한다. 적응적 전략은 궤적 최적화와 같이 제약조건의 비선형성이 강한 문제에서 단조 전략보다 안정적인 수렴을 보이는 것으로 알려져 있다 (Betts, 2010).

### 2.9 MUMPS 직접 솔버

각 Newton 반복에서 축소된 KKT 시스템(2.7절)의 계수 행렬은 대칭 부정치 희소 행렬(sparse symmetric indefinite matrix)이다. MUMPS(MUltifrontal Massively Parallel Sparse direct Solver)는 이러한 행렬에 대해 Bunch-Kaufman 피벗팅 기반의 $\mathbf{LDL}^\top$ 분해를 수행한다 (Amestoy et al., 2001).

$$
\mathbf{K} = \mathbf{P} \mathbf{L} \mathbf{D} \mathbf{L}^\top \mathbf{P}^\top
$$

여기서 $\mathbf{P}$는 순열 행렬, $\mathbf{L}$은 하삼각 행렬, $\mathbf{D}$는 $1 \times 1$ 및 $2 \times 2$ 블록 대각 행렬이다.

MUMPS는 다중전면(multifrontal) 알고리즘을 사용하여 fill-in을 최소화하는 재정렬(ordering)을 자동으로 수행한다. 본 문제(변수 수 $n \leq 551$)와 같은 중규모 NLP에서는 직접 솔버가 반복적 솔버(iterative solver)보다 안정적이고 효율적이다.

IPOPT는 $\mathbf{D}$의 관성(inertia)을 검사하여 헤시안의 양정치성을 간접적으로 확인한다. 관성이 올바르지 않으면 정규화 항 $\delta_w \mathbf{I}$를 헤시안에 추가하고 인수분해를 반복한다.

### 2.10 CasADi Opti Stack과 자동 미분

CasADi는 심볼릭 프레임워크를 통해 NLP를 자동으로 구축한다 (Andersson et al., 2019). Opti stack은 다음 과정을 자동화한다.

1. **심볼릭 그래프 구축**: `opti.variable()`, `opti.minimize()`, `opti.subject_to()`를 통해 결정변수, 목적함수, 제약조건의 심볼릭 표현을 생성한다.
2. **자동 미분(Automatic Differentiation, AD)**: 순전파(forward) 및 역전파(reverse) AD를 통해 야코비안과 헤시안을 정확하게 계산한다. 유한 차분 근사와 달리, 기계 정밀도 수준의 도함수를 얻는다.
3. **희소성 탐지**: 심볼릭 그래프를 분석하여 야코비안 및 헤시안의 희소 구조를 자동으로 파악한다. Direct collocation 문제에서 야코비안은 대역(banded) 구조를 가지며, CasADi는 이를 활용하여 0이 아닌 원소만 계산한다.
4. **NLP 인터페이스**: 구축된 심볼릭 NLP를 IPOPT의 C++ 인터페이스에 전달한다. 솔버가 야코비안/헤시안을 요청할 때마다 CasADi가 AD를 통해 효율적으로 평가한다.

이러한 자동화 덕분에, 사용자는 수학적 정식화에 집중할 수 있으며 도함수를 수동으로 유도할 필요가 없다.

### 2.11 Two-Pass 전략의 수학적 근거

Two-Pass 전략은 허용 오차 연속체(tolerance continuation)의 한 형태이다. 일반적으로 NLP의 수렴은 초기값의 품질과 허용 오차 수준에 민감하다. 느슨한 허용 오차에서 먼저 대략적인 해를 구하고, 이를 초기값으로 사용하여 엄격한 허용 오차에서 다시 풀면, 수렴 가능성이 향상된다 (Betts, 2010).

**Pass 1 (Hermite-Simpson, 탐색 단계)**:

- 허용 오차: $\epsilon_1 = 10^{-4}$
- 목적: 추력 프로파일의 전역적 구조(피크 위치, 개수) 파악
- 균일 격자($M = 30$)로 전체 시간 구간을 커버

**Pass 2 (Multi-Phase LGL, 정밀화 단계)**:

- 허용 오차: $\epsilon_2 = 10^{-6}$ ($= \epsilon_1 / 100$)
- 목적: Pass 1에서 발견한 추력 프로파일 구조를 기반으로 정밀 최적화
- 적응적 격자: 피크 구간에 15개, coasting 구간에 8개 LGL 노드 배치
- Warm start: Pass 1 해를 보간하여 초기값으로 사용

수학적으로, $\epsilon_1 > \epsilon_2$인 두 허용 오차에서의 해 $\mathbf{z}_1^*$, $\mathbf{z}_2^*$ 사이의 관계는 다음과 같다. $\epsilon_1$-최적해 $\mathbf{z}_1^*$가 $\epsilon_2$-최적해 $\mathbf{z}_2^*$의 수렴 반경 내에 있을 확률이, 임의의 초기값을 사용하는 것보다 높다. 이는 해의 연속 의존성(continuous dependence)에 근거한다.

### 2.12 수렴 실패 복구의 수학적 근거

궤적 최적화 NLP는 비볼록(non-convex)이므로, 초기값에 따라 국소 최적해가 달라지거나 수렴에 실패할 수 있다. `TwoPassOptimizer`는 다음 세 단계의 복구 전략을 순차적으로 적용한다.

**(1) 다중 시작점 전략 (Multi-start).**

$$
\nu_0^{(k)} \sim \mathcal{U}(0, 2\pi), \quad \nu_f^{(k)} \sim \mathcal{U}(0, 2\pi), \quad k = 1, 2, 3
$$

True anomaly $\nu_0$, $\nu_f$를 무작위로 변경하여 최대 3회 재시도한다. 경계조건이 $\nu_0$, $\nu_f$에 비선형적으로 의존하므로, 다른 $\nu$ 값은 NLP의 실행가능 영역의 기하학적 구조를 변경하여 새로운 수렴 경로를 열 수 있다. 원궤도에서 $\nu$는 argument of latitude와 동치이므로, $[0, 2\pi)$에서의 균일 샘플링은 궤도상의 모든 위치를 동등하게 탐색한다.

재현성을 위해 난수 생성기의 시드를 42로 고정한다(`np.random.default_rng(42)`).

**(2) 허용 오차 완화 (Tolerance relaxation).**

$$
\epsilon_{\text{relaxed}} = \epsilon_1 \times \alpha, \quad \alpha = 10
$$

허용 오차를 $10^{-4}$에서 $10^{-3}$으로 10배 완화한다. 이는 실행가능성 복원(feasibility restoration)의 한 형태로 볼 수 있다. 더 넓은 허용 오차는 KKT 조건의 만족 범위를 확대하여, 엄격한 조건에서는 수렴하지 않았을 해를 찾을 수 있게 한다.

**(3) 수렴 실패 반환.**

모든 복구 시도가 실패하면, 최종 반복의 해를 `converged = False`로 표시하여 반환한다. 비용은 $+\infty$로 설정되어, 후속 분석에서 해당 케이스가 유효하지 않음을 나타낸다.

---

## 3. 구현 매핑

### 3.1 solver.py: IPOPT 옵션 관리

`solver.py`(32행)는 두 개의 함수 `get_pass1_options`과 `get_pass2_options`으로 구성되며, Pass별 IPOPT 옵션 딕셔너리를 반환한다.

| 함수 | 행 | 기본 옵션 원천 | 용도 |
|------|------|------|------|
| `get_pass1_options(**overrides)` | 6--17행 | `IPOPT_OPTIONS_PASS1` | H-S collocation 솔버 설정 |
| `get_pass2_options(**overrides)` | 20--31행 | `IPOPT_OPTIONS_PASS2` | Multi-Phase LGL 솔버 설정 |

두 함수 모두 `**overrides` 키워드 인자를 받아 기본 설정을 덮어쓸 수 있다. 이 설계는 기본 설정과 케이스별 예외 설정을 분리하며, 수렴 실패 복구 시 특정 옵션(예: `tol`)만 선택적으로 변경하는 데 사용된다.

### 3.2 IPOPT 옵션 상세

Pass 1과 Pass 2의 IPOPT 옵션은 `config.py`에 정의되어 있다.

| 파라미터 | Pass 1 | Pass 2 | 의미 |
|----------|--------|--------|------|
| `tol` | $10^{-4}$ | $10^{-6}$ | 전체 KKT 오차 수렴 기준 |
| `constr_viol_tol` | $10^{-4}$ | $10^{-6}$ | 제약조건 위반 허용 오차 |
| `max_iter` | 500 | 1000 | 최대 반복 횟수 |
| `linear_solver` | MUMPS | MUMPS | 선형 시스템 직접 솔버 |
| `mu_strategy` | adaptive | adaptive | 장벽 파라미터 감소 전략 |
| `warm_start_init_point` | -- | yes | Warm start 활성화 |
| `warm_start_bound_push` | -- | $10^{-6}$ | Warm start 경계 여유 |
| `print_level` | 0 | 0 | 출력 억제 |

**수렴 허용 오차의 단계적 감소.** Pass 1에서 $10^{-4}$, Pass 2에서 $10^{-6}$으로 100배 엄격해진다. 이는 2.11절에서 논의한 허용 오차 연속체의 구현이다.

**Warm start 설정.** Pass 2는 `warm_start_init_point = "yes"`로 설정되어, IPOPT가 초기 장벽 파라미터와 슬랙 변수를 사용자 제공 초기값으로부터 추정한다. `warm_start_bound_push = 1e-6`은 초기값이 변수 경계에 매우 가까울 때의 수치적 문제를 방지하기 위한 최소 여유이다. Warm start는 cold start 대비 반복 횟수를 현저히 줄일 수 있다. Pass 1 해를 보간한 초기값이 이미 실행가능 영역 근방에 있으므로, IPOPT는 실행가능성 복원 단계를 건너뛰고 곧바로 최적성 개선에 집중할 수 있다.

### 3.3 two_pass.py: Two-Pass 최적화 파이프라인

`TwoPassOptimizer` 클래스(15--114행)는 전체 Two-Pass 파이프라인을 구현한다.

**클래스 구조:**

```
TwoPassOptimizer
├── __init__(config: TransferConfig)
└── solve() -> TrajectoryResult
```

**`solve()` 메서드의 실행 흐름:**

| 단계 | 행 | 동작 | NLP 솔버 호출 |
|------|------|------|------|
| 초기값 생성 | 40--42행 | `linear_interpolation_guess()` | -- |
| Pass 1 풀이 | 44--47행 | H-S collocation NLP 풀이 | IPOPT (tol=$10^{-4}$) |
| 복구: $\nu$ 변경 | 51--63행 | 최대 3회 재시도 | IPOPT (tol=$10^{-4}$) |
| 복구: 허용오차 완화 | 65--78행 | tol=$10^{-3}$으로 재시도 | IPOPT (tol=$10^{-3}$) |
| 실패 반환 | 80--82행 | `converged=False` 반환 | -- |
| 피크 탐지 | 87--94행 | 추력 크기 프로파일 분석 | -- |
| Phase 구조 결정 | 92--94행 | 피크 위치/폭 기반 phase 분할 | -- |
| 보간 (warm start) | 97--99행 | Pass 1 해 $\to$ Pass 2 격자 보간 | -- |
| Pass 2 풀이 | 102--106행 | Multi-Phase LGL NLP 풀이 | IPOPT (tol=$10^{-6}$, warm start) |
| Pass 2 실패 시 대체 | 109--112행 | Pass 1 결과로 대체 | -- |

### 3.4 수렴 실패 복구 흐름

수렴 실패 복구는 3단계 위계 구조를 가진다. 각 단계는 이전 단계가 실패할 때만 실행된다.

**단계 1: 기본 풀이 (44--47행).**

선형 보간 초기값(`linear_interpolation_guess`)과 기본 설정으로 Pass 1 NLP를 풀이한다. 대부분의 케이스는 이 단계에서 수렴한다.

**단계 2: True anomaly 랜덤 변경 (49--63행).**

Pass 1이 수렴하지 않으면, 고정 시드(42)의 균일 분포에서 $(\nu_0, \nu_f)$를 샘플링하여 최대 `MAX_NU_RETRIES = 3`회 재시도한다. 각 시도에서 초기값도 재생성한다.

```python
rng = np.random.default_rng(42)
for _ in range(MAX_NU_RETRIES):
    nu0_r = rng.uniform(0, 2 * np.pi)
    nuf_r = rng.uniform(0, 2 * np.pi)
    ...
```

**단계 3: 허용 오차 완화 (65--78행).**

$\nu$ 변경으로도 수렴하지 않으면, 허용 오차를 `TOL_RELAXATION_FACTOR = 10`배 완화한다.

```python
relaxed_opts = {
    'ipopt.tol': 1e-4 * TOL_RELAXATION_FACTOR,          # 1e-3
    'ipopt.constr_viol_tol': 1e-4 * TOL_RELAXATION_FACTOR,  # 1e-3
}
```

**단계 4: 최종 실패 (80--82행).**

모든 복구 시도가 실패하면 `converged = False`, `pass1_cost = None`으로 결과를 반환한다.

### 3.5 Pass 1에서 Pass 2로의 전환

Pass 1이 수렴하면, 추력 크기 프로파일에서 피크를 탐지하고 phase 구조를 결정한다.

```python
u_mag = np.linalg.norm(result1.u, axis=0)
n_peaks, peak_times, peak_widths = detect_peaks(result1.t, u_mag, self.config.T)
profile_class = classify_profile(n_peaks)
phases = determine_phase_structure(peak_times, peak_widths, self.config.T)
```

`phases`는 딕셔너리의 리스트로, 각 phase의 시작/종료 시각과 노드 수를 정의한다. 피크 구간에는 `LGL_NODES_PEAK = 15`개, coasting 구간에는 `LGL_NODES_COAST = 8`개의 LGL 노드가 배정된다.

Pass 1 해를 Pass 2 격자로 보간하는 `interpolate_pass1_to_pass2` 함수가 warm start 초기값을 생성한다. 이 보간 과정은 Hermite-Simpson의 균일 격자에서 LGL의 비균일 격자로의 사상(mapping)이다.

### 3.6 Pass 2 실패 시 대체 전략

Pass 2(LGL)가 수렴하지 않으면, Pass 1 결과를 최종 해로 대체한다(109--112행). 이는 "정밀화에 실패하더라도 대략적인 해를 확보한다"는 실용적 전략이다. Pass 1 비용(`pass1_cost`)은 두 Pass의 비용 비교를 위해 별도로 기록된다.

```python
if not result2.converged:
    result1.pass1_cost = pass1_cost
    return result1
```

---

## 4. 수치 검증

### 4.1 IPOPT 수렴 판정

IPOPT는 다음 세 조건을 모두 만족할 때 수렴을 선언한다 (Wachter and Biegler, 2006).

$$
\max\left(\frac{\|\nabla_\mathbf{z} \mathcal{L}\|_\infty}{s_d},\; \frac{\|\mathbf{c}\|_\infty}{s_c},\; \frac{\mu_j s_j - \mu}{s_c}\right) \leq \epsilon_{\text{tol}}
$$

여기서 $s_d$, $s_c$는 문제 크기에 의존하는 스케일링 인자이다. `tol = 1e-4` (Pass 1) 또는 `tol = 1e-6` (Pass 2)가 $\epsilon_{\text{tol}}$에 해당한다.

또한, 제약조건 위반의 무한대 노름이 별도의 허용 오차 이하여야 한다.

$$
\|\mathbf{c}\|_\infty \leq \epsilon_{\text{constr}}, \quad \min_j d_j(\mathbf{z}) \geq -\epsilon_{\text{constr}}
$$

여기서 $\epsilon_{\text{constr}}$은 `constr_viol_tol`에 해당한다.

### 4.2 레퍼런스 케이스 수렴 검증

테스트 코드(`tests/`)에서 다음 레퍼런스 케이스의 수렴을 검증한다.

| 케이스 | $\Delta a$ [km] | $\Delta i$ [deg] | $T/T_0$ | 검증 항목 |
|--------|----------------|-------------------|---------|-----------|
| R1 | +200 | 0 | 2.0 | 수렴, $J > 0$, 추력 상한 |
| R2 | +500 | 5 | 3.0 | 수렴, 복합 기동 |
| R3 | 0 | 10 | 2.5 | 수렴, 순수 면변환 |
| R4 | $-200$ | 0 | 1.5 | 수렴, 궤도 하강 |
| R5 | +1000 | 10 | 4.0 | 수렴, 대규모 전이 |

각 케이스에서 확인하는 항목은 다음과 같다.

1. **수렴 여부**: `result.converged == True`
2. **비용 유효성**: $0 < J < \infty$
3. **추력 상한 만족**: $\|\mathbf{u}_j\| \leq u_{\max} + 10^{-6}$, $\forall j$
4. **최소 고도 만족**: $\|\mathbf{r}_j\| - R_E \geq h_{\min} - 1.0$ km, $\forall j$
5. **경계조건 만족**: 출발/도착 궤도요소와의 일치 (제약조건 위반 허용 오차 이내)

### 4.3 Two-Pass 수렴 성능

Two-Pass 전략의 효과를 다음 지표로 평가한다.

**(1) Pass 2 비용 대 Pass 1 비용 비율.**

$$
\rho = \frac{J_2}{J_1}
$$

$\rho < 1$이면 Pass 2가 더 나은 해를 찾은 것이다. 적응적 격자 배치(피크 구간에 노드 집중)와 더 엄격한 허용 오차 덕분에, 대부분의 케이스에서 $\rho < 1$을 관측한다.

**(2) 수렴 실패율.** 파라미터 공간 전체에서 최종 수렴 실패(`converged = False`) 비율을 모니터링한다. 복구 전략이 적용된 후에도 수렴하지 않는 케이스는 파라미터 공간의 경계 영역(매우 짧은 전이 시간, 매우 큰 면변환 등)에 집중되는 경향이 있다.

### 4.4 수치 안정성 검증

**(1) 야코비안 희소성.** CasADi의 심볼릭 분석에 의해 야코비안의 비영 원소(nonzero) 비율을 확인한다. Hermite-Simpson collocation의 야코비안은 대역 구조를 가지며, 비영 원소 비율이 약 5--10%로 희소하다. 이 희소성이 MUMPS의 효율적 인수분해를 가능하게 한다.

**(2) KKT 행렬의 관성.** IPOPT는 매 반복에서 축소 KKT 행렬의 관성(양수/음수/영 고유값의 수)을 검사한다. 올바른 관성은 $(n, m_e, 0)$이다. 관성이 틀리면 정규화를 추가하고 재인수분해한다. 로그에서 "Number of trial factorizations"가 1이면 관성 보정 없이 진행된 것이다.

**(3) Warm start 효과.** Pass 2에서 warm start가 활성화되면, IPOPT의 초기 장벽 파라미터 $\mu_0$가 cold start보다 작은 값으로 설정된다. 이는 초기 반복에서 이미 실행가능 영역 근방에 있으므로, 장벽을 빠르게 줄여도 안전하기 때문이다.

---

## 5. 참고문헌

1. Betts, J. T. (1998). "Survey of numerical methods for trajectory optimization," *Journal of Guidance, Control, and Dynamics*, 21(2), 193--207.
2. Betts, J. T. (2010). *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed. SIAM.
3. Conway, B. A. (Ed.) (2010). *Spacecraft Trajectory Optimization*. Cambridge University Press.
4. Wachter, A. and Biegler, L. T. (2006). "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming," *Mathematical Programming*, 106(1), 25--57.
5. Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.
6. Fiacco, A. V. and McCormick, G. P. (1968). *Nonlinear Programming: Sequential Unconstrained Minimization Techniques*. John Wiley & Sons.
7. Amestoy, P. R., Duff, I. S., Koster, J., and L'Excellent, J.-Y. (2001). "A fully asynchronous multifrontal solver using distributed dynamic scheduling," *SIAM Journal on Matrix Analysis and Applications*, 23(1), 15--41.
8. Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., and Diehl, M. (2019). "CasADi: a software framework for nonlinear optimization and optimal control," *Mathematical Programming Computation*, 11(1), 1--36.
9. Gill, P. E., Murray, W., and Saunders, M. A. (2005). "SNOPT: An SQP algorithm for large-scale constrained optimization," *SIAM Review*, 47(1), 99--131.
