# B3. Multi-Phase Collocation 정식화

> 대응 코드: `src/orbit_transfer/collocation/multiphase_lgl.py`, `src/orbit_transfer/collocation/interpolation.py`

## 1. 목적

연속 추력 궤도전이 문제에서 최적 추력 프로파일은 일반적으로 매끄러운(smooth) 구간과 추력이 거의 0에 가까운 coasting 구간이 교대로 나타나는 구조를 보인다. 단일 pseudospectral 구간으로 이러한 프로파일 전체를 이산화하면, 추력이 급격히 변하는 구간에서 다항식 근사의 정확도가 저하되거나 과도한 노드 수가 필요하게 된다. 이는 Gibbs 현상과 관련된 spectral method의 근본적 한계에 기인한다.

Multi-phase collocation은 이 문제를 해결하기 위해 전이 시간 영역을 복수의 독립적인 phase로 분할하고, 각 phase에서 별도의 pseudospectral 이산화를 적용하는 접근법이다. 각 phase는 독립적인 상태 변수와 제어 변수를 갖되, phase 경계에서의 상태 연속성은 linkage 제약으로 보장한다.

본 구현에서는 Two-Pass 전략의 두 번째 단계(Pass 2)로 multi-phase LGL pseudospectral collocation을 사용한다. Pass 1에서 Hermite-Simpson 방법으로 초기 해를 구한 후, 추력 프로파일의 피크 탐지 결과를 기반으로 phase 구조를 결정하고, Pass 2에서 더 높은 정확도의 해를 구한다. 이 보고서는 multi-phase 최적 제어 문제의 수학적 정식화, phase 분할 알고리즘, LGL collocation의 multi-phase 확장, 그리고 Pass 1 해로부터의 warm start 보간을 상세히 기술한다.

---

## 2. 수학적 배경

### 2.1 Multi-Phase 최적 제어 문제

전이 시간 $[0, T]$를 $P$개의 연속 구간(phase)으로 분할한다.

$$
[0, T] = [t_0^{(1)}, t_f^{(1)}] \cup [t_0^{(2)}, t_f^{(2)}] \cup \cdots \cup [t_0^{(P)}, t_f^{(P)}]
$$

여기서 인접 phase의 경계는 일치한다: $t_f^{(p)} = t_0^{(p+1)}$, $p = 1, \ldots, P-1$.

각 phase $p$에서의 상태 벡터를 $\mathbf{x}^{(p)}(t) \in \mathbb{R}^6$, 제어 벡터를 $\mathbf{u}^{(p)}(t) \in \mathbb{R}^3$으로 정의하면, multi-phase 최적 제어 문제(OCP)는 다음과 같이 정식화된다.

**비용 함수:**

$$
J = \sum_{p=1}^{P} \int_{t_0^{(p)}}^{t_f^{(p)}} \|\mathbf{u}^{(p)}(t)\|^2 \, dt
$$

**Phase별 동역학 제약:**

$$
\dot{\mathbf{x}}^{(p)}(t) = \mathbf{f}(\mathbf{x}^{(p)}(t), \mathbf{u}^{(p)}(t)), \quad t \in [t_0^{(p)}, t_f^{(p)}], \quad p = 1, \ldots, P
$$

여기서 $\mathbf{f}$는 J2 섭동을 포함한 우주선 운동방정식이다. 상태 벡터 $\mathbf{x} = [\mathbf{r}^\top, \mathbf{v}^\top]^\top$은 ECI 좌표계에서의 위치 $\mathbf{r} \in \mathbb{R}^3$과 속도 $\mathbf{v} \in \mathbb{R}^3$으로 구성되며, 제어 $\mathbf{u}$는 추력 가속도 벡터 [km/s$^2$]이다. 동역학 함수는 다음과 같다.

$$
\mathbf{f}(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} \mathbf{v} \\ \mathbf{a}_\text{grav}(\mathbf{r}) + \mathbf{a}_{J2}(\mathbf{r}) + \mathbf{u} \end{bmatrix}
$$

**Linkage 제약 (상태 연속성):**

$$
\mathbf{x}^{(p)}(t_f^{(p)}) = \mathbf{x}^{(p+1)}(t_0^{(p+1)}), \quad p = 1, \ldots, P-1
$$

이 제약은 인접 phase의 경계에서 상태 벡터가 연속임을 보장한다. 제어 벡터에 대해서는 연속성을 요구하지 않으므로, phase 경계에서 추력 프로파일의 불연속이 허용된다.

**경계 조건:**

$$
\mathbf{x}^{(1)}(t_0^{(1)}) = \mathbf{x}_0(\nu_0), \quad \mathbf{x}^{(P)}(t_f^{(P)}) = \mathbf{x}_f(\nu_f)
$$

여기서 $\nu_0$, $\nu_f$는 각각 출발 궤도와 도착 궤도에서의 true anomaly이다. 초기/종단 궤도는 원궤도(이심률 $e = 0$)이므로, 궤도요소-ECI 변환에서 근점이각(argument of periapsis) $\omega = 0$으로 고정하고 $\nu$를 argument of latitude로 취급한다. $\nu_0$와 $\nu_f$는 NLP의 자유 변수로서 최적화기가 결정한다.

**경로 제약:**

$$
\|\mathbf{u}^{(p)}(t)\|^2 \leq u_\max^2, \quad \|\mathbf{r}^{(p)}(t)\| \geq R_E + h_\min
$$

추력 가속도의 크기가 상한 $u_\max$를 초과하지 않아야 하며, 우주선의 고도가 최소 허용 고도 $h_\min$ 이상이어야 한다.

### 2.2 Phase 분할 알고리즘

Phase 구조는 Pass 1 해의 추력 프로파일로부터 피크 탐지(peak detection) 결과에 기반하여 결정된다.

**피크 탐지.** Pass 1에서 얻은 추력 크기 프로파일 $\|\mathbf{u}(t)\|$에 이동평균 smoothing을 적용한 후, `scipy.signal.find_peaks`로 피크를 탐지한다. 피크 판별 기준은 두 가지이다.

- Prominence: $\text{prominence}_j > \alpha \cdot \max\|\mathbf{u}\|$, 여기서 $\alpha = 0.1$
- 최소 피크 간격: $\text{distance} > \beta \cdot N_t$, 여기서 $\beta = 0.1$, $N_t$는 시간 노드 수

탐지된 각 피크에 대해 반치폭(Full Width at Half Maximum, FWHM) $w_k$를 `scipy.signal.peak_widths`로 추정한다. FWHM은 피크 높이의 절반에 해당하는 폭을 인덱스 단위로 계산한 후, 평균 시간 간격을 곱하여 시간 단위로 변환한다.

**Phase 구조 결정 규칙.** 피크 개수 $n_\text{peaks}$에 따라 phase 구조가 결정된다.

| $n_\text{peaks}$ | Phase 구조 | Phase 수 |
|:-:|:--|:-:|
| 0 | 단일 coast phase $[0, T]$ | 1 |
| 1 (unimodal) | 단일 peak phase $[0, T]$ | 1 |
| 2 (bimodal) | peak - coast - peak | 3 |
| $n$ (multimodal) | peak - coast - $\cdots$ - coast - peak | $2n - 1$ |

Bimodal 이상의 경우, 각 피크 구간의 경계는 FWHM을 기준으로 설정한다. $k$번째 피크의 시각이 $t_k$, FWHM이 $w_k$일 때, peak phase의 경계는 다음과 같다.

$$
[t_k - w_k, \; t_k + w_k]
$$

첫 번째 피크의 시작 경계는 $\max(t_1 - w_1, \; 0)$으로, 마지막 피크의 종료 경계는 $\min(t_n + w_n, \; T)$로 클리핑한다. 인접 피크 사이의 구간은 coast phase로 분류한다.

**노드 수 배정.** Phase 유형에 따라 LGL 노드 수를 차등 배정한다.

- Peak phase: $N_\text{peak} = 15$ 노드
- Coast phase: $N_\text{coast} = 8$ 노드

Peak phase에서는 추력이 빠르게 변하므로 높은 다항식 차수가 필요하다. $N = 15$이면 14차 다항식 근사를 사용하며, LGL 구적법은 $2 \times 14 - 1 = 27$차 다항식까지 정확하게 적분한다. Coast phase에서는 추력이 거의 0이므로 낮은 차수로도 충분하다.

**짧은 Phase 병합.** Phase 길이가 전이 시간의 $\gamma = 0.05$ 미만인 경우, 해당 phase를 인접 phase에 병합한다.

$$
\Delta t^{(p)} = t_f^{(p)} - t_0^{(p)} < \gamma \cdot T \quad \Longrightarrow \quad \text{인접 phase에 흡수}
$$

병합 우선순위는 이전(왼쪽) phase에 부여한다. 병합 과정은 더 이상 짧은 phase가 없을 때까지 반복적으로 수행한다. 이 처리는 피크 간격이 매우 좁은 경우 과도하게 짧은 coast phase가 생성되는 것을 방지한다. 과도하게 짧은 phase는 시간 스케일링 행렬의 조건수를 악화시켜 NLP 솔버의 수렴성을 저해할 수 있다.

### 2.3 시간 영역 변환

LGL pseudospectral method는 정규화 구간 $[-1, 1]$에서 정의된다. 각 phase의 물리 시간 구간 $[t_0^{(p)}, t_f^{(p)}]$을 정규화 시간 $\tau \in [-1, 1]$로 변환하는 아핀(affine) 사상은 다음과 같다.

$$
t = t_0^{(p)} + \frac{\Delta t^{(p)}}{2}(\tau + 1), \quad \Delta t^{(p)} = t_f^{(p)} - t_0^{(p)}
$$

역변환은

$$
\tau = \frac{2(t - t_0^{(p)})}{\Delta t^{(p)}} - 1
$$

이다. 이 변환에서 $\tau = -1$은 $t = t_0^{(p)}$에, $\tau = 1$은 $t = t_f^{(p)}$에 대응한다.

시간 미분에 대한 chain rule은 다음과 같다.

$$
\frac{dt}{d\tau} = \frac{\Delta t^{(p)}}{2}, \quad \frac{d\mathbf{x}}{dt} = \frac{d\mathbf{x}}{d\tau} \cdot \frac{d\tau}{dt} = \frac{2}{\Delta t^{(p)}} \frac{d\mathbf{x}}{d\tau}
$$

따라서 동역학 방정식 $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$는 정규화 좌표에서

$$
\frac{2}{\Delta t^{(p)}} \frac{d\mathbf{x}}{d\tau} = \mathbf{f}(\mathbf{x}, \mathbf{u})
$$

로 변환된다.

적분에 대한 변환은 다음과 같다.

$$
\int_{t_0^{(p)}}^{t_f^{(p)}} g(t) \, dt = \frac{\Delta t^{(p)}}{2} \int_{-1}^{1} g(\tau) \, d\tau
$$

### 2.4 Phase별 LGL Collocation 이산화

$p$번째 phase에서 $N_p$개의 LGL 노드 $\{\tau_j^{(p)}\}_{j=0}^{N_p-1}$를 배치한다. 여기서 LGL 다항식의 차수는 $N_p - 1$이며, 노드 수는 $N_p$개이다. 각 노드에서의 상태 변수를 $\mathbf{X}_j^{(p)} \approx \mathbf{x}^{(p)}(\tau_j^{(p)})$, 제어 변수를 $\mathbf{U}_j^{(p)} \approx \mathbf{u}^{(p)}(\tau_j^{(p)})$로 정의한다.

LGL 미분 행렬 $\mathbf{D}^{(p)} \in \mathbb{R}^{N_p \times N_p}$를 사용하면, 정규화 좌표에서의 상태 미분 근사는

$$
\frac{d\mathbf{x}}{d\tau}\bigg|_{\tau = \tau_i^{(p)}} \approx \sum_{j=0}^{N_p-1} D_{ij}^{(p)} \, \mathbf{X}_j^{(p)}
$$

이다. 이를 동역학 제약에 대입하면, $p$번째 phase의 collocation 조건은 각 노드 $i = 0, 1, \ldots, N_p - 1$에 대해 다음과 같다.

$$
\frac{2}{\Delta t^{(p)}} \sum_{j=0}^{N_p-1} D_{ij}^{(p)} \, \mathbf{X}_j^{(p)} = \mathbf{f}(\mathbf{X}_i^{(p)}, \mathbf{U}_i^{(p)})
$$

이 조건은 각 phase에서 $6 \times N_p$개의 등호 제약을 생성한다. 행렬 형태로 표기하면

$$
\frac{2}{\Delta t^{(p)}} \, \mathbf{D}^{(p)} \, \boldsymbol{X}^{(p)} = \mathbf{F}^{(p)}(\boldsymbol{X}^{(p)}, \boldsymbol{U}^{(p)})
$$

여기서 $\boldsymbol{X}^{(p)} = [\mathbf{X}_0^{(p)}, \ldots, \mathbf{X}_{N_p-1}^{(p)}] \in \mathbb{R}^{6 \times N_p}$이고, $\mathbf{F}^{(p)}$는 각 열이 해당 노드에서의 운동방정식 값인 행렬이다.

### 2.5 비용 함수의 LGL 구적

비용 함수의 각 phase 기여분은 LGL 가중치 $\{w_j^{(p)}\}_{j=0}^{N_p-1}$을 이용한 구적법으로 근사한다.

$$
\int_{t_0^{(p)}}^{t_f^{(p)}} \|\mathbf{u}^{(p)}(t)\|^2 \, dt \approx \frac{\Delta t^{(p)}}{2} \sum_{j=0}^{N_p-1} w_j^{(p)} \, \|\mathbf{U}_j^{(p)}\|^2
$$

전체 비용 함수는 모든 phase의 합이다.

$$
J \approx \sum_{p=1}^{P} \frac{\Delta t^{(p)}}{2} \sum_{j=0}^{N_p-1} w_j^{(p)} \, \|\mathbf{U}_j^{(p)}\|^2
$$

$N_p - 1 = 14$차 LGL 구적법의 경우, $2 \times 14 - 1 = 27$차 이하의 다항식 피적분 함수에 대해 정확한 적분을 제공한다. 제어 변수 $\mathbf{U}_j^{(p)}$는 최대 14차 다항식으로 보간되므로, $\|\mathbf{U}^{(p)}\|^2$는 최대 28차 다항식이 된다. 따라서 비용 적분에는 약간의 구적 오차가 존재하나, spectral convergence에 의해 노드 수 증가에 따라 급속히 감소한다.

### 2.6 Linkage 제약

인접 phase의 경계에서 상태 연속성을 보장하는 linkage 제약은 다음과 같다.

$$
\mathbf{X}_{N_p-1}^{(p)} = \mathbf{X}_0^{(p+1)}, \quad p = 1, \ldots, P-1
$$

이 제약은 $p$번째 phase의 마지막 LGL 노드(정규화 좌표 $\tau = 1$에 대응)와 $(p+1)$번째 phase의 첫 번째 LGL 노드($\tau = -1$에 대응)에서의 상태 변수가 동일함을 강제한다. LGL 노드가 구간의 양 끝점을 포함하기 때문에 이러한 점 대 점(point-to-point) linkage가 자연스럽게 가능하다.

각 linkage 제약은 6개(위치 3, 속도 3)의 등호 조건을 생성하므로, 총 $6(P-1)$개의 linkage 제약이 존재한다.

### 2.7 NLP 문제 요약

Phase별 LGL 이산화와 linkage 제약을 결합하면 다음의 NLP 문제가 된다.

**결정 변수:**

$$
\mathbf{z} = \bigl[\boldsymbol{X}^{(1)}, \boldsymbol{U}^{(1)}, \ldots, \boldsymbol{X}^{(P)}, \boldsymbol{U}^{(P)}, \nu_0, \nu_f\bigr]
$$

총 결정 변수 수는 $\sum_{p=1}^{P} 9 N_p + 2$이다 (각 phase에서 상태 6개 + 제어 3개가 $N_p$개 노드에 존재, 여기에 $\nu_0$, $\nu_f$ 2개 추가).

**목적 함수:**

$$
\min_{\mathbf{z}} \; J = \sum_{p=1}^{P} \frac{\Delta t^{(p)}}{2} \sum_{j=0}^{N_p-1} w_j^{(p)} \, \|\mathbf{U}_j^{(p)}\|^2
$$

**등호 제약:**

- Phase별 collocation: $\sum_{p=1}^{P} 6 N_p$ 개
- Linkage: $6(P-1)$ 개
- 경계 조건: $6 + 6 = 12$ 개

**부등호 제약:**

- 추력 상한: $\sum_{p=1}^{P} N_p$ 개
- 최소 고도: $\sum_{p=1}^{P} N_p$ 개

예를 들어 bimodal 프로파일(3 phases, $N_p = 15, 8, 15$)의 경우, 총 결정 변수 수는 $9 \times 38 + 2 = 344$, 등호 제약 수는 $6 \times 38 + 6 \times 2 + 12 = 252$, 부등호 제약 수는 $2 \times 38 = 76$이다.

### 2.8 Pass 1 해의 Warm Start 보간

Pass 2 NLP의 수렴을 위해서는 양호한 초기 추정(initial guess)이 필수적이다. Pass 1에서 Hermite-Simpson 방법으로 구한 해를 Pass 2의 Multi-Phase LGL 노드에 보간하여 초기값으로 사용한다.

Pass 1의 해는 균일 구간(uniform segments) 위에서 정의된 시계열 데이터 $\{t_k, \mathbf{x}_k, \mathbf{u}_k\}_{k=1}^{N_1}$이다. 이를 Pass 2의 각 phase $p$에서의 LGL 노드 시각 $\{t_j^{(p)}\}$에 보간하기 위해 cubic spline interpolation을 사용한다.

**보간 절차:**

1. Pass 1 상태/제어 시계열에 대해 cubic spline 보간기를 구성한다.

$$
\tilde{\mathbf{x}}(t) = \text{CubicSpline}(\{t_k, \mathbf{x}_k\}), \quad \tilde{\mathbf{u}}(t) = \text{CubicSpline}(\{t_k, \mathbf{u}_k\})
$$

2. 각 phase $p$에서 LGL 노드의 물리 시각을 계산한다.

$$
t_j^{(p)} = t_0^{(p)} + \frac{\Delta t^{(p)}}{2}(\tau_j^{(p)} + 1), \quad j = 0, \ldots, N_p - 1
$$

3. 해당 시각에서 보간값을 평가한다.

$$
\mathbf{X}_j^{(p)} \leftarrow \tilde{\mathbf{x}}(t_j^{(p)}), \quad \mathbf{U}_j^{(p)} \leftarrow \tilde{\mathbf{u}}(t_j^{(p)})
$$

Cubic spline은 $C^2$ 연속성을 갖는 구간별 3차 다항식으로, 상태 변수(위치, 속도)의 보간에 적합하다. Pass 1 해가 물리적으로 타당한 궤적이라면, 보간된 초기값은 NLP의 실행 가능(feasible) 영역 근방에 위치하게 되어 IPOPT의 warm start와 결합하여 빠른 수렴을 기대할 수 있다.

IPOPT warm start 옵션(`warm_start_init_point = yes`, `warm_start_bound_push = 1e-6`)은 초기값이 실행 가능 영역에 가까울 때 barrier 파라미터의 초기값을 조정하여 수렴을 가속화한다. 이 설정은 Pass 2 전용 IPOPT 옵션에 포함되어 있다.

---

## 3. 구현 매핑

### 3.1 모듈 구조

Multi-phase collocation 구현은 세 개의 모듈로 구성된다.

| 모듈 | 파일 | 역할 |
|------|------|------|
| `MultiPhaseLGLCollocation` | `collocation/multiphase_lgl.py` | NLP 구성 및 풀기 |
| `interpolate_pass1_to_pass2` | `collocation/interpolation.py` | Pass 1 → Pass 2 보간 |
| `determine_phase_structure` | `classification/classifier.py` | Phase 분할 알고리즘 |

LGL 노드, 가중치, 미분 행렬의 계산은 `collocation/lgl_nodes.py`에서 제공하는 함수를 재사용한다.

### 3.2 NLP 구성: `MultiPhaseLGLCollocation.solve` (46--200행)

아래 표는 수학적 정식화와 구현 코드 사이의 대응 관계이다.

| 수학 표현 | 코드 위치 | 구현 방식 |
|-----------|-----------|-----------|
| Phase별 상태 변수 $\boldsymbol{X}^{(p)} \in \mathbb{R}^{6 \times N_p}$ | 64--70행 | `opti.variable(6, N_k)`, 리스트 `X_vars` |
| Phase별 제어 변수 $\boldsymbol{U}^{(p)} \in \mathbb{R}^{3 \times N_p}$ | 64--70행 | `opti.variable(3, N_k)`, 리스트 `U_vars` |
| True anomaly $\nu_0, \nu_f$ | 72--73행 | `opti.variable()` (스칼라) |
| LGL 노드 $\tau_j$, 가중치 $w_j$, 미분행렬 $D_{ij}$ | 76--87행 | `compute_lgl_nodes`, `compute_lgl_weights`, `compute_differentiation_matrix` 호출; `lgl_data` 딕셔너리에 저장 |
| 비용 $J = \sum_p (\Delta t^{(p)}/2) \sum_j w_j \|\mathbf{U}_j^{(p)}\|^2$ | 90--99행 | 이중 루프; `ca.dot(u_j, u_j)`로 내적 계산 |
| Collocation: $(2/\Delta t^{(p)}) \mathbf{D}^{(p)} \boldsymbol{X}^{(p)} = \mathbf{F}^{(p)}$ | 102--117행 | 삼중 루프 (phase, 노드 $i$, 노드 $j$); `eom_func` 호출 |
| Linkage: $\mathbf{X}^{(p)}_{N_p-1} = \mathbf{X}^{(p+1)}_0$ | 120--121행 | `X_vars[p][:, -1] == X_vars[p+1][:, 0]` |
| 경계 조건: 궤도요소 → ECI 변환 | 124--136행 | `oe_to_rv_casadi` 호출; $e = 0$, $\omega = 0$ 고정 |
| 추력 상한: $\|\mathbf{U}_j^{(p)}\|^2 \leq u_\max^2$ | 141--146행 | `ca.dot` ≤ `u_max_sq` |
| 최소 고도: $\|\mathbf{r}_j^{(p)}\| \geq R_E + h_\min$ | 147--149행 | `ca.dot` (위치 3성분) ≥ `r_min_sq` |
| IPOPT 설정 | 162--165행 | `IPOPT_OPTIONS_PASS2` 기본값 + 사용자 옵션 병합 |

**CasADi Opti Stack.** NLP는 CasADi의 `Opti` 인터페이스로 정식화한다. `Opti`는 심볼릭 변수 선언, 목적 함수, 제약 조건을 선언적으로 기술하고, 내부적으로 NLP를 자동으로 구성하여 IPOPT에 전달한다. 자동 미분(automatic differentiation)에 의해 Jacobian과 Hessian이 정확하게 계산된다.

### 3.3 결과 조립: `_assemble_result` (202--224행)

Phase별로 분리된 해를 단일 시계열로 재조립한다.

| 단계 | 설명 |
|------|------|
| 정규화 → 물리 시각 변환 | $t_j^{(p)} = t_0^{(p)} + (\Delta t^{(p)}/2)(\tau_j + 1)$ |
| Linkage 점 중복 제거 | $p > 0$인 phase에서 첫 번째 노드 제거 (이전 phase의 마지막 노드와 동일) |
| 배열 연결 | `np.concatenate` (시간), `np.hstack` (상태, 제어) |

### 3.4 보간 함수: `interpolate_pass1_to_pass2` (8--46행)

| 수학 표현 | 코드 위치 | 구현 방식 |
|-----------|-----------|-----------|
| Cubic spline 보간기 구성 | 28--31행 | `scipy.interpolate.interp1d(kind='cubic', fill_value='extrapolate')` |
| Phase별 LGL 노드 시각 계산 | 34--40행 | `compute_lgl_nodes(N-1)` → affine 변환 |
| 보간값 평가 | 43--44행 | `x_interp(t_phys)`, `u_interp(t_phys)` |

`fill_value='extrapolate'`는 Pass 1 시간 범위를 약간 벗어나는 LGL 노드에 대해 외삽을 허용한다. 이는 시간 영역 경계 근방의 수치적 안정성을 위한 것이다.

### 3.5 Phase 분할: `determine_phase_structure` (29--138행)

| 수학 표현 | 코드 위치 | 구현 방식 |
|-----------|-----------|-----------|
| Unimodal: 단일 phase | 67--76행 | $n_\text{peaks} = 1$이면 `[0, T]`, `N = 15` |
| Bimodal/Multimodal: 교대 배치 | 81--133행 | 피크 순회; FWHM 기반 경계; peak/coast 교대 |
| 짧은 phase 병합 | 136행, 141--174행 | `_merge_short_phases`: $\Delta t < 0.05T$ 반복 검사 |

### 3.6 설정 파라미터 요약

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `LGL_NODES_PEAK` | 15 | Peak phase 노드 수 (14차 다항식) |
| `LGL_NODES_COAST` | 8 | Coast phase 노드 수 (7차 다항식) |
| `MIN_PHASE_FRACTION` | 0.05 | 최소 phase 길이 / $T$ |
| `ipopt.tol` | $10^{-6}$ | NLP 허용 오차 |
| `ipopt.constr_viol_tol` | $10^{-6}$ | 제약 위반 허용 오차 |
| `ipopt.max_iter` | 1000 | 최대 반복 횟수 |
| `warm_start_init_point` | yes | Warm start 활성화 |
| `warm_start_bound_push` | $10^{-6}$ | Barrier 파라미터 초기 보정 |

---

## 4. 수치 검증

Multi-phase collocation의 정확성과 신뢰성은 다음 기준으로 검증한다.

### 4.1 Linkage 연속성

Phase 경계에서의 상태 불연속을 정량적으로 평가한다.

$$
\epsilon_\text{link}^{(p)} = \|\mathbf{X}_{N_p-1}^{(p)} - \mathbf{X}_0^{(p+1)}\|_\infty, \quad p = 1, \ldots, P-1
$$

NLP 솔버가 수렴한 경우, linkage 오차는 제약 위반 허용 오차(`constr_viol_tol = 1e-6`) 이하여야 한다. 실제로 IPOPT는 등호 제약을 내부적으로 $|g(\mathbf{z})| \leq \epsilon$ 형태로 처리하므로, 수렴된 해에서의 linkage 오차는 $10^{-6}$ 수준 이하가 된다.

### 4.2 비용 개선

Pass 2는 Pass 1보다 높은 정확도의 이산화를 사용하므로, 동일한 문제에 대해 Pass 2의 비용이 Pass 1 이하여야 한다.

$$
J_{\text{Pass 2}} \leq J_{\text{Pass 1}}
$$

이 관계가 성립하는 근거는 다음과 같다. Pass 2는 (1) 피크 구간에서 더 높은 차수의 다항식 근사를 사용하여 실행 가능 영역(feasible set)이 더 정밀하게 표현되고, (2) Pass 1 해를 warm start로 사용하므로 최소한 Pass 1 해에 준하는 국소 최적해를 탐색할 수 있다. 물론 NLP의 비볼록성으로 인해 다른 국소 최적해로 수렴할 가능성도 있으나, warm start에 의해 Pass 1 해의 basin of attraction 내에서 개선된 해를 찾는 것이 일반적이다.

### 4.3 Collocation 잔차

동역학 구속조건의 잔차를 평가한다.

$$
\mathbf{r}_i^{(p)} = \frac{2}{\Delta t^{(p)}} \sum_{j=0}^{N_p-1} D_{ij}^{(p)} \mathbf{X}_j^{(p)} - \mathbf{f}(\mathbf{X}_i^{(p)}, \mathbf{U}_i^{(p)})
$$

수렴된 해에서 $\|\mathbf{r}_i^{(p)}\|$는 `constr_viol_tol` = $10^{-6}$ 이하여야 한다. 이 잔차는 NLP 제약 조건 그 자체이므로, IPOPT의 수렴 판정 기준과 직접적으로 연결된다.

### 4.4 Phase 분할 일관성

Phase 분할 알고리즘의 일관성을 다음과 같이 검증한다.

- **전체 구간 커버**: $t_0^{(1)} = 0$, $t_f^{(P)} = T$이며, 모든 phase가 빈 구간 없이 연속적으로 이어진다.
- **Phase 순서**: $t_0^{(p)} < t_f^{(p)}$이며, $t_f^{(p)} = t_0^{(p+1)}$이다.
- **최소 길이**: 병합 후 모든 phase의 길이가 $\gamma T = 0.05T$ 이상이다.
- **교대 구조**: Bimodal 이상에서 peak phase와 coast phase가 교대로 배치된다.

### 4.5 수렴 실패 처리

NLP가 수렴하지 않는 경우(`RuntimeError` 발생), 다음과 같이 처리한다.

- `converged = False`, `cost = inf`로 설정
- `opti.debug.value`를 통해 마지막 iterate의 상태/제어 값을 추출 (디버깅 및 분석 목적)
- 수렴된 해와 동일한 `TrajectoryResult` 구조체로 반환하여 후속 파이프라인과의 호환성 유지

---

## 5. 참고문헌

1. Patterson, M. A. and Rao, A. V. (2014). "GPOPS-II: A MATLAB software for solving multiple-phase optimal control problems using hp-adaptive Gaussian quadrature collocation methods and sparse nonlinear programming," *ACM Transactions on Mathematical Software*, 41(1), 1--37.
2. Betts, J. T. (1998). "Survey of numerical methods for trajectory optimization," *Journal of Guidance, Control, and Dynamics*, 21(2), 193--207.
3. Garg, D., Patterson, M. A., Hager, W. W., Rao, A. V., Benson, D. A., and Huntington, G. T. (2010). "A unified framework for the numerical solution of optimal control problems using pseudospectral methods," *Automatica*, 46(11), 1843--1851.
4. Darby, C. L., Hager, W. W., and Rao, A. V. (2011). "An hp-adaptive pseudospectral method for solving optimal control problems," *Optimal Control Applications and Methods*, 32(4), 476--502.
5. Benson, D. A., Huntington, G. T., Thorvaldsen, T. P., and Rao, A. V. (2006). "Direct trajectory optimization and costate estimation via an orthogonal collocation method," *Journal of Guidance, Control, and Dynamics*, 29(6), 1435--1440.
6. Wächter, A. and Biegler, L. T. (2006). "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming," *Mathematical Programming*, 106(1), 25--57.
