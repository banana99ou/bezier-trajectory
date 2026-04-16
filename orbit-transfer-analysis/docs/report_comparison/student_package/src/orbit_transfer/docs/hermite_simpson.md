# B1. Hermite-Simpson Collocation 이론

> 대응 코드: `src/orbit_transfer/collocation/hermite_simpson.py`

## 1. 목적

연속 추력 궤도전이 최적 제어 문제는 무한 차원 함수 공간에서 정의된다. 상태 $\mathbf{x}(t)$와 제어 $\mathbf{u}(t)$는 시간의 연속 함수이며, 동역학은 상미분방정식(ODE) 구속조건으로 기술된다. 이러한 문제를 수치적으로 풀기 위해서는 유한 차원의 비선형 프로그래밍(NLP) 문제로 변환하는 이산화(transcription) 과정이 필요하다.

직접 배치법(direct collocation)은 시간 구간을 유한개의 부분 구간으로 나누고, 각 부분 구간에서 상태를 다항식으로 근사한 뒤, 선택된 배치점(collocation point)에서 ODE를 대수 제약조건으로 부과하는 방법이다. Hermite-Simpson collocation은 3차 Hermite 보간 다항식과 Simpson 적분 규칙을 결합한 직접 배치법으로, 4차 정확도($O(h^4)$ 전역 오차)를 제공하면서도 구현이 비교적 간단하여 궤적 최적화에 널리 사용된다.

본 모듈(`hermite_simpson.py`)은 LEO-to-LEO 연속 추력 궤도전이 문제의 1차 풀이(Pass 1)를 담당한다. $M = 30$개의 균일 구간에 $2M + 1 = 61$개의 배치점을 배치하고, 6차원 상태와 3차원 제어, 2개의 자유 true anomaly를 포함하여 총 551개의 결정변수를 갖는 NLP를 구성한다. 비용함수는 제어 노력의 적분으로 정의하며, Simpson 구적법으로 근사한다.

---

## 2. 수학적 배경

### 2.1 최적 제어 문제의 정식화

시간 구간 $[0, T]$에서 다음 최적 제어 문제를 고려한다.

$$
\min_{\mathbf{u}(t)} \quad J = \int_0^T \|\mathbf{u}(t)\|^2 \, dt
$$

subject to:

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)), \quad t \in [0, T]
$$

$$
\mathbf{x}(0) = \mathbf{x}_0, \quad \mathbf{x}(T) = \mathbf{x}_f
$$

$$
\|\mathbf{u}(t)\| \leq u_{\max}, \quad \|\mathbf{r}(t)\| \geq R_E + h_{\min}
$$

여기서 $\mathbf{x} = [\mathbf{r}^\top, \mathbf{v}^\top]^\top \in \mathbb{R}^6$은 ECI 좌표계의 위치-속도 상태벡터, $\mathbf{u} \in \mathbb{R}^3$은 추력 가속도 벡터, $\mathbf{f}$는 J2 섭동을 포함한 운동방정식이다.

### 2.2 시간 구간의 이산화

시간 구간 $[0, T]$를 $M$개의 균일 부분 구간으로 분할한다. 구간 폭은 $h = T/M$이며, 주 노드(mesh point)는 다음과 같다.

$$
t_k = k h, \quad k = 0, 1, \ldots, M
$$

각 부분 구간 $[t_k, t_{k+1}]$의 중간점을 추가로 도입한다.

$$
t_{k+\frac{1}{2}} = t_k + \frac{h}{2}, \quad k = 0, 1, \ldots, M - 1
$$

따라서 총 배치점의 수는 $M + 1$(주 노드) $+ M$(중간점) $= 2M + 1$이다. $M = 30$일 때 $2 \times 30 + 1 = 61$개의 배치점이 생성된다.

### 2.3 3차 Hermite 보간 다항식

각 부분 구간 $[t_k, t_{k+1}]$에서 상태를 3차 Hermite 다항식으로 근사한다. 3차 Hermite 보간 다항식은 구간 양 끝점의 함수값과 미분값, 총 4개의 조건으로 결정되는 유일한 3차 다항식이다.

**정의.** 구간 $[t_k, t_{k+1}]$에서 양 끝점의 값 $\mathbf{x}_k$, $\mathbf{x}_{k+1}$과 미분값 $\mathbf{f}_k = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)$, $\mathbf{f}_{k+1} = \mathbf{f}(\mathbf{x}_{k+1}, \mathbf{u}_{k+1})$이 주어질 때, 3차 Hermite 보간 다항식 $\mathbf{p}(t)$는 다음 4가지 조건을 만족한다.

$$
\mathbf{p}(t_k) = \mathbf{x}_k, \quad \mathbf{p}(t_{k+1}) = \mathbf{x}_{k+1}
$$

$$
\dot{\mathbf{p}}(t_k) = \mathbf{f}_k, \quad \dot{\mathbf{p}}(t_{k+1}) = \mathbf{f}_{k+1}
$$

**유도.** 정규화된 변수 $\tau = (t - t_k)/h \in [0, 1]$을 도입하면, $\mathbf{p}(\tau) = \mathbf{a}_0 + \mathbf{a}_1 \tau + \mathbf{a}_2 \tau^2 + \mathbf{a}_3 \tau^3$으로 놓을 수 있다. 4개의 보간 조건을 풀면 Hermite 기저 함수를 이용한 다음 표현을 얻는다.

$$
\mathbf{p}(\tau) = H_{00}(\tau)\, \mathbf{x}_k + H_{10}(\tau)\, h \mathbf{f}_k + H_{01}(\tau)\, \mathbf{x}_{k+1} + H_{11}(\tau)\, h \mathbf{f}_{k+1}
$$

여기서 Hermite 기저 함수는 다음과 같다.

$$
H_{00}(\tau) = 2\tau^3 - 3\tau^2 + 1, \quad H_{10}(\tau) = \tau^3 - 2\tau^2 + \tau
$$

$$
H_{01}(\tau) = -2\tau^3 + 3\tau^2, \quad H_{11}(\tau) = \tau^3 - \tau^2
$$

### 2.4 Hermite 중간점 조건의 유도

구간의 중간점 $\tau = 1/2$ (즉, $t = t_{k+1/2}$)에서 Hermite 보간 다항식의 값을 계산한다. 각 기저 함수를 $\tau = 1/2$에서 평가하면

$$
H_{00}\!\left(\tfrac{1}{2}\right) = 2 \cdot \tfrac{1}{8} - 3 \cdot \tfrac{1}{4} + 1 = \tfrac{1}{2}
$$

$$
H_{10}\!\left(\tfrac{1}{2}\right) = \tfrac{1}{8} - 2 \cdot \tfrac{1}{4} + \tfrac{1}{2} = \tfrac{1}{8}
$$

$$
H_{01}\!\left(\tfrac{1}{2}\right) = -2 \cdot \tfrac{1}{8} + 3 \cdot \tfrac{1}{4} = \tfrac{1}{2}
$$

$$
H_{11}\!\left(\tfrac{1}{2}\right) = \tfrac{1}{8} - \tfrac{1}{4} = -\tfrac{1}{8}
$$

이를 대입하면 중간점에서의 상태는 다음과 같다.

$$
\mathbf{x}_{k+\frac{1}{2}} = \mathbf{p}\!\left(\tfrac{1}{2}\right) = \frac{1}{2}(\mathbf{x}_k + \mathbf{x}_{k+1}) + \frac{h}{8}(\mathbf{f}_k - \mathbf{f}_{k+1})
$$

이것이 **Hermite 중간점 조건**이다. 이 조건은 중간점의 상태값이 양 끝점의 정보로부터 3차 다항식 보간에 의해 결정됨을 의미한다. 제1항 $\frac{1}{2}(\mathbf{x}_k + \mathbf{x}_{k+1})$은 선형 보간 성분이고, 제2항 $\frac{h}{8}(\mathbf{f}_k - \mathbf{f}_{k+1})$은 미분값 차이에 의한 3차 보정 성분이다.

### 2.5 Simpson 적분 규칙

**3점 Simpson 규칙의 유도.** 구간 $[a, b]$에서 함수 $g(t)$의 적분을 양 끝점과 중간점의 함수값을 이용하여 근사한다. $g(t)$를 이 세 점을 지나는 2차 다항식 $q(t)$로 보간하면

$$
q(t) = g(a) L_0(t) + g\!\left(\tfrac{a+b}{2}\right) L_1(t) + g(b) L_2(t)
$$

여기서 $L_0, L_1, L_2$는 세 점에 대한 Lagrange 기저 함수이다. 이 2차 다항식을 구간 $[a, b]$에서 적분하면 Simpson 규칙을 얻는다.

$$
\int_a^b g(t)\, dt \approx \frac{b - a}{6} \left[ g(a) + 4\, g\!\left(\tfrac{a+b}{2}\right) + g(b) \right]
$$

**국소 오차.** Simpson 규칙은 3차 이하의 다항식에 대해 정확하다. 일반적인 충분히 매끄러운 함수에 대한 국소 절단 오차(local truncation error)는 다음과 같다.

$$
E_{\text{local}} = -\frac{h^5}{2880}\, g^{(4)}(\xi), \quad \xi \in (a, b)
$$

여기서 $h = b - a$이다. 국소 오차가 $O(h^5)$이므로, $M$개의 부분 구간에 걸친 전역 오차는 다음과 같다.

$$
E_{\text{global}} = M \cdot O(h^5) = \frac{T}{h} \cdot O(h^5) = O(h^4)
$$

### 2.6 Simpson 연속성 조건의 유도

ODE $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$를 구간 $[t_k, t_{k+1}]$에서 적분하면

$$
\mathbf{x}_{k+1} - \mathbf{x}_k = \int_{t_k}^{t_{k+1}} \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t))\, dt
$$

우변의 적분에 Simpson 규칙을 적용하면

$$
\int_{t_k}^{t_{k+1}} \mathbf{f}\, dt \approx \frac{h}{6} \left( \mathbf{f}_k + 4\, \mathbf{f}_{k+\frac{1}{2}} + \mathbf{f}_{k+1} \right)
$$

여기서 $\mathbf{f}_{k+1/2} = \mathbf{f}(\mathbf{x}_{k+1/2}, \mathbf{u}_{k+1/2})$이다. 따라서 **Simpson 연속성 조건**은 다음과 같다.

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{h}{6}\left(\mathbf{f}_k + 4\, \mathbf{f}_{k+\frac{1}{2}} + \mathbf{f}_{k+1}\right)
$$

이 조건은 인접한 두 주 노드 사이의 상태 천이가 Simpson 적분에 의해 일관되어야 함을 의미한다. Hermite 중간점 조건(2.4절)과 결합하면, 각 구간에서 $6 + 6 = 12$개의 스칼라 등식 제약(6차원 상태에 대해 각 조건이 6개씩)이 부과된다.

### 2.7 Hermite-Simpson collocation의 구조

두 조건을 종합하면, 각 부분 구간 $[t_k, t_{k+1}]$에서 다음 두 세트의 대수 제약이 부과된다.

**(C1) Hermite 중간점 조건:**

$$
\mathbf{x}_{k+\frac{1}{2}} - \frac{1}{2}(\mathbf{x}_k + \mathbf{x}_{k+1}) - \frac{h}{8}(\mathbf{f}_k - \mathbf{f}_{k+1}) = \mathbf{0}
$$

**(C2) Simpson 연속성 조건:**

$$
\mathbf{x}_{k+1} - \mathbf{x}_k - \frac{h}{6}\left(\mathbf{f}_k + 4\, \mathbf{f}_{k+\frac{1}{2}} + \mathbf{f}_{k+1}\right) = \mathbf{0}
$$

이 두 조건의 역할은 다음과 같이 해석된다.

- **(C1)**은 중간점의 상태값을 3차 Hermite 다항식과 일관되도록 결정한다. 이 조건이 없으면 $\mathbf{x}_{k+1/2}$는 자유 변수로 남아, 중간점의 상태가 양 끝점의 동역학과 무관하게 된다.
- **(C2)**는 ODE의 적분형을 Simpson 규칙으로 근사한 것이다. 이 조건은 양 끝점과 중간점의 동역학을 모두 포함하므로, 중간점에서의 ODE 만족이 암묵적으로 보장된다.

### 2.8 오차 분석

Hermite-Simpson collocation의 전역 오차 수렴 차수는 $O(h^4)$이다. 이를 단계적으로 확인한다.

**Hermite 보간 오차.** 3차 Hermite 보간 다항식의 보간 오차는 다음과 같다.

$$
\mathbf{x}(t) - \mathbf{p}(t) = O(h^4)
$$

이는 3차 다항식이 4개의 조건(양 끝점의 값과 미분)으로 결정되므로, 오차에 $h^4$ 인자가 나타나는 것이다.

**Simpson 적분 오차.** 앞서 2.5절에서 보인 것처럼, Simpson 규칙의 국소 절단 오차는 $O(h^5)$이다.

**전역 오차.** $M$개의 구간에 걸쳐 국소 오차가 누적되므로

$$
E_{\text{global}} = M \cdot O(h^5) = \frac{T}{h} \cdot O(h^5) = O(h^4)
$$

따라서 구간 수 $M$을 2배로 늘리면(즉, $h$를 반으로 줄이면) 전역 오차는 약 $1/16$로 감소한다.

**다른 배치법과의 비교.** 참고로, 사다리꼴 배치법(trapezoidal collocation)은 $O(h^2)$ 전역 오차를 가지며, Gauss-Lobatto 4점 배치법은 $O(h^6)$이다. Hermite-Simpson은 정확도와 구현 복잡도 사이에서 좋은 균형을 이룬다.

### 2.9 비용함수의 Simpson 적분 근사

비용함수의 피적분 함수를 $L(\mathbf{u}) = \|\mathbf{u}\|^2 = \mathbf{u}^\top \mathbf{u}$로 정의하면, 전체 비용함수는

$$
J = \int_0^T \|\mathbf{u}(t)\|^2\, dt = \sum_{k=0}^{M-1} \int_{t_k}^{t_{k+1}} \|\mathbf{u}(t)\|^2\, dt
$$

각 부분 구간에 Simpson 규칙을 적용하면

$$
\int_{t_k}^{t_{k+1}} \|\mathbf{u}(t)\|^2\, dt \approx \frac{h}{6}\left(\|\mathbf{u}_k\|^2 + 4\|\mathbf{u}_{k+\frac{1}{2}}\|^2 + \|\mathbf{u}_{k+1}\|^2\right)
$$

따라서 이산화된 비용함수는

$$
J \approx \sum_{k=0}^{M-1} \frac{h}{6}\left(\|\mathbf{u}_k\|^2 + 4\|\mathbf{u}_{k+\frac{1}{2}}\|^2 + \|\mathbf{u}_{k+1}\|^2\right)
$$

이다. 이 비용함수는 NLP의 목적함수가 된다. 제어 노력의 2-노름 제곱을 적분하므로, 이른바 에너지 최적(energy-optimal) 정식화에 해당한다.

---

## 3. 구현 매핑

### 3.1 NLP 결정변수 구조

Hermite-Simpson collocation에서 결정변수는 다음과 같이 구성된다.

| 변수 | 기호 | 크기 | 설명 |
|------|------|------|------|
| 상태 | $\mathbf{X} \in \mathbb{R}^{6 \times 61}$ | 366 | 61개 배치점의 6차원 상태벡터 |
| 제어 | $\mathbf{U} \in \mathbb{R}^{3 \times 61}$ | 183 | 61개 배치점의 3차원 추력 가속도 |
| 출발 true anomaly | $\nu_0$ | 1 | 출발 궤도상 위치 (자유변수) |
| 도착 true anomaly | $\nu_f$ | 1 | 도착 궤도상 위치 (자유변수) |
| **합계** | | **551** | |

배치점의 인덱싱은 다음 규칙을 따른다. $k$번째 구간($k = 0, 1, \ldots, M-1$)에 대해

- 좌측 주 노드: 인덱스 $2k$
- 중간점: 인덱스 $2k + 1$
- 우측 주 노드: 인덱스 $2k + 2$

이 인덱싱에 의해, 주 노드는 짝수 인덱스(0, 2, 4, ..., 60)에, 중간점은 홀수 인덱스(1, 3, 5, ..., 59)에 배치된다.

### 3.2 경계조건: 궤도요소에서 ECI 상태벡터로의 변환

출발 궤도와 도착 궤도는 원궤도($e = 0$)로 정의되며, 궤도요소는 다음과 같다.

- 출발: $\boldsymbol{\alpha}_0 = (a_0,\; 0,\; i_0,\; 0,\; 0,\; \nu_0)$
- 도착: $\boldsymbol{\alpha}_f = (a_f,\; 0,\; i_f,\; 0,\; 0,\; \nu_f)$

여기서 $a$는 장반경, $i$는 경사각이다. 원궤도이므로 이심률 $e = 0$이고, 근지점 인수 $\omega$는 정의되지 않으므로 0으로 둔다. 이때 $\nu$는 true anomaly가 아닌 argument of latitude의 역할을 한다.

궤도요소에서 ECI 상태벡터로의 변환은 다음 과정으로 수행된다.

1. 반통경(semi-latus rectum) 계산: $p = a(1 - e^2) = a$ (원궤도에서)
2. 궤도면 좌표계(perifocal frame)에서의 위치와 속도:

$$
\mathbf{r}_{PQW} = a \begin{bmatrix} \cos\nu \\ \sin\nu \\ 0 \end{bmatrix}, \quad \mathbf{v}_{PQW} = \sqrt{\frac{\mu}{a}} \begin{bmatrix} -\sin\nu \\ \cos\nu \\ 0 \end{bmatrix}
$$

3. 회전 행렬 $\mathbf{R}(\Omega, \omega, i)$을 적용하여 ECI 좌표계로 변환:

$$
\mathbf{r} = \mathbf{R}\, \mathbf{r}_{PQW}, \quad \mathbf{v} = \mathbf{R}\, \mathbf{v}_{PQW}
$$

$\nu_0$와 $\nu_f$가 NLP의 자유변수이므로, 경계조건은 $\nu_0$, $\nu_f$에 대한 비선형 함수가 된다. 이는 솔버가 최적의 출발-도착 위치를 자동으로 결정할 수 있게 한다. CasADi의 심볼릭 미분(automatic differentiation)이 이 비선형 경계조건의 야코비안을 정확하게 계산한다.

### 3.3 부등식 제약조건

**(1) 추력 상한 제약.** 각 배치점 $j = 0, 1, \ldots, 2M$에서

$$
\mathbf{u}_j^\top \mathbf{u}_j \leq u_{\max}^2
$$

2차형(quadratic) 부등식으로 정식화되어 있다. 이 형태는 노름을 직접 사용하는 $\|\mathbf{u}_j\| \leq u_{\max}$와 수학적으로 동치이지만, 제곱근 연산을 피하여 미분 가능성을 보장하고, 야코비안 계산이 보다 안정적이다.

**(2) 최소 고도 제약.** 각 배치점 $j = 0, 1, \ldots, 2M$에서

$$
\mathbf{r}_j^\top \mathbf{r}_j \geq (R_E + h_{\min})^2
$$

여기서 $\mathbf{r}_j = \mathbf{X}_{1:3, j}$는 위치벡터이다. 이 제약은 궤적이 최소 고도 이상을 유지하도록 보장한다.

### 3.4 수학적 구성 요소와 코드의 대응

아래 표는 2절의 수학적 구성 요소와 `hermite_simpson.py`의 구현 사이의 대응 관계를 정리한 것이다.

| 수학 표현 | 코드 위치 | 구현 방식 |
|-----------|-----------|-----------|
| 결정변수 $\mathbf{X}, \mathbf{U}, \nu_0, \nu_f$ | 50--56행 | `opti.variable()` (CasADi Opti stack) |
| 구간 폭 $h = T/M$ | 59행 | `h = T / self.M` |
| Simpson 비용함수 | 62--72행 | `for k in range(self.M)` 루프, `ca.dot` 사용 |
| 배치점 인덱싱 ($2k$, $2k+1$, $2k+2$) | 64--66행 | `idx_k = 2*k`, `idx_m = 2*k+1`, `idx_k1 = 2*k+2` |
| 운동방정식 $\mathbf{f}(\mathbf{x}, \mathbf{u})$ | 88--90행 | `self.eom_func(x_k, u_k)` (CasADi Function) |
| Simpson 연속성 (C2) | 93--95행 | `opti.subject_to(x_k1 == x_k + (h/6)*(f_k + 4*f_m + f_k1))` |
| Hermite 중간점 (C1) | 98--100행 | `opti.subject_to(x_m == 0.5*(x_k + x_k1) + (h/8)*(f_k - f_k1))` |
| 경계조건 (궤도요소 $\to$ ECI) | 102--116행 | `oe_to_rv_casadi` 호출, `opti.subject_to` |
| 추력 상한 | 119--121행 | `ca.dot(U[:,k], U[:,k]) <= u_max_sq` |
| 최소 고도 | 124--126행 | `ca.dot(X[:3,k], X[:3,k]) >= r_min_sq` |
| NLP 솔버 호출 | 142--149행 | IPOPT via `opti.solver('ipopt', opts)` |

### 3.5 NLP 제약조건의 규모

$M = 30$ 구간에서 NLP의 제약조건 수는 다음과 같다.

| 제약 유형 | 개수 (스칼라) | 산출 |
|-----------|--------------|------|
| Simpson 연속성 (등식) | 180 | $6 \times 30$ |
| Hermite 중간점 (등식) | 180 | $6 \times 30$ |
| 출발 경계조건 (등식) | 6 | $6 \times 1$ |
| 도착 경계조건 (등식) | 6 | $6 \times 1$ |
| 추력 상한 (부등식) | 61 | $1 \times 61$ |
| 최소 고도 (부등식) | 61 | $1 \times 61$ |
| **합계** | **494** | |

등식 제약 372개, 부등식 제약 122개로 구성된 중규모 NLP이다.

### 3.6 IPOPT 솔버 설정

NLP는 내점법(interior-point method) 기반 솔버 IPOPT로 풀린다. Pass 1에서 사용되는 주요 설정 파라미터는 다음과 같다(`config.py` 참조).

| 파라미터 | 값 | 설명 |
|----------|---|------|
| `tol` | $10^{-4}$ | KKT 조건 수렴 허용 오차 |
| `constr_viol_tol` | $10^{-4}$ | 제약조건 위반 허용 오차 |
| `max_iter` | 500 | 최대 반복 횟수 |
| `linear_solver` | MUMPS | 선형 시스템 솔버 |
| `mu_strategy` | adaptive | 장벽 파라미터 갱신 전략 |

Pass 1은 초기 해를 생성하는 단계이므로, 상대적으로 느슨한 허용 오차($10^{-4}$)를 사용한다. 이후 Pass 2(LGL pseudospectral method)에서 $10^{-6}$ 수준으로 정밀화한다.

### 3.7 수렴 실패 처리

IPOPT가 `RuntimeError`를 발생시키는 경우(최대 반복 초과 또는 수렴 실패), `opti.debug.value()`를 통해 마지막 반복의 해를 추출한다(157--164행). 이 경우 `converged=False`가 반환되며, 비용함수는 `inf`로 설정된다. 이렇게 함으로써 수렴 실패 시에도 프로그램이 중단되지 않고, 후속 처리(재시도 등)가 가능하다.

---

## 4. 수치 검증

테스트 코드(`tests/test_collocation.py`)는 초기값 생성, 솔버 옵션, Hermite-Simpson collocation의 세 범주로 구성된다. 아래에 collocation 관련 핵심 테스트 항목을 정리한다.

### 4.1 초기화 검증

| 테스트 항목 | 검증 내용 | 비고 |
|------------|----------|------|
| `test_init` | $M = 30$, $N_{\text{points}} = 61$ | 기본 구간 수 |
| `test_custom_segments` | $M = 10 \Rightarrow N_{\text{points}} = 21$ | 사용자 지정 구간 |

### 4.2 NLP 수렴 테스트 (slow 마크)

다음 테스트들은 실제 IPOPT 풀이를 수행하며, `@pytest.mark.slow`로 표시되어 있다.

| 테스트 항목 | 문제 | 검증 내용 |
|------------|------|----------|
| `test_coplanar_transfer_R1` | $\Delta a = 200$ km, $\Delta i = 0$, $T = 2T_0$ | 수렴 여부, 비용 > 0, 추력 상한 만족 |
| `test_orbit_lowering_R4` | $\Delta a = -200$ km, $\Delta i = 0$, $T = 1.5T_0$ | 수렴 여부 (궤도 하강 시나리오) |
| `test_minimum_altitude_constraint` | $\Delta a = 200$ km, $\Delta i = 0$, $T = 2T_0$ | 모든 배치점에서 고도 $\geq h_{\min} - 1$ km |
| `test_result_fields` | $\Delta a = 200$ km, $\Delta i = 0$, $T = 2T_0$ | `TrajectoryResult` 필드 및 배열 크기 |

### 4.3 검증 기준

**수렴 판정.** `result.converged == True`를 확인한다. 이는 IPOPT가 지정된 허용 오차 내에서 KKT 조건을 만족하는 해를 찾았음을 의미한다. 내부적으로 defect(배치 잔차)는 `constr_viol_tol` = $10^{-4}$ 이내로 보장된다.

**추력 상한.** 모든 배치점에서 추력 크기가 상한 이내인지 확인한다.

$$
\|\mathbf{u}_j\| \leq u_{\max} + 10^{-6}, \quad \forall\, j = 0, 1, \ldots, 60
$$

$10^{-6}$의 여유는 수치적 허용 오차를 고려한 것이다.

**최소 고도.** 수렴한 해에 대해 모든 배치점의 고도가 최소 고도를 만족하는지 확인한다.

$$
\|\mathbf{r}_j\| - R_E \geq h_{\min} - 1.0 \;\text{km}, \quad \forall\, j
$$

1.0 km의 여유는 제약조건 위반 허용 오차(`constr_viol_tol` = $10^{-4}$)에 의한 수치적 마진을 반영한다.

### 4.4 초기값 생성 검증

NLP의 수렴 성능은 초기값의 품질에 크게 의존한다. 테스트 코드는 두 가지 초기값 생성 방법을 검증한다.

**(1) 선형 보간 (`TestLinearInterpolationGuess`).** 출발-도착 상태 사이를 단순 선형 보간하는 방법이다. 테스트는 배열 크기, 시간 범위, 제어 초기값(영벡터), 경계 위치의 궤도 반지름을 검증한다.

**(2) 케플러 전파 (`TestKeplerianGuess`).** 출발 궤도에서의 케플러 운동을 전파하여 초기값을 생성하는 방법이다. 원궤도 전파이므로 전 구간에서 궤도 반지름이 보존되어야 한다(`test_keplerian_radius_conservation`). Coplanar 전이에서는 $\nu_0 = 0$, plane change에서는 $\nu_0 = \pi/2$(교선 근방에서 면외 기동이 효율적)가 사용된다.

---

## 5. 참고문헌

1. Hargraves, C. R. and Paris, S. W. (1987). "Direct trajectory optimization using nonlinear programming and collocation," *Journal of Guidance, Control, and Dynamics*, 10(4), 338--342.
2. Herman, A. L. and Conway, B. A. (1996). "Direct optimization using collocation based on high-order Gauss-Lobatto quadrature rules," *Journal of Guidance, Control, and Dynamics*, 19(3), 592--599.
3. Betts, J. T. (1998). "Survey of numerical methods for trajectory optimization," *Journal of Guidance, Control, and Dynamics*, 21(2), 193--207.
4. Betts, J. T. (2010). *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed. SIAM.
5. Conway, B. A. (Ed.) (2010). *Spacecraft Trajectory Optimization*. Cambridge University Press.
6. Wachter, A. and Biegler, L. T. (2006). "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming," *Mathematical Programming*, 106(1), 25--57.
7. Stoer, J. and Bulirsch, R. (2002). *Introduction to Numerical Analysis*, 3rd ed. Springer.
