# 분할구간별 볼록화에 기반한 Bézier 궤적 초기화 기법


---

**초록**

본 논문에서는 구형 Keep-Out Zone(KOZ)을 연속적으로 회피하는 Bézier 기반 궤적 초기화 기법을 제안한다. 곡선 분할을 통해 구형 KOZ에 대한 비볼록 부등식 제약 조건을 선형화하고, 이를 제어점 공간에서의 선형 연산으로 표현함으로써 궤적 초기화 문제를 일련의 볼록 최적화 문제로 정식화한다. 생성된 궤적은 그 자체로 제약을 만족하며, 후속 고충실도 최적화에 초기값으로도 활용할 수 있다.

제안 기법을 단순화된 궤도전이 문제에 적용하여, 곡선의 분할 수와 차수가 해의 보수성 및 계산 비용에 미치는 영향을 분석한다. 이어서 두 단계로 이루어진 direct collocation 파이프라인에서 2단계를 동일하게 유지한 채 1단계 초기화만 서로 다르게 구성한 두 파이프라인을 비교하여, 제안 기법이 기존 초기화 단계를 대체할 수 있는 가능성을 검토한다.

---

## 1. 서론

제약이 있는 궤적 최적화 문제는 항공우주, 로봇공학, 자율 시스템 등 여러 분야에서 반복적으로 등장한다. 이때 중요한 요구 조건 가운데 하나는 궤적이 특정 금지 영역을 경로 전체에 걸쳐 지속적으로 회피해야 한다는 것이다. 그러나 일반적인 direct transcription 또는 direct collocation 방식 [3, 4]에서는 제약식이 주로 이산화된 노드에서만 부과되므로, 그 노드들에서 제약을 만족하더라도 노드 사이 구간에서 제약이 위반되는 노드 간 제약 위반(inter-sample constraint violation) [11]이 발생할 수 있다. 본 논문에서는 이산화된 노드만이 아니라 궤적 전 구간에서 제약이 성립하는 성질을 연속시간 제약 만족(continuous-time constraint satisfaction) [12]이라 부르며, 금지 영역 회피를 이러한 의미에서 보장하는 표현과 제약 방식이 필요하다.

또 다른 실용적 문제는 초기값의 품질이다. 많은 후속 solver는 초기값에 민감하며, 초기값이 좋지 않으면 제약을 만족하지 않는 해로 수렴하거나 반복 횟수가 크게 증가하거나 품질이 낮은 국소해에 머무를 수 있다. 이런 점에서 후속 고충실도 최적화에 앞서 매끄럽고 제약을 만족하는 초기 궤적을 생성하는 절차는 그 자체로 의미가 있다 [1, 2].

본 논문은 이러한 문제를 해결하기 위해 Bézier 곡선을 이용한 궤적 초기화 기법을 제안한다. 제안 기법의 핵심은 모든 계산을 제어점 공간에서 수행한다는 점이다. 곡선의 미분, 분할, 경계조건, KOZ 제약이 모두 제어점에 대한 선형 연산으로 정리되므로, 계산 구조가 비교적 단순하고 해석도 명확하다. 특히 구형 KOZ에 대해서는 각 분할구간에 supporting half-space(지지 반공간)을 부여하고, 그 반공간 안에 제어점이 놓이도록 함으로써 KOZ 제약의 연속시간 만족을 보수적으로 보장한다.

본 논문의 기여는 다음과 같이 정리할 수 있다. 첫째, Bézier 매개변수화를 기반으로 제어점 공간에서 직접 작동하는 궤적 초기화 정식화를 제시하여 제약 구성과 계산 구조를 단순화한다. 둘째, De Casteljau 분할과 지지 반공간을 이용하여 구형 KOZ 제약을 연속시간에서 만족하도록 하는 보수적 제약 구성 방식을 제안하고, 이를 표준적인 SCvx 틀 [6, 7]에 결합하여 각 반복에서 볼록 QP 하나를 푸는 알고리즘으로 정리한다. 셋째, 단순화된 궤도전이 문제에서 분할 수와 Bézier 차수에 대한 비교 실험을 수행하여 계산 비용과 성능의 관계를 분석한다. 넷째, 두 단계 direct collocation 파이프라인의 1단계 초기화를 제안 기법으로 대체하는 비교 실험을 수행하여, 정의된 작동 영역 안에서 기존 초기화 단계의 교체 가능성을 확인한다.

관련 연구는 크게 세 갈래로 나눌 수 있다. 첫째는 direct transcription 및 direct collocation 계열의 궤적 최적화 방법, 둘째는 장애물 회피를 위한 볼록화 및 보수적 근사 기법, 셋째는 후속 최적화를 위한 초기화와 warm start 생성 방법이다.

Direct transcription과 direct collocation은 제약이 있는 궤적 최적화에서 가장 널리 쓰이는 방법이다 [3, 4]. 이들 방법은 궤적을 여러 노드에서의 상태와 입력 변수로 이산화하고, 동역학을 등식 제약으로 부과한 뒤, 대규모 비선형 계획 문제를 푼다. 다양한 문제에 적용 가능하고 solver 생태계도 잘 갖추어져 있다는 장점이 있다.

다만 점별 이산화에 기반한 이러한 정식화에서는 제약의 연속시간 만족을 직접 다루기 어렵고, 초기값의 품질 또한 수렴 거동에 큰 영향을 줄 수 있다. 본 논문은 이러한 측면에서 제어점 공간 정식화와 KOZ 제약의 연속시간 만족을 보장하는 보수적 제약 구성을 제시한다. Direct collocation은 제안 기법이 연계될 수 있는 대표적인 비교 대상이다.

연속적인 장애물 회피에서는 이산화된 노드에서의 제약 만족만으로는 노드 간 제약 위반을 배제하기 어렵다. 이를 다루기 위한 여러 접근 가운데 일부는 특정 부류의 문제에 대해 무손실 볼록화를 사용하고 [5], 또 다른 일부는 순차 볼록화로 비볼록 제약을 반복적으로 선형화한다 [6, 7].

본 논문은 순차 볼록화의 틀을 사용하되, 점별 상태 제약을 부과하는 대신 Bézier 분할구간의 제어점에 제약을 가하는 형태로 적용한다. 구체적으로는 각 분할구간에 대해 구형 KOZ의 지지 반공간을 구성하고, 제어점이 그 반공간 안에 위치하도록 한다. 이 방식은 회피를 보수적으로 보장하며, 볼록 껍질 성질을 이용하여 분할구간 전체가 KOZ 바깥에 놓임을 보일 수 있다는 장점이 있다.

초기값의 품질이 비선형 궤적 최적화의 수렴 거동에 큰 영향을 준다는 점은 잘 알려져 있다. 실제로는 직선 보간, 경험 기반 형상 설계, 단순 모델 해, 데이터베이스 기반 초기화 등 다양한 방식이 사용된다 [1, 2]. 그러나 단순한 초기화 방법은 노드 간 제약 위반을 배제하지 못하는 경우가 많다.

본 논문의 제안 기법은 이러한 초기화 방법들과도 연결될 수 있다. 즉, 비교적 저차원인 제어점 공간에서 매끄럽고 제약을 만족하는 궤적을 먼저 만든 다음, 이를 후속 solver에 초기값으로 제공하는 방식이다 [1, 2]. 이 접근 방식은 초기화 문제의 차원을 작게 유지하면서도, 구형 KOZ 제약의 연속시간 만족을 명시적으로 다룰 수 있다는 장점이 있다.

본 논문의 이후 구성은 다음과 같다. 2절에서는 궤적의 제어점 공간 표현과 표기법을 정리하고, 3절에서는 제안 기법의 수학적 구성과 알고리즘을 설명한다. 4절에서는 실험 설정을 기술하고, 5절에서는 수치 결과를 제시한다. 6절에서는 한계와 함께 결론을 맺는다.

---

## 2. 제어점 공간에서의 궤적 표현

이 절에서는 제안 기법의 모든 계산이 이루어지는 제어점 공간을 정의하고, 3절의 볼록 최적화에 필요한 선형·이차 구성 요소를 차례로 마련한다. 먼저 궤적을 Bézier 곡선으로 표현하여 결정 변수를 정의하고(2.1절), 속도·가속도와 경계조건을 제어점에 대한 선형 연산으로 정리한 뒤(2.2절), 곡선을 따라 정의된 양의 크기를 제곱하여 적분한 값을 제어점에 대한 이차형식으로 나타낸다(2.3절). 이렇게 마련한 구성 요소를 바탕으로, 3절에서 KOZ 회피를 포함한 궤적 초기화 문제를 일련의 볼록 QP로 정식화할 수 있다.

### 2.1 궤적 표현과 결정 변수
우주비행체의 3차원 궤도전이 문제에서 궤적은 정규화 매개변수 $\tau \in [0,1]$ 위에서 정의된 차수 $N$의 Bézier 곡선으로 표현한다.

$$
\mathbf{r}(\tau) = \sum_{i=0}^{N} B_i^{N}(\tau)\,\mathbf{p}_i
$$

여기서 $B_i^N$은 Bernstein 기저다항식이고, $\mathbf{p}_i \in \mathbb{R}^3$은 $i$번째 제어점이다. 제어점을 행렬로 모으면

$$
P = [\mathbf{p}_0, \mathbf{p}_1, \ldots, \mathbf{p}_N]^{\mathsf{T}} \in \mathbb{R}^{(N+1)\times 3}
$$

가 되고, 이를 하나의 벡터로 쌓으면

$$
\mathbf{x} = \mathrm{vec}\!\left(P^{\mathsf{T}}\right) = [\mathbf{p}_0^{\mathsf{T}}, \mathbf{p}_1^{\mathsf{T}}, \ldots, \mathbf{p}_N^{\mathsf{T}}]^{\mathsf{T}} \in \mathbb{R}^{3(N+1)}
$$

를 얻는다. 본 논문에서 최적화의 결정 변수는 $\mathbf{x}$이며, 이후의 미분 연산, 분할, KOZ 제약은 모두 이 벡터에 대한 선형 연산으로 표현된다.

본 논문에서는 편의상 전이 시간 $T$를 고정한다. 물리 시간 $t$와 정규화 매개변수 $\tau$의 관계는

$$
t = T\tau
$$

로 둔다. 따라서 물리적인 속도와 가속도는 $\tau$에 대한 미분에 각각 $1/T$, $1/T^2$를 곱한 형태로 얻어진다.

### 2.2 미분 연산자와 경계조건

속도와 가속도를 제어점에 대한 선형 연산으로 표현하기 위해, 먼저 Bézier 곡선의 미분 구조를 정리한다. 이는 차분 행렬(difference matrix) $D_N$으로 나타낼 수 있으며, 여기서 $D_N = N[d_{il}] \in \mathbb{R}^{N\times(N+1)}$이고 $d_{i,i}=-1$, $d_{i,i+1}=1$, 그 밖의 항은 0이다.


$$
D_N
=
N
\begin{bmatrix}
-1 & 1 & 0 & \cdots & 0 \\\
0 & -1 & 1 & \ddots & \vdots \\\
\vdots & \ddots & \ddots & \ddots & 0 \\\\
0 & \cdots & 0 & -1 & 1
\end{bmatrix}
\in \mathbb{R}^{N\times(N+1)}
$$

또한 미분으로 얻은 제어점을 다시 원래 차수의 기저로 표현하기 위해 차수 상승 행렬(degree elevation matrix) $E_M \in \mathbb{R}^{(M+2)\times(M+1)}$을 사용한다. 이 행렬은 차수 $M$의 제어점을 차수 $M+1$의 기저로 옮긴다. 이를 이용하면 차수를 보존하는 속도·가속도 연산자를

$$
L_{1,N} = E_{N-1}D_N, \qquad L_{2,N} = E_{N-1}D_N E_{N-1}D_N
$$

로 쓸 수 있고, 대응하는 제어점은

$$
P^{[1]} = L_{1,N}P, \qquad P^{[2]} = L_{2,N}P
$$

이다.

물리적인 속도와 가속도는 다음과 같다.

$$
\dot{\mathbf{r}}(t) = \frac{1}{T}\frac{d\mathbf{r}}{d\tau}, \qquad \ddot{\mathbf{r}}(t) = \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2}
$$

끝점 위치 조건은 첫 제어점과 마지막 제어점을 고정하여 부과한다. 끝점 속도는

$$
\dot{\mathbf{r}}(0) = \frac{N}{T}(\mathbf{p}_1-\mathbf{p}_0), \qquad \dot{\mathbf{r}}(T) = \frac{N}{T}(\mathbf{p}_N-\mathbf{p}_{N-1})
$$

로 주어지며, 필요할 경우 끝점 가속도도 같은 방식으로 선형 제약식으로 표현할 수 있다.

### 2.3 Gram 행렬과 이차형식

3절의 목적함수는 곡선을 따라 정의된 벡터값 함수의 크기를 제곱하여 적분한 양이다. 이러한 적분을 결정 변수 $\mathbf{x}$에 대한 이차형식으로 옮기기 위해, Bernstein 기저다항식 사이의 내적을 모은 행렬을 마련한다. 이 행렬을 Gram 행렬이라 하며, Bernstein 기저에 대해서는 닫힌 형태로 계산할 수 있다.

$$
[G_N]_{il} = \frac{\binom{N}{i}\binom{N}{l}}{\binom{2N}{i+l}(2N+1)}, \qquad i,l=0,\ldots,N
$$

차수 $N$의 Bézier 곡선 $\mathbf{f}(\tau) = \sum_{i=0}^{N}B_i^N(\tau)\,\mathbf{f}_i$에 대해, 그 크기를 제곱하여 적분한 값은 제어점만으로

$$
\int_0^1 \|\mathbf{f}(\tau)\|_2^2\,d\tau = \sum_{i=0}^{N}\sum_{l=0}^{N}[G_N]_{il}\,\mathbf{f}_i^{\mathsf{T}}\mathbf{f}_l = \mathrm{tr}(F^{\mathsf{T}} G_N F), \qquad F = [\mathbf{f}_0,\ldots,\mathbf{f}_N]^{\mathsf{T}} \in \mathbb{R}^{(N+1)\times 3}
$$

로 정리된다. 이 식에는 이산화나 수치 적분이 들어 있지 않다. 따라서 제어점 $\mathbf{f}_i$가 결정 변수 $\mathbf{x}$에 대한 1차 함수이기만 하면, 위 적분은 $\mathbf{x}$에 대한 볼록 이차형식이 되고 그 값은 근사 없이 정확하다.

이 항등식은 2.2절의 미분 연산자와 결합하여 사용한다. $\mathbf{f}$를 가속도 곡선으로 두면 $F = L_{2,N}P$이므로, $\tilde G_N = L_{2,N}^{\mathsf{T}} G_N L_{2,N}$을 써서

$$
\int_0^1 \left\|\frac{d^2\mathbf{r}}{d\tau^2}\right\|_2^2 d\tau = \mathrm{tr}(P^{\mathsf{T}} \tilde G_N P) = \mathbf{x}^{\mathsf{T}} (\tilde G_N \otimes I_3)\mathbf{x}
$$

를 얻는다. 3.2절의 목적함수는 $\mathbf{f}$를 제어 가속도의 잔차 곡선으로 두어 같은 항등식을 적용한 것이다. 곡선을 따라 정의된 양의 적분을 제어점에 대한 이차형식으로 정확히 옮길 수 있다는 점은 제어점 공간에서 정식화하여 얻는 이점 가운데 하나이다.

---

## 3. 순차 볼록화 기반 궤적 초기화

이 절에서는 2절에서 마련한 제어점 공간의 선형·이차 구성 요소를 이용하여 제안 기법을 구성한다. 먼저 비볼록인 구형 KOZ 회피 조건을 제어점에 대한 볼록 제약으로 바꾸는 방법을 제시하고(3.1절), 이어서 궤적의 형태를 정하는 제어 비용 목적함수를 정의한 뒤(3.2절), 이 둘을 결합하여 매 반복(iteration)에서 하나의 볼록 QP를 푸는 SCvx 알고리즘으로 정리한다(3.3절).

### 3.1 분할과 지지 반공간을 이용한 구형 KOZ 처리

제안 기법의 첫 과제는 비볼록인 구형 KOZ 회피 조건을 제어점에 대한 볼록 제약으로 바꾸는 것이다. 먼저 구형 KOZ를 다음과 같이 정의한다.

$$
\mathcal{K} = \left\{\mathbf{r}\in\mathbb{R}^3 : \|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le R_{\mathrm{KOZ}} \right\}
$$

여기서 $\mathbf{c}_{\mathrm{KOZ}}$는 KOZ의 중심이며, 본 논문에서는 KOZ 중심을 원점에 둔다.

궤적 $\mathbf{r}(\tau)$가 KOZ 제약을 연속시간에서 만족한다는 것은, 모든 $\tau \in [0,1]$에 대해 $\mathbf{r}(\tau) \notin \operatorname{int}\mathcal{K}$, 즉 $\|\mathbf{r}(\tau)-\mathbf{c}_{\mathrm{KOZ}}\|_2 \ge R_{\mathrm{KOZ}}$가 성립함을 뜻한다. 이는 유한개의 노드에서만 회피를 요구하는 점별 제약 만족보다 강한 조건이다. 본 논문에서는 전이 시간 $T$가 고정되어 물리 시간 $t$와 매개변수 $\tau$가 일대일로 대응하므로, $\tau$ 전 구간에서의 만족은 곧 연속시간 만족과 같다. 제안 기법은 이 비볼록 조건을 직접 부과하는 대신, 이를 함의하는 볼록 충분조건(명제 1)을 부과한다.

제안 기법에서는 곡선을 $n_{\mathrm{seg}}$개의 분할구간으로 등분하기 위해 De Casteljau 분할 행렬 $S^{(s)}$를 사용한다. 그러면 $s$번째 분할구간의 제어점은

$$
P^{(s)} = S^{(s)}P
$$

가 된다. 이 분할구간의 제어점을 $\mathbf{q}^{(s)}_0, \ldots, \mathbf{q}^{(s)}_N$이라 하면, 대표점으로는 제어점의 중심점

$$
\mathbf{c}^{(s)} = \frac{1}{N+1}\sum_{m=0}^{N}\mathbf{q}^{(s)}_m
$$

를 사용한다. 분할 전후의 제어점과 그 볼록 껍질의 관계는 [그림 1](#fig-ctrl-subdivision)에 나타내었다.

<a id="fig-ctrl-subdivision"></a>
![그림 1. De Casteljau 분할에 따른 제어점과 볼록 껍질의 변화](../figures/f7_control_subdivision.png)
**그림 1 [F1].** 분할 행렬 $S^{(s)}$로 얻은 분할구간 제어점 $P^{(s)}$와 그 볼록 껍질. 분할구간마다 볼록 껍질이 곡선을 더 좁게 감싸며, 이 성질이 명제 1의 근거가 된다.

이때 외향 법선은

$$
\mathbf{n}^{(s)} = \frac{\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}}{\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\|_2}
$$

로 정의하며, $\mathbf{c}^{(s)} = \mathbf{c}_{\mathrm{KOZ}}$인 경우에는 법선 방향이 정의되지 않으므로 그 분할구간은 제약 구성에서 제외한다. 이에 따라 구의 지지 반공간은

$$
\mathcal{H}^{(s)} = \left\{\mathbf{r} : (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{r} \ge (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{c}_{\mathrm{KOZ}} + R_{\mathrm{KOZ}} \right\}
$$

로 쓸 수 있다.

본 논문에서는 각 분할구간의 모든 제어점이 이 반공간 안에 놓이도록 다음 부등식을 부과한다.

$$
(\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{q}^{(s)}_m \ge (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{c}_{\mathrm{KOZ}} + R_{\mathrm{KOZ}}, \qquad m=0,\ldots,N
$$

이 제약 구성이 연속시간 제약 만족을 보장한다는 사실은 다음 명제로 정리할 수 있다.

> **명제 1.** 분할구간 $s$의 제어점을 $P^{(s)} = S^{(s)}P$라 하고, 구형 KOZ를 $\mathcal{K} = \{\mathbf{r}\in\mathbb{R}^3 : \|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le R_{\mathrm{KOZ}}\}$라 하자. 위에서 정의한 지지 반공간 $\mathcal{H}^{(s)}$에 대해, 해당 분할구간의 모든 제어점 $\mathbf{q}^{(s)}_0, \ldots, \mathbf{q}^{(s)}_N$이 $\mathcal{H}^{(s)}$ 안에 놓이면, 그 분할구간의 Bézier 곡선 전체도 $\mathcal{H}^{(s)}$ 안에 놓이고, 따라서 $\mathcal{K}$ 바깥에 놓인다.

> **가정.** 이 명제는 다음 가정 하에서 성립한다.
> 1. 장애물은 구형이다.
> 2. 법선 $\mathbf{n}^{(s)}$은 분할구간 제어점의 중심점 $\mathbf{c}^{(s)}$으로부터 구성된다.
> 3. 동일한 지지 반공간 $\mathcal{H}^{(s)}$이 해당 분할구간의 모든 제어점에 부과된다.
> 4. 법선 구성 시 $\mathbf{c}^{(s)} \neq \mathbf{c}_{\mathrm{KOZ}}$이다.

> **증명.** Bézier 곡선은 제어점의 볼록 껍질 안에 놓인다. 구의 지지 반공간은 구의 내부를 배제하면서 경계에 접한다. 따라서 모든 제어점이 $\mathcal{H}^{(s)}$ 안에 있으면 볼록 껍질 전체도 $\mathcal{H}^{(s)}$ 안에 있고, 곡선도 $\mathcal{H}^{(s)} \cap \mathcal{K}^c$ 안에 놓인다. $\square$

각 $\mathbf{q}^{(s)}_m$는 원래 제어점의 선형결합이므로, 법선 $\mathbf{n}^{(s)}$을 하나의 방향으로 고정해 두면 위 부등식은 결정 변수 $\mathbf{x}$에 대해 선형이다. 그러나 법선 자체가 중심점 $\mathbf{c}^{(s)}$을 통해 $\mathbf{x}$에 의존한다. 명제 1의 조건을 $\mathbf{x}$의 함수로 적으면

$$
h(\mathbf{x}) = \sum_{s}\sum_{m=0}^{N}\max\left\{0,\ R_{\mathrm{KOZ}} + \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\mathbf{c}_{\mathrm{KOZ}} - \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\mathbf{q}^{(s)}_m(\mathbf{x})\right\}
$$

가 되며, 이 양은 $\mathbf{x}$에 대해 비볼록이다. 비볼록성은 법선을 만들 때의 정규화 $\mathbf{v}\mapsto\mathbf{v}/\|\mathbf{v}\|_2$에서만 나온다. $h(\mathbf{x}) = 0$이면 명제 1에 의해 곡선 전체가 KOZ 바깥에 놓이므로, $h$는 제안한 제약 구성의 실현 가능성을 그대로 재는 양이다. 3.3절의 하위 문제는 이 $h$를 기준점에서 1차 전개하여 볼록 제약으로 옮긴다. 반공간은 각 SCvx 반복에서 현재 해를 기준으로 다시 구성되므로, 이렇게 얻은 볼록 제약은 현재 해 주변에서 작동하는 보수적이고 국소적인 회피 제약 조건으로 이해할 수 있다.

명제 1은 법선의 선택과 무관하게 성립한다. 단위 벡터 $\mathbf{n}$에 대해 $\mathbf{n}^{\mathsf{T}}(\mathbf{q}-\mathbf{c}_{\mathrm{KOZ}}) \le \|\mathbf{q}-\mathbf{c}_{\mathrm{KOZ}}\|_2$이므로, 부등식이 성립하는 것만으로 거리 $\|\mathbf{q}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \ge R_{\mathrm{KOZ}}$가 이미 보장된다. 즉 중심점에서 법선을 만드는 규칙은 명제 1의 보수성만을 좌우하며, 타당성에는 영향을 주지 않는다.

보수성은 분할 수에 대해 정량적으로 예측할 수 있다. 지지 반공간은 구에 접하는 평면이므로, 접점에서 평면을 따라 거리 $L_{\mathrm{seg}}$만큼 떨어진 점은 구면보다 약 $L_{\mathrm{seg}}^2/2R_{\mathrm{KOZ}}$만큼 더 바깥에 놓여야 부등식을 만족한다. 분할구간이 평면을 따라 뻗는 폭 $L_{\mathrm{seg}}$은 분할 수에 반비례하므로, 곡선이 KOZ 경계로부터 필요 이상으로 떨어지는 정도는 $n_{\mathrm{seg}}^{-2}$에 비례하여 줄어든다. 이 예측은 5.2절에서 측정 결과와 함께 확인한다.

이상의 구성은 하나의 분할구간에 대해 [그림 2](#fig-subdivision)에 단계별로 나타내었다. (a)는 곡선 전체와 KOZ를 침범하는 분할구간을, (b)는 그 분할구간의 중심점에서 외향 법선과 지지 반공간을 구성하는 과정을, (c)는 모든 제어점이 반공간 안에 놓이도록 수정된 결과를 보인다. 이 그림은 회피 제약이 곡선 전체가 아니라 분할구간 단위로, 현재 해를 기준으로 국소적으로 구성됨을 보이기 위한 것이다. 그림이 보여 주는 것은 하나의 제어점 배치에서 명제 1의 조건 자체이며, 하위 문제가 실제로 푸는 볼록 제약은 이 조건을 기준점에서 1차 전개한 것이다(3.3절).

<a id="fig-subdivision"></a>
![그림 2. 분할과 지지 반공간을 이용한 연속시간 KOZ 제약 만족의 개념도](../figures/f1_koz_linearization.png)
**그림 2 [F2].** 하나의 분할구간에 대한 구형 KOZ 선형화 개념도. 분할구간 제어점의 중심점에서 구성한 지지 반공간 제약을 해당 분할구간의 모든 제어점에 부과함으로써 구형 KOZ를 보수적으로 배제한다.

### 3.2 제어 비용 목적함수

초기 궤적은 KOZ를 회피하면서도 제어 비용(control effort)이 지나치게 크지 않아야 한다. 이를 위해 궤적을 따라 요구되는 제어 가속도의 크기를 줄이는 목적함수를 구성한다. 임의의 위치에서 필요한 제어 가속도는 기하학적 가속도에서 중력 가속도를 뺀 값

$$
\mathbf{u}(t) = \ddot{\mathbf{r}}(t) - \mathbf{g}(\mathbf{r}(t))
$$

로 정의한다. 여기서 $\mathbf{g}$는 궤도 중력 모델로, 본 논문에서는 이체 문제(two-body problem) 항과 J2 섭동(perturbation) 항을 포함한다. 목적함수는 이 제어 가속도의 크기를 궤적 전체에 걸쳐 적분한 값으로 둔다.

$$
J(\mathbf{x}) = \int_0^1 \left\| \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2} - \mathbf{g}(\mathbf{r}(\tau)) \right\|_2^2 d\tau
$$

중력 가속도는 위치에 대해 비선형이므로 이 적분은 $\mathbf{x}$에 대해 비볼록이다. 하위 문제를 볼록하게 유지하기 위해, KOZ 분할과 별도로 곡선을 $n_{\mathrm{lin}}$개의 구간으로 나누고 각 구간에서 중력을 1차 테일러(Taylor) 전개로 근사한다. 이때 구간을 나누는 De Casteljau 분할 행렬은 KOZ 분할의 $S^{(s)}$와 구분하여 $\hat S^{(j)}$로 쓴다. 구간 $j$에서 전개의 기준점은 SCvx 반복 $k$의 기준 제어점이 정하는 그 구간의 중심점 $\mathbf{r}_j^{(k)}$이며,

$$
\mathbf{g}(\mathbf{r}) \approx \nabla\mathbf{g}_j^{(k)}\mathbf{r} + \mathbf{c}_j^{(k)}, \qquad \mathbf{c}_j^{(k)} = \mathbf{g}\!\left(\mathbf{r}_j^{(k)}\right) - \nabla\mathbf{g}_j^{(k)}\mathbf{r}_j^{(k)}
$$

로 쓴다. 여기서 $\nabla\mathbf{g}_j^{(k)} = \partial\mathbf{g}/\partial\mathbf{r}$는 기준점에서 계산한 중력 Jacobian이다. 구간을 나누는 목적은 중력 전개의 기준점을 곡선을 따라 여러 개 두는 데 있으며, 적분의 이산화와는 무관하다. 각 구간 안에서 중력은 위치에 대한 1차 함수로 고정된다.

구간 $j$를 $\xi \in [0,1]$로 다시 매개화하면, 그 구간에서 피적분 함수는 기하학적 가속도에서 선형화한 중력을 뺀 잔차

$$
\mathbf{f}^{(j)}(\xi) = \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2} - \left(\nabla\mathbf{g}_j^{(k)}\mathbf{r} + \mathbf{c}_j^{(k)}\right)
$$

이다. 우변은 $\xi$에 대응하는 $\tau$에서 평가하며, 두 항은 모두 $\xi$에 대한 차수 $N$의 Bézier 곡선이다. 그 제어점은 분할 행렬 $\hat S^{(j)}$와 2.2절의 가속도 연산자를 통해 결정 변수 $\mathbf{x}$에 대한 1차 함수로 얻어진다. 따라서 $\mathbf{f}^{(j)}$ 역시 제어점이 $\mathbf{x}$에 대한 1차 함수인 차수 $N$의 Bézier 곡선이고, 2.3절의 항등식을 그대로 적용할 수 있다. 구간별 적분을 더하면

$$
J^{(k)}(\mathbf{x}) = \sum_{j=1}^{n_{\mathrm{lin}}} \frac{1}{n_{\mathrm{lin}}}\int_0^1 \left\|\mathbf{f}^{(j)}(\xi)\right\|_2^2 d\xi = \sum_{j=1}^{n_{\mathrm{lin}}} \frac{1}{n_{\mathrm{lin}}}\,\mathrm{tr}\!\left(F_j(\mathbf{x})^{\mathsf{T}} G_N F_j(\mathbf{x})\right)
$$

를 얻는다. 여기서 $1/n_{\mathrm{lin}}$은 매개화에 따르는 척도 인자이고, $F_j(\mathbf{x})$는 $\mathbf{f}^{(j)}$의 제어점을 모은 행렬이다.

이 목적함수는 $\mathbf{x}$에 대한 볼록 이차형식이므로 각 SCvx 반복에서 볼록 QP의 목적함수로 그대로 사용한다. 적분은 표본점에서 근사한 값이 아니라 닫힌 형태로 정확히 계산된 값이며, $J^{(k)}$가 원래 목적함수 $J$와 다른 유일한 이유는 중력을 구간별로 선형화한 데 있다. 이 차이는 3.3절의 비 $\rho_k$가 재는 근사 오차의 하나가 된다.

### 3.3 볼록 하위 문제와 SCvx 알고리즘

3.1절의 지지 반공간과 3.2절의 중력 선형화는 모두 기준 제어점 근방에서만 유효한 국소 근사이므로, 제안 기법은 두 근사를 매 반복마다 다시 구성하면서 볼록 하위 문제를 푸는 신뢰 구간(trust region) 기반 순차 볼록화, 즉 SCvx [6, 7]의 틀을 따른다. 이 절에서는 하위 문제를 정의하고 그 해를 새로운 기준점으로 간주하는 조건을 밝힌 뒤, 전체 절차를 Algorithm 1로 정리한다.

SCvx 반복 $k$의 하위 문제는 다음의 볼록 QP이다.

$$
\min_{\mathbf{x},\,\boldsymbol{\nu}} \ \frac{1}{2}\mathbf{x}^{\mathsf{T}} H^{(k)}\mathbf{x} + (\boldsymbol{\ell}^{(k)})^{\mathsf{T}}\mathbf{x} + \mu\,\mathbf{1}^{\mathsf{T}}\boldsymbol{\nu}
$$

$$
A_{\mathrm{KOZ}}^{(k)}\mathbf{x} + \boldsymbol{\nu} \ge \mathbf{b}_{\mathrm{KOZ}}^{(k)}, \quad \boldsymbol{\nu} \ge \mathbf{0}, \qquad
A_{\mathrm{bc}}\mathbf{x} = \mathbf{b}_{\mathrm{bc}}, \qquad
\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \le \Delta_k
$$

여기서 $H^{(k)}$와 $\boldsymbol{\ell}^{(k)}$는 3.2절의 목적함수 $J^{(k)}$를 $\mathbf{x}$에 대해 전개하여 얻는 행렬과 벡터이며, 해에 영향을 주지 않는 상수항은 생략하였다.

KOZ 행 $A_{\mathrm{KOZ}}^{(k)}$, $\mathbf{b}_{\mathrm{KOZ}}^{(k)}$는 3.1절의 조건을 기준점 $\mathbf{x}^{(k)}$에서 1차 전개하여 얻는다. 분할구간 $s$의 제어점 $\mathbf{q}^{(s)}_m$에 대한 여유 거리를

$$
\gamma^{(s)}_m(\mathbf{x}) = \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\!\left(\mathbf{q}^{(s)}_m(\mathbf{x}) - \mathbf{c}_{\mathrm{KOZ}}\right) - R_{\mathrm{KOZ}}
$$

로 두면, 그 기울기는 분할 행렬 $S^{(s)}$와 중심점 가중치 $w^{(s)}_i = \frac{1}{N+1}\sum_{m=0}^{N} S^{(s)}_{mi}$를 써서

$$
\frac{\partial \gamma^{(s)}_m}{\partial \mathbf{p}_i} = S^{(s)}_{mi}\,\mathbf{n}^{(s)} + \frac{w^{(s)}_i}{\left\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\right\|_2}\left(I_3 - \mathbf{n}^{(s)}(\mathbf{n}^{(s)})^{\mathsf{T}}\right)\!\left(\mathbf{q}^{(s)}_m - \mathbf{c}_{\mathrm{KOZ}}\right)
$$

로 쓸 수 있다. 첫째 항은 법선을 고정했을 때의 기울기이고, 둘째 항은 제어점이 움직이면서 중심점이, 따라서 법선의 방향이 함께 도는 효과이다. 사영 $I_3 - \mathbf{n}^{(s)}(\mathbf{n}^{(s)})^{\mathsf{T}}$이 붙어 있으므로 둘째 항은 제어점이 반공간의 경계면을 따라 접점에서 멀리 놓일수록 커지고, 접점 위에 놓인 제어점에서는 사라진다. 이 기울기를 써서 각 KOZ 행을

$$
\gamma^{(s)}_m\!\left(\mathbf{x}^{(k)}\right) + \sum_{i=0}^{N}\left(\frac{\partial \gamma^{(s)}_m}{\partial \mathbf{p}_i}\right)^{\!\mathsf{T}}\!\left(\mathbf{p}_i - \mathbf{p}^{(k)}_i\right) + \nu^{(s)}_m \ \ge\ 0
$$

으로 부과한다. 둘째 항을 빼고 법선을 기준점 값으로 고정하면 하위 문제의 최적해는 다음 반복에서 다시 구성한 반공간을 만족하지 않는 점이 되며, 아래의 비 $\rho_k$가 그 불일치를 실제 개선으로 잘못 읽는다.

여유 변수(slack variable) $\boldsymbol{\nu}$는 선형화된 KOZ 제약의 실현 가능성을 보완해주는 항으로, SCvx의 virtual control [6, 7]에 해당한다. 기준점이 KOZ 안쪽에 놓이는 초기 단계에서는 선형화된 제약이 그 자체로 실현 불가능할 수 있으므로, 여유 변수로 이를 흡수하되 페널티 계수 $\mu$를 두어 수렴한 해에서는 $\boldsymbol{\nu} = \mathbf{0}$이 되도록 한다. $\mu$의 선택 기준은 4.1절에 제시한다. 신뢰 구간 제약 $\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \le \Delta_k$은 선형화가 유효한 범위 밖으로 벗어나는 해를 막으며, 신뢰 구간의 크기 $\Delta_k$는 아래의 기준과 연동하여 반복마다 조절된다.

하위 문제의 해 $\hat{\mathbf{x}}$를 새로운 기준점으로 삼을지는, 목적함수와 제약 조건 위반을 결합한 merit function으로 평가한다 [6, 7].

$$
\phi(\mathbf{x}) = J(\mathbf{x}) + \mu\,h(\mathbf{x})
$$

여기서 $J$는 중력을 비선형 그대로 둔 3.2절의 목적함수이고, $h$는 3.1절에서 정의한 실현 가능성 척도, 즉 $\mathbf{x}$ 자신에서 다시 구성한 지지 반공간에 대한 위반의 합이다. 앞의 볼록 QP가 최소화하는 목적함수는 $\phi$의 두 비볼록 요소를 각각 볼록 근사로 바꾼 $\phi^{(k)} = J^{(k)} + \mu h^{(k)}$와 일치한다. $J^{(k)}$는 중력을 구간별로 선형화한 목적함수이고, $h^{(k)}$는 $h$를 기준점에서 1차 전개한 양으로 하위 문제에서는 여유 변수의 합 $\mathbf{1}^{\mathsf{T}}\boldsymbol{\nu}$로 나타난다. 따라서 하위 문제가 계산한 예측 감소량에 대한 실제 감소량의 비

$$
\rho_k = \frac{\phi(\mathbf{x}^{(k)}) - \phi(\hat{\mathbf{x}})}{\phi^{(k)}(\mathbf{x}^{(k)}) - \phi^{(k)}(\hat{\mathbf{x}})}
$$

는 두 근사가 실제 개선을 얼마나 정확히 예측했는지를 잰다. $\rho_k$가 반영하는 근사 오차는 중력 선형화와 법선의 회전 두 가지이며, 어느 한쪽만 1차 근사로 두고 다른 쪽을 고정하면 $\phi$와 $\phi^{(k)}$가 서로 다른 제약을 재게 되어 이 비가 근사 오차를 재는 의미를 잃는다. $\rho_k$가 기준값 $\eta$를 넘으면 $\hat{\mathbf{x}}$를 새로운 기준점으로 간주하고, 1에 가까우면 신뢰 구간의 크기를 늘려 더 큰 이동을 허용한다. 반대로 $\rho_k \le \eta$이면 $\hat{\mathbf{x}}$를 버리고 신뢰 구간의 크기를 줄여 같은 기준점에서 다시 푼다. 한편 직선 보간으로 만든 초기 제어점은 속도 경계조건을 만족하지 않으므로, 이 등식 제약을 처음으로 만족하게 되는 첫 반복의 해는 위 비교 없이 새로운 기준점으로 삼는다.

수렴 조건은 두 가지를 함께 요구한다. 첫째는 새로운 기준점에서 merit function의 상대 변화가 허용오차보다 작다는 것이고, 둘째는 그 기준점이 $h = 0$, 즉 명제 1의 조건을 만족한다는 것이다. 두 조건을 연속된 $n_{\mathrm{conv}}$번의 반복에서 모두 만족할 때 수렴으로 간주한다. 한 번의 반복만으로 가리면, 법선이 반복마다 다시 구성되면서 제어점이 제약면을 따라 느리게 이동하는 구간에서 상대 변화가 일시적으로 작아지는 것을 수렴으로 잘못 읽을 수 있다. 같은 이유로 제어점 변화의 크기 자체는 수렴 조건으로 쓰지 않는다.

하위 문제가 더 이상 개선을 예측하지 못하는 경우도 같은 방식으로 처리한다. 예측 감소량이 $\phi$의 크기에 대해 허용오차보다 작고 기준점이 명제 1의 조건을 만족하는 반복이 연속 $n_{\mathrm{conv}}$번 나타나면 종료한다. 두 종료 조건은 모두 하위 문제가 정지점에 도달했다는 사실을 근거로 삼는다. 반면 신뢰 구간의 크기가 하한까지 줄어드는 것은 반복이 더 진행되지 못한 경우이므로, 수렴으로 보지 않고 실패로 기록한다.

이로써 제안 기법은 매 반복에서 볼록 QP 하나를 풀고 그 해를 merit function으로 평가하는, 신뢰 구간 기반의 국소 최적화 알고리즘으로 이해할 수 있다.

> **Algorithm 1** — 신뢰 구간 기반 궤적 초기화 (SCvx)
> 1. 초기 제어점 $\mathbf{x}^{(0)}$(직선 보간)과 신뢰 구간의 초기 크기 $\Delta_0$를 둔다
> 2. **for** $k = 0, 1, 2, \ldots$:
> 3. &nbsp;&nbsp; $\mathbf{x}^{(k)}$에서 De Casteljau 분할로 지지 반공간을 다시 구성하고, 그 조건을 1차 전개하여 $A_{\mathrm{KOZ}}^{(k)}$, $\mathbf{b}_{\mathrm{KOZ}}^{(k)}$를 얻는다 (3.1절)
> 4. &nbsp;&nbsp; 각 구간의 중심점에서 중력을 1차 전개하여 $H^{(k)}$, $\boldsymbol{\ell}^{(k)}$를 구성한다 (3.2절)
> 5. &nbsp;&nbsp; 볼록 QP를 풀어 해 $\hat{\mathbf{x}}$와 여유 변수 $\boldsymbol{\nu}$를 얻는다
> 6. &nbsp;&nbsp; $\rho_k$를 계산하여, $\rho_k > \eta$이면 $\hat{\mathbf{x}}$를 새로운 기준점으로 간주하고 $\rho_k$가 1에 가까우면 신뢰 구간의 크기를 늘린다; 그렇지 않으면 $\hat{\mathbf{x}}$를 버리고 크기를 줄인다
> 7. &nbsp;&nbsp; **수렴 판정**: merit function의 상대 변화가 허용오차보다 작은 반복, 또는 예측 감소량이 허용오차보다 작은 반복이 $h = 0$인 기준점에서 연속 $n_{\mathrm{conv}}$번 나타나면 종료한다
> 8. &nbsp;&nbsp; 신뢰 구간의 크기가 하한 미만으로 줄어들면 실패로 기록하고 중단한다

신뢰 구간의 초기 크기와 조절 계수, 기준값 $\eta$, 페널티 계수 $\mu$, 허용오차와 연속 만족 횟수 $n_{\mathrm{conv}}$의 값은 4.1절에 제시한다.

(이상의 절차를 [그림 3](#fig-scp-pipeline)에 도식화하였다. 그림에서는 한 번만 만들어 재사용하는 연산자와 매 반복에서 다시 구성하는 부분, 즉 지지 반공간과 중력 선형화를 구분하여 표시하였다. 또한 비 $\rho_k$가 기준값을 넘지 못하면 같은 기준점에서 신뢰 구간의 크기를 줄여 다시 푸는 경로와, 새로운 기준점으로 옮긴 뒤에도 수렴 조건을 만족하지 못하면 다음 반복으로 넘어가는 경로를 나누어 그렸다. 이 그림은 매 반복의 계산량이 어디에서 발생하는지와, 해를 새로운 기준점으로 간주하는 단계가 반복 구조의 어느 지점에 놓이는지를 함께 보이기 위한 것이다.)[[this but shorter. like we visualized this process to fig 3. or smh. it def can be shorter and def should be more organic.]]

<a id="fig-scp-pipeline"></a>
![그림 3. 제어점 공간에서의 SCvx 반복](../figures/f2_scp_pipeline.png)
**그림 3 [F3].** 제안 SCvx 반복의 제어점 공간 구현. 매 반복에서 지지 반공간과 중력 선형화를 다시 만들고 볼록 QP를 풀어 기준점을 옮겨 가는 구조를 보여준다.

---

## 4. 실험 설정

### 4.1 시연 문제와 평가 지표

실험에는 단순화된 3차원 궤도전이 문제를 사용하였다. 우주선은 지구 중심의 구형 KOZ를 회피하면서 주어진 초기 위치와 최종 위치 사이를 이동해야 한다. KOZ 반경은 $R_{\mathrm{KOZ}} = 6471$ km, 전이 시간은 $T = 1500$ s로 고정하였다. 양 끝점에서 위치를 고정하고, 초기 및 최종 속도 제약도 함께 부과하였다. 중력장은 이체 문제 항과 J2 섭동을 포함한다. 목적함수를 계산할 때 중력장은 각 구간의 중심점에서 1차 테일러 전개로 근사하지만, 그렇게 얻은 피적분 함수의 적분 자체는 2.3절의 항등식으로 정확히 계산한다.

시연 시나리오는 Progress 우주선의 ISS 접근 문제를 단순화한 단일 arc 궤도전이 시나리오로, 실제 궤도를 기반으로 하지만 분석 편의상 단순화하였다. (chaser는 고도 245 km의 원궤도, target은 고도 400 km 원궤도에서 시작하며, 두 궤도는 동일 평면(경사각 51.64 deg) 내에서 120 deg의 초기 위상차를 갖는다고 가정한다.)[[단순 궤도전이 문제인데 target/chaser 가 왜 나오는지?]]

이러한 시나리오를 선택한 이유는, 120 deg 위상차 시나리오에서는 초기 궤적이 KOZ 경계 근처까지 접근하는 경로가 자연스럽게 형성되기 때문이다. 즉, 분할 수 변화에 따른 수치적 보수성 차이가 실제로 뚜렷하게 드러나는 사례이다.

최적화에는 Rust로 구현한 QP solver를 사용하였다. SCvx 반복은 직선 보간으로 만든 초기 제어점에서 시작하며, 신뢰 구간의 초기 크기는 2000 km로 두고 $\rho_k$에 따라 2배로 늘리거나 절반으로 줄인다. 기준값은 $\eta = 0.1$, merit function의 상대 변화와 예측 감소량에 공통으로 적용하는 수렴 허용오차는 $10^{-8}$, 연속 만족 횟수는 $n_{\mathrm{conv}} = 3$, 신뢰 구간 크기의 하한은 $10^{-2}$ km, 반복 한도는 1000회로 두었다.

페널티 계수는 $\mu = 10^{-2}$로 두었다. L1 페널티 항이 정확한 페널티(exact penalty)로 작동하여 수렴한 해에서 여유 변수가 0이 되려면, 계수가 해당 제약의 쌍대변수 크기를 넘어야 한다 [13, 14]. 본 문제에서 KOZ 제약 쌍대변수의 크기는 약 $1.5\times10^{-7}$ 수준으로 측정되었으므로, $\mu = 10^{-2}$는 이 조건을 약 $10^5$배의 여유를 두고 만족한다. 동시에 이 값은 목적함수의 규모를 압도하지 않으므로 $\rho_k$가 목적 개선에 둔감해지지 않는다.

본 논문에서 사용하는 평가 지표는 다음과 같다. 성공 여부(solve success)는 최종 해가 모든 제약 조건을 만족하는지를, 안전 여유(safety margin)는 최종 궤적의 최소 반경에서 KOZ 반경을 뺀 값을 나타낸다. 제어 비용은 최종 궤적을 따라 요구되는 제어 가속도 $\|\mathbf{u}\|_2$의 평균 크기(m/s²)로 측정한다. 계산 시간(runtime)은 SCvx 반복 전체에 소요된 시간이며, 반복 횟수(iterations)는 종료 시점까지 수행된 횟수이다.

### 4.2 분할 수와 차수에 대한 비교 실험 설정

(첫 번째 비교 실험에서는 Bézier 차수를 $N=7$로 고정한 채, 분할 수 $n_{\mathrm{seg}} \in \{2,4,8,16,32,64\}$를 바꿔가며 그 영향을 측정한다. 이 실험의 목적은 분할 수가 커질수록 계산 비용은 늘어나는 대신 지지 반공간 근사의 보수성, 즉 곡선이 KOZ 경계로부터 필요 이상으로 떨어지는 정도가 줄어드는 상충 관계를 정량적으로 확인하고, 이러한 상충 관계가 초기 궤적이 KOZ 경계에 근접하는 (120 deg 위상차 시나리오)[[check scenario]]에서 특히 뚜렷하게 나타남을 보이는 것이다.

두 번째 비교 실험은 차수 $N \in \{6,7,8\}$에 대한 비교이다. 대표 비교 표는 $n_{\mathrm{seg}} = 16$에서 구성하였고, 전체 분할 수에 대해서도 차수에 따른 제어 비용과 계산 시간의 추세를 함께 확인하였다. 차수 비교에는 표현 자유도 변화와 변수 수 변화가 동시에 반영되므로, 결과는 표현 자유도와 변수 수가 함께 변한 효과로 해석한다.)[[표현이 너무 모호하고 명학성이 떨어짐. 전체적으로 재작성할것.]]

### 4.3 후속 direct collocation 파이프라인의 1단계 대체 비교 실험 설정

세 번째 비교 실험은 제안 기법이 후속 고충실도 최적화의 초기화 단계를 대신할 수 있는지를 측정하기 위한 것이다. 이를 위해, 두 단계로 이루어진 direct collocation 파이프라인에서 첫 단계만 서로 다르게 구성한 두 파이프라인을 동일한 조건에서 비교한다. 비교 대상은 다음 두 파이프라인이다. [[nees more explanation. what is this collocation thing? wha is it doing? is it using same scenario? where is this database mentioned below comming from?]]

- **Baseline (두 단계 direct collocation method, DCM)**: Pass 1로 Hermite-Simpson collocation을 사용하여 thrust profile과 phase 구조를 구하고, peak detection 절차로 phase 경계를 결정한 뒤, Pass 2로 multi-phase Legendre-Gauss-Lobatto collocation [8, 10]을 수행한다.
- **Proposed (Bézier-replaces-Pass-1)**: Pass 1을 본 논문의 Bézier SCP optimizer (차수 6, $n_{\mathrm{seg}}=16$)로 대체한다. Peak detection, phase 구조 결정, Pass 2 transcription, 동역학 모형, IPOPT [9] solver 허용오차, 경계조건 처리 방식은 baseline과 동일하게 유지한다.

두 파이프라인의 유일한 차이는 warm-start trajectory와 phase 구조 결정의 출처(Pass 1 Hermite-Simpson vs. Bézier SCP)이다. 따라서 본 실험은 후속 단계를 동일하게 고정한 채 1단계 구성만 바꾼 비교이며, 결과 해석은 1단계 초기화의 대체 가능성에 한정된다.

문제 사례는 (데이터베이스)[which database?]에 저장된 수렴한 궤적 가운데 선택하였다. 두 가지 실험을 수행한다. 첫째는 전이 시간이 비교적 짧고($T_{\mathrm{normed}} \le 0.5$) 시작·도착 궤도의 이심률이 모두 작은($\max(e_0, e_f) \le 0.1$) 10개 사례를 대상으로, 이심률이 제안 기법의 적용 가능 범위에 미치는 영향을 확인하는 실험이다. 둘째는 이심률이 거의 0인($\max(e_0, e_f) \le 0.01$) 수렴 사례 전체(112개)를 대상으로, 전이 시간이 길어질 때 두 파이프라인의 수렴 여부가 어떻게 달라지는지를 확인하는 실험이다. 각 사례에 대해 두 파이프라인의 수렴 여부, 단계별 계산 시간, 최종 비용 차이 $|\Delta \mathrm{cost}|$, 검출된 phase 경계(peak) 수, 그리고 전체 계산 시간 비(기준 파이프라인 총 시간 / 제안 파이프라인 총 시간)를 측정하고, 어느 한 파이프라인이라도 실패한 사례도 함께 기록한다.

---

## 5. 수치 결과

본 절에서는 다음 네 가지를 차례로 확인한다. 첫째, 제안 기법이 대상 궤도전이 문제에서 실현 가능한 궤적을 생성하는지 확인한다. 둘째, 분할 수가 계산 비용과 (안전 여유)[[]] 및 제어 비용에 어떻게 영향을 주는지 측정한다. 셋째, Bézier 차수가 제어 비용과 계산 시간에 미치는 차이를 확인한다. 넷째, 제안 기법이 두 단계 direct collocation 파이프라인의 1단계 초기화를 대체할 수 있는지 확인한다.

### 5.1 대표 궤적과 기본 실현 가능성

먼저 제안 기법이 대상 궤도전이 문제에서 실제로 실현 가능한 궤적을 생성하는지 확인한다. 대표 궤적의 예시는 [그림 4](#fig-trajectory)에 제시하였고, 정량 결과는 표 2에 요약하였다.

<a id="fig-trajectory"></a>
![그림 4. 대표 궤도전이 궤적 예시](../figures/f3_representative_settings.png)
**그림 4 [F4].** Bézier 차수별 궤도전이 궤적.

**표 2 [T2]. 대표 설정에서의 결과 요약**

| 차수 | 제어점 수 | $n_{\mathrm{seg}}$ | 성공 여부 | 안전 여유 (km) | 제어 비용 (m/s²) | 계산 시간 (s) | 반복 횟수 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 7 | 16 | 성공 | 15.57 | 4.652 | 0.090 | 8 |
| 7 | 8 | 16 | 성공 | 15.61 | 4.649 | 0.108 | 8 |
| 8 | 9 | 16 | 성공 | 15.87 | 4.635 | 0.129 | 8 |

> 표의 수치는 `tools/build_tables.py`가 생성하며, 생성에 쓰인 commit과 실행 환경은 `doc/results/paper_tables.md`에 함께 기록된다. 계산 시간은 15회 실행의 최소값으로, 실행 묶음 사이에서 약 ±5% 변동한다. 나머지 열은 재실행해도 모든 자리가 동일하다.

표 2에서 보듯이, 시험한 세 차수 모두에서 실현 가능한 궤적을 얻었다. 세 설정 모두 명제 1의 조건을 만족하는 기준점에서 3.3절의 수렴 조건에 따라 종료하였으며, 반복 한도에 도달한 사례는 없다. 제어 비용은 차수에 대해 단조 감소하고 계산 시간은 단조 증가하여, 대표 설정에서 표현력과 계산 비용 사이의 상충 관계가 관찰된다.

수렴 조건의 허용오차와 연속 만족 횟수는 4.1절에 명시하였으며, 대표 설정은 반복 한도보다 훨씬 적은 반복으로 종료한다.

### 5.2 분할 수에 따른 변화

다음으로 분할 수가 결과에 미치는 영향을 살펴본다. 정량 결과는 표 3에 정리하였고, 요약 추세는 [그림 5](#fig-subdivision-tradeoff)에 함께 제시하였다.

**표 3 [T3]. 분할 수에 대한 비교 실험 결과 ($N=7$)**

| $n_{\mathrm{seg}}$ | 성공 여부 | 안전 여유 (km) | 제어 비용 (m/s²) | 계산 시간 (s) | 반복 횟수 |
|---:|---:|---:|---:|---:|---:|
| 2 | 실패 | 145.00 | 96.824 | 3.479 | 1000 |
| 4 | 성공 | 145.00 | 8.159 | 0.102 | 12 |
| 8 | 성공 | 63.22 | 4.868 | 0.094 | 8 |
| 16 | 성공 | 15.61 | 4.649 | 0.108 | 8 |
| 32 | 성공 | 3.88 | 4.598 | 0.138 | 8 |
| 64 | 성공 | 0.97 | 4.585 | 0.215 | 8 |

> $n_{\mathrm{seg}}=2$의 성공 여부는 명제 1의 인증 기준으로 판정한 것이다. 이 설정의 궤적은 조밀 표본 검사에서는 KOZ를 침범하지 않으나, 반복 한도에 도달할 때까지 인증 조건($h \le 10^{-6}$ km)을 만족하지 못하였다. 생성 출처는 `doc/results/paper_tables.md`에 기록된다.

분할이 지나치게 거친 경우에는 명제 1의 조건을 만족하는 제어점 배치가 경계조건과 양립하기 어려워진다. 명제 1은 충분조건이므로 조건이 성립하면 회피는 보장되지만, 조건이 만족될 수 있는 영역이 좁아지면 하위 문제는 여유 변수가 0이 아닌 해를 내놓게 된다. $n_{\mathrm{seg}}=2$가 이 경우로, 반복 한도에 도달할 때까지 인증 조건을 만족하지 못하였다. 이때 얻은 궤적이 실제로 KOZ를 침범하지는 않았으나, 명제 1이 보장하는 것은 인증이 성립할 때의 회피이므로 이 궤적에는 그 보장이 적용되지 않는다. 제어 비용과 목적함수 값 또한 다른 설정보다 한 자릿수 이상 나쁘다. 즉 이 경우의 실패는 충분조건의 타당성에서 오는 것이 아니라 그 조건이 정의하는 영역의 크기에서 온다.

분할 수가 충분히 크면 안전 여유와 제어 비용이 분할 수에 대해 단조 감소한다. 가장 큰 개선 폭은 분할 수가 작은 쪽에서 나타나고, 분할 수가 커지면 개선이 미미해진다. 계산 시간은 반복 횟수가 같아지는 $n_{\mathrm{seg}} \ge 8$ 구간에서 분할 수에 따라 단조 증가하므로, 보수성 감소와 계산 비용 사이의 상충 관계가 존재한다.

여기서 보수성(conservatism)이란, 지지 반공간 구성이 부과하는 안전 여유와 곡선의 실제 최소 접근 거리 사이의 차이를 가리킨다. 이 차이는 제어점의 볼록 껍질이 곡선 자체보다 넓은 영역을 차지하기 때문에 발생한다. 분할 수가 증가하면 각 분할구간이 짧아지고 제어점이 곡선에 더 가까워지므로, 지지 반공간 제약이 실제 곡선-KOZ 거리를 보다 정밀하게 반영하게 된다. 3.1절에서는 이 차이가 분할구간이 평면을 따라 뻗는 폭 $L_{\mathrm{seg}}$에 대해 약 $L_{\mathrm{seg}}^2/2R_{\mathrm{KOZ}}$이고, 따라서 분할 수에 대해 $n_{\mathrm{seg}}^{-2}$로 줄어든다고 예측하였다. 표 3의 안전 여유 열은 이 보수성의 직접적인 척도이며, KOZ 제약이 작동하는 구간에서 분할 수를 두 배로 늘릴 때마다 안전 여유가 약 4분의 1로 줄어드는 경향이 이 예측과 일치한다.

<a id="fig-subdivision-tradeoff"></a>
![그림 5. 분할 수에 따른 계산 시간 및 결과 추세](../figures/f4_subdivision_tradeoff_N7.png)
**그림 5 [F5].** 120 deg 위상차 시나리오($N=7$)에서 분할 수에 따른 안전 여유·제어 비용·계산 시간의 변화.[[n_seg==2 infeasible case임을 별도로 표시]]

### 5.3 Bézier 차수에 따른 변화

다음으로 Bézier 차수 변화가 결과에 미치는 영향을 살펴본다. 차수가 높아지면 표현 자유도는 커지지만, 변수 수와 계산량도 함께 증가한다. 정량 결과는 표 4에 정리하였고, 전체 추세는 [그림 6](#fig-multi-order-trend)에 요약하였다.

**표 4 [T4]. 차수에 대한 비교 실험 결과 ($n_{\mathrm{seg}}=16$)**

| 차수 | 제어점 수 | $n_{\mathrm{seg}}$ | 성공 여부 | 안전 여유 (km) | 제어 비용 (m/s²) | 계산 시간 (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 7 | 16 | 성공 | 15.57 | 4.652 | 0.090 |
| 7 | 8 | 16 | 성공 | 15.61 | 4.649 | 0.108 |
| 8 | 9 | 16 | 성공 | 15.87 | 4.635 | 0.129 |

> 표 2와 같은 설정에서 측정한 값이며, 생성 출처는 `doc/results/paper_tables.md`에 기록된다.

표 4는 차수 변화에 따른 성능 지표를 비교한 것이다. 세 차수 모두 실현 가능성을 유지하며 안전 여유도 서로 비슷한 수준이다. 제어 비용은 차수에 대해 단조 감소하고 계산 시간은 단조 증가한다. 차수 간 목적함수 값의 상대 차이는 계산 시간의 상대 차이보다 작으므로, 차수 변화의 효과는 표현 자유도 향상에 따른 목적함수 개선과 계산 비용 증가 사이의 상충 관계로 해석하는 것이 적절하다.

<a id="fig-multi-order-trend"></a>
![그림 6. 차수별 성능 추세 요약](../figures/f5_multi_order_tradeoff_N678.png)
**그림 6 [F6].** 120 deg 위상차 시나리오에서 $N=6,7,8$ 차수에 대한 제어 비용(좌)과 계산 시간(우)의 추세.

다음 절에서는 후속 활용 가능성에 관한 비교 결과를 제시한다.

### 5.4 후속 direct collocation 파이프라인의 1단계 대체 비교

(두 비교 대상은 모두 두 단계로 이루어진 direct collocation 파이프라인이며, 1단계에서 얻은 궤적과 phase 구조를 2단계의 초기값으로 사용한다. 기준 파이프라인은 1단계에서 Hermite-Simpson collocation을, 제안 파이프라인은 1단계에서 본 논문의 Bézier SCP를 사용하며, 두 파이프라인은 동일한 다구간 Legendre-Gauss-Lobatto(LGL) 2단계 solver와 동역학 모델, 허용오차, 경계조건을 공유한다. 따라서 두 파이프라인의 차이는 1단계 초기화와 phase 구조 결정의 출처뿐이며, 본 비교는 동일한 절차 안에서 1단계를 Bézier SCP로 대체했을 때 최종 해와 전체 계산 시간이 어떻게 달라지는지를 확인한다. 제안 파이프라인이 작동할 수 있는 영역의 경계 또한 이하에서 함께 제시한다.)[[should be checked if agent with no context can understand this prose]]

> [그림 TODO] 두 파이프라인(기준: Hermite-Simpson 1단계 + LGL 2단계 / 제안: Bézier SCP 1단계 + LGL 2단계)의 단계 구조를 비교하는 개념도 필요 (교수 요청 #56).

**표 5 [T5]. 1단계 대체 비교 결과 (작동 영역 경계 탐색에서 두 파이프라인이 모두 수렴한 7개 원궤도 사례).**

| Case | $T_{\mathrm{normed}}$ | 초기 고도 (km) | $\Delta a$ (km) | $\Delta i$ (deg) | Baseline (s) | Bézier (s) | Pass 2 (s) | Proposed total (s) | 전체 시간 비 | $\|\Delta \mathrm{cost}\|$ | Peaks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 2   | 0.280 | 400 | −2.49  | 13.85 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 4   | 0.280 | 400 | −2.49  | 3.00  | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 6   | 0.280 | 400 | −2.49  | 13.85 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 20  | 1.920 | 400 | −78.47 | 0.83  | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 39  | 0.510 | 400 | 1262.69 | 9.39 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 114 | 2.052 | 400 | 1224.14 | 4.66 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 126 | 0.500 | 400 | 1137.93 | 4.14 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

> 표의 측정 수치(단계별 계산 시간·전체 계산 시간 비·비용 차이·peak 수)는 optimizer 루프 재작업 이후 재생성 예정 (TODO). 사례는 데이터베이스에서 두 파이프라인이 모두 수렴한 원궤도 사례를 추린 것이다.

표 5는 동일한 2단계 절차 하에서 두 파이프라인이 모두 수렴한 7개 원궤도 사례를 정리한 것이다. 이들은 앞서 기술한 데이터베이스에서 선별한 것으로, 정규화 전이 시간이 짧은 전이부터 여러 바퀴를 도는(multi-revolution) 전이까지 걸쳐 있다. 대부분의 사례에서 최종 비용은 기준 파이프라인과 제안 파이프라인이 기계 정밀도 수준에서 일치한다. 다만 한 사례는 예외로, Bézier 1단계가 기준보다 phase 경계를 하나 더 검출하여 2단계가 인접한 다른 국소해로 수렴하였고, 그 결과 작은 비용 차이가 나타났다. 따라서 제안 기법은 1단계의 완전한 교체라기보다는, 두 파이프라인의 phase 구조가 일치하는 경우에 성립하는 조건부 교체로 해석해야 한다.

전체 계산 시간 비(기준 파이프라인 시간 / 제안 파이프라인 총 시간)는 일부 사례에서 1보다 커서 제안 파이프라인이 더 빨랐고, 일부 사례에서는 1보다 작았다. Bézier 1단계 자체의 계산 시간은 모든 사례에서 일관되게 작으므로, 전체 계산 시간 비는 주로 2단계 계산 시간의 변동이 좌우한다. 특히 여러 바퀴를 도는 전이 사례에서는 제안 파이프라인의 2단계 계산 시간이 기준 파이프라인 전체보다 오히려 큰 경우가 있는데, 이는 Bézier 초기값이 2단계의 국소해와 부합하지 않을 수 있음을 보여주며, 1단계에서 절약한 시간이 항상 전체 계산 시간 개선으로 이어지지는 않음을 시사한다.

1단계 대체가 전체 계산 시간에 미치는 영향은 [그림 7](#fig-downstream-speedup)에 사례별로 정리하였다. 이 그림은 1단계 대체의 이득이 사례에 따라 달라진다는 점과, 그 변동이 1단계가 아니라 2단계 계산 시간에서 비롯된다는 점을 보이기 위한 것이다.

<a id="fig-downstream-speedup"></a>
![그림 7. 1단계 대체의 사례별 전체 계산 시간 비 및 구성](../figures/f6_downstream_speedup.png)
**그림 7 [F7].** 1단계 대체가 전체 계산 시간에 미치는 영향을 보이기 위한 사례별 비교. 좌측은 전체 계산 시간 비, 우측은 단계별 계산 시간 구성이다.

제안 파이프라인의 작동 영역은 두 가지 경계로 나뉜다. 첫째는 이심률 경계이다. 이심률 실험에서 시작·도착 궤도가 원궤도인 사례에서는 Bézier 1단계가 모두 실현 가능하였으나, 이심률이 작은 타원 궤도 사례에서는 모두 실현 불가능하였다. 모든 제어점이 출발 궤도 위에 놓여 있어도 곡선의 일부가 KOZ 안쪽으로 침범하기 때문이다. 둘째는 전이 시간 경계이다. 전이 시간이 길어질수록 Bézier 1단계가 실현 가능하더라도 2단계가 그 초기값으로부터 수렴하는 비율이 빠르게 낮아진다. 즉 이심률 축에서는 원궤도 여부가 뚜렷한 경계로 작용하고, 전이 시간 축에서는 2단계의 수렴 여부가 점진적인 병목으로 작용한다.

이러한 결과는 두 가지 한계 안에서 해석해야 한다. 첫째, 표 5의 결과는 동일한 2단계 절차와 작동 영역 안에서만 성립하는 초기화 교체 가능성에 대한 것이며, 제안 파이프라인 자체가 하나의 direct collocation 파이프라인이고 1단계 구현만 기준과 다르므로, 이는 direct collocation 대비 방법론적 우월성 주장과는 구분된다. 둘째, 작동 영역이 좁고 사례 수가 제한적이므로, 측정된 계산 시간 비를 2단계가 수렴하지 못하는 더 넓은 영역으로 외삽할 근거는 아직 없다. 요약하면, 제시한 가정 아래에서 Bézier 1단계는 초기화 단계의 제한적인 교체 수단이며, 후속 solver 일반을 가속하는 수단은 아니다.

---

## 6. 결론
--정량적 수치값을 직접 제시하는것은 지양하고, 정성적으로 서술할 것.

본 논문에서는 제어점 공간에서 직접 작동하는 Bézier 기반 궤적 초기화 기법을 제안하였다. 제안 기법은 각 분할구간의 제어점에 지지 반공간 제약을 부과함으로써, 구형 KOZ 제약의 연속시간 만족을 보수적으로 보장한다. 또한 전체 문제를 순차 볼록화 알고리즘 안에서 일련의 볼록 QP로 풀 수 있도록 구성하였다.

단순화된 궤도전이 문제에 대한 실험 결과, 제안 기법은 대표 차수 설정에서 실현 가능한 궤적을 생성할 수 있었다(다만 분할이 지나치게 거칠면 충분조건을 만족하는 영역이 좁아져 실현 불가능한 해가 나타날 수 있다). 분할 수 실험에서는 충분히 분할된 영역에서 안전 여유와 제어 비용이 분할 수에 대해 단조 감소하여, 분할 수 증가가 보수성을 실질적으로 줄임을 확인하였다. 차수 실험에서는 제어 비용이 차수에 대해 단조 감소하지만 계산 시간은 단조 증가하여 표현력-계산비용 상충 관계가 관찰되었다.

후속 활용 가능성에 관해서는, 동일한 두 단계 direct collocation 파이프라인에서 1단계 초기화를 Bézier SCP로 대체하는 비교를 수행하였다. 두 파이프라인이 모두 수렴한 원궤도 사례에서 최종 비용은 대부분 기계 정밀도 수준으로 보존되었고 일부 사례에서는 전체 계산 시간이 감소하였으나, 한 사례에서는 phase 구조 검출의 차이로 다른 국소해에 수렴하였다. 이 결과의 적용 영역은 Bézier 1단계가 실현 가능하고 후속 2단계가 수렴하는 원궤도 사례에 한정된다.

결론적으로, 제안 기법은 제어점 공간에서 연속시간 KOZ 제약을 구성하고 이를 SCvx 기반 최적화와 결합하는 하나의 정식화를 제공하며, 정의된 작동 영역 내에서 두 단계 direct collocation 파이프라인의 1단계 초기화를 대체하는 수단으로도 사용될 수 있다. 다만 본 논문의 연속시간 제약 만족 보장은 구형 KOZ와 고정 전이 시간 설정에 한정되고, 실험적 근거가 단일 시연 문제와 좁은 작동 영역에 기반한다는 한계가 있다. 향후 과제로는 타원 궤도에서의 Bézier 실현 가능성 확장(예: 여러 호(arc)로 분할한 Bézier 곡선), 다양한 전이 시간 영역에서의 2단계 수렴 특성 개선, 그리고 여러 문제 설정으로의 실험 확대와 시간 최적화 확장을 고려할 수 있다.

---

## 참고문헌

[1] Lee, S., and Kim, Y., "Optimal Output Trajectory Shaping Using Bézier Curves," *Journal of Guidance, Control, and Dynamics*, Vol. 44, No. 5, 2021, pp. 1027–1035. doi:10.2514/1.G005887

[2] Lee, S., "A Shape-based Approach Suited for Short-Duration Orbit Transfer Trajectory Design," *11th European Conference for AeroSpace Sciences (EUCASS)*, Rome, Italy, July 2025.

[3] Betts, J. T., "Survey of Numerical Methods for Trajectory Optimization," *Journal of Guidance, Control, and Dynamics*, Vol. 21, No. 2, 1998, pp. 193–207. doi:10.2514/2.4231

[4] Hargraves, C. R., and Paris, S. W., "Direct Trajectory Optimization Using Nonlinear Programming and Collocation," *Journal of Guidance, Control, and Dynamics*, Vol. 10, No. 4, 1987, pp. 338–342. doi:10.2514/3.20223

[5] Açıkmeşe, B., Carson, J. M., and Blackmore, L., "Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem," *IEEE Transactions on Control Systems Technology*, Vol. 21, No. 6, 2013, pp. 2104–2113. doi:10.1109/TCST.2012.2237346

[6] Mao, Y., Dueri, D., Szmuk, M., and Açıkmeşe, B., "Successive Convexification of Non-Convex Optimal Control Problems with State Constraints," *IFAC-PapersOnLine*, Vol. 50, No. 1, 2017, pp. 4063–4069. doi:10.1016/j.ifacol.2017.08.789

[7] Malyuta, D., Reynolds, T. P., Szmuk, M., Lew, T., Bonalli, R., Pavone, M., and Açıkmeşe, B., "Convex Optimization for Trajectory Generation: A Tutorial on Generating Dynamically Feasible Trajectories Reliably and Efficiently," *IEEE Control Systems Magazine*, Vol. 42, No. 5, 2022, pp. 40–113. doi:10.1109/MCS.2022.3187542

[8] Patterson, M. A., and Rao, A. V., "GPOPS-II: A MATLAB Software for Solving Multiple-Phase Optimal Control Problems Using hp-Adaptive Gaussian Quadrature Collocation Methods and Sparse Nonlinear Programming," *ACM Transactions on Mathematical Software*, Vol. 41, No. 1, 2014, pp. 1–37. doi:10.1145/2558904

[9] Wächter, A., and Biegler, L. T., "On the Implementation of an Interior-Point Filter Line-Search Algorithm for Large-Scale Nonlinear Programming," *Mathematical Programming*, Vol. 106, No. 1, 2006, pp. 25–57. doi:10.1007/s10107-004-0559-y

[10] Herman, A. L., and Conway, B. A., "Direct Optimization Using Collocation Based on High-Order Gauss-Lobatto Quadrature Rules," *Journal of Guidance, Control, and Dynamics*, Vol. 19, No. 3, 1996, pp. 592–599. doi:10.2514/3.21662

[11] Dueri, D., Mao, Y., Mian, Z., Ding, J., and Açıkmeşe, B., "Trajectory Optimization with Inter-Sample Obstacle Avoidance via Successive Convexification," *2017 IEEE 56th Annual Conference on Decision and Control (CDC)*, Melbourne, Australia, 2017, pp. 1150–1156. doi:10.1109/CDC.2017.8263811

[12] Elango, P., Luo, D., Kamath, A. G., Uzun, S., Kim, T., and Açıkmeşe, B., "Successive Convexification for Trajectory Optimization with Continuous-Time Constraint Satisfaction," arXiv:2404.16826, 2024. doi:10.48550/arXiv.2404.16826

[13] Han, S. P., and Mangasarian, O. L., "Exact Penalty Functions in Nonlinear Programming," *Mathematical Programming*, Vol. 17, No. 1, 1979, pp. 251–269. doi:10.1007/BF01588250

[14] Nocedal, J., and Wright, S. J., *Numerical Optimization*, 2nd ed., Springer, New York, 2006, Theorem 17.3. doi:10.1007/978-0-387-40065-5
