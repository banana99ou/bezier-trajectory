# 세그먼트별 컨벡스화에 기반한 Bézier 궤적 초기화 기법

부제: Orbital transfer 예제에 대한 적용

---

**초록**

본 논문에서는 구형 Keep-Out Zone(KOZ)을 연속적으로 회피하는 Bézier 기반 궤적 초기화 프레임워크를 제안한다. 제안 방법은 전적으로 제어점(control point) 공간에서 작동하며, 미분 연산자, 분할(subdivision) 행렬, 경계조건, KOZ 제약을 모두 제어점에 대한 선형 연산으로 표현한다. 구형 KOZ 회피는 De Casteljau 분할로 곡선을 여러 분할구간(sub-arc)으로 나눈 뒤, 각 분할구간의 제어다각형에 지지 반공간(supporting half-space) 제약을 부과하는 방식으로 보수적으로 처리한다. Bézier 곡선의 볼록 껍질(convex hull) 성질에 따르면 해당 분할구간의 모든 제어점이 반공간 제약을 만족할 때 곡선 전체도 KOZ 바깥에 놓이게 된다. 최종 최적화는 순차 컨벡스화(successive convexification) 절차 안에서 일련의 convex QP를 푸는 방식으로 수행된다.

제안된 기법을 단순화된 orbital transfer 문제에 적용하여 검증하였다. 분할 수에 대한 비교 실험에서는, 분할이 정밀해질수록 안전 여유가 약 145 km에서 약 1 km로 단조 감소하고 delta-v 대리 지표가 약 3.4배 개선되어, 보수적 KOZ 근사의 정밀화가 목적함수를 실질적으로 개선함을 확인하였다. Bézier 차수에 대한 비교 실험에서는 $N=8$이 가장 낮은 목적함수 값을 주지만 계산 시간도 가장 크고, $N=7$은 가장 빠른 수렴을 보이는 tradeoff가 관찰되었다. 제안한 프레임워크는 후속 solver에 전달할 수 있는 매끄럽고 연속적으로 안전한 warm start 초기값 생성에도 intended use를 가지지만, 이에 대한 matched downstream comparison은 현재 draft에서 placeholder로 남겨 둔다.

---

## 1. 서론

제약이 있는 궤적 최적화 문제는 항공우주, 로봇공학, autonomous systems 등 여러 분야에서 반복적으로 등장한다. 이때 중요한 요구 가운데 하나는 궤적이 특정 금지 영역을 경로 전체에 걸쳐 지속적으로 회피해야 한다는 점이다. 그러나 일반적인 direct transcription 또는 direct collocation 방식에서는 제약식이 주로 이산화된 지점에서만 강제되므로, 노드 사이 구간의 안전은 별도로 확인해야 하는 경우가 많다. 따라서 이산화된 지점만이 아니라 연속 구간 전체에서 안전을 다룰 수 있는 표현과 제약 방식이 필요하다.

또 다른 실용적 문제는 초기값의 품질이다. 많은 후속 solver는 초기값에 민감하며, 초기값이 좋지 않으면 제약을 만족하지 않는 해로 수렴하거나 반복 횟수가 크게 증가하거나 품질이 낮은 국소해에 머무를 수 있다. 이런 점에서 후속 고충실도 최적화에 앞서 매끄럽고 제약을 만족하는 초기 궤적을 생성하는 절차는 그 자체로 의미가 있다.

본 논문은 이러한 문제를 해결하기 위해 Bézier 곡선을 이용한 궤적 초기화 기법을 제안한다. 제안한 기법의 핵심은 모든 계산을 제어점 공간에서 수행한다는 점이다. 곡선의 미분, 분할, 경계조건, KOZ 제약이 모두 제어점에 대한 선형 연산으로 정리되므로, 계산 구조가 비교적 단순하고 해석도 명확하다. 특히 구형 KOZ에 대해서는 분할된 각 분할구간에 지지 반공간을 부여하고, 그 반공간 안에 제어다각형이 놓이도록 함으로써 연속 안전을 보수적으로 확보한다.

본 논문의 기여는 다음과 같이 정리할 수 있다.

1. Bézier parameterization을 기반으로 제어점 공간에서 직접 작동하는 궤적 초기화 정식화를 제시하여 제약 구성과 계산 구조를 단순화한다.
2. De Casteljau 분할과 지지 반공간을 이용하여 구형 KOZ를 연속적으로 회피하도록 하는 보수적 제약 구성 방식을 제안한다.
3. 논컨벡스 KOZ 제약을 각 SCP 반복에서 재선형화하면서 일련의 convex QP를 푸는 SCP 기반 절차를 정리한다.
4. 단순화된 orbital transfer 문제에서 분할 수와 Bézier 차수에 대한 비교 실험을 수행하여 계산 비용과 성능의 관계를 분석한다.

위의 네 가지 기여 외에, 후속 direct collocation warm-start 비교는 본 프레임워크의 intended-use를 평가하는 중요한 후속 실험 항목으로 유지하되 현재 draft에서는 placeholder로만 남겨 둔다.

이후 구성은 다음과 같다. 2절에서는 관련 연구와의 관계를 정리하고, 3절에서는 문제 설정과 표기법을 소개한다. 4절에서는 제안 기법의 수학적 구성과 알고리즘을 설명한다. 5절에서는 실험 설정을 기술하고, 6절에서는 결과를 제시한다. 7절에서는 한계와 해석 범위를 정리하며, 8절에서 결론을 맺는다.

---

## 2. 관련 연구 및 위치 설정

본 절에서는 제안 기법을 세 가지 맥락에서 살펴본다. 첫째는 direct transcription 및 direct collocation 계열의 궤적 최적화 방법, 둘째는 obstacle avoidance를 위한 컨벡스화 또는 보수적 안전 처리 방식, 셋째는 후속 최적화를 위한 초기화와 warm start 생성 방법이다.

### 2.1 Direct transcription 및 direct collocation과의 관계

Direct transcription과 direct collocation은 제약이 있는 궤적 최적화에서 가장 널리 쓰이는 방법들이다. 이들 방법은 궤적을 여러 노드에서의 상태와 입력 변수로 이산화하고, dynamics를 equality constraint로 부과한 뒤, 큰 규모의 nonlinear program을 푼다. 다양한 문제에 적용 가능하고 solver 생태계도 잘 갖추어져 있다는 장점이 있다.

다만 점별 이산화에 기반한 이러한 정식화에서는 연속 구간 전체의 안전을 직접 다루기 어렵고, 초기값의 품질 또한 수렴 거동에 큰 영향을 줄 수 있다. 본 논문은 이러한 지점에서 제어점 공간 정식화와 보수적 연속 안전 제약 구성을 제시하며, direct collocation은 제안 프레임워크가 연계될 수 있는 중요한 downstream 비교 대상 가운데 하나이다.

### 2.2 보수적 장애물 처리 및 컨벡스화

연속적인 장애물 회피에서는 이산화된 노드에서의 제약 만족만으로 노드 사이 구간의 안전을 보장하기 어렵다. 이를 다루기 위한 여러 접근 가운데 일부는 특정 문제 클래스에 대해 무손실 컨벡스화를 사용하고, 또 다른 일부는 순차 컨벡스화로 논컨벡스 제약을 반복적으로 선형화한다.

본 논문은 순차 컨벡스화의 틀을 사용하되, 점별 상태 제약이 아니라 Bézier 분할구간의 제어점에 제약을 가하는 형태로 적용한다. 구체적으로는 분할된 각 분할구간에 대해 구형 KOZ를 지지하는 반공간을 만들고, 제어점이 그 반공간 안에 위치하도록 한다. 이 방식은 보수적인 회피를 제공하며, 볼록 껍질 성질을 이용하여 분할구간 전체의 연속 안전을 논할 수 있다는 장점이 있다.

### 2.3 초기화 및 warm start 생성

초기값의 품질이 비선형 궤적 최적화의 수렴 거동에 큰 영향을 준다는 점은 잘 알려져 있다. 실제로는 직선 보간, 경험적 형상화, 단순 모델 해, 데이터베이스 기반 초기화 등 다양한 방식이 사용된다. 그러나 단순한 초기화는 연속 안전을 반영하지 못하는 경우가 많다.

본 논문의 제안 기법은 이러한 초기화 방법들과도 연결될 수 있다. 즉, 비교적 저차원인 control point 공간에서 
매끄럽고 안전한 궤적을 먼저 만든 다음, 이를 후속 solver에 초기값으로 제공하는 방식이다. 여기서 장점은 초기화 문제의 차원을 작게 유지하면서도, 기하학적인 안전 제약을 명시적으로 다룰 수 있다는 점이다.

---

## 3. 문제 설정 및 표기법

### 3.1 궤적 표현과 결정 변수
궤적은 정규화된 매개변수 $\tau \in [0,1]$ 위에서 정의된 차수 $N$의 Bézier 곡선으로 표현한다.

$$
\mathbf{r}(\tau) = \sum_{i=0}^{N} B_i^{N}(\tau)\,\mathbf{p}_i
$$

여기서 $B_i^N$은 Bernstein 기저다항식이고, $\mathbf{p}_i \in \mathbb{R}^3$은 $i$번째 제어점이다. 제어점을 행렬로 모으면

$$
P = [\mathbf{p}_0^{\mathsf{T}}, \mathbf{p}_1^{\mathsf{T}}, \ldots, \mathbf{p}_N^{\mathsf{T}}]^{\mathsf{T}} \in \mathbb{R}^{(N+1)\times 3}
$$

가 되고, 이를 하나의 벡터로 쌓으면

$$
\mathbf{x} = [\mathbf{p}_0^{\mathsf{T}}, \mathbf{p}_1^{\mathsf{T}}, \ldots, \mathbf{p}_N^{\mathsf{T}}]^{\mathsf{T}} \in \mathbb{R}^{3(N+1)}
$$

를 얻는다. 본 논문에서 최적화의 결정 변수는 $\mathbf{x}$이며, 이후의 미분 연산, 분할, KOZ 제약은 모두 이 벡터에 대한 선형 연산으로 표현된다.

본 연구에서는 편의 상 전이 시간 $T$를 고정한다. 물리 시간 $t$와 정규화 매개변수 $\tau$의 관계는

$$
t = T\tau
$$

로 둔다. 따라서 실제 속도와 가속도는 $\tau$에 대한 미분에 각각 $1/T$, $1/T^2$를 곱한 형태로 얻어진다.

### 3.2 미분 연산자와 경계조건

Bézier 곡선의 미분 구조는 표준 차분 행렬 $D_N$으로 정리할 수 있다. 여기서 $D_N = N[d_{ij}] \in \mathbb{R}^{N\times(N+1)}$이며, $d_{i,i}=-1$, $d_{i,i+1}=1$, 그 밖의 항은 0이다.


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

또한 미분된 제어점을 다시 원래 차수의 basis 위에서 표현하기 위해 degree elevation matrix $E_M$을 사용한다. 이를 이용하면 차수 보존형 속도 및 가속도 연산자를

$$
L_{1,N} = E_{N-1}D_N, \qquad L_{2,N} = E_{N-1}D_N E_{N-1}D_N
$$

로 쓸 수 있고, 대응하는 제어점은

$$
P^{(1)} = L_{1,N}P, \qquad P^{(2)} = L_{2,N}P
$$

이다.

물리적인 속도와 가속도는 다음과 같다.

$$
\dot{\mathbf{r}}(t) = \frac{1}{T}\frac{d\mathbf{r}}{d\tau}, \qquad \ddot{\mathbf{r}}(t) = \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2}
$$

끝점 위치는 첫 번째와 마지막 제어점을 고정하여 부과한다. 끝점 속도는

$$
\dot{\mathbf{r}}(0) = \frac{N}{T}(\mathbf{p}_1-\mathbf{p}_0), \qquad \dot{\mathbf{r}}(T) = \frac{N}{T}(\mathbf{p}_N-\mathbf{p}_{N-1})
$$

로 주어지며, 필요할 경우 끝점 가속도도 같은 방식으로 선형 제약식으로 표현할 수 있다.

### 3.3 Gram matrix와 quadratic form

Bernstein basis의 Gram matrix는 닫힌 형태로 계산할 수 있으며,

$$
[G_N]_{ij} = \frac{\binom{N}{i}\binom{N}{j}}{\binom{2N}{i+j}(2N+1)}, \qquad i,j=0,\ldots,N
$$

로 쓴다. 이를 가속도 연산자와 결합하면

$$
\tilde G_N = L_{2,N}^\top G_N L_{2,N}
$$

을 얻고,

$$
\int_0^1 \left\|\frac{d^2\mathbf{r}}{d\tau^2}\right\|_2^2 d\tau = \mathrm{tr}(P^\top \tilde G_N P) = \mathbf{x}^\top (\tilde G_N \otimes I_3)\mathbf{x}
$$

의 정확한 quadratic form으로 정리된다.

---

## 4. 제안 기법의 구성

### 4.1 분할과 지지 반공간을 이용한 구형 KOZ 처리

구형 KOZ를 다음과 같이 정의한다.

$$
\mathcal{K} = \left\{\mathbf{r}\in\mathbb{R}^3 : \|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le r_e \right\}
$$

여기서 $\mathbf{c}_{\mathrm{KOZ}}$는 KOZ의 중심이며, 현재 실험에서는 원점 중심의 경우를 사용한다.

제안 기법에서는 곡선을 $n_{\mathrm{seg}}$개의 분할구간으로 등분하기 위해 De Casteljau 분할 행렬 $S^{(s)}$를 사용한다. 그러면 $s$번째 분할구간의 제어다각형은

$$
P^{(s)} = S^{(s)}P
$$

가 된다. 이 분할구간의 제어점을 $\mathbf{q}^{(s)}_0, \ldots, \mathbf{q}^{(s)}_N$이라 하면, 대표점으로는 제어다각형의 중심점

$$
\mathbf{c}^{(s)} = \frac{1}{N+1}\sum_{k=0}^{N}\mathbf{q}^{(s)}_k
$$

를 사용한다.

이때 외향 법선은

$$
\mathbf{n}^{(s)} = \frac{\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}}{\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\|_2}
$$

로 정의하며, $\mathbf{c}^{(s)} = \mathbf{c}_{\mathrm{KOZ}}$인 경우에는 법선 방향이 정의되지 않으므로 제외한다. 이에 따라 구의 지지 반공간은

$$
\mathcal{H}^{(s)} = \left\{\mathbf{r} : (\mathbf{n}^{(s)})^\top \mathbf{r} \ge (\mathbf{n}^{(s)})^\top \mathbf{c}_{\mathrm{KOZ}} + r_e \right\}
$$

로 쓸 수 있다.

본 논문에서는 각 분할구간의 모든 제어점이 이 반공간 안에 놓이도록 다음 부등식을 부과한다.

$$
(\mathbf{n}^{(s)})^\top \mathbf{q}^{(s)}_k \ge (\mathbf{n}^{(s)})^\top \mathbf{c}_{\mathrm{KOZ}} + r_e, \qquad k=0,\ldots,N
$$

각 $\mathbf{q}^{(s)}_k$는 원래 제어점의 선형결합이므로 이 제약식은 결정 변수 $\mathbf{x}$에 대해 선형이다. 반공간은 각 SCP 반복에서 현재 해를 기준으로 다시 구성되며, 따라서 현재 해 주변에서 작동하는 보수적이고 국소적인 회피 제약조건으로 이해할 수 있다. 어떤 분할구간의 모든 제어점이 $\mathcal{H}^{(s)}$ 안에 있으면, Bézier 곡선의 볼록 껍질 성질에 의해 그 분할구간의 곡선 전체도 $\mathcal{H}^{(s)}$ 안에 놓이고, 따라서 $\mathcal{K}$ 바깥에 놓인다.

<a id="fig-subdivision"></a>
![그림 1. 분할과 지지 반공간을 이용한 연속 안전 제약의 개념도](../figures/f1_koz_linearization.png)
**그림 1.** 분할된 하나의 분할구간에 대한 구형 KOZ 선형화 개념도. 분할구간 제어다각형의 중심점에서 구성한 지지 반공간 제약을 해당 분할구간의 모든 제어점에 부과함으로써 구형 KOZ를 보수적으로 배제한다.

### 4.2 제어 가속도 대리 목적함수

본 논문에서는 affine하게 선형화한 중력장 아래에서 제어 가속도를 근사적으로 반영하는 대리 목적함수를 사용한다. 우선

$$
\mathbf{u}(t) = \ddot{\mathbf{r}}(t) - \mathbf{g}(\mathbf{r}(t))
$$

를 정의한다. 여기서 $\mathbf{g}$는 궤도 중력 모형이며, 현재 실험에서는 two-body 항과 J2 perturbation 항을 포함한다. 제안 기법은 곡선 전체에 걸쳐 dynamics를 정확히 강제하지 않고, 대표 분할구간 위치에서 중력장을 affine하게 선형화하여 목적함수를 구성한다.

목적함수 구성에는 KOZ 분할 수와 별도로 $n_{\mathrm{lin}}$개의 선형화용 분할구간을 사용한다. 대응하는 중심점 행을 이용하면 대표 위치와 기하학적 가속도를 모두 $\mathbf{x}$의 선형 함수로 쓸 수 있다. 반복 $k$에서 중력장은 현재 기준 위치 $\mathbf{r}_i(\mathbf{x}^{(k)})$에서 affine linearization으로 근사된다.

그 결과 각 샘플에 대해 선형화된 제어 가속도 residual

$$
\boldsymbol{\rho}_i^{(k)}(\mathbf{x}) = A_i\mathbf{x} - \left(B_i^{(k)}\mathbf{x} + \mathbf{c}_i^{(k)}\right)
$$

을 정의할 수 있다. 이 residual에 대해 IRLS(iteratively reweighted least squares) 방식의 quadratic majorization을 구성하면 다음 목적함수를 얻는다.

$$
J_{\mathrm{dv}}^{(k)}(\mathbf{x}) = \sum_{i=1}^{n_{\mathrm{lin}}}\omega_i^{(k)}\left\|\boldsymbol{\rho}_i^{(k)}(\mathbf{x})\right\|_2^2
$$

여기서 가중치 $\omega_i^{(k)}$는 현재 반복점의 residual 크기에 따라 갱신된다. 이 목적함수는 반복마다 제어 가속도 residual을 안정적으로 줄이도록 구성된 surrogate이며, 본 논문에서는 연속 안전을 갖는 초기 궤적 생성을 위한 목적함수로 사용한다.

### 4.3 Convex subproblem과 SCP 알고리즘

SCP 반복 $k$에서 KOZ 지지 반공간과 중력 선형화를 모두 고정하면, 내부 문제는 다음과 같은 convex QP가 된다.

$$
\min_{\mathbf{x}} \ \frac{1}{2}\mathbf{x}^\top H^{(k)}\mathbf{x} + (\mathbf{f}^{(k)})^\top \mathbf{x}
$$

제약조건은 다음과 같다.

$$
A_{\mathrm{KOZ}}^{(k)}\mathbf{x} \ge \mathbf{b}_{\mathrm{KOZ}}^{(k)}, \qquad A_{\mathrm{bc}}\mathbf{x} = \mathbf{b}_{\mathrm{bc}}, \qquad \boldsymbol{\ell} \le \mathbf{x} \le \mathbf{u}
$$

여기서 $A_{\mathrm{KOZ}}^{(k)}\mathbf{x} \ge \mathbf{b}_{\mathrm{KOZ}}^{(k)}$는 분할로부터 얻은 KOZ 제약, $A_{\mathrm{bc}}\mathbf{x}= \mathbf{b}_{\mathrm{bc}}$는 경계조건, $\boldsymbol{\ell}$과 $\mathbf{u}$는 경계점 위치를 포함한 bound이다.

필요한 경우 현재 iterate 주변에 다음과 같은 proximal regularization도 추가한다.

$$
\frac{\lambda}{2}\|\mathbf{x}-\mathbf{x}^{(k)}\|_2^2
$$

이는 문제의 convexity를 유지한 채 SCP 반복의 안정성을 높이는 역할을 한다.

전체 SCP 절차는 다음 순서로 수행된다.

1. 현재 제어다각형을 분할한다.
2. 각 분할구간에 대해 지지 반공간을 다시 계산한다.
3. 중력 선형화와 IRLS 가중치를 갱신한다.
4. 대응하는 convex QP를 푼다.
5. 새 제어점을 다음 SCP 반복의 기준점으로 사용한다.

수렴 판정은

$$
\|P^{(k+1)} - P^{(k)}\|_F < \mathrm{tol}
$$

을 만족할 때 또는 최대 SCP 반복 수에 도달했을 때 내린다. 추가로, 필요하면 QP 해 이후 step clipping을 적용하여 지나치게 큰 업데이트를 제한한다.

<a id="fig-scp-pipeline"></a>
![그림 2. 제어점 공간에서의 SCP 파이프라인](../figures/f2_scp_pipeline.png)
**그림 2 [F2].** SCP 파이프라인의 제어점 공간 구현. 제어다각형을 초기화하고, 재사용 가능한 연산자를 조립한 뒤, 현재 반복점을 분할하여 지지 반공간과 다른 국소 선형화를 다시 만들고, 볼록 2차 계획 문제를 풀어 갱신을 반복하는 구조이다. 이는 원래 문제를 한 번에 정확히 컨벡스화하는 것이 아니라, SCP 루프에서 연속적으로 컨벡스 QP를 푸는 구조임.

---

## 5. 실험 설정

### 5.1 시연 문제와 평가 지표

실험에는 단순화된 3차원 orbital transfer 문제를 사용하였다. 우주선은 지구 중심의 구형 KOZ를 회피하면서 주어진 초기 위치와 최종 위치 사이를 이동해야 한다. KOZ 반경은 $r_e = 6471$ km, 전이 시간은 $T = 1500$ s로 고정하였다. 양 끝점에서 위치를 고정하고, 초기 및 최종 속도 제약도 함께 부과하였다. 중력장은 two-body 항과 J2 perturbation을 포함하며, 목적함수 계산 과정에서는 대표 분할구간 위치에서 affine linearization을 사용한다.

시연 시나리오는 Progress-to-ISS approach 문제를 단순화한 단일-arc orbital transfer 시나리오로, 실제 궤도를 기반으로 하지만 분석 편의상 단순화하였다. Chaser는 고도 245 km의 원궤도, target은 고도 400 km 원궤도에서 시작하며, 두 궤도는 동일 평면(경사각 51.64 deg) 내에서 120 deg의 초기 위상차를 갖는다고 가정한다. 

이러한 세팅을 선택한 이유는, 120 deg 위상차 시나리오에서는 초기 궤적이 KOZ 경계 근처까지 접근하는 경로가 자연스럽게 형성되기 때문이다. 즉, 분할 수 변화에 따른 수치적 보수성 차이(conservatism gap)가 실제로 크게 드러날 수 있는 검증에 적합한 사례이기 때문이다.

Solver는 Rust 기반 QP 백엔드를 사용하였다. SCP 반복은 직선 형태의 초기 제어다각형에서 시작되고, 최대 10000회 반복까지 허용한다. 수렴 허용오차는 $\|P^{(k+1)} - P^{(k)}\|_F < 10^{-12}$, proximal regularization weight는 $10^{-6}$, step clipping 반경은 2000 km로 각각 고정하였다.

본 논문에서 사용하는 평가지표는 다음과 같다.

- **Solve success**: 최종 해가 제약조건을 모두 만족하는지 여부
- **Safety margin**: 최종 궤적의 최소 반경에서 KOZ 반경을 뺀 값
- **Delta-v proxy**: `dv` objective의 최종 값
- **Runtime**: 전체 SCP 계산 시간
- **iterations**: 종료 시점까지 수행된 SCP 반복 횟수

### 5.2 분할 수와 차수에 대한 비교 실험 설정

첫 번째 비교 실험에서는 Bézier 차수를 $N=7$로 고정한 채, 분할 수 $n_{\mathrm{seg}} \in \{2,4,8,16,32,64\}$를 변화시키는 sweep을 수행한다. 이 실험의 목적은, 분할 수가 커질수록 계산 비용과 안전 근사의 보수성(tradeoff)에 어떤 실질적인 변화가 나타나는지, 그리고 위상차 120 deg 시나리오라는 불리한(즉, KOZ 경계에 근접한) 조건에서 이 효과가 가장 뚜렷하게 관찰될 수 있음을 확인하는 것이다.

두 번째 비교 실험은 차수 $N \in \{6,7,8\}$에 대한 비교이다. 대표 비교 표는 $n_{\mathrm{seg}} = 16$에서 구성하였고, 전체 분할 수 비교 실험에 대해서도 차수별 추세를 함께 확인하였다. 차수 비교에는 표현 자유도 변화와 변수 수 변화가 동시에 반영되므로, 결과는 계산 비용과 표현력의 결합된 효과로 해석한다.

---

## 6. 결과

본 절에서는 제안 기법이 실제로 실행 가능한 궤적을 만들어 내는지, 분할 수가 계산 비용과 결과에 어떤 영향을 주는지, 그리고 Bézier 차수가 성능에 어떤 차이를 만드는지를 차례로 살펴본다.

### 6.1 대표 궤적과 기본 feasibility

먼저 제안 기법이 대상 orbital transfer 문제에서 실제로 실행 가능한 궤적을 생성하는지 확인한다. 대표 궤적의 예시는 [그림 3](#fig-trajectory)에 제시하였고, 정량 결과는 표 2에 요약하였다.

<a id="fig-trajectory"></a>
![그림 3. 대표 orbital transfer 궤적 예시](../figures/f3_representative_settings.png)
**그림 3 [F3].** 대표 orbital transfer 설정에서 얻은 representative trajectory 예시.

**표 2 [T2]. 대표 설정에서의 결과 요약**

| Setting | Degree | Control points | $n_{\mathrm{seg}}$ | Solve success | Safety margin (km) | Delta-v proxy $J_{\mathrm{dv}}$ (m/s) | Runtime (s) | iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `demo_N6_seg16` | 6 | 7 | 16 | True | 14.454 | 6,693.904 | 38.284 | 10000 |
| `demo_N7_seg16` | 7 | 8 | 16 | True | 14.832 | 6,909.823 | 2.054 | 292 |
| `demo_N8_seg16` | 8 | 9 | 16 | True | 12.673 | 6,299.037 | 107.346 | 10000 |

표 2에서 보듯이, 시험한 세 차수 모두에서 실행 가능한 궤적을 얻을 수 있었다. 안전 여유는 $N=6,7$에서 약 14–15 km, $N=8$에서 약 12.7 km로, KOZ 경계에 가까이 접근하면서도 제약을 만족하는 궤적이 생성되었다. Delta-v 대리 지표는 $N=8$에서 약 6,299 m/s로 가장 낮고, $N=6$에서 약 6,694 m/s, $N=7$에서 약 6,910 m/s 순서를 보여, 반드시 차수 순서대로 개선되지는 않는다. 계산 시간은 $N=7$이 292회 반복 만에 수렴하여 약 2.1 s로 가장 빠르고, $N=6$은 10000회 반복에 약 38.3 s, $N=8$은 10000회 반복에 약 107.3 s로 증가하였다. 따라서 대표 설정에서는 $N=8$이 가장 낮은 delta-v 대리 지표를 주지만 계산 시간도 가장 크며, $N=7$은 가장 빠른 수렴을 보여 차수 선택이 단순한 monotone 관계가 아닌 tradeoff임을 확인할 수 있다.

### 6.2 분할 수에 따른 변화

다음으로 분할 수가 결과에 미치는 영향을 살펴본다. 정량 결과는 표 3에 정리하였고, 요약 추세는 [그림 4](#fig-subdivision-tradeoff)에 함께 제시하였다.

**표 3 [T3]. 분할 수에 대한 비교 실험 결과 ($N=7$)**

| $n_{\mathrm{seg}}$ | Solve success | Safety margin (km) | Delta-v proxy $J_{\mathrm{dv}}$ (m/s) | Runtime (s) | iterations |
|---:|---:|---:|---:|---:|---:|
| 2 | True | 145.000 | 21,485.930 | 103.957 | 7214 |
| 4 | True | 137.115 | 14,012.521 | 0.368 | 12 |
| 8 | True | 60.585 | 7,468.268 | 1.008 | 209 |
| 16 | True | 14.832 | 6,909.823 | 2.054 | 292 |
| 32 | True | 3.595 | 6,316.465 | 87.176 | 10000 |
| 64 | True | 0.886 | 6,292.563 | 169.880 | 10000 |

표 3에서 가장 두드러지는 점은 분할 수가 증가할수록 안전 여유와 delta-v 대리 지표가 모두 단조적으로 감소한다는 것이다. 안전 여유는 $n_{\mathrm{seg}}=2$의 약 145 km에서 $n_{\mathrm{seg}}=64$의 약 0.9 km로 줄어들며, 이는 분할이 정밀해질수록 궤적이 KOZ 경계에 더 가까이 접근할 수 있음을 의미한다. 동시에 delta-v 대리 지표는 약 21,486 m/s에서 약 6,293 m/s로 약 3.4배 개선되었으며, 가장 큰 개선 폭은 $n_{\mathrm{seg}}=2$에서 $n_{\mathrm{seg}}=16$ 사이에서 나타난다. 여섯 설정 모두 feasible하므로, 이 비교 실험은 결측 없는 완전한 tradeoff 곡선을 제공한다. 계산 시간은 $n_{\mathrm{seg}}=4$의 0.4 s에서 $n_{\mathrm{seg}}=64$의 약 170 s로 증가하여, 보수성 감소(conservatism reduction)에 대한 명확한 계산 비용 tradeoff가 존재함을 확인할 수 있다.

<a id="fig-subdivision-tradeoff"></a>
![그림 4. subdivision count에 따른 runtime 및 outcome trend](../figures/f4_subdivision_tradeoff_N7.png)
**그림 4 [F4].** 120 deg 위상차 시나리오에서 $N=7$ 기준 subdivision count에 따른 runtime 및 outcome 변화. 분할 수가 증가할수록 safety margin은 약 145 km에서 약 1 km로, delta-v proxy는 약 21,486 m/s에서 약 6,293 m/s로 단조 감소하며, runtime은 sub-second에서 약 170 s로 증가한다. 이는 분할 정밀화에 따른 명확한 보수성 감소-계산 비용 tradeoff를 보여준다.

### 6.3 Bézier 차수에 따른 변화

이제 Bézier 차수 변화가 결과에 미치는 영향을 살펴본다. 차수가 높아지면 표현 자유도는 커지지만, 변수 수와 계산량도 함께 증가한다. 정량 결과는 표 4에 정리하였고, 전체 추세는 [그림 5](#fig-multi-order-trend)에 요약하였다.

**표 4 [T4]. 차수에 대한 비교 실험 결과 ($n_{\mathrm{seg}}=16$)**

| Setting | Degree | Control points | $n_{\mathrm{seg}}$ | Solve success | Safety margin (km) | Delta-v proxy $J_{\mathrm{dv}}$ (m/s) | Runtime (s) | Mean control accel (m/s²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `demo_N6_seg16` | 6 | 7 | 16 | True | 14.454 | 6,693.904 | 38.284 | 4.485 |
| `demo_N7_seg16` | 7 | 8 | 16 | True | 14.832 | 6,909.823 | 2.054 | 4.621 |
| `demo_N8_seg16` | 8 | 9 | 16 | True | 12.673 | 6,299.037 | 107.346 | 4.259 |

표 4는 차수 변화에 따른 성능 지표를 비교한 것이다. 세 차수 모두 실행 가능성을 유지하며, safety margin은 $N=6,7$에서 약 14–15 km, $N=8$에서 약 12.7 km로 유사한 수준이다. Delta-v 대리 지표는 $N=8$에서 약 6,299 m/s로 가장 낮지만, $N=7$은 약 6,910 m/s로 $N=6$의 약 6,694 m/s보다 오히려 높아, 차수 순서대로 단조 감소하지는 않는다. 평균 제어 가속도는 $N=8$이 가장 낮다. 계산 시간은 $N=7$이 292회 반복에서 수렴하여 2.1 s로 가장 빠르며, $N=6$은 10000회 반복에 38.3 s, $N=8$은 10000회 반복에 107.3 s가 소요되었다. 따라서 차수 변화의 효과는 blanket superiority가 아니라, 표현 자유도, 수렴 거동, 그리고 계산 비용 사이의 tradeoff로 해석하는 것이 적절하다.

<a id="fig-multi-order-trend"></a>
![그림 5. 다차수 성능 추세 요약](../figures/f5_multi_order_tradeoff_N678.png)
**그림 5 [F5].** 120 deg 위상차 시나리오에서 $N=6,7,8$ 차수에 대한 성능 추세. 좌측 패널은 subdivision count에 따른 delta-v proxy 변화를, 우측 패널은 같은 설정에서의 runtime을 보여준다. 세 차수 모두 $n_{\mathrm{seg}} \ge 8$에서 유사한 effort 수준에 수렴하며, $N=8$이 전반적으로 가장 낮은 effort를 보이나 runtime은 가장 크다. 차수 선택은 blanket superiority가 아닌 effort-runtime tradeoff로 해석해야 한다.

이상의 결과는 제안 정식화 자체의 실행 가능성과 분할 수 및 차수 변화에 따른 거동을 보여준다. 후속 direct collocation warm-start 비교는 본 논문의 intended-use 맥락에서는 중요하지만, 현재 draft pass에서는 placeholder로만 남겨 둔다.

### 6.4 Downstream direct collocation warm-start 비교 (placeholder)

이 절은 direct collocation 대비 우월성 주장을 위한 자리가 아니라, 동일한 downstream solver 설정에서 Bézier 기반 warm start가 initialization quality 측면에서 어떤 실용적 가치를 갖는지 평가하기 위한 planned comparison 자리이다.

**[T6] Downstream direct-collocation initialization comparison (draft placeholder).**

[추후 삽입: 동일한 downstream problem, mesh, tolerance, stopping rule 하에서 (i) naive 또는 DB-seeded initialization과 (ii) same DB seed -> Bézier warm start -> downstream direct collocation pipeline을 비교한다. 보고 지표는 solve success, iteration count, solve time, final objective, final constraint violation, optional safety margin으로 통일하고, 개선되지 않는 사례와 실패 사례도 함께 남긴다.]
---

## 7. 한계 및 해석 범위

본 연구의 결과는 다음과 같은 범위 안에서 해석해야 한다.

첫째, 연속 안전 논리는 구형 KOZ와 본 논문의 분할-지지-반공간 구성에 한정된다. 따라서 임의 형상의 장애물이나 시변 장애물에 대해 동일한 주장을 그대로 적용할 수는 없다.

둘째, 본 논문이 사용하는 목적함수는 실제 delta-v 자체가 아니라 제어 가속도를 반영하는 대리 지표이다. 따라서 결과는 연속적으로 안전한 초기 궤적을 생성하는 관점에서 해석한다.

셋째, 실험은 단일 orbital transfer 예제(위상차 120 deg)에 기반한다. 정식화 자체는 기하학적으로 더 일반적인 구조를 가지더라도, 현재 단계의 실험적 근거는 해당 시연 문제에 대해 제시된다.

넷째, 제안 기법은 일련의 convex QP를 푸는 방식으로 작동하지만, 전체 non-convex 문제에 대한 전역 최적성을 보장하지 않는다.

다섯째, 본 논문은 전이 시간을 고정한 설정을 다루므로, free final time이나 waiting strategy를 포함하는 시간 조절 문제는 현재 범위 밖에 있다.

---

## 8. 결론

본 논문에서는 제어점 공간에서 직접 작동하는 Bézier 기반 궤적 초기화 기법을 제안하였다. 제안 기법은 분할된 분할구간의 제어다각형에 지지 반공간 제약을 부과함으로써, 구형 KOZ에 대한 연속 안전을 보수적으로 처리한다. 또한 전체 문제를 순차 컨벡스화 절차 안에서 일련의 convex QP로 풀 수 있도록 구성하였다.

단순화된 orbital transfer 문제(위상차 120 deg)에 대한 실험 결과, 제안 기법은 대표 차수 설정 $N \in \{6,7,8\}$에서 실행 가능한 궤적을 생성할 수 있었다. Subdivision 실험에서는 $n_{\mathrm{seg}}$가 증가할수록 안전 여유가 약 145 km에서 약 1 km로 단조 감소하고, delta-v 대리 지표도 약 3.4배 개선되어, 분할 수 증가가 보수성을 실질적으로 줄여줌을 확인하였다. 차수 실험에서는 $N=8$이 가장 낮은 목적함수 값을 보이나 계산 시간도 가장 크고, $N=7$은 가장 빠른 수렴을 보이는 등 차수 간 tradeoff가 나타났다.
현재 데이터는 제안 정식화의 feasibility, 보수성 감소 효과, 그리고 차수-계산비용 tradeoff를 뒷받침하지만, downstream warm-start 효과는 아직 planned comparison 단계에 머물러 있다.

결론적으로, 제안 기법은 제어점 공간에서 연속 안전 제약을 구성하고 이를 SCP 기반 최적화와 결합하는 하나의 정식화를 제공한다. 후속 solver와의 연계 검증은 이 프레임워크의 중요한 활용 가능성을 평가하는 다음 단계이며, 이후에는 다양한 문제 설정에 대한 실험 확대와 시간 최적화를 포함하는 확장을 통해 적용 범위를 넓힐 수 있다.

---

## 참고문헌

[향후 추가 예정]
