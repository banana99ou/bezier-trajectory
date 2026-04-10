# 세그먼트별 볼록화에 기반한 Bézier 궤적 초기화 기법

부제: Orbital transfer 예제에 대한 적용

---

**초록**

본 논문에서는 구형 Keep-Out Zone(KOZ)을 연속적으로 회피하는 궤적을 생성하기 위한 Bézier 기반 초기화 기법을 제안한다. 제안하는 방법은 전적으로 control point 공간에서 작동하며, 미분 연산자, (subdivision)[[unnesseary english]] 행렬, 경계조건, KOZ 제약식을 모두 control point에 대한 선형 연산으로 기술한다. 구형 KOZ 회피는 De Casteljau (subdivision)[[unnesseary english]]으로 곡선을 여러 sub-arc로 분할한 뒤, 각 sub-arc의 control polygon에 supporting half-space 제약을 부과하는 방식으로 보수적으로 처리한다. Bézier 곡선의 convex hull 성질에 따르면 해당 sub-arc의 모든 control point가 반공간 제약을 만족할 때 곡선 전체도 KOZ 바깥에 놓이게 된다. 최종 최적화는 successive convexification 절차 안에서 일련의 convex quadratic subproblem을 푸는 방식으로 수행된다.

제안한 기법은 단순화된 orbital transfer 문제에 적용하여 검증하였다. (subdivision)[[unnesseary english]] 수에 대한 ablation 결과, (subdivision)[[unnesseary english]]을 세분할수록 계산 시간은 증가하지만 본 실험 설정에서는 safety margin과 delta-v proxy의 변화는 크지 않았다. Bézier 차수에 대한 ablation에서는 모든 시험 차수에서 feasible trajectory를 얻을 수 있었으며, 고차에서는 목적함수 값이 소폭 개선될 수 있으나 계산 비용이 뚜렷하게 증가하였다. 본( 논문은 direct collocation에 대한 우월성, 전역 최적성, 실제 연료 최적성을 주장하지 않는다.)[[bad tone]] 제안한 방법의 목적은 후속 최적화 문제에 사용할 수 있는 smooth하고 연속적으로 안전한 warm start를 제공하는 데 있다.

---

## 1. 서론

제약조건을 가지는 궤적 최적화 문제는 항공우주, 로봇공학, autonomous systems 등 여러 분야에서 반복적으로 등장한다. 이때 중요한 요구 가운데 하나는 궤적이 특정 금지 영역을 경로 전체에 걸쳐 지속적으로 회피해야 한다는 점이다. 그러나 일반적인 direct transcription 또는 direct collocation 방식에서는 제약식이 주로 이산화된 지점에서만 강제되므로, 노드 사이 구간의 안전은 별도로 확인해야 하는 경우가 많다. 따라서 이산화된 지점만이 아니라 연속 구간 전체에서 안전을 다룰 수 있는 표현과 제약 방식이 필요하다.

또 다른 실용적 문제는 초기값의 품질이다. 많은 downstream solver는 초기값에 민감하며, 초기값이 좋지 않으면 제약을 만족하지 않는 해로 수렴하거나, 반복 횟수가 크게 증가하거나, 품질이 낮은 국소해에 머무를 수 있다. 이런 점에서 후속 고충실도 최적화에 앞서 smooth하고 제약을 만족하는 초기 궤적을 만들어 주는 절차는 그 자체로 의미가 있다.

본 논문은 이러한 문제를 해결하기 위해 Bézier 곡선을 이용한 궤적 초기화 기법을 제안한다. 제안한 기법의 핵심은 모든 계산을 control point 공간에서 수행한다는 점이다. 곡선의 미분, (subdivision)[[unnesseary english]], 경계조건, KOZ 제약이 모두 control point에 대한 선형 연산으로 정리되므로, 계산 구조가 비교적 단순하고 해석도 명확하다. 특히 구형 KOZ에 대해서는 (subdivision)[[unnesseary english]]된 각 sub-arc에 supporting half-space를 부여하고, 그 half-space 안에 control polygon이 놓이도록 함으로써 연속 안전을 보수적으로 확보한다.

본 논문의 기여는 다음과 같이 정리할 수 있다.

1. Bézier parameterization을 기반으로 control point 공간에서 직접 작동하는 궤적 초기화 정식화를 제시하여 제약 구성과 계산 구조를 단순화한다.
2. De Casteljau (subdivision)[[unnesseary english]]과 supporting half-space를 이용하여 구형 KOZ를 연속적으로 회피하도록 하는 보수적 제약 구성 방식을 제안한다.
3. 비볼록 KOZ 제약을 각 SCP 반복에서 재선형화하면서 일련의 convex QP를 푸는 SCP 기반 절차를 정리한다.
4. 단순화된 orbital transfer 문제에서 분할 수와 Bézier 차수에 대한 ablation을 수행하여 계산 비용과 성능의 관계를 분석한다.

(본 논문은 제안 기법이 direct collocation을 대체한다고 주장하지 않는다. 또한 전역 최적성이나 실제 delta-v 최적성도 주장하지 않는다. 본 연구의 위치는 완성된 최종 해법보다는, 후속 solver에 전달할 수 있는 구조화된 warm start 생성 기법에 가깝다.)[[factual but weakens the paper reception. consider moving to non-claims section. or should be re-written to not undermine the impression of this paper. honesty is good, but honesty that damages the paper reception is bad.]]

이후 구성은 다음과 같다. 2절에서는 관련 연구와의 관계를 정리하고, 3절에서는 문제 설정과 표기법을 소개한다. 4절에서는 제안 기법의 수학적 구성과 알고리즘을 설명한다. 5절에서는 실험 설정을 기술하고, 6절에서는 결과를 제시한다. 7절에서는 한계와 해석 범위를 정리하며, 8절에서 결론을 맺는다.

---

## 2. 관련 연구 및 위치 설정

(제안 기법의 위치를 분명히 하기 위해, 본 절에서는 세 가지 맥락에서 관련 연구를 정리한다.)[[awkward]] 첫째는 direct transcription 및 direct collocation 계열의 trajectory optimization 방법, 둘째는 obstacle avoidance를 위한 convexification 또는 보수적 안전 처리 방식, 셋째는 downstream optimization을 위한 초기화 또는 warm start 생성 방법이다.

### 2.1 Direct transcription 및 direct collocation과의 관계

Direct transcription과 direct collocation은 제약이 있는 trajectory optimization에서 가장 널리 쓰이는 접근이다. 이들 방법은 궤적을 여러 노드에서의 상태와 입력 변수로 이산화하고, dynamics를 equality constraint로 부과한 뒤, 큰 규모의 nonlinear program을 푼다. 다양한 문제에 적용 가능하고 solver 생태계도 잘 갖추어져 있다는 장점이 있다.

(본 논문은 이러한 방법과 경쟁하는 대체 방법을 제안하는 것이 아니다. 오히려 direct collocation을 downstream 단계의 정교한 solver로 보고, 그 전에 전달할 수 있는 더 나은 (초기 추정)[[direct translation bad]]을 만드는 문제에 초점을 둔다. 따라서 본 연구의 핵심 질문은 “Bézier 정식화가 collocation을 대체할 수 있는가”가 아니라, “Bézier 기반 초기화가 downstream solver에 실질적으로 유용한 시작점을 줄 수 있는가”에 가깝다.)[[this is not a key perpose of this framework. should be demoted as a non trivial feature of this framework. and applied to claim docs also.]]

### 2.2 보수적 장애물 처리 및 볼록화

연속적인 장애물 회피는 점별 제약만으로는 다루기 어렵다. 노드 간 구간에서 어떤 일이 일어나는지는 추가 논증이 필요하기 때문이다. 이를 다루기 위한 여러 접근 가운데 일부는 특정 문제 클래스에 대해 무손실 볼록화를 사용하고, 또 다른 일부는 순차 볼록화로 비볼록 제약을 반복적으로 선형화한다.

본 논문은 순차 볼록화의 틀을 사용하되, 점별 상태 제약이 아니라 Bézier sub-arc의 제어점에 제약을 가하는 형태로 적용한다. 구체적으로는 분할된 각 sub-arc에 대해 구형 KOZ를 지지하는 반공간을 만들고, 제어점이 그 반공간 안에 위치하도록 한다. 이 방식은 정확한 (회피를 주는 것이)[[awkward to "give" avoidance]] 아니라 보수적인 회피를 제공하지만, convex hull 성질을 이용하여 sub-arc 전체에 대한 연속 안전을 (논할 수 있다는 장점이 있다.)[[unnatural. how about "보장된다는 장점이 있다"]]

### 2.3 초기화 및 시작점 생성

초기값의 품질이 비선형 궤적 최적화의 수렴 거동에 큰 영향을 준다는 점은 잘 알려져 있다. 실제로는 직선 보간, 경험적 형상화, 단순 모델 해, 데이터베이스 기반 초기화 등 다양한 방식이 사용된다. 그러나 단순한 초기화는 연속 안전을 반영하지 못하는 경우가 많다.

본 논문의 제안 기법은 이러한 초기화 방법들 가운데 하나로 이해할 수 있다. 즉, 비교적 저차원인 control point 공간에서 smooth하고 안전한 궤적을 먼저 만든 다음, 이를 downstream solver의 초기값으로 제공하는 방식이다. 여기서 장점은 초기화 문제의 차원을 작게 유지하면서도, 기하학적 안전 제약을 명시적으로 다룰 수 있다는 점이다.

---

## 3. 문제 설정 및 표기법

### 3.1 궤적 표현과 결정 변수

궤적은 정규화된 매개변수 $\\tau \\in [0,1]$ 위에서 정의된 차수 $N$의 Bézier 곡선으로 표현한다.

$$
\mathbf{r}(\tau) = \sum_{i=0}^{N} B_i^{N}(\tau)\,\mathbf{p}_i
$$

여기서 $B_i^N$은 Bernstein 기저다항식이고, $\mathbf{p}_i \in \mathbb{R}^3$은 $i$번째 control point이다. Control point를 행렬로 모으면

$$
P =
\begin{bmatrix}
\mathbf{p}_0^{\mathsf{T}} \\
\mathbf{p}_1^{\mathsf{T}} \\
\vdots \\
\mathbf{p}_N^{\mathsf{T}}
\end{bmatrix}
\in \mathbb{R}^{(N+1)\times 3}
$$

가 되고, 이를 하나의 벡터로 쌓으면

$$
\mathbf{x}
=
\begin{bmatrix}
\mathbf{p}_0^{\mathsf{T}} &
\mathbf{p}_1^{\mathsf{T}} &
\cdots &
\mathbf{p}_N^{\mathsf{T}}
\end{bmatrix}^{\mathsf{T}}
\in \mathbb{R}^{3(N+1)}
$$

를 얻는다. 본 논문에서 최적화의 결정 변수는 $\mathbf{x}$이며, 이후의 미분 연산, (subdivision)[[unnesseary english]], KOZ 제약은 모두 이 벡터에 대한 선형 연산으로 표현된다.

본 연구에서는 전이 시간 $T$를 고정한다. 물리 시간 $t$와 정규화 매개변수 $\tau$의 관계는

$$
t = T\tau
$$

로 둔다. 따라서 실제 속도와 가속도는 $\tau$에 대한 미분에 각각 $1/T$, $1/T^2$를 곱한 형태로 얻어진다. 본 논문은 free final time이나 timing allocation은 다루지 않는다.

### 3.2 미분 연산자와 경계조건

Bézier 곡선의 미분 구조는 표준 difference matrix를 이용하여 정리할 수 있다.

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

또한 미분된 control point를 다시 원래 차수의 basis 위에서 표현하기 위해 degree elevation matrix $E_M$을 사용한다. 이를 이용하면 차수 보존형 속도 및 가속도 연산자를

$$
L_{1,N} = E_{N-1}D_N,
\qquad
L_{2,N} = E_{N-1}D_N E_{N-1}D_N
$$

로 쓸 수 있고, 대응하는 control point는

$$
P^{(1)} = L_{1,N}P,
\qquad
P^{(2)} = L_{2,N}P
$$

이다.

물리적인 속도와 가속도는 다음과 같다.

$$
\dot{\mathbf{r}}(t) = \frac{1}{T}\frac{d\mathbf{r}}{d\tau},
\qquad
\ddot{\mathbf{r}}(t) = \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2}
$$

끝점 위치는 첫 번째와 마지막 control point를 고정하여 부과한다. 끝점 속도는

$$
\dot{\mathbf{r}}(0) = \frac{N}{T}(\mathbf{p}_1-\mathbf{p}_0),
\qquad
\dot{\mathbf{r}}(T) = \frac{N}{T}(\mathbf{p}_N-\mathbf{p}_{N-1})
$$

로 주어지며, 필요할 경우 끝점 가속도도 같은 방식으로 선형 제약식으로 표현할 수 있다.

### 3.3 Gram matrix와 quadratic form

Bernstein basis의 Gram matrix는 닫힌 형태로 계산할 수 있으며,

$$
[G_N]_{ij}
=
\frac{\binom{N}{i}\binom{N}{j}}{\binom{2N}{i+j}(2N+1)},
\qquad i,j=0,\ldots,N
$$

로 쓴다. 이를 가속도 연산자와 결합하면

$$
\tilde G_N = L_{2,N}^\top G_N L_{2,N}
$$

을 얻고,

$$
\int_0^1 \left\|\frac{d^2\mathbf{r}}{d\tau^2}\right\|_2^2 d\tau
=
\mathrm{tr}(P^\top \tilde G_N P)
=
\mathbf{x}^\top (\tilde G_N \otimes I_3)\mathbf{x}
$$

의 정확한 quadratic form으로 정리된다. 본 논문에서 이 구성은 재사용 가능한 연산자 체계의 일부이다.

---

## 4. 제안 방법

### 4.1 (subdivision)[[unnesseary english]]과 supporting half-space를 이용한 구형 KOZ 처리

구형 KOZ를 다음과 같이 정의한다.

$$
\mathcal{K}
=
\left\{
\mathbf{r}\in\mathbb{R}^3 :
\|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le r_e
\right\}
$$

여기서 $\mathbf{c}_{\mathrm{KOZ}}$는 KOZ의 중심이며, 현재 실험에서는 원점 중심의 경우를 사용한다.

제안 기법에서는 곡선을 $n_{\mathrm{seg}}$개의 sub-arc로 등분하기 위해 De Casteljau (subdivision)[[unnesseary english]] 행렬 $S^{(s)}$를 사용한다. 그러면 $s$번째 sub-arc의 control polygon은

$$
P^{(s)} = S^{(s)}P
$$

가 된다. 이 sub-arc의 control point를 $\mathbf{q}^{(s)}_0, \ldots, \mathbf{q}^{(s)}_N$이라 하면, 대표점으로는 control polygon의 centroid

$$
\mathbf{c}^{(s)} = \frac{1}{N+1}\sum_{k=0}^{N}\mathbf{q}^{(s)}_k
$$

를 사용한다.

이때 외향 법선은

$$
\mathbf{n}^{(s)}
=
\frac{\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}}
{\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\|_2}
$$

로 정의하며, centroid가 KOZ 중심과 일치하는 퇴화 경우는 제외한다. 이에 따라 구의 supporting half-space는

$$
\mathcal{H}^{(s)}
=
\left\{
\mathbf{r} :
(\mathbf{n}^{(s)})^\top \mathbf{r}
\ge
(\mathbf{n}^{(s)})^\top \mathbf{c}_{\mathrm{KOZ}} + r_e
\right\}
$$

로 쓸 수 있다.

본 논문에서는 각 sub-arc의 모든 control point가 이 반공간 안에 놓이도록 다음 부등식을 부과한다.

$$
(\mathbf{n}^{(s)})^\top \mathbf{q}^{(s)}_k
\ge
(\mathbf{n}^{(s)})^\top \mathbf{c}_{\mathrm{KOZ}} + r_e,
\qquad
k=0,\ldots,N
$$

각 $\mathbf{q}^{(s)}_k$는 원래 control point의 선형결합이므로 이 제약식은 (결정 변수)[[what does this supposed to mean?]] $\mathbf{x}$에 대해 선형이다. 반공간은 각 SCP 반복에서 현재 해를 기준으로 다시 구성된다. (따라서 이 제약은 전역적으로 정확한 회피를 보장하는 것이 아니라, 현재 해 주변에서 작동하는 보수적이고 국소적인 회피 제약으로 해석해야 한다.)[[using english sentence structure of "therefore X should be interpreted as Y" not bad but debatable]]

이 제약이 안전성을 보장하는 이유는 다음과 같다. 어떤 sub-arc의 모든 control point가 $\mathcal{H}^{(s)}$ 안에 있으면, convex hull 성질에 의해 그 sub-arc 전체도 동일한 반공간 안에 있다. 그리고 $\mathcal{H}^{(s)}$는 구형 KOZ를 지지하는 반공간이므로, 해당 sub-arc는 KOZ 바깥에 놓이게 된다. 이 주장은 구형 KOZ와 본 논문에서 사용하는 (subdivision)[[unnesseary english]]-half-space 구성에 대해서만 성립한다.

### 4.2 control effort surrogate objective

본 논문에서는 affine하게 선형화한 중력장 아래에서 제어 가속도를 근사적으로 반영하는 surrogate 목적함수를 사용한다.

우선

$$
\\mathbf{u}(t) = \\ddot{\\mathbf{r}}(t) - \\mathbf{g}(\\mathbf{r}(t))
$$

를 정의한다. 여기서 $\\mathbf{g}$는 orbital gravity model이며, (현재 구현)[[direct translation bad]]에서는 two-body 항과 J2 perturbation 항을 포함한다. 제안 기법은 곡선 전체에 걸쳐 dynamics를 정확히 강제하지 않고, representative sub-arc 위치에서 중력장을 affine하게 선형화하여 objective를 구성한다.

Objective 구성에는 KOZ (subdivision)[[unnesseary english]] 수와 별도로 $n_{\\mathrm{lin}}$개의 선형화용 sub-arc를 사용한다. 대응하는 centroid row를 이용하면 representative 위치와 geometric acceleration을 모두 $\\mathbf{x}$의 선형 함수로 쓸 수 있다. 외부 반복 $k$에서 gravity field는 현재 기준 위치 $\\mathbf{r}_i(\\mathbf{x}^{(k)})$에서 affine linearization으로 근사된다.

그 결과 각 샘플에 대해 선형화된 control effort residual

$$
\\boldsymbol{\\rho}_i^{(k)}(\\mathbf{x})
=
A_i\\mathbf{x} - \\left(B_i^{(k)}\\mathbf{x} + \\mathbf{c}_i^{(k)}\\right)
$$

를 정의할 수 있다. 구현은 이 residual에 대해 IRLS(iteratively reweighted least squares) 방식의 quadratic majorization을 구성하여 다음 objective를 푼다.

$$
J_{\\mathrm{dv}}^{(k)}(\\mathbf{x})
=
\\sum_{i=1}^{n_{\\mathrm{lin}}}
\\omega_i^{(k)}
\\left\\|
\\boldsymbol{\\rho}_i^{(k)}(\\mathbf{x})
\\right\\|_2^2
$$

여기서 가중치 $\\omega_i^{(k)}$는 현재 반복점의 residual 크기에 따라 갱신된다. 이 목적함수는 반복마다 제어 가속도 residual을 안정적으로 줄이도록 구성된 surrogate이며, (결과 해석에서도 실제 연료 최적화 지표와는 구분해서 다루어야 한다.)[[still bad]]

### 4.3 convex subproblem과 SCP 알고리즘

SCP 반복 $k$에서 KOZ (반공간)[[inconsistence choice of word. use 반공간 or half-space across all document]]과 중력 선형화를 모두 고정하면, 내부 문제는 다음과 같은 convex QP가 된다.

$$
\\begin{aligned}
\\min_{\\mathbf{x}} \\quad &
\\frac{1}{2}\\mathbf{x}^\\top H^{(k)}\\mathbf{x}
+ \\bigl(\\mathbf{f}^{(k)}\\bigr)^\\top \\mathbf{x} \\\\
\\text{s.t.} \\quad &
A_{\\mathrm{KOZ}}^{(k)}\\mathbf{x} \\ge \\mathbf{b}_{\\mathrm{KOZ}}^{(k)}, \\\\
&
A_{\\mathrm{bc}}\\mathbf{x} = \\mathbf{b}_{\\mathrm{bc}}, \\\\
&
\\boldsymbol{\\ell} \\le \\mathbf{x} \\le \\mathbf{u}
\\end{aligned}
$$

여기서 $A_{\\mathrm{KOZ}}^{(k)}\\mathbf{x} \\ge \\mathbf{b}_{\\mathrm{KOZ}}^{(k)}$는 (subdivision)[[unnesseary english]]에 의해 얻은 KOZ 제약, $A_{\\mathrm{bc}}\\mathbf{x}=\\mathbf{b}_{\\mathrm{bc}}$는 경계조건, $\\boldsymbol{\\ell}, \\mathbf{u}$는 끝점 위치를 포함한 bound이다.

필요한 경우 현재 iterate 주변에 다음과 같은 proximal regularization도 추가한다.

$$
\\frac{\\lambda}{2}\\|\\mathbf{x}-\\mathbf{x}^{(k)}\\|_2^2
$$

이는 문제의 convexity를 유지한 채 SCP 반복의 안정성을 높이는 역할을 한다.

(현재 orbital 실험에서는 prograde preservation 제약도 함께 사용한다. 초기 angular momentum 방향 $\\hat{\\mathbf{h}}$를 기준으로)

$$
c(\\tau;\\mathbf{x})
=
\\hat{\\mathbf{h}}^\\top
\\bigl(
\\mathbf{r}(\\tau;\\mathbf{x}) \\times \\dot{\\mathbf{r}}(\\tau;\\mathbf{x})
\\bigr)
$$

를 정의하고, 이를 몇 개의 내부 샘플 지점에서 선형화하여 부등식으로 추가한다. 이 제약은 본 기법의 일반 이론보다는 현재 실험 설정에 가까운 요소이므로, 재현성을 위해 명시한다.

전체 SCP 절차는 다음 순서로 수행된다.

1. 현재 control polygon을 (subdivision)[[unnesseary english]]한다.
2. 각 sub-arc에 대해 supporting half-space를 다시 계산한다.
3. gravity linearization과 IRLS 가중치를 갱신한다.
4. 대응하는 convex QP를 푼다.
5. 새 control point를 다음 SCP 반복의 기준점으로 사용한다.

((현재 구현)[[don't mention code]]에서는 quadratic model의 exact gradient와 Hessian을 사용하여 (내부 문제)[[direct translation bad]]를 수치적으로 (푼다)[[awkward]]. 수렴 판정은)[[what is the point of this whole paragraph?]]

$$
\\|P^{(k+1)} - P^{(k)}\\|_F < \\mathrm{tol}
$$

을 만족할 때 또는 최대 outer iteration 수에 도달했을 때 내린다. 추가로, 필요하면 QP 해 이후 step clipping을 적용하여 지나치게 큰 업데이트를 제한한다.

(### 4.4 가정과 해석 범위

제안 기법의 해석 범위는 분명히 제한되어 있다.

첫째, 연속 안전 논리는 구형 KOZ와 본 논문의 subdivision-half-space 구성에 한정된다. 이를 임의 형상의 장애물이나 시변 장애물로 그대로 확장할 수는 없다.

둘째, 전이 시간은 고정되어 있다. 따라서 time allocation, waiting behavior, free final time 문제는 다루지 않는다.

셋째, 논문 수준 objective는 surrogate이다. `dv` 모드는 delta-v 자체가 아니라 control effort에 대한 근사 지표이므로, 실제 연료 최적성에 대한 주장을 뒷받침하지 않는다.

넷째, gravity 처리는 국소적인 affine linearization에 기반하므로 근사적이다. 각 subproblem이 convex라는 사실이 원래 비볼록 문제 전체가 정확히 convexification되었음을 의미하지는 않는다.

(마지막으로, 본 논문은 제안 기법을 downstream solver를 위한 warm start 생성 도구로 제시한다. 그 downstream 효과는 결과에서 별도로 판단해야 하며, 방법론 자체만으로 자동으로 보장되는 것은 아니다.)[[non claim discaimer should be mentioned discreately. stretegically. not prominent like this]]

---

## 5. 실험 설정

### 5.1 시연 문제와 평가 지표

실험에는 단순화된 3차원 orbital transfer 문제를 사용하였다. 우주선은 지구 중심의 구형 KOZ를 회피하면서 주어진 초기 위치와 최종 위치 사이를 이동해야 한다. KOZ 반경은 $r_e = 6471$ km, 전이 시간은 $T = 1500$ s로 고정하였다. 양 끝점에서 위치를 고정하고, 초기 및 최종 속도 제약도 함께 부과하였다. 중력장은 two-body 항과 J2 perturbation을 포함하며, objective 계산 과정에서는 representative sub-arc 위치에서 affine linearization을 사용한다.



Solver는 Rust 기반 QP 백엔드를 사용하였다. SCP는 직선 형태의 초기 control polygon에서 시작하며, 최대 500회의 outer iteration까지 수행한다. 수렴 허용오차는 $\\|P^{(k+1)} - P^{(k)}\\|_F < 10^{-6}$으로 두었고, proximal regularization weight는 $10^{-6}$, step clipping 반경은 2000 km로 고정하였다.

본 논문에서 사용하는 평가지표는 다음과 같다.

- **Solve success**: 최종 해가 활성화된 제약식을 만족하는지 여부
- **Safety margin**: 최종 궤적의 최소 반경에서 KOZ 반경을 뺀 값
- **Delta-v proxy**: `dv` objective의 최종 값
- **Runtime**: 전체 SCP 계산 시간
- **Outer iterations**: 종료 시점까지 수행된 SCP 반복 횟수

### 5.2 분할 수와 차수에 대한 ablation 설정

첫 번째 ablation은 Bézier 차수를 $N=6$으로 고정한 상태에서 (subdivision)[[unnesseary english]] 수 $n_{\\mathrm{seg}} \\in \\{2,4,8,16,32,64\\}$를 변화시키는 실험이다. 목적은 (subdivision)[[unnesseary english]]을 늘릴수록 계산 비용과 안전 근사 정도 사이에 어떤 변화가 생기는지 확인하는 것이다.

두 번째 ablation은 차수 $N \\in \\{5,6,7\\}$에 대한 비교이다. 대표 비교 표는 $n_{\\mathrm{seg}} = 16$에서 구성하였고, 전체 (subdivision)[[unnesseary english]] sweep에 대해서도 차수별 추세를 함께 확인하였다.

(다만 차수를 바꾸면 표현력이 달라질 뿐 아니라 decision variable 수 자체도 바뀌므로, 차수 효과와 문제 크기 효과를 완전히 분리해 해석하기는 어렵다. 본 논문은 이 한계를 인정한 상태에서 결과를 정리한다.)[[honest. but dosen't help the paper. have to word this better than this.]]

---

## 6. 결과

본 절에서는 제안 기법이 실제로 feasible한 궤적을 만들어 내는지, (subdivision)[[unnesseary english]] 수가 계산 비용과 결과에 어떤 영향을 주는지, 그리고 Bézier 차수가 성능에 어떤 차이를 만드는지를 차례로 살펴본다. 마지막으로 downstream warm start로서의 가능성도 제한적으로 논의한다.

### 6.1 대표 궤적과 기본 feasibility

먼저 제안 기법이 대상 orbital transfer 문제에서 실제로 feasible trajectory를 생성하는지 확인한다. (이 절의 목적은 다른 방법과의 우열 비교가 아니라, 제안 기법이 최소한 본 논문에서 다루는 설정 안에서 안정적으로 작동하는지를 보여주는 데 있다.)[[unessary info]]

**표 2. 대표 설정에서의 결과 요약**

| Setting | Degree | Control points | $n_{\mathrm{seg}}$ | Solve success | Safety margin (km) | Delta-v proxy $J_{\mathrm{dv}}$ (m/s) | Runtime (s) | Outer iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `demo_N5_seg16` | 5 | 6 | 16 | True | 145.000 | 17,814.989 | 5.211 | 500 |
| `demo_N6_seg16` | 6 | 7 | 16 | True | 145.000 | 17,905.937 | 5.063 | 500 |
| `demo_N7_seg16` | 7 | 8 | 16 | True | 145.000 | 17,752.065 | 17.768 | 500 |

(표 2에서 보듯이, 시험한 세 차수 모두에서 feasible trajectory를 얻을 수 있었다. Safety margin은 세 경우 모두 145 km로 동일하였고, delta-v proxy는 17,752 m/s에서 17,906 m/s 사이로 큰 차이를 보이지 않았다. 반면 계산 시간은 차수에 따라 차이가 있었다. $N=5$와 $N=6$은 약 5초 수준이었으나, $N=7$은 약 18초로 증가하였다. 또한 모든 경우가 outer iteration 한계인 500회에 도달하여 종료되었다는 점도 함께 관찰되었다.

이 결과는 제안 기법이 적어도 현재 시연 문제에서는 smooth하고 feasible한 궤적을 안정적으로 생성할 수 있음을 보여준다. 다만 이는 단일 orbital transfer 예제에 대한 결과이며, 이를 곧바로 일반적인 성능 보장으로 해석할 수는 없다.)[[remember that we're writing boiler plate for yet to be comming numbers.]]

### 6.2 (subdivision)[[unnesseary english]] 수에 따른 변화

다음으로 (subdivision)[[unnesseary english]] 수가 결과에 미치는 영향을 살펴본다. ((subdivision)[[unnesseary english]]을 많이 할수록 KOZ 제약은 더 세밀하게 적용되지만, 그만큼 계산 부담도 증가할 것으로 예상된다.)[[unnatural for professional paper. unnatural to "expect" result]]

**표 3. (subdivision)[[unnesseary english]] 수에 대한 ablation 결과 ($N=6$)**

| Degree | $n_{\mathrm{seg}}$ | Solve success | Safety margin (km) | Delta-v proxy $J_{\mathrm{dv}}$ (m/s) | Runtime (s) | Outer iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 2 | True | 145.000 | 17,730.286 | 3.123 | 500 |
| 6 | 4 | True | 145.000 | 17,866.995 | 3.334 | 500 |
| 6 | 8 | True | 145.000 | 17,898.397 | 3.892 | 500 |
| 6 | 16 | True | 145.000 | 17,905.937 | 5.063 | 500 |
| 6 | 32 | True | 145.000 | 17,905.284 | 7.075 | 500 |
| 6 | 64 | True | 145.000 | 17,906.558 | 11.511 | 500 |

(표 3의 가장 뚜렷한 특징은 계산 시간의 증가이다. $n_{\\mathrm{seg}}=2$에서 약 3.1초였던 runtime이 $n_{\\mathrm{seg}}=64$에서는 약 11.5초까지 증가하였다. 반면 safety margin은 모든 경우에 145 km로 동일하였고, delta-v proxy 역시 전체 범위에서 거의 변하지 않았다.

즉, 현재 데이터는 (subdivision)[[unnesseary english]] 수를 늘릴수록 계산 비용이 분명히 증가한다는 점은 보여 주지만, 그에 상응하는 성능 개선이 현재 지표에서는 뚜렷하게 관찰되지는 않는다. 따라서 본 논문에서는 (subdivision)[[unnesseary english]] 증가가 “더 좋다”라고 단정하지 않고, 현재 실험 범위에서는 주로 계산 비용 증가가 확인되었다고 해석한다.)[[should be boillerplate not actual intepretation of result.]]

### 6.3 Bézier 차수에 따른 변화

이제 Bézier 차수 변화가 결과에 미치는 영향을 살펴본다. 차수가 높아지면 표현 자유도는 커지지만, 변수 수와 계산량도 함께 증가한다.

**표 4. 차수에 대한 ablation 결과 ($n_{\\mathrm{seg}}=16$)**

| Setting | Degree | Control points | $n_{\mathrm{seg}}$ | Solve success | Safety margin (km) | Delta-v proxy $J_{\mathrm{dv}}$ (m/s) | Runtime (s) | Mean control accel (m/s²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `demo_N5_seg16` | 5 | 6 | 16 | True | 145.000 | 17,814.989 | 5.211 | 11.966 |
| `demo_N6_seg16` | 6 | 7 | 16 | True | 145.000 | 17,905.937 | 5.063 | 12.034 |
| `demo_N7_seg16` | 7 | 8 | 16 | True | 145.000 | 17,752.065 | 17.768 | 11.938 |

(세 차수 모두 feasible trajectory를 생성하였다. Delta-v proxy는 $N=7$에서 가장 낮았지만, 그 차이는 크지 않았다. 반면 runtime은 $N=7$에서 크게 증가하였다. Mean control acceleration 역시 세 경우 모두 큰 차이를 보이지 않았다.

따라서 현재 실험에서는 고차 Bézier가 목적함수 측면에서 약간 유리할 수는 있으나, 그 이득은 계산 시간 증가와 함께 보아야 한다. 즉, 차수에 대해서도 단순한 우열 관계보다는 trade-off 관점이 더 적절하다.)[[should be boillerplate not actual intepretation of result.]]

(### 6.4 downstream warm start로서의 가능성

제안 기법의 실용적 가치는 downstream solver에 warm start를 제공할 수 있다는 점에 있다. 다만 현재 저장소에서 진행된 첫 번째 downstream 비교는 명확한 결론을 주지 못했다. naive initialization과 Bézier warm start 모두 direct collocation solve에 성공하였고, warm start 쪽이 최종 objective는 약간 더 낮았으나 iteration 수와 runtime은 더 컸다.

따라서 현재 단계에서 “warm start로서 유용하다”는 강한 실증적 결론을 내리기는 어렵다. 보다 공정하고 반복 가능한 비교가 추가로 필요하다. 본 논문에서는 이 부분을 이미 입증된 결과가 아니라, 제안 기법의 의도된 활용 방향으로 남겨 둔다.)[[it is unfair to mention result from buggy untested dcm implementaion that is under development. should be worded again expecting some positive result in specific region of input curves. (must refer to @dcm_db_experiment_note.md)]]

---

## 7. 한계 및 해석 범위

본 연구의 결과는 다음과 같은 한계 안에서 해석해야 한다.

(첫째, 연속 안전 논리는 구형 KOZ와 본 논문의 (subdivision)[[unnesseary english]]-half-space 구성에 한정된다. 임의 형상의 장애물이나 시변 장애물에 대해 동일한 주장을 그대로 적용할 수는 없다.

둘째, (본 논문이 사용하는 objective는 실제 delta-v가 아니라 surrogate이다. 따라서 결과를 실제 연료 최적화 성능과 직접 동일시해서는 안 된다.)[[true but sounds bad. should say this is fine cause we're doing this for mock case or initial guess gen perpose]]

셋째, 실험은 단일 orbital transfer 예제에 기반한다. 정식화 자체는 기하학적으로 더 일반적인 구조를 가지더라도, 실험적 증거는 아직 한정적이다.

넷째, 제안 기법은 일련의 convex QP를 푸는 방식으로 작동하지만, 전체 non-convex 문제에 대한 전역 최적성을 보장하지 않는다.

다섯째, 전이 시간은 고정되어 있으므로 free final time 문제나 waiting strategy는 현재 범위 밖에 있다.

(마지막으로, downstream warm start 효과는 아직 제한적으로만 확인되었다. 따라서 본 논문은 direct collocation 대비 우월성이나 일반적 수렴 개선을 주장하지 않는다.)[[again unfair conclusion. should be worded positively. removed or re-written like this if the final dcm still gives bad result.]])[[each shortcommings should have upshots. or rationalization etc to not harm the paper. honest is good as long as it doesn't hearts me.]]

---

## 8. 결론

본 논문에서는 control point 공간에서 직접 작동하는 Bézier 기반 궤적 초기화 기법을 제안하였다. 제안 기법은 (subdivision)[[unnesseary english]]된 sub-arc의 control polygon에 supporting half-space 제약을 부과함으로써, 구형 KOZ에 대한 연속 안전을 보수적으로 처리한다. 또한 전체 문제를 successive convexification 절차 안에서 일련의 convex QP로 풀 수 있도록 구성하였다.

단순화된 orbital transfer 문제에 대한 실험 결과, 제안 기법은 시험한 차수와 (subdivision)[[unnesseary english]] 설정 전반에서 feasible trajectory를 생성할 수 있었다. (subdivision)[[unnesseary english]] 수를 늘리면 계산 시간은 분명히 증가했지만, 현재 실험에서는 safety margin과 delta-v proxy의 변화는 크지 않았다. 차수 증가 역시 목적함수 측면에서 소폭 이득을 줄 수 있었으나, 계산 비용 증가가 함께 나타났다. 따라서 본 연구의 실험 결과는 (subdivision)[[unnesseary english]]과 차수 모두에 대해 단순한 우열보다는 trade-off 관점에서 이해하는 것이 적절하다.

결론적으로, 제안 기법은 control point 공간에서 연속 안전 제약을 구성하고 이를 SCP 기반 최적화와 결합하는 하나의 정식화를 제공한다. 향후에는 보다 공정한 downstream 비교, 다양한 문제 설정에 대한 검증, 그리고 시간 최적화까지 포함하는 확장이 필요하다.)[[should be worded positively. should not sound like we're giving up on the paper. should be worded like "future work" etc]]

---

## 참고문헌

[향후 추가 예정]

---

## Strategic Revision TODO

- [ ] 초록과 서론 앞부분의 non-claim 문장(`직접 collocation 대비 우월성 미주장`, `전역 최적성 미주장` 등)을 어디까지 전면에 둘지 다시 검토할 것.
- [ ] `warm start`와 `downstream solver` 관련 위치 설정을 방법론의 핵심 기여와 어떻게 분리해 제시할지 재정리할 것.
- [ ] 2.1의 direct collocation 관련 단락은 현재 연구의 정체성을 축소할 수 있으므로, 비교 대상 설명과 연구 기여 설명의 비중을 다시 조정할 것.
- [ ] 4.3의 prograde preservation 제약은 핵심 방법이 아니라 실험 설정에 가까우므로, 본문 유지 여부와 배치를 다시 판단할 것.
- [ ] 4.4의 `가정과 해석 범위` 단락은 필요하지만 현재는 서사상 다소 전면적이므로, 한계 절과의 역할 분담을 다시 검토할 것.
- [ ] 6.4의 downstream warm-start 서술은 아직 검증 중인 DCM 결과에 기대고 있으므로, `doc/dcm_db_experiment_note.md`의 범위 정의에 맞춰 별도 재서술할 것.
- [ ] 7절의 downstream 효과 관련 한계 문장은 정직성을 유지하되 연구 인상을 과도하게 약화하지 않도록 문구를 재조정할 것.
- [ ] 결론에서는 `warm start` positioning보다 control-point-space formulation과 continuous safety construction 자체의 기여가 먼저 드러나도록 문장 순서를 다시 다듬을 것.
