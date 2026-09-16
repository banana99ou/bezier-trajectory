# 분할구간별 볼록화에 기반한 Bézier 궤적 생성 기법

---

**초록**

본 논문에서는 구형 장애물을 회피하는 Bézier 궤적 생성 기법을 제안한다. 곡선을 분할하고 각 분할구간의 회피 제약을 제어점에 대한 선형 부등식으로 근사하여, 궤적 생성 문제를 일련의 볼록 최적화 문제로 정식화한다. 모든 분할구간에서 인증 조건을 만족하면 궤적 전 구간의 장애물 회피가 보장된다.

단순화된 궤도전이 문제에 적용하여 분할 수에 따른 보수성과 계산 비용의 변화를 분석하였다. 시험한 다섯 전이 시나리오 모두에서 1초 미만의 계산 시간으로 실현 가능한 궤적을 얻었다. 측정한 보수성은 분할구간 현 길이의 제곱에 비례한다는 예측과 수 % 이내로 일치하였으며, 분할 수의 제곱에 반비례하여 감소하였다.

---

## 1. 서론

항공우주, 로봇공학, 자율 시스템의 궤적 최적화에서는 경로 전 구간에 걸친 장애물 회피가 요구된다. 그러나 direct transcription과 direct collocation [1, 2]은 주로 이산화된 노드에 제약을 부과하므로, 노드에서 제약을 만족하더라도 노드 사이에서는 제약을 위반할 수 있다. 이를 노드 간 제약 위반(inter-sample constraint violation)이라 한다 [6]. 궤적 전 구간에서 제약을 만족하는 성질을 연속시간 제약 만족(continuous-time constraint satisfaction) [7]이라 하며, 이를 보장할 수 있는 궤적 표현과 제약 정식화가 필요하다.

Bézier 곡선의 제어점을 최적화 변수로 사용하는 궤적 생성 기법은 여러 연구에서 다루어졌다 [13, 14]. 최지웅·이수원 [18]은 미분·차수 상승 연산자와 Gram 행렬을 이용하여 제어점에 대한 이차형식 비용함수를 구성하고, 곡선 분할과 분리 평면 제약을 순차 볼록 프로그래밍에 적용하였다. 본 연구에서는 이를 바탕으로 구형 장애물 회피 제약의 보수성을 정량적으로 분석하고, 비선형 중력을 고려한 궤도전이 문제로 확장한다.

본 연구의 주요 내용은 다음과 같다. 첫째, 분할구간별 지지 반공간 제약의 보수성을 접평면 근사로 예측하고, 분할 수에 따른 변화를 실험으로 확인한다(3.1절, 5.2절). 둘째, 선행연구 [18]의 선형 Clohessy–Wiltshire(CW) 상대운동 모델을 비선형 중력 모델로 확장한다. 이체 중력과 $J_2$ 섭동을 구간별로 선형화하고, 기존의 Gram 행렬 정식화를 적용하여 제어 가속도 비용을 볼록 이차형식으로 구성한다(3.2절). 셋째, 제어점 변화에 따른 지지 반공간 법선의 변화까지 미분에 포함한다. 이를 신뢰영역 기반 순차 볼록화(successive convexification, SCvx) [4, 5]의 회피 제약 선형화와 감소량 평가에 반영한다(3.3절).

Direct transcription과 direct collocation은 노드별 상태와 입력을 변수로 두고 동역학을 등식 제약으로 부과하여 비선형 계획 문제를 푼다 [1, 2]. 적용 범위가 넓고 다양한 수치해법을 이용할 수 있지만, 점별 제약만으로는 연속시간 제약 만족을 보장하기 어렵다.

궤적 최적화의 비볼록 제약을 다루는 방법으로는 특정 문제에 적용할 수 있는 무손실 볼록화 [3]와, 제약을 반복적으로 선형화하는 순차 볼록화 [4, 5]가 있다.

장애물 외부의 자유 공간은 비볼록이며, 위상을 보존하는 좌표 변환만으로는 볼록화할 수 없다 [10]. 장애물을 둘러싼 경로는 한 점으로 축약되지 않는 반면 볼록 집합은 축약 가능하기 때문이다. 또한 자유 공간을 하나의 볼록 집합으로 완화하면 서로 다른 경로의 평균까지 실현 가능한 경로로 포함된다. 따라서 볼록 최적화를 이용한 회피 기법은 주로 두 가지 접근을 취한다. 첫째는 자유 공간을 여러 볼록 영역으로 덮고 통과할 영역을 선택하는 방법으로, 볼록 영역 생성 [11]과 볼록 집합 그래프 [12]가 이에 해당한다. 둘째는 장애물의 통과 방향을 정하고 지지 반공간으로 자유 공간을 제한하는 방법이다. 본 논문에서는 두 번째 방법을 사용한다.

볼록 껍질 성질을 이용하면 유한 개의 제어점에 대한 조건으로 곡선 전체의 안전을 보장할 수 있다 [13]. 안전 회랑 기법에서도 각 분할구간과 장애물의 쌍마다 하나의 반공간을 정하고, 해당 분할구간의 모든 제어점에 동일한 반공간 제약을 부과한다 [15, 16, 17]. 제어점마다 서로 다른 반공간을 적용하면 그 합집합의 볼록성이 보장되지 않으므로, 개별 제어점의 조건만으로 곡선 전체의 안전을 보장할 수 없다. 3.1절에서는 동일한 반공간의 적용을 제약 구성의 가정으로 명시한다.

2절에서는 제어점 공간의 궤적 표현과 표기법을 정리하고, 3절에서는 제안 기법의 정식화와 알고리즘을 설명한다. 4절과 5절에서는 각각 실험 설정과 수치 결과를 제시하며, 6절에서 결론과 한계를 기술한다.

---

## 2. 제어점 공간에서의 궤적 표현

선행연구 [18, 2.1–2.6절]의 제어점 공간 정식화에 따라 필요한 표기와 식을 정리한다. 미분·차수 상승 연산자와 Gram 행렬의 기본 성질은 문헌 [13, 18]을 따른다.

### 2.1 궤적 표현과 결정 변수

전이 시간 $T$를 고정하고 $t=T\tau$, $\tau\in[0,1]$로 두면, 3차원 궤적을 차수 $N$의 Bézier 곡선으로 나타낼 수 있다.

$$
\mathbf{r}(\tau) = \sum_{i=0}^{N} B_i^{N}(\tau)\,\mathbf{p}_i, \qquad
P = [\mathbf{p}_0,\ldots,\mathbf{p}_N]^{\mathsf{T}} \in \mathbb{R}^{(N+1)\times 3}
$$

여기서 $B_i^N$은 Bernstein 기저다항식이고, $\mathbf{p}_i\in\mathbb{R}^3$은 제어점이다. 최적화의 결정 변수는

$$
\mathbf{x} = \mathrm{vec}\!\left(P^{\mathsf{T}}\right) = [\mathbf{p}_0^{\mathsf{T}},\ldots,\mathbf{p}_N^{\mathsf{T}}]^{\mathsf{T}} \in \mathbb{R}^{3(N+1)}
$$

로 정의한다.

### 2.2 미분 연산자와 경계조건

미분 행렬 $D_N\in\mathbb{R}^{N\times(N+1)}$과 차수 상승 행렬 $E_{N-1}\in\mathbb{R}^{(N+1)\times N}$을 이용하여 차수를 보존하는 미분 연산자를 정의한다 [18].

$$
L_{1,N} = E_{N-1}D_N, \qquad L_{2,N} = E_{N-1}D_N E_{N-1}D_N
$$

물리 시간에 대한 속도·가속도 제어점은 각각 $L_{1,N}P/T$, $L_{2,N}P/T^2$이다. 본 논문에서 사용하는 끝점 위치·속도 조건은

$$
\mathbf{p}_0=\mathbf{r}_0, \qquad \mathbf{p}_N=\mathbf{r}_f, \qquad
\frac{N}{T}(\mathbf{p}_1-\mathbf{p}_0)=\mathbf{v}_0, \qquad
\frac{N}{T}(\mathbf{p}_N-\mathbf{p}_{N-1})=\mathbf{v}_f
$$

이며, 이를 $A_{\mathrm{bc}}\mathbf{x}=\mathbf{b}_{\mathrm{bc}}$ 형태로 정리한다. $\mathbf{r}_0$, $\mathbf{r}_f$, $\mathbf{v}_0$, $\mathbf{v}_f$는 각각 주어진 초기·최종 위치와 속도이다.

### 2.3 Gram 행렬과 이차형식

Bernstein Gram 행렬 $G_N$의 성분과 이에 따른 적분 항등식은 다음과 같다 [13, 18].

$$
[G_N]_{il} = \frac{\binom{N}{i}\binom{N}{l}}{\binom{2N}{i+l}(2N+1)}, \qquad i,l=0,\ldots,N
$$

$$
\int_0^1 \|\mathbf{f}(\tau)\|_2^2\,d\tau = \mathrm{tr}(F^{\mathsf{T}}G_NF), \qquad
\mathbf{f}(\tau)=\sum_{i=0}^{N}B_i^N(\tau)\mathbf{f}_i, \qquad
F=[\mathbf{f}_0,\ldots,\mathbf{f}_N]^{\mathsf{T}}
$$

$F$가 결정 변수 $\mathbf{x}$의 1차 함수이면 위 적분은 $\mathbf{x}$에 대한 볼록 이차형식으로 표현되며, 수치 적분 없이 정확히 계산할 수 있다. 3.2절에서는 이 식을 중력이 선형화된 제어 가속도 잔차에 적용한다. 비선형 중력을 사용한 목적함수의 평가는 3.3절에서 설명한다.

---

## 3. 순차 볼록화 기반 궤적 생성

2절의 연산자를 이용하여 장애물 회피 제약(3.1절)과 제어 비용 목적함수(3.2절)를 구성한다. 두 요소를 결합한 볼록 하위 문제와 SCvx 알고리즘은 3.3절에서 설명한다.

### 3.1 분할과 지지 반공간을 이용한 구형 장애물 처리

곡선 분할과 분리 평면을 이용한 회피 제약은 선행연구 [18, 2.7–2.8절]을 따른다. 이 절에서는 보수성 분석과 3.3절의 제약 미분에 필요한 식을 정리한다. 중심이 $\mathbf{c}_{\mathrm{KOZ}}$이고 반경이 $R_{\mathrm{KOZ}}$인 구형 장애물을 다음과 같이 정의한다.

$$
\mathcal{K} = \left\{\mathbf{r}\in\mathbb{R}^3 : \|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le R_{\mathrm{KOZ}} \right\}
$$

본 논문에서는 장애물 중심을 원점에 둔다.

연속시간 장애물 회피 조건은 모든 $\tau \in [0,1]$에서 $\mathbf{r}(\tau) \notin \operatorname{int}\mathcal{K}$, 즉 $\|\mathbf{r}(\tau)-\mathbf{c}_{\mathrm{KOZ}}\|_2 \ge R_{\mathrm{KOZ}}$를 만족하는 것이다. 이는 곡선상의 유한 개 노드에만 부과하는 조건보다 강하다. 전이 시간 $T$가 고정되어 $t$와 $\tau$가 일대일로 대응하므로, $\tau$ 전 구간에서 조건을 만족하면 물리 시간 전 구간에서도 조건을 만족한다. 제안 기법에서는 이 비볼록 조건을 보장하는 볼록 충분조건을 사용한다(명제 1).

곡선을 $n_{\mathrm{seg}}$개로 등분하는 De Casteljau 분할 행렬을 $S^{(s)}$라 하면 [18], $s$번째 분할구간의 제어점 행렬은

$$
P^{(s)} = S^{(s)}P
$$

가 된다. 각 행은 분할구간의 제어점 $\mathbf{q}^{(s)}_0, \ldots, \mathbf{q}^{(s)}_N$에 해당한다. 선행연구 [18]에서는 분할구간의 매개변수 중점에서 법선을 구성하였으나, 본 논문에서는 제어점의 무게중심

$$
\mathbf{c}^{(s)} = \frac{1}{N+1}\sum_{m=0}^{N}\mathbf{q}^{(s)}_m
$$

를 사용한다. 분할 전후의 제어점과 그 볼록 껍질의 관계는 [그림 1](#fig-ctrl-subdivision)에 나타내었다.

<a id="fig-ctrl-subdivision"></a>
![그림 1. De Casteljau 분할에 따른 제어점과 볼록 껍질의 변화](../figures/control_subdivision.png)
**그림 1 [F1].** 분할 행렬 $S^{(s)}$로 계산한 제어점 $P^{(s)}$와 볼록 껍질. 분할 후 볼록 껍질은 곡선에 더 가까워진다. 명제 1은 이 볼록 껍질 안에 곡선 전체가 포함된다는 성질을 이용한다.

이때 외향 법선은

$$
\mathbf{n}^{(s)} = \frac{\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}}{\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\|_2}
$$

로 정의한다. 이에 따라 구의 지지 반공간은

$$
\mathcal{H}^{(s)} = \left\{\mathbf{r} : (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{r} \ge (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{c}_{\mathrm{KOZ}} + R_{\mathrm{KOZ}} \right\}
$$

로 쓸 수 있다.

각 분할구간의 모든 제어점이 이 반공간에 속하도록 다음 제약을 부과한다.

$$
(\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{q}^{(s)}_m \ge (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{c}_{\mathrm{KOZ}} + R_{\mathrm{KOZ}}, \qquad m=0,\ldots,N
$$

이는 선행연구 [18, 식 (29)]에서 별도의 안전거리를 0으로 둔 제약이다. 볼록 껍질 성질 [13]에 따른 회피 충분조건을 명제 1로 정리하고, 이를 실현 가능성 판정에 사용한다.

> **명제 1.** 분할구간 $s$의 제어점을 $P^{(s)} = S^{(s)}P$, 구형 장애물을 $\mathcal{K} = \{\mathbf{r}\in\mathbb{R}^3 : \|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le R_{\mathrm{KOZ}}\}$라 하자. 모든 제어점 $\mathbf{q}^{(s)}_0, \ldots, \mathbf{q}^{(s)}_N$이 위에서 정의한 지지 반공간 $\mathcal{H}^{(s)}$에 속하면, 해당 분할구간의 Bézier 곡선 전체도 $\mathcal{H}^{(s)}$에 속하므로 $\operatorname{int}\mathcal{K}$와 만나지 않는다.

> **가정.** 명제 1의 적용 조건은 다음과 같다.
>
> 1. 장애물은 구형이다.
> 2. 법선 $\mathbf{n}^{(s)}$은 분할구간 제어점의 무게중심 $\mathbf{c}^{(s)}$에서 구성한다.
> 3. 해당 분할구간의 모든 제어점에 동일한 지지 반공간 $\mathcal{H}^{(s)}$의 제약을 부과한다.
> 4. 법선 구성 시 $\mathbf{c}^{(s)} \neq \mathbf{c}_{\mathrm{KOZ}}$이다.

> **증명.** 곡선은 제어점의 볼록 껍질 안에 있고 [13], $\mathcal{H}^{(s)}$는 볼록이며 장애물 내부와 만나지 않는다. 따라서 모든 제어점이 $\mathcal{H}^{(s)}$ 안에 있으면 곡선 전체도 장애물 내부와 만나지 않는다. ($\square$)

분할구간 제어점 $\mathbf{q}^{(s)}_m$은 원래 제어점의 선형결합이다. 따라서 법선 $\mathbf{n}^{(s)}$을 고정하면 위 부등식은 결정 변수 $\mathbf{x}$에 대해 선형이다. 그러나 법선도 무게중심 $\mathbf{c}^{(s)}$를 통해 $\mathbf{x}$에 의존한다. 명제 1의 조건에 대한 위반량을 다음과 같이 정의한다.

$$
h(\mathbf{x}) = \sum_{s}\sum_{m=0}^{N}\max\left\{0,\ R_{\mathrm{KOZ}} + \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\mathbf{c}_{\mathrm{KOZ}} - \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\mathbf{q}^{(s)}_m(\mathbf{x})\right\}
$$

$h(\mathbf{x})$는 법선의 정규화 $\mathbf{v}\mapsto\mathbf{v}/\|\mathbf{v}\|_2$ 때문에 비볼록이다. $h(\mathbf{x}) = 0$이면 명제 1에 따라 곡선 전체의 장애물 회피가 보장된다. 3.3절에서는 기준점에서 회피 조건을 1차 전개하여 볼록 하위 문제를 구성한다. 반공간은 매 SCvx 반복에서 현재 해를 기준으로 갱신하므로, 하위 문제의 회피 제약은 현재 해 주변의 국소 근사이다.

반공간의 경계가 구에 접하면 법선의 선택과 관계없이 명제 1의 회피 보장이 성립한다. 다만 법선의 방향에 따라 제약의 보수성이 달라진다. 여기서 보수성(conservatism)은 제어점의 볼록 껍질에 부과한 제약으로 인해 곡선이 장애물 경계에서 필요 이상으로 떨어지는 정도를 말한다.

분할구간의 두 끝점 사이 거리, 즉 현의 길이를 $L_{\mathrm{seg}}$로 정의한다. 반경 $R_{\mathrm{KOZ}}$인 짧은 원호에서 현과 원호 사이의 최대 거리는 약 $L_{\mathrm{seg}}^2/(8R_{\mathrm{KOZ}})$이다. 본 논문에서는 이 관계를 바탕으로 분할구간의 보수성을 추정하고, 5절에서 수치 결과와 비교한다. 분할구간의 길이가 분할 수에 반비례하면 이 추정값은 $n_{\mathrm{seg}}^{-2}$에 비례하여 감소한다.

하나의 분할구간에 대한 회피 제약 구성을 [그림 2](#fig-subdivision)에 나타내었다.

<a id="fig-subdivision"></a>
![그림 2. 분할과 지지 반공간을 이용한 연속시간 장애물 회피 제약 만족의 개념도](../figures/koz_linearization.png)
**그림 2 [F2].** 분할구간별 구형 장애물 회피 제약. (a) 전체 곡선과 장애물을 침범하는 분할구간, (b) 해당 분할구간의 무게중심에서 구성한 외향 법선과 지지 반공간, (c) 모든 제어점이 반공간에 속하도록 수정한 분할구간. 그림은 명제 1의 조건을 나타낸다. 실제 하위 문제에서는 각 분할구간의 현재 해를 기준으로 이 조건을 1차 전개한다(3.3절).

### 3.2 제어 비용 목적함수

장애물을 회피하는 데 필요한 제어 비용(control effort)을 줄이기 위해 제어 가속도에 대한 목적함수를 구성한다. 궤적을 따르는 데 필요한 제어 가속도는 궤적의 가속도와 중력 가속도의 차이로 정의한다.

$$
\mathbf{u}(t) = \ddot{\mathbf{r}}(t) - \mathbf{g}(\mathbf{r}(t))
$$

여기서 $\mathbf{g}$는 중력 가속도이며, 이체 문제(two-body problem) 항과 $J_2$ 섭동(perturbation) 항을 포함한다. $\mathbf{r} = (x, y, z)$, $r = \|\mathbf{r}\|_2$로 두면 다음과 같다.

$$
\mathbf{g}(\mathbf{r}) = -\frac{\mathrm{GM}}{r^{3}}\mathbf{r}
+ \frac{3 J_2 \mathrm{GM} R_\oplus^{2}}{2 r^{5}}
\begin{bmatrix}
\left(\dfrac{5z^{2}}{r^{2}} - 1\right) x \\[2pt]
\left(\dfrac{5z^{2}}{r^{2}} - 1\right) y \\[2pt]
\left(\dfrac{5z^{2}}{r^{2}} - 3\right) z
\end{bmatrix}
$$

$\mathrm{GM}$은 지구 중력상수, $R_\oplus$는 지구 반경, $J_2$는 2차 대상 조화항 계수이다. 지구의 대칭축은 좌표계의 $z$축과 일치한다고 가정한다. 목적함수는 제어 가속도 크기의 제곱을 궤적 전 구간에서 적분한 값으로 정의한다.

$$
J(\mathbf{x}) = \int_0^1 \left\| \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2} - \mathbf{g}(\mathbf{r}(\tau)) \right\|_2^2 d\tau
$$

중력 가속도가 위치에 대해 비선형이므로 목적함수 $J$는 $\mathbf{x}$에 대해 비볼록이다. 볼록 하위 문제를 구성하기 위해 곡선을 $n_{\mathrm{lin}}$개 구간으로 나누고, 각 구간에서 중력을 1차 테일러(Taylor) 전개로 근사한다. 이때 사용하는 De Casteljau 분할 행렬은 장애물 회피용 $S^{(s)}$와 구분하여 $\hat S^{(j)}$로 표기한다. 장애물 회피용 분할 수 $n_{\mathrm{seg}}$는 회피 제약의 보수성을, 중력 선형화 구간 수 $n_{\mathrm{lin}}$은 목적함수 근사의 정확도를 결정한다. 두 값을 독립적으로 설정하여, 5.2절의 분할 수 실험에서 중력 근사 오차의 영향을 분리한다. 반복 $k$에서 구간 $j$의 제어점 무게중심을 $\mathbf{r}_j^{(k)}$라 하면, 이 점에서 중력을 다음과 같이 선형화한다.

$$
\mathbf{g}(\mathbf{r}) \approx \nabla\mathbf{g}_j^{(k)}\mathbf{r} + \mathbf{c}_j^{(k)}, \qquad \mathbf{c}_j^{(k)} = \mathbf{g}\!\left(\mathbf{r}_j^{(k)}\right) - \nabla\mathbf{g}_j^{(k)}\mathbf{r}_j^{(k)}
$$

여기서 $\nabla\mathbf{g}_j^{(k)} = \partial\mathbf{g}/\partial\mathbf{r}$은 기준점에서 계산한 중력 Jacobian이다. 이 분할은 곡선을 따라 중력 선형화의 기준점을 배치하기 위한 것으로, 적분을 이산화하지는 않는다. 각 구간의 중력은 위치에 대한 1차 함수로 근사된다.

구간 $j$를 $\xi \in [0,1]$로 다시 매개화하고, 궤적의 가속도와 선형화된 중력 가속도의 차이를 잔차로 정의한다.

$$
\mathbf{f}^{(j)}(\xi) = \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2} - \left(\nabla\mathbf{g}_j^{(k)}\mathbf{r} + \mathbf{c}_j^{(k)}\right)
$$

우변은 $\xi$에 대응하는 $\tau$에서 평가한다. 두 항은 모두 $\xi$에 대한 차수 $N$의 Bézier 곡선으로 표현된다. 분할 행렬 $\hat S^{(j)}$와 2.2절의 가속도 연산자를 적용하면 각 항의 제어점은 $\mathbf{x}$의 1차 함수가 된다. 따라서 잔차 $\mathbf{f}^{(j)}$에도 2.3절의 적분 항등식을 적용할 수 있다. 구간별 적분을 합하면 다음을 얻는다.

$$
J^{(k)}(\mathbf{x}) = \sum_{j=1}^{n_{\mathrm{lin}}} \frac{1}{n_{\mathrm{lin}}}\int_0^1 \left\|\mathbf{f}^{(j)}(\xi)\right\|_2^2 d\xi = \sum_{j=1}^{n_{\mathrm{lin}}} \frac{1}{n_{\mathrm{lin}}}\,\mathrm{tr}\!\left(F_j(\mathbf{x})^{\mathsf{T}} G_N F_j(\mathbf{x})\right)
$$

여기서 $1/n_{\mathrm{lin}}$은 매개변수 변환에 따른 계수이며, $F_j(\mathbf{x})$는 잔차 $\mathbf{f}^{(j)}$의 제어점 행렬이다.

$J^{(k)}$는 $\mathbf{x}$에 대한 볼록 이차형식이므로 SCvx의 볼록 QP 목적함수로 사용할 수 있다. 선형화된 잔차의 적분은 해석적으로 정확히 계산되며, 원래 목적함수 $J$와의 차이는 중력 선형화에서 발생한다.

### 3.3 볼록 하위 문제와 SCvx 알고리즘

회피 제약과 중력의 선형화는 기준 제어점 주변에서만 유효하다. 따라서 매 반복에서 두 근사를 갱신하고, 신뢰영역(trust region) 기반 SCvx [4, 5]로 볼록 하위 문제를 푼다. 이 절에서는 하위 문제와 해의 수용 기준을 설명하고 전체 절차를 Algorithm 1로 정리한다.

SCvx 반복 $k$의 하위 문제는 다음의 볼록 QP이다.

$$
\min_{\mathbf{x},\,\boldsymbol{\nu}} \ \frac{1}{2}\mathbf{x}^{\mathsf{T}} H^{(k)}\mathbf{x} + (\boldsymbol{\ell}^{(k)})^{\mathsf{T}}\mathbf{x} + \mu\,\mathbf{1}^{\mathsf{T}}\boldsymbol{\nu}
$$

$$
A_{\mathrm{KOZ}}^{(k)}\mathbf{x} + \boldsymbol{\nu} \ge \mathbf{b}_{\mathrm{KOZ}}^{(k)}, \quad \boldsymbol{\nu} \ge \mathbf{0}, \qquad
A_{\mathrm{bc}}\mathbf{x} = \mathbf{b}_{\mathrm{bc}}, \qquad
\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \le \Delta_k
$$

$H^{(k)}$와 $\boldsymbol{\ell}^{(k)}$는 3.2절의 목적함수 $J^{(k)}$를 $\mathbf{x}$에 대해 전개한 행렬과 벡터이다. 해에 영향을 주지 않는 상수항은 생략하였다. 대표 설정 $N = 7$, $n_{\mathrm{seg}} = 16$에서는 결정 변수가 24개이며, 그중 12개는 경계조건으로 고정된다. 장애물 회피 제약은 $n_{\mathrm{seg}}(N+1) = 128$개이다.

장애물 회피 제약의 행렬 $A_{\mathrm{KOZ}}^{(k)}$와 벡터 $\mathbf{b}_{\mathrm{KOZ}}^{(k)}$는 3.1절의 조건을 기준점 $\mathbf{x}^{(k)}$에서 1차 전개하여 구한다. 분할구간 제어점 $\mathbf{q}^{(s)}_m$의 반공간 경계에 대한 여유를

$$
\gamma^{(s)}_m(\mathbf{x}) = \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\!\left(\mathbf{q}^{(s)}_m(\mathbf{x}) - \mathbf{c}_{\mathrm{KOZ}}\right) - R_{\mathrm{KOZ}}
$$

로 정의한다. 분할 행렬 $S^{(s)}$와 무게중심 가중치 $w^{(s)}_i = \frac{1}{N+1}\sum_{m=0}^{N} S^{(s)}_{mi}$를 이용하면 기울기는 다음과 같다.

$$
\frac{\partial \gamma^{(s)}_m}{\partial \mathbf{p}_i} = S^{(s)}_{mi}\,\mathbf{n}^{(s)} + \frac{w^{(s)}_i}{\left\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\right\|_2}\left(I_3 - \mathbf{n}^{(s)}(\mathbf{n}^{(s)})^{\mathsf{T}}\right)\!\left(\mathbf{q}^{(s)}_m - \mathbf{c}_{\mathrm{KOZ}}\right)
$$

첫째 항은 법선을 고정했을 때의 기울기이며, 둘째 항은 제어점 이동에 따른 무게중심과 법선의 변화를 반영한다. 둘째 항의 사영 행렬 $I_3 - \mathbf{n}^{(s)}(\mathbf{n}^{(s)})^{\mathsf{T}}$은 접평면에 평행한 성분만 남긴다. 따라서 제어점이 접평면 방향으로 접점에서 멀수록 둘째 항이 커지며, 접점에서는 0이 된다. 이 기울기를 이용하여 회피 제약을 다음과 같이 선형화한다.

$$
\gamma^{(s)}_m\!\left(\mathbf{x}^{(k)}\right) + \sum_{i=0}^{N}\left(\frac{\partial \gamma^{(s)}_m}{\partial \mathbf{p}_i}\right)^{\!\mathsf{T}}\!\left(\mathbf{p}_i - \mathbf{p}^{(k)}_i\right) + \nu^{(s)}_m \ \ge\ 0
$$

법선을 고정하면 위 기울기의 둘째 항이 누락되어 회피 제약의 1차 근사가 부정확해진다.

여유 변수(slack variable) $\boldsymbol{\nu}$는 선형화된 회피 제약을 완화하며, SCvx의 virtual control [4, 5]에 해당한다. 초기 궤적이 장애물 내부를 지나는 경우에는 선형화된 제약을 만족하는 해가 없을 수 있다. 이때 여유 변수로 제약 위반을 허용하되, 페널티 계수 $\mu$를 적용하여 수렴한 해에서는 $\boldsymbol{\nu} = \mathbf{0}$이 되도록 한다. $\mu$의 설정 근거는 4.1절에 제시한다. 신뢰영역 제약 $\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \le \Delta_k$는 제어점의 변화 범위를 제한한다. 신뢰영역 크기 $\Delta_k$는 선형화의 정확도에 따라 매 반복 조절한다.

하위 문제에서 구한 해 $\hat{\mathbf{x}}$의 수용 여부는 목적함수와 제약 위반량을 결합한 merit function으로 판단한다 [4, 5].

$$
\phi(\mathbf{x}) = J(\mathbf{x}) + \mu\,h(\mathbf{x})
$$

$J$는 비선형 중력을 사용한 3.2절의 목적함수이며, $h$는 해당 제어점 $\mathbf{x}$에서 다시 구성한 지지 반공간 제약의 위반량이다. 볼록 QP의 목적함수는 이 둘을 근사한 $\phi^{(k)} = J^{(k)} + \mu h^{(k)}$에 해당한다. $J^{(k)}$는 중력을 구간별로 선형화한 목적함수이고, $h^{(k)}$는 회피 조건을 1차 전개한 근사의 위반량이다. 하위 문제에서는 이 위반량을 여유 변수의 합 $\mathbf{1}^{\mathsf{T}}\boldsymbol{\nu}$로 나타낸다.

비선형 중력을 사용한 $J$에는 2.3절의 적분 항등식을 적용할 수 없으므로 수치 구적으로 평가한다. 감소량을 비교할 때에는 $J^{(k)}$도 같은 매개변수 지점에서 수치 구적으로 평가한다. 이는 두 값의 차이에서 구적 오차를 상쇄하고 중력 선형화와 법선 변화에 따른 오차를 평가하기 위한 것이다. 예측 감소량에 대한 실제 감소량의 비는 다음과 같다.

$$
\rho_k = \frac{\phi(\mathbf{x}^{(k)}) - \phi(\hat{\mathbf{x}})}{\phi^{(k)}(\mathbf{x}^{(k)}) - \phi^{(k)}(\hat{\mathbf{x}})}
$$

$\rho_k$는 근사 모델이 실제 감소량을 얼마나 정확히 예측하는지를 나타낸다. 중력 선형화와 법선 변화의 오차가 모두 반영되도록 두 요소를 함께 1차 근사한다. $\rho_k$가 기준값 $\eta = 0.1$을 넘으면 $\hat{\mathbf{x}}$를 수용하여 기준점을 갱신한다. $\rho_k > 0.9$이면 신뢰영역 크기를 두 배로 늘리되 초기 크기의 네 배를 넘지 않도록 한다. $\rho_k \le \eta$이면 해를 수용하지 않고 신뢰영역 크기를 절반으로 줄여 같은 기준점에서 다시 푼다. 직선 보간 초기 궤적은 속도 경계조건을 만족하지 않으므로, 첫 반복에서는 위 비교 없이 해를 수용하여 경계조건을 맞춘다.

기준점에서 merit function의 상대 변화가 $10^{-8}$보다 작고 회피 제약 위반량이 $h \le 10^{-6}$ km인 상태가 연속 $n_{\mathrm{conv}} = 3$회 유지되면 수렴으로 판정한다. 법선이 갱신되는 동안 제어점이 제약면을 따라 천천히 이동하면 상대 변화가 일시적으로 작아질 수 있으므로, 한 번의 반복 결과만으로 수렴을 판정하지 않는다. 같은 이유로 제어점 변화량 자체는 수렴 기준으로 사용하지 않는다.

예측 감소량이 $\phi$의 크기에 $10^{-8}$을 곱한 값보다 작고 회피 인증 조건을 만족하는 경우에도, 이 상태가 연속 $n_{\mathrm{conv}}$회 유지되면 종료한다. 두 종료 기준은 하위 문제의 정지점 조건에 근거한다. 신뢰영역 크기가 하한에 도달하여 반복을 진행할 수 없는 경우는 실패로 기록한다.

제안 기법은 매 반복에서 볼록 QP를 풀고 merit function으로 해를 평가하는 국소 최적화 알고리즘이다. 전체 절차는 Algorithm 1에, 계산 구조는 [그림 3](#fig-scp-pipeline)에 나타내었다.

> **Algorithm 1** — 신뢰영역 기반 궤적 생성 (SCvx)
>
> 1. 직선 보간으로 초기 제어점 $\mathbf{x}^{(0)}$를 구성하고 신뢰영역의 초기 크기 $\Delta_0$를 설정한다
> 2. **for** $k = 0, 1, 2, \ldots$:
> 3. &nbsp;&nbsp; $\mathbf{x}^{(k)}$에서 De Casteljau 분할로 지지 반공간을 구성하고, 회피 조건을 1차 전개하여 $A_{\mathrm{KOZ}}^{(k)}$, $\mathbf{b}_{\mathrm{KOZ}}^{(k)}$를 구한다 (3.1절)
> 4. &nbsp;&nbsp; 각 구간의 무게중심에서 중력을 1차 전개하여 $H^{(k)}$, $\boldsymbol{\ell}^{(k)}$를 구성한다 (3.2절)
> 5. &nbsp;&nbsp; 볼록 QP를 풀어 해 $\hat{\mathbf{x}}$와 여유 변수 $\boldsymbol{\nu}$를 얻는다
> 6. &nbsp;&nbsp; 기준점이 경계조건을 만족하지 못하면 $\hat{\mathbf{x}}$를 바로 수용하고 다음 반복으로 진행한다. 그렇지 않으면 $\rho_k$를 계산한다. $\rho_k > \eta$이면 해를 수용하고, $\rho_k > 0.9$이면 신뢰영역을 확대한다. $\rho_k \le \eta$이면 해를 수용하지 않고 신뢰영역을 축소한다
> 7. &nbsp;&nbsp; **수렴 판정**: 기준점이 $h \le 10^{-6}$ km를 만족하고, merit function의 상대 변화가 허용오차보다 작은 상태가 연속 $n_{\mathrm{conv}}$회 유지되면 종료한다. 같은 인증 조건에서 예측 감소량이 $\phi(\mathbf{x}^{(k)})$의 크기에 허용오차를 곱한 값보다 작은 상태가 연속 $n_{\mathrm{conv}}$회 유지되어도 종료한다. 두 조건의 연속 만족 횟수는 각각 기록한다
> 8. &nbsp;&nbsp; 신뢰영역 크기가 하한 미만이면 실패로 기록하고 종료한다
> 9. &nbsp;&nbsp; 반복 횟수가 한도 1000회에 도달하면 실패로 기록하고 종료한다

<a id="fig-scp-pipeline"></a>
![그림 3. 제어점 공간에서의 SCvx 반복](../figures/scp_pipeline.png)
**그림 3 [F3].** 제어점 공간에서의 SCvx 계산 절차. 선형 연산자는 한 번 계산하여 재사용하고, 지지 반공간과 중력 선형화는 매 반복 갱신한다.

신뢰영역 초기 크기와 페널티 계수 $\mu$의 설정 근거는 4.1절에 제시한다.

---

## 4. 실험 설정

### 4.1 실험 문제와 평가 지표

단순화된 3차원 궤도전이 문제에서 제안 기법을 평가하였다. 초기·최종 위치와 속도를 고정하고, 지구 중심의 구형 장애물을 회피하는 궤적을 생성하였다. 장애물 반경은 $R_{\mathrm{KOZ}} = 6471$ km로 지표면 고도 100 km에 해당하며, 전이 시간은 $T = 1500$ s로 설정하였다. 이 전이 시간은 최소 제어 비용 궤적이 장애물 경계에 접근하여 회피 제약이 작동하도록 정하였다. 요구 중심각이 자유 낙하로 이동하는 각거리인 약 100 deg보다 작으면 궤적이 장애물에서 멀어져 회피 제약이 비활성화되며, 중심각 70 deg가 이에 해당한다.

중력 모델에는 3.2절의 이체 중력과 $J_2$ 섭동을 사용하였다. 중력 선형화 구간 수는 $n_{\mathrm{lin}} = 100$으로 두고, 각 구간의 제어점 무게중심에서 중력을 1차 근사하였다. 선형화된 잔차의 적분은 2.3절의 항등식으로 계산하였다. 구간 수를 200과 400으로 늘렸을 때 수렴한 목적함수 $J$의 상대 변화는 $10^{-9}$ 이내로, 중력 선형화 구간 수에 따른 차이는 미미하였다.

> 중력 상수: $\mathrm{GM} = 3.986004418 \times 10^{5}$ km$^3$/s$^2$, $R_\oplus = 6371$ km, $J_2 = 1.08262668 \times 10^{-3}$.

기준 시나리오는 Progress 우주선의 ISS 접근 궤도 수치를 사용하되, 두 원궤도 위의 점을 하나의 호(arc)로 잇는 전이 문제로 단순화하였다. 출발·도착 고도는 각각 245 km와 400 km이며, 두 궤도는 경사각 51.64 deg의 동일 평면에 있다. 두 끝점의 중심각은 120 deg이고, 속도 경계조건은 각 원궤도의 원운동 속도이다. 이 기준에서 중심각을 70~170 deg로 바꾸거나 도착 궤도면을 변경하여 표 1의 다섯 시나리오를 구성하였다.

**표 1 [T1]. 시험한 전이 시나리오**

| 시나리오 | 출발 고도 (km) | 도착 고도 (km) | 경사각 (deg) | 승교점 적경 (deg) | 중심각 (deg) | 속도 경계조건 사이 각 (deg) | 신뢰영역의 초기 크기 (km) |
|:--|---:|---:|:--|:--|---:|---:|---:|
| 중심각 70 deg | 245 | 400 | 51.64 | 0 | 70.00 | 70.00 | 2000 |
| 중심각 120 deg (기준) | 245 | 400 | 51.64 | 0 | 120.00 | 120.00 | 2000 |
| 중심각 135 deg | 245 | 400 | 51.64 | 0 | 135.00 | 135.00 | 2000 |
| 중심각 170 deg | 245 | 400 | 51.64 | 0 | 170.00 | 170.00 | 4000 |
| 궤도면 변경 | 245 | 400 | 51.64 → 71.64 | 0 → 15 | 125.81 | 125.03 | 2000 |

중심각은 지구 중심에서 본 두 끝점 사이의 각이며, 속도 경계조건 사이 각은 두 끝점의 속도 벡터 사이 각이다. 동일 궤도면을 사용하는 네 시나리오에서는 두 각이 같다. 궤도면 변경 시나리오에서는 도착 궤도의 경사각과 승교점 적경을 바꾸어 출발 궤도면을 벗어나는 3차원 전이를 구성하였다.

중심각 120 deg는 분할 수 비교 실험의 기준으로 사용한다(4.2절). 중심각 135 deg, 170 deg와 궤도면 변경 시나리오에서는 분할 수를 고정하고, 회피 제약이 작동하는 분할구간의 현 길이에 따른 보수성을 비교한다. 중심각 170 deg는 직선 초기 궤적이 장애물 내부를 깊이 통과하는 경우이며, 중심각 70 deg는 회피 제약이 비활성인 대조군이다.

알고리즘은 Rust로 구현하였다. 계산 시간은 Apple M2(8코어)에서 측정하였다.

> 볼록 하위 문제는 내점법 기반 Clarabel 0.11.1로 풀었다. 허용오차는 절대 간극 $10^{-11}$, 상대 간극 $10^{-10}$, 실현 가능성 $10^{-9}$로 설정하였다.

페널티 계수는 $\mu = 10^{-2}$로 설정하였다. L1 페널티가 정확한 페널티(exact penalty)로 작용하여 수렴한 해의 여유 변수가 0이 되려면, 계수가 제약의 쌍대변수 크기보다 커야 한다 [8, 9]. 다섯 시나리오에서 측정한 쌍대변수의 최대 크기는 약 $1.9\times10^{-7}$로, 설정한 계수는 이 값의 약 $5\times10^{4}$배이다. 이때 페널티 항이 목적함수에 비해 지나치게 커지지 않아 $\rho_k$에 목적함수의 개선이 반영된다.

신뢰영역 초기 크기는 $\Delta_0 = 2000$ km로 설정하되, 중심각 170 deg에서는 4000 km를 사용하였다. 첫 반복에서는 속도 경계조건에 맞추어 둘째 제어점을 $\mathbf{p}_0 + (T/N)\mathbf{v}_0$로 이동시켜야 한다. $N = 7$에서 필요한 이동 거리는 중심각 170 deg의 경우 약 2400 km이며, 나머지 시나리오에서는 2000 km 미만이다. 신뢰영역이 이보다 작으면 첫 하위 문제에서 경계조건을 만족할 수 없다. 초기 크기를 더 늘려도 해는 제시한 유효숫자 범위에서 같았다.

성공 여부(solve success)는 최종 해가 경계조건과 명제 1의 인증 조건($h \le 10^{-6}$ km)을 모두 만족하는지로 판정한다. 최소 이격거리(minimum clearance)는 궤적과 장애물 표면 사이 거리의 최솟값이다. 추정 이격거리는 장애물 최근접점이 속한 분할구간의 현 길이 $L_{\mathrm{seg}}$를 이용하여 $L_{\mathrm{seg}}^2/(8R_{\mathrm{KOZ}})$로 계산한다. 측정값과 추정값을 비교하여 3.1절의 보수성 예측을 평가한다. 끝점에서 이격거리가 가장 짧은 경우는 보수성 비교에서 제외하고 표에 따로 표시한다.

제어 비용은 궤적을 따라 필요한 제어 가속도 $\|\mathbf{u}\|_2$의 평균 크기(m/s²)로 정의한다. 계산 시간(runtime)은 SCvx 반복과 후처리 검증·지표 계산에 소요된 시간이며, 반복 횟수(iterations)는 알고리즘이 종료될 때까지 수행한 횟수이다.

### 4.2 분할 수 비교 실험

표 1의 기준 시나리오에서 차수를 $N=7$로 고정하고, 분할 수를 $n_{\mathrm{seg}} \in \{2,4,8,16,32,64\}$로 바꾸어 비교하였다. 분할 수가 증가하면 제어점의 볼록 껍질이 곡선에 가까워져 회피 제약의 보수성과 제어 비용이 감소할 것으로 예상된다. 반면 제약 수가 늘어나므로 계산 시간은 증가할 수 있다. 이 실험에서는 분할 수에 따른 개선 폭과 계산 비용의 변화를 확인한다.

---

## 5. 수치 결과

### 5.1 시나리오별 인증과 보수성

다섯 시나리오의 궤적과 정량 결과를 각각 [그림 4](#fig-trajectory)와 표 2에 제시하였다.

<a id="fig-trajectory"></a>
![그림 4. 표 2의 다섯 전이 시나리오에서 얻은 궤적](../figures/representative_trajectories.png)
**그림 4 [F4].** 다섯 전이 시나리오의 궤적. (A)는 출발 궤도면 위에서 본 궤적이며, (B)는 곡선 매개변수에 따른 장애물과의 이격거리이다. (B)의 표식은 장애물 최근접점을 나타낸다. 중심각 70 deg에서는 출발점이 장애물에 가장 가깝다.

**표 2 [T2]. 시나리오별 결과 요약 ($N=7$, $n_{\mathrm{seg}}=16$)**

| 시나리오 | 성공 여부 | 최소 이격거리 (km) | 추정 이격거리 (km) | 제어 비용 (m/s²) | 계산 시간 (s) | 반복 횟수 |
|:--|---:|---:|---:|---:|---:|---:|
| 중심각 70 deg | 성공 | 145.00 † | — | 5.206 | 0.093 | 6 |
| 중심각 120 deg (기준) | 성공 | 15.61 | 16.13 | 4.649 | 0.107 | 8 |
| 중심각 135 deg | 성공 | 19.93 | 20.56 | 9.063 | 0.129 | 11 |
| 중심각 170 deg | 성공 | 31.89 | 32.39 | 21.166 | 0.133 | 12 |
| 궤도면 변경 | 성공 | 18.08 | 18.51 | 7.644 | 0.121 | 10 |

> † 출발점에서 장애물과의 이격거리가 가장 짧은 경우이다.

다섯 시나리오 모두 명제 1의 인증 조건을 만족하였고, 3.3절의 정지점 조건으로 종료하였다. 중심각 70 deg를 제외한 네 시나리오에서는 회피 제약이 작동하였다. 이때 최소 이격거리는 추정값의 96~99%로, 분할 수를 고정하고 시나리오를 바꾼 경우에도 3.1절의 보수성 예측과 일치하였다. 출발 궤도면을 벗어나는 궤도면 변경 시나리오에서도 같은 경향을 확인하였다. 중심각 170 deg에서는 직선 초기 궤적이 장애물 경계에서 안쪽으로 약 5900 km까지 진입하였으나, 여유 변수를 적용하여 인증 조건을 만족하는 해를 얻었다.

중심각 70 deg에서는 요구 중심각이 자유 낙하 각거리보다 작아 궤적이 장애물에서 멀어지고, 회피 제약이 비활성화되었다. 출발점에서 장애물과의 이격거리가 가장 짧으므로 추정 이격거리는 제시하지 않았다. 분할 수를 8, 16, 32로 바꾸었을 때 해의 차이는 $10^{-11}$ km 이내였다. 즉 회피 제약이 비활성인 경우에는 분할 수가 해에 영향을 주지 않았다. 제어 비용도 중심각 자체보다는 자유 낙하 각거리와의 차이에 따라 달라지므로, 중심각에 대해 단조적으로 변하지 않았다.

### 5.2 분할 수에 따른 변화

분할 수에 따른 결과를 표 3과 [그림 5](#fig-subdivision-tradeoff)에 제시하였다.

**표 3 [T3]. 분할 수에 대한 비교 실험 결과 (중심각 120 deg, $N=7$)**

| $n_{\mathrm{seg}}$ | 성공 여부 | 최소 이격거리 (km) | 추정 이격거리 (km) | 제어 비용 (m/s²) | 계산 시간 (s) | 반복 횟수 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 실패 | 145.00 † | — | 96.827 | 3.439 | 1000 |
| 4 | 성공 | 145.00 † | — | 8.159 | 0.101 | 12 |
| 8 | 성공 | 63.22 | 62.07 | 4.868 | 0.094 | 8 |
| 16 | 성공 | 15.61 | 16.13 | 4.649 | 0.107 | 8 |
| 32 | 성공 | 3.88 | 4.03 | 4.598 | 0.138 | 8 |
| 64 | 성공 | 0.97 | 1.01 | 4.585 | 0.216 | 8 |

> † 표 2와 같이 출발점에서 장애물과의 이격거리가 가장 짧은 경우이다. $n_{\mathrm{seg}}=2$에서는 조밀 표본 검사에서 장애물 침범이 관찰되지 않았으나, 반복 한도까지 인증 조건($h \le 10^{-6}$ km)을 만족하지 못하였다($h = 6.5 \times 10^{-6}$ km).

분할 수가 지나치게 작으면 명제 1의 충분조건과 경계조건을 동시에 만족하는 제어점의 범위가 제한된다. $n_{\mathrm{seg}}=2$에서는 여유 변수가 남아 반복 한도까지 인증에 실패하였고, 제어 비용도 다른 설정보다 한 자릿수 이상 컸다. 표본 검사에서 침범이 관찰되지 않았더라도, 인증 조건을 만족하지 못한 해에는 명제 1의 연속시간 회피 보장을 적용할 수 없다.

$n_{\mathrm{seg}} \ge 8$에서는 모든 해가 인증 조건을 만족하였으며, 분할 수에 따라 최소 이격거리와 제어 비용이 단조 감소하였다. 이 범위에서 이격거리의 감소는 회피 제약의 보수성이 줄었음을 보여준다. 분할 수를 8에서 64로 늘리면 최소 이격거리는 약 65분의 1로 줄었으나, 제어 비용의 감소율은 5.8%였다. 분할 수가 클수록 제어 비용의 추가 감소 폭은 작아졌다. 이 범위에서는 반복 횟수가 같고 계산 시간은 분할 수에 따라 증가하여, 보수성 감소에 따른 계산 비용의 증가를 확인하였다.

최소 이격거리는 분할 수를 두 배로 늘릴 때마다 약 4분의 1로 감소하여, 3.1절의 $n_{\mathrm{seg}}^{-2}$ 예측을 따랐다. 각 분할구간의 현 길이 $L_{\mathrm{seg}}$로 계산한 추정 이격거리 $\frac{L_{\mathrm{seg}}^2}{8R_{\mathrm{KOZ}}}$에 대한 측정값의 비는 0.96~1.02였다. 이 비교는 $n_{\mathrm{seg}} \ge 8$에 해당한다. $n_{\mathrm{seg}} \le 4$에서는 출발점 $\tau = 0$에서 장애물과의 이격거리가 가장 짧았고 그 값은 모두 145.00 km였으므로, 이를 보수성 비교에 사용하지 않았다.

<a id="fig-subdivision-tradeoff"></a>
![그림 5. 분할 수에 따른 계산 시간 및 결과 추세](../figures/subdivision_tradeoff_N7.png)
**그림 5 [F5].** 중심각 120 deg, $N=7$, $n_{\mathrm{seg}}=8,16,32,64$에서 분할 수에 따른 (a) 최소 이격거리의 측정값과 추정값, (b) 제어 비용, (c) 계산 시간. $n_{\mathrm{seg}}=2,4$의 결과는 표 3에 제시하였다.

차수 $N=6,7,8$을 비교한 보조 실험에서는 다섯 시나리오 모두 $n_{\mathrm{seg}}\ge4$에서 인증에 성공하였으며, $n_{\mathrm{seg}}=2$에서는 실패하였다. 세 차수가 모두 인증된 25개 설정에서 목적함수 $J$는 차수에 따라 감소하였으나, $N=6$에서 $N=8$로 높였을 때 감소율은 0.006–1.123%였다. 평균 제어 가속도는 차수에 따라 일관되게 감소하지 않았다. 계산 시간은 대체로 증가하였으나, 일부 설정에서는 반복 횟수의 변화로 반대 경향을 보였다.

---

## 6. 결론

본 논문에서는 분할구간별 지지 반공간 제약을 이용한 Bézier 궤적 생성 기법을 제안하였다. 구형 장애물 회피의 충분조건을 제어점 공간에서 구성하고, 비선형 중력을 선형화한 제어 비용 목적함수와 결합하여 SCvx로 풀었다. 각 분할구간의 인증 조건을 만족하면 궤적 전 구간의 장애물 회피가 보장된다.

단순화된 궤도전이 문제의 다섯 시나리오에서 모두 인증된 궤적을 얻었다. 회피 제약이 작동하는 설정의 최소 이격거리는 추정값 $L_{\mathrm{seg}}^2/(8R_{\mathrm{KOZ}})$와 수 % 이내로 일치하였다. 충분히 분할된 범위에서는 분할 수에 따라 최소 이격거리와 제어 비용이 단조 감소하였다. 다만 분할 수 비교 실험에서 $n_{\mathrm{seg}}=2$인 경우에는 반복 한도까지 인증 조건을 만족하지 못하였다.

볼록 껍질에 의한 회피 조건은 충분조건이므로 보수성이 남으며, 분할 수를 늘려 이를 줄일 수 있다. 본 논문의 회피 보장은 구형 장애물과 고정 전이 시간에 한정되며, 실험 범위도 단일 궤도전이 문제의 다섯 시나리오로 제한된다. 직선 보간 초기화는 속도 경계조건에 따른 접선 길이와 끝점 사이 거리가 비슷한 규모일 때를 전제로 한다. 전이 시간이 길어져 이 조건을 벗어나면 다른 초기화가 필요하다. 또한 제어 가속도의 상한은 제약으로 부과하지 않았다. 향후에는 여러 Bézier 호를 이용한 타원 궤도 전이, 다양한 문제 설정, 전이 시간 최적화로 연구를 확장할 수 있다.

---

## 참고문헌

[1] Betts, J. T., "Survey of Numerical Methods for Trajectory Optimization," *Journal of Guidance, Control, and Dynamics*, Vol. 21, No. 2, 1998, pp. 193–207. doi:10.2514/2.4231

[2] Hargraves, C. R., and Paris, S. W., "Direct Trajectory Optimization Using Nonlinear Programming and Collocation," *Journal of Guidance, Control, and Dynamics*, Vol. 10, No. 4, 1987, pp. 338–342. doi:10.2514/3.20223

[3] Açıkmeşe, B., Carson, J. M., and Blackmore, L., "Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem," *IEEE Transactions on Control Systems Technology*, Vol. 21, No. 6, 2013, pp. 2104–2113. doi:10.1109/TCST.2012.2237346

[4] Mao, Y., Dueri, D., Szmuk, M., and Açıkmeşe, B., "Successive Convexification of Non-Convex Optimal Control Problems with State Constraints," *IFAC-PapersOnLine*, Vol. 50, No. 1, 2017, pp. 4063–4069. doi:10.1016/j.ifacol.2017.08.789

[5] Malyuta, D., Reynolds, T. P., Szmuk, M., Lew, T., Bonalli, R., Pavone, M., and Açıkmeşe, B., "Convex Optimization for Trajectory Generation: A Tutorial on Generating Dynamically Feasible Trajectories Reliably and Efficiently," *IEEE Control Systems Magazine*, Vol. 42, No. 5, 2022, pp. 40–113. doi:10.1109/MCS.2022.3187542

[6] Dueri, D., Mao, Y., Mian, Z., Ding, J., and Açıkmeşe, B., "Trajectory Optimization with Inter-Sample Obstacle Avoidance via Successive Convexification," *2017 IEEE 56th Annual Conference on Decision and Control (CDC)*, Melbourne, Australia, 2017, pp. 1150–1156. doi:10.1109/CDC.2017.8263811

[7] Elango, P., Luo, D., Kamath, A. G., Uzun, S., Kim, T., and Açıkmeşe, B., "Successive Convexification for Trajectory Optimization with Continuous-Time Constraint Satisfaction," arXiv:2404.16826, 2024. doi:10.48550/arXiv.2404.16826

[8] Han, S. P., and Mangasarian, O. L., "Exact Penalty Functions in Nonlinear Programming," *Mathematical Programming*, Vol. 17, No. 1, 1979, pp. 251–269. doi:10.1007/BF01588250

[9] Nocedal, J., and Wright, S. J., *Numerical Optimization*, 2nd ed., Springer, New York, 2006, Theorem 17.3. doi:10.1007/978-0-387-40065-5

[10] Rimon, E., and Koditschek, D. E., "The Construction of Analytic Diffeomorphisms for Exact Robot Navigation on Star Worlds," *Transactions of the American Mathematical Society*, Vol. 327, No. 1, 1991, pp. 71–116.

[11] Deits, R., and Tedrake, R., "Computing Large Convex Regions of Obstacle-Free Space Through Semidefinite Programming," *Algorithmic Foundations of Robotics XI (WAFR)*, Springer, 2015, pp. 109–124.

[12] Marcucci, T., Petersen, M., von Wrangel, D., and Tedrake, R., "Motion Planning around Obstacles with Convex Optimization," *Science Robotics*, Vol. 8, No. 84, 2023.

[13] Kielas-Jensen, C., and Cichella, V., "Bernstein Polynomial-Based Transcription Method for Solving Optimal Trajectory Generation Problems," arXiv:2010.09992, 2020.

[14] Lee, S., and Kim, Y., "Optimal Output Trajectory Shaping Using Bézier Curves," *Journal of Guidance, Control, and Dynamics*, Vol. 44, No. 5, 2021, pp. 1027–1035. doi:10.2514/1.G005887

[15] Preiss, J. A., Hönig, W., Ayanian, N., and Sukhatme, G. S., "Downwash-Aware Trajectory Planning for Large Quadrotor Teams," *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2017, pp. 250–257.

[16] Tordesillas, J., and How, J. P., "MADER: Trajectory Planner in Multiagent and Dynamic Environments," *IEEE Transactions on Robotics*, Vol. 38, No. 1, 2022, pp. 463–476.

[17] Gao, F., Wang, L., Zhou, B., Han, L., Pan, J., and Shen, S., "Teach-Repeat-Replan: A Complete and Robust System for Aggressive Flight in Complex Environments," *IEEE Transactions on Robotics*, Vol. 36, No. 5, 2020, pp. 1526–1545.

[18] 최지웅, 이수원, "베지어 곡선 기반 컨벡스 궤적 최적화 및 순차 볼록 프로그래밍 충돌 회피 정식화," *한국항공우주학회지*, 2026, 게재 승인 원고(2026년 5월 14일 승인).
