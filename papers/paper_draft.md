# 분할구간별 볼록화에 기반한 Bézier 궤적 생성 기법

---

**초록**

본 연구에서는 비선형 중력장의 궤도전이에서 구형 장애물을 회피하는 Bézier 궤적 생성 기법을 제안한다. 곡선을 분할하고 분할구간마다 하나의 지지 반공간에 모든 제어점이 속하도록 제약하여 궤적 생성 문제를 순차 볼록화로 풀며, 이 제약을 만족하면 궤적 전 구간의 장애물 회피가 보장된다. 중력은 구간별로 선형화하여 제어 비용을 볼록 이차형식으로 구성하고, 회피 제약의 선형화에는 제어점 이동에 따른 반공간 법선의 변화를 포함한다.

중심각과 궤도면을 달리한 전이 시나리오와 기준 시나리오 주변에서 전이 조건을 교란한 몬테카를로 분석에서 모든 사례가 최적해에 수렴하였다. 장애물 경계 부근의 최소 이격거리는 원호 근사로 추정한 값에 가까웠고 분할 수의 제곱에 반비례하여 줄었다. 분할 수가 적으면 제어 비용이 크게 늘었고, 분할 수를 늘리면 제어 비용은 가장 세밀한 분할의 값에 빠르게 가까워졌다. 반면 회피 제약은 분할구간마다 부과되므로 그 수는 분할 수에 비례하여 늘어난다. 반공간 법선을 고정하여 선형화하면 대부분의 사례가 수렴하지 못하였다.

---

## 1. 서론

항공우주, 로봇공학, 자율 시스템의 궤적 최적화에서는 경로 전 구간에 걸친 장애물 회피가 요구된다. 그러나 direct transcription과 direct collocation [1, 2]은 주로 이산화된 노드에 제약을 부과하므로, 노드에서 제약을 만족하더라도 노드 사이에서는 제약을 위반할 수 있다 [3]. 궤적 전 구간에서 제약을 만족하는 성질을 연속시간 제약 만족 [4]이라 하며, 이를 보장할 수 있는 궤적 표현과 제약 정식화가 필요하다. Bézier 곡선의 제어점을 최적화 변수로 사용하는 궤적 생성 기법은 여러 연구에서 다루어졌다 [5, 6].

장애물 외부의 자유 공간은 볼록 집합이 아니므로, 장애물 회피 제약은 비볼록 제약이다. 궤적 최적화에서 비볼록 제약을 다루는 방법으로는 특정 문제에 적용할 수 있는 무손실 볼록화 [7]와 제약을 반복적으로 선형화하는 순차 볼록화(successive convexification, SCvx) [8, 9]가 있다. 장애물 회피를 볼록 최적화로 다루는 기법은 주로 두 가지 접근을 취한다. 첫째는 자유 공간을 여러 볼록 영역으로 덮고 통과할 영역을 선택하는 방법으로, 볼록 영역 생성 [10]과 볼록 집합 그래프 [11]가 이에 해당한다. 둘째는 장애물의 통과 방향을 정하고 지지 반공간으로 자유 공간을 제한하는 방법이다. 반공간을 고정하면 회피 제약은 선형 부등식이 된다. 그러나 반공간을 궤적에 맞추어 정하면 궤적이 바뀔 때마다 반공간도 다시 정해야 하므로, 반공간을 갱신하며 제약을 반복하여 선형화하는 SCvx를 적용할 수 있다.

두 번째 방법에서는 볼록 껍질 성질을 이용하여 제어점에 대한 조건으로 연속시간 제약 만족을 보장한다 [5]. 이 보장은 곡선을 나눈 한 분할구간의 모든 제어점이 같은 반공간 안에 있을 때 성립하며, 제어점마다 다른 반공간을 적용하면 볼록 껍질이 장애물과 겹칠 수 있다. 따라서 안전 회랑 기법 [12, 13, 14]은 분할구간마다 하나의 반공간을 정해 그 분할구간의 모든 제어점에 부과하며, 본 연구도 이 방식을 따른다. 다만 제어점의 볼록 껍질은 곡선보다 크므로, 이 조건을 만족하는 곡선은 장애물에서 필요 이상으로 떨어질 수 있다. 그 정도를 회피 제약의 보수성이라 한다.

[15]에서는 미분·차수 상승 연산자와 Gram 행렬을 이용하여 제어점에 대한 이차형식 비용함수를 구성하고, 곡선 분할과 분리 평면 제약을 순차 볼록 프로그래밍에 적용하였다. 본 연구에서는 이 정식화를 바탕으로 구형 장애물 회피 제약의 보수성을 정량적으로 분석하고, 비선형 중력을 고려한 궤도전이 문제로 확장한다. 주요 내용은 다음과 같다. 첫째, 장애물 경계 부근의 최소 이격거리를 원호 근사로 추정한다. 또한 분할 수를 늘릴 때 보수성과 제어 비용이 줄어드는 정도를 실험으로 확인하고, 이를 분할 수에 비례하여 늘어나는 회피 제약의 수와 비교한다. 둘째, 선행연구 [15]의 선형 Clohessy–Wiltshire(CW) 상대운동 모델을 비선형 중력 모델로 확장한다. 이체 중력과 $J_2$ 섭동을 구간별로 선형화하여 기존의 Gram 행렬 정식화로 제어 가속도 비용을 볼록 이차형식으로 구성한다. 셋째, 제어점 변화에 따른 지지 반공간 법선의 변화까지 미분에 포함한다. 이를 신뢰영역 기반 SCvx [8, 9]의 회피 제약 선형화와 감소량 평가에 반영하고, 법선을 고정한 경우와 비교하여 그 효과를 확인한다.

2절에서는 제어점 공간의 궤적 표현과 표기법을 정리하고, 3절에서는 제안 기법의 정식화와 알고리즘을 설명한다. 4절과 5절에서는 각각 실험 설정과 수치 결과를 제시하며, 6절에서 결론과 한계를 기술한다.

---

## 2. 제어점 공간에서의 궤적 표현

선행연구 [15]의 제어점 공간 정식화에 따라 필요한 표기와 식을 정리한다. 미분·차수 상승 연산자와 Gram 행렬의 기본 성질은 문헌 [5, 15]을 따른다.

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

미분 행렬 $D_N\in\mathbb{R}^{N\times(N+1)}$과 차수 상승 행렬 $E_{N-1}\in\mathbb{R}^{(N+1)\times N}$을 이용하여 차수를 보존하는 속도·가속도 연산자를 정의한다 [15].

$$
L_{1,N} = E_{N-1}D_N, \qquad L_{2,N} = E_{N-1}D_N E_{N-1}D_N
$$

물리 시간에 대한 속도·가속도 제어점은 각각 $L_{1,N}P/T$, $L_{2,N}P/T^2$이다. 본 연구에서 사용하는 끝점 위치·속도 조건은

$$
\mathbf{p}_0=\mathbf{r}_0, \qquad \mathbf{p}_N=\mathbf{r}_f, \qquad
\frac{N}{T}(\mathbf{p}_1-\mathbf{p}_0)=\mathbf{v}_0, \qquad
\frac{N}{T}(\mathbf{p}_N-\mathbf{p}_{N-1})=\mathbf{v}_f
$$

이며, 이를 $A_{\mathrm{bc}}\mathbf{x}=\mathbf{b}_{\mathrm{bc}}$ 형태로 정리할 수 있다. 여기서 $\mathbf{r}_0$, $\mathbf{r}_f$, $\mathbf{v}_0$, $\mathbf{v}_f$는 각각 주어진 초기·최종 위치와 속도이다.

### 2.3 Gram 행렬과 이차형식

본 연구의 목적함수는 곡선을 따라 정의된 벡터값 함수의 크기를 제곱하여 적분한 값이다. 이러한 적분은 Bernstein 기저다항식 사이의 내적을 모은 Gram 행렬 $G_N$을 이용하면 곡선의 제어점만으로 계산할 수 있다 [5, 15]. $G_N$의 성분과 적분 항등식은 다음과 같다.

$$
[G_N]_{il} = \frac{\binom{N}{i}\binom{N}{l}}{\binom{2N}{i+l}(2N+1)}, \qquad i,l=0,\ldots,N
$$

$$
\int_0^1 \|\mathbf{f}(\tau)\|_2^2\,d\tau = \mathrm{tr}(F^{\mathsf{T}}G_NF), \qquad
\mathbf{f}(\tau)=\sum_{i=0}^{N}B_i^N(\tau)\mathbf{f}_i, \qquad
F=[\mathbf{f}_0,\ldots,\mathbf{f}_N]^{\mathsf{T}}
$$

$F$가 결정 변수 $\mathbf{x}$의 1차 함수이면 위 적분은 $\mathbf{x}$에 대한 볼록 이차형식으로 표현되며, 수치 적분 없이 정확히 계산할 수 있다.

---

## 3. 순차 볼록화 기반 궤적 생성

### 3.1 분할과 지지 반공간을 이용한 구형 장애물 처리

곡선 분할과 분리 평면을 이용한 회피 제약은 [15]에서의 표현을 따른다. 중심이 $\mathbf{c}_{\mathrm{KOZ}}$이고 반경이 $R_{\mathrm{KOZ}}$인 구형 장애물을 다음과 같이 정의한다.

$$
\mathcal{K} = \left\{\mathbf{r}\in\mathbb{R}^3 : \|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le R_{\mathrm{KOZ}} \right\}
$$


연속시간 장애물 회피 조건은 모든 $\tau \in [0,1]$에서 $\mathbf{r}(\tau) \notin \operatorname{int}\mathcal{K}$, 즉 $\|\mathbf{r}(\tau)-\mathbf{c}_{\mathrm{KOZ}}\|_2 \ge R_{\mathrm{KOZ}}$를 만족하는 것이다. 이는 곡선 위의 이산 노드에만 부과하는 조건보다 강하다. 전이 시간 $T$가 고정되어 $t$와 $\tau$가 일대일로 대응하므로, $\tau$ 전 구간에서 조건을 만족하면 물리 시간 전 구간에서도 조건을 만족한다.

곡선을 $n_{\mathrm{seg}}$개로 등분하는 De Casteljau 분할 행렬을 $S^{(s)}$라 하면 $s$번째 분할구간의 제어점 행렬은

$$
P^{(s)} = S^{(s)}P
$$

가 된다. 각 행은 분할구간의 제어점 $\mathbf{q}^{(s)}_0, \ldots, \mathbf{q}^{(s)}_N$에 해당한다. 분할된 제어점은 $P$에서 계산되므로 독립적인 최적화 변수가 아니다.

분할구간 제어점의 무게중심을 다음과 같이 정의한다.

$$
\mathbf{c}^{(s)} = \frac{1}{N+1}\sum_{m=0}^{N}\mathbf{q}^{(s)}_m
$$

[그림 1](#fig-ctrl-subdivision)에서 보듯이, 분할 후 제어점의 볼록 껍질은 곡선에 더 가까워진다. 따라서 분할 수 $n_{\mathrm{seg}}$를 늘리면 같은 차수에서 회피 제약의 보수성을 줄일 수 있다.

<a id="fig-ctrl-subdivision"></a>
![그림 1. De Casteljau 분할에 따른 제어점과 볼록 껍질의 변화](../figures/control_subdivision.png)
**그림 1 [F1].** De Casteljau 분할에 따른 제어점과 볼록 껍질의 변화.

무게중심 $\mathbf{c}^{(s)}$가 장애물 중심 $\mathbf{c}_{\mathrm{KOZ}}$와 다르면, 장애물 중심에서 무게중심을 향하는 외향 법선은

$$
\mathbf{n}^{(s)} = \frac{\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}}{\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\|_2}
$$

로 정의한다. 이에 따라 구의 지지 반공간은

$$
\mathcal{H}^{(s)} = \left\{\mathbf{r} : (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{r} \ge (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{c}_{\mathrm{KOZ}} + R_{\mathrm{KOZ}} \right\}
$$

로 쓸 수 있다. 이 반공간의 경계면은 구에 접하며, 이를 지지평면이라 한다.

각 분할구간의 모든 제어점이 이 반공간에 속하도록 다음 제약을 부과한다.

$$
(\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{q}^{(s)}_m \ge (\mathbf{n}^{(s)})^{\mathsf{T}} \mathbf{c}_{\mathrm{KOZ}} + R_{\mathrm{KOZ}}, \qquad m=0,\ldots,N
$$

볼록 껍질 성질 [5]에 따른 회피 충분조건은 다음 명제 1로 정리할 수 있다.

> **명제 1.** 분할구간 $s$의 제어점을 $P^{(s)} = S^{(s)}P$, 구형 장애물을 $\mathcal{K} = \{\mathbf{r}\in\mathbb{R}^3 : \|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le R_{\mathrm{KOZ}}\}$라 하자. 모든 제어점 $\mathbf{q}^{(s)}_0, \ldots, \mathbf{q}^{(s)}_N$이 위에서 정의한 지지 반공간 $\mathcal{H}^{(s)}$에 속하면, 해당 분할구간의 Bézier 곡선 전체도 $\mathcal{H}^{(s)}$에 속하므로 $\operatorname{int}\mathcal{K}$와 만나지 않는다.

> **가정.** 명제 1의 적용 조건은 다음과 같다.
>
> 1. 장애물은 구형이다.
> 2. 해당 분할구간의 모든 제어점에 동일한 지지 반공간 $\mathcal{H}^{(s)}$의 제약을 부과한다.

> **증명.** 곡선은 제어점의 볼록 껍질 안에 있고 [5], $\mathcal{H}^{(s)}$는 볼록이며 장애물 내부와 만나지 않는다. 따라서 모든 제어점이 $\mathcal{H}^{(s)}$ 안에 있으면 곡선 전체도 장애물 내부와 만나지 않는다. ($\square$)

[그림 2](#fig-subdivision)는 명제 1을 하나의 분할구간에 적용한 예이다. 그림 2(a)에서는 분할구간 $s$의 곡선이 장애물 내부를 지난다. (b)는 이 분할구간 제어점의 무게중심에서 법선을 정하고 구에 접하는 지지 반공간을 구성한 것이다. (c)는 모든 제어점이 지지 반공간 안에 있어 분할구간의 곡선 전체가 장애물 밖에 놓인 경우이다.

<a id="fig-subdivision"></a>
![그림 2. 분할과 지지 반공간을 이용한 연속시간 장애물 회피 제약 만족의 개념도](../figures/koz_linearization.png)
**그림 2 [F2].** 분할구간별 회피 제약: (a) 장애물 침범 구간, (b) 지지 반공간 구성, (c) 회피 조건을 만족하는 구간.

분할구간 제어점 $\mathbf{q}^{(s)}_m$은 원래 제어점의 선형결합이다. 따라서 법선 $\mathbf{n}^{(s)}$을 고정하면 위 부등식은 결정 변수 $\mathbf{x}$에 대해 선형이다. 그러나 법선도 무게중심 $\mathbf{c}^{(s)}$를 통해 $\mathbf{x}$에 의존한다. 명제 1의 조건에 대한 위반량을 다음과 같이 정의한다.

$$
h(\mathbf{x}) = \sum_{s}\sum_{m=0}^{N}\max\left\{0,\ R_{\mathrm{KOZ}} + \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\mathbf{c}_{\mathrm{KOZ}} - \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\mathbf{q}^{(s)}_m(\mathbf{x})\right\}
$$

$h(\mathbf{x})$는 법선의 정규화 $\mathbf{v}\mapsto\mathbf{v}/\|\mathbf{v}\|_2$ 때문에 비볼록이다. $h(\mathbf{x}) = 0$이면 명제 1에 따라 곡선 전체의 장애물 회피가 보장된다.

법선을 어떤 단위벡터로 정하더라도 지지평면은 구에 접하므로 명제 1의 회피 보장이 성립한다. 다만 법선의 방향에 따라 제약의 보수성이 달라진다.

분할구간의 두 끝점 사이 거리, 즉 현의 길이를 $L_{\mathrm{seg}}$로 정의한다. 장애물 경계에 가까운 짧은 분할구간을 장애물과 중심이 같은 원호로 근사하고, 두 끝점이 원호 중점 방향의 지지평면에 놓인다고 가정한다. 이때 원호와 장애물 사이의 이격거리는

$$
\delta = \sqrt{R_{\mathrm{KOZ}}^2 + \frac{L_{\mathrm{seg}}^2}{4}} - R_{\mathrm{KOZ}}
\approx \frac{L_{\mathrm{seg}}^2}{8R_{\mathrm{KOZ}}}
$$

이다. 마지막 근사는 $L_{\mathrm{seg}} \ll R_{\mathrm{KOZ}}$일 때 성립하며, 현의 길이가 분할 수에 반비례하면 이 추정값은 $n_{\mathrm{seg}}^{-2}$에 비례하여 감소한다. 이 값은 5절에서 보수성을 비교하기 위한 추정값이며, 최종 궤적의 장애물 회피는 명제 1의 조건으로 판정한다.

### 3.2 제어 비용 목적함수

장애물을 회피하는 데 필요한 제어 비용을 줄이기 위해 제어 가속도에 대한 목적함수를 구성한다. 궤적을 따르는 데 필요한 제어 가속도는 궤적의 가속도와 중력 가속도의 차이로 정의한다.

$$
\mathbf{u}(t) = \ddot{\mathbf{r}}(t) - \mathbf{g}(\mathbf{r}(t))
$$

여기서 $\mathbf{g}$는 중력 가속도이며, 이체 문제 항과 $J_2$ 섭동 항을 포함한다. $\mathbf{r} = (x, y, z)$, $r = \|\mathbf{r}\|_2$로 두면 다음과 같다.

$$
\mathbf{g}(\mathbf{r}) = -\frac{\mathrm{GM}}{r^{3}}\mathbf{r}
+ \frac{3 J_2 \mathrm{GM} R_\oplus^{2}}{2 r^{5}}
\begin{bmatrix}
\left(\dfrac{5z^{2}}{r^{2}} - 1\right) x \\[2pt]
\left(\dfrac{5z^{2}}{r^{2}} - 1\right) y \\[2pt]
\left(\dfrac{5z^{2}}{r^{2}} - 3\right) z
\end{bmatrix}
$$

$\mathrm{GM}$은 지구 중력상수, $R_\oplus$는 지구 반경, $J_2$는 2차 대상 조화항 계수이다. 지구의 대칭축은 좌표계의 $z$축과 일치한다고 가정한다.

> 중력 상수: $\mathrm{GM} = 3.986004418 \times 10^{5}$ km$^3$/s$^2$, $R_\oplus = 6371$ km, $J_2 = 1.08262668 \times 10^{-3}$.

목적함수는 제어 가속도 크기의 제곱을 궤적 전 구간에서 적분한 값으로 정의한다.

$$
J(\mathbf{x}) = \int_0^1 \left\| \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2} - \mathbf{g}(\mathbf{r}(\tau)) \right\|_2^2 d\tau
$$

중력 가속도가 위치에 대해 비선형이므로 목적함수 $J$는 $\mathbf{x}$에 대해 비볼록이다. 볼록 이차형식의 목적함수를 얻기 위해 곡선을 $n_{\mathrm{lin}}$개 구간으로 등분하고, 각 구간에서 중력을 1차 테일러 전개로 근사한다. 이때 사용하는 De Casteljau 분할 행렬은 장애물 회피용 $S^{(s)}$와 구분하여 $\hat S^{(j)}$로 표기한다. 장애물 회피용 분할 수 $n_{\mathrm{seg}}$는 회피 제약의 보수성을, 중력 선형화 구간 수 $n_{\mathrm{lin}}$은 목적함수 근사의 정확도를 결정한다. 따라서 두 값은 독립적으로 설정한다. 반복 $k$에서 구간 $j$의 제어점 무게중심을 $\mathbf{r}_j^{(k)}$라 하면, 이 점에서 중력을 다음과 같이 선형화한다.

$$
\mathbf{g}(\mathbf{r}) \approx \nabla\mathbf{g}_j^{(k)}\mathbf{r} + \mathbf{c}_j^{(k)}, \qquad \mathbf{c}_j^{(k)} = \mathbf{g}\!\left(\mathbf{r}_j^{(k)}\right) - \nabla\mathbf{g}_j^{(k)}\mathbf{r}_j^{(k)}
$$

여기서 $\nabla\mathbf{g}_j^{(k)} = \partial\mathbf{g}/\partial\mathbf{r}$은 $\mathbf{r}_j^{(k)}$에서 계산한 중력 Jacobian이다. 이 분할은 중력을 선형화하는 점을 곡선을 따라 배치하기 위한 것으로, 적분을 이산화하지는 않는다. 각 구간의 중력은 위치에 대한 1차 함수로 근사된다.

구간 $j$를 $\xi \in [0,1]$로 다시 매개화하고, 궤적의 가속도와 선형화된 중력 가속도의 차이를 잔차로 정의한다.

$$
\mathbf{f}^{(j)}(\xi) = \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2} - \left(\nabla\mathbf{g}_j^{(k)}\mathbf{r} + \mathbf{c}_j^{(k)}\right)
$$

우변은 $\xi$에 대응하는 $\tau$에서 평가한다. 두 항은 모두 $\xi$에 대한 차수 $N$의 Bézier 곡선으로 표현된다. 분할 행렬 $\hat S^{(j)}$와 2.2절의 가속도 연산자를 적용하면 각 항의 제어점은 $\mathbf{x}$의 1차 함수가 된다. 따라서 잔차 $\mathbf{f}^{(j)}$에도 2.3절의 적분 항등식을 적용할 수 있다. 구간별 적분을 합하면 다음을 얻는다.

$$
J^{(k)}(\mathbf{x}) = \sum_{j=1}^{n_{\mathrm{lin}}} \frac{1}{n_{\mathrm{lin}}}\int_0^1 \left\|\mathbf{f}^{(j)}(\xi)\right\|_2^2 d\xi = \sum_{j=1}^{n_{\mathrm{lin}}} \frac{1}{n_{\mathrm{lin}}}\,\mathrm{tr}\!\left(F_j(\mathbf{x})^{\mathsf{T}} G_N F_j(\mathbf{x})\right)
$$

여기서 $1/n_{\mathrm{lin}}$은 매개변수 변환에 따른 계수이며, $F_j(\mathbf{x})$는 잔차 $\mathbf{f}^{(j)}$의 제어점 행렬이다.

$J^{(k)}$는 $\mathbf{x}$에 대한 볼록 이차형식이므로 SCvx의 볼록 이차 계획(quadratic programming, QP) 문제의 목적함수로 사용할 수 있다. 선형화된 잔차의 적분은 해석적으로 정확히 계산되며, 원래 목적함수 $J$와의 차이는 중력 선형화에서 발생한다.

### 3.3 볼록 부문제와 SCvx 알고리즘

반복 $k$에서는 그 반복의 제어점 $\mathbf{x}^{(k)}$를 기준으로 회피 제약과 중력을 선형화하며, 이 점을 기준점이라 한다. 반공간은 매 반복의 기준점에서 다시 구성하므로, 부문제의 회피 제약은 기준점 주변의 국소 근사에 의존한다. 중력의 선형화도 기준점 주변에서만 유효하다. 따라서 신뢰영역 기반 SCvx [8, 9]로 볼록 부문제를 반복하여 풀며, 전체 절차는 Algorithm 1과 같다. 첫 반복의 기준점 $\mathbf{x}^{(0)}$로는 두 끝점을 잇는 선분 위에 제어점을 같은 간격으로 놓은 직선 초기 궤적을 사용한다.

SCvx 반복 $k$의 부문제는 다음의 볼록 QP이다.

$$
\min_{\mathbf{x},\,\boldsymbol{\nu}} \ \frac{1}{2}\mathbf{x}^{\mathsf{T}} H^{(k)}\mathbf{x} + (\boldsymbol{\ell}^{(k)})^{\mathsf{T}}\mathbf{x} + \mu\,\mathbf{1}^{\mathsf{T}}\boldsymbol{\nu}
$$

$$
A_{\mathrm{KOZ}}^{(k)}\mathbf{x} + \boldsymbol{\nu} \ge \mathbf{b}_{\mathrm{KOZ}}^{(k)}, \quad \boldsymbol{\nu} \ge \mathbf{0}, \qquad
A_{\mathrm{bc}}\mathbf{x} = \mathbf{b}_{\mathrm{bc}}, \qquad
\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \le \Delta_k
$$

$H^{(k)}$와 $\boldsymbol{\ell}^{(k)}$는 3.2절의 목적함수 $J^{(k)}$를 $\mathbf{x}$에 대해 전개한 행렬과 벡터이고, $\boldsymbol{\nu}$는 선형화된 회피 제약마다 하나씩 두는 여유 변수를 모은 벡터이다. $\mu$는 여유 변수에 대한 페널티 계수이고, $\Delta_k$는 신뢰영역 크기이다. 해에 영향을 주지 않는 상수항은 생략하였다. 대표 설정 $N = 7$, $n_{\mathrm{seg}} = 16$에서는 궤적 변수 $\mathbf{x}$의 성분이 24개이며, 그중 12개는 경계조건으로 고정된다. 장애물 회피 제약은 $n_{\mathrm{seg}}(N+1) = 128$개이다. 분할 수를 늘리면 회피 제약과 여유 변수의 수가 함께 증가한다.

장애물 회피 제약의 행렬 $A_{\mathrm{KOZ}}^{(k)}$와 벡터 $\mathbf{b}_{\mathrm{KOZ}}^{(k)}$는 3.1절의 조건을 기준점 $\mathbf{x}^{(k)}$에서 1차 전개하여 구한다. 분할구간 제어점 $\mathbf{q}^{(s)}_m$의 지지평면에 대한 여유를

$$
\gamma^{(s)}_m(\mathbf{x}) = \left(\mathbf{n}^{(s)}(\mathbf{x})\right)^{\mathsf{T}}\!\left(\mathbf{q}^{(s)}_m(\mathbf{x}) - \mathbf{c}_{\mathrm{KOZ}}\right) - R_{\mathrm{KOZ}}
$$

로 정의한다. 분할 행렬 $S^{(s)}$와 무게중심 가중치 $w^{(s)}_i = \frac{1}{N+1}\sum_{m=0}^{N} S^{(s)}_{mi}$를 이용하면 기울기는 다음과 같다.

$$
\frac{\partial \gamma^{(s)}_m}{\partial \mathbf{p}_i} = S^{(s)}_{mi}\,\mathbf{n}^{(s)} + \frac{w^{(s)}_i}{\left\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\right\|_2}\left(I_3 - \mathbf{n}^{(s)}(\mathbf{n}^{(s)})^{\mathsf{T}}\right)\!\left(\mathbf{q}^{(s)}_m - \mathbf{c}_{\mathrm{KOZ}}\right)
$$

첫째 항은 법선을 고정했을 때의 기울기이며, 둘째 항은 제어점 이동에 따른 무게중심과 법선의 변화를 반영한다. 둘째 항의 사영 행렬 $I_3 - \mathbf{n}^{(s)}(\mathbf{n}^{(s)})^{\mathsf{T}}$은 지지평면에 평행한 성분만 남긴다. 따라서 둘째 항은 지지평면이 구에 닿는 접점에서 0이고, 제어점이 지지평면을 따라 접점에서 멀수록 커진다. 이 기울기를 이용하여 회피 제약을 다음과 같이 선형화한다.

$$
\gamma^{(s)}_m\!\left(\mathbf{x}^{(k)}\right) + \sum_{i=0}^{N}\left(\frac{\partial \gamma^{(s)}_m}{\partial \mathbf{p}_i}\right)^{\!\mathsf{T}}\!\left(\mathbf{p}_i - \mathbf{p}^{(k)}_i\right) + \nu^{(s)}_m \ \ge\ 0
$$

초기 궤적이 장애물 내부를 지나는 경우에는 선형화된 제약을 만족하는 해가 없을 수 있다. 따라서 $\boldsymbol{\nu}$의 성분 $\nu^{(s)}_m$로 선형화된 회피 제약을 완화하며, 이는 SCvx의 virtual control [8, 9]에 해당한다.

부문제에서 구한 해 $\hat{\mathbf{x}}$의 수용 여부는 목적함수와 제약 위반량을 결합한 merit function으로 판단한다 [8, 9].

$$
\phi(\mathbf{x}) = J(\mathbf{x}) + \mu\,h(\mathbf{x})
$$

$J$는 비선형 중력을 사용한 3.2절의 목적함수이며, $h$는 해당 제어점 $\mathbf{x}$에서 다시 구성한 지지 반공간 제약의 위반량이다. 볼록 QP의 목적함수는 이 둘을 근사한 $\phi^{(k)} = J^{(k)} + \mu h^{(k)}$에 해당한다. $J^{(k)}$는 중력을 구간별로 선형화한 목적함수이고, $h^{(k)}$는 명제 1의 조건을 1차 전개한 근사의 위반량이다. 부문제에서는 이 위반량을 여유 변수의 합 $\mathbf{1}^{\mathsf{T}}\boldsymbol{\nu}$로 나타낸다.

여유 변수로 제약 위반을 허용하되, 페널티 계수 $\mu$를 충분히 크게 두어 수렴한 해에서는 $\boldsymbol{\nu} = \mathbf{0}$이 되도록 한다 [16, 17]. 신뢰영역 제약 $\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \le \Delta_k$는 제어점의 변화 범위를 제한한다. 신뢰영역 크기 $\Delta_k$는 초기 크기 $\Delta_0$에서 시작하여 선형화의 정확도에 따라 매 반복마다 조절한다.

예측 감소량에는 2.3절의 적분 항등식으로 계산한 $J^{(k)}$를 사용한다. 실제 감소량을 평가할 때에는 비선형 중력과 선형화된 중력으로 계산한 잔차의 제곱노름 차이를 수치 적분하고, 이를 $J^{(k)}$에 더하여 $J$를 근사한다. 예측 감소량에 대한 실제 감소량의 비는 다음과 같다.

$$
\rho_k = \frac{\phi(\mathbf{x}^{(k)}) - \phi(\hat{\mathbf{x}})}{\phi^{(k)}(\mathbf{x}^{(k)}) - \phi^{(k)}(\hat{\mathbf{x}})}
$$

$\rho_k$는 근사 모델이 실제 감소량을 얼마나 정확히 예측하는지를 나타낸다. $\rho_k$에 따라 해를 수용하고 신뢰영역 크기를 조절하는 규칙을 [그림 3](#fig-acceptance)에 나타내었다. 직선 초기 궤적은 속도 경계조건을 만족하지 않으므로, 첫 반복에서는 $\rho_k$를 계산하지 않고 해를 수용하여 경계조건을 맞춘다.

<a id="fig-acceptance"></a>
![그림 3. 부문제 해의 수용과 신뢰영역 크기의 갱신](../figures/scp_pipeline.png)
**그림 3 [F3].** 부문제 해의 수용과 신뢰영역 크기의 갱신.

회피 제약 기울기의 둘째 항을 빼고 법선을 고정하면, 부문제의 해 $\hat{\mathbf{x}}$가 기준점에서 구성한 반공간을 만족하더라도 $\hat{\mathbf{x}}$에서 다시 구성한 반공간은 위반할 수 있다. 이 위반은 $h(\hat{\mathbf{x}})$에는 반영되지만 $h^{(k)}(\hat{\mathbf{x}})$에는 반영되지 않으므로, 실제 감소량이 예측 감소량보다 작아져 해가 기각되고 신뢰영역이 줄어든다.

merit function의 상대 변화나 예측 감소량이 허용오차보다 작고 명제 1의 회피 조건이 만족되는 상태가 연속 $n_{\mathrm{conv}} = 3$회 유지되면 SCvx 반복을 종료하며, 이를 종료 기준이라 한다. 연속 유지를 요구하는 것은 법선이 갱신되는 동안 제어점이 지지평면을 따라 천천히 움직이면서 merit function의 변화가 일시적으로 작아질 수 있기 때문이다. Algorithm 1의 $\sigma_1$과 $\sigma_2$는 각각 수용된 해의 merit function 상대 변화와 예측 감소량에 대해 이 상태가 연속된 횟수를 센다.

> **Algorithm 1** — SCvx 기반 궤적 생성
>
> 1. $\sigma_1 \leftarrow 0,\ \sigma_2 \leftarrow 0$
> 2. **for** $k = 0, 1, \ldots, 999$ **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp; $A_{\mathrm{KOZ}}^{(k)}, \mathbf{b}_{\mathrm{KOZ}}^{(k)} \leftarrow \gamma^{(s)}_m(\mathbf{x}^{(k)}),\ \partial\gamma^{(s)}_m/\partial\mathbf{p}_i$
> 4. &nbsp;&nbsp;&nbsp;&nbsp; $H^{(k)}, \boldsymbol{\ell}^{(k)} \leftarrow \nabla\mathbf{g}_j^{(k)}, \mathbf{c}_j^{(k)}$
> 5. &nbsp;&nbsp;&nbsp;&nbsp; $(\hat{\mathbf{x}}, \boldsymbol{\nu}) \leftarrow \mathrm{QP}(\mathbf{x}^{(k)}, \Delta_k)$
> 6. &nbsp;&nbsp;&nbsp;&nbsp; **if** $|\phi^{(k)}(\mathbf{x}^{(k)}) - \phi^{(k)}(\hat{\mathbf{x}})| < 10^{-8}|\phi(\mathbf{x}^{(k)})|$ **and** $h(\mathbf{x}^{(k)}) \le 10^{-6}$ km **then** $\sigma_2 \leftarrow \sigma_2 + 1$ **else** $\sigma_2 \leftarrow 0$
> 7. &nbsp;&nbsp;&nbsp;&nbsp; **if** $\sigma_2 = n_{\mathrm{conv}}$ **then return** $\mathbf{x}^{(k)}$
> 8. &nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{x}^{(k+1)}, \Delta_{k+1} \leftarrow$ 그림 3
> 9. &nbsp;&nbsp;&nbsp;&nbsp; **if** $\mathbf{x}^{(k+1)} = \hat{\mathbf{x}}$ **and** $|\phi(\mathbf{x}^{(k)}) - \phi(\hat{\mathbf{x}})| < 10^{-8}|\phi(\mathbf{x}^{(k)})|$ **and** $h(\hat{\mathbf{x}}) \le 10^{-6}$ km **then** $\sigma_1 \leftarrow \sigma_1 + 1$ **else** $\sigma_1 \leftarrow 0$
> 10. &nbsp;&nbsp;&nbsp;&nbsp; **if** $\sigma_1 = n_{\mathrm{conv}}$ **then return** $\mathbf{x}^{(k+1)}$
> 11. &nbsp;&nbsp;&nbsp;&nbsp; **if** $\Delta_{k+1} < 10^{-2}$ km **then return** 실패
> 12. **return** 실패

---

## 4. 실험 설정

단순화된 3차원 궤도전이 문제에서 제안 기법을 평가하였다. 초기·최종 위치와 속도를 고정하고, 지구 중심의 구형 장애물을 회피하는 궤적을 생성하였다. 장애물 반경은 $R_{\mathrm{KOZ}} = 6471$ km로 지표면 고도 100 km에 해당하며, 전이 시간은 $T = 1500$ s로 설정하였다. 이 전이 시간은 최소 제어 비용 궤적이 장애물 경계에 접근하여 회피 제약이 작동하도록 정하였다. 회피 제약이 작동한다는 것은 해에서 하나 이상의 지지 반공간 제약이 등호로 성립하는 것이며, 그렇지 않으면 회피 제약이 비활성이라 한다. 지구 중심에서 본 두 끝점 사이의 각을 중심각이라 할 때, 중심각이 자유 낙하로 이동하는 각거리인 약 100 deg보다 작으면 궤적이 장애물에서 멀어져 회피 제약이 비활성화된다.

중력 모델에는 3.2절의 이체 중력과 $J_2$ 섭동을 사용하였다. 중력 선형화 구간 수는 $n_{\mathrm{lin}} = 100$으로 두고, 각 구간의 제어점 무게중심에서 중력을 1차 근사하였다. 선형화된 잔차의 적분은 2.3절의 항등식으로 계산하였다. 기준 시나리오에서 구간 수를 200과 400으로 늘렸을 때 수렴한 목적함수 $J$의 상대 변화는 $10^{-9}$ 이내로, 중력 선형화 구간 수에 따른 차이는 미미하였다.

기준 시나리오는 Progress 우주선의 ISS 접근을 참고하여 두 원궤도 위의 점을 잇는 전이 문제로 단순화하였다. 출발·도착 고도는 Progress의 초기 원지점 고도 [18]와 ISS 운용 고도 범위 [19]를 참고하여 각각 245 km와 400 km로 정하였다. 두 궤도는 경사각 51.64 deg의 동일 평면에 있다. 두 끝점의 중심각은 120 deg이고, 속도 경계조건은 각 원궤도의 원운동 속도이다. 이 기준에서 중심각을 70~170 deg로 바꾸거나 도착 궤도면을 변경하여 표 1의 다섯 시나리오를 구성하였다.

**표 1 [T1]. 시험한 전이 시나리오**

| 시나리오 | 출발 고도 (km) | 도착 고도 (km) | 경사각 (deg) | 승교점 적경 (deg) | 중심각 (deg) | 속도 경계조건 사이 각 (deg) | 신뢰영역의 초기 크기 (km) |
|:--|---:|---:|:--|:--|---:|---:|---:|
| 1 | 245 | 400 | 51.64 | 0 | 70.00 | 70.00 | 2000 |
| 2 (기준) | 245 | 400 | 51.64 | 0 | 120.00 | 120.00 | 2000 |
| 3 | 245 | 400 | 51.64 | 0 | 135.00 | 135.00 | 2000 |
| 4 | 245 | 400 | 51.64 | 0 | 170.00 | 170.00 | 4000 |
| 5 | 245 | 400 | 51.64 → 71.64 | 0 → 15 | 125.81 | 125.03 | 2000 |

속도 경계조건 사이 각은 두 끝점의 속도 벡터 사이 각이다. 동일 궤도면을 사용하는 시나리오 1–4에서는 이 각이 중심각과 같다. 시나리오 5에서는 도착 궤도의 경사각과 승교점 적경을 바꾸어 출발 궤도면을 벗어나는 3차원 전이를 구성하였다.

시나리오 2는 분할 수 비교 실험의 기준으로 사용한다. 시나리오 3–5에서는 분할 수를 고정하고, 회피 제약이 작동하는 분할구간의 현 길이에 따른 보수성을 비교한다. 시나리오 4에서는 직선 초기 궤적이 장애물 내부를 가장 깊이 통과하므로, 여유 변수로 선형화된 회피 제약을 완화하는 효과를 확인한다. 시나리오 1은 회피 제약이 비활성인 대조군으로, 회피 제약이 작동하지 않을 때 분할 수에 따른 해의 변화를 비교한다.

알고리즘은 Rust로 구현하였다. 계산 시간은 Apple M2(8코어)에서 측정하였다.

> 볼록 부문제는 내점법 기반 Clarabel 0.11.1로 풀었다. 허용오차는 절대 간극 $10^{-11}$, 상대 간극 $10^{-10}$, 실현 가능성 $10^{-9}$로 설정하였다.

페널티 계수는 $\mu = 10^{-2}$, 신뢰영역 초기 크기는 $\Delta_0 = 2000$ km로 설정하였다. 시나리오 4에서는 첫 반복에서 속도 경계조건을 맞추는 데 필요한 제어점 이동 거리가 2000 km를 넘으므로 $\Delta_0 = 4000$ km를 사용하였다.

최적해 수렴 여부는 반복이 3.3절의 종료 기준으로 끝났는지로 판정하며, 이 기준은 명제 1의 회피 조건을 포함한다. 최소 이격거리는 궤적과 장애물 표면 사이 거리의 최솟값이다. 추정 이격거리는 장애물 최근접점이 속한 분할구간의 현 길이 $L_{\mathrm{seg}}$를 이용하여 $L_{\mathrm{seg}}^2/(8R_{\mathrm{KOZ}})$로 계산하며, 측정값과 비교하여 3.1절의 보수성 예측을 평가한다. 제어 비용은 최종 해에서 계산한 3.2절의 목적함수 $J$로, 제어 가속도 크기의 제곱을 전이 시간 동안 평균한 값이다. 계산 시간은 SCvx 반복과 후처리 검증·지표 계산을 포함한다.

---

## 5. 수치 결과

### 5.1 시나리오별 장애물 회피와 보수성

다섯 시나리오의 궤적과 정량 결과를 각각 [그림 4](#fig-trajectory)와 표 2에 제시하였다. 그림 4(A)는 출발 궤도면 위에서 본 궤적이며, (B)의 표식은 장애물 최근접점을 나타낸다.

<a id="fig-trajectory"></a>
![그림 4. 표 1의 다섯 전이 시나리오에서 얻은 궤적](../figures/representative_trajectories.png)
**그림 4 [F4].** 다섯 전이 시나리오의 (A) 궤적과 (B) 장애물과의 이격거리.

**표 2 [T2]. 시나리오별 결과 요약 ($N=7$, $n_{\mathrm{seg}}=16$)**

| 시나리오 | 최소 이격거리 (km) | 추정 이격거리 (km) | 제어 비용 $J$ (m²/s⁴) | 계산 시간 (s) | 반복 횟수 |
|:--|---:|---:|---:|---:|---:|
| 1 | 145.00 | — | 33.47 | 0.093 | 6 |
| 2 (기준) | 15.61 | 16.13 | 22.63 | 0.107 | 8 |
| 3 | 19.93 | 20.56 | 84.89 | 0.129 | 11 |
| 4 | 31.89 | 32.39 | 462.58 | 0.133 | 12 |
| 5 | 18.08 | 18.51 | 61.48 | 0.121 | 10 |

> 표 2와 그림 5의 계산 시간은 15회 실행 중 최솟값이다.

다섯 시나리오 모두 명제 1의 회피 조건을 만족하였고, 3.3절의 종료 기준을 만족하여 반복을 마쳤다. 시나리오 1을 제외한 네 시나리오에서는 회피 제약이 작동하였다. 이때 최소 이격거리는 추정값의 96~99%로, 분할 수를 고정하고 시나리오를 바꾼 경우에도 3.1절의 보수성 예측과 일치하였다. 출발 궤도면을 벗어나는 시나리오 5에서도 같은 경향을 확인하였다. 시나리오 4에서는 직선 초기 궤적이 장애물 경계에서 안쪽으로 약 5900 km까지 진입하였으나, 여유 변수로 선형화된 회피 제약을 완화하여 명제 1의 조건을 만족하는 해를 얻었다.

시나리오 1에서는 회피 제약이 비활성이었고, 출발점에서 장애물과의 이격거리가 가장 짧으므로 추정 이격거리는 제시하지 않았다. 분할 수를 8, 16, 32로 바꾸었을 때 해의 차이는 $10^{-11}$ km 이내였다. 즉 분할 수는 회피 제약이 작동할 때에만 해에 영향을 주며, 회피 제약의 보수성은 장애물 경계 부근에서만 나타난다. 제어 비용은 중심각 자체보다 자유 낙하 각거리와의 차이에 따라 달라지므로, 시나리오 1의 제어 비용은 중심각이 더 큰 기준 시나리오보다 컸다.

### 5.2 분할 수에 따른 변화

기준 시나리오에서 차수를 $N=7$로 고정하고 분할 수를 $n_{\mathrm{seg}} \in \{4,8,16,32,64\}$로 바꾸어 비교하였다. 중력 선형화 구간 수는 $n_{\mathrm{lin}} = 100$으로 고정하였으므로, 분할 수에 따른 제어 비용의 차이는 회피 제약의 보수성에서 비롯된다. 분할 수가 증가하면 제어점의 볼록 껍질이 곡선에 가까워져 회피 제약의 보수성과 제어 비용이 감소하지만, 제약 수가 늘어 계산 시간은 증가한다. 이 경향이 기준 시나리오 하나의 기하 조건에서만 나타나는 것이 아님을 확인하기 위해, 기준 시나리오의 전이 조건을 표 3의 범위에서 무작위로 바꾼 1024개 사례로 몬테카를로 분석을 수행하였다. 사례는 라틴 하이퍼큐브 표본추출로 뽑았으며, 각 사례를 모든 분할 수로 풀어 분할 수 사이의 비교가 같은 사례 안에서 이루어지도록 하였다.

**표 3 [T3]. 기준 시나리오에 준 교란의 범위**

| 변수 | 기준값 | 기준 주변 | 넓은 범위 |
|:--|---:|---:|---:|
| 궤도상 각 간격 (deg) | 120 | 110–130 | 100–170 |
| 출발 고도 (km) | 245 | 225–265 | 195–295 |
| 도착 고도 (km) | 400 | 380–420 | 350–450 |
| 도착 궤도 경사각의 변화 (deg) | 0 | ±5 | ±20 |
| 도착 궤도 승교점 적경의 변화 (deg) | 0 | ±5 | ±15 |
| 승교점에서 잰 도착점의 궤도상 각 (deg) | 45 | 0–360 | 0–360 |

> 궤도상 각 간격은 승교점에서 잰 출발점과 도착점의 궤도상 각의 차이로, 두 궤도가 같은 평면에 있으면 중심각과 같다. 각 변수는 주어진 범위에서 균등하게 뽑았다.

분할 수별 결과의 분포를 표 4와 [그림 5](#fig-subdivision-tradeoff)에 제시하였다. 모든 사례가 모든 분할 수에서 최적해에 수렴하였으므로 최적해 수렴 여부는 표에서 생략하였다. 분할 수 4에서는 회피 제약이 모든 사례에서 작동하였으나, 분할구간이 길어 곡선의 나머지 부분이 모두 출발점보다 장애물에서 멀리 놓였다. 이 경우 최소 이격거리는 출발 고도로 정해지므로 보수성 비교에서 제외하였다.

**표 4 [T4]. 분할 수에 따른 몬테카를로 분석 결과 (기준 주변 1024개 사례, $N=7$)**

| $n_{\mathrm{seg}}$ | 비교 사례 수 | 추정값 대비 최소 이격거리 | $n_{\mathrm{seg}}=64$ 대비 제어 비용 증가율 (%) | 반복 횟수 |
|---:|---:|---:|---:|---:|
| 4 | 0 | — | 214 [127, 490] | 12 [10, 15] |
| 8 | 716 | 0.967 [0.919, 0.990] | 12.0 [10.9, 13.0] | 8 [7, 9] |
| 16 | 782 | 0.970 [0.923, 0.997] | 2.63 [2.34, 2.87] | 8 [7, 9] |
| 32 | 823 | 0.968 [0.905, 0.998] | 0.515 [0.420, 0.552] | 8 [7, 9] |
| 64 | 842 | 0.971 [0.907, 0.999] | — | 8 [7, 9] |

> 각 값은 중앙값이며, 대괄호 안의 두 값은 5번째와 95번째 백분위수이다. 이격거리 비는 회피 제약이 작동하고, 최근접점이 곡선 내부에 있으며, 최근접점이 속한 분할구간이 지지평면에 닿은 사례로 계산하였다. 제어 비용 증가율은 같은 사례를 $n_{\mathrm{seg}}=64$로 푼 결과와 비교하였다.

최소 이격거리는 각 사례에서 분할 수를 두 배로 늘릴 때마다 약 4분의 1로 줄었다. 분할 수 8 이상에서 사례별로 최소 이격거리와 분할 수의 로그 기울기를 구하면 그 중앙값은 −1.99로, 3.1절의 $n_{\mathrm{seg}}^{-2}$ 예측과 일치하였다. 3.1절의 추정은 최근접점이 속한 분할구간의 두 끝점이 지지평면에 놓인다고 가정하므로, 표 4의 이격거리 비는 그 분할구간이 지지평면에 닿은 사례에서 계산하였다. 이 사례들에서 추정값 대비 최소 이격거리의 중앙값은 분할 수와 관계없이 0.97 안팎이었고, 측정값이 추정값보다 0.2% 넘게 큰 사례는 없었다. 최근접점이 지지평면에 닿지 않은 분할구간에 놓인 사례는 이웃 분할구간의 제약이 곡선을 밀어낸 경우로, 추정의 가정을 만족하지 않는다.

<a id="fig-subdivision-tradeoff"></a>
![그림 5. 분할 수에 따른 몬테카를로 분석 결과와 계산 시간](../figures/monte_carlo_subdivision.png)
**그림 5 [F5].** 분할 수에 따른 (a) 추정값 대비 최소 이격거리, (b) 제어 비용 증가율, (c) 기준 시나리오의 계산 시간.

그림 5의 상자는 사분위 범위를, 수염은 5번째부터 95번째 백분위수까지를 나타낸다. 제어 비용은 분할 수가 클수록 줄었고, 그 감소 폭은 분할 수가 커질수록 작아졌다. 분할 수 16에서 제어 비용은 분할 수 64보다 중앙값 기준 2.6% 컸다. 반복 횟수의 중앙값은 분할 수 8 이상에서 모두 8회였다. 반복마다 푸는 부문제의 회피 제약 수는 분할 수에 비례하므로, 그림 5(c)의 기준 시나리오에서 계산 시간은 분할 수 8부터 분할 수에 따라 증가하였다. 분할 수 4에서는 반복 횟수가 많아 분할 수 8보다 오래 걸렸다.

기준 주변보다 넓은 표 3의 범위에서도 4096개 사례를 같은 방법으로 풀었다. 이 범위는 표 1의 시나리오 3–5의 조건을 포함한다. 신뢰영역 초기 크기는 첫 반복에서 속도 경계조건을 맞추는 데 필요한 제어점 이동 거리를 넘을 때까지 2000 km를 두 배씩 늘려 정하였다. 최적해에 수렴하지 못한 사례는 분할 수 4의 58개뿐이었고, 분할 수 8 이상에서는 모든 사례가 수렴하였다. 수렴하지 못한 사례는 모두 반복 한도에 도달하거나 신뢰영역 크기가 하한에 이르러 종료되었다. 곡선 위 1000개 점에서 장애물 침범을 검사했을 때, 수렴한 해 가운데 침범이 관찰된 사례는 없었다.

### 5.3 법선 변화를 반영한 선형화의 효과

회피 제약의 선형화에 법선 변화를 포함한 효과를 확인하기 위해, 5.2절의 기준 주변 1024개 사례를 법선을 기준점에 고정한 부문제로 다시 풀고 그 결과를 표 5에 제시하였다. 두 경우는 3.3절 회피 제약 기울기의 둘째 항을 포함하는지만 다르며, merit function, 해의 수용 규칙, 페널티 계수, 신뢰영역 초기 크기, 종료 기준은 같다.

**표 5 [T5]. 법선을 고정한 경우의 몬테카를로 분석 결과 (기준 주변 1024개 사례, $N=7$)**

| $n_{\mathrm{seg}}$ | 최적해 수렴 비율 (%) | 반복 횟수 | 법선 변화를 포함한 경우 대비 제어 비용 증가율 (%) |
|---:|---:|---:|---:|
| 4 | 0.0 | 32 [23, 43] | — |
| 8 | 0.2 | 31 [27, 39] | 1.16 [0.221, 5.31] |
| 16 | 9.9 | 29 [10, 35] | 0.175 [0.0192, 0.829] |
| 32 | 11.4 | 26 [10, 31] | 0.0352 [0.00243, 0.0845] |
| 64 | 23.2 | 24 [9, 27] | 0.00976 [0.000888, 0.0172] |

> 반복 횟수와 제어 비용 증가율은 중앙값이며, 대괄호 안의 두 값은 5번째와 95번째 백분위수이다. 제어 비용 증가율은 같은 사례를 법선 변화를 포함하여 푼 결과와 비교하였으며, 두 해가 모두 명제 1의 조건을 만족한 사례만 사용하였다.

법선 변화를 포함하면 5.2절에서와 같이 모든 사례가 최적해에 수렴하였다. 법선을 고정하면 대부분의 사례에서 신뢰영역 크기가 하한에 이르러 반복이 종료되었고, 반복 횟수도 표 4보다 크게 늘었다. 표 1의 다섯 시나리오를 분할 수 4–64에서 같은 방법으로 비교했을 때, 법선을 고정한 경우 기각된 해의 대부분은 기준점에서 구성한 반공간을 만족하고 분할구간 제어점이 모두 장애물 밖에 있었으나, 그 해에서 다시 구성한 반공간을 위반하였다. 즉 3.3절에서 설명한 대로 법선 변화를 반영하지 않은 예측 감소량이 실제 감소량과 어긋나 해가 반복해서 기각되었다.

분할 수 8 이상에서는 법선을 고정해도 종료 시점의 해가 대부분 명제 1의 조건을 만족하였으나, 최적해에 도달하기 전에 종료되어 제어 비용이 더 컸다. 분할 수 4에서는 거의 모든 사례에서 종료 시점의 해가 명제 1의 조건을 만족하지 못했으므로 제어 비용을 비교하지 않았다.

---

## 6. 결론

본 연구에서는 분할구간마다 하나의 지지 반공간에 모든 제어점을 두어 궤적 전 구간의 장애물 회피를 보장하는 Bézier 궤적 생성 기법을 비선형 중력장의 궤도전이에 적용하였다. 이체 중력과 $J_2$ 섭동을 구간별로 선형화하여 [15]의 Gram 행렬 정식화를 비선형 중력장의 제어 가속도 비용으로 확장하였다. 지지 반공간의 법선은 제어점과 함께 움직이므로, 그 변화를 회피 제약의 선형화에 포함하여 SCvx가 실제 감소량을 더 정확히 예측하도록 하였다.

수치 실험에서 장애물 경계 부근의 최소 이격거리는 원호 근사로 추정한 값에 가까웠다. 기준 시나리오 주변의 교란 사례 전반에서도 이 관계가 유지되어, 회피 제약의 보수성을 분할구간의 현 길이로 예측할 수 있었다. 분할 수가 적으면 제어 비용이 크게 늘었고 교란 범위를 넓히면 일부 사례가 수렴하지 못하였다. 분할 수를 늘리면 제어 비용은 빠르게 줄었으나, 분할구간마다 부과되는 회피 제약의 수는 분할 수에 비례하여 늘어난다. 법선 변화를 선형화에 포함하지 않으면 해가 반복해서 기각되어 대부분의 사례가 수렴하지 못하였다. 따라서 반복이 수렴하려면 법선 변화를 선형화에 포함해야 한다.

본 연구는 구형 장애물과 고정 전이 시간의 단일 Bézier 궤적을 대상으로 하였으며, 제어 가속도 상한과 장시간 전이에 적합한 초기화는 다루지 않았다. 향후에는 여러 Bézier 호를 이용한 타원 궤도 전이와 전이 시간 최적화로 확장할 수 있다.

---

## 참고문헌

[1] Betts, J. T., "Survey of Numerical Methods for Trajectory Optimization," *Journal of Guidance, Control, and Dynamics*, Vol. 21, No. 2, 1998, pp. 193–207. doi:10.2514/2.4231

[2] Hargraves, C. R., and Paris, S. W., "Direct Trajectory Optimization Using Nonlinear Programming and Collocation," *Journal of Guidance, Control, and Dynamics*, Vol. 10, No. 4, 1987, pp. 338–342. doi:10.2514/3.20223

[3] Dueri, D., Mao, Y., Mian, Z., Ding, J., and Açıkmeşe, B., "Trajectory Optimization with Inter-Sample Obstacle Avoidance via Successive Convexification," *2017 IEEE 56th Annual Conference on Decision and Control (CDC)*, Melbourne, Australia, 2017, pp. 1150–1156. doi:10.1109/CDC.2017.8263811

[4] Elango, P., Luo, D., Kamath, A. G., Uzun, S., Kim, T., and Açıkmeşe, B., "Successive Convexification for Trajectory Optimization with Continuous-Time Constraint Satisfaction," arXiv:2404.16826, 2024. doi:10.48550/arXiv.2404.16826

[5] Kielas-Jensen, C., and Cichella, V., "Bernstein Polynomial-Based Transcription Method for Solving Optimal Trajectory Generation Problems," arXiv:2010.09992, 2020.

[6] Lee, S., and Kim, Y., "Optimal Output Trajectory Shaping Using Bézier Curves," *Journal of Guidance, Control, and Dynamics*, Vol. 44, No. 5, 2021, pp. 1027–1035. doi:10.2514/1.G005887

[7] Açıkmeşe, B., Carson, J. M., and Blackmore, L., "Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem," *IEEE Transactions on Control Systems Technology*, Vol. 21, No. 6, 2013, pp. 2104–2113. doi:10.1109/TCST.2012.2237346

[8] Mao, Y., Dueri, D., Szmuk, M., and Açıkmeşe, B., "Successive Convexification of Non-Convex Optimal Control Problems with State Constraints," *IFAC-PapersOnLine*, Vol. 50, No. 1, 2017, pp. 4063–4069. doi:10.1016/j.ifacol.2017.08.789

[9] Malyuta, D., Reynolds, T. P., Szmuk, M., Lew, T., Bonalli, R., Pavone, M., and Açıkmeşe, B., "Convex Optimization for Trajectory Generation: A Tutorial on Generating Dynamically Feasible Trajectories Reliably and Efficiently," *IEEE Control Systems Magazine*, Vol. 42, No. 5, 2022, pp. 40–113. doi:10.1109/MCS.2022.3187542

[10] Deits, R., and Tedrake, R., "Computing Large Convex Regions of Obstacle-Free Space Through Semidefinite Programming," *Algorithmic Foundations of Robotics XI (WAFR)*, Springer, 2015, pp. 109–124.

[11] Marcucci, T., Petersen, M., von Wrangel, D., and Tedrake, R., "Motion Planning around Obstacles with Convex Optimization," *Science Robotics*, Vol. 8, No. 84, 2023.

[12] Preiss, J. A., Hönig, W., Ayanian, N., and Sukhatme, G. S., "Downwash-Aware Trajectory Planning for Large Quadrotor Teams," *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2017, pp. 250–257.

[13] Tordesillas, J., and How, J. P., "MADER: Trajectory Planner in Multiagent and Dynamic Environments," *IEEE Transactions on Robotics*, Vol. 38, No. 1, 2022, pp. 463–476.

[14] Gao, F., Wang, L., Zhou, B., Han, L., Pan, J., and Shen, S., "Teach-Repeat-Replan: A Complete and Robust System for Aggressive Flight in Complex Environments," *IEEE Transactions on Robotics*, Vol. 36, No. 5, 2020, pp. 1526–1545.

[15] 최지웅, 이수원, "베지어 곡선 기반 컨벡스 궤적 최적화 및 순차 볼록 프로그래밍 충돌 회피 정식화," *한국항공우주학회지*, 2026, 게재 승인 원고(2026년 5월 14일 승인).

[16] Han, S. P., and Mangasarian, O. L., "Exact Penalty Functions in Nonlinear Programming," *Mathematical Programming*, Vol. 17, No. 1, 1979, pp. 251–269. doi:10.1007/BF01588250

[17] Nocedal, J., and Wright, S. J., *Numerical Optimization*, 2nd ed., Springer, New York, 2006, Theorem 17.3. doi:10.1007/978-0-387-40065-5

[18] NASA National Space Science Data Center, "SPACEWARN Bulletin No. 632," 1 July 2006, Progress-M 57 항목. SpaceRef 재수록본, <https://spaceref.com/status-report/nasa-spacewarn-bulletin-no-632-01-july-2006/>.

[19] European Space Agency, "ISS: International Space Station," <https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/International_Space_Station/ISS_International_Space_Station>, 접속일: 2026년 9월 22일.
