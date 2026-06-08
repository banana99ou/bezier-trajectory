# D1. 가우시안 프로세스 분류기

> 대응 코드: `src/orbit_transfer/sampling/gp_classifier.py`

## 1. 목적

본 프로젝트는 LEO-to-LEO 연속 추력 궤도전이의 최적 추력 프로파일을 3가지 유형으로 분류한다.

| 클래스 $c$ | 레이블 | 물리적 의미 |
|:----------:|:------:|:-----------|
| 0 | Unimodal | 단일 추력 arc (피크 1개) |
| 1 | Bimodal | 이중 추력 arc (피크 2개) |
| 2 | Multimodal | 다중 추력 arc (피크 3개 이상) |

분류의 입력은 3차원 정규화 파라미터 공간 $\mathbf{x} = [\tilde{T},\, \Delta a,\, \Delta i]^\top \in [0,1]^3$이다. 여기서 $\tilde{T}$는 정규화 추력, $\Delta a$는 반장축 변화량, $\Delta i$는 궤도경사 변화량이다. 각 점에서의 클래스 레이블은 direct collocation으로 산출한 최적 추력 프로파일의 피크 수로 결정된다.

이 분류 문제를 가우시안 프로세스 분류기(Gaussian Process Classifier, GPC)로 모델링한다. GPC는 결정 경계에서의 예측 불확실성을 확률적으로 정량화하므로, 능동 학습(active learning)에서 다음 샘플링 위치를 결정하는 acquisition function의 기반이 된다. 본 모듈(`gp_classifier.py`)은 scikit-learn의 `GaussianProcessClassifier`를 래핑하여, ARD(Automatic Relevance Determination) RBF 커널과 Laplace 근사 기반의 3-class GPC를 제공한다.


## 2. 수학적 배경

### 2.1 가우시안 프로세스 회귀의 기초

**가우시안 프로세스(Gaussian Process, GP)**는 함수에 대한 확률 분포이다. GP는 평균 함수 $m(\mathbf{x})$와 공분산 함수(커널) $k(\mathbf{x}, \mathbf{x}')$로 완전히 정의된다:

$$f \sim \mathcal{GP}\bigl(m(\mathbf{x}),\; k(\mathbf{x}, \mathbf{x}')\bigr)$$

이것은 임의의 유한 입력 집합 $\{\mathbf{x}_1, \dots, \mathbf{x}_N\}$에 대해, 함수값 벡터 $\mathbf{f} = [f(\mathbf{x}_1), \dots, f(\mathbf{x}_N)]^\top$가 다변량 가우시안 분포를 따른다는 의미이다:

$$\mathbf{f} \sim \mathcal{N}\bigl(\mathbf{m},\; \mathbf{K}\bigr)$$

여기서 $\mathbf{m}_i = m(\mathbf{x}_i)$이고 $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$이다.

**GP 회귀**에서는 관측값 $y_i = f(\mathbf{x}_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$를 가정한다. 가우시안 우도 덕분에 사후 분포가 해석적으로 계산된다. 학습 데이터 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$이 주어졌을 때, 새로운 입력 $\mathbf{x}_*$에서의 예측 분포는 다음과 같다:

$$f_* \mid \mathcal{D}, \mathbf{x}_* \sim \mathcal{N}\bigl(\bar{f}_*,\; \mathrm{var}(f_*)\bigr)$$

$$\bar{f}_* = \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$

$$\mathrm{var}(f_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

여기서 $\mathbf{k}_* = [k(\mathbf{x}_1, \mathbf{x}_*), \dots, k(\mathbf{x}_N, \mathbf{x}_*)]^\top$이다. GP 회귀의 핵심 장점은 예측값뿐 아니라 예측 불확실성을 동시에 제공한다는 점이다.


### 2.2 GP 회귀에서 GP 분류로의 확장

분류 문제에서 관측값 $y_i$는 이산적인 클래스 레이블이다. GP 회귀의 가우시안 우도 $p(y \mid f) = \mathcal{N}(y \mid f, \sigma_n^2)$를 그대로 사용할 수 없다. 대신 잠재 함수(latent function) $f(\mathbf{x})$를 비선형 링크 함수로 변환하여 클래스 확률을 정의한다.

**이진 분류**에서는 시그모이드(또는 probit) 링크를 사용한다:

$$p(y = 1 \mid f) = \sigma(f) = \frac{1}{1 + \exp(-f)}$$

**다중 클래스 분류** ($C \geq 3$)에서는 각 클래스 $c = 0, 1, \dots, C-1$에 대응하는 잠재 함수 $f_c(\mathbf{x})$를 도입하고, softmax 링크를 적용한다:

$$p(y = c \mid \mathbf{f}) = \frac{\exp(f_c)}{\sum_{c'=0}^{C-1} \exp(f_{c'})}$$

여기서 $\mathbf{f} = [f_0, f_1, \dots, f_{C-1}]^\top$이다.

이 비가우시안 우도로 인해 사후 분포 $p(\mathbf{f} \mid X, \mathbf{y})$는 해석적으로 계산되지 않는다. 이를 해결하기 위한 대표적인 근사 방법이 Laplace 근사와 Expectation Propagation (EP)이다. scikit-learn의 `GaussianProcessClassifier`는 Laplace 근사를 사용한다.


### 2.3 Laplace 근사

Laplace 근사는 사후 분포를 MAP(Maximum A Posteriori) 추정치 주변에서 2차 Taylor 전개하여 가우시안으로 근사하는 방법이다.

**사후 분포의 로그:**

$$\log p(\mathbf{f} \mid X, \mathbf{y}) = \log p(\mathbf{y} \mid \mathbf{f}) + \log p(\mathbf{f} \mid X) - \log p(\mathbf{y} \mid X)$$

여기서 $p(\mathbf{f} \mid X) = \mathcal{N}(\mathbf{f} \mid \mathbf{0}, \mathbf{K})$는 GP 사전 분포이다.

**Step 1: MAP 추정.** 비정규화 사후 분포를 최대화하는 $\hat{\mathbf{f}}$를 구한다:

$$\hat{\mathbf{f}} = \arg\max_{\mathbf{f}} \; \Psi(\mathbf{f})$$

여기서:

$$\Psi(\mathbf{f}) = \log p(\mathbf{y} \mid \mathbf{f}) - \frac{1}{2} \mathbf{f}^\top \mathbf{K}^{-1} \mathbf{f} - \frac{1}{2} \log |\mathbf{K}| - \frac{N}{2} \log(2\pi)$$

이 최적화는 Newton-Raphson 반복으로 수행한다. $\Psi(\mathbf{f})$의 그래디언트는 다음과 같다:

$$\nabla \Psi = \nabla \log p(\mathbf{y} \mid \mathbf{f}) - \mathbf{K}^{-1} \mathbf{f}$$

**Step 2: Hessian 계산.** MAP 추정치 $\hat{\mathbf{f}}$에서의 음의 Hessian을 계산한다:

$$\mathbf{W} = -\nabla\nabla \log p(\mathbf{y} \mid \mathbf{f})\big|_{\hat{\mathbf{f}}}$$

$\mathbf{W}$는 우도 함수의 곡률(curvature)을 나타내는 대각 행렬이다. Softmax 우도의 경우, $\mathbf{W}$는 블록 대각 구조를 갖는다.

**Step 3: 가우시안 근사.** 사후 분포를 다음과 같이 근사한다:

$$q(\mathbf{f} \mid X, \mathbf{y}) = \mathcal{N}\bigl(\hat{\mathbf{f}},\; (\mathbf{K}^{-1} + \mathbf{W})^{-1}\bigr)$$

근사 공분산 $(\mathbf{K}^{-1} + \mathbf{W})^{-1}$는 사전 분포의 공분산 $\mathbf{K}$보다 작다. 이는 관측 데이터가 잠재 함수에 대한 불확실성을 줄여주기 때문이다.


### 2.4 새로운 입력에서의 예측

학습 후, 새로운 입력 $\mathbf{x}_*$에서의 예측은 두 단계로 이루어진다.

**잠재 함수 예측.** GP 사후를 이용하여 잠재 함수의 예측 분포를 구한다:

$$f_* \mid \mathcal{D}, \mathbf{x}_* \sim q(f_*) = \mathcal{N}(\mu_*, \sigma_*^2)$$

$$\mu_* = \mathbf{k}_*^\top \mathbf{K}^{-1} \hat{\mathbf{f}}$$

$$\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top (\mathbf{K} + \mathbf{W}^{-1})^{-1} \mathbf{k}_*$$

**클래스 확률 예측.** 잠재 함수 분포를 링크 함수에 적분하여 클래스 확률을 구한다:

$$p(y_* = c \mid \mathcal{D}, \mathbf{x}_*) = \int p(y_* = c \mid \mathbf{f}_*) \, q(\mathbf{f}_*) \, d\mathbf{f}_*$$

이 적분은 해석적으로 풀리지 않으므로, 수치 적분 또는 Monte Carlo 근사를 사용한다. scikit-learn에서는 `max_iter_predict` 파라미터로 이 수치 적분의 반복 횟수를 제어한다.


### 2.5 주변 우도와 하이퍼파라미터 최적화

커널 함수의 하이퍼파라미터 $\boldsymbol{\theta}$를 학습 데이터로부터 결정하기 위해, 주변 우도(marginal likelihood)를 최대화한다:

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \; \log p(\mathbf{y} \mid X, \boldsymbol{\theta})$$

Laplace 근사 하에서 주변 우도의 로그는 다음과 같이 근사된다:

$$\log p(\mathbf{y} \mid X, \boldsymbol{\theta}) \approx -\frac{1}{2} \hat{\mathbf{f}}^\top \mathbf{K}^{-1} \hat{\mathbf{f}} + \log p(\mathbf{y} \mid \hat{\mathbf{f}}) - \frac{1}{2} \log |\mathbf{K} + \mathbf{W}^{-1}|$$

우변의 세 항은 각각 다음과 같이 해석된다.

| 항 | 의미 |
|:---|:----|
| $-\frac{1}{2}\hat{\mathbf{f}}^\top \mathbf{K}^{-1}\hat{\mathbf{f}}$ | 복잡도 벌칙 (complexity penalty). MAP 추정치가 사전 분포에서 벗어난 정도를 측정한다. |
| $\log p(\mathbf{y} \mid \hat{\mathbf{f}})$ | 데이터 적합도 (data fit). 관측 레이블을 잘 설명하는 정도를 측정한다. |
| $-\frac{1}{2}\log|\mathbf{K} + \mathbf{W}^{-1}|$ | Occam 인자 (Occam factor). 모델 복잡도에 대한 자동 벌칙이다. |

이 세 항의 균형이 GP의 자동 복잡도 조절 능력을 제공한다. 과적합 모델은 데이터 적합도는 높지만 복잡도 벌칙이 크고, 과소적합 모델은 그 반대이다.

주변 우도의 $\boldsymbol{\theta}$에 대한 그래디언트를 계산하여 경사 기반 최적화를 수행한다. 다중 시작점(multi-start) 전략으로 지역 최적해에 빠지는 것을 완화한다. 본 구현에서는 `n_restarts_optimizer=3`으로 설정하여, 초기 하이퍼파라미터를 4번(1회 기본 + 3회 재시작) 최적화하고 가장 높은 주변 우도를 갖는 해를 선택한다.


### 2.6 ARD RBF 커널

본 구현에서 사용하는 커널은 상수 커널과 ARD(Automatic Relevance Determination) RBF 커널의 곱이다:

$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{1}{2} \sum_{d=1}^{D} \frac{(x_d - x'_d)^2}{\ell_d^2}\right)$$

여기서:
- $\sigma_f^2$: 신호 분산(signal variance). 함수값의 전체적인 진폭을 제어한다.
- $\ell_d$: 차원 $d$의 길이 척도(length scale). 해당 차원에서의 함수 변동 규모를 결정한다.

**ARD의 물리적 해석.** 표준 RBF 커널은 모든 차원에 동일한 길이 척도 $\ell$을 사용한다. ARD RBF 커널은 각 차원에 개별 길이 척도 $\ell_d$를 부여한다. 학습 결과 $\ell_d$가 크면, 차원 $d$의 변화가 함수값에 미치는 영향이 작다는 의미이다. 반대로 $\ell_d$가 작으면, 해당 차원의 변화에 함수가 민감하게 반응한다.

본 프로젝트의 3차원 파라미터 공간에서 ARD 길이 척도는 다음과 같이 해석된다:

| 차원 $d$ | 파라미터 | 길이 척도 | 해석 |
|:--------:|:-------:|:---------:|:-----|
| 0 | $\tilde{T}$ (정규화 추력) | $\ell_0$ | 추력 크기가 프로파일 유형에 미치는 영향 |
| 1 | $\Delta a$ (반장축 변화) | $\ell_1$ | 궤도 크기 변화가 프로파일 유형에 미치는 영향 |
| 2 | $\Delta i$ (경사 변화) | $\ell_2$ | 궤도면 변경이 프로파일 유형에 미치는 영향 |

학습된 길이 척도의 비율 $\ell_0 : \ell_1 : \ell_2$는 세 파라미터의 상대적 영향력(relevance)을 정량적으로 나타낸다. 이는 어떤 물리 파라미터가 추력 프로파일 유형 전환에 가장 큰 영향을 미치는지 식별하는 데 활용된다.

**커널 파라미터의 초기값.** 코드에서 `ConstantKernel(1.0) * RBF(length_scale=np.ones(n_dims))`로 초기화한다. 상수 커널의 초기값은 $\sigma_f^2 = 1.0$이고, 모든 차원의 길이 척도 초기값은 $\ell_d = 1.0$이다. 정규화 좌표 $[0,1]^3$에서 길이 척도 1.0은 전체 범위에 해당하므로, 사전 정보가 없는 상태에서의 합리적인 출발점이다.


### 2.7 One-vs-Rest (OvR) 다중 클래스 전략

$C$개 클래스에 대한 다중 클래스 분류는 두 가지 대표적 전략으로 분해할 수 있다.

**One-vs-Rest (OvR).** 각 클래스 $c$에 대해 "클래스 $c$ vs. 나머지 전체"로 이진 분류기를 독립적으로 학습한다. 총 $C$개의 이진 GP 분류기를 학습하며, 예측 시 각 분류기의 출력 확률을 정규화하여 클래스 확률을 구한다:

$$p(y = c \mid \mathbf{x}) = \frac{p_c(\mathbf{x})}{\sum_{c'=0}^{C-1} p_{c'}(\mathbf{x})}$$

여기서 $p_c(\mathbf{x})$는 클래스 $c$에 대한 이진 분류기의 양성 확률이다.

**One-vs-One (OvO).** 모든 클래스 쌍 $(c, c')$에 대해 이진 분류기를 학습한다. $\binom{C}{2}$개의 분류기가 필요하다.

본 구현에서는 scikit-learn의 `multi_class='one_vs_rest'` 설정을 통해 OvR 전략을 사용한다. $C = 3$일 때 OvR은 3개의 이진 GP 분류기를 학습한다. 이는 OvO의 3개 분류기와 같은 수이지만, OvR은 각 분류기가 전체 데이터를 사용하므로 데이터 효율성이 높다.

OvR 전략에서 각 이진 분류기는 독립적으로 Laplace 근사를 수행한다. 따라서 각 분류기는 자체적인 MAP 추정치 $\hat{\mathbf{f}}_c$와 Hessian $\mathbf{W}_c$를 가진다. 커널 하이퍼파라미터도 분류기별로 독립적으로 최적화된다.

그러나 OvR 전략에서 각 이진 분류기가 독립적으로 학습되므로, 원리적으로는 하이퍼파라미터(길이 척도)가 분류기마다 다를 수 있다. scikit-learn 구현에서 `get_length_scales()` 메서드는 첫 번째(또는 합성된) 커널의 길이 척도를 반환한다.


## 3. 구현 매핑

### 3.1 클래스-수학 대응표

| 수학 개념 | 코드 요소 | 파일 위치 | 비고 |
|:---------|:---------|:---------|:----|
| GP 사전: $f \sim \mathcal{GP}(0, k)$ | `GaussianProcessClassifier(kernel=...)` | `gp_classifier.py:19–24` | zero-mean GP |
| ARD RBF 커널: $\sigma_f^2 \cdot \exp(\cdots)$ | `ConstantKernel(1.0) * RBF(length_scale=np.ones(n_dims))` | `gp_classifier.py:18` | 차원별 길이 척도 |
| Laplace 근사 | scikit-learn 내부 (기본값) | `gp_classifier.py:19` | `GaussianProcessClassifier` 기본 추론 |
| OvR 다중 클래스 | `multi_class='one_vs_rest'` | `gp_classifier.py:20` | 3개 이진 분류기 |
| 예측 적분 반복 | `max_iter_predict=100` | `gp_classifier.py:21` | 예측 시 수치 적분 반복 수 |
| 다중 시작 하이퍼파라미터 최적화 | `n_restarts_optimizer=3` | `gp_classifier.py:22` | 총 4회(1+3) 최적화 |
| 학습: $\hat{\boldsymbol{\theta}} = \arg\max \log p(\mathbf{y} \mid X, \boldsymbol{\theta})$ | `fit(X, y)` | `gp_classifier.py:27–30` | 주변 우도 최대화 |
| 확률 예측: $p(y_* = c \mid \mathcal{D}, \mathbf{x}_*)$ | `predict_proba(X)` | `gp_classifier.py:32–34` | $(N, C)$ 배열 반환 |
| 클래스 예측: $\hat{y}_* = \arg\max_c p(y_* = c)$ | `predict(X)` | `gp_classifier.py:36–38` | $(N,)$ 배열 반환 |
| ARD 길이 척도: $[\ell_0, \ell_1, \ell_2]$ | `get_length_scales()` | `gp_classifier.py:40–51` | 학습 후 커널 파라미터 추출 |

### 3.2 `GPClassifierWrapper` 클래스 구조

클래스는 다음 네 가지 공개 메서드를 제공한다.

**`__init__(self, n_dims=3)`** (lines 16–25). 커널과 분류기를 초기화한다. `n_dims`는 입력 차원수로, 기본값 3은 파라미터 공간의 차원에 대응한다. 커널을 `ConstantKernel(1.0) * RBF(length_scale=np.ones(n_dims))`로 구성하여 ARD RBF 커널을 정의한다. scikit-learn의 `Product` 커널 객체가 생성되며, `k1`이 `ConstantKernel`, `k2`가 `RBF`에 해당한다.

**`fit(self, X, y)`** (lines 27–30). 학습 데이터 $X \in \mathbb{R}^{N \times d}$와 레이블 $y \in \{0, 1, 2\}^N$으로 GP 분류기를 학습한다. 내부적으로 scikit-learn이 다음을 수행한다:
1. OvR 전략에 따라 3개의 이진 GP 분류기를 구성
2. 각 분류기에 대해 Laplace 근사로 MAP 추정
3. 주변 우도를 최대화하여 커널 하이퍼파라미터 최적화
4. `n_restarts_optimizer=3`에 따라 4회 반복, 최적 결과 선택

학습 후 `self.is_fitted = True`로 설정하고, 학습된 커널은 `self.gpc.kernel_` 속성에 저장된다. 초기 커널 `self.gpc.kernel`과 구분된다는 점에 유의해야 한다. 밑줄(`_`)이 붙은 속성이 학습 결과이다.

**`predict_proba(self, X)`** (lines 32–34). 새로운 입력 $X \in \mathbb{R}^{N_* \times d}$에서 클래스 확률 행렬 $(N_*, C)$를 반환한다. 각 행은 확률 벡터로, 합이 1이다. 이 확률 분포는 능동 학습의 predictive entropy 계산에 직접 사용된다.

**`get_length_scales(self)`** (lines 40–51). 학습된 ARD 길이 척도 $[\ell_0, \ell_1, \ell_2]$를 반환한다. 학습 전이면 `None`을 반환한다. scikit-learn의 `Product` 커널에서 `kernel_.k2`가 RBF 부분이므로, 이로부터 `length_scale` 속성을 추출한다. `hasattr` 검사로 커널 구조가 예상과 다른 경우에도 안전하게 동작한다.

### 3.3 scikit-learn 내부 구현 요약

`GPClassifierWrapper`는 scikit-learn의 `GaussianProcessClassifier`를 직접 래핑한다. scikit-learn의 내부 동작을 요약하면 다음과 같다.

**학습 과정 (`fit`):**
1. `multi_class='one_vs_rest'`이면, `OneVsRestClassifier`가 클래스 수 $C$개의 이진 `_BinaryGaussianProcessClassifierLaplace` 객체를 생성한다.
2. 각 이진 분류기에서:
   - 커널 행렬 $\mathbf{K}$를 계산한다 ($O(N^2 d)$).
   - Newton-Raphson으로 MAP 추정치 $\hat{\mathbf{f}}$를 구한다.
   - Hessian $\mathbf{W}$를 계산한다.
   - 주변 우도의 그래디언트를 이용하여 L-BFGS-B로 하이퍼파라미터를 최적화한다.
   - `n_restarts_optimizer`만큼 랜덤 초기점에서 반복하여 최적 해를 선택한다.

**예측 과정 (`predict_proba`):**
1. 각 이진 분류기에서 잠재 함수의 예측 평균 $\mu_*$과 분산 $\sigma_*^2$를 계산한다.
2. 시그모이드 적분을 수치적으로 수행하여 양성 확률 $p_c(\mathbf{x}_*)$를 구한다.
3. OvR 정규화: $p(y = c) = p_c / \sum_{c'} p_{c'}$를 적용한다.

**계산 복잡도.** 학습 시 커널 행렬의 역행렬 계산이 지배적이다. 시간 복잡도는 $O(N^3)$, 공간 복잡도는 $O(N^2)$이다. $N = 500$ (본 프로젝트의 최대 샘플 수) 수준에서는 수 초 이내에 학습이 완료된다.


## 4. 수치 검증

대응 테스트: `tests/test_sampling.py::TestGPClassifier`

### 4.1 합성 3-class 분류 정확도

```python
class TestGPClassifier:
    def test_synthetic_3class(self):
        """합성 3-class 데이터에서 정확도 > 80%."""
        rng = np.random.default_rng(42)
        N = 200
        X = rng.uniform(0, 1, (N, 3))
        # 간단한 규칙: class = 0 if x0<0.33, 1 if 0.33<x0<0.66, 2 otherwise
        y = np.where(X[:, 0] < 0.33, 0, np.where(X[:, 0] < 0.66, 1, 2))

        gpc = GPClassifierWrapper(n_dims=3)
        # 80% train, 20% test
        n_train = 160
        gpc.fit(X[:n_train], y[:n_train])
        pred = gpc.predict(X[n_train:])
        accuracy = np.mean(pred == y[n_train:])
        assert accuracy > 0.80, f"accuracy {accuracy:.2f} < 0.80"
```

이 테스트는 다음 사항을 검증한다.

**데이터 생성.** $[0,1]^3$ 공간에서 200개 점을 균일 분포로 추출한다. 클래스 레이블은 첫 번째 차원 $x_0$에 의해서만 결정된다: $x_0 < 0.33$이면 class 0, $0.33 \leq x_0 < 0.66$이면 class 1, $x_0 \geq 0.66$이면 class 2이다. 이는 결정 경계가 $x_0 = 0.33$과 $x_0 = 0.66$인 단순한 구조이다.

**학습/테스트 분할.** 160개로 학습, 40개로 테스트한다 (80/20 분할).

**정확도 기준.** 테스트 정확도 80% 이상을 요구한다. ARD RBF 커널은 학습 과정에서 $\ell_0$가 작아지고 $\ell_1, \ell_2$가 커지는 방향으로 수렴하여, $x_0$ 차원의 지배적 영향을 학습해야 한다.

**검증 의미.** 이 테스트는 (1) `fit`-`predict` 파이프라인이 정상 동작하고, (2) ARD 커널이 관련 차원을 식별할 수 있으며, (3) Laplace 근사가 충분한 분류 성능을 제공함을 확인한다.

### 4.2 길이 척도 추출 검증

합성 테스트에서 학습된 ARD 길이 척도를 `get_length_scales()`로 추출할 수 있다. 위의 데이터 구조에서, 클래스가 $x_0$에 의해서만 결정되므로, 학습 후 다음 관계가 기대된다:

$$\ell_0 \ll \ell_1 \approx \ell_2$$

$\ell_0$가 작다는 것은 함수가 $x_0$ 방향으로 빠르게 변한다는 의미이며, $\ell_1, \ell_2$가 크다는 것은 $x_1, x_2$ 방향의 변화가 함수에 거의 영향을 미치지 않는다는 의미이다. 이 관계는 ARD의 자동 관련성 결정 능력을 보여준다.

### 4.3 미학습 상태 안전성

`get_length_scales()` 메서드는 학습 전 (`is_fitted=False`) 호출 시 `None`을 반환한다. 이는 학습되지 않은 커널 파라미터에 접근하는 오류를 방지한다.

### 4.4 확률 출력 일관성

`predict_proba(X)`의 반환값은 $(N, C)$ 형상의 배열이며, 각 행의 합은 1이다. OvR 전략에서 각 이진 분류기의 양성 확률을 정규화하므로, 이 성질이 보장된다:

$$\sum_{c=0}^{C-1} p(y = c \mid \mathbf{x}) = 1, \quad \forall \mathbf{x}$$

### 4.5 능동 학습 맥락에서의 통합 검증

`tests/test_sampling.py::TestAdaptiveSampler::test_small_convergence` 테스트는 `GPClassifierWrapper`가 능동 학습 루프 내에서 반복적으로 학습-예측되는 전체 파이프라인을 검증한다. 20개의 초기 LHS 샘플에서 시작하여 최대 50개까지 적응적으로 샘플을 추가하며, GP 분류기가 매 반복마다 재학습된다. 이 테스트는 GP 분류기가 점진적 데이터 추가에 대해 안정적으로 동작하고, predictive entropy 기반 acquisition function과 올바르게 연동됨을 확인한다.


## 5. 참고문헌

- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Williams, C. K. I. & Barber, D. (1998). Bayesian Classification with Gaussian Processes. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(12), 1342–1351.
- Nickisch, H. & Rasmussen, C. E. (2008). Approximations for Binary Gaussian Process Classification. *Journal of Machine Learning Research*, 9, 2035–2078.
- Neal, R. M. (1998). Regression and Classification Using Gaussian Process Priors. In *Bayesian Statistics 6* (pp. 475–501). Oxford University Press.
- MacKay, D. J. C. (1992). The Evidence Framework Applied to Classification Networks. *Neural Computation*, 4(5), 720–736.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Settles, B. (2009). Active Learning Literature Survey. *Computer Sciences Technical Report 1648*, University of Wisconsin-Madison.
- Lee, S. (2023). A Generative Verification Framework on Statistical Stability for Data-Driven Controllers. *IEEE Access*, 11, 5267–5280.
