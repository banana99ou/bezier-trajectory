# D3. Latin Hypercube Sampling 설계

> 대응 코드: `src/orbit_transfer/sampling/lhs.py`

## 1. 목적

본 프로젝트는 LEO-to-LEO 연속 추력 궤도전이의 최적 추력 프로파일을 분류하기 위해, 3차원 파라미터 공간 $\mathcal{P} \subset \mathbb{R}^3$에서 초기 학습 샘플을 생성해야 한다. 파라미터 공간은 다음 세 변수로 구성된다.

| 차원 $d$ | 파라미터 키 (sorted) | 기호 | 범위 | 단위 |
|:---------:|:-------------------:|:----:|:----:|:----:|
| 0 | `T_normed` | $\tilde{T}$ | $[0.5,\; 5.0]$ | $T/T_0$ |
| 1 | `delta_a` | $\Delta a$ | $[-500,\; 2000]$ | km |
| 2 | `delta_i` | $\Delta i$ | $[0,\; 15]$ | deg |

키 정렬은 Python `sorted()` 기준으로 `['T_normed', 'delta_a', 'delta_i']` 순서이며, 코드 전체에서 이 순서를 일관되게 유지한다.

이 공간에서 초기 $n = 100$개 샘플을 효율적으로 배치하기 위해 Latin Hypercube Sampling (LHS)을 사용한다. LHS는 적은 샘플 수로도 각 차원의 주변분포(marginal distribution)를 균일하게 커버하여, 후속 GP(Gaussian Process) 분류기의 학습 효율을 높인다.


## 2. 수학적 배경

### 2.1 층화 추출 원리

LHS는 층화 추출(stratified sampling)의 다차원 확장이다.

**1차원 경우.** 구간 $[0, 1]$을 $n$개의 동일 폭 부분구간으로 분할한다:

$$I_i = \left[\frac{i}{n},\; \frac{i+1}{n}\right), \quad i = 0, 1, \dots, n-1$$

각 부분구간에서 정확히 하나의 샘플을 추출한다:

$$x_i = \frac{i + U_i}{n}, \quad U_i \sim \mathrm{Uniform}(0, 1)$$

이 구조는 $n$개 샘플만으로 전 구간을 빠짐없이 커버함을 보장한다.

**다차원 확장.** $d$차원 공간 $[0,1]^d$에서 LHS를 구성하려면, 각 차원마다 독립적인 랜덤 순열 $\pi_j$를 적용한다. $n$개 샘플로 구성된 LHS 행렬 $\mathbf{X} \in \mathbb{R}^{n \times d}$의 원소는 다음과 같다:

$$X_{\pi_j(i),\, j} = \frac{i + U_{i,j}}{n}, \quad i = 0, \dots, n-1, \quad j = 0, \dots, d-1$$

여기서 $\pi_j : \{0, \dots, n-1\} \to \{0, \dots, n-1\}$는 차원 $j$에 대한 랜덤 순열이고, $U_{i,j} \sim \mathrm{Uniform}(0, 1)$는 구간 내 랜덤 섭동이다. 이 구조는 Latin square의 다차원 일반화로서, 각 행과 열에 정확히 하나의 원소가 배치되는 성질을 $d$차원으로 확장한 것이다.

### 2.2 정규화 좌표와 물리 좌표 변환

LHS는 정규화 공간 $[0,1]^d$에서 생성된다. 물리 파라미터 공간 $\mathcal{P}$와의 변환은 각 차원별 선형 매핑으로 수행한다.

**정규화 (물리 $\to$ $[0,1]$):**

$$x_j^{\mathrm{norm}} = \frac{x_j - l_j}{h_j - l_j}$$

**역정규화 ($[0,1]$ $\to$ 물리):**

$$x_j = l_j + (h_j - l_j) \cdot x_j^{\mathrm{norm}}$$

여기서 $l_j$, $h_j$는 차원 $j$의 하한과 상한이다. 두 변환은 역함수 관계를 만족한다:

$$\mathrm{denormalize} \circ \mathrm{normalize} = \mathrm{id}$$

### 2.3 Space-Filling 성질

LHS가 갖는 핵심 성질은 **주변 균일성(marginal uniformity)**이다. 임의의 차원 $j$에 대해, $n$개 샘플의 $j$번째 좌표를 1차원으로 사영하면, $n$개 부분구간 각각에 정확히 하나의 점이 존재한다. 즉:

$$|\{i : X_{i,j} \in I_k\}| = 1, \quad \forall k = 0, \dots, n-1$$

이 성질은 어떤 차원이든 "빈 구간" 없이 전 범위를 커버함을 의미한다. 이는 순수 난수 샘플링에서 발생하는 클러스터링(clustering) 문제를 구조적으로 방지한다.

### 2.4 LHS vs 대안 기법 비교

| 기법 | 샘플 수 | 주변 균일성 | 분산 | 차원 확장성 |
|:----:|:------:|:---------:|:----:|:---------:|
| 순수 난수 (MC) | $n$ | 비보장 | $O(1/n)$ | 양호 |
| 정규 격자 (Grid) | $n^d$ | 보장 | 0 (결정론적) | 차원의 저주 |
| LHS | $n$ | 보장 | $\leq O(1/n)$ | 양호 |

**순수 난수 (Monte Carlo).** 각 좌표를 독립 균일분포에서 추출한다. 샘플 수 $n$에 대해 적분 추정의 분산이 $O(1/n)$이지만, 유한 $n$에서 점들이 한곳에 밀집(clustering)되거나 빈 영역이 생길 수 있다.

**정규 격자 (Full Factorial).** 각 차원을 $m$개 수준으로 나누면 총 $m^d$개 점이 필요하다. $d = 3$이고 차원당 10개 수준이면 $10^3 = 1000$개이다. 해석 비용이 높은 경우 비현실적이다.

**LHS.** $n$개 샘플만으로 모든 1차원 주변분포의 균일 커버를 보장한다. McKay et al. (1979)은 LHS의 적분 추정 분산이 순수 난수 대비 항상 같거나 작음을 증명하였다. 특히 출력이 주로 주효과(main effect)에 의존하는 함수에서 분산 감소 효과가 크다.

### 2.5 초기 샘플 수 결정

본 프로젝트에서 초기 LHS 샘플 수를 $n_{\mathrm{init}} = 100$으로 설정하였다. 근거는 다음과 같다.

- **차원 대비 배수:** $d = 3$ 차원에서 약 $33d \approx 100$으로, 컴퓨터 실험 설계의 경험적 지침인 $10d$를 충분히 상회한다 (Loeppky et al., 2009).
- **GP 학습 최소 요구량:** GP 분류기가 3-class 경계를 안정적으로 학습하려면, 각 클래스에 최소 수십 개 샘플이 필요하다. $n = 100$은 3개 클래스에 대해 클래스당 평균 $\sim 33$개를 확보한다.
- **후속 적응적 샘플링과의 균형:** 초기 샘플이 너무 적으면 GP 예측이 불안정하고, 너무 많으면 적응적 샘플링의 이점이 줄어든다. 최대 샘플 수 $n_{\max} = 500$ 대비 20%를 초기 탐색에 할당하는 것은 적절한 비율이다.


## 3. 구현 매핑

### 3.1 함수-수학 대응표

| 수학 표현 | Python 함수 | 파일 위치 | 비고 |
|:---------|:-----------|:---------|:----|
| LHS 알고리즘 (섹션 2.1) | `latin_hypercube_sample()` | `lhs.py:8–41` | 순열 + 구간 내 균일 랜덤 |
| $\mathcal{P} \to [0,1]^d$ 정규화 | `normalize_params()` | `lhs.py:44–53` | 차원별 선형 스케일링 |
| $[0,1]^d \to \mathcal{P}$ 역정규화 | `denormalize_params()` | `lhs.py:56–65` | 차원별 역선형 스케일링 |

### 3.2 알고리즘 흐름

`latin_hypercube_sample(n_samples, param_ranges, seed)` 함수의 실행 흐름은 다음과 같다.

1. **초기화:** `param_ranges`가 `None`이면 `config.PARAM_RANGES`를 사용한다. 키를 `sorted()`로 정렬하여 차원 순서를 결정한다.
2. **정규화 좌표 생성 (lines 29–33):**
   - 각 차원 $j = 0, \dots, d-1$에 대해:
     - $\pi_j \leftarrow$ `rng.permutation(n_samples)` — 랜덤 순열 생성
     - 각 구간 $i = 0, \dots, n-1$에 대해:
       - `samples_normed[π_j(i), j] = (i + U) / n` — $U \sim \mathrm{Uniform}(0,1)$
3. **물리 좌표 변환 (lines 36–39):**
   - 각 차원 $j$에 대해 `samples[:, j] = lo + (hi - lo) * samples_normed[:, j]`
4. **반환:** `(samples, samples_normed)` — 물리 좌표와 정규화 좌표를 함께 반환

`normalize_params`와 `denormalize_params`는 임의 shape의 배열을 지원한다. 마지막 축(axis $= -1$)의 크기가 $d$이면 되며, `...` 인덱싱으로 브로드캐스팅을 처리한다.

### 3.3 난수 생성기

`numpy.random.default_rng(seed)`를 사용하여 PCG-64 기반 난수 생성기를 초기화한다. `seed`를 명시하면 결과가 재현 가능하다. 이는 실험 재현성 확보에 필수적이다.


## 4. 수치 검증

대응 테스트: `tests/test_sampling.py::TestLHS`

### 4.1 형상 검증

```python
def test_shape(self):
    samples, normed = latin_hypercube_sample(50, seed=42)
    assert samples.shape == (50, 3)
    assert normed.shape == (50, 3)
```

$n = 50$개 샘플, $d = 3$ 차원에서 출력 행렬의 shape이 $(50, 3)$인지 확인한다.

### 4.2 범위 검증

```python
def test_range(self):
    _, normed = latin_hypercube_sample(100, seed=42)
    assert np.all(normed >= 0) and np.all(normed <= 1)
```

정규화 좌표가 $[0, 1]$ 범위 내에 있는지 확인한다. LHS 구성상 $x_{i,j} = (i + U_{i,j})/n$이므로, $U_{i,j} \in (0,1)$에서 $x_{i,j} \in (0/n,\; (n-1+1)/n) = (0, 1)$이 보장된다.

### 4.3 주변 균일성 검증

```python
def test_uniformity(self):
    _, normed = latin_hypercube_sample(1000, seed=42)
    for d in range(3):
        hist, _ = np.histogram(normed[:, d], bins=10)
        assert np.all(hist > 70)
        assert np.all(hist < 130)
```

$n = 1000$개 샘플의 각 차원을 10개 bin으로 히스토그램화한다. 이상적 LHS에서 각 bin에 정확히 $1000/10 = 100$개가 배치되어야 한다. 허용 범위 $[70, 130]$ (±30%)으로 검증한다. LHS의 층화 구조 덕분에 실제로는 이보다 훨씬 균일한 분포가 나타난다.

### 4.4 Round-Trip 일관성

```python
def test_normalize_denormalize_roundtrip(self):
    samples, normed = latin_hypercube_sample(20, seed=0)
    recovered = denormalize_params(normalize_params(samples))
    np.testing.assert_allclose(recovered, samples, atol=1e-12)
```

물리 좌표 → 정규화 → 역정규화 경로에서 원본이 복원되는지 확인한다. 허용 오차 $10^{-12}$는 부동소수점 산술의 기계 정밀도 수준이다.

### 4.5 키 정렬 순서

`sorted(PARAM_RANGES.keys())`의 결과는 `['T_normed', 'delta_a', 'delta_i']`이다. `latin_hypercube_sample`, `normalize_params`, `denormalize_params` 세 함수 모두 동일한 `sorted()` 호출을 사용하므로, 차원 인덱스 $d = 0, 1, 2$가 항상 같은 물리 파라미터에 대응된다.


## 5. 참고문헌

- McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. *Technometrics*, 21(2), 239–245.
- Sacks, J., Welch, W. J., Mitchell, T. J., & Wynn, H. P. (1989). Design and Analysis of Computer Experiments. *Statistical Science*, 4(4), 409–423.
- Loeppky, J. L., Sacks, J., & Welch, W. J. (2009). Choosing the Sample Size of a Computer Experiment: A Practical Guide. *Technometrics*, 51(4), 366–376.
