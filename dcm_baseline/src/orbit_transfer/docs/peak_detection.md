# D4. 위상적 지속성 기반 피크 탐지 알고리즘

> 대응 코드: `src/orbit_transfer/classification/peak_detection.py`

## 1. 목적

본 보고서는 연속 추력 궤도전이 최적화 결과에서 추력 크기 프로파일 $\|\mathbf{u}(t)\|$의 피크를 자동으로 탐지하고 정량화하는 알고리즘의 수학적 배경과 구현 상세를 기술한다.

추력 프로파일의 피크 개수는 궤적 분류의 기초이다. 피크가 1개면 unimodal (class 0), 2개면 bimodal (class 1), 3개 이상이면 multimodal (class 2)로 분류하며, 이 분류 결과는 Multi-Phase collocation의 phase 구조 결정에 직접 사용된다. 따라서 피크 탐지의 정확성과 강건성은 전체 최적화 파이프라인의 신뢰성을 좌우하는 핵심 요소이다.

본 구현은 **위상적 지속성(Topological Persistence)** 기반의 0차 persistent homology 알고리즘을 사용한다. 이 방법은 기존의 scipy `find_peaks` 기반 접근법에 비해 (i) 파라미터가 하나뿐이며, (ii) 경계 피크를 별도 로직 없이 자연스럽게 처리하고, (iii) 보간 아티팩트에 강건하다는 장점이 있다.

## 2. 수학적 배경

### 2.1 피크 탐지 문제의 정의

이산 시계열 $f = (f_0, f_1, \ldots, f_{N-1})$이 주어질 때, "피크"란 극대점(local maximum), 즉 $f_i > f_{i-1}$이고 $f_i > f_{i+1}$인 인덱스 $i$를 의미한다. 경계점($i = 0$ 또는 $i = N-1$)은 한쪽만 비교한다.

실제 데이터에서 모든 극대점을 피크로 간주하면 노이즈에 의한 위양성(false positive)이 발생한다. 핵심 과제는 **유의미한 피크와 노이즈 피크를 구분하는 임계값**을 체계적으로 결정하는 것이다.

### 2.2 위상적 지속성 (Topological Persistence)

#### 2.2.1 0차 호몰로지의 직관

함수 $f: [a, b] \to \mathbb{R}$의 그래프 위에 수평선 $y = c$를 놓고 $c$를 위에서 아래로 내려 보자. 처음에 수평선은 함수 그래프와 만나지 않다가, 전역 최대점에서 처음 접촉한다. 이때 하나의 **연결 성분(connected component)**, 즉 "섬"이 **탄생(birth)**한다. 수평선을 더 내리면 새로운 극대점에서 추가 섬이 탄생하고, 두 섬 사이의 극소점을 지날 때 두 섬이 **합병(merge)**한다.

합병 시 더 낮은 극대점을 가진 섬이 **사망(death)**한다. 이를 **elder rule**이라 하며, 더 높은 피크에 우선권을 부여한다. 각 피크의 **지속성(persistence)**은 다음과 같이 정의된다:

$$\text{pers}(p) = f(\text{birth}_p) - f(\text{death}_p)$$

전역 최대점의 경우 사망하지 않으므로 $\text{pers} = \infty$이다.

#### 2.2.2 1D 수열에 대한 알고리즘

이산 수열 $f = (f_0, \ldots, f_{N-1})$에 대해 0차 persistent homology를 계산하는 알고리즘은 다음과 같다.

**입력**: 길이 $N$의 수열 $f$

**출력**: 피크 리스트, 각 피크는 $(b, d, \ell, r)$을 가짐
- $b$: 탄생 인덱스 (birth)
- $d$: 사망 인덱스 (death), 전역 최대는 $d = \text{None}$
- $\ell, r$: 해당 섬의 좌/우 경계 인덱스

**알고리즘**:
1. 인덱스를 $f$ 값의 내림차순으로 정렬: $\sigma = \text{argsort}(-f)$
2. 보조 배열 $\text{owner}[0 \ldots N-1]$을 $\text{None}$으로 초기화
3. 정렬된 순서대로 각 인덱스 $i = \sigma_k$에 대해:
   - 좌측 이웃: $L = \text{owner}[i-1]$ (존재하면)
   - 우측 이웃: $R = \text{owner}[i+1]$ (존재하면)
   - **Case 1** ($L = R = \text{None}$): 새 섬 탄생. $\text{peak} \leftarrow (b=i, d=\text{None}, \ell=i, r=i)$
   - **Case 2** ($L \neq \text{None}, R = \text{None}$): 기존 섬 $L$에 흡수. $r_L \leftarrow i$
   - **Case 3** ($L = \text{None}, R \neq \text{None}$): 기존 섬 $R$에 흡수. $\ell_R \leftarrow i$
   - **Case 4** ($L \neq \text{None}, R \neq \text{None}$): 두 섬 합병. Elder rule 적용:
     - $f[b_L] > f[b_R]$이면 $R$ 사망: $d_R \leftarrow i$, $L$이 $R$의 영역 흡수
     - 그렇지 않으면 $L$ 사망: $d_L \leftarrow i$, $R$이 $L$의 영역 흡수
   - $\text{owner}[i] \leftarrow$ 해당 섬 인덱스

**시간 복잡도**: $O(N \log N)$ (Step 1의 정렬이 지배적. Step 3은 $O(N)$.)

**공간 복잡도**: $O(N)$

#### 2.2.3 지속성 다이어그램

각 피크를 $(b_p, d_p)$ 좌표의 점으로 표현하면 **지속성 다이어그램(persistence diagram)**을 얻는다. 대각선 $b = d$에서 먼 점일수록 지속성이 크고 유의미한 피크이다. 유의미한 피크의 선택은 단일 임계값 $\tau$로 결정된다:

$$\text{significant peaks} = \{p \mid \text{pers}(p) \geq \tau\}$$

본 구현에서는 $\tau = \alpha \cdot f_{\max}$으로 설정하며, $\alpha = 0.10$ (10%)를 기본값으로 사용한다.

### 2.3 경계 피크의 처리

전통적인 `find_peaks` 기반 방법은 경계점($i = 0$ 또는 $i = N-1$)에서의 피크를 별도 로직으로 탐지해야 한다. 위상적 지속성에서는 경계 피크가 자연스럽게 처리된다. 수열의 첫 번째 또는 마지막 원소가 극대점이면, 해당 위치에서 섬이 탄생하고 그 지속성이 계산된다. 별도 경계 탐지 로직이 필요 없으므로 구현이 간결하고 일관된다.

### 2.4 CubicSpline 보간과 Phase-Aware 전처리

#### 2.4.1 보간의 필요성

Pseudospectral collocation 해법은 15~30개의 비균일 노드에서만 정의된다. 이 소수의 이산점에 직접 피크 탐지를 적용하면 해상도 부족으로 피크 위치와 개수가 부정확해진다. CubicSpline 보간으로 $N_{\text{interp}} = 200$개의 균일 격자로 업샘플링하여 해결한다.

#### 2.4.2 Phase-Aware 보간

Multi-Phase LGL collocation은 각 phase별로 독립적인 LGL 노드를 사용한다. Phase 경계에서 노드 간격이 급격히 변하므로, 전체 시간 구간에 글로벌 CubicSpline을 적용하면 Runge 현상과 유사한 대형 진동이 발생한다. 이를 방지하기 위해 각 phase 내부에서 독립적으로 CubicSpline 보간을 수행한다.

Phase $k$의 시간 구간이 $[t_k^{\text{start}}, t_k^{\text{end}}]$이고 전체 구간이 $[0, T]$일 때, phase $k$에 배정되는 보간 점 수는:

$$N_k = \max\left(10, \left\lfloor N_{\text{interp}} \cdot \frac{t_k^{\text{end}} - t_k^{\text{start}}}{T} \right\rfloor\right)$$

각 phase별 보간 결과를 연결할 때, 경계점 중복을 방지하기 위해 $k > 0$인 phase에서는 첫 번째 보간점을 제거한다.

### 2.5 피크 반치폭 (FWHM) 추정

각 피크의 반치폭(Full Width at Half Maximum, FWHM)은 Multi-Phase 구조에서 peak/coast phase의 경계를 결정하는 데 사용된다.

내부 피크(경계가 아닌 피크)에 대해서는 scipy의 `peak_widths` 함수를 사용하여 반높이에서의 폭을 샘플 단위로 계산한 뒤 시간 단위로 변환한다:

$$\text{FWHM}_{\text{time}} = \text{FWHM}_{\text{samples}} \cdot \Delta t_{\text{mean}}$$

여기서 $\Delta t_{\text{mean}} = (t_{N-1} - t_0) / (N - 1)$이다.

경계 피크($i = 0$ 또는 $i = N-1$)에 대해서는 반높이 $f_i / 2$에 도달하는 지점까지의 거리를 측정하고, 대칭을 가정하여 2배로 추정한다:

$$\text{FWHM}_{\text{boundary}} = 2 \cdot d_{\text{half}} \cdot \Delta t_{\text{mean}}$$

### 2.6 대안 알고리즘과의 비교

본 구현을 선정하기 위해 4가지 피크 탐지 방법을 체계적으로 비교하였다.

| 방법 | 원리 | 파라미터 수 | 경계 피크 | LGL 희소 데이터 |
|------|------|------------|----------|----------------|
| `find_peaks` + smoothing | 균일 평활 후 prominence/distance 기반 | 3+ | 별도 로직 필요 | 아티팩트 발생 |
| **Topological Persistence** | 0차 persistent homology | **1** | **자동 처리** | **강건** |
| Savitzky-Golay 영교차 | S-G 1차 미분의 부호 변환점 | 3+ | 불완전 | 불안정 |
| CWT (`find_peaks_cwt`) | 연속 웨이블릿 변환 | 2+ | 불완전 | 과다 검출 |

8개 테스트 케이스에 대한 벤치마크 결과:

| 방법 쌍 | 일치율 |
|---------|--------|
| find_peaks vs Persistence | 88% (7/8) |
| find_peaks vs S-G | 62% (5/8) |
| find_peaks vs CWT | 50% (4/8) |
| Persistence vs S-G | 62% (5/8) |
| Persistence vs CWT | 50% (4/8) |

Topological Persistence는 baseline(find_peaks)과 가장 높은 일치율을 보였으며, 불일치하는 모든 케이스에서 Persistence가 시각적으로 올바른 결과를 제시하였다. 특히:

- **CWT**는 희소 LGL 노드(15~30점)에서 과다 검출(7~9개 피크)이 심각하였다.
- **Savitzky-Golay**는 윈도우 크기에 민감하여 동일 데이터에서도 결과가 불안정하였다.
- **find_peaks**는 과도한 스무딩으로 인접 피크를 병합하는 경향이 있었다.

## 3. 구현 매핑

| 수학 표현 | Python 함수/클래스 | 파일 | 비고 |
|-----------|-------------------|------|------|
| 피크 객체 $(b, d, \ell, r)$ | `_Peak` | `peak_detection.py` | `__slots__` 최적화 |
| $\text{pers}(p) = f[b] - f[d]$ | `_Peak.persistence(seq)` | `peak_detection.py` | $d = \text{None}$이면 $\infty$ |
| 0차 persistent homology | `_persistent_homology(seq)` | `peak_detection.py` | $O(N \log N)$, persistence 내림차순 반환 |
| 전체 파이프라인 | `detect_peaks(t, u_mag, T, phase_boundaries)` | `peak_detection.py` | 전처리 + 지속성 + FWHM |
| Phase-aware 보간 | `_interpolate_phased(t, u, n_interp, phase_boundaries)` | `peak_detection.py` | Phase별 독립 CubicSpline |
| 글로벌 보간 | `_interpolate_for_detection(t, u, n_interp, phase_boundaries)` | `peak_detection.py` | `phase_boundaries=None`이면 글로벌 |
| FWHM 추정 | `estimate_peak_widths(t, u_mag, peak_indices)` | `peak_detection.py` | scipy `peak_widths` + 경계 보정 |
| 임계값 $\alpha$ | `PEAK_PERSISTENCE_RATIO` | `config.py` | 기본값 0.10 |
| 보간 점 수 $N_{\text{interp}}$ | `PEAK_INTERP_POINTS` | `config.py` | 기본값 200 |

### 3.1 핵심 코드

**0차 Persistent Homology 계산**:

```python
def _persistent_homology(seq):
    n = len(seq)
    peaks = []
    idx_to_peak = [None] * n
    indices = sorted(range(n), key=lambda i: seq[i], reverse=True)

    for idx in indices:
        lft = idx_to_peak[idx - 1] if idx > 0 and idx_to_peak[idx - 1] is not None else None
        rgt = idx_to_peak[idx + 1] if idx < n - 1 and idx_to_peak[idx + 1] is not None else None

        if lft is None and rgt is None:
            peaks.append(_Peak(idx))
            idx_to_peak[idx] = len(peaks) - 1
        elif lft is not None and rgt is None:
            peaks[lft].right += 1
            idx_to_peak[idx] = lft
        elif lft is None and rgt is not None:
            peaks[rgt].left -= 1
            idx_to_peak[idx] = rgt
        else:
            # Elder rule: 낮은 피크가 사망
            if seq[peaks[lft].born] > seq[peaks[rgt].born]:
                peaks[rgt].died = idx
                peaks[lft].right = peaks[rgt].right
                idx_to_peak[peaks[lft].right] = idx_to_peak[idx] = lft
            else:
                peaks[lft].died = idx
                peaks[rgt].left = peaks[lft].left
                idx_to_peak[peaks[rgt].left] = idx_to_peak[idx] = rgt

    return sorted(peaks, key=lambda p: p.persistence(seq), reverse=True)
```

**유의미한 피크 필터링**:

```python
threshold = PEAK_PERSISTENCE_RATIO * np.max(u_work)
significant = [p for p in peaks if p.persistence(u_work) >= threshold]
```

## 4. 수치 검증

### 4.1 합성 가우시안 프로파일 테스트

가우시안 함수의 합으로 정의된 합성 추력 프로파일을 사용하여 검증한다:

$$u(t) = \sum_{k=1}^{K} A_k \exp\left(-\frac{(t - c_k)^2}{2\sigma_k^2}\right)$$

| 테스트 케이스 | $K$ | 기대 피크 수 | 검출 결과 | 위치 오차 |
|--------------|-----|-------------|----------|----------|
| 단봉 ($c = T/2$) | 1 | 1 | 1 | $< 0.05T$ |
| 쌍봉 ($c = T/4, 3T/4$) | 2 | 2 | 2 | $< 0.05T$ |
| 삼봉 ($c = T/6, T/2, 5T/6$) | 3 | 3 | 3 | $< 0.05T$ |

### 4.2 엣지 케이스 검증

| 케이스 | 입력 | 기대 결과 | 실제 결과 |
|--------|------|----------|----------|
| 영 신호 | $u(t) = 0$ | 0 피크 | 0 피크 |
| 상수 신호 | $u(t) = c > 0$ | 0 피크 | 0 피크 |
| 경계 피크 (시작) | $u(t) = e^{-3t/T}$ | 1 피크 ($t \approx 0$) | 1 피크 |
| 경계 피크 (끝) | $u(t) = e^{-3(T-t)/T}$ | 1 피크 ($t \approx T$) | 1 피크 |
| 양쪽 경계 + 내부 | 복합 | 3 피크 | 3 피크 |

### 4.3 노이즈 강건성 테스트

NLP 솔버의 해는 다항식/스플라인이므로 Gaussian 노이즈가 아닌 미세 수치 오차 수준의 섭동이 현실적이다. $\sigma = 10^{-3}$ 수준의 섭동을 추가하여 검증한다:

$$u_{\text{noisy}}(t) = \max\left(u_{\text{clean}}(t) + \varepsilon(t),\ 0\right), \quad \varepsilon \sim \mathcal{N}(0, 10^{-3})$$

모든 테스트 케이스에서 섭동 유무에 관계없이 동일한 분류 결과를 반환하였다.

### 4.4 저해상도 보간 테스트

| 입력 노드 수 | 프로파일 | 기대 | 결과 |
|-------------|---------|------|------|
| $N = 20$ | 쌍봉 | 2 피크 | 2 피크 |
| $N = 30$ | 삼봉 | 3 피크 | 3 피크 |

CubicSpline 보간이 15~30개 LGL 노드에서도 충분한 해상도를 제공함을 확인하였다.

### 4.5 기존 데이터베이스 재분류 결과

Topological Persistence 피크 탐지기로 기존 데이터베이스 전체(4개 고도, 1763건 수렴 궤적)를 재분류한 결과:

| 고도 [km] | 기존 class 0/1/2 | 신규 class 0/1/2 | 변경율 |
|-----------|------------------|------------------|--------|
| 400 | 266/61/4 | 25/219/87 | 82.5% |
| 600 | 233/113/77 | 12/167/244 | 79.4% |
| 800 | 214/74/1 | 12/217/60 | 76.8% |
| 1000 | 431/273/16 | 23/405/292 | 71.5% |
| **합계** | **1144/521/98** | **72/1008/683** | **76.3%** |

기존 `find_peaks` 기반 탐지기는 과도한 스무딩으로 인접 피크를 병합하여 unimodal을 과대 추정하고 있었음이 확인되었다. 신규 탐지기에서는 bimodal과 multimodal의 비율이 크게 증가하여 실제 추력 프로파일의 복잡성을 더 정확히 반영한다.

### 4.6 테스트 코드

총 26개 테스트가 `tests/test_peak_detection.py`에 구현되어 있으며, 모두 통과한다:

- `TestUnimodal`: 단봉 탐지 및 분류 (2 tests)
- `TestBimodal`: 쌍봉 탐지 및 분류 (2 tests)
- `TestMultimodal`: 삼봉 탐지 및 분류 (2 tests)
- `TestNoiseRobustness`: 미세 섭동 강건성 (3 tests)
- `TestPhaseStructure`: Phase 구조 결정 (5 tests)
- `TestEdgeCases`: 영 신호, 상수 신호 (2 tests)
- `TestEdgePeaks`: 경계 피크 탐지 (4 tests)
- `TestNonMonotonicTime`: 비단조 시간 처리 (2 tests)
- `TestInterpolation`: 저해상도 보간 (3 tests)

벤치마크 스크립트: `scripts/compare_peak_detectors.py`

## 5. 참고문헌

1. Huber, S. (2021). *Persistent Homology in Data Science*. In: Topological Data Analysis, Springer. — 위상적 지속성의 수학적 기초.
2. Edelsbrunner, H. and Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society. — Persistent homology의 이론적 배경.
3. Huber, S. "Topological Peak Detection." https://www.sthu.org/blog/13-perstopology-peakdetection/index.html — 본 구현의 직접적 참고. 1D 수열에 대한 0차 persistent homology 기반 피크 탐지.
4. Virtanen, P. et al. (2020). "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." *Nature Methods*, 17, 261–272. — `scipy.signal.peak_widths` 함수.
5. de Boor, C. (2001). *A Practical Guide to Splines*. Revised Edition, Springer. — CubicSpline 보간의 수학적 기초.
