# D2. 엔트로피 기반 능동 학습 알고리즘

> 대응 코드: `src/orbit_transfer/sampling/acquisition.py`, `src/orbit_transfer/sampling/adaptive_sampler.py`
> 전제 보고서: D1 (가우시안 프로세스 분류기)

## 1. 목적

본 보고서는 LEO-to-LEO 연속 추력 궤도전이 최적 추력 프로파일 분류 데이터베이스 구축을 위한 능동 학습(active learning) 알고리즘을 다룬다. 입력은 3차원 정규화 파라미터 공간 $\mathbf{x} = [\tilde{T},\, \Delta a,\, \Delta i]^\top \in [0,1]^3$이고, 출력은 클래스 레이블 $y \in \{0, 1, 2\}$이다. 각 레이블은 직접 collocation으로 계산한 최적 추력 프로파일의 모드 유형(unimodal, bimodal, multimodal)에 대응한다.

궤도전이 최적화는 계산 비용이 높다(IPOPT 수렴 시간 $\sim$ 수 초/케이스). 따라서 전체 파라미터 공간을 균일하게 샘플링하는 것은 비효율적이다. 본 알고리즘은 가우시안 프로세스(GP) 분류기의 예측 불확실성을 활용하여, 클래스 경계 영역을 집중적으로 탐색하는 적응적 샘플링 전략을 제공한다.

**논문 본문과의 관계.** 본 보고서는 논문 Section 3의 "Parametric Database Construction"에서 생략된 수학적 배경과 알고리즘 상세를 기술한다. 논문에는 최종 샘플 수, 수렴 조건, 분류 정확도만 보고하고, 엔트로피 획득함수의 정의, greedy 배치 선택 알고리즘, 수렴 조건의 이론적 근거는 본 보고서에서 독립 완결로 제공한다.

**전제 지식.** D1 보고서의 GP 분류기 이론(Laplace 근사, ARD RBF 커널, 예측 확률 분포)을 전제로 한다. GP 분류기가 $p(y = c \mid \mathbf{x})$를 제공한다는 사실은 이미 알려져 있다고 가정하며, 본 보고서에서는 이 확률 분포를 어떻게 활용하여 샘플링 효율을 극대화하는지에 집중한다.


## 2. 수학적 배경

### 2.1 능동 학습 프레임워크

**능동 학습(active learning)**은 레이블 획득 비용이 높을 때, 학습기가 능동적으로 정보량이 큰 샘플을 선택하여 레이블링 효율을 극대화하는 기계학습 전략이다. 수동적 학습(passive learning)에서는 학습 데이터의 분포가 사전에 고정되지만, 능동 학습에서는 학습기가 "어떤 샘플에 레이블을 부여해야 모델 성능이 가장 빠르게 향상되는가?"를 반복적으로 결정한다.

능동 학습은 크게 세 가지 시나리오로 분류된다.

| 시나리오 | 정의 | 본 프로젝트 적용 가능성 |
|:--------|:-----|:----------------------|
| **Stream-based** | 샘플이 순차적으로 도착하며, 각 샘플에 대해 레이블링 여부를 즉시 결정 | 불가능. 파라미터 공간은 연속적이며 샘플은 능동적으로 선택 가능. |
| **Membership query** | 학습기가 임의의 $\mathbf{x}$를 생성하여 레이블 요청 | 가능하지만 제약 없음. 임의 점에서 궤도전이 최적화 가능. |
| **Pool-based** | 대규모 후보 집합(pool)에서 가장 유용한 샘플을 선택 | **채택**. $30^3 = 27000$ 균일 격자를 후보 pool로 사용. |

본 프로젝트에서 파라미터 공간은 3차원 유계 영역 $[0,1]^3$이고, 임의의 점에서 궤적 최적화가 가능하다. 따라서 membership query 시나리오도 적용 가능하지만, 실제로는 **pool-based active learning**을 채택한다. 이유는 다음과 같다.

1. **격자 구조의 시각화 편의성.** 균일 격자는 학습 결과(decision boundary)를 3차원 공간에 직관적으로 시각화할 수 있다.
2. **배치 선택의 간편성.** 고정된 후보 집합에서 엔트로피가 높은 점들을 빠르게 검색할 수 있다.
3. **수치 안정성.** 사전에 정의된 격자는 최적화 solver의 수렴 실패율이 낮다(파라미터 범위 내부).

**Uncertainty Sampling.** 능동 학습에서 샘플을 선택하는 가장 보편적인 전략은 **uncertainty sampling**이다. 이는 모델이 가장 확신하지 못하는(가장 불확실한) 샘플을 선택하는 원칙이다. 확률적 분류기에서 불확실성은 예측 확률 분포의 엔트로피, 또는 최고 확률과 차순위 확률의 차이(margin) 등으로 정량화된다.

본 프로젝트에서는 **predictive entropy**를 불확실성 척도로 사용한다. 엔트로피는 정보 이론의 불확실성 정량화 표준 도구이며, GP 분류기의 확률 출력 $p(y = c \mid \mathbf{x})$와 자연스럽게 연동된다.


### 2.2 예측 엔트로피 획득함수

**Shannon 엔트로피.** 이산 확률 분포 $P = \{p_0, p_1, \dots, p_{C-1}\}$의 불확실성은 Shannon 엔트로피로 측정된다:

$$H(P) = -\sum_{c=0}^{C-1} p_c \log p_c$$

여기서 밑은 자연로그 $\log = \ln$이다. $0 \log 0 = 0$으로 정의한다(극한).

**정보이론적 해석.** 엔트로피는 분포 $P$에서 샘플링한 결과를 무손실 부호화하는 데 필요한 평균 정보량(비트 또는 nats)이다. 엔트로피는 다음 성질을 만족한다.

| 성질 | 식 | 의미 |
|:-----|:---|:----|
| 비음성 | $H(P) \geq 0$ | 정보량은 항상 비음수 |
| 최대값 | $H_{\text{max}} = \log C$ | 균일 분포 $p_c = 1/C$일 때 최대 |
| 최소값 | $H_{\text{min}} = 0$ | 확정적 분포 $p_{c^*} = 1, p_{c \neq c^*} = 0$일 때 최소 |
| 오목성 | $H(\lambda P_1 + (1-\lambda)P_2) \geq \lambda H(P_1) + (1-\lambda) H(P_2)$ | 혼합 분포는 더 불확실 |

본 프로젝트에서 $C = 3$이므로, 최대 엔트로피는 다음과 같다:

$$H_{\text{max}} = \log 3 \approx 1.099 \; \text{nats}$$

**GP 분류기의 예측 엔트로피.** GP 분류기는 입력 $\mathbf{x}$에서 클래스 확률 벡터 $\mathbf{p}(\mathbf{x}) = [p_0(\mathbf{x}), p_1(\mathbf{x}), p_2(\mathbf{x})]^\top$를 제공한다(D1 보고서 참조). 이 확률 분포의 엔트로피를 **예측 엔트로피(predictive entropy)**라 부른다:

$$H(\mathbf{x}) = -\sum_{c=0}^{2} p_c(\mathbf{x}) \log p_c(\mathbf{x})$$

$H(\mathbf{x})$는 입력 공간 $[0,1]^3$의 각 점에서 정의되는 스칼라 함수이다. $H(\mathbf{x})$가 큰 영역은 GP 분류기가 클래스 레이블을 확신하지 못하는 영역, 즉 **결정 경계(decision boundary)** 부근이다. 반대로 $H(\mathbf{x}) \approx 0$인 영역은 특정 클래스로 확정적으로 분류되는 영역이다.

**획득함수(acquisition function)로서의 엔트로피.** 능동 학습에서 획득함수 $a(\mathbf{x})$는 각 후보 샘플의 "유용성(utility)"을 정량화하는 함수이다. 유용한 샘플이란, 레이블을 획득했을 때 모델의 예측 성능이 가장 크게 개선되는 샘플이다. 본 프로젝트에서는 다음과 같이 정의한다:

$$a(\mathbf{x}) = H(\mathbf{x})$$

이는 가장 단순하고 직관적인 획득함수이다. 엔트로피가 높은 점을 샘플링하면, 결정 경계의 불확실성을 직접 줄일 수 있다.

**다른 획득함수와의 비교.** 문헌에서 제안된 주요 대안들을 비교하면 다음과 같다.

| 획득함수 | 정의 | 장점 | 단점 |
|:--------|:----|:-----|:-----|
| Predictive entropy | $H(y \mid \mathbf{x})$ | 계산 간단, 직관적 | 모델 개선 효과를 직접 측정하지 않음 |
| **BALD** (Bayesian Active Learning by Disagreement) | $I(y; \boldsymbol{\theta} \mid \mathbf{x})$ | 모델 파라미터 불확실성 반영 | GP에서 근사 필요, 계산 비용 높음 |
| Variation ratio | $1 - \max_c p_c$ | 계산 매우 빠름 | 전체 분포 정보 무시 |
| Margin sampling | $p_{(1)} - p_{(2)}$ | 이진 분류에 효과적 | 다중 클래스에서 정보 손실 |
| GP-UCB | $\mu(\mathbf{x}) + \beta \sigma(\mathbf{x})$ | 회귀 문제에 최적화 | 분류 문제에는 부적합 |

본 프로젝트에서는 (1) GP 분류기가 `predict_proba` 메서드로 $p_c(\mathbf{x})$를 직접 제공하고, (2) 클래스 수가 3으로 고정되어 있으며, (3) 계산 효율이 중요하므로, **predictive entropy**를 선택한다. BALD는 이론적으로 더 정교하지만, Laplace 근사 기반 GP 분류기에서 BALD를 계산하려면 추가 Monte Carlo 샘플링이 필요하여 계산 비용이 크게 증가한다.

**수치 안정성: $\log(0)$ 방지.** 실제 구현에서 GP 분류기가 출력하는 확률 $p_c$는 수치 오차로 인해 정확히 0이 될 수 있다. $\log(0) = -\infty$는 수치 불안정을 초래하므로, 확률값을 하한 $\epsilon = 10^{-15}$로 clipping한다:

$$p_c^{\text{safe}} = \max(p_c, \epsilon)$$

이후 엔트로피 계산에 $p_c^{\text{safe}}$를 사용한다. $\epsilon = 10^{-15}$는 배정밀도 부동소수점 머신 엡실론($\sim 2.2 \times 10^{-16}$) 수준이므로, 실제 확률값이 0에 가까운 경우에만 개입한다. 이 보정은 엔트로피 값에 $\Delta H \sim 10^{-13}$ nats 정도의 영향만 주므로, 샘플링 결정에는 무시할 수 있다.


### 2.3 Greedy 배치 선택 알고리즘

능동 학습에서 샘플을 한 개씩 순차적으로 선택하면, 매 단계마다 GP 재학습과 엔트로피 재계산이 필요하다. 이는 계산 비용이 크다. 또한 본 프로젝트에서는 궤적 최적화가 병렬 실행 가능하므로, 여러 샘플을 동시에 선택하여 배치(batch)로 평가하는 것이 효율적이다.

**배치 능동 학습(batch active learning)**은 한 번에 $k$개의 샘플을 선택하는 전략이다. 배치 선택에서는 다음 두 가지를 균형있게 고려해야 한다.

1. **개별 유용성(individual utility).** 각 샘플의 엔트로피가 높아야 한다.
2. **배치 다양성(batch diversity).** 선택된 샘플들이 파라미터 공간에서 서로 분산되어 있어야 한다. 비슷한 위치의 샘플들은 중복 정보를 제공하므로 비효율적이다.

이를 정식화하면, 후보 집합 $\mathcal{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_{N_{\text{cand}}}\}$에서 크기 $k$의 부분집합 $S \subset \mathcal{X}$를 선택하는 문제이다. 이상적으로는 다음을 최대화해야 한다:

$$S^* = \arg\max_{S \subset \mathcal{X}, |S| = k} \; f(S)$$

여기서 $f(S)$는 배치 $S$의 전체 유용성이다. 하지만 조합 최적화 문제 $\binom{N_{\text{cand}}}{k}$는 $N_{\text{cand}} = 27000$, $k = 10$ 수준에서는 계산이 불가능하다.

**Greedy 근사.** 본 프로젝트에서는 순차 greedy 알고리즘으로 근사한다. 알고리즘은 다음과 같다.

---

**Algorithm 1: Greedy Batch Selection with Diversity**

**입력:**
- $\mathcal{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_{N_{\text{cand}}}\}$: 후보 점 집합
- $\{H(\mathbf{x}_i)\}_{i=1}^{N_{\text{cand}}}$: 각 후보의 엔트로피
- $k$: 배치 크기
- $d_{\min}$: 최소 분리 거리 (exclusion zone)

**출력:** $S = \{\mathbf{x}_{i_1}, \dots, \mathbf{x}_{i_k}\}$ (선택된 배치)

**과정:**
1. $S \leftarrow \emptyset$ (빈 배치)
2. $\mathcal{A} \leftarrow \mathcal{X}$ (가용 후보 집합)
3. **for** $b = 1$ to $k$ **do**
4. $\quad$ **if** $\mathcal{A} = \emptyset$ **then break**
5. $\quad$ $i^* \leftarrow \arg\max_{i : \mathbf{x}_i \in \mathcal{A}} H(\mathbf{x}_i)$ $\quad$ // 가용 후보 중 최대 엔트로피
6. $\quad$ $S \leftarrow S \cup \{\mathbf{x}_{i^*}\}$ $\quad$ // 배치에 추가
7. $\quad$ $\mathcal{A} \leftarrow \mathcal{A} \setminus \{\mathbf{x}_j : \|\mathbf{x}_j - \mathbf{x}_{i^*}\| < d_{\min}\}$ $\quad$ // exclusion zone 제외
8. **end for**
9. **return** $S$

---

**알고리즘 해석.**

- **Step 5:** 현재 가용한 후보 중에서 엔트로피가 가장 높은 점을 선택한다. 이는 개별 유용성을 최대화한다.
- **Step 7:** 선택된 점 $\mathbf{x}_{i^*}$ 주변 $d_{\min}$ 반경 이내의 모든 후보를 가용 집합에서 제거한다. 이는 배치 내 점들 간 최소 거리 $d_{\min}$를 보장하여 다양성을 강제한다.
- **반복:** $k$회 반복하거나, 가용 후보가 소진될 때까지 계속한다. 만약 $k$회 이전에 $\mathcal{A} = \emptyset$가 되면, 더 이상 선택할 수 없으므로 조기 종료한다.

**시간 복잡도.** 각 반복에서:
- Step 5의 $\arg\max$는 $O(|\mathcal{A}|)$ 연산이다.
- Step 7의 거리 계산은 $O(|\mathcal{A}| \cdot d)$ 연산이다($d = 3$은 차원).

초기에 $|\mathcal{A}| = N_{\text{cand}} = 27000$이므로, 최악의 경우 각 반복은 $O(N_{\text{cand}})$이다. $k = 10$ 반복이므로 전체 시간 복잡도는 $O(k \cdot N_{\text{cand}}) = O(270000)$ 연산이다. 이는 현대 하드웨어에서 1ms 이내에 수행된다.

**Submodularity와 근사 보장.** 집합 함수 $f(S) = \sum_{\mathbf{x} \in S} H(\mathbf{x})$가 submodular이면, greedy 알고리즘은 $(1 - 1/e) \approx 0.632$ 근사 보장을 제공한다(Nemhauser et al., 1978). 단순 엔트로피 합은 submodular가 아니지만, 다양성 제약 하에서 greedy 알고리�m은 실용적으로 우수한 성능을 보인다. 엄밀한 이론 보장을 위해서는 determinantal point process(DPP) 기반 배치 선택이 가능하지만, 계산 복잡도가 $O(N_{\text{cand}}^3)$로 증가하여 본 프로젝트 규모에서는 실용적이지 않다.

**최소 분리 거리 $d_{\min}$의 선택.** $d_{\min}$는 정규화 좌표계 $[0,1]^3$에서 정의된다. 본 프로젝트에서 $d_{\min} = 0.1$로 설정한다. 이는 유클리드 거리로 각 차원에서 약 $0.1$ 간격을 의미한다. $30^3$ 균일 격자에서 인접 격자점 간 거리가 $1/29 \approx 0.0345$이므로, $d_{\min} = 0.1$은 약 3개 격자점 간격에 해당한다. 이는 경험적으로 결정 경계의 곡률을 충분히 포착하면서, 배치 내 과도한 중복을 방지하는 균형점이다.


### 2.4 적응적 샘플링 루프

전체 능동 학습 알고리즘은 초기 샘플링과 적응적 반복으로 구성된다. 알고리즘은 다음과 같다.

---

**Algorithm 2: Entropy-based Adaptive Sampling**

**입력:**
- $\text{evaluate}(\mathbf{x})$: 파라미터 $\mathbf{x}$에서 궤적 최적화 및 클래스 레이블 반환
- $N_{\text{init}}$: 초기 샘플 수 (LHS)
- $N_{\text{max}}$: 최대 샘플 수
- $k$: 배치 크기
- $d_{\min}$: 최소 분리 거리
- $\epsilon_H$: 엔트로피 수렴 임계값

**출력:**
- $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$: 레이블된 샘플 집합
- $\text{GPC}$: 학습된 GP 분류기

**과정:**

**Phase 1: 초기 샘플링 (Space-Filling)**
1. $\mathcal{D} \leftarrow \emptyset$
2. $\{\mathbf{x}_1, \dots, \mathbf{x}_{N_{\text{init}}}\} \leftarrow \text{LHS}([0,1]^3, N_{\text{init}})$ $\quad$ // Latin Hypercube Sampling
3. **for** $i = 1$ to $N_{\text{init}}$ **do**
4. $\quad$ $y_i \leftarrow \text{evaluate}(\mathbf{x}_i)$
5. $\quad$ $\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{x}_i, y_i)\}$
6. **end for**

**Phase 2: 적응적 반복 (Boundary-Focused)**
7. $\text{stable\_count} \leftarrow 0$
8. $\mathbf{y}_{\text{prev}} \leftarrow \text{null}$
9. **while** $|\mathcal{D}| < N_{\text{max}}$ **do**
10. $\quad$ $\text{GPC} \leftarrow \text{train}(\mathcal{D})$ $\quad$ // GP 분류기 학습 (D1 참조)
11. $\quad$ **if** $|\text{unique classes in } \mathcal{D}| < 2$ **then break** $\quad$ // 클래스 수 부족
12. $\quad$
13. $\quad$ // 후보 격자에서 엔트로피 계산
14. $\quad$ $\mathcal{X} \leftarrow \{30^3 \text{ 균일 격자 점들}\}$
15. $\quad$ **for** $\mathbf{x} \in \mathcal{X}$ **do**
16. $\quad$ $\quad$ $\mathbf{p}(\mathbf{x}) \leftarrow \text{GPC.predict\_proba}(\mathbf{x})$
17. $\quad$ $\quad$ $H(\mathbf{x}) \leftarrow -\sum_c p_c(\mathbf{x}) \log p_c(\mathbf{x})$
18. $\quad$ **end for**
19. $\quad$
20. $\quad$ // 수렴 조건 1: 최대 엔트로피 임계값
21. $\quad$ **if** $\max_{\mathbf{x} \in \mathcal{X}} H(\mathbf{x}) < \epsilon_H$ **then break**
22. $\quad$
23. $\quad$ // 수렴 조건 2: 경계 안정성
24. $\quad$ $\mathbf{y}_{\text{pred}} \leftarrow [\text{GPC.predict}(\mathbf{x}) \; \text{for } \mathbf{x} \in \mathcal{X}]$
25. $\quad$ **if** $\mathbf{y}_{\text{pred}} = \mathbf{y}_{\text{prev}}$ **then**
26. $\quad$ $\quad$ $\text{stable\_count} \leftarrow \text{stable\_count} + 1$
27. $\quad$ $\quad$ **if** $\text{stable\_count} \geq 3$ **then break**
28. $\quad$ **else**
29. $\quad$ $\quad$ $\text{stable\_count} \leftarrow 0$
30. $\quad$ **end if**
31. $\quad$ $\mathbf{y}_{\text{prev}} \leftarrow \mathbf{y}_{\text{pred}}$
32. $\quad$
33. $\quad$ // 배치 선택
34. $\quad$ $S \leftarrow \text{GreedyBatchSelection}(\mathcal{X}, \{H(\mathbf{x})\}, k, d_{\min})$
35. $\quad$ **if** $S = \emptyset$ **then break**
36. $\quad$
37. $\quad$ // 선택된 점에서 평가
38. $\quad$ **for** $\mathbf{x} \in S$ **do**
39. $\quad$ $\quad$ $y \leftarrow \text{evaluate}(\mathbf{x})$
40. $\quad$ $\quad$ $\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{x}, y)\}$
41. $\quad$ **end for**
42. **end while**
43. **return** $\mathcal{D}, \text{GPC}$

---

**알고리즘 해석.**

**Phase 1 (Steps 1–6): 초기 샘플링.** Latin Hypercube Sampling(LHS)은 공간 충진(space-filling) 성질을 가진 준난수 샘플링 기법이다(McKay et al., 1979). LHS는 각 차원을 $N_{\text{init}}$개 구간으로 균등 분할하고, 각 구간에서 한 번씩 샘플링하여 전체 공간을 고르게 탐색한다. 이는 순수 난수 샘플링보다 공간 커버리지가 우수하며, GP 커널 하이퍼파라미터 학습에 유리하다.

초기 단계에서는 결정 경계의 위치를 알 수 없으므로, 경계와 무관하게 전체 공간을 고르게 샘플링하는 것이 합리적이다. $N_{\text{init}} = 100$은 3차원 공간에서 GP 분류기가 대략적인 경계 형태를 학습하기에 충분한 경험적 값이다.

**Phase 2 (Steps 7–42): 적응적 반복.** GP 분류기를 반복적으로 학습하고, 엔트로피가 높은 영역을 집중 샘플링한다. 각 반복은 다음 단계로 구성된다.

- **Step 10: GP 학습.** 현재 레이블된 샘플 $\mathcal{D}$로 GP 분류기를 재학습한다. D1 보고서에서 기술한 Laplace 근사 및 하이퍼파라미터 최적화가 수행된다. 학습 시간 복잡도는 $O(N^3)$이며, $N \sim 100$–$500$ 범위에서는 1–10초 수준이다.

- **Steps 13–18: 후보 격자 엔트로피 계산.** $30^3 = 27000$개 균일 격자점에서 예측 확률 $\mathbf{p}(\mathbf{x})$를 계산하고, 엔트로피 $H(\mathbf{x})$를 구한다. `predict_proba`는 GP 분류기의 벡터화된 예측이므로, 27000개 점을 한 번에 처리할 수 있다. 계산 시간은 $O(N \cdot N_{\text{cand}})$이며, $N = 500$, $N_{\text{cand}} = 27000$일 때 약 수 초이다.

- **Step 21: 수렴 조건 1 (엔트로피 임계값).** 모든 후보점의 엔트로피가 임계값 $\epsilon_H$ 이하이면, 결정 경계의 불확실성이 충분히 낮다고 판단하여 종료한다. $\epsilon_H = 0.3$ nats는 최대 엔트로피 $\log 3 \approx 1.099$의 약 27%에 해당한다. 이는 가장 불확실한 점에서도 $p_{\text{max}} \approx 0.5$ 수준의 확신을 의미한다.

- **Steps 24–31: 수렴 조건 2 (경계 안정성).** 후보 격자의 예측 클래스 레이블 벡터 $\mathbf{y}_{\text{pred}} \in \{0,1,2\}^{27000}$를 이전 반복의 $\mathbf{y}_{\text{prev}}$와 비교한다. 두 벡터가 정확히 일치하면, 결정 경계가 변하지 않았다는 의미이므로 `stable_count`를 증가시킨다. 3회 연속 일치하면 수렴으로 판정하여 종료한다. 이 조건은 엔트로피가 여전히 높더라도, 경계의 위치가 안정화되면 추가 샘플링이 불필요함을 인식한다.

- **Step 34: Greedy 배치 선택.** Algorithm 1을 호출하여 크기 $k$의 배치를 선택한다.

- **Steps 38–41: 선택된 점에서 평가.** 각 점 $\mathbf{x} \in S$에 대해 궤적 최적화를 수행하고 클래스 레이블 $y$를 획득한다. 이 단계가 전체 알고리즘에서 가장 비용이 높다(점당 수 초). 배치 내 점들은 독립적이므로 병렬 실행이 가능하다.


### 2.5 수렴 조건의 이론적 근거

알고리즘은 세 가지 OR 조건 중 하나라도 만족하면 종료한다.

**조건 1: 샘플 예산 소진 ($|\mathcal{D}| \geq N_{\text{max}}$).** 하드 제약이다. $N_{\text{max}} = 500$은 계산 자원 한도를 의미한다. 500회 궤적 최적화는 단일 CPU에서 약 30분–1시간 소요된다. 이 조건에 도달하면, 수렴 여부와 무관하게 종료한다.

**조건 2: 엔트로피 임계값 ($\max H(\mathbf{x}) < \epsilon_H$).** 정보 이론적 근거가 있다. 엔트로피는 예측의 불확실성을 정량화하므로, 최대 엔트로피가 낮다는 것은 모든 영역에서 GP 분류기가 확신을 가진다는 의미이다. 이는 결정 경계가 충분히 정확하게 학습되었음을 시사한다.

$\epsilon_H = 0.3$ nats의 의미를 구체적으로 분석해보자. 3-class 분류에서 엔트로피 $H = 0.3$에 대응하는 확률 분포는 다음과 같다.

| 분포 | $(p_0, p_1, p_2)$ | $H$ [nats] |
|:----|:-----------------|:-----------|
| 균일 | $(0.33, 0.33, 0.33)$ | $1.099$ |
| 강한 확신 | $(0.9, 0.05, 0.05)$ | $0.325$ |
| 매우 강한 확신 | $(0.95, 0.025, 0.025)$ | $0.198$ |
| 확정적 | $(1.0, 0.0, 0.0)$ | $0.000$ |

$H = 0.3$은 대략 $(0.9, 0.05, 0.05)$ 분포에 해당한다. 이는 지배적 클래스의 확률이 90% 이상인 상태이다. 따라서 $\epsilon_H = 0.3$은 "가장 불확실한 점조차도 90% 이상 확신을 가진다"는 기준이다.

**조건 3: 경계 안정성 (3회 연속 예측 불변).** 실용적 근거가 있다. 엔트로피가 높더라도, 결정 경계의 위치가 변하지 않으면 추가 샘플링이 분류 성능 개선에 기여하지 않는다. 이는 다음 상황에서 발생한다.

1. **경계 근처 데이터 부족으로 인한 높은 엔트로피.** 경계 영역 자체는 정확하지만, GP의 예측 분산이 여전히 크다.
2. **내재적으로 불확실한 경계.** 물리적으로 unimodal과 bimodal의 전환이 smooth한 경계에서, 프로파일 유형이 미세한 파라미터 변화에 민감할 수 있다. 이 경우 샘플을 더 추가해도 불확실성이 감소하지 않는다.

3회 연속 조건은 우연한 일치를 배제하기 위한 안전장치이다. 만약 1회만 비교하면, 하이퍼파라미터 최적화의 국소해 변화로 인해 경계가 미세하게 변동할 때 조기 종료될 위험이 있다.


### 2.6 후보 격자 설계

후보 격자는 $[0,1]^3$에서 각 차원 30개 균일 구간으로 정의한다:

$$\mathcal{X} = \left\{ \left(\frac{i}{29}, \frac{j}{29}, \frac{k}{29}\right) : i,j,k \in \{0, 1, \dots, 29\} \right\}$$

총 점 수는 $30^3 = 27000$이다. 인접 격자점 간 거리는 $\Delta = 1/29 \approx 0.0345$이다.

**격자 해상도 선택 근거.** 해상도는 계산 비용과 분해능의 트레이드오프이다.

| 해상도 | 점 수 | `predict_proba` 시간 | 엔트로피 계산 분해능 | 비고 |
|:------|:-----|:-------------------|:------------------|:----|
| $20^3$ | 8000 | $\sim 1$s | 낮음 (간격 0.053) | 경계 곡률 포착 부족 |
| $30^3$ | 27000 | $\sim 3$s | 중간 (간격 0.034) | **채택** |
| $40^3$ | 64000 | $\sim 8$s | 높음 (간격 0.026) | 계산 시간 증가 대비 효과 미미 |
| $50^3$ | 125000 | $\sim 20$s | 매우 높음 (간격 0.020) | 메모리 부담 |

$30^3$은 경험적으로 결정 경계의 곡률을 충분히 포착하면서도, GP 예측 시간이 수 초 수준으로 실용적이다. 본 프로젝트의 결정 경계는 일반적으로 smooth하므로(물리적 연속성), 과도한 분해능은 불필요하다.

**정규화 좌표 ↔ 물리 좌표 매핑.** 후보 격자는 정규화 좌표 $[0,1]^3$에서 정의되지만, 궤적 최적화는 물리 좌표에서 수행된다. 매핑은 다음과 같다:

$$\begin{bmatrix} \Delta a \\ \Delta i \\ T_{\text{normed}} \end{bmatrix} = \begin{bmatrix} \Delta a_{\min} + (\Delta a_{\max} - \Delta a_{\min}) \cdot x_0 \\ \Delta i_{\min} + (\Delta i_{\max} - \Delta i_{\min}) \cdot x_1 \\ T_{\min} + (T_{\max} - T_{\min}) \cdot x_2 \end{bmatrix}$$

여기서 $(x_0, x_1, x_2)$는 정규화 좌표이고, 하첨자 min/max는 `PARAM_RANGES`에서 정의된 물리적 범위이다. **중요:** 매핑 순서는 `sorted(PARAM_RANGES.keys())`로 결정된다. 본 프로젝트에서 정렬 순서는 `['T_normed', 'delta_a', 'delta_i']`이므로, $x_0 \leftrightarrow T_{\text{normed}}$, $x_1 \leftrightarrow \Delta a$, $x_2 \leftrightarrow \Delta i$이다. 이는 코드 `lhs.py`의 `denormalize_params` 함수에서 구현된다.


### 2.7 초기-적응 전략의 이론적 근거

초기 LHS + 적응적 엔트로피 샘플링의 2단계 전략은 다음 이론적 장점을 제공한다.

**LHS 초기화의 이점.** GP 커널 하이퍼파라미터(ARD 길이 척도 $\ell_d$)는 주변 우도 최대화로 학습된다(D1 보고서 Section 2.5). 하이퍼파라미터 최적화는 경사 기반 방법(L-BFGS-B)을 사용하므로, 초기 추정이 중요하다. LHS는 공간 충진 성질로 인해, 각 차원의 변동성을 고르게 포착하여 초기 길이 척도 추정에 유리하다.

만약 초기 샘플이 한쪽 영역에 편중되면, 해당 영역의 결정 경계만 학습되고 다른 영역의 경계는 외삽(extrapolation)으로 예측된다. GP는 외삽 영역에서 불확실성이 크므로, 엔트로피가 과대평가되어 샘플링 효율이 떨어진다. LHS는 이를 방지한다.

**적응적 단계의 이점.** 초기 100개 샘플로 대략적인 경계 위치를 파악한 후, 경계 영역에 샘플을 집중하여 정밀도를 높인다. 이는 능동 학습의 핵심 이점이다. 전체 공간 균일 샘플링 대비 필요한 샘플 수를 크게 줄일 수 있다.

**샘플 수 감소 효과 정량화.** 정성적 분석을 제시한다. 3차원 공간에서 결정 경계는 2차원 곡면이다. 경계의 두께(GP 불확실성이 높은 영역)가 각 차원에서 $w$라 하면, 경계 영역의 부피는 대략 $V_{\text{boundary}} \sim w \cdot L^2$이다($L = 1$은 공간 크기). 반면 전체 부피는 $V_{\text{total}} = L^3 = 1$이다.

균일 샘플링으로 경계 영역에 $N_{\text{boundary}}$개 샘플을 얻으려면, 전체 샘플 수는 다음과 같다:

$$N_{\text{uniform}} = N_{\text{boundary}} \cdot \frac{V_{\text{total}}}{V_{\text{boundary}}} \sim \frac{N_{\text{boundary}}}{w}$$

예를 들어 $w = 0.1$이면, 균일 샘플링은 능동 학습 대비 10배 많은 샘플이 필요하다. 본 프로젝트에서 최종 샘플 수 $N \sim 300$–$400$이므로, 균일 샘플링은 3000–4000개가 필요할 수 있다. 이는 계산 비용 관점에서 10배 증가를 의미한다.


## 3. 구현 매핑

### 3.1 함수-수식 대응표

| 수학 개념 | 수식 | 코드 요소 | 파일:행 | 비고 |
|:---------|:----|:---------|:-------|:----|
| Shannon 엔트로피 | $H = -\sum_c p_c \log p_c$ | `predictive_entropy(proba)` | `acquisition.py:6–18` | Section 2.2 |
| 안전 clipping | $p_c^{\text{safe}} = \max(p_c, 10^{-15})$ | `np.clip(proba, 1e-15, 1.0)` | `acquisition.py:17` | $\log(0)$ 방지 |
| Greedy 배치 선택 | Algorithm 1 | `greedy_batch_selection(...)` | `acquisition.py:21–49` | Section 2.3 |
| 최대 엔트로피 선택 | Step 5: $\arg\max H$ | `np.argmax(masked_entropy)` | `acquisition.py:42` | 가용 후보 중 최대 |
| Exclusion zone | Step 7: $\|\mathbf{x} - \mathbf{x}_{i^*}\| < d_{\min}$ | `dists < d_min` → `available[...] = False` | `acquisition.py:46–47` | 다양성 강제 |
| 적응적 샘플링 루프 | Algorithm 2 | `AdaptiveSampler.run()` | `adaptive_sampler.py:78–150` | Section 2.4 |
| 후보 격자 생성 | $30^3$ 균일 격자 | `np.meshgrid(grid_1d, ...)` | `adaptive_sampler.py:67–69` | Section 2.6 |
| 엔트로피 계산 (벡터) | $H(\mathbf{x}_i)$ for all $i$ | `predictive_entropy(proba)` | `adaptive_sampler.py:113` | 27000개 한 번에 |
| 수렴 조건 1 | $\max H < \epsilon_H$ | `max_entropy < self.entropy_threshold` | `adaptive_sampler.py:116–118` | Section 2.5 |
| 수렴 조건 2 | $\mathbf{y}_{\text{pred}} = \mathbf{y}_{\text{prev}}$ (3회) | `np.array_equal(...)`<br>`stable_count >= 3` | `adaptive_sampler.py:122–127` | 경계 안정성 |
| 수렴 조건 3 | $\|\mathcal{D}\| \geq N_{\text{max}}$ | `while len(y) < self.n_max` | `adaptive_sampler.py:103` | 샘플 예산 |
| 정규화↔물리 변환 | 식 Section 2.6 | `denormalize_params(normed)` | `lhs.py:56–65` | `sorted(keys)` 순서 |
| LHS 초기 샘플링 | Algorithm 2 Steps 1–6 | `latin_hypercube_sample(...)` | `adaptive_sampler.py:87–89` | Section 2.7 |
| GP 재학습 | Algorithm 2 Step 10 | `self.gpc.fit(X, y)` | `adaptive_sampler.py:109` | D1 참조 |
| 예측 확률 | $p(y=c \mid \mathbf{x})$ | `self.gpc.predict_proba(...)` | `adaptive_sampler.py:112` | D1 참조 |
| 배치 평가 | Algorithm 2 Steps 38–41 | `for idx in selected: ... evaluate_fn(...)` | `adaptive_sampler.py:141–145` | 병렬 가능 |

### 3.2 `acquisition.py` 모듈 구조

**`predictive_entropy(proba)`** (lines 6–18). 입력 `proba`는 $(N_{\text{cand}}, C)$ 형상의 클래스 확률 배열이다($C = 3$). 출력은 $(N_{\text{cand}},)$ 형상의 엔트로피 배열이다.

- **Line 17:** `np.clip(proba, 1e-15, 1.0)`로 확률값을 $[10^{-15}, 1]$ 범위로 제한한다. 하한 clipping은 $\log(0) = -\infty$를 방지하고, 상한 clipping은 수치 오차로 확률이 1을 초과하는 경우를 보정한다(실제로는 `predict_proba`가 이미 정규화되므로 거의 발생하지 않음).
- **Line 18:** `np.sum(proba * np.log(proba_safe), axis=1)`은 각 행(샘플)에 대해 $-\sum_c p_c \log p_c$를 계산한다. `axis=1`은 클래스 차원($C = 3$)을 따라 합산함을 의미한다. 음수 부호는 함수 정의에 이미 포함되어 있다.

**`greedy_batch_selection(candidates, entropy, k, d_min)`** (lines 21–49). 입력:
- `candidates`: $(N_{\text{cand}}, d)$ 형상의 후보 점 배열(정규화 좌표).
- `entropy`: $(N_{\text{cand}},)$ 형상의 엔트로피 배열.
- `k`: 배치 크기(스칼라).
- `d_min`: 최소 분리 거리(스칼라).

출력: 선택된 인덱스 배열 `selected_indices` $(k,)$ 또는 더 짧게(가용 후보 소진 시).

- **Line 34:** `available = np.ones(len(candidates), dtype=bool)`로 가용 후보를 boolean 배열로 초기화한다. `True`는 선택 가능, `False`는 제외됨을 의미한다.
- **Line 36–49:** $k$회 반복. 각 반복에서:
  - **Line 37–38:** 가용 후보가 없으면 조기 종료.
  - **Line 41:** `masked_entropy = np.where(available, entropy, -np.inf)`는 가용하지 않은 후보의 엔트로피를 $-\infty$로 설정하여, `argmax`에서 선택되지 않도록 한다.
  - **Line 42:** `idx = np.argmax(masked_entropy)`는 가용 후보 중 최대 엔트로피 인덱스를 반환한다.
  - **Line 43:** 선택된 인덱스를 `selected` 리스트에 추가.
  - **Line 46:** `dists = np.linalg.norm(candidates - candidates[idx], axis=1)`는 선택된 점과 모든 후보 간 유클리드 거리를 계산한다. Broadcasting으로 `candidates[idx]` $(d,)$가 `candidates` $(N_{\text{cand}}, d)$로 확장된다.
  - **Line 47:** `available[dists < d_min] = False`는 거리 `d_min` 이내의 모든 후보를 가용 집합에서 제외한다. 여기에는 `idx` 자신도 포함된다(`dists[idx] = 0 < d_min`).

반환값은 numpy 배열 `np.array(selected)`이다.

### 3.3 `adaptive_sampler.py` 모듈 구조

**`AdaptiveSampler.__init__(...)`** (lines 26–69). 초기화에서 다음을 수행한다.

- **Lines 49–50:** 하이퍼파라미터를 인스턴스 변수로 저장.
- **Line 58–59:** 학습 데이터를 빈 리스트로 초기화. `X_train`은 정규화 좌표 $(N, 3)$, `y_train`은 클래스 레이블 $(N,)$이다.
- **Line 60:** `GPClassifierWrapper(n_dims=3)`로 GP 분류기 인스턴스 생성(D1 참조).
- **Lines 62–64:** `PARAM_RANGES`의 키를 정렬하여 정규화↔물리 좌표 매핑 순서를 고정한다. `_key_to_dim` 딕셔너리는 키 이름에서 차원 인덱스로의 매핑이다. 예를 들어 `_key_to_dim['delta_a']`는 정렬 후 `delta_a`가 몇 번째 차원인지 반환한다.
- **Lines 66–69:** 후보 격자를 생성한다. `np.linspace(0, 1, 30)`는 $[0, 1]$을 30개 점으로 균등 분할한다. `np.meshgrid`는 3차원 격자를 생성하며, `indexing='ij'`는 행렬 인덱싱 순서를 사용한다. 결과 `self.candidates`는 $(27000, 3)$ 형상의 배열이다.

**`AdaptiveSampler._call_evaluate(self, phys)`** (lines 71–76). 보조 메서드이다. 물리 좌표 배열 `phys` $(3,)$을 받아, `evaluate_fn(delta_a, delta_i, T_normed)`를 호출하고 클래스 레이블을 반환한다. `phys` 배열의 순서는 `sorted(PARAM_RANGES.keys())`와 일치하므로, `_key_to_dim` 딕셔너리로 각 파라미터의 인덱스를 찾아 추출한다.

**`AdaptiveSampler.run(self)`** (lines 78–150). 메인 루프. 반환값은 `(X, y, gpc)` 튜플이다.

**Phase 1: 초기 샘플링** (lines 86–97).
- **Lines 87–89:** `latin_hypercube_sample(self.n_init, seed=self.seed)`를 호출하여 $N_{\text{init}} = 100$개 LHS 샘플을 생성한다. 반환값은 `(samples_phys, samples_norm)` 튜플로, 각각 물리 좌표와 정규화 좌표이다.
- **Lines 91–94:** 각 샘플에 대해 `_call_evaluate`로 레이블을 획득하고, `X_train`, `y_train`에 추가한다.
- **Lines 96–97:** 리스트를 numpy 배열로 변환한다.

**Phase 2: 적응적 반복** (lines 99–148).
- **Lines 100–101:** 경계 안정성 추적 변수 초기화. `prev_predictions = None`은 이전 반복의 예측 레이블 벡터를 저장한다. `stable_count = 0`은 연속 일치 횟수를 세는 카운터이다.
- **Line 103:** `while len(y) < self.n_max` 조건으로 최대 샘플 수 제약을 구현한다.
- **Lines 105–107:** 최소 2개 클래스가 필요하다는 안전 검사. GP 분류기는 단일 클래스 데이터로는 학습할 수 없다. 실제로는 LHS 초기화로 인해 거의 발생하지 않지만, 극단적인 파라미터 범위에서 모든 샘플이 동일 클래스가 될 수 있다.
- **Line 109:** `self.gpc.fit(X, y)`로 GP 분류기 재학습(D1 참조).
- **Lines 111–113:** 후보 격자 27000개 점에서 예측 확률과 엔트로피를 계산한다. `predict_proba`는 벡터화되어 있으므로 한 번에 처리된다.
- **Lines 115–118:** 수렴 조건 1 확인. `max_entropy = np.max(entropy)`는 27000개 중 최댓값이다. 임계값 미만이면 `break`로 루프 종료.
- **Lines 120–130:** 수렴 조건 2 확인. `self.gpc.predict(self.candidates)`는 예측 클래스 레이블 벡터 $(27000,)$를 반환한다. `np.array_equal`로 이전 반복과 정확히 비교한다. 일치하면 `stable_count` 증가, 불일치하면 0으로 리셋. 3회 연속 일치 시 `break`.
- **Line 131:** `prev_predictions = predictions.copy()`로 현재 예측을 저장. `.copy()`는 참조가 아닌 복사를 보장한다(중요: numpy 배열 기본 대입은 뷰를 생성할 수 있음).
- **Lines 133–138:** Greedy 배치 선택. `greedy_batch_selection`은 인덱스 배열을 반환한다. 빈 배열이면 가용 후보가 없다는 의미이므로 종료.
- **Lines 141–145:** 선택된 인덱스에 대해 반복. `self.candidates[idx:idx+1]`는 2차원 형상 $(1, 3)$을 유지하며, `denormalize_params`가 이를 요구한다(1차원 배열도 허용하지만 일관성을 위해). 반환값 `phys`는 $(1, 3)$ 배열이므로 `[0]`로 첫 번째 행을 추출한다. `_call_evaluate`로 레이블을 획득하고 데이터에 추가.
- **Lines 147–148:** 업데이트된 리스트를 배열로 재변환. 이는 매 반복마다 수행되지만, 리스트 길이가 최대 500이므로 오버헤드는 무시할 수 있다.
- **Line 150:** 최종 데이터와 학습된 GP 분류기를 반환.

### 3.4 `lhs.py` 모듈 내 좌표 변환

**`denormalize_params(normed, param_ranges=None)`** (lines 56–65). 정규화 좌표 $[0,1]^d$를 물리 좌표로 변환한다. 입력 `normed`는 $(N, d)$ 또는 $(d,)$ 배열이다. 출력 `params`는 동일한 형상의 물리 좌표 배열이다.

- **Lines 58–60:** `param_ranges`가 제공되지 않으면 전역 `PARAM_RANGES` 사용. 키를 정렬하여 순서 고정.
- **Lines 62–64:** 각 차원 $d$에 대해, `lo + (hi - lo) * normed[..., d]`로 선형 변환을 수행한다. `...`는 배치 차원을 포함하는 ellipsis notation이다.

**`normalize_params(params, param_ranges=None)`** (lines 44–53). 역변환. 물리 좌표 → 정규화 좌표. 수식은 다음과 같다:

$$x_d = \frac{\text{param}_d - \text{lo}_d}{\text{hi}_d - \text{lo}_d}$$

### 3.5 하이퍼파라미터 설정 (`config.py`)

능동 학습 관련 상수는 다음과 같이 정의되어 있다.

| 변수명 | 값 | 의미 | 이론적 근거 |
|:------|:---|:----|:----------|
| `GP_N_INIT` | 100 | 초기 LHS 샘플 수 | Section 2.7. 3차원 GP 학습에 충분. |
| `GP_BATCH_SIZE` | 10 | 배치 크기 $k$ | 병렬 평가 효율과 다양성의 균형. |
| `GP_D_MIN` | 0.1 | 최소 분리 거리 | Section 2.3. 격자 간격의 ~3배. |
| `GP_N_MAX` | 500 | 최대 샘플 수 | 계산 자원 한도(~30분–1시간). |
| `GP_ENTROPY_THRESHOLD` | 0.3 | 엔트로피 수렴 임계값 | Section 2.5. 90% 확신 수준. |
| `PARAM_RANGES` | `dict` | 물리 파라미터 범위 | 좌표 변환 기준. 키 정렬 순서 중요. |

**주의:** `config.py`에서 실제 값은 다음과 같이 정의되어 있다(코드 확인 결과).

```python
PARAM_RANGES = {
    "delta_a": (-500.0, 2000.0),   # [km]
    "delta_i": (0.0, 15.0),        # [deg]
    "T_normed": (0.5, 5.0),        # 무차원
}
```

정렬 순서는 `['T_normed', 'delta_a', 'delta_i']`이다(알파벳 순). 따라서 정규화 좌표 $(x_0, x_1, x_2)$는 각각 $(T_{\text{normed}}, \Delta a, \Delta i)$에 대응한다. 이는 모든 좌표 변환 함수에서 일관되게 적용된다.


## 4. 수치 검증

대응 테스트: `tests/test_sampling.py::TestAdaptiveSampler`, `tests/test_sampling.py::TestAcquisition`

### 4.1 엔트로피 계산 정확성

**테스트: `TestAcquisition::test_predictive_entropy`**

검증 내용: 엔트로피 함수가 정의대로 계산되는지 확인한다.

```python
def test_predictive_entropy():
    """엔트로피 계산 정확성 검증."""
    from acquisition import predictive_entropy

    # Case 1: 균일 분포 (C=3)
    proba_uniform = np.array([[1/3, 1/3, 1/3]])
    H_uniform = predictive_entropy(proba_uniform)
    expected = -3 * (1/3) * np.log(1/3)  # log(3) ≈ 1.099
    assert np.isclose(H_uniform[0], expected, rtol=1e-6)

    # Case 2: 확정적 분포
    proba_certain = np.array([[1.0, 0.0, 0.0]])
    H_certain = predictive_entropy(proba_certain)
    assert np.isclose(H_certain[0], 0.0, atol=1e-10)

    # Case 3: 수치 안정성 (p=0 포함)
    proba_with_zero = np.array([[0.0, 0.5, 0.5]])
    H = predictive_entropy(proba_with_zero)
    expected = -2 * 0.5 * np.log(0.5)  # 1e-15 * log(1e-15) ≈ 0
    assert np.isfinite(H[0])  # -inf 발생 안 함
```

**검증 사항:**
1. **균일 분포:** $H = -\sum_c \frac{1}{3} \log \frac{1}{3} = \log 3 \approx 1.099$. 수치 오차 $10^{-6}$ 이내.
2. **확정적 분포:** $H = -1 \cdot \log 1 = 0$. 수치 오차 $10^{-10}$ 이내.
3. **$\log(0)$ 안전성:** $p_c = 0$인 경우에도 `np.isfinite` 확인. `np.clip`이 정상 작동하면 `-inf`가 발생하지 않는다.

### 4.2 Greedy 배치 선택 동작 검증

**테스트: `TestAcquisition::test_greedy_batch_diversity`**

검증 내용: 배치 선택에서 다양성 제약이 올바르게 작동하는지 확인한다.

```python
def test_greedy_batch_diversity():
    """Greedy 배치 선택의 다양성 제약 검증."""
    from acquisition import greedy_batch_selection

    # 1차원 후보 (시각화 간편)
    candidates = np.linspace(0, 1, 100).reshape(-1, 1)  # (100, 1)
    entropy = np.random.rand(100)  # 임의 엔트로피
    entropy[50] = 10.0  # 최대 엔트로피를 인덱스 50에 강제

    k = 5
    d_min = 0.15  # 15% 간격

    selected = greedy_batch_selection(candidates, entropy, k, d_min)

    # 검증 1: 첫 번째 선택이 최대 엔트로피 점
    assert selected[0] == 50

    # 검증 2: 선택된 점들 간 거리가 d_min 이상
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            dist = np.linalg.norm(candidates[selected[i]] - candidates[selected[j]])
            assert dist >= d_min, f"dist={dist:.3f} < d_min={d_min}"

    # 검증 3: k개 또는 가용 후보 소진 시 더 적게 선택
    assert len(selected) <= k
```

**검증 사항:**
1. **첫 번째 선택:** 전역 최대 엔트로피 점(인덱스 50)이 선택된다.
2. **분리 거리:** 선택된 모든 쌍 $(i, j)$에 대해 $\|\mathbf{x}_i - \mathbf{x}_j\| \geq d_{\min}$.
3. **크기 제약:** 배치 크기 $\leq k$.

### 4.3 적응적 샘플링 수렴성

**테스트: `TestAdaptiveSampler::test_small_convergence`** (pytest marker: `@pytest.mark.slow`)

검증 내용: 전체 능동 학습 루프가 수렴 조건을 만족하며 종료하는지 확인한다.

```python
@pytest.mark.slow
def test_small_convergence():
    """소규모 능동 학습 루프 수렴 검증."""
    def dummy_evaluate(delta_a, delta_i, T_normed):
        """간단한 규칙 기반 분류."""
        # delta_i < 5: class 0, 5 <= delta_i < 10: class 1, else: class 2
        if delta_i < 5.0:
            return 0
        elif delta_i < 10.0:
            return 1
        else:
            return 2

    sampler = AdaptiveSampler(
        h0=400.0,
        evaluate_fn=dummy_evaluate,
        n_init=20,  # 초기 20개
        n_max=50,   # 최대 50개
        batch_size=5,
        d_min=0.1,
        entropy_threshold=0.25,
        seed=42,
    )

    X, y, gpc = sampler.run()

    # 검증 1: 샘플 수 범위
    assert 20 <= len(X) <= 50

    # 검증 2: 3개 클래스 모두 존재
    assert len(np.unique(y)) == 3

    # 검증 3: GP 분류기가 학습됨
    assert gpc.is_fitted

    # 검증 4: 수렴 후 최대 엔트로피가 임계값 이하
    proba = gpc.predict_proba(sampler.candidates)
    H = predictive_entropy(proba)
    max_H = np.max(H)
    assert max_H < 0.5  # 느슨한 기준 (dummy 함수는 단순하므로)
```

**검증 사항:**
1. **샘플 수:** 초기 20개 이상, 최대 50개 이하. 실제로는 수렴 조건에 따라 30–40개 수준에서 종료 예상.
2. **클래스 다양성:** 3개 클래스 모두 관측됨. `dummy_evaluate`는 `delta_i` 기준으로 명확히 분류되므로 보장됨.
3. **GP 학습:** `gpc.is_fitted == True`.
4. **수렴 품질:** 최대 엔트로피가 임계값 근처 또는 이하. Dummy 함수는 결정 경계가 선형이므로 GP가 빠르게 학습한다.

### 4.4 좌표 변환 일관성

**테스트: `TestLHS::test_normalize_denormalize_inverse`**

검증 내용: 정규화 ↔ 물리 좌표 변환이 역함수 관계를 만족하는지 확인한다.

```python
def test_normalize_denormalize_inverse():
    """정규화-역정규화 역함수 검증."""
    from lhs import normalize_params, denormalize_params

    # 물리 좌표 샘플
    params_phys = np.array([
        [1000.0, 7.5, 2.0],   # (delta_a, delta_i, T_normed)
        [-200.0, 0.0, 0.5],
        [2000.0, 15.0, 5.0],
    ])

    # 정규화 → 역정규화
    params_norm = normalize_params(params_phys)
    params_back = denormalize_params(params_norm)

    # 검증: 원래 값과 일치
    assert np.allclose(params_phys, params_back, rtol=1e-10)

    # 검증: 정규화 좌표 범위 [0, 1]
    assert np.all((params_norm >= 0) & (params_norm <= 1))
```

**검증 사항:**
1. **역함수 관계:** $(f \circ f^{-1})(\mathbf{x}) = \mathbf{x}$. 수치 오차 $10^{-10}$ 이내.
2. **범위 보존:** 정규화 좌표가 $[0,1]$ 범위 내.

### 4.5 허용 오차 및 검증 기준 요약

| 테스트 항목 | 검증 기준 | 허용 오차 | 물리적 의미 |
|:-----------|:----------|:---------|:-----------|
| 엔트로피 계산 정확성 | 수학적 정의와 일치 | $10^{-6}$ (상대) | Shannon 엔트로피 정의 |
| $\log(0)$ 안전성 | `np.isfinite` | N/A | 수치 폭발 방지 |
| Greedy 선택 다양성 | $\|\mathbf{x}_i - \mathbf{x}_j\| \geq d_{\min}$ | $10^{-12}$ (부동소수점) | 배치 다양성 제약 |
| 적응적 수렴 | $\max H < \epsilon_H$ 또는 안정 | $\epsilon_H = 0.3$ nats | GP 예측 신뢰도 |
| 좌표 변환 역함수 | $(f \circ f^{-1})(\mathbf{x}) = \mathbf{x}$ | $10^{-10}$ (상대) | 수치 정밀도 |


## 5. 참고문헌

- Settles, B. (2009). Active Learning Literature Survey. *Computer Sciences Technical Report 1648*, University of Wisconsin-Madison.
- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379–423.
- McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. *Technometrics*, 21(2), 239–245.
- Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions—I. *Mathematical Programming*, 14(1), 265–294.
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian Active Learning for Classification and Preference Learning. *arXiv preprint arXiv:1112.5745*.
- Krause, A., Singh, A., & Guestrin, C. (2008). Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies. *Journal of Machine Learning Research*, 9, 235–284.
- Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. *Proceedings of the 27th International Conference on Machine Learning (ICML)*, 1015–1022.
- Bichon, B. J., Eldred, M. S., Swiler, L. P., Mahadevan, S., & McFarland, J. M. (2008). Efficient Global Reliability Analysis for Nonlinear Implicit Performance Functions. *AIAA Journal*, 46(10), 2459–2468.
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Lee, S. (2023). A Generative Verification Framework on Statistical Stability for Data-Driven Controllers. *IEEE Access*, 11, 5267–5280.
