# A3. 호만 전이 해석해 및 검증 벤치마크

> 대응 코드: `src/orbit_transfer/astrodynamics/hohmann.py`

---

## 1. 목적

호만 전이(Hohmann transfer)는 두 동심 원궤도 사이의 이-임펄스(two-impulse) 최소 $\Delta v$ 궤도전이이다. 본 프로젝트에서 호만 전이 해석해는 두 가지 역할을 수행한다.

첫째, 연속 추력 최적화 결과의 **물리적 타당성 검증 기준**이다. 연속 추력 해의 총 $\Delta v$는 이상적 임펄스 해의 $\Delta v$보다 항상 크거나 같아야 하므로, 호만 전이 $\Delta v$는 하한(lower bound)을 제공한다.

둘째, 궤적 최적화 코드의 **단위 테스트 기준값**이다. LEO-GEO 호만 전이처럼 교과서에서 널리 알려진 참조값과 비교하여 구현 정확도를 검증한다.

본 보고서는 vis-viva 방정식으로부터 호만 전이 공식을 유도하고, 코드 구현과의 일대일 대응을 보이며, 수치 검증 결과를 제시한다.

---

## 2. 수학적 배경

### 2.1 Vis-viva 방정식

이체 문제(two-body problem)에서 우주체의 비역학 에너지(specific mechanical energy)는 궤도상 어디서나 보존된다. 위치 $r$, 속력 $v$에서

$$
\varepsilon = \frac{v^2}{2} - \frac{\mu}{r}
$$

이며, 이 값은 궤도 장반경 $a$로만 결정된다:

$$
\varepsilon = -\frac{\mu}{2a}
$$

두 식을 연립하면 **vis-viva 방정식**을 얻는다:

$$
v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)
$$

이 관계는 케플러 궤도상 임의 위치에서의 속력을 궤도 크기($a$)와 현재 위치($r$)만으로 결정한다. 호만 전이의 모든 속도 계산은 이 단일 방정식에 기초한다 [1, Ch.6].

### 2.2 원궤도 속도

원궤도에서는 $r = a$이므로 vis-viva 방정식이 단순화된다:

$$
v_{\text{circ}} = \sqrt{\frac{\mu}{a}}
$$

초기 원궤도(반지름 $r_1 = a_1$)와 최종 원궤도(반지름 $r_2 = a_2$)의 속도는 각각

$$
v_{c1} = \sqrt{\frac{\mu}{a_1}}, \qquad v_{c2} = \sqrt{\frac{\mu}{a_2}}
$$

이다.

### 2.3 호만 전이 궤도 설계

호만 전이 궤도는 근지점이 초기 궤도에, 원지점이 최종 궤도에 접하는 타원이다 [2]. 따라서 전이 타원의 장반경은

$$
a_t = \frac{r_1 + r_2}{2} = \frac{a_1 + a_2}{2}
$$

이다. 전이 궤도상의 속도를 vis-viva 방정식으로 구한다.

**근지점 속도** ($r = r_1 = a_1$):

$$
v_{t1} = \sqrt{\mu\left(\frac{2}{a_1} - \frac{1}{a_t}\right)}
$$

**원지점 속도** ($r = r_2 = a_2$):

$$
v_{t2} = \sqrt{\mu\left(\frac{2}{a_2} - \frac{1}{a_t}\right)}
$$

### 2.4 $\Delta v$ 공식

호만 전이는 두 번의 접선 임펄스 기동으로 구성된다.

**제1 기동 (근지점, 초기 궤도 → 전이 궤도):**

$$
\Delta v_1 = v_{t1} - v_{c1} = \sqrt{\mu\left(\frac{2}{a_1} - \frac{1}{a_t}\right)} - \sqrt{\frac{\mu}{a_1}}
$$

궤도를 올리는 경우($a_2 > a_1$) $v_{t1} > v_{c1}$이므로 $\Delta v_1 > 0$이다. 순방향 접선 가속이다.

**제2 기동 (원지점, 전이 궤도 → 최종 궤도):**

$$
\Delta v_2 = v_{c2} - v_{t2} = \sqrt{\frac{\mu}{a_2}} - \sqrt{\mu\left(\frac{2}{a_2} - \frac{1}{a_t}\right)}
$$

원지점에서 전이 궤도 속도 $v_{t2}$는 최종 원궤도 속도 $v_{c2}$보다 작으므로 $\Delta v_2 > 0$이다.

**총 $\Delta v$:**

$$
\Delta v_{\text{total}} = |\Delta v_1| + |\Delta v_2|
$$

구현에서는 절대값을 사용하여 궤도를 내리는 경우($a_2 < a_1$)도 동일한 함수로 처리한다. 이 경우 $\Delta v_1 < 0$ (감속), $\Delta v_2 < 0$ (감속)이나 기동 크기의 합은 동일하다.

### 2.5 비행시간

전이 궤도의 반주기가 곧 비행시간이다. 케플러 제3법칙에 의해

$$
T = 2\pi\sqrt{\frac{a_t^3}{\mu}}
$$

이므로 반주기(근지점 → 원지점)는

$$
\text{TOF} = \pi\sqrt{\frac{a_t^3}{\mu}}
$$

이다 [1, Ch.6].

### 2.6 접선 기동의 최적성과 Oberth 효과

호만 전이가 이-임펄스 전이 중 최소 $\Delta v$를 달성하는 이유는 다음 두 원리에 기인한다.

**접선 기동의 에너지 효율.** 임펄스 $\Delta \mathbf{v}$가 속도 벡터에 평행하게 가해질 때 궤도 에너지 변화가 최대이다. 에너지 변화율은

$$
\Delta \varepsilon = \mathbf{v} \cdot \Delta \mathbf{v} + \frac{|\Delta \mathbf{v}|^2}{2}
$$

이므로, 동일한 $|\Delta \mathbf{v}|$에 대해 $\mathbf{v} \parallel \Delta \mathbf{v}$일 때 $\Delta \varepsilon$가 최대화된다 [3].

**Oberth 효과.** 속력이 큰 위치에서 동일한 $|\Delta v|$를 가하면 운동에너지 변화($\Delta E_k = mv\Delta v + \frac{1}{2}m\Delta v^2$)가 더 크다. 호만 전이는 제1 기동을 근지점(속력 최대)에서 수행하여 이 효과를 활용한다. Edelbaum [4]은 동심 원궤도 전이에서 이-임펄스 호만 전이가 최적임을 증명하였으며, 궤도 반지름 비 $a_2/a_1$이 약 11.94를 초과하면 삼-임펄스 bi-elliptic 전이가 더 효율적임을 보였다.

### 2.7 연속 추력 $L_2$ 비용과 임펄스 $\Delta v$의 관계

본 프로젝트의 궤적 최적화에서 사용하는 비용함수는 제어 입력의 $L_2$ 노름 제곱 적분이다:

$$
J = \int_0^{t_f} \|\mathbf{u}(t)\|^2 \, dt
$$

여기서 $\mathbf{u}(t)$는 추력 가속도 벡터이다. 임펄스 기동은 Dirac delta 형태의 제어 입력으로 표현된다:

$$
\mathbf{u}(t) = \sum_{k} \Delta \mathbf{v}_k \, \delta(t - t_k)
$$

임펄스 해는 연속 추력 해의 극한 형태로 해석할 수 있다. 연속 추력 해의 해공간 $\mathcal{S}_{\text{cont}}$는 임펄스 해의 해공간 $\mathcal{S}_{\text{imp}}$를 포함한다:

$$
\mathcal{S}_{\text{imp}} \subset \mathcal{S}_{\text{cont}}
$$

이 포함관계는 연속 추력 최적해의 비용이 임펄스 최적해의 비용보다 항상 작거나 같음을 의미한다 [5, 6]. 따라서 호만 전이 $\Delta v$는 연속 추력 해의 $\Delta v$ 하한을 제공하며, 이를 위반하는 수치 해는 물리적으로 부당하다.

구체적으로, 추력 크기 $T_{\max}$와 비행시간 $t_f$가 충분히 클 때, 연속 추력 최적 프로파일은 기동 구간이 집중된 "bang-off-bang" 형태로 수렴하며, 이는 임펄스 해에 접근한다 [6]. 본 프로젝트에서는 이 수렴 거동을 데이터베이스 전체에서 체계적으로 확인하는 데 호만 전이 해석해를 활용한다.

---

## 3. 구현 매핑

`hohmann.py`는 두 개의 함수로 구성되며, 각 함수는 2.3--2.5절의 수식을 직접 구현한다. 사용 상수는 `constants.py`에 정의된 $\mu_\oplus = 398600.4418$ km$^3$/s$^2$, $R_\oplus = 6378.137$ km이다.

### 3.1 함수별 매핑

| 수학 표현 | Python 구현 | 파일 위치 | 비고 |
|-----------|------------|-----------|------|
| $a_t = (a_1 + a_2)/2$ | `a_t = (a1 + a2) / 2.0` | hohmann.py:28 | 전이 타원 장반경 |
| $v_{c1} = \sqrt{\mu/a_1}$ | `v1 = np.sqrt(mu / a1)` | hohmann.py:31 | 초기 원궤도 속도 |
| $v_{c2} = \sqrt{\mu/a_2}$ | `v2 = np.sqrt(mu / a2)` | hohmann.py:32 | 최종 원궤도 속도 |
| $v_{t1} = \sqrt{\mu(2/a_1 - 1/a_t)}$ | `v_t1 = np.sqrt(mu * (2.0/a1 - 1.0/a_t))` | hohmann.py:35 | 전이 근지점 속도 |
| $v_{t2} = \sqrt{\mu(2/a_2 - 1/a_t)}$ | `v_t2 = np.sqrt(mu * (2.0/a2 - 1.0/a_t))` | hohmann.py:36 | 전이 원지점 속도 |
| $\Delta v_1 = \|v_{t1} - v_{c1}\|$ | `dv1 = abs(v_t1 - v1)` | hohmann.py:39 | 절대값 처리 |
| $\Delta v_2 = \|v_{c2} - v_{t2}\|$ | `dv2 = abs(v2 - v_t2)` | hohmann.py:40 | 절대값 처리 |
| $\Delta v_{\text{total}} = \Delta v_1 + \Delta v_2$ | `dv_total = dv1 + dv2` | hohmann.py:41 | 총 기동량 |
| $\text{TOF} = \pi\sqrt{a_t^3/\mu}$ | `tof = np.pi * np.sqrt(a_t**3 / mu)` | hohmann.py:64 | 전이 반주기 |

### 3.2 함수 인터페이스

**`hohmann_dv(a1, a2, mu)`**

- 입력: 초기 원궤도 반지름 $a_1$ [km], 최종 원궤도 반지름 $a_2$ [km], 중력 파라미터 $\mu$ [km$^3$/s$^2$]
- 출력: `(dv1, dv2, dv_total)` [km/s]
- $a_1 = a_2$이면 $\Delta v_1 = \Delta v_2 = 0$을 반환한다.
- $a_2 < a_1$ (궤도 하강)에서도 절대값 처리로 양수 $\Delta v$를 반환한다.

**`hohmann_tof(a1, a2, mu)`**

- 입력: 동일
- 출력: 비행시간 $\text{TOF}$ [s]
- 전이 타원의 반주기로 계산한다.

### 3.3 설계 결정

1. **절대값 사용**: $\Delta v_1$, $\Delta v_2$ 계산에 `abs()`를 적용하여 궤도 상승/하강 모두 동일 함수로 처리한다. 이는 호만 전이의 대칭성($a_1 \leftrightarrow a_2$이면 $\Delta v_{\text{total}}$ 동일)을 보존한다.
2. **NumPy 사용**: `np.sqrt`를 사용하여 배열 입력에도 대응 가능하나, 현재는 스칼라 입력만 사용한다.
3. **$\mu$ 외부 전달**: 중력 파라미터를 함수 인자로 받아 지구 외 천체에도 적용 가능하다.

---

## 4. 수치 검증

### 4.1 LEO → GEO 벤치마크

가장 널리 알려진 호만 전이 벤치마크인 LEO-GEO 전이를 검증 기준으로 사용한다 [1].

**입력 조건:**

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| 초기 고도 $h_1$ | 200 km | LEO |
| 최종 고도 $h_2$ | 35,786 km | GEO |
| $a_1 = R_\oplus + h_1$ | 6,578.137 km | |
| $a_2 = R_\oplus + h_2$ | 42,164.137 km | |
| $\mu$ | 398,600.4418 km$^3$/s$^2$ | |

**계산 과정:**

전이 타원 장반경:

$$
a_t = \frac{6578.137 + 42164.137}{2} = 24371.137 \text{ km}
$$

원궤도 속도:

$$
v_{c1} = \sqrt{\frac{398600.4418}{6578.137}} = 7.784 \text{ km/s}
$$

$$
v_{c2} = \sqrt{\frac{398600.4418}{42164.137}} = 3.075 \text{ km/s}
$$

전이 궤도 속도:

$$
v_{t1} = \sqrt{398600.4418 \times \left(\frac{2}{6578.137} - \frac{1}{24371.137}\right)} = 10.239 \text{ km/s}
$$

$$
v_{t2} = \sqrt{398600.4418 \times \left(\frac{2}{42164.137} - \frac{1}{24371.137}\right)} = 1.597 \text{ km/s}
$$

$\Delta v$ 결과:

$$
\Delta v_1 = 10.239 - 7.784 = 2.455 \text{ km/s}
$$

$$
\Delta v_2 = 3.075 - 1.597 = 1.478 \text{ km/s}
$$

$$
\Delta v_{\text{total}} = 2.455 + 1.478 \approx 3.935 \text{ km/s}
$$

비행시간:

$$
\text{TOF} = \pi\sqrt{\frac{24371.137^3}{398600.4418}} \approx 18,925 \text{ s} \approx 5.26 \text{ hours}
$$

**검증 기준:** 교과서 참조값 $\Delta v_{\text{total}} \approx 3.935$ km/s [1, Ch.6]와 0.01 km/s 이내 일치.

### 4.2 동일 궤도 전이 (퇴화 케이스)

$a_1 = a_2$이면 $a_t = a_1$이므로 $v_{t1} = v_{c1}$, $v_{t2} = v_{c2}$이다. 따라서

$$
\Delta v_1 = \Delta v_2 = 0, \qquad \Delta v_{\text{total}} = 0
$$

이 케이스는 구현의 수치적 안정성(0 반환)을 확인한다.

### 4.3 테스트 구현

검증은 `tests/test_orbital_elements.py`의 `TestHohmann` 클래스에서 수행된다.

**`test_leo_to_geo`**: LEO(200 km) → GEO(35,786 km) 전이에서 $|\Delta v_{\text{total}} - 3.935| < 0.01$ km/s를 확인하고, $\Delta v_1 > 0$, $\Delta v_2 > 0$을 검증한다.

**`test_hohmann_tof`**: 동일 조건에서 비행시간이 양수이며 약 5.0--5.5시간 범위에 있음을 확인한다.

### 4.4 검증 결과 요약

| 검증 항목 | 기대값 | 허용 오차 | 결과 |
|-----------|--------|-----------|------|
| $\Delta v_{\text{total}}$ (LEO→GEO) | 3.935 km/s | $\pm$ 0.01 km/s | 통과 |
| $\Delta v_1, \Delta v_2 > 0$ | 양수 | -- | 통과 |
| TOF (LEO→GEO) | 약 5.26 hours | 5.0--5.5 hours | 통과 |
| $a_1 = a_2$ 시 $\Delta v = 0$ | 0 | 수치 영점 | 통과 |

---

## 5. 참고문헌

[1] Curtis, H. D., *Orbital Mechanics for Engineering Students*, 4th ed., Butterworth-Heinemann, 2020.

[2] Hohmann, W., *The Attainability of Heavenly Bodies*, NASA Technical Translation F-44, 1960. (원저 1925)

[3] Lawden, D. F., *Optimal Trajectories for Space Navigation*, Butterworths, London, 1963.

[4] Edelbaum, T. N., "How Many Impulses?," *Astronautics & Aeronautics*, Vol. 5, No. 11, 1967, pp. 64--69.

[5] Taheri, E., Junkins, J. L., "Generic Smoothing for Optimal Bang-Off-Bang Spacecraft Maneuvers," *Journal of Guidance, Control, and Dynamics*, Vol. 41, No. 11, 2018, pp. 2470--2475.

[6] Prussing, J. E., "Primer Vector Theory and Applications," in *Spacecraft Trajectory Optimization*, Conway, B. A. (ed.), Cambridge University Press, 2010, pp. 16--36.
