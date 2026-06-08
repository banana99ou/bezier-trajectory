# C2. 초기값 생성 전략

> 대응 코드: `src/orbit_transfer/optimizer/initial_guess.py`

---

## 1. 목적

연속 추력 궤적 최적화는 비선형 프로그래밍(NLP)으로 정식화되며, Interior-Point 또는 SQP 계열 솔버로 풀린다. 이러한 솔버는 Newton-Raphson 기반 반복법이므로, 초기값(initial guess)의 품질이 수렴 성공 여부와 계산 효율에 결정적 영향을 미친다.

**수렴 반경 문제.** Newton-Raphson 반복의 국소 수렴 이론에 따르면, 해 $\mathbf{x}^*$ 주변의 수렴 반경(convergence radius) $\delta > 0$가 존재하여, $\|\mathbf{x}_0 - \mathbf{x}^*\| < \delta$인 초기값 $\mathbf{x}_0$에서만 이차 수렴이 보장된다 [1]. 궤적 최적화에서 결정 변수의 차원은 수백~수천에 달하므로, 무작위 초기값은 수렴 반경 밖에 놓일 가능성이 높다.

**물리적 타당성.** 초기값이 물리법칙에 부합하는 궤적에 가까울수록, NLP의 동역학 구속조건 위반(constraint violation)이 작아져 솔버의 실행가능성 복원(feasibility restoration) 단계가 줄어든다 [2]. 케플러 궤도를 기반으로 한 초기값은 무추력 상태에서의 물리적 해이므로, 저추력 문제에서 좋은 출발점을 제공한다.

본 모듈은 두 가지 초기값 생성 전략을 구현한다.

1. **선형 보간법**(linear interpolation guess): 출발/도착 궤도를 각각 케플러 전파한 후 cosine blending으로 혼합
2. **케플러 전파법**(Keplerian guess): 출발 궤도에서 단순 케플러 전파

두 방법 모두 true anomaly 초기값에 대한 물리적 휴리스틱을 포함한다.

---

## 2. 수학적 배경

### 2.1 케플러 전파: Universal Variable Formulation

케플러 전파(Kepler propagation)는 이체 문제에서 초기 상태벡터 $(\mathbf{r}_0, \mathbf{v}_0)$로부터 시간 $\Delta t$ 후의 상태 $(\mathbf{r}, \mathbf{v})$를 계산하는 과정이다. Universal Variable Formulation은 타원, 쌍곡선, 포물선 궤도를 단일 알고리즘으로 처리한다 [3, Algorithm 3.4].

**Stumpff 함수.** Universal variable $\chi$와 $\psi = \alpha \chi^2$ ($\alpha = 1/a$, 장반경의 역수)에 대해 Stumpff 함수 $c_2(\psi)$, $c_3(\psi)$를 정의한다.

$$
c_2(\psi) = \begin{cases}
\dfrac{1 - \cos\sqrt{\psi}}{\psi} & \psi > 0 \text{ (타원)} \\[6pt]
\dfrac{\cosh\sqrt{-\psi} - 1}{-\psi} & \psi < 0 \text{ (쌍곡선)} \\[6pt]
\dfrac{1}{2} & \psi = 0 \text{ (포물선)}
\end{cases}
$$

$$
c_3(\psi) = \begin{cases}
\dfrac{\sqrt{\psi} - \sin\sqrt{\psi}}{\psi\sqrt{\psi}} & \psi > 0 \\[6pt]
\dfrac{\sinh\sqrt{-\psi} - \sqrt{-\psi}}{(-\psi)\sqrt{-\psi}} & \psi < 0 \\[6pt]
\dfrac{1}{6} & \psi = 0
\end{cases}
$$

이들은 $\cos$ 및 $\cosh$의 Taylor 전개에서 자연스럽게 나타나는 함수이며, $\psi \to 0$에서 $c_2 \to 1/2$, $c_3 \to 1/6$으로 연속이다.

**역수 장반경.** 초기 상태벡터로부터 역수 장반경 $\alpha$를 계산한다.

$$
\alpha = \frac{2}{r_0} - \frac{v_0^2}{\mu}
$$

$\alpha > 0$이면 타원, $\alpha < 0$이면 쌍곡선, $\alpha = 0$이면 포물선 궤도이다. 본 프로젝트의 원궤도($e = 0$)에서는 항상 $\alpha > 0$이다.

**Universal Kepler 방정식.** 전파 시간 $\Delta t$에 대응하는 universal variable $\chi$를 다음 방정식의 근으로 구한다.

$$
F(\chi) = \chi^3 c_3(\psi) + \frac{\mathbf{r}_0 \cdot \mathbf{v}_0}{\sqrt{\mu}} \chi^2 c_2(\psi) + r_0 \chi (1 - \psi\, c_3(\psi)) - \sqrt{\mu}\,\Delta t = 0
$$

여기서 $\psi = \alpha \chi^2$이고, $r_0 = \|\mathbf{r}_0\|$이다.

**Newton-Raphson 반복.** $F(\chi)$의 도함수는 다음과 같이 주어진다.

$$
F'(\chi) = \chi^2 c_2(\psi) + \frac{\mathbf{r}_0 \cdot \mathbf{v}_0}{\sqrt{\mu}} \chi (1 - \psi\, c_3(\psi)) + r_0 (1 - \psi\, c_2(\psi))
$$

이 값은 전파 후 거리 $r(\chi)/\sqrt{\mu}$와 같다. 반복은 $|\delta\chi| < \varepsilon$ ($\varepsilon = 10^{-12}$)일 때 종료한다. 타원 궤도의 초기 추정값은 $\chi_0 = \sqrt{\mu}\,\Delta t\,\alpha$로 설정한다.

**Lagrange 계수.** 수렴한 $\chi$로부터 Lagrange 계수를 계산한다.

$$
f = 1 - \frac{\chi^2}{r_0} c_2(\psi), \qquad
g = \Delta t - \frac{\chi^3}{\sqrt{\mu}} c_3(\psi)
$$

$$
\dot{f} = \frac{\sqrt{\mu}}{r\,r_0} \chi (\psi\, c_3(\psi) - 1), \qquad
\dot{g} = 1 - \frac{\chi^2}{r} c_2(\psi)
$$

여기서 $r = \|\mathbf{r}\|$이다. 최종 상태벡터는 다음과 같다.

$$
\mathbf{r} = f\,\mathbf{r}_0 + g\,\mathbf{v}_0, \qquad
\mathbf{v} = \dot{f}\,\mathbf{r}_0 + \dot{g}\,\mathbf{v}_0
$$

Lagrange 계수는 항등식 $f\dot{g} - \dot{f}g = 1$을 만족하며, 이는 이체 문제에서 상태 전이의 정준적(canonical) 성질에 기인한다 [3, Ch.3].

### 2.2 선형 보간법 (Cosine Blending)

출발 궤도와 도착 궤도 각각을 케플러 전파하여 두 개의 참조 궤적을 생성한 후, 이를 시간에 따라 부드럽게 혼합한다.

**출발 궤도 전파 (forward).** 출발 궤도요소 $(a_0, e_0, i_0, \Omega_0, \omega_0, \nu_0)$에서 초기 상태벡터 $(\mathbf{r}_0, \mathbf{v}_0)$를 계산하고, 각 시간점 $t_k$ ($k = 0, 1, \ldots, N-1$)에서 케플러 전파한다.

$$
(\mathbf{r}_k^{\text{dep}}, \mathbf{v}_k^{\text{dep}}) = \text{Kepler}(\mathbf{r}_0, \mathbf{v}_0, t_k)
$$

**도착 궤도 역전파 (backward).** 도착 궤도요소 $(a_f, e_f, i_f, \Omega_f, \omega_f, \nu_f)$에서 최종 상태벡터 $(\mathbf{r}_f, \mathbf{v}_f)$를 계산하고, 각 시간점에서 역방향으로 전파한다. 역전파는 $\Delta t = t_k - T < 0$을 전파 시간으로 사용하여 수행한다.

$$
(\mathbf{r}_k^{\text{arr}}, \mathbf{v}_k^{\text{arr}}) = \text{Kepler}(\mathbf{r}_f, \mathbf{v}_f, t_k - T)
$$

**Cosine blending.** 혼합 가중치 $\alpha(t)$를 다음과 같이 정의한다.

$$
\alpha(t) = \frac{1}{2}\left(1 - \cos\frac{\pi t}{T}\right)
$$

이 함수는 다음 성질을 가진다.

- $\alpha(0) = 0$: 초기 시점에서 출발 궤도만 반영
- $\alpha(T) = 1$: 종단 시점에서 도착 궤도만 반영
- $\alpha'(0) = \alpha'(T) = 0$: 양 끝에서 기울기 0, 즉 경계에서 상태 변화가 부드러움
- $\alpha(T/2) = 1/2$: 중간 시점에서 두 궤적의 동일 가중 혼합
- $C^{\infty}$ 연속

선형 혼합 $\alpha_{\text{lin}}(t) = t/T$ 대비 cosine blending의 장점은 양단에서 기울기가 0이므로, 경계조건 부근의 상태 불연속이 완화된다는 점이다. 혼합된 초기값은

$$
\mathbf{x}_k^{\text{guess}} = (1 - \alpha_k)\,\mathbf{x}_k^{\text{dep}} + \alpha_k\,\mathbf{x}_k^{\text{arr}}
$$

이다. 여기서 $\mathbf{x}_k = [\mathbf{r}_k^{\top}, \mathbf{v}_k^{\top}]^{\top}$는 6차원 상태벡터이다.

**초기 제어 입력.** 초기 추력 벡터는 $\mathbf{u}_k^{\text{guess}} = \mathbf{0}$으로 설정한다. 이는 케플러 궤적(무추력 궤적)에서 출발한다는 전제와 일치하며, 솔버가 추력 프로파일을 자유롭게 탐색하도록 허용한다. 비영(nonzero) 추력 초기값은 NLP의 해공간 탐색을 특정 방향으로 편향시킬 수 있다 [4].

### 2.3 케플러 전파법 (Keplerian Guess)

도착 궤도 정보를 사용하지 않고, 출발 궤도에서의 순수 케플러 전파만으로 초기값을 생성하는 단순한 방법이다. 출발 궤도요소로부터 초기 상태벡터를 구한 후, 모든 시간점에서 전방 전파한다.

$$
(\mathbf{r}_k^{\text{guess}}, \mathbf{v}_k^{\text{guess}}) = \text{Kepler}(\mathbf{r}_0, \mathbf{v}_0, t_k), \quad k = 0, 1, \ldots, N-1
$$

이 방법은 도착 궤도 경계조건을 전혀 반영하지 않으므로, 장반경 변화 $\Delta a$나 경사각 변화 $\Delta i$가 큰 문제에서는 선형 보간법보다 초기 구속조건 위반이 크다. 그러나 구현이 단순하고 항상 물리적으로 타당한 궤적을 생성한다는 장점이 있어, 선형 보간법의 대안 또는 첫 번째 시도로 사용된다.

### 2.4 True Anomaly 초기값 휴리스틱

원궤도($e = 0$)에서는 근점 인수(argument of periapsis) $\omega$가 정의되지 않으므로, true anomaly $\nu$를 argument of latitude로 취급한다. 출발 true anomaly $\nu_0$는 자유 변수이며, 그 초기값은 기동 유형에 따라 다음과 같이 설정한다.

**Coplanar 전이 ($\Delta i \approx 0$): $\nu_0 = 0$.** 경사각 변화가 없는 궤도전이에서는 장반경만 변화시키면 되므로, 추력의 주성분이 속도 방향(접선 방향)이다. 에너지 변화율은

$$
\dot{\varepsilon} = \mathbf{v} \cdot \mathbf{a}_{\text{thrust}}
$$

이므로, 속력이 큰 위치에서 접선 추력의 에너지 변환 효율이 높다 (Oberth 효과 [5]). 원궤도에서 속력은 균일하나, 전이 과정 중 생성되는 타원 궤도에서는 근지점($\nu = 0$ 부근) 속력이 최대이다. 따라서 $\nu_0 = 0$에서 출발하면 기동 초기의 추력 효율이 최적화된다.

**경사각 변경 전이 ($\Delta i \ne 0$): $\nu_0 = \pi/2$.** 경사각 변경은 궤도면 법선 방향의 속도 성분을 바꾸는 기동이다. 경사각 변경에 필요한 $\Delta v$는

$$
\Delta v_i = 2v \sin\frac{\Delta i}{2}
$$

이며, 이 기동은 line of nodes (승교점-강교점 연결선) 방향에서 가장 효율적이다 [3, Ch.6]. 원궤도에서 $\Omega = 0$이면 line of nodes는 $x$축 방향이고, 이에 수직인 궤도면 내 방향은 argument of latitude $u = \pi/2$이다. 보다 정확히는, line of nodes 위의 점($u = 0$ 또는 $u = \pi$)에서 면외 기동을 수행해야 하나, 연속 추력 문제에서는 기동이 분산되므로, $\nu_0 = \pi/2$를 설정하여 기동 구간의 중심이 노드 근처에 오도록 배치한다.

이 휴리스틱은 NLP의 true anomaly 자유 변수에 대한 초기값만 제공하며, 최적화 과정에서 솔버가 자유롭게 조정한다.

### 2.5 Warm Start: Pass 1에서 Pass 2로의 보간

Two-Pass 최적화에서 Pass 1 (Hermite-Simpson collocation, 균일 격자)의 해를 Pass 2 (LGL pseudospectral, 비균일 격자)의 초기값으로 사용한다. 균일 시간점에서 구한 상태/제어 변수를 LGL 노드 위치에서 cubic spline으로 보간한다. 이 과정의 상세는 B3 보고서에서 다루며, 여기서는 전략적 의의만 기술한다.

Warm start는 Pass 1에서 이미 동역학 구속조건을 근사적으로 만족하는 해를 제공하므로, Pass 2의 초기 구속조건 위반이 크게 줄어든다. 이는 IPOPT의 실행가능성 복원 단계를 단축하고, Newton 반복 횟수를 감소시킨다 [2].

---

## 3. 구현 매핑

### 3.1 파일 구조

| 파일 | 역할 |
|------|------|
| `optimizer/initial_guess.py` | 초기값 생성 함수 2종 |
| `astrodynamics/kepler.py` | Universal Variable 케플러 전파 |
| `astrodynamics/orbital_elements.py` | 궤도요소 ↔ 상태벡터 변환 |
| `constants.py` | $\mu_\oplus = 398600.4418$ km$^3$/s$^2$ |
| `types.py` | `TransferConfig` 데이터 클래스 |

### 3.2 `linear_interpolation_guess` (11--79행)

| 수학 표현 | Python 구현 | 행 번호 | 비고 |
|-----------|------------|---------|------|
| $\nu_0 = 0,\; \nu_f = \pi$ | `nu0_guess = 0.0; nuf_guess = np.pi` | 39--40 | coplanar 기준 |
| $(a_0, 0, i_0, 0, 0, \nu_0) \to (\mathbf{r}_0, \mathbf{v}_0)$ | `oe_to_rv(oe0, MU_EARTH)` | 43--44 | 출발 궤도 |
| $(a_f, 0, i_f, 0, 0, \nu_f) \to (\mathbf{r}_f, \mathbf{v}_f)$ | `oe_to_rv(oef, MU_EARTH)` | 47--48 | 도착 궤도 |
| 전방 전파: $\text{Kepler}(\mathbf{r}_0, \mathbf{v}_0, t_k)$ | `kepler_propagate(r0, v0, dt_fwd, MU_EARTH)` | 59--63 | $k = 1, \ldots, N-1$ |
| 역전파: $\text{Kepler}(\mathbf{r}_f, \mathbf{v}_f, t_k - T)$ | `kepler_propagate(rf, vf, dt_bwd, MU_EARTH)` | 65--69 | $k = N-2, \ldots, 0$ |
| $\alpha_k = \frac{1}{2}(1 - \cos(\pi t_k/T))$ | `alpha = 0.5 * (1.0 - np.cos(np.pi * t[k] / T))` | 74 | cosine blending |
| $\mathbf{x}_k = (1-\alpha_k)\mathbf{x}_k^{\text{dep}} + \alpha_k \mathbf{x}_k^{\text{arr}}$ | `x_guess[:, k] = (1.0 - alpha) * x_depart[:, k] + alpha * x_arrive[:, k]` | 75 | 상태 혼합 |
| $\mathbf{u}_k = \mathbf{0}$ | `u_guess = np.zeros((3, N_points))` | 77 | 무추력 초기값 |

**반환값**: `(t, x_guess, u_guess, nu0_guess, nuf_guess)`. 시간 배열 `t`는 `np.linspace(0, T, N_points)`로 균일 격자이다.

### 3.3 `keplerian_guess` (82--125행)

| 수학 표현 | Python 구현 | 행 번호 | 비고 |
|-----------|------------|---------|------|
| 경사각 변경 판별 | `is_plane_change = abs(config.delta_i) > 1e-10` | 104 | 수치 허용값 |
| $\nu_0 = \pi/2$ (plane change) 또는 $0$ (coplanar) | `nu0_guess = np.pi / 2 if is_plane_change else 0.0` | 105 | 2.4절 휴리스틱 |
| $(a_0, 0, i_0, 0, 0, \nu_0) \to (\mathbf{r}_0, \mathbf{v}_0)$ | `oe_to_rv(oe0, MU_EARTH)` | 109--110 | 출발 궤도 |
| 전방 전파: $\text{Kepler}(\mathbf{r}_0, \mathbf{v}_0, t_k)$ | `kepler_propagate(r0, v0, dt, MU_EARTH)` | 117--121 | $k = 1, \ldots, N-1$ |
| $\mathbf{u}_k = \mathbf{0}$ | `u_guess = np.zeros((3, N_points))` | 123 | 무추력 초기값 |

이 함수는 도착 궤도 정보를 사용하지 않으므로 $\nu_f$는 $\pi$로 고정 반환한다 (106행).

### 3.4 `kepler_propagate` (kepler.py, 31--144행)

| 수학 표현 | Python 구현 | 행 번호 |
|-----------|------------|---------|
| $\alpha = 2/r_0 - v_0^2/\mu$ | `alpha = 2.0 / r0_mag - v0_mag**2 / mu` | 63 |
| $c_2(\psi)$ | `_stumpff_c2(psi)` | 9--16 |
| $c_3(\psi)$ | `_stumpff_c3(psi)` | 19--28 |
| $F(\chi) = 0$ | `fn = chi3*c3 + rdotv*chi2*c2 + r0_mag*chi*(1-psi*c3) - sqrt_mu*dt` | 103--105 |
| $F'(\chi)$ | `r_chi = chi2*c2 + rdotv*chi*(1-psi*c3) + r0_mag*(1-psi*c2)` | 108--110 |
| $\delta\chi = -F/F'$ | `delta_chi = -fn / r_chi` | 115 |
| $f = 1 - \chi^2 c_2 / r_0$ | `f = 1.0 - chi2 / r0_mag * c2` | 130 |
| $g = \Delta t - \chi^3 c_3 / \sqrt{\mu}$ | `g = dt - chi3 / sqrt_mu * c3` | 131 |
| $\dot{f} = \sqrt{\mu}\,\chi(\psi c_3 - 1)/(r\,r_0)$ | `fdot = sqrt_mu / (r_mag * r0_mag) * chi * (psi * c3 - 1.0)` | 138 |
| $\dot{g} = 1 - \chi^2 c_2 / r$ | `gdot = 1.0 - chi2 / r_mag * c2` | 139 |
| $\mathbf{r} = f\mathbf{r}_0 + g\mathbf{v}_0$ | `r = f * r0 + g * v0` | 134 |
| $\mathbf{v} = \dot{f}\mathbf{r}_0 + \dot{g}\mathbf{v}_0$ | `v = fdot * r0 + gdot * v0` | 142 |

Newton-Raphson 반복은 최대 100회, 허용 오차 $\varepsilon = 10^{-12}$로 설정되어 있다 (87--88행).

### 3.5 설계 결정

1. **Cosine blending 선택**: 단순 선형 보간($\alpha = t/T$) 대비 양 끝에서 기울기가 0이므로, 경계조건 부근의 상태 변화가 부드럽다. 이는 NLP 솔버의 초기 Jacobian 평가에서 수치적 안정성을 개선한다.

2. **역전파 사용**: 도착 궤도에서 음의 시간($t_k - T < 0$)으로 케플러 전파하여 역방향 궤적을 생성한다. Universal Variable Formulation은 $\Delta t < 0$을 자연스럽게 처리하므로 별도의 역전파 알고리즘이 불필요하다.

3. **True anomaly 자유 변수**: $\nu_0$, $\nu_f$는 NLP의 자유 변수이다. 초기값 함수는 이들의 초기 추정값만 반환하며, 최종 최적값은 솔버가 결정한다.

4. **초기 추력 0 설정**: 무추력 케플러 궤적에서 출발하므로 $\mathbf{u}_k = \mathbf{0}$이 물리적으로 일관된다. 또한 비용함수 $J = \int \|\mathbf{u}\|^2 dt$의 관점에서 초기 비용이 0이 되어, 솔버가 비용을 증가시키는 방향으로만 추력을 추가하게 된다.

---

## 4. 수치 검증

### 4.1 경계 궤도 반경 일치

선형 보간법의 초기값이 경계조건을 올바르게 반영하는지 검증한다.

**출발 경계 ($k = 0$).** $\alpha(0) = 0$이므로

$$
\mathbf{x}_0^{\text{guess}} = \mathbf{x}_0^{\text{dep}}
$$

이며, 이는 출발 궤도요소에서 변환한 상태벡터와 정확히 일치한다. 따라서

$$
\|\mathbf{r}_0^{\text{guess}}\| = a_0 \cdot \frac{1 - e_0^2}{1 + e_0 \cos\nu_0}
$$

이고, 원궤도($e_0 = 0$)에서 $\|\mathbf{r}_0\| = a_0 = R_\oplus + h_0$이다.

**도착 경계 ($k = N-1$).** $\alpha(T) = 1$이므로

$$
\mathbf{x}_{N-1}^{\text{guess}} = \mathbf{x}_{N-1}^{\text{arr}}
$$

이며, $\|\mathbf{r}_{N-1}\| = a_f = a_0 + \Delta a$이다.

**검증 기준**: $\left| \|\mathbf{r}_0\| - a_0 \right| < 10^{-10}$ km, $\left| \|\mathbf{r}_{N-1}\| - a_f \right| < 10^{-10}$ km.

### 4.2 True Anomaly 물리적 타당성

| 기동 유형 | $\nu_0$ 초기값 | 물리적 근거 | 검증 기준 |
|-----------|---------------|-------------|-----------|
| Coplanar ($\Delta i = 0$) | $0$ | Oberth 효과: 근지점에서 접선 기동 효율 최대 | $\nu_0 = 0.0$ 확인 |
| Plane change ($\Delta i \ne 0$) | $\pi/2$ | Line of nodes에서 면외 기동 효율 최대 | $\nu_0 = \pi/2$ 확인 |
| 선형 보간법 (항상) | $\nu_f = \pi$ | 반궤도 전이: 대칭적 기동 분포 | $\nu_f = \pi$ 확인 |

### 4.3 Cosine Blending 함수 성질 검증

$\alpha(t) = \frac{1}{2}(1 - \cos(\pi t/T))$에 대해 다음을 수치적으로 확인한다.

| 성질 | 기대값 | 허용 오차 |
|------|--------|-----------|
| $\alpha(0)$ | $0$ | $< 10^{-15}$ |
| $\alpha(T)$ | $1$ | $< 10^{-15}$ |
| $\alpha(T/2)$ | $0.5$ | $< 10^{-15}$ |
| $\alpha'(0) = \frac{\pi}{2T}\sin(0)$ | $0$ | $< 10^{-15}$ |
| $\alpha'(T) = \frac{\pi}{2T}\sin(\pi)$ | $0$ | $< 10^{-15}$ |
| $\alpha(t) \in [0, 1]$ for all $t \in [0, T]$ | 단조 증가 | 모든 격자점 확인 |

### 4.4 케플러 전파 보존량 검증

Universal Variable Formulation에 의한 케플러 전파가 이체 문제의 보존량을 유지하는지 검증한다.

**에너지 보존.** 비역학 에너지 $\varepsilon = v^2/2 - \mu/r$은 궤도상 어디서나 일정하다.

$$
\left| \varepsilon(t_k) - \varepsilon(0) \right| < 10^{-10} \text{ km}^2/\text{s}^2, \quad \forall k
$$

**각운동량 보존.** 비각운동량 $\mathbf{h} = \mathbf{r} \times \mathbf{v}$의 크기는 일정하다.

$$
\left| \|\mathbf{h}(t_k)\| - \|\mathbf{h}(0)\| \right| < 10^{-10} \text{ km}^2/\text{s}, \quad \forall k
$$

**Lagrange 계수 항등식.** $f\dot{g} - \dot{f}g = 1$이 수치적으로 만족되는지 확인한다.

### 4.5 검증 결과 요약

| 검증 항목 | 기대값 | 허용 오차 | 결과 |
|-----------|--------|-----------|------|
| 출발점 반경 $\|\mathbf{r}_0\| = a_0$ | $R_\oplus + h_0$ | $10^{-10}$ km | 통과 |
| 도착점 반경 $\|\mathbf{r}_{N-1}\| = a_f$ | $a_0 + \Delta a$ | $10^{-10}$ km | 통과 |
| $\nu_0$ 휴리스틱 (coplanar) | $0.0$ rad | 정확 일치 | 통과 |
| $\nu_0$ 휴리스틱 (plane change) | $\pi/2$ rad | 정확 일치 | 통과 |
| Blending $\alpha(0) = 0$, $\alpha(T) = 1$ | 경계값 | $10^{-15}$ | 통과 |
| 케플러 전파 에너지 보존 | $\Delta\varepsilon = 0$ | $10^{-10}$ km$^2$/s$^2$ | 통과 |
| 케플러 전파 각운동량 보존 | $\Delta h = 0$ | $10^{-10}$ km$^2$/s | 통과 |

---

## 5. 참고문헌

[1] Nocedal, J., Wright, S. J., *Numerical Optimization*, 2nd ed., Springer, New York, 2006.

[2] Wachter, A., Biegler, L. T., "On the Implementation of an Interior-Point Filter Line-Search Algorithm for Large-Scale Nonlinear Programming," *Mathematical Programming*, Vol. 106, No. 1, 2006, pp. 25--57.

[3] Curtis, H. D., *Orbital Mechanics for Engineering Students*, 4th ed., Butterworth-Heinemann, 2020.

[4] Betts, J. T., *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed., SIAM, Philadelphia, 2010.

[5] Oberth, H., *Ways to Spaceflight*, NASA Technical Translation TT F-622, 1972. (원저 1929)

[6] Prussing, J. E., Conway, B. A., *Orbital Mechanics*, Oxford University Press, 2nd ed., 2012.
