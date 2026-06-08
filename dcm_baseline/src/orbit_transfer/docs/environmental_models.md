# A2. 환경 모델: J2 섭동 및 대기 항력

> 대응 코드: `src/orbit_transfer/dynamics/` (4파일)

---

## 1. 목적

저궤도(LEO) 연속 추력 궤도전이 최적화에서 사용하는 **환경 모델**을 정의한다. 본 프로젝트의 동역학 모델은 다음 세 가지 가속도 성분의 합으로 구성된다.

$$
\ddot{\mathbf{r}} = \mathbf{a}_{\text{grav}} + \mathbf{a}_{J2} + \mathbf{a}_{\text{drag}} + \mathbf{u}
$$

여기서 $\mathbf{a}_{\text{grav}}$는 이체 중력 가속도, $\mathbf{a}_{J2}$는 지구 비구형(oblateness)에 의한 J2 섭동 가속도, $\mathbf{a}_{\text{drag}}$는 대기 항력 가속도, $\mathbf{u}$는 추력 가속도이다.

본 보고서는 각 환경 모델의 수학적 배경을 유도하고, 이를 코드에 매핑한 방법과 수치 검증 결과를 기술한다. LEO 고도(200--1000 km)에서 각 섭동원의 상대적 크기를 정량적으로 비교하여, 모델 선택의 타당성을 확인한다.


---

## 2. 수학적 배경

### 2.1 이체 중력 가속도

Newton의 만유인력 법칙에 따르면, 질량 $m$인 우주선이 질량 $M$인 지구로부터 거리 $r = \|\mathbf{r}\|$에서 받는 중력은 다음과 같다.

$$
\mathbf{F}_{\text{grav}} = -\frac{GMm}{r^3}\,\mathbf{r}
$$

여기서 $\mathbf{r}$은 지구 중심에서 우주선까지의 위치 벡터(ECI)이다. 지구 중력 상수 $\mu = GM$을 도입하면, 단위 질량당 중력 가속도(specific force)는 다음과 같다.

$$
\mathbf{a}_{\text{grav}} = -\frac{\mu}{r^3}\,\mathbf{r}
$$

이는 중심력(central force)으로, 케플러 운동의 기본이 된다. 추력이나 섭동이 없을 때 이 가속도 하에서 비에너지(specific orbital energy)는 보존된다.

$$
\mathcal{E} = \frac{1}{2}\|\dot{\mathbf{r}}\|^2 - \frac{\mu}{r} = \text{const.}
$$

본 프로젝트에서 사용하는 값은 $\mu = 398600.4418\;\text{km}^3/\text{s}^2$이다 (WGS-84).


### 2.2 지구 중력 포텐셜과 구면조화함수 전개

실제 지구의 중력장은 질량 분포의 비대칭성으로 인해 점질량 모델에서 벗어난다. 지구 중력 포텐셜은 구면조화함수(spherical harmonics)로 전개할 수 있다.

$$
U = \frac{\mu}{r}\left[1 - \sum_{n=2}^{\infty} J_n \left(\frac{R_E}{r}\right)^n P_n(\sin\phi)\right]
$$

여기서:
- $R_E = 6378.137\;\text{km}$: 지구 적도 반지름
- $J_n$: $n$차 대영(zonal) 조화 계수
- $P_n$: $n$차 르장드르 다항식
- $\phi$: 지심위도(geocentric latitude), $\sin\phi = z/r$

대영 조화 계수 중 $J_2 = 1.08263 \times 10^{-3}$이 가장 크며, $J_3 \sim 10^{-6}$, $J_4 \sim 10^{-6}$ 순서이다. LEO 궤도전이 문제에서 $J_2$는 궤도 요소에 유의미한 영향(secular drift)을 미치지만, $J_3$ 이상의 고차항은 본 프로젝트의 최적화 정밀도에 비해 무시할 수 있다. 따라서 $J_2$항만 고려한다.


### 2.3 J2 섭동 가속도

$J_2$항에 의한 중력 포텐셜 보정은 다음과 같다.

$$
U_{J2} = -\frac{\mu J_2 R_E^2}{2r^3}\left(3\sin^2\phi - 1\right)
$$

$\sin\phi = z/r$을 대입하면:

$$
U_{J2} = -\frac{\mu J_2 R_E^2}{2r^3}\left(\frac{3z^2}{r^2} - 1\right) = -\frac{\mu J_2 R_E^2}{2}\left(\frac{3z^2}{r^5} - \frac{1}{r^3}\right)
$$

J2 섭동 가속도는 이 포텐셜의 구배(gradient)로 구한다.

$$
\mathbf{a}_{J2} = \nabla U_{J2} = \left(\frac{\partial U_{J2}}{\partial x},\;\frac{\partial U_{J2}}{\partial y},\;\frac{\partial U_{J2}}{\partial z}\right)
$$

**편미분 유도.** 계산의 편의를 위해 $r^2 = x^2 + y^2 + z^2$이므로 $\partial r/\partial x = x/r$ 등의 관계를 이용한다.

$U_{J2}$를 $r$과 $z$의 함수로 정리하면:

$$
U_{J2} = -\frac{\mu J_2 R_E^2}{2}\left(3z^2 r^{-5} - r^{-3}\right)
$$

$x$-성분 편미분:

$$
\frac{\partial U_{J2}}{\partial x} = -\frac{\mu J_2 R_E^2}{2}\left[3z^2 \cdot (-5) r^{-6}\frac{x}{r} - (-3)r^{-4}\frac{x}{r}\right]
$$

$$
= -\frac{\mu J_2 R_E^2}{2}\left[-\frac{15z^2 x}{r^7} + \frac{3x}{r^5}\right]
$$

$$
= -\frac{\mu J_2 R_E^2}{2}\cdot\frac{3x}{r^5}\left[1 - \frac{5z^2}{r^2}\right]
$$

$$
= -\frac{3}{2}\frac{\mu J_2 R_E^2}{r^5}\cdot x\left(1 - 5\frac{z^2}{r^2}\right)
$$

$y$-성분도 동일한 구조이다:

$$
\frac{\partial U_{J2}}{\partial y} = -\frac{3}{2}\frac{\mu J_2 R_E^2}{r^5}\cdot y\left(1 - 5\frac{z^2}{r^2}\right)
$$

$z$-성분은 $U_{J2}$가 $z$에 직접 의존하므로 추가 항이 발생한다:

$$
\frac{\partial U_{J2}}{\partial z} = -\frac{\mu J_2 R_E^2}{2}\left[-\frac{15z^2}{r^7}\cdot z + \frac{6z}{r^5} + \frac{3z}{r^5}\right]
$$

$$
= -\frac{\mu J_2 R_E^2}{2}\cdot\frac{3z}{r^5}\left[3 - 5\frac{z^2}{r^2}\right]
$$

$$
= -\frac{3}{2}\frac{\mu J_2 R_E^2}{r^5}\cdot z\left(3 - 5\frac{z^2}{r^2}\right)
$$

**최종 J2 섭동 가속도 (ECI 성분):**

공통 계수 $C = -\dfrac{3}{2}\dfrac{\mu J_2 R_E^2}{r^5}$를 정의하면:

$$
\boxed{
\begin{aligned}
a_x &= C\,x\left(1 - 5\frac{z^2}{r^2}\right) \\[6pt]
a_y &= C\,y\left(1 - 5\frac{z^2}{r^2}\right) \\[6pt]
a_z &= C\,z\left(3 - 5\frac{z^2}{r^2}\right)
\end{aligned}
}
$$

이 공식이 `j2_perturbation.py`에 직접 구현되어 있다.


### 2.4 Secular Perturbation: RAAN Drift Rate

J2 섭동은 궤도 요소에 장기(secular) 변화를 유발한다. 특히 승교점경도(RAAN, $\Omega$)의 평균 변화율은 궤도역학에서 중요한 역할을 한다. 가우스 행성 운동 방정식(Gauss's variational equations)을 J2 섭동에 대해 한 주기 평균하면 다음을 얻는다 (Vallado, 2013, Sec. 9.4):

$$
\dot{\Omega} = -\frac{3}{2}\,n\,J_2\left(\frac{R_E}{p}\right)^2 \cos i
$$

여기서:
- $n = \sqrt{\mu/a^3}$: 평균 운동(mean motion)
- $p = a(1 - e^2)$: 반직현(semi-latus rectum)
- $i$: 궤도 경사각(inclination)
- $a$: 장반경(semi-major axis), $e$: 이심률

원궤도($e = 0$)에서는 $p = a$이므로:

$$
\dot{\Omega} = -\frac{3}{2}\,n\,J_2\left(\frac{R_E}{a}\right)^2 \cos i
$$

**유도 과정.** 가우스 행성 방정식에서 $\Omega$의 변화율은 다음과 같다:

$$
\frac{d\Omega}{dt} = \frac{r\sin(\omega + f)}{n a^2 \sqrt{1-e^2}\sin i}\,a_W
$$

여기서 $a_W$는 궤도면 법선 방향 섭동 가속도, $\omega$는 근지점 편각, $f$는 진근점 이각이다. J2 섭동의 법선 방향 성분을 대입하고 한 주기에 대해 평균하면 위의 secular rate를 얻는다.

**수치 예시.** $h = 400\;\text{km}$, $i = 45°$ 원궤도에서:
- $a = 6778.137\;\text{km}$
- $n = \sqrt{398600.4418 / 6778.137^3} = 1.1311 \times 10^{-3}\;\text{rad/s}$
- $\dot{\Omega} = -1.5 \times 1.1311 \times 10^{-3} \times 1.08263 \times 10^{-3} \times (6378.137/6778.137)^2 \times \cos(45°)$
- $\dot{\Omega} \approx -1.021 \times 10^{-6}\;\text{rad/s} \approx -5.06°/\text{day}$

이 해석식은 수치 검증(Sec. 4)에서 수치 적분 결과와의 비교 기준으로 사용된다.


### 2.5 지수 대기 모델 (Exponential Atmosphere)

LEO에서 대기 항력은 저고도(특히 500 km 이하)에서 무시할 수 없는 섭동원이다. 대기 밀도의 고도 의존성을 모델링하기 위해 지수 대기 모델(exponential atmosphere model)을 사용한다.

$$
\rho(h) = \rho_0 \exp\!\left(-\frac{h - h_{\text{ref}}}{H}\right)
$$

여기서:
- $h = \|\mathbf{r}\| - R_E$: 지표면 위 고도 [km]
- $\rho_0$: 기준 고도에서의 대기 밀도 [kg/m$^3$]
- $h_{\text{ref}}$: 기준 고도 [km]
- $H$: 대기 스케일 높이(scale height) [km]

실제 대기는 고도에 따라 온도와 조성이 변하므로 하나의 $(\rho_0, H)$ 쌍으로 전 고도를 표현할 수 없다. 본 프로젝트에서는 Vallado (2013, Table 8-4)의 층상 모델(layered model)을 채택하여, 고도를 여러 구간으로 나누고 각 구간마다 다른 계수를 사용한다.

**대기 모델 계수 테이블 (일부):**

| 고도 구간 [km] | $\rho_0$ [kg/m$^3$] | $h_{\text{ref}}$ [km] | $H$ [km] |
|:-:|:-:|:-:|:-:|
| 150 -- 180 | $2.070 \times 10^{-9}$ | 150.0 | 22.523 |
| 180 -- 200 | $3.845 \times 10^{-10}$ | 180.0 | 29.740 |
| 200 -- 250 | $1.585 \times 10^{-10}$ | 200.0 | 37.105 |
| 250 -- 300 | $3.233 \times 10^{-11}$ | 250.0 | 45.546 |
| 300 -- 350 | $8.152 \times 10^{-12}$ | 300.0 | 53.628 |
| 350 -- 400 | $2.477 \times 10^{-12}$ | 350.0 | 53.298 |
| 400 -- 450 | $8.484 \times 10^{-13}$ | 400.0 | 58.515 |
| 450 -- 500 | $3.027 \times 10^{-13}$ | 450.0 | 60.828 |
| 500 -- 600 | $1.514 \times 10^{-13}$ | 500.0 | 63.822 |
| 600 -- 700 | $3.614 \times 10^{-14}$ | 600.0 | 71.835 |
| 700 -- 800 | $1.170 \times 10^{-14}$ | 700.0 | 88.667 |
| 800 -- 900 | $3.614 \times 10^{-15}$ | 800.0 | 124.64 |
| 900 -- 1000 | $1.454 \times 10^{-15}$ | 900.0 | 181.05 |
| 1000 -- 1100 | $6.967 \times 10^{-16}$ | 1000.0 | 268.00 |

모델의 유효 범위는 150--1100 km이며, 14개 구간으로 구성된다. 고도가 증가할수록 $\rho_0$는 급격히 감소하고 $H$는 증가하는데, 이는 고고도에서 대기 밀도의 고도 변화율이 완만해짐을 의미한다.


### 2.6 항력 가속도

대기 항력에 의한 단위 질량당 가속도는 다음과 같다.

$$
\mathbf{a}_{\text{drag}} = -\frac{1}{2}\,C_D\,\frac{A}{m}\,\rho\,\|\mathbf{v}_{\text{rel}}\|\,\mathbf{v}_{\text{rel}}
$$

여기서:
- $C_D$: 항력 계수(drag coefficient), 무차원. 본 프로젝트에서는 $C_D = 2.2$ 사용
- $A/m$: 단면적 대 질량 비 [m$^2$/kg]. 본 프로젝트에서는 $A/m = 0.01\;\text{m}^2/\text{kg}$ 사용
- $\rho$: 대기 밀도 [kg/m$^3$] (Sec. 2.5에서 계산)
- $\mathbf{v}_{\text{rel}}$: 대기에 대한 상대 속도

**대기 공전 보정.** 대기는 지구와 함께 자전하므로, 우주선의 관성 속도 $\mathbf{v}$에서 대기의 공전 속도를 빼야 한다.

$$
\mathbf{v}_{\text{rel}} = \mathbf{v} - \boldsymbol{\omega}_E \times \mathbf{r}
$$

여기서 $\boldsymbol{\omega}_E = (0, 0, \omega_E)^T$이고 $\omega_E = 7.2921159 \times 10^{-5}\;\text{rad/s}$는 지구 자전 각속도이다.

**단위 변환.** 코드에서는 위치를 km, 속도를 km/s 단위로 사용한다. 항력 공식의 각 인자 단위를 추적하면:

$$
\frac{1}{2}\,C_D\,\frac{A}{m}\,\rho\,\|\mathbf{v}_{\text{rel}}\|\,\mathbf{v}_{\text{rel}} \;\sim\; [\text{무차원}]\cdot[\text{m}^2/\text{kg}]\cdot[\text{kg}/\text{m}^3]\cdot[\text{km}/\text{s}]\cdot[\text{km}/\text{s}]
$$

$$
= \frac{1}{\text{m}}\cdot\frac{\text{km}^2}{\text{s}^2} = \frac{\text{km}^2}{\text{m}\cdot\text{s}^2} = \frac{\text{km}}{\text{s}^2}\cdot\frac{\text{km}}{\text{m}} = \frac{\text{km}}{\text{s}^2}\cdot 10^3
$$

따라서 km/s$^2$ 단위로 변환하기 위해 $10^{-3}$을 곱한다:

$$
\mathbf{a}_{\text{drag}}\;[\text{km/s}^2] = -\frac{1}{2}\,C_D\,\frac{A}{m}\,\rho\,\|\mathbf{v}_{\text{rel}}\|\,\mathbf{v}_{\text{rel}} \times 10^{-3}
$$


### 2.7 운동방정식 통합

6차원 상태 벡터 $\mathbf{x} = (\mathbf{r},\;\mathbf{v})^T$에 대한 1차 상미분방정식(ODE) 형태의 운동방정식은 다음과 같다.

$$
\dot{\mathbf{x}} = \begin{pmatrix} \dot{\mathbf{r}} \\ \dot{\mathbf{v}} \end{pmatrix} = \begin{pmatrix} \mathbf{v} \\ \mathbf{a}_{\text{grav}} + \mathbf{a}_{J2} + \mathbf{a}_{\text{drag}} + \mathbf{u} \end{pmatrix}
$$

구성 요소를 정리하면:

| 가속도 성분 | 수식 | 조건 |
|:-:|:-:|:-:|
| $\mathbf{a}_{\text{grav}}$ | $-\mu/r^3\,\mathbf{r}$ | 항상 포함 |
| $\mathbf{a}_{J2}$ | Sec. 2.3의 공식 | `include_j2=True`일 때 |
| $\mathbf{a}_{\text{drag}}$ | Sec. 2.6의 공식 | `include_drag=True`일 때 (NumPy 전용) |
| $\mathbf{u}$ | 추력 가속도 | 최적화 제어 입력 |

CasADi 심볼릭 버전에서는 대기 항력을 포함하지 않는다. 이는 `_get_atmo_layer` 함수의 고도별 조건 분기(if-else)가 CasADi의 심볼릭 미분(automatic differentiation)과 호환되지 않기 때문이다. 최적화 문제에서 항력은 일반적으로 후처리(post-processing) 단계에서 고려하거나, 필요시 CasADi의 `if_else` 함수로 별도 구현해야 한다.


### 2.8 LEO 고도별 섭동 크기 비교

LEO에서 각 섭동원의 상대적 크기를 정량적으로 비교한다. 원궤도($e = 0$)를 가정하고, 각 가속도의 크기를 고도의 함수로 표현한다.

**이체 중력 가속도:**

$$
\|\mathbf{a}_{\text{grav}}\| = \frac{\mu}{r^2}
$$

$h = 400\;\text{km}$에서: $\|\mathbf{a}_{\text{grav}}\| = 398600.4418 / 6778.137^2 \approx 8.676\;\text{km/s}^2 \approx 8.676 \times 10^{-3}\;\text{m/s}^2 \times 10^3 = 8.676\;\text{m/s}^2$

**J2 섭동 가속도:**

$$
\|\mathbf{a}_{J2}\| \sim \frac{3}{2}\frac{\mu J_2 R_E^2}{r^4} \approx J_2 \cdot \left(\frac{R_E}{r}\right)^2 \cdot \|\mathbf{a}_{\text{grav}}\|
$$

$h = 400\;\text{km}$에서: $\|\mathbf{a}_{J2}\| / \|\mathbf{a}_{\text{grav}}\| \approx J_2 \cdot (R_E/r)^2 \approx 1.08 \times 10^{-3} \times (6378/6778)^2 \approx 9.57 \times 10^{-4}$

따라서 J2 가속도는 이체 중력의 약 $10^{-3}$배 수준이다.

**대기 항력 가속도:**

$h = 400\;\text{km}$, 원궤도 속도 $v \approx 7.67\;\text{km/s}$, $\rho \approx 8.484 \times 10^{-13}\;\text{kg/m}^3$일 때:

$$
\|\mathbf{a}_{\text{drag}}\| \approx \frac{1}{2} \times 2.2 \times 0.01 \times 8.484 \times 10^{-13} \times (7.67 \times 10^3)^2 \times 10^{-3}
$$
$$
\approx 5.5 \times 10^{-10}\;\text{km/s}^2 \approx 5.5 \times 10^{-7}\;\text{m/s}^2
$$

**제3체 섭동 (달/태양):** LEO에서 달과 태양의 조석 가속도는 대략 다음과 같다 (Montenbruck and Gill, 2000):

$$
\|\mathbf{a}_{\text{Moon}}\| \sim 5 \times 10^{-6}\;\text{m/s}^2, \quad \|\mathbf{a}_{\text{Sun}}\| \sim 2 \times 10^{-6}\;\text{m/s}^2
$$

**섭동 크기 비교 요약 ($h = 400\;\text{km}$):**

| 섭동원 | 가속도 크기 [m/s$^2$] | 대 중력 비 | 본 프로젝트 포함 여부 |
|:-:|:-:|:-:|:-:|
| 이체 중력 | $8.7$ | 1 | 포함 |
| J2 | $8.3 \times 10^{-3}$ | $\sim 10^{-3}$ | 포함 |
| 대기 항력 ($h=400$ km) | $5.5 \times 10^{-7}$ | $\sim 10^{-8}$ | 선택적 |
| 달 조석력 | $5 \times 10^{-6}$ | $\sim 10^{-7}$ | 미포함 |
| 태양 조석력 | $2 \times 10^{-6}$ | $\sim 10^{-7}$ | 미포함 |
| 태양 복사압 | $\sim 10^{-7}$ | $\sim 10^{-8}$ | 미포함 |

J2 섭동은 이체 중력 대비 $10^{-3}$ 수준으로, 수 주기에 걸친 궤도전이에서 RAAN drift 등 유의미한 궤도 변화를 유발하므로 반드시 포함해야 한다. 대기 항력은 고도 400 km 이상에서는 다른 섭동원보다 작지만, 저고도(< 300 km)에서는 밀도가 급증하여 중요해진다. 제3체 섭동과 태양 복사압은 단기 궤도전이(수 시간~수 일)에서는 영향이 미미하여 본 프로젝트에서 무시한다.

고도에 따른 항력 가속도 변화를 살펴보면:

| 고도 [km] | $\rho$ [kg/m$^3$] | $\|\mathbf{a}_{\text{drag}}\|$ [m/s$^2$] | 대 J2 비 |
|:-:|:-:|:-:|:-:|
| 200 | $1.585 \times 10^{-10}$ | $1.0 \times 10^{-4}$ | $\sim 10^{-2}$ |
| 300 | $8.152 \times 10^{-12}$ | $5.3 \times 10^{-6}$ | $\sim 10^{-3}$ |
| 400 | $8.484 \times 10^{-13}$ | $5.5 \times 10^{-7}$ | $\sim 10^{-5}$ |
| 600 | $3.614 \times 10^{-14}$ | $2.3 \times 10^{-8}$ | $\sim 10^{-6}$ |
| 800 | $1.170 \times 10^{-14}$ | $7.6 \times 10^{-9}$ | $\sim 10^{-6}$ |
| 1000 | $6.967 \times 10^{-16}$ | $4.5 \times 10^{-10}$ | $\sim 10^{-8}$ |

200 km에서는 항력이 J2 대비 $10^{-2}$ 수준으로 무시할 수 없으나, 600 km 이상에서는 $10^{-6}$ 이하로 실질적으로 무시 가능하다.


---

## 3. 구현 매핑

### 3.1 코드-수식 대응표

| 수학 표현 | Python 함수 | 파일 | 비고 |
|:--|:--|:--|:--|
| $\mathbf{a}_{\text{grav}} = -\mu/r^3\,\mathbf{r}$ | `gravity_acceleration(r, mu)` | `two_body.py:8-28` | NumPy/CasADi 겸용 |
| $\mathbf{a}_{J2}$ (Sec. 2.3 공식) | `j2_acceleration(r, mu, J2, R_E)` | `j2_perturbation.py:8-50` | NumPy/CasADi 겸용 |
| $\rho_0, h_{\text{ref}}, H$ 조회 | `_get_atmo_layer(h)` | `drag.py:14-36` | 고도별 구간 검색 |
| $\mathbf{a}_{\text{drag}}$ (Sec. 2.6 공식) | `exponential_drag(r, v, Cd, area_mass, omega_earth)` | `drag.py:39-78` | NumPy 전용 |
| $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$ | `spacecraft_eom_numpy(x, u, mu, include_j2, include_drag)` | `eom.py:12-38` | NumPy 전용, 전체 EOM |
| CasADi 심볼릭 EOM | `create_dynamics_function(mu, include_j2)` | `eom.py:41-68` | drag 제외, `ca.Function` 반환 |


### 3.2 NumPy/CasADi 겸용 설계

`gravity_acceleration`과 `j2_acceleration`은 동일한 수학 공식을 NumPy 배열과 CasADi 심볼릭 변수 양쪽에서 실행할 수 있도록 설계되었다. 입력 `r`의 타입을 런타임에 검사하여 분기한다.

```python
try:
    import casadi as ca
    if isinstance(r, (ca.MX, ca.SX, ca.DM)):
        r_norm = ca.norm_2(r)
        return -mu / r_norm**3 * r
except ImportError:
    pass

r_norm = np.linalg.norm(r)
return -mu / r_norm**3 * r
```

이 설계를 통해:
- **수치 적분**(RK4 등)에서는 NumPy 경로로 빠르게 실행
- **최적화**(direct collocation)에서는 CasADi 경로로 심볼릭 미분(AD) 지원

### 3.3 대기 모델 구간 검색

`_get_atmo_layer` 함수는 고도 $h$를 입력받아 `ATMO_PARAMS` 테이블에서 해당 구간의 $(\rho_0, h_{\text{ref}}, H)$를 반환한다. 선형 탐색으로 구현되어 있으며, 모델 범위(150--1100 km) 밖의 고도에 대해서는 `ValueError`를 발생시킨다.

```python
for h_lower, h_upper, rho0, h_ref, H in ATMO_PARAMS:
    if h_lower <= h < h_upper:
        return rho0, h_ref, H
```

### 3.4 상수 관리

모든 물리 상수는 `constants.py`에서 중앙 관리된다.

| 상수 | 변수명 | 값 | 단위 |
|:--|:--|:--|:--|
| 지구 중력 상수 | `MU_EARTH` | 398600.4418 | km$^3$/s$^2$ |
| 지구 적도 반지름 | `R_E` | 6378.137 | km |
| J2 계수 | `J2` | $1.08263 \times 10^{-3}$ | 무차원 |
| 항력 계수 | `CD` | 2.2 | 무차원 |
| 단면적/질량 비 | `AREA_MASS_RATIO` | 0.01 | m$^2$/kg |
| 지구 자전 각속도 | `OMEGA_EARTH` | $7.2921159 \times 10^{-5}$ | rad/s |

이 값들은 WGS-84 및 Vallado (2013)을 기준으로 한다.


---

## 4. 수치 검증

수치 검증은 `tests/test_dynamics.py`에 구현되어 있으며, 4가지 테스트로 구성된다.

### 4.1 원궤도 1주기 전파

**목적:** 수치 적분기와 동역학 모델의 기본 정확도를 확인한다.

**방법:** 적도 원궤도($h = 400\;\text{km}$)에서 $\mathbf{u} = \mathbf{0}$으로 1 케플러 주기를 RK4로 전파한다.

- **J2 미포함 케이스:** 이체 문제의 정확해로 원궤도가 폐곡선이므로, 1주기 후 초기 상태로 복귀해야 한다.
  - 판정 기준: 위치 오차 $< 10^{-8}\;\text{km}$, 속도 오차 $< 10^{-10}\;\text{km/s}$
  - RK4 스텝 수: 10,000

- **J2 포함 케이스:** J2 secular drift로 인해 카르테시안 상태의 완전 복귀는 기대할 수 없다. 대신 궤도 반경($\|\mathbf{r}\|$)이 보존되는지 확인한다.
  - 판정 기준: 궤도 반경 변화 $< 0.1\;\text{km}$, 속력 변화 $< 10^{-4}\;\text{km/s}$

### 4.2 에너지 보존

**목적:** 이체 문제($J2$ 미포함, $\mathbf{u} = \mathbf{0}$)에서 수치 적분의 에너지 보존 성능을 확인한다.

**방법:** 적도 원궤도에서 10주기(약 15.3시간)를 전파하며 매 주기 끝에서 비에너지를 계산한다.

$$
\mathcal{E} = \frac{1}{2}\|\mathbf{v}\|^2 - \frac{\mu}{\|\mathbf{r}\|}
$$

- 판정 기준: $|\Delta\mathcal{E}| < 10^{-10}\;\text{km}^2/\text{s}^2$
- RK4 총 스텝 수: 50,000 (주기당 5,000)

이 기준은 비에너지의 절대값($|\mathcal{E}| \approx 29.4\;\text{km}^2/\text{s}^2$)에 비해 $\sim 3 \times 10^{-12}$의 상대 오차에 해당하며, RK4 적분기의 충분한 정확도를 입증한다.


### 4.3 CasADi/NumPy 일관성

**목적:** 동일한 입력에 대해 NumPy 경로와 CasADi 경로가 동일한 결과를 산출하는지 확인한다.

**방법:** 세 가지 입력 조합에서 `spacecraft_eom_numpy`와 `create_dynamics_function`의 출력을 비교한다.
1. 적도 원궤도, J2 미포함, 추력 없음
2. 경사 원궤도($i = 45°$), J2 포함, 추력 없음
3. 경사 원궤도($i = 30°$, $h = 500\;\text{km}$), J2 포함, 추력 포함

- 판정 기준: 모든 성분에서 최대 오차 $< 10^{-14}$
- 이 수준은 부동소수점 산술의 기계 엡실론(machine epsilon, $\sim 2.2 \times 10^{-16}$)에 가까운 정밀도이다.

### 4.4 J2 RAAN Drift

**목적:** J2 구현의 물리적 정확도를 검증한다. 수치 적분으로 관측된 RAAN 변화량이 해석식(Sec. 2.4)의 예측과 일치하는지 확인한다.

**방법:**
- 초기 조건: $h = 400\;\text{km}$, $i = 45°$ 원궤도 ($\Omega_0 = 0$, ascending node가 $x$-축)
- 전파 시간: 10 케플러 주기
- RK4 스텝 수: 100,000

RAAN은 각운동량 벡터 $\mathbf{h} = \mathbf{r} \times \mathbf{v}$로부터 추출한다:

$$
\Omega = \text{atan2}(h_x, -h_y)
$$

RAAN 변화량의 해석적 예측:

$$
\Delta\Omega_{\text{analytical}} = \dot{\Omega} \cdot t_{\text{total}}
$$

여기서 $\dot{\Omega}$는 Sec. 2.4의 secular rate이다.

- **판정 기준:** 상대 오차 $< 0.5\%$

$$
\frac{|\Delta\Omega_{\text{numerical}} - \Delta\Omega_{\text{analytical}}|}{|\Delta\Omega_{\text{analytical}}|} < 0.005
$$

0.5% 이내의 오차는 secular rate 공식이 1차 평균화(first-order averaging)에 기반하므로 단주기 진동(short-period oscillation)에 의한 잔차를 반영한 것이다.


---

## 5. 참고문헌

1. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, 4th ed. Microcosm Press. -- 지구 중력 모델, J2 섭동, 지수 대기 모델(Table 8-4), RAAN secular drift 유도의 주요 참고서.

2. Montenbruck, O. and Gill, E. (2000). *Satellite Orbits: Models, Methods and Applications*. Springer-Verlag. -- 구면조화함수 전개, 제3체 섭동 크기 비교, 대기 모델 개관.

3. Battin, R. H. (1999). *An Introduction to the Methods of Astrodynamics*. AIAA Education Series. -- 이체 문제 기본 이론, 가우스 행성 방정식 유도.

4. Graham, K. F. and Rao, A. V. (2016). "Minimum-Time Trajectory Optimization of Low-Thrust Earth-Orbit Transfers with Eclipsing." *Journal of Spacecraft and Rockets*, 53(2), 289--303. -- LEO 연속 추력 궤도전이에서의 환경 모델 적용 사례.

5. Schaub, H. and Junkins, J. L. (2018). *Analytical Mechanics of Space Systems*, 4th ed. AIAA Education Series. -- J2 secular/long-period/short-period 섭동 분류 및 해석적 표현.
