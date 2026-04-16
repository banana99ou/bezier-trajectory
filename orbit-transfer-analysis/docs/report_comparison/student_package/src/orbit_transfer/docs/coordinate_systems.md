# A1. 좌표계 및 궤도요소 변환

> 대응 코드: `src/orbit_transfer/astrodynamics/orbital_elements.py`

## 1. 목적

본 문서는 지구 중심 관성 좌표계(ECI)와 고전 궤도요소(classical orbital elements) 사이의 상호 변환에 관한 수학적 배경 및 구현 세부사항을 기술한다. 구체적으로 다음 세 가지 변환을 다룬다.

1. **순변환 (OE $\to$ ECI)**: 6개의 고전 궤도요소로부터 ECI 위치/속도 벡터를 계산한다.
2. **역변환 (ECI $\to$ OE)**: ECI 위치/속도 벡터로부터 6개의 궤도요소를 복원한다.
3. **CasADi 심볼릭 변환**: 순변환의 CasADi 자동미분 호환 버전을 제공한다.

이 변환들은 LEO-to-LEO 연속 추력 궤도전이 최적화 문제의 경계 조건 설정 및 궤적 후처리에 핵심적으로 사용된다.

---

## 2. 수학적 배경

### 2.1 ECI 좌표계 정의

지구 중심 관성 좌표계(Earth-Centered Inertial, ECI)는 지구 중심을 원점으로 하는 관성 기준계이다. 세 축은 다음과 같이 정의된다.

- **$\hat{X}$축**: 춘분점(vernal equinox) 방향. 태양이 천구 적도를 남에서 북으로 통과하는 방향이다.
- **$\hat{Y}$축**: 적도면(equatorial plane) 내에서 $\hat{X}$축에 직교하며, 오른손 좌표계를 형성하는 방향이다. 즉 $\hat{Y} = \hat{Z} \times \hat{X}$이다.
- **$\hat{Z}$축**: 지구 자전축(북극) 방향. 적도면에 수직이다.

관성 기준계의 조건으로서, ECI 좌표계의 축 방향은 시간에 대해 고정되어 있다(세차, 장동 등은 무시하거나 별도 보정). 따라서 뉴턴 역학의 운동 방정식을 직접 적용할 수 있다.

### 2.2 고전 궤도요소

케플러 궤도를 기술하는 6개의 고전 궤도요소는 다음과 같다.

| 기호 | 명칭 | 정의 | 범위 |
|------|------|------|------|
| $a$ | 장반경 (semi-major axis) | 타원 궤도의 장축 절반 길이 | $a > 0$ |
| $e$ | 이심률 (eccentricity) | 궤도 형상의 타원 정도. $e=0$이면 원, $0<e<1$이면 타원 | $0 \le e < 1$ |
| $i$ | 경사각 (inclination) | 궤도면과 적도면 사이의 각도 | $0 \le i \le \pi$ |
| $\Omega$ | 승교점경도 (RAAN) | 춘분점에서 승교점(ascending node)까지의 적도면 위 각도 | $0 \le \Omega < 2\pi$ |
| $\omega$ | 근점인수 (argument of periapsis) | 승교점에서 근지점까지의 궤도면 위 각도 | $0 \le \omega < 2\pi$ |
| $\nu$ | 진근점이각 (true anomaly) | 근지점에서 위성 위치까지의 궤도면 위 각도 | $0 \le \nu < 2\pi$ |

물리적으로, $(a, e)$는 궤도의 **크기와 형상**, $(i, \Omega)$는 궤도면의 **공간적 자세**, $\omega$는 궤도면 내 타원의 **배향**, $\nu$는 궤도 위의 **위성 위치**를 각각 결정한다.

### 2.3 근지점 좌표계 (Perifocal Frame, PQW)

근지점 좌표계(perifocal frame)는 궤도면에 고정된 좌표계로, 다음 세 축으로 정의된다.

- **$\hat{P}$축**: 궤도 중심에서 근지점(periapsis) 방향
- **$\hat{Q}$축**: 궤도면 내에서 $\hat{P}$에 직교하며, 위성 운동 방향 쪽
- **$\hat{W}$축**: 궤도 각운동량 벡터 방향, $\hat{W} = \hat{P} \times \hat{Q}$

이 좌표계에서 위성의 위치와 속도는 진근점이각 $\nu$만의 함수로 간결하게 표현된다.

### 2.4 PQW 좌표에서의 위치 및 속도

#### 궤도 방정식

케플러 궤도에서 위성과 중심체 사이의 거리는 다음과 같다.

$$r = \frac{p}{1 + e \cos\nu}$$

여기서 $p$는 반통경(semi-latus rectum)으로, 장반경 및 이심률과 다음 관계를 가진다.

$$p = a(1 - e^2)$$

#### PQW 위치 벡터

근지점이각 $\nu$에서의 위치 벡터를 PQW 좌표로 표현하면 다음과 같다.

$$\mathbf{r}_{PQW} = \begin{bmatrix} r\cos\nu \\ r\sin\nu \\ 0 \end{bmatrix}$$

$\hat{W}$ 성분이 0인 이유는 위성이 궤도면 위에 있기 때문이다.

#### PQW 속도 벡터

속도 벡터의 유도를 위해, 위치 벡터를 시간에 대해 미분한다. 궤도 운동에서 $\dot{\nu} = h / r^2$ (각운동량 보존)이므로, 시간 미분 대신 $\nu$에 대한 미분과 $\dot{\nu}$의 곱으로 표현할 수 있다.

$$\mathbf{v}_{PQW} = \frac{d\mathbf{r}_{PQW}}{dt} = \frac{d\mathbf{r}_{PQW}}{d\nu}\dot{\nu}$$

$\mathbf{r}_{PQW}$를 $\nu$로 미분하면,

$$\frac{d\mathbf{r}_{PQW}}{d\nu} = \begin{bmatrix} \dot{r}\cos\nu - r\sin\nu \\ \dot{r}\sin\nu + r\cos\nu \\ 0 \end{bmatrix} \frac{1}{\dot{\nu}}$$

여기서 궤도 방정식으로부터

$$\dot{r} = \frac{pe\sin\nu}{(1+e\cos\nu)^2}\dot{\nu} = \frac{h}{p}e\sin\nu$$

이고 $h = \sqrt{\mu p}$이므로, 정리하면 다음과 같다.

$$\mathbf{v}_{PQW} = \sqrt{\frac{\mu}{p}} \begin{bmatrix} -\sin\nu \\ e + \cos\nu \\ 0 \end{bmatrix}$$

이 결과는 속도 벡터가 $\nu$와 궤도 파라미터 $(\mu, p, e)$만으로 결정됨을 보여준다.

### 2.5 Perifocal $\to$ ECI 회전 변환

PQW 좌표계에서 ECI 좌표계로의 변환은 3-1-3 오일러 회전(Euler rotation)으로 구성된다. 이 변환은 ECI 좌표계를 세 번의 회전을 통해 PQW 좌표계와 일치시키는 과정의 역으로 이해할 수 있다.

#### 회전 순서

ECI $\to$ PQW 변환은 다음 세 단계의 회전으로 수행된다.

1. **$\hat{Z}$축 기준 $\Omega$ 회전**: ECI의 $\hat{X}$축을 승교점 방향(교선, line of nodes)으로 정렬한다.
2. **$\hat{X}'$축 기준 $i$ 회전**: 적도면을 궤도면으로 기울인다.
3. **$\hat{Z}''$축 기준 $\omega$ 회전**: 교선 방향을 근지점 방향($\hat{P}$)으로 정렬한다.

따라서 ECI $\to$ PQW 변환 행렬은 다음과 같다.

$$\mathbf{R}_{ECI \to PQW} = \mathbf{R}_3(\omega) \, \mathbf{R}_1(i) \, \mathbf{R}_3(\Omega)$$

여기서 기본 회전 행렬들은 다음과 같다.

$$\mathbf{R}_3(\theta) = \begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{R}_1(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & \sin\theta \\ 0 & -\sin\theta & \cos\theta \end{bmatrix}$$

우리가 필요한 것은 **PQW $\to$ ECI** 변환이므로, 역변환(전치)을 취한다.

$$\mathbf{R}_{PQW \to ECI} = \mathbf{R}_{ECI \to PQW}^T = \mathbf{R}_3^T(\Omega) \, \mathbf{R}_1^T(i) \, \mathbf{R}_3^T(\omega)$$

이는 $\mathbf{R}_3(-\Omega) \, \mathbf{R}_1(-i) \, \mathbf{R}_3(-\omega)$와 동일하다.

#### 회전 행렬 원소 전개

세 행렬의 곱을 전개하면 PQW $\to$ ECI 회전 행렬 $\mathbf{R}$의 원소는 다음과 같다.

$$\mathbf{R} = \begin{bmatrix} R_{11} & R_{12} & R_{13} \\ R_{21} & R_{22} & R_{23} \\ R_{31} & R_{32} & R_{33} \end{bmatrix}$$

각 원소를 명시하면,

$$R_{11} = \cos\Omega\cos\omega - \sin\Omega\sin\omega\cos i$$

$$R_{12} = -\cos\Omega\sin\omega - \sin\Omega\cos\omega\cos i$$

$$R_{13} = \sin\Omega\sin i$$

$$R_{21} = \sin\Omega\cos\omega + \cos\Omega\sin\omega\cos i$$

$$R_{22} = -\sin\Omega\sin\omega + \cos\Omega\cos\omega\cos i$$

$$R_{23} = -\cos\Omega\sin i$$

$$R_{31} = \sin\omega\sin i$$

$$R_{32} = \cos\omega\sin i$$

$$R_{33} = \cos i$$

#### 최종 변환

ECI 좌표에서의 위치/속도 벡터는 다음과 같이 계산된다.

$$\mathbf{r}_{ECI} = \mathbf{R} \, \mathbf{r}_{PQW}, \quad \mathbf{v}_{ECI} = \mathbf{R} \, \mathbf{v}_{PQW}$$

이때 $\mathbf{R}$은 위치/속도에 동일하게 적용된다. 회전은 좌표계만 변환하며, 미분 구조(시간 미분)에 영향을 주지 않기 때문이다.

### 2.6 역변환: ECI $\to$ 궤도요소

ECI 상태벡터 $(\mathbf{r}, \mathbf{v})$로부터 궤도요소를 복원하는 절차는 다음과 같다.

#### (1) 각운동량 벡터

$$\mathbf{h} = \mathbf{r} \times \mathbf{v}, \quad h = \|\mathbf{h}\|$$

#### (2) 교선 벡터 (node vector)

$$\mathbf{n} = \hat{K} \times \mathbf{h}, \quad n = \|\mathbf{n}\|$$

여기서 $\hat{K} = [0, 0, 1]^T$은 ECI의 $\hat{Z}$축 단위벡터이다. $\mathbf{n}$은 승교점 방향을 가리킨다.

#### (3) 이심률 벡터

이심률 벡터는 근지점 방향을 가리키며, 크기가 이심률과 같다.

$$\mathbf{e} = \frac{1}{\mu}\left[\left(v^2 - \frac{\mu}{r}\right)\mathbf{r} - (\mathbf{r} \cdot \mathbf{v})\mathbf{v}\right], \quad e = \|\mathbf{e}\|$$

이 식은 운동 방정식으로부터 유도되는 라플라스-룽게-렌츠(Laplace-Runge-Lenz) 벡터의 무차원 형태이다.

#### (4) 장반경 (vis-viva equation)

궤도 에너지(specific orbital energy)는 다음과 같다.

$$\mathcal{E} = \frac{v^2}{2} - \frac{\mu}{r}$$

vis-viva 관계로부터 장반경을 구한다.

$$a = -\frac{\mu}{2\mathcal{E}}$$

#### (5) 경사각

$$i = \arccos\left(\frac{h_z}{h}\right)$$

여기서 $h_z$는 $\mathbf{h}$의 $z$성분이다. $\mathbf{h}$가 $\hat{Z}$축과 이루는 각도가 경사각이다.

#### (6) 승교점경도

$$\Omega = \arccos\left(\frac{n_x}{n}\right)$$

사분면 판별: $n_y < 0$이면 $\Omega = 2\pi - \Omega$이다. 이는 $\mathbf{n}$이 $\hat{X}$축 아래쪽(음의 $y$방향)에 있으면 $\Omega$가 $\pi$보다 크기 때문이다.

#### (7) 근점인수

$$\omega = \arccos\left(\frac{\mathbf{n} \cdot \mathbf{e}}{n \, e}\right)$$

사분면 판별: $e_z < 0$이면 $\omega = 2\pi - \omega$이다. 근지점이 궤도면 아래(남쪽)에 있으면 $\omega$는 $\pi$보다 크다.

#### (8) 진근점이각

$$\nu = \arccos\left(\frac{\mathbf{e} \cdot \mathbf{r}}{e \, r}\right)$$

사분면 판별: $\mathbf{r} \cdot \mathbf{v} < 0$이면 $\nu = 2\pi - \nu$이다. 위성이 원점에서 멀어지는 중($\mathbf{r} \cdot \mathbf{v} > 0$)이면 $\nu < \pi$이고, 가까워지는 중이면 $\nu > \pi$이다.

### 2.7 특이값 처리

고전 궤도요소는 특정 궤도 형상에서 정의가 불분명해지는 특이점(singularity)을 가진다.

#### (a) 원궤도 ($e = 0$): $\omega$ 미정

이심률이 0이면 근지점이 존재하지 않으므로 $\omega$가 정의되지 않는다. 이 경우 $\omega = 0$으로 설정하고, $\nu$를 **argument of latitude** $u$로 대체한다.

$$u = \omega + \nu$$

$\omega = 0$이므로 $u = \nu$가 되며, $\nu$는 승교점에서 위성 위치까지의 궤도면 위 각도를 의미한다.

구체적으로, $\nu$는 다음과 같이 계산한다.

$$\nu = \arccos\left(\frac{\hat{\mathbf{n}} \cdot \hat{\mathbf{r}}}{1}\right)$$

사분면 판별은 $(\mathbf{n} \times \mathbf{r}) \cdot \mathbf{h}$의 부호로 수행한다. 이 외적이 $\mathbf{h}$와 반대 방향이면 $\nu = 2\pi - \nu$이다.

#### (b) 적도궤도 ($i = 0$): $\Omega$ 미정

경사각이 0이면 궤도면이 적도면과 일치하여 교선이 존재하지 않으므로 $\Omega$가 정의되지 않는다. 이 경우 $\Omega = 0$으로 설정한다.

$e > 0$인 적도 타원궤도에서는 $\omega$를 이심률 벡터의 방향으로부터 결정한다.

$$\omega = \text{atan2}(e_y, e_x)$$

#### (c) 원형 적도궤도 ($e = 0$, $i = 0$)

$\omega$와 $\Omega$가 모두 미정이다. 이 경우 $\Omega = \omega = 0$으로 설정하고, $\nu$를 **true longitude** $\lambda$로 대체한다.

$$\lambda = \Omega + \omega + \nu$$

$\Omega = \omega = 0$이므로 $\lambda = \nu$이며, 이는 춘분점에서 위성 위치까지의 적도면 위 각도이다.

$$\nu = \text{atan2}(r_y, r_x)$$

이 처리는 관례(convention)에 해당하며, 순변환과 역변환에서 동일한 관례를 사용하면 round-trip 일관성이 보장된다.

### 2.8 CasADi 심볼릭 구현

궤적 최적화 문제에서는 경계 조건으로서 궤도요소-상태벡터 변환이 NLP(Nonlinear Programming) 제약 조건에 포함된다. 이 제약 조건의 기울기(gradient) 및 야코비안(Jacobian)을 효율적으로 계산하기 위해 **자동미분(automatic differentiation)**이 필요하다.

CasADi는 심볼릭 연산 그래프(symbolic expression graph)를 구축하여 순방향/역방향 자동미분을 지원하는 프레임워크이다. `oe_to_rv_casadi` 함수는 `oe_to_rv`와 동일한 수학을 CasADi의 `MX` 심볼릭 타입으로 구현한다.

주요 차이점은 다음과 같다.

- `np.cos/sin` $\to$ `ca.cos/sin`: CasADi 심볼릭 삼각함수
- `np.array` $\to$ `ca.vertcat`: 심볼릭 벡터 결합
- `np.sqrt` $\to$ `ca.sqrt`: 심볼릭 제곱근
- 행렬 구성: `ca.horzcat`과 `ca.vertcat`의 조합으로 $3 \times 3$ 회전 행렬 생성

이렇게 구성된 심볼릭 표현은 `ca.Function`으로 래핑한 후 IPOPT 등의 NLP 솔버에 직접 전달할 수 있다. CasADi가 내부적으로 연쇄 법칙(chain rule)을 적용하여 야코비안 $\partial \mathbf{r}/\partial \mathbf{oe}$, $\partial \mathbf{v}/\partial \mathbf{oe}$를 정확하게 계산한다.

역변환 `rv_to_oe`는 `arccos`, `clip`, 조건 분기 등 CasADi에서 직접 미분 불가능한 연산을 포함하므로, 심볼릭 버전을 제공하지 않는다. 최적화 문제에서는 순변환만 제약 조건에 사용되기 때문에 이것으로 충분하다.

---

## 3. 구현 매핑

### 3.1 함수 매핑 테이블

| 수학 표현 | Python 함수 | 파일 위치 | 백엔드 | 비고 |
|-----------|------------|-----------|--------|------|
| OE $\to$ $(\mathbf{r}, \mathbf{v})$ | `oe_to_rv` | `orbital_elements.py:7-70` | NumPy | 수치 평가용 |
| $(\mathbf{r}, \mathbf{v})$ $\to$ OE | `rv_to_oe` | `orbital_elements.py:73-170` | NumPy | 특이값 분기 포함 |
| OE $\to$ $(\mathbf{r}, \mathbf{v})$ | `oe_to_rv_casadi` | `orbital_elements.py:173-238` | CasADi | NLP 자동미분용 |

### 3.2 순변환 (`oe_to_rv`) 구현 상세

| 수학 | 코드 | 행 |
|------|------|----|
| $p = a(1-e^2)$ | `p = a * (1.0 - e**2)` | 33 |
| $r = p/(1+e\cos\nu)$ | `r_mag = p / (1.0 + e * np.cos(nu))` | 36 |
| $\mathbf{r}_{PQW}$ | `r_pqw = np.array([r_mag*np.cos(nu), r_mag*np.sin(nu), 0.0])` | 39-41 |
| $\mathbf{v}_{PQW}$ | `v_pqw = np.sqrt(mu/p) * np.array([-np.sin(nu), e+np.cos(nu), 0.0])` | 43-45 |
| $\mathbf{R}$ 행렬 원소 | `R = np.array([[...], [...], [...]])` | 55-65 |
| $\mathbf{r} = \mathbf{R}\mathbf{r}_{PQW}$ | `r = R @ r_pqw` | 67 |
| $\mathbf{v} = \mathbf{R}\mathbf{v}_{PQW}$ | `v = R @ v_pqw` | 68 |

### 3.3 역변환 (`rv_to_oe`) 구현 상세

| 수학 | 코드 | 행 |
|------|------|----|
| $\mathbf{h} = \mathbf{r} \times \mathbf{v}$ | `h = np.cross(r, v)` | 99 |
| $\mathbf{n} = \hat{K} \times \mathbf{h}$ | `n = np.cross(K, h)` | 104 |
| 이심률 벡터 | `e_vec = ((v_mag**2 - mu/r_mag)*r - np.dot(r,v)*v) / mu` | 108 |
| $\mathcal{E} = v^2/2 - \mu/r$ | `energy = 0.5*v_mag**2 - mu/r_mag` | 112 |
| $a = -\mu/(2\mathcal{E})$ | `a = -mu / (2.0*energy)` | 113 |
| $i = \arccos(h_z/h)$ | `i = np.arccos(np.clip(h[2]/h_mag, -1, 1))` | 116 |
| $e=0$ 분기 | `if e < 1e-10:` | 131 |
| $i=0$ 분기 | `if i < 1e-10:` | 119, 135 |

### 3.4 특이값 처리 분기 구조

```
rv_to_oe
├── i < 1e-10?  → Omega = 0
│   └── else    → Omega = arccos(n_x/n), 사분면 보정
├── e < 1e-10?  → omega = 0
│   ├── i < 1e-10?  → nu = atan2(r_y, r_x)     [true longitude]
│   └── else        → nu = arccos(n_hat . r_hat) [argument of latitude]
└── else (e > 0)
    ├── i < 1e-10?  → omega = atan2(e_y, e_x)   [longitude of periapsis]
    └── else        → omega = arccos(n.e / ne), 사분면 보정
                      nu = arccos(e.r / er), 사분면 보정
```

---

## 4. 수치 검증

수치 검증은 `tests/test_orbital_elements.py`에 구현되어 있다. 검증 항목은 다음 네 가지이다.

### 4.1 Round-trip 일관성 (OE $\to$ RV $\to$ OE)

다양한 궤도 유형에 대해 순변환 후 역변환을 수행하고, 원래 궤도요소와의 차이를 확인한다.

| 테스트 케이스 | $a$ [km] | $e$ | $i$ [deg] | 허용 오차 |
|---------------|----------|-----|-----------|-----------|
| LEO-ISS | $R_E + 400$ | 0.001 | 51.6 | $\Delta a < 10^{-8}$, $\Delta e < 10^{-12}$ |
| High eccentricity | $R_E + 500$ | 0.3 | 28.5 | 각도 차이 $< 10^{-12}$ rad |
| GEO-like | 42164 | 0.01 | 0.05 | 동일 |
| Molniya-like | 26600 | 0.74 | 63.4 | 동일 |

각도 비교 시 $2\pi$ 주기성을 고려한 차이 함수를 사용한다.

$$\Delta\theta = \min\left(|\theta_1 - \theta_2| \mod 2\pi, \; 2\pi - |\theta_1 - \theta_2| \mod 2\pi\right)$$

### 4.2 원궤도 특이 케이스 ($e = 0$)

- **원형 경사궤도** ($e=0$, $i=51.6^\circ$): 궤도 반경 $\|\mathbf{r}\| = a$, 원궤도 속력 $\|\mathbf{v}\| = \sqrt{\mu/a}$ 확인. 역변환 후 $e < 10^{-10}$ 확인.
- **원형 적도궤도** ($e=0$, $i=0$): 동일한 반경/속력 검증. 역변환 후 $e < 10^{-10}$, $i < 10^{-10}$ 확인.

### 4.3 적도궤도 특이 케이스 ($i = 0$)

- **적도 타원궤도** ($e=0.1$, $i=0$): ECI $z$성분이 0임을 확인 ($|r_z| < 10^{-10}$, $|v_z| < 10^{-10}$). 역변환 후 $\omega + \nu$의 합이 보존됨을 확인.

### 4.4 CasADi/NumPy 일관성

동일한 입력에 대해 `oe_to_rv`(NumPy)와 `oe_to_rv_casadi`(CasADi)의 출력 차이를 비교한다.

| 테스트 케이스 | 허용 오차 |
|---------------|-----------|
| LEO-ISS | $\max|\mathbf{r}_{CA} - \mathbf{r}_{NP}| < 10^{-14}$ |
| High eccentricity | $\max|\mathbf{v}_{CA} - \mathbf{v}_{NP}| < 10^{-14}$ |
| GEO circular ($e=0$) | 동일 |

$10^{-14}$ 수준의 일치는 두 구현이 동일한 수학 연산을 수행하며, 부동소수점 연산 순서까지 일관적임을 의미한다.

---

## 5. 참고문헌

1. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications* (4th ed.). Microcosm Press. -- ECI 좌표계 정의, 고전 궤도요소 변환, 특이값 처리에 관한 표준 참고서.
2. Curtis, H. D. (2014). *Orbital Mechanics for Engineering Students* (3rd ed.). Butterworth-Heinemann. -- 근지점 좌표계, 회전 행렬 유도, 역변환 알고리즘의 교재적 설명.
3. Battin, R. H. (1999). *An Introduction to the Methods of Astrodynamics*. AIAA Education Series. -- 이심률 벡터 유도 및 궤도 역학의 수학적 기초.
4. Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., & Diehl, M. (2019). CasADi: a software framework for nonlinear optimization and optimal control. *Mathematical Programming Computation*, 11(1), 1-36. -- CasADi 자동미분 프레임워크의 설계 및 구현.
