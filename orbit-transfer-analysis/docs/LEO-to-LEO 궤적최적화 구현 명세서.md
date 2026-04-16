# LEO-to-LEO 궤적최적화 구현 명세서

## 1. 개요

본 문서는 LEO-to-LEO 연속 추력 궤도전이 최적화를 위한 구현 방법론을 정리한다. 최적 추력 프로파일 데이터베이스 구축을 목표로 하며, 다양한 궤도 구성에 대해 체계적으로 최적해를 생성할 수 있는 안정적이고 효율적인 구현을 지향한다.

---

## 2. 문제 정의

### 2.1 최적 제어 문제

**상태 변수**: 위치 및 속도 (ECI 좌표계)
```
x = [r; v] ∈ ℝ⁶
```

**제어 변수**: 추력 가속도
```
u ∈ ℝ³
```

**비용함수**: 에너지 최소화 (L2 norm)
```
J = ∫₀ᵀ ||u(t)||² dt
```

**경계조건**:
- 출발 궤도: (a₀, e₀, i₀, Ω₀, ω₀) - true anomaly ν₀는 자유 변수
- 도착 궤도: (aᶠ, eᶠ, iᶠ, Ωᶠ, ωᶠ) - true anomaly νᶠ는 자유 변수

**제약조건**:
- 추력 상한: ||u(t)|| ≤ uₘₐₓ
- 최소 고도: ||r(t)|| ≥ Rₑ + hₘᵢₙ

### 2.2 문제 특성

| 항목 | 값/범위 |
|------|---------|
| 전이 시간 | ≤ 3일 (임무 특성에 따라) |
| 궤도 고도 | LEO (200~2000 km) |
| 섭동 모델 | Two-body + J2 (+ 선택적 대기 항력) |

---

## 3. 이산화 방법 선택

### 3.1 고려한 옵션들

#### Option A: 단일 Phase Hermite-Simpson
- **원리**: 3차 Hermite 다항식으로 상태 보간, Simpson 적분 규칙
- **노드 구조**: 균일 간격
- **장점**: 구현 단순, 전 구간 균일 커버
- **단점**: 높은 정확도 위해 많은 노드 필요

#### Option B: 단일 Phase LGL Pseudospectral
- **원리**: Legendre 다항식 기반 전역 보간, Gauss-Lobatto 적분
- **노드 구조**: 비균일 (양 끝점 밀집)
- **장점**: 적은 노드로 높은 정확도 (spectral convergence)
- **단점**: 피크 위치와 노드 분포 불일치 가능

#### Option C: hp-Adaptive Method
- **원리**: 구간별 다항식 차수(p) 및 구간 분할(h) 적응적 조절
- **장점**: 자동 mesh refinement
- **단점**: 구현 매우 복잡

#### Option D: Two-Pass Hybrid (Hermite-Simpson → Multi-Phase LGL)
- **원리**: 1단계에서 피크 탐지, 2단계에서 적응적 노드 배치
- **장점**: 피크 위치에 노드 집중, 논문 기여 보존
- **단점**: 2회 최적화 필요

### 3.2 선택: Two-Pass Hybrid Approach

**핵심 문제 인식:**
- LGL 노드는 양 끝점에 밀집 → 피크 위치와 무관
- 피크 위치를 사전에 알아야 적응적 노드 배치 가능
- Lambert 기반 예측은 논문 기여(피크 유형 예측)를 약화시킴

**해결책:**
- Pass 1: 균일 노드(Hermite-Simpson)로 피크 위치 탐지
- Pass 2: 탐지된 피크 기반 Multi-Phase LGL로 정밀 최적화

---

## 4. Two-Pass 최적화 구조

### 4.1 전체 흐름

```
┌────────────────────────────────────────────────────────────────┐
│  Pass 1: Hermite-Simpson (균일 노드)                           │
│  ────────────────────────────────────────────────────────────  │
│  • M = 30 균일 구간                                            │
│  • 빠른 수렴 (저정밀 허용)                                      │
│  • 출력: 대략적 추력 프로파일                                   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  피크 탐지                                                      │
│  ────────────────────────────────────────────────────────────  │
│  • ||u(t)|| local maxima 검출                                  │
│  • 피크 개수: n_peaks                                          │
│  • 피크 시각: [t₁*, t₂*, ...]                                  │
│  • 피크 폭 추정: [δ₁, δ₂, ...]                                 │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Pass 2: Multi-Phase LGL                                       │
│  ────────────────────────────────────────────────────────────  │
│  • Phase 구조: 피크/coasting 구간 분리                          │
│  • 피크 구간: 노드 밀집 (N=15)                                  │
│  • Coasting 구간: 노드 희소 (N=8)                              │
│  • Pass 1 해를 초기 추정값으로 활용                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  최종 해 및 분류                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Pass 1: Hermite-Simpson Collocation

**목적:** 피크 위치/개수 탐지 (정밀 해 불필요)

**노드 구조:**
```
주 노드:    t₀ ──── t₁ ──── t₂ ──── ... ──── tₘ
                │       │       │              │
중간점:      t₀.₅     t₁.₅     t₂.₅           tₘ₋₀.₅

• 균일 간격: h = T/M
• M = 30 구간 → 31개 주 노드 + 30개 중간점 = 61 collocation points
```

**Collocation 조건:**

상태 연속성 (Simpson rule):
```
x(tₖ₊₁) = x(tₖ) + (h/6)[f(tₖ) + 4f(tₖ₊₀.₅) + f(tₖ₊₁)]
```

중간점 조건 (Hermite interpolation):
```
x(tₖ₊₀.₅) = ½[x(tₖ) + x(tₖ₊₁)] + (h/8)[f(tₖ) - f(tₖ₊₁)]
```

**비용함수 (Simpson 적분):**
```
J ≈ Σₖ (h/6)[||uₖ||² + 4||uₖ₊₀.₅||² + ||uₖ₊₁||²]
```

**결정 변수:**
```
z = [x₀, x₀.₅, x₁, x₁.₅, ..., xₘ,
     u₀, u₀.₅, u₁, u₁.₅, ..., uₘ,
     ν₀, νᶠ]

변수 수: 9 × 61 + 2 = 551개
```

**장점:**
- 균일 노드로 전 구간 균일 커버 → 피크 누락 방지
- 구현 단순
- 피크 예측에 의존하지 않음 → 논문 기여 보존

### 4.3 피크 탐지 알고리즘

**입력:** Pass 1 최적화 결과 (시간 t, 제어 u)

**알고리즘:**
```matlab
function [n_peaks, peak_times, peak_widths] = detect_peaks(t, u)
    % 1. 추력 크기 계산
    u_mag = vecnorm(u, 2, 2);  % ||u(t)||

    % 2. Smoothing (noise 제거)
    u_smooth = movmean(u_mag, 5);

    % 3. Local maxima 탐지
    threshold = 0.1 * max(u_smooth);  % 최대값의 10% 이상
    [peaks, locs] = findpeaks(u_smooth, t, ...
        'MinPeakProminence', threshold, ...
        'MinPeakDistance', T/10);  % 최소 피크 간격

    % 4. 피크 폭 추정 (반치폭 기준)
    peak_widths = estimate_peak_widths(t, u_smooth, locs);

    n_peaks = length(peaks);
    peak_times = locs;
end
```

**분류 기준:**
| 피크 개수 | 분류 | Phase 구조 |
|-----------|------|------------|
| 1 | Unimodal | 1-phase (단일 LGL) |
| 2 | Bimodal | 3-phase (peak-coast-peak) |
| 3+ | Multimodal | 2N-1 phase |

### 4.4 Pass 2: Multi-Phase LGL

**목적:** 피크 근처 고정밀 해 획득

**Phase 구조 결정 (Bimodal 예시):**
```
피크 시각: t₁*, t₂*
피크 폭: δ₁, δ₂

Phase 1 (Peak 1):    [0, t₁* + δ₁]         N₁ = 15 노드
Phase 2 (Coasting):  [t₁* + δ₁, t₂* - δ₂]  N₂ = 8 노드
Phase 3 (Peak 2):    [t₂* - δ₂, T]         N₃ = 15 노드
```

**결정 변수 구조:**
```
z = [X⁽¹⁾, U⁽¹⁾, X⁽²⁾, U⁽²⁾, X⁽³⁾, U⁽³⁾, ν₀, νf, τ₁, τ₂]

여기서:
  X⁽ᵏ⁾ = Phase k의 상태 변수
  U⁽ᵏ⁾ = Phase k의 제어 변수
  τₖ  = Phase 경계 시각 (정규화, 최적화 변수)
```

**Phase별 LGL 이산화:**
```
Phase k의 시간 구간: [Tₖ₋₁, Tₖ]
정규화 좌표: σ ∈ [-1, 1]

물리 시간: t = Tₖ₋₁ + (Tₖ - Tₖ₋₁)/2 × (σ + 1)

상태 미분: dx/dt = 2/(Tₖ - Tₖ₋₁) × Σⱼ Dᵢⱼ xⱼ
```

**Phase 경계 연속성 (Linkage Constraints):**
```
x⁽ᵏ⁾(σ = 1) = x⁽ᵏ⁺¹⁾(σ = -1),  k = 1, ..., P-1
```

**Phase 경계 순서 조건:**
```
0 < τ₁ < τ₂ < ... < τₚ₋₁ < 1
τₖ₊₁ - τₖ ≥ εₘᵢₙ  (최소 phase 길이 보장, εₘᵢₙ ≈ 0.05)
```

**초기 추정값:**
- 상태/제어: Pass 1 해를 보간하여 사용 (warm start)
- Phase 경계: 피크 탐지 결과 기반

---

## 5. NLP 솔버 선택

### 5.1 고려한 옵션들 (MATLAB fmincon 알고리즘)

#### Option A: Sequential Quadratic Programming (SQP)
- **작동 원리**: 매 반복에서 QP 부문제 해결, BFGS Hessian 근사
- **장점**: 일반적인 NLP에 강건, 중간 규모 문제에 적합
- **권장 문제 규모**: 변수 수 ~100개 이하

#### Option B: Interior-Point
- **작동 원리**: Barrier function으로 부등식 제약 처리, Newton step
- **장점**: 대규모 희소 문제에 효율적, 메모리 효율적
- **권장 문제 규모**: 변수 수 ~1000개 이상

#### Option C: Active-Set
- **작동 원리**: Active constraint set 관리하며 반복
- **장점**: 소규모 문제에서 빠름
- **권장 문제 규모**: 변수 수 ~50개 이하

### 5.2 선택: Interior-Point Algorithm

**문제 규모 분석:**
```
Pass 1 (Hermite-Simpson):
  변수 수: ~550개

Pass 2 (Multi-Phase LGL, 3-phase 예시):
  Phase 1: 9 × 16 = 144
  Phase 2: 9 × 9 = 81
  Phase 3: 9 × 16 = 144
  추가 변수: 4 (ν₀, νf, τ₁, τ₂)
  총: ~373개
```

**선택 이유:**

1. **문제 규모**: 400~550개 변수 범위에서 Interior-Point 안정적
2. **희소 구조 활용**: Collocation 제약의 band-sparse Jacobian
3. **제약조건 처리**: 등식/부등식 제약 동시 처리 효율적
4. **데이터베이스 구축**: 수천 케이스 반복 시 평균 수렴 시간 중요

---

## 6. 구현 세부사항

### 6.1 Hermite-Simpson 구현 (Pass 1)

**NLP 구조:**
```matlab
function [c, ceq] = hermite_simpson_constraints(z, params)
    % z: 결정 변수 벡터
    % 상태/제어 추출
    [x, u, nu0, nuf] = unpack_variables(z, params);

    % 경계조건
    ceq_bc = [x(:,1) - oe2rv(params.alpha0, nu0);
              x(:,end) - oe2rv(params.alphaf, nuf)];

    % Collocation (각 구간)
    ceq_col = [];
    for k = 1:M
        % 주 노드
        xk = x(:, 2*k-1);
        xk_mid = x(:, 2*k);
        xk1 = x(:, 2*k+1);

        uk = u(:, 2*k-1);
        uk_mid = u(:, 2*k);
        uk1 = u(:, 2*k+1);

        fk = dynamics(xk, uk);
        fk_mid = dynamics(xk_mid, uk_mid);
        fk1 = dynamics(xk1, uk1);

        % Simpson continuity
        defect1 = xk1 - xk - (h/6)*(fk + 4*fk_mid + fk1);

        % Hermite midpoint
        defect2 = xk_mid - 0.5*(xk + xk1) - (h/8)*(fk - fk1);

        ceq_col = [ceq_col; defect1; defect2];
    end

    ceq = [ceq_bc; ceq_col];

    % 부등식 제약 (추력 상한, 고도 하한)
    c = inequality_constraints(x, u, params);
end
```

### 6.2 LGL 노드 및 미분 행렬 (Pass 2)

```matlab
function [tau, w, D] = lgl_nodes(N)
    % tau: LGL nodes in [-1, 1]
    % w: quadrature weights
    % D: differentiation matrix

    % 끝점
    tau = zeros(N+1, 1);
    tau(1) = -1;
    tau(N+1) = 1;

    % 내부 노드: P'_N(τ) = 0의 근
    % Newton iteration으로 계산
    tau(2:N) = lgl_interior_nodes(N);

    % 가중치
    P_N = legendre_poly(N, tau);
    w = 2 ./ (N*(N+1) * P_N.^2);

    % 미분 행렬
    D = zeros(N+1, N+1);
    for i = 1:N+1
        for j = 1:N+1
            if i ~= j
                D(i,j) = P_N(i) / (P_N(j) * (tau(i) - tau(j)));
            end
        end
        D(i,i) = -sum(D(i,:));
    end
end
```

### 6.3 Multi-Phase NLP 구조 (Pass 2)

```matlab
function [c, ceq] = multiphase_lgl_constraints(z, params, phase_info)
    % phase_info: 각 phase의 노드 수, 경계 정보

    ceq = [];

    % 각 Phase별 collocation
    for p = 1:n_phases
        [x_p, u_p] = extract_phase_vars(z, p, phase_info);
        [tau, ~, D] = lgl_nodes(phase_info(p).N);

        T_start = get_phase_start(z, p);
        T_end = get_phase_end(z, p);
        dt = T_end - T_start;

        % Collocation constraints
        for i = 1:phase_info(p).N+1
            dx_dt = (2/dt) * D(i,:) * x_p';
            f_i = dynamics(x_p(:,i), u_p(:,i));
            ceq = [ceq; dx_dt' - f_i];
        end
    end

    % Linkage constraints (phase 경계 연속성)
    for p = 1:n_phases-1
        x_end_p = get_phase_final_state(z, p);
        x_start_p1 = get_phase_initial_state(z, p+1);
        ceq = [ceq; x_end_p - x_start_p1];
    end

    % 경계조건
    ceq = [ceq; boundary_conditions(z, params)];

    % Phase 경계 순서 조건
    c = phase_ordering_constraints(z, phase_info);
end
```

### 6.4 Warm Start (Pass 1 → Pass 2)

```matlab
function z0_pass2 = generate_initial_guess(sol_pass1, phase_info)
    % Pass 1 해를 Pass 2 초기 추정값으로 변환

    t_pass1 = sol_pass1.t;
    x_pass1 = sol_pass1.x;
    u_pass1 = sol_pass1.u;

    z0_pass2 = [];

    for p = 1:n_phases
        % Phase p의 시간 구간
        T_start = phase_info(p).T_start;
        T_end = phase_info(p).T_end;

        % LGL 노드 시각 계산
        [tau, ~, ~] = lgl_nodes(phase_info(p).N);
        t_lgl = T_start + (T_end - T_start)/2 * (tau + 1);

        % Pass 1 해를 LGL 노드에서 보간
        x_p = interp1(t_pass1, x_pass1', t_lgl)';
        u_p = interp1(t_pass1, u_pass1', t_lgl)';

        z0_pass2 = [z0_pass2; x_p(:); u_p(:)];
    end

    % 경계 조건 변수
    z0_pass2 = [z0_pass2; sol_pass1.nu0; sol_pass1.nuf];

    % Phase 경계 시각 (정규화)
    for p = 1:n_phases-1
        tau_p = phase_info(p).T_end / sol_pass1.T;
        z0_pass2 = [z0_pass2; tau_p];
    end
end
```

### 6.5 fmincon 옵션 설정

```matlab
% Pass 1 옵션 (빠른 수렴, 낮은 정밀도 허용)
options_pass1 = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'SpecifyObjectiveGradient', true, ...
    'SpecifyConstraintGradient', true, ...
    'OptimalityTolerance', 1e-4, ...
    'ConstraintTolerance', 1e-4, ...
    'MaxIterations', 500, ...
    'Display', 'iter');

% Pass 2 옵션 (높은 정밀도)
options_pass2 = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'SpecifyObjectiveGradient', true, ...
    'SpecifyConstraintGradient', true, ...
    'OptimalityTolerance', 1e-6, ...
    'ConstraintTolerance', 1e-6, ...
    'MaxIterations', 1000, ...
    'Display', 'iter');
```

---

## 7. 검증 계획

### 7.1 단위 테스트
- Hermite-Simpson collocation 정확도 검증
- LGL 노드/가중치 정확도 검증 (Chebfun 대비)
- 미분 행렬 정확도 검증 (해석해 대비)
- 피크 탐지 알고리즘 검증 (합성 데이터)

### 7.2 통합 테스트
- 단순 케이스 (coplanar circular → circular)
  - Pass 1 단독 vs Pass 1+2 비용값 비교
  - Hohmann 전이 해석해와 비교
- Pass 1 피크 탐지 → Pass 2 phase 구조 일관성 확인
- Phase 경계 최적화가 피크 위치로 수렴하는지 확인

### 7.3 성능 테스트
- Pass 1 평균 수렴 시간
- Pass 2 평균 수렴 시간 (warm start 효과)
- 전체 파이프라인 시간 (피크 탐지 포함)
- 수렴 실패율

---

## 8. 예상 계산 비용

```
Pass 1 (Hermite-Simpson, M=30):
  변수: ~550개
  수렴 시간: ~3-5초

피크 탐지:
  시간: < 0.1초

Pass 2 (Multi-Phase LGL):
  변수: ~400개
  초기값: Pass 1 해 (warm start)
  수렴 시간: ~2-4초

총 시간: ~5-10초/케이스
데이터베이스 (1000 케이스): ~1.5-3시간
```

---

## 9. 요약

| 항목 | 선택 | 대안 | 선택 이유 |
|------|------|------|-----------|
| 전체 구조 | Two-Pass Hybrid | 단일 Phase, hp-Adaptive | 피크 위치에 적응적 노드 배치, 논문 기여 보존 |
| Pass 1 | Hermite-Simpson (M=30) | LGL | 균일 노드로 피크 누락 방지 |
| Pass 2 | Multi-Phase LGL | 단일 LGL | 피크 근처 고밀도 노드 |
| NLP 솔버 | fmincon (interior-point) | SQP, Active-Set | 희소 구조 활용, 대규모 제약 효율적 |
| Gradient | 해석적 제공 | 수치 미분 | 수렴 속도 및 안정성 향상 |

---

## 10. 참고문헌

- Hargraves, C.R. & Paris, S.W. (1987). Direct Trajectory Optimization Using Nonlinear Programming and Collocation
- Herman, A.L. & Conway, B.A. (1996). Direct Optimization Using Collocation Based on High-Order Gauss-Lobatto Quadrature Rules
- Rao, A.V. et al. (2025). Legendre-Gauss-Lobatto Collocation Method for Optimal Control
- Fahroo, F. & Ross, I.M. (2002). Direct Trajectory Optimization by a Chebyshev Pseudospectral Method
- Patterson, M.A. & Rao, A.V. (2014). GPOPS-II: A MATLAB Software for Solving Multiple-Phase Optimal Control Problems
- Betts, J.T. (1998). Survey of Numerical Methods for Trajectory Optimization

---

*문서 작성일: 2025-01-13*
*최종 수정: Two-Pass Hybrid 방식 채택*
