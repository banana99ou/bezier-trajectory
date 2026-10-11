# Argument map: `papers/paper_draft.md`

This is the whole-paper coherence pass. Sections 1–6 are phase 1, a map of what the draft says.
Section 7 is phase 2, the diagnosis and the ranked problem list. Phase 3 (deciding the structure)
builds on both.

- **Snapshot:** commit `b6358a6` (2026-10-08). Line numbers (`L`) refer to that commit; the opening
  words of each paragraph are given so a row can be found after the lines drift.
- **Paragraph numbers** `P01`–`P76` count prose blocks in reading order. Display equations, tables
  and figure images are not numbered; table notes and the proposition block are.
- **Role vocabulary:** *motivation*, *prior work*, *positioning*, *definition*, *derivation*,
  *claim* (something the paper asserts and must back), *evidence* (a result that backs a claim),
  *setup*, *illustration*, *transition*, *summary*, *limitation*.
- **Needs** = what the reader must already have from earlier paragraphs. **Hands on** = what later
  paragraphs use from this one. Both name the paragraph that supplies or uses the item.

## 1. Spine

| Section | Job in the argument |
|---|---|
| Abstract | Method in one paragraph, results in one paragraph. |
| 1 Introduction | Continuous-time avoidance is needed (P03) → [17] is the starting point (P04) → avoidance is non-convex, two ways to convexify it (P05) → the second way rests on the convex hull and one half-space per sub-arc, which costs conservatism (P06) → three contributions (P07). |
| 2 Control-point representation | Notation and the three tools the method needs: the decision vector, linear boundary conditions, and the exact quadratic integral. |
| 3.1 Avoidance constraint | Build the sufficient condition (Proposition 1), its violation measure, and the conservatism estimate. Supports contribution 1. |
| 3.2 Objective | Turn the non-convex control-acceleration objective into an exact convex quadratic by piecewise gravity linearization. Supports contribution 2. |
| 3.3 Subproblem and SCvx | The convex QP, the normal-aware constraint linearization, the acceptance and stopping rules. Supports contribution 3. |
| 4 Setup | Problem, scenarios, solver, parameters, metrics. Also carries the only evidence for contribution 2 (P52). |
| 5.1 Scenarios | Five scenarios: avoidance holds and clearance matches the estimate at fixed subdivision count. |
| 5.2 Subdivision count | Monte Carlo over perturbed baseline cases: the n_seg-squared law, the cost and constraint-count trade-off, robustness on a wider range. |
| 5.3 Normal term | Ablation: fixing the normal makes most cases fail to converge. Backs contribution 3. |
| 6 Conclusion | Method summary, results, limitations. |

## 2. Paragraph map

### Abstract

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P01 | 7 | 본 연구에서는 비선형 중력장의 | Method: one supporting half-space per sub-arc, solved by successive convexification, whole-trajectory guarantee; gravity linearized per interval into a convex quadratic cost; the normal's change is in the linearization. | summary | — | — |
| P02 | 9 | 중심각과 궤도면을 달리한 | Results: every case converged; clearance close to the arc estimate and shrinking with the square of the subdivision count; few sub-arcs cost much more; cost approaches the finest value quickly while constraint count grows linearly; a fixed normal makes most cases fail. | summary | — | — |

### 1 Introduction

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P03 | 15 | 항공우주, 로봇공학, 자율 시스템의 | Node-based transcription can violate constraints between nodes [6]; continuous-time constraint satisfaction [7] needs a suitable representation. | motivation | — | the term 연속시간 제약 만족 → P06, P17 |
| P04 | 17 | Bézier 곡선의 제어점을 최적화 변수로 | Control points as variables [12, 13]; [17] built the Gram quadratic cost and used subdivision with separating planes in sequential convex programming. | prior work | — | [17] as baseline → P07, P09, P12, P16 |
| P05 | 19 | 장애물 외부의 자유 공간은 볼록 집합이 | Avoidance is non-convex; lossless convexification [3] and successive convexification (SCvx) [4, 5]; two convex treatments of avoidance: covering free space [10, 11], or a chosen side with supporting half-spaces. | positioning | — | SCvx → P07, P39; "the second method" → P06 |
| P06 | 21 | 두 번째 방법에서는 볼록 껍질 성질을 | The second method guarantees continuous-time satisfaction through the convex hull, which needs one half-space for all control points of a sub-arc; safe corridors [14–16] and this work do that; defines 보수성 (conservatism). | positioning + definition | P03, P05 | 보수성 → P07, P18, P27, P28, P55 |
| P07 | 23 | 본 연구에서는 [17]의 정식화를 바탕으로 | Contributions: (1) arc estimate of clearance and its dependence on subdivision count; (2) extend [17]'s linear CW model to two-body plus J2 gravity, linearized per interval into the Gram quadratic, and show the result barely changes with more linearization intervals; (3) include the normal's change in the linearization and compare with a fixed normal. | claim | P04, P05, P06 | contributions 1–3 → section 3, sections 5 |
| P08 | 25 | 2절에서는 제어점 공간의 궤적 표현과 | Paper organization. | transition | — | — |

### 2 Control-point representation

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P09 | 31 | 선행연구 [17]의 제어점 공간 정식화에 | This section follows [17]; operator and Gram properties from [12, 17]. | transition (attribution) | P04 | — |
| P10 | 35 | 전이 시간 $T$를 고정하고 | Fixed transfer time, normalized parameter, degree-N Bézier trajectory. | definition | — | T, τ, N, P → everywhere |
| P11 | 42 | 여기서 $B_i^N$은 Bernstein | Basis and control points; decision vector x. | definition | P10 | x → everywhere |
| P12 | 52 | 미분 행렬 $D_N$ | Differentiation and degree-elevation matrices give degree-preserving derivative operators [17]. | definition | P10 | L1, L2 → P13, P36 |
| P13 | 58 | 물리 시간에 대한 속도·가속도 제어점은 | Velocity and acceleration control points; endpoint position and velocity conditions as a linear equality. | definition | P12 | the boundary-condition equality → P40 |
| P14 | 70 | Bernstein Gram 행렬 $G_N$의 | Gram matrix and the exact integral identity [12, 17]. | definition | P10 | the identity → P15, P36, P47, P52 |
| P15 | 82 | $F$가 결정 변수 $\mathbf{x}$의 1차 함수이면 | If the coefficients are affine in x, the integral is an exact convex quadratic with no numerical integration. | claim (tool) | P14 | → P36, P38 |

### 3.1 Avoidance constraint (section 3 has no opening paragraph)

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P16 | 90 | 곡선 분할과 분리 평면을 이용한 | The constraint follows [17]; spherical obstacle defined. | definition | P04 | obstacle center and radius → all of 3.1, 3.3 |
| P17 | 97 | 연속시간 장애물 회피 조건은 | Formal continuous-time avoidance condition; stronger than node conditions; fixed T makes parameter-wise equal to time-wise. | definition | P03 | "회피 조건" → P26, P29, … (see definitions table) |
| P18 | 99 | 곡선을 $n_{\mathrm{seg}}$개로 등분하는 | De Casteljau subdivision matrix and sub-arc control points; they are not independent variables; raising the degree adds freedom, raising the subdivision count lowers conservatism. | definition + claim | P10, P06 | sub-arc control points → P19–P25, P42; "more sub-arcs, less conservatism" → P67 |
| P19 | 107 | 분할구간 제어점의 무게중심을 | Sub-arc control-point centroid. | definition | P18 | centroid → P21, P24, P25, P42 |
| P20 | 113 | [그림 1]에서 보듯이 | After subdivision the hull hugs the curve (Figure 1). | illustration | P18 | — |
| P21 | 119 | 이때 외향 법선은 | Outward normal from the centroid; supporting half-space. | definition | P19 | normal, half-space → P22–P25, P42 |
| P22 | 133 | 각 분할구간의 모든 제어점이 이 반공간에 | Every sub-arc control point must lie in the half-space. | definition (constraint) | P21 | → P23 |
| P23 | 139 | 볼록 껍질 성질 [12]에 따른 회피 충분조건을 | The sufficient condition becomes Proposition 1 and is used "for feasibility judgment". | transition | P22 | "실현 가능성 판정" → no later paragraph uses this phrase |
| P24 | 141–150 | **명제 1.** / **가정.** / **증명.** | Proposition 1 with four assumptions and its proof. | claim + derivation | P18, P21, P22 | Proposition 1 → P25, P26, P28, P29, P49, P58, P61 |
| P25 | 152 | 분할구간 제어점 $\mathbf{q}^{(s)}_m$은 | Sub-arc points are linear in x; with a fixed normal the constraint is linear, but the normal depends on x; violation measure h. | derivation | P19, P21, P24 | h → P26, P45, P48, P50; "normal depends on x" → P42 |
| P26 | 158 | $h(\mathbf{x})$는 법선의 정규화 | h is non-convex; h = 0 gives avoidance; 3.3 linearizes at the reference point; half-spaces are rebuilt each iteration, so the subproblem constraint is local. | transition | P25 | → section 3.3 |
| P27 | 160 | 반공간의 경계가 구에 접하면 | A tangent half-space guarantees avoidance for any normal, but the normal's direction changes conservatism. | claim | P06, P21 | → P28 |
| P28 | 162–169 | 분할구간의 두 끝점 사이 거리 | Chord length; arc approximation gives the clearance estimate, which falls with the square of the subdivision count; the estimate is for comparison in section 5, avoidance is still judged by Proposition 1. | derivation (contribution 1) | P18, P27 | estimate → P58, P61, P66, P67 |
| P29 | 171 | [그림 2]는 하나의 분할구간에 적용한 | Figure 2 shows Proposition 1 on one sub-arc; the subproblem linearizes it. | illustration | P22, P24 | — |

### 3.2 Objective

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P30 | 179 | 장애물을 회피하는 데 필요한 제어 비용을 | Control acceleration = trajectory acceleration minus gravity. | definition | P10 | u → P33 |
| P31 | 185 | 여기서 $\mathbf{g}$는 중력 가속도이며 | Gravity has two-body and J2 terms; formula. | definition | — | g → P33, P34 |
| P32 | 197–199 | $\mathrm{GM}$은 지구 중력상수 | Constants named; Earth's axis along z; constant values (block moved here per R3). | definition | P31 | — |
| P33 | 201 | 목적함수는 제어 가속도 크기의 제곱을 | Objective J = integral of squared control acceleration. | definition | P30, P31 | J → P34, P38, P45, P52, P58 |
| P34 | 207 | 중력 가속도가 위치에 대해 비선형이므로 | J is non-convex; split into n_lin intervals, first-order Taylor gravity; the split matrix is named apart from the avoidance one; n_seg sets conservatism, n_lin sets objective accuracy, independent so 5.2 isolates gravity error; linearization point is the interval centroid. | derivation (contribution 2) | P18, P33 | n_lin → P52; "5.2 isolates gravity error" → 5.2 never refers back |
| P35 | 213 | 여기서 $\nabla\mathbf{g}_j^{(k)}$ | Gravity Jacobian at the reference point; the split places linearization points, it does not discretize the integral. | definition | P34 | — |
| P36 | 215–221 | 구간 $j$를 $\xi$로 다시 매개화하고 | Per-interval residual; both terms are degree-N Bézier curves whose control points are affine in x, so the Gram identity applies; summing gives J^(k). | derivation (contribution 2) | P12, P14, P15, P34 | J^(k) → P38, P41, P45, P47 |
| P37 | 227 | 여기서 $1/n_{\mathrm{lin}}$은 | The 1/n_lin factor and the residual control-point matrix. | definition | P36 | — |
| P38 | 229 | $J^{(k)}$는 $\mathbf{x}$에 대한 볼록 이차형식이므로 | J^(k) is a convex quadratic usable as the QP objective, computed exactly; its only error against J is gravity linearization. | claim (contribution 2) | P15, P36 | → P40, P41 |

### 3.3 Subproblem and SCvx

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P39 | 233 | 회피 제약과 중력의 선형화는 기준 제어점 | Both linearizations are local, so they are rebuilt every iteration inside trust-region SCvx [4, 5]; Algorithm 1. | transition | P26, P38 | → P40 |
| P40 | 235 | SCvx 반복 $k$의 부문제는 | The convex QP at iteration k (uses the penalty weight μ and trust-region size before they are described). | definition | P13, P36 | QP → P41–P50 |
| P41 | 247 | $H^{(k)}$와 $\boldsymbol{\ell}^{(k)}$는 | Objective matrix and vector; the slack vector; problem size at N = 7, n_seg = 16 (24 variables, 12 fixed, 128 avoidance constraints); constraints and slacks grow with the subdivision count. | definition + setup | P36, P40 | constraint count → P63, P68, P02, P75 |
| P42 | 249–255 | 장애물 회피 제약의 행렬 | Avoidance rows come from linearizing the 3.1 condition at the reference point; margin to the half-space boundary; centroid weights; gradient. | derivation (contribution 3) | P18, P19, P21, P25 | gradient → P43, P44, P48, P50 |
| P43 | 261 | 첫째 항은 법선을 고정했을 때의 기울기이며 | First term = fixed-normal gradient; second term = centroid and normal change; the projection keeps only tangential components, zero at the contact point. | derivation (contribution 3) | P42 | "second term" → P48, P70 |
| P44 | 264–267 | (linearized constraint) / $\nu^{(s)}_m$는 | Linearized constraint with a slack per row; slack plays the role of SCvx's virtual control [4, 5]; no feasible point exists when the initial path is inside the obstacle. | definition | P41, P42 | slack → P45, P46, P61 |
| P45 | 269–275 | 부문제에서 구한 해 / $J$는 비선형 중력을 | Acceptance uses the merit function J + μh [4, 5]; J uses nonlinear gravity, h is rebuilt at the point; the QP objective is the model of the merit function, its h-term is the slack sum. | definition | P25, P33, P36, P44 | merit function → P47, P48, P49, P70 |
| P46 | 277 | 여유 변수로 제약 위반을 허용하되 | μ large enough that converged slacks vanish [8, 9]; the trust region limits the step; its size is adjusted each iteration. | definition | P40, P44 | μ value → P57 |
| P47 | 279–285 | 예측 감소량에는 2.3절의 적분 항등식으로 | Predicted reduction via the exact identity; actual reduction adds the numerically integrated gravity-linearization error; ratio ρ_k; accept-and-resize rules in Figure 3; the first iteration is accepted without the ratio. | definition | P14, P36, P45 | ρ_k → Figure 3, P48, P50 |
| P48 | 291 | 회피 제약 기울기의 둘째 항을 빼고 | Prediction: with a fixed normal, a step can satisfy the old half-spaces yet violate the rebuilt ones, so actual reduction falls short and steps get rejected. | claim (contribution 3) | P43, P45, P47 | tested by P72 |
| P49 | 293 | merit function의 상대 변화나 예측 감소량이 | Stop when relative merit change or predicted reduction is below tolerance with Proposition 1 satisfied for three consecutive iterations; why consecutive. | definition | P24, P45, P47 | stopping rule → P58 ("종료 기준"), P61 ("정지점 조건") |
| P50 | 295–308 | **Algorithm 1** | Symbolic algorithm: counters σ1, σ2, iteration cap 1000, tolerances, trust-region floor 10⁻² km. | definition | P40–P49 | — |

### 4 Setup

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P51 | 314 | 단순화된 3차원 궤도전이 문제에서 | Simplified 3-D transfer, Earth-centered spherical obstacle 6471 km, T = 1500 s chosen so the constraint becomes active; below about 100 deg the constraint is inactive. | setup | P16 | 자유 낙하 각거리 → P62; "inactive" → P55, P62 |
| P52 | 316 | 중력 모델에는 3.2절의 이체 중력과 | Two-body plus J2, n_lin = 100, centroid linearization; on the baseline, 200 and 400 intervals change J by under 10⁻⁹. | setup + **evidence (contribution 2)** | P14, P34 | — (the result is not referred to again) |
| P53 | 318 | 기준 시나리오는 Progress 우주선의 | Baseline from Progress to ISS [18, 19]; 245 and 400 km, 51.64 deg, 120 deg, circular velocities; five scenarios. | setup | — | scenarios → Table 1, P54, P55 |
| P54 | 330 | 중심각은 지구 중심에서 본 | Central angle and velocity angle; scenarios 1–4 coplanar, 5 leaves the plane. | setup | Table 1 | — |
| P55 | 332 | 시나리오 2는 분할 수 비교 실험의 기준으로 | Role of each scenario: 2 is the subdivision baseline, 3–5 compare conservatism at fixed n_seg, 4 starts deep inside, 1 is the inactive control. | setup | Table 1 | → P61, P62, P63 |
| P56 | 334–336 | 알고리즘은 Rust로 구현하였다 / > 볼록 부문제는 | Implementation, machine, QP solver tolerances. | setup | — | — |
| P57 | 338 | 페널티 계수는 $\mu = 10^{-2}$ | μ and initial trust-region size; scenario 4 uses 4000 km because its first step exceeds 2000 km. | setup | P46 | → P69 (wide-range rule) |
| P58 | 340 | 최적해 수렴 여부는 반복이 3.3절의 종료 기준으로 | Metrics: convergence (stopped by the 3.3 criteria, which include Proposition 1), minimum clearance, estimated clearance, control cost = J, runtime. | definition | P28, P33, P49 | metrics → sections 5 |

### 5.1 Scenarios

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P59 | 348 | 다섯 시나리오의 궤적과 정량 결과를 | Where the results are (Figure 4, Table 2). | transition | — | — |
| P60 | 364 | > 표 2와 그림 5의 계산 시간은 | Runtimes are the minimum of 15 runs. | setup (note) | — | — |
| P61 | 366 | 다섯 시나리오 모두 명제 1의 회피 조건을 | All five satisfy Proposition 1 and stop by "the 3.3 stationarity condition"; constraint active in 2–5, clearance 96–99 % of the estimate; same out of plane; scenario 4 penetrates about 5900 km initially but slack recovers it. | evidence (contribution 1, convergence) | P24, P28, P49, P55 | → P02, P75 |
| P62 | 368 | 시나리오 1에서는 요구 중심각이 | Scenario 1: inactive constraint, closest point at departure, n_seg 8/16/32 give the same solution, cost not monotone in central angle. | evidence (control case) | P51, P55 | — (no claim or conclusion uses it) |

### 5.2 Subdivision count

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P63 | 372 | 기준 시나리오에서 차수를 $N=7$로 고정하고 | Design: N = 7, n_seg 4–64; expected trade-off; 1024 perturbed baseline cases by Latin hypercube, each solved at every n_seg. | setup | P18, P41 | Table 3, Table 4 |
| P64 | 385 | > 궤도상 각 간격은 | Defines the in-orbit angle spacing; uniform sampling. | definition (note) | Table 3 | — |
| P65 | 387 | 분할 수별 결과의 분포를 | Results in Table 4 and Figure 5; all converged; n_seg = 4 excluded from the conservatism comparison because the minimum sits at departure. | evidence + exclusion rule | P58 | → P67 |
| P66 | 399 | > 각 값은 중앙값이며 | Statistics; the clearance ratio uses only cases where the constraint is active, the closest point is interior, and its sub-arc "touches the supporting plane"; cost increase relative to n_seg = 64. | definition (note) | P28 | → P67 |
| P67 | 401 | 최소 이격거리는 각 사례에서 분할 수를 두 배로 | Clearance quarters per doubling, median log slope −1.99; ratio about 0.97 at every n_seg, none above the estimate by more than 0.2 %; cases off the assumption explained. | evidence (contribution 1) | P28, P66 | → P02, P75 |
| P68 | 407 | 그림 5의 상자는 사분위 범위를 | Box-plot reading; cost falls with diminishing returns, n_seg 16 is 2.6 % above 64; median iterations 8; constraint count is linear in n_seg, so baseline runtime grows. | evidence (trade-off) | P41 | → P02, P75 |
| P69 | 409 | 기준 주변보다 넓은 표 3의 범위에서도 | 4096 wide-range cases; trust-region rule; only 58 non-converged, all at n_seg = 4; failure modes; 1000-point sampling found no intrusion. | evidence (robustness, guarantee) | P24, P57 | → P75 |

### 5.3 Normal term

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P70 | 413 | 회피 제약의 선형화에 법선 변화를 포함한 효과를 | Ablation design: same 1024 cases with the normal fixed; everything else identical. | setup | P43, P63 | Table 5 |
| P71 | 425 | > 반복 횟수와 제어 비용 증가율은 | Table 5 statistics; cost comparison only where both solutions satisfy avoidance. | definition (note) | — | — |
| P72 | 427 | 법선 변화를 포함하면 5.2절에서와 같이 | With the normal term all converge; fixed normal mostly stops at the trust-region floor with more iterations; on the five scenarios, rejected steps satisfied the old half-spaces but violated the rebuilt ones, as P48 predicted. | evidence (contribution 3) | P48, P65 | → P02, P75 |
| P73 | 429 | 분할 수 8 이상에서는 법선을 고정해도 | Fixed normal at n_seg ≥ 8 ends feasible but early and costlier; at 4 mostly infeasible. | evidence (contribution 3) | P72 | — |

### 6 Conclusion

| P | L | Opens with | Point | Role | Needs | Hands on |
|---|---|---|---|---|---|---|
| P74 | 435 | 본 연구에서는 분할구간별 지지 반공간 제약을 | Method summary (half-space per sub-arc, piecewise gravity quadratic, normal-aware linearization, SCvx). | summary | — | — |
| P75 | 437 | 수치 실험에서 장애물 경계 부근의 | Clearance matches the arc estimate across perturbed cases, so conservatism is predictable from chord length; few sub-arcs cost much and fail on the wide range; more sub-arcs cut cost quickly while constraints grow linearly; the normal term is needed for convergence. | summary (results) | P61, P67, P68, P69, P72 | — |
| P76 | 439 | 본 연구는 구형 장애물과 고정 전이 시간의 | Limitations and future work. | limitation | — | — |

## 3. Claims table

Each claim the paper makes in the abstract or introduction, traced to where it is built, where it is
shown, and where the conclusion states it. A blank cell is a break in the chain.

| Claim | Promised | Built | Shown | Concluded |
|---|---|---|---|---|
| **Contribution 1.** The clearance near the obstacle follows the arc estimate and falls with the square of the subdivision count. | P07, P02 | P28 (P18, P27) | P61 + Table 2 (five scenarios at n_seg 16); P67 + Table 4 + Figure 5(a) (1024 cases) | P75 |
| **Contribution 2.** Two-body plus J2 gravity, linearized per interval, gives an exact convex quadratic cost; the result barely changes with more intervals. | P07, P01 | P34–P38 | P52, one sentence in the **setup** section, baseline scenario only | P74 restates the method; **no result sentence** in P75 or P02 |
| **Contribution 3.** Including the normal's change in the linearization is needed for convergence. | P07, P01, P02 | P42–P43, prediction in P48 | P72–P73 + Table 5 | P75 |
| Whole-curve avoidance is guaranteed when the condition holds. | P01, P06 | P24 (Proposition 1) | P69 (1000-point sampling, no intrusion); P61 (all five satisfy Proposition 1) | P74 |
| Every case converged. | P02 | P49 (stopping rule) | P61, P65, P69 (wide range: 58 failures, all at n_seg 4) | P75 (wide-range failures mentioned) |
| More sub-arcs cut cost with diminishing returns while constraint count grows linearly. | P02 (**not** among the P07 contributions) | P41 (constraint count) | P68 + Table 4 + Figure 5(b, c) | P75 |
| A deep initial penetration is recovered by the slack. | — | P44 | P61 (scenario 4) | — |
| With the constraint inactive, the subdivision count does not change the solution. | — | — | P62 (scenario 1) | — |

## 4. Definitions table

Terms and symbols used before they are defined, or used under more than one name. Everything not
listed is defined at or before its first use in the body (abstract uses are not counted).

| Term or symbol | First body use | Defined | Note |
|---|---|---|---|
| 부문제 (subproblem) | P26 L158, P29 L171 | P40 L235 | Two uses in 3.1 before 3.3 defines it. |
| 기준점 (reference point) | P26 L158, P35 L213 | P42 L249 ("기준점 $\mathbf{x}^{(k)}$") | Used in 3.1 and 3.2 before it is tied to the iterate. |
| penalty weight `\mu` | P40 L238 (in the QP) | P46 L277 | 39 lines between use and description. |
| trust-region size `\Delta_k` | P40 L244 (in the QP) | P46 L277 | Same gap. |
| counters `\sigma_1`, `\sigma_2` | P50 L297 | — | Appear only inside Algorithm 1; the text never says what they count. |
| 정지점 조건 (stationarity condition) | P61 L366 | — | The stopping rule is called 종료 기준 in P58 and described without this name in P49. Two names, one undefined. |
| 회피 조건 (avoidance condition) | P17 L97 | P17 defines the **exact** continuous-time condition | Later uses mean Proposition 1's **sufficient** condition: qualified in P29, P49, P58, P61, unqualified in P26 L158, P45 L275, P73 L429. |
| 지지평면 (supporting plane) | P28 L162 | — | Implicitly the boundary of the half-space; P66's selection rule ("touches the supporting plane") depends on it. |
| 작동/비활성 (constraint active/inactive) | P51 L314 | — | Used as a scenario property and a selection rule (P55, P62, P66) without a definition. |
| 실현 가능성 판정 (feasibility judgment) | P23 L139 | — | Proposition 1 is announced for this use; no later paragraph uses or names it (the metric is 최적해 수렴 여부). |

## 5. Noticed while mapping (input for phase 2; nothing changed)

1. **Reference [19] carries your own note**, `(...)[[this url has to render as url. ...]]`. The "("
   opens the note's span; it is not a renumbering leftover. The note entered with `50382b2`.
   (Corrected in phase 2. The first version of this item misread the "(" as a stray from `29cc993`.)
2. **Contribution 2's evidence is in the setup section** (P52), on one scenario, and the conclusion
   has no result sentence for it. Contribution 2 is the only contribution without a results-section
   paragraph.
3. **The abstract and conclusion lead with the cost and constraint-count trade-off**, which the
   introduction does not list as a contribution.
4. **Two results connect to no claim:** scenario 1's inactive case (P62) and scenario 4's recovery
   (in P61).
5. **Hand-ons that nobody picks up:** P23's "feasibility judgment" and P34's "5.2 isolates gravity
   error" (5.2 never mentions gravity).
6. **Figure placement.** Figure 1 (P20) sits between the centroid (P19) and the normal that uses it
   (P21). Figure 2 (P29) illustrates the constraint of P22–P24 but comes after the conservatism
   estimate (P28).
7. **P55 assigns roles to the scenarios.** Its scenario-2 sentence is accurate, because the 5.2 Monte
   Carlo is centred on scenario 2. The roles given to scenarios 1 and 4 support no claim (phase 2,
   7.5 item 9). (Corrected in phase 2. The first version called the scenario-2 sentence stale.)
8. **P18's remark on raising the degree** stands alone now that the degree experiment is cut
   (`64b44c0`).

## 6. Checker state at the snapshot

- `check_advisor_rules.py`: 0 errors, 2 warnings (the parentheses in the Table 4 and Table 5 titles).
- `check_notation.py`: no violations.

## 7. Phase 2: diagnosis

### 7.1 How the feedback history was mapped

- **Marks.** All 186 marks in the three transcripts:
  - R1, `paper_draft_korean_PF.md`: 94;
  - R2, `paper_draft_korean_rev2_PF.md`: 31;
  - R3, `paper_draft_feedback_applied_11c772e_PF.md`: 61.

  A mark is one `[[...]]` comment or sign, one `{...}` replacement or insertion, or one `~~...~~`
  strike. The number extracted from each file equals that file's raw count of mark tokens. That check
  caught two marks the first parser missed: a comment inside a display equation (R1) and a
  replacement whose text is math (R3).
- **Unit.** Each mark goes to the current unit holding what the marked text became, and moved text
  follows the move. For example, the origin sentence that R1 and R3 marked in 3.1 is now in P51.
  Figures and tables are units of their own (F1–F5, T1–T5). Marks on content that was later cut are
  **GONE**.
- **Assignment.**
  - A script made two proposals for each mark by character-trigram overlap: one matching the
    reviewed paragraph, one matching the reviewed sentence. I read every mark against them.
  - 105 of the 142 placed marks equal a proposal. The other 37 were decided by reading, mostly in
    R1, whose text has since been rewritten.
  - A second assignment, made independently in an earlier session, agrees on 134 of the 167 marks
    the two share. 23 of the differences are convention (figures as their own units, moved text
    placed at its destination). Recomputing 7.3 with the other run's choice for the 10 real
    differences gives the same 20 units.
- **Class.**
  - **D**: definition or clarity.
  - **O**: organization: merge, move, delete, list to prose, level of the abstract or conclusion.
  - **A**: argument or evidence: citation, justification, experiment.
  - **S**: sentence: wording, 번역투, a sentence struck as unneeded, a pointer sentence.
  - **F**: figure or table.
  - **T**: mechanical: term swap, English gloss, parenthesized section reference, parentheses,
    bold, unit.

  Every class except T counts as substantive.
- **Status.**
  - Values: resolved, open, partial, gone, unverified.
  - Strikes and glosses marked resolved were checked by script: none of their spans remain in the
    draft's prose. The script's four hits were all false: an anchor id, and section numbers that
    remain without their parentheses.
  - Figure marks were checked by viewing the five figure files.
- The full per-mark ledger is Appendix A.

### 7.2 What the history shows

- **Cut, not repaired.** 42 marks target content that was cut rather than repaired. 41 are from R1:
  the downstream experiment, the surrogate objective, the warm-start framing, the proximal term and
  the degree experiment. One is from R2: the degree section. R3's deletion of the section 3 opener
  is counted as resolved, because deletion was what that mark asked for.
- **Still open.** Of the other 144 marks, 138 are resolved. The rest:
  - Open or partial, all from R1:
    - the Gram paragraphs in 2.3 (R1-29, R1-30, R1-33);
    - the name 가속도 연산자 (R1-31);
    - the text describing Figure 2 (R1-42).
  - Unverified: Figure 1's font size (R3-19).
- **The complaints changed character.**
  - R1 was mostly D: 38 of 94 marks were about undefined terms.
  - R3 was mostly T (23 of 61) and S (16 of 61).

  The draft became easy to follow. What remains is the logic inside paragraphs, which word-level
  fixes do not reach.

### 7.3 Units marked in two or more rounds

| Unit | Substantive marks | What was asked | Now |
|---|---|---|---|
| **P05** non-convexity, two approaches | R1 O O; R2 S; R3 O; plus your own note in `50382b2` on [10] and the "있다." junction | Merge the related-work section into the introduction, drop the rhetoric, join two paragraphs. | **Open.** Each round moved or trimmed text. The paragraph still lists two classifications without relating them or saying where SCvx enters: lossless convexification vs. SCvx, and covering the free space vs. half-spaces. |
| **P07** contributions | R1 O O; R2 S S | Merge items 2 and 3, prose instead of a list, 세운다. | Wording resolved. **Open at the claim level**: contribution 2's promised check has no results paragraph (7.5 item 1). |
| **P15** exact quadratic | R1 D; R3 S | Why a quadratic matters for the formulation. In R3, a struck pointer to 3.2. | **Open.** R1's request was answered with a pointer sentence to 3.2, which R3 struck. Section 2.3 is back to giving no reason. P14 shares the problem (R1-29 and R1-30 open). |
| **P26** h is non-convex; the linearization is local | R2 S; R3 S | 번역투 in the h sentence; "국소 근사이다" → "국소 근사에 의존한다". | **Open.** Both fixes swapped words. The paragraph still ends with a pointer to 3.3 that uses 부문제 and 기준점 before 3.3 defines them, and it repeats a claim made three more times (7.4). |
| P06 convex hull, one half-space per sub-arc | R2 S S S S D; R3 S | 번역투, 유한 개, define 보수성 earlier. | Rewritten structurally in `a963f31`. The professor has not seen it yet. |
| P03 motivation | R1 A D D O O; R2 S | Citation, what 안전 means, merge 2.1. | Resolved. Its link to P04 is weak (7.4). |
| P04 [17] | R2 A; R3 S | Cite [17]; "[17]에서는". | Resolved. Its position is a phase 3 question (7.5 item 3). |
| P17 continuous-time condition | R2 S; R3 S | Insert 곡선상의; strike one sentence. | Resolved. Both were one-word or one-sentence fixes. |
| P50 Algorithm 1 | R1 O; R3 O | Algorithm format, then symbols instead of Korean. | Resolved. The counters σ1 and σ2 are still unexplained in the text. |
| P51 setup (origin sentence) | R1 D; R3 O | What "원점 중심" means; move it to section 4. | Resolved: now 지구 중심, in section 4. |
| P57 penalty weight and initial trust region | R2 S; R3 D A S | 인쇄한 자릿수; 쌍대변수 undefined; why the initial trust region differs; trust-region know-how not needed. | Resolved. Since 쌍대변수 was cut, μ = 10⁻² has no stated basis. P69 brings back a trust-region rule, which risks the same mark again. |
| P58 metrics | R2 D; R3 D S | The two L_seg definitions disagree; is 제어 비용 the same as J? | Resolved. The stopping rule it names, 종료 기준, is called 정지점 조건 in P61. |
| P63 subdivision experiment design | R1 D D D S S S S; R3 O A | Vague design text; absorb 4.2; Monte Carlo. | Resolved: rewritten, and the Monte Carlo study was run. |
| P75 conclusion, results | R1 O; R3 S | No numbers; insert 수치. | Resolved. It has no result for contribution 2. |
| F1 | R1 F; R3 F F | Add a figure; fonts; the centroid is missing. | Resolved: centroids appear in both panels. The font size can't be checked from the image file. |
| F2 | R1 F; R2 F | The text doesn't describe the figure; symbol styles. | Symbols resolved. **Partial**: P29 doesn't describe panels (a)–(c), and the figure comes after P28 (7.4). |
| F3 | R2 F; R3 F | Make it compact and fix the font; flow in one direction. | Resolved: it flows left to right. |
| F4 | R1 F F; R2 F; R3 F | Legend and the Δv label; black-and-white line styles; scenario numbers. | Resolved: five line styles and a numbered legend. |
| F5 | R1 F; R2 F | Mark the n_seg = 2 case; what the hollow and filled markers mean. | Resolved: the figure was replaced. |
| T2 | R1 A; R3 F F | Iterations at the cap; drop a column and the †. | Resolved. |

**Reading.** 14 of these 20 units are paragraphs. Four of them still have the problem their marks
were circling: P05, P07, P15 and P26. In each, the earlier fixes were local (a move, a trim, a word
swap, a pointer). The defect is in what the paragraph is for.

### 7.4 Paragraph-job check of the current draft

The six symptoms from the plan were checked against all 76 paragraphs. Two tools helped:

- a detector for similar sentences in different paragraphs (trigram overlap of at least 0.35);
- a list of every sentence that names a section, figure or table.

Everything else was found by reading.

| Symptom | Where | Evidence |
|---|---|---|
| Pointer sentence | P26 L158 | "3.3절에서는 기준점에서 회피 조건을 1차 전개하여 볼록 부문제를 구성한다." The checker misses it because 구성한다 is not in its verb list. |
| | P29 L171 | The whole paragraph is a pointer to Figure 2 plus a repeat of the linearization claim. |
| | P34 L207 | "5.2절의 분할 수 실험에서 중력 근사 오차의 영향을 분리한다." Section 5.2 never comes back to this. |
| | P23 L139 | Announces Proposition 1 for "실현 가능성 판정"; no later paragraph does that. |
| Same claim twice | P26, P29, P39, P42 | "The avoidance constraint is linearized at the current solution" appears four times. The first two come before 3.3 defines 부문제 and 기준점. |
| | P74 and P01 | The conclusion's method summary nearly copies the abstract (sentence overlap 0.74–0.75). |
| | P51 and P62 | The inactive-constraint explanation appears in the setup and again in the results (0.51). |
| | P57 and P69 | The initial trust region is explained for scenario 4, then again as a doubling rule (0.58). |
| | P66 and P67 | The table note's selection rule is restated in the body (0.38). |
| | P18 and P63 | "More sub-arcs, less conservatism" is stated in 3.1 before its mechanism (P20, P27–P28) is given, and again as the expectation in 5.2. |
| Reason after the claim | P18 L105 | The claim 분할 수를 늘리면 보수성을 줄일 수 있다 comes before its reason. Its half about raising the degree no longer connects to anything. |
| | P44 L267 | The slack is named as SCvx's virtual control first, and only then is the reason given (the initial path may run inside the obstacle). |
| Broken link to the previous paragraph | P21 L119 | "이때 외향 법선은" refers to the centroid in P19, but Figure 1 (P20) sits between them. |
| | P04 to P05 | P04 introduces the specific baseline [17]; P05 then steps back to general non-convexity. P03's need, a continuous-time guarantee, is met by the convex hull only in P06. |
| Stacked nouns | P75 L437 | "법선 변화의 선형화는 반복의 수렴에 필요하였다." The only instance found. |
| Citations | whole draft | Every reference is cited, but the numbers are not in order of first citation: 1, 2, 6, 7, 12, 13, 17, 3, 4, 5, 10, 11, 14–16, 8, 9, 18, 19. Check the journal's rule; the repo has no author guide. |
| Inconsistency | P24 and P27 | Proposition 1 lists assumption 2 (normal built from the centroid) as a condition. The proof doesn't use it, and P27 says the guarantee holds for any normal. |
| | P68 and Figure 5(c) | The text says "계산 시간은 분할 수에 따라 증가한다", but 5(c) shows 0.101 s at 4 sub-arcs and 0.094 s at 8. |
| | P36 and P12 | P36's 가속도 연산자 is the operator P12 calls 차수를 보존하는 미분 연산자: one operator, two names. |
| | P26, P35, P39, P42 | 기준점, 기준 제어점 and 현재 해 all name the same point. |

The definitions table (section 4) lists the remaining use-before-definition cases.

**Checkers.**

- `check_advisor_rules.py`: 0 errors and 2 warnings (the Table 4 and 5 titles); its self-test
  passes all 44 cases.
- `check_notation.py`: clean, and its self-test passes.

### 7.5 Ranked problems

Ranking rule:

1. it breaks the claims chain;
2. it was marked in two or more rounds and still shows the symptom;
3. it blocks the reader (definitions, names);
4. it was marked in one round, or was newly found;
5. it is small.

Within a rank, problems earlier in the paper come first, because later paragraphs depend on them.

1. **Contribution 2 has no evidence in the results.** P07 promises "선형화 구간 수를 늘려도
   목적함수가 거의 변하지 않음을 확인한다". The only support is one sentence in the setup (P52), on
   the baseline scenario only, and P02 and P75 report nothing. *Decision:* move the check into
   section 5 and say what it shows, or drop the promise.
2. **The headline trade-off is not promised.** P02 and P75 lead with cost falling quickly while the
   constraint count grows linearly, but P07 lists no such contribution. *Decision:* promise it in
   P07, or demote it in the abstract and conclusion.
3. **The introduction's logic (P03–P07).** P05 was marked in all three rounds and in your own note.
   It does not relate its two classifications or say where SCvx enters. P04's specific baseline
   sits between the need (P03) and the general background (P05).
   - A candidate order for phase 3: P03 → P05 → P06 → P04 → P07.
   - P05 or P06 would then say that the half-space constraint stays non-convex because the normal
     depends on the control points, which is why SCvx is used.

   *Decision:* the order.
4. **Section 2.3 gives no reason for the Gram matrix (P14–P15).** R1 marked it twice and R3 once.
   The fix has to be a sentence of content inside 2.3, not a pointer: the 3.2 objective is the
   integral of a squared curve, and this identity makes it an exact quadratic in x.
5. **The linearization claim appears four times (P26, P29, P39, P42).** Keep it where 3.3 actually
   does the linearization (P39, P42). P26 can end at "h = 0 guarantees avoidance", and P29 can keep
   only what Figure 2 shows. That also removes the use of 부문제 and 기준점 in 3.1 before they are
   defined.
6. **Names, and symbols used early.**
   - The stopping rule has two names: 종료 기준 and 정지점 조건.
   - 회피 조건 names both the exact condition and the sufficient condition.
   - 지지평면 and 작동/비활성 are never defined.
   - 기준점, 기준 제어점 and 현재 해 name the same point.
   - 가속도 연산자 vs. 차수를 보존하는 미분 연산자.
   - σ1 and σ2 are never explained.
   - μ and Δ_k are used about 35 lines before P46 describes them.
7. **Figures 1 and 2.** Figure 1 separates the centroid from the normal. Figure 2 comes after the
   estimate, and the text does not describe its panels (R1-42, partial).
8. **Assumption 2 vs. P27.** *Question:* is building the normal from the centroid a condition of
   Proposition 1, or a construction choice? If it is a choice, it does not belong among the
   proposition's assumptions.
9. **Results that back no claim.** Scenario 1 (P62) and scenario 4's recovery (P61) support no
   claim, yet P55 gives both scenarios a role. P51 and P62 explain the inactive case twice.
   *Decision:* tie each result to a claim, or cut it.
10. **Repeats and loose ends.**
    - P74 nearly copies P01 (rule 22 asks for a stronger conclusion).
    - P66 and P67 state the selection rule twice.
    - P57 and P69 both explain trust-region sizing, which R3 said isn't needed.
    - P18's remark about raising the degree.
    - P34's promise about 5.2.
    - P23's "실현 가능성 판정".
    - P44 gives the reason after the claim.
    - P68's runtime sentence contradicts Figure 5(c).
    - P75's stacked nouns.
11. **Small items.**
    - Your note on [19] is still open: a bare web address as a reference, which should at least
      render as a link.
    - The citation numbers are not in order of first citation (check the journal's rule).
    - μ = 10⁻² has no stated basis since 쌍대변수 was cut.
    - Figure 1's font size is unverified.
    - The checker warns on the Table 4 and 5 titles.
    - The checker misses 구성한다 pointer sentences. Fixing that is a dotfiles change with a
      self-test case; it has not been made.

## 8. Phase 3: decisions

Each decision names the 7.5 item it settles and what phase 4 will change. Nothing in the draft
changes until phase 4.

| # | Item | Decision (date) | Phase 4 change |
|---|---|---|---|
| D1 | 7.5 item 1, contribution 2's evidence | **A: contribution 2 is a formulation contribution** (2026-10-08). The n_lin check justifies the setting of 100. It is not a finding. | P07: drop "선형화 구간 수를 늘려도 목적함수가 거의 변하지 않음을 확인한다". P52 stays in section 4 as the reason for n_lin = 100. Claims-table row becomes: promised P07, built P34–P38, used by every section 5 experiment, concluded P74. Revisit under item 9 whether scenario 1's cost pattern is physical evidence for it. |
| D2 | 7.5 item 2, the unpromised trade-off | **A: fold it into contribution 1** (2026-10-08). The subdivision count is one setting with three effects: conservatism and cost fall, and the subproblem grows. | P07: contribution 1 covers conservatism, cost and subproblem size against the subdivision count. P02 and P75 keep their order, but state the constraint-count growth as a consequence of the construction, not as a measured result; the measured side is cost (Table 4, Figure 5(b)) and baseline runtime (Figure 5(c)). |
| D3 | 7.5 item 3, introduction order and P05 | **Split P04; P05 relates its pieces** (2026-10-09). | P04's first sentence (control points as variables [12, 13]) stays after P03. Its [17] sentence opens P07. P05 stays one paragraph (R3-07 joined it) and says that a fixed half-space gives a linear constraint, but the half-space moves with the trajectory, so it is re-linearized every iteration (SCvx); lossless convexification [3] is the other route, limited to specific problems. Supports §0 rule 3 (R1, A2) and your note in `50382b2`. |
| D4 | 7.5 item 4, why the Gram matrix | **Add the reason inside 2.3** (2026-10-09). | P14 says what the Gram matrix is (inner products of the Bernstein basis polynomials) and why it is there (an integral of a squared curve, such as the acceleration cost, becomes an exact quadratic in the control points). No section pointer (rule 17), no term defined later (rule 1). Answers R1-30 and R1-33 (rule 6). |
| D5 | 7.5 item 5, the linearization claim four times | **Keep it in 3.3 only** (2026-10-09). | P26: delete the pointer sentence (rule 17). Move its last sentence, in the professor's R3 wording "국소 근사에 의존한다" (rule 20), into P39, where 부문제 is defined. P29: delete "실제 부문제에서는 이 조건을 현재 해에서 1차 전개한다". The professor kept that sentence in R3 (he struck only its "(3.3절)"), but after D7 it would use 부문제 before its definition (rule 1); its content stays in P39 and P42. The change log names both. |
| D6 | 7.5 item 6, names and early symbols | **One name per thing, chosen from words the paper already uses** (2026-10-09). | Stopping rule: 종료 기준 everywhere (P61's 정지점 조건 changes); P49 introduces the name. 회피 조건: "연속시간 회피 조건" for P17's exact condition, "명제 1의 조건" for the sufficient one; qualify P26, P45 and P73. 지지평면: define it in P21 as the half-space's boundary plane; P43's 접평면 becomes 지지평면. 작동: define it at first use (P51): at least one half-space constraint holds with equality at the solution (the `_active` test in `tools/build_monte_carlo.py`); 비활성 stays as its stated opposite. 기준점: kept for x^(k), defined where 3.3 first uses it (P39's 기준 제어점 changes). P35's 기준점 for the gravity point becomes the paper's own "구간 무게중심 r_j^(k)", not a new name. P12 names L₁ and L₂ 속도·가속도 연산자. P49 ties σ₁ and σ₂ to their two tests. P41 names μ and Δ_k in one clause (rule 1, as with R3-34's ν). How they are set stays in P46, after the merit function, as R3-33 asked. |
| D7 | 7.5 item 7, Figures 1 and 2 | **Figure 1 stays; Figure 2 moves** (2026-10-09). | P21 opens by naming the centroid instead of "이때". Figure 2 and P29 move to just after Proposition 1, and P29 walks through panels (a)–(c) (R1-42, rule 44). |
| D8 | 7.5 item 8, assumption 2 vs P27 | **Follow the design freeze** (2026-10-09). | Proposition 1 keeps: spherical obstacle; one supporting half-space tangent to it; every control point of the sub-arc inside it (`design_freeze.md` §8 list). Assumptions 2 (centroid normal) and 4 (c ≠ c_KOZ) move into the construction text in P21 (freeze §9: the centroid rule picks the witness and is not part of the constraint). In the same commit, the freeze's "Assumption 3 of Proposition 1" becomes assumption 2. |
| D9 | 7.5 item 9, orphan results | **Keep both and tie each to a claim** (2026-10-09). | Scenario 1 shows contribution 1's scope: the subdivision count changes the solution only when the constraint is active. P62 drops its repeat of P51's free-fall explanation and keeps the cost sentence that explains Table 2. Scenario 4 shows the slack's purpose stated in P44; one clause in P61. P55's roles then match. |
| D10 | 7.5 item 10, repeats and loose ends | **As listed** (2026-10-09). | P74: stop copying P01 (rule 22, R1 and A2). P18: delete the degree remark and its claim; P20 states the claim with its reason. P34's 5.2 promise becomes content in P63 (n_lin is fixed, so cost differences across n_seg come from conservatism); P34 keeps its reason for separating the counts. P23: drop "이를 실현 가능성 판정에 사용한다". P44: reason before the slack's name. P68: Figure 5(c) in words, with no new numbers (rule 30, A2): runtime grows from 8 sub-arcs up, and 4 takes longer than 8 because it needs more iterations (Table 4). P66 keeps the rule and P67 only the reason. P57 and P69 unchanged (R3-45). P75: unstack the nouns (rule 14). |
| D11 | 7.5 item 11, small items | **As listed** (2026-10-09). | [18] and [19]: print the full addresses, answering your note. Renumber citations in first-citation order after D3, once the journal guide is checked. μ = 10⁻² unchanged. Figure 1 font checked in phase 5. Table 4 and 5 title parentheses unchanged: R3 left the same pattern in the old Tables 2 and 3 unmarked. Add 구성한다 to the checker's pointer rule, with a self-test case. Confirm whether Table 4's cost column counts only active cases (`tools/build_monte_carlo.py` near line 534); if so, the table note says so. |

**Check against the advisor rules and your preferences (2026-10-09).** Every row above was checked
against the 46 rules in `korean-prose` §0 (R1, A2, R2, R3), the transcripts, `CLAUDE.md`, the
design freeze and your saved preferences. Four recommendations changed:

- **D5:** the sentence the professor reworded in R3 moves to P39 instead of being deleted, so his
  wording survives.
- **D6:** "선형화 점" was a new name. It became the paper's existing "구간 무게중심", following
  your rule never to relabel the paper's terms.
- **D10:** the runtime fix adds no numbers to the body, following rule 30 (A2).
- **D8:** the freeze's reference to assumption 3 is updated in the same commit, so the two documents
  stay consistent.

## Appendix A. Mark ledger

One row per mark. **Line** is the line in that round's transcript (`papers/feedback/*_PF.md`). Unit,
class and status are defined in 7.1.

<details><summary>186 marks</summary>

| Mark | Line | Unit | Class | Status | Note |
|---|---:|---|---|---|---|
| R1-01 | 6 | P01 | O | resolved | abstract less detailed |
| R1-02 | 8 | P01 | S | resolved | prof's model sentence for the method |
| R1-03 | 12 | GONE | D | gone | downstream terms |
| R1-04 | 12 | GONE | D | gone | downstream terms |
| R1-05 | 12 | GONE | D | gone | downstream terms |
| R1-06 | 12 | GONE | D | gone | downstream terms |
| R1-07 | 12 | GONE | D | gone | downstream terms |
| R1-08 | 12 | P02 | O | resolved | no numbers in abstract |
| R1-09 | 18 | P03 | A | resolved | [1, 2] added |
| R1-10 | 18 | P03 | D | resolved | 안전 replaced by constraint violation wording |
| R1-11 | 18 | P03 | D | resolved | same |
| R1-12 | 20 | GONE | A | gone | initialization framing cut |
| R1-13 | 28 | P07 | O | resolved | contribution list rebuilt |
| R1-14 | 30 | GONE | D | gone | Pass 1 |
| R1-15 | 30 | GONE | S | gone | replacement-possibility wording |
| R1-16 | 30 | P07 | O | resolved | list to prose |
| R1-17 | 36 | GLOBAL | T | resolved | English overuse; checker rule 7 |
| R1-18 | 40 | P03 | O | resolved | related work 2.1 merged into intro |
| R1-19 | 40 | P03 | O | resolved | same |
| R1-20 | 44 | GONE | S | gone | 이러한 지점 |
| R1-21 | 44 | GONE | T | gone | downstream |
| R1-22 | 46 | P05 | O | resolved | related work 2.2 merged into intro |
| R1-23 | 46 | P05 | O | resolved | same |
| R1-24 | 52 | GONE | O | gone | warm-start subsection cut |
| R1-25 | 52 | GONE | O | gone | same |
| R1-26 | 54 | GONE | A | gone | initialization practice |
| R1-27 | 54 | GONE | D | gone | 초기화는 연속 안전 |
| R1-28 | 56 | GONE | D | gone | 기하학적인 안전 제약 |
| R1-29 | 137 | P14 | O | open | 2.3 still opens with the formula and no reason for it |
| R1-30 | 139 | P14 | D | open | Gram matrix: what it is and why it appears is still not said |
| R1-31 | 145 | P36 | D | partial | 가속도 연산자 is used in P36; P12 names it only 차수를 보존하는 미분 연산자 |
| R1-32 | 157 | P15 | T | resolved | quadratic form -> 이차형식 |
| R1-33 | 157 | P15 | D | partial | why a quadratic matters for the formulation appears only in P38 |
| R1-34 | 171 | P51 | T | resolved | 현재 실험 (origin sentence, now in P51) |
| R1-35 | 171 | P51 | T | resolved | same |
| R1-36 | 171 | P51 | D | resolved | 원점 중심 -> 지구 중심 |
| R1-37 | 173 | P18 | T | resolved | 제어다각형 |
| R1-38 | 173 | P18 | T | resolved | same |
| R1-39 | 173 | F1 | F | resolved | Figure 1 added |
| R1-40 | 179 | P19 | T | resolved | 제어다각형 (centroid sentence) |
| R1-41 | 179 | P19 | T | resolved | same |
| R1-42 | 223 | F2 | F | partial | text still does not walk through panels (a)-(c) |
| R1-43 | 227 | P30 | T | resolved | 논문 -> 연구 |
| R1-44 | 227 | P30 | T | resolved | same |
| R1-45 | 227 | P30 | D | resolved | linearization explained (P34) before use |
| R1-46 | 227 | P30 | S | resolved | u(t) defined |
| R1-47 | 233 | P31 | D | resolved | linearization explained (P34) before use; g paragraph |
| R1-48 | 241 | P36 | D | resolved | sampled residual became the per-interval residual, now explained |
| R1-49 | 243 | GONE | A | gone | IRLS |
| R1-50 | 249 | GONE | D | gone | 연속 안전 |
| R1-51 | 262 | P40 | T | resolved | bold ell |
| R1-52 | 267 | GONE | A | gone | proximal term |
| R1-53 | 289 | P50 | O | resolved | Algorithm 1 |
| R1-54 | 293 | F3 | T | resolved | 제어다각형 in old caption |
| R1-55 | 303 | P53 | D | resolved | chaser/target replaced by Progress/ISS origin |
| R1-56 | 313 | P63 | D | resolved | 안전 근사 |
| R1-57 | 313 | P63 | T | resolved | (tradeoff) |
| R1-58 | 313 | P63 | D | resolved | 어떤 변화 |
| R1-59 | 313 | P63 | S | resolved | 불리한 |
| R1-60 | 313 | P63 | S | resolved | 즉 |
| R1-61 | 313 | P63 | D | resolved | 이 효과 |
| R1-62 | 313 | P63 | S | resolved | 가장 |
| R1-63 | 315 | GONE | D | gone | degree trend |
| R1-64 | 315 | P63 | S | resolved | experiment opener rewritten |
| R1-65 | 317 | GONE | D | gone | downstream |
| R1-66 | 319 | GONE | D | gone | downstream |
| R1-67 | 319 | GONE | D | gone | downstream |
| R1-68 | 326 | GONE | D | gone | downstream |
| R1-69 | 332 | GONE | S | gone | results opener in question form |
| R1-70 | 332 | GONE | D | gone | 어떤 |
| R1-71 | 332 | GONE | D | gone | 어떤 |
| R1-72 | 340 | F4 | F | resolved | caption replaced |
| R1-73 | 340 | F4 | F | resolved | legend, Δv label gone |
| R1-74 | 348 | T2 | A | resolved | iterations 6-12, stopping rule stated |
| R1-75 | 350 | P61 | S | resolved | phrase gone |
| R1-76 | 350 | GONE | D | gone | Δv proxy |
| R1-77 | 352 | P49 | A | resolved | tolerances in Algorithm 1 and P56 |
| R1-78 | 375 | F5 | F | resolved | n_seg 2 gone, n_seg 4 exclusion stated |
| R1-79 | 399 | GONE | D | gone | downstream |
| R1-80 | 399 | GONE | S | gone | question form |
| R1-81 | 399 | GONE | S | gone | downstream |
| R1-82 | 399 | GONE | D | gone | downstream |
| R1-83 | 401 | GONE | D | gone | circular cases |
| R1-84 | 411 | GONE | D | gone | downstream cases |
| R1-85 | 413 | GONE | D | gone | downstream |
| R1-86 | 413 | GONE | D | gone | downstream |
| R1-87 | 413 | GONE | D | gone | downstream |
| R1-88 | 415 | GONE | D | gone | downstream |
| R1-89 | 415 | GONE | S | gone | downstream figure pointer |
| R1-90 | 419 | GONE | F | gone | caption length (rule 44 still applies paper-wide) |
| R1-91 | 423 | GONE | S | gone | nominalization (rule 14 still applies paper-wide) |
| R1-92 | 423 | GONE | A | gone | observation listing (rule 6 still applies paper-wide) |
| R1-93 | 427 | P76 | O | resolved | limitations in conclusion, 2 sentences |
| R1-94 | 445 | P75 | O | resolved | no numbers in conclusion |
| R2-01 | 13 | P01 | T | resolved | KOZ |
| R2-02 | 13 | P01 | T | resolved | KOZ |
| R2-03 | 19 | P04 | A | resolved | [17] cited in P04; section 2 reduced to notation |
| R2-04 | 25 | P07 | S | resolved | 세우는 |
| R2-05 | 25 | P07 | S | resolved | 세운다 |
| R2-06 | 25 | P07 | T | resolved | acronyms |
| R2-07 | 31 | P03 | S | resolved | 직접 다루기 어렵다 |
| R2-08 | 35 | P05 | T | resolved | 장애물 circled |
| R2-09 | 35 | P05 | T | resolved | same |
| R2-10 | 35 | P05 | S | resolved | topology rhetoric removed |
| R2-11 | 37 | P06 | S | resolved | 유한 개 (a963f31) |
| R2-12 | 37 | P06 | S | resolved | 번역투 (a963f31) |
| R2-13 | 37 | P06 | S | resolved | 번역투 (a963f31) |
| R2-14 | 37 | P06 | S | resolved | 어색 (a963f31) |
| R2-15 | 66 | P11 | T | resolved | KOZ |
| R2-16 | 162 | P17 | S | resolved | 곡선 위의 노드 |
| R2-17 | 170 | P18 | D | resolved | sub-arc points named after S |
| R2-18 | 220 | P26 | S | resolved | 실현 가능성을 재는 양 |
| R2-19 | 222 | P28 | D | resolved | L_seg = chord in both places |
| R2-20 | 227 | F2 | F | resolved | italic symbols in figure (image checked) |
| R2-21 | 316 | P48 | D | resolved | rho defined in P47 first |
| R2-22 | 352 | F3 | F | resolved | compact flowchart, font (image checked) |
| R2-23 | 383 | P56 | D | resolved | Rust only |
| R2-24 | 389 | P57 | S | resolved | 인쇄한 자릿수 |
| R2-25 | 391 | P58 | D | resolved | L_seg consistent |
| R2-26 | 410 | F4 | F | resolved | black-and-white line styles (image checked) |
| R2-27 | 450 | P06 | D | resolved | 보수성 defined in the introduction |
| R2-28 | 453 | F5 | F | resolved | hollow/filled markers (image checked) |
| R2-29 | 456 | GONE | O | gone | degree section cut |
| R2-30 | 479 | P75 | T | resolved | parentheses |
| R2-31 | 481 | P76 | T | resolved | parentheses |
| R3-01 | 20 | P03 | T | resolved | gloss |
| R3-02 | 20 | P03 | T | resolved | gloss |
| R3-03 | 22 | P04 | S | resolved | [17]에서는 |
| R3-04 | 24 | P07 | T | resolved | (3.1절, 5.2절) |
| R3-05 | 24 | P07 | T | resolved | (3.2절) |
| R3-06 | 24 | P07 | T | resolved | (3.3절) |
| R3-07 | 26 | P05 | O | resolved | two paragraphs joined |
| R3-08 | 30 | P06 | S | resolved | rewritten (a963f31) |
| R3-09 | 38 | P09 | T | resolved | citation locator |
| R3-10 | 73 | P13 | S | resolved | 정리할 수 있다. 여기서 |
| R3-11 | 89 | P15 | S | resolved | pointer to 3.2 struck; P15 lost its only link to the objective |
| R3-12 | 95 | GONE | S | resolved | section 3 opener deleted |
| R3-13 | 99 | P16 | S | resolved | [17]에서의 표현 |
| R3-14 | 99 | P16 | S | resolved | pointer struck |
| R3-15 | 105 | P51 | O | resolved | origin sentence moved to section 4 |
| R3-16 | 107 | P17 | S | resolved | sentence struck |
| R3-17 | 109 | P18 | T | resolved | citation struck |
| R3-18 | 117 | P19 | S | resolved | prior-work comparison struck |
| R3-19 | 126 | F1 | F | unverified | font, panel titles (labels present; size depends on PDF scale) |
| R3-20 | 126 | F1 | F | resolved | centroid marker (image checked) |
| R3-21 | 149 | P23 | S | resolved | sentence struck |
| R3-22 | 168 | P26 | T | resolved | 부문제 |
| R3-23 | 168 | P26 | S | resolved | 에 의존한다 |
| R3-24 | 181 | P29 | T | resolved | (3.3절) |
| R3-25 | 189 | P30 | T | resolved | gloss |
| R3-26 | 195 | P31 | T | resolved | gloss |
| R3-27 | 213 | P34 | T | resolved | gloss |
| R3-28 | 239 | P39 | T | resolved | 부문제 |
| R3-29 | 239 | P39 | T | resolved | 부문제 |
| R3-30 | 241 | P40 | T | resolved | 부문제 |
| R3-31 | 275 | P44 | T | resolved | gloss |
| R3-32 | 275 | P44 | T | resolved | nu symbol |
| R3-33 | 275 | P46 | O | resolved | moved after the merit function |
| R3-34 | 275 | P44 | D | resolved | nu defined in P41 |
| R3-35 | 275 | P46 | S | resolved | pointer struck |
| R3-36 | 275 | P46 | S | resolved | 마다 |
| R3-37 | 291 | P47 | O | resolved | flowchart (Figure 3) |
| R3-38 | 299 | P50 | O | resolved | pseudocode |
| R3-39 | 312 | F3 | F | resolved | one-direction flow (image checked) |
| R3-40 | 327 | P32 | O | resolved | constants moved to 3.2 |
| R3-41 | 333 | T1 | F | resolved | scenario numbers |
| R3-42 | 343 | P55 | T | resolved | (4.2절) |
| R3-43 | 349 | P57 | T | resolved | gloss |
| R3-44 | 349 | P57 | D | resolved | 쌍대변수 deleted |
| R3-45 | 351 | P57 | A | resolved | reason for 4000 km given |
| R3-46 | 351 | P57 | S | resolved | trust-region know-how struck (P69 now carries a doubling rule) |
| R3-47 | 353 | P58 | T | resolved | 최적해 수렴 여부 |
| R3-48 | 353 | P58 | T | resolved | gloss |
| R3-49 | 355 | P58 | D | resolved | control cost = J stated |
| R3-50 | 355 | P58 | T | resolved | unit |
| R3-51 | 355 | P58 | T | resolved | gloss |
| R3-52 | 355 | P58 | S | resolved | iterations sentence struck |
| R3-53 | 357 | P63 | O | resolved | 4.2 absorbed into 5.2 |
| R3-54 | 370 | F4 | F | resolved | legend scenario numbers (image checked) |
| R3-55 | 375 | T2 | F | resolved | column deleted |
| R3-56 | 377 | T2 | F | resolved | dagger deleted |
| R3-57 | 385 | P62 | F | resolved | table note moved to text |
| R3-58 | 393 | P65 | A | resolved | reason for excluding n_seg 4 |
| R3-59 | 399 | T4 | F | resolved | n_seg 2 row deleted |
| R3-60 | 410 | P63 | A | resolved | Monte Carlo done |
| R3-61 | 426 | P75 | S | resolved | 수치 실험 |

</details>
