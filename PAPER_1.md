# Paper 1 — offline space-time trajectory optimization, known obstacle motion

**Central doc for paper 1.** Everything about this paper lives here or is linked from here. One
linear document: there is no Part A / Part B split and no "where these disagree, X wins" rule.
That device is what let ten contradictions accumulate by 2026-08-21.

**Scope boundary.** Obstacle motion is known and deterministic. The solve is offline. There is no
sensing model and nothing to re-estimate, which is why receding-horizon replanning is explicitly
declined. The online / uncertain-hazard regime is a *separate* paper: [`PAPER_2.md`](PAPER_2.md).
Stating this assumption plainly is what makes paper 2's contribution legible as a contribution
rather than a correction.

**Supporting evidence, kept separate on purpose** (verification records, not paper content):
- [`doc/refs/novelty_positioning.md`](doc/refs/novelty_positioning.md) — prior-art map, comparison against Osburn, per-source verification status
- [`doc/refs/safe_corridor_references.md`](doc/refs/safe_corridor_references.md) — external ground truth for one plane per (segment, obstacle)
- [`doc/refs/advisor_review_20260820.md`](doc/refs/advisor_review_20260820.md) — the PI's review; outranks any agent's assertion about scope, venue, or the objective

---

# 제출 일정

**두 학회에 같은 논문을 분량만 달리하여 낸다.** 항공우주 2페이지, 기계학회 6페이지. 어느 쪽으로
갈지는 아직 열려 있다 — 판단 근거는 위 검토의견 사이드카에 있다.

| | 한국항공우주학회 추계 | 대한기계학회 추계 |
|---|---|---|
| 발표신청 마감 | — | **2026-08-26(수) 자정** |
| 논문 제출 마감 | **2026-09-04(금)** | **2026-09-02(수) 자정** |
| 분량 | 2페이지 + 별도 400자 초록 | 1~6페이지 |
| 학술대회 | 11/10(화)~13(금) | 11/11(수)~14(토) |
| 장소 | 하이원리조트 (강원 정선) | ICC 제주 |

**두 일정은 11/11~13에 겹친다.** 다른 도(道)에서 동시에 열리므로 두 곳 모두 직접 발표하는 것은
불가능하다. 현재 방향은 제주에 직접 가고 정선 발표는 공저자에게 맡기는 것.

**8/26이 진짜 마감이다.** 비회원은 사전등록 결제를 완료해야 발표신청이 접수된다. 연구비 계정과
품의 절차를 이번 주에 확인해야 한다. 기계학회는 심사결과 9/22(화) 통보, 수정파일 9/30(수) 마감.

항공우주 마감일 재확인 — `ksas.or.kr`는 HTTPS로 응답하지 않는다(타임아웃). HTTP로만 열리고
인코딩은 EUC-KR이므로 웹 조회 도구는 실패한다. 결과가 비면 마감이 바뀐 것이다:

```bash
curl -s "http://ksas.or.kr/Conference/ConferenceView.asp?AC=0&CODE=CC20260701" \
  | iconv -f EUC-KR -t UTF-8 | grep "9월 4일"
```

---

# The claim

> The obstacle's convex over-approximation is built over a **local neighbourhood of the current
> iterate in the lifted space**, and rebuilt on every successive-convexification iteration, instead
> of free space being decomposed into convex cells before the solve begins.

That single property is what keeps arbitrary known non-straight obstacle motion affordable, and
what permits timing — including stopping and waiting — to be optimised. Waiting and free arrival
time are **consequences**, not the mechanism.

**Do not claim "time as a coordinate."** Osburn, Peterson and Salmon published that in August 2025,
along with the space-time lift, finite-height tubes, the Bézier convex-hull argument in the lifted
space, and time monotonicity on control points. **Never write "novel combination."**

**Do not claim speed.** Osburn solves a single-moving-obstacle case in about half a second on the
same solver we use. The argument is how cost scales with dimension, not wall-clock on a
two-dimensional toy.

## 핵심 기여

본 논문은 이동 장애물의 볼록 외부 근사를 풀기 전에 고정하지 않고, 리프팅된 공간에서 현재
반복점의 국소 이웃 위에 순차 볼록 최적화(sequential convex programming, SCP) 반복마다 다시
구성하는 궤적 최적화 정식화를 제안한다. 자유 공간을 볼록 영역으로 미리 분할하지 않으며, 어느
영역을 지날지 고르는 조합적 계층도 두지 않는다.

기여는 세 가지이다. 첫째, 장애물의 운동이 임의로 주어져 관이 휘더라도 **관 전체가 볼록일 필요가
없다** — 국소 조각의 볼록 껍질만 있으면 지지 반공간(supporting half-space)이 성립한다. 둘째, 그
반공간을 분할구간과 장애물 쌍마다 하나씩 두고 해당 분할구간의 모든 제어점에 부과하면 연속
시간에 대한 볼록 껍질 보장이 유지된다. 제어점마다 다른 평면을 부과하면 이 보장은 유한 개의 점
조건으로 약해진다. 셋째, 동일한 구성이 휘어진 KOZ 관과 관측자에 대한 차폐 영역을 모두 처리한다.

각 기여가 틀렸음을 보이는 조건도 함께 적어 둔다. 첫째 기여는 장애물 근사를 반복마다 국소적으로
다시 구성하는 선행 연구가 있으면 무너진다. 둘째 기여는 모든 지지 반공간을 여유 없이 만족한 해가
장애물의 실제 궤적에 대해 재검증했을 때 침범으로 판정되면 무너진다. 셋째 기여는 차폐 제약이
KOZ와 다른 구성을 요구하는 사례가 나오면 무너진다.

## What is different

| approach | free space | obstacle over-approximation | time allocation |
|---|---|---|---|
| Osburn, and the graph-of-convex-sets family | decomposed in advance into convex cells, plus a combinatorial layer to choose among them | not used | chosen, but only inside static cells |
| MADER, semantic-corridor and driving-domain corridor methods | none | convex hull over a fixed interval | fixed by the spline knots |
| ours | none | convex hull over an **adaptive local neighbourhood** | a decision variable |

- MADER structurally cannot wait — its time allocation is baked into the knot vector before the
  solve.
- The decomposition is a **recurring** cost, not a setup cost. Osburn's own sweep shows a 17%
  better trajectory costing 17× the compute, with free-space coverage still incomplete at the top
  of the sweep. Both halves of that machine — sampling convex regions, then searching the graph —
  degrade as dimensions are added, and we are adding one.

---

# The formulation

Changed 2026-08-21. Everything above the horizontal rule below is the current construction; the
frozen decisions that survived the change follow it.

## The lift, and why no axis is special

Three spatial coordinates plus time, all four coordinates of **one** Bézier. The convex-hull
property holds verbatim in the lifted space with no side conditions — unlike corridor methods that
write position as a function of time and need an extra sufficient condition, because there the
spatial axes and the temporal axis are not equivalent.

**This equivalence is load-bearing, not cosmetic.** `SPACETIME_AXIS_SCALE = 1.0` used to be a
declared modelling convenience. It is now structural: the construction below measures distance and
takes a ball in the lifted space, and neither is meaningful until the axes are commensurable.
Nothing in the formulation may single out the time axis — no time windows, no per-segment time
intervals, no clipping along t.

## The obstacle

Known, deterministic, **arbitrary** motion. The swept volume is a tube of radius `r` around a
curved centreline in the lifted space.

Curved, therefore **not convex**, therefore it has **no supporting half-space at all** — no single
plane holds the whole tube on one side, so the plane would certify nothing and a segment's hull
could straddle it. That is the problem the construction exists to solve. Not conservatism, not
performance.

*Constant velocity is the special case where the centreline is straight and the tube is a capsule.
It is a special case and must never be the setup.*

## The clip

Per (segment, obstacle), per SCP iteration:

- `c` — the segment's centroid in the lifted space, a linear function of the control points
- `f` — the nearest point of the obstacle's centreline to `c`, measured in the lifted space
- `R = |c − f|` — so the ball `B(c, R)` is **tangent to the centreline** at `f`
- capped at `R_max = r + E + Δ√(d+1)`, where `E` is the segment's radius about its centroid
- **clipped piece = tube ∩ `B(c, min(R, R_max))`**

A lens at the contact point. Always non-empty, since the tube's nearest material sits at `R − r`
from `c`. The radius scales itself with proximity — far segments get a large ball they do not need,
close segments get a small one exactly where the constraint acts.

**The cap is free.** Beyond `R_max` the segment provably cannot reach the tube within one trust
step, so the ball stops touching it and no row is emitted. It bounds both the worst-case hull size
and the subdivision work, and — see (7) — it never weakens soundness. It is also the one place a
clip parameter may be tied to Δ without breaking attribution in the ratio test, because it acts
only where the row cannot bind.

**No free parameter, and no axis privileged.** A ball, not a slab, not an interval.

## The over-approximation and the half-space

Take the convex hull of the clipped piece. Two facts make this cheap and exact: hulling and
inflating commute, so it is enough to convexify the **centreline** and inflate the result by `r`;
and if the obstacle's motion is polynomial, its lifted centreline is itself a Bézier, so **De
Casteljau subdivision** hands you exact control points for the stretch inside the ball.

**Do not sample the centreline.** A curve bulges outside the chords between its samples, so the
hull of sampled points need not contain the tube — containment fails and the certificate with it.
Recovering it costs a chord-error inflation. Subdividing the obstacle's own Bézier removes the
error instead of bounding it.

Then emit **exactly one supporting half-space** — the plane through the projection of the centroid
onto that hull — and apply it to **every control point of the segment.**

**The plane must come from the hull, not from the tube.** A plane tangent to the tube at its
closest surface point is valid only if the tube is convex. Measured counterexample in the formal
section: with the obstacle turning and the segment inside the turn, 12.35% of the clipped piece
sits on the *allowed* side of the tangent plane. The correct plane differs by 1.19° in normal and
0.76 in offset.

One plane per segment-and-obstacle pair, never one per control point. Per-control-point planes
prove only that each point individually is outside its own plane — exactly as strong as sampling
the curve — and opposing normals let the hull wrap around the tube.

The plane's normal has a nonzero **time component** because the centreline is slanted in the lifted
space. It is not optional; without it the plane does not support the tube at all. *For the
constant-velocity special case that component reduces to minus the dot product of the spatial
normal with the obstacle's velocity. That expression is the special case, not the definition.*

## The hole, stated as a hole

The plane protects against tube material on the obstacle side of it. **Tube material lying outside
the ball *and* on the safe side of the plane is not protected** — and at a bend, the far arm is
exactly that.

Soundness would require the ball's radius to exceed the segment's own extent plus the trust radius.
With `R` set to the distance from the centroid to the centreline, that fails precisely when the
segment gets close to the obstacle — which is when the constraint matters.

**So this construction is not sound by construction.** It is adopted because its failure is
**detectable**, by the check below.

**It is one line from being sound.** Statement (8) in the formal section: clamp the clip radius
below by `E + Δ√(d+1)` instead of letting it shrink to the tangent value, and the certificate
covers the full tube unconditionally. The price is conservatism exactly where the constraint is
active. **Which of the two to use is an open experimental question** — the tangent form goes in
first, and the fallbacks stand: more segments, centre on the contact point, reach filtering,
adaptive growth. If a reviewer asks for a guarantee, there is not one — there is
a check. Say so in the limitations section rather than letting it be found.

## Verification — the only thing that certifies

Sample the **whole returned trajectory** densely and measure clearance against the obstacle's true
motion. Not per segment. Never against the hulls the solver used.

`compute_min_clearance` (`rust_optimizer/core/src/spacetime_optimizer.rs:58`) already does exactly
this — 1500 samples, spatial distance at each sample's own time, obstacles skipped outside their
own windows, and it touches neither the planes nor the axis scale. It is genuinely independent of
the construction it checks. Reuse it; do not write a second checker.

This is also condition 2 of the risk below, and it is what makes the hole acceptable.

## The risk this construction creates

An adaptively-built hull is valid only inside the neighbourhood it was built to cover. If the next
iterate's segment slides outside that neighbourhood, the constraint becomes **silently
meaningless** — the solver reports zero slack on a row that asserts nothing, and a feasibility gate
passes a trajectory that penetrates an obstacle.

Two conditions, neither automatic:

1. the segment must stay inside the ball its hull was built from — by explicit constraint, or by a
   trust region small enough to guarantee it;
2. the gate must re-verify clearance against the true obstacle trajectory, as above.

**This makes the trust region correctness-critical rather than a convergence aid.** Without it,
nothing keeps the iterate inside the region where its own constraints are valid.

---

# Formulation — rigorous statement

The prose above is the reading guide; this section is the object. Every claim the paper makes
about correctness must be traceable to a numbered statement here.

## Notation

Spatial dimension $d$; the lifted space is $\mathbb{R}^{d+1}$ with a point written
$z=(x,\tau)$, $x\in\mathbb{R}^{d}$, $\tau\in\mathbb{R}$. **The axis scale is $1$**, so the
Euclidean norm $\lVert\cdot\rVert$ on $\mathbb{R}^{d+1}$ mixes space and time on equal terms.
Every ball, distance and projection below is taken in that norm. This is the formal content of
"no axis is privileged" — remove it and none of what follows is defined.

Decision variable: control points $P_0,\dots,P_N\in\mathbb{R}^{d+1}$, $P_j=(x_j,t_j)$, stacked
into $\mathbf p\in\mathbb{R}^{(N+1)(d+1)}$.

Curve, with $u\in[0,1]$ the curve parameter — **not** time:

$$Z(u)=\sum_{j=0}^{N}B_j^N(u)\,P_j,\qquad B_j^N(u)=\binom{N}{j}u^j(1-u)^{N-j},\qquad
\tau(u)=\sum_{j=0}^{N}B_j^N(u)\,t_j .$$

De Casteljau subdivision into $M$ segments gives row-stochastic matrices $A^{(k)}$
($A^{(k)}_{ij}\ge 0$, $\sum_j A^{(k)}_{ij}=1$) with segment control points and centroid

$$Q^{(k)}_i=\sum_j A^{(k)}_{ij}P_j,\qquad
c^{(k)}=\tfrac1{N+1}\sum_i Q^{(k)}_i=\sum_j w^{(k)}_j P_j,\quad w^{(k)}_j\ge0,\ \textstyle\sum_j w^{(k)}_j=1 .$$

**(H) Hull property.** For $u$ in segment $k$'s parameter interval, $Z(u)\in\operatorname{conv}\{Q^{(k)}_i\}_{i=0}^{N}$.

Segment radius: $E^{(k)}=\max_i\lVert Q^{(k)}_i-c^{(k)}\rVert$.

## Obstacle

Obstacle $m$ has known motion $\pi_m$ on $[T^0_m,T^1_m]$. Lifted centreline and keep-out zone:

$$\gamma_m(\tau)=\bigl(\pi_m(\tau),\tau\bigr),\qquad
\Gamma_m=\gamma_m\!\left([T^0_m,T^1_m]\right),\qquad
\mathcal K_m=\Gamma_m\oplus \bar B(0,r_m).$$

$\Gamma_m$ is a curve, so $\mathcal K_m$ is **not convex** unless $\pi_m$ is affine. It therefore
admits no supporting half-space, and (H) alone certifies nothing against it.

## The clip

With $c=c^{(k)}$ fixed at the reference iterate:

$$\tau^\star\in\arg\min_{\tau\in[T^0_m,T^1_m]}\lVert c-\gamma_m(\tau)\rVert,\qquad
f=\gamma_m(\tau^\star),\qquad R=\lVert c-f\rVert,$$

$$\boxed{\ \rho=\min\bigl(R,\;R_{\max}\bigr),\qquad
R_{\max}=r_m+E^{(k)}+\Delta\sqrt{d+1}\ }$$

$$\mathcal K_m^{(k)}=\mathcal K_m\cap \bar B(c,\rho)\qquad\text{(the clipped piece).}$$

$\bar B(c,R)$ is tangent to $\Gamma_m$ at $f$. $\tau^\star$ need not be unique; any minimiser
serves, and non-uniqueness is exactly the bend case that makes $f$ discontinuous in $\mathbf p$.

## Over-approximation

**(1) Containment of the clip in a centreline band.** Let
$I=\{\tau:\lVert\gamma_m(\tau)-c\rVert\le\rho+r_m\}$. Then

$$\mathcal K_m\cap\bar B(c,\rho)\ \subseteq\ \gamma_m(I)\oplus\bar B(0,r_m).$$

*Proof.* Let $z$ be in the left side. Some $\tau$ has $\lVert z-\gamma_m(\tau)\rVert\le r_m$, and
$\lVert z-c\rVert\le\rho$, so $\lVert\gamma_m(\tau)-c\rVert\le r_m+\rho$, i.e. $\tau\in I$. $\square$

**(2) Hulling and inflating commute.** $\operatorname{conv}(S\oplus\bar B(0,r))=\operatorname{conv}(S)\oplus\bar B(0,r)$.

So it suffices to convexify the **centreline**, never the tube.

**(3) Exact convexification when the motion is polynomial.** If $\pi_m$ has degree $N_m$ then
$\gamma_m$ is a Bézier curve in $\mathbb{R}^{d+1}$ with control points $G_0,\dots,G_{N_m}$. Take
any interval $[\alpha,\beta]\supseteq I$ — e.g. $[\min I,\max I]$, valid even when $I$ is
disconnected — and let $\tilde G_0,\dots,\tilde G_{N_m}$ be its De Casteljau subdivision control
points. By (H) applied to the obstacle,
$\gamma_m([\alpha,\beta])\subseteq\mathcal G:=\operatorname{conv}\{\tilde G_l\}$. Define

$$\boxed{\ \mathcal H=\mathcal G\oplus\bar B(0,r_m)\ }$$

Then $\mathcal H$ is convex and $\mathcal K_m^{(k)}\subseteq\mathcal H$ by (1)–(3).

**No sampling anywhere.** Hulling sampled centreline points would give a set that a curve bulges
*outside of*, breaking containment; recovering it needs a chord-error inflation. Subdividing the
obstacle's own Bézier removes the error rather than bounding it.

## The supporting half-space

Let $y^\star=\arg\min_{y\in\mathcal G}\lVert c-y\rVert$ (projection onto a polytope; a small QP).
Assume $\lVert c-y^\star\rVert>r_m$, i.e. $c\notin\mathcal H$. Set

$$\boxed{\ n=\frac{c-y^\star}{\lVert c-y^\star\rVert},\qquad b=n^\top y^\star+r_m\ }$$

**(4) $\mathcal H\subseteq\{z:n^\top z\le b\}$.**

*Proof.* $\mathcal G$ is convex and $y^\star$ is the projection of $c$ onto it, so the variational
inequality $(c-y^\star)^\top(y-y^\star)\le0$ holds for all $y\in\mathcal G$, i.e.
$n^\top y\le n^\top y^\star$. Any $z\in\mathcal H$ is $y+v$ with $y\in\mathcal G$,
$\lVert v\rVert\le r_m$, so $n^\top z\le n^\top y^\star+r_m=b$. $\square$

**This is the step the tangent plane fails.** Taking $n$ from $c-f$ and $b$ from the tube surface
at $f$ satisfies (4) only when $\mathcal K_m$ is convex. Measured counterexample: obstacle on a
circle of radius $10$ at $0.5$ rad/s, $r_m=1$, $c$ inside the turn — $12.35\%$ of the clipped
piece lands strictly on the safe side of the tangent plane, worst point $0.77$ inside it and
$0.99$ from the centreline. The projection onto $\mathcal G$ moves the normal by $1.19^\circ$ and
the offset by $0.76$, and removes every violation.

## Rows and the segment certificate

Impose on **every** control point of segment $k$:

$$n^\top Q^{(k)}_i\ \ge\ b,\qquad i=0,\dots,N
\qquad\Longleftrightarrow\qquad
n^\top\!\!\sum_j A^{(k)}_{ij}P_j\ \ge\ b .$$

**(5) Segment certificate.** If those $N+1$ rows hold, then $Z(u)\notin\operatorname{int}\mathcal H$
for every $u$ in segment $k$, hence $Z(u)\notin\operatorname{int}\bigl(\mathcal K_m\cap\bar B(c,\rho)\bigr)$.

*Proof.* By (H), $Z(u)=\sum_i\lambda_iQ^{(k)}_i$ with $\lambda\ge0$, $\sum\lambda_i=1$. Linearity
gives $n^\top Z(u)\ge b$; apply (4). $\square$

One plane per $(k,m)$. Per-control-point planes destroy (5): the step from $Q_i$ to $Z(u)$ uses
the *same* $n$ for all $i$, and with differing normals the hull can wrap the tube.

## Reachability, and exactly when the certificate covers $\mathcal K_m$

(5) certifies against $\mathcal K_m\cap\bar B(c,\rho)$, **not** against $\mathcal K_m$. The
difference is closed by the trust region and nothing else.

**(6) Reachability.** If $\lVert\mathbf p^{+}-\mathbf p\rVert_\infty\le\Delta$ then for every $u$
in segment $k$, $Z^{+}(u)\in\bar B\!\left(c,\;E^{(k)}+\Delta\sqrt{d+1}\right)$.

*Proof.* $Q^{+}_i-Q_i=\sum_jA^{(k)}_{ij}(P^{+}_j-P_j)$ is a convex combination of vectors of
$\infty$-norm at most $\Delta$, so $\lVert Q^{+}_i-Q_i\rVert_\infty\le\Delta$ and
$\lVert Q^{+}_i-Q_i\rVert\le\Delta\sqrt{d+1}$. Hence
$\lVert Q^{+}_i-c\rVert\le E^{(k)}+\Delta\sqrt{d+1}$, and the ball is convex, so it contains
$\operatorname{conv}\{Q^{+}_i\}\ni Z^{+}(u)$. $\square$

**(7) Soundness condition.**

$$\boxed{\ \rho\ \ge\ E^{(k)}+\Delta\sqrt{d+1}\ }$$

Under (7), $\bar B(c,\rho)$ contains the whole next-iterate segment, so (5) certifies it against
the **full** $\mathcal K_m$. Note (7) does not involve $r_m$ or the obstacle at all.

Three regimes for $\rho=\min(R,R_{\max})$:

| regime | $\rho$ | (7) holds? |
|---|---|---|
| $R>R_{\max}$ — obstacle out of reach | $R_{\max}=r_m+E+\Delta\sqrt{d+1}$ | **yes**, and the clip is empty so no row is emitted |
| $E+\Delta\sqrt{d+1}\le R\le R_{\max}$ | $R$ | **yes** |
| $r_m<R<E+\Delta\sqrt{d+1}$ — segment close | $R$ | **no — this is the hole** |

So the cap is free: it only acts where the row cannot bind, and it preserves (7). The hole is at
*small* $R$, i.e. exactly when the constraint binds.

**(8) The one-line repair, and its price.** Replacing the clip radius by

$$\rho=\operatorname{clip}\!\left(R,\ E^{(k)}+\Delta\sqrt{d+1},\ R_{\max}\right)$$

makes (7) hold unconditionally and the construction **sound by construction**. It costs
conservatism precisely where the constraint is active, because the ball is then larger than the
distance to the obstacle and $\mathcal G$ covers more centreline. **Which of the two is used is an
open experimental question, not a settled one** — the tangent form is being implemented first.

## The subproblem

$$\min_{\mathbf p,\ \sigma\ \ge 0}\quad
\tfrac12\,\mathbf p^\top H\,\mathbf p\;+\;w_t\,t_N\;+\;\lambda\!\sum\sigma$$

subject to, with $\mathbf p^{\mathrm{ref}}$ the reference iterate:

| | |
|---|---|
| boundary | $P_0=z^{\text{start}}$; $x_N=x^{\text{goal}}$; $t_N$ free iff free-arrival |
| monotonicity | $t_{j+1}-t_j\ \ge\ \delta>0$ |
| slant limit | $\lVert x_{j+1}-x_j\rVert\ \le\ v_{\max}\,(t_{j+1}-t_j)$ — second-order cone, **no slack** |
| keep-out | $n_{km}^\top Q^{(k)}_i+\sigma_{kmi}\ \ge\ b_{km}$ |
| occlusion | same form, per $(k,m,\text{station})$ |
| trust region | $\lVert\mathbf p-\mathbf p^{\mathrm{ref}}\rVert_\infty\ \le\ \Delta$ |

**$H$ is blind to timing.** It is assembled only from second differences of the spatial
coordinates, so $He_{t_j}=0$ for every time-coordinate basis vector: two control-point sets with
equal spatial coordinates and any time coordinates score identically. *Checked by test.*

**(9) The slant limit is sufficient.** $x'(u)=N\sum_{j}B^{N-1}_j(u)(x_{j+1}-x_j)$ and
$\tau'(u)=N\sum_jB^{N-1}_j(u)(t_{j+1}-t_j)$. Since $B^{N-1}_j\ge0$, the triangle inequality gives

$$\lVert x'(u)\rVert\le N\!\sum_j B^{N-1}_j(u)\lVert x_{j+1}-x_j\rVert
\le v_{\max}\,N\!\sum_j B^{N-1}_j(u)(t_{j+1}-t_j)=v_{\max}\,\tau'(u),$$

so physical speed $\lVert x'\rVert/\tau'\le v_{\max}$ everywhere. **Clearing the denominator
requires $\tau'>0$**, which is what makes monotonicity physics-load-bearing rather than a sanity
constraint. $\square$

## The certificate

$$\boxed{\ \mu(\mathbf p)=\min_{u\in[0,1]}\ \ \min_{m:\ \tau(u)\in[T^0_m,T^1_m]}\ \
\bigl\lVert x(u)-\pi_m(\tau(u))\bigr\rVert-r_m\ }$$

Spatial distance at each sample's **own** time, minus the radius, obstacles skipped outside their
own existence window. It touches neither $n$, $b$, $\mathcal G$, $\rho$, nor the axis scale, so it
is independent of everything it checks. The run is feasible iff $\mu>0$.

$\mu$ is the only statement made about $\mathcal K_m$ itself. Everywhere (7) fails, it is the only
statement there is.

# Frozen decisions that survived the change

Recorded 2026-08-19, unaffected by the 08-21 construction change. Full guardrail form in
`CLAUDE.md` §"Formulation decisions".

**No quadratic acceleration energy exists.** The lifted Bézier is drawn by a parameter, and
converting to real time makes physical velocity a *ratio* of two Béziers and physical acceleration
a ratio with a cubic denominator. Neither is polynomial in the control points, so no matrix makes
acceleration energy a quadratic form. "Re-derive the energy matrix" is not a task that can succeed.
Recorded so it is not attempted a second time.

**The objective is a parameter-domain smoothness regularizer, and it is blind to timing.** It loops
over the spatial coordinates only and never writes into the time column, so its value is a function
of the spatial control points alone. Hold those fixed and move the times however you like — wait
nine seconds then dash across in one, or cruise evenly for ten — and it returns the **same number**.
Minimising it improves nothing about physical velocity or acceleration. Physics goes into
constraints. *Checked: two control-point sets with equal spatial coordinates and different times
must score identically.* **The phrase "minimizes spatial acceleration" is permanently forbidden**,
not merely deferred.

**No obstacle penalty in the objective** (PI's instruction, 2026-08-20). Avoidance is a hard
constraint through the supporting half-space. The elastic slack variables and their penalty term
may appear as a **numerical device**, and the write-up must say plainly that this is not the
avoidance mechanism.

**The speed cap is a slant limit, and it is provably sufficient.** Clear the denominator: the
spatial gap per parameter, in norm, is at most the speed limit times the time gap per parameter.
Both sides are then polynomials of the same degree and weighted averages of their own control
points with non-negative weights summing to one, so by the triangle inequality a per-control-point
condition implies the curve-wide one. It reduces to: *the spatial distance between consecutive
control points is at most the speed limit times the time between them.* Convex, acts on control
points directly, conservative in the safe direction. **Clearing the denominator is legal only
because time strictly increases along the curve — which makes time monotonicity physics-load-
bearing, not a sanity constraint.**

**The acceleration bound is bilinear, so linearize it** each iteration exactly as the keep-out rows
are, and let the trust region keep it valid. No new machinery.

**Arrival time is free and its penalty is linear**, because a Bézier passes through its last
control point, so arrival time *is* that point's time coordinate. Quadratic smoothness plus a
linear time penalty keeps the subproblem a QP.

> **Trap, recorded so it is recognised rather than rediscovered.** Add the time penalty *without*
> the speed cap and arrival time collapses to the only floor that exists — minimum time separation
> times the number of control-point gaps — regardless of scenario and regardless of the weight. A
> run reporting that has produced an artifact, not a result. The two must land together.

**The energy-versus-time weight is a reported scenario parameter**, not a tuned constant. Its
sweep — arrival time against energy — is a legitimate figure.

**No numbers until re-measurement.** *Re-armed 2026-08-21.* The 2026-08-20 one-pass measurement was
made against constant-velocity capsules and a plan-horizon clip. The construction change
invalidates all of it. Nothing measured before the new construction lands goes in the paper.

---

# Occlusion

## The shadow lemma

Let the observer be a single point and the obstacle any convex body not containing it. Define the
shadow as the set of positions whose straight line of sight back to the observer is interrupted.
**Then the shadow is convex.**

*Proof in words.* Measure from the observer. A position is in shadow exactly when it is some
obstacle point stretched away from the observer by a factor of at least one. Take two shadow
positions and any weighted average with non-negative weights summing to one. Each is an obstacle
point times its own stretch factor. Let the combined stretch factor be the same weighted average of
the two; since each is at least one, so is it. Divide the averaged position by that combined factor
and what remains is a weighted average of the two original obstacle points — a point of the
obstacle, since the obstacle is convex. So the average is an obstacle point stretched by a factor
of at least one, which is a shadow position. ∎

Holds in any dimension, so spheres in three spatial dimensions are covered unchanged. **Derived
here, not taken from a source** — if it appears in the paper, state it as our result.

Consequence — hiding is cheaper than avoiding:

| constraint | region | convex? | machinery |
|---|---|---|---|
| stay out of the keep-out zone | outside a tube | no | supporting half-space plus a committed side |
| **stay out of the shadow** (keep the link) | outside a cone | no | supporting half-space plus a committed side — the same code |
| stay inside the shadow (hide) | inside a cone | yes | plain linear rows, no side to commit |

## The shadow in the lifted space

At each instant the shadow is convex, but the cone swings as the obstacle moves, so the shadow
volume in the lifted space is **not** convex. It gets the identical treatment as the tube: clip to
the same local ball, take the convex hull of the clipped piece, one supporting half-space.

**Measured 2026-08-19 — the check ran and it can fail.** The space-time shadow of a moving occluder
is not convex, and the mechanism is not the one originally predicted: the half-angle change from
varying observer distance is *minor*; the dominant contributor is the **cone rotating** as the
occluder crosses the sight line. Purely tangential motion, where distance barely changes, violates
convexity an order of magnitude worse than radial motion. The stationary control case comes out
exactly convex. **Conclusion: the conservative outer approximation is mandatory, not optional.**

⚠️ *That sweep measured pieces cut along the time axis. It does not transfer to a ball-defined
piece and must be re-measured before it is quoted.* When quoting, quote the violation magnitude and
the window length — the failure *fraction* depends on the sampling box and is a property of the
measure, not the geometry.

---

# Demo scenario

## Recorded, not used

Kept so the choice is legible and so a discarded option cannot quietly reappear:

- **Wait for the door.** The curve stops and waits for a time-gated obstacle to clear. Verified
  who else can wait: Safe Interval Path Planning and space-time RRT can (discrete / sampled
  output); MADER cannot (knots fix the timing); Osburn and the graph-of-convex-sets family have not
  demonstrated it. Same assumption, different output class.
- **Hide inside the shadow.** Stay concealed from an observer behind zigzagging occluders. Convex,
  cheap, one-line claim — and *rejected for that reason*: it is the easy side.

## The scenario the paper uses — stay **outside** the shadow

Maintain line of sight to a ground station while avoiding a moving obstacle, at **three spatial
dimensions plus time**.

**Moving occluder, non-straight path.** The static alternative was rejected: a static shadow
extruded along time is a convex prism, exactly what a graphs-of-convex-sets method handles
natively — cheap for us *and* for the nearest prior work, so it demonstrates nothing. Only the
twisting shadow of a moving occluder earns the claim.

**One figure carries both consequences.** Two pages hold one demonstrated consequence, so there is
no separate dimensional-scaling figure — the occlusion demo itself runs in three spatial dimensions
and does double duty.

**The third dimension must be load-bearing.** Built so going around loses contact and going over
keeps it: the climb exists only because there is a third spatial dimension, and the reason it
happens is the occlusion constraint. Causally linked in one maneuver, not co-located in one plot.

**The occlusion constraint picks the passing side.** The station is placed so the natural avoidance
direction is the one that breaks the line of sight, so the rows *select the passing class* rather
than decorate a trajectory that would have looked the same.

**Occluder and keep-out obstacle are the same body.** Staying visible implies not being inside the
occluder, so occlusion subsumes collision avoidance here and the keep-out rows are present but
never binding. One honest sentence; the keep-out machinery is demonstrated by other scenarios.

**Geometry.** A wide, low moving fence on a **curved** path, with the station beyond it at low
altitude. Going around laterally is long or hits the workspace bound; going over is short and exits
the shadow because the fence is low. *The `wall` scenario is straight and constant-velocity — it no
longer exercises the paper's own case and must be replaced.*

**The baseline that must be able to fail.** Same scenario, occlusion rows removed: it must lose
contact for a measurable interval. If the baseline also keeps line of sight, the constraint was
slack and the figure proves nothing — move the station and re-run.

**Figure: one figure, two panels.** Three spatial dimensions plus time has no natural projection,
so the picture and the proof are different panels — a 3D spatial view that sells the climb, and a
line-of-sight-margin-versus-time panel where the baseline dips below zero and the constrained run
stays above. The margin panel is dimension-free and is the part a reviewer checks.

---

# Decided against

- **Convexifying free space by a coordinate transform** (inversion, conformal, Nyquist-style). The
  obstruction is topological, not geometric: a convex set is simply connected, free space around an
  obstacle is not, and continuous invertible maps preserve that. Inversion moves the hole to the
  origin; it does not remove it. The hole *is* the fact that passing left and passing right are
  distinct plans. The strongest published version is Rimon and Koditschek's navigation functions,
  whose titles concede the point and whose guarantee holds only from almost all initial conditions.
- **Receding-horizon replanning.** Motion is deterministic and known; nothing to re-estimate.
- **The radar equation, detection probability, target fluctuation.** Binary occlusion only.
- **Competing on supporting more constraint types.** That is Osburn's own stated contribution.
  Compete instead on the *kind* of constraint — nonconvex and time-varying.

---

# What must be measured

One pass, after the new construction lands. Nothing before it may be quoted.

- the stay-visible scenario keeping line of sight start to finish, with the baseline failing;
- **minimum clearance against the true obstacle trajectory**, never against the solver's hulls;
- how often the clip's hole actually fires — a returned iterate where every half-space is satisfied
  but the trajectory-wide check reports penetration. This is the number the limitations section
  needs;
- the shadow convexity sweep, re-measured for ball-defined pieces;
- solve time against Osburn's half-second single-obstacle and roughly four-second cluttered cases,
  on the same solver and comparable hardware.

# Weakest links

Ranked by damage if wrong. **Re-ranked 2026-08-21 — the previous top item is no longer the top
item.**

1. **The clip is not sound by construction.** Its hole is at bends, which is exactly the case this
   paper is about, and the mitigation is detection rather than prevention. A reviewer asking for a
   guarantee gets a check instead. This is new, it is the largest risk, and it must be stated in
   the paper rather than found.
2. **MADER's fixed-knot time allocation** is the pivot of the comparison table, and it reached us
   through a fetch summariser rather than the source. The solver workstream read MADER in full and
   its quoted equation is consistent, but that specific sentence is unconfirmed. Confirm it before
   the formulation section relies on it.
3. **The shadow lemma is ours** and nobody else has reviewed it.
4. **"No paper does this" is weaker than a proof.** No decomposition-free space-time formulation
   was found, but absence of evidence is not proof of absence.
5. **Erdmann and Lozano-Pérez, 1987, is unverified** — the archived copy is a scan with no text
   layer. Lineage citation only; nothing load-bearing rests on it.

---

# Open

- **Venue.** Both, or 항공우주 only. Decides the title (mission-first for 항공우주, method-first
  for 기계학회 — the PI supplied a candidate for each) and it is on the 8/26 clock.
- **`CLAUDE.md` §"Formulation decisions" item 8** still asserts the keep-out tube is a capsule and
  therefore convex, which is now false. Needs the same correction as this file; not done here.
- **Occlusion mission motivation.** The PI's review requires a statement of *which mission, and
  why* line-of-sight maintenance is needed in aerospace practice. Currently the constraint is
  presented only as a methodological differentiator. Required in both manuscript and talk.
