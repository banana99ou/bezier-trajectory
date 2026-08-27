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

**한국항공우주학회 2026년도 추계학술대회로 확정 — 2026-08-25.** 대한기계학회 추계는 포기했다.
두 학회가 11/11~13에 겹치고 다른 도(道)에서 동시에 열려 한 곳만 고를 수 있으며, 지도교수 역시
둘 중 하나를 고르라는 입장이었다. **검토의견 §1의 "6페이지 기계학회에도 낸다" 제안은 이 결정으로
닫혔다 — 다시 꺼내지 말 것.**

| | 한국항공우주학회 추계 |
|---|---|
| 논문 제출 마감 | **2026-09-04(금)** |
| 분량 | 2페이지 + 별도 400자 초록 |
| 학술대회 | 11/10(화)~13(금) |
| 장소 | 하이원리조트 (강원 정선) |

**400자 초록은 원고가 아니라 제출 웹페이지에 입력한다.** 별도 산출물이고 마감은 같다. 제출에는
회비 납부와 사전등록 결제 완료가 전제이며, 결제 기한은 사전등록 마감이 아니라 논문 마감 전이다.

**6페이지가 필요했던 논증은 저널로 간다.** 학회 발표 이후 내용을 보완해 저널에 제출하는 것이
후속 계획이다. 지지 반공간 구성, 매 반복 재구성, 볼록성 논의는 거기서 전개하고, 2페이지 원고는
그 논증을 다 담으려 하지 말고 현장 토론용으로 유지한다.

마감일 재확인 — `ksas.or.kr`는 HTTPS로 응답하지 않는다(타임아웃). HTTP로만 열리고 인코딩은
EUC-KR이므로 웹 조회 도구는 실패한다. 결과가 비면 마감이 바뀐 것이다:

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
없다** — 클리핑된 KOZ 볼륨에 대한 지지 반공간(supporting half-space)만 세우면 된다. 둘째, 그
반공간을 분할구간·장애물·클리핑된 성분마다 하나씩 두고 해당 분할구간의 모든 제어점에 부과하면
연속 시간에 대한 볼록 껍질 보장이 유지된다. 제어점마다 다른 평면을 부과하면 이 보장은 유한 개의 점
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

*Symbols, fixed 2026-08-25. The **clipping radius is `r_clip`**. `ρ` is reserved for the merit-function
ratio of the successive-convexification loop and never denotes a clipping radius. `d` is the
distance from the segment centroid to the nearest centreline point. `r_clip_max` is the cap on `r_clip`. In
the formal section the spatial dimension is written `D`, not `d`, for exactly this reason.*

Per (segment, obstacle), per SCP iteration:

- `c` — the segment's centroid in the lifted space, a linear function of the control points
- `f` — the nearest point of the obstacle's centreline to `c`, measured in the lifted space
- `d = |c − f|` — the distance from the centroid to the nearest centreline point, so the ball
  `B(c, d)` is **tangent to the centreline** at `f`
- capped at `r_clip_max = r + E + Δ√(d_spatial+1)` — the obstacle radius, plus the segment's own radius `E`
  about its centroid, plus the furthest one trust step of size `Δ` can carry a control point in a
  lifted space of `d_spatial+1` coordinates
- **floored at the obstacle radius `r`** — see below; the floor binds exactly when `c` is inside
  the KOZ, and nowhere else
- **clip radius `r_clip = clamp(d, r, r_clip_max)` = `max(min(d, r_clip_max), r)`**
- **clipped KOZ volume = KOZ ∩ `B(c, r_clip)`**

A lens at the contact point. Always non-empty, since the tube's nearest material sits at `d − r`
from `c`. The radius scales itself with proximity — far segments get a large ball they do not need,
close segments get a small one exactly where the constraint acts.

**The floor at `r` binds only when the centroid is inside the KOZ.** `r_clip_max` is `r` plus a
non-negative reach, so `min(d, r_clip_max)` can fall below `r` only when `d < r` — which is
precisely the deep-penetration case. Everywhere else the floor is inert and the radius is `min(d,
r_clip_max)` unchanged. What it buys there is measured: with the centroid `0.55` inside a tube of
radius `0.9`, the unfloored ball has radius `0.35`, is **smaller than the obstacle**, and the wall
retreats onto the ball instead of the tube — offset `0.60`, centroid margin exactly `0.00`, a
`0.55`-deep penetration reported as *on the boundary*. Floored to `0.9`, the same case gives
centroid margin `−0.879` and the row says what it should.

**The cap exists to ignore obstacles that are too far away.** Beyond `r_clip_max` the segment
provably cannot reach the tube within one trust step, so no row is needed and none is emitted. It
also bounds the worst-case work, and it never weakens soundness. *(It is a reachability test, not
an emptiness test — the ball still grazes the tube a little past the cap. That costs nothing,
because a row there could not bind anyway.)* It bounds both the worst-case hull size
and the subdivision work, and — see the soundness condition in the formal section — it never
weakens soundness. It is also the one place a clip parameter may be tied to Δ without breaking
attribution in the ratio test, because it acts only where the row cannot bind.

A ball, not a slab, not an interval.

## Where the ball is centred — open, and measured rather than asserted

Three knobs, and **none of the three is settled**:

1. **Where the ball is centred.** The segment centroid `c` (what the code does now); the nearest
   centreline point `f`; the nearest point of the KOZ surface.
2. **How big it is.** The tangent value `d`; the reach-sized value `E + Δ√(d_spatial+1)`; or `d` clamped
   between the two.
3. Whether the answers to 1 and 2 are independent — a centring choice can change which radius is
   sound.

These are experimental questions. They are to be measured on real runs, not decided by argument in
this file.

**The centreline-centred variant is PINNED, not rejected**, and its measured properties are
recorded here so it is not rediscovered:

- Centre the ball at `f`, radius `min(|c − f|, r_clip_max)`. The segment centroid then sits exactly on
  the ball's boundary, so it is an extreme point of the ball. It can therefore never lie inside the
  convex hull of the clipped volume, and **a wall always exists** — the existence gap of
  centroid-centring does not arise.
- At ball radius equal to the obstacle radius the construction is **exact**: the ball is inscribed
  in the tube's cross-section, so the clipped volume is the ball and the wall sits on the true tube
  surface.
- **Below that radius it is OPTIMISTIC.** The ball lies wholly inside the tube, the wall sits on
  the ball rather than on the tube, and the clearance it claims overstates the truth by exactly
  (obstacle radius − ball radius). Measured: 0.453 claimed against 0.252 true.
- It is pinned for one reason only: a ball centred on the centreline yields **at most one wall**,
  so it cannot represent several approaches. For the centroid-centred ball that count is now
  decided by the approach cut (2026-08-26: one wall per interior local maximum of the distance
  profile); no analogue has been worked out for centreline-centring.

**The floor `r_clip ≥ r` applies to centroid-centring too — corrected 2026-08-26.** This file
previously said the floor "belongs to centreline-centring alone" and that with the ball on the
segment centroid "there is nothing to floor". That was wrong, and the reasoning behind it — *an
empty clipped volume is the correct outcome* — applies to a ball that misses the tube, not to a
ball that is **inside** it. When `d < r` the centroid is within the KOZ, the unfloored ball of
radius `d` lies wholly inside the tube, and the clipped volume is the whole ball. Nothing is empty:
the wall gets built against the ball rather than against the obstacle, and it understates the
penetration by exactly `r − d`. The floor is what makes the clipped volume a piece of the
**obstacle** in that case.

## The outer approximation — what the code did before, and why this is not that

**This section is history, not the construction.** It is kept because the code implemented it for
months, the paper described it, and its failure mode is the reason the real construction is stated
the way it is below. Nothing here is a step of the recipe.

Two facts are true and useful, and they are what made the route tempting. Hulling and inflating
**commute**: the convex hull of a tube equals the convex hull of its centreline, inflated by the
same radius. And if the obstacle's motion is polynomial, its lifted centreline is itself a Bézier,
so **De Casteljau subdivision** hands you exact control points for the stretch of centreline a
parameter interval selects — no sampling anywhere.

**What is *not* true is that this reproduces the clipped KOZ volume.** Intersection distributes
over neither hulling nor inflation. The clipped KOZ volume is
(centreline ⊕ ball of radius `r`) ∩ `B(c, r_clip)`; convexifying the selected centreline stretch and
inflating by `r` gives a set that **strictly contains** it. It is an **outer approximation**, and
the earlier claim in this file that it was *exact* was wrong.

**Measured, on the pre-2026-08-26 code** — scenario `curve`, SCP iteration 1, all six (segment,
obstacle) pairs: the set the plane was actually built against reaches **1.13 to 2.23 further from
the segment centroid than the clip radius**, i.e. as much as 4.7 times that radius. The ball never
truncated anything. **It only picked a parameter range.**

**And that is why it can never yield more than one wall.** Selecting a parameter interval and then
hulling it fuses whatever the ball caught into a single convex lump — the two arms of a bend and
everything between them. The gap the curve could pass through disappears into the hull. Clipping
the **volume** keeps the gap, which is the whole reason the construction below replaces this one.

**Do not sample the centreline** *(for the hull route)*. A curve bulges outside the chords between
its samples, so the hull of sampled points need not contain the tube — containment fails and the
certificate with it. Recovering it costs a chord-error inflation. Subdividing the obstacle's own
Bézier removes the error instead of bounding it.

### The wall, built against the clipped KOZ volume itself

Per (segment, obstacle), and per **local approach** of the obstacle to the segment — the
parameter band cut at every interior local maximum of `|γ(τ) − c|`, one piece per dip of the
distance profile:

- `L` — the **clipped KOZ volume**: KOZ ∩ `B(c, r_clip)`, the ball centred at the segment centroid with
  the clip radius above
- `y*` — the point of `L` closest to `c`
- `n = (c − y*) / |c − y*|` — the unit direction pointing from that closest point back to the
  centroid
- `b` — the **largest value of `n·z` over all points `z` of `L`**: the support of the clipped KOZ
  volume in direction `n`. The plane therefore sits on the volume's most protruding point along
  `n`, not on the point nearest the centroid
- the wall is **`n·z ≥ b`**, imposed on **every control point of the segment**

**Convexity of `L` is irrelevant to CONTAINMENT, and that is precisely why this is well-posed.**
The support function of a set and of its convex hull are identical in every direction — the
furthest a set reaches along `n` is the furthest its convex hull reaches along `n`, because hulling
adds no new extreme point. So the wall built against `L` and the wall built against the convex hull
of `L` are **the same wall**, and `L ⊆ {z : n·z ≤ b}` holds whether or not `L` is convex.
Convexifying `L` buys nothing and is never required.

**Convexity is not irrelevant to SEPARATION, and the two must not be conflated.** Containment says
the piece is on the forbidden side. It does not say the centroid is on the free side. When a piece
wraps around `c` the support reaches past the centroid and `n·c < b`. **That is a legal wall with a
negative margin, not a failure.** The row then reads "you are this far in, climb out along `n`",
which is exactly what a sequential convex solver with elastic slack consumes — the deep-penetration
case (centroid inside the KOZ) measures `−0.889` and is exactly this.

**Wrapping is also why the grouping is per approach and not per connected component.** A bend can
curl around the centroid while staying ONE connected lump; the centroid then sits inside the lump's
convex hull, so a single wall against it is non-separating **by theorem**, whatever its normal.
Measured through the builder: the fused wall reported a centroid margin of `−1.154` on a segment
whose centroid is `0.205` **outside** the keep-out zone — a clear segment read as 1.15 deep, and a
row unsatisfiable inside the default trust radius (it needed `≈1.04` against `0.5`), leaving the
elastic slack to absorb a violation that did not exist. Cutting the band at the crest between the
two approaches — same clip radius, same clipped volume, same offset rule — gives two walls with
margins `+0.529` and `−0.162`; the residual `−0.162` is the support ceiling's conservatism, a
step-size cost, not a correctness cost.

**The plane must not be taken tangent to the tube.** A plane tangent to the tube at its closest
surface point is valid only if the tube is convex. Measured counterexample, in the formal section:
with the obstacle turning and the segment inside the turn, 12.35% of the clipped KOZ volume sits on
the *allowed* side of the tangent plane. *(That measurement compared the tangent plane against the
hull-of-band plane, which is the outer approximation this section just retired. The 12.35% figure
stands as evidence that the tangent plane is invalid; the accompanying "1.19° in normal, 0.76 in
offset" describes the distance to the **old** plane and is **stale** — re-measure it against the
support plane before quoting it.)*

One plane per (segment, obstacle, approach), never one per control point.
Per-control-point planes prove only that each point individually is outside its own plane — exactly
as strong as sampling the curve — and opposing normals let the hull wrap around the tube.

The plane's normal has a nonzero **time component** because the centreline is slanted in the lifted
space. It is not optional; without it the plane does not support the tube at all. *For the
constant-velocity special case that component reduces to minus the dot product of the spatial
normal with the obstacle's velocity. That expression is the special case, not the definition.*

### When the segment centroid is inside the KOZ

Deep penetration is a real case, and it needs **no separate recipe** — corrected 2026-08-26. This
file previously ruled that `c` be projected onto the un-inflated centreline hull and the offset
pushed out by the obstacle radius, a deliberate fallback to the outer approximation. That is
retired along with the rest of the hull route.

**`n` never actually depends on `y*`.** Let `f` be the component's own nearest centreline point.
The nearest point of the KOZ to `c` is `f + r·(c − f)/|c − f|`, which lies **on the ray from `f`
to `c`**, at distance `d − r` from `c`; since the clip radius is floored at `r` we have
`d − r ≤ r_clip`, so that point is inside the clip ball and is therefore `y*`. Hence

$$n \;=\; \frac{c - y^\star}{\lVert c - y^\star\rVert} \;=\; \frac{c - f}{\lVert c - f\rVert}$$

wherever the quotient is defined at all. When `c` is inside the KOZ, `y* = c` and the written
quotient goes `0/0` — but the **ray does not vanish with it**. Taking the direction directly from
`f` is continuation of the same formula, not a fallback to a different set. The only genuinely
undefined case is `c` landing exactly on the centreline, where every direction is equally good.

`b` is then the support of the component along that `n`, unchanged. Nothing about the wall
degenerates when the centroid is inside; only one way of writing the direction down does.
Measured: centroid `0.55` inside a tube of radius `0.9`, floored clip radius `0.9` — normal exactly
`(1, 0)` by symmetry, offset `1.479`, centroid margin `−0.879`, signed clearance exactly
`0.35 − 0.9 = −0.55`.

**Refusing a wall here was a measured defect, and that has not changed.** On `diverse` at 8
segments it left the single penetrating (segment, obstacle) pair with no row at all, and the
constraint-residual certificate then reported 4.6e-13 — clean — for a trajectory penetrating by
0.219. **A component in reach always yields a wall.** The only outcome that legitimately carries
none is the clip ball missing the KOZ entirely.

### What conservatism costs, and what it does not

**A conservative wall is a step-size cost, not a correctness cost.** This is sequential convex
programming: the wall is rebuilt at every iteration around the new iterate. If the curve wants to
come closer than one iteration's wall allows, it moves as far as that wall permits, the wall is
rebuilt closer, and the conservatism shrinks. Over-conservatism buys extra iterations. It does not
buy a wrong answer.

**So the two things that matter about a wall are that it EXISTS and that it is SOUND.** A wall that
is missing leaves a pair unconstrained and the certificate then certifies nothing about it; a wall
that is optimistic claims clearance the trajectory does not have. Both are correctness failures.
Tightness is not. **This reorders which failure modes this paper should worry about**, and it is
the reason the existence gap of centroid-centring, and the optimism of a sub-radius centreline
ball, are recorded above as the properties that decide the question.

## The hole, stated as a hole

The wall protects against tube material inside the clipping ball. **Tube material lying outside the
ball *and* on the safe side of the wall is not protected** — and at a bend, the far arm is exactly
that.

With the volume **genuinely clipped**, coverage is exactly one condition and nothing else:

> **clip radius `r_clip` ≥ `E + Δ√(d_spatial+1)`** — at least the segment's own radius about its centroid, plus
> the furthest one trust step can carry a control point.

Set `r_clip` to the tangent value `d` and that fails precisely when the segment gets close to the
obstacle, which is when the constraint matters. **There is no accidental extra coverage to lean
on**: the moment the wall is built against the clipped volume rather than against the overshooting
band, the band's 1.13–2.23 of reach beyond the clip radius stops contributing anything.

**Measured, on the current code, whose set is the band and not the clipped volume** — `curve`,
iteration 1, the same six (segment, obstacle) pairs: the band covers 100%, 86.9%, 100%, 100%,
61.1% and 99.8% of the true tube material lying inside the reach ball. So **today the soundness
flag is more pessimistic than today's geometry** — it reports "not covered" on pairs the band
happens to cover in full, and it cannot report the two pairs where coverage is genuinely partial as
anything more specific. After the fix the flag and the geometry agree: what the flag reports is
what the wall actually protects. *(These percentages describe the pre-fix code. They are diagnostic
numbers, not paper numbers, and must be re-measured after the change.)*

**So this construction is not sound by construction.** It is adopted because its failure is
**detectable**, by the check below.

**It is one line from being sound.** The one-line repair in the formal section: raise the clip
radius' lower clamp from `r` to `E + Δ√(d_spatial+1)` instead of letting it shrink to the tangent
value, and the certificate covers the full tube unconditionally. *(That is a **different, larger**
floor than the `r` floor in "The clip" above. The `r` floor stops the ball being smaller than the
obstacle when the centroid is inside; this one stops the segment reaching tube material outside the
ball. Applying the first does not give you the second.)* The price is conservatism exactly where the
constraint is active — and by the argument above, conservatism is a step-size cost, not a
correctness cost. **Which of the two to use is an open experimental question** — the tangent form
goes in first, and the fallbacks stand: more segments, centre on the contact point, reach
filtering, adaptive growth. If a reviewer asks for a guarantee, there is not one — there is a
check. Say so in the limitations section rather than letting it be found.

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
about correctness must be traceable to a statement here. **The statements carry no numbers** —
refer to each by what it says ("the result that a Bézier segment lies in the hull of its control
points"), not by a label, so that inserting or removing one cannot silently invalidate a
cross-reference elsewhere.

## Notation

**Spatial dimension $d_{\mathrm{spatial}}$**, spelled out because the bare $d$ is the
centroid-to-centreline distance. The lifted space is $\mathbb{R}^{d_{\mathrm{spatial}}+1}$ with a point written
$z=(x,\tau)$, $x\in\mathbb{R}^{d_{\mathrm{spatial}}}$, $\tau\in\mathbb{R}$. **The axis scale is $1$**, so the
Euclidean norm $\lVert\cdot\rVert$ on $\mathbb{R}^{d_{\mathrm{spatial}}+1}$ mixes space and time on equal terms.
Every ball, distance and projection below is taken in that norm. This is the formal content of
"no axis is privileged" — remove it and none of what follows is defined.

Decision variable: control points $P_0,\dots,P_N\in\mathbb{R}^{d_{\mathrm{spatial}}+1}$, $P_j=(x_j,t_j)$, stacked
into $\mathbf p\in\mathbb{R}^{(N+1)(d_{\mathrm{spatial}}+1)}$.

Curve, with $u\in[0,1]$ the curve parameter — **not** time:

$$Z(u)=\sum_{j=0}^{N}B_j^N(u)\,P_j,\qquad B_j^N(u)=\binom{N}{j}u^j(1-u)^{N-j},\qquad
\tau(u)=\sum_{j=0}^{N}B_j^N(u)\,t_j .$$

De Casteljau subdivision into $M$ segments gives row-stochastic matrices $A^{(k)}$
($A^{(k)}_{ij}\ge 0$, $\sum_j A^{(k)}_{ij}=1$) with segment control points and centroid

$$Q^{(k)}_i=\sum_j A^{(k)}_{ij}P_j,\qquad
c^{(k)}=\tfrac1{N+1}\sum_i Q^{(k)}_i=\sum_j w^{(k)}_j P_j,\quad w^{(k)}_j\ge0,\ \textstyle\sum_j w^{(k)}_j=1 .$$

**Hull property.** For $u$ in segment $k$'s parameter interval, $Z(u)\in\operatorname{conv}\{Q^{(k)}_i\}_{i=0}^{N}$ — *a Bézier segment lies in the convex hull of its own control points.*

Segment radius: $E^{(k)}=\max_i\lVert Q^{(k)}_i-c^{(k)}\rVert$.

## Obstacle

Obstacle $m$ has known motion $\pi_m$ on $[T^0_m,T^1_m]$. Lifted centreline and keep-out zone:

$$\gamma_m(\tau)=\bigl(\pi_m(\tau),\tau\bigr),\qquad
\Gamma_m=\gamma_m\!\left([T^0_m,T^1_m]\right),\qquad
\mathcal K_m=\Gamma_m\oplus \bar B(0,r_m).$$

$\Gamma_m$ is a curve, so $\mathcal K_m$ is **not convex** unless $\pi_m$ is affine. It therefore
admits no supporting half-space covering the whole of it, and the hull property alone certifies
nothing against it.

## The clip

With $c=c^{(k)}$ fixed at the reference iterate:

$$\tau^\star\in\arg\min_{\tau\in[T^0_m,T^1_m]}\lVert c-\gamma_m(\tau)\rVert,\qquad
f=\gamma_m(\tau^\star),\qquad d=\lVert c-f\rVert,$$

$$\boxed{\ r_clip=\operatorname{clip}\bigl(d,\;r_m,\;r_{\mathrm{clip,max}}\bigr)
=\max\bigl(\min(d,\,r_{\mathrm{clip,max}}),\;r_m\bigr),\qquad
r_{\mathrm{clip,max}}=r_m+E^{(k)}+\Delta\sqrt{d_{\mathrm{spatial}}+1}\ }$$

$$\mathcal L^{(k)}_m=\mathcal K_m\cap \bar B(c,r_clip)\qquad\text{(the \textbf{clipped KOZ volume}).}$$

In words: $d$ is the distance from the segment centroid to the nearest point of the obstacle's
lifted centreline; $r_{\mathrm{clip}}$ is the clipping radius, that distance capped above and
floored below; the clipped KOZ volume is the part of the keep-out zone that lies inside the ball of
radius $r_{\mathrm{clip}}$ about the centroid. When the floor is inactive, $\bar B(c,d)$ is tangent
to $\Gamma_m$ at $f$. $\tau^\star$ need not be unique; any minimiser serves, and non-uniqueness is
exactly the bend case that makes $f$ discontinuous in $\mathbf p$.

**The floor is active exactly when $d<r_m$.** Since
$r_{\mathrm{clip,max}}=r_m+E^{(k)}+\Delta\sqrt{d_{\mathrm{spatial}}+1}\ge r_m$, we have
$\min(d,r_{\mathrm{clip,max}})<r_m \iff d<r_m$, i.e. iff $c\in\mathcal K_m$. Elsewhere
$r_{\mathrm{clip}}=\min(d,r_{\mathrm{clip,max}})$ unchanged. Its purpose is to keep
$\mathcal L^{(k)}_m$ a piece of the **obstacle** rather than a ball strictly inside it: at $d<r_m$
the unfloored $\bar B(c,d)\subset\mathcal K_m$, so $\mathcal L^{(k)}_m=\bar B(c,d)$ and the support
plane sits on the ball, understating the penetration by $r_m-d$.

**Two different floors appear in this document and they are not the same number.** $r_m$, boxed
above, is part of the construction and always applied. The larger floor
$E^{(k)}+\Delta\sqrt{d_{\mathrm{spatial}}+1}$ appears in the soundness section below; it is
**optional**, it addresses a different failure (the next iterate leaving the clipped ball), and it
is reachable in the code as a flag rather than as a default. Do not conflate them.

**One wall per local approach.** Cut the band $I$ at every interior local maximum of
$\tau\mapsto\lVert\gamma_m(\tau)-c\rVert$; each resulting sub-interval $I_\ell$ is one
**approach** of the obstacle to the segment, and defines the piece

$$\mathcal L^{(k)}_{m,\ell}=\bigl(\gamma_m(I_\ell)\oplus\bar B(0,r_m)\bigr)\cap\bar B(c,r_{\mathrm{clip}}),
\qquad \mathcal L^{(k)}_m=\textstyle\bigcup_\ell\mathcal L^{(k)}_{m,\ell}.$$

**Each approach gets its own wall.** One wall per $(k,m,\ell)$, never one wall per $(k,m)$. This
covers both failure modes with one rule: two separated lumps in the ball are two approaches (the
distance profile leaves and re-enters the band), and a single connected lump that wraps the
centroid is ALSO two approaches (the profile dips twice with a crest between). Fusing approaches is
not a conservative simplification; it is the opposite of what is wanted, because the fused wall
spans the corridor — or the mouth of the bend — the trajectory is entitled to pass through, and on
a wrapping lump it is non-separating by theorem. The cut position is a quality knob, never a
correctness one: the pieces cover $\mathcal L^{(k)}_m$ for ANY partition of the band, so both
over-splitting and under-splitting keep every wall sound.

## The outer approximation, and why it is not the clipped volume

**Containment of the clip in a centreline band.** Let
$I=\{\tau:\lVert\gamma_m(\tau)-c\rVert\le r_clip+r_m\}$. Then

$$\mathcal K_m\cap\bar B(c,r_clip)\ \subseteq\ \gamma_m(I)\oplus\bar B(0,r_m).$$

*Proof.* Let $z$ be in the left side. Some $\tau$ has $\lVert z-\gamma_m(\tau)\rVert\le r_m$, and
$\lVert z-c\rVert\le r_clip$, so $\lVert\gamma_m(\tau)-c\rVert\le r_m+r_clip$, i.e. $\tau\in I$. $\square$

**Hulling and inflating commute.** $\operatorname{conv}(S\oplus\bar B(0,r))=\operatorname{conv}(S)\oplus\bar B(0,r)$ — so
convexifying the **centreline** and inflating is the same as convexifying the tube.

**Exact convexification of the band when the motion is polynomial.** If $\pi_m$ has degree $N_m$
then $\gamma_m$ is a Bézier curve in $\mathbb{R}^{d_{\mathrm{spatial}}+1}$ with control points $G_0,\dots,G_{N_m}$.
Decompose $I$ into its maximal intervals $I_1,\dots,I_C$; for each interval take
$[\alpha_\ell,\beta_\ell]=[\min I_\ell,\max I_\ell]$ and let $\tilde G^{(\ell)}_0,\dots,\tilde
G^{(\ell)}_{N_m}$ be its De Casteljau subdivision control points. By the hull property applied to
the obstacle, $\gamma_m([\alpha_\ell,\beta_\ell])\subseteq\mathcal
G_\ell:=\operatorname{conv}\{\tilde G^{(\ell)}_l\}$, and

$$\mathcal H_\ell=\mathcal G_\ell\oplus\bar B(0,r_m)$$

is convex. **Never take $[\min I,\max I]$ across all of $I$**: that bridges the gaps between
approaches and fuses walls that must stay separate.

**$\mathcal H_\ell$ is an OUTER APPROXIMATION of the clipped KOZ volume, not equal to it.** The
inclusion $\mathcal L^{(k)}_m\subseteq\bigcup_\ell\mathcal H_\ell$ holds, and it is **strict in
general**: intersection distributes over neither the hull nor the Minkowski sum, so
$(\Gamma_m\oplus\bar B(0,r_m))\cap\bar B(c,r_clip)$ is strictly smaller than
$\operatorname{conv}(\text{selected centreline})\oplus\bar B(0,r_m)$. **The ball selects a
parameter range; it does not cut the set.** Measured on the current code, scenario `curve`,
iteration 1, all six $(k,m)$ pairs: $\mathcal H$ extends $1.13$ to $2.23$ beyond the clip radius
$r_{\mathrm{clip}}$ — up to $4.7r_clip$. *(Pre-fix diagnostic; re-measure after the change.)*

**No sampling anywhere.** Hulling sampled centreline points would give a set that a curve bulges
*outside of*, breaking containment; recovering it needs a chord-error inflation. Subdividing the
obstacle's own Bézier removes the error rather than bounding it.

## The wall — a support of the clipped KOZ volume

Work per approach; write $\mathcal L$ for one piece $\mathcal L^{(k)}_{m,\ell}$ — the tube over
one sub-interval of the band, clipped — and assume $c\notin\mathcal L$. Set

$$\boxed{\ y^\star\in\arg\min_{y\in\mathcal L}\lVert c-y\rVert,\qquad
n=\frac{c-y^\star}{\lVert c-y^\star\rVert},\qquad
b=\max_{z\in\mathcal L} n^\top z\ =\ h_{\mathcal L}(n)\ }$$

In words: $y^\star$ is the point of the clipped KOZ volume nearest the segment centroid; $n$ points
from it back to the centroid; and $b$ is the **support** of the clipped KOZ volume in direction
$n$ — the largest value the linear functional $n^\top z$ takes on it. The plane $n^\top z=b$
therefore rests on the volume's most protruding point along $n$, which is in general **not**
$y^\star$.

**Support functions do not see convexity.** For any compact set $S$ and any direction $n$,

$$h_S(n)=\max_{z\in S}n^\top z=\max_{z\in\operatorname{conv}S}n^\top z=h_{\operatorname{conv}S}(n),$$

because the maximum of a linear functional over the hull of a compact set is attained at an extreme
point of that hull, and every extreme point of $\operatorname{conv}S$ belongs to $S$ itself. *In
words: hulling never lets a set reach further in any direction, so a set and its convex hull have
identical supports.* **Consequence: the wall built against
$\mathcal L$ and the wall built against $\operatorname{conv}\mathcal L$ are the same wall.**
Convexifying $\mathcal L$ costs nothing and gains nothing, and **the convexity of $\mathcal L$ is
irrelevant to the construction** — this is why no convex over-approximation of the clipped volume
is needed to define the plane.

**Half-space containment.** $\mathcal L\subseteq\{z:n^\top z\le b\}$, immediately from the
definition of $b$; and by the identity above, $\operatorname{conv}\mathcal L$ satisfies the same
inclusion. No convexity assumption on $\mathcal K_m$ is used anywhere in this step.

**This is the step the tangent plane fails.** Taking $n$ from $c-f$ and $b$ from the tube surface
at $f$ is valid only when $\mathcal K_m$ is convex, because then and only then is the tube's
tangent plane a support of it. Measured counterexample: obstacle on a circle of radius $10$ at
$0.5$ rad/s, $r_m=1$, $c$ inside the turn — $12.35\%$ of the clipped KOZ volume lands strictly on
the safe side of the tangent plane, worst point $0.77$ inside it and $0.99$ from the centreline.
*(The accompanying "normal moves $1.19^\circ$, offset moves $0.76$" was measured against the
projection onto $\mathcal G$, i.e. against the outer approximation, and is **stale** for the
support plane. Re-measure before quoting.)*

**The direction does not depend on $y^\star$.** Let $f_\ell=\gamma_m(\tau^\star_\ell)$ be the
nearest centreline point *within sub-interval $\ell$*. If $c\notin\mathcal K_m$, the nearest point of
$\mathcal K_m$ to $c$ is $f_\ell+r_m(c-f_\ell)/\lVert c-f_\ell\rVert$, at distance
$\lVert c-f_\ell\rVert-r_m$ from $c$; the floor $r_{\mathrm{clip}}\ge r_m$ puts it inside
$\bar B(c,r_{\mathrm{clip}})$, so it is $y^\star$, and it lies on the segment from $f_\ell$ to $c$.
Hence

$$\boxed{\ n=\frac{c-y^\star}{\lVert c-y^\star\rVert}=\frac{c-f_\ell}{\lVert c-f_\ell\rVert}\ }$$

whenever the left-hand quotient exists at all. Two consequences:

- **Centroid inside the KOZ (deep penetration).** Then $y^\star=c$ and the left-hand quotient is
  $0/0$, but the right-hand one is not: the direction is still $c-f_\ell$ normalised. **No fallback
  set, no hull projection, no separate recipe** — this replaces the earlier rule that projected $c$
  onto the un-inflated centreline hull $\mathcal G_\ell$, which is retired with the rest of the
  outer approximation. $b$ remains the support of $\mathcal L^{(k)}_{m,\ell}$ along $n$. The only
  undefined case is $c=f_\ell$, i.e. the centroid exactly on the centreline, where every direction
  is equally valid. **An approach in reach always yields a wall. Refusing one was a measured
  defect** — on `diverse` at $8$ segments the single penetrating pair got no row, and the
  constraint-residual certificate reported $4.6\times10^{-13}$ for a trajectory penetrating by
  $0.219$.
- **Wrapping is not an existence problem — and it is a grouping problem the cut solves.**
  $\mathcal L^{(k)}_{m,\ell}$ may reach past $c$ along
  $n$, giving $n^\top c<b$. Containment still holds — that is the inclusion above, which uses no
  convexity — so the wall is valid; the margin is simply negative, and a negative margin is the row
  telling the solver how far it has to climb out. What fails in that case is *separation*, not
  existence, and separation is not what the certificate rests on. Centring the ball on $f$ instead
  would make $c$ an extreme point of the ball and force $n^\top c\ge b$, but it yields **at most one
  wall**; that variant is **pinned**, with its measured properties, in the prose section "Where the
  ball is centred".

## Rows and the segment certificate

Impose on **every** control point of segment $k$:

$$n^\top Q^{(k)}_i\ \ge\ b,\qquad i=0,\dots,N
\qquad\Longleftrightarrow\qquad
n^\top\!\!\sum_j A^{(k)}_{ij}P_j\ \ge\ b .$$

**Segment certificate.** If those $N+1$ rows hold, then $n^\top Z(u)\ge b$ for every $u$ in
segment $k$, hence $Z(u)\notin\operatorname{int}\mathcal L$ — *the curve does not enter the
interior of the clipped KOZ volume.*

*Proof.* By the hull property, $Z(u)=\sum_i\lambda_iQ^{(k)}_i$ with $\lambda\ge0$,
$\sum\lambda_i=1$; linearity gives $n^\top Z(u)\ge b$. If $z\in\operatorname{int}\mathcal L$ then
some ball about $z$ lies in $\mathcal L$, so $n^\top z+\varepsilon\le b$ for some $\varepsilon>0$,
i.e. $n^\top z<b$. A point with $n^\top Z(u)\ge b$ is therefore not an interior point of
$\mathcal L$. $\square$

*Note what this proof does not use: nowhere does it assume $\mathcal L$ is convex. Only the hull
property, linearity, and the definition of the support $b$ are used.*

One plane per $(k,m,\ell)$ — segment, obstacle, approach. Per-control-point planes
destroy the certificate: the step from $Q_i$ to $Z(u)$ uses the *same* $n$ for all $i$, and with
differing normals the hull can wrap the tube.

## Reachability, and exactly when the certificate covers $\mathcal K_m$

The segment certificate certifies against $\mathcal K_m\cap\bar B(c,r_clip)$, **not** against
$\mathcal K_m$. The difference is closed by the trust region and nothing else.

**Reachability.** If $\lVert\mathbf p^{+}-\mathbf p\rVert_\infty\le\Delta$ then for every $u$
in segment $k$, $Z^{+}(u)\in\bar B\!\left(c,\;E^{(k)}+\Delta\sqrt{d_{\mathrm{spatial}}+1}\right)$.

*Proof.* $Q^{+}_i-Q_i=\sum_jA^{(k)}_{ij}(P^{+}_j-P_j)$ is a convex combination of vectors of
$\infty$-norm at most $\Delta$, so $\lVert Q^{+}_i-Q_i\rVert_\infty\le\Delta$ and
$\lVert Q^{+}_i-Q_i\rVert\le\Delta\sqrt{d_{\mathrm{spatial}}+1}$. Hence
$\lVert Q^{+}_i-c\rVert\le E^{(k)}+\Delta\sqrt{d_{\mathrm{spatial}}+1}$, and the ball is convex, so it contains
$\operatorname{conv}\{Q^{+}_i\}\ni Z^{+}(u)$. $\square$

**Soundness condition.**

$$\boxed{\ r_clip\ \ge\ E^{(k)}+\Delta\sqrt{d_{\mathrm{spatial}}+1}\ }$$

*In words: the clipping radius is at least the segment's own radius about its centroid plus the
furthest one trust step can carry a control point.* Under it, $\bar B(c,r_clip)$ contains the whole
next-iterate segment, so the segment certificate certifies against the **full** $\mathcal K_m$.
Note it does not involve $r_m$ or the obstacle at all.

**This is not the floor in the boxed clip radius, and the two must not be merged.** The
construction floors $r_{\mathrm{clip}}$ at $r_m$; that floor is unconditional, it binds only when
$c\in\mathcal K_m$, and it exists so the clipped volume is a piece of the obstacle rather than a
ball inside it. The condition here is a *larger*, *optional* floor at
$E^{(k)}+\Delta\sqrt{d_{\mathrm{spatial}}+1}$, addressing a different failure: material of
$\mathcal K_m$ that the next iterate can reach but that lies outside $\bar B(c,r_{\mathrm{clip}})$
and is therefore constrained by nothing. Applying $r_m$ does not imply this one, and the gap
between them is the regime table below.

**With the wall built against the clipped KOZ volume, this condition is exactly coverage — there
is nothing else.** Under the retired outer approximation the wall also happened to hold back
material out to $\mathcal H$'s overshoot, so the flag could be pessimistic relative to the
geometry; measured on the current code, the band covers $100\%$, $86.9\%$, $100\%$, $100\%$,
$61.1\%$ and $99.8\%$ of the true tube material inside the reach ball over the six pairs of
`curve` at iteration 1. Once the wall is the support of $\mathcal L$, no such accidental coverage
exists and flag and geometry agree. *(Pre-fix diagnostic; re-measure after the change.)*

Four regimes for the construction's radius
$r_clip=\operatorname{clip}(d,\ r_m,\ r_{\mathrm{clip,max}})$:

| regime | $r_{\mathrm{clip}}$ | sound? |
|---|---|---|
| $d>r_{\mathrm{clip,max}}$ — obstacle out of reach | $r_{\mathrm{clip,max}}=r_m+E+\Delta\sqrt{d_{\mathrm{spatial}}+1}$ | **yes**, and no row is needed: the segment cannot reach $\mathcal K_m$ in one step. (The clip is not yet *empty* here — that happens only past $d=r_{\mathrm{clip,max}}+r_m$ — so this is a reachability test, not an emptiness test.) |
| $E+\Delta\sqrt{d_{\mathrm{spatial}}+1}\le d\le r_{\mathrm{clip,max}}$ | $d$ | **yes** |
| $r_m<d<E+\Delta\sqrt{d_{\mathrm{spatial}}+1}$ — segment close | $d$ | **no — this is the hole** |
| $d\le r_m$ — centroid inside the KOZ | $r_m$ (**floor active**) | **no** unless $r_m\ge E+\Delta\sqrt{d_{\mathrm{spatial}}+1}$; the floor fixes the *understated penetration*, not the coverage |

So the cap is free: it only acts where the row cannot bind, and it preserves soundness. The hole is
at *small* $d$, i.e. exactly when the constraint binds — and the $r_m$ floor does not close it,
because $r_m$ and the reach are unrelated quantities.

**The one-line repair, and its price.** Raising the floor from $r_m$ to the reach,

$$r_clip=\operatorname{clip}\!\left(d,\ \max\bigl(r_m,\ E^{(k)}+\Delta\sqrt{d_{\mathrm{spatial}}+1}\bigr),\ r_{\mathrm{clip,max}}\right)$$

makes the soundness condition hold unconditionally and the construction **sound by construction**.
It costs conservatism precisely where the constraint is active, because the ball is then larger
than the distance to the obstacle and more of the tube is clipped in. By the step-size argument in
the prose, that conservatism is paid in iterations, not in correctness. **Which of the two is used
is an open experimental question, not a settled one** — it is reachable in the code as the
`sound_clip` flag, and the count of pairs failing the condition is exported per iteration as
`unsound_clips` so the hole is a number rather than a caveat.

## The subproblem

$$\min_{\mathbf p,\ \sigma\ \ge 0}\quad
\tfrac12\,\mathbf p^\top H\,\mathbf p\;+\;w_t\,t_N\;+\;\lambda\!\sum\sigma$$

subject to, with $\mathbf p^{\mathrm{ref}}$ the reference iterate:

| | |
|---|---|
| boundary | $P_0=z^{\text{start}}$; $x_N=x^{\text{goal}}$; $t_N$ free iff free-arrival |
| monotonicity | $t_{j+1}-t_j\ \ge\ \delta>0$ |
| slant limit | $\lVert x_{j+1}-x_j\rVert\ \le\ v_{\max}\,(t_{j+1}-t_j)$ — second-order cone, **no slack** |
| keep-out | $n_{km\ell}^\top Q^{(k)}_i+\sigma_{km\ell i}\ \ge\ b_{km\ell}$ — one row block per segment, obstacle **and approach** $\ell$ |
| occlusion | same form, per $(k,m,\ell,\text{station})$ |
| trust region | $\lVert\mathbf p-\mathbf p^{\mathrm{ref}}\rVert_\infty\ \le\ \Delta$ |

**$H$ is blind to timing.** It is assembled only from second differences of the spatial
coordinates, so $He_{t_j}=0$ for every time-coordinate basis vector: two control-point sets with
equal spatial coordinates and any time coordinates score identically. *Checked by test.*

**The slant limit is sufficient.** $x'(u)=N\sum_{j}B^{N-1}_j(u)(x_{j+1}-x_j)$ and
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
own existence window. It touches neither $n$, $b$, $\mathcal L$, $r_{\mathrm{clip}}$, nor the axis scale, so it
is independent of everything it checks. The run is feasible iff $\mu>0$.

$\mu$ is the only statement made about $\mathcal K_m$ itself. Everywhere the soundness condition
fails, it is the only statement there is.

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
volume in the lifted space is **not** convex. It gets the identical treatment as the tube, and that
treatment is now the corrected one: clip the shadow volume to the same local ball about the segment
centroid to get the **clipped shadow volume**; take the point of that volume nearest the centroid;
take the unit direction from it back to the centroid; and set the offset to the **support of the
clipped shadow volume in that direction** — its most protruding point along the normal. One wall
per time-windowed piece of the shadow, imposed on every control point of the segment. *(The KOZ
side's approach cut — one wall per interior local maximum of the distance profile, 2026-08-26 —
has no occlusion analogue yet: a shadow piece is one time window, not one approach.)*

**Do not take the convex hull of the clipped shadow volume and build the plane against that.** As with the
KOZ, that hull is a strict **outer approximation** of the clipped shadow volume, because
intersection distributes over neither hulling nor inflation. It is also unnecessary: a set and its
convex hull have the same support in every direction, so the wall against the clipped shadow volume
*is* the wall against its hull. **Convexity of the clipped shadow volume is not required by the
construction** — which is what makes the non-convexity measured below survivable rather than fatal.

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

# Open

- **Venue.** Both, or 항공우주 only. Decides the title (mission-first for 항공우주, method-first
  for 기계학회 — the PI supplied a candidate for each) and it is on the 8/26 clock.
- **`CLAUDE.md` §"Formulation decisions" item 8** still asserts the keep-out tube is a capsule and
  therefore convex, which is now false. Needs the same correction as this file; not done here.
- **Occlusion mission motivation.** The PI's review requires a statement of *which mission, and
  why* line-of-sight maintenance is needed in aerospace practice. Currently the constraint is
  presented only as a methodological differentiator. Required in both manuscript and talk.
