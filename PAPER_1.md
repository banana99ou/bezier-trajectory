# Paper 1 — offline space-time trajectory optimization, known obstacle motion

**Central doc for paper 1.** Everything about this paper lives here or is linked from here.

**Scope boundary.** Obstacle motion is known and deterministic. The solve is offline. There is no
sensing model and nothing to re-estimate, which is why receding-horizon replanning is explicitly
declined in Part A §7. The online / uncertain-hazard regime is a *separate* paper:
[`PAPER_2.md`](PAPER_2.md). Stating this assumption plainly is what makes paper 2's contribution
legible as a contribution rather than a correction.

**Supporting evidence, kept separate on purpose** (it is a verification record, not paper content):
- [`doc/refs/novelty_positioning.md`](doc/refs/novelty_positioning.md) — prior-art map, comparison against Osburn, per-source verification status
- [`doc/notes/_shared/safe_corridor_references.md`](doc/notes/_shared/safe_corridor_references.md) — external ground truth for one plane per (segment, obstacle)

**Structure of this file**
- **Part A** — claim, what is different, method, demo scenarios, the shadow lemma, the risk the contribution creates, decided-against, what must be measured, weakest links. *(English; was `PAPER_CLAIM.md`.)*
- **Part B** — 논문 뼈대: section-by-section skeleton for the 2-page paper. *(Korean; was `PAPER_OUTLINE.md`.)*

Where Part A and Part B disagree, Part B §핵심 기여 is the authority — it is the section that was
fixed first, and the rest is aligned to it.

---

# Decisions — 2026-08-19

Taken in conversation. Recorded here because they change what Part A and Part B are
allowed to claim. **Part A below is unmodified** — it is byte-identical to the
committed `PAPER_CLAIM.md` — so where this section and Part A disagree, this section
is newer. The full formulation reasoning lives in `CLAUDE.md` §"Formulation
decisions"; this is the paper-facing half.

## The one-liner

Causal, not a list. The mechanism is decomposition-free; three spatial dimensions
plus time and the occlusion constraint are its consequences; freeing the arrival
time removes a weakness and is not a claim.

> Because moving obstacles enter as linearized supporting half-spaces on a lifted
> space-time tube rather than a precomputed convex cell complex, the formulation
> runs at three spatial dimensions plus time, and absorbs constraints — such as
> line-of-sight occlusion by a moving obstacle — that a convex-decomposition
> framework structurally cannot express.

Never write "novel combination" and never claim the lift itself: Osburn et al.
published it in August 2025. Never open with "time as a coordinate."

## What the formulation change means for the paper

1. **"Minimizes spatial acceleration" is currently false of the code.** The
   objective measures bending against the curve *parameter*, not against real time,
   and those coincide only when timing is uniform — the case this method exists to
   escape. A plan that waits then dashes can score smooth while being physically
   violent. Do not write the sentence until the re-derivation lands.
2. **The speed cap enters as a constraint** — a slant limit in space-time — not as
   a cost term.
3. **Arrival time is freed**, and the objective becomes energy plus a weighted time
   penalty. Until then, §formulation must state plainly that both endpoints pin
   time, because the nearest prior work has arrival time free and a reviewer reads
   that column first.
4. **The energy-versus-time weight is a reported parameter**, not a tuned constant.
   Its sweep is a legitimate figure: arrival time against energy.
5. **No numbers until the freeze.** Every measurement in the repository predates the
   objective change. The draft the PI reads carries none — as the abstract slot in
   Part B already assumes.

## Waiting: shown, but not the differentiator

Decided 2026-08-19. The paper **does** state that the method produces waiting — it is true of the
code and it is the visible consequence of per-segment time extents being decision variables, so
Part B §핵심 기여's first contribution stands and the behaviour goes in the text.

What it must not do is carry the differentiation. Osburn et al. have finite-height prisms *and* a
free arrival time, so a wait-for-the-door trajectory comes out of their method too. Waiting
demonstrates that time allocation is being optimized; it does not demonstrate that anything new is
being done to obtain it. The differentiator is the mechanism: the obstacle's convex outer
approximation is rebuilt from the current iterate every SCP iteration rather than fixed before the
solve, and the segment time extents that approximation is built against are themselves decision
variables. Convex-set-graph methods fix their regions before solving; MADER fixes its intervals in
the knot vector.

So: claim waiting as a capability, claim decomposition-free as the contribution.

## The demo scenario and its figure

Decided 2026-08-19, in stages. The tube-chain geometry behind it is derived in
[`doc/notes/005_formulation_freeze.md`](doc/notes/005_formulation_freeze.md) Part 3 — that note
carries the math; this section carries every paper-facing choice.

**Occlusion goes into paper 1, with a moving occluder on a non-straight path.** The
static-occluder alternative was considered and rejected: a static shadow extruded along time is a
convex prism, which is exactly what a graphs-of-convex-sets method handles natively — cheap for
us *and* for the nearest prior work, so it demonstrates nothing. The twisting shadow of a moving
occluder is the case that shows the seam: one linearized row per segment-obstacle-station triple
for us, against a shadow that a convex decomposition must carve out of free space with a set
count that explodes. Only the moving case earns the second clause of the one-liner.

**The middle path: one figure carries both consequences.** Two pages hold one demonstrated
consequence, not two. So there is no separate dimensional-scaling figure: the occlusion demo
itself runs at three spatial dimensions plus time, and the one figure does double duty.

**The third dimension must be load-bearing, not decorative.** If the vehicle merely happens to be
in three dimensions, a reviewer reads it as a rendering choice. The scenario is built so that
going around loses contact and going over keeps it: the climb exists only because there is a
third spatial dimension, and the reason the climb happens is the occlusion constraint. The two
consequences are causally linked in one maneuver, not co-located in one plot.

**The occlusion constraint picks the passing side.** The station is placed so the natural
avoidance direction is the one that breaks the line of sight. The occlusion rows then select the
passing class rather than decorate a trajectory that would have looked the same anyway — a much
stronger claim than "the constraint was satisfied."

**The occluder and the keep-out obstacle are the same body.** Staying visible to the station
implies not being inside the occluder — a sight line that starts inside the body is blocked by
definition — so the occlusion constraint subsumes collision avoidance for that body, and the
keep-out rows are present but never binding in this scenario. State that in one honest sentence;
it is a small true observation, and the keep-out machinery is already demonstrated by the
existing scenarios.

**Concrete geometry.** A wide, low moving fence — the `wall` scenario lifted to three spatial
dimensions — with the station beyond it at low altitude. Going around laterally is long, or runs
into the workspace bound; going over is short, and exits the shadow because the fence is low.

**The baseline that must be able to fail.** Run the same scenario with the occlusion rows
removed: it must lose contact for a measurable interval. If the baseline also keeps line of
sight, the constraint was slack and the figure proves nothing — move the station and re-run. Two
runs, identical in everything but one constraint block. This is the difference between evidence
and decoration.

**Figure plan: one figure, two panels** (this fills the figure-slot definition, item A6, for the
occlusion figure). Three spatial dimensions plus time has no natural projection, so the picture
and the proof are different panels:

- a 3D spatial view, which only has to establish the dimensionality and sell the climb;
- a line-of-sight-margin-versus-time panel, which proves the claim: the baseline dips below
  zero, the constrained run stays above it. This panel is dimension-free and is the part a
  reviewer actually checks.

**Gate before any code** (solver item B12, the occlusion constraint builder): run the numerical
convexity check on the space-time shadow set, per tube piece. The shadow of a convex body from a
point observer is convex at a single instant, but the observer-to-occluder distance changes with
time, so the cone's half-angle varies nonlinearly along the time axis. If the per-piece shadow is
not convex, §3.2's outer approximation becomes mandatory rather than optional.

## The question for the PI

Not "does the skeleton look good." The one thing only they can answer:

> Osburn et al. published the space-time Bezier lift in August 2025. Our delta is
> that supporting half-spaces are generated directly against the obstacle tube
> instead of precomputing a convex cell decomposition. **Is that delta enough for
> this venue?**

---

# Part A — claim, evidence, and risks

## 1. The claim

> A decomposition-free space-time formulation in which each Bezier segment's time extent is
> itself a decision variable, so the obstacle's convex over-approximation is rebuilt from the
> current iterate on every successive-convexification iteration rather than being fixed before
> the solve begins.

That single property is what permits timing to be optimised — including stopping and waiting —
and what keeps arbitrary known non-straight obstacle motion affordable.

Do not claim "time as a coordinate." That was published in August 2025 by Osburn, Peterson and
Salmon, along with the space-time lift, finite-height tubes, the Bezier convex-hull argument in
the lifted space, and time monotonicity on control points. Four of the five items listed in the
project instruction file under "What is genuinely new" appear in that paper.

Do not claim speed either. Osburn solves a single-moving-obstacle case in about half a second
using the same solver we use. Any advantage on small problems is a small factor, not an order of
magnitude. The argument is how cost scales with dimension, not wall-clock time on a
two-dimensional toy problem.

## 2. What is different

Everyone else does one of two things that we do not.

| approach | free space | obstacle over-approximation | time allocation |
|---|---|---|---|
| Osburn, and the graph-of-convex-sets family generally | decomposed in advance into convex cells, plus a combinatorial layer to choose among them | not used | chosen, but only inside static cells |
| MADER, the semantic-corridor work, and driving-domain corridor methods | none | convex hull over a fixed time interval | fixed by the spline knots |
| ours | none | convex hull over an adaptive time interval | a decision variable |

Two consequences worth stating in the paper:

- MADER structurally cannot wait. Its time allocation is baked into the knot vector before the
  solve. Waiting is a behaviour that exists only when the time extent is free.
- The decomposition is a recurring cost, not a setup cost. Osburn's own sweep shows a seventeen
  per cent better trajectory costing seventeen times the compute, with free-space coverage still
  incomplete at the top of the sweep. Both halves of that machine — seeding convex regions by
  random sampling, then searching the resulting graph — degrade as dimensions are added, and we
  are adding one.

## 3. Method

1. Lift into the four-dimensional space of three spatial coordinates and time, with time as an
   explicit Bezier coordinate. The convex-hull property then holds verbatim in four dimensions
   with no side conditions, unlike corridor methods that write position as a function of time,
   which need an extra condition because the spatial axes and the temporal axis are not
   equivalent.
2. On each successive-convexification iteration, and for each pairing of one segment with one
   obstacle: take a convex over-approximation of the obstacle over that segment's current time
   span, emit exactly one supporting half-space, and apply it to every control point of that
   segment. One plane per segment-and-obstacle pair, never one per control point. In the lifted
   space the plane's normal is the spatial unit normal paired with a time component equal to
   minus the dot product of that spatial normal with the obstacle's velocity. The time component
   is not optional; it is what makes the plane support the tube at all.
3. For non-straight obstacle motion, adopt MADER's construction: sample the obstacle's known
   positions across the span, inflate each sample, and take the convex hull of the result. The
   whole tube need not be convex — only the slice overlapping one segment's span.
4. Enforce time monotonicity on the control points' time components, with a minimum separation.
5. The same construction covers all three constraint types: straight keep-out tubes, bent tubes
   from zigzagging obstacles, and shadow cones as described in section five.

Design rule: the segment count must be high enough that each segment's span sees roughly monotone
obstacle motion. A span covering a direction reversal produces a fat hull. This is the same
monotone knob as everywhere else — more segments give tighter hulls and a strictly larger
feasible set.

## 4. Demo scenarios

These replace the three scenarios listed in the project instruction file.

### Scenario one — wait for the door

The curve must stop and wait for a time-gated obstacle to clear. This proves the formulation:
time is a real optimisation dimension, and here is a behaviour that requires it.

Who else can wait, verified:

| method | waits? | output |
|---|---|---|
| Safe Interval Path Planning, Phillips and Likhachev, 2011 | yes — its edges are combined wait-then-move actions that jump through multiple time steps to a time when an obstacle has passed | discrete grid path |
| Space-Time Rapidly-exploring Random Tree, 2022 | yes | sampled path |
| MADER | no — the knots fix the timing | smooth trajectory |
| Osburn, and graph-of-convex-sets methods generally | not demonstrated | smooth trajectory |
| the temporal-logic graph-of-convex-sets paper | no — it does not describe robots waiting in place, and achieves timing compliance through path selection instead; its "doors" are logical key-precedence, not time gates | smooth trajectory |

Safe Interval Path Planning assumes that dynamic obstacles occupy locations during known time
intervals, which is our exact assumption. Same assumption, different output class. The sentence
for the paper: waiting exists in discrete and sampling-based planners under exactly our
assumptions, and does not exist in the continuous-optimisation family, because that family fixes
timing before solving.

### Scenario two — sneak past a radar

Hide inside the shadow cast by zigzagging moving obstacles. The mirrored variant is to avoid
those shadows in order to maintain line of sight to a ground controller. This proves the method
choice: a nonconvex, time-varying mission constraint that a decomposition-based method would have
to re-carve around.

Observer model: binary line-of-sight occlusion from a point source. No radar equation, no
detection probability. The claim is extensibility to real radar avoidance, not accurate radar
avoidance.

Prior art on the stealth half is games and graphics, and discrete — stealthy path planning against
dynamic observers, corridor-map stealth planning, covert planning against imperfect observers. All
of them use static terrain as the occluder. No continuous trajectory optimisation hiding behind a
moving occluder was found.

## 5. The shadow lemma

Statement. Let the observer be a single point, and let the obstacle be any convex body that does
not contain the observer. Define the shadow as the set of all positions whose straight line of
sight back to the observer is interrupted by the obstacle. Then the shadow is convex.

Proof, in words. Measure everything from the observer. A position lies in the shadow exactly when
you can reach it by starting at the observer, travelling to some point of the obstacle, and then
continuing outward along that same ray without turning — that is, the position is some obstacle
point stretched away from the observer by a factor of at least one. So the shadow, measured from
the observer, is the set of all obstacle points stretched by any factor of one or more.

Now take two shadow positions and form any weighted average of them, with non-negative weights
summing to one. Each of the two is an obstacle point multiplied by its own stretch factor. Let
the combined stretch factor be the same weighted average of the two stretch factors; since each
of them is at least one, so is that average. Divide the weighted average of the two positions by
the combined stretch factor. What remains is a weighted average of the two original obstacle
points, again with non-negative weights summing to one, so it is itself a point of the obstacle
because the obstacle is convex. The weighted average of the two shadow positions is therefore an
obstacle point multiplied by a stretch factor of at least one — which is exactly the description
of a shadow position. So the average lies in the shadow, and the shadow is convex.

This holds for any convex obstacle in any number of dimensions, so it covers spheres in three
spatial dimensions unchanged. It was derived here, not taken from a source. It is elementary and
independently checkable, but if it appears in the paper it should be stated as our result rather
than cited to someone else.

Consequence — hiding is cheaper than avoiding:

| constraint | region | convex? | machinery |
|---|---|---|---|
| stay out of the keep-out zone | outside a tube | no | supporting half-space plus a committed side |
| stay out of the shadow, to keep the controller link | outside a cone | no | supporting half-space plus a committed side — the same code |
| stay inside the shadow, to hide from the radar | inside a cone | yes | plain linear rows, with no side to commit and no passing class |

Space-time caveat. At each instant the shadow is convex, but the cone swings as the obstacle
moves, so the shadow volume in the lifted four-dimensional space is not convex. The remedy is the
same as for the bent tube: over a segment's span, require the position to lie in the intersection
of the shadows across that span. An intersection of convex sets is convex, and it is a sound inner
approximation, because being inside all of them implies being inside the one at the actual time.
Conservative, and it tightens with segment count.

Two honest failure modes, both physically correct. The intersection goes empty if the obstacle
moves far within one span — you cannot hide behind something that has left. And the choice of
which obstacle to hide behind, in what order, is a discrete choice with the same combinatorial
structure as the passing class. That is where solver item B6, procedural seeds and multi-start,
earns its place.

Suggested demo form: a hard constraint requiring the vehicle to stay hidden for the whole
traverse, rather than minimising exposure. It is convex, it is a one-line claim, and the figure
reads at a glance.

## 6. The risk this contribution creates

Read this before implementing solver item B7, the feasibility gate.

An adaptively-built hull is valid only inside the time span it was built to cover. If the hull is
constructed for a span taken from the current iterate, and the next iterate's segment slides
outside that span, the constraint becomes silently meaningless — the curve occupies a time the
hull never accounted for. The solver then reports total slack as zero on a constraint that
asserts nothing, and the feasibility gate passes a trajectory that penetrates an obstacle.

Two conditions, neither of them automatic:

1. the segment's control-point time components must stay within the span its hull covers, either
   by explicit constraint or by a trust region tight enough to guarantee it; and
2. the feasibility gate must re-verify clearance against the true obstacle trajectory, not
   against the hulls the solver used.

This makes the trust region from solver item B5 — porting the successive-convexification
machinery from the main branch, landed in commit 848bf3b — correctness-critical rather than a
convergence aid. Without it, nothing keeps the iterate inside the region where its own
constraints are valid.

Also for the solver workstream: the door scenario needs the finite-height tube, meaning the
branch of the keep-out-zone builder that clips against an obstacle's start and end times. The
project instruction file records that obstacle time windows default to plus and minus infinity,
which makes that branch unreachable. The scenario's headline behaviour may run through code that
never executes.

## 7. Decided against

- Convexifying free space by a coordinate transform, such as inversion, a conformal map, or a
  Nyquist-style transform. The obstruction is topological, not geometric: a convex set is simply
  connected, free space around an obstacle is not, and continuous invertible maps preserve that
  distinction. Inversion about the obstacle moves the hole to the origin; it does not remove it.
  The hole is the fact that passing left and passing right are genuinely distinct plans. The
  strongest published version of this idea is Rimon and Koditschek's navigation functions, whose
  titles concede the point — geometrically complicated but topologically simple spaces — and
  whose convergence guarantee holds only from almost all initial conditions, the excluded set
  being the saddle points that topology forces to exist.
- Receding-horizon replanning. Obstacle motion is deterministic and known, so there is nothing to
  re-estimate.
- The radar equation, detection probability, and target fluctuation models. Binary occlusion only.
- Competing on supporting more constraint types. That is Osburn's own stated contribution, so
  competing there head-on loses. Compete instead on the kind of constraint — nonconvex and
  time-varying — that a convex-cell framework cannot absorb.

## 8. What must be measured

No current performance number exists. The benchmarks file predates commit 848bf3b and the
uncommitted solver work on top of it. Its numbers describe a solver that no longer exists and
must not be quoted.

To be measured once the two constraint defects are fixed, meaning solver items B1 and B4:

- the door scenario demonstrably waiting;
- the stealth scenario hidden from start to finish;
- minimum clearance verified against the true obstacle trajectory, not against the solver's hulls;
- solve time against Osburn's half-second single-obstacle case and roughly four-second cluttered
  case, on the same solver and a comparable class of hardware.

## 9. Weakest links

Ranked by how much damage each would do if it turned out to be wrong.

1. MADER's fixed-knot time allocation is the pivot of section two, and it reached us through a
   fetch summariser rather than the source document. The solver workstream read MADER in full for
   references item C3 and its quoted equation is consistent, but that specific sentence is
   unconfirmed. Confirm it before paper item A2, the formulation section, relies on it.
2. The shadow lemma is ours, and nobody else has reviewed it.
3. "No paper does this" is weaker than a proof. No decomposition-free space-time formulation was
   found, but absence of evidence is not proof of absence.
4. Erdmann and Lozano-Perez, 1987, is unverified — the archived copy is a scan with no text
   layer. It is a lineage citation only, and nothing load-bearing rests on it.

---

# Part B — 논문 뼈대 (2페이지 학회 논문)

뼈대만 정리한 절이다. 각 절에 무엇을 담고 무엇을 담지 않을지를 적었으며, 본문 문장은 아직 쓰지
않았다. 주장의 근거와 검증 상태는 위 Part A에 있다.

## 핵심 기여

이 절은 본 문서에서 가장 먼저 확정된 부분이며, 다른 절의 서술이 이 절과 어긋나면 이 절을 기준으로
맞춘다. 논문 본문에 옮길 때에도 표현만 다듬고 내용은 그대로 쓴다.

한 문장으로 정리하면 다음과 같다. 본 논문은 각 분할구간의 시간 구간을 결정 변수로 두어, 이동
장애물의 볼록 외부 근사를 순차 볼록 최적화(sequential convex programming, SCP) 반복마다 현재
반복점으로부터 다시 구성하는 궤적 최적화 정식화를 제안한다. 자유 공간을 볼록 영역으로 미리
분할하지 않으며, 어느 영역을 지날지 고르는 조합적 계층도 두지 않는다.

기여는 다음 세 가지이다. 첫째, 분할구간의 시간 구간이 최적화의 입력이 아니라 결정 변수가 되므로
시간 배분 자체가 최적화 대상이 된다. 통과 시점을 기다리는 동작이 이를 드러내는 사례이며, 이
동작은 시간 구간이 자유로울 때에만 나타난다. 둘째, 리프팅된 공간에서 지지 반공간은 시간 성분을
가져야 하며, 그 값은 공간 법선과 장애물 속도의 내적에 음의 부호를 붙인 것이다. 이 반공간을
분할구간과 장애물 쌍마다 하나씩 두고 해당 분할구간의 모든 제어점에 부과하면 연속 시간에 대한
볼록 껍질 보장이 유지된다. 제어점마다 다른 평면을 부과하면 이 보장은 유한 개의 점 조건으로
약해진다. 셋째, 동일한 구성이 직선 형태의 구형 Keep-Out Zone(KOZ), 등속이 아닌 운동에서 생기는
휘어진 관, 그리고 관측자에 대한 차폐 영역을 모두 처리한다. 비볼록이고 시간에 따라 변하는 임무
제약을 추가하는 비용은 분할구간과 장애물과 관측자 조합마다 선형화된 제약 한 줄이다.

주장하지 않는 것을 함께 적어 둔다. 시간을 곡선의 좌표로 두는 것 자체는 2025년 8월에 이미
발표되었다. 등속 장애물이 정적인 관이 된다는 관찰은 1987년으로 거슬러 올라간다. 분할구간마다
하나의 평면을 두면 볼록 껍질 보장이 유지된다는 사실은 안전 통로 문헌의 표준이다. 계산 속도도
주장하지 않는다. 작은 문제에서는 동일한 solver 위에서 선행 연구와 큰 차이가 없다. 전역 최적성과
초기값 없이 푸는 성질은 오히려 선행 연구 쪽이 낫고, 이는 한계 절에 명시한다.

각 기여가 틀렸음을 보이는 조건도 미리 적어 둔다. 첫째 기여는 분할구간의 시간 구간을 자유롭게
두면서 장애물 근사를 반복마다 다시 구성하는 선행 연구가 있으면 무너진다. 확인한 범위에서 볼록
집합 그래프 계열은 영역을 풀기 전에 고정하고, MADER는 구간을 매듭으로 고정한다. 둘째 기여는
모든 지지 반공간을 여유 없이 만족한 해가 장애물의 실제 궤적에 대해 재검증했을 때 침범으로
판정되면 무너진다. 셋째 기여는 시간 배분이 미리 고정된 기법에서 대기 동작이 시연되면 무너진다.

## 제목 후보

1. 시간을 좌표로 갖는 Bézier 곡선을 이용한 이동 장애물 회피 궤적 최적화
2. 공간 분할 없는 공간-시간 지지 반공간 기반 궤적 최적화

## 초록 (5–6문장, 구체 수치 없이)

이동 장애물 환경에서 경로와 통과 시점을 함께 결정해야 한다는 문제 제시로 시작한다. 기존
볼록 최적화 계열이 자유 공간을 미리 볼록 영역으로 분할하고 그 위에서 조합적 탐색을 수행한다는
구조적 특징을 한 문장으로 정리한다. 본 논문은 시간을 Bézier 곡선의 좌표로 포함하고 각
분할구간의 시간 구간 자체를 결정 변수로 두는 정식화를 제안한다는 점을 밝힌다. 그 결과 통과
시점을 기다리는 동작과 관측자에 대한 차폐를 유지하는 동작을 하나의 연속 최적화로 얻는다는
정성적 결과로 마무리한다.

## 1. 서론

1문단은 이동 장애물 회피에서 경로 결정과 시점 결정이 분리되지 않는다는 점을 든다.

2문단은 기존 접근의 구조적 제약을 정리한다. 볼록 영역 분할을 사전에 계산하는 계열과, 시간을
곡선의 매개변수로 두어 시간 배분이 매듭에 의해 고정되는 계열로 나누어 각각 한 문장씩 적는다.

3문단은 기여를 줄글로 열거한다. 첫째, 각 분할구간의 시간 구간을 결정 변수로 두어 SCP 반복마다 장애물의 볼록 외부 근사를 다시 구성하는
정식화를 제안한다. 둘째, 리프팅된 공간에서 지지 반공간의 시간 성분이 생략될 수 없음을 보이고,
분할구간과 장애물 쌍마다 하나의 지지 반공간을 모든 제어점에 부과하여 볼록 껍질 성질을 유지한다.
셋째, 통과 시점 대기와 관측자 차폐 유지를 동일한 제약 구성으로 처리할 수 있음을 보인다.

기여 서술에서 시간을 좌표로 두는 것 자체를 새로움으로 주장하지 않는다. 선행 연구가 이미
제시한 부분이므로 관련 연구 절에서 명시적으로 인정한다.

## 2. 문제 정식화

### 2.1 공간-시간 리프팅

등속으로 움직이는 장애물이 리프팅된 공간에서 기울어진 정적 관이 된다는 점을 서술한다. 시간이
곡선의 매개변수가 아니라 좌표이므로 볼록 껍질 성질이 추가 조건 없이 그대로 성립한다는 점을
덧붙인다. 시간을 매개변수로 두는 통로 기반 기법이 별도의 충분조건을 필요로 한다는 사실을 인용
한 문장으로 대비시킨다.

### 2.2 Bézier 표현과 결정 변수

제어점을 결정 변수로 정의하고, De Casteljau 분할로 얻은 분할구간을 도입한다. 시간 좌표에 대한
단조성 제약과 최소 시간 간격을 여기서 정의한다.

## 3. 제안 기법

### 3.1 분할구간과 장애물 쌍에 대한 지지 반공간

supporting half-space(지지 반공간)를 첫 등장에서 병기하고 이후 한글 표기로 통일한다. 외향
법선의 공간 성분과 시간 성분을 말로 서술한다. 시간 성분은 공간 법선과 장애물 속도의 내적에
음의 부호를 붙인 값이며, 이 항이 없으면 해당 평면은 관을 지지하지 못한다는 점을 분명히 한다.

하나의 분할구간에 속한 모든 제어점에 동일한 반공간을 부과한다는 점을 강조한다. 제어점마다
다른 평면을 부과하면 볼록 껍질 성질을 사용할 수 없고 유한 개의 점 조건으로 약화된다는 점을
한 문장으로 정리한다.

### 3.2 시간 구간이 결정 변수인 경우의 외부 근사

장애물 운동이 등속이 아닐 때 관이 휘어 비볼록이 되므로, 분할구간의 시간 구간에 겹치는 부분만
볼록 외부 근사로 덮는다는 구성을 제시한다. 시간 구간이 결정 변수이므로 이 근사는 SCP 반복마다
다시 구성된다는 점이 본 논문의 핵심이다.

외부 근사가 자신이 덮는 시간 구간 밖에서는 아무것도 보장하지 않으므로, 신뢰 구간의 크기가
반복점을 그 구간 안에 묶어 두어야 한다는 점을 명시한다. 이는 수렴을 돕는 장치가 아니라
정당성을 위한 조건이다.

### 3.3 관측자 차폐 제약

관측자를 하나의 점으로 두고, 시선이 장애물에 가려지는 위치들의 집합을 차폐 영역으로 정의한다.
장애물이 볼록이면 차폐 영역도 볼록이라는 명제를 제시하고 증명은 두세 문장으로 줄인다. 차폐를
유지하는 제약은 볼록이므로 통과 방향을 미리 정할 필요가 없고, 차폐를 피하는 제약은 비볼록이어서
장애물 회피와 같은 구성을 그대로 쓴다는 대비를 적는다.

시간에 따라 차폐 영역이 회전하므로 리프팅된 공간에서는 볼록이 아니라는 점, 분할구간의 시간
구간에 걸친 교집합을 사용하면 볼록이면서 보수적인 내부 근사가 된다는 점을 잇는다.

## 4. 실험

### 4.1 시나리오 1 — 통과 시점 대기

시간 제한이 있는 장애물이 사라질 때까지 기다렸다가 통과하는 사례를 제시한다. 대기 동작이
연속 최적화 하나로 얻어진다는 점이 이 시나리오의 목적이다.

### 4.2 시나리오 2 — 관측자 차폐 유지

여러 개의 지그재그로 움직이는 장애물 뒤에 머무르며 관측자의 시선을 피하는 사례를 제시한다.
관측자 모형은 점 광원에 대한 이진 시선 차폐로 한정하고, 탐지 확률 모형은 사용하지 않는다는
점을 실험 설정에서 밝힌다.

### 4.3 비교

계산 시간과 최소 여유 거리를 표로 제시한다. 구체 수치는 이 절에만 둔다. 최소 여유 거리는
최적화가 사용한 외부 근사가 아니라 장애물의 실제 궤적에 대해 재검증한 값이어야 한다.

## 5. 한계 및 결론

국소해만 얻으며 초기값에 의존한다는 점을 먼저 적는다. 장애물 운동이 결정적이고 알려져 있다는
가정을 명시한다. 볼록 외부 근사의 보수성은 분할구간 수로 조절되며, 분할구간 수를 늘리면
실현 가능 영역이 넓어진다는 점을 덧붙인다.

## 그림 슬롯

1. 리프팅 개념도. 움직이는 장애물이 기울어진 관이 되는 그림.
2. 지지 반공간 그림. 시간 성분이 없는 평면이 관을 지지하지 못함을 보이는 대비 그림.
3. 시나리오 1 결과. 위에서 본 경로와 시간 축을 포함한 등각 투상 두 장.
4. 시나리오 2 결과. 차폐 영역과 궤적을 함께 표시.

## 관련 연구에서 반드시 인용할 것

Erdmann과 Lozano-Perez의 구성 공간-시간, Safe Interval Path Planning, 공간-시간
Rapidly-exploring Random Tree, 볼록 집합 그래프 계열과 그 공간-시간 확장, MADER,
시간 의존 통로에서 볼록 껍질 성질이 성립하기 위한 충분조건을 다룬 연구.
