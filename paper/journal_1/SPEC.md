# Idea 1 — journal paper specification

**What this file is.** The complete specification of the journal manuscript: the claim as it will
be stated, the formulation as numbered definitions and lemmas, the algorithm as one procedure, and
the experiments as things that can fail. It is compiled from
[`idea/spacetime.md`](../../idea/spacetime.md) (§"Formulation — rigorous statement", §"Occlusion",
§"Novelty and prior art"), from [`SOLVER.md`](../../SOLVER.md) §Established facts, and from the
code where the code is more specific than either. Where the code and the idea document differ, this
file follows the **code** and says so in place.

**What it is not.** It owns no measurement and quotes no measured number. Algorithm constants
appear as symbols with their current code value beside them, marked as parameters. Measured
numbers live in figure-grade sidecars and reach the manuscript only through them.

**Companion files, and what outranks what.** **This file owns the claim, the formulation, the
algorithm and the experiment plan, and it outranks every other document in this directory.**
[`README.md`](README.md) is the short artifact document (figure slots, status); its claim and
consequence 3 are marked superseded in place. [`pi_report_2026-09-09.md`](pi_report_2026-09-09.md)
is a superseded snapshot of one session's recommendations — **read its precedence header before
using anything in it**; it is still the only source for the schedule, demo A's stop rule, and the
baseline options, and its argument against self-implementation is an open conflict recorded in §6.
The four demo specifications in [`demo_specs/`](demo_specs/) are build specifications and keep
their own detail; §4 below only assigns each a job and outranks them on which consequence each
carries.

**Numbering.** Definitions D1…, Lemmas L1…, one Problem P, one Algorithm A, one Condition C. The
labels are for cross-reference *inside this file and the manuscript*; `idea/spacetime.md`
deliberately carries none, so a statement there is referred to by what it says.

---

## 1. Claim and positioning

**Framing decided 2026-09-20, after the audience decision (§6).** The paper is written for the
planning and trajectory-optimization methods community, and the claim is stated as an *interface
requirement*, not as the absence of a competitor's machinery. "Decomposition-free" describes what
we do not do; it also picks a fight that cannot be fully won, because a space-time decomposition
can carve any closed set and can optimize arrival time inside itself (Tang et al.). What can be
claimed without qualification is what this construction *consumes*.

### 1.1 The claim

> Continuous-time avoidance constraints can be built from a **support oracle on the hazard**
> instead of a representation of free space. For each (segment, obstacle, approach) the
> construction asks the keep-out set for a nearest point and for a rigorous bound on its support
> in one direction, and returns one supporting half-space imposed on the segment's control points.
> The resulting subproblem is a quadratic program with second-order cones whose size is linear in
> segments × obstacles and carries no discrete structure — no integer variables, no region graph,
> no cell selection — and whose conservatism is bounded by the trust region rather than by the
> geometric complexity of the free space.

**What the construction requires of a keep-out set $\mathcal K$, and nothing more.** Given the
current segment's centroid $c$ and a clipping ball $\bar B(c,r_{\mathrm{clip}})$:

| | requirement | used for |
|---|---|---|
| **O1** | the nearest point of $\mathcal K$ to $c$, or equivalently the direction from $\mathcal K$ back to $c$ | the wall's normal $n$ (D9, L7) |
| **O2** | a **rigorous upper bound** on $\max\{n^\top z:\ z\in\mathcal K\cap\bar B(c,r_{\mathrm{clip}})\}$ | the wall's offset $\hat b$ (D9) |
| **O3** | a partition of $\mathcal K\cap\bar B(c,r_{\mathrm{clip}})$ into pieces, one wall each | approaches (D8) |
| **O4** | *optional* — the active face of $\mathcal K$ at the nearest point | the rotation term of the linearised row (§2.8); without it the rows are still sound, only slower to converge |

O3 is sound for **any** partition, so it is a quality knob, never a correctness one. O4 changes
only the rate. The load-bearing requirements are O1 and O2.

**Why this is enough, and why convexity never appears.** A set and its convex hull have identical
support in every direction (L5), so O2 can be answered on any superset sharing supports and
$\mathcal K$ need not be convex, connected, or even representable. The continuous-time guarantee
then comes from the Bézier hull property alone (L1, L8): a half-space satisfied at every control
point of a segment is satisfied along the whole segment, for all times in it.

**The evidence that the interface is real, not a retrospective story.** The same construction and
the same row builder produce the walls for a ball, for a curving lifted tube, and for the shadow
a moving body casts from a ground station — a set that twists, widens with distance and is not
convex in space-time. There is no occlusion-specific branch anywhere in the derivation or in the
code: the obstacle and its shadow are one keep-out set read at two stretch factors, and with no
station present every formula reduces to the plain tube identically (L3). A hazard nobody would
choose to carve costs one paragraph, because the oracle does not care what shape it is.

**Title direction.** Support-oracle constraints for continuous-time trajectory optimization among
moving hazards. **Never open with "time as a coordinate"** — that lift is Osburn, Peterson and
Salmon, August 2025.

### 1.2 Consequences — narrowed 2026-09-17, reframed under the oracle 2026-09-20

Each consequence is evidence that the interface buys something. Each is stated against the
specific prior work it actually separates us from; a consequence overstated against the whole
corridor family is the sentence a reviewer who knows Tang rejects.

| consequence | separates us from | does **not** separate us from | demo that carries it |
|---|---|---|---|
| **1. Hazard shape is free.** The oracle is all the construction sees, so an obstacle of arbitrary polynomial motion, a finite lifetime, or a twisting shadow produces the same rows as a static ball. | Osburn (static or constant-velocity only; "more complex motions require finer sampling and will only approximate"); ST-GCS (piecewise-constant velocity); every space-only decomposition | — | **B (the shadow — the thesis demonstration)**, A (rotating appendages) |
| **2. Timing is a decision inside the same convex subproblem.** Arrival time, the speed cap and the time penalty are variables and rows of the one QP; there is no path-then-time-allocation stage. | Osburn (end point pinned at a fixed space-time point; cost is spatial path length, so time is never optimised); space-only corridor methods | **ST-GCS** (Tang et al.: time-optimal, free arrival, inside a space-time graph) | A (when to cross) |
| **3. Cost is independent of the geometry of the keep-out set.** Rows are linear in segments × obstacles whatever the shape; answering the oracle is local work at one point. A decomposition's set count grows with the geometry it must carve, and must be built before the solve. | the whole corridor family, on **cost** — not on expressibility | — | **C (obstacle-count axis — the cost evidence)** |

**Retired wording, do not reintroduce:** "a decomposition cannot express visibility" (false as
stated — a space-time decomposition can carve any closed set); "a decomposition must be fixed
before the timing that decides which cells are free" (true only of space-only decomposition).
Consequence 3 is about *cost*, and the paper says so.

**The prior-art table's two arrival-time cells were corrected 2026-09-21** (`idea/spacetime.md`
§"The corridor / convex-opt family in detail"): Osburn now reads *pinned* and ours reads *free,
optionally*. Recorded here because the correction is the evidence for consequence 2.
**The submitted KSAS manuscript and the draft poster both carry the retired timing sentence**
(`paper/ksas_2026_fall/manuscript.md`, and `poster/poster.tex` 「통과 시각도 그 셀 배치에 매인다」).
The manuscript is submitted and cannot change; the poster is a draft and must be corrected before
November, and the journal must not inherit the sentence from either.

### 1.3 The guarantee — what was not given up

- **Continuous-time feasibility by hull containment** (L1, L8). Standard machinery; Bézier and
  B-spline planners have used it for years. Cite it, do not claim it. What is ours is that it is
  made to work against a set given only by an oracle.
- **The coverage condition** (C, §2.7) — ours, and the paper's second thesis. A construction built
  from a *local* oracle owes the reader a statement of what it covers: the half-spaces provably
  contain the clipped keep-out volume, and a stated, testable condition on the clipping radius
  makes the segment certificate cover the *whole* keep-out zone rather than the clipped piece.
  The residual hole is named, counted per iteration, and gates every figure. **Nobody in the
  corridor family reports coverage at all; they report that the solver converged.** The paper
  proposes this as the thing such methods should report.
- **The independent certificate** (D11). Every reported guarantee is re-checked by a computation
  that shares nothing with the construction it checks.

### 1.4 What is not claimed

- Global optimality. The method is local; the passing homotopy class comes from the initialization.
  This is the price of a local oracle instead of global structure, and it is stated as a design
  position, not conceded as an oversight.
- Stationarity. `converged` asserts feasibility plus no further progress; there is no KKT residual.
- Speed on small problems. Osburn's small cases take a fraction of a second; "we are fast" cannot
  carry the paper. The argument is how cost scales with dimension and hazard geometry.
- **A plugin interface.** O1–O4 describe what the *derivation* consumes and what the one row
  builder implements for the tube and the center surface. The code exposes no abstract oracle type,
  and the paper must not imply one.
- Any physical model of the mission: no gravity or Δv (A), no wake physics, no radar equation or
  detection probability (B). Binary occlusion, exact known motion, offline solve.
- "Minimises spatial acceleration." Forbidden phrase; the objective is a parameter-domain
  smoothness regulariser blind to timing (§2.8).

### 1.5 Nearest prior work — the sentences the introduction must get right

- **Osburn, Peterson, Salmon (arXiv 2508.10203, 2025).** Space-time GCS with Bézier control
  points carrying an explicit time component, IRIS-sampled convex sets in 2D + time, Clarabel.
  Obstacles static or constant-velocity prisms. End point pinned in space-time; cost is spatial
  path length. Their Table I is the evidence that the decomposition is the cost: set count and
  solve time rise with IRIS samples while coverage stays incomplete. **Mandatory citation, the
  twin.**
- **Tang et al., ST-GCS (IROS 2025, arXiv 2503.00583; search-based variant arXiv 2607.00444).**
  Space-time graph of convex sets for multi-robot planning; time-optimal with free arrival;
  piecewise-constant velocity; other robots' planned paths as the only dynamic obstacles.
- **SPOT (arXiv 2602.01189).** The only genuinely 3D + time member of the family; horizon capped
  at two seconds with binned time. Read via summariser only — verify before citing detail.
- **Marcucci et al.** (GCS, SIAM J. Opt. 2024; Science Robotics 2023) and **Deits & Tedrake**
  (IRIS) for framing; **Zhang, Yadmellat, Gao (arXiv 2110.00065)** for the hull-property
  sufficient condition in corridors; **MADER** for separating planes as decision variables, the
  alternative to linearising them.
- The three strengths of the graph family that every limitations section states: no initial guess
  needed; global with respect to their graph; small cases solved in under a second.

**Blocking the introduction:** four of these are marked "search-level only, bibliography not
verified" in `idea/spacetime.md` §Confidence and known gaps — GCS, IRIS, Zhang, SPOT. This
audience's reviewers are those authors. Read them from source before sections 1 and 2 are written;
Osburn and ST-GCS are already read from PDF.

---

## 2. Formulation

### 2.1 The lift

**D1 (lifted space).** Spatial dimension $d_s$ (two or three). The lifted space is
$\mathbb R^{d_s+1}$ with points $z=(x,\tau)$, $x\in\mathbb R^{d_s}$, $\tau\in\mathbb R$. The axis
scale is one: the Euclidean norm $\lVert\cdot\rVert$ on $\mathbb R^{d_s+1}$ mixes space and time on
equal terms, and every ball, distance and projection below is taken in it. Nothing in the
formulation may single out the time axis — no time windows on the trajectory, no per-segment time
intervals, no clipping along $\tau$.

**D2 (trajectory).** Decision variable: control points $P_0,\dots,P_N\in\mathbb R^{d_s+1}$,
$P_j=(x_j,t_j)$, stacked into $\mathbf p\in\mathbb R^{(N+1)(d_s+1)}$. The curve, with $u\in[0,1]$
the curve parameter — not time:

$$Z(u)=\sum_{j=0}^{N}B_j^N(u)\,P_j,\qquad B_j^N(u)=\binom{N}{j}u^j(1-u)^{N-j},\qquad
\tau(u)=\sum_{j}B_j^N(u)\,t_j .$$

**D3 (segments).** De Casteljau subdivision into $M$ segments gives row-stochastic matrices
$A^{(k)}$ ($A^{(k)}_{ij}\ge0$, $\sum_jA^{(k)}_{ij}=1$), segment control points, centroid and
radius

$$Q^{(k)}_i=\sum_jA^{(k)}_{ij}P_j,\qquad
c^{(k)}=\tfrac1{N+1}\sum_iQ^{(k)}_i=\sum_jw^{(k)}_jP_j,\qquad
E^{(k)}=\max_i\lVert Q^{(k)}_i-c^{(k)}\rVert ,$$

with $w^{(k)}\ge0$, $\sum_jw^{(k)}_j=1$. The subdivision matrix is dense on purpose;
sparsifying it breaks L8.

**L1 (hull property).** For $u$ in segment $k$'s parameter interval,
$Z(u)\in\operatorname{conv}\{Q^{(k)}_i\}_{i=0}^N$. *Standard; cite.*

### 2.2 The obstacle

**D4 (keep-out zone).** Obstacle $m$ has known motion $\pi_m$ on its existence window
$[T^0_m,T^1_m]$ and radius $r_m$. Lifted centreline and keep-out zone:

$$\gamma_m(\tau)=\bigl(\pi_m(\tau),\tau\bigr),\qquad
\Gamma_m=\gamma_m\!\left([T^0_m,T^1_m]\right),\qquad
\mathcal K_m=\Gamma_m\oplus\bar B(0,r_m).$$

If $\pi_m$ is polynomial of degree $N_m$ then $\gamma_m$ is a Bézier curve in $\mathbb R^{d_s+1}$
with control points $G_0,\dots,G_{N_m}$, and the existence window *is* the time coordinate of its
first and last control point — a finite lifetime costs nothing extra. $\mathcal K_m$ is **not
convex unless $\pi_m$ is affine**; it admits no single supporting half-space, and the hull
property alone certifies nothing against it. That sentence is why the rest of §2 exists.

### 2.3 The station and the shadow

**D5 (center surface).** A ground station is a point $p\in\mathbb R^{d_s}$ present at every
instant. Stretching the centreline away from it,

$$\Sigma_m(\tau,u)=\bigl(p+u\,(\pi_m(\tau)-p),\ \tau\bigr),\quad u\ge1,\qquad
\mathcal K_m^{p}=\bigcup_{\tau,\,u\ge1}\bar B\bigl(\Sigma_m(\tau,u),\ u\,r_m\bigr).$$

$u=1$ is the body itself, radius $r_m$, unchanged. What widens with $u$ is the shadow: a point
source at finite distance casts an umbra of constant *angular* thickness. Time is not stretched —
the station maps each instant to itself — so every shadow wall carries a nonzero time component
like every other wall.

**L2 (shadow lemma — ours).** For a point observer and a convex body not containing it, at one
instant, the set of positions whose line of sight to the observer meets the body is convex. *Proof:
a shadow position is a body point scaled from the observer by a factor at least one; a convex
combination of two such is a convex combination of body points scaled by the averaged factor,
which is at least one.* Holds in any dimension. State as our result.

**L3 (the sweep is the umbra exactly).** $q$ is occluded at instant $\tau$ iff $q=p+u(z-p)$ for
some body point $z$ and $u\ge1$, iff $q\in\mathcal K_m^p$ at that $\tau$. Body and shadow are one
connected set with no seam; the construction has no occlusion-specific branch. With no station,
$u\equiv1$ and every formula reduces to D4 bit for bit — that reduction is the regression the
generalisation stands on.

**Remark.** The *space-time* shadow of a *moving* occluder is not convex — the cone rotates as the
occluder crosses the sight line — which is exactly why the wall below is a support of a clipped
piece and not a tangent plane, for the shadow as for the tube. For fixed $u$ the stretch is
affine, so $\Sigma_m(\cdot,u)$ is again a Bézier curve with control points $(p+u(X_l-p),T_l)$;
subdivision and the support bounds apply to it unchanged. Two readings of one zone are kept on
purpose: every obstacle contributes its body rows ($u=1$), and a station adds a second reading over
$u\ge1$ that contains the body again — sound, and what lets keep-out and line-of-sight be two
separately reported certificates. Several stations: the link must hold to every one; "any one
suffices" is a disjunction, needs binaries, decided against.

### 2.4 The clip

Everything in §2.4–2.6 is done per segment $k$ and obstacle $m$ with the centroid $c=c^{(k)}$
fixed at the reference iterate; the segment index is dropped where no confusion arises.

**D6 (nearest point and clipping radius).**

$$\tau^\star\in\arg\min_{\tau\in[T^0_m,T^1_m]}\lVert c-\gamma_m(\tau)\rVert,\qquad
f=\gamma_m(\tau^\star),\qquad d=\lVert c-f\rVert,$$

$$r_{\mathrm{clip}}=\operatorname{clip}\bigl(d,\ r_{\mathrm{floor}},\ r_{\mathrm{clip,max}}\bigr)
=\max\bigl(\min(d,\,r_{\mathrm{clip,max}}),\ r_{\mathrm{floor}}\bigr),\qquad
r_{\mathrm{clip,max}}=r_m+E^{(k)}+\Delta\sqrt{d_s+1},$$

where $\Delta$ is the trust radius of the current iteration and the floor is one of two:

$$r_{\mathrm{floor}}=\begin{cases}
r_m & \text{construction floor, always applied}\\[2pt]
\max\bigl(r_m,\ E^{(k)}+\Delta\sqrt{d_s+1}\bigr) & \text{reach floor — the sound-clip flag, default on}
\end{cases}$$

The two floors address different failures and must not be conflated: $r_m$ keeps the clipped
volume a piece of the *obstacle* rather than a ball strictly inside it (it binds iff
$c\in\mathcal K_m$, and without it the wall understates a penetration by $r_m-d$); the reach floor
is Condition C of §2.7 made unconditional. $\tau^\star$ need not be unique — non-uniqueness is
the bend case that makes $f$ discontinuous in $\mathbf p$, and any minimiser serves.

> **Code note — closed 2026-09-21.** An experiment labelled `E1 (2026-09-09)` once replaced the
> clamp under the reach floor with the minimal sound radius, discarding $d$; it was measured a
> regression, **refused 2026-09-18 and is no longer in the tree** (verified 2026-09-21). The rule
> above is the one the code runs. Four demo specifications still describe a "post-E1 build
> installed now" and are wrong on that point.

**D7 (clipped keep-out volume).** $\mathcal L^{(k)}_m=\mathcal K_m\cap\bar B(c,r_{\mathrm{clip}})$.

**L4 (band containment).** Let $I=\{\tau:\lVert\gamma_m(\tau)-c\rVert\le r_{\mathrm{clip}}+r_m\}$.
Then $\mathcal L^{(k)}_m\subseteq\gamma_m(I)\oplus\bar B(0,r_m)$. *Proof: a point of the left
side is within $r_m$ of some $\gamma_m(\tau)$ and within $r_{\mathrm{clip}}$ of $c$; the triangle
inequality puts $\tau$ in $I$.* The ball **selects a parameter range; it does not cut the set** —
the containing band is an outer approximation, and the wall is never built against it (§2.5).

**D8 (approaches).** Cut $I$ at every interior local maximum of
$\tau\mapsto\lVert\gamma_m(\tau)-c\rVert$. Each resulting sub-interval $I_\ell$ is one
**approach** of the obstacle to the segment, with its piece

$$\mathcal L^{(k)}_{m,\ell}=\bigl(\gamma_m(I_\ell)\oplus\bar B(0,r_m)\bigr)\cap\bar B(c,r_{\mathrm{clip}}),
\qquad\mathcal L^{(k)}_m=\textstyle\bigcup_\ell\mathcal L^{(k)}_{m,\ell}.$$

One wall per $(k,m,\ell)$, never one per $(k,m)$: two separated lumps in the ball are two
approaches, and one connected lump that *wraps* the centroid is also two — its single wall is
non-separating. The pieces cover $\mathcal L^{(k)}_m$ for **any** partition of the band, so the cut
position is a quality knob, never a correctness one. **Never bridge $[\min I,\max I]$ across all
of $I$**: that fuses walls that must stay separate and spans the corridor the trajectory is
entitled to pass through. (With a station the profile is over $(\tau,u)$ with the nearest stretch
in closed form for each $\tau$; the cut rule is the same.)

### 2.5 The wall

Fix one piece $\mathcal L=\mathcal L^{(k)}_{m,\ell}$.

**D9 (supporting half-space).**

$$y^\star\in\arg\min_{y\in\mathcal L}\lVert c-y\rVert,\qquad
n=\frac{c-y^\star}{\lVert c-y^\star\rVert},\qquad
b=h_{\mathcal L}(n)=\max_{z\in\mathcal L}n^\top z .$$

$n$ points from the nearest point of the piece back to the centroid; $b$ is the **support** of the
piece in direction $n$. The plane $n^\top z=b$ rests on the piece's most protruding point along
$n$, which is in general not $y^\star$.

**L5 (support functions do not see convexity).** For compact $S$ and any $n$,
$h_S(n)=h_{\operatorname{conv}S}(n)$. *Proof: a linear functional attains its maximum over the
hull at an extreme point, and every extreme point of the hull belongs to $S$.* Consequence: the
wall against $\mathcal L$ and the wall against $\operatorname{conv}\mathcal L$ are the same wall.
The convexity of $\mathcal L$ is irrelevant to the construction, and **no convex
over-approximation of the clipped volume is ever formed.** This is the step at which the
tangent plane fails: a tangent to the tube at $f$ is a support of $\mathcal K_m$ only when
$\mathcal K_m$ is convex.

**L6 (half-space containment).** $\mathcal L\subseteq\{z:n^\top z\le b\}$, immediately from D9. No
convexity of $\mathcal K_m$ is used.

**L7 (the direction does not depend on $y^\star$).** Let $f_\ell=\gamma_m(\tau^\star_\ell)$ be
the nearest centreline point within approach $\ell$. Then

$$n=\frac{c-f_\ell}{\lVert c-f_\ell\rVert}$$

whenever D9's quotient exists, and this form is used *also when it does not*: with the centroid
inside the keep-out zone, $y^\star=c$ and D9 reads $0/0$, but $c-f_\ell$ is still defined and $b$
is still the support of the piece along it. No fallback set, no hull projection, no separate
recipe. The only undefined case is $c=f_\ell$ — and, for a station, a centreline passing within
$r_m$ of the station, whose shadow is all of space beyond the body. Both are **dropped walls,
counted** per iteration, never silently out of reach: an approach in reach always yields a wall,
and refusing one was a measured defect.

**Wrapping.** A piece may reach past $c$ along $n$, giving $n^\top c<b$ — a negative margin. L6
still holds, so the wall is valid; what fails is *separation*, which the certificate never rests
on. The negative margin is the row telling the solver how far it has to climb out.

**Computing $b$.** $h_{\mathcal L}(n)$ is bounded from above by De Casteljau subdivision of the
approach's centreline (and of the stretch parameter $u$ when a station is present): by L1 applied
to the *obstacle*, the support of a Bézier arc plus a ball is at most the largest $n^\top$ of the
subdivided control points plus $r_m$, with the analogous bound over cells in $u$. The code uses
this **rigorous ceiling** $\hat b\ge b$; L6 holds a fortiori. Hulling sampled centreline points
is forbidden — a curve bulges outside such a hull.

### 2.6 Rows and the segment certificate

**D10 (exact rows).** For every control point of segment $k$ and every wall $(k,m,\ell)$:

$$n_{km\ell}^\top Q^{(k)}_i\ \ge\ \hat b_{km\ell},\qquad i=0,\dots,N
\quad\Longleftrightarrow\quad
n_{km\ell}^\top\sum_jA^{(k)}_{ij}P_j\ \ge\ \hat b_{km\ell}.$$

**L8 (segment certificate).** If the $N+1$ rows of one wall hold, then $n^\top Z(u)\ge\hat b$ for
every $u$ in segment $k$, hence $Z(u)\notin\operatorname{int}\mathcal L$: the curve does not enter
the interior of the clipped keep-out volume for all times in the segment — not at collocation
nodes. *Proof: L1 gives $Z(u)$ as a convex combination of the $Q_i$, linearity gives
$n^\top Z(u)\ge\hat b$, and an interior point of $\mathcal L$ has $n^\top z<b\le\hat b$.* Nowhere
is $\mathcal L$ assumed convex. One plane per $(k,m,\ell)$: per-control-point planes destroy the
proof, because the step from $Q_i$ to $Z(u)$ uses the same $n$ for all $i$.

### 2.7 Reachability and coverage — the guarantee that is ours

L8 certifies against $\mathcal L^{(k)}_m=\mathcal K_m\cap\bar B(c,r_{\mathrm{clip}})$, **not**
against $\mathcal K_m$. The trust region closes the difference, and nothing else does.

**L9 (reachability).** If $\lVert\mathbf p^+-\mathbf p\rVert_\infty\le\Delta$ then for every $u$
in segment $k$, $Z^+(u)\in\bar B\bigl(c,\ E^{(k)}+\Delta\sqrt{d_s+1}\bigr)$. *Proof: each
$Q^+_i-Q_i$ is a convex combination of vectors of $\infty$-norm at most $\Delta$, so
$\lVert Q^+_i-c\rVert\le E^{(k)}+\Delta\sqrt{d_s+1}$, and the ball is convex.*

**Condition C (coverage).**

$$r_{\mathrm{clip}}\ \ge\ E^{(k)}+\Delta\sqrt{d_s+1}.$$

*In words: the clipping radius is at least the segment's own radius about its centroid plus the
furthest one trust step can carry a control point.* Under C, the ball contains the whole
next-iterate segment, so L8 certifies against the **full** $\mathcal K_m$. C does not involve
$r_m$ or the obstacle. With the wall built against the clipped volume, C is exactly coverage —
there is no accidental coverage beyond it.

Four regimes of the construction floor ($r_{\mathrm{floor}}=r_m$), writing $R=E^{(k)}+\Delta\sqrt{d_s+1}$:

| regime | $r_{\mathrm{clip}}$ | C holds? |
|---|---|---|
| $d>r_{\mathrm{clip,max}}$ — out of reach | $r_{\mathrm{clip,max}}$ | yes, and no row is needed: the segment cannot reach $\mathcal K_m$ in one step |
| $R\le d\le r_{\mathrm{clip,max}}$ | $d$ | yes |
| $r_m<d<R$ — segment close | $d$ | **no — this is the hole** |
| $d\le r_m$ — centroid inside | $r_m$ | no unless $r_m\ge R$; the floor fixes the understated penetration, not coverage |

The hole sits at small $d$, exactly where the constraint binds. **The reach floor**
(`sound_clip`, default on since 2026-09-01) raises $r_{\mathrm{floor}}$ to $\max(r_m,R)$ and makes
C hold unconditionally — sound by construction — at the price of conservatism precisely where the
row is active, paid in iterations rather than correctness. Which floor a run used is a reported
parameter. In either case the number of walls failing C is exported per iteration as
`unsound_clips`, and the figure gate (D12) refuses any run whose count is nonzero. **The hole is a
number, never a caveat.**

### 2.8 The subproblem

**P (the convex subproblem at reference $\mathbf p^{\mathrm{ref}}$).**

$$\min_{\mathbf p,\ \sigma\ge0}\quad
\tfrac12\,\mathbf p^\top H\mathbf p\;+\;w_t\,t_N\;+\;\lambda\sum\sigma$$

| | rows | slack |
|---|---|---|
| boundary | $P_0=z^{\mathrm{start}}$; $x_N=x^{\mathrm{goal}}$; $t_N$ free iff `free_arrival_time`, else pinned | none |
| workspace box | $\underline z\le P_j\le\bar z$ per coordinate, spatial and time bounds | none |
| monotonicity | $t_{j+1}-t_j\ge\delta>0$ | none |
| slant limit (speed cap) | $\lVert x_{j+1}-x_j\rVert\le v_{\max}\,(t_{j+1}-t_j)$ — second-order cone | **none** |
| keep-out, per wall $(k,m,\ell)$ and control point $i$ | the linearised row below, $+\ \sigma_{km\ell i}$ | elastic |
| line of sight, per wall $(k,m,\ell,\text{station})$ | same form | elastic |
| trust region | $\lVert\mathbf p-\mathbf p^{\mathrm{ref}}\rVert_\infty\le\Delta$ | none |

**The linearised keep-out row — what the QP is given.** The exact row D10 freezes $n$ and $\hat b$
at the reference, but both move with the centroid once the solver moves the control points. Write
$g_{ki}(\mathbf p)=n^\top Q^{(k)}_i-\hat b$ and let $s=c+(\hat b-n^\top c)\,n$ be the support
point on the plane. The row imposed is the first-order model

$$g_{ki}(\mathbf p^{\mathrm{ref}})+\sum_j\bigl(A^{(k)}_{ij}\,n+w^{(k)}_j\,\kappa_{ki}\bigr)^{\!\top}(P_j-P_j^{\mathrm{ref}})\ \ge\ 0,\qquad
\kappa_{ki}=\frac{(I-P_F)(I-nn^\top)\,(Q^{(k)}_i-s)}{\lVert c-y^\star\rVert},$$

where $P_F$ projects onto the active face of the piece at $y^\star$ ($P_F=0$ at a cap end,
$P_F=ee^\top/\lVert e\rVert^2$ on a straight tube body with axis $e$ — the capsule builder's two
branches, which is why a degree-one obstacle produces identical rows from the old code and this).
The term $\kappa$ is the derivative of the aiming direction through the centroid; without it the
plane pivots out from under an accepted step and the ratio test rejects almost everything. **This
row is a first-order model and is not a conservative restriction**: a point satisfying it may have
$g_{ki}<0$. Soundness is untouched because every certificate, merit and gate rebuilds the **exact**
rows D10 at the iterate it grades. The manuscript states both rows and which one certifies.

**$H$ is blind to timing.** It is assembled from second differences of the spatial coordinates
only, so $He_{t_j}=0$ for every time-coordinate basis vector; two control-point sets with equal
spatial coordinates and any time coordinates score identically. *Checked by test.* There is no
quadratic acceleration energy to be had: physical velocity is a ratio of two Béziers, acceleration
a ratio with a cubic denominator, neither polynomial in the control points. Physics enters through
the constraints.

**L10 (the slant limit is sufficient).** With $x'(u)=N\sum_jB^{N-1}_j(u)(x_{j+1}-x_j)$ and
$\tau'(u)=N\sum_jB^{N-1}_j(u)(t_{j+1}-t_j)$, non-negativity of the Bernstein weights and the
triangle inequality give $\lVert x'(u)\rVert\le v_{\max}\,\tau'(u)$, so physical speed
$\lVert x'\rVert/\tau'\le v_{\max}$ everywhere. Clearing the denominator requires $\tau'>0$, which
is what makes monotonicity physics-load-bearing. The cone carries no slack: the penalty may relax
keep-out rows, never the physics.

**Arrival time.** A Bézier passes through its last control point, so arrival time is $t_N$ and its
penalty $w_tt_N$ is linear; the subproblem stays a QP with cones. **Trap, guarded in code:** a
time penalty with no speed cap collapses arrival to $\delta\times N$ regardless of weight; the two
land together or not at all. The energy-versus-time weight is a reported scenario parameter and its
sweep is a legitimate figure.

**No obstacle penalty in the objective** (PI instruction, 2026-08-20). Avoidance is the hard
half-space row; the slack and its one-norm penalty are a numerical device — SNOPT's elastic mode —
and the manuscript says so in those words.

### 2.9 The certificate and the figure gate

**D11 (independent certificate).**

$$\mu(\mathbf p)=\min_{u\in[0,1]}\ \min_{m:\ \tau(u)\in[T^0_m,T^1_m]}\ \bigl\lVert x(u)-\pi_m(\tau(u))\bigr\rVert-r_m ,$$

spatial distance at each sample's own time minus the radius, obstacles skipped outside their
existence window; the line-of-sight margin is the analogous sampled segment-distance check against
the station. $\mu$ touches neither $n$, $\hat b$, $\mathcal L$, $r_{\mathrm{clip}}$ nor the axis
scale, so it is independent of everything it checks. The run is feasible iff $\mu>0$. It is
evaluated at the **returned** iterate, and where C fails it is the only statement there is.

**D12 (`figure_grade`).** A run may feed a figure or a table iff *all* of: `converged`; the
hull-row certificate at the returned iterate at most the certificate tolerance; $\mu>0$; total
slack at most the slack tolerance; the line-of-sight certificate where a station exists;
`unsound_clips` equal to zero. Each condition has been shown to sink the gate alone. The figure
pipeline refuses to draw a run that fails it, and a build that cannot find a figure-grade sidecar
fails.

---

## 3. Algorithm

**A (successive convexification with a ratio test and in-loop elastic weight).** State carried
across iterations: reference $\mathbf p$, trust radius $\Delta$ with floor and ceiling and its
initial value $\Delta_0$, elastic weight $\lambda$ with raise count, the three streak counters,
and the best feasible iterate seen.

1. **Build rows at the reference.** For every segment and obstacle (and station): D6–D9, the
   exact rows D10, the linearised rows of P, the per-wall `sound` flag against C, and the dropped-
   wall count. $\Delta$ enters here through $r_{\mathrm{clip,max}}$ and the reach floor.
2. **Solve P** with Clarabel (interior-point conic), reading the dual vector. Candidate
   $\mathbf p^{\mathrm c}$, slack totals, largest keep-out dual.
3. **Bootstrap.** If the reference violates a row that carries no slack (box, monotonicity — the
   straight-line seed generally does), accept the candidate unconditionally as a repair step and
   go to 1. A run that cannot repair in one trust step from its seed stops with
   `FIRST_QP_INFEASIBLE`, a label, not a repair.
4. **Ratio test.** Convex merit $L(\cdot)=J(\cdot)+\lambda\cdot$(violation of the rows the QP was
   given); true merit $T(\cdot)=J(\cdot)+\lambda\cdot$(violation of rows **rebuilt** at the
   point). $J$ is exactly quadratic, so the two differ for one reason only: the half-spaces moved.
   $\mathrm{pred}=L(\mathbf p)-L(\mathbf p^{\mathrm c})$,
   $\mathrm{act}=T(\mathbf p)-T(\mathbf p^{\mathrm c})$, $\rho=\mathrm{act}/\mathrm{pred}$;
   a prediction below the cancellation floor is a null step, accepted iff it does not make things
   worse.
5. **Accept / reject / trust update.** Accept iff $\rho>\eta$ (current value $0.1$); expand
   $\Delta\leftarrow\min(2\Delta,\Delta_{\max})$ iff $\rho>0.9$; on reject halve $\Delta$.
   Accepted iterates update the best-feasible record.
6. **Success stops — both require the exact certificate.** `MERIT_STREAK`: $K$ consecutive
   accepted steps (current $K=3$) with negligible relative merit change **and** rebuilt exact
   violation at most the certificate tolerance. `STATIONARY`: $K$ consecutive iterations with
   $|\mathrm{pred}|$ below tolerance at a certified reference. A single quiet iteration is not
   evidence when the half-spaces re-aim between iterations.
7. **Elastic weight (SNOPT's rule, this code's schedule).** Raise $\lambda\leftarrow10\lambda$ up
   to the cap (current $10^5$) and **continue from the same iterate with $\Delta$ reset to
   $\Delta_0$**, when either (a) the trust radius falls below its floor at a violating reference —
   the elastic problem at this weight is finished and slack remains — or (b) the stall trigger
   fires: the exact violation has not improved by one percent on the minimum of the last five
   iterations for ten consecutive infeasible iterations, or has set no one-percent-better level
   best for twenty. Feasible iterations freeze both counters; the weight is never raised at a
   feasible iterate. The escalate-and-continue rule is Gill, Murray & Saunders' elastic mode; the
   fixed base and ×10 schedule are ours; the trigger is a stall heuristic with no published form
   claimed. Exactness of the one-norm penalty at finite weight is Han & Mangasarian; the measured
   exactness margin is the largest keep-out dual against $\lambda$, and complementarity (slack
   active iff that dual sits at the weight) is tested from both sides.
8. **Failure stops.** `TRUST_COLLAPSE` with the weight at its cap or held (converged iff the
   reference is certified, otherwise a genuine failure reported as one); `ITERATION_CAP`;
   `QP_FAILURE`. A stall the weight cannot escape does **not** stop the loop; it runs to its own
   stops.
9. **Return** the final reference — unless it penetrates ($\mu<0$) and a feasible iterate was
   seen, in which case the best-clearance feasible iterate is returned and the swap is flagged as
   `returned_best_iterate`, never silent. Evaluate D11 and D12 **at the returned iterate**; export per-iteration `unsound_clips`, dropped walls, $\rho$
   statistics, weight raises and the start weight the row ran from.

**What `converged` asserts and does not.** Feasibility (exact rows and $\mu$) plus no further
progress. Not stationarity of the original problem: duals are read, but no KKT residual is tested.
The manuscript states this in the limitations, not the appendix.

**Properties worth one paragraph each in the manuscript.**
- $\rho$ is a pure measurement of half-space motion (step 4) — the paper's per-iterate
  reconstruction figure is this number over iterations.
- The row count of P is $\sum_k\#\{(m,\ell)\text{ in reach of }k\}\times(N+1)$, bounded by
  $M\times(\text{obstacles})\times(\text{approaches})\times(N+1)$, with no term that depends on
  the shape of any keep-out set. This is consequence 3.
- The solver is dimension-generic: $d_s$ comes from the array shape. Three spatial dimensions
  plus time is a scenario definition, not a solver change.

**Parameters reported with every run** (never tuned silently): $N$, $M$, $\Delta_0$ and its
bounds, $\lambda_0$ and whether escalation was held, $w_t$, $v_{\max}$, $\delta$,
`free_arrival_time`, `sound_clip`, the seed, and the solver build.

---

## 4. Experiments — each stated as something that can fail

### 4.1 Rules

- **One measurement pass, on a frozen solver.** (E1 is closed — see §2.4.) Three different
  builds produced the numbers currently scattered through the demo specs; none may be compared
  with another and none reaches the manuscript.
- Numbers reach the manuscript only from figure-grade sidecars (D12), injected at build time;
  the LaTeX build fails on a missing or non-figure-grade sidecar.
- Every run records the solver build (the compiled extension's mtime, not the package's), the git
  sha, the machine, and the full parameter list of §3.
- Timing protocol: wall-clock of the whole solve including row construction, repeats reported as a
  spread, same machine for us and for the baseline.
- Each demo is a **pair**: the constrained run against a baseline that differs in exactly one
  named thing. If the baseline also succeeds, the constraint was slack and the figure proves
  nothing — change the scene, do not soften the claim.

### 4.2 The demos and their jobs

| demo | scene | carries | must show | fails if | status |
|---|---|---|---|---|---|
| **B — C2-link continuity** ([spec](demo_specs/B_c2_link.md)) | `loiter`: occluder on a circular lap above a ground station, corridor tangent to the shadow ring, $d_s=3$ | consequence 1 — **the thesis demonstration**; the guarantee via the line-of-sight certificate | constrained run keeps the link start to finish while the pair loses it for a measurable interval; the escape is retiming, not a sidestep; the third dimension is load-bearing | the pair also keeps the link; the figure and the tested pair run different configurations (audit finding, open) | scene, pair and tests exist; journal figure, mission paragraph and one honest baseline arm missing |
| **C — traffic** ([spec](demo_specs/C_traffic_and_backbone.md)) | scheduled traffic family, obstacle count as the axis, every obstacle moving, 2 + t and 3 + t twins with the same conflict schedule | consequence 3 — **the cost evidence**; the scaling experiment | walls and rows grow linearly in obstacle count and stay inside the segments × obstacles bound; solve time against count; success rate at each count | a vacuous count axis (obstacles never approached produce no wall — placement rule); a confounded axis (horizon knob moving the geometry); no wall-clock number | families module and tests exist uncommitted; `tools/bench.py` and `tools/make_tables.py` not written |
| **A — rotating target** ([spec](demo_specs/A_rotating_target.md)) | chaser to a fixed port on a stabilised bus whose appendages rotate; target-fixed frame, kinematic only | consequences 1 and 2 in one picture; the cross-domain demonstration | the decision is *when* to cross, made inside one subproblem; a cluster of balls on Bézier circular paths costs the same kind of row as a static ball | the figure implies gravity, Δv, tumbling or a rendezvous result; the pair (fixed timing) also crosses cleanly | specification and a cost spike only; no production scenario |
| ~~**WAKE**~~ ([spec](demo_specs/WAKE_vortex.md)) | — | — | — | — | **OUT, decided 2026-09-21.** It demonstrates nothing B and C do not. The specification stays on disk as a record; no track builds it. |

> **`DECISIONS.md` does not exist and never did** (checked across all branches, 2026-09-21). The
> three demo specifications cite it about ten times, sometimes with quoted text. Those citations
> are unresolvable: treat any claim sourced to it as unverified, and this section as the authority
> on which demo carries what.

**Where the centre of the paper is, stated once so it is not stated three ways.** The *thesis* —
that constraints can be built from a support oracle — is demonstrated by **B**, because the shadow
is the hazard nobody would choose to carve. The *cost* consequence that makes the thesis worth
having is carried by **C**. The *mechanism* is shown by figure slot 2 (per-iterate reconstruction),
which belongs to no demo. Those are three different jobs, not three candidates for one job.

Order in the manuscript: B first as the thesis demonstration, C as the quantitative evidence, A as
the generality demonstration. WAKE is out (§6).

### 4.3 The comparison baseline — the experiment that decides the paper's impact

**Baseline: Osburn's space-time GCS, re-implemented, dimension-agnostic from the first line.**
IRIS-sampled convex sets over the lifted space, a graph of convex sets with Bézier control points
per vertex, time-monotonicity and velocity-cone rows, Clarabel. Written so that the spatial part
of every vector is `x[:-1]` and the time part `x[-1]`; the space-time obstacle is the convex hull
of start and end vertices in $d_s+1$ dimensions (a prism), for any $d_s$. Whether to reimplement
the graph-of-convex-sets machinery directly (CVXPY, as Osburn did) or on Drake's n-dimensional
IRIS and `GraphOfConvexSets` with Clarabel is a ten-minute check when the work starts: the deciding
question is whether Drake accepts the velocity cone and time-monotonicity as custom rows.

**Stage 1 — 2D + time.** The twin of Osburn's one-moving-obstacle case (the agent speeds up to slip
past, timing as a real decision — behaviourally our `wall`) and of his cluttered twenty-obstacle
case, run on both methods, same solver, same machine. Report: feasible or not by the independent
$\mu$ (theirs checked by the same $\mu$, not by their own sets), path length or the smoothness
value, solve time, and for the baseline the IRIS sample count swept as in his Table I. **What
refutes us:** the baseline matches our trajectory quality at comparable time with no initial guess
— then the paper's contribution is consequences 1 and 3 only, and it says so.

**Stage 2 — 3D + time, decided after Stage 1's cost is known.** The same code with $d_s=3$; our
side is `door3d` / `fence3d` / the 3 + t traffic twin, which already run. The baseline needs a
4D obstacle builder for our ball obstacles (a polytope approximation of the ball, swept) and the
same sample sweep. **This is the only experiment that tests the sentence the claim rests on** —
practical at three spatial dimensions plus time, where the cell complex is the dominant cost.
Expected failure mode of the baseline: IRIS coverage in 4D too poor to find any path. That is a
result, and the sample sweep is the defence against "you did not tune the baseline."

**Honest counter-position, stated in the same section:** no initial guess needed on their side;
global with respect to their graph; sub-second on small cases. We are local, seeded, and the
passing homotopy class is the seed's.

**A second baseline, local.** A plain local trajectory optimiser without the per-iterate
construction — e.g. the same QP with the tangent plane at $f$ instead of the support of the
clipped piece, and no approach cut. Tests whether the construction adds value beyond ordinary
successive convexification. Cheap: it is the retired construction, and the measured defects
(non-separating walls on a wrapping bend, penetration certified as feasible) are the expected
outcome; re-measure on the frozen build before quoting.

### 4.4 The scaling experiment

Axis: obstacle count in the traffic family, the placement rule guaranteeing every obstacle is
approached. Report per count: walls, rows ($=$ walls $\times(N+1)$ — an identity the bench
asserts), solve time with spread, iterations, weight raises, success rate, `unsound_clips`. The
linear-size claim is overlaid as the bound, not fitted. The 2 + t and 3 + t twins share a
conflict schedule so dimension is the only difference between them.

### 4.5 Coverage — the limitations section's number

With the reach floor **off**: how often C fails at a returned iterate (`unsound_clips` rate per
iteration and per run), and how often a returned iterate satisfies every half-space while
$\mu\le0$ — the hole actually firing. With the reach floor **on**: what it costs in iterations
and in minimum clearance on the same configurations. Both arms on the same 28-configuration
battery already used for `SOLVER.md` §Measurements.

### 4.6 Ablations worth one table

- Rotation term $\kappa$ on and off: acceptance ratio and iterations.
- Reach floor on and off (§4.5).
- Free arrival with the time weight swept: arrival time against the smoothness value — the
  energy-versus-time trade-off as a curve.
- Start weight $\lambda_0$: the escalation path depends on it; report that no single start
  dominates rather than hide it behind a retry.

### 4.7 Reliability — the bar solver work stops at (decided 2026-09-29)

**Why this section exists.** Solver reliability was worked on for a month with no finish line.
The 28-configuration battery of §4.5 varies discretization on seven hand-built scenes, so a pass
rate over it is a rate over tuning choices, not over problems. And a timing-only retry on the
unchanged 2026-09-08 build passed four of that battery's five failures
(`artifacts/koz-principled-20260922/seed_controls/portfolio.json`, a Codex run, not re-run here):
most of what the wall-construction fixes were chasing was initialization, which no wall
construction can fix. The battery stays as the regression suite and the §4.5 coverage battery. It
is not the reliability scoreboard.

**The benchmark is §4.4's run, not a second experiment.** The traffic family
(`spacetime_bezier/families.py`) in two arms that share every seed and every crossing: `traffic`
(straight tracks, which lift to convex capsules) and `traffic_curved` (every aircraft turning
through the same crossing point, time and velocity, so the tube is not convex). Counts 1, 2, 4, 8,
12, 16, 20; twenty seeds per count; degree 8 and 8 segments fixed; one frozen build for the whole
pass. The arm-to-arm difference is what the non-convex case costs.

**Every run lands in exactly one bin, read at the returned iterate.**

| bin | test | bar |
|---|---|---|
| false certificate | hull-row certificate within tolerance, yet the independent $\mu\le0$ (D11) | **zero, across every run.** One occurrence is a bug and stops the pass. |
| solved | `figure_grade` (D12) | reported per count and per arm |
| safe, uncertified | $\mu>0$, gate fails | reported: conservatism |
| failed | $\mu\le0$, no certificate claimed | reported, and split by the seed test |

**The seed test.** Every run outside *solved* is re-run from a fixed set of timing seeds —
straight, wait, hurry. If any seed solves it, it is an **initialization failure**; if none does, a
**construction failure**. Wait and hurry are not implemented in the package yet
(`build_initial_guess` knows `straight` and `quadratic_bow`); the bench defines them, and the row
names the seed it came from.

**The stopping rule.**
- False certificates are zero.
- Solver work may target **construction failures only**. Initialization failures go to the
  limitations section, which §1.4 already commits to: the method is local and seeded.
- Solved with seed retries at least 90% at every count, on both arms. **The 90 is a guess**, made
  2026-09-29 before any data; it is revisited once, after the first pass, and never again.

**What refutes it.** Any false certificate. A curved arm that solves far less often than the
straight arm. Construction failures clustering at high counts.

**Named risk — the curved arm may be too gentle.** At the initial guess the two arms put nearly the
same walls in front of the solver, one approach per wall. So does the hand-built `curve` scene,
whose failures are real, so the initial guess cannot settle it. If the first pass shows the two
arms' outcomes identical run for run, the turn is not exercising the construction, and the turn
rate goes up before any number is quoted.

---

## 5. Figure slots, limitations, and what the README keeps

**Figure slots** (from README, with the section that fills each):

1. Formulation schematic — D6–D9 in one picture: clipped volume, clipping ball, supporting
   half-space, trust box; C should be readable from it.
2. Per-iterate reconstruction — the same segment across iterations, the half-space re-aiming,
   with $\rho$ beneath. **The central claim lives in this figure.**
3. Line-of-sight pair — demo B; sight margin over time under the geometry.
4. Timing pair — demo A; the agent waits for the gap.
5. Scaling curve — §4.4, with the baseline's sample-sweep curve on the same axes where Stage 1 or
   2 has run.
6. Coverage — §4.5.

**Limitations, in their own section, in this order:** local and seeded; `converged` is not
stationarity; the hole and its measured rate; conservatism of the reach floor; the baseline's
strengths; the scope boundary (deterministic known motion, offline, binary occlusion).

**README** keeps the short form — claim shape, figure slots, status — and points here. Its status
line ("Nothing in this directory but this file") is stale and is to be updated when the README is
next edited.

---

## 6. Decisions taken, and what is still open

### Taken

- **Audience — the planning and trajectory-optimization methods community. Decided 2026-09-20.**
  Not aerospace guidance and control, and not air-traffic systems. Reasons, in order: the
  contribution is a construction rather than a mission result; the only physics in the formulation
  is a speed cap, which this audience treats as normal and an aerospace-control audience treats as
  disqualifying; the baseline (IRIS, GCS, ST-GCS) is this audience's own literature; idea 2
  (`idea/risk_field.md`) addresses the same audience, so the portfolio coheres; and the aerospace
  community already has the KSAS conference version and will see the poster in November.
  **Consequences the rest of this file must reflect:** the shadow is the thesis demonstration, not
  a feature; scaling is the cost evidence; the rotating-target demo demonstrates generality; code
  release becomes expected rather than optional.
- **Framing — the support oracle (§1.1). Decided 2026-09-20.** Stated as what the construction
  consumes, not as the absence of a decomposition.
- **E1 — refused. Decided 2026-09-18** (a hail mary, not a fix). The committed clip rule of §2.4
  stands. → `WORKSTREAM.md` `journal.freeze`
- **WAKE — out. Decided 2026-09-21.** It demonstrates nothing B and C do not. Its specification
  stays as a record; no track builds it.

### Open

1. **Venue.** The advisor declines to advise until he has read a draft, so the deliverable that
   unblocks this is a **skeleton draft** — introduction, contributions, figure list and experiment
   table, roughly five pages, no results — not a full manuscript. Room-A venues want the same paper
   shape, so writing is not blocked while this is open. Graduation is 2027-02.
2. **Baseline by public implementation or by re-implementation — an open conflict, recorded
   2026-09-21.** §4.3 specifies re-implementing Osburn. `pi_report_2026-09-09.md` §"비교 대상"
   recommends the opposite: run a *public* graph-of-convex-sets implementation on our scenarios,
   two weeks, dropping to ablation-only if it will not run — because a self-built baseline that
   comes out weak leaves a reviewer unable to tell a method difference from an implementation
   difference, and running published code removes that objection entirely. The audience decision
   sharpens this rather than settling it: room-A reviewers are the authors of those
   implementations. **§4.3 has not answered the objection and must, before `journal.baseline`
   starts.** A candidate resolution, not yet chosen: public implementation for the 2D + time
   fairness argument, our own only for 3D + time, where no public implementation runs.
3. **Baseline Stage 2 go / no-go**, after Stage 1's cost is known (§4.3). Stage 2 is the
   only answer to this audience's central objection — *your method is local, GCS is global* — so
   "no-go" costs the paper its rebuttal, not merely an experiment.
4. **Submitted or accepted by 2027-02?** Whether the degree requires the journal paper to be
   *submitted* or *accepted* by graduation has been asked and not answered. It decides whether a
   slow venue is eligible at all, and the advisor's venue advice lands on top of it.
5. **The mission citation for demo B** — naming a regulation or a continuity requirement is a
   literature decision (demo B spec §7). Lower stakes now: for this audience the mission is
   motivation, not the result.
