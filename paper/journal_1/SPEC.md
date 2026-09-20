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

**Companion files.** [`README.md`](README.md) is the short artifact document (claim shape, figure
slots, status). The four demo specifications in [`demo_specs/`](demo_specs/) are build
specifications and keep their own detail; §4 below only assigns each a job.

**Numbering.** Definitions D1…, Lemmas L1…, one Problem P, one Algorithm A, one Condition C. The
labels are for cross-reference *inside this file and the manuscript*; `idea/spacetime.md`
deliberately carries none, so a statement there is referred to by what it says.

---

## 1. Claim and positioning

### 1.1 The claim

> A **decomposition-free** convexification of space-time collision avoidance, rebuilt **per
> iterate** around the current trajectory. Each (segment, obstacle, approach) triple contributes one
> supporting half-space of a clipped piece of the keep-out volume; the subproblem is a QP with
> second-order cones, its size is linear in segments × obstacles, and it carries no discrete
> structure — no integer variables, no region graph, no cell selection. Conservatism is bounded by
> the trust region, not by the geometric complexity of the free space.

Opening sentence rule, inherited: **never "time as a coordinate."** That lift is Osburn, Peterson
and Salmon, August 2025. What is new is that nothing is decomposed.

### 1.2 Consequences — narrowed 2026-09-17

Each consequence is evidence that the claim buys something. Each is stated against the specific
prior work it actually separates us from; a consequence overstated against the whole corridor
family is the sentence a reviewer who knows Tang rejects.

| consequence | separates us from | does **not** separate us from | demo that carries it |
|---|---|---|---|
| **1. Moving obstacles of arbitrary polynomial motion are structurally free.** An obstacle is a lifted Bézier centreline of any degree with a radius; a curving, accelerating body produces the same rows as a static one. | Osburn (static or constant-velocity only; "more complex motions require finer sampling and will only approximate"); ST-GCS (piecewise-constant velocity trajectories); every space-only decomposition | — | A (rotating appendages), B (occluder on a circular lap) |
| **2. Timing is a decision inside the same convex subproblem.** Arrival time, the speed cap and the time penalty are variables and rows of the one QP; there is no path-then-time-allocation stage. | Osburn (end point pinned at a fixed space-time point; cost is spatial path length, so time is never optimised); space-only corridor methods | **ST-GCS** (Tang et al.: time-optimal, free arrival, inside a space-time graph) | A (when to cross), WAKE (wait for decay) if built |
| **3. Row count is independent of the geometric complexity of the keep-out set.** Rows are linear in segments × obstacles whatever the shape of the volume; the twisting shadow of a moving station costs one row block, the same as a ball. A decomposition's set count grows with the geometry it has to carve. | the whole corridor family, on cost — not on expressibility | — | B (line of sight), C (obstacle count axis) |

**Retired wording, do not reintroduce:** "a decomposition cannot express visibility" (false as
stated — a space-time decomposition can carve any closed set); "a decomposition must be fixed
before the timing that decides which cells are free" (true only of space-only decomposition; a
space-time decomposition never commits to timing). Consequence 3 is about *cost*, and the paper
says so.

**Two stale cells in the prior-art table** at `idea/spacetime.md` §"The corridor / convex-opt
family in detail", row "arrival time": Osburn is recorded as *free* — his examples pin the end
point at a fixed time and his cost is spatial length; and ours is recorded as *fixed* — the
`free_arrival_time` flag exists. Both cells are to be corrected when that file is next edited.

### 1.3 The guarantee — what was not given up

- **Continuous-time feasibility by hull containment** (L1, L8). Standard machinery; Bézier and
  B-spline planners have used it for years. Cite it, do not claim it.
- **The coverage condition** (C, §2.7) — ours. The half-spaces provably contain the clipped
  keep-out volume, and a stated, testable condition on the clipping radius makes the segment
  certificate cover the *whole* keep-out zone rather than the clipped piece. The residual hole is
  named, counted per iteration, and gates every figure. Nobody in the corridor family reports
  coverage; they report that the solver converged.
- **The independent certificate** (D11). Every reported guarantee is re-checked by a computation
  that shares nothing with the construction it checks.

### 1.4 What is not claimed

- Global optimality. The method is local; the passing homotopy class comes from the initialization.
- Stationarity. `converged` asserts feasibility plus no further progress; there is no KKT residual.
- Speed on small problems. Osburn's small cases take a fraction of a second; "we are fast" cannot
  carry the paper. The argument is how cost scales with dimension and obstacle count.
- Any physical model of the mission: no gravity or Δv (A), no wake physics (WAKE), no radar
  equation or detection probability (B). Binary occlusion, exact known motion, offline solve.
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
  sufficient condition in corridors; **MADER** for planes as decision variables, the alternative to
  linearising them.
- The three strengths of the graph family that every limitations section states: no initial guess
  needed; global with respect to their graph; small cases solved in under a second.

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

> **Code note.** An uncommitted experiment labelled `E1 (2026-09-09)` in
> `rust_optimizer/core/src/spacetime_obstacle.rs` replaces the clamp under `sound_clip` with the
> minimal sound radius $\min(r_{\mathrm{floor}},r_{\mathrm{clip,max}})$, discarding $d$. The
> 2026-09-09 measurement pass reported it as a regression. This specification states the
> **committed** rule above; E1 is landed or reverted before the measurement pass, never both.

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

- **One measurement pass, on a frozen solver, after E1 is landed or reverted.** Three different
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
| **B — C2-link continuity** ([spec](demo_specs/B_c2_link.md)) | `loiter`: occluder on a circular lap above a ground station, corridor tangent to the shadow ring, $d_s=3$ | consequences 1 and 3; the guarantee via the line-of-sight certificate | constrained run keeps the link start to finish while the pair loses it for a measurable interval; the escape is retiming, not a sidestep; the third dimension is load-bearing | the pair also keeps the link; the figure and the tested pair run different configurations (audit finding, open) | scene, pair and tests exist; journal figure, mission paragraph and one honest baseline arm missing |
| **C — traffic** ([spec](demo_specs/C_traffic_and_backbone.md)) | scheduled traffic family, obstacle count as the axis, every obstacle moving, 2 + t and 3 + t twins with the same conflict schedule | **the claim itself** and consequence 3; the scaling experiment | walls and rows grow linearly in obstacle count and stay inside the segments × obstacles bound; solve time against count; success rate at each count | a vacuous count axis (obstacles never approached produce no wall — placement rule); a confounded axis (horizon knob moving the geometry); no wall-clock number | families module and tests exist uncommitted; `tools/bench.py` and `tools/make_tables.py` not written |
| **A — rotating target** ([spec](demo_specs/A_rotating_target.md)) | chaser to a fixed port on a stabilised bus whose appendages rotate; target-fixed frame, kinematic only | consequences 1 and 2 in one picture; the cross-domain demonstration | the decision is *when* to cross, made inside one subproblem; a cluster of balls on Bézier circular paths costs the same kind of row as a static ball | the figure implies gravity, Δv, tumbling or a rendezvous result; the pair (fixed timing) also crosses cleanly | specification and a cost spike only; no production scenario |
| **WAKE** ([spec](demo_specs/WAKE_vortex.md)) | follower crosses a leader's decayed wake: chain of balls with birth and death times drifting with wind | consequence 2 (finite obstacle lifetime priced) | the run holds until the wake at the crossing has died and crosses not a second later | any sentence reads as a wake-encounter result | specification and exploratory solve; **in only if it shows something A and B do not** |

Order in the manuscript: B as the introductory example (why timing matters to a mission), C as
the quantitative evidence, A as the cross-domain demonstration. WAKE is a decision for §6.

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

---

## 5. Figure slots, limitations, and what the README keeps

**Figure slots** (from README, with the section that fills each):

1. Formulation schematic — D6–D9 in one picture: clipped volume, clipping ball, supporting
   half-space, trust box; C should be readable from it.
2. Per-iterate reconstruction — the same segment across iterations, the half-space re-aiming,
   with $\rho$ beneath. **The central claim lives in this figure.**
3. Line-of-sight pair — demo B; sight margin over time under the geometry.
4. Timing pair — demo A (or WAKE); the agent waits for the gap.
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

## 6. Open decisions — the user's and the PI's, not an agent's

1. **Venue.** Page budget, reviewer expectations on the baseline, and template all follow from it.
   Graduation is 2027-02.
2. **Baseline Stage 2 go / no-go**, after Stage 1's cost is known (§4.3).
3. **E1 — revert or land** before the measurement pass (§2.4 code note).
4. **WAKE in or out** (§4.2).
5. **The mission citation for demo B** — naming a regulation or a continuity requirement is a
   literature decision (demo B spec §7).
