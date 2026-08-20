# C3 — Bézier / B-spline safe-corridor formulations: external ground truth

Task C3. Everything below is from published sources, read directly (PDF text extracted, not
recalled). Where a source could not be read, it says so. Nothing here is inferred from this
repo's code or docs.

Sources actually read in full text: FASTER (arXiv:2001.04420), MADER (arXiv:2010.11061v2),
Preiss et al. (arXiv:1704.04852), Teach-Repeat-Replan (arXiv:1907.00520), Deits & Tedrake
(MIT CSAIL PDF), Kielas-Jensen & Cichella (arXiv:2010.09992), Deolasee et al.
(arXiv:2209.15150). Read via rendered HTML only: EGO-Planner (ar5iv 2008.08835), Fast-Planner
(ar5iv 1907.01531), Zhang et al. (ar5iv 2110.00065). **Not read: Gao et al. ICRA 2018** — see
"Disagreements / uncertainty".

---

## Q1. One half-space per (segment, obstacle) applied to ALL control points — or one per (segment, control point, obstacle)?

**Answer: one convex region / one half-space per (segment, obstacle-or-corridor), applied to
every control point of that segment.** The literature is unanimous. No certificate-carrying
paper found assigns a different plane to different control points of the same segment.

| Source | Statement |
|---|---|
| Preiss, Hönig, Ayanian, Sukhatme, *Downwash-Aware Trajectory Planning for Large Quadrotor Teams*, IROS 2017 (arXiv:1704.04852) | QP (14): `y^i_{k,d} ∈ P^i_k  ∀ i, k, d` — the same polyhedron `P^i_k` for **all** control points `d` of piece `k`. `P^i_k` is itself built (Sect. IV-A) as "the intersection of: N−1 half-spaces separating r_i from r_j … N_obs half-spaces separating r_i from O_1 … O_{N_obs}", i.e. **one half-space per (segment, other object)**, each from the SVM (9). |
| Tordesillas & How, *FASTER: Fast and Safe Trajectory Planner for Navigation in Unknown Environments*, IROS 2019 / arXiv:2001.04420, §IV (local planner), MIQP (1) | `b_np = 1 ⟹ { A_p r_n0 ≤ c_p, A_p r_n1 ≤ c_p, A_p r_n2 ≤ c_p, A_p r_n3 ≤ c_p } ∀n, ∀p`. Prose: "As a Bézier curve is contained in the convex hull of its control points, we can ensure that the trajectory will be completely contained in this convex corridor by forcing that **all the control points of an interval n are in the same polyhedron**." |
| Gao, Wang, Zhou, Han, Pan, Shen, *Teach-Repeat-Replan*, IEEE T-RO 36(5), 2020 (arXiv:1907.00520), §V-A-3 "Safety Constraints", Eq. (8) | "Thanks to the convex hull property, an entire Bézier curve is confined within the convex hull formed by all its control points. Therefore we constrain control points using hyperplane functions obtained in Eq. 1." Eq. (8) applies the *same* face list `a^x_0·c + a^y_0·c + a^z_0·c ≤ k_0, …, ≤ k_n` to the `i`-th control point `c^i_j` of the `j`-th piece, for every `i`. |
| Tordesillas & How, *MADER*, IEEE T-RO 38(1), 2022 (arXiv:2010.11061), §V-A, Eq. (3) | `n_ij^T c + d_ij > 0  ∀c ∈ C_ij, ∀i∈I, j∈J` and `n_ij^T q + d_ij < 0  ∀q ∈ Q^MV_j, ∀j∈J`. One plane `π_ij` per **(obstacle i, interval j)**; every control point `q` of interval `j` is on one side, every obstacle vertex `c` on the other. |
| Deits & Tedrake, *Efficient Mixed-Integer Planning for UAVs in Cluttered Environments*, ICRA 2015, Eq. (25) | Obstacle-face variant: `Σ_{r=1..N_faces} H_{o,r,j} = 1  ∀j ∈ {1,…,N}`, for every obstacle `o`. Exactly **one face (half-space) selected per (obstacle, trajectory segment)** — never per coefficient. |
| Deolasee, Lin, Li, Dolan, *Spatio-temporal Motion Planning for Autonomous Vehicles with Trapezoidal Prism Corridors and Bézier Curves*, arXiv:2209.15150 (2022), Prop. 1 | "If a trajectory has control points in each time interval satisfying `c^k_i ∈ Ω^k_cub = {…, i = 0,1,…,n, k = 0,1,…,m}`, `f(t)` is guaranteed to be safe" — one corridor `Ω^k` per piece `k`, quantified over **all** `i`. |

**The one counterexample, and it is explicitly not a certificate.** Zhou, Wang, Xu, Gao,
*EGO-Planner: An ESDF-free Gradient-based Local Planner for Quadrotors*, IEEE RA-L 6(2), 2021
(arXiv:2008.08835) assigns each colliding control point `Q_i` its **own** `{p_ij, v_ij}` pair
(anchor point on the obstacle surface + repulsion direction), distance `d_ij = (Q_i − p_ij)·v_ij`
(Eq. 1); "each {p,v} pair only belongs to one specific control point". But this is a **soft
penalty** (Eq. 5, a piecewise-cubic cost), not a constraint, and the paper does **not** claim a
continuous-time guarantee from it. The convex hull property is invoked in EGO-Planner only for
the *dynamic feasibility* limits (Eq. 8), where a single box does bound all derivative control
points. The same split appears in Zhou, Gao, Wang, Liu, Shen, *Robust and Efficient Quadrotor
Trajectory Generation for Fast Autonomous Flight* (Fast-Planner), IEEE RA-L, 2019
(arXiv:1907.01531): the B-spline convex hull property carries the velocity/acceleration bounds;
collision is an ESDF penalty evaluated at control points with an iterative re-check
(§ "post-process … we check collisions … increase the collision term").

---

## Q2. What breaks if each control point gets its own plane?

**The single step that converts finitely many point constraints into a continuous-time claim is
lost.** State it precisely:

A Bézier curve is `C(t) = Σ_i b_i(t) P_i` with `b_i(t) ≥ 0` and `Σ_i b_i(t) = 1` for all `t` —
so `C(t) ∈ conv{P_0,…,P_n}` (Kielas-Jensen & Cichella, arXiv:2010.09992, **Property 1**; also
Preiss et al. §IV-B; Deolasee et al. §III-B). A half-space `H = {x : aᵀx ≤ b}` is convex, and
for any convex `H`: `conv{P_i} ⊆ H  ⟺  P_i ∈ H for every i`. That equivalence is the whole
argument, and it requires **the same `(a, b)` for every `i`**.

With per-control-point planes `(a_i, b_i)`:

- **What you can still certify:** `P_i ∈ H_i` for each `i`, and — by the endpoint property
  (arXiv:2010.09992, Property 2: `C(t_0) = P_0`, `C(t_f) = P_n`) — that the segment's two
  endpoints are on their assigned safe sides. That is it.
- **What you can no longer certify:** anything about `C(t)` for `t` strictly inside the segment.
  `conv{P_i}` is not contained in `∪_i H_i` (a union of half-spaces is not convex, so it is not
  a superset of the hull of its members), and there is no reason for it to be contained in
  `∩_i H_i` either. So the constraint set has the same epistemic status as *sampling the curve
  at n+1 parameter values*: a finite collection of point conditions with no interpolation
  guarantee. The convex hull property is simply not invoked.
- **Additionally, for obstacle avoidance specifically:** the free space outside a convex obstacle
  is nonconvex, and the supporting half-space *is* the convex restriction that picks a side. If
  different control points of one segment are given normals pointing to different sides, then
  `conv{P_i}` provably **contains the obstacle** (it contains points on both sides of the
  obstacle's supporting planes). The constraint system is then not merely uncertified — it is
  satisfiable by a curve that goes straight through the obstacle.
- **Coupling pathology:** each per-point plane is still a linear row over the decision variables,
  so with conflicting normals the QP is solving a tug-of-war whose feasible set can be empty or
  can only be reached by the elastic/slack relaxation. Papers that keep the certificate avoid
  this by construction, because there is only one normal per (segment, obstacle) to conflict
  about.

No source states this failure in those words — none of them contemplate the per-control-point
variant for a hard constraint. The argument above is elementary convexity (see e.g. Boyd &
Vandenberghe, *Convex Optimization*, §2.1.4, for `conv S ⊆ C ⟺ S ⊆ C` when `C` is convex), and
the empirical corroboration is EGO-Planner: the one method that does assign per-control-point
planes treats them as soft and re-checks collisions afterwards.

---

## Q3. Opposite sides / how is the side (homotopy class) committed?

**Universal rule: a single segment is never allowed to straddle. The side is committed per
(segment, obstacle), and if the trajectory must change sides, the change happens at a segment
boundary — you add segments, you do not split a segment's control points across sides.**

Four mechanisms in the literature:

1. **Mixed-integer, side as a binary decision.**
   Deits & Tedrake (ICRA 2015): `H ∈ {0,1}^{R×N}`, `H_{r,j} ⟹ P_j(t) ∈ G_r ∀t ∈ [0,1]` (Eq. 1),
   with `Σ_r H_{r,j} = 1` (Eq. 2, restated as Eq. 23) — segment `j` is assigned to exactly one
   convex safe region. Obstacle-face variant Eq. (25): `Σ_{r} H_{o,r,j} = 1` per obstacle — one
   face per (obstacle, segment). Solved to **global** optimality over the discrete choice
   (MISOCP for cubic segments, Eqs. 16–17).
   FASTER (arXiv:2001.04420, MIQP 1): `b_np` binaries with `Σ_p b_np ≥ 1 ∀n`, "the optimizer is
   free to choose the specific interval allocation (i.e., which interval is inside which
   polyhedron)".

2. **Front-end commits, back-end is convex.** A discrete/graph planner fixes the corridor
   sequence; the trajectory QP then never sees a side choice. Preiss et al. (IROS 2017): discrete
   ILP plan → spatial partition into `P^i_k` → independent per-robot QP (14). Gao et al.
   (Teach-Repeat-Replan, T-RO 2020): flight corridor of convex polyhedra generated along the
   teaching path (§IV), then the piecewise-Bézier QP of §V-A. This is the family that keeps the
   subproblem a plain QP, at the price of not searching over classes.
   Preiss et al. §V-C is explicit that when a class conflict arises the fix is **more segments**:
   "Subdivision of discrete plan ensures that this situation cannot occur" (Fig. 4b).

3. **Plane as a continuous decision variable, warm-started by a search.** MADER
   (arXiv:2010.11061, §V-A/§V-D): `n_ij, d_ij` are decision variables alongside the control
   points; "This problem is clearly nonconvex since we are minimizing over the control points and
   the planes `π_ij`." Solved with augmented Lagrangian + MMA (NLopt). The class is committed by
   the **initial guess**: the Octopus Search (Alg. 1) returns "both the control points
   `{q_0,…,q_n}^BS` and the planes `π_ij`". This is the closest published analogue of an
   SCP/linearization loop like ours — and note that even there, the plane per (interval,
   obstacle) is one object, jointly optimized, never per control point.

4. **Convex relaxation over a graph of regions.** Marcucci, Petersen, von Wrangel, Tedrake,
   *Motion planning around obstacles with convex optimization*, Science Robotics 8(84), 2023
   (arXiv:2205.04422): Bézier curves + shortest path in a Graph of Convex Sets; the discrete
   region (hence class) assignment is a graph path, and the convex relaxation is tight enough
   that cheap rounding yields globally optimal trajectories. Current state of the art for the
   side-selection problem specifically.

---

## Q4. Does the certificate survive De Casteljau subdivision into per-segment control points?

**Yes — unconditionally as geometry, subject to three usage conditions.**

*Why it survives.* De Casteljau subdivision (Kielas-Jensen & Cichella, arXiv:2010.09992,
**Property 5**) computes

```
P⁰_{i,n} = P_{i,n}
Pʲ_{i,n} = ((t_f − t_div)/(t_f − t_0)) · Pʲ⁻¹_{i,n} + ((t_div − t_0)/(t_f − t_0)) · Pʲ⁻¹_{i+1,n}
```

and splits the curve into two `n`-th order Bernstein polynomials with coefficients
`{P⁰_{0,n}, P¹_{0,n}, …, Pⁿ_{0,n}}` and `{Pⁿ_{0,n}, Pⁿ⁻¹_{1,n}, …, P⁰_{n,n}}`. Both blending
weights are nonnegative and sum to 1, so **every subdivided control point is a convex combination
of the parent control points**: the subdivision matrix is nonnegative and row-stochastic. Hence

- `conv{sub-control-points} ⊆ conv{parent control points}` (the certificate is never weakened), and
- the sub-curve on the sub-interval **is** exactly the Bézier curve of those sub-control points,
  so Property 1 applies to it verbatim.

*Conditions for the certificate to actually hold:*

1. The map must be the **exact** de Casteljau / blossom subdivision for the parameter split
   actually used (nonnegative, rows summing to 1). Any other linear map — e.g. one whose rows do
   not sum to 1, or a basis change with negative weights that is not an established outer
   enclosure — breaks the convex-combination step.
2. The half-space must be applied to **all** control points of that sub-segment (Q1 again).
3. The sub-intervals must **cover** the whole parameter domain, or the uncovered part of the
   curve is uncertified.

*Density of the subdivision matrix is not a defect.* In general a sub-segment's control points
depend on **all** parent control points — that is the normal, correct situation, and the
certificate transfers regardless, because the dependence is a convex combination. What is not
allowed is a different plane per control point.

*Subdivision is the standard tool for tightening.* Repeated splitting shrinks the hulls toward
the curve, and this is used operationally: BeBOT (arXiv:2010.09992, **Algorithm 2** minimum
distance and **Algorithm 3** collision detection) computes GJK distance between the convex hulls
of the Bernstein coefficients as a *lower* bound (valid by Property 1), then recursively splits
until the upper/lower bound gap is below tolerance. Degree elevation (Property 6, Eq. 12) gives
the same tightening with the bound `max_i |P_{i,m} − C_m(i/m·(t_f−t_0)+t_0)| ≤ k/m`.

*Why you need it:* the un-subdivided hull constraint is **sufficient but not necessary** and is
genuinely conservative. Preiss et al. state this outright (§IV-C): "for a given safe polyhedron
`P^i_k`, there exist degree-D polynomials that lie inside the polyhedron but cannot be expressed
as a Bézier curve with control points that are contained within `P^i_k`. … this problem is most
significant when the desired trajectory is near the faces of the polyhedron rather than the
center." Deits & Tedrake avoid the conservatism entirely by using an exact SOS nonnegativity
condition instead of the hull (Eqs. 9–11 — necessary *and* sufficient for univariate
nonnegativity on `[0,1]`), at the cost of an SDP/SOCP.

*Related, in MADER:* the outer polyhedral representation is `Q^MV_j = Q^BS_j A^BS_pos(j)
(A^MV_pos)^{-1}` (Eq. 2) — a linear map of **that interval's** control points producing an
enclosing simplex, whose vertices then all face the same plane `π_ij`. Same pattern: change of
representation is fine; one plane per (segment, obstacle) is the invariant.

---

## What this means for our code

Our builder emits **one plane per (segment, control point, obstacle)** and spreads each row across
**all** global control points via a dense subdivision matrix. Two separate issues, and only one of
them is a defect:

1. **The dense subdivision matrix is fine.** De Casteljau rows are nonnegative and sum to 1, so
   each segment control point is a convex combination of the global ones and the certificate
   transfers (Q4). "Each row touches all global variables" is expected, not a bug. Do not "fix"
   this by sparsifying — that would break the convex-combination property and destroy the
   certificate.

2. **The per-control-point plane is the defect, and it is fatal to the guarantee.** Nothing in the
   safe-corridor literature does this for a hard constraint. Under it, the convex hull property is
   never invoked, so what the QP enforces is a finite set of point conditions with no
   continuous-time meaning — equivalent in strength to sampling the curve at `n+1` points. Worse,
   because the free space outside a KOZ tube is nonconvex, per-control-point normals let one
   segment's hull straddle the tube, and a hull that straddles **contains** the tube. The paper
   cannot claim a convex-hull safety certificate while this is in the code.

The literature-conformant target, stated in our variables:

- For each (segment `s`, obstacle `o`), build **one** supporting half-space `(a_{s,o}, b_{s,o})`
  in the full lifted `(x, y, t)` space.
- Apply it to **every** local control point of segment `s`: `a_{s,o}ᵀ (M_s P)_i ≤ b_{s,o}` for all
  `i`, where `M_s` is the (dense, row-stochastic) subdivision matrix. Same `a`, same `b`, all `i`.
- Commit the side per (segment, obstacle) — by the passing class from the seed / previous iterate
  (MADER's mechanism: plane comes from the initial guess and is then a decision variable), or by a
  front-end that fixes it (Preiss / TRR mechanism). Never let the sign vary within a segment.
- If the trajectory must change sides of an obstacle, that change belongs at a **segment
  boundary**. If a class conflict makes a segment infeasible, the literature's answer is
  subdivide further (Preiss §V-C), not per-point planes.
- Conservatism is then tuned by segment count, monotonically and safely: more segments → tighter
  hulls → strictly larger feasible set (nested hulls, Q4). This is the honest knob, and it is also
  the invariant to test with: *increasing `n_seg` cannot shrink the feasible set*.

One extra note relevant to our core claim, in our favour: two AV papers observe that when time is
the *parameter* of the Bézier curve and the corridor is time-dependent, the naive convex-hull
argument needs an extra condition — Zhang, Yadmellat, Gao, *A Sufficient Condition for Convex Hull
Property in General Convex Spatio-Temporal Corridors* (arXiv:2110.00065, 2021), which states "the
convex hull property does not necessarily hold for time-dependent corridors, and depends on the
shape of corridors" because "the spatial axis and the temporal axis are not equivalent"; and
Deolasee et al. (arXiv:2209.15150), whose Prop. 1 holds only for *cuboidal* corridors and whose
Prop. 2 is the relaxed trapezoidal-prism condition. Our formulation does **not** have this problem,
because time is an explicit Bézier *coordinate*: the curve is a genuine Bézier curve in `R³` and
Property 1 applies verbatim to `(x, y, t)`. That is a defensible, citable point of distinction for
the paper — provided the code actually applies one plane per (segment, obstacle) to all control
points in the lifted space.

---

## Disagreements / uncertainty

- **Gao, Wu, Lin, Shen, "Online Safe Trajectory Generation for Quadrotors Using Fast Marching
  Method and Bernstein Basis Polynomial", ICRA 2018, DOI 10.1109/ICRA.2018.8462878 — COULD NOT
  READ.** Metadata verified via the Semantic Scholar API (authors Fei Gao, William Wu, Yi Lin,
  Shaojie Shen; venue ICRA; year 2018; `openAccessPdf` status `CLOSED`). No open PDF was found. I
  am therefore **not** quoting any equation number from it. The same group's journal paper
  (Teach-Repeat-Replan, T-RO 2020, arXiv:1907.00520 §V-A-3 Eq. 8) states the formulation
  explicitly and is read in full; cite that when an equation is needed. Code is at
  `github.com/HKUST-Aerial-Robotics/Btraj` (not inspected).
- **Deits & Tedrake do not use the Bézier convex hull.** If anyone asserts "Deits & Tedrake
  constrain Bézier control points", that is wrong. They use an exact sums-of-squares nonnegativity
  certificate on `q(t) = b_{r,ℓ} − Σ_k (a_{r,ℓ}ᵀ C_{j,k}) Φ_k(t) ≥ 0 ∀t ∈ [0,1]` (Eqs. 9–11), which
  is *necessary and sufficient*, unlike the hull condition. They are still the canonical citation
  for "one half-space per (segment, obstacle-face), selected by a binary" (Eq. 25).
- **MADER is B-spline + MINVO, not Bernstein.** Its per-interval outer polyhedron is the MINVO
  simplex, which is tighter than the Bernstein hull ("2.36 … times smaller" volume than Bernstein,
  §IV-A). The one-plane-per-(interval, obstacle) structure is basis-independent, so the citation
  holds for our purpose, but do not describe MADER as a Bézier method.
- **Zhang et al. (arXiv:2110.00065) — venue not verified.** Read via ar5iv only; I could not
  confirm peer-reviewed publication. Its setting (Bézier `s(t)` with `t` as parameter, corridor
  time-dependent, control points at equal time spacing) is **not** identical to ours. Treat as a
  caution about a formulation we do not use, not as authority on ours. Its Theorem II.1 / Lemma
  III.4 numbering came from the ar5iv rendering and was not cross-checked against the PDF.
- **Deolasee et al. (arXiv:2209.15150)** — preprint, venue not verified. Prop. 1 text quoted above
  is from the extracted PDF; Prop. 2 (the trapezoidal-prism condition) was located but its full
  statement is not quoted here.
- **No readable dedicated survey of Bernstein-polynomial trajectory optimization was found.** The
  closest is the properties review in Kielas-Jensen & Cichella, arXiv:2010.09992 §II (Properties
  1–7), which is read in full and used above. The related MDPI paper (*Bernstein Polynomial-Based
  Method for Solving Optimal Trajectory Generation Problems*, Sensors 22(5):1869, 2022) returned
  HTTP 403 and was **not** read; it is not cited for any specific claim.
- **Absence of the per-control-point-plane variant is not proof of impossibility** — I found no
  paper that assigns one plane per (segment, control point, obstacle) and claims a guarantee, but
  "no paper does it" is weaker than a proof. The proof is the convexity argument in Q2: a union of
  half-spaces is not convex, so no containment claim about `conv{P_i}` follows from
  `P_i ∈ H_i`. That argument stands on its own and does not need a citation.
- **FASTER**: I cite arXiv:2001.04420 (the "FASTER: Fast and Safe Trajectory Planner for
  Navigation in Unknown Environments" version). There are several closely-related versions/titles
  by the same authors (FaSTraP, IROS 2019 version, IJRR version). Equation (1) as quoted is from
  arXiv:2001.04420. Pin the exact version when citing in the paper.
