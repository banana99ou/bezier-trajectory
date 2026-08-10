# The Problem
solve non-convex problem of generating bezier curve that connects orbit A to B that satisfies following constraints while minimizing following objective function.
## Objective Function
J = Σ‖a_geom(P) − a_grav(P) − a_J2(P)‖²
where a_geom is geometric acceleration of the resulting bezier curve. given by EDED P EDED^T =  Pᵀ G̃ P. a_grav is gravitational acceleration at each curve point. sample based. [[need to think about sample counts makes sense tho.]] a_J2 is J2 done similary
## constraints
for all tau dist to CG of KOZ > rad of KOZ. which gets converted to supporting halfspace constraint per de kasteljau segments.
V_0=V_0, V_f=V_f.
## SCP
QP needs linear problem. and the KOZ constraint and the Grav/J2 term is not linear. (1/r^2). 
so we linearize the grav term around P at each Iteration of P with tailor serise expention. and KOZ with de kasteljau.
one is around specific P. and one is around specific P and curve segment.

~~subproblem? does it mean that the scp is subproblem of QP in this case?~~ -> SCPloop(QP) SCP makes easyer version of QP.\
~~add virtual control???/slack??? does this mean adding additional control parameter that let's the subproblem feasiable? but how is this additional parameter determined?~~ -> virtual control/slack is another variable that gets optimized to 0. which is neccesary bc some lienarized points are genuinely infeasiable. (A_koz x ≥ b_koz becomes A_koz x + s ≥ b_koz with s ≥ 0.)\
~~what does x − xₖ mean? what is X if Xk is control point matrix P at iteration k? what does it mean to subtract Pk from P(???)???~~ -> new - prev. displacement. trust region is about limiting how much we can advance in linearized space.\
~~and how is rk set? is this arbiturary?~~ -> is arbiturary but dosen't matter bc it auto adapts with ρ ≈ ΔM/ΔL.\
merit fcn is just true non-approx obj fcn + constraint violation with weights\
M(xₖ) doesn't neccesarily have 0 constraints violation section.\
~~but where does ΔL comes from? what is predicted reduction? what is convex model?~~ -> L(x) Linerized Obj fcn and constraints. convex(ified) model. The inner QP loop.\
~~why grow ρ if ρ ≈ 1? what does ρ ≈ 1 mean?~~ -> convexified model is close to real model. we can trust it more.\
~~step here means outer QP loop yes? and inner loop is SCP loop? so we're re-doing SCP every step? which is re-linearizing every step?~~ -> do scp once. then solve linearized problem with qp once again. acept the qp result only when p<1(linearized problem stayed within expected boundary).

so the regular QP is opt parameter+obj fcn+const. and SCP is just one of the technique to linearize something non-linear? and the trust region is also just another technique being slapped on? or all of these is one mathmatical thing? -> SCP is one mathmatical thing. including trust region.

LP ⊂ QP ⊂ SOCP ⊂ SDP ⊂ general convex, with NLP (nonconvex) above all. our problme has cost function in sum of squares -> Quadratic.\
If you kept the KOZ in norm form ‖r‖ ≥ r_KOZ, you'd need SOCP — that's what the "IRLS hack" in the handoff was dodging -> what does this mean? what is the diff btw QP and SOCP? and what does SOCP stands for?\
SCP is the method. and is the method to linearize nonconvex problem in to linear approximation and solve it as QP.\
and so linearization at given Xk (SCP) -> solve linearized (sub)problem as QP (inner loop)?\
what is the difference btw SCP and SCvx? -> (more strict. our combination of SCP+trust_region+slack_variable flavor) SCvx ⊂ SCP (more broader).

# how can we assure the fix IS a fix
1. I want independent cross-check. can you setup ScPy trust-constr or IPOPT/CasADi as a direct NLP and compare the optimal cost and trajectory?
2. Isolation / ablation (closes the review's gap). Add a toggle to disable the freeze, then run a small table:
  - trust off + proximal on → should crawl to the cap (reproduces the bug),
  - trust on + proximal off + re-linearize every step (canonical) → should converge fast,
  - trust on + freeze on (current) → converges fast.
This proves which change is the fix and whether the freeze matters — and that all converging configs reach the same optimum within tolerance. (Right now there's no toggle, so this needs a one-line code change first.)
3. Optimality + feasibility check at the solution. At the returned x*, verify the KKT conditions of the original nonconvex problem: KOZ satisfied on a dense τ-grid (min‖r‖ ≥ r_KOZ), boundary conditions met, and the objective gradient projected onto feasible directions ≈ 0. A KKT point = a legitimate local optimum, not just "the loop stopped."
4. Diagnostics + regression. Log per-iteration ρ, trust radius, and merit: a correct run shows ρ near 1, radius growing then settling, merit monotonically decreasing, slack → 0. Then sweep N × n_seg × scenarios and confirm all converge (iter < cap), all feasible, costs sensible vs n_seg — with n_seg=4 flagged as the known holdout.
---

spacetime-bezier ideas for demo/application.
LOS minimization. (PESA radar beam avoidance.)
Missile evade manuever
re-entry refer to prof doc for param.
GCS