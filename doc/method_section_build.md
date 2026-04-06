# Method Section Draft

## Brief Verification Summary

### What was confirmed from the implementation

- The optimization variable is the stacked control-point vector `x = [p_0^T p_1^T ... p_N^T]^T`, with `P in R^{(N+1) x 3}` the control-point matrix used for convenience in the implementation.
- Transfer time is fixed in the present formulation. Physical time and normalized parameter are related by `t = T tau`, with `T = TRANSFER_TIME_S` fixed rather than optimized.
- The derivative operator is the exact Bézier difference matrix `D_N`, and the degree-elevation operator is the exact Bézier elevation matrix `E_M` implemented in `orbital_docking/bezier.py`.
- Velocity and acceleration control points are represented in the original degree-`N` basis through `E_{N-1} D_N P` and `E_{N-1} D_N E_{N-1} D_N P`.
- The Bernstein Gram matrix `G_N` is implemented in closed form and is used to build the exact quadratic control-acceleration-energy matrix `\tilde G_N = L_{2,N}^T G_N L_{2,N}`.
- Equal-parameter subdivision is implemented through segment matrices returned by `segment_matrices_equal_params(N, n_seg)`, with each sub-arc control polygon given by `P^{(s)} = S^{(s)} P`.
- The KOZ supporting half-space is built from the centroid of each subdivided control polygon, not from a curve midpoint or arbitrary representative point.
- The primary `dv` mode is an IRLS-weighted quadratic majorization of an L1-style proxy for control effort `||a_geom - g||`, with gravity modeled as two-body plus J2 and linearized affinely at representative sub-arc positions.
- Each SCP outer iteration solves a convex quadratic subproblem with linear constraints and endpoint bounds. Optional proximal regularization is part of the subproblem objective.
- The implementation can also apply an optional post-solve step clipping radius. This is a safeguard on the accepted SCP step, not an explicit trust-region constraint inside the QP.
- The stopping rule is based on the control-point update norm: terminate when `||P^{(k+1)} - P^{(k)}|| < tol`, or when the maximum outer-iteration count is reached.

### What was corrected relative to the previous build draft

- The KOZ representative point is now fixed precisely as the centroid of the subdivided control polygon.
- The primary manuscript objective is now the implementation's `dv` mode rather than the earlier quadratic-energy placeholder.
- The Gram-matrix term is no longer presented as the primary paper objective. Under the locked decisions, it is retained as part of the operator framework and as the basis of the legacy L2 energy mode and optional geometric regularization.
- The SCP subsection now distinguishes clearly between the convex QP itself and the optional post-solve step clipping.
- The objective-linearization segment count is distinct from the KOZ subdivision count in the implementation and should be notated separately in the paper.

### Contradictions found

- No direct contradiction was found between the locked decisions and the implementation.
- One scientific caution remains: because the locked primary objective is the IRLS L1-style `dv` mode, the paper should not write as though Gram-matrix reuse is the defining mechanism of the main objective. The narrow safe fix is to present the Gram matrix as part of the reusable derivative/objective machinery and as the basis of the alternative L2 mode or optional regularization.

## Updated Manuscript-Ready Method Draft

### 3. Problem Setup and Notation

#### 3.1 Trajectory parameterization

We represent the trajectory by a degree-`N` Bézier curve over the normalized parameter domain `tau in [0,1]`:

$$
\mathbf{r}(\tau) = \sum_{i=0}^{N} B_i^N(\tau)\,\mathbf{p}_i,
$$

where `B_i^N` denotes the Bernstein basis polynomial of degree `N`, and `\mathbf{p}_i in R^3` is the `i`th control point. For compact matrix expressions, we collect the control points row-wise as

$$
P =
\begin{bmatrix}
\mathbf{p}_0^\top \\
\mathbf{p}_1^\top \\
\vdots \\
\mathbf{p}_N^\top
\end{bmatrix}
\in \mathbb{R}^{(N+1)\times 3},
$$

and use the stacked control-point vector

$$
\mathbf{x}
=
\begin{bmatrix}
\mathbf{p}_0^\top &
\mathbf{p}_1^\top &
\cdots &
\mathbf{p}_N^\top
\end{bmatrix}^\top
\in \mathbb{R}^{3(N+1)}.
$$

as the primary optimization variable. This makes the method a control-point-space formulation: the optimization variables, derivative operators, subdivision operators, and KOZ constraints are all expressed directly in terms of `\mathbf{x}` or equivalently `P` (see planned `T1`).

The present formulation uses a fixed transfer time `T`. Physical time and normalized parameter are related by

$$
t = T\tau,
$$

so the method does not optimize time allocation or final time. This point should be stated explicitly in the paper, because the physical interpretation of velocity and acceleration depends on the fixed time scaling.

#### 3.2 Derivative mappings, boundary conditions, and Gram-matrix structure

The derivative structure is implemented through the standard Bézier difference matrix

$$
D_N
=
N
\begin{bmatrix}
-1 & 1 & 0 & \cdots & 0 \\
0 & -1 & 1 & \ddots & \vdots \\
\vdots & \ddots & \ddots & \ddots & 0 \\
0 & \cdots & 0 & -1 & 1
\end{bmatrix}
\in \mathbb{R}^{N\times(N+1)},
$$

and the degree-elevation matrix `E_M in R^{(M+2) x (M+1)}`, whose nonzero entries are

$$
[E_M]_{0,0} = 1,\qquad [E_M]_{M+1,M} = 1,
$$

and, for `i = 1, ..., M`,

$$
[E_M]_{i,i-1} = \frac{i}{M+1}, \qquad
[E_M]_{i,i} = \frac{M+1-i}{M+1}.
$$

For a degree-`N` trajectory, the implementation uses `E_{N-1}` to lift derivative control points back into the degree-`N` basis. Defining

$$
L_{1,N} = E_{N-1} D_N,
\qquad
L_{2,N} = E_{N-1} D_N E_{N-1} D_N,
$$

the degree-preserving velocity and acceleration control-point maps are

$$
P^{(1)} = L_{1,N} P,
\qquad
P^{(2)} = L_{2,N} P.
$$

Physical derivatives then follow from the fixed time scaling:

$$
\dot{\mathbf{r}}(t) = \frac{1}{T}\frac{d\mathbf{r}}{d\tau},
\qquad
\ddot{\mathbf{r}}(t) = \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2}.
$$

The endpoint position constraints are imposed by fixing the first and last control points. Optional endpoint derivative conditions are linear equalities in the control points:

$$
\dot{\mathbf{r}}(0) = \frac{N}{T}(\mathbf{p}_1 - \mathbf{p}_0),
\qquad
\dot{\mathbf{r}}(T) = \frac{N}{T}(\mathbf{p}_N - \mathbf{p}_{N-1}),
$$

and, for `N >= 2`,

$$
\ddot{\mathbf{r}}(0) = \frac{N(N-1)}{T^2}(\mathbf{p}_2 - 2\mathbf{p}_1 + \mathbf{p}_0),
$$

$$
\ddot{\mathbf{r}}(T) = \frac{N(N-1)}{T^2}(\mathbf{p}_N - 2\mathbf{p}_{N-1} + \mathbf{p}_{N-2}).
$$

The Bernstein Gram matrix is implemented in closed form as

$$
[G_N]_{ij}
=
\frac{\binom{N}{i}\binom{N}{j}}{\binom{2N}{i+j}(2N+1)},
\qquad i,j = 0,\ldots,N.
$$

Together with `L_{2,N}`, it yields the exact quadratic matrix

$$
\tilde G_N = L_{2,N}^\top G_N L_{2,N},
$$

which satisfies

$$
\int_0^1 \left\|\frac{d^2 \mathbf{r}}{d\tau^2}\right\|_2^2 d\tau
=
\mathrm{tr}(P^\top \tilde G_N P)
=
\mathbf{x}^\top (\tilde G_N \otimes I_3)\mathbf{x}.
$$

This Gram-matrix construction is part of the reusable operator framework used by the implementation. Under the locked objective choice below, it is not the primary paper objective, but it remains mathematically useful because it defines the legacy L2 energy mode and the optional geometric regularization term.

### 4. Method

#### 4.1 Conservative spherical-KOZ handling by subdivision and supporting half-spaces

Let the keep-out zone be the sphere

$$
\mathcal{K}
=
\left\{
\mathbf{r} \in \mathbb{R}^3 :
\|\mathbf{r} - \mathbf{c}_{\mathrm{KOZ}}\|_2 \le r_e
\right\}.
$$

To impose conservative continuous avoidance, the curve is subdivided into `n_seg` equal-parameter sub-arcs using De Casteljau segment matrices `S^{(s)} in R^{(N+1) x (N+1)}`, `s = 1, ..., n_seg`. The corresponding sub-arc control polygons are

$$
P^{(s)} = S^{(s)} P.
$$

Writing the control points of the `s`th sub-arc as `\mathbf{q}^{(s)}_0, ..., \mathbf{q}^{(s)}_N`, the implementation chooses the representative point

$$
\mathbf{c}^{(s)}
=
\frac{1}{N+1}
\sum_{k=0}^{N}
\mathbf{q}^{(s)}_k,
$$

that is, the centroid of the subdivided control polygon. The supporting-half-space normal is then

$$
\mathbf{n}^{(s)}
=
\frac{\mathbf{c}^{(s)} - \mathbf{c}_{\mathrm{KOZ}}}
{\|\mathbf{c}^{(s)} - \mathbf{c}_{\mathrm{KOZ}}\|_2},
$$

whenever the denominator is nonzero. The associated supporting half-space is

$$
\mathcal{H}^{(s)}
=
\left\{
\mathbf{r} :
(\mathbf{n}^{(s)})^\top \mathbf{r}
\ge
(\mathbf{n}^{(s)})^\top \mathbf{c}_{\mathrm{KOZ}} + r_e
\right\}.
$$

The method then constrains every control point of every sub-arc to lie in its corresponding supporting half-space:

$$
(\mathbf{n}^{(s)})^\top \mathbf{q}^{(s)}_k
\ge
(\mathbf{n}^{(s)})^\top \mathbf{c}_{\mathrm{KOZ}} + r_e,
\qquad
k = 0,\ldots,N.
$$

Since `\mathbf{q}^{(s)}_k = \sum_{j=0}^{N} S^{(s)}_{kj}\mathbf{p}_j`, each of these inequalities is linear in the stacked decision vector `\mathbf{x}`. The half-spaces are rebuilt at every SCP iteration from the current iterate, so the KOZ constraints are conservative and local rather than globally exact (see planned `F1`).

Proposition. Fix a sub-arc `s` and a supporting half-space `\mathcal{H}^{(s)}` constructed as above. If all control points of `P^{(s)}` lie in `\mathcal{H}^{(s)}`, then the entire Bézier sub-arc lies in `\mathcal{H}^{(s)}` and therefore outside `\mathcal{K}`.

The proof is immediate from two facts: a Bézier curve lies in the convex hull of its control points, and `\mathcal{H}^{(s)}` is a supporting half-space of the sphere. The statement is narrow on purpose. It does not establish a general obstacle-avoidance theorem, and it should not be written as one.

#### 4.2 Objective construction and interpretation boundary

The current paper uses the implementation's `dv` mode as its primary optimization objective. This objective should be described as an L1-style control-effort proxy, not as a true delta-v-optimal objective. The relevant continuous quantity is the control-acceleration magnitude

$$
\mathbf{u}(t) = \ddot{\mathbf{r}}(t) - \mathbf{g}(\mathbf{r}(t)),
$$

where `\mathbf{g}` denotes the orbital gravity model. In the present implementation, `\mathbf{g}` includes a two-body term plus a J2 perturbation term. Because the paper is not primarily about gravity modeling, it is sufficient in the method section to state only how this field enters the optimization: through an affine linearization at representative sub-arc positions.

The objective linearization uses `n_lin` equal-parameter sub-arcs, which are distinct from the KOZ subdivision count `n_seg` in the implementation. Let `\hat S^{(i)}` denote the corresponding segment matrices, and let

$$
\mathbf{w}^{(i)} = \frac{1}{N+1}\mathbf{1}^\top \hat S^{(i)}
$$

be the row vector that averages the control points of the `i`th sub-arc. The representative position and geometric acceleration are then linear maps of `\mathbf{x}`:

$$
\mathbf{r}_i(\mathbf{x}) = R_i \mathbf{x},
\qquad
R_i = \mathbf{w}^{(i)} \otimes I_3,
$$

$$
\mathbf{a}_i(\mathbf{x}) = A_i \mathbf{x},
\qquad
A_i = \frac{1}{T^2}\bigl(\mathbf{w}^{(i)} L_{2,N}\bigr)\otimes I_3.
$$

At SCP iteration `k`, the gravity field is linearized affinely about the reference position `\mathbf{r}_i(\mathbf{x}^{(k)})`:

$$
\mathbf{g}_i^{(k)}(\mathbf{x})
\approx
B_i^{(k)} \mathbf{x} + \mathbf{c}_i^{(k)},
$$

where `B_i^{(k)} = J_i^{(k)} R_i`, `J_i^{(k)}` is the Jacobian of the gravity model evaluated numerically at `\mathbf{r}_i(\mathbf{x}^{(k)})`, and

$$
\mathbf{c}_i^{(k)}
=
\mathbf{g}\bigl(\mathbf{r}_i(\mathbf{x}^{(k)})\bigr)
-
J_i^{(k)} \mathbf{r}_i(\mathbf{x}^{(k)}).
$$

Define the linearized control-effort residual by

$$
\boldsymbol{\rho}_i^{(k)}(\mathbf{x})
=
A_i \mathbf{x}
-
\left(
B_i^{(k)} \mathbf{x} + \mathbf{c}_i^{(k)}
\right).
$$

The implementation then builds an IRLS majorization of an L1-style objective:

$$
\omega_i^{(k)}
=
\frac{1/n_{\mathrm{lin}}}
{\sqrt{
\left\|
\boldsymbol{\rho}_i^{(k)}(\mathbf{x}^{(k)})
\right\|_2^2

+ \varepsilon}},
$$

$$
J_{\mathrm{dv}}^{(k)}(\mathbf{x})
=
\sum_{i=1}^{n_{\mathrm{lin}}}
\omega_i^{(k)}
\left\|
\boldsymbol{\rho}_i^{(k)}(\mathbf{x})
\right\|_2^2.
$$

This is the manuscript-safe description of the primary objective: it is an iterative quadratic majorization of an L1-style proxy for control effort. It is not exact mission delta-v optimization, and the paper should not imply otherwise.

For completeness, the implementation also retains an L2-style energy mode based on the same residual structure, together with the Gram-matrix-based quadratic form `\mathbf{x}^\top (\tilde G_N \otimes I_3)\mathbf{x}/T^4`. Under the locked paper decisions, that alternative mode should be mentioned only briefly as implementation context or optional regularization, not as a parallel paper claim.

#### 4.3 SCP subproblem and update loop

At SCP iteration `k`, the method fixes the KOZ supporting half-spaces and the affine gravity linearizations built from the current control-point vector `\mathbf{x}^{(k)}`. The resulting subproblem is a convex quadratic program of the form

$$
\begin{aligned}
\min_{\mathbf{x}} \quad &
\frac{1}{2}\mathbf{x}^\top H^{(k)} \mathbf{x}
+
\bigl(\mathbf{f}^{(k)}\bigr)^\top \mathbf{x} \\
\text{s.t.} \quad &
A_{\mathrm{KOZ}}^{(k)} \mathbf{x} \ge \mathbf{b}_{\mathrm{KOZ}}^{(k)}, \\
&
A_{\mathrm{bc}} \mathbf{x} = \mathbf{b}_{\mathrm{bc}}, \\
&
\boldsymbol{\ell} \le \mathbf{x} \le \mathbf{u}.
\end{aligned}
$$

Here `A_{\mathrm{KOZ}}^{(k)}` and `\mathbf{b}_{\mathrm{KOZ}}^{(k)}` encode the centroid-based supporting-half-space constraints, `A_{\mathrm{bc}} \mathbf{x} = \mathbf{b}_{\mathrm{bc}}` collects the optional endpoint velocity and acceleration conditions, and the bounds `\boldsymbol{\ell}` and `\mathbf{u}` fix the initial and final positions. Under the primary `dv` mode, the matrices `H^{(k)}` and `\mathbf{f}^{(k)}` come from the weighted least-squares objective above. When enabled, a proximal regularization term

$$
\frac{\lambda}{2}
\left\|
\mathbf{x} - \mathbf{x}^{(k)}
\right\|_2^2
$$

is added directly to the subproblem objective.

The current implementation initializes the SCP loop with a straight-line control polygon between the endpoint positions. It then repeats the following steps: subdivide the current control polygon, rebuild the KOZ half-spaces, rebuild the affine gravity/J2 linearization and IRLS weights, solve the convex QP, and update the reference control points. The outer loop terminates when

$$
\left\|
P^{(k+1)} - P^{(k)}
\right\|_F < \mathrm{tol},
$$

or when the prescribed maximum number of SCP iterations is reached.

If enabled, the implementation also clips the accepted SCP step to a prescribed radius in control-point-vector norm. This step clipping is a practical safeguard, not part of the QP itself. The method section should therefore avoid claiming that the implemented subproblem includes an explicit trust-region constraint unless the code is changed accordingly. This distinction matters because the paper's defensible claim is that the method solves a sequence of convex QPs, not that it solves a trust-region-constrained convex reformulation of the original nonconvex problem (see planned `F2`).

#### 4.4 Assumptions and formulation boundary

The formulation should close with its assumption boundary rather than relying on later sections to repair overstatement. First, the continuous-exclusion argument is tied to a spherical KOZ and to the supporting-half-space construction on subdivided sub-arcs. The resulting statement is conservative and assumption-dependent. It should not be presented as a general result for arbitrary obstacle geometry.

Second, transfer time is fixed. The present formulation does not optimize timing, free final time, or trajectory duration. All derivative quantities and all objective terms are interpreted under the fixed scaling `t = T tau`.

Third, the paper's primary objective is a surrogate. The `dv` mode is an IRLS-based proxy for control effort built from geometric acceleration minus affine-linearized gravity. It is suitable for trajectory initialization, but it does not justify claims of true fuel optimality, physical superiority over alternative surrogates, or planner-class superiority.

Fourth, the method section should describe the framework as designed to produce warm-start trajectories for downstream solvers. Demonstrated downstream usefulness is a results-level claim and should not be asserted by the method section alone.

Finally, the SCP wording should remain precise. Each outer iteration solves a convex QP after fixing local linearizations, but the original trajectory problem is not thereby converted into a single exact convex program. No claim of global optimality is made.

## Final Risk List

- The phrase "delta-v proxy" remains reviewer-vulnerable if it is not immediately qualified as an IRLS L1-style surrogate on control acceleration relative to linearized gravity. Do not call it a true delta-v objective.
- The Gram matrix is verified, but under the locked objective choice it is not the defining mechanism of the main objective. Do not overstate its role in the primary `dv` formulation.
- The continuous-safety statement is safe only when it is written with its assumptions: spherical KOZ, fixed supporting half-spaces, and control-point satisfaction on each subdivided sub-arc.
- The gravity/J2 term is used through affine linearization at representative sub-arc positions. Avoid language suggesting exact dynamics enforcement along the full continuous curve.
- The optional trust radius in the implementation is post-solve step clipping, not a trust-region constraint inside the convex QP. This wording must remain precise.
- The code skips the KOZ normal construction if a sub-arc centroid coincides with the KOZ center. The manuscript should avoid implying that the supporting normal is defined in that degenerate case.
- Warm-start usefulness should remain an intended-use statement in the method section unless the results section later supplies the downstream comparison evidence.
