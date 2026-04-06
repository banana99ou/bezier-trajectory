# Method Section Draft

## 3. Problem Setup and Notation

### 3.1 Trajectory parameterization and decision variables

We represent the trajectory by a degree-$N$ Bezier curve over the normalized parameter domain $\tau \in [0,1]$:

$$
\mathbf{r}(\tau) = \sum_{i=0}^{N} B_i^N(\tau)\,\mathbf{p}_i,
$$

where $B_i^N$ denotes the Bernstein basis polynomial of degree $N$, and $\mathbf{p}_i \in \mathbb{R}^3$ is the $i$th control point. The control points are collected row-wise as

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

and the stacked control-point vector is

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

The optimization is therefore carried out entirely in control-point space: the decision variable is $\mathbf{x}$, and all derivative maps, subdivision operators, and KOZ constraints are written directly in terms of $\mathbf{x}$ or, equivalently, $P$.

The present formulation uses a fixed transfer time $T$. Physical time and normalized parameter are related by

$$
t = T\tau.
$$

The method does not optimize free final time or timing allocation. This fixed time scaling is important because all physical velocity and acceleration quantities are obtained from derivatives with respect to $\tau$ by factors of $1/T$ and $1/T^2$.

Although the control-point operators themselves are geometric and dimension-agnostic, the present objective and demonstration specialize to $\mathbb{R}^3$ because the implementation uses a three-dimensional orbital gravity model.

### 3.2 Derivative operators, endpoint conditions, and Gram-matrix structure

The derivative structure is encoded by the standard Bezier difference matrix

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
\in \mathbb{R}^{N\times(N+1)}.
$$

To express derivative control points back in the original degree-$N$ basis, the implementation uses the degree-elevation matrix $E_M \in \mathbb{R}^{(M+2)\times(M+1)}$, defined by

$$
[E_M]_{0,0} = 1, \qquad [E_M]_{M+1,M} = 1,
$$

and, for $i=1,\ldots,M$,

$$
[E_M]_{i,i-1} = \frac{i}{M+1}, \qquad
[E_M]_{i,i} = \frac{M+1-i}{M+1}.
$$

For a degree-$N$ curve, let

$$
L_{1,N} = E_{N-1}D_N,
\qquad
L_{2,N} = E_{N-1}D_N E_{N-1}D_N.
$$

Then the degree-preserving velocity and acceleration control-point maps are

$$
P^{(1)} = L_{1,N}P,
\qquad
P^{(2)} = L_{2,N}P.
$$

Physical derivatives follow from the fixed-time scaling:

$$
\dot{\mathbf{r}}(t) = \frac{1}{T}\frac{d\mathbf{r}}{d\tau},
\qquad
\ddot{\mathbf{r}}(t) = \frac{1}{T^2}\frac{d^2\mathbf{r}}{d\tau^2}.
$$

Endpoint position constraints are imposed by fixing the first and last control points. Optional endpoint velocity constraints are linear equalities:

$$
\dot{\mathbf{r}}(0) = \frac{N}{T}(\mathbf{p}_1-\mathbf{p}_0),
\qquad
\dot{\mathbf{r}}(T) = \frac{N}{T}(\mathbf{p}_N-\mathbf{p}_{N-1}),
$$

and, for $N \ge 2$, optional endpoint acceleration constraints are

$$
\ddot{\mathbf{r}}(0) = \frac{N(N-1)}{T^2}(\mathbf{p}_2-2\mathbf{p}_1+\mathbf{p}_0),
$$

$$
\ddot{\mathbf{r}}(T) = \frac{N(N-1)}{T^2}(\mathbf{p}_N-2\mathbf{p}_{N-1}+\mathbf{p}_{N-2}).
$$

The Bernstein Gram matrix is implemented in closed form as

$$
[G_N]_{ij}
=
\frac{\binom{N}{i}\binom{N}{j}}{\binom{2N}{i+j}(2N+1)},
\qquad i,j=0,\ldots,N.
$$

Together with $L_{2,N}$, it gives the exact quadratic form

$$
\tilde G_N = L_{2,N}^\top G_N L_{2,N},
$$

which satisfies

$$
\int_0^1 \left\|\frac{d^2\mathbf{r}}{d\tau^2}\right\|_2^2 d\tau
=
\mathrm{tr}(P^\top \tilde G_N P)
=
\mathbf{x}^\top (\tilde G_N \otimes I_3)\mathbf{x}.
$$

In the present paper this Gram-matrix construction is part of the reusable operator framework rather than the primary paper-level objective. It remains relevant because it defines the legacy L2 control-acceleration energy mode in the implementation and, under the current `dv` mode, can be used as an optional geometric regularizer.

### 3.3 Compact map to the implementation

The manuscript notation corresponds directly to the implementation objects:

- $D_N$ corresponds to `get_D_matrix(N)`.
- $E_M$ corresponds to `get_E_matrix(M)`.
- $S^{(s)}$ in the next section corresponds to the equal-parameter segment matrices returned by `segment_matrices_equal_params(N, n_seg)`.
- $\tilde G_N$ corresponds to the quadratic form assembled in `BezierCurve.G_tilde`.

This mapping is intentionally small: the paper uses compact notation, but every operator used below has a direct implementation counterpart.

## 4. Method

### 4.1 Conservative spherical-KOZ handling by subdivision and supporting half-spaces

Let the keep-out zone be the sphere

$$
\mathcal{K}
=
\left\{
\mathbf{r}\in\mathbb{R}^3 :
\|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le r_e
\right\}.
$$

For generality we write the center as $\mathbf{c}_{\mathrm{KOZ}}$, although the current optimization path uses the origin-centered case $\mathbf{c}_{\mathrm{KOZ}}=\mathbf{0}$.

To impose conservative continuous avoidance, the curve is subdivided into $n_{\mathrm{seg}}$ equal-parameter sub-arcs using De Casteljau segment matrices $S^{(s)} \in \mathbb{R}^{(N+1)\times(N+1)}$, $s=1,\ldots,n_{\mathrm{seg}}$. The control polygon of the $s$th sub-arc is

$$
P^{(s)} = S^{(s)}P.
$$

Writing the control points of this sub-arc as $\mathbf{q}^{(s)}_0,\ldots,\mathbf{q}^{(s)}_N$, the implementation chooses the representative point

$$
\mathbf{c}^{(s)}
=
\frac{1}{N+1}\sum_{k=0}^{N}\mathbf{q}^{(s)}_k,
$$

that is, the centroid of the subdivided control polygon. The outward normal is then

$$
\mathbf{n}^{(s)}
=
\frac{\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}}
{\|\mathbf{c}^{(s)}-\mathbf{c}_{\mathrm{KOZ}}\|_2},
$$

whenever the denominator is nonzero. The corresponding supporting half-space is

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

The method enforces this half-space on every control point of every sub-arc:

$$
(\mathbf{n}^{(s)})^\top \mathbf{q}^{(s)}_k
\ge
(\mathbf{n}^{(s)})^\top \mathbf{c}_{\mathrm{KOZ}} + r_e,
\qquad
k=0,\ldots,N.
$$

Because each $\mathbf{q}^{(s)}_k$ is a linear combination of the original control points, these inequalities are linear in $\mathbf{x}$. The half-spaces are rebuilt at each outer iteration from the current iterate, so the KOZ treatment is conservative and local rather than globally exact.

The key guarantee is narrow but useful. If all control points of a given sub-arc lie in its supporting half-space $\mathcal{H}^{(s)}$, then the entire Bezier sub-arc lies in $\mathcal{H}^{(s)}$ and therefore outside the spherical KOZ. This follows from the convex-hull property of Bezier curves together with the fact that $\mathcal{H}^{(s)}$ is a supporting half-space of the sphere. The statement should not be generalized beyond the spherical-KOZ setting or beyond the stated subdivision-and-half-space construction.

### 4.2 Control-effort surrogate objective

The paper uses the implementation's `dv` mode as its sole paper-level objective. This objective should be described as an L1-style control-effort surrogate, not as a true delta-v-optimal objective.

Let

$$
\mathbf{u}(t) = \ddot{\mathbf{r}}(t) - \mathbf{g}(\mathbf{r}(t)),
$$

where $\mathbf{g}$ is the orbital gravity model. In the implementation, $\mathbf{g}$ consists of a two-body term plus a J2 perturbation term. Rather than enforcing these dynamics exactly along the full continuous curve, the optimization uses an affine linearization of $\mathbf{g}$ at representative sub-arc positions.

The objective linearization uses $n_{\mathrm{lin}}$ equal-parameter sub-arcs, distinct from the KOZ subdivision count $n_{\mathrm{seg}}$. In the code this count is passed as `sample_count`. Let $\hat S^{(i)}$ denote the corresponding segment matrices and define the centroid row

$$
\mathbf{w}^{(i)} = \frac{1}{N+1}\mathbf{1}^\top \hat S^{(i)}.
$$

Then the representative position and geometric acceleration are linear functions of the stacked control-point vector:

$$
\mathbf{r}_i(\mathbf{x}) = R_i\mathbf{x},
\qquad
R_i = \mathbf{w}^{(i)} \otimes I_3,
$$

$$
\mathbf{a}_i(\mathbf{x}) = A_i\mathbf{x},
\qquad
A_i = \frac{1}{T^2}\bigl(\mathbf{w}^{(i)}L_{2,N}\bigr)\otimes I_3.
$$

At SCP iteration $k$, the gravity field is linearized affinely about the reference position $\mathbf{r}_i(\mathbf{x}^{(k)})$:

$$
\mathbf{g}_i^{(k)}(\mathbf{x})
\approx
B_i^{(k)}\mathbf{x} + \mathbf{c}_i^{(k)},
$$

where $B_i^{(k)} = J_i^{(k)}R_i$, $J_i^{(k)}$ is the Jacobian of the gravity model evaluated numerically at the reference point, and

$$
\mathbf{c}_i^{(k)}
=
\mathbf{g}\bigl(\mathbf{r}_i(\mathbf{x}^{(k)})\bigr)
- J_i^{(k)}\mathbf{r}_i(\mathbf{x}^{(k)}).
$$

Define the linearized control-effort residual

$$
\boldsymbol{\rho}_i^{(k)}(\mathbf{x})
=
A_i\mathbf{x} - \left(B_i^{(k)}\mathbf{x} + \mathbf{c}_i^{(k)}\right).
$$

The implementation then builds an iteratively reweighted quadratic majorization of an L1-style objective:

$$
\omega_i^{(k)}
=
\frac{1/n_{\mathrm{lin}}}
{\sqrt{\left\|\boldsymbol{\rho}_i^{(k)}(\mathbf{x}^{(k)})\right\|_2^2 + \varepsilon}},
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

This is the manuscript-safe interpretation of the objective: an IRLS-weighted quadratic majorization of an L1-style proxy for control effort in the presence of affinely linearized gravity. It is suitable for warm-start generation, but it does not justify claims of true fuel optimality or physical superiority over other objective surrogates.

For completeness, the implementation also retains an L2 control-acceleration energy mode based on the same residual structure, together with the Gram-matrix quadratic form above. Under the locked paper decisions, that mode should appear only as implementation context or optional regularization, not as a parallel paper-level objective.

### 4.3 Convex subproblem and SCP update loop

At SCP iteration $k$, the method fixes the KOZ supporting half-spaces and the affine gravity linearizations constructed from the current control-point vector $\mathbf{x}^{(k)}$. The resulting subproblem is a convex quadratic program of the form

$$
\begin{aligned}
\min_{\mathbf{x}} \quad &
\frac{1}{2}\mathbf{x}^\top H^{(k)}\mathbf{x}
+ \bigl(\mathbf{f}^{(k)}\bigr)^\top \mathbf{x} \\
\text{s.t.} \quad &
A_{\mathrm{KOZ}}^{(k)}\mathbf{x} \ge \mathbf{b}_{\mathrm{KOZ}}^{(k)}, \\
&
A_{\mathrm{bc}}\mathbf{x} = \mathbf{b}_{\mathrm{bc}}, \\
&
\boldsymbol{\ell} \le \mathbf{x} \le \mathbf{u}.
\end{aligned}
$$

Here $A_{\mathrm{KOZ}}^{(k)}\mathbf{x} \ge \mathbf{b}_{\mathrm{KOZ}}^{(k)}$ collects the centroid-based supporting-half-space constraints, $A_{\mathrm{bc}}\mathbf{x}=\mathbf{b}_{\mathrm{bc}}$ collects any optional endpoint velocity or acceleration equalities, and the bounds $\boldsymbol{\ell}$ and $\mathbf{u}$ fix the endpoint positions.

When enabled, the implementation also adds a proximal regularization term around the current iterate,

$$
\frac{\lambda}{2}\|\mathbf{x}-\mathbf{x}^{(k)}\|_2^2,
$$

which preserves convexity and acts as an outer-loop stabilizer.

The current driver script used for the orbital experiments also enables an additional prograde-preservation constraint. Let $\hat{\mathbf{h}}$ denote the normalized initial angular-momentum direction, and define

$$
c(\tau;\mathbf{x}) = \hat{\mathbf{h}}^\top \bigl(\mathbf{r}(\tau;\mathbf{x}) \times \dot{\mathbf{r}}(\tau;\mathbf{x})\bigr).
$$

At a fixed set of interior parameter samples, the implementation linearizes this quantity at $\mathbf{x}^{(k)}$ and appends inequalities of the form

$$
c(\tau_j;\mathbf{x}^{(k)}) +
\nabla_{\mathbf{x}} c(\tau_j;\mathbf{x}^{(k)})^\top
(\mathbf{x}-\mathbf{x}^{(k)})
\ge 0.
$$

This constraint is configuration-specific rather than central to the paper's main claim. If it remains enabled in the reported experiments, it should be disclosed explicitly either here or in the experimental-setup section; it should not be omitted from the paper if it materially shapes the reported trajectories.

The outer loop is initialized from a straight-line control polygon between the endpoints. Each SCP iteration then:

1. subdivides the current control polygon;
2. rebuilds the KOZ supporting half-spaces;
3. rebuilds the affine gravity linearization and IRLS weights;
4. solves the convex QP;
5. updates the reference control points.

In the current implementation the convex subproblem is solved numerically with `trust-constr` using the exact gradient and Hessian of the quadratic model. Convergence is declared when the control-point update satisfies

$$
\|P^{(k+1)} - P^{(k)}\|_F < \mathrm{tol},
$$

or when the prescribed maximum number of outer iterations is reached.

If enabled, the implementation also clips the accepted SCP step to a prescribed control-point-vector radius after the QP solve. This is a practical step-acceptance safeguard, not a trust-region constraint inside the convex QP itself. The distinction matters because the defensible claim is that the method solves a sequence of convex QPs, not that it solves a trust-region-constrained exact reformulation of the original nonconvex problem.

### 4.4 Assumptions and scope boundary

The formulation should close with its limits stated explicitly.

First, the continuous-exclusion argument is tied to a spherical KOZ together with the specific subdivision-and-supporting-half-space construction. The resulting statement is conservative and assumption-dependent. It should not be phrased as a general theorem for arbitrary obstacle geometry.

Second, transfer time is fixed. The present formulation does not optimize timing, free final time, or waiting behavior. All derivative and objective terms are interpreted under the fixed relation $t=T\tau$.

Third, the paper-level objective is a surrogate. The `dv` mode is an IRLS-based proxy built from geometric acceleration minus affinely linearized gravity. It is appropriate for trajectory initialization, but it does not support claims of true delta-v optimality, global optimality, or physical dominance over alternative surrogate objectives.

Fourth, the affine gravity treatment is local. The method linearizes the two-body plus J2 field at representative sub-arc positions, so the objective model is approximate even though each outer subproblem is convex.

Finally, the method section should describe the framework as designed to generate warm-start trajectories for downstream solvers. Demonstrated downstream usefulness is a results-level claim and should not be asserted by the method section alone.
