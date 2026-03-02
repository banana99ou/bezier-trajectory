# Chat notes: time scaling, SCP, and gravity/J2 objectives

Date: 2026-02-24  
Workspace: `bezier-trajectory/`

This document captures the key conclusions and file pointers from our chat about:

- Reading GitHub repos vs. local copies
- How the downloaded Bézier drone optimizer handles time scaling (`T`) and SCP
- How the orbital docking optimizer in this repo defines its cost function
- How to move toward an SCP scheme with a **linearized gravity** (and optionally **J2**) term

---

## 1) GitHub access (what worked / didn’t)

- Public repos can be read via raw URLs or the GitHub API.
- The provided URL was a **private** repo, so GitHub returned **404** without authentication.
- Since a local copy existed, analysis proceeded directly on the downloaded folder instead of GitHub.

---

## 2) Downloaded repo: `Convex_Trajectory_Optimization_used_Bezier_drone-main/`

### 2.1 How `T` is defined

- `T` is a **user-selected fixed constant**, set in `main.py` and passed to the solver.
- There is **no algorithm** inside that codebase that derives or optimizes `T`.

### 2.2 Why boundary conditions scale with `T`

That codebase uses a normalized curve parameter \(\tau \in [0,1]\) and maps it to time via:

\[
t = \tau T
\]

So derivatives transform as:

\[
\frac{d}{dt} = \frac{1}{T}\frac{d}{d\tau},\qquad
\frac{d^2}{dt^2} = \frac{1}{T^2}\frac{d^2}{d\tau^2}
\]

Therefore, when boundary conditions are specified in physical units (m/s, m/s²), the constraints written using \(\tau\)-derivative rows must scale the RHS by `T` and `T**2` to remain consistent.

Key file:

- `Condition/Boundary_Conditions.py` (velocity RHS scaled by `T`, acceleration RHS scaled by `T**2`)

### 2.3 What SCP is doing there

- SCP is implemented for **obstacle avoidance only**:
  - It repeatedly rebuilds linearized obstacle half-space constraints around the current trajectory and resolves a QP.
- Gravity is treated as a **constant vector** in the drone model, so there is no nonlinear gravity term to SCP-linearize.
- J2 is not present in that downloaded codebase.

Key file:

- `Solver/Solving_QP.py` (outer SCP loop for obstacle constraints)

---

## 3) In this repo: `Orbital_Docking_Optimizer.py` and `orbital_docking/`

### 3.1 Where the cost function is defined

`Orbital_Docking_Optimizer.py` is the runner; the objective is defined in:

- `orbital_docking/optimization.py`

The current objective is a **quadratic form** based on a precomputed matrix `G_tilde` and **does not include gravity in the optimization objective** (gravity is used in visualization).

Cost helper:

- `_compute_cost_only(...)` in `orbital_docking/optimization.py`:
  - Computes `cost = tr(P^T G_tilde P)`

Gradient/Hessian:

- `cost_function_gradient_hessian(...)` in `orbital_docking/optimization.py`:
  - \(\nabla J = 2 G_{\tilde{}} P\)
  - \(H = 2\,\mathrm{kron}(I_{dim}, G_{\tilde{}})\)

### 3.2 What `G_tilde` represents

`G_tilde` is built in:

- `orbital_docking/bezier.py` in `BezierCurve.__init__`

It is:

\[
G_{\tilde{}} = (E D E D)^T\; G\; (E D E D)
\]

This corresponds to an \(L^2\)-style penalty on the curve’s **second derivative w.r.t. \(\tau\)** (not physical time).

### 3.3 KOZ SCP-like iteration (what is being linearized)

The iteration in `optimize_orbital_docking(...)` repeatedly rebuilds **KOZ constraints** based on the current curve and resolves the constrained optimization. This is analogous to SCP, but it is only convexifying the **constraint geometry** (KOZ), not gravity dynamics.

Files:

- `orbital_docking/constraints.py` (linear KOZ constraints)
- `orbital_docking/optimization.py` (outer loop that updates KOZ constraints)

### 3.4 Gravity in this repo today

- Gravity is computed for plotting in `orbital_docking/visualization.py` using:
  - \(\displaystyle a_g(r) = -\mu \frac{r}{\|r\|^3}\) (with `EARTH_MU_SCALED`)
- But that gravity term is not currently part of the optimizer’s cost or constraints.

Files:

- `orbital_docking/constants.py` (`EARTH_MU_SCALED`)
- `orbital_docking/visualization.py` (gravity vectors for plots)

---

## 4) Moving to a cost with linearized gravity (SCP concept)

Goal concept (typical for continuous-thrust proxy):

\[
J = \int_0^T \|u(t)\|^2 dt,\qquad
u(t)=\ddot r(t) - a_g(r(t))
\]

Nonconvexity comes from \(a_g(r(t))\).

### 4.1 SCP linearization

At SCP iteration \(k\), linearize gravity about reference \(r_k(t)\):

\[
a_g(r) \approx a_g(r_k) + A_k (r - r_k),
\quad A_k=\left.\frac{\partial a_g}{\partial r}\right|_{r_k}
\]

For two-body gravity:

\[
A_k = -\mu\left(\frac{I}{\|r_k\|^3} - \frac{3 r_k r_k^T}{\|r_k\|^5}\right)
\]

Then a convex quadratic surrogate cost is:

\[
J_k \approx \sum_i w_i \left\|\ddot r(P,t_i) - \left(a_g(r_{k,i}) + A_{k,i}(r(P,t_i)-r_{k,i})\right)\right\|^2
\]

Inside the norm is **affine in** \(P\) (since both \(r(P,t_i)\) and \(\ddot r(P,t_i)\) are linear maps of control points), so \(J_k\) becomes a **convex quadratic** in \(P\).

### 4.2 Important: time scaling is required if comparing against gravity

To compare \(\ddot r\) to gravity (km/s²), the Bézier “acceleration” must be with respect to physical time:

\[
\ddot r(t) = \frac{1}{T^2}\frac{d^2 r}{d\tau^2}
\]

So implementing gravity-consistent costs/constraints implies introducing an explicit `T` and applying \(1/T^2\) scaling to second derivatives (and related scalings in boundary conditions/objective), similar in spirit to the downloaded drone codebase.

---

## 5) Adding J2 (optional) via SCP

If adding J2 acceleration \(a_{J2}(r)\), you can SCP-linearize similarly:

\[
a_{J2}(r) \approx a_{J2}(r_k) + A^{J2}_k (r-r_k)
\]

Where \(A^{J2}_k\) can be computed:

- analytically (messy), or
- via finite differences around \(r_k\) (often good enough for SCP prototypes).

Then include \(a_g + a_{J2}\) in the same linearized residual used in the objective (or dynamics constraints).

---

## 6) Suggested next step (implementation direction)

To incorporate linearized gravity into optimization **while staying convex per SCP iteration**:

- Keep the existing outer loop structure in `orbital_docking/optimization.py`
- Add an outer-loop update that:
  - samples reference points \(r_k(t_i)\)
  - computes \(a_g(r_{k,i})\) and Jacobians \(A_{k,i}\)
  - builds a quadratic surrogate objective \(J_k(P)\)
- Solve the resulting constrained problem (still using `trust-constr`, or move to a QP solver if everything is quadratic + linear constraints).

