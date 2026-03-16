# Bézier-curve-based trajectory optimization (Earth-orbit rendezvous demo)

This project optimizes a **Bézier curve trajectory** for a simplified orbital rendezvous/docking approach scenario.
The trajectory is represented in **ECI coordinates** and is constrained to stay outside an **Earth-centric spherical keep-out zone (KOZ)** (i.e., minimum geocentric radius).

The main technical contribution is a practical way to enforce an “outside-a-sphere” constraint by **convexifying it segment-by-segment**:
split the Bézier curve via **De Casteljau** subdivision and approximate the nonlinear constraint by a set of **supporting half-spaces** per segment.

---

## Coordinate frame, units, and scaling

- **Frame**: Earth-Centered Inertial (ECI).
- **Units**: km, km/s, km/s².
- **Parameter**: Bézier parameter `τ ∈ [0, 1]`.

### Fixed time scaling (for now)

We use a **fixed, arbitrary transfer time** `T := t_f - t_0` (seconds) to map `τ` to physical time:

$$
τ = (t - t0) / T
ṙ(t) = (1 / T) * r′(τ)
r̈(t) = (1 / T^2) * r″(τ)
$$

**Note**: without `T`, “acceleration w.r.t. `τ`” is not a physical acceleration.

---

## Scenario (Progress-to-ISS fast-rendezvous inspired)

Mission basis:
- **Progress fast-rendezvous timeline**: Progress MS-09 completed a two-orbit rendezvous to the ISS in about 3 h 40 min.
- **Progress insertion orbit**: published mission profile values for Progress MS give an insertion orbit of 193 km x 245 km at 51.67 deg inclination.
- **ISS orbit**: NASA reports the ISS operates at roughly 370-460 km altitude with 51.6 deg inclination.

Reduced-order modeling choice used in this project:
- **Start orbit**: circularized Progress-like parking orbit at **245 km** altitude.
- **Target orbit**: ISS-like circular orbit at **400 km** altitude.
- **Inclination**: **51.64 deg**.
- **RAAN**: **0 deg**.
- **Target argument of latitude**: **45 deg**.
- **Progress phase lag behind ISS**: **30 deg**.
- **Earth-centric KOZ altitude**: **100 km AMSL**.

This is intentionally **not** a literal reconstruction of Progress MS-09. The sourced mission values define the orbital regime; the circularized start orbit, fixed RAAN, and 30 deg phase lag are simplified single-arc choices for the present Bézier-based optimizer.

Derived Bézier endpoint positions used by the current simplified scenario:
- **Start endpoint** `P0` (Progress-like): `[6390.565267, 1062.683295, 1342.697206] km`
- **End endpoint** `PN` (ISS-like): `[4787.820015, 2971.323532, 3754.258511] km`

Source URLs for later paper references:
- NASA ISS facts: <https://www.nasa.gov/international-space-station/space-station-facts-and-figures>
- TASS report on Progress MS-09 fast rendezvous: <https://tass.com/science/1012518>
- Progress MS mission profile / insertion orbit: <https://spaceflight101.com/progress-ms/progress-ms-flight-profile/>

Interpretation (Earth-centric KOZ):
$$
||r(τ)|| ≥ r_KOZ := R_E + 100 km
$$

---

## Trajectory representation (Bézier curve)

Let `r(τ)` be a Bézier curve of degree `N` with control points:
```text
P = [P0, P1, ..., PN],   Pi ∈ R^3
```

We solve for the control points (typically only the interior points are free; endpoints fixed).

---

## Derivatives in control-point space (D/E matrices)

We compute derivatives efficiently via matrices acting on the control points:

- **Derivative matrix** `D` (degree `N → N-1`):
$$
[\mathbf{D}]_{i,j} = N \times
\begin{cases}
-1, & \text{if } j = i \\
1, & \text{if } j = i + 1 \\
0, & \text{otherwise}
\end{cases}
$$

- **Degree-elevation matrix** `E_{N→N+1}` (degree `N → N+1`):
$$
[\mathbf{E}_{\,{N \to N+1}}]_{i,j} =
\begin{cases}
1, & \text{if } i = 1 \text{ and } j = 1 \\
1, & \text{if } i = N + 2 \text{ and } j = N + 1 \\
\dfrac{N + 2 - i}{N + 1}, & \text{if } 2 \le i \le N + 1 \text{ and } j = i - 1 \\
\dfrac{i - 1}{N + 1}, & \text{if } 2 \le i \le N + 1 \text{ and } j = i \\
0, & \text{otherwise}
\end{cases}
$$

Then:
$$
V = E D P
A = E D E V = E D E D P
$$
which represent the (parameter-space) velocity and acceleration control points.

---

## Optimization problem statement (decision variables / objective / constraints)

### Decision variables

- Full control-point stack `x = vec(P)`.
- Endpoint positions are fixed by bounds on `P0` and `PN`, so effective free variables are interior points.
- fixed transfer time `T`.

### Constraints

- **Boundary conditions (equality constraints)**:
  - Position: `r(0)=P_0`, `r(1)=P_N`.
  - Optional: velocity and acceleration endpoint constraints in physical units (ECI), written as linear equalities in `P`:
    $$
    v(0) = (N/T) * (P1 - P0)
    v(1) = (N/T) * (PN - P_{N-1})

    a(0) = (N*(N-1)/T^2) * (P2 - 2*P1 + P0)
    a(1) = (N*(N-1)/T^2) * (PN - 2*P_{N-1} + P_{N-2})
    $$

- **Earth-centric KOZ (inequality, nonlinear)**:
  $$
  ||r(τ)|| ≥ r_{KOZ} \quad \text{for all} \quad τ ∈ [0, 1]
  $$
  Enforced approximately by segment-wise convexification (next section).

### Objective (fixed-`T` baseline, current implementation)

# key cost to minimize is **delta-v surrogate**. anything else claiming or suggesting otherwise is simply false or error.

The optimizer uses a control-acceleration least-squares objective with fixed transfer time `T`.
At each SCP iteration, gravity (two-body + J2) is linearized around the current reference trajectory,
yielding a convex quadratic subproblem in control points.

1) **Geometric acceleration-energy term**:
$$
J_{\text{geom}}(P) = \int_{0}^{1} \left\| \frac{1}{T^2} \, \mathbf{r}''(\tau) \right\|^2 \, d\tau
$$

2) **Control-acceleration term with linearized gravity/J2**:
$$
\begin{align*}
g_{\text{total}}(\mathbf{r}) &= g_{\text{two body}}(\mathbf{r}) + g_{J2}(\mathbf{r}) \\
g_{\text{total}}(\mathbf{r}(\tau)) &\approx M_{\text{grav}}(\tau)\, \mathbf{P} + \mathbf{b}_{\text{grav}}(\tau)
\end{align*}
$$
so the iteration objective is convex quadratic in `P`.

In implementation form, each outer iteration solves:
$$
\min_{x} \;\; \frac{1}{2} x^\top H x + f^\top x
$$
where `H,f` come from:
- exact geometric term `(1/T^4) * (G_tilde ⊗ I_3)`
- sampled segment-wise linearized residual terms for `a_geom - g_total`

and reports the full least-squares energy with constant term:
$$
J_{\text{true}}(x) = 0.5\, x^\top H x + f^\top x + c
$$

---

## KOZ convexification via De Casteljau segmentation (core method)

The KOZ constraint `||r(τ)|| ≥ r_KOZ` is nonconvex.
We approximate it by:

1) **Split** the Bézier curve into `K` segments using De Casteljau subdivision (equal parameter partition).
   Each segment `i` has control points:
   $$
   Q_i = A_i P
   $$

2) For each segment `i`, compute the centroid `c_i = mean(Q_i)`.

3) Construct an outward unit normal (supporting half-space normal):
   $$
   n_i = \frac{c_i - c_{\mathrm{KOZ}}}{\left\|c_i - c_{\mathrm{KOZ}}\right\|}
   $$
   $$
   \text{(Earth-centric case: } c_{\mathrm{KOZ}} = 0 \text{)}
   $$

4) Enforce **linear half-space constraints** for all control points of segment `i`:
   $$
   n_i^T q_{i,k} ≥ r_KOZ    for all k ∈ {0, ..., N}
   $$

Because Bézier curves lie in the convex hull of their control points, enforcing this on segment control points
pushes the segment away from the sphere in a conservative and computationally efficient way.

This is iterated (fixed-point / SCP-style): update normals from the current solution, resolve, repeat until convergence.

---

## Solver

- **Per-iteration solver**: SciPy `minimize(..., method="trust-constr")`
- **Overall method**: iterative convexification (update KOZ half-spaces and update gravity/J2 linearization every outer iteration)

---

## Outputs / figures (baseline reproducibility targets)

- Trajectory comparison for `K ∈ {2, 4, 8, 16, 32, 64}`
- Performance figures (runtime / iterations / feasibility vs `K`)
- Time vs curve order `N ∈ {2, 3, 4}` (quadratic/cubic/4th-degree)
- Position/velocity/acceleration profiles vs `τ` for each `K`

---

## TODO / next steps

- **Realistic scenario**: if time allows later, fetch an epoch-specific ISS state and a matching visiting-vehicle state instead of using the current Progress-inspired reduced-order geometry.
- **Time scaling**: select a meaningful fixed `T` (e.g., minutes-hours) and report how `J` changes; later add `T` as a variable with thrust limits.
- **Dynamics-aware objective**: implemented in baseline form (gravity + J2 linearization each SCP iteration).
  - Remaining work: improve linearization robustness (e.g., trust region/proximal regularization) and report convergence quality vs `K`.
- **Clarify KOZ meaning for publication**: Earth-centric minimum-altitude KOZ vs ISS-centric keep-out sphere (different physical meaning).
