# Notation lock

**This file is the single source of truth for every mathematical symbol used in
this repository's papers, method drafts, figure captions, and design docs.**

It outranks every draft. If a draft and this file disagree, the draft is wrong.
If you need a symbol that is not here, add it here *first*, in the same edit.

Enforced by `tools/check_notation.py`. Run it before committing any doc change:

```
python3 tools/check_notation.py
```

Superseded on 2026-08-10: the "Frozen method notation lock" in
`doc/paper_execution_state.md` and the `T1` table in `doc/method_artifact_pack.md`.
Both are stale (they still mandate Δv / IRLS wording that `CLAUDE.md` forbids)
and are retained only as historical record.

---

## 1. Index letters — one letter, one role

Every index below has exactly one meaning. Never reuse one for another role,
even locally, even "obviously from context".

| Letter | Role | Range |
|---|---|---|
| `i` | control-point / Bernstein basis index of the **whole** curve | `0 … N` |
| `m` | control-point index **within one subdivided sub-arc** | `0 … N` |
| `l` | second index of a matrix entry (`[G_N]_{il}`, `d_{il}`) | `0 … N` |
| `s` | KOZ subdivision sub-arc | `1 … n_seg` |
| `j` | gravity-linearization interval | `1 … n_lin` |
| `k` | SCvx iteration — **superscript `(k)` only, never a subscript** | `0, 1, 2, …` |

`i` and `m` coexist in the KOZ gradient, which differentiates a sub-arc control
point with respect to a whole-curve control point:
`∂γ^{(s)}_m / ∂p_i`, with weights `S^{(s)}_{mi}` and `w^{(s)}_i`.

## 2. Superscript grammar

The parenthesized superscript was previously overloaded four ways. It is now
partitioned:

| Form | Means | Example |
|---|---|---|
| `^{(k)}` | SCvx iteration — **reserved, never anything else** | `x^{(k)}`, `H^{(k)}` |
| `^{(s)}` | KOZ sub-arc | `S^{(s)}`, `q^{(s)}_m` |
| `^{(j)}` | gravity-linearization interval | `f^{(j)}`, `Ŝ^{(j)}` |
| `^{[q]}` | derivative order (**square** brackets) | `P^{[1]}`, `P^{[2]}` |
| `^{\mathsf{T}}` | transpose | `P^{\mathsf{T}}` |

Square brackets for derivative order exist because `P^{(2)}` was ambiguous:
it meant both "second-derivative control points" and "sub-arc 2 control
points", and `n_seg ≥ 2` always, so both readings were always live.

## 3. Typography

| Class | Style | Examples |
|---|---|---|
| Latin vector | `\mathbf{}` | `\mathbf{r}`, `\mathbf{p}_i`, `\mathbf{x}`, `\mathbf{u}` |
| Greek / script vector | `\boldsymbol{}` | `\boldsymbol{\nu}`, `\boldsymbol{\ell}` |
| Matrix | upright-italic capital, **no bold** | `P`, `G_N`, `S^{(s)}`, `H^{(k)}` |
| Scalar | plain italic | `T`, `N`, `\rho_k`, `\eta`, `\mu` |
| Set | `\mathcal{}` | `\mathcal{K}`, `\mathcal{H}^{(s)}` |
| Transpose | `^{\mathsf{T}}` — **never `\top`** | `\mathbf{x}^{\mathsf{T}}` |
| Norm | `\|\cdot\|_2` — **never `\lVert`**, never an unsubscripted `\|\cdot\|` where the 2-norm is meant | `\|\mathbf{r}\|_2` |

## 4. Curve and decision variables

| Symbol | Type | Meaning | Code |
|---|---|---|---|
| `\tau` | `[0,1]` | normalized curve parameter | `tau` |
| `t = T\tau` | s | physical time | — |
| `T` | s | transfer time (**fixed**) | `T` |
| `N` | int | Bézier degree | `N` |
| `B_i^N(\tau)` | scalar | Bernstein basis polynomial | `_bernstein_basis` |
| `\mathbf{r}(\tau)` | `R^3`, km | position | — |
| `\dot{\mathbf{r}}, \ddot{\mathbf{r}}` | — | derivatives **with respect to `t`** | — |
| `\mathbf{p}_i` | `R^3` | `i`-th control point | — |
| `P = [\mathbf{p}_0,\ldots,\mathbf{p}_N]^{\mathsf{T}}` | `R^{(N+1)×3}` | control-point matrix, **one control point per row** | `P` |
| `\mathbf{x} = \mathrm{vec}(P^{\mathsf{T}})` | `R^{3(N+1)}` | decision vector, **point-major** | `P_flat` |
| `\mathbf{u}(t)` | `R^3`, km/s² | control acceleration | — |

`P` and `\mathbf{x}` must never be given the same right-hand side. `P` is a
matrix of rows; `\mathbf{x}` is the point-major stacking of that matrix.
Point-major ordering is what makes `\tilde G_N \otimes I_3` the correct
Kronecker order — reversing it silently transposes the objective.

## 5. Linear operators

| Symbol | Dimensions | Definition | Code |
|---|---|---|---|
| `D_N` | `N×(N+1)` | Bézier difference matrix | `get_D_matrix(N)` |
| `E_M` | `(M+2)×(M+1)` | degree elevation, degree `M` → `M+1` | `get_E_matrix(M)` |
| `L_{1,N}` | `(N+1)×(N+1)` | `E_{N-1} D_N`, degree-preserving velocity map | — |
| `L_{2,N}` | `(N+1)×(N+1)` | `E_{N-1} D_N E_{N-1} D_N`, acceleration map | — |
| `P^{[1]}, P^{[2]}` | `(N+1)×3` | `L_{1,N}P`, `L_{2,N}P` | — |
| `G_N` | `(N+1)×(N+1)` | Bernstein Gram matrix | — |
| `\tilde G_N` | `(N+1)×(N+1)` | `L_{2,N}^{\mathsf{T}} G_N L_{2,N}` | `BezierCurve.G_tilde` |
| `S^{(s)}` | `(N+1)×(N+1)` | De Casteljau subdivision matrix, **KOZ** sub-arc `s` | `segment_matrices_equal_params(N, n_seg)` |
| `\hat S^{(j)}` | `(N+1)×(N+1)` | De Casteljau subdivision matrix, **gravity** interval `j` | `segment_matrices_equal_params(N, n_lin_seg)` |
| `I_3` | `3×3` | identity | — |
| `\otimes` | — | Kronecker product | `np.kron` |

The hat on `\hat S^{(j)}` is load-bearing: the KOZ subdivision and the gravity
linearization use **different** counts (`n_seg` vs `n_lin`) and therefore
different matrices. Writing both as `S` implies a coupling that does not exist.

## 6. KOZ construction

| Symbol | Type | Meaning | Code |
|---|---|---|---|
| `\mathcal{K}` | set | keep-out zone (sphere) | — |
| `\mathbf{c}_{\mathrm{KOZ}}` | `R^3` | KOZ center (origin in this paper) | — |
| `R_{\mathrm{KOZ}}` | km | **KOZ radius** | `r_e` arg of `optimize_orbital_docking` |
| `n_{\mathrm{seg}}` | int | KOZ subdivision count | `n_seg` |
| `P^{(s)} = S^{(s)}P` | `(N+1)×3` | sub-arc control polygon | — |
| `\mathbf{q}^{(s)}_m` | `R^3` | `m`-th control point of sub-arc `s` | — |
| `\mathbf{c}^{(s)}` | `R^3` | centroid of `P^{(s)}` | — |
| `\mathbf{n}^{(s)}` | `R^3`, unit | outward supporting normal | — |
| `\mathcal{H}^{(s)}` | set | supporting half-space | — |
| `\gamma^{(s)}_m(\mathbf{x})` | km | clearance of control point `m` of sub-arc `s` | — |
| `w^{(s)}_i` | scalar | centroid weight, `\frac{1}{N+1}\sum_m S^{(s)}_{mi}` | — |
| `h(\mathbf{x})` | km | total half-space violation (feasibility measure) | — |
| `L_{\mathrm{seg}}` | km | chord length between the two endpoints of a sub-arc | — |

> **`R_{\mathrm{KOZ}}`, not `r_e`.** In `orbital_docking/optimization.py` the
> identifier `r_e` means the **KOZ radius** (6471 km) everywhere except inside
> `_build_ctrl_accel_quadratic`, where line 253 rebinds it to
> `EARTH_RADIUS_KM` (6371 km) for the J2 term. That is currently safe only
> because `_build_ctrl_accel_quadratic` takes no `r_e` parameter. Adding one
> would silently change the gravity model. Do not propagate `r_e` into the
> paper, and do not add an `r_e` parameter to that function.

## 7. Objective and gravity

| Symbol | Type | Meaning | Code |
|---|---|---|---|
| `\mathbf{g}(\mathbf{r})` | `R^3` | gravity: two-body + J2 | `_accel_total` |
| `\mathrm{GM}` | km³/s² | Earth gravitational parameter, `3.986004418e5` | `EARTH_MU_SCALED` |
| `R_\oplus` | km | Earth radius, `6371` — **not** `R_{\mathrm{KOZ}}` | `EARTH_RADIUS_KM` |
| `J_2` | — | second zonal harmonic, `1.08262668e-3` | `EARTH_J2` |
| `n_{\mathrm{lin}}` | int | gravity-linearization interval count | `n_lin_seg` |
| `\mathbf{r}_j^{(k)}` | `R^3` | reference position of interval `j` at iteration `k` | `r_ref` |
| `\nabla\mathbf{g}_j^{(k)}` | `3×3` | gravity Jacobian at that reference point | `J_s` |
| `\mathbf{c}_j^{(k)}` | `R^3` | affine offset of the gravity model | — |
| `\xi` | `[0,1]` | local parameter within one interval | — |
| `\mathbf{f}^{(j)}(\xi)` | `R^3` | control-acceleration residual on interval `j` | — |
| `F_j(\mathbf{x})` | `(N+1)×3` | control points of `\mathbf{f}^{(j)}` | — |
| `J(\mathbf{x})` | scalar | objective, **exact** gravity | — |
| `J^{(k)}(\mathbf{x})` | scalar | objective, gravity linearized at iteration `k` | — |
| `\mathbf{w}^{(j)}` | `1×(N+1)` | centroid row, `\frac{1}{N+1}\mathbf{1}^{\mathsf{T}}\hat S^{(j)}` | `w_seg` |
| `R_j = \mathbf{w}^{(j)} \otimes I_3` | `3×3(N+1)` | extracts interval `j`'s reference position from `\mathbf{x}` | — |
| `\Lambda_j` | `3×3(N+1)` | extracts interval `j`'s geometric acceleration from `\mathbf{x}` | — |
| `\Gamma_j^{(k)} = \nabla\mathbf{g}_j^{(k)} R_j` | `3×3(N+1)` | composed affine gravity map | — |

`\nabla\mathbf{g}` spells out the gravity Jacobian rather than reusing `J`,
which already denotes the objective (and, as `J^{(k)}` vs `J_j^{(k)}`, differed
from the Jacobian by a single subscript). `J_2` is the zonal harmonic coefficient and is the one admitted exception to
the `J`-is-the-objective rule: a bare numeric subscript with no argument and no
iteration index cannot be read as `J(\mathbf{x})` or `J^{(k)}`, and every
aerospace reader expects that spelling. `\mathrm{GM}` rather than `\mu` for the
gravitational parameter, because `\mu` is the exact-penalty weight (§8).

## 8. SCvx algorithm

| Symbol | Type | Meaning | Code |
|---|---|---|---|
| `k` | int | iteration counter | `it` |
| `\mathbf{x}^{(k)}` | `R^{3(N+1)}` | reference iterate | `P_ref` |
| `\hat{\mathbf{x}}` | `R^{3(N+1)}` | subproblem solution (candidate) | — |
| `H^{(k)}, \boldsymbol{\ell}^{(k)}` | — | QP quadratic / linear terms | `Hf`, `ff` |
| `A_{\mathrm{KOZ}}^{(k)}, \mathbf{b}_{\mathrm{KOZ}}^{(k)}` | row stack | linearized KOZ rows | — |
| `A_{\mathrm{bc}}, \mathbf{b}_{\mathrm{bc}}` | row stack | boundary-condition rows | — |
| `\boldsymbol{\nu} \ge \mathbf{0}` | — | virtual control (slack); component `\nu^{(s)}_m` | `elastic` slack |
| `\mu` | scalar | exact-penalty weight | `elastic_weight` |
| `\Delta_k` | km | trust-region radius; `\Delta_0` initial | `scp_trust_radius` |
| `\phi(\mathbf{x}) = J + \mu h` | scalar | true merit function | — |
| `\phi^{(k)} = J^{(k)} + \mu h^{(k)}` | scalar | convex model merit | — |
| `\rho_k` | scalar | actual / predicted merit reduction | — |
| `\eta` | scalar | acceptance threshold | — |
| `n_{\mathrm{conv}}` | int | consecutive iterations required to declare convergence | — |

Warrants for the letter choices, per the freeze's "textbook-with-citation"
rule: `\boldsymbol{\nu}` for virtual control follows Malyuta et al. [5]; `\mu`
for the penalty weight and `\Delta_k` for the trust-region radius follow
Nocedal & Wright [9], already cited in the paper.

## 9. Experiment and results symbols

| Symbol | Meaning |
|---|---|
| `e`, `e_0`, `e_f` | eccentricity — **never `ecc`** |
| `T_{\mathrm{normed}}` | normalized transfer time |
| `\Delta a`, `\Delta i` | semi-major-axis / inclination difference |
| initial altitude | **spell out** in table headers; no symbol (`h` is the feasibility measure) |
| `\lvert \Delta\mathrm{cost} \rvert` | absolute cost difference — single bars, not `\|` |

## 10. Banned spellings

Each maps to its replacement. `tools/check_notation.py` flags all of these.

| Banned | Use instead | Because |
|---|---|---|
| `\top` | `\mathsf{T}` | 17 vs 15 split in rev2, both in one display |
| `\lVert` / `\rVert` | `\|` | single norm macro |
| `r_e` | `R_{\mathrm{KOZ}}` | collides with Earth radius in code; with `r_k`, `\mathbf{r}` in prose |
| `r_k`, `r_0` (radius) | `\Delta_k`, `\Delta_0` | `r` already means position and KOZ radius |
| `w_s` | `\mu` | `s` collides with the sub-arc index; `w^{(s)}_i` is the centroid weight |
| `\mathbf{s}`, `s^{(s)}_k` (slack) | `\boldsymbol{\nu}`, `\nu^{(s)}_m` | `s` is the sub-arc index |
| `J_i`, `J_s` (Jacobian) | `\nabla\mathbf{g}_j` | `J` is the objective |
| `P^{(1)}`, `P^{(2)}` (derivative) | `P^{[1]}`, `P^{[2]}` | collides with sub-arc superscript |
| `u` (local parameter) | `\xi` | `\mathbf{u}` is control acceleration |
| `K` (streak length) | `n_{\mathrm{conv}}` | collides with `\mathcal{K}` and `k` |
| `h_0` (altitude) | spell out | `h` is the feasibility measure |
| `L(x)`, `T(x)` (merits) | `\phi^{(k)}`, `\phi` | `L` is a derivative operator, `T` the transfer time |
| `g_k(x)` (clearance) | `\gamma^{(s)}_m` | `\mathbf{g}` is gravity |
| `\mathrm{ecc}` | `e` | one spelling per quantity |
| `p` (reference iterate) | `\mathbf{x}^{(k)}` | `\mathbf{p}_i` is a control point |
| `A` (subdivision map) | `S^{(s)}` / `\hat S^{(j)}` | `A_{\mathrm{KOZ}}`, `A_{\mathrm{bc}}` are constraint matrices |
| `A_i` (acceleration map) | `\Lambda_j` | `A_*` is reserved for constraint matrices |
| `B_i^{(k)}` (gravity map) | `\Gamma_j^{(k)}` | `B_i^N` is the Bernstein basis |
| `\boldsymbol{\rho}_i^{(k)}` (residual) | `\mathbf{f}_j^{(k)}` | `\rho_k` is the merit ratio |

## 11. Scope

Applies to: `doc/*.md`, figure captions, table headers, and any equation in a
commit message or handoff note. Code identifiers are **not** renamed by this
lock — the "Code" columns above are a mapping, not a mandate. Where a code name
contradicts the lock (`r_e`, `J_s`, `ff`), the mapping table is the bridge.
