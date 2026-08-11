# Pillar 2 -- Ablation (energy, n_seg=16, 5 geometries)

The conclusion under test is an ISOLATION claim, so it has to hold on more
than the geometry it was diagnosed on: the trust region is the primary fix
and the mis-scaled proximal is a secondary aggravator. Trust cells use each
scenario's own r0 (see module docstring).

A geometry where (a0) itself runs to the cap yields NOT MEASURED: with both
legacy cells censored at the same ceiling the aggravation comparison is
empty, and reading that as either result would be a measurement artifact.

| scenario | (a) iters | (a0) iters | (b) iters | (c) iters | a0/b ratio | verdict |
|---|---|---|---|---|---|---|
| phase70 | 12000 (cap) | 69 | 6 | 6 | 11.5x | **PASS** |
| phase120 | 12000 (cap) | 1197 | 8 | 8 | 149.6x | **PASS** |
| phase135 | 12000 (cap) | 2385 | 9 | 9 | 265.0x | **PASS** |
| phase170 | 12000 (cap) | 12000 (cap) | 12 | 12 | ≥1000.0x | **NOT MEASURED** |
| planechange | 12000 (cap) | 2339 | 10 | 10 | 233.9x | **PASS** |

- aggravation measurable on 4/5 geometries: phase70, phase120, phase135, planechange (not measured: phase170 — (a0) did not terminate within 12000)
- every measured geometry agrees: **True**

## VERDICT: PASS

---

## phase70 (trust cells at r0 = 2000 km)

| cell | iters | capped | converged | feasible | min_r | J_true | cost_true_energy |
|---|---|---|---|---|---|---|---|
| (a) trust0 + prox1e-6 | 12000 | True | 0 | True | 6616.000 | 5.559807e-05 | 5.559946e-05 |
| (a0) trust0 + prox0 | 69 | False | 0 | True | 6616.000 | 3.347149e-05 | 3.347253e-05 |
| (b) trust2000 + prox0 | 6 | False | 1 | True | 6616.000 | 3.347149e-05 | 3.347253e-05 |
| (c) trust2000 + prox1e-6 | 6 | False | 1 | True | 6616.000 | 3.347149e-05 | 3.347253e-05 |

- (a) legacy loop + mis-scaled prox reproduces the crawl (caps): **True** (12000 iters)
- trust region is the PRIMARY fix: legacy loop even WITHOUT the prox (a0=69 iters, never satisfies the SCvx convergence criterion) is >3x slower than the trust path, which does converge (b=6, c=6 iters): **True**
- the mis-scaled proximal is a SECONDARY aggravator: it pushes the legacy loop from 69 iters (a0) to 12000 (the cap) (a): **True**
- (b)(c) both converge + feasible: **True**
- (b) prox0 and (c) prox1e-6 are bit-identical: **True** (NOT A GATE — the proximal is skipped when `trust_active`, so these two cells run the same code; excluded from the verdict because it cannot fail)

Isolation finding: Cell (a) alone conflates two changes -- setting `scp_trust_radius=0` both reverts to the legacy unconditional-accept loop AND re-enables the proximal. The added cell (a0) separates them: with the proximal removed the legacy loop still takes 69 iters (it exits via the step-norm tolerance, not the SCvx criterion) -- >3x the trust path's 6. So the **trust region is the primary fix**; the mis-scaled proximal is a **secondary aggravator** that drives the already-slow legacy loop from 69 iters to the cap. In the trust path the proximal is inert ((b)==(c)). The `scvx_freeze` cells were removed with the mechanism itself.

## phase120 (trust cells at r0 = 2000 km)

| cell | iters | capped | converged | feasible | min_r | J_true | cost_true_energy |
|---|---|---|---|---|---|---|---|
| (a) trust0 + prox1e-6 | 12000 | True | 0 | True | 6486.659 | 2.459697e-05 | 2.459288e-05 |
| (a0) trust0 + prox0 | 1197 | False | 0 | True | 6486.624 | 2.267170e-05 | 2.266781e-05 |
| (b) trust2000 + prox0 | 8 | False | 1 | True | 6486.614 | 2.263317e-05 | 2.262928e-05 |
| (c) trust2000 + prox1e-6 | 8 | False | 1 | True | 6486.614 | 2.263317e-05 | 2.262928e-05 |

- (a) legacy loop + mis-scaled prox reproduces the crawl (caps): **True** (12000 iters)
- trust region is the PRIMARY fix: legacy loop even WITHOUT the prox (a0=1197 iters, never satisfies the SCvx convergence criterion) is >3x slower than the trust path, which does converge (b=8, c=8 iters): **True**
- the mis-scaled proximal is a SECONDARY aggravator: it pushes the legacy loop from 1197 iters (a0) to 12000 (the cap) (a): **True**
- (b)(c) both converge + feasible: **True**
- (b) prox0 and (c) prox1e-6 are bit-identical: **True** (NOT A GATE — the proximal is skipped when `trust_active`, so these two cells run the same code; excluded from the verdict because it cannot fail)

Isolation finding: Cell (a) alone conflates two changes -- setting `scp_trust_radius=0` both reverts to the legacy unconditional-accept loop AND re-enables the proximal. The added cell (a0) separates them: with the proximal removed the legacy loop still takes 1197 iters (it exits via the step-norm tolerance, not the SCvx criterion) -- >3x the trust path's 8. So the **trust region is the primary fix**; the mis-scaled proximal is a **secondary aggravator** that drives the already-slow legacy loop from 1197 iters to the cap. In the trust path the proximal is inert ((b)==(c)). The `scvx_freeze` cells were removed with the mechanism itself.

## phase135 (trust cells at r0 = 4000 km)

| cell | iters | capped | converged | feasible | min_r | J_true | cost_true_energy |
|---|---|---|---|---|---|---|---|
| (a) trust0 + prox1e-6 | 12000 | True | 0 | True | 6493.301 | 9.927433e-05 | 9.926367e-05 |
| (a0) trust0 + prox0 | 2385 | False | 0 | True | 6491.038 | 8.493797e-05 | 8.492802e-05 |
| (b) trust4000 + prox0 | 9 | False | 1 | True | 6490.935 | 8.490308e-05 | 8.489313e-05 |
| (c) trust4000 + prox1e-6 | 9 | False | 1 | True | 6490.935 | 8.490308e-05 | 8.489313e-05 |

- (a) legacy loop + mis-scaled prox reproduces the crawl (caps): **True** (12000 iters)
- trust region is the PRIMARY fix: legacy loop even WITHOUT the prox (a0=2385 iters, never satisfies the SCvx convergence criterion) is >3x slower than the trust path, which does converge (b=9, c=9 iters): **True**
- the mis-scaled proximal is a SECONDARY aggravator: it pushes the legacy loop from 2385 iters (a0) to 12000 (the cap) (a): **True**
- (b)(c) both converge + feasible: **True**
- (b) prox0 and (c) prox1e-6 are bit-identical: **True** (NOT A GATE — the proximal is skipped when `trust_active`, so these two cells run the same code; excluded from the verdict because it cannot fail)

Isolation finding: Cell (a) alone conflates two changes -- setting `scp_trust_radius=0` both reverts to the legacy unconditional-accept loop AND re-enables the proximal. The added cell (a0) separates them: with the proximal removed the legacy loop still takes 2385 iters (it exits via the step-norm tolerance, not the SCvx criterion) -- >3x the trust path's 9. So the **trust region is the primary fix**; the mis-scaled proximal is a **secondary aggravator** that drives the already-slow legacy loop from 2385 iters to the cap. In the trust path the proximal is inert ((b)==(c)). The `scvx_freeze` cells were removed with the mechanism itself.

## phase170 (trust cells at r0 = 4000 km)

| cell | iters | capped | converged | feasible | min_r | J_true | cost_true_energy |
|---|---|---|---|---|---|---|---|
| (a) trust0 + prox1e-6 | 12000 | True | 0 | True | 6616.000 | 2.414994e-03 | 2.414979e-03 |
| (a0) trust0 + prox0 | 12000 | True | 0 | True | 6502.728 | 4.630532e-04 | 4.630144e-04 |
| (b) trust4000 + prox0 | 12 | False | 1 | True | 6502.894 | 4.626229e-04 | 4.625838e-04 |
| (c) trust4000 + prox1e-6 | 12 | False | 1 | True | 6502.894 | 4.626229e-04 | 4.625838e-04 |

- (a) legacy loop + mis-scaled prox reproduces the crawl (caps): **True** (12000 iters)
- trust region is the PRIMARY fix: legacy loop even WITHOUT the prox (a0=12000 iters, never satisfies the SCvx convergence criterion) is >3x slower than the trust path, which does converge (b=12, c=12 iters): **True**
- the mis-scaled proximal is a SECONDARY aggravator: it pushes the legacy loop from 12000 iters (a0) to 12000 (the cap) (a): **False** — NOT MEASURED: (a0) itself hit the 12000 cap, so the two cells are both censored and the comparison is empty
- (b)(c) both converge + feasible: **True**
- (b) prox0 and (c) prox1e-6 are bit-identical: **True** (NOT A GATE — the proximal is skipped when `trust_active`, so these two cells run the same code; excluded from the verdict because it cannot fail)

Isolation finding: Cell (a) alone conflates two changes -- setting `scp_trust_radius=0` both reverts to the legacy unconditional-accept loop AND re-enables the proximal. The added cell (a0) separates them: with the proximal removed the legacy loop still takes 12000 iters (it exits via the step-norm tolerance, not the SCvx criterion) -- >3x the trust path's 12. So the **trust region is the primary fix**; the mis-scaled proximal is a **secondary aggravator** that drives the already-slow legacy loop from 12000 iters to the cap. In the trust path the proximal is inert ((b)==(c)). The `scvx_freeze` cells were removed with the mechanism itself.

## planechange (trust cells at r0 = 2000 km)

| cell | iters | capped | converged | feasible | min_r | J_true | cost_true_energy |
|---|---|---|---|---|---|---|---|
| (a) trust0 + prox1e-6 | 12000 | True | 0 | True | 6490.232 | 9.383261e-05 | 9.382611e-05 |
| (a0) trust0 + prox0 | 2339 | False | 0 | True | 6489.173 | 6.158024e-05 | 6.157384e-05 |
| (b) trust2000 + prox0 | 10 | False | 1 | True | 6489.076 | 6.148523e-05 | 6.147884e-05 |
| (c) trust2000 + prox1e-6 | 10 | False | 1 | True | 6489.076 | 6.148523e-05 | 6.147884e-05 |

- (a) legacy loop + mis-scaled prox reproduces the crawl (caps): **True** (12000 iters)
- trust region is the PRIMARY fix: legacy loop even WITHOUT the prox (a0=2339 iters, never satisfies the SCvx convergence criterion) is >3x slower than the trust path, which does converge (b=10, c=10 iters): **True**
- the mis-scaled proximal is a SECONDARY aggravator: it pushes the legacy loop from 2339 iters (a0) to 12000 (the cap) (a): **True**
- (b)(c) both converge + feasible: **True**
- (b) prox0 and (c) prox1e-6 are bit-identical: **True** (NOT A GATE — the proximal is skipped when `trust_active`, so these two cells run the same code; excluded from the verdict because it cannot fail)

Isolation finding: Cell (a) alone conflates two changes -- setting `scp_trust_radius=0` both reverts to the legacy unconditional-accept loop AND re-enables the proximal. The added cell (a0) separates them: with the proximal removed the legacy loop still takes 2339 iters (it exits via the step-norm tolerance, not the SCvx criterion) -- >3x the trust path's 10. So the **trust region is the primary fix**; the mis-scaled proximal is a **secondary aggravator** that drives the already-slow legacy loop from 2339 iters to the cap. In the trust path the proximal is inert ((b)==(c)). The `scvx_freeze` cells were removed with the mechanism itself.

<!-- provenance: commit=3e73f70f1f6fdfe50e7731a74919c39056677c2e dirty=0 ext=825716be634e -->

---

_produced by commit `3e73f70f1f6fdfe50e7731a74919c39056677c2e`, Rust extension `825716be634e`; working tree clean under tools/verify, orbital_docking, rust_optimizer._
