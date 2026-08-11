# Codebase Design Freeze — Bézier SCvx Trajectory Initializer

_Authoritative record of locked design decisions and the evidence behind them.
Read this before touching solver code or the paper. Supersedes all prior
handoff docs (`scvx_fix_handoff.md`, `paper_revision_handoff.md` — deleted;
history in git). Terse rules live in `CLAUDE.md`; this file holds the full
design and the evidence. In-flight work lives in `doc/session_handoff.md`._

**NOTATION: `doc/notation.md` is the single source of truth for every
mathematical symbol, and it outranks this file.** Symbols here were aligned to
it on 2026-08-10 (`R_KOZ`, `Δ_k`, `μ`, `∇g_j`, `c_j`, `φ`/`φ⁽ᵏ⁾`, `γ⁽ˢ⁾_m`,
`n_conv`, sub-arc index `s` vs gravity interval `j`). Enforce with
`python3 tools/check_notation.py`.

**PRECEDENCE: the code + the 5-pillar verification are ground truth.** The
paper (`doc/paper_draft_korean_rev2.md`) currently LAGS the code at
§2 roadmap / §2.3 / §3.2 (it still shows the removed smoothness term and the
Gram-for-smoothness story), at §4.1 (수렴 허용오차 still 10⁻⁶; locked value is
10⁻⁸), at §3.3 / Algorithm 1 (no K-consecutive requirement stated), and at
§5.1/§5.3 (10000회-era numbers). It is being rewritten TOWARD the code. Never
"fix" the code to match the paper.

**DEFECT STATUS (2026-08-09):** the four defects found by the 2026-08-08 rigor
review are FIXED and covered by tests — the KOZ-row skip is now a counted,
reported mode (`DegenerateNormal`, `koz_degenerate_segments`); `pred_floor` is
relative; Clarabel tolerances are set explicitly; `AlmostSolved` is counted
separately (`qp_almost_solved`). A fifth, larger defect was found and fixed on
2026-08-09 — the merit graded candidates against rows the subproblem had not
optimized; see §9. The core Gram-integral mathematics was independently
re-derived and is CLEAN.

**OPTIMALITY:** first gated externally on 2026-08-09 by Pillar 5
(`tools/verify/optimality.py`) — KKT stationarity residual + a feasible-descent
search, both computed independently of the solver. All five scenarios PASS. The
exact-penalty condition μ ≥ ‖λ_KOZ‖_∞ is now MEASURED (§5), not estimated.

## 0. Exact problem formulation (locked 2026-08-09)

Decision variable: the control net x = (P₀, …, P_N), P_i ∈ ℝ³, of one
degree-N Bézier curve r(τ) = Σ_i P_i B_{i,N}(τ), τ ∈ [0,1]. Transfer time T
is a fixed input; physical time t = T·τ, physical velocity = r′(τ)/T.

**The problem:**

    minimize    J(x) = ∫₀¹ ‖ r″(τ)/T² − g(r(τ)) ‖² dτ
    subject to  P₀ = r₀,  P_N = r_f                       (position BCs, pinned endpoint rows)
                (N/T)(P₁ − P₀) = v₀                        (velocity BC, optional)
                (N/T)(P_N − P_{N−1}) = v_f                 (velocity BC, optional)
                (N(N−1)/T²)(P₀ − 2P₁ + P₂) = a₀            (acceleration BC, optional)
                (N(N−1)/T²)(P_{N−2} − 2P_{N−1} + P_N) = a_f (acceleration BC, optional)
                ‖ r(τ) − c_KOZ ‖ ≥ R_KOZ   for all τ ∈ [0,1] (keep-out sphere)

- g = true gravity (two-body + J2) — the ONLY non-linear ingredient.
- r″(τ)/T² is physical geometric acceleration, so the integrand is the
  squared control (thrust) acceleration: J is control-effort energy (§1).
- All BC rows are linear equalities in x (verified against
  `constraints.rs:184-266`; endpoints pinned at `optimizer.rs:945`).
- The continuous KOZ constraint is never imposed directly. It is replaced by
  the 명제 1 sufficient condition: De Casteljau subdivision Q⁽ˢ⁾ = S⁽ˢ⁾ x into
  n_seg sub-arcs; sub-arc s is certified iff some unit witness n⁽ˢ⁾ has
  n⁽ˢ⁾·(q − c_KOZ) ≥ R_KOZ for EVERY control point q of Q⁽ˢ⁾. Witness-based and
  existential — see §9.

**Convex subproblem at iterate k (reference x⁽ᵏ⁾, trust radius Δ_k):**

    minimize    φ⁽ᵏ⁾(x) = ∫₀¹ ‖ r″/T² − (∇g_j r + c_j) ‖² dτ + μ·Σ_rows max(0, b_row − a_row·x)
    subject to  BC rows,  ‖x − x⁽ᵏ⁾‖_∞ ≤ Δ_k

- Gravity affine per sub-arc j: ∇g_j, c_j = Jacobian/offset of g at the sub-arc
  centroid of x⁽ᵏ⁾; the integral is EXACT via the Bernstein Gram matrix (§1).
  Note the two subdivisions are DISTINCT: KOZ uses n_seg sub-arcs indexed by s
  (matrices S⁽ˢ⁾), gravity uses n_lin intervals indexed by j (matrices Ŝ⁽ʲ⁾).
- KOZ rows (elastic, L1-penalized by μ): the first-order model about x⁽ᵏ⁾ of
  the clearance γ⁽ˢ⁾_m(x) = n⁽ˢ⁾(x)·(q⁽ˢ⁾_m − c_KOZ) − R_KOZ, with the centroid
  rule's witness n⁽ˢ⁾(x) and ITS rotation included
  (`build_koz_constraints_linearized`). Freezing n at x⁽ᵏ⁾ instead drops a
  first-order term and makes the QP's optimum a point the centroid rule will
  disagree with next iteration — the 2026-08-09 defect (§9).

**Acceptance:**

    ρ = [φ(x⁽ᵏ⁾) − φ(x⁺)] / [φ⁽ᵏ⁾(x⁽ᵏ⁾) − φ⁽ᵏ⁾(x⁺)],  φ(x) = J(x) + μ·h(x)

- φ uses TRUE gravity g; φ⁽ᵏ⁾ uses the affine model. h(x) is the EXACT hull
  violation with the witnesses rebuilt at x (h = 0 ⇔ x carries the 명제 1
  certificate), so ρ measures the two model errors that exist: gravity
  linearization and witness re-aiming. Accept if ρ > η = 0.1.

## 1. Objective (locked 2026-08-07)

Singular, hard-coded — the exact integral of control-acceleration energy:

    J = ∫₀¹ ‖ a_geom(τ)/T² − (∇g_j r(τ) + c_j) ‖² dτ

- Gravity = two-body + J2, affine per De Casteljau segment (∇g_j, c_j at the
  segment centroid); `sample_count` = number of linearization segments.
- Computed in CLOSED FORM per segment via the degree-N Bernstein Gram matrix:
  residual control points f_k are linear in x, ∫ = Σ_{kl} G_{kl} f_k·f_l.
  No sampling anywhere in the objective.
- Why exact: sampled objectives have near-null "soft directions" that fed
  wall-crawl pathologies; and computing curve derivatives/integrals in
  control-point space in closed form is the formulation's raison d'être
  (the compute-time story of the paper).
- NO auxiliary terms, NO modes/flags. The historical full-weight smoothness
  term (1/T⁴)xᵀ(G̃⊗I)x biased the optimum against free-fall arcs; its apparent
  optimization benefit was a premature-convergence artifact (evidence #3).
- Δv / fuel: never an objective, never mentioned in the paper in ANY form
  (not even as a limitation). Exclude `dv_proxy_m_s` from regenerated tables.

## 2. Constraints and the feasibility certificate

- KOZ: per-segment supporting half-spaces imposed on De Casteljau control
  points (`constraints.rs`). All rows satisfied ⇒ the whole continuous curve
  is outside the sphere (paper 명제 1). Conservative inner approximation; the
  gap shrinks as n_seg grows.
- Feasibility is REPORTED via the exact certificate: `final_hull_violation_km`
  (rows rebuilt at the final iterate) = 0 ⇒ certified. The dense min-radius
  sweep (1001 samples) is a diagnostic, not the guarantee.
- Boundary conditions are hard equality rows; the trust region is a hard box
  ‖x−p‖_∞ ≤ Δ_k inside the QP.

## 3. Acceptance (canonical SCvx, Mao et al.)

- The elastic (virtual-control) QP is solved UNCONDITIONALLY on the trust
  path: slack on KOZ rows with L1 penalty μ. The QP therefore minimizes
  exactly the convex merit φ⁽ᵏ⁾(x) = J⁽ᵏ⁾(x) + μ·Σ max(0, b − a·x), so the
  predicted merit reduction is ≥ 0 by construction (given a reference feasible
  for the hard rows — see bootstrap).
- True merit φ(x) = J_true(x) + μ·h(x), where h(x) = hull-row violation with
  rows REBUILT AT x (the certifiability function; h=0 ⇔ 명제 1 certificate).
  Penalizing sphere-distance instead deadlocks the ratio test (conservatism gap
  never closes) — measured, do not revisit. The rows-rebuilt-at-x definition of
  h REQUIRES the subproblem to model the normal re-aim; pairing it with
  frozen-normal rows is the 2026-08-09 defect (§9). The `scvx_freeze` clause
  that used to sit here is void — the mechanism is deleted.
- ρ = actual/predicted merit reduction. Accept if ρ > η = 0.1; grow trust ×2
  if ρ > 0.9 (finite ρ only); shrink ×0.5 on rejection.
- Affine bootstrap: if the reference violates any hard (non-KOZ) row by >1e-9
  — true at iteration 1, where the straight-line init violates the velocity-BC
  equalities — the merit comparison is meaningless; the repairing candidate is
  accepted unconditionally (ρ recorded as NaN).
- pred_floor = 1e-12·|φ⁽ᵏ⁾(x⁽ᵏ⁾)|: pred below it is a null step, accepted iff
  non-worsening. Relative, not `1+|φ⁽ᵏ⁾(x⁽ᵏ⁾)|` — the old form was absolute in
  practice (|φ⁽ᵏ⁾(x⁽ᵏ⁾)| ~ 2e-5) and so depended on the choice of units.
- Termination is NOT this floor. Two principled exits, both n_conv = 3
  consecutive: the merit streak (stop_reason 1) and model stationarity,
  pred < tol_f·|φ(x⁽ᵏ⁾)|
  with the certificate held (stop_reason 4). stop_reason 0/2/3 (cap, trust
  collapse, QP failure) all mean the loop gave up and are gated as failures.

## 4. Convergence (locked 2026-08-07)

- Converged ⇔ relative merit change |Δφ|/|φ(x⁽ᵏ⁾)| < tol_f on n_conv = 3 CONSECUTIVE
  accepted steps AND the accepted iterate carries the certificate
  (h ≤ 1e-6 km aggregate). tol_f = max(tol, 1e-8); defaults tol = 1e-8
  (Python signature and verify harness).
- Second criterion — trust-radius collapse. **DECISION 2026-08-11 (author):
  gate this on CRITICALITY, not feasibility.** Reference for the whole
  trust-region stopping question, and the book this project defers to on it:

  > Conn, A. R., Gould, N. I. M., and Toint, Ph. L., *Trust-Region Methods*,
  > MOS-SIAM Series on Optimization, SIAM, Philadelphia, 2000.
  > doi:10.1137/1.9780898719857 · ISBN 978-0-89871-460-9 · xix+942 pp.
  > Ch. 6 (Global Convergence of the Basic Algorithm) covers collapse as a
  > stopping rule; Ch. 12 (Projection Methods for Convex Constraints) is the
  > relevant chapter for the criticality measure in our setting, since the
  > subproblem's constraints — half-spaces, the ∞-norm box, the BC equalities —
  > are all convex. Chapter titles verified against the SIAM listing 2026-08-11;
  > SUBSECTION numbers have NOT been checked, so do not cite them.

  Current code (`optimizer.rs:1468-1476`) sets
  `converged_scvx = vlin_p <= 1e-6 && hard_viol_p <= 1e-9` — a FEASIBILITY
  test. Feasible ≠ optimal, so it cannot separate a converged point from a
  deadlocked one. That is why the pillars were moved off `scvx_converged` onto
  `scvx_stop_reason`; the switch hid the symptom without fixing the test.

  **IMPLEMENTED 2026-08-11**, and NOT in the trust-collapse branch — measurement
  (§7 entry 9) showed every run reaching collapse on phase120 was already
  critical (pred/|φ| ≤ 1.2e-9 against tol_f = 1e-8). One guard was blocking the
  stationarity exit: it required `pred >= 0.0`, so a numerically-zero pred of
  −9.1e-15 failed it and reset the streak permanently. The test now compares
  **|pred| < tol_f·|φ|**. This widens the accepting set by exactly
  −tol_f·|φ| < pred < 0 — only where the sign is below the merit's own
  resolution; a pred negative by more than tol_f·|φ| is a genuine model defect
  and still resets the streak. The certificate guard (`vlin_p <= 1e-6`) is
  unchanged in the code — but note it is **UNTESTED**: an adversarial review
  (2026-08-11) rebuilt with that clause deleted and every test in
  `tests/regression/test_stationarity_exit.py` stayed green. The n_seg=2 case
  that was believed to probe it is gated entirely by pred magnitude instead
  (min |pred|/|φ| = 1.082e-05 over 1000 iterations, against tol_f = 1e-8; the
  certificate holds on 30 of those iterations and the two conditions co-occur on
  ZERO). Closing that gap needs a configuration that is model-stationary AND
  uncertified at once; none is currently known. Do not cite the guard as
  covered.

  `pred_floor = 1e-12·|φ|` was left alone deliberately. It also sits below the
  merit's resolution (~1e-14 absolute, 4e-10 relative), but it governs which
  candidate becomes the next reference, so changing it would change ANSWERS. The
  |pred| fix changes only where the loop stops. Revisit only with evidence.

  Do NOT "fix" this by evaluating pred at the collapsed radius — a tiny box
  always predicts tiny improvement, so that test cannot fail and would be
  worthless evidence. The criticality comparison must be against |φ|.

  Regression guard: `tests/regression/test_stationarity_exit.py`. No verification
  pillar covers N=8/n_seg=16 or N=7/n_seg=4, so a revert would otherwise pass the
  whole suite. Verified 2026-08-11 that reverting the comparison and rebuilding
  turns both assertions red.
- Why K-consecutive: a one-shot test fires during the slow crawl along
  re-aimed KOZ walls — measured stopping 33% above the optimum at tol 1e-6
  (evidence #3/#4). Step-norm tests are invalid here: the walls re-aim every
  iteration, so the iterate jitters along a moving constraint.

## 5. Parameters (each with its selection principle)

| parameter | value | principle |
|---|---|---|
| μ (`elastic_weight`) | 1e-2 | exact-penalty rule: above the KOZ dual scale, far below objective-swamping — μ=1e4 measurably degraded optima 4–5× via penalty noise in ρ (evidence #2). **‖λ_KOZ‖_∞ MEASURED: max 1.895e-7 across five scenarios (2026-08-11, with the corrected KOZ rows; the 2026-08-09 figure of 1.5e-7 came from the frozen-normal rows now known to be wrong), so μ=1e-2 clears it by 5 orders.** Previously recorded as "estimated ~1e-6" and never measured. |
| r₀ (`scp_trust_radius`) | 2000 km | must exceed the iteration-1 BC-repair distance (~1650 km in the demo); r₀ ≤ 1000 fails at iteration 1 (known open item) |
| η | 0.1 | textbook SCvx acceptance threshold [Mao et al.] |
| grow / shrink | ×2 @ ρ>0.9 / ×0.5 | textbook trust-region schedule |
| trust_min | 1e-2 km | collapse floor = second stopping criterion |
| tol | 1e-8 | below the wall-crawl plateau (~3e-7 relative steps), above QP solver noise |
| K | 3 | consecutive-streak guard against plateau false convergence |

## 6. Non-canonical machinery still present (status, not endorsement)

- `scvx_freeze`: **DELETED** 2026-08-09 (commit `9c673b0`). No literature
  basis, measurably worse on objective and wall time where the KOZ binds
  (evidence #5), and a permanent config-provenance hazard.
- `freeze_gravity_jacobian`/`freeze_after_iter`, `scp_prox_weight`, and the
  legacy fixed-point path (`scp_trust_radius=0`): legacy knobs, unused by the
  paper configuration. These two freeze knobs are STILL LIVE — signature
  `optimization.py:373-374`, forwarded at `:476-477`, declared in
  `rust_optimizer/pybind/src/lib.rs:25-26`. Only `scvx_freeze` was deleted;
  earlier wording here wrongly called them deleted. The proximal is skipped
  entirely when `trust_active`, so the pillar-2 prox cell compares identical
  code paths — documentation of intent, NOT evidence, and excluded from that
  pillar's verdict.
- The nine `tools/probe_*.py` freeze-era investigation scripts were DELETED
  2026-08-10: they passed the `objective_mode` argument removed in `1581e54`,
  so they raised on use, and they probed `scvx_freeze`, itself since deleted.
  History in git.

## 7. Evidence log (what killed what)

1. **2026-08-05** — two-phase acceptance → canonical merit. Three adversarial
   agents + literature survey: the scale-mismatch justification was
   self-contradicted by μ; pred<0 hole confirmed; real precedent family is
   phase-I/II, CCP, and 2024–26 two-phase SCP — not filter methods.
2. **2026-08-05** — μ grid: 1e4 → +460% optimality gap; 1e-2 → +3.4%;
   robustness 12/12 configs. Exact-penalty rule adopted.
3. **2026-08-06** — smoothness term removed → apparent 33% degradation → basin
   forensics: a restart FROM the "second minimum" escaped it; tol sweep
   (1e-6 → 1e-8) reached the same optimum from every config. The "basin" was a
   premature stop; the term buys nothing. Deleted for good.
4. **2026-08-07** — K=3 + tol 1e-8 adopted; trust-radius sweep showed the
   r₀≤1000 iteration-1 failure and that basin selection via r₀ is unreliable.
5. **2026-08-07** — exact-integral (Gram) objective replaced sampling:
   internal cost vs dense true energy now 0.04–0.9% (pure gravity
   linearization); freeze A/B (12 configs) → freeze harmful where KOZ binds.
6. **Verification state**: all 5 pillars report PASS — ρ ∈ [0.985, 1.001],
   sweep 24/24 fine-mesh, golden refreshed. **CAVEAT (2026-08-08 review):** the
   quoted gap "6.5% (n_seg=16) → 1.3% (n_seg=64)" was measured with the freeze
   ON, because pillars 1/3/4b never got the freeze-off switch. The freeze-off
   baseline is ≈4.2% at n_seg=16. Re-run before quoting any of these numbers —
   see `session_handoff.md` step 2. The `artifacts/verify/` tree is gitignored
   and carries no producing-commit provenance.

7. **2026-08-09** — verification harness defects found and fixed while building
   Pillar 5: (a) `J_true`, the harness's canonical cross-solver objective, was a
   uniform-mean Riemann sum converging only as O(1/n) — ~1.6e-3 relative error at
   its n_dense=1200 default, LARGER than the A/B gap it was being used to resolve.
   It manufactured 562 phantom descent directions on phase70 whose best step
   reversed sign (−1.4e-5 → +5.0e-5) once the quadrature converged. Replaced with
   Gauss-Legendre (exact to f64 at 24 nodes); it now agrees with the Rust exact-Gram
   integral to 1.8e-6 relative, the cross-implementation check the objective never
   had. (b) Pillar 4a gained `best_ok` — the returned iterate must be the best
   visited. rho and per-phase monotonicity CANNOT catch a run that drifts off its
   own optimum, because both compare only within one iteration while the merit
   itself changes between them; variant (B) drifted 0.178% with rho = 1.000 on all
   139 steps and every gate green. (c) Pillar 2's `prox_inert` gate compared two
   bit-identical code paths and was removed from the verdict; its remaining cells
   now gate on `scvx_stop_reason`, not the weak `scvx_converged` flag.
8. **2026-08-09** — scenario grid widened from 2 to 5: phase135, phase170 (nearly
   antipodal; its straight-line init passes 5420 km INSIDE the KOZ, so it needs
   r0=4000 — r0 is now a per-scenario field, not a global constant) and
   `planechange` (23.9° orbit-plane difference, so the two velocity BCs differ in
   direction, not just phase). All five pass Pillars 1–5 with (A).

9. **2026-08-11** — the trust-collapse test is wrong in BOTH directions, measured
   on phase120 with Pillar 5's solver-blind KKT check. Two configurations stop on
   collapse (stop_reason 2) at an identical final radius of 0.00763 km:
   **N=8, n_seg=16** has KKT relative residual **1.229e-02** — *better* than the
   clean-exit control (N=7, n_seg=16, stop 1) at 1.420e-02 — and 0/604 descent
   hits. It is at an optimum and is labelled a failure. **N=7, n_seg=4** has KKT
   residual **3.723e-01**, seven times the 5e-2 gate.

   **The first reading of that pair — "n_seg=4 is genuinely stuck" — was WRONG,
   and was refuted the same day.** Restarting each collapsed run from its own
   final point with a fresh 2000 km radius moved neither: max control-point
   motion 0.0000 km, objective unchanged to all digits, both collapsing again in
   exactly 18 halvings (2000·2⁻¹⁸ = 0.0076 km, i.e. every step rejected). The
   per-iteration trace then gave the criticality number directly: **pred/|φ| is
   4.0e-10 … 1.2e-9 for BOTH collapsed runs**, against tol_f = 1e-8. Both are
   critical. So is the clean-exit control (stop 1) at 4.3e-14 … 1.6e-9.

   The real defect is numerical, and it is twofold (`optimizer.rs`):
   (a) the stationarity test requires `pred >= 0.0`, so a pred of −9.1e-15 —
   numerically zero — fails it and resets `stat_streak` forever; and
   (b) `pred_floor = 1e-12·|φ| ≈ 2.3e-17` sits ~400× BELOW the merit's actual
   resolution (~1e-14 absolute, 4e-10 relative), so a noise-level negative `act`
   is read as genuine worsening, the null step is rejected instead of accepted,
   and the radius halves to the floor. The clean run differs only in that its
   null steps land on the accept side of that same threshold.

   This does not overturn the §4 decision — gating on criticality is still the
   right rule, and it is now better supported, since the criticality measure
   separates cleanly (all three critical) where radius and feasibility do not.
   It does change the fix: the change is to the stationarity test and the noise
   floor, not to the trust-collapse branch.

   **Pillar 5 blind spot — CONFIRMED and FIXED 2026-08-11.**
   `tools/verify/optimality.py:koz_rows` omitted the witness-rotation term, on
   the stated ground that it is second order at an active point. That ground is
   false wherever a sub-arc has appreciable lateral reach, which is every mesh
   the paper reports. Measured on phase120, residual with the term vs without:

   | n_seg | with | without (shipped) | inflation |
   |---|---|---|---|
   | 4 | 2.835e-06 | 3.723e-01 | 131000× |
   | 8 | 1.044e-05 | 3.127e-02 | 2995× |
   | 16 | 1.131e-05 | 1.420e-02 | 1255× |
   | 32 | 1.166e-05 | 6.933e-03 | 595× |
   | 64 | 1.168e-05 | 3.442e-03 | 295× |

   With the term the residual is FLAT across the mesh — the signature of a
   genuine KKT point. Without it the residual tracked mesh coarseness, i.e. the
   test was measuring its own approximation. **The n_seg=4 "3.723e-01, NOT
   OPTIMAL" verdict recorded above was therefore an artifact: corrected, that
   point has the LOWEST residual of the five.** The rows that "passed" passed
   for a reason unrelated to what the test claims to measure.

   Two consequences, both adopted: `KKT_TOL` tightened 5e-2 → **1e-3** (88×
   margin on the worst of 1.131e-05; the old gate could not have failed on the
   broken rows, the new one does — verified by reverting the term and watching
   phase120 go FAIL at 1.420e-02). And `ACTIVE_TOL` swept 1e-5…1e0 on all five
   scenarios: the residual is IDENTICAL over 1e-5…1e-1, so the verdict does not
   depend on that threshold — the open sensitivity question is closed.

   (Independently: n_seg=4's 0 descent hits rest on only 88 certified trials
   versus 604 and 644, because its feasible set is small enough that most random
   directions leave it.)

   **Outcome after the |pred| fix (same day).** Every trust-collapse row
   converted to model stationarity: N=8/n_seg=16 stop 2 @ 27 iters → stop 4 @ 8;
   N=7/n_seg=4 stop 2 @ 30 → stop 4 @ 12. Iteration counts fell across the board
   (n_seg=16: 20 → 8, n_seg=32: 21 → 8, n_seg=64: 13 → 8) because the loop no
   longer spends its tail rejecting noise-level null steps. n_seg=2 still runs to
   the cap. All 6 pillars PASS, 24 Rust tests, Python suite green.

   **Correction (adversarial review, same day).** The first write-up of this
   entry said "no objective moved by a single digit". That was measured on
   phase120 only and is FALSE as stated. A 5-scenario × 8-config pre/post sweep
   found **0 runs worse and 0 costs moved by more than 1e-9 relative**, but seven
   configurations DID move, by ~1e-12 relative — e.g. phase135/N=8/n_seg=16 at
   1.02e-12, planechange/N=7/n_seg=64 at 5.70e-12. Mechanism: those configs
   flipped stop 1 → stop 4 at the same iteration count, and the stop-4 `break`
   fires BEFORE `p = x_new`, so it returns the REFERENCE where stop 1 returned
   the accepted candidate. Correct claim: identical to ~12 significant digits,
   four orders inside tol_f. The fix changes what is returned, negligibly; it
   does not change what the loop converges to.

10. **2026-08-11** — `harness_common.J_true_and_grad` was still the uniform-mean
    Riemann sum that was removed from `J_true` on 2026-08-09. The fix never
    reached the gradient twin, and nothing compared them, so it survived two days
    behind a green suite. Measured relative error at the `n_dense=600` its caller
    used: 0.196% (n_seg=8) … **0.207% (n_seg=64)**.

    It feeds **Pillar 1** (`nlp_crosscheck.py`, the objective AND gradient of the
    independent scipy NLP) and **Pillar 3** (`kkt_check.py`). Pillar 1 therefore
    optimized one function and was scored with another (`J_true`, corrected), and
    its `res_cold.optimality` certified a stationary point of the wrong function.
    Worse, **the error EXCEEDED the gap being reported**: 0.207% against a
    n_seg=64 gap of 0.165%. That is the identical defect shape as evidence #7(a)
    — an oracle asked to resolve a difference finer than itself.

    Fixed to Gauss-Legendre on the same nodes as `J_true`. The two now agree to
    1.2e-15 relative, and the analytic gradient matches central differences at
    **2.5e-09**, from 4.147e-03 before — a disagreement that was INDEPENDENT of
    the finite-difference step, which is the signature of two different
    integrands rather than differencing noise.

    Corrected Pillar 1 gaps (phase120, cold start, Rust-blind): **12.570%
    (n_seg=8), 2.882% (16), 0.717% (32), 0.175% (64)** — the values barely moved,
    but they are now measurements rather than noise, and the NLP reaches
    `optimality=9.37e-09` with cold and warm starts agreeing to 0.000%.

    `tests/regression/test_harness_oracle.py` now asserts the two oracles agree,
    the gradient matches finite differences, that check can itself fail, and the
    quadrature is node-count invariant (a Riemann sum cannot be). Proven able to
    fail by reverting the quadrature in place.

10. **2026-08-11 — adversarial review of the |pred| fix.** An independent agent was
    told to REFUTE seven claims about the change, with "nothing found" declared an
    acceptable answer. Four core claims survived; two of the author's own
    statements were refuted. **Read this before re-deriving any of it.**

    **REFUTED — `test_uncertified_run_still_cannot_claim_stationarity` did not test
    what its docstring claimed.** Rebuilding with `&& vlin_p <= 1e-6` DELETED from
    `optimizer.rs` left all three tests in
    `tests/regression/test_stationarity_exit.py` GREEN. Reproduce the proof without
    rebuilding anything:

    ```
    SCVX_TRACE=1 .venv/bin/python -c "
    import sys; sys.path.insert(0,'.')
    import warnings; warnings.filterwarnings('ignore')
    from tools.verify import harness_common as H
    H.run_rust(H.make_scenario('phase120', N=7), n_seg=2)" 2>&1 | grep '^SCVXTRACE' \
      | awk -F, '$2 ~ /^[0-9]+$/ {p=($5<0?-$5:$5); t=($9<0?-$9:$9); v=$15+0;
          if(t>0){r=p/t; n++; if(n==1||r<m)m=r; if(r<1e-8)b++; if(v<=1e-6)c++;
          if(r<1e-8&&v<=1e-6)both++}}
        END{printf "iters=%d min|pred|/|phi|=%.3e band=%d cert=%d BOTH=%d\n",n,m,b+0,c+0,both+0}'
    ```

    Gives `iters=1000 min|pred|/|phi|=1.082e-05 band=0 cert=30 BOTH=0`. The pred
    band never opens on n_seg=2 (three orders above tol_f = 1e-8), so the
    certificate guard is never the binding condition there and a regression in it
    passes unnoticed. **The certificate guard on the stationarity exit is UNTESTED.**
    Closing that needs a configuration simultaneously model-stationary AND
    uncertified; none is known. Test renamed and its docstring corrected to claim
    only what it checks. Tests 1 and 2 ARE falsifiable — both go red on the pre-fix
    binary, re-confirmed independently.

    **REFUTED — "no objective moved by a single digit" was phase120-only.** See the
    correction embedded in entry 9. Correct claim: identical to ~12 significant
    digits, 0 runs worse, 0 costs moved >1e-9 relative across 5 scenarios × 8
    configs.

    **SURVIVED, with the evidence, so nobody re-runs these:**
    - *pred >= 0 is exact, negatives are pure noise.* Verified structurally, not by
      plausibility: `l_p`/`l_c` use the same `quad_form(&h_obj, &f_obj, ·)` and the
      same `koz_row_violation` over the same rows, so the dropped constant cancels;
      the proximal is gated `!trust_active` so `h_mat == h_obj`; `obj_scale` divides
      H, f AND the elastic weight identically so `argmin` is unchanged; trust rows
      are `[p−Δ, p+Δ]`, an ∞-norm box exactly centred at p, so `d = 0` is always
      feasible. Max `hard_viol_p` among negative-pred iterations: **6.66e-15**.
    - *The accepted band is wide enough but not too wide.* 571 non-bootstrap
      iterations, 35 configs: max |negative pred| **5.1232e-14**, min band
      `tol_f·|φ|` **2.2038e-13**, ratio **4.30×**, negatives falling outside the
      band **0 of 110**. The author's worry that the band is "10–20× wider than the
      noise" was misconceived: since the true pred is ≥ 0, an observed negative
      inside the band implies the true value is inside it, provided noise < band.
      Width is not the criterion; only noise < band is.
    - *No premature stop.* Impossibility test — restart every run from its own final
      point at a fresh FULL trust radius (more freedom cannot help a true optimum).
      35 configs × 5 scenarios: **0 improved**, max relative gain 1.34e-10.

    **NEW SOFT SPOTS, all pre-existing, none introduced by the fix:**
    - **`pred_floor` is ~114× below the merit's noise floor** (median 4.50e-16 vs
      measured 5.12e-14). Over 571 iterations, **56 had pred inside the noise and 27
      of those were REJECTED**, halving the trust region on a ratio of one noise
      number by another. The |pred| fix escapes the *symptom* by exiting before that
      cascade collapses the radius; it does not remove the mechanism. Largest
      remaining soft spot. Changing it would change ANSWERS (§4) — do not touch
      without a measurement campaign.
    - **`n_conv = 3` is weaker than §4 implies.** On the reject path the reference
      does not move and the box only shrinks, so pred is monotonically
      non-increasing — three consecutive qualifying iterations are NOT three
      independent observations. Applies equally to the old `pred >= 0` form.
    - **phase170/N=7/n_seg=4 fires stop 4 at trust = 0.0305 km** (16 halvings), the
      near-collapsed-box hazard §4 warns about. Checked directly: re-evaluated at
      the full 4000 km radius, pred = 1.59e-12 and 7.09e-13 against a band of
      1.32e-11 — 8× inside, genuinely critical. Fires pre-fix too.
    - **`SolverStatus::AlmostSolved` increments `qp_almost_solved` but returns the
      solution as if exact**; nothing in the loop gates on it, so a degraded QP does
      feed pred. Measured 0 occurrences across all 75 sweep runs. The protection the
      old `pred >= 0` guard incidentally gave covered only QP errors smaller than
      the convergence tolerance, which by definition cannot change the answer.

    **Method notes worth keeping.** (a) The reviewer's wheel-swap protocol was
    silently broken — `pip install /tmp/pre_fix.whl` fails on non-PEP-427 filenames
    and the notice scrolls past in `tail`, so its first "pre-fix" measurement was
    actually the post-fix binary. It recovered by asserting the installed `.so`
    SHA-1 before every measurement. **Any A/B of two solver builds must verify the
    installed binary's hash, not assume the install took.** (b) It declined to run
    `tools/verify/*.py` because they overwrite `artifacts/verify/`, which is
    gitignored and holds the only recorded 6/6 PASS evidence — an interrupted run
    would destroy it unrecoverably. That is the provenance problem in §7's caveats,
    biting in practice. **Resolved 2026-08-11 — see entry 11.**

11. **2026-08-11 — the pillars only ever ran one geometry, and extending them
    exposed five blind spots.** Four of six pillars ran phase120 alone, the case
    they were debugged against. A check that runs only where it was calibrated
    reports that that case still works, not that the solver does. All five
    defects below are invisible on phase120 and were found by running the
    existing `_SCENARIOS` set.

    **(a) Artifacts were not evidence.** `artifacts/verify/` was gitignored and
    carried no producing commit, so no pillar number was traceable from a clean
    checkout, and the six summaries silently mixed 29 Jul with 9 Aug results into
    one PASS. Now committed, and every file carries commit + dirty-flag + the
    sha256 of the compiled Rust extension — the `.so` does not rebuild when
    `rust_optimizer/` changes, so a clean source tree can still be running a
    stale binary. `verdict.py` refuses an overall PASS across disagreeing stamps
    (INCONCLUSIVE); run against the old unstamped artifacts it fires, exiting 1
    where it previously exited 0 on the same inputs.

    **(b) Pillar 2's ablation was CENSORED, not passing.** Its 2000-iteration cap
    sat below what the legacy loop needs on the harder geometries, so cells (a)
    and (a0) both read exactly 2000 and `a > a0` was false — recorded as "the
    proximal does not aggravate". Lift the cap and (a0) terminates at 69 / 1197 /
    2385 / 2339 iterations on phase70 / phase120 / phase135 / planechange while
    (a) still runs past 12000: **the proximal aggravates by 11.5×–265×**. The
    gate now requires (a0) to terminate; two cells pinned to one ceiling report
    NOT MEASURED. phase170 remains NOT MEASURED at CAP=12000.

    **(c) phase135 at N=6 could not take a single step** — first QP failed
    (stop_reason 3, iterations 1), iterate still 3910 km inside the KOZ, at every
    mesh. The iteration-1 BC-repair distance depends on the DEGREE as well as the
    geometry: fewer control points means each must move further. r0=4000 fixes
    it. Threshold, not tuning — the answer does not move above it: r0 ∈ {4000,
    8000} × N ∈ {6,7,8} gives clearance 19.811 / 19.935 / 19.910 km and J
    8.497359e-05 / 8.490308e-05 / 8.486337e-05, identical in every digit and
    identical at N=7,8 to what r0=2000 already produced. Entry 8's "r0 is a
    per-scenario field" is therefore incomplete: it is per (scenario, degree),
    and 4000 covers the whole degree range for both phase135 and phase170.

    **(d) Pillar 4b gated on `scvx_converged`**, which the trust-collapse exit
    also sets whenever the iterate happens to be feasible. Recording
    `scvx_stop_reason` beside it shows the two disagree on phase70/n_seg=2 at all
    three degrees (stop=2, flag=1) — coarse cells that were never gated, but the
    same disagreement on a fine-mesh cell would have passed silently. Fine-mesh
    now gates on stop_reason, as Pillars 2 and 4a already did.

    **(e) Pillar 1's ORACLE was the limiting factor, not the solver.** The first
    five-geometry run reported FAIL on phase135, phase170 and planechange, and on
    phase170 the gap read **−6.2%** — Rust apparently *below* the true optimum,
    which would mean it violates the KOZ. It does not. All three failures share
    one cause: the independent SciPy solve never reached a first-order point. The
    straight-line start passes 3120–5890 km INSIDE the keep-out sphere, and
    trust-constr spent its whole budget restoring feasibility.

    Two fixes, both measured. First, `cold_start` pushes the interior control
    points radially out to the KOZ surface — still Rust-blind, since it uses only
    the KOZ radius and the initial guess. phase170 goes from optimality 1.5e-02
    to **9.73e-09 in 597 iterations**, landing on J = 4.493266e-04, which is the
    Rust-warm solve's value **to all seven digits**; two starts of opposite
    provenance agreeing is what makes that the optimum rather than a basin.
    Against it Rust at n_seg=64 is **+0.18%**, the expected sign. phase120 (310
    iters) and planechange (459) likewise converge, phase120 to exactly the J the
    old start reported. Second, convergence is now tested on the KKT residual and
    constraint violation rather than on trust-constr's exit code: planechange had
    returned `optimality=1.15e-07` with `status=0` and was recorded as a
    non-reference for an exit code rather than for a number.

    **phase135 has NO REFERENCE and this is not a budget problem.** Both starts,
    given 6000 iterations, return exactly the J they already had at 800
    (projected 8.319355e-05, straight-line 8.307425e-05) with optimality flat at
    2.1e-03 and 9.5e-04. Stalled, not starved. `MAXITER=1500` therefore stands at
    2.5× the worst converging case rather than chasing it. A geometry without a
    converged reference now reports **NO REFERENCE**, not FAIL: an oracle that
    cannot converge says nothing about the code under test, and scoring it as a
    failure would attribute the oracle's limits to the solver. It is not a pass
    either — the summary carries an explicit coverage line. There is deliberately
    no fallback to the warm solve, which is seeded from the solver under test and
    could never be independent. Pillar 2 uses the same three-state rule for the
    same reason (b).

    **Unchanged by all of this:** Pillars 3 and 4a pass on all five geometries —
    15/15 cells primally feasible, all five traces stopping on model stationarity
    with zero merit drift.

    **phase70's KOZ never binds**, and that is worth stating because its numbers
    otherwise look like a spectacular safety margin. Its minimum radius is at
    τ = 0 exactly — the departure point — so its 145.00 km "clearance" is the
    Progress orbit's altitude above the KOZ, not clearance the method produced.
    Three independent confirmations: τ* = 0.0000 on a 20001-point grid; the
    solution is bit-identical across n_seg ∈ {8,16,32}, which is impossible if a
    KOZ constraint were active; and its equality-nullspace gradient ratio is
    1.6e-07 against 0.24–0.42 on the other four, because the projection only
    omits an active set when there is one. The same is true of phase120 at
    n_seg ∈ {2,4}: both τ* = 0, both 145.00 km, which is why those two rows of
    the paper's mesh table are not a conservatism measurement.

### Caveats on this evidence log (2026-08-08 adversarial review)

Entries #2–#5 were measured in scratchpad scripts whose outputs were not
persisted to `artifacts/`, and #2's numbers predate the exact-integral
objective. The conclusions they support (μ scale, smoothness removal, freeze
harmfulness) are believed sound and were each reproduced at the time, but the
numbers are not currently reproducible from a clean checkout. Re-derive before
citing any of them in the paper.

## 8. Paper terminology

`~/.claude/skills/korean-prose/references/korean_writing_case_collection.md` §6
판정표 is the vocabulary authority; the `korean-prose` skill is mandatory for
Korean output. Both now live in the dotfiles repo (`claude/skills/korean-prose/`)
and are symlinked into every project — this repo no longer carries a copy. "merit function"
stays in ENGLISH (확대 목적함수 rejected — a single printed occurrence in an
adjacent field is not precedent; that rule is now general).

## 9. KOZ constraint — exact formulation (settled 2026-08-09)

Settled during the ρ-test investigation. This is the FORMULATION of record;
it supersedes §3's rows-REBUILT-AT-x definition of h. The matching code
change is pending explicit approval and is NOT yet in the tree.

- True requirement: the continuous curve stays outside the KOZ sphere.
  Non-convex.
- 명제 1 is WITNESS-based: one unit normal per segment, with every subdivided
  control point beyond the supporting half-space, certifies the whole segment.
  ANY unit normal is a valid witness — a single satisfied row is already a
  distance guarantee on its own (unit n ⇒ n·(q−c) ≤ ‖q−c‖, so a satisfied row
  gives ‖q−c‖ ≥ R_KOZ). The normal choice affects conservatism, never validity.
  Verified against the as-built rows (`constraints.rs:125-144`; adversarial
  check 2026-08-09: SURVIVES).
- The centroid rule is the heuristic that PICKS the witness each iteration.
  It is scaffolding, not part of the constraint. With the normal fixed, the
  rows are LINEAR in the control points: the supporting half-space IS the
  convexification, and there is no separate "correct KOZ model" for ρ to be
  measured against. The only genuine model error in the subproblem is gravity
  linearization. The paper already presents it this way: §3.1 and
  `figures/f1_koz_linearization.py` show one SCvx iteration as normals frozen
  at the reference.

### Defect (measured 2026-08-09)

The merit graded candidates on rows rebuilt AT THE CANDIDATE while the QP
optimized rows built at the reference — optimizing one quantity, grading
another. phase120/n_seg=16: 24 rejected steps, trust-collapse stop
(stop_reason 2), 0.06–2.5% objective left on the table vs a mismatch-free
run (three variants). Two cures exist; both remove the mismatch, so that
measurement does NOT discriminate between them:

- **(A) — RESOLUTION OF RECORD.** Add the normal-rotation term to the
  subproblem's rows (`build_koz_constraints_linearized`), so the QP's optimum
  is a point the centroid rule still agrees with after it re-aims. The merit
  keeps grading h at the candidate, which is what makes the convergence test
  assert the certificate actually reported.
- (B) — measured and REJECTED. Grade the candidate on the same rows the QP
  optimized (vtrue_c := vlin_c) and delete (A). Preserved in commit `12b5b06`
  for reproducibility.

### Why (A), on measurement (2026-08-09, phase120, trust=2000, μ=1e-2)

| | (A) | (B) |
|---|---|---|
| n_seg=16 | **20 iters**, stop 1 | 172 iters, stop 2 (gave up) |
| n_seg=16 objective | **2.263313e-05** | 2.267205e-05 (+0.172%) |
| n_seg=32 | **21 iters**, stop 4 | 261 iters, stop 2 |
| n_seg=64 | — | 445 iters, stop 2 |

Objective values above are the Rust exact-Gram internal merit, deliberately NOT
the harness `J_true`: the A/B gap (1.7e-3) is the same size as the quadrature
error the old `J_true` carried, so it could not resolve this comparison. That
oracle bug was found and fixed the same day (§7 entry 8); the corrected
Gauss-Legendre `J_true` now agrees with the Rust integral to 1.8e-6 relative,
which is the cross-implementation check the objective never previously had.

(B) removes the mismatch but replaces a POISONED ratio test with a BLIND one:
the KOZ term cancels from act−pred, so ρ reads 1.000 on all 139 accepted steps
of a run that walks past its own best merit (step 26) and ratchets 0.178%
above it before collapsing. The re-aim is physically real and something must
account for it; (A) accounts for it in the constraint, which is the right
place. **(A)'s Jacobian is NOT a physical model** — the original framing, and
the reason the fix looked like a hack — it is a self-consistency device for
the step. That reframing is what (B) bought us.

Measured geometry of the phantom violation (iteration 6, n_seg=16, segment 7;
reproduce with `SCVX_TRACE=1` and the `hown_c`/`cpviol_c` columns): the
accepted iterate satisfies its own rows EXACTLY (0.0) yet misses the rows
rebuilt at itself by 68 m, while clearing the KOZ sphere by 16.0 km. The 68 m
is pure pivot — 455 km of lateral reach along the plane × a 31 arcsec normal
rotation. It cannot be meshed away: 451 m / 68 m / 6.3 m / 0.8 m at
n_seg = 8/16/32/64, still 800× above the 1e-6 km certificate gate at n_seg=64.

Guards adopted with (A): the stationarity exit (stop 4) is K-consecutive like
the merit streak — cost ~14 iterations on phase120/n_seg=16, removes the
one-shot premature-stop failure mode (§4). The (B)-era own-rows convergence
gate is REDUNDANT under (A) and was removed: vtrue_c is already measured
against rows rebuilt at the candidate.

Canonicity note for the paper: (A) is canonical SCvx applied to h — the
subproblem carries h's true first-order model and the merit evaluates h.
§3's sphere-distance rejection stands (a third pairing; still wrong).
**Paper conflict to resolve:** §3.1 and `figures/f1_koz_linearization.py`
present one SCvx iteration as normals FROZEN at the reference, which describes
(B), not the shipped subproblem. The figure is a correct picture of the
*certificate*; it is not a picture of the *subproblem rows*. Reconcile before
publication.

### Conservatism of 명제 1 (measured 2026-08-09) — paper-relevant

A single plane per segment against a round sphere forces the trajectory
further out than the constraint requires. Measured minimum clearance above
R_KOZ at the converged iterate, phase120: 63.9 km (n_seg=8), 15.6 km (16),
3.9 km (32), 0.97 km (64) — falling as ~1/n_seg², exactly the tangent-plane
geometry (lateral reach L per segment ⇒ over-clearance ≈ L²/2R_KOZ). This is
the real cost of the sufficient condition and belongs in the paper as a
quantitative characterization, not a footnote.
