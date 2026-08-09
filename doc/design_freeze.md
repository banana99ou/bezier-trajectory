# Codebase Design Freeze — Bézier SCvx Trajectory Initializer

_Authoritative record of locked design decisions and the evidence behind them.
Read this before touching solver code or the paper. Supersedes all prior
handoff docs (`scvx_fix_handoff.md`, `paper_revision_handoff.md` — deleted;
history in git). Terse rules live in `CLAUDE.md`; this file holds the full
design and the evidence. In-flight work lives in `doc/session_handoff.md`._

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

**WHAT IS STILL UNVERIFIED:** no optimality evidence exists for ANY reported
answer. See §10.

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
                ‖ r(τ) − c_KOZ ‖ ≥ r_e   for all τ ∈ [0,1] (keep-out sphere)

- g = true gravity (two-body + J2) — the ONLY non-linear ingredient.
- r″(τ)/T² is physical geometric acceleration, so the integrand is the
  squared control (thrust) acceleration: J is control-effort energy (§1).
- All BC rows are linear equalities in x (verified against
  `constraints.rs:184-266`; endpoints pinned at `optimizer.rs:945`).
- The continuous KOZ constraint is never imposed directly. It is replaced by
  the 명제 1 sufficient condition: De Casteljau subdivision Q_i = A_i x into
  n_seg segments; segment i is certified iff some unit witness n_i has
  n_i·(q − c_KOZ) ≥ r_e for EVERY control point q of Q_i. Witness-based and
  existential — see §9.

**Convex subproblem at iterate k (reference p, trust radius r_k):**

    minimize    L(x) = ∫₀¹ ‖ r″/T² − (J_s r + c_s) ‖² dτ + w_s·Σ_rows max(0, b_row − a_row·x)
    subject to  BC rows,  ‖x − p‖_∞ ≤ r_k

- Gravity affine per segment: J_s, c_s = Jacobian/offset of g at the segment
  centroid of p; the integral is EXACT via the Bernstein Gram matrix (§1).
- KOZ rows (elastic, L1-penalized by w_s): n_i·(A_i x)_k ≥ n_i·c_KOZ + r_e
  with witnesses n_i chosen by the centroid rule AT p. With n_i fixed these
  are linear inequalities — the convexification itself, not a model of
  anything (§9).

**Acceptance (formulation of record per §9; code change pending):**

    ρ = [T(p) − T(x⁺)] / [L(p) − L(x⁺)],  where T(x) = J(x) + w_s·(violation of the SAME KOZ rows as L)

- T uses TRUE gravity g; L uses the affine model. The penalty rows are
  IDENTICAL in T and L, so the KOZ contributes zero mismatch between
  numerator and denominator: ρ tests exactly one thing, the gravity
  linearization. §3's older rebuilt-rows definition of T is superseded (§9).

## 1. Objective (locked 2026-08-07)

Singular, hard-coded — the exact integral of control-acceleration energy:

    J = ∫₀¹ ‖ a_geom(τ)/T² − (J_s r(τ) + c_s) ‖² dτ

- Gravity = two-body + J2, affine per De Casteljau segment (J_s, c_s at the
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
  ‖x−p‖_∞ ≤ r_k inside the QP.

## 3. Acceptance (canonical SCvx, Mao et al.)

- The elastic (virtual-control) QP is solved UNCONDITIONALLY on the trust
  path: slack on KOZ rows with L1 penalty w_s. The QP therefore minimizes
  exactly the convex merit L(x) = J⁽ᵏ⁾(x) + w_s·Σ max(0, b − a·x), so the
  predicted merit reduction is ≥ 0 by construction (given a reference feasible
  for the hard rows — see bootstrap).
- True merit T(x) = J_true(x) + w_s·h(x), where h(x) = hull-row violation with
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
- pred_floor = 1e-12·(1+|L(p)|): pred below it is a null step; accepted iff
  non-worsening (this is how stationarity terminates the loop).

## 4. Convergence (locked 2026-08-07)

- Converged ⇔ relative merit change |ΔT|/|T(p)| < tol_f on K = 3 CONSECUTIVE
  accepted steps AND the accepted iterate carries the certificate
  (h ≤ 1e-6 km aggregate). tol_f = max(tol, 1e-8); defaults tol = 1e-8
  (Python signature and verify harness).
- Second criterion: trust radius < 1e-2 km at a reference satisfying the
  certificate AND all hard rows ⇒ converged (standard trust-region collapse
  stopping); otherwise converged stays false.
- Why K-consecutive: a one-shot test fires during the slow crawl along
  re-aimed KOZ walls — measured stopping 33% above the optimum at tol 1e-6
  (evidence #3/#4). Step-norm tests are invalid here: the walls re-aim every
  iteration, so the iterate jitters along a moving constraint.

## 5. Parameters (each with its selection principle)

| parameter | value | principle |
|---|---|---|
| w_s (`elastic_weight`) | 1e-2 | exact-penalty rule: above the KOZ dual scale (~1e-6, estimated), far below objective-swamping — w_s=1e4 measurably degraded optima 4–5× via penalty noise in ρ (evidence #2) |
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
  paper configuration. The proximal is skipped entirely when `trust_active`,
  so the pillar-2 `prox_inert` cell compares identical code paths — it is
  documentation of intent, NOT evidence. Nine `tools/probe_*.py` still
  reference the deleted freeze knobs.

## 7. Evidence log (what killed what)

1. **2026-08-05** — two-phase acceptance → canonical merit. Three adversarial
   agents + literature survey: the scale-mismatch justification was
   self-contradicted by w_s; pred<0 hole confirmed; real precedent family is
   phase-I/II, CCP, and 2024–26 two-phase SCP — not filter methods.
2. **2026-08-05** — w_s grid: 1e4 → +460% optimality gap; 1e-2 → +3.4%;
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

### Caveats on this evidence log (2026-08-08 adversarial review)

Entries #2–#5 were measured in scratchpad scripts whose outputs were not
persisted to `artifacts/`, and #2's numbers predate the exact-integral
objective. The conclusions they support (w_s scale, smoothness removal, freeze
harmfulness) are believed sound and were each reproduced at the time, but the
numbers are not currently reproducible from a clean checkout. Re-derive before
citing any of them in the paper.

## 8. Paper terminology

`doc/korean_writing_case_collection.md` §6 판정표 is the vocabulary authority;
the local `korean-prose` skill is mandatory for Korean output. "merit function"
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
  gives ‖q−c‖ ≥ r_e). The normal choice affects conservatism, never validity.
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

### Why (A), on measurement (2026-08-09, phase120, trust=2000, w_s=1e-2)

| | (A) | (B) |
|---|---|---|
| n_seg=16 | **20 iters**, stop 1 | 172 iters, stop 2 (gave up) |
| n_seg=16 J_true | **2.265623e-05** | 2.269524e-05 (+0.17%) |
| n_seg=32 | **21 iters**, stop 4 | 261 iters, stop 2 |
| n_seg=64 | — | 445 iters, stop 2 |

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
r_e at the converged iterate, phase120: 63.9 km (n_seg=8), 15.6 km (16),
3.9 km (32), 0.97 km (64) — falling as ~1/n_seg², exactly the tangent-plane
geometry (lateral reach L per segment ⇒ over-clearance ≈ L²/2r_e). This is
the real cost of the sufficient condition and belongs in the paper as a
quantitative characterization, not a footnote.
