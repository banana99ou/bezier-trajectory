# Session Handoff — paper-alignment workstream

_Ephemeral: rewrite this file each session. Design and locked decisions live in
`doc/design_freeze.md` — READ IT FIRST; it outranks the paper. Rules are in
`CLAUDE.md`. Mathematical symbols are locked in `doc/notation.md` — read it
before writing any equation, and verify with `python3 tools/check_notation.py`._

## State (2026-08-10)

The solver is now trusted. That was the blocker on the paper rewrite, and it is
gone. **15 unpushed commits.** Today's four:

| commit | what |
|---|---|
| `12b5b06` | variant (B) — merit graded on the rows the QP solved. PRESERVED, REJECTED. |
| `278d07a` | user's F1 figure rewrite (KOZ linearization from real constraint projection) |
| `48f6273` | **(A) — the design of record.** Models the KOZ normal re-aim in the subproblem rows. |
| `d8e9f2d` | Pillar 5 (gated optimality) + three gates that could not fail + `J_true` quadrature fix |

Verification: **all 6 pillars PASS**, 24 Rust tests, 86 Python tests green.
Optimality is gated for the first time in the project's history.

Do NOT trust older chat summaries or docs describing the solver: anything
mentioning two-phase acceptance, a dv/IRLS mode, a smoothness term, a sampled
objective, or `scvx_freeze` is obsolete.

## Decisions locked this session

1. **(A) over (B) for the KOZ subproblem rows.** The centroid rule re-aims the
   supporting half-space every iteration. (A) adds the normal-rotation term
   (`build_koz_constraints_linearized`) so the QP's optimum is a point the
   centroid rule still agrees with after re-aiming. (B) instead graded the
   candidate on the same frozen rows the QP solved. Measured phase120/n_seg=16:
   (A) 20 iters / stop 1 / 2.263313e-05 versus (B) 172 iters / trust collapse /
   2.267205e-05. **(B) is preserved in `12b5b06`; do not resurrect it without
   reading design_freeze §9 first.**
2. **How to DESCRIBE (A) — this matters for the paper.** It is **not** a physical
   model of the keep-out zone and must never be written as one. The sphere is the
   constraint; the half-space is its convexification. (A) is a *self-consistency
   device for the step*. The original framing ("the true first-order model of the
   clearance") is what made the fix look like a hack, and the user was right to
   attack it. The detour through (B) is what produced the correct framing.
3. **K-consecutive stationarity exit kept.** Costs ~14 iterations on
   phase120/n_seg=16 (6 → 20), removes the one-shot premature-stop mode that §4
   exists to prevent. Both principled exits now require K = 3.
4. **The (B)-era own-rows convergence gate was dropped** — redundant under (A),
   where the merit already grades rows rebuilt at the candidate.
5. **`scvx_freeze` deleted** (`9c673b0`). Decision made; §6 of design_freeze
   updated. Nine `tools/probe_*.py` still reference the dead knobs.
6. **`r0` is now a per-scenario field, not a global 2000.** It must exceed the
   iteration-1 BC-repair distance, which is geometry-dependent.

## Discoveries (measured, with numbers)

### The KOZ pivot — the defect that started the whole audit

The merit graded candidates against half-spaces rebuilt AT the candidate while
the QP had optimized rows built at the reference. Optimizing one quantity,
grading another. Symptom: ρ ∈ [−1.07e5, −1] on every rejected step, 24
rejections, trust collapse.

The geometry, measured at iteration 6, n_seg=16, segment 7 (reproduce with
`SCVX_TRACE=1`, columns `vtrue_c` / `cpviol_c` / `minrad_c`):

- The optimizer presses the segment's two outermost control points **flush onto
  the plane** — clearance exactly 0.0.
- Those points sit **±455 km laterally** along the plane from its tangent point.
- The normal then pivots **1.5e-4 rad (31 arcsec)**.
- 455 km × 1.5e-4 rad = **68 m**: one end drops below the new plane, the other
  gains 68 m.
- Meanwhile that control point is **16.0 km ABOVE the actual sphere**. True
  sphere penetration was 0.000 on all 101 crawl steps measured.

So the "violation" is pure pivot, never danger. **It cannot be meshed away:**
451 m / 68 m / 6.3 m / 0.8 m at n_seg = 8/16/32/64, still 800× above the 1e-6 km
certificate gate at n_seg=64.

### Conservatism of 명제 1 — quantified, and PAPER-RELEVANT

A single flat plane per segment against a round sphere forces the trajectory
further out than the constraint requires. Over-clearance ≈ L²/2r_e where L is the
segment's lateral reach. Measured minimum clearance above r_e at the converged
iterate, phase120: **63.9 km (n_seg=8), 15.6 km (16), 3.9 km (32), 0.97 km (64)**
— falling as ~1/n_seg². This is a real quantitative account of the method's cost
and currently appears NOWHERE in the paper.

### `J_true` was broken — and it is the harness's own source of truth

The canonical cross-solver objective was a uniform-mean Riemann sum converging
only as O(1/n): **~1.6e-3 relative error at its n_dense=1200 default**, which is
LARGER than the A/B gap (1.7e-3) it was being used to resolve. It manufactured
562 phantom descent directions on phase70 whose best step **reversed sign**
(−1.4e-5 → +5.0e-5) once the quadrature converged.

Now Gauss-Legendre, exact to f64 at 24 nodes. **It agrees with the Rust
exact-Gram integral to 1.8e-6 relative** — the cross-implementation check the
objective never had, and a citable validation.

⚠️ **Any objective number quoted from before `d8e9f2d` is suspect.** The A/B
comparison in design_freeze §9 is deliberately quoted from the Rust internal
merit, which the bug never touched.

### Three gates could not fail

- **Pillar 4a had no gate against drift.** ρ and per-phase monotonicity only ever
  compare within one iteration, while the merit function itself changes between
  iterations (the rows re-aim). (B) drifted 0.178% away from its own best point
  with ρ = 1.000 on all 139 accepted steps and every gate green. Added `best_ok`:
  the returned iterate must be the best visited.
- **Pillar 2's `prox_inert`** compared two bit-identical code paths (the proximal
  is skipped when `trust_active`). Removed from the verdict, kept as a labelled
  identity check.
- **Ablation gated on `scvx_converged`**, which is also set by the trust-collapse
  exit. Switched to `scvx_stop_reason ∈ {1,4}`.

### `w_s` is no longer an estimate — closes a §5 citation gap

‖λ_KOZ‖_∞ **MEASURED at 1.5e-7** (max across five scenarios, Pillar 5). With
w_s = 1e-2 the ℓ1 exact-penalty condition `w_s ≥ ‖λ*‖_∞` (Han & Mangasarian 1979;
Nocedal & Wright 2e Thm 17.3) holds with **5 orders of margin**. design_freeze §5
previously said "~1e-6, estimated" and nobody had ever measured it.

### Scenario grid: 2 → 5

`phase70`, `phase120`, `phase135`, `phase170`, `planechange`. All pass Pillars
1–5 under (A).

- **`phase170`** is nearly antipodal; its straight-line init passes **5420 km
  INSIDE the KOZ**, so r0=2000 fails with stop_reason=3. r0=4000 converges, and
  to an identical answer for every r0 in 4000..12000.
- **`planechange`** has a **23.9° orbit-plane difference** (inc 51.64→71.64,
  RAAN 0→15), so the two velocity BCs differ in *direction*, not just phase. Use
  `H.bc_angle_deg(sc)` for the BC-angle metric.

## Next steps (priority order)

### 1. THE PAPER — this is the actual workstream (see the dedicated section below)

### 2. Two loose ends in what was just built

- **The new scenarios only run in Pillar 5.** Pillars 1/2/3/4a use phase120 only;
  4b uses phase120+phase70. So "everything rests on one binding geometry" is only
  half fixed. Cheap to extend, and it strengthens every structural claim.
- ~~**Pillar 5 may be under-resolved**~~ **RESOLVED 2026-08-11, and the cause was
  not the one suspected.** The four-orders spread (3.1e-6 phase70 vs 1.4e-2
  phase120) was not `ACTIVE_TOL` — that threshold turns out to be irrelevant
  (residual identical over 1e-5…1e-1 on all five scenarios). It was
  `optimality.py:koz_rows` omitting the witness-rotation term, which inflated the
  residual by 295×–131000× in proportion to mesh coarseness. Fixed; residuals are
  now flat at 3.1e-6…1.1e-5, and `KKT_TOL` is tightened 5e-2 → 1e-3. See
  design_freeze §7 entry 9.

### 3. Housekeeping

- **Push** (15 commits).
- Nine `tools/probe_*.py` reference the deleted freeze knobs.
- Pre-existing unused-variable warning, `optimizer.rs:390` (`total_rows`).
- (B) was never re-measured against the fixed `J_true` — the internal-merit
  comparison makes this optional. A `git checkout <commit> -- <path>` + rebuild
  was **blocked by the permission classifier**; needs explicit approval.

## PAPER REWRITE — everything needed

**Method: D1** — surgical edits at the affected spans only, `.bak-YYYYMMDD`
first, **`korean-prose` skill loaded before writing any Korean**.

### The formulation of record

`design_freeze.md` **§0** now holds the exact problem statement (objective, all
BC rows, the continuous KOZ constraint, the convex subproblem, the acceptance
test) verified against the code line by line. **Write the paper from §0, not from
memory.**

### Span-by-span work list

| span | what must change |
|---|---|
| **§3.1 + Figure 1** | **BLOCKING, and it is the user's own work (`278d07a`).** Both present an SCvx iteration as normals FROZEN at the reference. That describes rejected variant (B), not the shipped subproblem. Probable resolution: the figure is a correct picture of the **certificate** (which genuinely uses frozen normals) but not of the **subproblem rows** — so reframing may suffice without redrawing. Decide this first; §3 text builds on it. |
| §4.1 | 수렴 허용오차 **10⁻⁶ → 10⁻⁸**. Describe the REAL termination: two principled exits, each requiring **K = 3 consecutive** qualifying iterations — the merit streak (`stop_reason` 1) and model stationarity with the certificate held (`stop_reason` 4). Already correct in §4.1: r₀=2000 km, ×2/÷2, η=0.1, floor 1e-2 km, w_s=1e-2. |
| §2 roadmap + §2.3 | The Gram matrix now serves the **control-energy** integral — rewrite the section's ROLE, do not delete it. The smoothness term is gone. |
| §3.2 | Single exact-integral objective display. |
| §3.3 + Algorithm 1 | State the K-consecutive requirement (it is nowhere in the paper). |
| §5.1 / §5.3 | STALE (10000회-era numbers). Regenerate. **Also a live Δv leak + unit error**: lines ~361/382/404 still quote 제어 비용 as "약 6,694 m/s", "9,288 m/s → 6,292 m/s" — those are `dv_proxy_m_s` values while the metric is defined in m/s² at line 318. |
| **NEW section** | The 명제 1 conservatism result (63.9/15.6/3.9/0.97 km, ~1/n_seg²). Genuinely new material, and it is the honest cost accounting of the method's central contribution. |

### Numbers that are safe to quote

- **Objective comparisons: use the Rust exact-Gram internal merit**, or the
  post-`d8e9f2d` `J_true`. Never a pre-`d8e9f2d` `J_true` number.
- Pillar 1 (independent scipy NLP, Rust-blind, cold start) gap versus n_seg,
  **re-measured 2026-08-11 with the corrected oracle**: **12.570% (n_seg=8),
  2.882% (16), 0.717% (32), 0.175% (64)**. The 2026-08-09 figures (12.559 /
  2.871 / 0.706 / 0.165) are DEAD: that run optimized the stale uniform-mean
  `J_true_and_grad`, whose own error (0.207%) exceeded the n_seg=64 gap it
  reported (0.165%) — see design_freeze §7 entry 10. The older "6.5% → 1.3%" was
  freeze-ON and is doubly dead.
- (A) iteration counts, phase120, **post-|pred| fix**: 8 iters at n_seg=8, 16,
  32 and 64 alike (was 8 / 20 / 21 before `0d130fc`).
- The 1.8e-6 agreement between the Python Gauss-Legendre integral and the Rust
  Gram integral is the strongest validation claim available for the objective.

### Hard prohibition (standing)

**Δv / fuel is never mentioned in the paper in ANY form** — not as objective, not
as limitation, not as future work, not as "we don't do Δv". Exclude
`dv_proxy_m_s` from every regenerated table. `orbital_docking/visualization.py`
(uncommitted, user's) still renders `Δv proxy: … m/s` in figure panel titles.

### Citation / rigor gaps still open in the method section

- **η=0.1 / grow ×2 / shrink ×0.5 is Nocedal & Wright Alg. 4.1 style, NOT Mao et
  al.** Mao's is a four-parameter update with a shrink-on-accept band. Cite
  correctly or adopt Mao's schedule.
- **Trust-collapse-as-convergence is citable** (Conn–Gould–Toint, *Trust-Region
  Methods* §6.4); the principled alternative is a criticality measure χ_k (ibid.
  §8.1/§12.1). Note the code now gates trust collapse as a FAILURE
  (`stop_reason` 2), so this is about wording, not behaviour.
- **`hard_viol_p` sums rows with different physical units** (km, km/s, km/s²,
  km²/s) into one scalar compared against 1e-9 (`optimizer.rs`, the hard-row loop
  in the trust_active block). Use per-row-family relative residuals.
- **The `1e-6 km` hull threshold is mesh-dependent** — an ℓ1 sum over
  `n_seg·(N+1)` rows, so 8× tighter per row at n_seg=64 than at n_seg=8, across
  the very sweep the paper reports. Normalize or state it.
- **The affine bootstrap is a restoration phase without the citation** and can
  recur for prograde rows, bypassing the ratio test. Not exercised by the paper
  config (`enforce_prograde=False`), but latent. Close via virtual control on
  every relaxable row (Mao/Szmuk/Açıkmeşe CDC 2016; Malyuta et al. IEEE CSM 2022
  §III) or by naming it a restoration phase (Fletcher & Leyffer, Math. Prog.
  91:239).
- **`eval_residual_terms` quadrature**: the boundary-straddling bug is fixed
  (`n_q = 1000.div_ceil(n_lin) * n_lin`), but no error bound is stated.
  Per-segment Gauss-Legendre would respect the piecewise structure by
  construction — and `J_true` now demonstrates the technique works here.
- **Units of w_s**: J is km²/s⁴, h is km, so w_s carries km/s⁴. State it.
- **J2 uses R = 6371 km** (mean) where the harmonic expansion is defined against
  6378.137 km (equatorial) — −0.22% on that term. Consistent across Rust and
  Python so cross-checks are unaffected, but a reviewer will catch it.

### Verification-provenance problem (blocks citing any pillar number)

**`artifacts/verify/**` is gitignored and records no producing commit.** A clean
checkout has zero evidence and nothing ties a number to a SHA. Either un-ignore
the verify artifacts or write the producing SHA into `VERDICT.md` on every run.
This is the same unversionable-state problem the project bans project memory
files for.

Also: design_freeze evidence-log entries #2–#5 were measured in scratchpad
scripts never persisted, and #2 predates the exact-integral objective. Re-derive
before citing any of them.

## Rules that must not be lost

- **Never compare the Rust `cost_true_energy` surrogate across solvers** — it is
  a linearized surrogate. Cross-solver truth is `harness_common.J_true`
  (Gauss-Legendre, true gravity, post-`d8e9f2d`).
- **Why the mis-scaled proximal caused the historical 10000-iteration crawl**:
  `scp_prox_weight = 1e-6` is ~7 orders above the objective Hessian scale
  (Gram/T⁴ ≈ 1e-13), so every QP degenerated into a tiny anchor-to-p step. Do not
  re-enable it. Inert on the trust path by construction (`!trust_active`).
- **Self-reference vocabulary (author-locked, do not re-litigate)**: 제안 기법
  (프레임워크 금지) · 파이프라인 (한글) · 후속 (downstream 금지) · 분할구간
  (세그먼트 금지). These are in neither 사례집 §6 nor the skill — they live here.
- **"merit function" stays in ENGLISH** (메리트 함수 has 0 printed occurrences;
  aerospace SCP papers name only the "L1 페널티 항"). 신뢰 구간 over 신뢰영역;
  예측 감소량 / 실제 감소량 validated in 3 papers incl. JKSAS 2011.
- **Lab self-reference is 국민대** — never 본 연구실 / KMU / 저희 연구실.
- `doc/paper_decision_ledger.md` is **stale** (ends at D6, 2026-07-08); its D2
  "HOLD §3 substantive rewrites" is **dead** (§3.3 rewritten in `4589fe1`). It is
  the source of the D1 editing method; treat nothing else in it as current.
- A gate that cannot fail is not evidence. Three were found this session — assume
  more exist.

## Operational gotchas

1. **Rust rebuild** (the .so does not auto-update):
   `cd rust_optimizer/pybind && ../../.venv/bin/maturin build --release && cd ../.. && .venv/bin/pip install --force-reinstall --no-deps rust_optimizer/target/wheels/bezier_opt-0.1.0-cp311-cp311-macosx_11_0_arm64.whl`
   (`maturin develop` fails. `cargo test` on the workspace fails to LINK the
   pybind crate — a pyo3 lib-test issue, not a build failure; use
   `cargo test --release -p bezier_opt_core`.)
2. **Cache never hashes the binary**: bump `CACHE_VERSION` (now
   `14.0-self-consistent-koz-rows`) or pass `use_cache=False`. The verify harness
   is always cache-off. The key now DOES hash `elastic_weight`,
   `enforce_prograde`, `prograde_n_samples`, `transfer_time`, `strict_koz_normals`.
3. Use `.venv/bin/python` (3.11); bare `python3` is a different interpreter.
4. pytest runs clean now (86 passed, 2 skipped) — dymos has an `importorskip`.
   Golden refresh: `UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/regression/test_golden_run.py`.
   The golden now exercises the **SCvx** path with real assertions, not the legacy
   fixed-point path.
5. Korean output: load `korean-prose` first; authority is 사례집 §6. New terms
   need multiple printed occurrences in the target community — one citation is
   not precedent.
6. `SCVX_TRACE=1` dumps a 28-column per-iteration CSV to stderr including
   REJECTED steps. This is the instrument that made the pivot visible — keep it.
7. ⚠️ Two commits in history carry AI co-author trailers (`f4b4579`, `3f0161e`,
   both 2026-02-23, `Co-authored-by: Cursor`). Pre-existing, from a different
   tool; removing them requires rewriting history.
8. Syncthing conflict duplicates poison grep-based agents. Two were renamed to
   `.rs.bak` (gitignored) because dots in filenames break Cargo's test
   auto-discovery. Others may remain.

## Still open, lower priority

- **Table regeneration** T2/T3/T4 (§5.1/§5.3) and T6 (§5.4): the downstream
  pipeline is broken since the dv removal. The fix is to **delete** the
  `objective_mode` kwarg (it no longer exists on `optimize_orbital_docking`) in
  `orbital_docking/downstream_collocation.py:232,246,260,276`,
  `orbital_docking/dymos_t6.py:125,133`, `tools/downstream_dc_compare.py:49`.
- **`doc/tmp.md`**: only the QP-vs-SOCP question remains open. Also carries
  user-authored forward direction recorded nowhere else — application ideas for
  demo: LOS minimization (PESA radar beam avoidance), missile evade maneuver,
  re-entry (refer to prof doc), GCS.
- **Korean terminology histogram**: 5/8 clusters done. 3 (penalty-slack,
  iteration-init, duality-exact) died on a session limit — resume with Workflow
  `scriptPath` `~/.claude/projects/-Users-hyeon-yongjeong-code-bezier-trajectory/bf8217cd-de6e-4c9e-a1da-a9bb71c5d860/workflows/scripts/scvx-korean-term-histogram-wf_af30bd8a-9b2.js`,
  `resumeFromRunId` `wf_af30bd8a-9b2` (completed clusters replay from cache).
  Then attach counts to 사례집 §6.
- Dense-probe resolution (1001 samples, ~2–4 m blind spot) vs the 1 mm
  `feasible` threshold — the certificate carries the guarantee, so the thresholds
  could be made honest.
- `optimization.py:393` docstring says the trust region is a 2-norm (it is an
  ℓ∞ box); `solve_qp` contains dead first-pass constraint assembly plus a
  `// Wait, let me re-read Clarabel's convention...` comment in production
  source; 12 unresolved content annotations in rev2 (`grep -n '\[\['`);
  §1 roadmap ↔ §2 title mismatch; `T_normed` symbol definition.
- `.claude/skills/korean-prose` versioning (`.claude/` is gitignored).

## Uncommitted, not mine — do not clobber

`CLAUDE.md`, `doc/paper_draft_korean_rev2.md`, `doc/tmp.md`,
`orbital_docking/visualization.py` (new `_solver_info_line()` surfacing
termination_reason / iterations / min_radius / koz_linear_max_violation),
deletions of `doc/paper_revision_handoff.md` and `doc/scvx_fix_handoff.md`.

## Key pointers

- Design + formulation of record: `doc/design_freeze.md` (§0 formulation,
  §9 the KOZ decision) · rules: `CLAUDE.md`
- Solver: `rust_optimizer/core/src/optimizer.rs` (objective:
  `build_ctrl_accel_quadratic`; acceptance: the `trust_active` block) ·
  KOZ rows: `rust_optimizer/core/src/constraints.rs`
  (`build_koz_constraints`, `build_koz_constraints_linearized`,
  `koz_clearances`) · Python mirror: `_build_ctrl_accel_quadratic` in
  `orbital_docking/optimization.py`
- Verification: `tools/verify/*.py` → `artifacts/verify/VERDICT.md` (gitignored).
  Order: `nlp_crosscheck, ablation, kkt_check, diagnostics, sweep, optimality`
  then `verdict`.
- Paper: `doc/paper_draft_korean_rev2.md` (backups `.bak-20260807`,
  `.bak-20260805-canonical`) · terminology:
  `~/.claude/skills/korean-prose/references/korean_writing_case_collection.md`
