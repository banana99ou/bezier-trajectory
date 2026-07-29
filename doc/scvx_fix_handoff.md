# SCvx optimizer fix + verification — session handoff

_Curated context for continuing this work. Last updated end of the SCvx-fix + verification session._

## TL;DR state
- **The fix is done, committed (`04c3f07`), and verified** — a trust-region SCvx rewrite of `optimize_orbital_docking` that converges in ~4 iters (energy) / ~10–12 (dv) instead of hitting the 10000 cap.
- **⚠️ UNCOMMITTED and fragile (commit first):** the Stage-B optimizer changes (`disable_scvx_freeze` toggle + 6 per-iteration diagnostic histories) and the entire `tools/verify/` verification harness. Suggested split:
  - `feat(optimizer): disable_scvx_freeze toggle + per-iteration SCvx diagnostics`
  - `test(verify): SCvx fix-verification harness (4 pillars, all pass)`
- **Verification: OVERALL PASS** (all 4 pillars). See `artifacts/verify/VERDICT.md`.
- **One red test, NOT a regression:** `tests/regression/test_golden_run.py` fails on a *stale* golden (`9.005e-05`, committed in `e849ff7` "rust migration prep", pre-elastic era). Current legacy value `5.455e-05` is stable across Stage-B and matches the Rust golden. Refresh with `UPDATE_GOLDEN=1` or leave.

## How we got here (session flow)
Chronological arc, including dead-ends worth **not** re-walking:

1. **Started from the paper.** Prof's critiques (`paper_draft_korean_PF.md`, 87 inline annotations → distilled in `paper_revision_action_list.md`). Confirmed scope, then built `paper_draft_korean_rev1.md` (folded §2 into §1; rewrote abstract/contributions/§5; tables → TODO; figure placeholders). **§4 method was held** — because…
2. **…the prof's real objection was the optimizer, not the prose:** "a correct optimization converges in a few iterations; 10000 means the loop is broken." The user had already tried freezing the linearization and gotten *worse* numbers.
3. **Diagnosed the loop** (`optimizer.rs`): it wasn't a real SCvx — unconditional step accept, a fixed trust radius applied as *post-hoc clipping*, and a nanometre step-norm tolerance → it always ran to the cap. Taught the missing piece (trust-region ρ-test acceptance) plus adjacent methods (SOCP vs the IRLS hack, KOZ linearization).
4. **Fixed it, through several dead-ends (do not re-explore):**
   - single-phase merit with a μ-penalty → returned *infeasible* (violation-scale mismatch);
   - switched to a two-phase merit (restoration → optimality) with a step-norm convergence test;
   - **"freezing the KOZ regressed energy to 10000" was a red herring** — it was the *step-norm criterion* failing under moving constraints, not the freeze;
   - reverted to an objective-change convergence test;
   - **root cause finally found:** the mis-scaled proximal (1e-6 vs ~1e-13 Hessian). Skipping it in the trust path → energy 4 / dv 12 iters. Committed `04c3f07`.
5. **User ran it themselves → stale `.so`.** A month-old wheel silently ran the OLD optimizer (first an `elastic_weight` TypeError, then 1000-iter caps that looked like the fix had failed). Rebuilt into the venv; this is where gotchas #1–#3 below were learned.
6. **`n_seg=4` still capped → investigated, ruled out two wrong theories:** it is **NOT** a limit cycle (the curve is comfortably feasible) and **NOT** the IRLS/dv hack (it fails in energy mode too). Real cause: persistent elastic KOZ slack + slow linear convergence, and it's *narrow* (only `phase120` N≥7).
7. **User handed over `doc/tmp.md`** (conceptual questions + a 4-pillar "prove the fix is real" checklist) → plan mode → built the `tools/verify/` harness (Stage A no-rebuild pillars; Stage B Rust toggle + diagnostics; Stage C ablation + sweep). All pillars PASS; dv disabled (energy default). Then this handoff.

## The fix — root cause (the non-obvious part)
The "crawl to the 10000 cap" was **not** the ρ-test or IRLS. It was a **scale bug in the proximal term**: the paper's `scp_prox_weight = 1e-6` is ~7 orders of magnitude larger than the energy Hessian (`Gram/T⁴ ≈ 1e-13`), so every QP became `min (λ/2)‖x−p‖²` — a tiny anchor-to-current-point step. The fix, gated behind `scp_trust_radius > 0` (legacy path untouched): trust-region ρ-test acceptance + adaptive radius as a QP constraint + feasibility-restoration phase + **skip the proximal in the trust path** + freeze gravity/KOZ linearization at the first feasible iterate. Entry point: `rust_optimizer/core/src/optimizer.rs::optimize_orbital_docking`.

## Operational gotchas (each cost real time — read before running)
1. **Stale `.so` runs the OLD optimizer silently.** The venv `bezier_opt.so` is not auto-rebuilt. After ANY Rust change, rebuild + reinstall, and sanity-check `.so` mtime vs source. Exact commands (venv is Python **3.11.9**; repo is symlinked from `/Volumes/Sandisk`):
   ```
   cd rust_optimizer/pybind
   /Users/hyeon-yongjeong/code/bezier-trajectory/.venv/bin/maturin build --release -i /Users/hyeon-yongjeong/code/bezier-trajectory/.venv/bin/python3.11
   cd ../.. && .venv/bin/pip install --force-reinstall --no-deps rust_optimizer/target/wheels/bezier_opt-0.1.0-cp311-cp311-macosx_11_0_arm64.whl
   ```
   (`maturin develop` fails here — it can't resolve the interpreter name; build a wheel + pip install instead.)
2. **The cache does NOT hash the Rust binary** — only params + `CACHE_VERSION` (`orbital_docking/cache.py`). After a rebuild, stale results return as cache hits. Bump `CACHE_VERSION` (now `7.0-scvx-energy`) OR pass `use_cache=False`. The harness always uses `use_cache=False`.
3. **CLI/interpreter mismatch:** run with `.venv/bin/python` (3.11), not bare `python3` (a 3.11 framework build elsewhere). The `.so` is interpreter-specific.

## Verification harness (`tools/verify/`, run from repo root)
`.venv/bin/python tools/verify/{harness_common is a lib; nlp_crosscheck, kkt_check, ablation, diagnostics, sweep}.py` then `verdict.py`. Outputs → `artifacts/verify/`. `harness_common.py` holds `J_true` (the canonical dense true-gravity objective — **never** compare the Rust `cost_true_energy` surrogate across solvers), scenarios `phase120` (aggressive) / `phase70` (mild), and `run_rust`.

Findings worth carrying forward:
- **Pillar 1:** an independent SciPy `trust-constr` NLP, warm-started at x*_rust, sits on the true KOZ boundary; the Rust conservatism gap **shrinks 17%→1.4% as n_seg grows** (validates the paper's subdivision thesis).
- **Pillar 2 (ablation):** the prox-skip is the real fix; **the freeze is arguably droppable** — it saves ~1 iteration but locks in ~2.8% extra conservatism, and canonical re-linearize-every-step SCvx converges fine (5 iters).
- **Pillar 4 (sweep):** `n_seg=4` is a **narrow** holdout — only fails at `phase120` N=7 & N=8; it converges for milder geometry / N=6. Cause: persistent elastic KOZ slack (~53) + slow linear convergence, mode-independent (fails in dv AND energy).

## Open decisions / next steps (roughly prioritized)
1. **Commit the uncommitted work** (see TL;DR) — most important.
2. **Consider dropping the freeze** — Pillar 2 shows it's marginal and adds conservatism.
3. **dv mode is disabled (energy default) but the Rust code is dormant, not deleted.** Full deletion + the pure-Gram redesign (drop the sampled gravity-residual term so energy is exactly `Gram/T⁴`, no sampling) is a larger follow-up with paper implications.
4. **`n_seg=4`:** footnote/drop in the paper, or fix via a tighter (satisfiable) KOZ linearization.
5. **Paper `doc/paper_draft_korean_rev1.md`:** §4 method held pending the SCvx framing decision; tables are TODO and become energy-based now that dv is retired.

## Working preferences (assistant should honor)
- **No AI co-author trailers** in commits (and warn if any are found).
- **No `.claude` memory folder** — keep persistent context in the repo (this file), version-controlled and visible.
- **Don't over-claim.** Validate on the REAL config, not a simplified probe (I once reported probe numbers as if they were the real Progress→ISS config — they weren't; the real problem has boundary velocities, sample_count=100, degree 8).
- Ask good clarifying questions; remind to commit when changes accumulate.
- Lab self-reference in Korean writing = **국민대** (never 본 연구실 / KMU).

## Key pointers
- Fix + entry point: `rust_optimizer/core/src/optimizer.rs` (commit `04c3f07`).
- pybind/Python surface: `rust_optimizer/pybind/src/lib.rs`, `orbital_docking/optimization.py`, `orbital_docking/cache.py`.
- Harness + results: `tools/verify/`, `artifacts/verify/VERDICT.md`.
- Paper: `doc/paper_draft_korean_rev1.md` (+ `paper_revision_action_list.md`).
- The approved verification plan lived at `~/.claude/plans/fancy-coalescing-forest.md` (not repo-level); it's fully realized by `tools/verify/`, so this file supersedes it.
