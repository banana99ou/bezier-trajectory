# Session Handoff — paper workstream

_Ephemeral: rewrite this file each session. Design and locked decisions live in
`doc/design_freeze.md` — READ IT FIRST; it outranks the paper. Rules are in
`CLAUDE.md`. Mathematical symbols are locked in `doc/notation.md` — read it
before writing any equation, and verify with `python3 tools/check_notation.py`.
Korean output requires the `korean-prose` skill; this paper's own term decisions
are in design_freeze §8.1._

## State (2026-09-01)

**The paper carries zero `[[…]]` annotations and zero TODOs**, for the first
time. The solver has not been the blocker for three weeks; the paper is now the
whole workstream.

- Verification: **6/6 pillars PASS**, stamped with the producing commit and the
  Rust extension hash in `artifacts/verify/VERDICT.md`.
- Tests: 24 Rust, 89 Python (1 skipped).
- Tables 2/3/4 and figures 4/5/6 all come from `tools/build_tables.py` and the
  builders beside it, so a figure and its table cannot disagree.
- **108 commits unpushed** to `gitlab/main`. The remote is `gitlab`, not
  `origin` — `git log origin/main..HEAD` silently reports 0.

### Uncommitted right now

The related-work draft and the terminology pull, in
`doc/paper_draft_korean_rev2.md` and `tools/build_tables.py`. Suggested message:
`docs(paper): 용어를 동반 논문에 맞추고 관련 연구를 안전 회랑 문헌에 접지한다`.

## What changed this session

1. **Framing loosened** — the method is 궤적 생성, not 궤적 초기화. Forced, not
   cosmetic: with the pipeline comparison cut there is no experiment left in the
   paper that demonstrates initialization. 초기화 survives only in related work
   and in the sentence saying the output can also serve as an initial guess.
2. **§4.2 rewritten** — it drew 전체적으로 재작성 from the professor because it
   never said what was varied or measured. Each experiment now states what is
   fixed, what varies, which way the result should move and why, and why the
   baseline geometry is the one where the comparison exists at all.
3. **`n_lin` = 100 is justified, and it is not tied to `n_seg`.** Swept 5→400;
   the objective is converged by ~50 and 100→400 moves it by <1e-9 relative,
   three orders below the 1.8e-6 Python/Rust agreement. Measurements and the
   reproduction command are in design_freeze §7 entry 11.
4. **`sample_count` → `n_lin_seg`** through Rust, pybind, Python, tests, tools
   and docs. It never counted samples; the objective integral is the exact Gram
   closed form. The name had actively misled us.
5. **Figure 4 rebuilt** from the five geometries in table 2, in the departure
   orbit frame, solved fresh with the cache off.
6. **Related work grounded in the safe-corridor literature** — refs [15]–[19],
   the two-moves framing, and Assumption 3 of Proposition 1 tied to the
   convention six papers share. Source and caveats: design_freeze §10.
7. **Terminology pulled from the companion manuscript** — 신뢰영역, 무게중심.
   Two divergences kept deliberately: the slack-penalty symbol stays μ, and SCvx stays SCvx
   rather than SCP. See §8.1.

## Context that is otherwise lost

- **The `/code-review ultra` of the solver math produced nothing.** Launched
  2026-08-12 00:42 from the `review/core` worktree against `1581e54~1`, 21 files;
  the cloud session exceeded 30 minutes and returned no findings. **The solver
  math has had no external adversarial review.** Do not read the 6/6 PASS as one.
- **`experiments/downstream/` has never been run.** The DCM comparison was
  redesigned from first principles (`doc/dcm_downstream_experiment_design.md`,
  2×2 factorial, pre-declared cases, independent grader) and the runner was
  committed in `831297d`, but no session has executed it. The old T6 numbers and
  `doc/dcm_downstream_pack.md` are stale — produced by the April solver.
- **The pipeline comparison is cut from the paper** (`e60f6e8`). Cleanly: the
  abstract, intro and conclusion no longer promise it. Expect the professor to
  ask where it went. The honest answer is that the old comparison reported only
  the 7 cases where both pipelines happened to succeed, which is selection on the
  outcome.
- **The conclusion's non-circular-orbit claim was deleted**, not overlooked. Its
  evidence came from the boundary sweep run by `tools/dcm_downstream_experiment.py`,
  a tool deleted in `831297d`, and those findings are formally demoted to
  hypotheses by the redesign doc.
- **The professor's §2 instruction is already satisfied.** His §2 was
  관련 연구 및 위치 설정 and it is now inside §1. Today's §2 is the control-point
  machinery, which he never asked to fold. `doc/paper_revision_action_list.md`
  does not know this — it is superseded, see below.

## Before this goes to the professor

1. **One end-to-end read.** The draft has been edited in pieces by several
   sessions and nobody has read it start to finish since the pipeline section
   came out. That is where a paragraph that no longer connects to its neighbours
   will show up.
2. **Render a PDF.** `rev2` has never been rendered; he reviewed a PDF last time.
3. Optional, and a good thing to *ask* rather than fix: whether decoupling the
   gravity-linearization partition from the KOZ subdivision is standard, or needs
   its own justification. Standard SCvx linearizes dynamics on the discretization
   mesh. Draft question in design_freeze §7 entry 11.

## Superseded documents — do not act on them

- `doc/paper_revision_action_list.md` — statuses are fiction; written before the
  paper was rewritten against the code. Several items are moot (IRLS, Δv proxy)
  and several are already done.
- `doc/paper_execution_state.md` — 2026-04-17. dv objective, 10000 iterations,
  T6. Every configuration line in it is wrong.

## Operational gotchas

1. **Rust rebuild** (the `.so` does not auto-update):
   `cd rust_optimizer/pybind && ../../.venv/bin/maturin build --release && cd ../.. && .venv/bin/pip install --force-reinstall --no-deps rust_optimizer/target/wheels/bezier_opt-0.1.0-cp311-cp311-macosx_11_0_arm64.whl`
   (`maturin develop` fails; use `cargo test --release -p bezier_opt_core`, the
   workspace test fails to LINK the pybind crate.)
2. **The cache never hashes the binary.** Bump `CACHE_VERSION` (now
   `17.0-n-lin-seg-rename`) or pass `use_cache=False`. The verify harness is
   always cache-off.
3. Use `.venv/bin/python` (3.11); bare `python3` is a different interpreter.
4. `SCVX_TRACE=1` dumps a 28-column per-iteration CSV to stderr including
   REJECTED steps. This is the instrument that made the KOZ pivot visible.
5. `figures/*.png` is gitignored — figures are regenerated, not committed.
6. Two commits carry AI co-author trailers (`f4b4579`, `3f0161e`, 2026-02-23,
   `Co-authored-by: Cursor`). Pre-existing, from a different tool; removing them
   means rewriting history.
7. Syncthing conflict duplicates poison grep-based searches. Two were renamed to
   `.rs.bak`; others may remain.

## Key pointers

- Design and formulation of record: `doc/design_freeze.md` — §0 formulation,
  §7 evidence log, §8.1 terminology, §9 the KOZ decision, §10 companion papers
- Rules: `CLAUDE.md` · symbols: `doc/notation.md`
- Solver: `rust_optimizer/core/src/optimizer.rs` (objective
  `build_ctrl_accel_quadratic`; acceptance in the `trust_active` block) ·
  KOZ rows: `rust_optimizer/core/src/constraints.rs`
- Verification: `tools/verify/*.py` → `artifacts/verify/VERDICT.md`. Order:
  `nlp_crosscheck, ablation, kkt_check, diagnostics, sweep, optimality`, then `verdict`.
- Paper: `doc/paper_draft_korean_rev2.md` · numbers: `doc/results/paper_tables.md`
- Companion papers: `../bezier-trajectory-merge/` — see design_freeze §10
