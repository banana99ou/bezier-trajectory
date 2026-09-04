# CLAUDE.md

Direction and guardrails only: **what you need before you know which document to open.** Everything
else has an owner, and [`README.md`](README.md) §Repo map routes to all of them.

## Goal

- **End goal** — develop these ideas and publish them.
- **Now** — KSAS 2026년도 추계학술대회 **제출 완료(2026-09-03)** → 포스터 → idea 1 저널 투고. The journal
  draft runs here in [`paper/journal_1/`](paper/journal_1/) — the separate `bezier-trajectory-journal`
  worktree was retired 2026-09-04.
- **What is open** — [`WORKSTREAM.md`](WORKSTREAM.md). Open items only; `git log` carries the rest.

## Traps

Each line is a mistake that already cost something, and an unbuilt guard: when a test or a raised
error makes the mistake impossible, **delete the line**. One line, imperative, with a link — if it
needs a paragraph it is a finding, and it belongs to its owner instead.

- **Never `git merge` or rebase `main`.** The merge-base carries no Rust; both lineages wrote
  `rust_optimizer/` from scratch, so it is an add/add conflict on every file. Bringing solver work
  across is a file-level port. → [`SOLVER.md`](SOLVER.md)
- **Never open the novelty claim with "time as a coordinate."** The lift was published in August
  2025. What is new here is *decomposition-free*. → [`idea/spacetime.md`](idea/spacetime.md)
- **A zero time coefficient on a keep-out row is a reportable defect**, shadow rows included —
  never a modelling choice. → [`SOLVER.md`](SOLVER.md)
- **`doc/notes/` is not part of this repository.** Each note is its own git repo on disk and the
  folder is ignored here. → [`doc/notes/README.md`](doc/notes/README.md)
- **One test is red on purpose** (`test_a_clearing_run_can_still_be_standing_on_slack`). Do not fix
  it, weaken it, or revert the density that turned it red. → [`SOLVER.md`](SOLVER.md) §Known Issues
- **A number in a markdown file is not evidence.** Put a measured number where a test re-measures
  it and cite the test; a document quotes a measurement pass, it never owns one.

## Session protocol

The commit prefix names the owner: `solver:`, `idea1:`, `idea2:`, `paper(ksas):`, `refs:`. Last act
of a session is to update the owning document and commit. **`git log` is the status board** — no
document keeps one.
