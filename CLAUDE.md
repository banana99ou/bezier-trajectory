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
  2025. What is new is the **support oracle on the hazard** — decomposition-free describes what we
  do not do. Guarded in artifact text only by `tests/unit/test_retired_wording.py`; believing it
  is still your job. → [`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) §1.1
- **Never write that a decomposition binds the passing time.** False against a space-time
  decomposition, and it is already in the submitted KSAS manuscript and the draft poster — the
  poster's copy must go before 2026-11-10. Guarded in artifact text by
  `tests/unit/test_retired_wording.py`. → [`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) §1.2
- **The journal's reader is the planning / trajectory-optimization community**, not aerospace
  guidance and control — decided 2026-09-20, and it outranks the PI report's JGCD recommendation,
  which predates it. → [`paper/journal_1/SPEC.md`](paper/journal_1/SPEC.md) §6
- **A zero time coefficient on a keep-out row is a reportable defect**, shadow rows included —
  never a modelling choice. → [`SOLVER.md`](SOLVER.md)
- **`doc/notes/` is not part of this repository.** Each note is its own git repo on disk and the
  folder is ignored here. → [`doc/notes/README.md`](doc/notes/README.md)
- **Two tests are red on purpose, each waiting on the user's decision** — each lost its specimen
  to a solver improvement. Do not fix, weaken, or re-anchor them. → [`WORKSTREAM.md`](WORKSTREAM.md)
  `repo.red-test`
- **A number in a markdown file is not evidence.** Put a measured number where a test re-measures
  it and cite the test; a document quotes a measurement pass, it never owns one.
- **Exclude `.claude/worktrees/` from every repo-wide search** — it shadows every document at an
  older commit, so each hit returns twice and the wrong copy is citable. → [`README.md`](README.md)

## Session protocol

The commit prefix names the owner: `solver:`, `idea1:`, `idea2:`, `paper(ksas):`, `refs:`. Last act
of a session is to update the owning document and commit. **`git log` is the status board** — no
document keeps one.
