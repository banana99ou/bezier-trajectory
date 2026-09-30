# Professor-facing revision - 2026-09-23

The outgoing manuscript is the baseline for the professor's next feedback.
`papers/paper_draft.md` and the separate exported Markdown are synchronized.
The historical hash in the export filename identifies its starting point;
`SOURCE.txt` records the subsequent approved corrections.

## Later work retained

- Qualitative results paragraphs in the abstract and conclusion; concise limitations.
- Short captions, with Figure 2's necessary explanation retained in the body.
- Removal of the invalid 3D obstacle-loop argument and duplicated introduction text.
- Explanation that subdivision control points are computed from the original points; corrected trajectory-variable versus slack-variable count.
- Conditional clearance estimate and separation from the avoidance certificate, using a concise derivation.
- Accurate Gram-based model cost and numerical-correction description.
- Baseline-scenario qualifier on the gravity-discretization sensitivity result.
- Progress/ISS source citations with a concise scenario explanation.
- One sentence stating that table runtimes are minima of 15 runs.
- Figure 3 content corrections and a compact layout at body-text size.
- Figure 4 axis-spacing correction, regenerated from the saved trajectories.

## Later work reserved

The alternate title, abstract opening, Section 3.2 opening, and conclusion opening remain in the exact saved later draft at `papers/archive/later_draft_20260923/files/papers/paper_draft.md`. They can be reconsidered against the next feedback. Most later edits have been retained or adapted; the reserved alternatives have not been deleted from the repository.

All pre-existing modified and untracked files were copied before this pass, with 27 file checksums and four deletions recorded. Unrelated experiment and repository-organization changes remain untouched in the working tree. No commit or stash was made.

## Verification

- All 13 final pages rendered and visually inspected, including formulas, figures, captions, tables, notes, and references.
- Bold vector symbols restored with an explicit bold math font version.
- Algorithm 1 and compact Figure 3 fit together; the previous nearly empty page is gone.
- Short formula lead-ins remain with their equations; table notes remain with their tables.
- Original title retained. All three table contents unchanged from the feedback-applied baseline.
- The two Markdown versions are text-identical after normalizing five image paths; corresponding figures are byte-identical.
- Notation checker passed for both Markdown versions.
- Five figures and 20 reference labels present in the searchable PDF.
- No missing characters, math-font fallback, undefined commands, or overfull horizontal boxes in the final LaTeX log.
- Snapshot checksums intact and Git HEAD unchanged. No optimizer reruns were performed.

## Diffs

- `changes_from_11c772e.diff`: approved professor-facing manuscript changes relative to the original feedback-applied text.
- `changes_from_later_draft.diff`: what was restored or shortened relative to the preserved later draft.

The rendering sources and preserved Figure 4 data are stored beside this record; the rebuild command is in `SOURCE.txt`.
