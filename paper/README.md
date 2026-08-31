# Writing a paper in this repository

Everything under `paper/` is a manuscript in a society's own template, plus the machinery that
makes a binary template file behave like source. One directory per venue
(`ksas_2026_fall/`); this file is what applies to all of them.

**This file is venue-agnostic machinery.** What a given paper argues lives with its idea
([`idea/spacetime.md`](../idea/spacetime.md)); the deadline, that society's rules, which figures go
in and what state the draft is in live in the artifact's own README next to its manuscript
([`ksas_2026_fall/README.md`](ksas_2026_fall/README.md)). This file is only how to get words into
the file the society accepts.

## The writing loop

```bash
./tools/watch_paper.sh          # leave running in a terminal tab
```

Then edit the manuscript with `tools/docx_edit.py` (below) or by hand in LibreOffice. Every save
re-renders two things beside the `.docx`:

- **`manuscript.md`** — one line per paragraph, prefixed with its **template style name**. This is
  what `git diff` and the VS Code diff view show you. It is a render: **never edit it.**
- **`manuscript.pdf`** — the real layout, and the only honest source of a page count.

Tracked in git: `.md` and `.docx`. Ignored: `.pdf`, regenerable at any time.

The sidecar's header carries the source's sha256 and mtime. If the watcher dies, the sidecar keeps
the old hash while the `.docx` moves on — a dead watcher becomes visible in the diff instead of
silently serving you a stale page count.

## The four tools, and what each one actually guarantees

| tool | guarantee |
|---|---|
| [`tools/render_paper.py`](../tools/render_paper.py) | Resolves style **names**, not the opaque ids (`ab`, `ad`) the template stores, so a paragraph that loses its style shows up as a changed tag. Page count comes from the rendered PDF, cross-checked two ways, and reports `unknown` rather than guess when they disagree. |
| [`tools/watch_paper.sh`](../tools/watch_paper.sh) | fswatch on the manuscript's directory, with a **content-hash guard**: the `.md` and `.pdf` it writes land in that same watched directory, and without the guard each render would trigger the next one forever. |
| [`tools/make_manuscript_skeleton.py`](../tools/make_manuscript_skeleton.py) | Rebuilds the empty manuscript from the template with every guide sentence stripped. Pinned to the template's sha256 and refuses to run against a template it does not recognise; refuses to overwrite drafted work without `--force`. |
| [`tools/docx_edit.py`](../tools/docx_edit.py) | Surgical edits, so hand edits in LibreOffice survive. An anchor must match **exactly one** paragraph or the tool exits. `^(1)` in the text becomes a real superscript run — the citation form the template requires. |

```bash
python3 tools/docx_edit.py paper/ksas_2026_fall/manuscript.docx \
  --set '〈국문 제목〉' '실제 제목' \
  --after '참고문헌' '참고문헌' '1)Author, A. B., “Title,” Venue, 2025.'
```

## Invariants — what silently breaks if you are careless

**The title block's single-column section hangs off one paragraph.** In the KSAS template, body
paragraph index 9 carries the `sectPr` that *ends* the single-column title block; the two-column
setting is the document-level `sectPr` at the end of the body. Delete paragraph 9 and the
two-column body survives — but it now starts at the top of the document, so the title, authors and
Key Words get split across two narrow columns. The page count does not change (still 2), which is
why the render alone will not tell you. `make_manuscript_skeleton.py` clones that paragraph
deliberately. If you write your own transform, assert that **both** `<w:cols w:space="0">` and
`<w:cols w:num="2">` survive — measured 2026-08-21 by deleting it and re-rendering.

**The skeleton's paragraph indices belong to one specific template file.** They were read off the
2026-08-20 download. The sha256 pin is what turns "the society re-uploaded the template" into a
loud failure instead of a scrambled manuscript.

## Where the rest went

| you want | read |
|---|---|
| what the paper claims, and whether it is novel | [`idea/spacetime.md`](../idea/spacetime.md) |
| 제출 일정, 분량, venue rules, manuscript state, poster | [`ksas_2026_fall/README.md`](ksas_2026_fall/README.md) |
| any number that could go in a paper | [`SOLVER.md`](../SOLVER.md) §Measurements |

**A number that is not in a figure-grade sidecar does not go in a paper.** `figure_grade` means
converged, certificate ≤ 1e-6, clearance > 0, slack ≤ 1e-6; `tools/make_paper_figure.py` refuses to
draw a run that fails it.
