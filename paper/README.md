# Writing a paper in this repository

Everything under `paper/` is a manuscript in a society's own template, plus the machinery that
makes a binary template file behave like source. One directory per venue
(`ksas_2026_fall/`); this file is what applies to all of them.

The claims themselves are not here. [`PAPER_1.md`](../PAPER_1.md) carries what the paper argues and
why; this file carries how to get it into the file the society accepts.

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

## Venue rules that change what you write

Full verified list, with source URLs and fetch date:
[`ksas_2026_fall/template/PROVENANCE.md`](ksas_2026_fall/template/PROVENANCE.md). The ones that
constrain the writing rather than the formatting:

- **참고문헌 5개 이내**, all in English. Related work has to make its whole case in five citations.
- The **400자 초록** is typed into the submission web page, **not** the manuscript. Separate
  deliverable, same deadline.
- Citations are **superscript numbers in parentheses**; captions in English, table above, figure
  below; every template guide sentence deleted before submission.
- Submitting requires a **paid membership and completed 사전등록 결제** — the registration payment
  is due before the paper deadline, not before the 사전등록 deadline.
- A **late oral request is demoted to poster**. Submitting early is what buys the oral slot.

## Two things not to assert

1. **No page limit is stated** for the regular conference. Two pages is the length of the template
   file, nothing more. The "A4 4쪽 이내" on the same page belongs to the separate 산업·정책
   경진대회, a different submission path. Write to two pages by choice; do not cite it as a rule.
2. **`ksas.or.kr` has no working HTTPS.** The TLS handshake hangs, port 443 accepts and then dies.
   `WebFetch` force-upgrades http to https, so it can never reach this site. Use
   `curl -4` over `http://`.

## Who wins when sources disagree

| question | authority |
|---|---|
| what the paper claims | [`PAPER_1.md`](../PAPER_1.md) Part B §핵심 기여 |
| whether something is novel | [`doc/refs/novelty_positioning.md`](../doc/refs/novelty_positioning.md) |
| scope, venue, objective function | [`doc/refs/advisor_review_20260820.md`](../doc/refs/advisor_review_20260820.md) |
| any number | [`README.md`](../README.md) §Measurements and `figures/paper1/*.json` |

**A number that is not in a figure-grade sidecar does not go in the paper.** `figure_grade` means
converged, certificate ≤ 1e-6, clearance > 0, slack ≤ 1e-6; `tools/make_paper_figure.py` refuses to
draw a run that fails it.

The advisor's instruction from the 2026-08-20 review is already applied and must not be undone:
**장애물 페널티 항은 목적함수에서 뺀다.** §2.2 of the manuscript states that avoidance is the
supporting half-space constraint and that the SCP slack variables are a numerical device, not the
avoidance mechanism.

## State of the KSAS manuscript, 2026-08-21

Written: title block, 서론 (three paragraphs), 본론 2.1–2.5, 결론, five references. Two pages.

Open, each needing the author rather than an agent:

- **후 기 is empty** — funding and acknowledgment text.
- **The figure is not in the document.** It exists and is figure-grade —
  `figures/paper1/occlusion_figure.png`, sidecar `occlusion_figure.json` recording LOS margin
  +0.3379, clearance +1.1625, occlusion certificate 0.0, 124 iterations, git `950a5ec`. Placing it
  will cost space the current two pages do not have.
- **The mission motivation for the line-of-sight constraint is missing.** The advisor's review
  requires it in both manuscript and talk; the constraint is currently presented only as something
  convex-decomposition methods cannot express.
- **Author names, affiliation and romanization are unverified guesses** made by an agent from
  repository filenames. Confirm every one before submission.
