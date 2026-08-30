#!/usr/bin/env python3
"""Render a .docx manuscript into a diff-able .md sidecar and a .pdf.

The sidecar exists so VS Code can diff the manuscript. A .docx is a zip of XML,
so git shows nothing useful; the sidecar is plain text and diffs line by line.

Every paragraph is prefixed with its *template style name* — not the styleId.
The KSAS template names its styles in Korean ("초록 및 본문 내용", "그림표캡션")
but stores them under opaque ids ("ab", "ad"), so a paragraph that silently
loses its style shows up in the diff as a changed tag.

The header block carries the source file's hash and mtime. If the watcher dies,
the sidecar keeps the *old* hash while the docx moves on — a dead watcher is
visible instead of silent.

Usage:
    python3 tools/render_paper.py paper/ksas_2026_fall/manuscript.docx

Requires: LibreOffice (`soffice`) for the PDF. Everything else is stdlib.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W, "m": M}

# soffice refuses to run headless against a profile a GUI instance holds open.
SOFFICE_PROFILE = "file:///tmp/claude-soffice-profile"


def style_names(zf: zipfile.ZipFile) -> dict[str, str]:
    """styleId -> human style name, as the template spells it."""
    try:
        root = ET.fromstring(zf.read("word/styles.xml"))
    except KeyError:
        return {}
    out = {}
    for style in root.iterfind(f"{{{W}}}style"):
        sid = style.get(f"{{{W}}}styleId")
        name = style.find(f"{{{W}}}name")
        if sid and name is not None:
            out[sid] = name.get(f"{{{W}}}val", sid)
    return out


def paragraph_text(p: ET.Element) -> str:
    """Text of one paragraph, including math and a marker for images."""
    parts: list[str] = []
    for node in p.iter():
        tag = node.tag
        if tag == f"{{{W}}}t" or tag == f"{{{M}}}t":
            parts.append(node.text or "")
        elif tag == f"{{{W}}}tab":
            parts.append("\t")
        elif tag == f"{{{W}}}br":
            parts.append(" / ")
        elif tag == f"{{{W}}}drawing" or tag == f"{{{W}}}pict":
            parts.append("〔그림〕")
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def paragraph_style(p: ET.Element, names: dict[str, str]) -> str:
    ppr = p.find(f"{{{W}}}pPr")
    if ppr is not None:
        ps = ppr.find(f"{{{W}}}pStyle")
        if ps is not None:
            sid = ps.get(f"{{{W}}}val", "")
            return names.get(sid, f"?{sid}")
    return "기본"


def render_body(zf: zipfile.ZipFile) -> list[str]:
    """Document body in reading order, one line per paragraph or table row."""
    root = ET.fromstring(zf.read("word/document.xml"))
    body = root.find(f"{{{W}}}body")
    names = style_names(zf)
    lines: list[str] = []
    if body is None:
        return lines

    for el in body:
        if el.tag == f"{{{W}}}p":
            lines.append(f"[{paragraph_style(el, names)}] {paragraph_text(el)}".rstrip())
        elif el.tag == f"{{{W}}}tbl":
            lines.append("[표 시작]")
            for row in el.iterfind(f"{{{W}}}tr"):
                cells = [
                    " ".join(paragraph_text(p) for p in cell.iterfind(f"{{{W}}}p")).strip()
                    for cell in row.iterfind(f"{{{W}}}tc")
                ]
                lines.append("[표] | " + " | ".join(cells) + " |")
            lines.append("[표 끝]")
    return lines


def make_pdf(docx: Path) -> tuple[Path | None, str]:
    """Convert to PDF next to the source. Returns (path, note)."""
    if shutil.which("soffice") is None:
        return None, "soffice not on PATH — PDF not rendered"
    # Export filter options as JSON (LibreOffice >= 7.4). The defaults recompress
    # every image as JPEG and downsample it to 300 dpi -- measured on the
    # manuscript: a 1360x620 PNG came out as a 944x430 JPEG. The figure is
    # authored at 600 dpi so that the printed column is crisp; the exporter must
    # not undo that.
    pdf_filter = (
        'pdf:writer_pdf_Export:{'
        '"UseLosslessCompression":{"type":"boolean","value":"true"},'
        '"ReduceImageResolution":{"type":"boolean","value":"false"}}'
    )
    cmd = [
        "soffice",
        f"-env:UserInstallation={SOFFICE_PROFILE}",
        "--headless",
        "--convert-to",
        pdf_filter,
        str(docx),
        "--outdir",
        str(docx.parent),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None, "soffice timed out after 180 s"
    pdf = docx.with_suffix(".pdf")
    if proc.returncode != 0 or not pdf.exists():
        tail = (proc.stderr or proc.stdout).decode("utf8", "replace").strip()[-200:]
        return None, f"soffice failed: {tail or 'no output'}"
    return pdf, ""


def page_count(pdf: Path) -> int | None:
    """Page count, cross-checked two ways; None when they disagree."""
    data = pdf.read_bytes()
    by_type = len(re.findall(rb"/Type\s*/Page[^s]", data))
    counts = [int(n) for n in re.findall(rb"/Type\s*/Pages.{0,200}?/Count\s+(\d+)", data, re.S)]
    by_count = max(counts) if counts else None
    if by_count is not None and by_count != by_type:
        return None
    return by_type or by_count


# --- author critique markers ------------------------------------------------
#
# The sidecar is regenerated on every save, and the author annotates it in
# place: `(원문 조각){비평}`, or a bare `{비평}` hung off the end of a phrase.
# A plain rewrite throws those away — it did, on 2026-08-27, and the critiques
# were only recoverable by hand.
#
# So the renderer reads the annotations out of the previous sidecar and puts
# them back on the freshly rendered text. Two rules, and they are the whole
# design:
#
#   1. Nothing here ever deletes a critique. A resolved one is removed by hand.
#   2. An annotation whose span no longer appears is NOT dropped. The source
#      sentence changed, which is exactly what the author needs to see, so it
#      is carried into a block at the end of the file instead.
BARE_ANCHOR_LEN = 40
CARRY_HEADING = "## 자리를 찾지 못한 비평 — 본문이 바뀌었습니다"


def _match_open(s: str) -> int | None:
    """Index of the '(' matching the ')' that ends `s`, or None."""
    depth = 0
    for k in range(len(s) - 1, -1, -1):
        if s[k] == ")":
            depth += 1
        elif s[k] == "(":
            depth -= 1
            if depth == 0:
                return k
    return None


EQUATION_TAG = "[수식]"


def _split_line(line: str) -> tuple[str, list[tuple[str, str, str]]]:
    """One paragraph line: strip its markers, return the plain text and them."""
    plain = ""
    annots: list[tuple[str, str, str]] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch != "{":
            plain += ch
            i += 1
            continue
        j = line.find("}", i + 1)
        if j == -1:  # an unpaired brace is ordinary text
            plain += ch
            i += 1
            continue
        comment = line[i + 1 : j]
        if plain.endswith(")"):
            k = _match_open(plain)
            if k is not None:
                span = plain[k + 1 : -1]
                plain = plain[:k] + span
                annots.append(("span", span, comment))
                i = j + 1
                continue
        annots.append(("tail", plain[-BARE_ANCHOR_LEN:], comment))
        i = j + 1
    return plain, annots


def split_annotations(body: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Strip the markers. Returns the plain body and (kind, anchor, comment).

    Equation paragraphs are skipped whole. Braces are set-builder notation
    there, and reading `b = max { n·x : x in L }` as a critique both invented
    an annotation and would have deleted the braces from the manuscript on the
    next render."""
    plains: list[str] = []
    annots: list[tuple[str, str, str]] = []
    for line in body.split("\n"):
        if line.startswith(EQUATION_TAG):
            plains.append(line)
            continue
        plain, found = _split_line(line)
        plains.append(plain)
        annots.extend(found)
    return "\n".join(plains), annots


def _hangul(ch: str) -> bool:
    return "\uac00" <= ch <= "\ud7a3"


def _mid_word(body: str, hit: int, anchor: str) -> bool:
    """True when the match starts inside a longer Korean word.

    Short anchors are the danger: `집합` re-placed itself inside `교집합`,
    splitting the word. Korean has no space before a particle, so only the
    LEADING edge can be tested — a trailing 이지만 after `연결 성분` is a
    legitimate match and must not be rejected."""
    return hit > 0 and _hangul(anchor[0]) and _hangul(body[hit - 1])


def reapply_annotations(
    body: str, annots: list[tuple[str, str, str]]
) -> tuple[str, list[tuple[str, str, str]]]:
    """Put the markers back. Returns the annotated body and what did not fit."""
    edits: list[tuple[int, str, int, str]] = []
    used: list[tuple[int, int]] = []
    missing: list[tuple[int, tuple[str, str, str]]] = []
    # Longest anchor first, so a critique on a whole paragraph claims it before
    # a one-word critique can land inside that paragraph. Without the ordering
    # the two compete and the winner changes from render to render.
    order = sorted(range(len(annots)), key=lambda i: -len(annots[i][1]))
    for idx in order:
        kind, anchor, comment = annots[idx]
        pos, start = -1, 0
        while anchor:
            hit = body.find(anchor, start)
            if hit == -1:
                break
            free = not any(a < hit + len(anchor) and hit < b for a, b in used)
            if free and not _mid_word(body, hit, anchor):
                pos = hit
                break
            start = hit + 1
        if pos == -1:
            missing.append((idx, (kind, anchor, comment)))
            continue
        end = pos + len(anchor)
        used.append((pos, end))
        opener = "(" if kind == "span" else ""
        closer = ("){" if kind == "span" else "{") + comment + "}"
        edits.append((pos, opener, end, closer))
    for pos, opener, end, closer in sorted(edits, key=lambda e: -e[0]):
        body = body[:pos] + opener + body[pos:end] + closer + body[end:]
    return body, [a for _, a in sorted(missing)]


def carry_block(missing: list[tuple[str, str, str]], strays: list[str]) -> list[str]:
    if not missing and not strays:
        return []
    out = ["", "---", "", CARRY_HEADING, ""]
    for kind, anchor, comment in missing:
        # Written back in the marker's own syntax, so the next render recovers
        # it with `split_annotations` and re-places it verbatim the moment the
        # author restores the sentence it belonged to.
        marker = f"({anchor})" if kind == "span" else anchor
        out.append(f"- `{marker}{{{comment}}}`")
    out.extend(strays)
    return out


def previous_body(md: Path) -> str:
    """The body of an existing sidecar — everything past the header rule."""
    if not md.is_file():
        return ""
    lines = md.read_text(encoding="utf8").splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            return "\n".join(lines[idx + 1 :])
    return ""


def previous_annotations(previous: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Every critique the last sidecar carried, from the body AND from the
    carry block, plus any carry line too malformed to parse.

    The carry block is read back on purpose: a critique that could not be
    placed last time gets another chance this time — the author may have put
    the sentence back. It is parsed strictly, one marker per `- ` + backticks
    line, so a carried entry cannot decay into a longer and longer anchor by
    being re-read as ordinary text. Anything that fails that parse is handed
    back verbatim rather than dropped."""
    cut = previous.find(CARRY_HEADING)
    body = previous if cut == -1 else previous[:cut]
    _, annots = split_annotations(body)
    strays: list[str] = []
    if cut != -1:
        for line in previous[cut:].splitlines()[1:]:
            if not line.strip():
                continue
            match = re.match(r"^- `(.*)`\s*$", line)
            if match is None:
                strays.append(line)
                continue
            _, carried = split_annotations(match.group(1))
            if carried:
                annots.extend(carried)
            else:
                strays.append(line)
    return annots, strays


def build_sidecar(docx: Path, previous: str = "") -> str:
    raw = docx.read_bytes()
    with zipfile.ZipFile(docx) as zf:
        lines = render_body(zf)

    pdf, note = make_pdf(docx)
    if pdf is None:
        pages = f"unknown — {note}"
    else:
        n = page_count(pdf)
        pages = str(n) if n is not None else "unknown — PDF page count is ambiguous"

    mtime = datetime.fromtimestamp(docx.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    head = [
        f"# {docx.name} — rendered view",
        "",
        "Generated by `tools/render_paper.py`. Edit the .docx, not the prose here —",
        "but `(원문){비평}` markers written into this file survive every re-render",
        "and are only ever removed by hand.",
        "Each line is one paragraph, prefixed with its template style name.",
        "",
        f"- source: `{docx.name}`",
        f"- source sha256: `{hashlib.sha256(raw).hexdigest()[:16]}`",
        f"- source modified: {mtime}",
        f"- pages: {pages}",
        f"- rendered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    annots, strays = previous_annotations(previous)
    body, missing = reapply_annotations("\n".join(lines), annots)
    return "\n".join(head + body.split("\n") + carry_block(missing, strays)) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print(f"usage: {Path(argv[0]).name} <manuscript.docx>", file=sys.stderr)
        return 2
    docx = Path(argv[1]).resolve()
    if not docx.is_file():
        print(f"no such file: {docx}", file=sys.stderr)
        return 1

    md = docx.with_suffix(".md")
    md.write_text(build_sidecar(docx, previous_body(md)), encoding="utf8")

    pages = next(
        (ln.split(": ", 1)[1] for ln in md.read_text(encoding="utf8").splitlines()
         if ln.startswith("- pages: ")),
        "?",
    )
    print(f"{datetime.now().strftime('%H:%M:%S')}  {md.name} + {docx.stem}.pdf  ({pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
