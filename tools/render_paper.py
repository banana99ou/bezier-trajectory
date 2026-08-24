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
    cmd = [
        "soffice",
        f"-env:UserInstallation={SOFFICE_PROFILE}",
        "--headless",
        "--convert-to",
        "pdf",
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


def build_sidecar(docx: Path) -> str:
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
        "Generated by `tools/render_paper.py`. **Do not edit this file** — edit the .docx.",
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
    return "\n".join(head + lines) + "\n"


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
    md.write_text(build_sidecar(docx), encoding="utf8")

    pages = next(
        (ln.split(": ", 1)[1] for ln in md.read_text(encoding="utf8").splitlines()
         if ln.startswith("- pages: ")),
        "?",
    )
    print(f"{datetime.now().strftime('%H:%M:%S')}  {md.name} + {docx.stem}.pdf  ({pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
