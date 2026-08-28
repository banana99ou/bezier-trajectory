#!/usr/bin/env python3
"""Surgical text edits on the manuscript .docx, so hand edits survive.

The alternative — regenerating the whole file from a text source — would silently
clobber anything typed in LibreOffice. Every operation here targets one paragraph
matched by its current text, so an edit that no longer applies fails loudly
instead of writing to the wrong place.

    # replace a paragraph's text (anchor must match exactly one paragraph)
    python3 tools/docx_edit.py FILE --set '〈국문 제목〉' '진짜 제목'

    # insert a new paragraph after an anchor, in a named template style
    python3 tools/docx_edit.py FILE --after '참고문헌' '참고문헌' '1)Osburn, C., ...'

Operations apply in the order given. --after inserts immediately after the anchor,
so repeated --after calls on the same anchor come out in reverse; anchor each one
on the line it should follow instead.
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
ET.register_namespace("w", W)


def text_of(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(f"{{{W}}}t")).strip()


def style_id_of(p: ET.Element) -> str | None:
    ppr = p.find(f"{{{W}}}pPr")
    if ppr is None:
        return None
    ps = ppr.find(f"{{{W}}}pStyle")
    return None if ps is None else ps.get(f"{{{W}}}val")


def set_text(p: ET.Element, text: str) -> None:
    """Replace paragraph content, keeping pPr and the first run's rPr.

    `^(1)` becomes a superscript run reading "(1)" — the citation form the KSAS
    template requires ("상첨자 ( )"). Everything else is written literally.
    """
    runs = p.findall(f"{{{W}}}r")
    rpr = None
    if runs:
        found = runs[0].find(f"{{{W}}}rPr")
        if found is not None:
            rpr = copy.deepcopy(found)
    for child in list(p):
        if child.tag != f"{{{W}}}pPr":
            p.remove(child)
    if not text:
        return

    for chunk, is_sup in split_superscripts(text):
        run = ET.SubElement(p, f"{{{W}}}r")
        run_pr = copy.deepcopy(rpr) if rpr is not None else None
        if is_sup:
            if run_pr is None:
                run_pr = ET.Element(f"{{{W}}}rPr")
            ET.SubElement(run_pr, f"{{{W}}}vertAlign").set(f"{{{W}}}val", "superscript")
        if run_pr is not None:
            run.append(run_pr)
        t = ET.SubElement(run, f"{{{W}}}t")
        t.text = chunk
        t.set(XML_SPACE, "preserve")


def split_superscripts(text: str) -> list[tuple[str, bool]]:
    """[(chunk, is_superscript)] — `^(1)` marks a superscript citation."""
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in re.finditer(r"\^\(([^)]*)\)", text):
        if m.start() > pos:
            out.append((text[pos : m.start()], False))
        out.append((f"({m.group(1)})", True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def find_one(body: ET.Element, anchor: str) -> tuple[int, ET.Element]:
    hits = [
        (i, el)
        for i, el in enumerate(body)
        if el.tag == f"{{{W}}}p" and text_of(el) == anchor
    ]
    if not hits:
        loose = [
            (i, el)
            for i, el in enumerate(body)
            if el.tag == f"{{{W}}}p" and anchor and anchor in text_of(el)
        ]
        if len(loose) == 1:
            return loose[0]
        if loose:
            raise SystemExit(
                f"anchor {anchor!r} is a substring of {len(loose)} paragraphs and an exact "
                "match for none — give more of the paragraph's text"
            )
        raise SystemExit(f"anchor not found: {anchor!r}")
    if len(hits) > 1:
        raise SystemExit(f"anchor matches {len(hits)} paragraphs, must match one: {anchor!r}")
    return hits[0]


def style_map(zf: zipfile.ZipFile) -> dict[str, str]:
    """style name -> styleId."""
    root = ET.fromstring(zf.read("word/styles.xml"))
    out = {}
    for style in root.iterfind(f"{{{W}}}style"):
        sid = style.get(f"{{{W}}}styleId")
        name = style.find(f"{{{W}}}name")
        if sid and name is not None:
            out[name.get(f"{{{W}}}val", sid)] = sid
    return out


def prototype_for(body: ET.Element, style_id: str) -> ET.Element | None:
    for el in body:
        if el.tag == f"{{{W}}}p" and style_id_of(el) == style_id:
            return el
    return None


def squeeze_blanks(body: ET.Element, styles: dict[str, str], keep_before_heading: bool = True) -> int:
    """Drop blank body paragraphs that only pad the layout.

    The 소제목 style carries no space-before, so ONE blank paragraph ahead of a
    section heading is doing real typographic work and stays. Every other blank
    is slack: a second blank in a run, a gap left between subsections, the
    trailing one at the end of the references.

    Never touched: a paragraph carrying the section break (`sectPr`) — that one
    keeps the title block out of the two-column body — or one holding a
    drawing, which is blank only in the sense of having no text.
    """
    heading = styles.get("소제목") if keep_before_heading else None
    kept_before_heading: set[int] = set()
    ps = [(i, el) for i, el in enumerate(body) if el.tag == f"{{{W}}}p"]
    for pos, (i, el) in enumerate(ps):
        if style_id_of(el) != heading:
            continue
        # walk back over the blank run and keep its LAST member
        j = pos - 1
        if j >= 0 and not text_of(ps[j][1]):
            kept_before_heading.add(ps[j][0])

    drop = []
    for i, el in ps:
        if text_of(el) or i in kept_before_heading:
            continue
        if el.find(f".//{{{W}}}sectPr") is not None:
            continue
        if el.find(f".//{{{W}}}drawing") is not None or el.find(f".//{{{W}}}pict") is not None:
            continue
        drop.append(el)
    for el in drop:
        body.remove(el)
    return len(drop)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("docx", type=Path)
    ap.add_argument("--set", nargs=2, action="append", metavar=("ANCHOR", "TEXT"), default=[])
    ap.add_argument(
        "--after", nargs=3, action="append", metavar=("ANCHOR", "STYLE", "TEXT"), default=[]
    )
    ap.add_argument("--delete", action="append", metavar="ANCHOR", default=[])
    ap.add_argument(
        "--tight",
        action="store_true",
        help="with --squeeze, drop the blank before section headings too",
    )
    ap.add_argument(
        "--squeeze",
        action="store_true",
        help="drop padding blank paragraphs, keeping one before each 소제목",
    )
    args = ap.parse_args(argv[1:])

    docx = args.docx.resolve()
    if not docx.is_file():
        print(f"no such file: {docx}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(docx) as zin:
        items = [(i, zin.read(i.filename)) for i in zin.infolist()]
        styles = style_map(zin)
        doc_xml = dict(  # noqa: C416 - explicit for clarity
            (i.filename, data) for i, data in items
        )["word/document.xml"]

    root = ET.fromstring(doc_xml)
    body = root.find(f"{{{W}}}body")
    if body is None:
        raise SystemExit("no w:body")

    applied = 0
    for anchor, text in args.set:
        _, p = find_one(body, anchor)
        set_text(p, text)
        applied += 1

    for anchor, style_name, text in args.after:
        idx, _ = find_one(body, anchor)
        sid = styles.get(style_name)
        if sid is None:
            raise SystemExit(f"no style named {style_name!r} in this document")
        proto = prototype_for(body, sid)
        if proto is None:
            new = ET.Element(f"{{{W}}}p")
            ppr = ET.SubElement(new, f"{{{W}}}pPr")
            ET.SubElement(ppr, f"{{{W}}}pStyle").set(f"{{{W}}}val", sid)
        else:
            new = copy.deepcopy(proto)
        set_text(new, text)
        body.insert(idx + 1, new)
        applied += 1

    for anchor in args.delete:
        _, target = find_one(body, anchor)
        if target.find(f".//{{{W}}}sectPr") is not None:
            raise SystemExit(
                f"refusing to delete {anchor!r}: it carries the section break that keeps "
                "the title block out of the two-column body"
            )
        if target.find(f".//{{{W}}}drawing") is not None:
            raise SystemExit(f"refusing to delete {anchor!r}: it holds a figure")
        body.remove(target)
        applied += 1

    if args.squeeze:
        n = squeeze_blanks(body, styles, keep_before_heading=not args.tight)
        print(f"squeezed {n} blank paragraph(s)")
        applied += n

    new_doc = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    tmp = docx.with_suffix(".docx.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in items:
            zout.writestr(info, new_doc if info.filename == "word/document.xml" else data)
    shutil.move(tmp, docx)
    print(f"applied {applied} edit(s) to {docx.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
