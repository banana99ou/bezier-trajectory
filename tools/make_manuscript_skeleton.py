#!/usr/bin/env python3
"""Build manuscript.docx from the KSAS template: styles intact, guide text gone.

The submitted file must contain none of the template's instruction sentences
("가이드 문장은 모두 삭제한다"), so they are stripped once, here, rather than
hunted down on the deadline. Everything that carries formatting is preserved by
*cloning* the template's own paragraphs — in particular paragraph 9, whose pPr
holds the sectPr that ends the single-column title block and starts the
two-column body. Rebuilding that from scratch loses the layout.

Idempotent: overwrites the output every run. It refuses to run against a
template it does not recognise, so a re-downloaded or edited template fails
loudly instead of silently producing a skeleton with the wrong indices.

Usage:
    python3 tools/make_manuscript_skeleton.py            # writes manuscript.docx
    python3 tools/make_manuscript_skeleton.py --force    # overwrite existing work
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "paper/ksas_2026_fall/template/ksas_2026_fall_template.docx"
OUTPUT = REPO / "paper/ksas_2026_fall/manuscript.docx"

# The template this script's paragraph indices were read off, 2026-08-20.
TEMPLATE_SHA256 = "ac1ee4987cf190063cbd5cd43cfd82f42c372baf0b4dc30f2fc6b86784e0ba79"

# What survives, in order. (template paragraph index, replacement text or None)
#   None  -> keep the paragraph exactly as the template has it
#   ""    -> keep the paragraph and its style, empty out the text
#   "..." -> keep the style, replace the text
#
# Paragraph 9 must stay: its pPr carries the single-column sectPr.
KEEP: list[tuple[int, str | None]] = [
    (0, "〈국문 제목〉"),
    (1, "〈저자명*〉"),
    (2, "〈소속〉"),
    (3, ""),
    (4, "〈English Title〉"),
    (5, "〈Author Name*〉"),
    (6, None),                 # ━━━ divider, verbatim
    (7, "Key Words : "),
    (8, ""),
    (9, ""),                   # <- carries the sectPr. Never drop.
    (12, "서 론"),
    (13, ""),
    (17, ""),
    (18, "본 론"),
    (19, ""),
    (22, ""),
    (36, "결 론"),
    (37, ""),
    (38, ""),
    (39, "후 기"),
    (40, ""),
    (41, ""),
    (42, "참고문헌"),
    (44, ""),
]

# Sanity anchors: index -> text the template must start with there.
ANCHORS = {
    0: "한국항공우주학회 2026 추계학술대회 논문 한글제목",
    9: "",
    12: "서 론",
    18: "본 론",
    36: "결 론",
    39: "후 기",
    42: "참고문헌",
    44: "1)Ahn",
}


def para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(f"{{{W}}}t"))


def set_text(p: ET.Element, text: str) -> None:
    """Replace a paragraph's content, keeping its pPr and its first run's rPr."""
    runs = p.findall(f"{{{W}}}r")
    rpr = None
    if runs:
        first_rpr = runs[0].find(f"{{{W}}}rPr")
        if first_rpr is not None:
            rpr = copy.deepcopy(first_rpr)
    for child in list(p):
        if child.tag != f"{{{W}}}pPr":
            p.remove(child)
    if text:
        run = ET.SubElement(p, f"{{{W}}}r")
        if rpr is not None:
            run.append(rpr)
        t = ET.SubElement(run, f"{{{W}}}t")
        t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def build_document_xml(src: bytes) -> bytes:
    root = ET.fromstring(src)
    body = root.find(f"{{{W}}}body")
    if body is None:
        raise SystemExit("template has no w:body")
    paras = list(body)

    for idx, expected in ANCHORS.items():
        got = para_text(paras[idx]).strip()
        if not got.startswith(expected):
            raise SystemExit(
                f"template paragraph {idx} reads {got[:40]!r}, expected {expected!r} — "
                "the template changed; re-read its structure before trusting these indices"
            )

    tail_sectpr = paras[-1]
    if tail_sectpr.tag != f"{{{W}}}sectPr":
        raise SystemExit("last body element is not the document sectPr")

    for child in list(body):
        body.remove(child)
    for idx, text in KEEP:
        p = copy.deepcopy(paras[idx])
        if text is not None:
            set_text(p, text)
        body.append(p)
    body.append(copy.deepcopy(tail_sectpr))

    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def main(argv: list[str]) -> int:
    force = "--force" in argv[1:]
    if not TEMPLATE.is_file():
        print(f"template missing: {TEMPLATE}", file=sys.stderr)
        return 1

    digest = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    if digest != TEMPLATE_SHA256:
        print(
            f"template sha256 is {digest[:16]}…, expected {TEMPLATE_SHA256[:16]}… — "
            "paragraph indices in this script were read off the 2026-08-20 download.\n"
            "Re-read the template structure before re-running.",
            file=sys.stderr,
        )
        return 1

    if OUTPUT.exists() and not force:
        print(
            f"{OUTPUT.relative_to(REPO)} already exists — refusing to overwrite drafted work.\n"
            "Pass --force if you really want a fresh skeleton.",
            file=sys.stderr,
        )
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TEMPLATE) as zin:
        new_doc = build_document_xml(zin.read("word/document.xml"))
        tmp = OUTPUT.with_suffix(".docx.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = new_doc if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)
    shutil.move(tmp, OUTPUT)
    print(f"wrote {OUTPUT.relative_to(REPO)}  ({len(KEEP)} paragraphs, guide text stripped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
