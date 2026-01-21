#!/usr/bin/env python3
"""
Compare two artifact snapshot directories and write a single markdown report with:
- SHA256 + file sizes for all artifacts
- PNG pixel-diff images + summary (when PNGs exist)
- Cache (.pkl) load summary (type/shape/keys)
- CSV diff summary (if any)

This script is dependency-free beyond the standard library + numpy (already used by this repo).
"""

import argparse
import csv
import datetime as _dt
import hashlib
import os
import pickle
import struct
import textwrap
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sizeof(path: Path) -> int:
    return path.stat().st_size


def _iter_files(root: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(root)).replace("\\", "/")
            files[rel] = p
    return files


def _mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _paeth(a: int, b: int, c: int) -> int:
    # Paeth predictor (per PNG spec)
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _read_png_rgba_bytes(path: Path) -> Tuple[int, int, bytes]:
    """
    Minimal PNG decoder for non-interlaced 8-bit RGB/RGBA/Grayscale.
    Returns (width, height, rgba_bytes) where rgba_bytes is length width*height*4.
    """
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file (bad signature)")
    i = len(PNG_SIGNATURE)

    width = height = None
    bit_depth = color_type = interlace = None
    idat_parts: List[bytes] = []

    while i < len(data):
        if i + 8 > len(data):
            break
        length = struct.unpack(">I", data[i : i + 4])[0]
        tag = data[i + 4 : i + 8]
        i += 8
        chunk_data = data[i : i + length]
        i += length
        _crc = data[i : i + 4]
        i += 4

        if tag == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif tag == b"IDAT":
            idat_parts.append(chunk_data)
        elif tag == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("PNG missing IHDR")
    if interlace != 0:
        raise ValueError("Interlaced PNG not supported")
    if bit_depth != 8:
        raise ValueError(f"Only 8-bit PNG supported, got bit_depth={bit_depth}")
    if color_type not in (0, 2, 6):
        raise ValueError(f"Unsupported color_type={color_type} (supported: 0,2,6)")

    raw = zlib.decompress(b"".join(idat_parts))

    if color_type == 0:
        channels = 1
    elif color_type == 2:
        channels = 3
    else:
        channels = 4

    bpp = channels  # bytes per pixel (since bit_depth==8)
    stride = width * bpp
    expected = (stride + 1) * height
    if len(raw) != expected:
        raise ValueError(f"Unexpected decompressed size: got {len(raw)}, expected {expected}")

    prev = bytearray(stride)

    pos = 0
    rgba_out = bytearray(width * height * 4)
    out_pos = 0
    for y in range(height):
        filt = raw[pos]
        pos += 1
        scan = raw[pos : pos + stride]
        pos += stride

        if filt == 0:
            recon = bytearray(scan)
        elif filt == 1:  # Sub
            recon = bytearray(scan)
            for i in range(bpp, stride):
                recon[i] = (recon[i] + recon[i - bpp]) & 0xFF
        elif filt == 2:  # Up
            recon = bytearray(stride)
            for i in range(stride):
                recon[i] = (scan[i] + prev[i]) & 0xFF
        elif filt == 3:  # Average
            recon = bytearray(scan)
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                up = prev[i]
                recon[i] = (recon[i] + ((left + up) // 2)) & 0xFF
        elif filt == 4:  # Paeth
            recon = bytearray(scan)
            for i in range(stride):
                a = recon[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                recon[i] = (recon[i] + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter type: {filt}")

        prev = recon

        # Expand to RGBA into output buffer
        if channels == 4:
            rgba_out[out_pos : out_pos + width * 4] = recon
            out_pos += width * 4
        elif channels == 3:
            for x in range(width):
                base = x * 3
                rgba_out[out_pos + 0] = recon[base + 0]
                rgba_out[out_pos + 1] = recon[base + 1]
                rgba_out[out_pos + 2] = recon[base + 2]
                rgba_out[out_pos + 3] = 255
                out_pos += 4
        else:  # grayscale
            for x in range(width):
                g = recon[x]
                rgba_out[out_pos + 0] = g
                rgba_out[out_pos + 1] = g
                rgba_out[out_pos + 2] = g
                rgba_out[out_pos + 3] = 255
                out_pos += 4

    return width, height, bytes(rgba_out)


def _write_png_rgba_bytes(path: Path, width: int, height: int, rgba: bytes) -> None:
    """
    Minimal PNG writer for 8-bit RGBA, non-interlaced.
    """
    if len(rgba) != width * height * 4:
        raise ValueError("rgba must be width*height*4 bytes")

    # Build filtered scanlines using filter 0 (None)
    raw_lines = []
    for y in range(height):
        start = y * width * 4
        raw_lines.append(b"\x00" + rgba[start : start + width * 4])
    compressed = zlib.compress(b"".join(raw_lines), level=6)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = bytearray()
    png += PNG_SIGNATURE
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    path.write_bytes(bytes(png))


def _summarize_obj(obj, max_items: int = 50, max_str: int = 200) -> str:
    def _type(o) -> str:
        return type(o).__name__

    # numpy (optional)
    try:
        import numpy as _np  # type: ignore

        if isinstance(obj, _np.ndarray):
            return f"ndarray shape={obj.shape} dtype={obj.dtype}"
    except Exception:
        pass

    # dict-like
    if isinstance(obj, dict):
        keys = list(obj.keys())
        parts = [f"dict nkeys={len(keys)}"]
        for k in keys[:max_items]:
            v = obj[k]
            parts.append(f"- {repr(k)[:max_str]}: {_summarize_obj(v, max_items=10, max_str=max_str)}")
        if len(keys) > max_items:
            parts.append(f"- ... ({len(keys) - max_items} more keys)")
        return "\n".join(parts)

    # list/tuple
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        parts = [f"{_type(obj)} len={n}"]
        for idx, v in enumerate(obj[: min(n, max_items)]):
            parts.append(f"- [{idx}]: {_summarize_obj(v, max_items=10, max_str=max_str)}")
        if n > max_items:
            parts.append(f"- ... ({n - max_items} more items)")
        return "\n".join(parts)

    # scalar-ish
    r = repr(obj)
    if len(r) > max_str:
        r = r[: max_str - 3] + "..."
    return f"{_type(obj)} {r}"


def _load_pickle_summary(path: Path) -> Tuple[bool, str]:
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        return True, _summarize_obj(obj)
    except Exception as e:
        return False, f"ERROR {type(e).__name__}: {e}"


@dataclass(frozen=True)
class FileCompareRow:
    relpath: str
    exists_a: bool
    exists_b: bool
    sha_a: Optional[str]
    sha_b: Optional[str]
    size_a: Optional[int]
    size_b: Optional[int]
    same: bool


def _compare_files(a_root: Path, b_root: Path) -> List[FileCompareRow]:
    a = _iter_files(a_root)
    b = _iter_files(b_root)
    all_rel = sorted(set(a.keys()) | set(b.keys()))
    rows: List[FileCompareRow] = []
    for rel in all_rel:
        pa = a.get(rel)
        pb = b.get(rel)
        exists_a = pa is not None
        exists_b = pb is not None
        sha_a = _sha256_file(pa) if pa else None
        sha_b = _sha256_file(pb) if pb else None
        size_a = _sizeof(pa) if pa else None
        size_b = _sizeof(pb) if pb else None
        same = (exists_a and exists_b and sha_a == sha_b and size_a == size_b)
        rows.append(
            FileCompareRow(
                relpath=rel,
                exists_a=exists_a,
                exists_b=exists_b,
                sha_a=sha_a,
                sha_b=sha_b,
                size_a=size_a,
                size_b=size_b,
                same=same,
            )
        )
    return rows


def _pixel_diff_png(a_png: Path, b_png: Path, out_diff_png: Path) -> Dict[str, object]:
    wa, ha, ra = _read_png_rgba_bytes(a_png)
    wb, hb, rb = _read_png_rgba_bytes(b_png)
    if wa != wb or ha != hb:
        return {
            "ok": False,
            "reason": f"dimension mismatch: A={wa}x{ha} B={wb}x{hb}",
        }
    n_pixels = wa * ha
    n_diff = 0
    max_diff = 0
    sum_diff = 0
    vis = bytearray(n_pixels * 4)
    for i in range(n_pixels):
        base = i * 4
        dr = abs(ra[base + 0] - rb[base + 0])
        dg = abs(ra[base + 1] - rb[base + 1])
        db = abs(ra[base + 2] - rb[base + 2])
        intensity = dr if dr >= dg and dr >= db else (dg if dg >= db else db)
        if intensity != 0:
            n_diff += 1
        if intensity > max_diff:
            max_diff = intensity
        sum_diff += intensity
        vis[base + 0] = intensity
        vis[base + 1] = 0
        vis[base + 2] = 0
        vis[base + 3] = 255
    mean_diff = (sum_diff / n_pixels) if n_pixels else 0.0

    _mkdirp(out_diff_png.parent)
    _write_png_rgba_bytes(out_diff_png, wa, ha, bytes(vis))
    return {
        "ok": True,
        "width": wa,
        "height": ha,
        "n_diff_pixels": n_diff,
        "max_channel_diff": max_diff,
        "mean_channel_diff": mean_diff,
        "diff_image": str(out_diff_png),
    }


def _csv_diff_summary(a_csv: Path, b_csv: Path, tol: float = 0.0) -> Dict[str, object]:
    # Basic diff: row/col counts + mismatches
    def read_rows(p: Path) -> List[List[str]]:
        with p.open("r", newline="") as f:
            return list(csv.reader(f))

    ra = read_rows(a_csv)
    rb = read_rows(b_csv)
    nra, nrb = len(ra), len(rb)
    nca = max((len(r) for r in ra), default=0)
    ncb = max((len(r) for r in rb), default=0)

    mismatches = 0
    numeric_mismatches = 0
    compared = 0

    def to_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    for i in range(max(nra, nrb)):
        row_a = ra[i] if i < nra else None
        row_b = rb[i] if i < nrb else None
        if row_a is None or row_b is None:
            mismatches += 1
            continue
        for j in range(max(len(row_a), len(row_b))):
            ca = row_a[j] if j < len(row_a) else None
            cb = row_b[j] if j < len(row_b) else None
            compared += 1
            if ca == cb:
                continue
            fa = to_float(ca) if ca is not None else None
            fb = to_float(cb) if cb is not None else None
            if fa is not None and fb is not None and tol > 0:
                if abs(fa - fb) <= tol:
                    continue
                numeric_mismatches += 1
            mismatches += 1

    return {
        "rows_a": nra,
        "rows_b": nrb,
        "max_cols_a": nca,
        "max_cols_b": ncb,
        "compared_cells": compared,
        "mismatches": mismatches,
        "numeric_mismatches": numeric_mismatches,
        "tolerance": tol,
    }


def _find_latest_snapshot(artifacts_dir: Path, prefixes: Sequence[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for p in artifacts_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if any(name.startswith(pref) for pref in prefixes):
            candidates.append(p)
    if not candidates:
        # Also support nested layout like refactor_modular-structure/<timestamp>/
        for p in artifacts_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if any(name.startswith(pref.rstrip("/")) for pref in prefixes) and any(c.is_dir() for c in p.iterdir()):
                # pick latest child directory
                subdirs = [c for c in p.iterdir() if c.is_dir()]
                if subdirs:
                    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    candidates.append(subdirs[0])
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _write_report(
    report_path: Path,
    a_root: Path,
    b_root: Path,
    rows: List[FileCompareRow],
    png_diffs: Dict[str, Dict[str, object]],
    pkl_summaries: Dict[str, Dict[str, object]],
    csv_summaries: Dict[str, Dict[str, object]],
    diffs_dir: Path,
) -> None:
    exact = sum(1 for r in rows if r.same)
    only_a = sum(1 for r in rows if r.exists_a and not r.exists_b)
    only_b = sum(1 for r in rows if r.exists_b and not r.exists_a)
    both_diff = sum(1 for r in rows if (r.exists_a and r.exists_b and not r.same))

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# Artifact comparison report\n\nGenerated: `{now}`\n")
    lines.append("## Inputs\n")
    lines.append(f"- **A**: `{a_root}`\n")
    lines.append(f"- **B**: `{b_root}`\n")
    lines.append(f"- **Diff images dir**: `{diffs_dir}`\n")
    lines.append("\n## High-level summary\n")
    lines.append(f"- **Exact matches**: {exact}\n")
    lines.append(f"- **Different (both exist)**: {both_diff}\n")
    lines.append(f"- **Only in A**: {only_a}\n")
    lines.append(f"- **Only in B**: {only_b}\n")

    lines.append("\n## Checksums + sizes (all files)\n")
    lines.append("| file | status | size(A) | size(B) | sha256(A) | sha256(B) |\n")
    lines.append("|---|---:|---:|---:|---|---|\n")
    for r in rows:
        if r.same:
            status = "MATCH"
        elif not r.exists_a:
            status = "ONLY_B"
        elif not r.exists_b:
            status = "ONLY_A"
        else:
            status = "DIFF"
        lines.append(
            f"| `{r.relpath}` | {status} | {r.size_a if r.size_a is not None else ''} | {r.size_b if r.size_b is not None else ''} | "
            f"`{r.sha_a or ''}` | `{r.sha_b or ''}` |\n"
        )

    # PNG section
    png_files = sorted([r.relpath for r in rows if r.relpath.lower().endswith(".png") and r.exists_a and r.exists_b])
    lines.append("\n## PNG pixel-diff\n")
    if not png_files:
        lines.append("_No PNGs found in both snapshots._\n")
    else:
        lines.append("| file | pixel-diff | details |\n")
        lines.append("|---|---:|---|\n")
        for rel in png_files:
            row = next((x for x in rows if x.relpath == rel), None)
            if row is not None and row.same:
                lines.append(f"| `{rel}` | MATCH | identical bytes |\n")
                continue

            info = png_diffs.get(rel)
            if not info:
                lines.append(f"| `{rel}` | DIFF | (no pixel diff produced) |\n")
                continue
            if not info.get("ok"):
                lines.append(f"| `{rel}` | FAILED | {info.get('reason','unknown')} |\n")
                continue
            diff_img = info.get("diff_image", "")
            diff_link = f"`{diff_img}`" if diff_img else ""
            details = f"n_diff_pixels={info.get('n_diff_pixels')} max={info.get('max_channel_diff')} mean={info.get('mean_channel_diff'):.3f}"
            lines.append(f"| `{rel}` | {diff_link} | {details} |\n")

    # PKL section
    pkl_files = sorted([r.relpath for r in rows if r.relpath.lower().endswith(".pkl") and r.exists_a and r.exists_b])
    lines.append("\n## Cache (.pkl) load summary\n")
    if not pkl_files:
        lines.append("_No PKL files found in both snapshots._\n")
    else:
        for rel in pkl_files:
            info = pkl_summaries.get(rel, {})
            lines.append(f"\n### `{rel}`\n")
            lines.append(f"- **A load**: {'OK' if info.get('a_ok') else 'FAIL'}\n")
            lines.append(f"- **B load**: {'OK' if info.get('b_ok') else 'FAIL'}\n")
            lines.append("\n**A summary**:\n\n```\n")
            lines.append((info.get("a_summary") or "").rstrip() + "\n")
            lines.append("```\n\n**B summary**:\n\n```\n")
            lines.append((info.get("b_summary") or "").rstrip() + "\n")
            lines.append("```\n")

    # CSV section
    csv_files = sorted([r.relpath for r in rows if r.relpath.lower().endswith(".csv") and r.exists_a and r.exists_b])
    lines.append("\n## CSV diff\n")
    if not csv_files:
        lines.append("_No CSVs found in both snapshots._\n")
    else:
        lines.append("| file | rows(A) | rows(B) | cols(A) | cols(B) | mismatches | numeric_mismatches | tol |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for rel in csv_files:
            s = csv_summaries.get(rel, {})
            lines.append(
                f"| `{rel}` | {s.get('rows_a','')} | {s.get('rows_b','')} | {s.get('max_cols_a','')} | {s.get('max_cols_b','')} | "
                f"{s.get('mismatches','')} | {s.get('numeric_mismatches','')} | {s.get('tolerance','')} |\n"
            )

    report_path.write_text("".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare artifact snapshots and write artifacts/compare_report.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python tools/compare_artifacts.py --a artifacts/revert-... --b artifacts/refactor-...
              python tools/compare_artifacts.py   # auto-picks latest revert-* vs refactor-*
            """
        ),
    )
    ap.add_argument("--a", type=str, default=None, help="Snapshot dir A (default: latest revert-*)")
    ap.add_argument("--b", type=str, default=None, help="Snapshot dir B (default: latest refactor-*)")
    ap.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts root (default: artifacts)",
    )
    ap.add_argument(
        "--report",
        type=str,
        default="artifacts/compare_report.md",
        help="Output markdown report path",
    )
    ap.add_argument(
        "--diffs-dir",
        type=str,
        default=None,
        help="Directory to store PNG diff images (default: artifacts/compare_diffs/<timestamp>/)",
    )
    ap.add_argument("--csv-tol", type=float, default=0.0, help="Numeric tolerance for CSV numeric comparisons")
    args = ap.parse_args(argv)

    root = Path(args.artifacts_dir).resolve()
    if not root.exists():
        raise SystemExit(f"Artifacts dir does not exist: {root}")

    a_root = Path(args.a).resolve() if args.a else _find_latest_snapshot(root, prefixes=["revert-"])
    b_root = Path(args.b).resolve() if args.b else _find_latest_snapshot(root, prefixes=["refactor-", "refactor_"])

    if a_root is None:
        raise SystemExit("Could not auto-find A snapshot (prefix revert-). Pass --a explicitly.")
    if b_root is None:
        raise SystemExit("Could not auto-find B snapshot (prefix refactor-/refactor_). Pass --b explicitly.")

    report_path = Path(args.report).resolve()
    if args.diffs_dir:
        diffs_dir = Path(args.diffs_dir).resolve()
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        diffs_dir = (root / "compare_diffs" / ts).resolve()

    rows = _compare_files(a_root, b_root)

    # PNG diffs
    png_diffs: Dict[str, Dict[str, object]] = {}
    for r in rows:
        if not (r.exists_a and r.exists_b):
            continue
        if not r.relpath.lower().endswith(".png"):
            continue
        # only do pixel diff when files differ (otherwise diff is empty noise)
        if r.same:
            continue
        a_png = a_root / r.relpath
        b_png = b_root / r.relpath
        out_png = diffs_dir / (r.relpath.replace("/", "__") + ".diff.png")
        try:
            png_diffs[r.relpath] = _pixel_diff_png(a_png, b_png, out_png)
        except Exception as e:
            png_diffs[r.relpath] = {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    # PKL summaries
    pkl_summaries: Dict[str, Dict[str, object]] = {}
    for r in rows:
        if not (r.exists_a and r.exists_b):
            continue
        if not r.relpath.lower().endswith(".pkl"):
            continue
        ok_a, sum_a = _load_pickle_summary(a_root / r.relpath)
        ok_b, sum_b = _load_pickle_summary(b_root / r.relpath)
        pkl_summaries[r.relpath] = {
            "a_ok": ok_a,
            "b_ok": ok_b,
            "a_summary": sum_a,
            "b_summary": sum_b,
        }

    # CSV summaries
    csv_summaries: Dict[str, Dict[str, object]] = {}
    for r in rows:
        if not (r.exists_a and r.exists_b):
            continue
        if not r.relpath.lower().endswith(".csv"):
            continue
        try:
            csv_summaries[r.relpath] = _csv_diff_summary(
                a_root / r.relpath, b_root / r.relpath, tol=float(args.csv_tol)
            )
        except Exception as e:
            csv_summaries[r.relpath] = {"error": f"{type(e).__name__}: {e}"}

    _mkdirp(report_path.parent)
    _write_report(
        report_path=report_path,
        a_root=a_root,
        b_root=b_root,
        rows=rows,
        png_diffs=png_diffs,
        pkl_summaries=pkl_summaries,
        csv_summaries=csv_summaries,
        diffs_dir=diffs_dir,
    )

    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

