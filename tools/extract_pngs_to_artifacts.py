#!/usr/bin/env python3
"""
Extract PNGs from a given git commit and write them into an artifacts snapshot folder
under a canonical path layout: figure/figures/<name>.png

This is meant to support comparing outputs across branches that used different output dirs:
- revert commits may have written to `figures/`
- refactor commits may have written to `figure/figures/` (and sometimes `figure/`)

We prioritize sources in this order when multiple sources map to the same canonical path:
1) figure/figures/
2) figures/
3) figure/
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def run_git(args: List[str], cwd: Path) -> bytes:
    return subprocess.check_output(["git", *args], cwd=str(cwd))


def list_candidate_pngs(commit: str, cwd: Path) -> List[str]:
    out = run_git(["ls-tree", "-r", "--name-only", commit, "--", "figure", "figures"], cwd=cwd)
    paths = [line.strip() for line in out.decode("utf-8", errors="replace").splitlines()]
    return [p for p in paths if p.lower().endswith(".png")]


def canonicalize(p: str) -> Tuple[str, int]:
    # returns (canonical_path, priority_rank) where lower rank wins
    if p.startswith("figure/figures/"):
        return p, 0
    if p.startswith("figures/"):
        return "figure/figures/" + p[len("figures/") :], 1
    if p.startswith("figure/"):
        return "figure/figures/" + p[len("figure/") :], 2
    # fallback: treat as filename
    return "figure/figures/" + os.path.basename(p), 9


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract PNGs from git commit into artifacts snapshot folder.")
    ap.add_argument("--repo", required=True, help="Path to git repo root")
    ap.add_argument("--commit", required=True, help="Commit-ish to extract from (branch, tag, SHA)")
    ap.add_argument("--out", required=True, help="Artifacts snapshot folder to write into")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pngs = list_candidate_pngs(args.commit, cwd=repo)
    chosen: Dict[str, Tuple[int, str]] = {}
    for src in pngs:
        canon, rank = canonicalize(src)
        cur = chosen.get(canon)
        if cur is None or rank < cur[0]:
            chosen[canon] = (rank, src)

    wrote = 0
    for canon, (_rank, src) in sorted(chosen.items()):
        dst = out_root / canon
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob = run_git(["show", f"{args.commit}:{src}"], cwd=repo)
        dst.write_bytes(blob)
        wrote += 1

    print(f"Extracted {wrote} PNG(s) from {args.commit} into {out_root}/figure/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

