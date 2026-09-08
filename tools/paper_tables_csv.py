"""Read the committed result table CSV that the paper's figures must agree with.

`tools/build_tables.py` is the single source of the paper's numbers: it runs the
solver with the cache off and writes `doc/results/paper_tables.{md,csv}` together
with the producing commit SHA. Any figure that plots the same measurements has to
read the same file, or it will drift from the table printed beside it -- which is
exactly what happened to the results figures before this module existed.

The CSV carries no table identifier: `build_tables.py:172` writes `rows = t2 + t3
+ t4` as one flat list, and the same (degree, n_seg) configuration appears in more
than one block. Blocks are therefore located by their (degree, n_seg) signature
rather than by position, so a T3 prefix appearing later cannot silently shift
which rows a figure picks up. If a signature stops matching, that is a real
change in the tables and the figure should fail rather than plot the wrong rows.
"""

from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[1] / "doc" / "results" / "paper_tables.csv"

# Section 4.2, first experiment: subdivision sweep at N = 7.
T4_SIGNATURE = [(7, n) for n in (2, 4, 8, 16, 32, 64)]
# Section 4.2, second experiment: degree sweep at n_seg = 16.
T5_SIGNATURE = [(6, 16), (7, 16), (8, 16)]

_FLOAT = ("margin_km", "ctrl_cost_ms2", "objective", "runtime_s",
          "hull_violation_km")
_INT = ("degree", "n_ctrl", "n_seg", "iters", "stop")
_BOOL = ("certified", "dense_probe_ok", "disagree")


def load_rows(path: Path = CSV_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python tools/build_tables.py")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in _FLOAT:
            if k in r:
                r[k] = float(r[k])
        for k in _INT:
            if k in r:
                r[k] = int(float(r[k]))
        for k in _BOOL:
            if k in r:
                r[k] = str(r[k]).strip().lower() == "true"
    return rows


def block(signature: list[tuple[int, int]], what: str,
          path: Path = CSV_PATH) -> list[dict]:
    """Return the contiguous run of rows matching a (degree, n_seg) signature."""
    rows = load_rows(path)
    keys = [(r["degree"], r["n_seg"]) for r in rows]
    n = len(signature)
    for i in range(len(keys) - n + 1):
        if keys[i:i + n] == signature:
            return rows[i:i + n]
    raise SystemExit(
        f"{what}: no rows matching {signature} in {path}.\n"
        f"  found: {keys}\n"
        "The tables changed shape. Regenerate with `python tools/build_tables.py`,\n"
        "or if the block genuinely moved, add a table-id column there so the\n"
        "figures do not have to guess.")


def endpoint_attained(rows: list[dict]) -> list[bool]:
    """Flag rows whose safety margin is NOT a measure of the method.

    Section 5.2 makes this argument itself: where the minimum radius sits at an
    endpoint, the boundary conditions rather than the KOZ set it, so the margin
    is the departure orbit's altitude. The paper's own evidence is that two
    settings then report *exactly* the same margin -- "두 설정의 안전 여유가
    145.00 km로 정확히 같은 것이 그 근거이다." Detect it the same way instead of
    hard-coding which n_seg are affected.
    """
    counts: dict[float, int] = {}
    for r in rows:
        counts[r["margin_km"]] = counts.get(r["margin_km"], 0) + 1
    return [counts[r["margin_km"]] > 1 for r in rows]
