"""
Stage D -- aggregate all pillar verdicts into artifacts/verify/VERDICT.md.

Reads each pillarN_*/summary.md, extracts its `## VERDICT:` line and its
provenance stamp, and writes one table.
Run after the individual pillars:  .venv/bin/python tools/verify/verdict.py

An overall PASS asserts something about ONE codebase, so it requires more than
six PASSes: every pillar must carry a stamp, the stamps must name the same
commit and the same compiled Rust extension, and none may report a dirty tree.
Six pillars run at six code versions are six results, not a verdict -- that case
reports INCONCLUSIVE, which is neither a pass nor an accusation of failure.
(This is not a hypothetical: before the stamps existed, five of these artifacts
were from 29 Jul and Pillar 5 was from 9 Aug, and VERDICT.md presented the mix
as a single PASS.)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from tools.verify import harness_common as H

PILLARS = [
    ("Pillar 1 -- independent NLP cross-check", "pillar1_nlp"),
    ("Pillar 2 -- ablation (which change is the fix)", "pillar2_ablation"),
    ("Pillar 3 -- KKT / feasibility at x*", "pillar3_kkt"),
    ("Pillar 4a -- per-iteration diagnostics", "pillar4_diag"),
    ("Pillar 4b -- regression sweep", "pillar4_sweep"),
    ("Pillar 5 -- optimality (external KKT + descent search)", "pillar5_optimality"),
]


def _read(sub):
    """(verdict, stamp) for one pillar; stamp is None when unstamped."""
    sm = H.ARTIFACT_ROOT / sub / "summary.md"
    if not sm.exists():
        return "MISSING", None
    text = sm.read_text()
    m = re.search(r"##\s*VERDICT:\s*(PASS|FAIL)", text)
    return (m.group(1) if m else "UNKNOWN"), H.read_stamp(text)


def main():
    rows = [(title, sub, *_read(sub)) for title, sub in PILLARS]

    all_pass = all(v == "PASS" for _, _, v, _ in rows)
    stamps = [s for *_, s in rows]
    unstamped = [t for t, _, _, s in rows if s is None]
    dirty = [t for t, _, _, s in rows if s and s[1]]
    commits = {s[0] for s in stamps if s}
    exts = {s[2] for s in stamps if s}
    consistent = not unstamped and not dirty and len(commits) == 1 and len(exts) == 1

    lines = ["# SCvx fix verification -- VERDICT", "",
             "| pillar | verdict | commit | Rust ext |", "|---|---|---|---|"]
    for title, _, verdict, stamp in rows:
        if stamp is None:
            prov = "| _unstamped_ | _unstamped_ |"
        else:
            sha, is_dirty, ext = stamp
            prov = f"| `{sha[:9]}`{' **+dirty**' if is_dirty else ''} | `{ext}` |"
        lines.append(f"| {title} | **{verdict}** {prov}")

    if all_pass and consistent:
        overall = "PASS"
    elif not all_pass:
        overall = "FAIL"
    else:
        overall = "INCONCLUSIVE (every pillar passed, but not on one codebase)"

    lines += ["", f"## OVERALL: {overall}", ""]

    if not consistent:
        lines.append("### Why the provenance is not consistent")
        lines.append("")
        if unstamped:
            lines.append(f"- unstamped (produced before stamping existed, or by hand): "
                         f"{', '.join(unstamped)}")
        if dirty:
            lines.append(f"- produced from a dirty working tree, so the commit does not "
                         f"describe what ran: {', '.join(dirty)}")
        if len(commits) > 1:
            lines.append(f"- {len(commits)} different commits: "
                         f"{', '.join('`' + c[:9] + '`' for c in sorted(commits))}")
        if len(exts) > 1:
            lines.append(f"- {len(exts)} different compiled Rust extensions: "
                         f"{', '.join('`' + e + '`' for e in sorted(exts))} -- the .so does "
                         f"not rebuild when rust_optimizer/ changes, so this can differ even "
                         f"at one commit")
        lines.append("")
        lines.append("Re-run every pillar on one clean tree to resolve.")
        lines.append("")

    lines.append("Details in `artifacts/verify/pillar*/summary.md`. Run the whole harness with "
                 "`tools/verify/{nlp_crosscheck,ablation,kkt_check,diagnostics,sweep,"
                 "optimality}.py` then this.")
    H.write_text(H.ARTIFACT_ROOT / "VERDICT.md", "\n".join(lines) + H.provenance())
    print("\n".join(lines))
    return all_pass and consistent


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
