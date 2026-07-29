"""
Stage D -- aggregate all pillar verdicts into artifacts/verify/VERDICT.md.

Reads each pillarN_*/summary.md, extracts its `## VERDICT:` line, writes one table.
Run after the individual pillars:  .venv/bin/python tools/verify/verdict.py
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
]


def main():
    lines = ["# SCvx fix verification -- VERDICT", ""]
    lines.append("| pillar | verdict |")
    lines.append("|---|---|")
    all_pass = True
    for title, sub in PILLARS:
        sm = H.ARTIFACT_ROOT / sub / "summary.md"
        verdict = "MISSING"
        if sm.exists():
            m = re.search(r"##\s*VERDICT:\s*(PASS|FAIL)", sm.read_text())
            verdict = m.group(1) if m else "UNKNOWN"
        all_pass = all_pass and (verdict == "PASS")
        lines.append(f"| {title} | **{verdict}** |")
    lines.append("")
    lines.append(f"## OVERALL: {'PASS' if all_pass else 'FAIL'}")
    lines.append("")
    lines.append("Details in `artifacts/verify/pillar*/summary.md`. Run the whole harness with "
                 "`tools/verify/{nlp_crosscheck,ablation,kkt_check,diagnostics,sweep}.py` then this.")
    H.write_text(H.ARTIFACT_ROOT / "VERDICT.md", "\n".join(lines))
    print("\n".join(lines))
    return all_pass


if __name__ == "__main__":
    main()
