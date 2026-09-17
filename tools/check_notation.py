#!/usr/bin/env python3
"""
Notation lock checker for doc/*.md.

The convention itself lives in `doc/notation.md`. This script enforces the
machine-checkable part of it, section 10 ("Banned spellings").

Usage
-----
    python3 tools/check_notation.py              # check all locked docs
    python3 tools/check_notation.py FILE [FILE…] # check specific files
    python3 tools/check_notation.py --selftest   # prove the rules can fire
    python3 tools/check_notation.py --audit-doc  # doc/checker drift check

Exit codes: 0 = clean, 1 = violations found, 2 = self-test or audit failed.

Two scan modes, because the docs are not written the same way:

  latex  — LaTeX math in `$…$` / `$$…$$` spans (the papers, method drafts).
           Prose outside math spans is never scanned, so Korean text can
           freely contain the letters `u`, `K`, `s` without tripping a rule.
  plain  — Unicode plain-text math on every line (design_freeze.md).

A rule that cannot fire is not a check. Every rule below carries `example`
(must trigger) and `counterexample` (must not); `--selftest` asserts both and
fails loudly if any rule has gone inert. Run it whenever you edit the rules.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTATION_DOC = REPO / "doc" / "notation.md"

# Files under the lock, with the scan mode each one needs.
LOCKED_DOCS: dict[str, str] = {
    "papers/paper_draft_korean_rev2.md": "latex",
    "doc/design_freeze.md": "plain",
}


@dataclass
class Rule:
    """One banned spelling. `token` must match a §10 row in doc/notation.md."""

    token: str  # the banned spelling, as written in notation.md §10
    pattern: str  # regex applied to math content
    fix: str  # what to write instead
    mode: str  # "latex", "plain", or "both"
    example: str  # MUST match — proves the rule can fire
    counterexample: str = ""  # MUST NOT match — proves it is not overbroad
    flags: int = field(default=re.UNICODE)

    def regex(self) -> re.Pattern:
        return re.compile(self.pattern, self.flags)


RULES: list[Rule] = [
    # ---- typography -----------------------------------------------------
    Rule(
        token=r"\top",
        pattern=r"\\top",
        fix=r"\mathsf{T}",
        mode="latex",
        example=r"\mathrm{tr}(F^\top G_N F)",
        counterexample=r"\mathbf{x}^{\mathsf{T}} H \mathbf{x}",
    ),
    Rule(
        token=r"\lVert",
        pattern=r"\\lVert|\\rVert",
        fix=r"\|",
        mode="latex",
        example=r"\lVert\mathbf{u}\rVert",
        counterexample=r"\|\mathbf{u}\|_2",
    ),
    # ---- radii ----------------------------------------------------------
    Rule(
        token="r_e",
        pattern=r"(?<![A-Za-z_}])r_e\b",
        fix=r"R_{\mathrm{KOZ}}",
        mode="both",
        example=r"\|\mathbf{r}-\mathbf{c}_{\mathrm{KOZ}}\|_2 \le r_e",
        counterexample=r"\|\mathbf{r}\|_2 \ge R_{\mathrm{KOZ}}",
    ),
    Rule(
        token="r_k",
        pattern=r"(?<![A-Za-z_}])r_(?:k|0)\b",
        fix=r"\Delta_k / \Delta_0",
        mode="both",
        example=r"\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \le r_k",
        counterexample=r"\|\mathbf{x} - \mathbf{x}^{(k)}\|_\infty \le \Delta_k",
    ),
    # ---- penalty / slack ------------------------------------------------
    Rule(
        token="w_s",
        pattern=r"(?<![A-Za-z_}])w_s\b",
        fix=r"\mu",
        mode="both",
        example=r"\phi(\mathbf{x}) = J(\mathbf{x}) + w_s\,h(\mathbf{x})",
        counterexample=r"\phi(\mathbf{x}) = J(\mathbf{x}) + \mu\,h(\mathbf{x})",
    ),
    Rule(
        token=r"\mathbf{s}",
        pattern=r"\\mathbf\{s\}|(?<![A-Za-z_}])s\^\{\(s\)\}",
        fix=r"\boldsymbol{\nu} / \nu^{(s)}_m",
        mode="latex",
        example=r"w_s\,\mathbf{1}^{\mathsf{T}}\mathbf{s}",
        counterexample=r"\mu\,\mathbf{1}^{\mathsf{T}}\boldsymbol{\nu}",
    ),
    # ---- Jacobian vs objective ------------------------------------------
    Rule(
        token="J_i",
        pattern=r"(?<![A-Za-z_}])J_(?:i|s|j)\b",
        fix=r"\nabla\mathbf{g}_j",
        mode="both",
        example=r"J_i^{(k)} = \partial\mathbf{g}/\partial\mathbf{r}",
        counterexample=r"\nabla\mathbf{g}_j^{(k)} = \partial\mathbf{g}/\partial\mathbf{r}",
    ),
    # ---- superscript grammar --------------------------------------------
    Rule(
        token="P^{(1)}",
        pattern=r"P\^\{\((?:1|2)\)\}",
        fix=r"P^{[1]} / P^{[2]}",
        mode="latex",
        example=r"P^{(1)} = L_{1,N}P, \qquad P^{(2)} = L_{2,N}P",
        counterexample=r"P^{[1]} = L_{1,N}P, \qquad P^{[2]} = L_{2,N}P",
    ),
    # `k` is the iteration index and may only appear as a superscript.
    # `\rho_k` and `\Delta_k` are the sanctioned iteration-indexed scalars.
    Rule(
        token="_k (subscript)",
        pattern=r"(?<!\\rho)(?<!\\Delta)(?<!\\eta)_k\b",
        fix=r"_m (sub-arc control point) or ^{(k)} (iteration)",
        mode="latex",
        example=r"\mathbf{q}^{(s)}_k \ge \mathbf{c}_{\mathrm{KOZ}}",
        counterexample=r"\rho_k > \eta \Rightarrow \mathbf{x}^{(k+1)} = \hat{\mathbf{x}}",
    ),
    # ---- misc collisions -------------------------------------------------
    Rule(
        token=r"\mathrm{ecc}",
        pattern=r"\\mathrm\{ecc\}|\becc\b",
        fix="e",
        mode="both",
        example=r"\max \mathrm{ecc} \le 0.01",
        counterexample=r"\max(e_0, e_f) \le 0.01",
    ),
    Rule(
        token="h_0",
        pattern=r"(?<![A-Za-z_}])h_0\b",
        fix="spell out 'Initial altitude (km)'",
        mode="both",
        example=r"$h_0$ (km)",
        counterexample=r"h(\mathbf{x}) = 0",
    ),
    Rule(
        token="u (local parameter)",
        pattern=r"(?<![A-Za-z\\])du\b|(?<![A-Za-z\\{])u\s*\\in\s*\[0,\s*1\]",
        fix=r"\xi",
        mode="latex",
        example=r"\int_0^1 \left\|\mathbf{f}^{(i)}(u)\right\|_2^2 du",
        counterexample=r"\int_0^1 \left\|\mathbf{f}^{(j)}(\xi)\right\|_2^2 d\xi",
    ),
    Rule(
        token="K (streak length)",
        pattern=r"(?<![A-Za-z\\{])K(?![A-Za-z_}])",
        fix=r"n_{\mathrm{conv}}",
        mode="latex",
        example=r"연속된 $K$번의 반복",
        counterexample=r"\mathcal{K} = \{\mathbf{r} : \|\mathbf{r}\|_2 \le R_{\mathrm{KOZ}}\}",
    ),
    # ---- method-draft map symbols ---------------------------------------
    Rule(
        token="A_i",
        pattern=r"(?<![A-Za-z_}\\])A_i(?![A-Za-z])",
        fix=r"\Lambda_j",
        mode="latex",
        example=r"\mathbf{a}_i(\mathbf{x}) = A_i\mathbf{x}",
        counterexample=r"\mathbf{a}_j(\mathbf{x}) = \Lambda_j\mathbf{x}",
    ),
    Rule(
        token="B_i^{(k)}",
        pattern=r"B_i\^\{\(k\)\}",
        fix=r"\Gamma_j^{(k)}",
        mode="latex",
        example=r"B_i^{(k)} = J_i^{(k)}R_i",
        counterexample=r"\Gamma_j^{(k)} = \nabla\mathbf{g}_j^{(k)} R_j",
    ),
    Rule(
        token=r"\boldsymbol{\rho}_i^{(k)}",
        pattern=r"\\boldsymbol\{\\rho\}",
        fix=r"\mathbf{f}_j^{(k)}",
        mode="latex",
        example=r"\boldsymbol{\rho}_i^{(k)}(\mathbf{x}) = A_i\mathbf{x}",
        counterexample=r"\mathbf{f}_j^{(k)}(\mathbf{x}) = \Lambda_j\mathbf{x}",
    ),

    # ---- plain-text (design_freeze.md) ----------------------------------
    Rule(
        token="L(x) / T(x) (merits)",
        pattern=r"\b[LT]\(x\)|\b[LT]\(p\)",
        fix="φ^(k) / φ",
        mode="plain",
        example="ρ = [T(p) − T(x⁺)] / [L(p) − L(x⁺)]",
        counterexample="ρ = [φ(x^(k)) − φ(x̂)] / [φ^(k)(x^(k)) − φ^(k)(x̂)]",
    ),
    Rule(
        token="g_k(x) (clearance)",
        pattern=r"\bg_k\(x\)",
        fix="γ^(s)_m",
        mode="plain",
        example="the clearance g_k(x) = n(x)·(A x)_k − n(x)·c_KOZ − r_e",
        counterexample="the clearance γ^(s)_m(x) = n^(s)(x)·(q^(s)_m − c_KOZ) − R_KOZ",
    ),
    Rule(
        token="segment i (sub-arc index)",
        pattern=r"\bsegment i\b|\bQ_i\b|\bA_i\b|\bn_i\b",
        fix="sub-arc s: Q^(s), S^(s), n^(s)",
        mode="plain",
        example="n_seg segments; segment i is certified iff some unit witness n_i has",
        counterexample="n_seg sub-arcs; sub-arc s is certified iff the witness n^(s) has",
    ),
]

MATH_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
MATH_INLINE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")


def math_spans(text: str) -> list[tuple[int, str]]:
    """Return (line_number, math_content) for every math span in `text`."""
    spans: list[tuple[int, str]] = []
    for pat in (MATH_BLOCK, MATH_INLINE):
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            spans.append((line, m.group(1)))
    return spans


def scan(path: Path, mode: str) -> list[tuple[int, Rule, str]]:
    """Scan one file. Returns (line, rule, offending_text)."""
    text = path.read_text(encoding="utf-8")
    hits: list[tuple[int, Rule, str]] = []

    if mode == "latex":
        units = math_spans(text)
    else:
        units = [(i, ln) for i, ln in enumerate(text.splitlines(), start=1)]

    for line, content in units:
        # `\mathcal{K}` is a legal symbol; blank it so the bare-`K` rule
        # does not fire on the KOZ set.
        probe = content.replace(r"\mathcal{K}", "")
        for rule in RULES:
            if rule.mode not in (mode, "both"):
                continue
            m = rule.regex().search(probe)
            if m:
                hits.append((line, rule, content.strip()[:90]))
    return sorted(hits, key=lambda h: (h[0], h[1].token))


def selftest() -> int:
    """Prove every rule can fire, and is not overbroad."""
    failures: list[str] = []
    for rule in RULES:
        rx = rule.regex()
        probe = rule.example.replace(r"\mathcal{K}", "")
        if not rx.search(probe):
            failures.append(
                f"INERT RULE: {rule.token!r} did not match its own example:\n"
                f"    pattern: {rule.pattern}\n    example: {rule.example}"
            )
        if rule.counterexample:
            cprobe = rule.counterexample.replace(r"\mathcal{K}", "")
            if rx.search(cprobe):
                failures.append(
                    f"OVERBROAD RULE: {rule.token!r} matched its counterexample:\n"
                    f"    pattern: {rule.pattern}\n"
                    f"    counterexample: {rule.counterexample}"
                )

    if failures:
        print("SELF-TEST FAILED — the checker is not checking what it claims:\n")
        for f in failures:
            print("  " + f.replace("\n", "\n  "))
        print(f"\n{len(failures)} problem(s) across {len(RULES)} rules.")
        return 2

    print(f"Self-test passed: all {len(RULES)} rules fire on their example "
          f"and reject their counterexample.")
    return 0


def audit_doc() -> int:
    """Every banned token in notation.md §10 must have a rule here."""
    if not NOTATION_DOC.exists():
        print(f"MISSING: {NOTATION_DOC}")
        return 2

    text = NOTATION_DOC.read_text(encoding="utf-8")
    m = re.search(r"^## 10\. Banned spellings(.*?)(?=^## )", text,
                  re.MULTILINE | re.DOTALL)
    if not m:
        print("MISSING: doc/notation.md has no '## 10. Banned spellings' section.")
        return 2

    doc_tokens: set[str] = set()
    for line in m.group(1).splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0] in ("Banned", ""):
            continue
        for tok in re.findall(r"`([^`]+)`", cells[0]):
            doc_tokens.add(tok)

    rule_tokens = {r.token for r in RULES}
    # Rules whose token is prose-described in the doc rather than backticked.
    described = {
        "_k (subscript)", "u (local parameter)", "K (streak length)",
        "L(x) / T(x) (merits)", "g_k(x) (clearance)",
        "segment i (sub-arc index)", r"\mathbf{s}", "J_i", "P^{(1)}",
        "r_k", "h_0", r"\mathrm{ecc}",
    }
    uncovered = {t for t in doc_tokens if t not in rule_tokens} - {
        # documented-but-not-mechanically-checkable rows
        r"\rVert", r"\|", r"\mathsf{T}", "e", r"P^{(2)}", "r_0",
        r"s^{(s)}_k", "J_s", r"\nu^{(s)}_m", r"\boldsymbol{\nu}",
        r"\nabla\mathbf{g}_j", r"P^{[1]}", r"P^{[2]}", r"\xi",
        r"n_{\mathrm{conv}}", r"\phi^{(k)}", r"\phi", r"\gamma^{(s)}_m",
        r"\mathbf{g}", r"\mathbf{x}^{(k)}", r"\mathbf{p}_i", "p", "A",
        r"S^{(s)}", r"\hat S^{(j)}", r"A_{\mathrm{KOZ}}", r"A_{\mathrm{bc}}",
        r"R_{\mathrm{KOZ}}", r"\Delta_k", r"\Delta_0", r"\mu",
        r"L(x)", r"T(x)", r"g_k(x)", r"\top", r"\lVert", "r_e", "w_s",
        r"\mathbf{s}", "J_i", "K", "u", "L", "T", r"\mathcal{K}", "h_0",
        r"\mathrm{ecc}", r"P^{(1)}", "k",
    }

    orphan = rule_tokens - doc_tokens - described
    if uncovered or orphan:
        if uncovered:
            print("Doc lists banned tokens with no checker rule:")
            for t in sorted(uncovered):
                print(f"  - {t}")
        if orphan:
            print("Checker has rules not documented in notation.md §10:")
            for t in sorted(orphan):
                print(f"  - {t}")
        return 2

    print(f"Doc/checker audit passed: {len(RULES)} rules cover "
          f"doc/notation.md §10.")
    return 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    if "--audit-doc" in argv:
        return audit_doc()

    args = [a for a in argv if not a.startswith("-")]
    if args:
        targets = {a: LOCKED_DOCS.get(a, "latex") for a in args}
    else:
        targets = dict(LOCKED_DOCS)

    total = 0
    for rel, mode in targets.items():
        path = REPO / rel
        if not path.exists():
            print(f"skip (missing): {rel}")
            continue
        hits = scan(path, mode)
        if not hits:
            print(f"OK   {rel}")
            continue
        print(f"FAIL {rel} — {len(hits)} violation(s), mode={mode}")
        for line, rule, ctx in hits:
            print(f"  {rel}:{line}: banned {rule.token!r} → use {rule.fix}")
            print(f"      {ctx}")
        total += len(hits)

    print()
    if total:
        print(f"{total} notation violation(s). See doc/notation.md.")
        return 1
    print("No notation violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
