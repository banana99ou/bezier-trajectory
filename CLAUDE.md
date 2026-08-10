## Permission required for changes

- Do not commit unless the user explicitly says to commit. A request to "fix", "implement", or "do" something is **not** permission to commit.
- Do not edit files unless the user explicitly asks for an edit. Requests to investigate, plan, analyze, or discuss are **not** permission to edit.
- When in doubt, ask before acting.

## Commit hygiene

- No AI co-author trailers. Do not add `Co-Authored-By: Claude ...` lines to commit messages.
- Split unrelated changes into separate commits — one commit per work context.

## Locked design decisions (author, 2026-08-07 — do not revisit without explicit user approval)

**Before touching solver code or the paper, read `doc/design_freeze.md`** — it
holds the full design, the principle behind every parameter, and the evidence
log. It outranks the paper: the code plus the 5-pillar verification are ground
truth, and the paper is being rewritten toward them, never the reverse.
In-flight work is in `doc/session_handoff.md`.

## Mathematical notation

**Before writing any equation, symbol, figure caption, or table header in
`doc/`, read `doc/notation.md`.** It is the single source of truth for every
mathematical symbol and it outranks every draft — if a draft disagrees, the
draft is wrong. Adding a symbol means adding it to `doc/notation.md` in the
same edit. Verify with:

```
python3 tools/check_notation.py            # must exit 0
python3 tools/check_notation.py --selftest # proves the rules can still fire
```

Do not reuse an index letter for a second role, and do not reintroduce a
spelling listed in `doc/notation.md` §10 (`\top`, `r_e`, `w_s`, `r_k`, `J_i`,
`K`, `P^{(2)}` for a derivative, `\mathbf{s}` for slack, …).

- **The cost function is control-acceleration energy, hard-coded, singular,
  computed as the EXACT integral** (closed form via the Bernstein Gram matrix —
  computing curve derivatives/integrals in control-point space is the whole
  point of the formulation):
  `J = integral_0^1 ||a_geom(tau)/T^2 - (grad_g_j r(tau) + c_j)||^2 dtau`,
  gravity affine per De Casteljau interval `j` (`n_lin` of them — distinct from
  the `n_seg` KOZ sub-arcs indexed by `s`). There are NO objective modes, flags,
  sampling schemes, or auxiliary terms (no standalone smoothness term, no
  dv/IRLS mode, no multi-stage schemes). Do not add any. History: a full-weight
  smoothness term biased the optimum (its apparent benefit was a
  premature-convergence artifact, fixed by the `n_conv`-consecutive convergence
  test); a sampled objective had soft directions the exact integral eliminates.
- **Δv / fuel is never mentioned in the paper.** Not as an objective, not as a
  limitation, not as future work, not as "we don't do Δv" — no mention at all.
  When regenerating result tables, exclude the `dv_proxy_m_s` column.
- The solver machinery is canonical SCvx (penalized-merit ratio test, virtual
  control, trust region). Any deviation from textbook must be justified from a
  principle in the paper or removed.
- machinery maximally boring — every mechanism is textbook-with-citation or principle-derived