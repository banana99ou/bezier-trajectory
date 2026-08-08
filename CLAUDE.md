## Permission required for changes

- Do not commit unless the user explicitly says to commit. A request to "fix", "implement", or "do" something is **not** permission to commit.
- Do not edit files unless the user explicitly asks for an edit. Requests to investigate, plan, analyze, or discuss are **not** permission to edit.
- When in doubt, ask before acting.

## Commit hygiene

- No AI co-author trailers. Do not add `Co-Authored-By: Claude ...` lines to commit messages.
- Split unrelated changes into separate commits — one commit per work context.

## Locked design decisions (author, 2026-08-07 — do not revisit without explicit user approval)

- **The cost function is control-acceleration energy, hard-coded, singular,
  computed as the EXACT integral** (closed form via the Bernstein Gram matrix —
  computing curve derivatives/integrals in control-point space is the whole
  point of the formulation):
  `J = integral_0^1 ||a_geom(tau)/T^2 - (J_s r(tau) + c_s)||^2 dtau`,
  gravity affine per De Casteljau segment. There are NO objective modes, flags,
  sampling schemes, or auxiliary terms (no standalone smoothness term, no
  dv/IRLS mode, no multi-stage schemes). Do not add any. History: a full-weight
  smoothness term biased the optimum (its apparent benefit was a
  premature-convergence artifact, fixed by the K-consecutive convergence test);
  a sampled objective had soft directions the exact integral eliminates.
- **Δv / fuel is never mentioned in the paper.** Not as an objective, not as a
  limitation, not as future work, not as "we don't do Δv" — no mention at all.
  When regenerating result tables, exclude the `dv_proxy_m_s` column.
- The solver machinery is canonical SCvx (penalized-merit ratio test, virtual
  control, trust region). Any deviation from textbook must be justified from a
  principle in the paper or removed.
- machinery maximally boring — every mechanism is textbook-with-citation or principle-derived