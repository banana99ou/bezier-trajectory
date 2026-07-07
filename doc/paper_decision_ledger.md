# Paper decision ledger

Standing decisions for the Korean paper revision (`paper_draft_korean_rev2.md`).
One row per decision. Append new rows at the bottom; never rewrite history — if a
decision is reversed, add a new row that supersedes it and link back.

Working file: `doc/paper_draft_korean_rev2.md` (surgical in-place edits of `paper_draft_korean_PF.md`).
Backups: `paper_draft_korean_rev2.md.bak-YYYYMMDD` siblings, created before every edit batch.
Revert = `cp <backup> paper_draft_korean_rev2.md`.

| # | Date | Decision | Rationale | Scope / where |
|---|------|----------|-----------|---------------|
| D1 | 2026-06-12 | Revision process = copy PF → edit **in place, only at critique sites**. No from-scratch rewrites. | rev1 (from-scratch rewrite by agent) touched unflagged prose and was rejected. Surgical edits keep the diff auditable against the prof's `[[…]]` annotations. | whole file; `rev1` abandoned |
| D2 | 2026-06 | **HOLD §3 (method) substantive rewrites** (#27, #31, #34, #35, #39: IRLS explanation, notation block, Algorithm form, new figure). Term/notation fixes only. | Optimizer loop is structurally wrong (no trust-region acceptance test → not true SCvx; 10000-iter cap). Method prose must not describe a method the code doesn't implement. Rewrite after the SCvx rework. | §3 전체 |
| D3 | 2026-06 | **All measured table cells (T2/T3/T4/T6) = TODO**; results prose keeps qualitative trends only, no specific numbers. | Numbers come from the non-converged loop (tol=1e-12, always hits iteration cap). Prof: code structure is a mess, rerun pointless until fixed. | §5 tables + prose, abstract, conclusion |
| D4 | 2026-06-13 | **Ban coined term "연속 안전"**; do not launder vague "안전" behind a homemade label. | Prof flagged "안전" as unclear; defining a coinage doesn't remove the vagueness. Say the concrete thing instead. | paper-wide (10 sites replaced) |
| D5 | 2026-07-05 | Adopt literature term **연속시간 제약 만족 (continuous-time constraint satisfaction)** as the property name; **노드 간 제약 위반 (inter-sample constraint violation)** names the failure mode. Informal definition in §1, formal (∀τ, ‖r(τ)−c‖≥r_e, boundary contact allowed) in §3.1 with τ↔t fixed-T caveat and hook to 명제 1. | Checked literature: ct-SCvx (Elango et al. 2024) is the standard term in the direct lineage of refs [6][7]; Dueri et al. 2017 names the failure mode. Property (goal) vs. half-space containment (mechanism/sufficient condition) kept distinct. | §1 L16, §3.1 |
| D6 | 2026-07-08 | Added refs **[11] Dueri et al. CDC 2017** (doi:10.1109/CDC.2017.8263811), **[12] Elango et al. arXiv:2404.16826** ; cited at first use of each term in §1. Term used consistently paper-wide (§1 기여/관련연구, 명제 1 lead-in, F1 caption, §3.2, §5.2, 결론). "안전 여유(safety margin)" kept — it is a metric name, not the property. Abstract left in plain language (term not yet defined there). | Terms of art should be cited; consistent naming separates property from mechanism throughout. | refs, ~14 sites |
