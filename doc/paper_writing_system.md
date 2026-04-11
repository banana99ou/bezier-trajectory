# Paper Writing System

This document defines how to write, review, and assess the paper in a way that prevents fragmented drafting, unsupported claims, and comparison sloppiness.

The basic rule is simple:

**Do not draft prose faster than the claim boundary and evidence boundary are defined.**

Corollary:

- the paper's identity should be framework-first
- downstream warm-start usefulness remains a major paper component, but it should not become the paper's sole or primary identity

## Source documents

Use these documents as the paper's control system:

- `doc/paper_claim_scope_nonclaims.md`: claim boundary and explicit non-claims
- `doc/paper_evidence_map.md`: claim-to-evidence map and gap analysis
- `doc/dcm_db_experiment_note.md`: provisional DB-seeded DCM comparison framing and fairness conditions
- `doc/figure_table_per_claim_plan.md`: figure/table inventory tied to claims
- `doc/paper_technical_skeleton.md`: current Stage 3 source-of-truth section architecture
- `doc/paper_quality_rubric.md`: scoring rubric for readiness checks
- `doc/paper_positioning_strategy_report.md`: advisory positioning guidance, not canonical claim control

Use these implementation references when writing the method section:

- `orbital_docking/bezier.py`
- `orbital_docking/de_casteljau.py`
- `orbital_docking/constraints.py`
- `orbital_docking/optimization.py`
- `Orbital_Docking_Optimizer.py`

## Recommended writing workflow

## Stage 0: Claim lock

Purpose:

- freeze the paper's main claim, scope, non-claims, and comparison stance before drafting

Required output:

- `doc/paper_claim_scope_nonclaims.md`

Pass criteria:

- the core claim fits in two sentences
- the framework contribution is clearer than the comparison framing
- the paper's scope is explicit
- the paper's non-claims are explicit
- no sentence implies superiority or broad validation without evidence

Failure mode to avoid:

- writing an ambitious introduction first and trying to backfill evidence later
- making the paper sound as if it exists mainly to be compared against direct collocation

## Stage 1: Evidence lock

Purpose:

- ensure every important claim has matching proof, experiment, ablation, or explicit limitation

Required output:

- `doc/paper_evidence_map.md`

Pass criteria:

- every major claim is labeled `present`, `partial`, or `missing`
- every `missing` claim is either assigned new work or deleted from the target paper
- the downstream DCM / direct-collocation comparison is explicitly classified as a required major paper component or the paper scope is deliberately rewritten
- the downstream comparison is defined on matched cases, with the same downstream settings and the same metrics across pipelines
- any case-selection rule is reproducible and not phrased as "good looking" or other aesthetic filtering

Failure mode to avoid:

- letting "intended use" silently become "demonstrated capability"
- treating the downstream comparison as optional while still writing the paper as if its usefulness case has been established
- allowing subjective case selection or hidden filtering to masquerade as experiment design

## Stage 2: Figure and table plan

Purpose:

- define the evidence before writing narrative around it

Required output:


- a list of all planned figures, tables, and metrics: `doc/figure_table_per_claim_plan.md`

Minimum recommended items:

- method diagram: Bézier curve -> subdivision -> supporting half-spaces -> SCP loop
- KOZ linearization figure for one representative sub-arc
- degree/segment-count ablation table
- compute-time versus degree/segment-count figure
- downstream comparison table: `DB init -> DCM` versus `same DB init -> upstream optimizer -> DCM`
- case-selection rule and case-count summary for the downstream comparison set
- region-of-improvement summary if the proposed pipeline helps only on a subset of cases

Pass criteria:

- every major result claim points to a specific figure or table
- no figure exists only because it looks nice; each must answer a paper question

Failure mode to avoid:

- producing many trajectory plots and still failing to answer the comparison question

## Stage 3: Technical skeleton

Purpose:

- build the section-by-section logic before polishing prose

Required output:

- a section outline with purpose, key message, and evidence source for each section: `doc/paper_technical_skeleton.md`

Pass criteria:

- each section solves one problem in the paper's argument
- the conservative eight-section structure is fixed as the current source-of-truth
- there is no section whose real job is only "background filler"

Failure mode to avoid:

- an introduction that meanders across orbital motivation, Bézier history, and planner taxonomy without a single sharp gap statement

## Stage 4: Problem setup and method draft

Purpose:

- write Sections 3 and 4 from the actual formulation, not from remembered intent

Writing order inside the technical core:

1. problem setup and notation
2. control-point-space derivative structure (`D`, `E`, Gram matrix)
3. KOZ subdivision and supporting-half-space construction
4. objective definitions and what they do and do not mean
5. SCP loop and subproblem structure
6. assumptions and limitations of the formulation

Pass criteria:

- every variable is defined once and used consistently
- the safety argument states its assumptions explicitly
- the paper says "sequence of convex QPs" rather than implying exact convexification

Failure mode to avoid:

- mixing mathematical claims with implementation convenience and hoping the reviewer will not notice

## Stage 5: Experiment draft

Purpose:

- show evidence that matches the paper's real claim

Required comparisons:

- ablation over subdivision count
- ablation over Bézier degree
- downstream comparison: matched DB-seeded baseline versus matched DB-seeded upstream-optimized pipeline

Pass criteria:

- each experiment answers a specific claim from `doc/paper_evidence_map.md`
- baselines are fair and clearly defined
- captions explain what conclusion the reader should draw
- if results are still moving, captions and prose remain provisional and avoid locking in pessimistic or over-strong interpretations
- the downstream comparison states the exact baseline and proposed pipelines in input-output form
- the downstream comparison retains failed or non-improving cases rather than reporting wins only
- if the proposed pipeline helps only in a subset of cases, the prose says so explicitly instead of implying broad superiority

Failure mode to avoid:

- comparing against a weakly tuned baseline and calling the result meaningful
- writing ablation prose that freezes a negative or sweeping interpretation before the evidence stabilizes
- cherry-picking visually appealing cases without a reproducible selection rule
- changing downstream settings between the baseline and proposed pipelines
- describing subset improvement as if it were general downstream superiority

## Stage 6: Introduction, abstract, title

Purpose:

- write the framing after the technical and evidence core are fixed

Writing rule:

- the title and abstract must be narrower than the full intellectual ambition, not broader

Pass criteria:

- title, abstract, and conclusion do not promise more than the evidence shows
- the paper is application-agnostic in framing but honest about its demonstration base
- the opening framing leads with the framework contribution before the downstream usefulness angle
- downstream usefulness claims are conditional and matched-case-specific unless broader evidence actually exists

Failure mode to avoid:

- trying to make the work sound universal by hiding the limitations

## Stage 7: Red-team review

Purpose:

- attack the paper before reviewers do

Review questions:

- what exactly is new here?
- what is guaranteed, and under which assumptions?
- what does the paper demonstrate versus merely suggest?
- why is the comparison fair?
- where would a skeptical reviewer accuse the paper of overclaiming?
- which sentence in the abstract would be hardest to defend in rebuttal?

Pass criteria:

- the paper survives these questions without rhetorical evasiveness

## Quality assessment at each stage

Use `doc/paper_quality_rubric.md` after stages 1, 3, 5, and 7.

Recommended minimum scores before submission drafting:

- Claim Clarity: `4`
- Scope Discipline: `4`
- Positioning: `4`
- Technical Soundness: `4`
- Evidence Fit: `4`
- Comparison Fairness: `4`
- Reviewer Defensibility: `4`

If any of these remain at `3` or below, the problem is probably structural, not stylistic.

## Review checklist

Use this checklist every time a major section is revised.

- Does the section make exactly one main point?
- Is that point already authorized by `doc/paper_claim_scope_nonclaims.md`?
- Does the point have matching support in `doc/paper_evidence_map.md`?
- Is the framework identity clearer than the direct-collocation comparison angle?
- Are any broad phrases hiding narrow evidence?
- Is "application-agnostic" being used carefully, without pretending cross-domain validation exists?
- Is "continuous safety" stated with its assumptions?
- Is the comparison framed as warm-start value rather than full-method replacement?
- Is any provisional ablation wording prematurely pessimistic or stronger than the still-moving evidence?
- Is any scenario-specific or local feasibility device being mispresented as part of the reusable framework?
- Is any baseline being weakened by bad initialization, unequal tuning, or mismatched scope?
- Is the downstream baseline defined as the actual DB-seeded DCM pipeline rather than an easier substitute?
- Is the case-selection rule reproducible rather than aesthetic?
- Are failed, neutral, and losing downstream cases retained in the reported evidence?
- Does any sentence turn subset improvement into a global downstream claim?
- Is any objective interpretation being oversold?
- Is the prose using paper-level concepts rather than unnecessary code identifiers, mode names, or repository narration?
- Would a skeptical reviewer call this sentence fair?

If the answer to the last question is "probably not," rewrite it.

## Drafting order

Only after the earlier stages are locked should prose drafting proceed in this order:

1. problem setup and notation
2. method
3. experiment setup and figure captions
4. results section
5. limitations and scope
6. related work and positioning
7. introduction
8. conclusion
9. abstract
10. title

This order is deliberate. Weak papers often do the reverse.

## Final discipline rules

- Do not claim replacement if the evidence only shows warm-start value.
- Do not let the downstream comparison become the paper's primary identity when the framework itself is the main contribution.
- Do not claim generality if the paper only demonstrates one domain.
- Do not claim physical optimality if the objective is a surrogate.
- Do not use "convexification" in a way that implies the original problem has been solved exactly.
- Do not let scenario-specific protocol details quietly become framework claims.
- Do not write the paper as a narrated tour of code internals unless those internals are analytically necessary.
- Do not let convenience phrasing outrun proof or experiment.
- Do not use aesthetic filtering such as "good looking curve" unless it has been replaced by a reproducible case-selection rule.
- Do not turn conditional downstream wins on a subset of cases into a claim of general superiority.

If a sentence sounds stronger than the evidence map, the sentence is wrong, not bold.
