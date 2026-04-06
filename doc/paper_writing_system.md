# Paper Writing System

This document defines how to write, review, and assess the paper in a way that prevents fragmented drafting, unsupported claims, and comparison sloppiness.

The basic rule is simple:

**Do not draft prose faster than the claim boundary and evidence boundary are defined.**

## Source documents

Use these documents as the paper's control system:

- `doc/paper_claim_scope_nonclaims.md`: claim boundary and explicit non-claims
- `doc/paper_evidence_map.md`: claim-to-evidence map and gap analysis
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
- the paper's scope is explicit
- the paper's non-claims are explicit
- no sentence implies superiority or broad validation without evidence

Failure mode to avoid:

- writing an ambitious introduction first and trying to backfill evidence later

## Stage 1: Evidence lock

Purpose:

- ensure every important claim has matching proof, experiment, ablation, or explicit limitation

Required output:

- `doc/paper_evidence_map.md`

Pass criteria:

- every major claim is labeled `present`, `partial`, or `missing`
- every `missing` claim is either assigned new work or deleted from the target paper
- the direct-collocation comparison is explicitly classified as required or out of scope

Failure mode to avoid:

- letting "intended use" silently become "demonstrated capability"

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
- downstream comparison table: naive direct collocation vs warm-started direct collocation

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
- downstream comparison: direct collocation from naive initialization vs from Bézier-based warm start

Pass criteria:

- each experiment answers a specific claim from `doc/paper_evidence_map.md`
- baselines are fair and clearly defined
- captions explain what conclusion the reader should draw

Failure mode to avoid:

- comparing against a weakly tuned baseline and calling the result meaningful

## Stage 6: Introduction, abstract, title

Purpose:

- write the framing after the technical and evidence core are fixed

Writing rule:

- the title and abstract must be narrower than the full intellectual ambition, not broader

Pass criteria:

- title, abstract, and conclusion do not promise more than the evidence shows
- the paper is application-agnostic in framing but honest about its demonstration base

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
- Are any broad phrases hiding narrow evidence?
- Is "application-agnostic" being used carefully, without pretending cross-domain validation exists?
- Is "continuous safety" stated with its assumptions?
- Is the comparison framed as warm-start value rather than full-method replacement?
- Is any baseline being weakened by bad initialization, unequal tuning, or mismatched scope?
- Is any objective interpretation being oversold?
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
- Do not claim generality if the paper only demonstrates one domain.
- Do not claim physical optimality if the objective is a surrogate.
- Do not use "convexification" in a way that implies the original problem has been solved exactly.
- Do not let convenience phrasing outrun proof or experiment.

If a sentence sounds stronger than the evidence map, the sentence is wrong, not bold.


## Global Placeholder Policy
This paper-writing workflow forbids invented evidence, invented trends, invented comparative outcomes, and invented conclusions.
If a section is structurally ready but evidence is not yet available, the agent must use explicit placeholders rather than fabricate completeness.
Allowed placeholder types:
- missing numeric results
- missing figure/table outputs
- missing trend descriptions
- missing interpretation that depends on real runs
- blocked comparisons awaiting implementation or fair evaluation
Required placeholder format:
- `[INSERT T3 RESULTS]`
- `[INSERT F4 TREND DESCRIPTION AFTER RUNS]`
- `[INSERT T6 DOWNSTREAM COMPARISON DATA]`
- `[EVIDENCE PENDING]`
- `[INTERPRET ONLY AFTER ACTUAL RESULTS ARE AVAILABLE]`
Rules:
1. Write missing-evidence sections in near-final form when possible, but keep all unsupported content explicitly marked.
2. Do not invent:
   - numbers
   - trends
   - success claims
   - superiority claims
   - convergence claims
   - physical interpretations not supported by evidence
3. If a conclusion depends on missing evidence, write it as conditional or placeholder text.
4. If evidence is missing and the section cannot be honestly drafted, mark it as blocked instead of smoothing it over.
5. A polished placeholder is acceptable. A fabricated result is not.
Audit rule:
Any draft that presents missing evidence as if it already exists fails review and must be revised before further drafting.