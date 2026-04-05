# Figure/Table Per-Claim Plan

This document is the Stage 2 evidence-design plan derived from `doc/paper_evidence_map.md`, constrained by `doc/paper_claim_scope_nonclaims.md` and `doc/paper_writing_system.md`.

It is intentionally claim-driven rather than figure-driven. The goal is not to brainstorm visuals. The goal is to decide which figures and tables are worth building because they defend specific claims, and which claims should instead be weakened or removed.

## Operating rules

- A script existing in the repository is not the same thing as paper-ready evidence.
- A figure is only worth building if it answers a paper question tied to a claim.
- Methodological claims often still require equations, definitions, or a proposition; figures can clarify them but cannot replace them.
- Comparative claims require matched protocols and exact numbers; attractive trajectory plots are not enough.
- If a claim is weaker than the current evidence base, the right action is to weaken or remove the claim, not to build a decorative figure around it.

## Claim 1: The method operates entirely in control-point space

- Worth keeping: Yes
- Priority: primary
- Claim type: methodological
- Evidence needed: A compact formulation showing that the optimization variables are control points and that derivatives, subdivision, KOZ constraints, and boundary constraints are all written as linear maps on those control points.
- Best support form: table
- Why this support form is appropriate: This claim is about exact mathematical structure. A compact operator table is more auditable than a picture and reduces notation confusion.
- Exact figure or table concept: `T1. Control-point-space objects and linear maps.` Columns should include object/operator, dimensions, definition, and role in the optimization.
- Required data, experiment, or analysis inputs:
  - Final paper notation for the control-point matrix/vector
  - Definitions of subdivision matrices
  - Definitions of derivative maps and objective matrices
  - Definitions of boundary and KOZ constraint rows
- Input status: partially present
- Why the inputs are marked this way: The code structure exists in `orbital_docking/bezier.py`, `orbital_docking/de_casteljau.py`, `orbital_docking/constraints.py`, and `orbital_docking/optimization.py`, but the paper-ready operator table and notation lock do not yet exist.
- Production priority: must-have
- Shared support opportunities:
  - `T1` also supports Claim 3.
  - `T1` can absorb the defensible part of Claim 4 if that claim is weakened from "efficient" to "structured and reusable."

## Claim 2: Continuous spherical-KOZ avoidance is handled conservatively by De Casteljau subdivision plus supporting half-spaces

- Worth keeping: Yes, but only with explicit assumptions
- Priority: primary
- Claim type: methodological
- Evidence needed: A precise safety statement at proposition or lemma level, plus a geometric explanation of sub-arc control polygons, supporting half-spaces, and the convex-hull logic.
- Best support form: figure
- Why this support form is appropriate: The geometry is hard to parse from prose alone. A representative sub-arc figure can make the argument legible and auditable. It still must be paired with a formal statement in the text.
- Exact figure or table concept: `F1. Representative KOZ linearization on one sub-arc.` Recommended panels:
  - original curve with segmented sub-arcs and one highlighted violating sub-arc
  - control polygon of that sub-arc with KOZ, support point, normal, and half-space
  - corrected sub-arc or feasible control polygon on the safe side of the support boundary
- Required data, experiment, or analysis inputs:
  - One representative violating case
  - Sub-arc control points from subdivision
  - KOZ center and radius
  - Support point and normal construction
  - Formal assumption list for what the geometric statement does and does not guarantee
- Input status: partially present
- Why the inputs are marked this way: There is an illustration script in `figures/legacy/constraint_linearization_figures.py`, but there is no paper-ready exported figure and no formal proposition-level statement yet.
- Production priority: must-have
- Shared support opportunities:
  - `F1` also helps Claim 1 by making the control-point-space KOZ handling concrete.
  - `F1` can support Claim 5 indirectly by showing how feasibility is enforced, but it should not be treated as empirical proof.
- Critical note: Do not let this figure imply general obstacle avoidance or unconditional continuous safety. The caption must state the assumptions explicitly.

## Claim 3: The overall algorithm is an SCP method that solves a sequence of convex QPs

- Worth keeping: Yes
- Priority: primary
- Claim type: methodological
- Evidence needed: A decomposition of the optimization loop showing which pieces are fixed, which are relinearized, and what makes each subproblem convex.
- Best support form: figure
- Why this support form is appropriate: Reviewers need to see the algorithm flow quickly. A pipeline figure prevents sloppy wording such as implying one-shot convexification.
- Exact figure or table concept: `F2. SCP pipeline in control-point space.` Recommended flow:
  - initialize control points
  - assemble derivative/objective operators
  - subdivide into sub-arcs
  - build supporting half-spaces from current iterate
  - solve convex QP
  - update and repeat until convergence
- Required data, experiment, or analysis inputs:
  - Final naming of the optimization variables and operators
  - Clear separation between convex components and relinearized components
  - Final algorithm step order for the paper
- Input status: partially present
- Why the inputs are marked this way: The logic is implemented in code and described in broad terms, but no paper-specific schematic currently exists.
- Production priority: must-have
- Shared support opportunities:
  - `F2` also supports Claim 1.
  - `F2` can absorb the defensible portion of Claim 4.
  - `F2` should be drawn with domain-neutral labels so it indirectly supports Claim 10 without overclaiming validation.
- Critical note: The caption should say "sequence of convex QPs" and should not imply exact convexification of the original nonconvex problem.

## Claim 4: The structured `D`/`E`/Gram-matrix construction gives an efficient and organized derivative/objective formulation

- Worth keeping: Yes, but only in weakened form
- Priority: supporting
- Claim type: explanatory
- Evidence needed: Operator-level derivation and reuse logic. If the word "efficient" is retained, an actual benchmark against a naive implementation is needed.
- Best support form: neither
- Why this support form is appropriate: A figure cannot prove efficiency, and a dedicated table would mostly duplicate Claim 1 unless the wording is narrowed. This is primarily a wording and exposition issue.
- Exact figure or table concept: No dedicated figure or table. Fold the defensible content into `T1` and `F2`.
- Required data, experiment, or analysis inputs:
  - Definitions of `D`, `E`, and Gram-matrix use
  - Clear statement of what is reused and why
  - Optional benchmark design if speed claims are kept
- Input status: partially present
- Why the inputs are marked this way: The structured formulation exists in code, but there is no benchmark supporting a comparative efficiency claim.
- Production priority: optional
- Shared support opportunities:
  - `T1` can carry the operator mapping.
  - `F2` can show where the structured pieces enter the loop.
- Critical note: This claim is currently too weak to justify dedicated visual support in its current "efficient" wording. The safer claim is "structured, reusable, and computationally organized."

## Claim 5: The framework produces smooth, feasible trajectories for the demonstration problem

- Worth keeping: Yes
- Priority: primary
- Claim type: empirical
- Evidence needed: Representative trajectory outcomes plus quantitative feasibility and objective-aligned metrics.
- Best support form: both
- Why this support form is appropriate: The geometry must be visible, but smoothness and feasibility cannot be defended from a picture alone. A companion summary table is needed.
- Exact figure or table concept:
  - `F3. Representative optimized trajectories for selected settings.` This should be curated, not exhaustive.
  - `T2. Demonstration outcome summary.` Suggested columns: setting, solve success, minimum radius or clearance margin, objective-aligned effort metric, runtime, and iteration count.
- Required data, experiment, or analysis inputs:
  - Selected optimized trajectories
  - KOZ geometry and start/end configuration
  - Feasibility metric
  - Objective-aligned effort metric
  - Runtime and convergence metadata
- Input status: partially present
- Why the inputs are marked this way: The optimizer and visualization code exist, and metric alignment has already been addressed in `orbital_docking/visualization.py`, but there are no paper-ready generated outputs or distilled tables in the repository.
- Production priority: must-have
- Shared support opportunities:
  - `F3` and `T2` can share data with Claims 6 and 7.
  - `T2` can serve as the anchor table for a concise results subsection if carefully curated.
- Critical note: Do not mistake a gallery for evidence. If a trajectory panel does not support a concrete claim, it should be cut.

## Claim 6: Increasing subdivision count reduces conservatism and changes the cost/runtime trade-off

- Worth keeping: Yes
- Priority: primary
- Claim type: empirical
- Evidence needed: A matched sweep over subdivision count with explicit operational definitions of conservatism reduction, cost behavior, and runtime cost.
- Best support form: both
- Why this support form is appropriate: The trade-off needs exact values and an interpretable trend. A table or a figure alone would be incomplete.
- Exact figure or table concept:
  - `T3. Subdivision-count ablation.` Suggested columns: `n_seg`, solve success, minimum radius or clearance margin, objective-aligned effort metric, runtime, and outer iterations.
  - `F4. Runtime and outcome trends versus subdivision count.` Prefer separate panels rather than overloaded dual axes.
- Required data, experiment, or analysis inputs:
  - Runs across `n_seg = [2, 4, 8, 16, 32, 64]`
  - Consistent metric definitions across runs
  - A precise interpretation of "reduced conservatism"
  - Runtime metadata
- Input status: partially present
- Why the inputs are marked this way: The code already sweeps segment counts and reports times and costs, but there is no paper-ready aggregation, no explicit definition of conservatism reduction, and no final table/figure pair.
- Production priority: must-have
- Shared support opportunities:
  - `T3` and `F4` also support Claim 5.
  - These can share the same experimental batch with Claim 7 if degree is stratified carefully.
- Critical note: "Reduced conservatism" must be defined in paper terms. It cannot mean only that the plotted path looks tighter.

## Claim 7: Higher Bezier degree changes flexibility, boundary-condition accommodation, and performance

- Worth keeping: Yes, but with disciplined wording
- Priority: secondary
- Claim type: empirical
- Evidence needed: A matched degree sweep with a clear statement of what changes: feasibility, objective, smoothness, boundary-condition accommodation, or runtime.
- Best support form: both
- Why this support form is appropriate: A table is needed for exact comparisons, while a compact trend figure helps readers see how degree interacts with subdivision count.
- Exact figure or table concept:
  - `T4. Degree ablation under matched settings.` Suggested columns: degree, number of control points, solve success, safety metric, objective-aligned effort metric, runtime, and selected smoothness indicator.
  - `F5. Multi-order performance trends.` Use degree-coded series over subdivision count for one or two carefully chosen metrics.
- Required data, experiment, or analysis inputs:
  - Runs for `N = 2, 3, 4` under matched settings
  - Common boundary-condition protocol
  - Clearly defined comparison metrics
- Input status: partially present
- Why the inputs are marked this way: Multi-order optimization and plotting support already exist, but there is no distilled paper-ready ablation and no disciplined final interpretation.
- Production priority: should-have
- Shared support opportunities:
  - `T4` and `F5` can reuse the same run matrix as Claim 6.
  - A carefully designed `T4` may reduce the need for multiple extra trajectory panels.
- Critical note: Do not write "higher degree is better" unless a criterion is named and defended.

## Claim 8: The L1 delta-v proxy and L2 control-acceleration energy surrogate are alternative objective surrogates with different optimization behavior

- Worth keeping: No
- Priority: supporting
- Claim type: explanatory
- Evidence needed: None beyond enough method text to define the objective actually used in the paper.
- Best support form: neither
- Why this support form is appropriate: This claim has been dropped from the target paper. A dedicated comparison would add cognitive and narrative load without strengthening the core argument.
- Exact figure or table concept: None in the main paper.
- Required data, experiment, or analysis inputs:
  - Final definition of the objective used in the paper
  - One sentence of interpretation boundary if needed
- Input status: present
- Why the inputs are marked this way: The implementation contains both objective modes, but the paper no longer needs to turn that implementation detail into a paper claim.
- Production priority: optional
- Shared support opportunities:
  - No dedicated visual support should be allocated.
- Critical note: If objective variants are mentioned at all, they should appear only as narrow implementation context, not as a results claim.

## Claim 9: The method is useful as a warm-start generator for downstream direct collocation

- Worth keeping: Yes, but only if the comparison is actually built
- Priority: primary
- Claim type: comparative
- Evidence needed: A fair downstream comparison between direct collocation from a naive initialization and direct collocation from a Bezier-based warm start, under identical solver settings and evaluation criteria.
- Best support form: table
- Why this support form is appropriate: The central question is comparative performance under matched conditions. A table is the cleanest way to report success, convergence, time, and final quality without visual ambiguity.
- Exact figure or table concept: `T6. Downstream direct-collocation initialization comparison.` Suggested columns: initialization type, solve success, solve time, iteration count, final objective, and final constraint satisfaction. If enough cases exist, report medians and spread rather than one anecdotal run.
- Required data, experiment, or analysis inputs:
  - A direct-collocation implementation and solver protocol
  - A clearly defined naive initialization
  - Export of the Bezier result into the downstream representation
  - Matched problem instances
  - Fairness protocol with identical tolerances and stopping rules
- Input status: missing
- Why the inputs are marked this way: The evidence map identifies this as missing, and no repository artifact currently shows the downstream comparison.
- Production priority: must-have
- Shared support opportunities:
  - `T6` is the single most important table for the external-value claim.
  - If `T6` exists, it replaces any temptation to make a stronger unsupported superiority claim.
- Critical note: If `T6` cannot be produced, weaken the paper's warm-start language from demonstrated usefulness to intended use.

## Claim 10: The formulation is application-agnostic in principle

- Worth keeping: Yes, but only as a narrow positioning statement
- Priority: supporting
- Claim type: positioning
- Evidence needed: Domain-neutral derivation and wording. A second demonstration would strengthen this, but it is not currently required for the narrower phrasing.
- Best support form: neither
- Why this support form is appropriate: Without cross-domain validation, a dedicated figure would mostly advertise breadth that has not been demonstrated.
- Exact figure or table concept: No dedicated figure or table. Instead, draw `F2` and `T1` in domain-neutral notation and keep the paper wording narrow: application-agnostic in construction, not broadly validated in practice.
- Required data, experiment, or analysis inputs:
  - Domain-neutral notation and framing
  - Optional second demo only if actually completed
- Input status: partially present
- Why the inputs are marked this way: The mathematical construction is geometric, but the empirical evidence base remains single-domain.
- Production priority: optional
- Shared support opportunities:
  - `F2` and `T1` should be designed so they do not unnecessarily hard-code orbital semantics.
- Critical note: This claim is currently too weak to justify dedicated visual support.

## Claim 11: The method is better than direct collocation

- Worth keeping: No
- Priority: supporting
- Claim type: comparative
- Evidence needed: A broad benchmark suite with fair tuning, clear success criteria, and scope-matched interpretation.
- Best support form: neither
- Why this support form is appropriate: A figure or table here would suggest a claim the current paper should not pursue.
- Exact figure or table concept: None. Replace this claim with the narrower Claim 9 comparison if that experiment is built.
- Required data, experiment, or analysis inputs:
  - Broad benchmark suite
  - Careful baseline tuning
  - Clear definition of "better"
- Input status: missing
- Why the inputs are marked this way: The evidence map explicitly marks this claim as unsupported.
- Production priority: optional
- Critical note: This claim should be removed rather than illustrated.

## Claim 12: The method gives a true delta-v-optimal solution

- Worth keeping: No
- Priority: supporting
- Claim type: explanatory
- Evidence needed: True fuel-optimal formulation with appropriate dynamics, controls, and objective.
- Best support form: neither
- Why this support form is appropriate: No visual support can rescue a claim whose mathematical basis is not present in the current paper.
- Exact figure or table concept: None.
- Required data, experiment, or analysis inputs:
  - True delta-v objective
  - Proper control parameterization
  - Matching optimization formulation
- Input status: missing
- Why the inputs are marked this way: The evidence map explicitly states that current objectives are surrogates, not true mission delta-v optimization.
- Production priority: optional
- Critical note: This claim should be removed completely rather than visualized.

## Claim 13: The framework already supports dynamic obstacles through space-time extension

- Worth keeping: No for the present paper; future work only
- Priority: supporting
- Claim type: positioning
- Evidence needed: Implemented space-time formulation plus demonstrations and evidence for the present paper.
- Best support form: neither
- Why this support form is appropriate: Including a dedicated visual here would blur the paper's contribution boundary and invite reviewer criticism about scope creep.
- Exact figure or table concept: None in the present paper. At most, mention as future work in prose.
- Required data, experiment, or analysis inputs:
  - Implemented 3+1 formulation
  - Demonstrations
  - Evidence integrated into the actual paper scope
- Input status: missing
- Why the inputs are marked this way: The evidence map explicitly classifies this as not part of the present supported contribution.
- Production priority: optional
- Critical note: Do not try to rescue this by adding a teaser illustration. That would expand scope without closing the evidence gap.

## Cross-claim support consolidation

The paper should avoid one-figure-per-claim inflation. The better architecture is a small number of claim-dense figures and tables:

- `T1. Control-point-space objects and linear maps`
  - Directly supports Claims 1 and 3
  - Indirectly supports Claim 4 in weakened form
  - Can be drawn in a domain-neutral way to avoid overspecifying orbital language
- `F1. Representative KOZ linearization on one sub-arc`
  - Directly supports Claim 2
  - Helps readers understand the geometric content behind Claims 1 and 5
- `F2. SCP pipeline in control-point space`
  - Directly supports Claim 3
  - Also supports Claim 1
  - Carries the defensible portion of Claim 4 if wording is weakened
  - Can indirectly support Claim 10 if drawn with domain-neutral notation
- `F3. Representative optimized trajectories` plus `T2. Demonstration outcome summary`
  - Directly support Claim 5
  - Supply representative cases reused by Claims 6 and 7
- `T3. Subdivision-count ablation` plus `F4. Runtime and outcome trends versus subdivision count`
  - Directly support Claim 6
  - Also strengthen Claim 5
- `T4. Degree ablation` plus `F5. Multi-order performance trends`
  - Directly support Claim 7
  - Reuse the same experimental matrix as Claim 6
- `T6. Downstream direct-collocation initialization comparison`
  - Directly supports Claim 9
  - Functionally replaces any temptation to make Claim 11

## Claims too weak to justify dedicated visual support

- Claim 4 in its current "efficient" wording
- Claim 8
- Claim 10 as a broad portability claim without a second demonstration
- Claim 11
- Claim 12
- Claim 13

## Claims that should be weakened or removed rather than illustrated

- Claim 4: weaken from "efficient and organized" to "structured, reusable, and computationally organized" unless a real benchmark is added
- Claim 8: drop from the target paper as an explicit claim
- Claim 9: weaken to intended use if `T6` cannot be built
- Claim 10: phrase as application-agnostic in construction, not broadly validated in practice
- Claim 11: remove
- Claim 12: remove
- Claim 13: future work only

## Short production roadmap

### 1. Must-build figures/tables

- `T1. Control-point-space objects and linear maps`
- `F1. Representative KOZ linearization on one sub-arc`
- `F2. SCP pipeline in control-point space`
- `F3. Representative optimized trajectories for selected settings`
- `T2. Demonstration outcome summary`
- `T3. Subdivision-count ablation`
- `F4. Runtime and outcome trends versus subdivision count`
- `T6. Downstream direct-collocation initialization comparison`

### 2. Secondary figures/tables

- `T4. Degree ablation under matched settings`
- `F5. Multi-order performance trends`

### 3. Claims that should be reframed or dropped

- Reframe Claim 4 to remove unsupported efficiency language
- Drop Claim 8 from the target paper
- Reframe Claim 9 to intended use unless `T6` exists
- Reframe Claim 10 to application-agnostic construction only
- Drop Claim 11
- Drop Claim 12
- Keep Claim 13 only as future work

