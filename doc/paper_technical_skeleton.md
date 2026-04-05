# Paper Technical Skeleton

This document is the Stage 3 technical skeleton for the paper. It is not draft prose. Its job is to define the paper's argument architecture before method and results drafting begins.

## Stage 3 reasonability check

### Subjectiveness

- Low to moderate subjectiveness.
- Section naming and compression choices retain some editorial judgment, but the main structure should be determined by claims, evidence, and non-claims rather than personal style.

### Logicalness

- High.
- Stage 3 is logically necessary now because Stage 0-2 already fixed the claim boundary, evidence boundary, and figure/table plan.
- Without a technical skeleton, later drafting would likely duplicate claims across sections, create filler background, and let the comparison logic drift.

### Scientificalness

- High.
- A section-by-section argument skeleton is scientifically appropriate because it forces each section to justify a specific claim, evidence source, or limitation.
- It is especially important here because the paper is vulnerable to overclaiming, comparison sloppiness, and method/result leakage.

## Reference shorthand

### Claim IDs

- `C1`: The method operates entirely in control-point space.
- `C2`: Continuous spherical-KOZ avoidance is handled conservatively by De Casteljau subdivision plus supporting half-spaces.
- `C3`: The overall algorithm is an SCP method that solves a sequence of convex QPs.
- `C4`: The structured `D`/`E`/Gram-matrix construction gives a structured and reusable derivative/objective formulation.
- `C5`: The framework produces smooth, feasible trajectories for the demonstration problem.
- `C6`: Increasing subdivision count reduces conservatism and changes the cost/runtime trade-off.
- `C7`: Higher Bézier degree changes flexibility, boundary-condition accommodation, and performance.
- `C8`: Retired. Objective-mode comparison was dropped from the target paper and should not receive positive argumentative space.
- `C9`: The method is useful as a warm-start generator for downstream direct collocation.
- `C10`: The formulation is application-agnostic in construction, not broadly validated in practice.

### Claims that should not receive positive argumentative space

- `X1`: The method is better than direct collocation.
- `X2`: The method gives a true delta-v-optimal solution.
- `X3`: The present paper already supports dynamic obstacles through space-time extension.

### Figure/table IDs

- `T1`: Control-point-space objects and linear maps
- `F1`: Representative KOZ linearization on one sub-arc
- `F2`: SCP pipeline in control-point space
- `F3`: Representative optimized trajectories for selected settings
- `T2`: Demonstration outcome summary
- `T3`: Subdivision-count ablation
- `F4`: Runtime and outcome trends versus subdivision count
- `T4`: Degree ablation under matched settings
- `F5`: Multi-order performance trends
- `T6`: Downstream direct-collocation initialization comparison

## Paper-level architecture

Two viable skeletons exist. Both can be defended. One is more conservative and journal-shaped. The other is leaner and better aligned with the current claim discipline.

### Alternative A: Conservative journal structure


| Section                         | Role in the paper's argument                                                                                | Main dependencies                         | Keep / merge / defer notes                                  |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| 1. Introduction                 | State the problem, gap, contribution, and comparison stance.                                                | All later sections stabilize its wording. | Keep, but draft late.                                       |
| 2. Related work and positioning | Show where the method sits relative to collocation, convexification, and trajectory initialization methods. | Section 1, Section 7.                     | Risk of filler. Keep only if venue expectations require it. |
| 3. Problem setup and notation   | Define trajectory parameterization, decision variables, transfer time, and boundary conditions.             | `C1`, `T1`.                               | Keep.                                                       |
| 4. Method                       | Explain derivative structure, KOZ construction, safety assumptions, objectives, and SCP loop.               | `C2`, `C3`, `C4`, `F1`, `F2`, `T1`.       | Keep.                                                       |
| 5. Experimental setup           | Define scenario, metrics, ablation protocol, and downstream comparison fairness.                            | `T2`-`T6`.                                | Keep.                                                       |
| 6. Results                      | Present feasibility, ablations, and downstream comparison results.                                          | `C5`, `C6`, `C7`, `C9`, `F3`, `T2`-`T6`. | Keep.                                                       |
| 7. Limitations and scope        | Close overclaim loopholes and distinguish demonstrated facts from intended use.                             | `C10`, non-claims.                        | Keep.                                                       |
| 8. Conclusion                   | Compress the defended contribution after evidence is fixed.                                                 | Sections 1-7.                             | Keep, but draft last.                                       |


#### Assessment

- Strength: familiar structure, journal-safe, easy for reviewers to parse.
- Weakness: high risk of filler in Section 2 and of an oversized Section 4.
- Failure mode: generic paper anatomy with too much positioning/background and too little claim-density.

### Alternative B: Lean claim-maximizing structure


| Section                                        | Role in the paper's argument                                                                       | Main dependencies                   | Keep / merge / defer notes |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------- | -------------------------- |
| 1. Introduction and claim boundary             | Authorize the exact paper question, contribution, and non-claims.                                  | Sections 2-7 stabilize its wording. | Keep, but draft late.      |
| 2. Problem formulation in control-point space  | Prove that the method's variables and operators live in control-point space.                       | `C1`, `C4`, `T1`.                   | Keep.                      |
| 3. Conservative KOZ handling                   | Prove the geometric logic behind continuous spherical-KOZ handling under explicit assumptions.     | `C2`, `F1`.                         | Keep.                      |
| 4. Objective and SCP algorithm                 | Show what is optimized, what is linearized, and why each outer iteration solves a convex QP.       | `C3`, `C4`, `F2`, `T1`.             | Keep.                      |
| 5. Experimental design and comparison protocol | Define exactly how each claim will be tested and how fairness is enforced.                         | `T2`-`T6`.                          | Keep.                      |
| 6. Results                                     | Deliver the evidence claim-by-claim.                                                               | `C5`, `C6`, `C7`, `C9`, `F3`, `T2`-`T6`. | Keep.                    |
| 7. Limitations and claim boundary              | State what the paper does not prove and how portability/comparison language should be interpreted. | `C10`, non-claims.                  | Keep.                      |
| 8. Conclusion                                  | Compress only what has been defended.                                                              | Sections 1-7.                       | Keep, but draft last.      |


#### Assessment

- Strength: each section answers one argumentative question.
- Strength: lower filler risk than the conservative structure.
- Strength: cleaner separation between method logic, experiment logic, and claim boundaries.
- Weakness: some venues expect a more explicit related-work section.
- Failure mode: if Section 1 becomes too compressed, positioning can become underexplained.

### Recommended structure

Recommend `Alternative B: Lean claim-maximizing structure`.

Why:

- It is less likely to produce a generic AI-written paper.
- It directly matches the Stage 3 rule that each section must solve one problem in the paper's argument.
- It suppresses the two biggest filler temptations: standalone background and bloated related-work taxonomy.
- It makes the direct-collocation comparison live where it belongs: first in protocol, then in results, not as vague motivation.

This choice is now fixed for the paper unless a later venue constraint forces a structural adaptation.

### Dependency order

The logical dependency order is not the same as the final paper order.

#### Logical dependency order for drafting

1. Section 2: Problem formulation in control-point space
2. Section 3: Conservative KOZ handling
3. Section 4: Objective and SCP algorithm
4. Section 5: Experimental design and comparison protocol
5. Section 6: Results
6. Section 7: Limitations and claim boundary
7. Section 1: Introduction and claim boundary
8. Section 8: Conclusion

#### Final paper order

1. Introduction and claim boundary
2. Problem formulation in control-point space
3. Conservative KOZ handling
4. Objective and SCP algorithm
5. Experimental design and comparison protocol
6. Results
7. Limitations and claim boundary
8. Conclusion

### Sections that should be merged, deferred, or removed

- Merge any standalone Bézier background into Section 2 unless a definition is strictly needed.
- Merge any standalone related-work taxonomy into Section 1 unless a venue later requires a dedicated section.
- Remove any standalone orbital-background section. The orbital scenario belongs in experimental design, not in paper framing.
- Remove any standalone future-work section. Future work belongs only as a short tail in Section 7 or Section 8.
- Defer abstract and title until Stage 6.
- Defer polished introduction and conclusion drafting until after Sections 2-7 stabilize.

## Detailed skeleton for the recommended structure

This section provides the detailed skeleton for `Alternative B`.

## Section 1. Introduction and claim boundary

- Purpose: authorize the exact paper question, contribution, scope, and comparison stance.
- Key message: the paper presents a control-point-space trajectory-initialization framework for continuous spherical-KOZ avoidance that produces smooth, safety-respecting warm starts through conservative subdivision-based constraints and SCP.
- What question this section answers: what exactly is the paper trying to do, and what is it explicitly not trying to prove?
- Supports claims: `C1`, `C2`, `C3`, `C5`, `C9` as target claim, `C10` in narrow form.
- Depends on figures/tables: none directly, but final wording must remain consistent with `T6`.
- What must appear here:
  - narrow problem statement
  - the gap being solved
  - contribution statement
  - comparison stance: warm-start value, not planner replacement
  - explicit non-claims
- What must be excluded:
  - long orbital mission motivation
  - broad planner taxonomy
  - Bézier history survey
  - results preview beyond one-sentence paper roadmap if needed
  - any superiority implication over direct collocation
- Likely reviewer objections:
  - "What is actually new here?"
  - "Is this just another SCP planner?"
  - "Are you claiming generality from one domain?"
  - "Are you implying replacement of collocation?"
- Current status: premature

### 1.1 Problem setting and gap

- Purpose: define the narrow problem the paper solves.
- Key message: continuous spherical-KOZ avoidance with smooth warm-start generation is awkward under pointwise formulations, and the paper proposes a control-point-space alternative.
- What question this subsection answers: what concrete technical problem motivates the formulation?
- Supports claims: `C1`, `C2`, `C3`
- Depends on figures/tables: none
- Must include:
  - one clean problem statement
  - why continuity of safety matters
  - why warm-start generation is the relevant role
- Exclude:
  - orbital mission narrative
  - full related-work review
  - language about global optimality or universal applicability
- Likely reviewer objections:
  - "Why not use direct collocation directly?"
  - "Why is control-point space worth the extra notation?"
- Current status: premature

### 1.2 Contribution statement and non-claims

- Purpose: lock the contribution boundary early.
- Key message: the contribution is a framework-level combination of control-point-space formulation, conservative KOZ handling, and SCP-based warm-start generation.
- What question this subsection answers: what is the paper claiming, and what is it refusing to claim?
- Supports claims: `C1`, `C2`, `C3`, `C5`, `C9`, `C10`
- Depends on figures/tables: none
- Must include:
  - concise contribution list
  - explicit non-claims from `doc/paper_claim_scope_nonclaims.md`
  - narrow framing of `C10`
- Exclude:
  - any claim corresponding to `X1`, `X2`, or `X3`
  - claims of real-time capability
  - claims of exact convexification
- Likely reviewer objections:
  - "This sounds overframed relative to the evidence."
  - "Where is the warm-start usefulness actually shown?"
- Current status: premature

## Section 2. Problem formulation in control-point space

- Purpose: establish the mathematical objects, variables, and linear operator structure before KOZ logic or algorithmic details appear.
- Key message: the optimization variables are Bézier control points, and derivative/boundary structure is expressed as linear maps on them.
- What question this section answers: what is the actual optimization representation?
- Supports claims: `C1`, `C4`
- Depends on figures/tables: `T1`
- What must appear here:
  - trajectory parameterization
  - decision variables
  - fixed transfer time treatment
  - boundary-condition encoding
  - derivative mappings via `D` and `E`
  - operator summary through `T1`
- What must be excluded:
  - KOZ supporting-half-space derivation
  - SCP loop logic
  - empirical claims
  - efficiency superiority claims
- Likely reviewer objections:
  - "Are the variables really only control points?"
  - "How are physical derivatives recovered?"
  - "Is time optimized or fixed?"
- Current status: ready

### 2.1 Trajectory parameterization and decision variables

- Purpose: define the curve, its degree, control points, and the fixed-time setting.
- Key message: the paper solves over control points rather than pointwise state/control samples.
- What question this subsection answers: what exactly is being optimized?
- Supports claims: `C1`
- Depends on figures/tables: `T1`
- Must include:
  - Bézier curve definition
  - vectorized decision variable definition
  - dimensional conventions
  - fixed transfer time assumption
- Exclude:
  - algorithm pseudocode
  - safety proof language
  - scenario-specific constants unless strictly needed
- Likely reviewer objections:
  - "Why fixed transfer time?"
  - "What role do endpoints and optional boundary derivatives play?"
- Current status: ready

### 2.2 Boundary conditions and derivative operator structure

- Purpose: show how velocity and acceleration structure are obtained without leaving control-point space.
- Key message: derivative and boundary information comes from structured linear maps, not repeated pointwise differentiation.
- What question this subsection answers: how are derivatives and boundary constraints represented?
- Supports claims: `C1`, `C4`
- Depends on figures/tables: `T1`
- Must include:
  - `D` and `E` mappings
  - how velocity and acceleration control points are formed
  - boundary equality form
  - the precise weakened wording of `C4`
- Exclude:
  - runtime benchmark rhetoric
  - KOZ construction
  - downstream comparison discussion
- Likely reviewer objections:
  - "Is this really a contribution or just notation?"
  - "What exactly is reused?"
- Current status: ready

## Section 3. Conservative KOZ handling

- Purpose: establish the geometric safety logic and its assumptions.
- Key message: continuous spherical-KOZ avoidance is enforced conservatively by subdividing the curve into sub-arcs and constraining each sub-arc control polygon with a supporting half-space.
- What question this section answers: why does the method's KOZ handling support a continuous-safety claim, and under what assumptions?
- Supports claims: `C2`
- Depends on figures/tables: `F1`, secondarily `T1`
- What must appear here:
  - De Casteljau subdivision into sub-arcs
  - sub-arc control polygons
  - support point and normal construction
  - formal safety statement with assumptions
  - explicit scope: spherical KOZ only
- What must be excluded:
  - non-spherical or dynamic obstacle generalization
  - experimental trade-off interpretation
  - claims of exact safety beyond assumptions
- Likely reviewer objections:
  - "Where exactly does the guarantee come from?"
  - "Is the guarantee geometric or numerical?"
  - "What fails if assumptions are violated?"
- Current status: ready

### 3.1 De Casteljau subdivision into sub-arcs

- Purpose: define the sub-arc objects on which safety constraints are imposed.
- Key message: subdivision creates control polygons for smaller curve pieces while staying in control-point space.
- What question this subsection answers: how do we localize the continuous curve into manageable pieces?
- Supports claims: `C2`
- Depends on figures/tables: `F1`, `T1`
- Must include:
  - segmentation operator or matrix description
  - sub-arc control-point definition
  - notation consistent with Section 2
- Exclude:
  - experimental metrics
  - SCP loop discussion
- Likely reviewer objections:
  - "Why equal-parameter subdivision?"
  - "Does the number of sub-arcs affect correctness or only conservatism?"
- Current status: ready

### 3.2 Supporting-half-space construction

- Purpose: define how a local linear safety constraint is built for each sub-arc.
- Key message: each sub-arc receives a tangent-like supporting half-space anchored to the spherical KOZ geometry.
- What question this subsection answers: how is the local safety constraint constructed?
- Supports claims: `C2`
- Depends on figures/tables: `F1`
- Must include:
  - centroid ~~or representative-point choice~~
  - support point
  - outward normal
  - control-point half-space inequalities
- Exclude:
  - language implying exact obstacle representation
  - dynamic obstacles
- Likely reviewer objections:
  - "Why is the centroid-based construction appropriate?"
  - "What is conservative here, precisely?"
- Current status: ready

### 3.3 Safety statement, assumptions, and scope boundary

- Purpose: convert the geometric construction into a precise defensible statement.
- Key message: the method provides conservative continuous spherical-KOZ avoidance under explicit assumptions; it is not a general obstacle-avoidance theorem.
- What question this subsection answers: what is guaranteed, and what is not?
- Supports claims: `C2`
- Depends on figures/tables: `F1`
- Must include:
  - proposition or lemma-level statement
  - explicit assumptions
  - one short explanation of the convex-hull argument
  - explicit non-extension to general obstacles
- Exclude:
  - informal convenience phrasing like "continuous safety guarantee" without assumptions
  - future-work content
- Likely reviewer objections:
  - "This still sounds heuristic."
  - "Is the guarantee about the curve, the control polygon, or sampled points?"
- Current status: ready

## Section 4. Objective and SCP algorithm

- Purpose: explain what is optimized, how nonconvex terms are handled, and why each outer iteration is a convex QP solve.
- Key message: the method solves a sequence of convex QPs in control-point space by updating KOZ half-spaces and linearized gravity/J2-dependent objective terms around the current iterate.
- What question this section answers: what exactly is the optimization problem at each iteration, and what is the status of convexity?
- Supports claims: `C3`, `C4`
- Depends on figures/tables: `F2`, `T1`
- What must appear here:
  - objective definitions
  - gravity/J2 linearization logic
  - convex subproblem form
  - SCP update loop
  - precise wording that this is a sequence of convex QPs
- What must be excluded:
  - claims of exact convexification
  - claims of true delta-v optimality
  - runtime superiority rhetoric
- Likely reviewer objections:
  - "What exactly is convex here?"
  - "What is relinearized each iteration?"
  - "What do the objectives mean physically?"
- Current status: ready

### 4.1 Objective definitions and interpretation boundary

- Purpose: define the objectives and their limits.
- Key message: the objectives are optimization surrogates, not exact mission-optimality claims.
- What question this subsection answers: what is the method minimizing, and what should the reader not infer from it?
- Supports claims: `C4`
- Depends on figures/tables: none directly
- Must include:
  - mathematical definition of the main objective
  - role of gravity/J2 linearization in the surrogate
  - explicit statement that neither objective proves true delta-v optimality
- Exclude:
  - claims of physical superiority for one surrogate
  - any objective-mode subplot treated as a paper contribution
- Likely reviewer objections:
  - "Why should I care about this surrogate?"
  - "Are you overselling the objective's physical meaning?"
- Current status: ready

### 4.2 Convex subproblem structure

- Purpose: show the per-iteration problem statement precisely.
- Key message: once KOZ half-spaces and linearized terms are fixed, the subproblem is a convex quadratic program over control points.
- What question this subsection answers: what is solved at each outer iteration?
- Supports claims: `C3`
- Depends on figures/tables: `F2`, `T1`
- Must include:
  - variable stack
  - quadratic objective structure
  - linear constraints
  - statement of what remains fixed within one iteration
- Exclude:
  - convergence overclaiming
  - detailed numerical implementation trivia
- Likely reviewer objections:
  - "Why call this convex?"
  - "What about the original nonconvex problem?"
- Current status: ready

### 4.3 SCP loop and update logic

- Purpose: explain how the method moves across iterations.
- Key message: the method alternates between rebuilding conservative linear constraints / linearized terms and solving the resulting convex subproblem.
- What question this subsection answers: how do the outer iterations work, and what is their role?
- Supports claims: `C3`
- Depends on figures/tables: `F2`
- Must include:
  - iteration order
  - what gets updated
  - stopping logic
  - careful wording about algorithmic role versus theoretical guarantee
- Exclude:
  - exaggerated convergence claims
  - detailed ablation interpretation
- Likely reviewer objections:
  - "What prevents drift or poor local behavior?"
  - "How sensitive is this to initialization?"
- Current status: ready

## Section 5. Experimental design and comparison protocol

- Purpose: define the evidence structure before showing results.
- Key message: each experiment exists to answer a specific claim, and the downstream comparison is framed as a fairness-sensitive initialization study rather than a winner-take-all benchmark.
- What question this section answers: what exactly is being measured, and how is comparison fairness enforced?
- Supports claims: `C5`, `C6`, `C7`, `C9`
- Depends on figures/tables: `T2`, `T3`, `F4`, `T4`, `F5`, `T6`
- What must appear here:
  - scenario definition
  - metrics
  - ablation protocol
  - downstream direct-collocation comparison protocol
  - explicit fairness rules
- What must be excluded:
  - interpretation of results
  - any superiority language
  - broad cross-domain claims
- Likely reviewer objections:
  - "Are the metrics aligned with the objective?"
  - "Is the collocation comparison fair?"
  - "Are the ablations actually diagnostic?"
- Current status: blocked

### 5.1 Demonstration scenario and reporting metrics

- Purpose: define the base scenario and common metrics.
- Key message: the demonstration section will use objective-aligned metrics, feasibility checks, and runtime/convergence metadata rather than only visual trajectory appeal.
- What question this subsection answers: what is the core scenario, and how will outcomes be measured?
- Supports claims: `C5`
- Depends on figures/tables: `F3`, `T2`
- Must include:
  - scenario specification
  - minimum-radius or clearance metric
  - objective-aligned effort metric
  - runtime and iteration count
- Exclude:
  - interpretation of ablation trends
  - downstream comparison claims
- Likely reviewer objections:
  - "Are these metrics aligned with the objective or just visually convenient?"
- Current status: ready

### 5.2 Subdivision-count and degree ablation protocol

- Purpose: define how `C6` and `C7` are tested.
- Key message: segment-count and degree sweeps are controlled diagnostic studies, not fishing expeditions.
- What question this subsection answers: how will trade-offs be measured and interpreted?
- Supports claims: `C6`, `C7`
- Depends on figures/tables: `T3`, `F4`, `T4`, `F5`
- Must include:
  - sweep ranges
  - matched settings
  - operational definition of reduced conservatism
  - exact metrics reported
- Exclude:
  - statements that higher degree is automatically better
  - trajectory galleries used as evidence
- Likely reviewer objections:
  - "What exactly does reduced conservatism mean?"
  - "Are these trends general or scenario-specific?"
- Current status: ready

### 5.3 Downstream direct-collocation comparison protocol

- Purpose: define a fair test of the warm-start claim.
- Key message: the downstream comparison is an initialization comparison under a shared solver/problem setup, not a broad planner-class contest.
- What question this subsection answers: how will naive initialization and Bézier warm starts be compared fairly?
- Supports claims: `C9`
- Depends on figures/tables: `T6`
- Must include:
  - downstream problem definition
  - naive initialization definition
  - warm-start export mapping
  - common solver settings and stopping rules
  - outcome metrics: success, time, iterations, final objective, final constraint satisfaction
- Exclude:
  - claims that direct collocation is the wrong baseline
  - any language implying general superiority
- Likely reviewer objections:
  - "Is this comparison fair?"
  - "Is the downstream problem aligned with the upstream one?"
  - "Is the naive initialization artificially weak?"
- Current status: blocked

## Section 6. Results

- Purpose: deliver evidence claim-by-claim with explicit claim-to-evidence linkage.
- Key message: the method demonstrates feasible warm-start trajectories, interpretable trade-offs, and potentially downstream initialization value if the comparison is completed fairly.
- What question this section answers: what does the evidence actually show, and what does it not show?
- Supports claims: `C5`, `C6`, `C7`, `C9`
- Depends on figures/tables: `F3`, `T2`, `T3`, `F4`, `T4`, `F5`, `T6`
- What must appear here:
  - one conclusion per subsection tied to one claim
  - tables/figures before interpretation
  - explicit restraint where evidence is weak
- What must be excluded:
  - re-derivation of the method
  - broad portability rhetoric
  - any suggestion that the method replaces collocation
- Likely reviewer objections:
  - "Do these results actually support the wording in the introduction?"
  - "Are the comparisons fair?"
  - "Are the conclusions stronger than the data?"
- Current status: blocked

### 6.1 Demonstration feasibility and representative trajectories

- Purpose: show that the method produces smooth, feasible, safety-respecting trajectories on the target scenario.
- Key message: the framework works on the demonstration problem and the representative trajectories are supported by quantitative safety and effort metrics.
- What question this subsection answers: does the method actually work on the demonstration case?
- Supports claims: `C5`
- Depends on figures/tables: `F3`, `T2`
- Must include:
  - representative trajectories
  - solve success
  - safety metric
  - objective-aligned effort metric
- Exclude:
  - generalized claims beyond the demonstration case
  - excessive gallery content
- Likely reviewer objections:
  - "Is this just a nice picture?"
- Current status: blocked

### 6.2 Subdivision-count trade-off

- Purpose: show how segment count affects conservatism and computational cost.
- Key message: increasing `n_seg` refines the supporting-half-space approximation; this subsection evaluates how that affects conservatism and computational cost.
- What question this subsection answers: what does `n_seg` change, and at what computational cost?
- Supports claims: `C6`
- Depends on figures/tables: `T3`, `F4`
- Must include:
  - operational interpretation of reduced conservatism
  - quantitative trade-off
  - scenario-specific caution if needed
- Exclude:
  - blanket monotonicity claims not supported by results
  - claims that tighter-looking paths automatically mean better solutions
- Likely reviewer objections:
  - "How is conservatism being measured?"
  - "Are the trends robust or anecdotal?"
- Current status: blocked

### 6.3 Degree trade-off

- Purpose: show what higher degree changes and what it does not.
- Key message: degree changes expressiveness and performance characteristics, but "better" must be tied to a named metric.
- What question this subsection answers: what, specifically, does degree buy or cost?
- Supports claims: `C7`
- Depends on figures/tables: `T4`, `F5`
- Must include:
  - a clearly named criterion
  - matched comparisons
  - restrained interpretation
- Exclude:
  - vague claims of flexibility without consequence
  - claims that higher degree dominates lower degree
- Likely reviewer objections:
  - "Why is degree here if its effect is not clearly defined?"
- Current status: blocked

### 6.4 Downstream warm-start comparison

- Purpose: test the paper's most important external-value claim.
- Key message: if the comparison succeeds, the method shows value as an initializer for downstream direct collocation under a fair shared setup.
- What question this subsection answers: does the Bézier result help downstream direct collocation relative to a naive initialization?
- Supports claims: `C9`
- Depends on figures/tables: `T6`
- Must include:
  - fair baseline definition
  - quantitative comparison
  - explicit statement of what the comparison does and does not prove
- Exclude:
  - planner-class superiority language
  - generalization beyond the tested setup
- Likely reviewer objections:
  - "Was the baseline weakened?"
  - "Is this only one scenario?"
  - "Does warm-start value mean method superiority?"
- Current status: blocked

## Section 7. Limitations and claim boundary

- Purpose: expose the paper's limits before reviewers do.
- Key message: the paper demonstrates a coherent, conservative initialization framework in one domain under specific assumptions; it does not prove planner superiority, cross-domain validation, or true fuel-optimality.
- What question this section answers: what is the exact boundary between demonstrated fact and extrapolation?
- Supports claims: `C10` and all explicit non-claims
- Depends on figures/tables: optionally references `T6`, otherwise none
- What must appear here:
  - assumption boundary
  - evidence boundary
  - comparison boundary
  - portability boundary
  - future-work boundary
- What must be excluded:
  - apology tone
  - speculative research agenda
  - re-litigation of results
- Likely reviewer objections:
  - "Are the limitations stated early enough and clearly enough?"
  - "Does the paper quietly rely on claims it later disowns?"
- Current status: ready

### 7.1 What is guaranteed and under which assumptions

- Purpose: restate the exact safety and optimization-status boundary.
- Key message: the conservative safety logic has a narrow assumption set, and the algorithm remains an SCP method rather than an exact convex reformulation.
- What question this subsection answers: what should the reader treat as formally supported?
- Supports claims: `C2`, `C3`
- Depends on figures/tables: may point back to `F1` and `F2`
- Must include:
  - spherical-KOZ assumption
  - supporting-half-space assumption
  - sequence-of-convex-QPs language
- Exclude:
  - generic safety rhetoric
- Likely reviewer objections:
  - "Is the guarantee stronger in the results section than here?"
- Current status: ready

### 7.2 What the paper demonstrates versus merely suggests

- Purpose: separate empirical evidence from intended use.
- Key message: demonstration, ablation, and downstream warm-start evidence have different epistemic status and should not be merged rhetorically.
- What question this subsection answers: which statements are directly shown, and which are only plausible?
- Supports claims: `C5`, `C6`, `C7`, `C9`
- Depends on figures/tables: `T2`-`T6`
- Must include:
  - one sentence each on what the paper demonstrates
  - one sentence each on what remains suggested rather than shown
  - fallback wording if any evidence remains incomplete
- Exclude:
  - strong performance rhetoric
- Likely reviewer objections:
  - "Are you smuggling intended use into demonstrated capability?"
- Current status: ready

### 7.3 Portability and future-work boundary

- Purpose: keep `C10` narrow and push non-present extensions out of the contribution.
- Key message: the formulation is application-agnostic in construction, but the evidence base is single-domain and the present paper does not cover dynamic obstacles.
- What question this subsection answers: how broad is the formulation in principle, and how narrow is the current validation?
- Supports claims: `C10`
- Depends on figures/tables: none
- Must include:
  - narrow portability wording
  - explicit exclusion of dynamic-obstacle claims from the present paper
- Exclude:
  - teaser-style future-work marketing
  - domain-general validation rhetoric
- Likely reviewer objections:
  - "Why mention generality at all without more demonstrations?"
- Current status: ready

## Section 8. Conclusion

- Purpose: compress only the defended contribution and its evidence-backed significance.
- Key message: the contribution is a conservative control-point-space warm-start framework, not a replacement for downstream planners or a proof of general optimality.
- What question this section answers: after all evidence and limitations are accounted for, what remains as the paper's defensible takeaway?
- Supports claims: `C1`, `C2`, `C3`, `C5`, `C9`, `C10`
- Depends on figures/tables: all main results are stabilized first
- What must appear here:
  - narrow contribution summary
  - one sentence on evidence-backed value
  - one sentence on limitations
- What must be excluded:
  - any new claim
  - universal framing
  - speculative future-work inflation
- Likely reviewer objections:
  - "Does the conclusion overstate the body?"
- Current status: premature

## Claim coverage audit

### Claim-to-section map


| Claim | Primary home                                             | Secondary mentions allowed                 | Audit note                                                       |
| ----- | -------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| `C1`  | Section 2                                                | Sections 1 and 4                           | Good. Keep the technical burden in Section 2.                    |
| `C2`  | Section 3                                                | Sections 1 and 7                           | Good. Do not fragment the safety logic across the paper.         |
| `C3`  | Section 4                                                | Sections 1 and 7                           | Good. Keep convexity language centralized in Section 4.          |
| `C4`  | Section 2 and Section 4                                  | None beyond brief mention                  | Risk of overpromotion. Keep it supporting, not headline.         |
| `C5`  | Section 6.1                                              | Section 5.1 protocol, Section 7.2 boundary | Good.                                                            |
| `C6`  | Section 6.2                                              | Section 5.2 protocol                       | Good.                                                            |
| `C7`  | Section 6.3                                              | Section 5.2 protocol                       | Good, but secondary in narrative priority.                       |
| `C8`  | Retired                                                  | None                                       | Dropped claim. Do not reuse this ID for a live paper claim.      |
| `C9`  | Section 5.3 protocol and Section 6.4 results             | Section 1 motivation, Section 7.2 boundary | Necessary two-home claim. Protocol and evidence must both exist. |
| `C10` | Section 7.3                                              | Section 1 narrow framing                   | Good if kept narrow.                                             |


### Claims that should not have a positive home

- `X1` should appear only as an explicit non-claim.
- `X2` should appear only as an explicit non-claim.
- `X3` should appear only in future-work boundary language.

### Sections that do not serve a real argumentative purpose

These should not exist as standalone sections in the current paper:

- standalone Bézier background
- standalone orbital background
- standalone planner taxonomy / generic related work
- standalone application-agnostic discussion
- standalone future-work section

### Sections at risk of duplication or filler

- Section 1 and Section 7 can duplicate non-claims if not kept distinct.
  - Section 1 should state the boundary early.
  - Section 7 should revisit the boundary after evidence.
- Section 2 and Section 4 can duplicate operator content if `C4` is allowed to sprawl.
  - Section 2 should carry representation and operator definitions.
  - Section 4 should carry optimization meaning and algorithmic role.
- Section 5 and Section 6 can duplicate claims if protocol and interpretation are mixed.
  - Section 5 defines how evidence will be obtained.
  - Section 6 states what the evidence shows.

### Generic-paper warning

If the outline drifts toward:

- a long motivation section,
- a broad related-work survey,
- a Bézier tutorial,
- a generic orbital background section,
- and a discussion section that mostly restates results,

then it will likely produce a generic AI-written paper rather than a sharp method-and-evidence paper.

## Decision minimization

The major structure decisions are now fixed:

- Lean claim-maximizing skeleton adopted.
- Degree ablation remains in the main text as a secondary result.
- Objective-mode comparison is dropped as a paper claim rather than retained as an optional subplot.

No further paper-architecture decisions are required at this stage.

The remaining open decisions are experiment-protocol choices for Section 5.3:

- exact downstream problem definition
- exact naive initialization definition
- exact warm-start export mapping
- exact shared solver and stopping-rule protocol

## Fault-detection pass

### Overclaim risk

- Section 1 can easily overpromise if it implies:
  - exact convexification
  - general obstacle avoidance
  - broad cross-domain validity
  - superiority to direct collocation
- Section 6.4 can easily overclaim if warm-start value is written as planner replacement.
- Section 7 must explicitly neutralize these risks rather than assuming the reader will infer the boundary.

### Unsupported transitions

- Transition risk from Section 3 to Section 6:
  - geometric conservatism does not automatically imply empirical usefulness
- Transition risk from Section 4 to Section 6:
  - a coherent optimization loop does not automatically imply better results
- Transition risk from Section 6.1 to Section 6.4:
  - demonstration success does not automatically imply downstream initialization value

### Weak comparison logic

- The direct-collocation comparison is the strongest potential claim and the easiest place to fail scientifically.
- Section 5.3 must define fairness tightly:
  - same downstream problem
  - same stopping rules
  - same solver tolerances
  - same constraint reporting
  - honest naive initialization
- If Section 5.3 stays vague, Section 6.4 becomes non-defensible.

### Vague contribution wording

- `C4` becomes weak if phrased as efficiency without measurement.
- `C10` becomes weak if phrased as generality rather than construction-level portability.

### Unnecessary background

- The paper does not need:
  - a Bézier history lesson
  - a long orbital rendezvous background section
  - a general tutorial on trajectory optimization methods
- Any such section would be filler unless a venue later requires a minimal related-work section.

### Premature sections

These should not be drafted in polished form yet:

- Section 1: Introduction and claim boundary
- Section 6: Results
- Section 8: Conclusion

These are especially dependent on not-yet-built evidence:

- Section 5.3: Downstream direct-collocation comparison protocol
- Section 6.4: Downstream warm-start comparison

### Structural warning

If the paper tries to make all of `C5`, `C6`, `C7`, `C9`, and `C10` feel equally central, the structure will become diffuse.

The actual hierarchy should be:

1. `C1`, `C2`, `C3` as the method core
2. `C5` and `C9` as the main evidence-backed value claims
3. `C6` as the main ablation
4. `C7` and `C10` as secondary/supporting material

## End-state summary

### 1. Sections ready to draft now

- Section 2: Problem formulation in control-point space
- Section 3: Conservative KOZ handling
- Section 4: Objective and SCP algorithm
- Section 5.1: Demonstration scenario and reporting metrics
- Section 5.2: Subdivision-count and degree ablation protocol
- Section 7: Limitations and claim boundary

### 2. Sections blocked by missing evidence

- Section 5.3: Downstream direct-collocation comparison protocol
- Section 6: Results
- Section 6.4: Downstream warm-start comparison

### 3. Sections that should be postponed until later stages

- Section 1: Introduction and claim boundary
- Section 8: Conclusion

### 4. Minimal set of decisions you need to make next

No additional paper-architecture decisions are required now. The remaining open decisions are experiment-protocol choices for Section 5.3.

