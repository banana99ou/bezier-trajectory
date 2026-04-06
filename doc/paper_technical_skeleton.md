# Paper Technical Skeleton

This document is the Stage 3 technical skeleton for the paper. It is not draft prose. Its job is to define the paper's argument architecture before method and results drafting begins.

## Stage 3 reasonability check

### Subjectiveness

- Moderate.
- The rollback from the lean structure to the conservative structure is a paper-shape decision, not a change in what the method or evidence can support.
- That subjectiveness is acceptable only if the structure improves reviewability without diluting claim density.

### Logicalness

- Conditionally high.
- Stage 3 remains logically necessary because Stage 0 through Stage 2 already fixed the claim boundary, evidence boundary, and figure/table plan.
- The conservative structure is logically acceptable if it keeps setup, method, protocol, and results distinct while resisting filler in related work.

### Scientificalness

- Neutral to positive.
- The science does not become stronger merely by changing section names.
- The scientific value of this rollback is that assumptions, protocol, and result boundaries may become easier to audit in a reviewer-familiar journal shape.

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

The adopted source-of-truth structure is:

### Alternative A: Conservative journal structure

This rollback is now fixed for Stage 3 control purposes.

Important qualification:

- `Section 2: Related work and positioning` remains part of the conservative source-of-truth.
- That section must stay minimal and claim-serving.
- If a later venue clearly does not want a standalone related-work section, it may be merged into the introduction without changing the claim/evidence system.
- Until that later venue adaptation exists, the conservative eight-section structure remains the control structure.

| Section                         | Role in the paper's argument                                                                               | Main dependencies                   | Keep / merge / defer notes                                                  |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------- |
| 1. Introduction                 | State the problem, gap, contribution, and comparison stance.                                               | Sections 2-7 stabilize its wording. | Keep, but draft late.                                                       |
| 2. Related work and positioning | Place the method relative to collocation, convexification, and trajectory-initialization work.            | Sections 1 and 7.                   | Keep as a short section only. Do not let it become taxonomy filler.         |
| 3. Problem setup and notation   | Define trajectory parameterization, decision variables, transfer time, boundary conditions, and notation. | `C1`, supporting part of `C4`, `T1` | Keep. This consolidates the lean formulation split into a setup section.    |
| 4. Method                       | Explain derivative structure, KOZ construction, safety assumptions, objective, and SCP loop.              | `C2`, `C3`, `C4`, `F1`, `F2`, `T1` | Keep. Large but defensible if tightly organized.                            |
| 5. Experimental setup           | Define scenario, metrics, ablation protocol, and downstream comparison fairness.                           | `T2`-`T6`                           | Keep. Protocol belongs here, not in Results.                                |
| 6. Results                      | Present demonstration, ablations, and downstream comparison results.                                       | `C5`, `C6`, `C7`, `C9`, `F3`, `T2`-`T6` | Keep. Deliver evidence claim by claim.                                  |
| 7. Limitations and scope        | Close overclaim loopholes and separate demonstrated facts from intended use.                               | `C10`, non-claims                   | Keep. This is where narrow framing is reasserted after results.             |
| 8. Conclusion                   | Compress only what has actually been defended.                                                             | Sections 1-7                        | Keep, but draft last.                                                       |

## Structural discipline rules

- Do not create a standalone Bezier background section.
- Do not create a standalone orbital background section.
- Do not create a standalone future-work section.
- Do not let Section 2 become a literature dump. It exists only to sharpen positioning and comparison stance.
- Do not let Section 4 absorb experimental fairness or result interpretation.
- Do not let Section 6 restate protocol instead of reporting findings.
- Do not revive objective-mode comparison as a positive paper claim.

## Drafting dependency order

The logical dependency order is not the same as the final paper order.

### Logical dependency order for drafting

1. Section 3: Problem setup and notation
2. Section 4: Method
3. Section 5: Experimental setup
4. Section 6: Results
5. Section 7: Limitations and scope
6. Section 2: Related work and positioning
7. Section 1: Introduction
8. Section 8: Conclusion

### Final paper order

1. Introduction
2. Related work and positioning
3. Problem setup and notation
4. Method
5. Experimental setup
6. Results
7. Limitations and scope
8. Conclusion

## Sections that should be merged, deferred, or removed

- Merge any Bezier history or generic Bernstein background into Section 3 only if a definition is strictly needed.
- Keep Section 2 short. If it starts to look like a planner taxonomy, cut it.
- Remove any standalone orbital-motivation section. The orbital scenario belongs in Sections 5 and 6.
- Remove any standalone objective-variant section. Objective-mode comparison is not a live paper claim.
- Defer title and abstract until after Sections 3-7 stabilize.
- Defer polished Introduction and Conclusion until after evidence is fixed.

## Detailed skeleton for the adopted structure

## Section 1. Introduction

- Purpose: state the paper question, gap, contribution, and comparison stance.
- Key message: the paper presents a Bezier-based trajectory-initialization framework for continuous spherical-KOZ avoidance that operates in control-point space and is intended to generate smooth, safety-respecting warm starts.
- Supports claims: `C1`, `C2`, `C3`, `C5`, narrow `C9`, narrow `C10`.
- What must appear:
  - narrow problem statement
  - why continuous safety plus smooth initialization is awkward under pointwise formulations
  - contribution statement
  - comparison stance: warm-start value, not planner replacement
  - explicit non-claims
- What must be excluded:
  - long orbital mission motivation
  - generic trajectory-optimization tutorial
  - broad literature taxonomy that belongs in Section 2

## Section 2. Related work and positioning

- Purpose: position the method relative to the specific comparison classes that matter for the paper's claims.
- Key message: the paper sits at the intersection of trajectory initialization, conservative continuous-safety handling, and SCP-style constrained optimization, but it is not claiming dominance over direct collocation or a full replacement of downstream planners.
- Supports claims: mainly Section 1 framing and Section 7 boundary discipline; no new empirical claim should originate here.
- What must appear:
  - short positioning against direct collocation/direct transcription as downstream optimizers rather than defeated baselines
  - short positioning against conservative obstacle-handling or convexification-style methods
  - short positioning against trajectory-initialization work, if needed to justify `C9`
- What must be excluded:
  - exhaustive taxonomy
  - generic survey prose
  - portability rhetoric unsupported by the current evidence base
- Discipline rule: if a paragraph does not clarify the paper's gap or comparison fairness, cut it.

## Section 3. Problem setup and notation

- Purpose: define the trajectory representation, decision variables, fixed-time treatment, and notation before the method logic begins.
- Key message: the paper's variables remain in control-point space, and the optimization objects are built around that representation rather than a pointwise state/control discretization.
- Supports claims: `C1`, supporting part of `C4`.
- Depends on figures/tables: `T1`.
- What must appear:
  - Bezier trajectory definition and stacked control-point decision variable
  - normalized parameter and physical-time convention
  - boundary-condition setup
  - notation discipline for control points, derivative-space objects, subdivision objects, and QP objects
- What must be excluded:
  - detailed safety proof
  - SCP loop explanation
  - results-oriented claims about usefulness or performance

## Section 4. Method

- Purpose: explain how the formulation constructs derivative/objective operators, conservative KOZ constraints, and the SCP loop.
- Key message: the method combines a structured control-point-space representation, conservative sub-arc half-space constraints, and iterative convex QP solves.
- Supports claims: `C2`, `C3`, `C4`.
- Depends on figures/tables: `T1`, `F1`, `F2`.

### Section 4.1 Derivative structure and objective assembly

- Purpose: show how derivative quantities, boundary rows, and quadratic objective terms are assembled through reusable operators.
- Key message: `D`, `E`, and Gram-matrix structure provide a coherent operator framework, not a proved speed advantage.
- Supports claims: supporting part of `C4`, with `C1` carried forward from Section 3.
- What must appear:
  - derivative operator definition at the paper-ready level
  - degree-alignment or degree-elevation role
  - quadratic surrogate assembly
  - explicit warning that the objective is a surrogate, not proof of true delta-v optimality
- What must be excluded:
  - unsupported efficiency superiority language
  - objective-mode comparison as a result claim

### Section 4.2 Conservative spherical-KOZ handling

- Purpose: state the geometric construction behind continuous conservative KOZ handling.
- Key message: De Casteljau subdivision plus supporting half-spaces conservatively excludes each spherical-KOZ sub-arc under explicit assumptions.
- Supports claims: `C2`.
- Depends on figures/tables: `F1`, supporting notation from `T1`.
- What must appear:
  - sub-arc construction
  - support point and normal construction
  - control-point half-space inequalities
  - convex-hull argument or proposition-level statement
  - explicit assumption list
- What must be excluded:
  - rhetoric implying arbitrary obstacle-geometry generality
  - rhetoric implying unconditional continuous safety outside the stated assumptions

### Section 4.3 SCP algorithm and convex subproblem structure

- Purpose: define what is fixed, what is rebuilt, and why each outer iteration solves a convex QP.
- Key message: the method is a sequence of convex QPs inside an outer SCP loop; it does not exactly convexify the original nonconvex problem in one shot.
- Supports claims: `C3`, supporting part of `C4`.
- Depends on figures/tables: `F2`, `T1`.
- What must appear:
  - per-iteration convex QP statement
  - equality constraints versus iteration-dependent inequalities
  - update loop and stopping-rule language
  - explicit distinction between convex subproblem status and original-problem nonconvexity
- What must be excluded:
  - global convergence rhetoric unsupported by analysis
  - loose wording such as "the problem becomes convex"

## Section 5. Experimental setup

- Purpose: define how each empirical or comparative claim will be tested before the results are shown.
- Key message: fairness, metric definitions, and reporting protocol are fixed here so that Section 6 can report evidence instead of inventing interpretation.
- Supports claims: protocol support for `C5`, `C6`, `C7`, `C9`.
- Depends on figures/tables: `T2`, `T3`, `F4`, `T4`, `F5`, `T6`.

### Section 5.1 Demonstration scenario and reporting metrics

- Purpose: define the baseline demonstration problem and common metrics.
- Supports claims: protocol support for `C5`, `C6`, `C7`.
- What must appear:
  - scenario definition
  - boundary-condition setup
  - safety metric
  - objective-aligned effort metric
  - runtime and iteration reporting rule

### Section 5.2 Subdivision-count and degree ablation protocol

- Purpose: define the matched sweep for `C6` and `C7`.
- Supports claims: protocol support for `C6`, `C7`.
- What must appear:
  - `n_seg` sweep
  - degree sweep
  - fairness conditions across runs
  - declared interpretation boundary for "reduced conservatism" and "degree benefit"

### Section 5.3 Downstream direct-collocation comparison protocol

- Purpose: define a fair test of the warm-start claim.
- Supports claims: protocol support for `C9`.
- Depends on figures/tables: `T6`.
- What must appear:
  - same downstream problem for naive and Bezier-based initialization
  - same solver settings, tolerances, and stopping rules
  - honest naive initialization
  - exact warm-start export mapping
  - fixed reporting metrics for success, time, iterations, final objective, and final constraint satisfaction
- What must be excluded:
  - any language implying planner superiority
  - vague fairness rules

## Section 6. Results

- Purpose: deliver evidence claim by claim, using the protocol already defined in Section 5.
- Key message: the paper demonstrates feasible trajectories, interpretable ablations, and possibly downstream warm-start value if `T6` is actually completed.
- Supports claims: `C5`, `C6`, `C7`, `C9`.

### Section 6.1 Demonstration feasibility and representative trajectories

- Purpose: establish `C5`.
- Depends on figures/tables: `F3`, `T2`.
- Safe conclusion: the framework produces smooth, feasible, safety-respecting trajectories for the demonstrated setting.

### Section 6.2 Subdivision-count trade-off

- Purpose: establish `C6`.
- Depends on figures/tables: `T3`, `F4`.
- Safe conclusion: larger subdivision counts can change approximation conservatism and computation burden under the reported metric definitions.

### Section 6.3 Degree trade-off

- Purpose: establish `C7`.
- Depends on figures/tables: `T4`, `F5`.
- Safe conclusion: degree changes expressiveness and performance characteristics under the stated protocol; no blanket "higher degree is better" claim is allowed.

### Section 6.4 Downstream warm-start comparison

- Purpose: establish `C9` only if the comparison exists.
- Depends on figures/tables: `T6`.
- Safe conclusion if `T6` exists: the Bezier-based result improves downstream initialization behavior on the tested setup.
- Required fallback if `T6` does not exist: keep warm-start language as intended use only.

## Section 7. Limitations and scope

- Purpose: close every major overclaim loophole after the evidence is presented.
- Key message: the paper demonstrates a conservative control-point-space warm-start framework under explicit assumptions; it does not prove planner superiority, true physical optimality, or broad validated portability.
- Supports claims: `C10`, explicit reinforcement of non-claims.
- What must appear:
  - spherical-KOZ scope
  - surrogate-objective interpretation boundary
  - intended-use versus demonstrated-value distinction for warm starts
  - single-domain evidence boundary
  - no global-optimality claim
- What must be excluded:
  - apologetic filler
  - future-work sprawl

## Section 8. Conclusion

- Purpose: restate only the contribution actually defended by Sections 3-7.
- Key message: the paper contributes a conservative Bezier-based trajectory-initialization framework with control-point-space formulation, conservative spherical-KOZ handling, and SCP-based optimization, demonstrated on the current setting with tightly bounded claims.
- What must be excluded:
  - any new claim
  - any stronger wording than the evidence and limitations sections allow

## Claim-to-section map

| Claim | Primary home                         | Secondary mentions allowed               | Audit note                                                                    |
| ----- | ------------------------------------ | ---------------------------------------- | ----------------------------------------------------------------------------- |
| `C1`  | Section 3                            | Sections 1 and 4                         | Keep the technical burden in setup/notation.                                  |
| `C2`  | Section 4.2                          | Sections 1 and 7                         | Keep the safety logic centralized.                                            |
| `C3`  | Section 4.3                          | Sections 1 and 7                         | Keep convexity language centralized in the SCP subsection.                    |
| `C4`  | Sections 3 and 4.1                   | Section 4.3 only in supporting form      | Keep it supporting. Do not overpromote efficiency rhetoric.                   |
| `C5`  | Section 6.1                          | Section 5.1 protocol, Section 7 boundary | Good.                                                                         |
| `C6`  | Section 6.2                          | Section 5.2 protocol                     | Good.                                                                         |
| `C7`  | Section 6.3                          | Section 5.2 protocol                     | Secondary claim only.                                                         |
| `C8`  | Retired                              | None                                     | Do not reuse this ID for a live claim.                                        |
| `C9`  | Section 5.3 protocol and 6.4 results | Section 1 motivation, Section 7 boundary | Necessary two-home claim. Protocol and evidence must both exist.              |
| `C10` | Section 7                            | Section 1 narrow framing                 | Keep it narrow: application-agnostic in construction, not broadly validated.  |

## Claims that should not have a positive home

- `X1` should appear only as an explicit non-claim.
- `X2` should appear only as an explicit non-claim.
- `X3` should appear only in future-work boundary language.

## Decision minimization

The major structure decisions are now fixed:

- Conservative journal-shaped skeleton adopted as the Stage 3 source-of-truth.
- Section 2 remains minimal and mergeable later, but is currently retained in the source structure.
- Objective-mode comparison remains dropped as a paper claim.
- Degree ablation remains in the main text as a secondary result.

No further architecture decisions are required now.

The remaining open decisions are experiment-protocol choices for Section 5.3:

- exact downstream problem definition
- exact naive initialization definition
- exact warm-start export mapping
- exact shared solver and stopping-rule protocol

## Fault-detection pass

### Overclaim risks

- Section 1 can overpromise if it implies exact convexification, general obstacle avoidance, broad cross-domain validity, or superiority to direct collocation.
- Section 2 can become filler if it expands beyond the specific comparison logic needed for the paper.
- Section 4 can overclaim if the objective is described as physically optimal or if the SCP loop is described as exact convexification.
- Section 6.4 can overclaim if warm-start value is written as planner replacement.
- Section 7 must explicitly neutralize these risks.

### Unsupported transitions

- Section 4.2 to Section 6: geometric conservatism does not automatically imply empirical usefulness.
- Section 4.3 to Section 6: a coherent optimization loop does not automatically imply better results.
- Section 6.1 to Section 6.4: demonstration success does not automatically imply downstream warm-start value.

### Unnecessary background

- The paper does not need a Bezier history lesson.
- The paper does not need a long orbital rendezvous background section.
- The paper does not need a generic tutorial on trajectory optimization methods.

## End-state summary

### Sections ready to draft now

- Section 3: Problem setup and notation
- Section 4: Method
- Section 5.1: Demonstration scenario and reporting metrics
- Section 5.2: Subdivision-count and degree ablation protocol
- Section 7: Limitations and scope

### Sections blocked by missing evidence

- Section 5.3: Downstream direct-collocation comparison protocol
- Section 6.4: Downstream warm-start comparison
- Any wording that treats `C9` as already demonstrated without `T6`

### Sections that should still be drafted late

- Section 1: Introduction
- Section 2: Related work and positioning
- Section 8: Conclusion
