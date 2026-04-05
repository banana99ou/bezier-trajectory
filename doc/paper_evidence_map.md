# Paper Evidence Map

This document maps each major paper claim to the evidence required to support it. The point is not to sound ambitious. The point is to stop unsupported claims from slipping into the title, abstract, introduction, or conclusion.

Status labels:

- `present`: already supported in the repository or by straightforward code-level inspection
- `partial`: partly supported, but still vulnerable without added derivation, experiments, or cleaner exposition
- `missing`: not currently supported and should either be added or removed from the paper

## Evidence table

| Claim | Evidence required | Status | Why | Must be added, revised, or removed |
|---|---|---|---|---|
| The method operates entirely in control-point space. | Clear formulation showing optimization variables are Bézier control points; derivative, subdivision, and constraints expressed as linear maps on control points. | `present` | The codebase already builds the method around control points and matrix operators in `orbital_docking/bezier.py`, `orbital_docking/de_casteljau.py`, `orbital_docking/constraints.py`, and `orbital_docking/optimization.py`. | Add a compact derivation and a figure or notation table so the reader sees this immediately. |
| Continuous spherical-KOZ avoidance is handled conservatively by De Casteljau subdivision plus supporting half-spaces. | Formal explanation of sub-arc control points, supporting half-space construction, and the convex-hull argument; ideally a short proposition or lemma. | `partial` | The implementation matches this logic, but code is not a proof. The paper still needs a precise statement of what is guaranteed and under what assumptions. | Add a theorem/lemma-level statement and explicitly state the assumptions. Do not overstate this as a general obstacle-avoidance theorem. |
| The overall algorithm is an SCP method that solves a sequence of convex QPs. | Optimization problem statement showing what is linearized and what remains convex at each iteration. | `present` | `orbital_docking/optimization.py` clearly rebuilds linearized KOZ constraints and a quadratic objective, then solves with convex structure per outer iteration. | Explain this carefully. Avoid saying the original problem becomes convex in one step. |
| The structured `D`/`E`/Gram-matrix construction gives an efficient and organized derivative/objective formulation. | Derivation of velocity and acceleration mappings; explanation of degree elevation; description of Gram-matrix reuse. | `partial` | The formulation is present in code, but "efficient" is only partly supported. The code shows reuse and structure, but not yet a rigorous runtime comparison against a naive implementation. | Safe claim: "structured and reusable." Risky claim: "faster." Either add a benchmark or weaken the efficiency wording. |
| The framework produces smooth, feasible trajectories for the demonstration problem. | Successful runs, trajectory figures, feasibility checks, and trajectory metrics. | `present` | The repo already generates optimized trajectories, measures minimum radius, and reports objective and timing information. | Present this as demonstration evidence, not as proof of broad superiority. |
| Increasing subdivision count reduces conservatism and changes the cost/runtime trade-off. | Ablation over segment count with metrics such as minimum clearance, surrogate objective, feasibility, and computation time. | `partial` | The code and existing figures support the existence of a segment-count sweep, but the paper needs a clean and quantitative interpretation rather than just plots. | Add a compact ablation table and explain what "reduced conservatism" means operationally. |
| Higher Bézier degree changes flexibility, boundary-condition accommodation, and performance. | Degree sweep with matched settings and clear interpretation. | `partial` | The repository studies multiple degrees, but the paper still needs disciplined wording. "More freedom" is plausible; "better" requires evidence and a stated criterion. | Add a degree-ablation subsection. Define exactly what improves: feasibility, surrogate cost, smoothness, or downstream usefulness. |
| The L1 delta-v proxy and L2 control-acceleration energy surrogate are alternative objective surrogates with different optimization behavior. | Mathematical definitions, clear interpretation, and empirical comparison or at least qualitative discussion. | `partial` | Both modes exist in the code, but the repo does not yet show a careful analysis of when one behaves differently from the other. | Remove this as an explicit paper claim. At most, mention objective variants briefly as implementation context without turning them into a paper result. |
| The method is useful as a warm-start generator for downstream direct collocation. | Direct comparison between direct collocation from a naive initial guess and direct collocation initialized by the Bézier-based result; report convergence behavior, solve success, cost, and time. | `missing` | This is the paper's most important external-value claim, and it is not supported yet. | Add this experiment. If not added, weaken all warm-start utility claims from demonstrated fact to intended use. |
| The formulation is application-agnostic in principle. | A geometric derivation that is not orbital-specific; preferably a second demonstration or at minimum very disciplined wording. | `partial` | The formulation itself is geometric, but the evidence base is currently one orbital demonstration. That supports portability in principle, not in validated practice. | Frame as application-agnostic in construction, not broadly validated across domains. Add a second demo only if it is genuinely finishable. |
| The method is better than direct collocation. | Broad benchmark suite, fair tuning, clear success criteria, and careful interpretation. | `missing` | This claim is not supported and should not be pursued under the current strategy. | Remove the claim. Compare warm-started vs naive direct collocation instead. |
| The method gives a true delta-v-optimal solution. | Proper dynamics, control parameterization, time treatment, and a true fuel-optimal objective. | `missing` | The current objectives are surrogates, not exact mission delta-v optimization. | Remove the claim completely. |
| The framework already supports dynamic obstacles through space-time extension. | Implemented 3+1 formulation, demonstrations, and evidence. | `missing` | This belongs to future work, not the present paper. | Keep only as future work. Do not blur it into the present contribution. |

## Claim-by-claim implications

## Claims that are safe now

- The method operates in control-point space.
- It uses De Casteljau subdivision and supporting half-spaces to conservatively handle spherical-KOZ avoidance.
- It is implemented as an SCP loop that solves a sequence of convex quadratic subproblems.
- It is demonstrated on a simplified orbital-transfer example.

These claims still need good exposition, but they are not fundamentally unsupported.

## Claims that are usable only with disciplined wording

- The method provides a continuous-safety guarantee.
- The structured formulation is computationally efficient.
- The framework is application-agnostic.
- Segment-count and degree sweeps reveal interpretable trade-offs.

These are not false, but they are easy to overstate.

## Claims that should be removed unless new evidence is added

- The method is superior to direct collocation.
- The objective-mode difference warrants dedicated paper-claim status.
- The method is broadly validated across domains.
- The method optimizes true delta-v.
- The present paper already covers dynamic-obstacle planning.

## Minimum additions needed for a defensible paper

1. A precise statement of the conservative safety argument for the spherical-KOZ case.
2. A clean ablation on segment count and degree with quantitative interpretation.
3. A downstream comparison: naive direct collocation versus direct collocation initialized by the Bézier-based warm start.
4. A clearly written limitations section that states what the current paper does not prove.

## If time is limited

If schedule pressure forces cuts, the order of sacrifice should be:

1. broad portability rhetoric
2. any claim of efficiency superiority without measurement
3. any claim about objective physicality
4. any claim that sounds like replacement of downstream planners

Do **not** sacrifice the claim discipline to make the paper sound larger. That only makes the paper easier to reject.
