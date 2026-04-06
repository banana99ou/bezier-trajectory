# Results Section Build

This document is the Stage 5 experiment/results build for the paper. It is a claim-driven drafting scaffold, not polished paper prose. It does not assume that missing evidence already exists.

## Stage 5 reasonability check

### Subjectiveness

- Low to moderate.
- The task is mostly constrained by the claim boundary, evidence map, figure/table plan, and technical skeleton.
- Editorial judgment still enters in subsection ordering and how aggressively to compress or defer blocked material, but the results logic should be driven by claims rather than style.

### Logicalness

- High.
- Stage 5 is the next logical step after claim lock, evidence lock, figure/table planning, and the technical skeleton.
- The requested comparisons are not arbitrary. They are already identified in `doc/paper_writing_system.md` and `doc/paper_evidence_map.md` as the minimum evidence structure for the empirical and comparative parts of the paper.
- A major logic constraint remains: the downstream warm-start claim cannot be written as an achieved result unless the direct-collocation comparison is actually built under a fair protocol.

### Scientificalness

- Conditionally high.
- The task is scientifically appropriate because it forces each experiment to answer a specific approved claim and forces fairness conditions to be stated before interpretation.
- The task becomes scientifically weak if missing evidence is hidden behind narrative. The main danger is the warm-start claim: `doc/paper_evidence_map.md` marks it as `missing`, so Stage 5 must preserve that boundary.
- Therefore the correct scientific output is a results build that distinguishes:
  - what can already be framed as supported or partially supported,
  - what can only be drafted as a planned comparison,
  - what must be weakened or withheld if the evidence is not produced.

## Approved empirical and comparative claims for Stage 5

Only the following claims should receive positive results-section space:

- `C5`: The framework produces smooth, feasible trajectories for the demonstration problem.
- `C6`: Increasing subdivision count reduces conservatism and changes the cost/runtime trade-off.
- `C7`: Higher Bézier degree changes flexibility, boundary-condition accommodation, and performance.
- `C9`: The method is useful as a warm-start generator for downstream direct collocation.

Claim status from the evidence map:

- `C5`: `present`
- `C6`: `partial`
- `C7`: `partial`
- `C9`: `missing`

Claims that must not be smuggled into the results section:

- The method is better than direct collocation.
- The method gives true delta-v-optimal solutions.
- The present paper already establishes dynamic-obstacle support.
- The structured formulation is faster than alternatives, unless an actual benchmark exists.

## Results-section architecture

Recommended subsection order, consistent with `doc/paper_technical_skeleton.md`:

1. `6.1 Demonstration feasibility and representative trajectories`
2. `6.2 Subdivision-count trade-off`
3. `6.3 Degree trade-off`
4. `6.4 Downstream warm-start comparison`

This order is defensible because:

- `6.1` establishes that the method works on the target demonstration case before ablations are interpreted.
- `6.2` carries the main ablation claim (`C6`).
- `6.3` carries the secondary ablation claim (`C7`).
- `6.4` carries the strongest external-value claim (`C9`) and should appear last because it is fairness-sensitive and currently blocked by missing evidence.

Results-section discipline rules:

- Each subsection must answer one question and support one main claim.
- Tables and figures should appear before interpretation.
- Interpretations must stay scenario-specific unless broader evidence exists.
- If a subsection lacks the required evidence, it should say so explicitly rather than inflating partial evidence into a claim.

## Subsection build details

### 6.1 Demonstration feasibility and representative trajectories

- Question answered:
  - Does the method actually produce smooth, feasible, safety-respecting trajectories on the demonstration problem?
- Supports claim(s):
  - `C5`
- Depends on:
  - `F3. Representative optimized trajectories for selected settings`
  - `T2. Demonstration outcome summary`
- Purpose:
  - Establish that the framework works on the target demonstration problem in a way that is visible and quantitatively checkable.
- Exact comparison being made:
  - Not a planner-versus-planner comparison.
  - A curated report of selected optimized cases under the paper's demonstration setup, with figures supported by matched quantitative outcomes in `T2`.
- Fairness conditions required:
  - The reported cases must come from the same scenario definition and boundary-condition setup.
  - The figure cases must be traceable to the quantitative entries in `T2`.
  - Representative trajectories must not be a cherry-picked gallery detached from the reported metrics.
  - The same metric definitions must be used across all reported cases.
- Metrics to report:
  - solve success
  - minimum radius or clearance margin
  - objective-aligned effort metric
  - runtime
  - iteration count
- Likely failure modes or reviewer attacks:
  - "This is just a nice trajectory picture."
  - "The trajectories are cherry-picked and the table is incomplete."
  - "Feasibility is being implied visually rather than reported quantitatively."
  - "Smoothness is asserted but not connected to an actual reported metric or objective-aligned quantity."
- Safe conclusions:
  - The framework can produce smooth, feasible, safety-respecting trajectories for the demonstrated problem setting.
  - The demonstration evidence is scenario-level evidence, not a broad validation claim.
- Overclaims:
  - The method works broadly across domains.
  - The results prove general continuous safety beyond the assumptions already stated in the method section.
  - The method is useful for downstream solvers solely because the trajectories look good.

### 6.2 Subdivision-count trade-off

- Question answered:
  - What does increasing subdivision count change, and what computational cost comes with it?
- Supports claim(s):
  - `C6`
- Depends on:
  - `T3. Subdivision-count ablation`
  - `F4. Runtime and outcome trends versus subdivision count`
- Purpose:
  - Test whether subdivision count changes the conservatism/computation trade-off in an interpretable way.
- Exact comparison being made:
  - A matched sweep over subdivision count, using the planned grid from the figure/table plan:
    - `n_seg = [2, 4, 8, 16, 32, 64]`
  - The comparison is within the same method, not against an external baseline.
- Fairness conditions required:
  - Hold scenario, objective, boundary-condition protocol, and solver settings fixed across the sweep.
  - Use the same initialization rule across `n_seg` values unless a deviation is explicitly justified.
  - Report failed runs rather than silently dropping them.
  - Use one operational definition of "reduced conservatism" consistently across the subsection.
- Metrics to report:
  - subdivision count `n_seg`
  - solve success
  - minimum radius or clearance margin
  - objective-aligned effort metric
  - runtime
  - outer iteration count
- Required interpretation discipline:
  - "Reduced conservatism" must not mean only that a path looks tighter.
  - The safe operational reading is: how the safety metric and the objective-aligned effort change under matched settings as the approximation is refined.
  - If the data do not support a clean monotone trend, the text must report that instead of forcing a story.
- Likely failure modes or reviewer attacks:
  - "Conservatism is undefined."
  - "You are reading visual path tightness as evidence."
  - "Runtime goes up, but the claimed benefit is vague."
  - "The trend is scenario-specific or noisy, yet the prose sounds universal."
- Safe conclusions:
  - Increasing subdivision count changes the approximation quality and computation burden.
  - If supported by the actual runs, one may say that larger subdivision counts can reduce safety-approximation conservatism at increased runtime.
  - The conclusion should remain tied to the tested scenario and metric definitions.
- Overclaims:
  - Higher subdivision is always better.
  - The trade-off is monotone unless the data clearly show it.
  - The ablation proves a universal property of all Bézier obstacle-avoidance problems.

### 6.3 Degree trade-off

- Question answered:
  - What does Bézier degree change, and under what criterion?
- Supports claim(s):
  - `C7`
- Depends on:
  - `T4. Degree ablation under matched settings`
  - `F5. Multi-order performance trends`
- Purpose:
  - Test the secondary claim that degree changes expressiveness, boundary-condition accommodation, and performance, while preventing vague "higher degree is better" rhetoric.
- Exact comparison being made:
  - A matched sweep over degree using the planned range:
    - `N = 5, 6, 7`
  - The comparison must explicitly state the regime being compared:
    - fixed scenario and boundary-condition protocol,
    - fixed subdivision protocol,
    - fixed solver settings,
    - and a declared interpretation of how differing control-point counts are handled.
- Fairness conditions required:
  - Use a common boundary-condition protocol across degrees.
  - Hold scenario, objective, and solver settings fixed.
  - State clearly whether the comparison is at fixed subdivision count, across the full subdivision sweep, or under another matched design.
  - Do not blur expressiveness effects with trivial variable-count effects; if they cannot be separated, say so.
- Metrics to report:
  - degree
  - number of control points
  - solve success
  - safety metric
  - objective-aligned effort metric
  - runtime
  - selected smoothness indicator
- Likely failure modes or reviewer attacks:
  - "The comparison is confounded by changing decision-space size."
  - "You say degree adds flexibility, but you do not define what improved."
  - "The trend is weak, yet the prose implies dominance."
  - "Boundary-condition accommodation is asserted but not actually shown in the data."
- Safe conclusions:
  - Degree changes the optimization's expressiveness and performance characteristics under the stated protocol.
  - If supported, specific improvements must be tied to named metrics such as solve success, safety metric, objective-aligned effort, runtime, or the selected smoothness indicator.
  - Any benefit must be phrased as conditional on the tested setup.
- Overclaims:
  - Higher degree is better in general.
  - Degree alone explains all quality differences without acknowledging other coupled settings.
  - The degree result validates broad design recommendations outside the demonstrated setup.

### 6.4 Downstream warm-start comparison

- Question answered:
  - Does a Bézier-based initialization help downstream direct collocation relative to a naive initialization under a matched downstream setup?
- Supports claim(s):
  - `C9`
- Depends on:
  - `T6. Downstream direct-collocation initialization comparison`
- Purpose:
  - Test the paper's main comparative value claim without turning it into a planner-superiority claim.
- Exact comparison being made:
  - Direct collocation from naive initialization versus direct collocation from Bézier-based warm start.
  - The downstream problem instance must be the same for both initializations.
- Fairness conditions required:
  - identical downstream problem definition
  - identical solver settings
  - identical tolerances and stopping rules
  - identical constraint reporting
  - an honest naive initialization definition
  - a clearly specified export mapping from the Bézier result into the downstream transcription
  - matched evaluation criteria for success, time, iterations, final objective, and final constraint satisfaction
- Metrics to report:
  - solve success
  - solve time
  - iteration count
  - final objective
  - final constraint satisfaction
  - if multiple cases exist, medians and spread rather than one anecdotal run
- Likely failure modes or reviewer attacks:
  - "The naive baseline is artificially weak."
  - "The downstream problem is not actually matched to the upstream one."
  - "This is only one anecdotal case."
  - "Warm-start utility is being overstated as method superiority."
  - "The solver was going to converge anyway; the result only changes initial transients."
- Safe conclusions:
  - Only if `T6` is actually built under the stated fairness conditions:
    - the Bézier result can be argued to improve downstream initialization behavior on the tested setup
    - the method has demonstrated warm-start value for that tested setup
  - Without `T6`, the only safe wording is:
    - the method is intended as a warm-start generator for downstream solvers
- Overclaims:
  - The method is better than direct collocation.
  - Warm-start benefit on one setup proves broad planner superiority.
  - Downstream comparison success validates cross-domain usefulness.

## Planned results figures and tables

This section covers only the results figures/tables. Method-support figures such as `T1`, `F1`, and `F2` belong to the method architecture, not the Stage 5 results build.

### F3. Representative optimized trajectories for selected settings

- Working caption:
  - `Representative optimized trajectories for selected settings in the demonstration problem. Each panel shows the trajectory geometry relative to the spherical keep-out zone, while the corresponding quantitative outcomes are reported in T2.`
- What the reader should conclude:
  - The method produces plausible, smooth, safety-respecting trajectories on the demonstration problem.
  - The figure is illustrative support for `C5`, not standalone evidence.
- Missing evidence or implementation work still blocking it:
  - paper-ready case selection criteria
  - final curated trajectory exports
  - explicit linkage from displayed cases to `T2`
  - confirmation that the shown cases are representative rather than decorative

### T2. Demonstration outcome summary

- Working caption:
  - `Demonstration outcome summary for selected optimization settings. Reported quantities include solve success, safety margin, objective-aligned effort, runtime, and iteration count to support the claim that the framework produces feasible trajectories on the target problem.`
- What the reader should conclude:
  - The demonstration claim rests on quantitative evidence, not pictures alone.
  - The reported trajectories satisfy the paper's reporting logic for feasibility, effort, and computational behavior.
- Missing evidence or implementation work still blocking it:
  - final paper-ready aggregation of selected cases
  - locked metric definitions for safety margin and objective-aligned effort
  - a policy for reporting failed or incomplete runs

### T3. Subdivision-count ablation

- Working caption:
  - `Subdivision-count ablation under matched settings. The table reports how solve success, safety metric, objective-aligned effort, runtime, and outer-iteration count change as the number of subdivided sub-arcs increases.`
- What the reader should conclude:
  - Subdivision count materially changes the approximation/computation trade-off.
  - Any claim about reduced conservatism must be read through the reported metrics, not visual intuition.
- Missing evidence or implementation work still blocking it:
  - complete matched sweep across the planned `n_seg` values
  - explicit operational definition of reduced conservatism
  - final aggregation and formatting of runtime and outcome metrics
  - inclusion of failed runs if any occur

### F4. Runtime and outcome trends versus subdivision count

- Working caption:
  - `Runtime and outcome trends versus subdivision count. Separate panels summarize how computational cost and key trajectory metrics vary with refinement of the subdivision-based safety approximation.`
- What the reader should conclude:
  - The subdivision ablation has an interpretable trade-off structure rather than a single scalar winner.
  - Runtime and trajectory-quality metrics must be read together.
- Missing evidence or implementation work still blocking it:
  - cleaned trend data from the full subdivision sweep
  - panel design that avoids overloaded dual-axis plotting
  - confirmation of whether trends are stable enough to summarize graphically

### T4. Degree ablation under matched settings

- Working caption:
  - `Degree ablation under matched settings. The table reports solve success, safety, effort, runtime, and a selected smoothness indicator for quadratic, cubic, and quartic Bézier parameterizations under a common protocol.`
- What the reader should conclude:
  - Degree changes performance characteristics, but any advantage must be tied to a named metric.
  - The paper is not claiming blanket superiority of higher order.
- Missing evidence or implementation work still blocking it:
  - a declared fairness regime for comparing different degrees
  - final sweep outputs for `N = 2, 3, 4`
  - a justified selected smoothness indicator
  - final interpretation boundary for what degree is allowed to claim

### F5. Multi-order performance trends

- Working caption:
  - `Multi-order performance trends across Bézier degree. Degree-coded curves summarize how one or two selected metrics vary across the matched experimental sweep without implying universal dominance of any single degree.`
- What the reader should conclude:
  - Degree interacts with the experimental settings in interpretable but limited ways.
  - The role of this figure is comparative structure, not "higher degree wins."
- Missing evidence or implementation work still blocking it:
  - final trend data from the degree study
  - choice of one or two metrics worth plotting
  - enough stability in the data to justify a trend figure rather than only a table

### T6. Downstream direct-collocation initialization comparison

- Working caption:
  - `Downstream direct-collocation initialization comparison under matched solver settings. The table compares naive initialization and Bézier-based warm-start initialization using solve success, time, iteration count, final objective, and final constraint satisfaction.`
- What the reader should conclude:
  - If the table exists and is fair, the paper has actual evidence about downstream initialization value.
  - The table supports a narrow warm-start claim, not a planner-superiority claim.
- Missing evidence or implementation work still blocking it:
  - an implemented direct-collocation comparison pipeline
  - a defined naive initialization
  - a defined warm-start export mapping
  - locked fairness rules and solver settings
  - multiple matched cases if the paper wants anything stronger than an anecdotal observation

## Writing guidance for the actual results prose

- Open each subsection with the question being answered, not with a conclusion.
- Report the table/figure content first.
- Separate three layers explicitly:
  - observation: what the table or figure shows
  - interpretation: what that means for the tested claim
  - boundary: what the result does not justify
- Use scenario-specific wording unless the experiment truly spans more than one setup.
- Do not let the ablation sections imply warm-start value.
- Do not let the warm-start section imply superiority over direct collocation.

## Evidence gaps and blocked claims

### Gap 1: Warm-start usefulness remains blocked without `T6`

- A publishable warm-start claim requires the downstream direct-collocation comparison identified in the evidence map.
- What is still needed:
  - a matched downstream problem definition
  - naive versus Bézier-warm-start initialization protocol
  - identical solver settings, stopping rules, and constraint reporting
  - reported success, time, iterations, final objective, and final constraint satisfaction
- Until this exists, `C9` is not an achieved results claim.
- Required fallback wording:
  - the method is intended as a warm-start generator for downstream solvers

### Gap 2: Subdivision ablation remains only partially supported without a disciplined interpretation

- `C6` is not publishable as a strong claim if the paper only shows plots without a clear operational definition of conservatism reduction.
- What is still needed:
  - a completed matched sweep over subdivision count
  - a stable reporting table
  - a stated definition of how the reported metrics reflect conservatism/computation trade-off
  - explicit reporting of failures or non-monotone behavior

### Gap 3: Degree ablation remains weak unless the comparison regime is declared

- `C7` is easy to overstate because different degrees alter representation freedom and problem size at the same time.
- What is still needed:
  - a declared fairness regime
  - matched reporting across degree values
  - a named criterion for any claimed improvement
- Without this, degree should remain a secondary result with restrained wording.

### Gap 4: Demonstration evidence needs paper-ready aggregation

- `C5` is marked `present`, but it is not yet publication-ready if the paper only has code outputs and informal plots.
- What is still needed:
  - curated representative trajectories
  - a paper-ready summary table
  - locked metric definitions
  - a consistent policy for reporting success and failure

### Claims that remain unsupported and should not be rehabilitated in Stage 5

- Better than direct collocation
- True delta-v optimality
- Broad cross-domain validation
- Dynamic-obstacle support in the present paper
- Comparative efficiency superiority without measurement

If evidence for these does not exist, the correct action is not to phrase them more cleverly. The correct action is to keep them out of the results section.
