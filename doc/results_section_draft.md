# Results Section Draft

## 6. Results

This section evaluates the proposed framework at the level supported by the current evidence plan. The results are organized around four questions: whether the method produces feasible trajectories on the demonstration problem, how subdivision count changes the approximation-versus-runtime trade-off, how Bézier degree changes performance under a matched protocol, and whether the resulting trajectory is useful as an initializer for a downstream direct-collocation solve. The first three questions can be drafted in near-final form using the current evidence boundary, although numerical statements and trend descriptions still require insertion of the actual aggregated outputs. The downstream warm-start comparison remains structurally important but empirically incomplete, so the subsection below is written in final form with explicit placeholders rather than inferred conclusions.

### 6.1 Demonstration feasibility and representative trajectories

We first assess the basic demonstration claim of the paper: whether the proposed control-point-space framework can produce smooth, feasible, safety-respecting trajectories for the target orbital-transfer setup. This subsection is not intended as a planner-versus-planner comparison. Its purpose is narrower. It establishes that the method produces valid demonstration outcomes on the problem class actually studied in the paper, and that those outcomes are supported by quantitative reporting rather than by trajectory plots alone.

Figure 3 presents representative optimized trajectories for selected settings, and Table 2 summarizes the corresponding quantitative outcomes. The reported metrics should include solve success, a minimum-radius or clearance-based safety metric, an objective-aligned effort metric, runtime, and outer-iteration count. These quantities are the minimum needed to support the claim that the method produces feasible and smooth trajectories in the demonstrated setting rather than merely visually plausible curves.

[INSERT FIGURE 3 HERE]

[INSERT TABLE 2 HERE]

As summarized in Table 2, `[INSERT T2 RESULTS]`. Figure 3 shows `[INSERT F3 VISUAL SUMMARY]`. The text accompanying Figure 3 should describe only what is visible and relevant to the demonstration claim, such as qualitative smoothness of the trajectory family, the relationship to the spherical keep-out zone, and any visible differences among the selected settings. It should not treat visual proximity or apparent smoothness as a substitute for the reported safety and effort metrics.

Taken together, Figure 3 and Table 2 can support the narrow claim that the proposed framework produces smooth, feasible, safety-respecting trajectories for the demonstrated orbital-transfer problem. The safe interpretation boundary is important here. These results would provide demonstration evidence for the tested setup only. They would not, by themselves, establish broad cross-domain validity, justify any claim of superiority over other planner classes, or prove more than the paper's stated continuous-safety assumptions support. They also would not establish downstream warm-start value, which requires the separate comparison in Section 6.4.

### 6.2 Subdivision-count trade-off

We next examine how the number of De Casteljau subdivisions affects the empirical behavior of the method. This ablation addresses the paper's claim that increasing subdivision count changes the conservatism of the spherical-KOZ approximation and therefore changes the cost-runtime trade-off. The purpose of the subsection is not to argue that more subdivision is always better, but to quantify what changes when the approximation is refined under otherwise matched settings.

Table 3 reports the subdivision-count ablation, and Figure 4 summarizes the corresponding runtime and outcome trends. The planned sweep is over `[INSERT EXACT SUBDIVISION GRID, e.g., n_seg = 2, 4, 8, 16, 32, 64]`, with scenario definition, objective, boundary-condition protocol, and solver settings held fixed across the sweep. In this subsection, any reference to reduced conservatism must be interpreted operationally through whether comparable certified clearance can be maintained with lower objective-aligned effort, or whether both quantities improve, under matched settings. It must not be inferred from visual path tightness alone.

[INSERT TABLE 3 HERE]

[INSERT FIGURE 4 HERE]

Table 3 reports `[INSERT T3 ABLATION RESULTS]`, and Figure 4 shows `[INSERT F4 TREND DESCRIPTION AFTER RUNS]`. The actual interpretation should remain tied to those reported quantities. If the completed sweep shows that larger subdivision counts maintain comparable certified clearance at lower cost while increasing runtime or iteration burden, that is the claim this subsection may support. If the sweep instead shows weak, mixed, or non-monotone behavior, then the paper should say so directly rather than forcing a monotonic narrative.

The safe conclusion from this subsection is therefore conditional and metric-specific. The experiment can support the claim that subdivision count materially affects approximation quality and computational burden in the tested setting. It cannot support the broader claim that higher subdivision is universally preferable, that any observed trend must persist across all scenarios, or that tighter-looking trajectories are automatically better solutions. `[INTERPRET ONLY AFTER ACTUAL RESULTS ARE AVAILABLE]`

### 6.3 Degree trade-off

We then study the effect of Bézier degree on the demonstrated optimization behavior. This subsection addresses the secondary empirical claim that degree changes flexibility, boundary-condition accommodation, and performance, but it does so under a deliberately narrow interpretation. The relevant question is not whether higher degree is better in general. The relevant question is what degree changes under a matched protocol and under which named metric any improvement, degradation, or trade-off should be understood.

Table 4 reports the degree ablation at the fixed representative setting `n_seg = 16`, and Figure 5 summarizes the corresponding multi-order trends across the full `n_seg = [2, 4, 8, 16, 32, 64]` sweep. Following the figure/table plan, the intended degree study is over `N = 5, 6, 7` under a common scenario definition, common boundary-condition protocol, and common solver settings. Because degree changes both representation flexibility and problem size, this subsection must remain careful not to blur expressiveness effects with the trivial effect of changing the number of decision variables.

[INSERT TABLE 4 HERE]

[INSERT FIGURE 5 HERE]

Table 4 reports `[INSERT T4 DEGREE ABLATION RESULTS]`, and Figure 5 shows `[INSERT F5 TREND DESCRIPTION AFTER RUNS]`. The final prose should identify which quantities are actually improved, worsened, or unchanged across the tested degree settings. Suitable reported quantities include solve success, safety metric, objective-aligned effort, runtime, and a declared control-acceleration summary used as a smoothness-adjacent secondary indicator. If boundary-condition accommodation is retained as part of the claim, the text should state exactly how that effect appears in the reported outcomes rather than relying on a general statement about flexibility.

The interpretation boundary is especially important here. This experiment can support only metric-specific conclusions under the stated protocol. It cannot support a blanket claim that higher degree is better, that one degree dominates across all criteria, or that the observed behavior generalizes beyond the demonstrated setting. If the completed data do not reveal a clean or important difference, the subsection should report that limited outcome directly. `[INTERPRET ONLY AFTER ACTUAL RESULTS ARE AVAILABLE]`

### 6.4 Downstream warm-start comparison

The final results question is the paper's narrow external-value claim: whether the trajectory produced by the proposed framework is useful as an initializer for a downstream direct-collocation solve on the same problem setup. This is an initialization comparison, not a planner-class benchmark. The subsection should therefore compare downstream behavior under two initialization strategies while avoiding any implication that the proposed method replaces or outperforms direct collocation as a method class.

We compare direct collocation from a naive initialization against direct collocation initialized from the Bézier-based trajectory produced by the proposed framework. For this comparison to support a paper claim, the downstream optimization problem must be identical across both initializations: the same dynamics model, objective, path and boundary constraints, transcription choices, solver implementation, tolerances, stopping rules, and reporting criteria must be used in both cases. The only intended difference between the two runs is the initial guess supplied to the downstream solver. If any of these conditions are not matched in the actual experiment, this subsection should be described as exploratory rather than evidentiary.

The naive initialization should be defined explicitly as `[INSERT EXACT NAIVE INITIALIZATION DEFINITION]`. That definition must be concrete enough for a reviewer to judge whether it is a reasonable baseline rather than an artificially weak starting point. The Bézier-based warm start should be defined by first solving the upstream control-point-space problem and then exporting that trajectory into the downstream direct-collocation representation via `[INSERT EXACT WARM-START EXPORT MAPPING]`. The export procedure should state how states, controls, and any additional collocation variables are initialized, and it should make clear whether the downstream transcription receives only a state trajectory guess or a fuller initialization package.

Under this matched protocol, the primary reported metrics are solve success, solve time, solver iteration count, final objective value, and final constraint satisfaction. If the comparison is run on multiple matched instances, Table 6 should report medians and spread for each metric. If only a single matched instance is available, Table 6 should report that case explicitly and the text should acknowledge the resulting limitation in scope. In either case, failed solves should be reported rather than omitted.

[INSERT TABLE 6 HERE]

Table 6 summarizes the downstream direct-collocation comparison for `[INSERT NUMBER OF MATCHED INSTANCES OR "THE TESTED INSTANCE"]` and should report `[INSERT T6 DOWNSTREAM COMPARISON DATA]`. For the current paper pass, this subsection is intentionally placeholder-only because the downstream direct-collocation workflow is still under active development. The interpretation should therefore remain conditional on the completed table. `[INTERPRET ONLY AFTER ACTUAL RESULTS ARE AVAILABLE]`

If the completed comparison shows that the Bézier-based initialization improves one or more of the reported downstream metrics under the matched protocol, then the paper may state the narrow conclusion that the proposed method is useful as a warm-start generator for the tested downstream setup. If the comparison instead shows mixed behavior, no material difference, or a trade-off between reliability, computational cost, and final solution quality, then that outcome should be reported directly and the warm-start claim should be weakened accordingly. If Table 6 is not available at submission time, the subsection should not present warm-start usefulness as a demonstrated result; the paper should revert to the weaker wording that the proposed trajectories are intended as warm starts for downstream solvers.

Two interpretation boundaries must remain explicit. First, even a favorable Table 6 would support only an initialization claim under the present downstream protocol; it would not show that the proposed method is better than direct collocation or that it should replace downstream solvers. Second, an unfavorable or inconclusive Table 6 would not invalidate the upstream formulation itself. It would show only that the current paper has not yet established downstream warm-start value strongly enough to keep that claim in demonstrated form.
