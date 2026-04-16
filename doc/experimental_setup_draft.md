# Experimental Setup Draft

## 5. Experimental Setup

This section defines the demonstration scenario, reporting metrics, and ablation protocols used in Section 6. Its purpose is to fix the evaluation logic before results are presented, so that interpretation in Section 6 can report findings rather than invent methodology.

### 5.1 Demonstration scenario and reporting metrics

The framework is demonstrated on a simplified three-dimensional orbital-transfer problem. The spacecraft must transfer between two prescribed orbital positions while avoiding a spherical keep-out zone centered at the primary body. The KOZ radius is $r_e = 6471$ km. The transfer time is fixed at $T = 1500$ s. Endpoint positions are fixed, and endpoint velocity constraints are enforced at both the initial and final boundaries. The gravity model is a two-body field with J2 perturbation, linearized affinely at representative sub-arc positions as described in Section 4.2.

The demonstration uses the IRLS-weighted L1-style control-effort surrogate objective defined in Section 4.2. No alternative objective mode is reported as a paper-level result. The prograde-preservation constraint is disabled for all reported runs; an earlier implementation contained a bug that caused premature SCP termination when this constraint was active, so it was removed to ensure clean convergence behavior.

The solver backend uses a Rust-based QP implementation. The SCP outer loop is initialized from a straight-line control polygon between the endpoints and runs for a maximum of 10000 outer iterations with a convergence tolerance of $\|P^{(k+1)} - P^{(k)}\|_F < 10^{-12}$. A proximal regularization weight of $10^{-6}$ and a control-point step-clipping radius of 2000 km are used across all reported runs. The phase lag between the chaser and target orbits is set to 120 deg, which produces a transfer geometry where the trajectory approaches the KOZ boundary, making the subdivision ablation informative.

All reported outcomes use the following metric definitions:

- **Solve success**: binary feasibility flag from the solver output, indicating whether the final iterate satisfies all active constraints within the solver's tolerance.
- **Safety margin**: $\min_\tau \|\mathbf{r}(\tau)\|_2 - r_e$, reported in km. This is the minimum distance between the trajectory and the KOZ boundary, sampled along the final trajectory.
- **Objective-aligned effort**: the L1-style delta-v proxy value $J_{\mathrm{dv}}$ at the final iterate, reported in m/s.
- **Runtime**: wall-clock elapsed time for the full SCP solve, reported in seconds.
- **Outer iterations**: the number of completed SCP outer iterations before termination.

Failed or infeasible runs are retained and reported. If a planned setting produces valid solver metadata but an infeasible final iterate, the row is kept with solve success marked false. If a planned setting does not produce valid metadata, it is reported as missing rather than silently dropped.

### 5.2 Subdivision-count and degree ablation protocol

Two ablation studies are reported in Section 6: one over the subdivision count $n_{\mathrm{seg}}$ at fixed degree, and one over the Bézier degree $N$ under matched settings.

**Subdivision-count ablation.** The primary ablation sweeps $n_{\mathrm{seg}} \in \{2, 4, 8, 16, 32, 64\}$ at fixed degree $N = 7$. The choice of seventh degree as the ablation baseline is a workflow decision: $N = 7$ is the middle order in the active study range and avoids anchoring the subdivision claim to either the lowest or highest tested degree. All other settings are held constant across the sweep: the same scenario, endpoint conditions, objective, solver configuration, and initialization rule apply to every entry.

The purpose of this sweep is to test whether subdivision count materially changes the computation-versus-approximation trade-off. For the present paper, *reduced conservatism* is interpreted operationally: whether increasing $n_{\mathrm{seg}}$ preserves comparable certified clearance while reducing objective-aligned effort, or improves both, under matched settings. If the data instead show that the safety margin and effort remain effectively flat while runtime increases, the paper reports that outcome directly rather than forcing a monotone narrative. Under the current 120-deg phase-lag scenario, the trajectory approaches the KOZ boundary, so the subdivision sweep produces a clear conservatism-reduction trend rather than a flat feasible region.

**Degree ablation.** The secondary ablation compares $N \in \{6, 7, 8\}$ under a common boundary-conditioned protocol. A fixed-setting comparison at $n_{\mathrm{seg}} = 16$ provides the primary cross-degree table. The full subdivision sweep across all three degrees provides a supplementary trend figure showing how degree interacts with subdivision count.

The degree study uses the same scenario definition, endpoint conditions, objective mode, solver configuration, and reporting metrics as the subdivision ablation. Because degree changes both representation flexibility and the number of decision variables ($N+1$ control points per spatial dimension), the comparison cannot cleanly separate expressiveness effects from problem-size effects. This confound is acknowledged rather than hidden: if the data do not reveal a clear or important difference, the subsection reports that limited outcome directly.

### 5.3 Downstream direct-collocation comparison protocol

The paper's narrowest external-value claim is that the proposed framework is useful as a warm-start generator for downstream direct-collocation solvers. Testing this claim requires a fair comparison between direct collocation from a naive initialization and direct collocation from a Bézier-based warm start, under an identical downstream problem setup.

[PLACEHOLDER: This subsection is structurally necessary but empirically blocked. The downstream direct-collocation comparison pipeline is under active development. The protocol below defines the fairness requirements that must be satisfied before Section 6.4 can report demonstrated warm-start value rather than intended use.]

For the comparison to support a paper claim, the following conditions must hold:

- The downstream optimization problem must be identical across both initializations: the same dynamics model, objective, path and boundary constraints, transcription, solver, tolerances, and stopping rules.
- The naive initialization must be defined explicitly and concretely enough for a reviewer to judge whether it is a reasonable baseline rather than an artificially weak starting point. [PLACEHOLDER: exact naive initialization definition]
- The Bézier-based warm start must be defined by a stated export mapping from the upstream control-point-space solution into the downstream direct-collocation representation. [PLACEHOLDER: exact warm-start export mapping]
- Reported metrics must include solve success, solve time, iteration count, final objective, and final constraint satisfaction, computed identically for both initialization strategies.
- Failed solves must be reported rather than omitted.

Until this protocol is implemented and stabilized, the paper uses the narrow fallback wording: the framework is intended as a warm-start generator for downstream solvers. The stronger demonstrated-value wording requires a completed and fair comparison table.
