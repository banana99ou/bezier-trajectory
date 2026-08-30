# DCM Downstream Experiment Design

**Status: rewritten from first principles on 2026-08-12.** This file replaces the
earlier design in place. Everything below supersedes the pre-2026-08-12 content
of this same file; there is deliberately no second design document.

Companion files, and what to believe in each:

| File | Status |
|---|---|
| `doc/dcm_experiment_findings.md` | Findings 1–4, measured 2026-04-16/17. **Demoted to hypotheses** — see §1.3. |
| `doc/dcm_downstream_pack.md` | The old `T6` result pack. **Stale**; produced by the April solver. |
| `doc/conditional_branch_decisions.md` | Holds the locked claim wording for `T6` / §6.4. **Wording still applies.** |
| `doc/design_freeze.md` | The upstream method of record. Outranks this file on anything about the Bézier solver. |
| `doc/notation.md` | Symbol lock. Outranks this file on any symbol. |

---

## 0. The test in plain words

Two pipelines produce a spacecraft trajectory for the same transfer case. The
second half of both pipelines is the *same code*. Only the first stage differs:
the baseline's own Hermite-Simpson pass, versus the Bézier SCP method of this
paper. The experiment measures what changes when only that stage is swapped:
how often the pipeline succeeds, what the final trajectory costs, and how long
it takes.

The previous version of this experiment reported only the seven cases where
*both* pipelines happened to succeed. That is selection on the outcome — the
equivalent of comparing two students using only the questions both answered
correctly. The redesign fixes this by fixing the list of test cases before any
run, and reporting every case, including every failure.

---

## 1. What is actually being compared

### 1.1 The downstream solver (`dcm_baseline/`), as measured from source

Not from memory — every row below was read out of the code on 2026-08-12.

| Property | Value | Source |
|---|---|---|
| Framework | CasADi `Opti` + IPOPT (MUMPS) | [hermite_simpson.py:52](dcm_baseline/src/orbit_transfer/collocation/hermite_simpson.py#L52) |
| Dynamics | two-body **+ J2**, in *both* passes | [eom.py:41](dcm_baseline/src/orbit_transfer/dynamics/eom.py#L41) |
| State / control | ECI position+velocity; control is thrust acceleration | [types.py:74](dcm_baseline/src/orbit_transfer/types.py#L74) |
| Objective | `∫‖u‖² dt` (Simpson in Pass 1, LGL quadrature in Pass 2) | [hermite_simpson.py:70-89](dcm_baseline/src/orbit_transfer/collocation/hermite_simpson.py#L70-L89) |
| Thrust cap | `‖u‖ ≤ u_max`, default `0.01` km/s², **at nodes only** | [hermite_simpson.py:133-136](dcm_baseline/src/orbit_transfer/collocation/hermite_simpson.py#L133-L136) |
| Altitude floor | `‖r‖ ≥ R_E + h_min`, default `h_min = 150` km, **at nodes only** | [hermite_simpson.py:138-141](dcm_baseline/src/orbit_transfer/collocation/hermite_simpson.py#L138-L141) |
| Boundary conditions | departure/arrival **orbits**, with true anomalies `nu0`, `nuf` as **free decision variables in both passes** | [hermite_simpson.py:118-131](dcm_baseline/src/orbit_transfer/collocation/hermite_simpson.py#L118-L131) |
| Earth radius | `6378.137` km (equatorial) | [constants.py:7](dcm_baseline/src/orbit_transfer/constants.py#L7) |

The pipeline stages:

| Stage | What it does | Key settings |
|---|---|---|
| Pass 1 | Hermite-Simpson collocation on a uniform mesh | `M = 30` (61 nodes); transfer time **free** in `[0.15·T0, T_max]`; IPOPT tol `1e-4` |
| Peak detection | topological persistence on the thrust magnitude profile | threshold `0.10 × max`, on a 200-point cubic-spline resample |
| Phase structure | alternating peak/coast intervals from the peaks | 15 nodes per peak interval, 8 per coast; intervals shorter than `0.05·T` merged |
| Pass 2 | multi-phase Legendre-Gauss-Lobatto collocation | horizon **fixed** to Pass 1's transfer time; IPOPT tol `1e-6`; warm-started; `nu0`/`nuf` free again |

Two behaviours in [two_pass.py](dcm_baseline/src/orbit_transfer/optimizer/two_pass.py)
dominate the experimental design and must be handled explicitly:

1. **A recovery ladder.** If Pass 1 fails, the code retries up to `MAX_NU_RETRIES = 3`
   times with random true anomalies (seeded `42`), then retries once more with
   tolerances relaxed 10×
   ([two_pass.py:86-115](dcm_baseline/src/orbit_transfer/optimizer/two_pass.py#L86-L115)).
2. **A silent fallback.** If Pass 2 fails, the pipeline returns the *Pass 1*
   result, flagged by the dynamic attribute `used_pass1_fallback`
   ([two_pass.py:159-165](dcm_baseline/src/orbit_transfer/optimizer/two_pass.py#L159-L165)).
   A caller that reads only `converged` cannot tell a two-pass success from a
   one-pass fallback.

Also note `converged` means only "IPOPT did not raise" — it includes
`Solved_To_Acceptable_Level`. The strict flag is `solve_succeeded`, derived from
the return status
([multiphase_lgl.py:205-212](dcm_baseline/src/orbit_transfer/collocation/multiphase_lgl.py#L205-L212)).
**Neither flag is accepted as evidence in this experiment** — see §5.

Finally, `TwoPassOptimizer.solve` already accepts `match_T_max`
([two_pass.py:130-133](dcm_baseline/src/orbit_transfer/optimizer/two_pass.py#L130-L133)),
which rescales the Pass-1 time axis so Pass 2 runs on the same horizon as the
proposed pipeline. §4.3 justifies using it.

### 1.2 The upstream stage under test

The Bézier SCP method of this paper, exactly as frozen in `doc/design_freeze.md`
§0 — the control-acceleration-energy objective computed as the exact integral,
the subdivided KOZ half-space constraints, canonical SCvx acceptance. No
variant, no tuning for this experiment. Whatever configuration is used is
recorded per run (§7).

### 1.3 Why the old findings are now hypotheses, not facts

Findings 1–4 were measured on 2026-04-16/17. Since then the upstream solver has
changed in ways that touch every number: the objective became the exact integral,
the KOZ subproblem rows were rebuilt as the self-consistent variant (A), the
initial trust radius became per-scenario, and the cross-solver objective
(`J_true`) was found to be broken and replaced. Additionally the bridge scripts
were left non-functional after the Δv removal (`tools/downstream_dc_compare.py`
and friends still pass a removed keyword argument).

So the following are **claims to be re-tested**, not inputs to the design:

- Finding 1 — a single Bézier arc cannot represent multi-revolution transfers,
  cutoff around `T_normed ≈ 0.5`.
- Finding 2 — a direct warm start does not improve the two-pass pipeline.
- Finding 3 — the Bézier stage can replace Pass 1 at preserved cost.
- Finding 4 — the Bézier stage rescues none of the failed cases.

Each becomes a measured outcome of the new run. Where a re-measured result
contradicts one of them, the finding file gets a dated correction; it is not
silently overwritten.

---

## 2. What the database actually is

Probed directly on 2026-08-12 from `dcm_baseline/data/trajectories.duckdb`.

| Property | Measured |
|---|---|
| Rows | 249 (`method = 'collocation'` for all) |
| **Distinct parameter vectors** | **214** — 22 vectors appear more than once (e.g. ids 2 and 6 are identical) |
| Provenance columns | `run_id` and `param_config` are **NULL on every row** |
| Creation window | 2026-03-11, 17:11 to 18:17 — a single unlabelled batch |
| Initial altitude | **`400` km on every row**, though the config declares four slices |
| `T_normed` range | `0.2796` to `4.9081` |
| Eccentricity | `e_0 ∈ [0, 0.0503]`, `e_f ∈ [0, 0.0788]` |
| Outcome labels | 220 `converged`, 29 not |
| Stored curves | `trajectory_file` points at `data/trajectories/*.npz` — **that directory does not exist in this repo** |
| Thrust cap / altitude floor | **not stored at all** — re-runs silently inherit the current defaults |

Four consequences that shape the design:

1. **The outcome columns are stale labels, not ground truth.** Finding 4 already
   showed 14 of 29 "failed" rows now converge. The only scientifically usable
   content of this database is the **case-parameter vectors**. Every outcome —
   for both arms — is re-measured fresh on one tree.

2. **The free transfer time never leaves its upper bound on this population.**
   All 220 rows labelled converged have `T_f` equal to `T_normed × T0` to within
   `1e-3` relative. Zero exceptions. This is what makes a matched-horizon
   comparison honest rather than a distortion — but it is re-asserted per case
   in the fresh run rather than assumed (§4.3).

3. **The database sits mostly outside its own declared parameter box.** The
   config declares `T_normed ∈ [0.15, 1.2]`
   ([config.py:7-13](dcm_baseline/src/orbit_transfer/config.py#L7-L13)); only
   **23 of 249 rows (20 distinct cases, 10 of them circular)** fall inside it.
   The database predates the current config. Neither one is wrong, but a claim
   must say which population it is about.

4. **The regime where the claim lives is nearly empty in this database.**
   Circular and `T_normed < 0.5`: **3 rows**. Circular and `T_normed < 1.0`:
   **7 rows**. The old `T6`'s "seven cases" was not a sample of the in-regime
   population — it *was* the entire in-regime population. That is the single
   strongest reason for a designed second stratum (§4.1).

Peak counts in the database run from 1 to 20, with 1–3 in the short circular
region and the large counts concentrated in the multi-revolution tail.

---

## 3. Design principles

Each principle exists because a specific failure mode was identified in the old
design. They are stated as principles so future edits can be checked against
them.

**P1 — The population is declared before any run, from parameters only.**
Never from outcomes. A conditional analysis ("among cases where both succeeded")
is permitted only as a clearly labelled secondary, after the full outcome
taxonomy over the declared population has been reported.

**P2 — The first stage delivers two treatments, not one.** It supplies (a) a
warm-start trajectory and (b) the phase structure. (b) changes the downstream
problem itself, because the peak count sets the interval count and node
allocation. So "matched Pass 2" means matched *solver and protocol*, not a
matched *problem*. Any comparison that does not separate these two effects
cannot attribute a difference to either. Hence the factorial design in §4.2.

**P3 — Where parity is claimed, the final optimization problem must be
identical.** Same horizon, same boundary-condition protocol, same constraints,
same tolerances. Otherwise "the costs agree" is a statement about two different
problems.

**P4 — Protocol symmetry.** Recovery machinery available to one arm must be
available to the other, or disabled for both. The primary comparison disables it
for both.

**P5 — Solver self-reports are claims, not evidence.** `converged` is set by the
absence of an exception. An independent grader that never calls the solver
decides success (§5). This follows the project rule: a check that cannot fail is
not evidence.

**P6 — Every reported number carries its producing commit.** The verification
pillars adopted this after `artifacts/verify/**` was found to be unversioned and
untraceable. The same rule applies here from the start (§7).

---

## 4. The experiment

### 4.1 Population — two strata, both declared in advance

| Stratum | Definition | Size | Purpose |
|---|---|---|---|
| **A — inherited** | all **214 distinct** parameter vectors in `trajectories.duckdb`, duplicates collapsed | 214 | Provenance: this is the downstream method's *own* dataset. Also measures the regime boundaries across the full `T_normed` range up to 4.9. |
| **B — designed** | Latin-hypercube sample over the ranges the downstream config itself declares: `T_normed ∈ [0.15, 1.2]`, `Δa ∈ [−500, 2000]` km, `Δi ∈ [0, 15]°`, `e_0, e_f ∈ [0, 0.1]`, initial altitude `400` km | ~100 | Statistical coverage where the claim actually lives. Sampling the baseline's declared box removes any "you picked the battlefield" objection. |

Stratum B fixes initial altitude at 400 km to stay comparable with A; the other
three declared altitude slices are named as future work, not silently omitted.

Both strata are frozen to a case file (`case_id`, parameters, stratum,
sampling seed) **before the first run**, and the case file is committed. Results
join to it by `case_id`.

The thrust cap and altitude floor are not in the database, so they are declared
explicitly in the case file at the current defaults (`0.01` km/s², `150` km) and
applied identically to both arms and both strata.

### 4.2 Arms — a 2×2 factorial, not a two-way race

Per P2, the first stage contributes two separable things. The full design:

| Arm | Phase structure from | Warm start from | Role |
|---|---|---|---|
| `BASE` | Pass 1 | Pass 1 | The baseline pipeline, unmodified. |
| `PROP` | Bézier | Bézier | The proposed pipeline. This is the arm the paper's claim is about. |
| `MIX-S` | Bézier | Pass 1 | Isolates the effect of the *structure* decision. |
| `MIX-W` | Pass 1 | Bézier | Isolates the effect of the *warm start*. |

`BASE` and `PROP` are run on every case in both strata. `MIX-S` and `MIX-W` are
run on every case where **both** first stages produced a usable profile — they
are diagnostic, so conditioning on that is legitimate and is labelled as such.

The mixed arms are what turn a result like the old case 114 ("1.7% cost delta,
and also the peak count differed 5 versus 4") from a confounded observation into
an attributed one.

### 4.3 Matched downstream problem

- **Horizon.** Both arms run Pass 2 on the same transfer time, `T_max`, using the
  existing `match_T_max` path. Justified by §2 item 2 — the free horizon sits at
  its bound on the entire converged population — and **re-asserted per case**:
  every `BASE` run records whether its free `T_f` left the bound, and any case
  where it does is flagged and reported separately rather than quietly rescaled.
- **Boundary conditions.** `nu0` and `nuf` remain free in Pass 2 for both arms.
  The Bézier stage necessarily fixes its own endpoints; correcting that is Pass
  2's job in both arms equally.
- **Everything else identical**: dynamics, thrust cap, altitude floor, IPOPT
  options, node-allocation rule, merge rule.
- `l1_lambda = 0` throughout. It is not part of this experiment.

### 4.4 Protocol symmetry

Primary runs, both arms:

- no true-anomaly retries (`MAX_NU_RETRIES = 0`),
- no tolerance relaxation,
- the Pass-2-failed fallback counted as a **stage-2 failure**, never as a result.

A secondary "as shipped" run enables the full recovery ladder **for both arms**
and is reported separately. Asymmetric protocols are never reported.

### 4.5 Outcome taxonomy

Every case in every arm lands in exactly one bucket, and all buckets are
reported:

| Outcome | Meaning |
|---|---|
| `STAGE1_FAIL` | the first stage produced no usable thrust profile |
| `STAGE2_FAIL` | Pass 2 did not solve, including the fallback path |
| `SUCCESS_UNVERIFIED` | Pass 2 returned a solution that the grader rejected |
| `SUCCESS_VERIFIED` | Pass 2 returned a solution the grader accepted |

Only `SUCCESS_VERIFIED` counts as a success anywhere in the analysis.

---

## 5. The grader — an independent oracle

Per P5, the solver does not get to certify itself. For each returned solution,
independently of the solver that produced it:

1. Take the returned control history and integrate the downstream method's own
   equations of motion forward from the returned initial state with a tight
   adaptive Runge-Kutta scheme (control interpolated as the transcription
   defines it).
2. Measure, on the *propagated* trajectory:
   - terminal boundary-condition error against the target orbit at the returned
     `nuf`,
   - minimum altitude over the continuous path — not only at nodes, which is
     where the transcription's own constraint is enforced,
   - maximum thrust magnitude against the cap,
   - the objective, recomputed by quadrature.
3. Accept or reject against thresholds **written into the case file before the
   first run**.

Three self-checks on the grader itself, because an oracle that cannot fail is
worth nothing:

- **Resolution check.** Re-run the grader at doubled resolution; the reported
  quantities must agree to a stated tolerance, or the grader is not converged
  and its verdicts are void. (This is the failure that broke `J_true`: a
  quadrature whose own error exceeded the difference it was being used to
  resolve.)
- **Zero-residual invariant.** The bridge constructs the Bézier thrust as
  `u = r̈/T² − g(r)` using the downstream method's own gravity. Fed back through
  those same equations of motion, the dynamics residual must be zero to machine
  precision. A non-zero value means the bridge and the grader disagree about the
  model, and everything downstream of it is suspect.
- **Deliberate corruption.** The grader must reject a solution perturbed by a
  known amount that exceeds the thresholds. If it accepts, it is not a gate.

The same grader, with the same thresholds, judges all four arms.

---

## 6. Hypotheses, each with its falsifier

Registered before the run. Each states what result would refute it.

| # | Hypothesis | Refuted by |
|---|---|---|
| **H1** | Within the representable regime, the Bézier stage recovers the same phase structure as Pass 1. | Peak-count disagreement on a material fraction of in-regime cases. |
| **H2** | On cases where both arms verify **and** the structures agree, the final objectives match within a tolerance derived from the downstream tolerance plus the grader's own quadrature error. | Any systematic gap beyond that tolerance, in either direction. |
| **H3** | Over the declared population, the proposed arm's verified-success rate is at least the baseline's within the representable regime; outside it, the boundary is measured rather than assumed. | Proposed verifying strictly less often in-regime. |
| **H4** | End-to-end median wall-clock time of the proposed arm is below the baseline's, and the saving is attributable to the first stage. | Median ratio at or above 1, **or** a saving that the stage breakdown does not attribute to stage 1. |

H1–H4 are reported whichever way they come out. A refuted hypothesis is a
result, not a problem to be fixed by re-running.

The two regime boundaries — eccentricity, and multi-revolution transfer time —
are **primary reported outputs**, measured under the current solver as curves
over the population, not inherited from the April findings.

---

## 7. Metrics, artifacts, and provenance

One row per (`case_id`, arm), written to CSV plus JSON:

| Group | Fields |
|---|---|
| Identity | `case_id`, `stratum`, `arm`, all case parameters |
| Outcome | the §4.5 bucket; downstream `return_status`; `used_pass1_fallback`; whether the free horizon left its bound |
| Structure | peak count, interval count, interval boundaries, node allocation |
| Objective | value reported by the solver, and value recomputed by the grader |
| Grader | terminal boundary-condition error, minimum altitude over the continuous path, maximum thrust, verdict, threshold values used |
| Timing | stage 1, bridge + peak detection, Pass 2, total; IPOPT iteration counts per pass |
| Provenance | producing commit SHA, dirty-tree flag, upstream solver configuration hash, grader thresholds, host, seeds |

Rules:

- **Timing is median of `K` repeats** on one machine with a pinned thread count.
  Objective values are deterministic and single-run; wall-clock is not.
- Upstream caching is **off** for every timed run.
- `Δv` — the time integral of thrust magnitude — is **never computed, stored, or
  reported**, in any arm, in any file. This is a standing project prohibition
  (`CLAUDE.md`), not a preference for this experiment.
- Artifacts are committed with the producing SHA in the file, per P6.
- The whole run is reproducible from the committed case file plus one command.

---

## 8. Claim wording

Unchanged from `doc/conditional_branch_decisions.md`, which remains the
authority. This redesign strengthens the *evidence* for the existing scoped
claim; it does not widen the claim. In particular the following remain
prohibited: any method-class superiority over direct collocation, any general
"Bézier is faster" statement, and any extension of a timing result outside the
regime in which it was measured.

Korean text for §4.3 / §5.4 follows the locked vocabulary in
`doc/session_handoff.md` (제안 기법 · 파이프라인 · 후속 · 분할구간) and the
`korean-prose` skill, loaded before writing.

---

## 9. Open questions

1. **Bézier configuration for this experiment.** Degree and subdivision count
   are inherited from the paper's demonstrated configuration; whether the
   in-regime population needs a different subdivision count is a measurement,
   not an assumption. Decide from stratum B, and record it.
2. **Fixed endpoints.** The Bézier stage fixes departure and arrival positions
   while the downstream method optimizes them. §4.3 leaves the correction to
   Pass 2 in both arms. A true-anomaly search variant would be a **separate
   labelled treatment**, never folded into the proposed arm.
3. **The other three altitude slices.** Declared in the downstream config,
   absent from the database, out of scope here, named as future work.
4. **Stratum B size.** ~100 is a starting point; the final number follows from
   the in-regime feasibility rate observed in the first batch.

---

## 10. Revision history

### 2026-08-12: Rewritten from first principles

Rebuilt the whole design after auditing the downstream source and probing the
database directly. Changes of substance:

- The population is declared from parameters in advance, in two strata; the old
  "seven both-converged cases" was selection on the outcome (P1).
- Added the 2×2 factorial so a difference can be attributed to the phase
  structure or to the warm start, instead of being one confounded lump (P2).
- Disabled the recovery ladder and the Pass-1 fallback for the primary
  comparison, symmetrically (P4).
- Added an independent grader with its own self-checks; `converged` is no longer
  accepted as evidence (P5).
- Registered four hypotheses with explicit falsifiers.
- Recorded what the database is: 214 distinct cases, one altitude, unlabelled
  provenance, missing curve files, stale outcome labels, and only 20 distinct
  cases inside the downstream method's own declared parameter box.
- Demoted Findings 1–4 to hypotheses, since the solver that produced them no
  longer exists.

### 2026-04-16: Pivot to Pass 1 replacement (historical)

The original design tested whether the Bézier optimizer, used as a warm starter
for the full two-pass pipeline, improved its results. Findings 1–2 returned a
negative result, and the goal was revised to testing whether the Bézier stage
could replace Pass 1 as a cheaper way to obtain the thrust peak structure and a
warm start for Pass 2. The 2026-08-12 rewrite keeps that framing and rebuilds
the method around it.
