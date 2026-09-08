# SOLVER.md — the code, and what it is known to do

Everything about the implementation: how it is put together, what has already been established
about it, what is deliberately not being done, and what is currently broken. **The ideas it serves
live in [`idea/`](idea/)** — geometry derivations and novelty belong there, not here.

Two rules govern this file:

- **A number lives where a test re-measures it.** A number written here is a report of a
  measurement pass, dated and reproducible from the registered configuration; it is never the
  authority. The authority is the assertion in [`tests/`](tests/).
- **A fact earns its line by having cost something.** Three separate agents each spent ~50 tool
  calls rediscovering the section below, which is why it exists.

---

## Why this code exists

The goal is not "make the solver good." It is: **make the idea's claims true of the code.** Two
failure modes it exists to exclude:

1. A paper claims a convex-hull guarantee the code does not implement. (Closed by the G1 and G2
   fixes; it is what the certificate exists to keep closed.)
2. A figure comes from a run where elastic slack hid a constraint violation. (Closed 2026-08-20 by
   the `figure_grade` gate — `tools/make_paper_figure.py` refuses to draw a run that fails it.)

Performance is housekeeping: the numbers need to be reasonable, not impressive.

**The bad pattern, which was defect G1 and is fixed:** evaluate the obstacle at `pos0 + vel*t`,
build the normal from spatial coordinates only, emit a zero coefficient on the time coordinate.
That is a 2D obstacle re-evaluated per time slice, not a static obstacle in space-time. The zone is
now generated in the full lifted space by `Generator` in
`rust_optimizer/core/src/spacetime_generator.rs`, and **a zero time coefficient is a reportable
defect on every row, the shadow's included.** `SPACETIME_AXIS_SCALE` stays pinned at 1.0 as a
declared modelling choice — the consequence, and the measured conservatism it buys, are documented
at `rust_optimizer/core/src/spacetime_constraints.rs:97`.

---

## Architecture

Rust is the sole optimizer backend; the Python optimizer has been removed. Clarabel (interior-point
conic) solves the QP subproblems, and elastic relaxation — slack variables on the KOZ rows —
handles infeasible ones.

Five rules keep it honest. They were written for the post-paper sandbox and they govern how code is
written here now:

- **One canonical execution model.** Same request + same configuration = same execution = same
  result = same traceable explanation. `scp_step` in the Rust core is both the batch iteration and
  the debug step; a debug session never simulates an alternative stepper.
- **One frontend, two tempos.** Diagnostic mode is a drawer over the same scene, sharing scenario
  state, camera, and server — never a separate page or a reload.
- **Rust is the sole engine.** It owns the SCP loop, production KOZ half-space generation,
  candidate acceptance, final iterate selection, and trace emission. Python is not a reference
  solver and not an oracle.
- **Trace is observability over the real run,** not a parallel optimizer. Reconstruction from
  partial logs is an acceptable bridge; fake stage synthesis is not.
- **Geometry authenticity.** Anything drawn either came from the backend that ran, or is explicitly
  marked derived. No time-sliced 2D plane labeled as a 3D supporting surface.

Trace frames must distinguish *current iterate / raw candidate / accepted iterate / best feasible
iterate / final returned iterate*. These are different concepts; never conflate them.
---

## Entrypoints

Full quickstart — install, build, run, test — is [`README.md`](README.md) §Quickstart. The three
things here that are guardrails rather than instructions:

```bash
python3 -m spacetime_bezier          # THE frontend — config panel, solve, one page on 8767
python3 -m spacetime_bezier.viewer   # superseded sanity viewer, still on disk
./tools/watch_paper.sh               # manuscript .docx -> .md sidecar + .pdf, on every save
```

- **Two ways in, never two servers.** Both bind 8767, and both exit 1 when it is held, naming the
  pid and warning when that process loaded the Rust extension before your last build. The shared
  port is the mutual exclusion — do not work around it by changing the port.
- `python3 -m spacetime_bezier.io` and `.sandbox` are **not** entrypoints, as of 2026-08-18. They
  bound a second default port (8765 against 8767), which is how two sandboxes ended up live at
  once — the older four days stale on a different interpreter, answering with pre-G1/G2 geometry
  and no signal in the UI.
- **Reuse, do not rewrite.** `orbital_docking/` supplies the dimension-agnostic building blocks the
  spacetime code imports: `bezier.py` (D/E/G matrices, `BezierCurve`) and `de_casteljau.py`
  (`segment_matrices_equal_params`).
- **Writing a paper is its own workflow** — [`paper/README.md`](paper/README.md) before touching a
  manuscript.

---

## Established facts — do not re-derive

Three separate agents have each spent ~50 tool calls rediscovering these. **Every number below is
carried by a test that fails when it goes stale** — `tests/unit/test_spacetime_koz_geometry.py`
and `tests/integration/test_scvx_invariants.py` are where the evidence lives.

- **The merge-base `e849ff7` contains no Rust.** Both lineages wrote `rust_optimizer/` from scratch, so `git merge main` is an add/add conflict on every file. Bringing solver work over from `main` is a **file-level port**, never a merge or a rebase.
- `bezier.rs` and `de_casteljau.rs` were **byte-identical** between `main` and this branch until
  `8dc5a4b` (2026-09-04) rewrote their kernels allocation-free on this branch; results are
  bit-identical (28-config golden battery, IEEE-754 hex), the text is not. A port from `main` now
  has to carry the kernel rewrite forward, not overwrite it.
- **The SCP loop has a real ratio test.** Since `9b9c3d3` there is a merit function, a predicted reduction, and a rho governing accept/reject and the trust update. Since 2026-08-31 Clarabel's **duals are extracted** (they drive the elastic-weight rule and the exactness margin), but `converged` still tests no KKT residual: it asserts feasibility plus no-further-progress — *not* stationarity of the original problem.
- **G1 — FIXED 2026-08-17.** The keep-out normal's time component now falls out of the tube's slanted centreline instead of being written as zero.
- **G2 — FIXED 2026-08-17, and the key widened 2026-08-26 (twice).** One plane per **(segment, obstacle, local approach)** — the band cut at every interior local maximum of the centreline's distance to the segment centroid — aimed at the segment centroid, plus a rotation term in the subproblem. It was one plane per (segment, obstacle) until the wall moved onto the clipped volume, then one per connected component until the same day's second correction: a bend that WRAPS the centroid is one component, and its single wall is non-separating by theorem (panel B2 measured it at margin −1.154 on a clear segment; the cut gives two walls at +0.529/−0.162). The wire name `component_idx` survives and now indexes approaches. Grouping rows without the index folds two genuinely different walls together. The dense subdivision matrix was never part of this defect — sparsifying it would *break* the guarantee.
- **The keep-out wall is the SUPPORT of the clipped KOZ volume, not a hull projection.** Since `0a8bc9e`+ the plane is built against `K_m ∩ B(c, r_clip)` itself: normal from the component's own nearest centreline point, offset a rigorous De Casteljau ceiling on that component's support. **The retired route — select a parameter interval, hull the subdivided centreline, project the centroid, push out by `r_m` — is an OUTER approximation**, measured 1.13–2.23 past the clip radius on all six `curve` pairs, and it can never yield more than one wall. Do not reintroduce it. `r_clip = clamp(d, r_m, r_clip_max)`; the floor binds exactly when the centroid is inside the keep-out zone.
- **`wall` and `diverse` are FEASIBLE.** Corrected 2026-08-19. Both clear and certify once the elastic penalty weight exceeds the scenario's exact-penalty threshold; the weight was pinned at 100 inside the Rust binding and unreachable from Python. Below threshold, a penetrating curve is genuinely the cheaper answer — the solver was right about the wrong problem.
- **The elastic weight escalates IN-LOOP; the Python ladder of cold restarts is gone** (2026-08-31,
  branch `solver/dual-elastic-weight`; trigger repaired 2026-09-06). The escalate-and-continue RULE
  is SNOPT's elastic mode (Gill, Murray & Saunders, SIAM Review 47(1), 2005 reprint, §6.5 steps
  6–7 and eq. 2.10 — verified against the reprint 2026-09-08; the 2002 SIAM J. Optim. original's
  numbering is unchecked);
  the SCHEDULE — unconditionally elastic, x10 per raise from a fixed start, cap 1e5 (the old top
  rung) — is this code's adaptation, not SNOPT's (theirs scales the base by the objective-gradient
  norm and accelerates 10x, 100x, 1000x). The raise TRIGGER is a stall heuristic on the exact
  violation with no published algorithm claimed for its form (an earlier claim of Nocedal & Wright
  Alg. 17.4 was wrong — that is LANCELOT's forcing-tolerance test with multiplier updates — and is
  retracted in the code): two tiers, either of which raises — no 1% improvement on the minimum of
  the last 5 iterations for 10 consecutive infeasible iterations, or no new 1%-better level best for
  20; feasible iterations freeze both counters rather than resetting them; the trust radius
  collapsing while violating raises too. A stall the weight cannot escape (cap reached, or held
  with `escalate_elastic_weight=False`) does NOT stop the loop — a stall stop was shipped and
  removed on 2026-09-07; the loop runs on to its own stops (trust collapse or the iteration cap),
  and the evidence for the removal is mixed and recorded in the trigger comment in
  `spacetime_optimizer.rs`. On a raise the SAME iterate continues with the trust radius reset. Clarabel's dual vector is read (`solve_qp_with_socs_duals`):
  `info["max_koz_dual"]` vs `info["final_elastic_weight"]` is the measured exactness margin (Han &
  Mangasarian 1979), paired honestly (NaN when a raise fired after the last solved subproblem), and
  complementarity — slack active iff the largest keep-out dual sits AT the weight — is tested from
  both sides in `tests/integration/test_scvx_invariants.py::TestElasticComplementarity` (4 active-
  slack runs, 4 zero-slack runs, and per-iteration through the stepper). Measured old→new on the
  committed extension: `wall` (8,16) 20.5 s → 2.7 s, `diverse` (8,4) 0.94 s → 0.43 s, both at
  IDENTICAL clearance; configs that never needed the ladder are unchanged.
  Three trigger forms were measured wrong on the way: a merit-flatness streak fired mid-grind and
  wrecked runs that were winning; a never-expiring running minimum (shipped in `f5a0ac1`) let one
  lucky iterate count ten "stalls" on a descending run, and its feasible-iterate reset blinded it to
  a limit cycle (`curve` N8_seg16, 158 iterations burned); a 5-iteration windowed minimum alone
  aliased on a 6-iteration cycle (`curve` N8_seg4, 200 iterations burned at weight 100). The
  windowed minimum plus the level-best backstop is what survived measurement. Removing the ladder
  had regressed `curve` N8_seg8 and N10_seg8 out of figure grade (over-escalation to the cap,
  certificates 5.1e-6 and 1.5e-6); the repaired trigger returns both (8.8e-9 at 1e5 after 3 raises,
  1.2e-9 at 1e4 after 2). `curve` N8_seg4 and N8_seg16 were red under the ladder and are red now;
  the escalated N8_seg4 run lands in a slack-0.58 basin at the cap, and NO start weight certifies
  it (measured 2026-09-08: held at 1e4, 1.3e-05; escalating from 1e3 / 1e4 / 1e5, 1.95e-05 /
  2.09e-06 / 1.97e-05 — a previous version of this line said it certifies from a cold start at
  1e4, which was wrong) — the basin, not the weight, see `solver.multistart`.
  Two stale beliefs died on the way: `wall` N8_seg2 does NOT clear at weight 800
  at this HEAD (the +0.1035 comment predates the current geometry; every rung fails, old and new,
  and the new best-effort −0.098 beats the old −0.129), and `diverse` N8_seg4 from weight 100 now
  converges figure-grade instead of relying on the best-iterate fallback (its fallback tests pin
  `max_iter=15` to stop before the rescue).
- **The certificate is evaluated at the RETURNED iterate.** Fixed 2026-08-19; it used to report the loop's final *reference* point, which is a different trajectory whenever the best-iterate fallback fires.
- **`obstacle_pos0` / `obstacle_vel` are not parameters of anything — and the drift they caused is CLOSED.** `0918df5` moved the Rust API to lifted control points and did not update the tests. 75 of 338 were red at that commit; 52 were still red as late as `e9d953f` on 2026-08-31, and `test_los_margin.py` and `test_clearance_sampling.py` were among them — the independent line-of-sight check behind the occlusion claim, and the computation behind every clearance number in §Measurements, neither of which had executed a single assertion since `0918df5` because they failed at the CALL in 0.09 s. Repaired by `d3a1426` + `8c948a7`, merged from `paper/journal-1` on 2026-09-01. **Suite is now 1 failed / 364 passed / 1 skipped (2026-09-08; 363 on 2026-09-06 before the start-weight provenance guard, 352 before the escalation-trigger repair added eleven), and the one red is the deliberate one.** Convert `pos0`/`vel` obstacles with `spacetime_bezier.geometry.obstacle_array_bundle`; a legacy obstacle with no window needs one, because the active window is now intrinsic to the control points.
- **The dead Python triple is gone** — `327a28e` deleted `spacetime_bezier/constraints.py`, `debug_stepper.py`, `tests/unit/test_spacetime_constraints.py` and `BENCHMARKS.md` on 2026-08-20. The lesson they carried is worth keeping: those KOZ tests could not fail on G1, because every one used zero velocity and one asserted the time column was zero, enshrining the bug. **The Rust builder's coverage is `tests/unit/test_spacetime_koz_geometry.py`** (and `tests/unit/test_center_surface_geometry.py` since `46a15f7`); nothing else covers it.
- **The keep-out zone and the shadow are ONE set at two stretch factors.** There is no occlusion
  builder and there are not two keep-out zones: the centreline generalises to a **center surface**,
  the shadow of the centreline with the ground station as a point light. `u = 1` is the body — the
  obstacle's radius never changes and the umbra widens, because a point source at finite distance
  casts constant *angular* thickness. For fixed `u` the stretch is affine, so the surface is again a
  Bezier curve and subdivision, hodographs and the De Casteljau support bounds carry over unchanged.
  **No station means `u = 1` identically and every formula reduces to the plain lifted tube** — that
  reduction is the regression the rewrite stands on, witnessed by
  `tests/unit/test_spacetime_koz_geometry.py`. Derivation and falsification test:
  `rust_optimizer/core/src/spacetime_generator.rs` module docstring and
  `tests/unit/test_center_surface_geometry.py`.
  - **API.** `spacetime_occlusion_rows_exact` is gone. `spacetime_koz_rows_exact` takes `stations`
    and returns both wall kinds in one row set, split on a per-row station index (`-1` is the
    obstacle's own zone). Rows stay elastic-relaxable, `occlusion_violation_reference` at the
    returned iterate still feeds `figure_grade`, and `compute_los_margin` in `geometry.py` is still
    the independent pure-Python check.
  - **RETIRED, do not reintroduce:** the per-piece conservative containing ball and the single flat
    wall against it; and clipping the BODY by proximity, then casting the cone from the shrunken
    body — the shadow depends on the whole occluder, so that destroys containment.
- **`figure_grade` is converged AND certificate ≤ 1e-6 at the returned iterate AND clearance > 0
  AND slack ≤ 1e-6 AND the occlusion certificate where a station exists AND — since `bb8b50b` —
  `koz_unsound_clips` == 0.** The slack tolerance sits **inside an eight-order gap between the two
  populations** — 117× above the worst good residue, six orders below the smallest bad one — not on
  its edge. Each condition was tested to sink the gate alone. `tools/make_paper_figure.py` refuses
  to draw a run that fails it.
- **A time penalty without a speed cap is a measured trap, and it raises.** `time_weight > 0` with
  no `v_max` raises `UncappedTimePenaltyError`; the trap was measured before the guard was written —
  arrival collapses to exactly `min_dt × gaps` and ignores the weight entirely. `free_arrival_time`
  is an explicit flag. The speed cap is a **second-order cone, not a linear fallback**, and it
  carries no elastic slack: the penalty may relax keep-out rows, never the physics.
- **`cap_bulge_ratio` is dead code, measured flat.** The 28-run pass swept `elastic_weight` through
  the ladder and left it untouched because it moves nothing.
- **The solver is already dimension-generic.** `dim` comes from the array shape; the only guard is `dim >= 2` (`rust_optimizer/pybind/src/lib.rs:144`). **A run at three spatial dimensions plus time costs one scenario definition and one run, not a solver change** — which is how the paper's largest weakness was retired, by the `fence3d` scenario.

---

## Decided against — do not re-propose

- `git merge` or rebase of `main` (no shared Rust ancestor; see above)
- MIQP / big-M binary side variables — Clarabel has no integer support, and it kills the real-time story
- Purging `orbital_docking` before the deadline — zero paper value, and the SCvx machinery being ported lives in the shared `optimizer.rs`
- A SECOND profiling pass. One was done (`8dc5a4b`, 2026-09-04: allocator profile, allocation-free
  Bezier/De Casteljau kernels, results bit-identical); what remains of a 200-iteration run is a
  limit cycle at the weight cap (`curve` N8_seg16), not a performance problem
- A retry-from-other-start-weights fallback in `optimize_scenario` — tried 2026-09-07, removed
  2026-09-08. Cold starts at 1e3 / 1e4 / 1e5 on every config the default run failed, keep the best:
  it changed no verdict (23 of 28 either way), it moved the floor-OFF `Unsound walls` count onto a
  different trajectory from the floor-ON arm's (`curve` N10_seg8: 0 → 2), and the row reported the
  retried run under the requested start weight (8 of 56 rows read "start 100, one raise, ended at
  1e5"). The one fact worth keeping: escalation is a homotopy in the weight and its path depends on
  the start — no single start dominates (`curve` N8_seg4 best-effort certificate 0.578 from 100,
  2.09e-06 from 1e4, neither a certificate; `wall` N8_seg3/4 −0.306 / −0.481 from 100, −0.300 /
  −0.410 cold at 1e5). That is `solver.multistart`, and it belongs to seeds, not to a hidden ladder.
  `test_a_row_is_traceable_to_the_start_weight_it_reports` is the guard.
- Porting `main`'s full 5-pillar verification harness — the `figure_grade` gate buys the honesty that's needed

---

## Key Files

The full tree is [`README.md`](README.md) § Repo map — not repeated here. Only the entries that
carry a warning or are easy to mistake:

- `spacetime_bezier/__main__.py` -- serves `frontend.py` as of 2026-08-21, THE entrypoint. `io.py` and `sandbox.py` are not entrypoints (2026-08-18), and the sandbox is no longer what `-m spacetime_bezier` starts
- `spacetime_bezier/frontend.py` + `static/frontend.html` -- the one frontend (`fe5c510`): config panel, solve endpoint, axis picker with server-supplied banners (t always vertical when shown), opt-in layers all default-off, diagnostics drawer. All verdicts server-side via `figure_grade_failures`; replay reuses the trace viewer's child script by import so parameters cannot drift. **Covered by `tests/integration/test_frontend.py` (38 tests)** -- request/response path, port mutual exclusion, plane patches proven against their own rows, result/replay cache, per-frame replay planes, cancel kills a running solve
- `spacetime_bezier/viewer.py` -- superseded by `frontend.py` 2026-08-21, no longer an entrypoint. Nothing stored, the solve
  path is just the solve, and the client only draws -- verdict fields are computed server-side from the
  solver's own numbers. Serves `static/viewer.html`; shares port 8767 so it cannot run beside the sandbox
- `spacetime_bezier/optimize.py` -- public API (`escalate_elastic_weight=False` holds a weight), the `figure_grade` gate, `optimize_scenario` (one solve per config from the starting weight; the ladder is gone)
- `rust_optimizer/core/src/spacetime_generator.rs` -- **the set itself**: `Generator`, the centreline read through a station as a point light source. `station: None` is the plain lifted tube. The module docstring carries the derivation and why the widening radius is required rather than cosmetic
- `rust_optimizer/core/src/spacetime_obstacle.rs` -- the wall built on that set: `clip_band`, `component_support`, `clip_geometry`, `rotation_correction`
- `rust_optimizer/core/src/spacetime_constraints.rs` -- row assembly and bookkeeping, `SPACETIME_AXIS_SCALE`; both fixed defects lived here, and the retired occlusion builder did too until `46a15f7`
- `rust_optimizer/core/src/spacetime_optimizer.rs` -- SCP loop, ratio test, certificate at the returned iterate
- `rust_optimizer/core/src/optimizer.rs` -- `solve_qp` + `solve_qp_with_socs`; second-order cones landed with the slant-limit speed cap (that cone carries no elastic slack, by design)
- `paper/` -- one directory per venue, holding that venue's manuscript **and its own README, which is the artifact's document** (schedule, venue rules, manuscript state). [`paper/README.md`](paper/README.md) is the shared handbook: the render/watch pipeline, the verified venue rules, the invariants that break silently (one template paragraph carries the section break that keeps the title block out of the two-column body), and the state of the draft. `tools/render_paper.py`, `watch_paper.sh`, `make_manuscript_skeleton.py`, `docx_edit.py` are its machinery
- `doc/notes/001_problem_formulation/` -- a separate repo, ignored by this one. **Contradicts itself**; do not seed the paper from it

---

## Demo Scenarios

Registry keys are in `spacetime_bezier/scenarios.py` (`SCENARIO_MAP`). What each one shows, the
last measured pass, and the elastic weight each result needs: §Measurements below.

The objective rename, the slant-limit speed cap and the freed arrival time froze together, and the one-pass re-measurement ran **2026-08-20**; §Measurements is current at defaults. Runs enabling `v_max` / `time_weight` / `free_arrival_time` are different problems — re-measure per scenario.

**`loiter` is the paper's demo scenario and its elastic weight is MEASURED.** Corrected
2026-08-31 — this paragraph previously said the weight was unmeasured, that `loiter` was absent from
README §Measurements, and that the registry held 1e6. All three were wrong. The registry holds
**1e5** (`SCENARIO_ELASTIC_WEIGHT` in `spacetime_bezier/scenarios.py`), the measurement was taken
2026-08-30 across six runs — defaults and priced, each with and without `sound_clip` — and README
§Measurements carries it in both the main table and its own dated block. Every rung certifies and
returns the same trajectory; the rungs differ only in iteration count, which is why 1e5 is
registered rather than the first rung that certifies.

**Re-measured 2026-08-31 on the current build: all six runs reproduce exactly** — iterations
3/3/3/3/10/12, weights 100 ×4 then 1e5 ×2, clearances 35.3204 / 35.3767 / 36.8417 / 36.8459 /
24.3026 / 30.5243, arrival 80.000 pinned ×4 then 51.502 and 55.747, hull and occlusion certificates
0.000 throughout, slack ≤ 3e-12, figure-grade on all six. The full record is the comment above
`"loiter"` in `scenarios.py`; do not quote a rung from here.

---

## Measurements

Last full pass **2026-09-08**, every registered configuration run twice — once with the reach floor
on, which is the default, and once with `sound_clip=False` as the comparison arm. **28
configurations, 56 runs:** `original` 5, `curve` 4, `loiter` 2, `diverse` 5, `wall` 6, `fence3d` 4,
`door3d` 2. Defaults otherwise: speed cap off, arrival pinned, elastic weight starting at 100 and
escalating in-loop (x10 per raise, cap 1e5). A run that enables `v_max` / `time_weight` /
`free_arrival_time` is a different problem and must be re-measured.

**FIGURE-GRADE** is the gate, checked per run: converged AND certificate ≤ 1e-6 at the returned
iterate AND clearance > 0 against the true obstacle trajectories AND total slack ≤ 1e-6 AND the
occlusion certificate where a station exists AND `koz_unsound_clips` == 0 AND the resolved weight
reported.

**Provenance.** Taken on this branch (`solver/dual-elastic-weight`, `8dc5a4b` plus the
escalation-trigger repair and the stall-stop removal that carry this table), extension built
2026-09-08T19:18:23 into this worktree's venv, arm64 Darwin, Python 3.14.6 (PyO3 0.24 built with
`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`, ABI-only). One escalating solve per configuration from a
start of 100 and no retry from other starts (the 2026-09-07 fallback is removed, §Decided against);
every row satisfies `Weight == 100 × 10^Raises`, which
`test_a_row_is_traceable_to_the_start_weight_it_reports` asserts on a raising row. Against the
2026-09-06 pass (same trigger, stall stop still in) 25 rows are unchanged at the table's precision;
`curve` N8_seg4 and N8_seg16 run on past the stall (135 → 172 iterations by trust collapse, 82 →
200 by the iteration cap) and `curve` N10_seg8's floor-OFF arm moves +0.1000 → +0.0956.
The 2026-09-01 pass (`paper/journal-1` at `d42db55`, ladder era) is superseded: its
`Weight` column held ladder rungs (800, 3000) the in-loop schedule cannot produce, and its `Iters`
counted one rung, where iterations now accumulate across raises. **Every configuration that never
raises is bit-identical to that pass's re-measured rows** (`original` N8_seg4, `loiter` N8_seg8);
`wall` N8_seg16 moved +0.0556 → +0.0575 under the repaired trigger.

The **Unsound walls** column is measured on the floor-OFF arm only. With the floor on it is zero for
every configuration by construction, so a column of zeros would say nothing. **Weight** is the
weight the run ENDED at, **Raises** how many x10 steps it took to get there from 100, and **Stop**
is the loop's own reason (`stationary` and `merit_streak` are the success claims; `trust_collapse`
is success only if certified, which the Grade column settles).

| Scenario | Config | Unsound walls (floor OFF) | Clearance ON | Clearance OFF | Cert ON | Iters ON | Raises | Weight | Stop | Grade ON | Grade OFF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `original` | N4_seg4 | 4 | +0.6189 | +0.6189 | 0.00e+00 | 11 | 0 | 100 | stationary | **yes** | no |
| `original` | N4_seg8 | 4 | +0.3058 | +0.3060 | 0.00e+00 | 8 | 0 | 100 | stationary | **yes** | no |
| `original` | N6_seg8 | 10 | +0.3231 | +0.3231 | 0.00e+00 | 9 | 0 | 100 | stationary | **yes** | no |
| `original` | N8_seg4 | 10 | +0.6204 | +0.6204 | 0.00e+00 | 9 | 0 | 100 | stationary | **yes** | no |
| `original` | N8_seg8 | 10 | +0.3230 | +0.3230 | 8.80e-09 | 10 | 0 | 100 | stationary | **yes** | no |
| `curve` | N8_seg4 | 1 | +0.1000 | +0.1000 | 5.78e-01 | 172 | 3 | 100000 | trust_collapse | no | no |
| `curve` | N8_seg8 | 3 | +0.0999 | +0.0894 | 8.85e-09 | 111 | 3 | 100000 | trust_collapse | **yes** | no |
| `curve` | N8_seg16 | 2 | +0.1000 | +0.1000 | 1.02e+00 | 200 | 3 | 100000 | iteration_cap | no | no |
| `curve` | N10_seg8 | 0 | +0.0989 | +0.0956 | 1.16e-09 | 88 | 2 | 10000 | trust_collapse | **yes** | no |
| `loiter` | N8_seg8 | 2 | +35.3767 | +35.3204 | 0.00e+00 | 3 | 0 | 100 | stationary | **yes** | no |
| `loiter` | N8_seg16 | 0 | +36.8459 | +36.8417 | 0.00e+00 | 3 | 0 | 100 | stationary | **yes** | yes |
| `diverse` | N8_seg4 | 25 | +0.1136 | +0.1136 | 1.21e-10 | 39 | 1 | 1000 | stationary | **yes** | no |
| `diverse` | N8_seg8 | 32 | +0.0279 | +0.0279 | 1.15e-07 | 60 | 2 | 10000 | stationary | **yes** | no |
| `diverse` | N8_seg16 | 29 | +0.0051 | +0.0051 | 3.02e-08 | 58 | 2 | 10000 | stationary | **yes** | no |
| `diverse` | N10_seg4 | 28 | +0.1085 | +0.1085 | 5.25e-08 | 31 | 1 | 1000 | stationary | **yes** | no |
| `diverse` | N10_seg16 | 11 | +0.0074 | +0.0074 | 7.59e-08 | 47 | 2 | 10000 | stationary | **yes** | no |
| `wall` | N8_seg2 | 22 | -0.0978 | -0.0978 | 4.50e+00 | 48 | 3 | 100000 | trust_collapse | no | no |
| `wall` | N8_seg3 | 17 | -0.3056 | -0.3083 | 2.66e+00 | 44 | 3 | 100000 | trust_collapse | no | no |
| `wall` | N8_seg4 | 14 | -0.4808 | -0.4784 | 2.88e+00 | 43 | 3 | 100000 | trust_collapse | no | no |
| `wall` | N8_seg16 | 130 | +0.0575 | +0.0556 | 5.90e-11 | 55 | 2 | 10000 | merit_streak | **yes** | no |
| `wall` | N10_seg16 | 130 | +0.2018 | +0.2018 | 0.00e+00 | 41 | 2 | 10000 | stationary | **yes** | no |
| `wall` | N10_seg24 | 202 | +0.0667 | +0.0667 | 0.00e+00 | 41 | 2 | 10000 | stationary | **yes** | no |
| `fence3d` | N8_seg2 | 22 | +0.1623 | +0.1623 | 0.00e+00 | 9 | 0 | 100 | stationary | **yes** | no |
| `fence3d` | N8_seg4 | 49 | +0.0342 | +0.0342 | 0.00e+00 | 13 | 0 | 100 | stationary | **yes** | no |
| `fence3d` | N8_seg8 | 43 | +0.0087 | +0.0087 | 0.00e+00 | 10 | 0 | 100 | stationary | **yes** | no |
| `fence3d` | N10_seg8 | 42 | +0.0088 | +0.0088 | 0.00e+00 | 10 | 0 | 100 | stationary | **yes** | no |
| `door3d` | N8_seg4 | 36 | +0.9828 | +0.9827 | 0.00e+00 | 8 | 0 | 100 | stationary | **yes** | no |
| `door3d` | N8_seg8 | 60 | +1.0044 | +1.0009 | 0.00e+00 | 9 | 0 | 100 | stationary | **yes** | no |

**Figure-grade: 23 of 28 with the floor on, 1 of 28 with it off.** Three things have to travel with
that, or the table misleads:

1. **The "Grade OFF" column is not a quality judgement on those runs.** 22 of its 27 no's are the
   uncovered-walls condition *alone* — every other gate condition passes (measured 2026-09-06 by
   reading `figure_grade_reasons` per run; re-read 2026-09-08, still 22). It measures the hole,
   not the trajectories. The other five no's are `curve` N8_seg4 and `wall` N8_seg2/3/4 — four of
   the five that fail with the floor on — plus `curve` N10_seg8's floor-OFF arm, which runs to the
   iteration cap unconverged while its floor-ON arm is figure-grade. `curve` N8_seg16's floor-OFF
   arm is one of the 22: converged at 1000, two uncovered walls.
2. **Four of the five that fail with the floor ON fail for reasons unrelated to the clip.** `wall`
   N8_seg2/3/4 penetrate either way (−0.0978, −0.3056, −0.4808; the 2026-09-01 pass had −0.1292,
   −0.3002, −0.4098 from the ladder's cold restarts), which is the 2026-08-24 densification; `curve`
   N8_seg4 does not converge either way and was a standing failure before any of this — it now
   escalates to the cap and ends by trust collapse at 172 iterations holding 0.58 of slack, and no
   start weight certifies it (held at 1e4, 1.3e-05; escalating from 1e3 / 1e4 / 1e5, 1.95e-05 /
   2.09e-06 / 1.97e-05, measured 2026-09-08), so what it lacks is the basin, not the weight
   (`WORKSTREAM.md` `solver.multistart`).
3. **The fifth is the floor's own cost, and it is named rather than dropped.** `curve` N8_seg16
   certifies at 2.39e-12 with the floor off (converged at 1000 after one raise, though not
   figure-grade there either: 2 uncovered walls) and with it on escalates to the weight cap and
   runs to the iteration cap inside an exact 5-iteration limit cycle, returning a certificate of
   1.02. The stall stop that used to catch this cycle at iteration 82 with 2.87e-03 was removed on
   2026-09-07 (trigger comment in `spacetime_optimizer.rs`); no iterate of the cycle is feasible,
   so the best-iterate fallback has nothing to keep and the returned iterate is whatever phase of
   the cycle the cap lands on. On the one scenario whose tube is genuinely non-convex, a rigorous
   support ceiling is sometimes paid in convergence.
   The clearance cost the 2026-09-01 pass recorded on `wall` N8_seg16 and N10_seg24 is gone at this
   pass: +0.0575 ON against +0.0556 OFF, and +0.0667 both arms.

Everything else is free — `original`, `diverse` and `fence3d` are identical to three decimals across
all fourteen of their configurations, and `door3d` and `loiter` come out slightly better with the
floor on. `curve` N8_seg8 and N10_seg8, which the ladder certified and which `f5a0ac1`'s
never-expiring stall reference had over-escalated out of figure grade (certificates 5.1e-6 and
1.5e-6 at the cap), are figure-grade again under the repaired trigger.

### `loiter`, measured 2026-08-30 — the pass that found the gate blind to an unsound clip

`loiter` was not in the pass above; these six runs are its own, all on one build of the extension
(checked: same mtime before and after). `sound_clip` floors the clip radius at the reach, `idea/spacetime.md`
statement 8; **since `43224b4` it is the DEFAULT** (it was off when this pass ran, which is why
the table has a with/without pair per row). `sound_clip=False` survives so this comparison stays
reproducible — it is a measurement setting, not a way to run the solver. `koz_unsound_clips` is statement 7 counted at the returned iterate: **walls** — one count per
emitted plane, per generator, so a two-approach clip counts twice — whose clip ball did not reach
the segment radius plus the trust-box reach, so the wall was built against a clipped piece **the
next iterate could leave**.

| Run | Iters | Weight | Clearance | Min sight margin | Arrival | Unsound clips | `figure_grade` |
|---|---|---|---|---|---|---|---|
| N8_seg8 defaults | 3 | 100 | +35.3204 | +12.067 | 80.00 pinned | **2** | yes |
| N8_seg8 defaults + `sound_clip` | 3 | 100 | +35.3767 | +12.090 | 80.00 pinned | 0 | yes |
| N8_seg16 defaults | 3 | 100 | +36.8417 | +12.646 | 80.00 pinned | 0 | yes |
| N8_seg16 defaults + `sound_clip` | 3 | 100 | +36.8459 | +12.647 | 80.00 pinned | 0 | yes |
| N8_seg8 priced | 10 | 1e5 | +24.3026 | +7.365 | 51.502 | **24** | yes |
| N8_seg8 priced + `sound_clip` — **the figure** | 12 | 1e5 | +30.5243 | +10.642 | 55.747 | 0 | yes |

Priced is free arrival, `time_weight` 10, `v_max` 5, weight pinned at the registered 1e5 — the
registry value at measurement time, `c32e88b` plus the center-surface working tree. Both
certificates are 0.000, total slack ≤ 3e-12 and the speed-cap violation is 0.0 on all six. Every
row was taken twice, once through the elastic ladder and once pinned at the rung it settled on, and
the two clearances agree exactly; the second pass exists only because the result row does not carry
`koz_unsound_clips`.

**Measured twice, down two different call paths.** The table above came through
`optimize_scenario` and its ladder; the same six configurations were run independently through
`optimize_spacetime` via `make_paper_figure`'s own `solve_pair`, and the counts agree — 2 at
N8_seg8 defaults, 0 with the floor, 0 at N8_seg16 either way, 24 priced at the default clip, 0
priced with the floor. Two paths that share only the Rust core had to agree, and do.

**The last two columns together were the finding, and it is now closed.** At the time of this pass
`figure_grade` passed the priced default-clip run while 24 of its clipped volumes escaped the ball
their wall was built against: `figure_grade_failures` did not read `koz_unsound_clips`, the key was
not propagated into the result row, and **on that condition the gate could not fail.** Only
`tools/make_paper_figure.py` refused, and only for the figure.

Closed 2026-09-01 by importing `bb8b50b` and `43224b4` from `paper/journal-1`. The count is now
propagated into the result row (NaN when the extension does not emit it, so a stale build refuses
rather than grading as sound), the predicate reads it, and the frontend forwards it. **The
`figure_grade` column in the table above is therefore the OLD gate's verdict** — the two rows with
a nonzero count would not pass today.

**The ladder settles at 100 at defaults, not at the registered 1e5.** The registered weight belongs
to the priced problem; a defaults row quoting it would be quoting a rung this scenario never needs
with the arrival pinned.

**What the clipped-volume construction changed, measured against the same configurations built
with the retired hull-of-band plane.** Where the obstacle is straight its tube is convex and the
two constructions agree on the plane, so `original`, `fence3d` and `door3d` move only in the last
digits. Where it curves they differ, and the difference is iterations rather than answers: `curve`
N8_seg8 went from 11 iterations to 39 and from +0.0894 to +0.0919 of clearance, and `curve`
N10_seg8 then needed the ladder's top rung (1e5) where it used to certify at 800 (ladder era; with
the in-loop escalation it certifies at 1e4 after two raises, 2026-09-06). That is the
conservatism of a rigorous support ceiling being paid in step size, which is what sequential convex
programming pays it in. Nothing that certified before stopped certifying.

**The clip-radius floor fires on the real scenario, not only in the unit tests.** On `curve` at the
straight seed, segment 1 against the arc has its centroid 0.598 from the centreline of a tube of
radius 0.9 — inside the keep-out zone — so the clip radius is floored to 0.900 instead of shrinking
to 0.598. Sweeping the clip radius over every segment and both obstacles at four trust radii, the
clipped volume has **exactly one connected component everywhere** and nothing is dropped: time
monotonicity keeps the lifted tube self-avoiding, so no ball can cut a piece out of its middle.

**Walls group by local approach, not by connected component — changed 2026-08-26.** The band is cut
at every interior local maximum of the centreline's distance to the segment centroid, one wall per
dip of the profile. The reason is the wrapping bend of panel B2: one connected lump curling around
the centroid got one non-separating wall (margin −1.154 on a segment whose centroid is 0.205
OUTSIDE the keep-out zone, a row needing trust ≈1.04 against the default 0.5); the cut gives the
same geometry two walls at +0.529/−0.162 with nothing else changed. At the straight seed the cut
emits **exactly the same row count as the component grouping on all seven registered scenarios** —
every obstacle here approaches each segment once — so the table above did not move. The multi-wall
path is exercised by the panel tests (`test_panel_b1_...`, `test_panel_b2_...`) and by the Rust
in-crate tests in `spacetime_obstacle.rs`, but by no scenario in this repository.

The elastic weight is part of the result, not a tuning knob: above the exact-penalty threshold the
penalized and constrained problems share a solution, below it they do not, and the threshold
depends on the optimal multipliers so it differs per scenario and per segment count.
The solver escalates it in-loop from the starting weight (x10 per raise, cap 1e5; see
`ELASTIC_STALL_WINDOW` in `spacetime_optimizer.rs`) and `optimize_scenario` records the weight the
run ended at. Segment count reversed direction when one-plane-per-segment landed — `original` reaches
+0.620 at 4 segments against +0.323 at 8 — so old config lists that start at 8 segments miss the
best result entirely.

Timing profiles at these defaults are objective artifacts (the smoothness term is blind to the
time coordinate; control-point times can sit on the `min_dt` floor). Geometry is evidence; timing
is not, until a run sets the speed cap and time penalty.

---

### The clip floor, measured 2026-09-01 on this build

Two things were imported from `paper/journal-1` (`bb8b50b`, `43224b4`) and re-measured here rather
than quoted.

**The `station_fence` measurements that were here are gone with the scenario.** They recorded that
all four of its configurations returned `koz_unsound_clips == 0` at both clip settings, which is
why its two controls pinned `sound_clip=False`. `8092a08` removed the scenario on 2026-09-01 and
those controls now run against `loiter`, which *does* return uncovered walls at the default clip,
so upstream's `sound_clip=True` is the correct pin there and is what the merge took.

**`original` priced, trust-radius sweep** — `v_max` 5, `time_weight` 10, free arrival, N8_seg4:

| Trust radius | Floor off | Floor on |
|---|---|---|
| 2.0 | 9 unsound, 11 iters, **not figure-grade** | 0 unsound, 11 iters, figure-grade |
| 4.0 | 12 unsound, 9 iters, **not figure-grade** | 0 unsound, 9 iters, figure-grade |
| 8.0 | 12 unsound, 9 iters, **not figure-grade** | 0 unsound, 9 iters, figure-grade |

The gate now refuses every default-clip run in that sweep and accepts every floored one, which is
the imported change doing its job. **The floor costs no iterations at this configuration.**

**There is one configuration where it costs everything, and it is a corner, not a trend.** Through
`_solve` in `test_speed_cap_diagnostics.py` — `original` N8_seg4, `v_max` 1.0, `time_weight` 1.0,
`elastic_weight` 1e5, `time_ub_scale` 6.0, free arrival, `max_iter` 300 — the floor takes trust 8.0
from 13 iterations to the 300 cap (trust 2.0 is 21 → 22, trust 4.0 is 13 → 13). Reproduced here
exactly.

It is tempting to read that as "a large trust radius and a sound clip are in tension," and that
reading is wrong. Starting from the configuration above the table and moving **one knob at a time**
toward it, at trust 8.0 with the floor on: baseline 9 iterations, `v_max` 1.0 → 8, `time_weight`
1.0 → 8, `elastic_weight` 1e5 → 14, `time_ub_scale` 6.0 → 9, **all four together → the 300 cap.**
No single parameter reproduces it. What survives is only the arithmetic — the reach is the segment
radius plus one trust step, so at trust 8.0 the floor is about 14 on a 10-unit scene and the clip
has stopped localizing — and it does **not** follow that this costs convergence.

**Superseded.** The 28-configuration pass that came with those commits is now §Measurements above,
with its provenance and a three-row spot-check on this branch's own build.

---

### `loiter` is graded as a PAIR, and the pair test reaches into `tools/`

`tests/integration/test_loiter_scenario.py` (merged `ccae73f`, 2026-09-01) carries the coverage the
`station_fence` removal lost: two runs differing only in the occlusion rows, every assertion on the
independent `compute_los_margin` rather than on the certificate under test, plus the schedule graft
in both directions — constrained path on the baseline schedule −2.1086 (link lost), baseline path on
the constrained schedule +10.6198 (link held). The path is interchangeable; the schedule is not.

**It imports `solve_pair` out of `tools/make_paper_figure.py` by path, deliberately.** A test
reaching into `tools/` is unusual and it is kept on purpose: the run the suite grades is then the
same run the figure is drawn from, so a drift in the trust radius the tool forwards or the weight it
takes from the registry fails in the suite instead of silently producing a figure nobody re-checked.
Reimplementing the pair would cost 15 lines and lose that property.

**One deliberate false zero is itself asserted.** The baseline solves with `stations=None`, so it
builds no shadow rows, so `occlusion_violation_reference` returns **0.0 for a trajectory that spends
1.56 s blocked**. The test asserts that false zero so nobody later "simplifies" the file by trusting
the number the solver reports for a constraint it was never given.

**Falsified in both directions, 2026-09-01.** Doctoring the pair so the two halves stop differing by
the occlusion rows turns the file red, and which tests fire depends on which half is doctored:
making the *baseline* constrained sinks three of four; making the *constrained* run a baseline sinks
two of four (the link assertion, and the graft, which reports the schedules 0.0000 s apart).

**Known gap, small:** `test_the_pair_differs_by_the_occlusion_rows_and_nothing_else` does **not**
detect a pair whose halves fail to differ — with both halves solved as baselines it still passes,
because every assertion in it is satisfied by two baselines. Its name claims more than it checks.
The condition is caught by the neighbouring `test_the_constrained_run_holds_the_link`, so the file
is sound; the name is the defect.

---

## Known Issues

- **ONE TEST IS RED, deliberately, and the decision is the user's.** `wall` was densified
  2026-08-24 at the user's request (spacing 0.8 → 0.5, 13 → 21 circles). The forbidden slab barely
  moves, but 8 more per-segment planes make the relaxation more conservative, and the consequence
  is measured: `wall` N8_seg2 went from +0.102 (clears while standing on 1.35 of slack) to −0.129
  (penetrates outright). That row was the ONLY real specimen of "clears by sampling yet fails the
  certificate", so `tests/integration/test_figure_grade_gate.py::test_a_clearing_run_can_still_be_standing_on_slack`
  now fails. 19 configurations across `wall`, `diverse` and `original` were probed; no replacement
  specimen exists. **Do not "fix" this by weakening the test, re-anchoring it on a synthetic row,
  or reverting the density** — the user has the evidence and is choosing between keeping the
  density, restoring it, and dropping the test. **Corrected 2026-08-31:** this used to add that
  every `wall` row in README §Measurements is stale for the same reason and marked so. Neither half
  held — the README pass is dated **2026-08-26**, two days after the densification, and its `wall`
  row already attributes N8_seg2/3/4 penetrating to the densification rather than to the clipped-volume
  change. The rows are current and carry no stale marker.
- **`spacetime_bezier/viewer.py` is UNVERIFIED.** Tracked 2026-08-20 because it was already
  documented as live, not because it was checked. No test exercises it -- nothing in `tests/`
  imports it, and no test opens a browser, so the server, the static page, and the verdict fields
  it computes have never been asserted against anything. It compiles and it starts; that is the
  whole of the evidence. Treat any picture it draws as unconfirmed until a test covers the
  request/response path.
- **A superseded viewer stack is still on disk and still documented as live.**
  `spacetime_bezier/viewer.py` replaced it, but `spacetime_bezier/sandbox.py` (478 lines),
  `figures/spacetime_bezier_interactive.html` (3,868 lines), the `--bake` +
  `sync_interactive_viewer` path in `io.py`, and `figures/spacetime_scenarios.json` were never
  deleted — that was left as a separate, explicit decision. Meanwhile the interactive page's inline
  `const SCENARIOS` blob is committed **test mock data** (one control point at the origin, no
  `diverse`, no `wall`); the source was fixed 2026-08-18 but the committed data was not, and only
  `--bake` regenerates it. Serving over HTTP is unaffected — only `file://` shows the mock.
- **Baked and live runs use different solver parameters.** `--bake` defaults to `tol=1e-12,
  max_iter=10000`; the sandbox posts `tol=1e-6, max_iter=30`; the scenario table was made with
  neither, and there is no CLI flag for trust radius at all. A live violation of "one canonical
  execution model", left alone deliberately — changing a default silently changes every number.
- The venv may be broken (built against a Homebrew Python since upgraded). If `python3 -m
  spacetime_bezier` fails with a dyld error, rebuild it and re-run `maturin develop --release`.

---

## Open work

Moved. Open solver items — the test drift that leaves `compute_los_margin` uncovered, multi-start
seeds, the missing KKT residual, and the unread SCvx / topology references — are in
[`WORKSTREAM.md`](WORKSTREAM.md). This file records what is **established**; that one records what
is **not done**.

