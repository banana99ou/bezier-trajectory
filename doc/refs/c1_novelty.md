# C1 — Novelty positioning: external ground truth

Workstream C, item C1. **Blocks A2.**

Everything below was read from the source, not recalled. Where a source was read only at
abstract level, or through a fetch summarizer rather than the PDF, it says so. PDFs of the
load-bearing sources are in [`papers/`](papers/).

**Chosen claim: (a) decomposition-free.** See "The claim that survives" below.

---

## Verdict

**The core idea as `CLAUDE.md` states it is not novel.** "Lift moving obstacles into space-time
so a constant-velocity obstacle becomes a static tube, then apply Bezier convex-hull supporting
half-spaces in the lifted space" was published in August 2025 by Osburn, Peterson & Salmon.

Four of the five items in `CLAUDE.md` § "What is genuinely new" appear in that paper. This
section of `CLAUDE.md` needs revision; it has not been edited by this workstream.

| `CLAUDE.md` claim | Osburn et al. 2025 |
|---|---|
| 1. moving obstacle → static tube in space-time | yes, §F |
| 2. time as a Bezier coordinate; hull applies in lifted space | yes — control points carry an explicit `t` component |
| 3. finite-height tubes for time-limited obstacles | yes — prism runs from initial time to final time |
| 4. time monotonicity `P[i+1,t] − P[i,t] ≥ min_dt` | yes — `(x_{v,i+1} − x_{v,i})_t ≥ 0`, plus an explicit "minimum time separation" variant |
| 5. objective penalizes only spatial acceleration | no — they minimize `(x,y)` path length via the control-polygon upper bound |

---

## Why free space cannot be convexified (settles a recurring proposal)

Recorded because it will be re-proposed: *can a coordinate transform (inversion, conformal
map, Nyquist-style) turn the non-convex free space into a convex one?*

**No, and the obstruction is topological, not geometric.**

- A convex set is contractible, hence simply connected.
- Free space around an obstacle is not simply connected — a loop around the obstacle cannot be
  contracted.
- Homeomorphisms preserve the fundamental group.

Therefore no continuous invertible map takes free space onto a convex set. Sphere inversion
`x → x/‖x‖²` maps the exterior of the unit disc to the *punctured* interior: the obstacle centre
goes to infinity and infinity comes to the origin, exactly as the proposal intends — and the hole
survives as the puncture. The hole is not an artifact. It is the fact that passing left and
passing right are genuinely distinct plans; convexity would imply their average is also a
solution, and the average is "straight through the obstacle."

Prior art for the strongest version of this idea: **Rimon & Koditschek**, navigation functions —
*"Exact robot navigation in geometrically complicated but topologically simple spaces"*, and
*"The construction of analytic diffeomorphisms for exact robot navigation on star worlds"*
(Trans. AMS, 1991). They deform complicated geometry into simple geometry; they do not attempt to
change the topology. The tell is that convergence holds from *"almost all"* initial
configurations — the excluded set is the saddle points the topology forces to exist, one per
obstacle. (Read at abstract/summary level only; PDF not archived here.)

**Consequence, and a usable framing for the paper's introduction:** there are exactly two moves
available. Cover free space with convex pieces (IRIS + GCS), or cut it to one convex piece per
obstacle by committing to a side (supporting half-spaces). Everything in the prior art is one of
these two. What the space-time lift actually buys is not convexity — it is turning a
**time-varying** constraint into a **static** one.

---

## The prior-art map

Four families, all with time as a genuine dimension.

| family | representative | how time enters |
|---|---|---|
| configuration-time space | Erdmann & Lozano-Pérez 1987 | the origin; represents space-time **approximately, by 2D slices** |
| search-based | SIPP (Phillips & Likhachev 2011), CBS variants | discrete space-time graph |
| sampling-based | ST-RRT\* (Grothe et al., ICRA 2022, arXiv:2203.02176; in OMPL) | samples `(x,t)` directly, free arrival time |
| corridor / convex-opt | SSC 2019 → GCS 2023 → ST-GCS 2025 → Osburn 2025 → Tang 2026 | convex sets **in space-time**, curve constrained inside |

### The corridor / convex-opt family in detail

| | ST-GCS (2503.00583) | SPOT (2602.01189) | **Osburn (2508.10203)** | us |
|---|---|---|---|---|
| time | coordinate, continuous | coordinate, **binned Δt=0.2 s, 2 s horizon** | coordinate, continuous | coordinate, continuous |
| curve | **piecewise linear** | MINCO polynomial | **Bezier** | **Bezier** |
| dynamic obstacle | other robots' planned paths only | perceived bounding boxes | any constant-velocity **polygon** | any constant-velocity **capsule** |
| free space | extrude 2D sets + exact carve (ECD) | inflate along an RRT\* path | **IRIS sampled in 3D** | **none — half-space per (segment, obstacle)** |
| optimality | time-optimal, global | local | **global w.r.t. its graph** | local |
| arrival time | free | — | free | **fixed** (endpoint pins `t`) |
| solver | Mosek (commercial) | L-BFGS | **Clarabel** | **Clarabel** |
| spatial dims | 2 | **3** | 2 | 2 today, 3 intended |

**Osburn is the only real competitor.** ST-GCS produces no smooth trajectory; SPOT cannot plan
past two seconds. Nobody occupies *continuous time + smooth curve + 3 spatial dimensions* at once.

---

## The claim that survives

Every method in the corridor family requires a **convex decomposition of free space-time computed
before optimization**, plus a combinatorial layer to select cells. Ours requires neither:
supporting half-spaces are generated directly against the tube, each SCP iteration.

> **Decomposition-free**: moving obstacles enter as linearized supporting half-spaces on a lifted
> tube rather than as a precomputed convex cell complex, so there is no cell-selection search.
> This is what makes the lift practical at 3 spatial dimensions plus time, where building and
> searching the cell complex is the dominant cost.

Do **not** phrase the hook as "time as a coordinate" — that is Osburn's, published.

### Evidence that the decomposition is the bottleneck, from their own papers

Osburn Table I — cluttered scenario, 20 obstacles, 100 runs per row, **3D (2 spatial + time)**:

| IRIS samples | convex sets | edges | cost (m) | time (s) |
|---|---|---|---|---|
| 80 | 15.22 | 22.71 | 1.24 | 4.75 |
| 100 | 17.05 | 26.45 | 1.23 | 5.28 |
| 250 | 27.58 | 52.28 | 1.12 | 11.50 |
| 500 | 40.16 | 95.27 | 1.05 | 26.80 |
| 1000 | 55.62 | 148.44 | 1.03 | 82.30 |

17 % better trajectory for 17× the compute, and coverage is still incomplete at the bottom row:
*"the sets themselves do not provide complete coverage of the free space… This results in a
trajectory that is not globally optimal with respect to the geometric minimum distance."* Their
conclusion concedes it: *"Although the convex set generation method limits solution quality…"*

Both halves of the machine degrade with dimension. IRIS places seeds by **random sampling** —
Osburn: *"can generate collision-free convex sets in dimensions higher than R³… However, its main
drawback is that it requires sampling."* GCS then solves over whatever graph results.

SPOT is the corroborating data point: the only genuinely 4D method in the family, and it caps its
horizon at **2 seconds** for tractability with time binned at 0.2 s. A decomposition-free
formulation has no such term — a 20-second horizon costs what a 2-second one costs, because the
tube is one static object either way. **This is the `wall` scenario's reason to exist.**

### Honest counter-position

Their strengths are real and must be stated in A4:

- Osburn and ST-GCS need **no initial guess**. We need seeds and multi-start (B6).
- They are **globally optimal** (w.r.t. their graph). We are local.
- Osburn's small dynamic-obstacle case takes **0.52 s**, and his static case 0.29 s. These are not
  slow. Whatever our current numbers turn out to be, the gap on small problems is unlikely to be
  more than a small factor — **"we are fast" cannot carry the paper.** The argument is how the
  cost scales with dimension, not the wall-clock on a 2D toy.

---

## Benchmark comparison

Osburn: Python, CVXPY, **Clarabel**, AMD Ryzen 7 9700X / 32 GB. Ours: Rust, **Clarabel**, Apple M1
(`BENCHMARKS.md`). Same solver — an unusually fair comparison.

| | scenario | time | feasible |
|---|---|---|---|
| Osburn | 1 static obstacle | 0.29 s | yes, globally optimal |
| Osburn | **1 moving obstacle**, agent speeds up to pass then slows | **0.52 s** | yes, globally optimal |
| Osburn | 20 obstacles, cluttered | 4.12 s (→ 82.3 s for 17 % better) | yes, optimal w.r.t. its graph |
| us | — | **no current measurement** | — |

**Our side of this table is deliberately empty.** `BENCHMARKS.md` predates commit `848bf3b`
(B5, SCvx ratio test) and the large uncommitted solver diff on top of it; its numbers
(`original` 177 ms feasible, `wall` 12.3 s infeasible −0.112, `diverse` 4.4 s infeasible −0.495)
describe a solver that no longer exists and **must not be quoted**. `wall` is reported to wait
correctly on the current build. Re-measure before any number here reaches the paper.

Osburn's 0.52 s dynamic case is behaviourally our `wall`: the agent speeds up to slip past a
moving obstacle, using timing as a real decision. That is the comparison the paper has to win,
and it needs a fresh measurement to be made at all.

---

## Differentiation beyond (a) — for A3/A4, not for implementation

Adding constraints is Osburn's own stated contribution (*"the derivation of general
GCS-compatible constraints"*), so competing there head-on loses. The seam is the **kind** of
constraint GCS structurally cannot absorb: GCS's vocabulary is a convex set per vertex and a
convex cost per edge, so a constraint must be expressible as "stay inside this convex region."

| constraint | shape in `(x,y,z,t)` | cost to us | cost to GCS |
|---|---|---|---|
| **max** range from a moving station | ball swept along a line — convex tube | one linear row | fine, convex |
| **min** range keep-out | keep-out tube — non-convex free side | one linearized supporting half-space — *machinery we already have* | must be carved around |
| **line-of-sight / occlusion** by an obstacle | shadow cast from a *moving* station — twisting, neither convex nor a straw | one linearized row per (segment, obstacle, station) | IRIS must carve a twisting shadow; set count explodes |

A minimum-range keep-out around a moving station is *the same object as our KOZ tube* and comes
free once B1 lands. Supporting evidence for the seam: the visibility-aware / perception-aware
trajectory literature (RAPTOR arXiv:2007.03465, SVPTO, FOV-constrained flight arXiv:2403.17067)
is entirely local NLP/SQP over splines — nobody does it by convex decomposition. The two
literatures do not overlap.

**Naming:** do not call this "semantic." SSC (Ding et al. 2019) owns that word here and it means
traffic lights and speed limits. Use *mission constraints* or *sensing-constrained*.

**Scope:** this is a positioning paragraph and future work. Do not implement before the deadline.

---

## Cite-list (≈15 for 2 pages)

- Erdmann & Lozano-Pérez, *On multiple moving objects*, Algorithmica 2:477–521, 1987 — origin of configuration space-time
- Phillips & Likhachev, *SIPP: Safe Interval Path Planning for Dynamic Environments*, ICRA 2011
- Grothe, Hartmann, Orthey, Toussaint, *ST-RRT\**, ICRA 2022, arXiv:2203.02176
- Marcucci, Umenberger, Parrilo, Tedrake, *Shortest Paths in Graphs of Convex Sets*, SIAM J. Opt. 34(1):507–532, 2024, arXiv:2101.11565
- Marcucci, Petersen, von Wrangel, Tedrake, *Motion planning around obstacles with convex optimization*, Science Robotics 8(84), 2023, arXiv:2205.04422
- Deits & Tedrake, *Computing Large Convex Regions of Obstacle-Free Space Through Semidefinite Programming* (IRIS), WAFR 2015
- **Osburn, Peterson, Salmon, arXiv:2508.10203, 2025 — the nearest prior work; mandatory**
- Tang, Mao, Yang, Ma, *Space-Time Graphs of Convex Sets*, IROS 2025, arXiv:2503.00583
- Ding, Zhang, Chen, Shen, *Safe Trajectory Generation … Spatio-Temporal Semantic Corridor*, RA-L 2019
- Tordesillas & How, *MADER*, IEEE T-RO 38(1), 2022, arXiv:2010.11061
- Zhang, Yadmellat, Gao, *A Sufficient Condition for Convex Hull Property in General Convex Spatio-Temporal Corridors*, arXiv:2110.00065, 2021
- Rimon & Koditschek, navigation functions (only if the convexification question is addressed in text)

## Read-list, ranked

1. Osburn arXiv:2508.10203 — ~40 min, read §E–§F and Table I closely. Our twin. **PDF archived.**
2. Zhang/Yadmellat/Gao arXiv:2110.00065 — ~25 min, short, load-bearing for the hull argument
3. MADER §V-A separating planes — ~30 min; plane-as-decision-variable is the alternative to our
   linearization, and MINVO gives a hull 2.36× tighter than Bernstein
4. Marcucci et al. GCS — for framing; perspective-operator relaxation is why "one-shot" works

## Lead-list — found, not verified

- SPOT arXiv:2602.01189 — read via fetch summarizer only; PDF exceeds fetch size limit
- Deolasee et al. arXiv:2209.15150 — trapezoidal prism corridors
- "Mao et al. 2024, collision avoidance with Bezier curves" — cited by arXiv:2607.00444, unverified

---

## Confidence and known gaps

- **Read from PDF text, in full:** Osburn 2508.10203; ST-GCS 2503.00583.
- **Read via fetch summarizer (HTML/abstract):** GCS 2205.04422, Marcucci SIAM 2101.11565,
  Zhang 2110.00065, SPOT 2602.01189, Tang 2607.00444, IRIS (search summary only).
- **Search-level only, bibliography not verified against the source:** SSC 2019, MADER, SIPP,
  velocity obstacles, Rimon & Koditschek.
- **Not verified: Erdmann & Lozano-Pérez 1987.** The archived PDF is a scan with no text layer;
  `pdftotext` yields nothing. No verbatim quote is available without OCR. The abstract's
  "two-dimensional slices" phrasing, used above, comes from a search summary of the Springer page
  and is **not** confirmed against the paper.
- **Bibliographic data that passes through a summarizer is unreliable.** Concrete instance: a
  fetch of arXiv:2607.00444 reported MADER as "Tordesillas & Beard, 2021"; it is Tordesillas &
  How, T-RO 2022. Verify every author list before it reaches the `.bib`.
- **"No paper does X" is weaker than a proof.** No decomposition-free space-time formulation was
  found, but absence of evidence is not proof of absence.
