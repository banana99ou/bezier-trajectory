# Paper 2 — online space-time planning against uncertain hazards

**Central doc for paper 2.** Future work. Nothing here belongs in paper 1.

**Scope boundary.** Hazard motion is *not* known. Sensing range is limited. The field is
re-estimated as observations arrive, and the plan is recomputed on a receding horizon. Paper 1
([`PAPER_1.md`](PAPER_1.md)) assumes known deterministic motion and declines receding-horizon
replanning for that reason. Paper 2 relaxes exactly that assumption — it is the sequel, not a
correction.

**Formal write-up.** The mathematics of §5 is written up properly, in Korean, with its own
figures, in [`doc/notes/002_probabilistic_koz/main.tex`](doc/notes/002_probabilistic_koz/) —
density vs confidence, lobe radius and expiry, convexity by covariance growth model,
correlation-determined shape, merging at the saddle, convex hull of the merged cluster, and the
link to the existing SCP-QP formulation. **That note is the paper draft; this file is the
positioning and prior-art record it does not contain** (its `references.bib` currently holds two
entries, so §3 and §4 here are not yet reflected in it).

**Figure duplication to resolve.** `doc/notes/002_probabilistic_koz/` (fig1–fig3 +
`make_figures.py`) and [`figures/risk_field/`](figures/risk_field/) are two independent renderings
of the same three figures. Keep the note's copies; `figures/risk_field/` is the throwaway.

---

## 1. The idea

There are no separate obstacles. There is **one probability field over space-time** — the
likelihood that a hazard occupies `(x, t)`. Every hazard contributes to it. You never avoid *an
obstacle*; you avoid a region of the field.

Each contribution starts as a sphere of unbounded range. Knowing the hazard's velocity **focuses**
it into a lobe. A probability threshold `ε` **cuts** the lobe to something finite.

The "fog" is not a second model. An unobserved hazard has no velocity estimate, so its
contribution stays unfocused — wide, flat, low. Same equation, less information. Detection
collapses it into a lobe. The uniform background and the sharp lobes are the same object at
different confidence levels.

Working metaphor: walking to a campsite through foggy forest, avoiding bees. Bees you can see have
lobes. Bees you cannot are a diffuse pressure that makes you cautious without forbidding anything.

## 2. Vocabulary

Recorded because these terms are load-bearing and non-obvious.

- **Class** — a distinct *way past*. Two hazards give three: left of both, between, right of both.
  Two paths are the same class if one can be slid onto the other without ever crossing a hazard.
- **Committing to a class** — making that discrete choice before optimizing. A continuous
  optimizer polishes *within* a class and can never jump between classes, which is why something
  upstream must choose. Solver item B6 (procedural seeds, multi-start) is that something.
- **Filtration** — the whole nested family of thresholded free spaces as `ε` sweeps, treated as
  one object rather than one map per `ε`.
- **Barcode** — one horizontal bar per class, spanning the range of `ε` over which that class is
  passable. Long bar = works whether timid or bold. Short bar = exists only at loose `ε`.
- **Persistence** — bar length. The persistent classes are the ones robust to not knowing your own
  risk tolerance.
- **Homology vs homotopy** — homotopy is "can one path be deformed into the other"; homology is a
  coarser, computable shadow of it. Bhattacharya & Ghrist further use `Z₂` coefficients, counting
  obstacle crossings mod 2, which discards winding — a deliberate trade for tractability.

## 3. Prior art: which ingredients are taken, and what regime is open

**Ingredient by ingredient, this is a crowded field.** Three of the four load-bearing ideas are
published. Do not open a paper claiming any of them.

| idea | status | source |
|---|---|---|
| one field, **no objects or tracks** | **taken, 2006** | Bayesian Occupancy Filter |
| `ε` changes the **topology** of free space; sweep it instead of choosing | **taken, 2015, and answered better** | Bhattacharya & Ghrist |
| moving obstacles → **homotopy classes in configuration-time** | **taken, 2024** | Topology-Driven Parallel Trajectory Optimization |
| side lobes = multi-modal intent → branch the plan | **taken**, standard | contingency / branch / scenario-tree MPC |

### Bayesian Occupancy Filter — "no separate obstacles", 2006
Coué, Pradalier, Laugier, Fraichard, Bessière, IJRR 25(1), 2006. Review: Adarve et al., Sensors
2017. Their own statement of the design: *"concepts such as objects or tracks do not exist; they
are replaced by more useful properties such as occupancy or risk, which are directly estimated for
each cell of the grid."* The 4D variant already carries two spatial and two velocity dimensions per
cell. This is §1 above, twenty years old, velocity included.

### Bhattacharya & Ghrist — the `ε` question, 2015
*Persistent Homology for Path Planning in Uncertain Environments*, IEEE T-RO 2015.
Read in full from the preprint.

They pose the exact question: *"it is unclear how to select this threshold. Low values of threshold
result in suboptimal paths while higher values may result in unsafe trajectories."* And they reject
the cost-based alternative too: *"these approaches lack robustness because there are cases when the
penalties for some edges are not high enough to offset the incentive offered by a shorter path, and
the resulting plan may very well pass through regions with high probability of occupancy."*

**That second sentence settles the cost-vs-constraint fork.** A cost will always trade risk for
distance when the distance saving is large enough. It does not take calibrated risk; it takes
whatever risk the geometry makes cheap.

Their answer: don't choose `ε`. Sweep it and take the class *"free of obstacles over the largest
range of threshold values."* Persistence replaces threshold selection.

**How close is it to §1?** Close on one axis, absent on every other:

| | Bhattacharya & Ghrist | paper 2 |
|---|---|---|
| field lives on | static 2D space | space-time, 3 + 1 D |
| field means | `P(cell blocked)`, fixed | `P(hazard at (x,t))`, evolving |
| velocity | none | central — it focuses the lobe |
| sensing | none; the map is given | limited range; detection collapses the cloud |
| update | **explicitly out of scope** | the entire point |
| output | grid path from A\* | continuous Bezier |
| time as a class | does not exist | wait / hurry / ahead / behind |
| cost | 246 s, 400×320 grid, 100 `ε` values | must be online |

They bracket the online case out by name: incremental planners *"do not address the fundamental
question of how to plan a path for a given probability map."*

**So B&G is not a competitor. It is the tool, and the mandatory citation.** It owns "persistence
instead of threshold selection" completely. It owns none of the fog model, the lobe geometry, the
space-time filtration, or the online loop.

Related: Pokorny, Hawasly & Ramamoorthy, IJRR 35(1–3), 2016.

### Space-time homotopy classes, 2024
*Topology-Driven Parallel Trajectory Optimization in Dynamic Environments*, IEEE T-RO 2024
(arXiv:2401.06021). Already states our framing: moving obstacles *"puncture the configuration-time
space, rendering the collision-free manifold highly non-convex, which results in the emergence of
multiple distinct homotopy classes"*, then optimizes one per class in parallel.

### Adjacent, different mechanism
TRUST-Planner (arXiv:2508.14610, Aug 2025) — topology-guided, uncertain obstacles, spatio-temporal.
Works from a *"dynamic enhanced visible probabilistic roadmap"* and a *"dynamic distance field"*,
not a thresholded risk field. Nearest by keyword, not by method. Full PDF not read.

## 4. The open question

Every prior barcode is computed **once**, from a field that never changes. Paper 2's field changes
on every observation.

> **What does persistence mean when the filtration is over space-time *and* the field is updating?**

Three concrete sub-questions, none of which exist in the 2015 setting:

1. **The barcode moves.** The most-persistent class at `t₀` can be dead by `t₁`. The tool is
   probably **vineyards** — persistence diagrams evolving over a second parameter (Cohen-Steiner,
   Edelsbrunner, Morozov, *Vines and vineyards by updating persistence in linear time*).
   **Not verified to apply.**
2. **You must commit against a barcode you cannot see yet.** "Longest bar" is the wrong criterion
   when the bars will move. The right one is closer to *the class most likely to still be viable
   after the next observation* — a decision problem, not a topology computation.
3. **Longest ≠ lowest, and the disagreement is physical.** B&G's Figure 12 is two high-occupancy
   blobs joined by a low-occupancy bridge; *"the most persistent class dies at a low value of
   threshold due to the presence of the bridge, and two other classes survive at even lower values
   of ε."* They name this and do not fix it. In paper 2 the bridge is two fast hazards whose lobes
   are merging, so there is a physical reason to prefer the class that survives *lowest* over the
   class that survives *longest*.

Sub-question 3 is the strongest candidate contribution found so far.

Second known crack in the 2015 method, also unfixed: with a noisy map *"a large number of spurious
classes show up due to the noise"* — the barcode fills with junk routes from sensor speckle.

## 5. Geometry that is worked out

### Threshold density, not confidence
The two criteria give different shapes, and the distinction was not found stated anywhere.

- **confidence** threshold ("the 95 % region") → radius `∝ σ(t)` → an unbounded **cone**
- **density** threshold ("risk per unit volume `≥ ε`") → radius rises, peaks, returns to zero → a
  closed **lens**, with a closed-form expiry

For an isotropic Gaussian in `n` spatial dimensions with `σ(t)` growing in time:

```
r(t) = σ(t) · sqrt( 2 · ln( C / σ(t)^n ) ),     C = 1 / ( ε · (2π)^(n/2) )
t_expire :  σ(t) = C^(1/n)          beyond this the region is empty
```

Consequences worth keeping:

- The region is finite **by construction**, not by imposing a horizon.
- Near the observation the density criterion is *stricter* than 2σ (≈3σ at the measured
  parameters); far from it, looser. Caution scales with how well the hazard is localized — the
  correct direction, and the structural answer to the freezing-robot problem (Trautman & Krause,
  *Unfreezing the Robot*, IROS 2010). A confidence threshold demands the same 95 % forever and so
  gets steadily more conservative the less it knows.
- `r(t)` concave ⟹ the lens is **convex in space-time** ⟹ paper 1's supporting-half-space
  machinery and convex-hull certificate carry over unchanged. Finite extent is the capped-tube
  geometry the door scenario already needs.
- Convexity by growth law: constant or `∝ t` or `∝ √t` → concave → convex. `∝ t^{3/2}` or `∝ t²`
  (bounded-acceleration reachable set) → not concave → not convex, but over a finite horizon a
  convex `r(t)` lies below its own secant line, so the secant cone is a valid concave outer bound.

### Chaining lobes: predictability trades space for time
Measured, reproducible via `figures/risk_field/lobes.py` (also fig2 in the LaTeX note):

| | ballistic (heading held) | diffusive (dancing) |
|---|---|---|
| covariance | `∝ t²`, anisotropic | `∝ t`, isotropic |
| widest extent | 2.13 × 0.55 | 1.08 × 1.08 |
| `t_expire` | **8.9** | **4.5** |

Same mean speed, same `ε`; only the velocity autocorrelation differs. **Predictability extends a
hazard's reach in time and narrows it in space.** A readable hazard constrains far ahead along a
thin line — sidestep it. An unreadable one constrains hard nearby then stops mattering — wait it
out. Opposite evasion strategies, selected by the field with no extra encoding, and visible only
in the lift. The interpolating process is Ornstein–Uhlenbeck velocity with one decorrelation time
`τ`; "dancing vs consistent" is `t/τ`.

### Merging: `ε` decides how many obstacles there are
Measured, `figures/risk_field/merge.py` (also fig3 in the LaTeX note). Single-lobe peak 0.526; saddle between two lobes 0.365.

- `ε > 0.365` → two obstacles, three classes (left / between / right)
- `ε < 0.365` → one peanut, one way around, **no class decision to make**

The topology changes exactly when `ε` crosses a **critical point** of the field: maxima create
obstacles, saddles merge them. Convex hull of a merged cluster restores convexity at the cost of
forbidding the gap — which is the desired behaviour, since threading a closing gap between two
fast hazards is the failure mode object-based methods have.

Pipeline: cluster on lobe overlap → convex hull each cluster → one supporting half-space per
(segment, cluster). That is exactly paper 1's B4 machinery; the new work sits upstream of the
solver.

## 6. Do not claim

- Any of the four ingredients in §3.
- `ε`-controls-topology, or merge trees over `ε`. Standard construction, and the literature moved
  past it in 2015.
- Real-time performance until measured. SPOT (arXiv:2602.01189) capped its horizon at **2 seconds**
  with time binned at 0.2 s, while doing less than this. That constraint will apply here too.

## 7. Verification status

- **Read in full from PDF:** Bhattacharya & Ghrist 2015.
- **Abstract / publisher page / search summary only:** Bayesian Occupancy Filter, T-RO 2024,
  TRUST-Planner, Pokorny et al., Trautman & Krause, vineyards.
- **Derived here, unreviewed:** the lens/`t_expire` result in §5, and the ballistic-vs-diffusive
  and saddle numbers (reproducible from the scripts, but the *claim of novelty* is unchecked).
- **Not searched at all:** whether the density-vs-confidence threshold distinction is folklore in
  the estimation literature. Check before treating it as a contribution.
