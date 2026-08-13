# Chat Notes - 2026-02-16 (Re-entry path optimizer w/ restricted airspace KOZ)

## Context / Motivation

You want a more exciting scenario than the current orbital-docking demo:
- **Problem**: generate a re-entry + landing path (e.g., to **Jeju** or **Goheung**) while **avoiding neighboring countries’ restricted airspace**, modeled as KOZ.
- **Core appeal**: “geolocation restriction” / inclination + downrange/crossrange constraints around Korea, with explicit geofencing.

## Proposed “minimal” scenario (as stated in chat)

- **Start A**: a point on/near an ISS-like orbit “somewhere near South Korea”.
- **End B**: a point at Jeju airstrip (Earth-fixed).
- **KOZ**: combination of simple **3D boxes** (geofenced volumes) representing other countries’ airspace.
- **Dynamics**: simplified; no bank angle / pitch / attitude.
  - Drag: simple function of drag coefficient and altitude.
  - Heating: simple function of speed and altitude.
- **Objective**: optimize for **control acceleration** and **heating** (multi-objective via weights).

## Key clarification (framing for a paper)

With “no bank/pitch” and proxy drag/heating, this is best framed as:
- **Trajectory optimization / path planning with simplified entry-physics proxies and geofencing**, not full entry guidance.

This is still publishable if the paper claims match the modeling level and the algorithm/constraints handling is solid.

## Minimal mathematical formulation (paper-friendly)

Represent trajectory as a 3D Bézier curve:
- \(r(\tau) \in \mathbb{R}^3\), \(\tau \in [0,1]\), control points \(P=[P_0,\dots,P_N]\).
- Use a fixed transfer time \(T\) to map to physical time:
  - \( \dot r(t) = \frac{1}{T} r'(\tau)\)
  - \( \ddot r(t) = \frac{1}{T^2} r''(\tau)\)

Define a “required control acceleration”:
- \(a_u(t) = \ddot r(t) - g(r(t)) - a_D(r(t), v_{\mathrm{rel}}(t))\)

Suggested simple models:
- Gravity: two-body (optionally J2), as in the current project’s direction.
- Atmosphere: \( \rho(h) = \rho_0 \exp(-h/H)\), \(h=\|r\|-R_E\).
- Drag acceleration surrogate:
  - \(a_D = -\tfrac12\,\rho(h)\,\beta^{-1}\,\|v_{\mathrm{rel}}\|\,v_{\mathrm{rel}}\), \(\beta=\frac{m}{C_D A}\).
- Optional “rotating atmosphere” realism:
  - \(v_{\mathrm{rel}} = v - \omega_E \times r\) (if working in an inertial frame).

Heating proxy (common in academic entry approximations):
- \(\dot q = k \sqrt{\rho(h)} \|v_{\mathrm{rel}}\|^3\).

Objective:
- \(J = w_u \int_0^T \|a_u(t)\|^2 dt + w_h \int_0^T \dot q(t)\,dt\)

Boundary conditions:
- Position: \(r(0)=A\), \(r(T)=B(t_f)\).
- **Important**: runway B is **Earth-fixed**, so either:
  - fix epoch \(t_0\) and \(T\), convert runway ECEF → ECI at \(t_f=t_0+T\), or
  - formulate in ECEF with appropriate pseudo forces (more complexity).

## KOZ / restricted airspace modeling (main technical difficulty)

### Core issue
A box (convex) is easy; **avoiding** a box is **nonconvex** (disjunctive: outside at least one face).
This is harder than the current “outside a sphere” constraint.

### Recommended approach (matches current project pattern)
Use **separating half-spaces updated iteratively** (SCP-style), analogous to current KOZ convexification:

For each Bézier segment \(i\):
- compute segment control points \(Q_i = A_i P\)
- compute representative point \(c_i\) (e.g., centroid of segment control points)
- for each obstacle box \(\mathcal{B}\), compute closest point \(p_i=\Pi_{\mathcal{B}}(c_i)\)
- define outward normal \(n_i = \frac{c_i - p_i}{\|c_i - p_i\|}\)
- impose (with safety margin \(s>0\)):
  - \(n_i^\top q_{i,k} \ge n_i^\top p_i + s \;\;\forall k\)

Iterate:
- build constraints from current trajectory → solve convexified subproblem → repeat until convergence.

Practical note:
- You will likely need a **trust region / proximal term** to prevent oscillations around corners and ensure stable convergence.

### Coordinate frame note (ECEF vs ECI)
Airspace volumes are naturally defined in **ECEF**. If the optimization runs in **ECI**, KOZ boxes become time-varying in ECI.
Paper should explicitly state the frame choice and transform method.

## What needs to be done for a paper (high-level checklist)

- **Problem statement**: entry-to-landing trajectory with geofencing + proxy physics.
- **Models**: atmosphere, drag, heating surrogate; define where simplifications apply.
- **Constraints**:
  - KOZ boxes (geofences) + safety margin
  - optional: altitude envelope, max dynamic pressure proxy, max g-load proxy (even if simple) to improve credibility
- **Algorithm**:
  - Bézier parameterization
  - De Casteljau segmentation
  - supporting/separating half-spaces updated iteratively (SCP)
  - trust region / step control
- **Experiments**:
  - Jeju vs Goheung targeting
  - KOZ density/complexity sweep
  - segmentation count sweep (K) and runtime/feasibility study
  - objective-weight sensitivity (w_u vs w_h)
- **Figures**:
  - 3D trajectory + map/ground-track overlay with KOZ
  - altitude/velocity/heating proxies vs time
  - constraint margins and runtime/iterations

## Difficulty / risk assessment (objective)

- **Medium difficulty** if you stick to proxies and SCP half-space separation.
- Biggest risks:
  - frame/time consistency (Earth-fixed runway + KOZ vs inertial states)
  - nonconvex obstacle avoidance convergence (requires trust region / margins / good initialization)
  - tuning objective weights so trajectories look plausible

## Timeline estimate (for this modeling level)

- Prototype + first plots: **2–3 weeks**
- Robust optimizer + clean experiment suite: **6–10 weeks**
- Writing + revisions: **2–4 weeks**

Total: **~2–3 months** for a solid submission-quality paper (assuming steady work).

## “Is it good for a paper?”

Yes, if framed honestly as **geofenced trajectory optimization with simplified entry proxies** and if the algorithmic contribution is clear:
- reliable SCP-style convexification for many box-like geofences
- reproducible experiments (Jeju/Goheung, KOZ sweeps, sensitivity studies)

Avoid over-claiming “full re-entry guidance” unless you later add lift/bank dynamics and validate against standard entry constraints.

