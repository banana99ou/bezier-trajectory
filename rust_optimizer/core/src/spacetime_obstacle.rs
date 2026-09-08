//! The keep-out zone, and its local convexification.
//!
//! **Everything here is driven by a `Generator`**, not by an obstacle index. A
//! generator is an obstacle's centreline, optionally read through a point light
//! source, and the two readings are the same keep-out zone at two stretch
//! factors — see `spacetime_generator`. With no station the stretch is pinned at
//! one and every formula below reduces to the plain lifted tube, bit for bit;
//! that reduction is the regression test the generalization stands on.
//!
//! This module is idea/spacetime.md §"Formulation — rigorous statement", statements (1)
//! through (8), in code. Read that section before changing anything here; every
//! constant and every branch below is traceable to a numbered statement, and a
//! change that is not is a change that breaks the paper.
//!
//! **The obstacle is not constant-velocity.** Its motion is a polynomial in time
//! of degree `N_m`, so the lifted centreline `gamma_m(tau) = (pi_m(tau), tau)` is
//! a Bezier curve, and the keep-out zone `K_m = Gamma_m (+) B(0, r_m)` is a
//! CURVED tube — not convex, so it admits no supporting half-space at all. That
//! is the problem this module exists to solve. The straight-capsule builder it
//! replaces was sound only because its tube happened to be convex.
//!
//! **Time is affine in the obstacle's own parameter.** Not a restriction added
//! here: statement (3) says the motion is a polynomial *in time*, and lifting
//! such a motion gives a time coordinate that is affine in the parameter by
//! definition. It buys `s = (tau - T0) / (T1 - T0)` in closed form, so evaluating
//! the obstacle at a given instant — which is what the certificate does 1500
//! times per run — stays a direct evaluation with no root-solve.

use crate::bezier;
use crate::de_casteljau;
use crate::spacetime_generator::Generator;

/// Obstacles as lifted Bezier control points.
///
/// `ctrl` is `(n_obs, n_ctrl, spatial_dim + 1)` row-major; the last coordinate of
/// each control point is time. All obstacles are degree-elevated to a common
/// `n_ctrl` before they get here, which is exact, so the array can stay
/// rectangular without any obstacle losing shape.
///
/// There is no `t_start` / `t_end` field on purpose. The active window is
/// `G_0[t]` to `G_last[t]`, so it is derived rather than carried, and the two can
/// never disagree.
pub struct SpacetimeObstacleData<'a> {
    pub ctrl: &'a [f64],
    pub n_ctrl: usize,
    pub radii: &'a [f64],
    pub n_obs: usize,
    pub spatial_dim: usize,
}

impl SpacetimeObstacleData<'_> {
    #[inline]
    pub fn dim(&self) -> usize {
        self.spatial_dim + 1
    }

    #[inline]
    pub fn degree(&self) -> usize {
        self.n_ctrl - 1
    }

    #[inline]
    pub fn ctrl_of(&self, obs: usize) -> &[f64] {
        let d = self.dim();
        &self.ctrl[obs * self.n_ctrl * d..(obs + 1) * self.n_ctrl * d]
    }

    /// First instant the obstacle exists — the time coordinate of `G_0`.
    #[inline]
    pub fn t_start(&self, obs: usize) -> f64 {
        let d = self.dim();
        self.ctrl_of(obs)[d - 1]
    }

    /// Last instant the obstacle exists — the time coordinate of the last control
    /// point.
    #[inline]
    pub fn t_end(&self, obs: usize) -> f64 {
        let d = self.dim();
        self.ctrl_of(obs)[(self.n_ctrl - 1) * d + (d - 1)]
    }

    /// The lifted centreline at the obstacle's own parameter.
    pub fn lifted_at(&self, obs: usize, s: f64) -> Vec<f64> {
        bezier::evaluate(self.ctrl_of(obs), self.n_ctrl, self.dim(), s.clamp(0.0, 1.0))
    }

    /// Obstacle parameter corresponding to an instant, or `None` outside the
    /// obstacle's own window. Affine, by the module's time convention.
    #[inline]
    pub fn param_at_time(&self, obs: usize, t: f64) -> Option<f64> {
        let (t0, t1) = (self.t_start(obs), self.t_end(obs));
        if t < t0 || t > t1 {
            return None;
        }
        let span = t1 - t0;
        Some(if span > 1e-15 { (t - t0) / span } else { 0.0 })
    }

    /// SPATIAL position at an instant, or `None` outside the window. This is
    /// `pi_m(tau)`, the only obstacle quantity the certificate touches.
    pub fn position_at_time(&self, obs: usize, t: f64) -> Option<Vec<f64>> {
        let s = self.param_at_time(obs, t)?;
        let mut p = self.lifted_at(obs, s);
        p.truncate(self.spatial_dim);
        Some(p)
    }

    /// Lipschitz bound on `||gamma_m'||` from the control polygon:
    /// `N * max_l ||G_{l+1} - G_l||`. Standard for a Bezier, and it is what makes
    /// the interval search in `clip_band` conservative rather than hopeful.
    pub fn speed_bound(&self, obs: usize) -> f64 {
        let d = self.dim();
        let c = self.ctrl_of(obs);
        let mut worst: f64 = 0.0;
        for l in 0..self.degree() {
            let mut acc = 0.0;
            for k in 0..d {
                let delta = c[(l + 1) * d + k] - c[l * d + k];
                acc += delta * delta;
            }
            worst = worst.max(acc.sqrt());
        }
        worst * self.degree() as f64
    }
}

/// The clipped stretch of one obstacle's centreline, exactly convexified.
///
/// This is the shared object of idea/spacetime.md statements (1)-(3), and the literal
/// content of "the shadow gets the identical treatment as the tube" — which is
/// now literal rather than aspirational: the shadow is the SAME keep-out zone
/// read at a larger stretch of the same generator, so there is one band, one
/// clip, one wall construction, and no occlusion-specific branch anywhere below.
pub struct ClipBand {
    /// The MAXIMAL subintervals of the clipped stretch, in increasing order.
    ///
    /// **Fusing these into one span is what made more than one wall structurally
    /// impossible.** A generator that leaves the ball and re-enters puts two
    /// separate lumps inside the same ball, and the gap between them is a
    /// corridor the trajectory is entitled to use; hulling the fused span
    /// swallows it.
    pub intervals: Vec<(f64, f64)>,
    /// The clip radius actually used.
    pub rho: f64,
    /// Distance from the segment centroid to the nearest point of the generator
    /// — the centreline on a plain tube, the center surface under a station.
    pub r_nearest: f64,
    /// The stretch factor at that nearest point, and the local ball radius
    /// `u_star * r_m` there. Both are `1.0` and `r_m` on a plain tube.
    pub u_star: f64,
    pub local_radius: f64,
    /// Whether statement (7) holds here: `rho >= E + Delta * sqrt(d+1)`.
    ///
    /// **False is not an error and not a caveat to mention later — it is the
    /// hole, at this segment, at this iteration.** When it is false the rows
    /// certify against `K_m ∩ B(c, rho)` only, and the next iterate can leave
    /// that ball. Counted per iteration so the hole is a number.
    pub sound: bool,
}

/// One supporting half-space of the locally convexified keep-out zone, plus
/// everything the caller needs to grade and differentiate it.
pub struct ClipGeometry {
    /// Unit outward normal `n = (c - y*) / ||c - y*||`. The free side is
    /// `n . z >= b`.
    pub normal: Vec<f64>,
    /// `b`, the SUPPORT of the component along `n`: the largest `n . z` over the
    /// clipped keep-out volume. **Not `n . y* + r_m`** — that is the offset of
    /// the retired hull construction, and the two agree only when the tube is
    /// straight. The plane rests on the component's most protruding point along
    /// `n`, not on the point nearest the centroid.
    pub offset: f64,
    /// The point of this component nearest the segment centroid. The projection
    /// onto the tube when the centroid is outside it, and the centroid itself
    /// when it is inside — where the distance is zero and no direction comes from
    /// here. Carried for the trace; the row reads `offset`.
    pub y_star: Vec<f64>,
    /// `||c - f||`, the distance from the centroid to the component's nearest
    /// CENTRELINE point — the denominator of `dn/dc`, not the clearance. Strictly
    /// positive; smaller than `r_m` exactly when the centroid is inside the
    /// keep-out zone, which gives a valid plane with a negative margin, which is
    /// a constraint to satisfy and not a reason to emit nothing.
    pub dist: f64,
    /// Orthonormal basis of the directions the anchor point can slide along.
    /// `P_F = sum b b^T` and the rotation term is `(I - P_F)(I - n n^T) d_k /
    /// dist`. One vector — the centreline's unit tangent — when the nearest
    /// parameter is interior to the component, because the anchor slides along
    /// the curve as the centroid moves. Empty at an interval endpoint, where it
    /// cannot slide: the frozen-nearest-point case.
    pub face: Vec<Vec<f64>>,
    pub rho: f64,
    pub r_nearest: f64,
    pub sound: bool,
}

/// How finely the obstacle parameter is scanned when locating the clipped
/// stretch. Cost is linear and tiny; the padding below makes correctness
/// independent of this number, so it is a speed/tightness knob and nothing else.
const PARAM_SAMPLES: usize = 64;

/// Control points of the derivative curve: `N (G_{l+1} - G_l)`, degree `N-1`.
fn deriv_ctrl(ctrl: &[f64], n_ctrl: usize, dim: usize) -> (Vec<f64>, usize) {
    let deg = n_ctrl - 1;
    if deg == 0 {
        return (vec![0.0; dim], 1);
    }
    let mut out = vec![0.0; deg * dim];
    for l in 0..deg {
        for k in 0..dim {
            out[l * dim + k] = deg as f64 * (ctrl[(l + 1) * dim + k] - ctrl[l * dim + k]);
        }
    }
    (out, deg)
}

/// Newton polish of the nearest parameter, on the stationarity condition
/// `g(s) = gamma'(s) . (c - gamma(s)) = 0`.
///
/// **The ternary search cannot do this on its own.** It compares VALUES of
/// `|c - gamma(s)|^2`, which is flat at its minimum, so the comparison drowns in
/// rounding once the bracket is about `sqrt(eps)` wide — roughly `1e-8` in the
/// parameter. That was invisible while the normal came from a hull projection,
/// which is linear algebra and exact; the normal now reads `c - gamma(s*)`
/// directly, so the parameter's error lands straight in the normal. Measured: a
/// STATIONARY obstacle, whose tube is vertical and whose normal must therefore
/// have an exactly zero time component, came back with `8.4e-9` of one.
///
/// Newton uses the derivative instead of the value, so the flat minimum costs it
/// nothing. On a straight obstacle `g` is linear and one step is exact. A step is
/// taken only when it does not increase the distance, so a runaway into a nearby
/// maximum cannot happen.
/// **On a plain tube this is the routine it replaces, unchanged.** With a
/// station the stretch is held at its current optimum, which turns the surface
/// back into a Bezier curve — `Sigma(., u)` has control points
/// `(p + u(X_l - p), T_l)` — so the same Newton runs on the stretched control
/// points. Three outer rounds re-solve `u` afterwards; each round is accepted
/// only if the true distance did not increase, so alternating cannot diverge.
fn polish_nearest(gen: &Generator<'_>, c: &[f64], s0: f64, lo: f64, hi: f64) -> f64 {
    let dim = gen.dim;
    let dist_sq = |s: f64| -> f64 { gen.dist_sq(s, c).map(|(d2, _)| d2).unwrap_or(f64::INFINITY) };
    let rounds = if gen.casts_shadow() { 3 } else { 1 };
    let mut s = s0;
    let mut best = dist_sq(s);
    for _ in 0..rounds {
        let u = match gen.nearest_u(s, c) {
            Ok(u) => u,
            Err(_) => break,
        };
        let ctrl = gen.stretched_ctrl(u);
        let (d1, n1) = deriv_ctrl(&ctrl, gen.n_ctrl, dim);
        let (d2, n2) = deriv_ctrl(&d1, n1, dim);
        // The stationarity condition of the FIXED-stretch curve. Newton uses the
        // derivative rather than the value, so the flat minimum costs it nothing.
        let curve_dist_sq = |s: f64| -> f64 {
            let p = bezier::evaluate(&ctrl, gen.n_ctrl, dim, s.clamp(0.0, 1.0));
            p.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
        };
        let mut t = s;
        let mut curve_best = curve_dist_sq(t);
        for _ in 0..16 {
            let p = bezier::evaluate(&ctrl, gen.n_ctrl, dim, t.clamp(0.0, 1.0));
            let v = bezier::evaluate(&d1, n1, dim, t);
            let a = bezier::evaluate(&d2, n2, dim, t);
            let diff: Vec<f64> = (0..dim).map(|k| c[k] - p[k]).collect();
            let g: f64 = (0..dim).map(|k| v[k] * diff[k]).sum();
            let gp: f64 = (0..dim).map(|k| a[k] * diff[k] - v[k] * v[k]).sum();
            if !gp.is_finite() || gp.abs() < 1e-300 {
                break;
            }
            let next = (t - g / gp).clamp(lo, hi);
            let val = curve_dist_sq(next);
            if !(val <= curve_best) {
                break;
            }
            if (next - t).abs() <= 1e-16 {
                t = next;
                break;
            }
            t = next;
            curve_best = val;
        }
        // Re-solving `u` can only move the true distance down, but the Newton
        // step was taken on a frozen stretch, so check before accepting.
        let val = dist_sq(t);
        if !(val <= best) {
            break;
        }
        if (t - s).abs() <= 1e-16 {
            s = t;
            break;
        }
        s = t;
        best = val;
    }
    s
}

/// Nearest point of the lifted centreline to `c`, as an obstacle parameter.
///
/// Scan then refine. **Precision here cannot make the result unsound.** The scan
/// returns an over-estimate of the true minimum distance, `rho` is built from it,
/// and a larger `rho` gives a wider interval, a larger hull, and a plane pushed
/// further out — the conservative direction. Refinement buys tightness, not
/// validity.
fn nearest_param(gen: &Generator<'_>, c: &[f64]) -> (f64, f64) {
    let dist_sq =
        |s: f64| -> f64 { gen.dist_sq(s, c).map(|(d2, _)| d2).unwrap_or(f64::INFINITY) };

    let mut best_s = 0.0;
    let mut best = f64::INFINITY;
    for i in 0..=PARAM_SAMPLES {
        let s = i as f64 / PARAM_SAMPLES as f64;
        let v = dist_sq(s);
        if v < best {
            best = v;
            best_s = s;
        }
    }
    let h = 1.0 / PARAM_SAMPLES as f64;
    let (mut lo, mut hi) = ((best_s - h).max(0.0), (best_s + h).min(1.0));
    for _ in 0..60 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if dist_sq(m1) < dist_sq(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let s = polish_nearest(gen, c, 0.5 * (lo + hi), lo, hi);
    (s, dist_sq(s).sqrt())
}

/// Clip the generator to the ball around the segment centroid and convexify the
/// result exactly.
///
/// **One reach rule, for tubes and shadows alike.** The cap of idea/spacetime.md statement
/// (7) says: skip the pair when nothing inside the trust region can touch this
/// keep-out zone. Written on the SIGNED distance — the distance from the centroid
/// to the generator, less the local ball radius there — that is
/// `r_nearest - local_radius > reach`, and it is correct for both readings.
///
/// This is what dissolves the old special case. When the generator was the
/// obstacle's centreline the rule had to be disabled for occlusion, because an
/// occluder far from the trajectory still casts its shadow onto it and capping on
/// proximity dropped exactly the occluders that mattered. The center surface
/// REACHES the vehicle when its shadow does, so proximity to the generator is the
/// right localizer again and no caller has to opt out.
///
/// Returns `None` when the generator is out of reach.
pub fn clip_band(
    centroid: &[f64],
    seg_radius: f64,
    trust_radius: f64,
    gen: &Generator<'_>,
    sound_clip: bool,
) -> Option<ClipBand> {
    let dim = gen.dim;
    let r_m = gen.r_m;

    // Statement (6): an l-inf trust box of half-width Delta moves any segment
    // control point by at most Delta * sqrt(d+1) in Euclidean norm, because the
    // De Casteljau matrix is row-stochastic. `reach` is the radius that
    // statement (7) requires the clip to cover.
    let reach = seg_radius + trust_radius * (dim as f64).sqrt();

    let (s_star, r_nearest) = nearest_param(gen, centroid);
    // The stretch at the nearest approach, and the ball radius there. `1.0` and
    // `r_m` on a plain tube, which is what makes every line below reduce.
    let u_star = gen.nearest_u(s_star, centroid).ok()?;
    let local_radius = gen.radius(u_star);
    let r_max = local_radius + reach;

    if r_nearest > r_max {
        return None;
    }

    // TWO floors, and they are different numbers. The construction's own is the
    // LOCAL ball radius, so the clip ball is never smaller than the keep-out
    // material it is cutting. It binds only when the centroid is inside the
    // keep-out zone (`r_max >= local_radius` always, so
    // `min(r_nearest, r_max) < local_radius` iff `r_nearest < local_radius`), and
    // what it buys there is that the clipped volume stays a piece of the ZONE
    // instead of a ball floating inside it — an unfloored ball of radius 0.35
    // inside a tube of radius 0.9 puts the wall on the ball and reports a
    // 0.55-deep penetration as margin exactly 0.00.
    //
    // `sound_clip` raises the floor to `reach`, a different and LARGER condition
    // addressing a different failure: material the next iterate can reach but
    // that lies outside the ball, constrained by nothing. Applying one does not
    // give you the other. Which is used is an open experimental question, so both
    // are reachable and both are measurable; `sound` counts the pairs where the
    // larger condition fails.
    let floor = if sound_clip { reach.max(local_radius) } else { local_radius };
    let rho = r_nearest.clamp(floor.min(r_max), r_max);
    let sound = rho >= reach - 1e-12;

    // Statement (1): the clipped piece sits inside the band of centreline within
    // `rho + r_m` of the centroid. Locate that band in the obstacle parameter.
    //
    // **Sampling is sound HERE and forbidden for the hull.** Widening the
    // interval only enlarges the hull, which is conservative; sampling the
    // centreline to build the hull directly would produce a set the curve bulges
    // outside of. The padding makes the widening provable rather than hoped for:
    // with `L` a Lipschitz bound and spacing `h`, any parameter inside the true
    // band has a sample within `h/2` whose distance exceeds the band by at most
    // `L*h/2`, so testing against the padded threshold catches every one of them,
    // and widening the result by `h` covers the gap back to the parameter itself.
    //
    // The runs of qualifying samples are kept SEPARATE here -- fusing them is what
    // made more than one wall structurally impossible. Padding each run by
    // `h` and merging runs that then overlap errs toward FEWER, LARGER intervals,
    // which is the safe direction: over-splitting only adds walls that each still
    // contain their own lump, while under-splitting hides a corridor.
    let h = 1.0 / PARAM_SAMPLES as f64;
    let pad = 0.5 * gen.param_lipschitz(centroid) * h;
    let mut runs: Vec<(f64, f64)> = Vec::new();
    let mut open: Option<(f64, f64)> = None;
    for i in 0..=PARAM_SAMPLES {
        let s = i as f64 / PARAM_SAMPLES as f64;
        let Ok((d2, u)) = gen.dist_sq(s, centroid) else {
            continue;
        };
        let d = d2.sqrt();
        // The material this parameter generates comes within `rho` of the
        // centroid when `d - u*r_m <= rho`. Written with the radius on the right
        // so a pinned stretch reproduces the tube's own test exactly.
        let threshold = rho + gen.radius(u) + pad;
        if d <= threshold {
            open = Some(match open {
                None => (s, s),
                Some((a, _)) => (a, s),
            });
        } else if let Some(run) = open.take() {
            runs.push(run);
        }
    }
    if let Some(run) = open.take() {
        runs.push(run);
    }
    if runs.is_empty() {
        // The nearest point is always in the band, so this is unreachable in
        // exact arithmetic. Fall back to a cell around it rather than trusting
        // that claim numerically.
        runs.push((s_star, s_star));
    }
    let mut intervals: Vec<(f64, f64)> = Vec::new();
    for (a, b) in runs {
        let (a, b) = ((a - h).max(0.0), (b + h).min(1.0));
        match intervals.last_mut() {
            Some(last) if a <= last.1 + 1e-15 => last.1 = last.1.max(b),
            _ => intervals.push((a, b)),
        }
    }
    let _ = r_m;

    Some(ClipBand {
        intervals,
        rho,
        r_nearest,
        u_star,
        local_radius,
        sound,
    })
}

/// EXACT support point of the lens `Ball(a, r) ∩ Ball(c, big_r)` along unit `n`.
///
/// `None` when the lens is empty. Three branches and exactly one is right:
///
/// * **A** — `a + r n` lies in `Ball(c, big_r)`. It maximises `n . z` over the
///   whole of `Ball(a, r)`, a superset of the lens, and it is IN the lens, so it
///   is the global maximiser.
/// * **B** — symmetric, with `c + big_r n`.
/// * **C** — neither cap point is feasible, so the maximiser sits on the rim
///   where the two boundary spheres meet: centre `m = a + h u` with
///   `u = (c-a)/|c-a|`, `h = (D^2 + r^2 - big_r^2)/(2D)`, radius
///   `sqrt(r^2 - h^2)`, inside the hyperplane orthogonal to `u`. The maximiser
///   is `m` pushed along the component of `n` orthogonal to `u`.
///
/// No iteration and no sampling: the error is floating point only.
pub fn lens_support_point(a: &[f64], r: f64, c: &[f64], big_r: f64, n: &[f64]) -> Option<Vec<f64>> {
    let dim = a.len();
    let diff: Vec<f64> = (0..dim).map(|k| c[k] - a[k]).collect();
    let dd = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
    if dd > r + big_r + 1e-12 {
        return None;
    }
    let norm_to = |p: &[f64], q: &[f64]| -> f64 {
        (0..dim).map(|k| (p[k] - q[k]) * (p[k] - q[k])).sum::<f64>().sqrt()
    };
    let pa: Vec<f64> = (0..dim).map(|k| a[k] + r * n[k]).collect();
    if norm_to(&pa, c) <= big_r + 1e-12 {
        return Some(pa);
    }
    let pb: Vec<f64> = (0..dim).map(|k| c[k] + big_r * n[k]).collect();
    if norm_to(&pb, a) <= r + 1e-12 {
        return Some(pb);
    }
    // Rim. `dd > 0` here: `dd == 0` puts one ball inside the other, which is
    // branch A or branch B.
    let u: Vec<f64> = diff.iter().map(|v| v / dd).collect();
    let hh = (dd * dd + r * r - big_r * big_r) / (2.0 * dd);
    let rad = (r * r - hh * hh).max(0.0).sqrt();
    let m: Vec<f64> = (0..dim).map(|k| a[k] + hh * u[k]).collect();
    let nu: f64 = (0..dim).map(|k| n[k] * u[k]).sum();
    let mut w: Vec<f64> = (0..dim).map(|k| n[k] - nu * u[k]).collect();
    let nw = w.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nw < 1e-12 {
        return Some(m);
    }
    for v in w.iter_mut() {
        *v /= nw;
    }
    Some((0..dim).map(|k| m[k] + rad * w[k]).collect())
}

/// How finely each band interval is scanned for interior maxima of the distance
/// to the segment centroid. Over-splitting and under-splitting are both SOUND --
/// each sub-interval's wall is a rigorous ceiling on the support of its own
/// material, so any partition covers the clipped volume -- which makes this a
/// quality knob and never a correctness one: too coarse and two approaches share
/// one wall, too fine and sampling noise buys extra walls.
const SPLIT_SAMPLES: usize = 32;

/// Fraction of the shorter adjacent piece by which each cut is pushed INTO both
/// neighbours. Overlap is free for soundness -- material near a crest is covered
/// twice -- and it is what keeps `nearest_param_in`'s `interior` flag true: that
/// test rejects a nearest point within `width/32` of an endpoint, and a cut
/// landing on a crest can put a piece's minimum exactly there. Padding by
/// `width/16` moves the endpoint past the crest without moving the minimum (the
/// distance keeps rising on the far side of a crest), leaving the minimum at
/// least `width/16` from the end against a threshold near `width/30` -- interior
/// with room to spare, unconditionally.
const CUT_OVERLAP: f64 = 1.0 / 16.0;

/// Split each band interval at every INTERIOR local maximum of
/// `s -> |gamma(s) - c|` -- one sub-interval per local APPROACH of the obstacle
/// to the segment. Nothing is ever merged.
///
/// This replaced grouping by connected components of the clipped volume, and the
/// reason is panel B2: a bend can wrap the centroid while staying one connected
/// lump, and the single wall against that lump is non-separating -- the centroid
/// sits inside the lump's convex hull, so NO separating half-space exists there.
/// Cutting at the crest between the two approaches yields two walls that both
/// clear, at the same clip radius. Where one approach ends and the next begins
/// is a property of the distance profile alone, which is why the signature
/// carries no radius.
///
/// Soundness does not depend on where a cut lands: every point of the clipped
/// volume has its nearest centreline parameter inside the band, hence inside
/// some sub-interval, and that sub-interval's offset (`component_support`) is a
/// rigorous ceiling on the support of exactly that material. The cut position
/// only tunes conservatism, so the maxima are located by sampling plus a
/// parabolic-vertex refinement and nothing sharper: `polish_nearest`'s Newton
/// guard accepts only distance-DECREASING steps, so it cannot be reused for a
/// maximum, and a mirrored Newton would buy sub-cell precision that nothing
/// measures.
pub(crate) fn split_at_distance_maxima(
    gen: &Generator<'_>,
    c: &[f64],
    intervals: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let dist_sq =
        |s: f64| -> f64 { gen.dist_sq(s, c).map(|(d2, _)| d2).unwrap_or(f64::INFINITY) };
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(intervals.len());
    for iv in intervals {
        let w = iv.1 - iv.0;
        if !(w > 0.0) {
            out.push(*iv); // degenerate interval (or NaN); nothing to split
            continue;
        }
        let cell = w / SPLIT_SAMPLES as f64;
        let at = |i: usize| iv.0 + w * (i as f64) / (SPLIT_SAMPLES as f64);
        let d: Vec<f64> = (0..=SPLIT_SAMPLES).map(|i| dist_sq(at(i))).collect();

        // A crest is a STRICT rise followed, possibly across a plateau, by a
        // STRICT fall. Endpoints can never qualify: a profile still rising at
        // the last sample leaves `rise` set and unemitted, and one falling from
        // the first sample never sets it -- so no cut can touch an interval end
        // and no zero-width sub-interval is possible.
        let mut cuts: Vec<f64> = Vec::new();
        let mut rise: Option<usize> = None;
        for i in 1..=SPLIT_SAMPLES {
            if d[i] > d[i - 1] {
                rise = Some(i);
            } else if d[i] < d[i - 1] {
                if let Some(j) = rise.take() {
                    cuts.push(if j == i - 1 {
                        // Vertex of the parabola through the three samples
                        // around the crest, clamped to its own cell. The vertex
                        // of the SQUARED distance sits a hair off the true
                        // crest; sub-cell precision is explicitly not worth
                        // paying for here.
                        let (dm, d0, dp) = (d[j - 1], d[j], d[j + 1]);
                        let den = dm - 2.0 * d0 + dp;
                        if den.abs() < 1e-300 {
                            at(j)
                        } else {
                            at(j) + cell * (0.5 * (dm - dp) / den).clamp(-1.0, 1.0)
                        }
                    } else {
                        // Plateau crest: cut at its middle.
                        0.5 * (at(j) + at(i - 1))
                    });
                }
            }
        }
        // A clamped vertex can still land within half a cell of an interval end
        // when the crest sits in the first or last cell; a cut there would mint
        // a spurious sliver of a wall.
        cuts.retain(|s| *s > iv.0 + 0.5 * cell && *s < iv.1 - 0.5 * cell);
        if cuts.is_empty() {
            out.push(*iv);
            continue;
        }
        let mut bounds = Vec::with_capacity(cuts.len() + 2);
        bounds.push(iv.0);
        bounds.extend(cuts.iter().copied());
        bounds.push(iv.1);
        for k in 0..bounds.len() - 1 {
            let (a, b) = (bounds[k], bounds[k + 1]);
            let pad_l = if k == 0 {
                0.0
            } else {
                CUT_OVERLAP * (b - a).min(a - bounds[k - 1])
            };
            let pad_r = if k + 2 == bounds.len() {
                0.0
            } else {
                CUT_OVERLAP * (b - a).min(bounds[k + 2] - b)
            };
            out.push(((a - pad_l).max(iv.0), (b + pad_r).min(iv.1)));
        }
    }
    out
}

/// Nearest centreline point to `c` RESTRICTED to a component's intervals.
///
/// Returns `(parameter, distance, interior)`. `interior` is false when the
/// minimum sits on an interval endpoint, where the nearest point cannot slide
/// with the centroid — the frozen case for the linearisation below.
fn nearest_param_in(
    gen: &Generator<'_>,
    c: &[f64],
    intervals: &[(f64, f64)],
) -> (f64, f64, bool) {
    let dist_sq =
        |s: f64| -> f64 { gen.dist_sq(s, c).map(|(d2, _)| d2).unwrap_or(f64::INFINITY) };
    let mut best = (f64::INFINITY, 0.0f64, (0.0f64, 1.0f64));
    for iv in intervals {
        for i in 0..=PARAM_SAMPLES {
            let s = iv.0 + (iv.1 - iv.0) * (i as f64) / (PARAM_SAMPLES as f64);
            let v = dist_sq(s);
            if v < best.0 {
                best = (v, s, *iv);
            }
        }
    }
    let iv = best.2;
    let h = (iv.1 - iv.0) / PARAM_SAMPLES as f64;
    let (mut lo, mut hi) = ((best.1 - h).max(iv.0), (best.1 + h).min(iv.1));
    for _ in 0..60 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if dist_sq(m1) < dist_sq(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let s = polish_nearest(gen, c, 0.5 * (lo + hi), iv.0, iv.1);
    let edge = 1e-9 + 2.0 * h;
    let interior = (s - iv.0) > edge && (iv.1 - s) > edge;
    (s, dist_sq(s).sqrt(), interior)
}

/// How many De Casteljau pieces each parameter interval is cut into when bounding
/// the component's support. The bound is RIGOROUS at any value and tightens as
/// this grows, so it is a tightness knob and never a correctness one.
const SUPPORT_PIECES: usize = 32;

/// How many stretch cells the `u` direction is cut into. Same status as
/// `SUPPORT_PIECES`: rigorous at any value, tighter as it grows. **One cell when
/// the generator has no station**, because the stretch is pinned there and the
/// whole `u` loop collapses to the tube's single pass.
const SUPPORT_U_PIECES: usize = 8;

/// A RIGOROUS upper bound on `max n.z` over one approach's piece of the clipped
/// keep-out volume.
///
/// Each interval is De Casteljau-subdivided in the parameter and bisected in the
/// stretch. On a cell the generator lies inside `conv(sub^{u_lo} ∪ sub^{u_hi})`
/// — the subdivided control points stretched by the cell's two endpoints — and
/// that hull is EXACT, not sampled, for two reasons: the stretch map is affine in
/// the control points at fixed `u`, and it is affine in `u` at fixed control
/// point, so the cell's surface lies in the hull of the four corners' images.
/// (This is the shadow lemma in coordinates: the shadow of a convex set is
/// convex.) The cell's balls have radius at most `u_hi * r_m`, so its material is
/// contained in ALL THREE of
///
/// ```text
///   conv(sub_u) (+) Ball(0, R)        support  max_l n.sub_l + R
///   Ball(m, rad + R) ∩ Ball(c, rho)   support  lens_support_point(..)  [exact]
///   Ball(c, rho)                      support  n.c + rho
/// ```
///
/// with `R = u_hi * r_m`, and the smallest of the three is taken; a cell whose
/// containing ball misses the clip ball carries no material and is skipped. The
/// middle term is what makes this CONVERGE: as the cells shrink, `rad -> 0` and
/// the lens collapses onto the true slice.
///
/// **This is the emitted offset, not a diagnostic.** Taking a sampled maximum
/// instead would risk stepping over a narrow peak, and an under-estimated offset
/// puts part of the keep-out zone on the ALLOWED side — the one way this
/// construction could be silently unsound. A rigorous ceiling cannot do that, and
/// what it costs is conservatism, which sequential convex programming pays for in
/// step size rather than in correctness.
fn component_support(
    gen: &Generator<'_>,
    centroid: &[f64],
    rho: f64,
    intervals: &[(f64, f64)],
    n: &[f64],
) -> f64 {
    let dim = gen.dim;
    let n_ctrl = gen.n_ctrl;
    let r_m = gen.r_m;
    let ball_cap: f64 = (0..dim).map(|k| n[k] * centroid[k]).sum::<f64>() + rho;

    // The stretch cells. A pinned stretch gives the single cell [1,1], and then
    // `stretched_ctrl` returns the centreline itself and the loop below is the
    // tube's own pass, arithmetic included.
    let u_top = gen.u_ceiling_in_ball(centroid, rho, gen.speed_bound());
    if !u_top.is_finite() {
        // No usable ceiling: the station is on the body somewhere. The clip
        // ball's own support is sound, looser, never wrong.
        return ball_cap;
    }
    let u_cells: Vec<(f64, f64)> = if u_top <= 1.0 {
        vec![(1.0, 1.0)]
    } else {
        (0..SUPPORT_U_PIECES)
            .map(|j| {
                let a = 1.0 + (u_top - 1.0) * (j as f64) / (SUPPORT_U_PIECES as f64);
                let b = 1.0 + (u_top - 1.0) * ((j + 1) as f64) / (SUPPORT_U_PIECES as f64);
                (a, b)
            })
            .collect()
    };

    let mut best = f64::NEG_INFINITY;
    // Reused across all pieces: the subdivision matrix and the subdivided
    // control points. Same values every iteration as the fresh-allocation
    // form; only the allocations are hoisted out of the SUPPORT_PIECES loop.
    let mut sub_matrix: Vec<f64> = Vec::new();
    let mut sub: Vec<f64> = Vec::new();
    for iv in intervals {
        for i in 0..SUPPORT_PIECES {
            let lo = iv.0 + (iv.1 - iv.0) * (i as f64) / (SUPPORT_PIECES as f64);
            let hi = iv.0 + (iv.1 - iv.0) * ((i + 1) as f64) / (SUPPORT_PIECES as f64);
            de_casteljau::subdivide_between_into(gen.degree(), lo, hi, &mut sub_matrix);
            bezier::matmul_into(&sub_matrix, n_ctrl, n_ctrl, gen.ctrl, dim, &mut sub);

            for (u_lo, u_hi) in u_cells.iter().copied() {
                let cell_radius = gen.radius(u_hi);
                // The cell's corners: the subdivided control points at the two
                // stretch ends. Identical points when the stretch is pinned, and
                // then only one copy is taken so the mean and the max distance
                // are computed over exactly the tube's own N points.
                let stretches: &[f64] =
                    if u_lo == u_hi { &[u_lo] } else { &[u_lo, u_hi] };
                let mut corners: Vec<f64> = Vec::with_capacity(stretches.len() * n_ctrl * dim);
                for u in stretches.iter().copied() {
                    for l in 0..n_ctrl {
                        let point = gen.stretch_point(&sub[l * dim..(l + 1) * dim], u);
                        corners.extend_from_slice(&point);
                    }
                }
                let n_corner = corners.len() / dim;

                let mut hull_cap = f64::NEG_INFINITY;
                let mut m = vec![0.0; dim];
                for l in 0..n_corner {
                    let v: f64 = (0..dim).map(|k| n[k] * corners[l * dim + k]).sum();
                    hull_cap = hull_cap.max(v);
                    for k in 0..dim {
                        m[k] += corners[l * dim + k] / n_corner as f64;
                    }
                }
                hull_cap += cell_radius;

                let mut rad: f64 = 0.0;
                for l in 0..n_corner {
                    let d2: f64 = (0..dim)
                        .map(|k| (corners[l * dim + k] - m[k]) * (corners[l * dim + k] - m[k]))
                        .sum();
                    rad = rad.max(d2.sqrt());
                }
                // Empty lens: this cell carries no material inside the clip ball,
                // so it contributes nothing rather than a loose bound.
                let Some(p) = lens_support_point(&m, rad + cell_radius, centroid, rho, n)
                else {
                    continue;
                };
                let lens_cap: f64 = (0..dim).map(|k| n[k] * p[k]).sum();
                best = best.max(hull_cap.min(ball_cap).min(lens_cap));
            }
        }
    }
    let _ = r_m;
    // Every cell came back empty. `L ⊆ Ball(c, rho)` always, so its support is a
    // sound fallback; it is looser, never wrong.
    if best.is_finite() {
        best
    } else {
        ball_cap
    }
}

/// What one keep-out clip attempt produced — including what it could not produce.
///
/// The two non-`Components` readings look the same to a row count and are
/// opposites in meaning. `OutOfReach` means no row is NEEDED: nothing within the
/// trust region can touch this obstacle, so silence is correct. A `dropped`
/// approach means no row is POSSIBLE: the centroid sits exactly ON the
/// centreline, so no direction exists. Reporting the second as a satisfied
/// constraint is how a violation becomes a certificate of 0.0.
pub enum ClipOutcome {
    OutOfReach,
    Components(ClipComponents),
}

/// One wall per local approach of the obstacle to the segment: the band is cut
/// at every interior local maximum of `|gamma(s) - c|` and each sub-interval
/// yields one wall. (Until 2026-08-26 this was one wall per connected component
/// of the clipped volume, which handed a wrapping bend a single non-separating
/// wall -- panel B2.)
pub struct ClipComponents {
    /// One entry per approach that admitted a direction, ordered by parameter.
    pub planes: Vec<ClipGeometry>,
    /// Approaches in reach that admitted none. A hole, counted, never silent.
    pub dropped: usize,
}

/// Keep-out reading of the clip: **the wall is built against the clipped KOZ
/// volume itself**, one per local approach.
///
/// ```text
///   L      = K_m ∩ Ball(c, rho)          the clipped KOZ volume
///   L_j    = the material of L whose nearest centreline parameter falls in
///            sub-interval j -- the band cut at each interior local maximum
///            of |gamma(s) - c|  (split_at_distance_maxima)
///   f_j    = the centreline point of piece j nearest c
///   n_j    = (c - f_j) / |c - f_j|
///   b_j    = max over L_j of n_j . z     the SUPPORT of the component
///   wall:    n_j . z >= b_j              on every control point of the segment
/// ```
///
/// **`n` never actually depends on `y*`.** The nearest point of `K_m` to `c` is
/// `f_j + r_m (c - f_j)/|c - f_j|`, which lies ON the ray from `f_j` to `c`; the
/// clip radius is floored at `r_m`, so that point is inside the clip ball and is
/// therefore `y*`. Hence `n = (c - y*)/|c - y*| = (c - f_j)/|c - f_j|` wherever
/// the first quotient exists at all. When the centroid is inside the keep-out
/// zone, `y* = c` and the first quotient is `0/0` — but the ray is not, so the
/// direction is taken straight from `f_j`. **That is continuation of the same
/// formula, not a fallback to a different set**, and it is what replaced the old
/// projection onto the un-inflated centreline hull.
///
/// **A negative margin is a wall, not a failure.** `b` is the support of the
/// component, so the component lies on the forbidden side whichever side the
/// centroid is on. A component wrapping the centroid gives `n . c < b`, and the
/// row then reads "you are this far in, climb out along `n`", which is exactly
/// what the elastic penalty consumes. Refusing a row there was a measured defect:
/// on `diverse` N8_seg8 it left the one penetrating pair with no constraint at
/// all, and the certificate — which sums row violations — reported 4.6e-13 for a
/// trajectory that penetrated by 0.219. A check that cannot fail.
pub fn clip_geometry(
    centroid: &[f64],
    seg_radius: f64,
    trust_radius: f64,
    gen: &Generator<'_>,
    sound_clip: bool,
) -> ClipOutcome {
    let dim = gen.dim;
    // A generator that is not defined anywhere on its range cannot produce a
    // wall, and that is a HOLE, not silence. Checked before the reach test on
    // purpose: "out of reach" means no row is needed, which is the opposite
    // claim, and letting a fault fall through to it is how "cannot be certified"
    // becomes "certified 0.0".
    if gen.fault().is_some() {
        return ClipOutcome::Components(ClipComponents {
            planes: Vec::new(),
            dropped: 1,
        });
    }
    let Some(band) = clip_band(centroid, seg_radius, trust_radius, gen, sound_clip) else {
        return ClipOutcome::OutOfReach;
    };

    let scale = centroid.iter().fold(1.0f64, |acc, v| acc.max(v.abs()));
    let subs = split_at_distance_maxima(gen, centroid, &band.intervals);

    let mut planes: Vec<ClipGeometry> = Vec::with_capacity(subs.len());
    let mut dropped = 0usize;
    for iv in subs.iter() {
        // A one-interval slice, so `nearest_param_in` / `component_support`
        // keep their signatures and the loop body stays untouched.
        let ivs = std::slice::from_ref(iv);
        let (s_star, r_near, interior) = nearest_param_in(gen, centroid, ivs);
        // The generator point this approach is aimed from: the centreline point
        // on a plain tube, the surface point at its own best stretch otherwise.
        let Ok(u_star) = gen.nearest_u(s_star, centroid) else {
            // The station lies inside the obstacle: the shadow is everything
            // beyond the body and no supporting half-space exists. A hole,
            // counted, never silent.
            dropped += 1;
            continue;
        };
        let local_radius = gen.radius(u_star);
        let f = gen.point(s_star, u_star);
        let diff: Vec<f64> = (0..dim).map(|k| centroid[k] - f[k]).collect();
        let dist = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        if dist <= 1e-12 * scale {
            // The centroid sits ON the generator. Every direction is equally
            // valid and none is determined; count it rather than inventing one.
            dropped += 1;
            continue;
        }
        let normal: Vec<f64> = diff.iter().map(|v| v / dist).collect();
        let offset = component_support(gen, centroid, band.rho, ivs, &normal);
        // `y*` — the point of this component nearest the centroid. It is the
        // projection onto the zone when the centroid is outside, and the centroid
        // itself when it is inside, where the distance is zero by definition.
        let y_star: Vec<f64> = if r_near > local_radius {
            (0..dim).map(|k| f[k] + local_radius * normal[k]).collect()
        } else {
            centroid.to_vec()
        };
        // The linearisation's face. `n` is anchored on the GENERATOR, not a
        // polytope: as the centroid moves the nearest point slides along the
        // generator, so to first order `df/dc` is the projection onto its tangent
        // space and `dist` is `|c - f|`, the denominator that goes with it. At an
        // interval endpoint the parameter cannot slide, which is the frozen case;
        // at `u = 1` the stretch cannot slide either, so the surface contributes
        // one tangent instead of two. A plain tube has no stretch at all and this
        // is the single unit tangent the tube builder used.
        let u_interior = gen.casts_shadow() && u_star > 1.0 + 1e-12;
        let face = gen.tangents(s_star, u_star, interior, u_interior);
        planes.push(ClipGeometry {
            face,
            normal,
            offset,
            y_star,
            dist,
            rho: band.rho,
            r_nearest: r_near,
            sound: band.sound,
        });
    }
    ClipOutcome::Components(ClipComponents { planes, dropped })
}

/// The rotation term of idea/spacetime.md's linearized row:
/// `corr = (I - P_F)(I - n n^T) d_k / ||c - y*||`.
///
/// **This is not a new derivation — it is the old one, generalized.**
/// Differentiating `g_k = n . (Q_k - y*) - r_m` through the centroid's effect on
/// the aiming direction, the `d y*` term vanishes because `n` is orthogonal to
/// the active face, leaving the expression above. Setting `P_F = 0` reproduces
/// the capsule builder's cap-end branch and `P_F = e e^T / |e|^2` reproduces its
/// tube-body branch exactly, which is why a degree-1 obstacle gets the same rows
/// from the old code and this one.
///
/// Like its predecessor this is a first-order model and is NOT a conservative
/// restriction: a point satisfying the linearized row may still violate the exact
/// one. Soundness is unaffected because the certificate always rebuilds the exact
/// rows.
pub fn rotation_correction(geom: &ClipGeometry, d_k: &[f64]) -> Vec<f64> {
    let dim = d_k.len();
    // (I - n n^T) d_k
    let s: f64 = geom.normal.iter().zip(d_k.iter()).map(|(a, b)| a * b).sum();
    let mut v: Vec<f64> = (0..dim).map(|d| d_k[d] - s * geom.normal[d]).collect();
    // (I - P_F) applied to that
    for basis in &geom.face {
        let c: f64 = basis.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        for d in 0..dim {
            v[d] -= c * basis[d];
        }
    }
    for value in v.iter_mut() {
        *value /= geom.dist;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Degree-2 lifted obstacle whose spatial path is the parabola `y = x^2`,
    /// exactly: control points (-1,1), (0,-1), (1,1) give `x = 2s - 1` and
    /// `y = (2s-1)^2`. All times zero. Against the centroid `(0, 1, 0)` the
    /// squared distance `x^2 + (x^2 - 1)^2` has ONE interior maximum at
    /// `s = 0.5` (d = 1) between two minima at `x = +-sqrt(0.5)` (d = sqrt(3)/2),
    /// and with `r_m = 0.2`, `seg_radius = 0`, `trust = 1` the band threshold
    /// `rho + r_m + pad = 1.101` exceeds the curve's farthest point (d = 1), so
    /// `clip_band` returns the single interval [0, 1]. The component grouping
    /// this replaced produced ONE wall here; the split must produce two.
    const PARABOLA: [f64; 9] = [-1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0];
    const RADII: [f64; 1] = [0.2];
    const CENTROID: [f64; 3] = [0.0, 1.0, 0.0];

    /// The plain tube reading of the parabola: no station, stretch pinned at one.
    fn data() -> Generator<'static> {
        Generator {
            ctrl: &PARABOLA,
            n_ctrl: 3,
            dim: 3,
            r_m: RADII[0],
            station: None,
            obstacle_idx: 0,
            station_idx: None,
        }
    }

    #[test]
    fn one_wall_per_approach_where_components_gave_one() {
        let d = data();
        let ClipOutcome::Components(c) = clip_geometry(&CENTROID, 0.0, 1.0, &d, false)
        else {
            panic!("the obstacle is in reach; OutOfReach is wrong");
        };
        assert_eq!(c.dropped, 0);
        // FAILS IF the split silently stops splitting: the band is one interval,
        // so anything grouping by parameter runs or by connectivity yields 1.
        assert_eq!(c.planes.len(), 2, "one wall per local approach");
        // The two approaches sit on opposite sides of the crest in x, so the
        // normals must too.
        assert!(
            c.planes[0].normal[0] * c.planes[1].normal[0] < 0.0,
            "normals should straddle the crest: n0_x = {}, n1_x = {}",
            c.planes[0].normal[0],
            c.planes[1].normal[0]
        );
        // FAILS IF CUT_OVERLAP stops doing its job: each piece's nearest point
        // must stay interior so the rotation face survives the cut.
        assert_eq!(c.planes[0].face.len(), 1, "piece 0 lost its rotation face");
        assert_eq!(c.planes[1].face.len(), 1, "piece 1 lost its rotation face");
    }

    #[test]
    fn each_wall_contains_its_own_piece_of_the_clipped_volume() {
        let d = data();
        let band = clip_band(&CENTROID, 0.0, 1.0, &d, false).unwrap();
        let subs = split_at_distance_maxima(&d, &CENTROID, &band.intervals);
        let ClipOutcome::Components(c) = clip_geometry(&CENTROID, 0.0, 1.0, &d, false)
        else {
            panic!("in reach");
        };
        assert_eq!(subs.len(), c.planes.len(), "walls must map 1:1 onto sub-intervals");
        // Sample the keep-out material carried by sub-interval j -- centreline
        // pushed out by r_m along coordinate directions AND along the wall's own
        // normal, so the samples reach the piece's true support -- keep what lies
        // inside the clip ball, and demand wall j contains every bit of it.
        // FAILS IF the offset drops below the piece's true support along n.
        // Detection floor: the offset is a rigorous CEILING, conservative by the
        // De Casteljau slack (measured 0.15 on this fixture at 32 pieces), so an
        // offset error smaller than that slack is not unsound and is not caught.
        for (iv, pl) in subs.iter().zip(c.planes.iter()) {
            let np = [pl.normal[0], pl.normal[1], pl.normal[2]];
            let dirs: [[f64; 3]; 9] = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                np,
                [-np[0], -np[1], -np[2]],
            ];
            for k in 0..=200 {
                let s = iv.0 + (iv.1 - iv.0) * (k as f64) / 200.0;
                let g = d.centre_at(s);
                for dir in dirs.iter() {
                    let z: Vec<f64> = (0..3).map(|q| g[q] + RADII[0] * dir[q]).collect();
                    let r: f64 = (0..3)
                        .map(|q| (z[q] - CENTROID[q]) * (z[q] - CENTROID[q]))
                        .sum::<f64>()
                        .sqrt();
                    if r > band.rho + 1e-12 {
                        continue; // outside the clip ball: not claimed
                    }
                    let v: f64 = (0..3).map(|q| pl.normal[q] * z[q]).sum();
                    assert!(
                        v <= pl.offset + 1e-9,
                        "sub-lump escaped its own wall at s = {s}: n.z = {v} > b = {}",
                        pl.offset
                    );
                }
            }
        }
    }

    #[test]
    fn a_monotone_profile_is_never_cut() {
        // Centroid off to one side: the distance is strictly decreasing over the
        // whole parameter range, so there is no interior maximum and the split
        // must hand the interval back verbatim. FAILS IF sampling noise mints
        // cuts, which would quietly inflate row counts on every run.
        let d = data();
        let c = [3.0, 0.0, 0.0];
        let ivs = [(0.0, 1.0)];
        let subs = split_at_distance_maxima(&d, &c, &ivs);
        assert_eq!(subs, ivs.to_vec());
    }
}
