//! The obstacle, and the local convexification of its keep-out zone.
//!
//! This module is PAPER_1 §"Formulation — rigorous statement", statements (1)
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
/// This is the shared object of PAPER_1 statements (1)-(3), and the literal
/// content of "the shadow gets the identical treatment as the tube": the keep-out
/// half-space and the occlusion half-space are two readings of this one band.
pub struct ClipBand {
    /// Obstacle-parameter interval, already widened to contain the true band.
    /// This is the FUSED span, `[min alpha_j, max beta_j]` over `intervals` — the
    /// occlusion prism wants one span and nothing else.
    pub alpha: f64,
    pub beta: f64,
    /// The MAXIMAL subintervals of the clipped stretch, in increasing order.
    ///
    /// **Fusing these into `[alpha, beta]` is what made more than one wall
    /// structurally impossible.** A centreline that leaves the ball and re-enters
    /// puts two separate lumps of tube inside the same ball, and the gap between
    /// them is a corridor the trajectory is entitled to use; hulling the fused
    /// span swallows it. The keep-out builder reads this field. `band_over_times`
    /// fills it with the single span it was asked for, which is correct: a time
    /// window is one interval by construction.
    pub intervals: Vec<(f64, f64)>,
    /// Control points of the sub-Bezier over `[alpha, beta]`, `n_ctrl * dim`
    /// row-major. `G = conv{g_tilde}` and `H = G (+) B(0, r_m)`.
    pub g_tilde: Vec<f64>,
    /// The clip radius actually used.
    pub rho: f64,
    /// Distance from the segment centroid to the nearest centreline point.
    pub r_nearest: f64,
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
fn polish_nearest(
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    c: &[f64],
    s0: f64,
    lo: f64,
    hi: f64,
) -> f64 {
    let dim = obs_data.dim();
    let (d1, n1) = deriv_ctrl(obs_data.ctrl_of(obs), obs_data.n_ctrl, dim);
    let (d2, n2) = deriv_ctrl(&d1, n1, dim);
    let dist_sq = |s: f64| -> f64 {
        let p = obs_data.lifted_at(obs, s);
        p.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
    };
    let mut s = s0;
    let mut best = dist_sq(s);
    for _ in 0..16 {
        let p = obs_data.lifted_at(obs, s);
        let v = bezier::evaluate(&d1, n1, dim, s);
        let a = bezier::evaluate(&d2, n2, dim, s);
        let diff: Vec<f64> = (0..dim).map(|k| c[k] - p[k]).collect();
        let g: f64 = (0..dim).map(|k| v[k] * diff[k]).sum();
        let gp: f64 = (0..dim).map(|k| a[k] * diff[k] - v[k] * v[k]).sum();
        if !gp.is_finite() || gp.abs() < 1e-300 {
            break;
        }
        let next = (s - g / gp).clamp(lo, hi);
        let val = dist_sq(next);
        if !(val <= best) {
            break;
        }
        if (next - s).abs() <= 1e-16 {
            s = next;
            break;
        }
        s = next;
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
fn nearest_param(obs_data: &SpacetimeObstacleData<'_>, obs: usize, c: &[f64]) -> (f64, f64) {
    let dist_sq = |s: f64| -> f64 {
        let p = obs_data.lifted_at(obs, s);
        p.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
    };

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
    let s = polish_nearest(obs_data, obs, c, 0.5 * (lo + hi), lo, hi);
    (s, dist_sq(s).sqrt())
}

/// Clip the obstacle centreline to the ball around the segment centroid and
/// convexify the result exactly.
///
/// `apply_reach_cap` selects the `R_max` cap of PAPER_1 statement (7), row 1.
///
/// **The cap is correct for keep-out and wrong for occlusion.** For keep-out,
/// `R > R_max` means nothing in the trust region can touch this obstacle, so
/// emitting no row is right. An occluder, though, blocks a sight line from
/// wherever it happens to be: a body far from the trajectory but sitting between
/// it and the station casts its shadow onto the trajectory all the same. Capping
/// there would silently drop exactly the occluders that matter. PAPER_1
/// §"The shadow in the lifted space" says "clip to the same local ball" without
/// distinguishing the two, so the occlusion caller passes `false` and the ball
/// stays tangent to the centreline however far away that is.
///
/// Returns `None` only when the cap applies and the obstacle is out of reach.
pub fn clip_band(
    centroid: &[f64],
    seg_radius: f64,
    trust_radius: f64,
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    sound_clip: bool,
    apply_reach_cap: bool,
    radius_floor: f64,
) -> Option<ClipBand> {
    let dim = obs_data.dim();
    let r_m = obs_data.radii[obs];

    // Statement (6): an l-inf trust box of half-width Delta moves any segment
    // control point by at most Delta * sqrt(d+1) in Euclidean norm, because the
    // De Casteljau matrix is row-stochastic. `reach` is the radius that
    // statement (7) requires the clip to cover.
    let reach = seg_radius + trust_radius * (dim as f64).sqrt();
    let r_max = r_m + reach;

    let (s_star, r_nearest) = nearest_param(obs_data, obs, centroid);

    if apply_reach_cap && r_nearest > r_max {
        return None;
    }

    // TWO floors, and they are different numbers. `radius_floor` is the
    // construction's own: the keep-out caller passes `r_m`, so the clip ball is
    // never smaller than the obstacle. It binds only when the centroid is inside
    // the keep-out zone (`r_max >= r_m` always, so `min(r_nearest, r_max) < r_m`
    // iff `r_nearest < r_m`), and what it buys there is that the clipped volume
    // stays a piece of the OBSTACLE instead of a ball floating inside it — an
    // unfloored ball of radius 0.35 inside a tube of radius 0.9 puts the wall on
    // the ball and reports a 0.55-deep penetration as margin exactly 0.00.
    //
    // `sound_clip` raises the floor to `reach`, a different and LARGER condition
    // addressing a different failure: tube material the next iterate can reach
    // but that lies outside the ball, constrained by nothing. Applying one does
    // not give you the other. Which is used is an open experimental question, so
    // both are reachable and both are measurable; `sound` counts the pairs where
    // the larger condition fails.
    let floor = if sound_clip { reach.max(radius_floor) } else { radius_floor };
    let rho = if apply_reach_cap {
        r_nearest.clamp(floor.min(r_max), r_max)
    } else {
        r_nearest.max(floor)
    };
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
    // The runs of qualifying samples are kept SEPARATE here; `alpha`/`beta` below
    // fuse them, and only the occlusion prism reads those. Padding each run by
    // `h` and merging runs that then overlap errs toward FEWER, LARGER intervals,
    // which is the safe direction: over-splitting only adds walls that each still
    // contain their own lump, while under-splitting hides a corridor.
    let h = 1.0 / PARAM_SAMPLES as f64;
    let threshold = rho + r_m + 0.5 * obs_data.speed_bound(obs) * h;
    let mut runs: Vec<(f64, f64)> = Vec::new();
    let mut open: Option<(f64, f64)> = None;
    for i in 0..=PARAM_SAMPLES {
        let s = i as f64 / PARAM_SAMPLES as f64;
        let p = obs_data.lifted_at(obs, s);
        let d: f64 = p
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
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
    let alpha = intervals[0].0;
    let beta = intervals[intervals.len() - 1].1;

    // Statements (2) and (3): hulling and inflating commute, so only the
    // centreline is convexified — and subdividing the obstacle's own Bezier makes
    // that convexification EXACT. No sampling, no chord error to bound.
    let n_ctrl = obs_data.n_ctrl;
    let sub_matrix = de_casteljau::subdivide_between(obs_data.degree(), alpha, beta);
    let g_tilde = bezier::matmul(&sub_matrix, n_ctrl, n_ctrl, obs_data.ctrl_of(obs), dim);

    Some(ClipBand {
        alpha,
        beta,
        intervals,
        g_tilde,
        rho,
        r_nearest,
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

/// Intersections of two circles given in plane coordinates. Empty when they miss,
/// are nested, or are concentric.
fn circle_circle_2d(p1: [f64; 2], r1: f64, p2: [f64; 2], r2: f64, tol: f64) -> Vec<[f64; 2]> {
    let d = [p2[0] - p1[0], p2[1] - p1[1]];
    let dd = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if dd < tol || dd > r1 + r2 + tol || dd < (r1 - r2).abs() - tol {
        return Vec::new();
    }
    let aa = (dd * dd + r1 * r1 - r2 * r2) / (2.0 * dd);
    let hh = (r1 * r1 - aa * aa).max(0.0).sqrt();
    let base = [p1[0] + aa * d[0] / dd, p1[1] + aa * d[1] / dd];
    let perp = [-d[1] / dd, d[0] / dd];
    vec![
        [base[0] + hh * perp[0], base[1] + hh * perp[1]],
        [base[0] - hh * perp[0], base[1] - hh * perp[1]],
    ]
}

/// EXACT test: do three balls in `R^dim` share at least one point?
///
/// Any candidate may be replaced by its orthogonal projection onto the affine
/// hull of the three centres without increasing any of the three distances, so
/// the question is at most 2-dimensional. Inside a plane containing all three
/// centres, an intersection of discs is either a whole disc — then that disc's
/// centre works — or is bounded by arcs whose corners are circle-circle
/// intersection points. Centres plus pairwise intersections is therefore a
/// complete candidate set. No optimiser, no sampling.
pub fn balls_have_common_point(centres: [&[f64]; 3], radii: [f64; 3], tol: f64) -> bool {
    let dim = centres[0].len();
    if radii.iter().any(|r| *r < -tol) {
        return false;
    }
    // Two orthonormal directions spanning a plane through the three centres. The
    // identity columns are appended so a collinear or coincident triple still
    // yields a genuine 2-plane rather than a degenerate coordinate system.
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut seeds: Vec<Vec<f64>> = Vec::new();
    for i in 1..3 {
        seeds.push((0..dim).map(|k| centres[i][k] - centres[0][k]).collect());
    }
    for k in 0..dim {
        let mut e = vec![0.0; dim];
        e[k] = 1.0;
        seeds.push(e);
    }
    for seed in seeds {
        if basis.len() == 2 {
            break;
        }
        let mut w = seed;
        for b in basis.iter() {
            let c: f64 = (0..dim).map(|k| w[k] * b[k]).sum();
            for k in 0..dim {
                w[k] -= c * b[k];
            }
        }
        let nw = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nw > 1e-10 {
            for v in w.iter_mut() {
                *v /= nw;
            }
            basis.push(w);
        }
    }
    while basis.len() < 2 {
        basis.push(vec![0.0; dim]); // dim == 1; the second coordinate is inert
    }

    let project = |p: &[f64]| -> [f64; 2] {
        let mut out = [0.0f64; 2];
        for (j, b) in basis.iter().enumerate() {
            out[j] = (0..dim).map(|k| (p[k] - centres[0][k]) * b[k]).sum();
        }
        out
    };
    let pc: Vec<[f64; 2]> = centres.iter().map(|c| project(c)).collect();

    let mut cands: Vec<[f64; 2]> = pc.clone();
    for i in 0..3 {
        for j in (i + 1)..3 {
            cands.extend(circle_circle_2d(pc[i], radii[i], pc[j], radii[j], tol));
        }
    }
    cands.iter().any(|z| {
        (0..3).all(|k| {
            let dx = z[0] - pc[k][0];
            let dy = z[1] - pc[k][1];
            (dx * dx + dy * dy).sqrt() <= radii[k] + tol
        })
    })
}

/// How finely a pair of parameter intervals is scanned when deciding whether
/// their lumps touch. A missed merge over-splits, which is the safe direction.
const MERGE_GRID: usize = 17;

/// Do the lumps carried by two parameter intervals share a point?
///
/// `z` is shared iff `Ball(gamma(t), r_m) ∩ Ball(gamma(u), r_m) ∩ Ball(c, rho)`
/// is nonempty for some `(t, u)`. Each pair is decided EXACTLY by
/// [`balls_have_common_point`]; the search over pairs is a grid plus a refined
/// closest approach, so the only approximation is which pairs get tested.
///
/// **Missing a merge over-splits a component, which is safe** — each wall still
/// contains its own lump and imposing both keeps the segment out of the union.
/// Merging wrongly is not safe: it fuses two genuine components into one lump
/// that spans the corridor between them.
fn lumps_touch(
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    r_m: f64,
    centroid: &[f64],
    rho: f64,
    iv_i: (f64, f64),
    iv_j: (f64, f64),
) -> bool {
    let at = |s: f64| obs_data.lifted_at(obs, s);
    let gap = |t: f64, u: f64| -> f64 {
        let (a, b) = (at(t), at(u));
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
    };
    let grid = |iv: (f64, f64), i: usize| -> f64 {
        if MERGE_GRID <= 1 {
            iv.0
        } else {
            iv.0 + (iv.1 - iv.0) * (i as f64) / ((MERGE_GRID - 1) as f64)
        }
    };

    let mut best = (f64::INFINITY, iv_i.0, iv_j.0);
    for i in 0..MERGE_GRID {
        for j in 0..MERGE_GRID {
            let (t, u) = (grid(iv_i, i), grid(iv_j, j));
            let g = gap(t, u);
            if g < best.0 {
                best = (g, t, u);
            }
            if g <= 2.0 * r_m + 1e-9
                && balls_have_common_point([&at(t), &at(u), centroid], [r_m, r_m, rho], 1e-9)
            {
                return true;
            }
        }
    }
    // Refine the closest approach and retest there: the grid can straddle a
    // narrow contact.
    let (mut t0, mut u0) = (best.1, best.2);
    let mut span_t = (iv_i.1 - iv_i.0) / (MERGE_GRID.max(2) - 1) as f64;
    let mut span_u = (iv_j.1 - iv_j.0) / (MERGE_GRID.max(2) - 1) as f64;
    for _ in 0..40 {
        let mut local = (gap(t0, u0), t0, u0);
        for dt in [-span_t, 0.0, span_t] {
            for du in [-span_u, 0.0, span_u] {
                let t = (t0 + dt).clamp(iv_i.0, iv_i.1);
                let u = (u0 + du).clamp(iv_j.0, iv_j.1);
                let g = gap(t, u);
                if g < local.0 {
                    local = (g, t, u);
                }
            }
        }
        t0 = local.1;
        u0 = local.2;
        span_t *= 0.5;
        span_u *= 0.5;
    }
    gap(t0, u0) <= 2.0 * r_m + 1e-9
        && balls_have_common_point([&at(t0), &at(u0), centroid], [r_m, r_m, rho], 1e-9)
}

/// Group parameter intervals into the connected components of the clipped volume.
///
/// Over one interval the lump is connected: every slice
/// `Ball(gamma(tau), r_m) ∩ Ball(c, rho)` is nonempty and convex and varies
/// continuously, so the projection of `c` onto the slices traces a curve meeting
/// all of them. Hence every component is a union of WHOLE intervals, and the
/// search reduces to union-find over pairs.
fn group_components(
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    r_m: f64,
    centroid: &[f64],
    rho: f64,
    intervals: &[(f64, f64)],
) -> Vec<Vec<(f64, f64)>> {
    let n = intervals.len();
    if n == 1 {
        return vec![vec![intervals[0]]];
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri == rj {
                continue;
            }
            if lumps_touch(obs_data, obs, r_m, centroid, rho, intervals[i], intervals[j]) {
                parent[ri] = rj;
            }
        }
    }
    let mut groups: Vec<(usize, Vec<(f64, f64)>)> = Vec::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        match groups.iter_mut().find(|(r, _)| *r == root) {
            Some((_, ivs)) => ivs.push(intervals[i]),
            None => groups.push((root, vec![intervals[i]])),
        }
    }
    groups.sort_by(|a, b| a.1[0].0.partial_cmp(&b.1[0].0).unwrap());
    groups.into_iter().map(|(_, ivs)| ivs).collect()
}

/// Nearest centreline point to `c` RESTRICTED to a component's intervals.
///
/// Returns `(parameter, distance, interior)`. `interior` is false when the
/// minimum sits on an interval endpoint, where the nearest point cannot slide
/// with the centroid — the frozen case for the linearisation below.
fn nearest_param_in(
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    c: &[f64],
    intervals: &[(f64, f64)],
) -> (f64, f64, bool) {
    let dist_sq = |s: f64| -> f64 {
        let p = obs_data.lifted_at(obs, s);
        p.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
    };
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
    let s = polish_nearest(obs_data, obs, c, 0.5 * (lo + hi), iv.0, iv.1);
    let edge = 1e-9 + 2.0 * h;
    let interior = (s - iv.0) > edge && (iv.1 - s) > edge;
    (s, dist_sq(s).sqrt(), interior)
}

/// Unit tangent of the lifted centreline, from the hodograph. Zero-length when
/// the curve is stationary there, which the caller reads as "no face".
fn unit_tangent(obs_data: &SpacetimeObstacleData<'_>, obs: usize, s: f64) -> Option<Vec<f64>> {
    let dim = obs_data.dim();
    let deg = obs_data.degree();
    if deg == 0 {
        return None;
    }
    let ctrl = obs_data.ctrl_of(obs);
    let mut hodo = vec![0.0; deg * dim];
    for l in 0..deg {
        for k in 0..dim {
            hodo[l * dim + k] = deg as f64 * (ctrl[(l + 1) * dim + k] - ctrl[l * dim + k]);
        }
    }
    let mut v = bezier::evaluate(&hodo, deg, dim, s.clamp(0.0, 1.0));
    let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if nv < 1e-12 {
        return None;
    }
    for x in v.iter_mut() {
        *x /= nv;
    }
    Some(v)
}

/// How many De Casteljau pieces each parameter interval is cut into when bounding
/// the component's support. The bound is RIGOROUS at any value and tightens as
/// this grows, so it is a tightness knob and never a correctness one.
const SUPPORT_PIECES: usize = 32;

/// A RIGOROUS upper bound on `max n.z` over one component of the clipped volume.
///
/// Each interval is De Casteljau-subdivided into pieces. On a piece the
/// centreline lies inside `conv(sub)`, which lies inside `Ball(m, rad)` with `m`
/// the mean of the subdivided control points. So the part of the component
/// carried by that piece is contained in ALL THREE of
///
/// ```text
///   conv(sub) (+) Ball(0, r_m)        support  max_l n.sub_l + r_m
///   Ball(m, rad + r_m) ∩ Ball(c, rho) support  lens_support_point(..)  [exact]
///   Ball(c, rho)                      support  n.c + rho
/// ```
///
/// and the smallest of the three is taken; a piece whose containing ball misses
/// the clip ball carries no material at all and is skipped. The middle term is
/// what makes this CONVERGE: as the pieces shrink, `rad -> 0` and the lens
/// collapses onto the true slice.
///
/// **This is the emitted offset, not a diagnostic.** Taking a sampled maximum
/// instead would risk stepping over a narrow peak, and an under-estimated offset
/// puts part of the keep-out zone on the ALLOWED side — the one way this
/// construction could be silently unsound. A rigorous ceiling cannot do that, and
/// what it costs is conservatism, which sequential convex programming pays for in
/// step size rather than in correctness.
fn component_support(
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    r_m: f64,
    centroid: &[f64],
    rho: f64,
    intervals: &[(f64, f64)],
    n: &[f64],
) -> f64 {
    let dim = obs_data.dim();
    let n_ctrl = obs_data.n_ctrl;
    let ball_cap: f64 = (0..dim).map(|k| n[k] * centroid[k]).sum::<f64>() + rho;
    let mut best = f64::NEG_INFINITY;
    for iv in intervals {
        for i in 0..SUPPORT_PIECES {
            let lo = iv.0 + (iv.1 - iv.0) * (i as f64) / (SUPPORT_PIECES as f64);
            let hi = iv.0 + (iv.1 - iv.0) * ((i + 1) as f64) / (SUPPORT_PIECES as f64);
            let sub_matrix = de_casteljau::subdivide_between(obs_data.degree(), lo, hi);
            let sub = bezier::matmul(&sub_matrix, n_ctrl, n_ctrl, obs_data.ctrl_of(obs), dim);

            let mut hull_cap = f64::NEG_INFINITY;
            let mut m = vec![0.0; dim];
            for l in 0..n_ctrl {
                let v: f64 = (0..dim).map(|k| n[k] * sub[l * dim + k]).sum();
                hull_cap = hull_cap.max(v);
                for k in 0..dim {
                    m[k] += sub[l * dim + k] / n_ctrl as f64;
                }
            }
            hull_cap += r_m;

            let mut rad: f64 = 0.0;
            for l in 0..n_ctrl {
                let d2: f64 = (0..dim)
                    .map(|k| (sub[l * dim + k] - m[k]) * (sub[l * dim + k] - m[k]))
                    .sum();
                rad = rad.max(d2.sqrt());
            }
            // Empty lens: this piece's tube carries no material inside the clip
            // ball, so it contributes nothing rather than a loose bound.
            let Some(p) = lens_support_point(&m, rad + r_m, centroid, rho, n) else {
                continue;
            };
            let lens_cap: f64 = (0..dim).map(|k| n[k] * p[k]).sum();
            best = best.max(hull_cap.min(ball_cap).min(lens_cap));
        }
    }
    // Every piece came back empty. `L ⊆ Ball(c, rho)` always, so its support is a
    // sound fallback; it is looser, never wrong.
    if best.is_finite() {
        best
    } else {
        ball_cap
    }
}

/// A spatial ball containing the obstacle body over the clipped stretch.
///
/// The spatial part of the curve lies in the hull of the spatial parts of the
/// subdivided control points, so a ball centred on their average with radius
/// `max_l ||G_l - centre|| + r_m` contains the body at every instant in the
/// stretch. Larger than the truth, never smaller.
///
/// **At degree 1 this is exactly the formula it replaces.** The two subdivided
/// control points sit at the obstacle's positions at the ends of the stretch,
/// their average is the position at the midpoint, and the max distance is
/// `|v| * (stretch length) / 2` — the old `occluder_piece_body` verbatim. A
/// divergence at degree 1 means the subdivision is wrong.
pub fn spatial_containing_ball(band: &ClipBand, spatial_dim: usize, r_m: f64) -> (Vec<f64>, f64) {
    let dim = spatial_dim + 1;
    let n_ctrl = band.g_tilde.len() / dim;
    let mut center = vec![0.0; spatial_dim];
    for l in 0..n_ctrl {
        for d in 0..spatial_dim {
            center[d] += band.g_tilde[l * dim + d];
        }
    }
    for value in center.iter_mut() {
        *value /= n_ctrl as f64;
    }
    let mut radius: f64 = 0.0;
    for l in 0..n_ctrl {
        let mut acc = 0.0;
        for d in 0..spatial_dim {
            let delta = band.g_tilde[l * dim + d] - center[d];
            acc += delta * delta;
        }
        radius = radius.max(acc.sqrt());
    }
    (center, radius + r_m)
}

/// The obstacle's material over `[t_lo, t_hi]`, exactly convexified.
///
/// **This is the occlusion band, and it is deliberately NOT the clip ball.**
///
/// PAPER_1 §"The shadow in the lifted space" says the shadow gets "the identical
/// treatment as the tube: clip to the same local ball". Implementing that
/// literally is wrong, and the reason is not a detail. The clip ball localizes by
/// DISTANCE TO THE SEGMENT CENTROID, which is exactly right for keep-out — only
/// tube material the segment can reach is able to hit it — and exactly wrong for
/// occlusion, because a body blocks a sight line from wherever it happens to be.
/// An occluder far from the trajectory and squarely between it and the station
/// blocks the link completely. Localizing by proximity shrinks the body until it
/// no longer CONTAINS the true occluder, and containment is the only reason the
/// half-space certifies anything. Measured on `station_fence`: the ball-clipped
/// body left the run losing line of sight with an occlusion certificate of 1.96
/// even at elastic weight 1e5, where the coexisting-material body certifies.
///
/// What localizes an occluder is therefore TIME, not distance: it blocks only
/// while it exists. That is not privileging the time axis over the spatial ones,
/// it is the definition of occlusion. The gate is load-bearing because an
/// occlusion row's normal is SPATIAL with a zero time coefficient by design (the
/// shadow extruded across the band is a prism with time-parallel walls), so the
/// row forbids its side FOR ALL TIME and a row built against an occluder that has
/// vanished would go on blocking the vehicle forever.
///
/// Returns `None` when the obstacle does not exist during `[t_lo, t_hi]`.
pub fn band_over_times(
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    t_lo: f64,
    t_hi: f64,
) -> Option<ClipBand> {
    let (obs_t0, obs_t1) = (obs_data.t_start(obs), obs_data.t_end(obs));

    // Overlap is decided in TIME, before any clamping. Clamping first is a real
    // defect and it was measured: with the obstacle live over [0, 5.2] and the
    // segment over [8.75, 10], the parameters 1.68 and 1.92 both clamp to 1.0
    // and yield a degenerate band at the obstacle's final instant — so a body
    // that vanished 3.5 s earlier still emitted a shadow row, and because that
    // row is time-blind it forbade its side forever. On `station_fence` it made
    // the pinned goal infeasible (margin -0.89 at the last control point) for a
    // trajectory whose true line-of-sight margin was +0.44 throughout.
    if t_hi < obs_t0 || t_lo > obs_t1 {
        return None;
    }

    let span = obs_t1 - obs_t0;
    // A degenerate-in-time obstacle exists at one instant only.
    let (s_lo, s_hi) = if span > 1e-15 {
        (((t_lo - obs_t0) / span), ((t_hi - obs_t0) / span))
    } else {
        (0.0, 1.0)
    };
    let alpha = s_lo.clamp(0.0, 1.0);
    let beta = s_hi.clamp(0.0, 1.0);
    if beta < alpha {
        return None;
    }

    let dim = obs_data.dim();
    let n_ctrl = obs_data.n_ctrl;
    let sub_matrix = de_casteljau::subdivide_between(obs_data.degree(), alpha, beta);
    let g_tilde = bezier::matmul(&sub_matrix, n_ctrl, n_ctrl, obs_data.ctrl_of(obs), dim);
    Some(ClipBand {
        alpha,
        beta,
        // A time window is one interval by construction.
        intervals: vec![(alpha, beta)],
        g_tilde,
        // No ball was involved, so there is no clip radius and statement (7) has
        // nothing to say here. Reported as sound because the band covers every
        // instant the segment occupies: there is no material left outside it for
        // the next iterate to reach.
        rho: f64::INFINITY,
        r_nearest: f64::INFINITY,
        sound: true,
    })
}

/// What one keep-out clip attempt produced — including what it could not produce.
///
/// The two non-`Components` readings look the same to a row count and are
/// opposites in meaning. `OutOfReach` means no row is NEEDED: nothing within the
/// trust region can touch this obstacle, so silence is correct. A `dropped`
/// component means no row is POSSIBLE: the centroid sits exactly ON the
/// centreline, so no direction exists. Reporting the second as a satisfied
/// constraint is how a violation becomes a certificate of 0.0.
pub enum ClipOutcome {
    OutOfReach,
    Components(ClipComponents),
}

/// One wall per connected component of the clipped keep-out volume.
pub struct ClipComponents {
    /// One entry per component that admitted a direction, ordered by parameter.
    pub planes: Vec<ClipGeometry>,
    /// Components in reach that admitted none. A hole, counted, never silent.
    pub dropped: usize,
}

/// Keep-out reading of the clip: **the wall is built against the clipped KOZ
/// volume itself**, one per connected component.
///
/// ```text
///   L      = K_m ∩ Ball(c, rho)          the clipped KOZ volume
///   L_j    = its connected components
///   f_j    = the centreline point of component j nearest c
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
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    sound_clip: bool,
) -> ClipOutcome {
    let dim = obs_data.dim();
    let r_m = obs_data.radii[obs];
    let Some(band) = clip_band(
        centroid,
        seg_radius,
        trust_radius,
        obs_data,
        obs,
        sound_clip,
        true,
        // The construction's own floor: the clip ball is never smaller than the
        // obstacle. Binds exactly when the centroid is inside the keep-out zone.
        r_m,
    ) else {
        return ClipOutcome::OutOfReach;
    };

    let scale = centroid.iter().fold(1.0f64, |acc, v| acc.max(v.abs()));
    let comps = group_components(obs_data, obs, r_m, centroid, band.rho, &band.intervals);

    let mut planes: Vec<ClipGeometry> = Vec::with_capacity(comps.len());
    let mut dropped = 0usize;
    for ivs in comps.iter() {
        let (s_star, r_near, interior) = nearest_param_in(obs_data, obs, centroid, ivs);
        let f = obs_data.lifted_at(obs, s_star);
        let diff: Vec<f64> = (0..dim).map(|k| centroid[k] - f[k]).collect();
        let dist = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        if dist <= 1e-12 * scale {
            // The centroid sits ON the centreline. Every direction is equally
            // valid and none is determined; count it rather than inventing one.
            dropped += 1;
            continue;
        }
        let normal: Vec<f64> = diff.iter().map(|v| v / dist).collect();
        let offset = component_support(obs_data, obs, r_m, centroid, band.rho, ivs, &normal);
        // `y*` — the point of this component nearest the centroid. It is the
        // projection onto the tube when the centroid is outside, and the centroid
        // itself when it is inside, where the distance is zero by definition.
        let y_star: Vec<f64> = if r_near > r_m {
            (0..dim).map(|k| f[k] + r_m * normal[k]).collect()
        } else {
            centroid.to_vec()
        };
        // The linearisation's face. `n` is anchored on a CURVE, not a polytope:
        // as the centroid moves, the nearest parameter slides along the tangent,
        // so to first order `df/dc` is the projection onto the unit tangent and
        // `dist` is `|c - f|`, the denominator that goes with it. At an interval
        // endpoint the nearest point cannot slide, which is the frozen case and
        // an empty face. For a straight obstacle this reproduces what the hull
        // projection produced, because the hull's active face there IS the
        // centreline direction.
        let face = match (interior, unit_tangent(obs_data, obs, s_star)) {
            (true, Some(t)) => vec![t],
            _ => Vec::new(),
        };
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

/// The rotation term of PAPER_1's linearized row:
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
