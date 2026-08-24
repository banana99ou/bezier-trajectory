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
use crate::minnorm;

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
    pub alpha: f64,
    pub beta: f64,
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
    /// `b = n . y* + r_m`.
    pub offset: f64,
    /// The projection of the segment centroid onto the convexified centreline.
    pub y_star: Vec<f64>,
    /// `||c - y*||`. Strictly positive, but NOT necessarily greater than `r_m`:
    /// a centroid inside the keep-out zone gives a valid plane with a negative
    /// margin, which is a constraint to satisfy, not a reason to emit nothing.
    pub dist: f64,
    /// Orthonormal basis of the active face's direction space. `P_F = sum b b^T`
    /// and the rotation term is `(I - P_F)(I - n n^T) d_k / dist`. Empty when the
    /// projection lands on a vertex, which is the frozen-nearest-point case.
    pub face: Vec<Vec<f64>>,
    pub rho: f64,
    pub r_nearest: f64,
    pub sound: bool,
}

/// How finely the obstacle parameter is scanned when locating the clipped
/// stretch. Cost is linear and tiny; the padding below makes correctness
/// independent of this number, so it is a speed/tightness knob and nothing else.
const PARAM_SAMPLES: usize = 64;

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
    let s = 0.5 * (lo + hi);
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

    // Statement (8) is a one-line switch, not a rewrite: clamping rho from BELOW
    // by `reach` makes statement (7) hold unconditionally and the construction
    // sound by construction, at the cost of conservatism exactly where the row
    // binds. Which of the two is better is an open experimental question, so both
    // are reachable and both are measurable.
    let rho = match (sound_clip, apply_reach_cap) {
        (true, true) => r_nearest.clamp(reach, r_max),
        (true, false) => r_nearest.max(reach),
        (false, true) => r_nearest.min(r_max),
        (false, false) => r_nearest,
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
    let h = 1.0 / PARAM_SAMPLES as f64;
    let threshold = rho + r_m + 0.5 * obs_data.speed_bound(obs) * h;
    let mut alpha = f64::INFINITY;
    let mut beta = f64::NEG_INFINITY;
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
            alpha = alpha.min(s);
            beta = beta.max(s);
        }
    }
    if !alpha.is_finite() {
        // The nearest point is always in the band, so this is unreachable in
        // exact arithmetic. Fall back to a cell around it rather than trusting
        // that claim numerically.
        alpha = s_star;
        beta = s_star;
    }
    let alpha = (alpha - h).max(0.0);
    let beta = (beta + h).min(1.0);

    // Statements (2) and (3): hulling and inflating commute, so only the
    // centreline is convexified — and subdividing the obstacle's own Bezier makes
    // that convexification EXACT. No sampling, no chord error to bound.
    let n_ctrl = obs_data.n_ctrl;
    let sub_matrix = de_casteljau::subdivide_between(obs_data.degree(), alpha, beta);
    let g_tilde = bezier::matmul(&sub_matrix, n_ctrl, n_ctrl, obs_data.ctrl_of(obs), dim);

    Some(ClipBand {
        alpha,
        beta,
        g_tilde,
        rho,
        r_nearest,
        sound,
    })
}

/// The supporting half-space of PAPER_1 statement (4), from the projection of the
/// centroid onto the convexified centreline.
///
/// **The plane is valid whether or not the centroid is inside `H`.** Statement
/// (4)'s proof needs only that `y*` is the projection of `c` onto `G`: the
/// variational inequality then puts all of `H` on one side, whoever `c` is.
/// PAPER_1 writes "assume `||c - y*|| > r_m`", but that assumption is about the
/// row being SATISFIED at the reference, not about the half-space existing. When
/// `||c - y*|| < r_m` the centroid is inside the keep-out zone, the row's margin
/// is negative, and driving it positive is exactly the elastic penalty's job.
///
/// Refusing to emit a row there was a real defect, measured: on `diverse`
/// N8_seg8 it left the one penetrating (segment, obstacle) pair with no
/// constraint at all, and the certificate — which sums row violations — reported
/// 4.6e-13 for a trajectory that penetrated by 0.219. A check that cannot fail.
///
/// Returns `None` only when the normal is genuinely undefined: the centroid sits
/// ON the convexified centreline, so there is no direction to push it. The
/// caller counts that, and the certificate refuses rather than passing.
pub fn support_plane(centroid: &[f64], band: &ClipBand, r_m: f64, dim: usize) -> Option<ClipGeometry> {
    let n_ctrl = band.g_tilde.len() / dim;

    // Statement (4): the plane must come from the projection onto the HULL. The
    // tube's nearest surface point satisfies the variational inequality only when
    // the tube is convex; for a curved one it leaves part of the clipped piece on
    // the free side. Translate so the projection is a minimum-norm point.
    let translated: Vec<f64> = (0..n_ctrl * dim)
        .map(|i| band.g_tilde[i] - centroid[i % dim])
        .collect();
    let mnp = minnorm::min_norm_point(&translated, n_ctrl, dim);

    let dist = mnp.point.iter().map(|v| v * v).sum::<f64>().sqrt();
    // Scale-relative, because `dist` is a length in the lifted space and a fixed
    // epsilon would be a different test in a different scene.
    let scale = band
        .g_tilde
        .iter()
        .fold(1.0f64, |acc, v| acc.max(v.abs()));
    if dist <= 1e-12 * scale {
        return None;
    }

    // n points from y* toward c, so `n . z >= b` is the free side.
    let normal: Vec<f64> = mnp.point.iter().map(|v| -v / dist).collect();
    let y_star: Vec<f64> = (0..dim).map(|d| centroid[d] + mnp.point[d]).collect();
    let offset = normal
        .iter()
        .zip(y_star.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + r_m;

    Some(ClipGeometry {
        face: minnorm::face_basis(&translated, dim, &mnp.active),
        normal,
        offset,
        y_star,
        dist,
        rho: band.rho,
        r_nearest: band.r_nearest,
        sound: band.sound,
    })
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
/// The two non-`Plane` outcomes look the same to a row count and are opposites in
/// meaning. `OutOfReach` means no row is NEEDED: nothing within the trust region
/// can touch this obstacle, so silence is correct. `NoPlane` means no row is
/// POSSIBLE: the centroid is already inside the convexified keep-out zone, so
/// there is no separating half-space to hand the solver. Reporting the second as
/// a satisfied constraint is how a violation becomes a certificate of 0.0.
pub enum ClipOutcome {
    OutOfReach,
    NoPlane,
    Plane(Box<ClipGeometry>),
}

/// Keep-out reading of the clip: band, then supporting plane, with the reach cap.
pub fn clip_geometry(
    centroid: &[f64],
    seg_radius: f64,
    trust_radius: f64,
    obs_data: &SpacetimeObstacleData<'_>,
    obs: usize,
    sound_clip: bool,
) -> ClipOutcome {
    let Some(band) = clip_band(
        centroid,
        seg_radius,
        trust_radius,
        obs_data,
        obs,
        sound_clip,
        true,
    ) else {
        return ClipOutcome::OutOfReach;
    };
    match support_plane(centroid, &band, obs_data.radii[obs], obs_data.dim()) {
        Some(g) => ClipOutcome::Plane(Box::new(g)),
        None => ClipOutcome::NoPlane,
    }
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
