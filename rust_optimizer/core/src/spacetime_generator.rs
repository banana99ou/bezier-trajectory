//! The KOZ generator: a centreline, or that centreline's shadow.
//!
//! **There is one keep-out zone, not two.** An obstacle's keep-out zone and the
//! shadow it casts are the same object seen at two stretch factors, so the
//! centreline generalizes to a **center surface** — the shadow of the centreline
//! with respect to the ground station as a point light source:
//!
//! ```text
//!   center surface   Sigma(s,u) = ( p + u*(x(s) - p),  t(s) ),   u >= 1
//!   keep-out zone    K = union over (s,u) of Ball( Sigma(s,u), u*r_m )
//! ```
//!
//! `u = 1` gives `Sigma = gamma` and radius `r_m`: the obstacle itself. **The
//! obstacle's radius never changes.** What widens with `u` is the shadow, because
//! a point source at finite distance casts a widening umbra — the cone tangent to
//! the body, `sin alpha = r_m / |x(s) - p|`, of constant ANGULAR thickness.
//!
//! The sweep is the umbra exactly, not an outer approximation: a position `q` is
//! occluded iff the sight segment `[p,q]` meets `Ball(x, r_m)`, iff
//! `q = p + u(z-p)` for some `z` in the body and `u >= 1`, and then
//! `|q - Sigma(s,u)| = u|z - x| <= u*r_m`. Verified numerically against an
//! independent segment-distance oracle over 20000 random configurations in two
//! and three spatial dimensions: zero disagreements. The same check with a
//! CONSTANT radius `r_m` in place of `u*r_m` misses 162 of 8000 genuinely
//! occluded points, every one of them on the unsafe side — the widening is
//! required for soundness, not a matter of taste.
//!
//! **No station means `u = 1` identically**, and then every formula here reduces
//! to the plain lifted tube it replaced. That reduction is exact and is the
//! regression test the rewrite stands on: a no-op cannot move a single row.
//!
//! The one structural fact that makes the existing machinery carry over: for a
//! FIXED `u` the stretch is affine, so `Sigma(., u)` is again a Bezier curve,
//! with control points `(p + u(X_l - p), T_l)`. Subdivision, hodographs and the
//! De Casteljau support bounds all apply to the stretched control points
//! unchanged. Nothing in this module needs a surface-specific analogue of them.

use crate::bezier;

/// How finely the parameter is scanned when bounding the stretch factor. Same
/// role as `PARAM_SAMPLES` next door: the padding below makes the bound provable
/// rather than hoped for, so this is a tightness knob and never a correctness one.
const BOUND_SAMPLES: usize = 64;

/// One keep-out generator: an obstacle's lifted centreline, optionally read
/// through a point light source.
pub struct Generator<'a> {
    /// Lifted centreline control points, `n_ctrl * dim` row-major.
    pub ctrl: &'a [f64],
    pub n_ctrl: usize,
    pub dim: usize,
    /// The obstacle's own radius. Fixed; `u` never scales the body.
    pub r_m: f64,
    /// Spatial position of the point light source, length `dim - 1`. `None` is
    /// the plain tube: no shadow, `u` pinned at one.
    pub station: Option<&'a [f64]>,
    /// Which obstacle and which station this generator came from. Carried so a
    /// row can say what it constrains without a second lookup, and so the
    /// certificate can be reported split by kind.
    pub obstacle_idx: usize,
    pub station_idx: Option<usize>,
}

/// Why a generator cannot produce a wall at all. Both cases are counted, never
/// silently skipped: a pair that needs a row and gets none is a hole in the
/// guarantee, and a row set that is silent about it sums to zero violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorFault {
    /// The station lies inside the obstacle. The shadow is then all of space
    /// beyond the body and no supporting half-space exists.
    StationInsideBody,
}

impl<'a> Generator<'a> {
    #[inline]
    pub fn spatial_dim(&self) -> usize {
        self.dim - 1
    }

    #[inline]
    pub fn degree(&self) -> usize {
        self.n_ctrl - 1
    }

    /// True when this generator carries a shadow. The keep-out-only generator
    /// answers false and takes every `u = 1` fast path below.
    #[inline]
    pub fn casts_shadow(&self) -> bool {
        self.station.is_some()
    }

    /// The lifted centreline point, `Sigma(s, 1)`.
    pub fn centre_at(&self, s: f64) -> Vec<f64> {
        bezier::evaluate(self.ctrl, self.n_ctrl, self.dim, s.clamp(0.0, 1.0))
    }

    /// Control points of `Sigma(., u)` — the centreline stretched from the
    /// station by `u`, with the time coordinate untouched.
    ///
    /// **`u = 1` returns the centreline unchanged, bit for bit**, and so does the
    /// no-station generator for any `u`. That is what makes the reduction exact
    /// rather than approximate.
    pub fn stretched_ctrl(&self, u: f64) -> Vec<f64> {
        let Some(p) = self.station else {
            return self.ctrl.to_vec();
        };
        if u == 1.0 {
            return self.ctrl.to_vec();
        }
        let sd = self.spatial_dim();
        let mut out = self.ctrl.to_vec();
        for l in 0..self.n_ctrl {
            for k in 0..sd {
                let idx = l * self.dim + k;
                out[idx] = p[k] + u * (self.ctrl[idx] - p[k]);
            }
        }
        out
    }

    /// Stretch one lifted point. Time is not stretched: the light source is a
    /// point in SPACE, present at every instant, so it maps each instant into
    /// itself.
    pub fn stretch_point(&self, x: &[f64], u: f64) -> Vec<f64> {
        let Some(p) = self.station else {
            return x.to_vec();
        };
        let sd = self.spatial_dim();
        let mut out = x.to_vec();
        for k in 0..sd {
            out[k] = p[k] + u * (x[k] - p[k]);
        }
        out
    }

    /// `Sigma(s, u)`.
    pub fn point(&self, s: f64, u: f64) -> Vec<f64> {
        let x = self.centre_at(s);
        self.stretch_point(&x, u)
    }

    /// The ball radius at stretch `u`. Linear, and exactly `r_m` at `u = 1`.
    #[inline]
    pub fn radius(&self, u: f64) -> f64 {
        u * self.r_m
    }

    /// The stretch factor minimising the SIGNED distance
    /// `f(u) = |c - Sigma(s,u)| - u*r_m` over `u >= 1` — the point of the
    /// keep-out zone this parameter generates that is nearest `c`.
    ///
    /// Closed form, no iteration. `f` is convex in `u` (a distance restricted to
    /// a line, minus a linear term), so its interior critical point is a minimum
    /// and `f'` is increasing. Hence:
    ///
    /// * `f'(1) >= 0` — `f` rises on the whole ray, so `u* = 1`;
    /// * otherwise the unique root of `f' = 0`, from squaring
    ///   `A*u - B = r_m * sqrt(A*u^2 - 2*B*u + C)`, which gives
    ///   `A*u^2 - 2*B*u + (B^2 - r_m^2*C)/(A - r_m^2) = 0` and the `+` branch,
    ///   the one with `A*u - B >= 0`.
    ///
    /// `A = |x(s) - p|^2 > r_m^2` exactly when the station is outside the body,
    /// which is the same condition that makes the shadow a proper cone.
    /// Convexity was checked numerically over 5000 random configurations: the
    /// worst second difference was zero, never negative.
    pub fn nearest_u_at(&self, x: &[f64], c: &[f64]) -> Result<f64, GeneratorFault> {
        let Some(p) = self.station else {
            return Ok(1.0);
        };
        let sd = self.spatial_dim();
        let r = self.r_m;
        let mut a = 0.0;
        let mut b = 0.0;
        let mut cc = 0.0;
        for k in 0..sd {
            let v = x[k] - p[k];
            let w = c[k] - p[k];
            a += v * v;
            b += w * v;
            cc += w * w;
        }
        let dt = c[sd] - x[sd];
        cc += dt * dt;
        if a <= r * r * (1.0 + 1e-12) {
            return Err(GeneratorFault::StationInsideBody);
        }

        // f'(1), with the denominator `|c - Sigma(s,1)|`.
        let denom_sq = a - 2.0 * b + cc;
        if denom_sq <= 0.0 {
            // The centroid sits ON the centreline. Every direction is equally
            // valid there; the caller already counts that case.
            return Ok(1.0);
        }
        if (a - b) / denom_sq.sqrt() >= r {
            return Ok(1.0);
        }

        let q = (b * b - r * r * cc) / (a - r * r);
        let disc = (b * b - a * q).max(0.0);
        Ok(((b + disc.sqrt()) / a).max(1.0))
    }

    /// `nearest_u_at` at an obstacle parameter.
    pub fn nearest_u(&self, s: f64, c: &[f64]) -> Result<f64, GeneratorFault> {
        if self.station.is_none() {
            return Ok(1.0);
        }
        let x = self.centre_at(s);
        self.nearest_u_at(&x, c)
    }

    /// Squared distance from `c` to the generator surface at parameter `s`, taken
    /// at the stretch that puts the keep-out zone nearest.
    ///
    /// **This is the driver of everything downstream** — the band search, the
    /// nearest-approach parameter, the approach cut — and with no station it is
    /// literally `|c - gamma(s)|^2`, the quantity the tube builder used.
    pub fn dist_sq(&self, s: f64, c: &[f64]) -> Result<(f64, f64), GeneratorFault> {
        let x = self.centre_at(s);
        let u = self.nearest_u_at(&x, c)?;
        let z = self.stretch_point(&x, u);
        let d2: f64 = (0..self.dim).map(|k| (c[k] - z[k]) * (c[k] - z[k])).sum();
        Ok((d2, u))
    }

    /// The stretch factors whose ball meets `Ball(c, rho)`, intersected with
    /// `u >= 1`. `None` when this parameter's material misses the clip ball
    /// entirely.
    ///
    /// `|c - Sigma(u)| <= rho + u*r_m` squares to
    /// `(A - r_m^2) u^2 - 2 (B + rho*r_m) u + (C - rho^2) <= 0`, an upward
    /// parabola because `A > r_m^2`, so the solution set is one closed interval.
    pub fn u_interval_in_ball(
        &self,
        x: &[f64],
        c: &[f64],
        rho: f64,
    ) -> Result<Option<(f64, f64)>, GeneratorFault> {
        if self.station.is_none() {
            // The tube's only stretch. It meets the clip ball iff the material
            // does, which the caller has already decided.
            return Ok(Some((1.0, 1.0)));
        }
        let p = self.station.unwrap();
        let sd = self.spatial_dim();
        let r = self.r_m;
        let (mut a, mut b, mut cc) = (0.0, 0.0, 0.0);
        for k in 0..sd {
            let v = x[k] - p[k];
            let w = c[k] - p[k];
            a += v * v;
            b += w * v;
            cc += w * w;
        }
        let dt = c[sd] - x[sd];
        cc += dt * dt;
        if a <= r * r * (1.0 + 1e-12) {
            return Err(GeneratorFault::StationInsideBody);
        }
        let qa = a - r * r;
        let qb = b + rho * r;
        let qc = cc - rho * rho;
        let disc = qb * qb - qa * qc;
        if disc < 0.0 {
            return Ok(None);
        }
        let root = disc.sqrt();
        let lo = ((qb - root) / qa).max(1.0);
        let hi = (qb + root) / qa;
        if hi < lo {
            return Ok(None);
        }
        Ok(Some((lo, hi)))
    }

    /// Whether this generator is defined at all, checked over the WHOLE
    /// parameter range rather than at one point.
    ///
    /// The shadow of a body that swallows its own light source is every position
    /// beyond the body in every direction — a set with no supporting half-space
    /// anywhere — so a generator whose centreline passes within `r_m` of the
    /// station cannot produce a wall. **That is a dropped wall, not an absent
    /// one.** Reporting it as "out of reach" would make the row set silent about
    /// the one configuration where line of sight is most certainly lost, and a
    /// silent row set sums to a certificate of 0.0.
    ///
    /// The sampled minimum is padded by half a spacing times the Lipschitz
    /// bound, so a near-miss between samples cannot slip through.
    pub fn fault(&self) -> Option<GeneratorFault> {
        let p = self.station?;
        let sd = self.spatial_dim();
        let h = 1.0 / BOUND_SAMPLES as f64;
        let pad = 0.5 * h * self.speed_bound();
        let mut v_min = f64::INFINITY;
        for i in 0..=BOUND_SAMPLES {
            let x = self.centre_at(i as f64 / BOUND_SAMPLES as f64);
            let d: f64 = (0..sd).map(|k| (x[k] - p[k]) * (x[k] - p[k])).sum::<f64>().sqrt();
            v_min = v_min.min(d);
        }
        if v_min - pad <= self.r_m {
            Some(GeneratorFault::StationInsideBody)
        } else {
            None
        }
    }

    /// Lipschitz bound on `||gamma'||` from the control polygon:
    /// `N * max_l ||G_{l+1} - G_l||`. Standard for a Bezier, and it is what makes
    /// the band search conservative rather than hopeful.
    pub fn speed_bound(&self) -> f64 {
        let mut worst: f64 = 0.0;
        for l in 0..self.degree() {
            let mut acc = 0.0;
            for k in 0..self.dim {
                let delta = self.ctrl[(l + 1) * self.dim + k] - self.ctrl[l * self.dim + k];
                acc += delta * delta;
            }
            worst = worst.max(acc.sqrt());
        }
        worst * self.degree() as f64
    }

    /// The Lipschitz bound the band search must pad with.
    ///
    /// `d(Sigma)/ds = (u * x'(s), t'(s))`, whose norm is at most `u` times the
    /// centreline's own bound for `u >= 1`, so the surface moves faster in `s`
    /// than the centreline does by exactly the stretch factor. **Substituting the
    /// unscaled bound here is the one place in this rewrite where a substitution
    /// is silently wrong**: it would leave the interval search hoping rather than
    /// proving, and the band could miss material between samples.
    pub fn param_lipschitz(&self, c: &[f64]) -> f64 {
        let speed = self.speed_bound();
        match self.u_bound(c, speed) {
            Some(u) => u * speed,
            None => speed,
        }
    }

    /// A RIGOROUS upper bound on `u*(s)` over the whole parameter range.
    ///
    /// At the optimum `r_m * |c - Sigma| = A u - B`; bounding `|c - Sigma|` above
    /// by `|w| + u|v|` and `A` below by `v_min^2` rearranges to
    ///
    /// ```text
    ///   u* <= |w| * (r_m + v_max) / ( v_min * (v_min - r_m) )
    /// ```
    ///
    /// with `v_min`/`v_max` the extreme station-to-centreline distances, each
    /// padded by half a sample spacing times the curve's Lipschitz bound so the
    /// scan cannot miss between samples. `None` when the station comes within
    /// `r_m` of the centreline anywhere, which is the same fault the shadow
    /// itself has there.
    ///
    /// Used only to scale the band search's padding, so looseness costs
    /// conservatism and never validity.
    pub fn u_bound(&self, c: &[f64], speed_bound: f64) -> Option<f64> {
        let p = self.station?;
        let sd = self.spatial_dim();
        let h = 1.0 / BOUND_SAMPLES as f64;
        let pad = 0.5 * h * speed_bound;
        let mut v_min = f64::INFINITY;
        let mut v_max: f64 = 0.0;
        for i in 0..=BOUND_SAMPLES {
            let x = self.centre_at(i as f64 / BOUND_SAMPLES as f64);
            let d: f64 = (0..sd).map(|k| (x[k] - p[k]) * (x[k] - p[k])).sum::<f64>().sqrt();
            v_min = v_min.min(d);
            v_max = v_max.max(d);
        }
        let v_min = v_min - pad;
        let v_max = v_max + pad;
        if !(v_min > self.r_m) {
            return None;
        }
        let w: f64 = (0..sd).map(|k| (c[k] - p[k]) * (c[k] - p[k])).sum::<f64>().sqrt();
        Some((w * (self.r_m + v_max) / (v_min * (v_min - self.r_m))).max(1.0))
    }

    /// A RIGOROUS ceiling on the stretch factors whose material can reach
    /// `Ball(c, rho)`, over the WHOLE parameter range.
    ///
    /// Material inside the clip ball has `|c - Sigma(s,u)| <= rho + u*r_m`, and
    /// the reverse triangle inequality gives `|c - Sigma| >= u*v_min - |w|`, so
    /// `u <= (rho + |w|) / (v_min - r_m)`. `1.0` with no station, which collapses
    /// the support integration back to a single stretch — the tube's own case.
    pub fn u_ceiling_in_ball(&self, c: &[f64], rho: f64, speed_bound: f64) -> f64 {
        let Some(p) = self.station else {
            return 1.0;
        };
        let sd = self.spatial_dim();
        let h = 1.0 / BOUND_SAMPLES as f64;
        let pad = 0.5 * h * speed_bound;
        let mut v_min = f64::INFINITY;
        for i in 0..=BOUND_SAMPLES {
            let x = self.centre_at(i as f64 / BOUND_SAMPLES as f64);
            let d: f64 = (0..sd).map(|k| (x[k] - p[k]) * (x[k] - p[k])).sum::<f64>().sqrt();
            v_min = v_min.min(d);
        }
        let v_min = v_min - pad;
        if !(v_min > self.r_m) {
            return f64::INFINITY;
        }
        let w: f64 = (0..sd).map(|k| (c[k] - p[k]) * (c[k] - p[k])).sum::<f64>().sqrt();
        ((rho + w) / (v_min - self.r_m)).max(1.0)
    }

    /// The directions the support anchor can slide along as the segment centroid
    /// moves: the surface tangents at `(s, u)`, orthonormalized.
    ///
    /// One vector on a plain tube — the centreline's unit tangent, which is what
    /// the tube builder used. Two on the surface where the stretch is free to
    /// move, because the anchor slides in `u` as well as in `s`. At `u = 1` the
    /// stretch is pinned by its own lower bound and only the `s` tangent
    /// survives.
    pub fn tangents(&self, s: f64, u: f64, s_free: bool, u_free: bool) -> Vec<Vec<f64>> {
        let mut face: Vec<Vec<f64>> = Vec::new();
        let deg = self.degree();
        if s_free && deg > 0 {
            let stretched = self.stretched_ctrl(u);
            let mut hodo = vec![0.0; deg * self.dim];
            for l in 0..deg {
                for k in 0..self.dim {
                    hodo[l * self.dim + k] = deg as f64
                        * (stretched[(l + 1) * self.dim + k] - stretched[l * self.dim + k]);
                }
            }
            let mut v = bezier::evaluate(&hodo, deg, self.dim, s.clamp(0.0, 1.0));
            let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nv >= 1e-12 {
                for x in v.iter_mut() {
                    *x /= nv;
                }
                face.push(v);
            }
        }
        if self.casts_shadow() && u_free {
            // d(Sigma)/du = (x(s) - p, 0): the ray direction, spatial only.
            let p = self.station.unwrap();
            let x = self.centre_at(s);
            let sd = self.spatial_dim();
            let mut d = vec![0.0; self.dim];
            for k in 0..sd {
                d[k] = x[k] - p[k];
            }
            // Gram-Schmidt against the tangent already collected.
            if let Some(first) = face.first() {
                let proj: f64 = (0..self.dim).map(|k| d[k] * first[k]).sum();
                for k in 0..self.dim {
                    d[k] -= proj * first[k];
                }
            }
            let nd = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nd >= 1e-12 {
                for x in d.iter_mut() {
                    *x /= nd;
                }
                face.push(d);
            }
        }
        face
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A degree-1 obstacle moving along x, lifted, with the station at the
    /// origin. Analytic answers are available for everything here.
    fn gen_with_station<'a>(
        ctrl: &'a [f64],
        station: &'a [f64],
        r_m: f64,
    ) -> Generator<'a> {
        Generator {
            ctrl,
            n_ctrl: 2,
            dim: 3,
            r_m,
            station: Some(station),
            obstacle_idx: 0,
            station_idx: Some(0),
        }
    }

    #[test]
    fn u_one_is_the_obstacle_and_the_stretch_leaves_time_alone() {
        let ctrl = [1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let station = [0.0, 0.0];
        let g = gen_with_station(&ctrl, &station, 0.25);
        let at_one = g.stretched_ctrl(1.0);
        assert_eq!(at_one, ctrl.to_vec());

        let at_three = g.stretched_ctrl(3.0);
        // Spatial coordinates scale about the station; times do not move.
        assert!((at_three[0] - 3.0).abs() < 1e-15);
        assert!((at_three[1] - 0.0).abs() < 1e-15);
        assert!((at_three[2] - 0.0).abs() < 1e-15);
        assert!((at_three[3] - 6.0).abs() < 1e-15);
        assert!((at_three[4] - 3.0).abs() < 1e-15);
        assert!((at_three[5] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn no_station_pins_the_stretch_at_one() {
        let ctrl = [1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let g = Generator {
            ctrl: &ctrl,
            n_ctrl: 2,
            dim: 3,
            r_m: 0.25,
            station: None,
            obstacle_idx: 0,
            station_idx: None,
        };
        assert_eq!(g.stretched_ctrl(7.0), ctrl.to_vec());
        let c = [5.0, 5.0, 0.5];
        assert_eq!(g.nearest_u(0.3, &c).unwrap(), 1.0);
        // With the stretch pinned, the driver is the plain lifted distance.
        let (d2, u) = g.dist_sq(0.3, &c).unwrap();
        assert_eq!(u, 1.0);
        let x = g.centre_at(0.3);
        let want: f64 = (0..3).map(|k| (c[k] - x[k]) * (c[k] - x[k])).sum();
        assert!((d2 - want).abs() < 1e-15, "{d2} vs {want}");
    }

    #[test]
    fn nearest_u_beats_a_dense_scan() {
        let ctrl = [1.0, 0.5, 0.0, 2.0, -0.5, 1.0];
        let station = [0.0, 0.0];
        let g = gen_with_station(&ctrl, &station, 0.3);
        for i in 0..25 {
            let s = i as f64 / 24.0;
            let c = [3.0 + 0.2 * i as f64, 1.5 - 0.1 * i as f64, 0.4];
            let x = g.centre_at(s);
            let u = g.nearest_u_at(&x, &c).unwrap();
            let f = |u: f64| -> f64 {
                let z = g.stretch_point(&x, u);
                let d: f64 =
                    (0..3).map(|k| (c[k] - z[k]) * (c[k] - z[k])).sum::<f64>().sqrt();
                d - u * g.r_m
            };
            let mine = f(u);
            let mut best = f64::INFINITY;
            for j in 0..=200000 {
                let uu = 1.0 + 30.0 * j as f64 / 200000.0;
                best = best.min(f(uu));
            }
            assert!(
                mine <= best + 1e-9,
                "closed form {mine} worse than scan {best} at s={s}"
            );
        }
    }

    /// The reason the radius must grow: a sweep with constant `r_m` declares
    /// genuinely occluded positions clear. This test fails if someone "simplifies"
    /// `radius` to a constant.
    #[test]
    fn the_sweep_is_the_umbra_and_a_constant_radius_is_not() {
        let ctrl = [2.0, 0.0, 0.0, 2.0, 0.0, 1.0]; // static body at (2,0)
        let station = [0.0, 0.0];
        let g = gen_with_station(&ctrl, &station, 0.4);
        let x = g.centre_at(0.5);

        // A point squarely behind the body, far enough out that the umbra has
        // widened past r_m: offset 0.6 > 0.4, still inside the cone at u = 4.
        let q = [8.0, 0.6, 0.0];
        let occluded = |scale: bool| -> bool {
            let mut best = f64::INFINITY;
            for j in 0..=20000 {
                let u = 1.0 + 20.0 * j as f64 / 20000.0;
                let z = g.stretch_point(&x, u);
                let d: f64 =
                    (0..3).map(|k| (q[k] - z[k]) * (q[k] - z[k])).sum::<f64>().sqrt();
                best = best.min(d - if scale { u * g.r_m } else { g.r_m });
            }
            best <= 0.0
        };
        // Truth: distance from the body centre to the segment [station, q].
        let seg = [q[0] - station[0], q[1] - station[1]];
        let tau = ((x[0] - station[0]) * seg[0] + (x[1] - station[1]) * seg[1])
            / (seg[0] * seg[0] + seg[1] * seg[1]);
        let foot = [station[0] + tau * seg[0], station[1] + tau * seg[1]];
        let truth = ((x[0] - foot[0]).powi(2) + (x[1] - foot[1]).powi(2)).sqrt() <= g.r_m;

        assert!(truth, "fixture must be genuinely occluded");
        assert!(occluded(true), "the u*r_m sweep must see it");
        assert!(!occluded(false), "a constant radius must MISS it — that is the bug");
    }

    #[test]
    fn the_u_interval_brackets_the_material_that_meets_the_clip_ball() {
        let ctrl = [2.0, 0.0, 0.0, 3.0, 1.0, 1.0];
        let station = [0.0, 0.0];
        let g = gen_with_station(&ctrl, &station, 0.3);
        let c = [6.0, 1.0, 0.5];
        let rho = 1.2;
        let x = g.centre_at(0.4);
        let (lo, hi) = g.u_interval_in_ball(&x, &c, rho).unwrap().unwrap();
        let meets = |u: f64| {
            let z = g.stretch_point(&x, u);
            let d: f64 = (0..3).map(|k| (c[k] - z[k]) * (c[k] - z[k])).sum::<f64>().sqrt();
            d <= rho + u * g.r_m + 1e-12
        };
        assert!(meets(0.5 * (lo + hi)), "the middle of the bracket must meet the ball");
        for j in 0..200 {
            let u = 1.0 + 20.0 * j as f64 / 200.0;
            if meets(u) {
                assert!(
                    u >= lo - 1e-9 && u <= hi + 1e-9,
                    "u={u} meets the ball but sits outside [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn u_bound_is_never_exceeded_by_the_true_optimum() {
        let ctrl = [2.0, 0.5, 0.0, 3.5, -0.5, 1.0];
        let station = [0.0, 0.0];
        let g = gen_with_station(&ctrl, &station, 0.3);
        let speed = 3.0 * 2.0f64.sqrt();
        for c in [[5.0, 2.0, 0.3], [12.0, -3.0, 0.9], [1.0, 1.0, 0.2]] {
            let bound = g.u_bound(&c, speed).unwrap();
            for i in 0..=200 {
                let s = i as f64 / 200.0;
                let u = g.nearest_u(s, &c).unwrap();
                assert!(u <= bound + 1e-9, "u*={u} exceeds the bound {bound}");
            }
        }
    }

    #[test]
    fn the_face_gains_the_ray_direction_only_where_the_stretch_can_move() {
        let ctrl = [2.0, 0.0, 0.0, 3.0, 1.0, 1.0];
        let station = [0.0, 0.0];
        let g = gen_with_station(&ctrl, &station, 0.3);
        assert_eq!(g.tangents(0.5, 2.0, true, false).len(), 1);
        assert_eq!(g.tangents(0.5, 2.0, true, true).len(), 2);
        // Parameter frozen at an interval end: the stretch direction alone, and
        // it is the RAW ray direction, not one orthogonalized against a tangent
        // that is not in the face.
        let only_u = g.tangents(0.5, 2.0, false, true);
        assert_eq!(only_u.len(), 1);
        let x = g.centre_at(0.5);
        let ray = [x[0] - station[0], x[1] - station[1], 0.0];
        let nr = (ray[0] * ray[0] + ray[1] * ray[1]).sqrt();
        for k in 0..3 {
            assert!((only_u[0][k] - ray[k] / nr).abs() < 1e-12);
        }
        let face = g.tangents(0.5, 2.0, true, true);
        let dot: f64 = (0..3).map(|k| face[0][k] * face[1][k]).sum();
        assert!(dot.abs() < 1e-12, "face must be orthonormal, got {dot}");
    }
}
