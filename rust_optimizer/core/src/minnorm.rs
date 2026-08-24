//! Minimum-norm point in the convex hull of finitely many points.
//!
//! This is the projection `y* = argmin_{y in G} ||c - y||` of PAPER_1 statement
//! (4), and it is the whole reason the construction is sound for a curved tube.
//!
//! **Why not the tube's nearest surface point.** Taking the normal from `c - f`
//! with `f` the closest point of the obstacle centreline satisfies statement (4)
//! only when the keep-out zone is convex. Measured counterexample, obstacle on a
//! circle of radius 10 at 0.5 rad/s with `r = 1` and the segment centroid inside
//! the turn: 12.35% of the clipped piece lands strictly on the safe side of that
//! plane, worst point 0.77 inside it. The projection onto the HULL moves the
//! normal by 1.19 degrees and the offset by 0.76, and removes every violation.
//! The difference is not a tolerance — it is the difference between a supporting
//! half-space and a plane that merely touches.
//!
//! **Why an exact method and not Frank-Wolfe.** Frank-Wolfe leaves a residual,
//! and a residual on this quantity is a plane that cuts slightly into the hull —
//! precisely the failure the construction exists to exclude. Wolfe's algorithm
//! terminates finitely at the exact minimiser (up to rounding), and the corral it
//! terminates with is the face `y*` lies in, which the rotation term needs.

/// The projection of the origin onto `conv{points}`.
pub struct MinNormPoint {
    /// The closest point of the hull to the origin.
    pub point: Vec<f64>,
    /// Barycentric weights of `point`, one per input point (zero off the corral).
    pub weights: Vec<f64>,
    /// The corral: indices of the affinely independent subset carrying `point`.
    /// This is the face of the hull the solution lies in, and its direction space
    /// is what `face_basis` turns into the projector `P_F`.
    pub active: Vec<usize>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Solve `A u = b` in place by Gaussian elimination with partial pivoting.
/// Returns `None` when the system is numerically singular, which is how an
/// affinely dependent corral announces itself.
fn solve_dense(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if best <= 1e-14 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / d;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut acc = b[col];
        for k in (col + 1)..n {
            acc -= a[col * n + k] * x[k];
        }
        x[col] = acc / a[col * n + col];
    }
    Some(x)
}

/// Minimum-norm point of the AFFINE hull of the corral, in barycentric weights.
///
/// Wolfe's system: with `E[i][j] = 1 + p_i . p_j`, solving `E u = 1` and
/// normalising `u` by its sum gives the weights of the affine-hull minimiser.
/// `E` is positive definite exactly when the corral is affinely independent, so a
/// singular solve is the signal to drop a point rather than a failure.
fn affine_min_norm(points: &[f64], dim: usize, corral: &[usize]) -> Option<Vec<f64>> {
    let m = corral.len();
    let mut e = vec![0.0; m * m];
    for i in 0..m {
        let pi = &points[corral[i] * dim..(corral[i] + 1) * dim];
        for j in 0..m {
            let pj = &points[corral[j] * dim..(corral[j] + 1) * dim];
            e[i * m + j] = 1.0 + dot(pi, pj);
        }
    }
    let mut rhs = vec![1.0; m];
    let u = solve_dense(&mut e, &mut rhs, m)?;
    let total: f64 = u.iter().sum();
    if total.abs() <= 1e-14 {
        return None;
    }
    Some(u.iter().map(|v| v / total).collect())
}

fn combine(points: &[f64], dim: usize, corral: &[usize], w: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    for (slot, &idx) in corral.iter().enumerate() {
        let p = &points[idx * dim..(idx + 1) * dim];
        for d in 0..dim {
            out[d] += w[slot] * p[d];
        }
    }
    out
}

/// Wolfe's minimum-norm-point algorithm.
///
/// `points` is `n_pts * dim`, row-major. Translate before calling: to project a
/// query `c` onto `conv{G_l}`, pass `G_l - c` and add `c` back to the result.
///
/// Terminates finitely. The iteration caps are a runaway guard, not a tolerance:
/// each major loop strictly decreases `||x||`, and the corral can only grow to
/// `dim + 1` points before a drop is forced.
pub fn min_norm_point(points: &[f64], n_pts: usize, dim: usize) -> MinNormPoint {
    assert!(n_pts >= 1 && dim >= 1);
    // Relative tolerance: the optimality test compares inner products, whose
    // magnitude scales with the SQUARE of the point coordinates. A fixed epsilon
    // would be a different test at different scene scales.
    let scale = (0..n_pts)
        .map(|i| dot(&points[i * dim..(i + 1) * dim], &points[i * dim..(i + 1) * dim]))
        .fold(0.0f64, f64::max)
        .max(1.0);
    let tol = 1e-12 * scale;

    // Start at the input point closest to the origin.
    let mut best = 0usize;
    let mut best_sq = f64::INFINITY;
    for i in 0..n_pts {
        let s = dot(&points[i * dim..(i + 1) * dim], &points[i * dim..(i + 1) * dim]);
        if s < best_sq {
            best_sq = s;
            best = i;
        }
    }
    let mut corral = vec![best];
    let mut alpha = vec![1.0f64];
    let mut x = points[best * dim..(best + 1) * dim].to_vec();

    for _ in 0..(4 * n_pts + 32) {
        // Which point most reduces the objective; `<x, p_j>` is the linearisation.
        let mut j = 0usize;
        let mut min_ip = f64::INFINITY;
        for i in 0..n_pts {
            let ip = dot(&x, &points[i * dim..(i + 1) * dim]);
            if ip < min_ip {
                min_ip = ip;
                j = i;
            }
        }
        // Optimality: no point lies strictly inside the supporting half-space at
        // x, i.e. <x, p_j> >= <x, x> for every j.
        if min_ip >= dot(&x, &x) - tol {
            break;
        }
        if corral.contains(&j) {
            break; // numerical stall; x is already optimal to working precision
        }
        corral.push(j);
        alpha.push(0.0);

        // Minor loop: walk to the affine minimiser, dropping points as their
        // weights hit zero, until it lands strictly inside the corral.
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 2 * (dim + 2) + 8 {
                break;
            }
            let Some(lambda) = affine_min_norm(points, dim, &corral) else {
                // Affinely dependent: drop the point just added and stop.
                corral.pop();
                alpha.pop();
                break;
            };
            if lambda.iter().all(|&v| v > tol) {
                x = combine(points, dim, &corral, &lambda);
                alpha = lambda;
                break;
            }
            // Step from x toward the affine minimiser until the first weight
            // would go negative.
            let mut theta = 1.0f64;
            for i in 0..corral.len() {
                if lambda[i] <= tol {
                    let denom = alpha[i] - lambda[i];
                    if denom > 1e-18 {
                        theta = theta.min(alpha[i] / denom);
                    } else {
                        theta = 0.0;
                    }
                }
            }
            theta = theta.clamp(0.0, 1.0);
            for i in 0..corral.len() {
                alpha[i] += theta * (lambda[i] - alpha[i]);
            }
            let mut keep_c = Vec::with_capacity(corral.len());
            let mut keep_a = Vec::with_capacity(corral.len());
            for i in 0..corral.len() {
                if alpha[i] > tol {
                    keep_c.push(corral[i]);
                    keep_a.push(alpha[i]);
                }
            }
            if keep_c.is_empty() {
                keep_c.push(corral[0]);
                keep_a.push(1.0);
            }
            let total: f64 = keep_a.iter().sum();
            for v in keep_a.iter_mut() {
                *v /= total;
            }
            corral = keep_c;
            alpha = keep_a;
            x = combine(points, dim, &corral, &alpha);
        }
    }

    let mut weights = vec![0.0; n_pts];
    for (slot, &idx) in corral.iter().enumerate() {
        weights[idx] = alpha[slot];
    }
    MinNormPoint {
        point: x,
        weights,
        active: corral,
    }
}

/// Orthonormal basis of the corral's direction space — `span{p_i - p_i0}`.
///
/// This is the `F` of PAPER_1's rotation term. `P_F = sum_b b b^T` projects onto
/// it, and `corr = (I - P_F)(I - n n^T) d_k / ||c - y*||` is the derivative of the
/// row's clearance through the aiming direction. A single-point corral gives an
/// empty basis, so `P_F = 0`, which reproduces the frozen-nearest-point case the
/// capsule builder called `clamped`.
pub fn face_basis(points: &[f64], dim: usize, active: &[usize]) -> Vec<Vec<f64>> {
    if active.len() < 2 {
        return Vec::new();
    }
    let base = &points[active[0] * dim..(active[0] + 1) * dim];
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for &idx in &active[1..] {
        let p = &points[idx * dim..(idx + 1) * dim];
        let mut v: Vec<f64> = (0..dim).map(|d| p[d] - base[d]).collect();
        for b in &basis {
            let c = dot(&v, b);
            for d in 0..dim {
                v[d] -= c * b[d];
            }
        }
        let norm = dot(&v, &v).sqrt();
        if norm > 1e-10 {
            basis.push(v.iter().map(|x| x / norm).collect());
        }
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f64]) -> f64 {
        dot(v, v).sqrt()
    }

    /// Independent check by exhaustive face enumeration. Exponential, so it is a
    /// test oracle only — but it cannot share a bug with Wolfe's algorithm,
    /// which is the point.
    fn brute_force(points: &[f64], n_pts: usize, dim: usize) -> Vec<f64> {
        let mut best: Option<Vec<f64>> = None;
        for mask in 1u32..(1u32 << n_pts) {
            let subset: Vec<usize> = (0..n_pts).filter(|i| mask & (1 << i) != 0).collect();
            let Some(w) = affine_min_norm(points, dim, &subset) else {
                continue;
            };
            if w.iter().any(|&v| v < -1e-9) {
                continue; // outside the hull; some other face carries it
            }
            let cand = combine(points, dim, &subset, &w);
            if best.as_ref().is_none_or(|b| norm(&cand) < norm(b)) {
                best = Some(cand);
            }
        }
        best.unwrap()
    }

    /// Deterministic linear congruential noise — no rand dependency, and the
    /// same sequence every run so a failure is reproducible.
    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }

    #[test]
    fn matches_brute_force_on_random_polytopes() {
        let mut state = 0x2026_08_24u64;
        for dim in 2..=4 {
            for n_pts in 1..=6 {
                for _ in 0..40 {
                    let pts: Vec<f64> = (0..n_pts * dim).map(|_| lcg(&mut state) * 5.0).collect();
                    let got = min_norm_point(&pts, n_pts, dim);
                    let want = brute_force(&pts, n_pts, dim);
                    assert!(
                        (norm(&got.point) - norm(&want)).abs() < 1e-7,
                        "dim={dim} n={n_pts}: wolfe |x|={} brute |x|={}",
                        norm(&got.point),
                        norm(&want)
                    );
                }
            }
        }
    }

    /// The variational inequality is the property statement (4) actually uses:
    /// `(0 - y*).(y - y*) <= 0` for every hull point `y`. If this fails, the
    /// half-space built from `y*` does not support the hull.
    #[test]
    fn satisfies_the_variational_inequality() {
        let mut state = 0xfeed_beefu64;
        for dim in 2..=4 {
            for n_pts in 2..=6 {
                for _ in 0..40 {
                    let pts: Vec<f64> = (0..n_pts * dim).map(|_| lcg(&mut state) * 5.0).collect();
                    let r = min_norm_point(&pts, n_pts, dim);
                    let y = &r.point;
                    for i in 0..n_pts {
                        let p = &pts[i * dim..(i + 1) * dim];
                        let ip: f64 = (0..dim).map(|d| (-y[d]) * (p[d] - y[d])).sum();
                        assert!(ip <= 1e-7 * norm(y).max(1.0), "VI violated by {ip}");
                    }
                }
            }
        }
    }

    #[test]
    fn origin_inside_the_hull_gives_zero() {
        // A simplex straddling the origin.
        let pts = vec![1.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, -1.5];
        let r = min_norm_point(&pts, 5, 3);
        assert!(norm(&r.point) < 1e-9, "expected 0, got {}", norm(&r.point));
    }

    #[test]
    fn face_basis_is_orthonormal_and_spans_the_corral() {
        // A triangle in the plane z = 2 that STRADDLES the origin's shadow, so
        // the projection lands in the relative interior of the 2-face rather
        // than on a vertex.
        let pts = vec![-1.0, -1.0, 2.0, 2.0, -1.0, 2.0, -1.0, 2.0, 2.0];
        let r = min_norm_point(&pts, 3, 3);
        assert!((r.point[2] - 2.0).abs() < 1e-12);
        assert!(r.point[0].abs() < 1e-9 && r.point[1].abs() < 1e-9, "{:?}", r.point);
        let basis = face_basis(&pts, 3, &r.active);
        for b in &basis {
            assert!((norm(b) - 1.0).abs() < 1e-12);
        }
        for i in 0..basis.len() {
            for j in (i + 1)..basis.len() {
                assert!(dot(&basis[i], &basis[j]).abs() < 1e-12);
            }
        }
        assert_eq!(basis.len(), 2, "corral {:?}", r.active);
    }

    /// When the projection lands on a VERTEX the corral is a single point, the
    /// face basis is empty, and `P_F = 0`. That is not a degenerate case to
    /// guard against — it is exactly the frozen-nearest-point branch the capsule
    /// builder called `clamped`, recovered from the general formula.
    #[test]
    fn a_vertex_projection_gives_an_empty_face_basis() {
        let pts = vec![0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0];
        let r = min_norm_point(&pts, 3, 3);
        assert_eq!(r.active, vec![0]);
        assert!(face_basis(&pts, 3, &r.active).is_empty());
    }
}
