/// De Casteljau subdivision for Bézier curve segmentation.
use std::cell::RefCell;

/// Compute subdivision coefficients for a single basis vector, into reused
/// buffers. The in-place forward sweep computes, per level, exactly the values
/// the old per-level allocation computed -- `w[j] = (1-tau)*w[j] + tau*w[j+1]`
/// ascending reads only not-yet-overwritten entries -- so the results are
/// bit-identical; only the allocations are gone.
fn split_1d_into(
    n: usize,
    tau: f64,
    basis_index: usize,
    w: &mut Vec<f64>,
    left: &mut Vec<f64>,
    right: &mut Vec<f64>,
) {
    w.clear();
    w.resize(n + 1, 0.0);
    w[basis_index] = 1.0;
    left.clear();
    right.clear();
    left.push(w[0]);
    right.push(w[n]);

    let mut len = n + 1;
    for _ in 1..=n {
        for j in 0..len - 1 {
            w[j] = (1.0 - tau) * w[j] + tau * w[j + 1];
        }
        len -= 1;
        left.push(w[0]);
        right.push(w[len - 1]);
    }
    right.reverse();
}

thread_local! {
    /// Scratch for `split_matrices`: the basis vector and the two coefficient
    /// columns of `split_1d_into`. `component_support` subdivides 32 pieces per
    /// wall, each piece two split calls -- this buffer removes the roughly
    /// n-squared small allocations each of those used to make.
    static SPLIT_SCRATCH: RefCell<(Vec<f64>, Vec<f64>, Vec<f64>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

/// Compute subdivision matrices S_left and S_right (each (N+1) x (N+1), row-major).
pub fn split_matrices(n: usize, tau: f64) -> (Vec<f64>, Vec<f64>) {
    let sz = n + 1;
    let mut s_left = vec![0.0; sz * sz];
    let mut s_right = vec![0.0; sz * sz];
    SPLIT_SCRATCH.with(|cell| {
        let (w, l, r) = &mut *cell.borrow_mut();
        for j in 0..sz {
            split_1d_into(n, tau, j, w, l, r);
            for i in 0..sz {
                s_left[i * sz + j] = l[i];
                s_right[i * sz + j] = r[i];
            }
        }
    });
    (s_left, s_right)
}

/// Generate segment matrices for equal-parameter splitting.
/// Returns list of (N+1) x (N+1) matrices (row-major).
pub fn segment_matrices_equal_params(n: usize, n_seg: usize) -> Vec<Vec<f64>> {
    assert!(n_seg >= 1);
    let sz = n + 1;
    if n_seg == 1 {
        let mut eye = vec![0.0; sz * sz];
        for i in 0..sz {
            eye[i * sz + i] = 1.0;
        }
        return vec![eye];
    }

    let mut mats = Vec::with_capacity(n_seg);
    // remainder = I
    let mut remainder = vec![0.0; sz * sz];
    for i in 0..sz {
        remainder[i * sz + i] = 1.0;
    }

    for k in (2..=n_seg).rev() {
        let tau = 1.0 / k as f64;
        let (s_l, s_r) = split_matrices(n, tau);
        // mats.push(S_L @ remainder)
        mats.push(crate::bezier::matmul(&s_l, sz, sz, &remainder, sz));
        // remainder = S_R @ remainder
        remainder = crate::bezier::matmul(&s_r, sz, sz, &remainder, sz);
    }
    mats.push(remainder);
    mats
}

/// Control points of the piece of a degree-`n` Bezier lying over `[alpha, beta]`,
/// as an (N+1) x (N+1) row-major matrix acting on the original control points.
///
/// This is what makes idea/spacetime.md statement (3) exact. The alternative — sampling the
/// curve over `[alpha, beta]` and hulling the samples — produces a set the curve
/// bulges OUTSIDE of between samples, so containment fails and every certificate
/// resting on it fails with it. Subdivision removes that error instead of
/// bounding it: the sub-curve's control points are computed, not estimated, and
/// the hull property applies to them verbatim.
///
/// Composed from two splits: restrict to `[alpha, 1]` first, then take the left
/// part at the position `beta` occupies in the restricted parameter.
pub fn subdivide_between(n: usize, alpha: f64, beta: f64) -> Vec<f64> {
    let mut out = Vec::new();
    subdivide_between_into(n, alpha, beta, &mut out);
    out
}

/// `subdivide_between` into a reused buffer; identical branches and
/// arithmetic. The two intermediate split matrices are still built fresh --
/// what this removes is the per-call result allocation in loops that subdivide
/// dozens of pieces per wall.
pub fn subdivide_between_into(n: usize, alpha: f64, beta: f64, out: &mut Vec<f64>) {
    let sz = n + 1;
    let (a, b) = if beta >= alpha { (alpha, beta) } else { (beta, alpha) };
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    // Whole domain: nothing to do, and the general path would divide by zero.
    if a <= 0.0 && b >= 1.0 {
        out.clear();
        out.resize(sz * sz, 0.0);
        for i in 0..sz {
            out[i * sz + i] = 1.0;
        }
        return;
    }

    let (_, s_right) = split_matrices(n, a);
    // Where beta sits once the curve has been restricted to [alpha, 1].
    let denom = 1.0 - a;
    if denom <= 1e-15 {
        // Degenerate interval at the very end of the domain: every control point
        // collapses onto the endpoint, which is the correct (single-point) hull.
        out.clear();
        out.resize(sz * sz, 0.0);
        for i in 0..sz {
            out[i * sz + n] = 1.0;
        }
        return;
    }
    let b_local = ((b - a) / denom).clamp(0.0, 1.0);
    let (s_left, _) = split_matrices(n, b_local);
    crate::bezier::matmul_into(&s_left, sz, sz, &s_right, sz, out);
}

#[cfg(test)]
mod tests {
    use super::*;


    /// The subdivided control points must describe the SAME curve over
    /// [alpha, beta], reparameterised to [0, 1]. This is what makes statement (3)
    /// exact rather than approximate.
    #[test]
    fn subdivision_reproduces_the_curve_exactly() {
        let n = 3usize;
        let dim = 2usize;
        // A curve that genuinely bends, so a wrong matrix cannot pass by symmetry.
        let ctrl = vec![0.0, 0.0, 1.0, 4.0, 5.0, -3.0, 6.0, 1.0];
        for &(a, b) in &[(0.0, 1.0), (0.2, 0.8), (0.0, 0.35), (0.61, 1.0), (0.4, 0.45)] {
            let m = subdivide_between(n, a, b);
            let sub = crate::bezier::matmul(&m, n + 1, n + 1, &ctrl, dim);
            for k in 0..=20 {
                let s = k as f64 / 20.0;
                let got = crate::bezier::evaluate(&sub, n + 1, dim, s);
                let want = crate::bezier::evaluate(&ctrl, n + 1, dim, a + s * (b - a));
                for d in 0..dim {
                    assert!(
                        (got[d] - want[d]).abs() < 1e-12,
                        "[{a},{b}] s={s} d={d}: {} vs {}",
                        got[d],
                        want[d]
                    );
                }
            }
        }
    }

    /// The falsifier for the construction idea/spacetime.md statement (3) forbids.
    ///
    /// Hulling SAMPLES of the curve over [alpha, beta] gives a set the curve
    /// bulges outside of between samples. Here that hull — the chord, for two
    /// samples — misses the curve by a wide margin, while the subdivided control
    /// hull contains it. If this test ever reports the chord as containing the
    /// curve, the fixture stopped bending and the test stopped being evidence.
    #[test]
    fn sampling_the_curve_does_not_contain_it_but_subdivision_does() {
        let n = 3usize;
        let dim = 2usize;
        let ctrl = vec![0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0, 0.0];
        let (a, b) = (0.1, 0.9);

        let m = subdivide_between(n, a, b);
        let sub = crate::bezier::matmul(&m, n + 1, n + 1, &ctrl, dim);

        // Signed distance to the hull, measured along the outward normal of the
        // chord between the two endpoint samples.
        let p0 = crate::bezier::evaluate(&ctrl, n + 1, dim, a);
        let p1 = crate::bezier::evaluate(&ctrl, n + 1, dim, b);
        let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
        let len = (dx * dx + dy * dy).sqrt();
        let (nx, ny) = (-dy / len, dx / len);

        let mut worst_chord: f64 = 0.0;
        let mut worst_sub: f64 = 0.0;
        for k in 0..=200 {
            let s = k as f64 / 200.0;
            let z = crate::bezier::evaluate(&ctrl, n + 1, dim, a + s * (b - a));
            let h = (z[0] - p0[0]) * nx + (z[1] - p0[1]) * ny;
            worst_chord = worst_chord.max(h.abs());
            // Distance outside the subdivided control hull, along the same normal.
            let hull_max = (0..=n)
                .map(|i| (sub[i * dim] - p0[0]) * nx + (sub[i * dim + 1] - p0[1]) * ny)
                .fold(f64::NEG_INFINITY, f64::max);
            worst_sub = worst_sub.max(h - hull_max);
        }
        assert!(
            worst_chord > 1.0,
            "fixture no longer bends: curve stays {worst_chord} from the chord"
        );
        assert!(
            worst_sub < 1e-12,
            "subdivided hull failed to contain the curve by {worst_sub}"
        );
    }

    #[test]
    fn test_segment_count_1() {
        let mats = segment_matrices_equal_params(3, 1);
        assert_eq!(mats.len(), 1);
        // Should be identity
        let sz = 4;
        for i in 0..sz {
            for j in 0..sz {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((mats[0][i * sz + j] - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_segment_count_4_first_matrix() {
        let mats = segment_matrices_equal_params(3, 4);
        assert_eq!(mats.len(), 4);
        // A[0][0,0] should be 1.0 (first control point of first segment = P0)
        assert!((mats[0][0] - 1.0).abs() < 1e-14);
        // A[0][1,1] should be 0.25
        assert!((mats[0][1 * 4 + 1] - 0.25).abs() < 1e-14);
    }
}
