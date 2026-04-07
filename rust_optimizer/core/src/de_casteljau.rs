/// De Casteljau subdivision for Bézier curve segmentation.

/// Compute subdivision coefficients for a single basis vector.
/// Returns (left, right) each of length N+1.
fn split_1d(n: usize, tau: f64, basis_index: usize) -> (Vec<f64>, Vec<f64>) {
    let mut w = vec![0.0; n + 1];
    w[basis_index] = 1.0;
    let mut left = Vec::with_capacity(n + 1);
    let mut right = Vec::with_capacity(n + 1);
    left.push(w[0]);
    right.push(w[n]);

    let mut ww = w;
    for _ in 1..=n {
        let mut new_w = vec![0.0; ww.len() - 1];
        for j in 0..new_w.len() {
            new_w[j] = (1.0 - tau) * ww[j] + tau * ww[j + 1];
        }
        left.push(new_w[0]);
        right.push(new_w[new_w.len() - 1]);
        ww = new_w;
    }
    right.reverse();
    (left, right)
}

/// Compute subdivision matrices S_left and S_right (each (N+1) x (N+1), row-major).
fn split_matrices(n: usize, tau: f64) -> (Vec<f64>, Vec<f64>) {
    let sz = n + 1;
    let mut s_left = vec![0.0; sz * sz];
    let mut s_right = vec![0.0; sz * sz];
    for j in 0..sz {
        let (l, r) = split_1d(n, tau, j);
        for i in 0..sz {
            s_left[i * sz + j] = l[i];
            s_right[i * sz + j] = r[i];
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
