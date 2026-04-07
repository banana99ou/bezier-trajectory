/// Bézier curve matrices and evaluation.
use std::f64;

/// Binomial coefficient C(n, k).
pub fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Derivative matrix D for degree N. Shape: N x (N+1), row-major.
pub fn get_d_matrix(n: usize) -> Vec<f64> {
    let rows = n;
    let cols = n + 1;
    let mut d = vec![0.0; rows * cols];
    let nf = n as f64;
    for i in 0..rows {
        d[i * cols + i] = -nf;
        d[i * cols + i + 1] = nf;
    }
    d
}

/// Elevation matrix E from degree N to N+1. Shape: (N+2) x (N+1), row-major.
pub fn get_e_matrix(n: usize) -> Vec<f64> {
    let rows = n + 2;
    let cols = n + 1;
    let mut e = vec![0.0; rows * cols];
    // First row
    e[0] = 1.0;
    // Last row
    e[(rows - 1) * cols + (cols - 1)] = 1.0;
    // Middle rows (1-indexed: 2..=N+1, 0-indexed: 1..=N)
    for row_idx in 1..=n {
        let i_1idx = row_idx + 1; // 1-indexed
        e[row_idx * cols + (row_idx - 1)] = (i_1idx - 1) as f64 / (n + 1) as f64;
        e[row_idx * cols + row_idx] = (n + 2 - i_1idx) as f64 / (n + 1) as f64;
    }
    e
}

/// Gram matrix G for degree N. Shape: (N+1) x (N+1), row-major.
pub fn get_g_matrix(n: usize) -> Vec<f64> {
    let sz = n + 1;
    let mut g = vec![0.0; sz * sz];
    for i in 0..sz {
        for j in 0..sz {
            g[i * sz + j] =
                binom(n, i) * binom(n, j) / binom(2 * n, i + j) / (2 * n + 1) as f64;
        }
    }
    g
}

/// Multiply two row-major matrices: A (m x k) * B (k x n) -> C (m x n).
pub fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Transpose a row-major matrix (m x n) -> (n x m).
pub fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * m];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// Compute G_tilde = (EDED)^T G (EDED) for degree N.
/// Returns (N+1) x (N+1) row-major, or None if N < 2.
pub fn get_g_tilde(n: usize) -> Option<Vec<f64>> {
    if n < 2 {
        return None;
    }
    let sz = n + 1;

    // D: N x (N+1)
    let d = get_d_matrix(n);
    // E: elevates from N-1 to N, shape N+1 x N
    let e = get_e_matrix(n - 1);

    // ED: (N+1 x N) * (N x (N+1)) = (N+1) x (N+1)
    let ed = matmul(&e, sz, n, &d, sz);
    // EDED: (N+1 x N+1) * (N+1 x N+1) = (N+1) x (N+1)
    let eded = matmul(&ed, sz, sz, &ed, sz);
    // G: (N+1) x (N+1)
    let g = get_g_matrix(n);
    // EDED^T: (N+1) x (N+1)
    let eded_t = transpose(&eded, sz, sz);
    // EDED^T * G
    let tmp = matmul(&eded_t, sz, sz, &g, sz);
    // (EDED^T * G) * EDED
    Some(matmul(&tmp, sz, sz, &eded, sz))
}

/// Bernstein basis weights for degree N at tau. Returns vec of length N+1.
pub fn bernstein_basis(n: usize, tau: f64) -> Vec<f64> {
    let mut b = vec![0.0; n + 1];
    for i in 0..=n {
        b[i] = binom(n, i) * tau.powi(i as i32) * (1.0 - tau).powi((n - i) as i32);
    }
    b
}

/// Bernstein derivative weights for degree N at tau.
/// Returns vec of length N+1 such that dp/dtau = sum_j w[j] * P_j.
pub fn bernstein_derivative_weights(n: usize, tau: f64) -> Vec<f64> {
    if n == 0 {
        return vec![0.0];
    }
    // Basis for degree N-1
    let nm1 = n - 1;
    let mut b_low = vec![0.0; n];
    for i in 0..n {
        b_low[i] = binom(nm1, i) * tau.powi(i as i32) * (1.0 - tau).powi((nm1 - i) as i32);
    }
    let nf = n as f64;
    let mut w = vec![0.0; n + 1];
    w[0] = -nf * b_low[0];
    for j in 1..n {
        w[j] = nf * (b_low[j - 1] - b_low[j]);
    }
    w[n] = nf * b_low[n - 1];
    w
}

/// Evaluate a Bézier curve at tau.
/// control_points: (N+1) x dim, row-major. Returns vec of length dim.
pub fn evaluate(control_points: &[f64], np1: usize, dim: usize, tau: f64) -> Vec<f64> {
    let n = np1 - 1;
    let basis = bernstein_basis(n, tau);
    let mut out = vec![0.0; dim];
    for i in 0..np1 {
        for d in 0..dim {
            out[d] += basis[i] * control_points[i * dim + d];
        }
    }
    out
}

/// Evaluate acceleration of a Bézier curve at tau using EDED matrix.
/// control_points: (N+1) x dim, row-major. Returns vec of length dim.
pub fn evaluate_acceleration(control_points: &[f64], np1: usize, dim: usize, tau: f64) -> Vec<f64> {
    let n = np1 - 1;
    if n < 2 {
        return vec![0.0; dim];
    }
    let sz = np1;
    // Compute EDED
    let d = get_d_matrix(n);
    let e = get_e_matrix(n - 1);
    let ed = matmul(&e, sz, n, &d, sz);
    let eded = matmul(&ed, sz, sz, &ed, sz);

    // Acceleration control points = EDED @ P
    let mut a_ctrl = vec![0.0; np1 * dim];
    for i in 0..np1 {
        for dd in 0..dim {
            let mut sum = 0.0;
            for j in 0..np1 {
                sum += eded[i * np1 + j] * control_points[j * dim + dd];
            }
            a_ctrl[i * dim + dd] = sum;
        }
    }

    // Evaluate using Bernstein basis of degree N
    let basis = bernstein_basis(n, tau);
    let mut out = vec![0.0; dim];
    for i in 0..np1 {
        for dd in 0..dim {
            out[dd] += basis[i] * a_ctrl[i * dim + dd];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binom() {
        assert!((binom(3, 0) - 1.0).abs() < 1e-15);
        assert!((binom(3, 1) - 3.0).abs() < 1e-15);
        assert!((binom(3, 2) - 3.0).abs() < 1e-15);
        assert!((binom(3, 3) - 1.0).abs() < 1e-15);
        assert!((binom(6, 3) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_d_matrix_n3() {
        let d = get_d_matrix(3);
        // 3x4 matrix
        assert_eq!(d.len(), 12);
        let expected = vec![
            -3.0, 3.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0, 0.0, 0.0, -3.0, 3.0,
        ];
        for (a, b) in d.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-15, "{a} != {b}");
        }
    }

    #[test]
    fn test_e_matrix_n2() {
        let e = get_e_matrix(2);
        // Elevates from degree 2 to 3: 4x3
        assert_eq!(e.len(), 12);
        let expected: Vec<f64> = vec![
            1.0, 0.0, 0.0,
            1.0 / 3.0, 2.0 / 3.0, 0.0,
            0.0, 2.0 / 3.0, 1.0 / 3.0,
            0.0, 0.0, 1.0,
        ];
        for (a, b) in e.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-15, "{a} != {b}");
        }
    }

    #[test]
    fn test_g_matrix_n3() {
        let g = get_g_matrix(3);
        assert_eq!(g.len(), 16);
        // Check G[0][0] = C(3,0)^2 / (C(6,0) * 7) = 1/7
        assert!((g[0] - 1.0 / 7.0).abs() < 1e-15);
    }

    #[test]
    fn test_bernstein_partition_of_unity() {
        for &tau in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let b = bernstein_basis(3, tau);
            let sum: f64 = b.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "tau={tau}: sum={sum}");
        }
    }
}
