/// Constraint building functions for KOZ and boundary conditions.

/// A linear constraint: lb <= A @ x <= ub.
/// A is stored row-major with shape (n_rows, n_vars).
#[derive(Clone)]
pub struct LinearConstraint {
    pub a: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    pub n_rows: usize,
    pub n_vars: usize,
}

/// How to handle a segment whose control-point centroid coincides with the KOZ
/// centre, where the direction `centroid - c_koz` has no defined orientation.
///
/// Soundness note: the half-space `n·q >= n·c_koz + r_e` is tangent to the sphere
/// for EVERY unit `n`, so any unit normal keeps the Prop-1 certificate sound — a
/// poorly aimed normal can only make the subproblem infeasible (a loud failure),
/// never certify a curve that intersects the KOZ. Emitting no rows is the only
/// option that breaks soundness, because the certificate is evaluated on these
/// same rows and a skipped segment is silently absent from it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DegenerateNormal {
    /// Emit no rows for the segment. Historical behaviour, kept as the default so
    /// existing results reproduce bit-for-bit. UNSOUND: the segment is both
    /// unconstrained and invisible to the feasibility certificate.
    Skip,
    /// Fall back to the deterministic direction `c_koz -> the segment control point
    /// farthest from c_koz`, and report the occurrence via the returned count.
    /// Deterministic rather than normalizing float noise, which would give a
    /// platform- and optimization-level-dependent direction.
    Fallback,
}

/// Build KOZ (Keep Out Zone) linear constraints for all segments.
///
/// For each segment j:
/// 1. Compute centroid of control points: Qi = Ai @ P
/// 2. Generate unit vector nj from c_koz to centroid
/// 3. Create half-space constraint: nj^T @ Qi >= r_e
///
/// Returns the constraint set and the number of segments that hit the degenerate
/// case (centroid on the KOZ centre); a nonzero count under `Skip` means the
/// certificate does not cover the whole curve.
pub fn build_koz_constraints(
    a_list: &[Vec<f64>],
    p: &[f64],     // (Np1, dim) row-major
    np1: usize,
    dim: usize,
    r_e: f64,
    c_koz: &[f64], // (dim,)
    degenerate: DegenerateNormal,
) -> (LinearConstraint, usize) {
    let n_vars = np1 * dim;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut lbs: Vec<f64> = Vec::new();
    let mut n_degenerate = 0usize;

    for a_seg in a_list {
        // Qi = Ai @ P, shape (Np1, dim)
        let mut qi = vec![0.0; np1 * dim];
        for i in 0..np1 {
            for d in 0..dim {
                let mut sum = 0.0;
                for j in 0..np1 {
                    sum += a_seg[i * np1 + j] * p[j * dim + d];
                }
                qi[i * dim + d] = sum;
            }
        }

        // Centroid
        let mut ci = vec![0.0; dim];
        for i in 0..np1 {
            for d in 0..dim {
                ci[d] += qi[i * dim + d];
            }
        }
        for d in 0..dim {
            ci[d] /= np1 as f64;
        }

        // Unit vector from c_koz to centroid
        let mut nj = vec![0.0; dim];
        for d in 0..dim {
            nj[d] = ci[d] - c_koz[d];
        }
        let mut nj_norm: f64 = nj.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nj_norm < 1e-12 {
            n_degenerate += 1;
            match degenerate {
                DegenerateNormal::Skip => continue,
                DegenerateNormal::Fallback => {
                    // Deterministic replacement: aim at the segment control point
                    // farthest from the KOZ centre. If every control point sits on
                    // the centre the segment is genuinely uncoverable by any
                    // half-space; a fixed axis then yields rows the QP reports as
                    // infeasible, which is the correct loud outcome.
                    let mut best = 0usize;
                    let mut best_d2 = -1.0f64;
                    for i in 0..np1 {
                        let d2: f64 = (0..dim)
                            .map(|d| (qi[i * dim + d] - c_koz[d]).powi(2))
                            .sum();
                        if d2 > best_d2 {
                            best_d2 = d2;
                            best = i;
                        }
                    }
                    if best_d2 > 1e-24 {
                        for d in 0..dim {
                            nj[d] = qi[best * dim + d] - c_koz[d];
                        }
                    } else {
                        nj[0] = 1.0;
                        for d in 1..dim {
                            nj[d] = 0.0;
                        }
                    }
                    nj_norm = nj.iter().map(|x| x * x).sum::<f64>().sqrt();
                }
            }
        }
        for d in 0..dim {
            nj[d] /= nj_norm;
        }

        // Constraint for each control point in segment
        for k in 0..np1 {
            let mut row = vec![0.0; n_vars];
            for j in 0..np1 {
                let coeff = a_seg[k * np1 + j];
                let start = j * dim;
                for d in 0..dim {
                    row[start + d] += coeff * nj[d];
                }
            }
            rows.push(row);

            // n^T c_koz + r_e
            let dot: f64 = (0..dim).map(|d| nj[d] * c_koz[d]).sum();
            lbs.push(dot + r_e);
        }
    }

    if rows.is_empty() {
        // Every segment was degenerate and skipped: the KOZ has silently vanished.
        // n_degenerate carries that fact to the caller.
        return (
            LinearConstraint {
                a: vec![0.0; n_vars],
                lb: vec![f64::NEG_INFINITY],
                ub: vec![f64::INFINITY],
                n_rows: 1,
                n_vars,
            },
            n_degenerate,
        );
    }

    let n_rows = rows.len();
    let mut a = vec![0.0; n_rows * n_vars];
    for (i, row) in rows.iter().enumerate() {
        a[i * n_vars..(i + 1) * n_vars].copy_from_slice(row);
    }
    let ub = vec![f64::INFINITY; n_rows];

    (
        LinearConstraint {
            a,
            lb: lbs,
            ub,
            n_rows,
            n_vars,
        },
        n_degenerate,
    )
}

/// Build boundary condition equality constraints.
///
/// Returns a list of LinearConstraints (each with n_rows = dim).
pub fn build_boundary_constraints(
    np1: usize,
    dim: usize,
    t: f64,
    v0: Option<&[f64]>,
    v1: Option<&[f64]>,
    a0: Option<&[f64]>,
    a1: Option<&[f64]>,
) -> Vec<LinearConstraint> {
    let n = np1 - 1;
    let n_vars = np1 * dim;
    let vel_scale = 1.0 / t;
    let accel_scale = 1.0 / (t * t);
    let nf = n as f64;
    let mut constraints = Vec::new();

    if let Some(v0) = v0 {
        // v(0) = (N/T) (P1 - P0)
        let mut a_mat = vec![0.0; dim * n_vars];
        for d in 0..dim {
            a_mat[d * n_vars + 0 * dim + d] = -nf * vel_scale;
            a_mat[d * n_vars + 1 * dim + d] = nf * vel_scale;
        }
        constraints.push(LinearConstraint {
            a: a_mat,
            lb: v0.to_vec(),
            ub: v0.to_vec(),
            n_rows: dim,
            n_vars,
        });
    }

    if let Some(v1) = v1 {
        // v(1) = (N/T) (PN - PN-1)
        let mut a_mat = vec![0.0; dim * n_vars];
        for d in 0..dim {
            a_mat[d * n_vars + (np1 - 2) * dim + d] = -nf * vel_scale;
            a_mat[d * n_vars + (np1 - 1) * dim + d] = nf * vel_scale;
        }
        constraints.push(LinearConstraint {
            a: a_mat,
            lb: v1.to_vec(),
            ub: v1.to_vec(),
            n_rows: dim,
            n_vars,
        });
    }

    if let Some(a0) = a0 {
        if n >= 2 {
            let mut a_mat = vec![0.0; dim * n_vars];
            let c = nf * (nf - 1.0) * accel_scale;
            for d in 0..dim {
                a_mat[d * n_vars + 0 * dim + d] = c;
                a_mat[d * n_vars + 1 * dim + d] = -2.0 * c;
                a_mat[d * n_vars + 2 * dim + d] = c;
            }
            constraints.push(LinearConstraint {
                a: a_mat,
                lb: a0.to_vec(),
                ub: a0.to_vec(),
                n_rows: dim,
                n_vars,
            });
        }
    }

    if let Some(a1) = a1 {
        if n >= 2 {
            let mut a_mat = vec![0.0; dim * n_vars];
            let c = nf * (nf - 1.0) * accel_scale;
            for d in 0..dim {
                a_mat[d * n_vars + (np1 - 3) * dim + d] = c;
                a_mat[d * n_vars + (np1 - 2) * dim + d] = -2.0 * c;
                a_mat[d * n_vars + (np1 - 1) * dim + d] = c;
            }
            constraints.push(LinearConstraint {
                a: a_mat,
                lb: a1.to_vec(),
                ub: a1.to_vec(),
                n_rows: dim,
                n_vars,
            });
        }
    }

    constraints
}

/// Exact KOZ clearance values g_k(x) for every (segment, control point) row, in the
/// same row order as `build_koz_constraints`.
///
/// ```text
/// g_k(x) = n_j(x) . (q_k(x) - c_koz) - r_e,   n_j(x) = v/||v||,
/// v = centroid_j(x) - c_koz
/// ```
///
/// g_k >= 0 for all k is exactly the Proposition-1 hull certificate. Degenerate
/// segments (centroid on the KOZ centre) are skipped, matching `DegenerateNormal::Skip`.
pub fn koz_clearances(
    a_list: &[Vec<f64>],
    x: &[f64],
    np1: usize,
    dim: usize,
    r_e: f64,
    c_koz: &[f64],
) -> Vec<f64> {
    let mut out = Vec::new();
    for a_seg in a_list {
        let mut qi = vec![0.0; np1 * dim];
        for i in 0..np1 {
            for d in 0..dim {
                let mut s = 0.0;
                for j in 0..np1 {
                    s += a_seg[i * np1 + j] * x[j * dim + d];
                }
                qi[i * dim + d] = s;
            }
        }
        let mut ci = vec![0.0; dim];
        for i in 0..np1 {
            for d in 0..dim {
                ci[d] += qi[i * dim + d];
            }
        }
        for d in 0..dim {
            ci[d] /= np1 as f64;
        }
        let v: Vec<f64> = (0..dim).map(|d| ci[d] - c_koz[d]).collect();
        let vn = v.iter().map(|t| t * t).sum::<f64>().sqrt();
        if vn < 1e-12 {
            continue;
        }
        let n: Vec<f64> = v.iter().map(|t| t / vn).collect();
        for k in 0..np1 {
            let s_k: f64 = (0..dim).map(|d| n[d] * (qi[k * dim + d] - c_koz[d])).sum();
            out.push(s_k - r_e);
        }
    }
    out
}

#[cfg(test)]
mod koz_row_tests {
    use super::*;
    use crate::de_casteljau;

    fn sample_polygon(np1: usize) -> Vec<f64> {
        let mut p = vec![0.0; np1 * 3];
        for i in 0..np1 {
            let s = i as f64 / (np1 - 1) as f64;
            let ang = 0.9 * s;
            let r = 6900.0 + 250.0 * s;
            p[i * 3] = r * ang.cos();
            p[i * 3 + 1] = r * ang.sin();
            p[i * 3 + 2] = 120.0 * s * (1.0 - s);
        }
        p
    }

    /// The rows must reproduce the exact clearance AT the reference point, so the
    /// merit's KOZ term is exact at x = p (this is what makes `vtrue_p == vlin_p`
    /// in the ratio test rather than an approximation).
    #[test]
    fn koz_rows_exact_at_reference() {
        let (np1, dim, r_e) = (8usize, 3usize, 6471.0);
        let c_koz = vec![0.0; dim];
        let nvars = np1 * dim;
        let p = sample_polygon(np1);
        let a_list = de_casteljau::segment_matrices_equal_params(np1 - 1, 4);
        let g_p = koz_clearances(&a_list, &p, np1, dim, r_e, &c_koz);
        let lc = build_koz_constraints(
            &a_list, &p, np1, dim, r_e, &c_koz, DegenerateNormal::Skip,
        ).0;
        assert_eq!(g_p.len(), lc.n_rows, "row count mismatch");
        for r in 0..lc.n_rows {
            let model: f64 =
                (0..nvars).map(|i| lc.a[r * nvars + i] * p[i]).sum::<f64>() - lc.lb[r];
            let rel = (model - g_p[r]).abs() / g_p[r].abs().max(1.0);
            assert!(rel < 1e-9, "row {r} wrong at reference: {model} vs {}", g_p[r]);
        }
    }

    /// SOUNDNESS OF THE WITNESS ARGUMENT (design_freeze section 9).
    ///
    /// The whole of Reading B rests on this: rows built at a reference `p` remain a
    /// valid Proposition-1 certificate for ANY other point `x` that satisfies them.
    /// The normal is a unit vector, so `n . (q - c_koz) <= ||q - c_koz||`; a satisfied
    /// row therefore lower-bounds the true distance regardless of which iterate chose
    /// `n`. If this test fails, grading the candidate on the reference's rows would be
    /// unsound and the merit must go back to rebuilding rows at the candidate.
    #[test]
    fn rows_built_at_reference_certify_any_satisfying_point() {
        let (np1, dim, r_e) = (8usize, 3usize, 6471.0);
        let c_koz = vec![0.0; dim];
        let nvars = np1 * dim;
        let p = sample_polygon(np1);

        for &n_seg in &[1usize, 4, 16] {
            let a_list = de_casteljau::segment_matrices_equal_params(np1 - 1, n_seg);
            let lc = build_koz_constraints(
                &a_list, &p, np1, dim, r_e, &c_koz, DegenerateNormal::Skip,
            ).0;

            // Displace far enough that the centroid rule would pick very different
            // normals at x than it did at p.
            for scale in [0.0f64, 50.0, 400.0, 2000.0] {
                let x: Vec<f64> = (0..nvars)
                    .map(|i| {
                        let s = (i % 7) as f64 / 7.0 - 0.5;
                        p[i] + scale * s
                    })
                    .collect();

                // Every row of the reference-built set that x satisfies must imply the
                // corresponding subdivided control point of x clears the sphere.
                let mut checked = 0usize;
                for (seg, a_seg) in a_list.iter().enumerate() {
                    for k in 0..np1 {
                        let r = seg * np1 + k;
                        if r >= lc.n_rows {
                            continue;
                        }
                        let ax: f64 = (0..nvars).map(|i| lc.a[r * nvars + i] * x[i]).sum();
                        if ax < lc.lb[r] {
                            continue; // row violated: nothing is claimed
                        }
                        // q_k(x) for this segment
                        let mut q = vec![0.0; dim];
                        for d in 0..dim {
                            for j in 0..np1 {
                                q[d] += a_seg[k * np1 + j] * x[j * dim + d];
                            }
                        }
                        let dist = (0..dim)
                            .map(|d| (q[d] - c_koz[d]).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        assert!(
                            dist >= r_e - 1e-6,
                            "reference-built row {r} satisfied at scale {scale} \
                             (n_seg={n_seg}) but the point is INSIDE the KOZ: \
                             dist={dist} < r_e={r_e}"
                        );
                        checked += 1;
                    }
                }
                assert!(checked > 0, "test vacuous at scale {scale}, n_seg={n_seg}");
            }
        }
    }
}
