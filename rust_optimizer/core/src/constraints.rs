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

/// Build SELF-CONSISTENT KOZ rows about the reference `p`.
///
/// `build_koz_constraints` fixes the normal at the reference. Those rows are sound
/// (satisfying them certifies the curve, whoever aimed the normal) but they describe
/// a half-space that will no longer be the one the centroid rule picks once the
/// solver has moved. A step optimized against them lands flush on a plane that then
/// pivots out from under it: measured on phase120/n_seg=16, the accepted iterate
/// satisfies its rows exactly (violation 0.0) yet misses the rows rebuilt at itself
/// by 68 m -- while clearing the KOZ sphere by 16.0 km. That phantom violation is
/// pure pivot: 455 km of lateral reach along the plane times a 31 arcsec rotation.
///
/// These rows add the term that makes the subproblem ANTICIPATE the pivot, so its
/// optimum is a point the centroid rule still agrees with. That is a self-consistency
/// device for the step, NOT a physical model: the KOZ is a sphere and the half-space
/// is the convexification of it, so there is no "truer" constraint being approximated
/// here (design_freeze.md section 9). Its purpose is to keep the point the QP returns
/// on the same footing as the point the next iteration will grade.
///
/// With c = centroid_j, v = c - c_koz, n = v/||v||, d_k = q_k - c_koz, s_k = n.d_k,
/// and w_m the centroid weights (w_m = mean_i A[i][m]):
///
/// ```text
/// dn/dP[m]    = (w_m/||v||) (I - n n^T)
/// grad g_k[m] = A[k][m] n + (w_m/||v||) (d_k - s_k n)
/// ```
///
/// The row is grad g_k(p) and the bound is grad g_k(p).p - g_k(p), so the constraint
/// grad g_k.x >= grad g_k(p).p - g_k(p) reproduces g_k(p) >= 0 exactly at x = p.
///
/// NOTE this row is a first-order model, NOT a conservative restriction: a point
/// satisfying it may have g_k(x) < 0. Soundness of the REPORTED guarantee is
/// unaffected -- the certificate is always evaluated with the exact rows via
/// `koz_clearances` / `build_koz_constraints`.
pub fn build_koz_constraints_linearized(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    r_e: f64,
    c_koz: &[f64],
    degenerate: DegenerateNormal,
) -> (LinearConstraint, usize) {
    let n_vars = np1 * dim;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut lbs: Vec<f64> = Vec::new();
    let mut n_degenerate = 0usize;

    for a_seg in a_list {
        // Segment control points q_i = (A_seg P)_i
        let mut qi = vec![0.0; np1 * dim];
        for i in 0..np1 {
            for d in 0..dim {
                let mut s = 0.0;
                for j in 0..np1 {
                    s += a_seg[i * np1 + j] * p[j * dim + d];
                }
                qi[i * dim + d] = s;
            }
        }
        // Centroid weights w_m = mean_i A[i][m], and the centroid itself.
        let mut w_row = vec![0.0; np1];
        for i in 0..np1 {
            for m in 0..np1 {
                w_row[m] += a_seg[i * np1 + m];
            }
        }
        for m in 0..np1 {
            w_row[m] /= np1 as f64;
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

        let mut v: Vec<f64> = (0..dim).map(|d| ci[d] - c_koz[d]).collect();
        let mut vn = v.iter().map(|t| t * t).sum::<f64>().sqrt();
        if vn < 1e-12 {
            n_degenerate += 1;
            match degenerate {
                DegenerateNormal::Skip => continue,
                DegenerateNormal::Fallback => {
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
                            v[d] = qi[best * dim + d] - c_koz[d];
                        }
                    } else {
                        v[0] = 1.0;
                        for d in 1..dim {
                            v[d] = 0.0;
                        }
                    }
                    vn = v.iter().map(|t| t * t).sum::<f64>().sqrt();
                }
            }
        }
        let n: Vec<f64> = v.iter().map(|t| t / vn).collect();

        for k in 0..np1 {
            let d_k: Vec<f64> = (0..dim).map(|d| qi[k * dim + d] - c_koz[d]).collect();
            let s_k: f64 = (0..dim).map(|d| n[d] * d_k[d]).sum();
            // t = (d_k - s_k n) / ||v||  -- the normal-rotation contribution
            let t_vec: Vec<f64> = (0..dim).map(|d| (d_k[d] - s_k * n[d]) / vn).collect();

            let mut row = vec![0.0; n_vars];
            for m in 0..np1 {
                let a_km = a_seg[k * np1 + m];
                let w_m = w_row[m];
                let base = m * dim;
                for d in 0..dim {
                    row[base + d] += a_km * n[d] + w_m * t_vec[d];
                }
            }
            // lb = grad.p - g_k(p), with g_k(p) = s_k - r_e
            let grad_dot_p: f64 = (0..n_vars).map(|i| row[i] * p[i]).sum();
            lbs.push(grad_dot_p - (s_k - r_e));
            rows.push(row);
        }
    }

    if rows.is_empty() {
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
        LinearConstraint { a, lb: lbs, ub, n_rows, n_vars },
        n_degenerate,
    )
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

    fn direction(nvars: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..nvars)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            })
            .collect()
    }

    /// The self-consistent rows must be the actual first-order Taylor model of the
    /// exact clearance g_k, i.e. the residual must shrink QUADRATICALLY in the step.
    ///
    /// This is the discriminating test. The frozen-normal rows of
    /// `build_koz_constraints` reproduce g_k(p) exactly at p but drop the dn/dx term,
    /// so their residual shrinks only LINEARLY. Halving t must therefore quarter the
    /// self-consistent residual while merely halving the frozen-normal one -- and that
    /// order gap is what decides whether the QP's optimum is a point the centroid rule
    /// still agrees with after it re-aims.
    #[test]
    fn linearized_koz_rows_are_second_order_accurate() {
        let (np1, dim, r_e) = (8usize, 3usize, 6471.0);
        let c_koz = vec![0.0; dim];
        let nvars = np1 * dim;
        let p = sample_polygon(np1);

        for &n_seg in &[1usize, 4, 8] {
            let a_list = de_casteljau::segment_matrices_equal_params(np1 - 1, n_seg);
            let (lin, _) = build_koz_constraints_linearized(
                &a_list, &p, np1, dim, r_e, &c_koz, DegenerateNormal::Skip,
            );
            let (froz, _) = build_koz_constraints(
                &a_list, &p, np1, dim, r_e, &c_koz, DegenerateNormal::Skip,
            );
            let g_p = koz_clearances(&a_list, &p, np1, dim, r_e, &c_koz);
            assert_eq!(g_p.len(), lin.n_rows, "row count mismatch");

            for seed in 0..3u64 {
                let e = direction(nvars, seed + 1);
                let mut prev_lin = f64::NAN;
                let mut prev_froz = f64::NAN;
                // Two step sizes a factor of 2 apart. Range matters: the measured
                // order is a clean 4.000 from t = 5e-2 down to ~1.5e-3, then decays
                // (3.10 at t = 3.9e-4) as the residual reaches the f64 noise floor
                // against a ~6900 km coordinate scale. These values sit in the
                // asymptotic regime with margin at both ends.
                for &t in &[2.5e-2f64, 1.25e-2] {
                    let x: Vec<f64> = (0..nvars).map(|i| p[i] + t * e[i]).collect();
                    let g_x = koz_clearances(&a_list, &x, np1, dim, r_e, &c_koz);

                    let mut err_lin: f64 = 0.0;
                    let mut err_froz: f64 = 0.0;
                    for r in 0..lin.n_rows {
                        let model_lin: f64 = (0..nvars)
                            .map(|i| lin.a[r * nvars + i] * x[i])
                            .sum::<f64>()
                            - lin.lb[r];
                        let model_froz: f64 = (0..nvars)
                            .map(|i| froz.a[r * nvars + i] * x[i])
                            .sum::<f64>()
                            - froz.lb[r];
                        err_lin = err_lin.max((model_lin - g_x[r]).abs());
                        err_froz = err_froz.max((model_froz - g_x[r]).abs());
                    }

                    if prev_lin.is_finite() {
                        // Halving t: quadratic error falls ~4x, linear only ~2x.
                        let ratio_lin = prev_lin / err_lin.max(1e-300);
                        let ratio_froz = prev_froz / err_froz.max(1e-300);
                        assert!(
                            ratio_lin > 3.2,
                            "self-consistent rows are not 2nd order (n_seg={n_seg} seed={seed}): \
                             error ratio {ratio_lin:.3} on halving t (expect ~4)"
                        );
                        assert!(
                            ratio_froz < 3.0,
                            "frozen-normal rows unexpectedly 2nd order (n_seg={n_seg}): \
                             ratio {ratio_froz:.3} (expect ~2)"
                        );
                        assert!(
                            err_lin < err_froz,
                            "self-consistent model no better than frozen: {err_lin:.3e} vs {err_froz:.3e}"
                        );
                    }
                    prev_lin = err_lin;
                    prev_froz = err_froz;
                }
            }
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
