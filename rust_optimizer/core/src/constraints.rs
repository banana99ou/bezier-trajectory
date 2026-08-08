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
