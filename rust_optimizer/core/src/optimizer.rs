/// SCP optimization loop for orbital docking trajectories.
use crate::bezier;
use crate::constraints;
use crate::de_casteljau;
use crate::gravity;
use std::collections::HashMap;

/// Orbital constants (matching Python constants.py).
pub struct OrbitalConstants {
    pub mu: f64,
    pub r_e_km: f64,
    pub j2: f64,
}

impl Default for OrbitalConstants {
    fn default() -> Self {
        Self {
            mu: 398600.4418,     // km^3/s^2
            r_e_km: 6371.0,     // km
            j2: 0.00108262668,
        }
    }
}

/// Result from optimization.
pub struct OptResult {
    pub p_opt: Vec<f64>,     // (Np1, dim) row-major
    pub np1: usize,
    pub dim: usize,
    pub info: HashMap<String, f64>,
    pub feasible: bool,
    pub iterations: usize,
    /// Per-iteration, per-segment Frobenius drift of the gravity Jacobian
    /// relative to its iter-1 value. Outer index = SCP iteration (0-based,
    /// iter 1 is index 0 and is identically zero), inner index = segment.
    pub jacobian_drift_history: Vec<Vec<f64>>,
    /// Per-accepted-outer-step SCvx diagnostics (trust path only). One entry per
    /// accepted step. `rho_history` is the penalized-merit ratio (actual/predicted
    /// reduction; +inf marks a null step where the model saw no improvement).
    /// `merit_history` is the true penalized merit T(x) after the step.
    /// `phase_history`: 0 = iterate still violates the KOZ control-point condition,
    /// 1 = iterate satisfies it (informational; acceptance is single-phase).
    pub rho_history: Vec<f64>,
    pub trust_history: Vec<f64>,
    pub merit_history: Vec<f64>,
    pub step_norm_history: Vec<f64>,
    pub slack_history: Vec<f64>,
    pub phase_history: Vec<f64>,
}

/// Per-segment gravity linearization quantities.
/// For a given reference control-polygon P_ref, each segment's centroid r_i has the
/// affine gravity model g(r) ≈ J_i r + c_i, applied across the segment's whole
/// parameter subinterval by the exact-integral objective.
#[derive(Clone)]
struct SegmentGravLin {
    j_i: [[f64; 3]; 3], // gravity Jacobian at r_ref (also drift diagnostic)
    c_i: [f64; 3],      // affine offset: g_ref - J_i r_ref
    a_seg: Vec<f64>,    // (np1 x np1) De Casteljau segment matrix for this subinterval
}

fn frobenius_norm_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let d = a[i][j] - b[i][j];
            s += d * d;
        }
    }
    s.sqrt()
}

/// Compute the per-segment gravity linearization at a reference control polygon.
fn compute_segment_lin(
    p_ref: &[f64],
    np1: usize,
    dim: usize,
    _t: f64,
    n_lin_seg: usize,
    consts: &OrbitalConstants,
) -> Vec<SegmentGravLin> {
    let n = np1 - 1;
    let n_lin_seg = n_lin_seg.max(1);
    let a_seg_list = de_casteljau::segment_matrices_equal_params(n, n_lin_seg);

    let mut out = Vec::with_capacity(a_seg_list.len());

    for a_seg in &a_seg_list {
        // w_row = mean of rows of A_seg (segment centroid weights)
        let mut w_row = vec![0.0; np1];
        for i in 0..np1 {
            for j in 0..np1 {
                w_row[j] += a_seg[i * np1 + j];
            }
        }
        for j in 0..np1 {
            w_row[j] /= np1 as f64;
        }

        // r_ref = sum_j w_row[j] * P_ref[j, :]
        let mut r_ref = [0.0f64; 3];
        for j in 0..np1 {
            for d in 0..dim {
                r_ref[d] += w_row[j] * p_ref[j * dim + d];
            }
        }

        let g_ref = gravity::accel_total(&r_ref, consts.mu, consts.r_e_km, consts.j2);
        let j_i = gravity::jacobian_analytic(&r_ref, consts.mu, consts.r_e_km, consts.j2);

        let mut c_i = [0.0f64; 3];
        for row in 0..3 {
            let mut jr = 0.0;
            for col in 0..3 {
                jr += j_i[row][col] * r_ref[col];
            }
            c_i[row] = g_ref[row] - jr;
        }

        out.push(SegmentGravLin { j_i, c_i, a_seg: a_seg.clone() });
    }
    out
}

/// Build quadratic objective: 0.5 x^T H x + f^T x + c
/// for control acceleration energy with gravity+J2 linearized about P_ref.
///
/// If `precomputed_lin` is provided, the gravity linearization (J_i, c_i, M_i)
/// is taken from that slice instead of being recomputed from `p_ref`.
fn build_ctrl_accel_quadratic(
    p_ref: &[f64],  // (Np1, dim) row-major
    np1: usize,
    dim: usize,
    t: f64,
    n_lin_seg: usize,
    consts: &OrbitalConstants,
    precomputed_lin: Option<&[SegmentGravLin]>,
) -> (Vec<f64>, Vec<f64>, f64) {
    let nvars = np1 * dim;

    let mut big_q = vec![0.0; nvars * nvars];
    let mut q_vec = vec![0.0; nvars];
    let mut c_const = 0.0f64;

    // DESIGN DECISION (author, 2026-08-07 — do not revisit without explicit user
    // approval): the objective is control-acceleration energy and NOTHING else,
    // computed as the EXACT integral (closed form via the Bernstein Gram matrix —
    // the point of the control-point-space formulation):
    //     J = ∫₀¹ || a_geom(τ)/T² − g_lin(r(τ)) ||² dτ
    // with gravity affine per De Casteljau segment (J_s, c_s at the segment
    // centroid). Per segment, the integrand is a degree-N Bernstein polynomial
    // squared, so ∫ = Σ_{k,l} G_{kl} f_k·f_l exactly — no sampling, hence no
    // quadrature soft directions and no aliasing. No mode switches, no flags, no
    // auxiliary terms (the historical full-weight smoothness term biased the
    // optimum; its apparent benefit was a premature-stop artifact).

    let n = np1 - 1;

    // Acceleration operator L = E·D·E·D: a(τ) is the degree-N curve with control
    // points (L P), in units of d²r/dτ² (divide by T² for physical acceleration).
    let d_mat = bezier::get_d_matrix(n);
    let e_mat = bezier::get_e_matrix(n - 1);
    let ed = bezier::matmul(&e_mat, np1, n, &d_mat, np1);
    let l_mat = bezier::matmul(&ed, np1, np1, &ed, np1);
    let g_mat = bezier::get_g_matrix(n); // (np1 x np1) Bernstein Gram: ∫ B_k B_l dτ

    // Gravity/J2 linearization (computed fresh or reused from precomputed_lin)
    let owned_lin: Vec<SegmentGravLin>;
    let lin: &[SegmentGravLin] = match precomputed_lin {
        Some(p) => p,
        None => {
            owned_lin = compute_segment_lin(p_ref, np1, dim, t, n_lin_seg, consts);
            &owned_lin[..]
        }
    };

    let n_lin_seg = lin.len().max(1);
    let w_seg = 1.0 / n_lin_seg as f64; // dτ = du / n_seg on each subinterval
    let t2_inv = 1.0 / (t * t);

    // Per segment s: residual control points f_k = Σ_j (U_kj I − V_kj J_s) p_j − c_s
    // with U = W_s L / T², V = W_s (W_s = De Casteljau segment matrix). Then
    //   ∫_seg ‖f(u)‖² du = Σ_{kl} G_{kl} f_k·f_l
    //     = xᵀ[Σ_{jj'} (P1_{jj'} I − P2_{jj'} J − P2_{j'j} Jᵀ + P3_{jj'} JᵀJ)]x
    //       − 2 Σ_j (u_j c − v_j Jᵀc)·p_j + (1ᵀG1) cᵀc
    // where P1 = UᵀGU, P2 = UᵀGV, P3 = VᵀGV, u = gᵀU, v = gᵀV, g_k = Σ_l G_{kl}.
    for seg in lin {
        let w_s = &seg.a_seg;
        let jm = &seg.j_i;
        let c_i = &seg.c_i;

        // U = (W_s · L) / T², V = W_s
        let wl = bezier::matmul(w_s, np1, np1, &l_mat, np1);
        let u_mat: Vec<f64> = wl.iter().map(|x| x * t2_inv).collect();
        let v_mat = w_s;

        // P1 = UᵀGU, P2 = UᵀGV, P3 = VᵀGV  (np1 x np1 each)
        let gu = bezier::matmul(&g_mat, np1, np1, &u_mat, np1);
        let gv = bezier::matmul(&g_mat, np1, np1, v_mat, np1);
        let mut p1 = vec![0.0; np1 * np1];
        let mut p2 = vec![0.0; np1 * np1];
        let mut p3 = vec![0.0; np1 * np1];
        for j in 0..np1 {
            for jp in 0..np1 {
                let (mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0);
                for k in 0..np1 {
                    s1 += u_mat[k * np1 + j] * gu[k * np1 + jp];
                    s2 += u_mat[k * np1 + j] * gv[k * np1 + jp];
                    s3 += v_mat[k * np1 + j] * gv[k * np1 + jp];
                }
                p1[j * np1 + jp] = s1;
                p2[j * np1 + jp] = s2;
                p3[j * np1 + jp] = s3;
            }
        }

        // 3x3 helpers: J, Jᵀ, JᵀJ, Jᵀc
        let mut jtj = [[0.0f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mut s = 0.0;
                for r in 0..3 {
                    s += jm[r][a] * jm[r][b];
                }
                jtj[a][b] = s;
            }
        }
        let mut jtc = [0.0f64; 3];
        for a in 0..3 {
            for r in 0..3 {
                jtc[a] += jm[r][a] * c_i[r];
            }
        }

        // Quadratic blocks
        for j in 0..np1 {
            for jp in 0..np1 {
                let a1 = p1[j * np1 + jp];
                let b2 = p2[j * np1 + jp];
                let c2 = p2[jp * np1 + j];
                let d3 = p3[j * np1 + jp];
                for a in 0..dim {
                    for b in 0..dim {
                        let mut val = 0.0;
                        if a == b {
                            val += a1;
                        }
                        val -= b2 * jm[a][b];
                        val -= c2 * jm[b][a];
                        val += d3 * jtj[a][b];
                        big_q[(j * dim + a) * nvars + (jp * dim + b)] += w_seg * val;
                    }
                }
            }
        }

        // Linear + constant terms: g_k = Σ_l G_kl; u_j = Σ_k g_k U_kj; v_j = Σ_k g_k V_kj
        let mut g_row = vec![0.0; np1];
        let mut g_total = 0.0;
        for k in 0..np1 {
            let mut s = 0.0;
            for l in 0..np1 {
                s += g_mat[k * np1 + l];
            }
            g_row[k] = s;
            g_total += s;
        }
        for j in 0..np1 {
            let (mut uj, mut vj) = (0.0, 0.0);
            for k in 0..np1 {
                uj += g_row[k] * u_mat[k * np1 + j];
                vj += g_row[k] * v_mat[k * np1 + j];
            }
            for a in 0..dim {
                q_vec[j * dim + a] += w_seg * (-(uj * c_i[a] - vj * jtc[a]));
            }
        }
        c_const += w_seg * g_total * (c_i[0] * c_i[0] + c_i[1] * c_i[1] + c_i[2] * c_i[2]);
    }

    // H = 2Q, f = 2q
    let h: Vec<f64> = big_q.iter().map(|x| 2.0 * x).collect();
    let f: Vec<f64> = q_vec.iter().map(|x| 2.0 * x).collect();
    (h, f, c_const)
}

/// Solve a QP: min 0.5 x^T P x + q^T x  s.t.  l <= Ax <= u
/// using Clarabel (interior-point conic solver). Returns the solution x.
fn solve_qp(
    h: &[f64],         // (n, n) row-major
    f: &[f64],         // (n,)
    constraints_a: &[f64], // (m, n) row-major
    constraints_lb: &[f64], // (m,)
    constraints_ub: &[f64], // (m,)
    n: usize,
    m: usize,
    // Incremented when Clarabel terminates on its REDUCED tolerances rather than the
    // requested ones. Such a solution feeds the merit/ratio test as if exact, so the
    // count must reach the caller instead of being discarded with the solver handle.
    almost_solved: &mut usize,
) -> Option<Vec<f64>> {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus};

    // Build P as upper-triangular CSC
    let mut p_col_ptr = vec![0usize; n + 1];
    let mut p_row_idx = Vec::new();
    let mut p_vals = Vec::new();
    for col in 0..n {
        for row in 0..=col {
            let val = h[row * n + col];
            if val.abs() > 1e-20 {
                p_row_idx.push(row);
                p_vals.push(val);
            }
        }
        p_col_ptr[col + 1] = p_row_idx.len();
    }
    let p_csc = CscMatrix::new(n, n, p_col_ptr, p_row_idx, p_vals);

    // Clarabel handles l <= Ax <= u as two sets of conic constraints:
    //   Ax - l >= 0  (nonneg cone)  =>  Ax >= l
    //   u - Ax >= 0  (nonneg cone)  =>  Ax <= u
    // For equality constraints (lb == ub), use zero cone.
    // For one-sided (ub = inf), only need the lower bound constraint.

    // Separate into equality and inequality constraints
    let mut a_rows: Vec<Vec<(usize, f64)>> = Vec::new(); // sparse rows
    let mut b_vals: Vec<f64> = Vec::new();
    let mut cones: Vec<clarabel::solver::SupportedConeT<f64>> = Vec::new();

    let mut n_eq = 0usize;
    let mut n_ineq = 0usize;

    // First pass: equality constraints (lb == ub)
    for row in 0..m {
        if (constraints_lb[row] - constraints_ub[row]).abs() < 1e-12 {
            let mut sparse_row = Vec::new();
            for col in 0..n {
                let val = constraints_a[row * n + col];
                if val.abs() > 1e-20 {
                    sparse_row.push((col, val));
                }
            }
            a_rows.push(sparse_row);
            b_vals.push(constraints_lb[row]);
            n_eq += 1;
        }
    }
    if n_eq > 0 {
        cones.push(clarabel::solver::SupportedConeT::ZeroConeT(n_eq));
    }

    // Second pass: inequality constraints (lb <= Ax, ub may be inf)
    for row in 0..m {
        if (constraints_lb[row] - constraints_ub[row]).abs() >= 1e-12 {
            // Lower bound: Ax - lb >= 0
            if constraints_lb[row].is_finite() {
                let mut sparse_row = Vec::new();
                for col in 0..n {
                    let val = constraints_a[row * n + col];
                    if val.abs() > 1e-20 {
                        sparse_row.push((col, val));
                    }
                }
                a_rows.push(sparse_row);
                b_vals.push(constraints_lb[row]);
                n_ineq += 1;
            }
            // Upper bound: ub - Ax >= 0 => -Ax + ub >= 0
            if constraints_ub[row].is_finite() {
                let mut sparse_row = Vec::new();
                for col in 0..n {
                    let val = constraints_a[row * n + col];
                    if val.abs() > 1e-20 {
                        sparse_row.push((col, -val));
                    }
                }
                a_rows.push(sparse_row);
                b_vals.push(-constraints_ub[row]);
                n_ineq += 1;
            }
        }
    }
    if n_ineq > 0 {
        cones.push(clarabel::solver::SupportedConeT::NonnegativeConeT(n_ineq));
    }

    // Build A as CSC
    // Clarabel convention: A x + s = b, s in cone
    // For zero cone: Ax = b (equality)
    // For nonneg cone: Ax + s = b, s >= 0 => Ax <= b
    // So for equality Ax = lb: we store A as-is, b = lb
    // For inequality Ax >= lb: we need -Ax + s = -lb, s >= 0 => Ax >= lb
    // Wait, let me re-read Clarabel's convention...
    // Clarabel: min 0.5 x^T P x + q^T x  s.t.  Ax + s = b, s in K
    // ZeroCone: s = 0 => Ax = b
    // NonnegCone: s >= 0 => Ax <= b (since Ax + s = b => Ax = b - s <= b)
    // So for Ax >= lb: -Ax <= -lb => need rows of -A and b = -lb in NonnegCone

    // Rebuild with correct sign conventions
    let mut a_rows2: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut b_vals2: Vec<f64> = Vec::new();
    let mut cones2: Vec<clarabel::solver::SupportedConeT<f64>> = Vec::new();
    let mut n_eq2 = 0usize;
    let mut n_ineq2 = 0usize;

    // Equalities first
    for row in 0..m {
        if (constraints_lb[row] - constraints_ub[row]).abs() < 1e-12 {
            let mut sparse_row = Vec::new();
            for col in 0..n {
                let val = constraints_a[row * n + col];
                if val.abs() > 1e-20 {
                    sparse_row.push((col, val));
                }
            }
            a_rows2.push(sparse_row);
            b_vals2.push(constraints_lb[row]);
            n_eq2 += 1;
        }
    }
    if n_eq2 > 0 {
        cones2.push(clarabel::solver::SupportedConeT::ZeroConeT(n_eq2));
    }

    // Inequalities: Ax >= lb  =>  -Ax <= -lb  =>  -A row, b = -lb in NonnegCone
    for row in 0..m {
        if (constraints_lb[row] - constraints_ub[row]).abs() >= 1e-12 {
            if constraints_lb[row].is_finite() {
                let mut sparse_row = Vec::new();
                for col in 0..n {
                    let val = constraints_a[row * n + col];
                    if val.abs() > 1e-20 {
                        sparse_row.push((col, -val)); // negate for Ax >= lb
                    }
                }
                a_rows2.push(sparse_row);
                b_vals2.push(-constraints_lb[row]);
                n_ineq2 += 1;
            }
            if constraints_ub[row].is_finite() {
                let mut sparse_row = Vec::new();
                for col in 0..n {
                    let val = constraints_a[row * n + col];
                    if val.abs() > 1e-20 {
                        sparse_row.push((col, val)); // Ax <= ub as-is
                    }
                }
                a_rows2.push(sparse_row);
                b_vals2.push(constraints_ub[row]);
                n_ineq2 += 1;
            }
        }
    }
    if n_ineq2 > 0 {
        cones2.push(clarabel::solver::SupportedConeT::NonnegativeConeT(n_ineq2));
    }

    let total_rows2 = a_rows2.len();

    // Build CSC from sparse rows
    let mut a_col_ptr = vec![0usize; n + 1];
    // Count entries per column
    for row_data in &a_rows2 {
        for &(col, _) in row_data {
            a_col_ptr[col + 1] += 1;
        }
    }
    // Prefix sum
    for col in 0..n {
        a_col_ptr[col + 1] += a_col_ptr[col];
    }
    let nnz = a_col_ptr[n];
    let mut a_row_indices = vec![0usize; nnz];
    let mut a_values = vec![0.0f64; nnz];
    let mut col_pos = a_col_ptr[..n].to_vec();
    for (row_idx, row_data) in a_rows2.iter().enumerate() {
        for &(col, val) in row_data {
            let pos = col_pos[col];
            a_row_indices[pos] = row_idx;
            a_values[pos] = val;
            col_pos[col] += 1;
        }
    }

    let a_csc = CscMatrix::new(total_rows2, n, a_col_ptr, a_row_indices, a_values);

    // EXPERIMENT (2026-08-09): Clarabel's default gap tolerances are ~1e-8 ABSOLUTE,
    // while the objective here is ~2e-5, i.e. only ~4e-4 relative — four orders
    // coarser than the 1e-8 RELATIVE merit change the outer convergence test asks
    // for. Tightened to see whether the endgame is solver-resolution-limited.
    let settings = DefaultSettingsBuilder::default()
        .max_iter(200)
        .tol_gap_abs(1e-11)
        .tol_gap_rel(1e-10)
        .tol_feas(1e-9)
        .verbose(false)
        .build()
        .unwrap();

    let mut solver = match DefaultSolver::new(&p_csc, f, &a_csc, &b_vals2, &cones2, settings) {
        Ok(s) => s,
        Err(_) => return None,
    };
    solver.solve();

    match solver.solution.status {
        SolverStatus::Solved => Some(solver.solution.x.clone()),
        SolverStatus::AlmostSolved => {
            *almost_solved += 1;
            Some(solver.solution.x.clone())
        }
        _ => None,
    }
}

/// Evaluate 0.5 x^T H x + f^T x (constant term omitted — it cancels in differences).
fn quad_form(h: &[f64], f: &[f64], x: &[f64], n: usize) -> f64 {
    let mut val = 0.0;
    for i in 0..n {
        let base = i * n;
        let mut hx = 0.0;
        for j in 0..n {
            hx += h[base + j] * x[j];
        }
        val += 0.5 * x[i] * hx + f[i] * x[i];
    }
    val
}

/// Control-acceleration residual energy at `x`, returned as
/// (model_res, true_res): the first uses the *linearized* per-segment gravity
/// (J_s r + c_s), the second the *true* gravity g(r). Both are midpoint-rule
/// quadratures of ∫‖a_geom(τ)/T² − g(r(τ))‖² dτ on the SAME nodes, so their
/// difference is purely the gravity linearization error the ratio test measures
/// (the shared quadrature error cancels in the difference).
fn eval_residual_terms(
    x: &[f64],
    lin: &[SegmentGravLin],
    np1: usize,
    dim: usize,
    t: f64,
    consts: &OrbitalConstants,
) -> (f64, f64) {
    let t2_inv = 1.0 / (t * t);
    let n_lin = lin.len().max(1);
    // The integrand is piecewise: each De Casteljau segment carries its own affine
    // gravity model, so it jumps at segment boundaries. A midpoint cell straddling a
    // boundary integrates the wrong model over part of itself, which degrades the
    // rule from O(h^2) to O(h). Rounding n_q up to a multiple of n_lin keeps every
    // cell inside one segment and restores second-order accuracy.
    let n_q = 1000usize.div_ceil(n_lin) * n_lin;
    let w = 1.0 / n_q as f64;
    let mut model_res = 0.0f64;
    let mut true_res = 0.0f64;
    for m in 0..n_q {
        let tau = (m as f64 + 0.5) / n_q as f64;
        let s = ((tau * n_lin as f64) as usize).min(n_lin - 1);
        let seg = &lin[s];
        let r = bezier::evaluate(x, np1, dim, tau);
        let a = bezier::evaluate_acceleration(x, np1, dim, tau);
        let r3 = [r[0], r[1], r[2]];
        let g_true = gravity::accel_total(&r3, consts.mu, consts.r_e_km, consts.j2);
        let mut m_sq = 0.0;
        let mut t_sq = 0.0;
        for d in 0..3 {
            let a_phys = a[d] * t2_inv;
            let mut g_lin = seg.c_i[d];
            for dd in 0..3 {
                g_lin += seg.j_i[d][dd] * r[dd];
            }
            let rm = a_phys - g_lin;
            let rt = a_phys - g_true[d];
            m_sq += rm * rm;
            t_sq += rt * rt;
        }
        model_res += w * m_sq;
        true_res += w * t_sq;
    }
    (model_res, true_res)
}

/// Minimum radius of the curve over a dense parameter sweep (KOZ feasibility probe).
fn min_radius_of(x: &[f64], np1: usize, dim: usize) -> f64 {
    let n_check = 1000;
    let mut min_r = f64::INFINITY;
    for i in 0..=n_check {
        let tau = i as f64 / n_check as f64;
        let pt = bezier::evaluate(x, np1, dim, tau);
        let r: f64 = pt.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r < min_r {
            min_r = r;
        }
    }
    min_r
}

/// Sum of linearized KOZ row violations at `x`: Σ max(0, lb_r − a_r·x) over the KOZ
/// rows of the assembled system. This is exactly what the elastic slack absorbs, so
/// the convex merit built from it matches the penalized objective the QP minimizes.
fn koz_row_violation(
    x: &[f64],
    a_rows: &[f64],
    lbs: &[f64],
    koz_row_start: usize,
    n_koz: usize,
    nvars: usize,
) -> f64 {
    let mut total = 0.0f64;
    for r in koz_row_start..koz_row_start + n_koz {
        let lb = lbs[r];
        if !lb.is_finite() {
            continue; // empty-KOZ placeholder row
        }
        let mut ax = 0.0;
        for j in 0..nvars {
            ax += a_rows[r * nvars + j] * x[j];
        }
        total += (lb - ax).max(0.0);
    }
    total
}

/// Sum of true (non-linearized) KOZ violations of the subdivided control points at
/// `x`: Σ_{s,k} max(0, r_e − ‖q_k^(s) − c_koz‖). This is the nonlinear counterpart
/// of the half-space rows — each supporting hyperplane under-approximates the
/// distance to the sphere, so this is ≤ the linearized row violation everywhere.
fn true_cp_violation(
    x: &[f64],
    a_list: &[Vec<f64>],
    np1: usize,
    dim: usize,
    r_e: f64,
    c_koz: &[f64],
) -> f64 {
    let mut total = 0.0f64;
    for a_seg in a_list {
        for k in 0..np1 {
            let mut d_sq = 0.0;
            for d in 0..dim {
                let mut q = 0.0;
                for j in 0..np1 {
                    q += a_seg[k * np1 + j] * x[j * dim + d];
                }
                let diff = q - c_koz[d];
                d_sq += diff * diff;
            }
            total += (r_e - d_sq.sqrt()).max(0.0);
        }
    }
    total
}

/// Solve the elastic (virtual-control) QP: slack s_k ≥ 0 on each KOZ row with L1
/// penalty `elastic_weight·Σs` — i.e. minimize the penalized convex merit exactly.
/// Returns (x, total_slack, max_slack).
fn solve_qp_elastic(
    h_mat: &[f64],
    f_vec: &[f64],
    all_a_rows: &[f64],
    all_lb: &[f64],
    all_ub: &[f64],
    nvars: usize,
    total_rows: usize,
    koz_row_start: usize,
    n_koz: usize,
    elastic_weight: f64,
    almost_solved: &mut usize,
) -> Option<(Vec<f64>, f64, f64)> {
    let nvars_ext = nvars + n_koz;
    let ext_nrows = total_rows + n_koz;

    let mut h_ext = vec![0.0; nvars_ext * nvars_ext];
    for i in 0..nvars {
        for j in 0..nvars {
            h_ext[i * nvars_ext + j] = h_mat[i * nvars + j];
        }
    }

    let mut f_ext = vec![0.0; nvars_ext];
    f_ext[..nvars].copy_from_slice(f_vec);
    for k in 0..n_koz {
        f_ext[nvars + k] = elastic_weight;
    }

    let mut a_ext = vec![0.0; ext_nrows * nvars_ext];
    let mut lb_ext = Vec::with_capacity(ext_nrows);
    let mut ub_ext = Vec::with_capacity(ext_nrows);

    for r in 0..total_rows {
        let dst = r * nvars_ext;
        let src = r * nvars;
        a_ext[dst..dst + nvars].copy_from_slice(&all_a_rows[src..src + nvars]);
        if r >= koz_row_start && r < koz_row_start + n_koz {
            a_ext[dst + nvars + (r - koz_row_start)] = 1.0;
        }
        lb_ext.push(all_lb[r]);
        ub_ext.push(all_ub[r]);
    }

    for k in 0..n_koz {
        let r = total_rows + k;
        a_ext[r * nvars_ext + nvars + k] = 1.0;
        lb_ext.push(0.0);
        ub_ext.push(f64::INFINITY);
    }

    let x_full = solve_qp(
        &h_ext, &f_ext, &a_ext, &lb_ext, &ub_ext, nvars_ext, ext_nrows, almost_solved,
    )?;
    let total_slack = x_full[nvars..].iter().sum::<f64>();
    let max_slack = x_full[nvars..].iter().cloned().fold(0.0f64, f64::max);
    Some((x_full[..nvars].to_vec(), total_slack, max_slack))
}

/// Main optimization entry point.
pub fn optimize_orbital_docking(
    p_init: &[f64],   // (Np1, dim) row-major
    np1: usize,
    dim: usize,
    n_seg: usize,
    r_e: f64,
    max_iter: usize,
    tol: f64,
    transfer_time: f64,
    n_lin_seg: usize,
    scp_prox_weight: f64,
    scp_trust_radius: f64,
    v0: Option<&[f64]>,
    v1: Option<&[f64]>,
    a0: Option<&[f64]>,
    a1: Option<&[f64]>,
    enforce_prograde: bool,
    prograde_n_samples: usize,
    elastic_weight: f64,
    koz_degenerate_mode: constraints::DegenerateNormal,
) -> OptResult {
    let consts = OrbitalConstants::default();
    let n = np1 - 1;
    let nvars = np1 * dim;
    let t = transfer_time;

    let mut p = p_init.to_vec();

    // Segment matrices (computed once)
    let a_list = de_casteljau::segment_matrices_equal_params(n, n_seg);

    // Bounds: fix endpoints
    let mut lb = vec![f64::NEG_INFINITY; nvars];
    let mut ub = vec![f64::INFINITY; nvars];
    for d in 0..dim {
        lb[d] = p[d];
        ub[d] = p[d];
        lb[nvars - dim + d] = p[nvars - dim + d];
        ub[nvars - dim + d] = p[nvars - dim + d];
    }

    // Boundary constraints
    let bc_constraints = constraints::build_boundary_constraints(np1, dim, t, v0, v1, a0, a1);

    // Prograde angular momentum reference
    let mut h_hat: Option<[f64; 3]> = None;
    if enforce_prograde && v0.is_some() && dim == 3 {
        let r0 = [p[0], p[1], p[2]];
        let v0_phys = v0.unwrap();
        let h0 = [
            r0[1] * v0_phys[2] - r0[2] * v0_phys[1],
            r0[2] * v0_phys[0] - r0[0] * v0_phys[2],
            r0[0] * v0_phys[1] - r0[1] * v0_phys[0],
        ];
        let h0_norm = (h0[0] * h0[0] + h0[1] * h0[1] + h0[2] * h0[2]).sqrt();
        if h0_norm > 0.0 {
            h_hat = Some([h0[0] / h0_norm, h0[1] / h0_norm, h0[2] / h0_norm]);
        }
    }

    let mut iterations = 0;
    let mut last_delta = f64::NAN;
    let mut last_total_slack = 0.0f64;
    let mut last_max_slack = 0.0f64;

    // SCvx trust-region state. Enabled when a positive trust radius is supplied;
    // otherwise the legacy fixed-point loop (unconditional accept + step-norm tol)
    // is used unchanged. When enabled, each QP step is accepted/rejected by the
    // ratio of actual (true-gravity) to predicted (linearized) merit reduction,
    // and the radius adapts — turning the loop into a real successive-convexification
    // method that converges in a handful of outer iterations.
    let trust_active = scp_trust_radius > 0.0;
    let mut trust = scp_trust_radius;
    let trust_min = 1e-2_f64; // km — below this we declare "no further progress"
    let trust_max = scp_trust_radius * 4.0;
    let eta_accept = 0.1_f64; // accept the step if rho exceeds this
    // Relative-merit convergence tolerance. The legacy `tol` is a nanometre-scale
    // step norm (1e-12) that can never trigger under a moving linearization; for
    // the SCvx path we reinterpret it as a relative merit-change tol with a sane floor.
    let tol_f = if tol > 0.0 { tol.max(1e-8) } else { 1e-8 };
    // Convergence requires K consecutive accepted steps below tol_f. A single-step
    // test cannot distinguish a stationary point from the slow crawl along
    // re-aimed KOZ walls (measured: the crawl produces isolated sub-tol steps,
    // then breaks through to a much better value — a one-shot test at 1e-6
    // stopped 33% above the optimum on phase120).
    let conv_streak_required = 3usize;
    let mut conv_streak = 0usize;
    // The model-stationarity test carries the same K-consecutive guard, for the same
    // reason: a single quiet `pred` is not evidence of stationarity when the walls
    // re-aim between iterations, and a one-shot test is exactly the premature-stop
    // failure K = 3 exists to prevent. Costs a few iterations, removes a whole
    // failure mode.
    let mut stat_streak = 0usize;
    let mut converged_scvx = false;
    // Which criterion actually ended the loop. `converged_scvx` alone cannot say:
    // both the K-consecutive merit streak and trust-region collapse set it true, and
    // they are very different claims — the streak asserts merit stationarity, the
    // collapse only says the ratio test stopped accepting anything.
    //   0 = iteration cap, 1 = K-consecutive merit streak,
    //   2 = trust-region collapse, 3 = QP failure,
    //   4 = K-consecutive model stationarity (|pred| < tol_f*|phi| with the
    //       certificate held) — the DOMINANT exit since 2026-08-11; 1 and 4 are
    //       the principled ones, 0/2/3 all mean the loop gave up.
    let mut stop_reason = 0.0f64;
    // QP-health counters, surfaced in `info` so a reduced-accuracy or degenerate
    // solve cannot pass unnoticed into the merit/ratio test.
    let mut qp_almost_solved = 0usize;
    let mut koz_degenerate_max = 0usize;

    // Per-iteration trace of EVERY step, accepted or rejected, to stderr when the
    // SCVX_TRACE env var is set. The *_history vectors record only ACCEPTED steps,
    // which hides precisely the rejected steps that collapse the trust radius — the
    // blind spot that concealed the ratio-test deadlock. Off unless asked for.
    let trace_scvx = std::env::var("SCVX_TRACE").is_ok();
    let mut null_step_count = 0usize;
    let mut reject_count = 0usize;
    if trace_scvx {
        eprintln!(
            "SCVXTRACE,it,outcome,rho,pred,act,l_p,l_c,t_p,t_c,rel,trust_before,trust_after,\
step_norm,vlin_p,vlin_c,vtrue_c,hard_viol_p,conv_streak,pred_floor,total_slack,\
quad_p,quad_c,gaperr_p,gaperr_c,cpviol_c,minrad_c,kozslack_min_p"
        );
    }

    // Per-accepted-step SCvx diagnostics (for the verification harness / paper figures).
    let mut rho_history: Vec<f64> = Vec::new();
    let mut trust_history: Vec<f64> = Vec::new();
    let mut merit_history: Vec<f64> = Vec::new();
    let mut step_norm_history: Vec<f64> = Vec::new();
    let mut slack_history: Vec<f64> = Vec::new();
    let mut phase_history: Vec<f64> = Vec::new();

    // Cache of the iter-1 gravity linearization: the drift-diagnostic baseline.
    let mut lin_baseline: Option<Vec<SegmentGravLin>> = None;
    let mut jacobian_drift_history: Vec<Vec<f64>> = Vec::new();

    for it in 1..=max_iter {
        iterations = it;

        let c_koz = vec![0.0; dim];

        // KOZ + gravity linearization, rebuilt fresh at the current reference every
        // iteration — this is what makes the method successive convexification.
        // (Also updates the Jacobian-drift diagnostic against the iter-1 baseline.)
        // KOZ rows: one supporting half-space per De Casteljau segment, normal aimed
        // from the KOZ centre at the segment centroid. The centroid rule re-aims every
        // iteration, so the half-space the QP optimizes against is NOT the one the next
        // iteration will grade with — unless the subproblem accounts for the pivot.
        //
        // On the trust path it does: `build_koz_constraints_linearized` adds the
        // normal-rotation term, so the QP's optimum is a point the centroid rule still
        // agrees with after re-aiming. Without it the step lands flush on a plane that
        // then pivots out from under it (measured phase120/n_seg=16: iterate satisfies
        // its own rows to 0.0 yet misses the rows rebuilt at itself by 68 m, while
        // clearing the sphere by 16.0 km), and the solver spends ~100 iterations
        // negotiating with a phantom violation.
        //
        // This is a self-consistency device for the STEP, not a physical model — see
        // design_freeze.md section 9. Soundness is unaffected: the certificate and the
        // true merit are always evaluated with the EXACT rows (`build_koz_constraints`).
        // The legacy fixed-point path accepts unconditionally with no ratio test, so it
        // keeps the conservative rows.
        let (koz, koz_degen) = if trust_active {
            constraints::build_koz_constraints_linearized(
                &a_list, &p, np1, dim, r_e, &c_koz, koz_degenerate_mode,
            )
        } else {
            constraints::build_koz_constraints(
                &a_list, &p, np1, dim, r_e, &c_koz, koz_degenerate_mode,
            )
        };
        koz_degenerate_max = koz_degenerate_max.max(koz_degen);
        let lin_t = compute_segment_lin(&p, np1, dim, t, n_lin_seg, &consts);
        if it == 1 {
            lin_baseline = Some(lin_t.clone());
            jacobian_drift_history.push(vec![0.0; lin_t.len()]);
        } else if let Some(base) = lin_baseline.as_deref() {
            let mut drift_row = Vec::with_capacity(lin_t.len());
            for (s_t, s_0) in lin_t.iter().zip(base.iter()) {
                drift_row.push(frobenius_norm_3x3(&s_t.j_i, &s_0.j_i));
            }
            jacobian_drift_history.push(drift_row);
        }
        let lin_for_qp_ref: &[SegmentGravLin] = &lin_t;

        // Build quadratic objective. Keep the unregularized (H_obj, f_obj) for the
        // SCvx merit/ratio computation; the solved system adds proximal damping on top.
        let (h_obj, f_obj, c_const) = build_ctrl_accel_quadratic(
            &p,
            np1,
            dim,
            t,
            n_lin_seg,
            &consts,
            Some(lin_for_qp_ref),
        );
        let mut h_mat = h_obj.clone();
        let mut f_vec = f_obj.clone();

        // Proximal regularization. Skipped in the SCvx trust path: the trust region
        // already bounds the step, and the legacy fixed weight (e.g. 1e-6) is ~7 orders
        // of magnitude larger than the objective Hessian (Gram/T^4 ~ 1e-13), so it would
        // dominate the QP and reduce every step to a tiny anchor-to-p move (the crawl).
        if scp_prox_weight > 0.0 && !trust_active {
            let lam = scp_prox_weight;
            for i in 0..nvars {
                h_mat[i * nvars + i] += lam;
                f_vec[i] -= lam * p[i];
            }
        }

        // Assemble all constraint matrices into one big A, lb, ub
        let mut all_a_rows: Vec<f64> = Vec::new();
        let mut all_lb: Vec<f64> = Vec::new();
        let mut all_ub: Vec<f64> = Vec::new();
        let mut total_rows = 0;

        // Bounds as constraints (2*dim rows for fixed endpoints)
        for i in 0..nvars {
            if lb[i] == ub[i] {
                let mut row = vec![0.0; nvars];
                row[i] = 1.0;
                all_a_rows.extend_from_slice(&row);
                all_lb.push(lb[i]);
                all_ub.push(ub[i]);
                total_rows += 1;
            }
        }

        // KOZ constraints (track row range for elastic relaxation)
        let koz_row_start = total_rows;
        for r in 0..koz.n_rows {
            all_a_rows.extend_from_slice(&koz.a[r * nvars..(r + 1) * nvars]);
            all_lb.push(koz.lb[r]);
            all_ub.push(koz.ub[r]);
            total_rows += 1;
        }
        let n_koz = total_rows - koz_row_start;

        // Boundary constraints
        for bc in &bc_constraints {
            for r in 0..bc.n_rows {
                all_a_rows.extend_from_slice(&bc.a[r * nvars..(r + 1) * nvars]);
                all_lb.push(bc.lb[r]);
                all_ub.push(bc.ub[r]);
                total_rows += 1;
            }
        }

        // Prograde constraints
        if let Some(hh) = h_hat {
            let n_loc = n;
            let n_pts = prograde_n_samples.max(1) + 2;
            let taus: Vec<f64> = (1..n_pts - 1)
                .map(|i| i as f64 / (n_pts - 1) as f64)
                .collect();

            for tau in &taus {
                let w = bezier::bernstein_basis(n_loc, *tau);
                let w_der = bezier::bernstein_derivative_weights(n_loc, *tau);

                // Reference r, v at this tau
                let mut r_ref = [0.0f64; 3];
                let mut v_tau_ref = [0.0f64; 3];
                let p_mat = &p;
                for j in 0..np1 {
                    for d in 0..3 {
                        r_ref[d] += w[j] * p_mat[j * dim + d];
                        v_tau_ref[d] += w_der[j] * p_mat[j * dim + d];
                    }
                }
                let v_ref = [v_tau_ref[0] / t, v_tau_ref[1] / t, v_tau_ref[2] / t];

                // h_ref = cross(r_ref, v_ref)
                let h_ref = [
                    r_ref[1] * v_ref[2] - r_ref[2] * v_ref[1],
                    r_ref[2] * v_ref[0] - r_ref[0] * v_ref[2],
                    r_ref[0] * v_ref[1] - r_ref[1] * v_ref[0],
                ];
                let c_ref: f64 = hh[0] * h_ref[0] + hh[1] * h_ref[1] + hh[2] * h_ref[2];

                // alpha = cross(h_hat, v_ref), beta = cross(r_ref, h_hat)
                let alpha = [
                    hh[1] * v_ref[2] - hh[2] * v_ref[1],
                    hh[2] * v_ref[0] - hh[0] * v_ref[2],
                    hh[0] * v_ref[1] - hh[1] * v_ref[0],
                ];
                let beta = [
                    r_ref[1] * hh[2] - r_ref[2] * hh[1],
                    r_ref[2] * hh[0] - r_ref[0] * hh[2],
                    r_ref[0] * hh[1] - r_ref[1] * hh[0],
                ];

                let mut g = vec![0.0f64; nvars];
                for j in 0..np1 {
                    for d in 0..3 {
                        g[j * 3 + d] = w[j] * alpha[d] + (w_der[j] / t) * beta[d];
                    }
                }

                let g_dot_x: f64 = (0..nvars).map(|i| g[i] * p[i]).sum();
                let rhs = -c_ref + g_dot_x;

                all_a_rows.extend_from_slice(&g);
                all_lb.push(rhs);
                all_ub.push(f64::INFINITY);
                total_rows += 1;
            }
        }

        // SCvx trust region: |x_i - p_i| <= trust for every free variable. Enforcing
        // it as a *constraint* in the QP (rather than clipping the step afterwards)
        // is what makes the convex subproblem a faithful local model. Fixed endpoints
        // already carry equality rows, so they are skipped. These rows are appended
        // before the elastic extension so they are carried as hard constraints there too.
        if trust_active {
            for i in 0..nvars {
                if lb[i] == ub[i] {
                    continue; // fixed endpoint
                }
                let mut row = vec![0.0; nvars];
                row[i] = 1.0;
                all_a_rows.extend_from_slice(&row);
                all_lb.push(p[i] - trust);
                all_ub.push(p[i] + trust);
                total_rows += 1;
            }
        }

        // Solve QP. On the SCvx trust path the elastic (virtual-control) form is
        // solved *unconditionally*: the subproblem then minimizes exactly the
        // penalized convex merit L(x) = J^(k)(x) + w_s·viol_lin(x) that the ratio
        // test measures, which makes the predicted merit reduction nonnegative by
        // construction (x = p with s = viol_lin(p) is always elastic-feasible).
        // The legacy path keeps the original hard-first / elastic-fallback order.
        let (x_new, iter_total_slack, iter_max_slack);
        let elastic_available = elastic_weight > 0.0 && n_koz > 0;

        // Solve the subproblem in STEP coordinates d = x − p, not absolute position.
        //
        // The absolute form hands the QP variables of order 7e3 (km from the Earth's
        // centre) with a Hessian of order 1e-13, i.e. ~17 orders of magnitude between
        // the variable scale and the curvature being minimized. Clarabel loses enough
        // precision there to return points WORSE than the reference while reporting
        // `Solved` — measured 26 times on phase120, up to 8% suboptimal. That is
        // impossible for an exact solve (d = 0 with s = viol(p) is always elastic
        // feasible), and the giveaway was two iterations sharing a reference,
        // linearization and rows where the LARGER trust region returned the WORSE
        // optimum. A superset feasible region cannot have a worse optimum.
        //
        // d = x − p is an exact affine change of variables — same feasible set, same
        // optimum, the constant ½pᵀHp + fᵀp drops out of every difference:
        //     H' = H,  f' = Hp + f,  lb' = lb − Ap,  ub' = ub − Ap
        // Measured effect: every negative-pred event disappears, iterations fall
        // 32–40% at n_seg 16/32, and the objective improves 2.2% at n_seg=8.
        let (h_use, f_use, lb_use, ub_use, shift_applied): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, bool) =
            if trust_active {
                let mut fs = f_vec.clone();
                for i in 0..nvars {
                    let mut hp = 0.0;
                    for j in 0..nvars {
                        hp += h_mat[i * nvars + j] * p[j];
                    }
                    fs[i] += hp;
                }
                let mut lbs = all_lb.clone();
                let mut ubs = all_ub.clone();
                for r in 0..total_rows {
                    let mut ap = 0.0;
                    for j in 0..nvars {
                        ap += all_a_rows[r * nvars + j] * p[j];
                    }
                    if lbs[r].is_finite() {
                        lbs[r] -= ap;
                    }
                    if ubs[r].is_finite() {
                        ubs[r] -= ap;
                    }
                }
                (h_mat.clone(), fs, lbs, ubs, true)
            } else {
                (h_mat.clone(), f_vec.clone(), all_lb.clone(), all_ub.clone(), false)
            };

        // Normalize the objective's MAGNITUDE (the step shift above normalized the
        // variable magnitudes). H is ~1e-13 here and the achievable improvement near
        // the optimum ~1e-17 against a merit of ~3.8e-7 -- about 11 orders down, past
        // what f64 resolves at this conditioning. Left unscaled, the QP returns points
        // WORSE than the reference: measured 100 such events on the migration-baseline
        // scenario, driving a stable accept/reject 2-cycle to the iteration cap.
        //
        // Scaling the objective is mathematically neutral -- argmin(J/s) = argmin(J) --
        // PROVIDED the elastic penalty is scaled identically so the objective/slack
        // trade-off is untouched. s is an exact power of two, so the rescaling itself
        // introduces no rounding: a conditioning artifact must vanish under it, real
        // mathematics must survive it unchanged.
        //
        // Measured: negative-pred events 100 -> 0 and 200 -> 24 iterations on the
        // baseline scenario; on phase120 the step shift alone already gave 0, and this
        // holds that at 0 across n_seg 8/16/32.
        let obj_scale: f64 = if trust_active {
            let max_h = h_use.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            if max_h > 0.0 && max_h.is_finite() {
                (2.0f64).powi(max_h.log2().floor() as i32)
            } else {
                1.0
            }
        } else {
            1.0
        };
        let (h_use, f_use, w_use) = if obj_scale != 1.0 {
            (
                h_use.iter().map(|v| v / obj_scale).collect::<Vec<f64>>(),
                f_use.iter().map(|v| v / obj_scale).collect::<Vec<f64>>(),
                elastic_weight / obj_scale,
            )
        } else {
            (h_use, f_use, elastic_weight)
        };

        if trust_active && elastic_available {
            match solve_qp_elastic(
                &h_use, &f_use, &all_a_rows, &lb_use, &ub_use, nvars, total_rows,
                koz_row_start, n_koz, w_use, &mut qp_almost_solved,
            ) {
                Some((x, ts, ms)) => {
                    x_new = if shift_applied {
                        (0..nvars).map(|i| p[i] + x[i]).collect()
                    } else {
                        x
                    };
                    iter_total_slack = ts;
                    iter_max_slack = ms;
                }
                None => {
                    stop_reason = 3.0;
                    break;
                }
            }
        } else if let Some(x_sol) = solve_qp(
            &h_mat, &f_vec, &all_a_rows, &all_lb, &all_ub, nvars, total_rows,
            &mut qp_almost_solved,
        ) {
            x_new = x_sol;
            iter_total_slack = 0.0;
            iter_max_slack = 0.0;
        } else if elastic_available {
            match solve_qp_elastic(
                &h_mat, &f_vec, &all_a_rows, &all_lb, &all_ub, nvars, total_rows,
                koz_row_start, n_koz, elastic_weight, &mut qp_almost_solved,
            ) {
                Some((x, ts, ms)) => {
                    x_new = x;
                    iter_total_slack = ts;
                    iter_max_slack = ms;
                }
                None => {
                    stop_reason = 3.0;
                    break;
                }
            }
        } else {
            break;
        }

        last_total_slack = iter_total_slack;
        last_max_slack = iter_max_slack;

        if trust_active {
            // ---- Canonical SCvx step acceptance (penalized merit; Mao et al.) ----
            // Convex merit  L(x) = J^(k)(x) + w_s·Σ max(0, b_r − a_r·x)   — exactly what
            //                      the elastic QP minimizes over the trust box.
            // True merit    T(x) = J(x)     + w_s·h(x)
            // where J(x) swaps the linearized gravity residual for the true one, and
            // h(x) is the *hull-certifiability* violation: the half-space rows rebuilt
            // at x itself (h = 0 ⇔ x carries the Prop-1 certificate, which is the
            // property actually being claimed at the end of the run). The convex rows
            // are h's first-order model, so rho = actual/predicted measures the two
            // model errors that exist: gravity linearization and normal re-aiming.
            //
            // Grading h at the candidate is what makes the convergence test mean
            // "carries its own certificate" rather than "satisfied last iteration's
            // rows". It requires the subproblem to model the re-aim (it does, above);
            // pairing this merit with frozen-normal rows is the defect that started
            // this work. Penalizing the sphere distance of the control points instead
            // would NOT work either: the half-space is conservative, so its violation
            // need not vanish where the sphere constraint holds, and the mismatch
            // deadlocks the ratio test (pred large, act ≈ 0).
            let cand = &x_new;
            let trust_before = trust;
            let step_norm: f64 = (0..nvars)
                .map(|i| (cand[i] - p[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            last_delta = step_norm;

            let w_s = elastic_weight.max(0.0);
            let quad_p = quad_form(&h_obj, &f_obj, &p, nvars);
            let quad_c = quad_form(&h_obj, &f_obj, cand, nvars);
            let vlin_p =
                koz_row_violation(&p, &all_a_rows, &all_lb, koz_row_start, n_koz, nvars);
            let vlin_c =
                koz_row_violation(cand, &all_a_rows, &all_lb, koz_row_start, n_koz, nvars);
            let (mres_p, tres_p) = eval_residual_terms(&p, lin_for_qp_ref, np1, dim, t, &consts);
            let (mres_c, tres_c) = eval_residual_terms(cand, lin_for_qp_ref, np1, dim, t, &consts);
            // h(p) and h(cand): EXACT rows rebuilt at each point. The reference's own
            // rows were already built at p this iteration, and they reproduce the exact
            // clearance there (`koz_rows_exact_at_reference`), so vlin_p == h(p) and
            // only the candidate needs its own re-aimed walls.
            let (vtrue_p, vtrue_c) = {
                let (koz_c, _) = constraints::build_koz_constraints(
                    &a_list, cand, np1, dim, r_e, &c_koz, koz_degenerate_mode,
                );
                let h_c = koz_row_violation(cand, &koz_c.a, &koz_c.lb, 0, koz_c.n_rows, nvars);
                (vlin_p, h_c)
            };

            // Diagnostics only — never fed to the merit. cpviol_c = true sphere
            // penetration of the candidate's subdivided control points; if this is 0
            // while vtrue_c > 0 the step is only failing the algorithm's own re-aimed
            // supporting half-space, not the actual keep-out sphere.
            // kozslack_min_p = tightest KOZ row slack at the reference (is the KOZ
            // even active?).
            let (cpviol_c, minrad_c, kozslack_min_p) = if trace_scvx {
                let mut smin = f64::INFINITY;
                for r in koz_row_start..koz_row_start + n_koz {
                    if !all_lb[r].is_finite() {
                        continue;
                    }
                    let mut ax = 0.0;
                    for j in 0..nvars {
                        ax += all_a_rows[r * nvars + j] * p[j];
                    }
                    smin = smin.min(ax - all_lb[r]);
                }
                (
                    true_cp_violation(cand, &a_list, np1, dim, r_e, &c_koz),
                    min_radius_of(cand, np1, dim),
                    smin,
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            };


            // c_const completes the model objective so merit values are absolute
            // (comparable across iterations); it cancels in both differences.
            let l_p = quad_p + c_const + w_s * vlin_p;
            let l_c = quad_c + c_const + w_s * vlin_c;
            let t_p = quad_p + c_const + (tres_p - mres_p) + w_s * vtrue_p;
            let t_c = quad_c + c_const + (tres_c - mres_c) + w_s * vtrue_c;
            // pred ≥ 0 up to solver tolerance PROVIDED x = p is elastic-feasible,
            // i.e. p satisfies every hard (non-KOZ) row — checked below.
            let pred = l_p - l_c;
            let act = t_p - t_c;
            // Null-step floor, scaled to the merit's own magnitude. The previous form
            // 1e-12*(1.0 + |L(p)|) was absolute in practice (|L(p)| ~ 2.3e-5, so the
            // `1.0 +` dominated), which made the threshold depend on the choice of
            // units rather than on the problem. Measured: purely cosmetic here — the
            // branch fires only where pred < 0, which is below any positive floor, so
            // both forms give bit-identical results on every tested configuration.
            // Kept in the relative form so no arbitrary absolute constant remains.
            let pred_floor = 1e-12 * l_p.abs().max(1e-30);

            // Hard-row (non-KOZ) violation of the reference: endpoints, boundary
            // equalities, prograde. These carry no slack, so the pred ≥ 0 argument
            // requires this to be zero. It is NOT zero at iteration 1 (the
            // straight-line init violates the velocity-BC equality rows) and can
            // recur for prograde rows, which re-aim each iteration. (Trust rows are
            // centred at p and contribute 0.)
            let mut hard_viol_p = 0.0f64;
            for r in 0..total_rows {
                if r >= koz_row_start && r < koz_row_start + n_koz {
                    continue;
                }
                let mut ax = 0.0;
                for j in 0..nvars {
                    ax += all_a_rows[r * nvars + j] * p[j];
                }
                if all_lb[r].is_finite() {
                    hard_viol_p += (all_lb[r] - ax).max(0.0);
                }
                if all_ub[r].is_finite() {
                    hard_viol_p += (ax - all_ub[r]).max(0.0);
                }
            }
            if hard_viol_p > 1e-9 {
                // Affine bootstrap: against a reference that violates exact linear
                // constraints the merit comparison is meaningless (pred can be
                // negative, and rejecting would loop forever on the repair step).
                // The candidate satisfies every hard row exactly, so accept it
                // unconditionally and skip the convergence test.
                if trace_scvx {
                    eprintln!(
                        "SCVXTRACE,{},bootstrap,{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},\
{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},\
{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                        it, f64::NAN, pred, act, l_p, l_c, t_p, t_c, f64::NAN,
                        trust, trust, step_norm, vlin_p, vlin_c, vtrue_c, hard_viol_p,
                        0, pred_floor, iter_total_slack, quad_p, quad_c,
                        tres_p - mres_p, tres_c - mres_c, cpviol_c, minrad_c, kozslack_min_p
                    );
                }
                p = x_new;
                conv_streak = 0;
                rho_history.push(f64::NAN);
                trust_history.push(trust);
                merit_history.push(t_c);
                step_norm_history.push(step_norm);
                slack_history.push(iter_total_slack);
                phase_history.push(0.0);
                continue;
            }

            // Stationarity test (standard trust-region "predicted reduction" criterion).
            // Reference: Conn, Gould & Toint, *Trust-Region Methods*, MOS-SIAM Series on
            // Optimization, SIAM 2000, doi:10.1137/1.9780898719857 — Ch. 6 for the basic
            // convergence theory and Ch. 12 (Projection Methods for Convex Constraints)
            // for the criticality measure in this setting, since the subproblem's
            // constraints (half-spaces, the inf-norm box, the BC equalities) are convex.
            // Also Nocedal & Wright 2e Ch. 4. `pred` is the model's own
            // estimate of the improvement available from this reference. Once that is
            // negligible relative to the merit, the model reports no achievable progress
            // and the iterate is stationary FOR THE MODEL — continuing only feeds the
            // ratio test differences smaller than the merit's numerical resolution,
            // which it then rejects forever until the trust region collapses.
            //
            // Measured on the migration-baseline scenario: the merit is frozen to all 17
            // digits from iteration 5 onward, yet the loop ran to 24 — 83% of iterations
            // accomplishing nothing. pred/|T| there is 1.4e-11, against 2.8e-7 on the
            // last productive step and 1.2e-5 on phase120's (genuinely non-stationary)
            // deadlock, so tol_f = 1e-8 separates all three with >25x margin either way.
            //
            // Guarded on the certificate (vlin_p) so a penalized-stationary but
            // UNCERTIFIED point cannot report success. Reaching here already implies
            // hard_viol_p <= 1e-9 — the bootstrap above `continue`s otherwise. Note this
            // is stationarity of the CONVEX MODEL; it coincides with stationarity of the
            // true problem only where the model is faithful, which is what rho certifies.
            //
            // K-consecutive, like the merit streak (see `stat_streak`).
            //
            // The comparison is on |pred|, not pred, and the reason is numerical. Near a
            // stationary point pred is a difference of two merits that agree to ~1e-14
            // absolute (~4e-10 relative), so its SIGN is noise: the phase120 preds at the
            // stopping point measure -9.095e-15, -1.819e-14, -2.729e-14. The former
            // `pred >= 0.0` guard therefore reset the streak on every such iteration and
            // this exit could never fire. The loop instead rejected the null steps and
            // halved the trust region 18 consecutive times to the floor, reporting
            // stop_reason 2 (failure) at a point whose pred/|phi| was 1.2e-9 — three
            // orders INSIDE tol_f. Measured 2026-08-11 on N=8/n_seg=16 and N=7/n_seg=4;
            // both are critical, and both were mislabelled as failures.
            //
            // This widens the accepting set by exactly -tol_f*|T| < pred < 0, i.e. only
            // where the sign sits below the merit's own resolution. A pred negative by
            // MORE than tol_f*|T| is a genuine model defect and still resets the streak.
            if pred.abs() < tol_f * t_p.abs() && vlin_p <= 1e-6 {
                stat_streak += 1;
                if stat_streak >= conv_streak_required {
                    converged_scvx = true;
                    stop_reason = 4.0;
                    break;
                }
            } else {
                stat_streak = 0;
            }

            let rho = if pred < pred_floor {
                null_step_count += 1;
                // Model sees (numerically) no merit improvement: the reference is
                // stationary for the current subproblem. Treat as a null step —
                // accept if non-worsening so the convergence test below can fire.
                // (pred < 0 beyond noise lands here too and can only accept an
                // *improving* step, never divide by a negative prediction.)
                if act >= -pred_floor {
                    f64::INFINITY
                } else {
                    -1.0
                }
            } else {
                act / pred
            };

            // Relative merit change against the merit's own magnitude. The
            // objective is O(1e-4), so a "+1"-style denominator would turn
            // this into an absolute test 4 orders below the problem scale.
            let rel = act.abs() / t_p.abs().max(1e-12);
            if rho > eta_accept {
                p = x_new;
                if rho > 0.9 && rho.is_finite() {
                    trust = (trust * 2.0).min(trust_max); // model trustworthy — be bolder
                }
                if trace_scvx {
                    eprintln!(
                        "SCVXTRACE,{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},\
{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},\
{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                        it,
                        if pred < pred_floor { "accept_null" } else { "accept" },
                        rho, pred, act, l_p, l_c, t_p, t_c, rel,
                        trust_before, trust, step_norm, vlin_p, vlin_c, vtrue_c, hard_viol_p,
                        conv_streak, pred_floor, iter_total_slack, quad_p, quad_c,
                        tres_p - mres_p, tres_c - mres_c, cpviol_c, minrad_c, kozslack_min_p
                    );
                }
                rho_history.push(rho);
                trust_history.push(trust);
                merit_history.push(t_c);
                step_norm_history.push(step_norm);
                slack_history.push(iter_total_slack);
                // 0 = uncertified iterate (feasibility restoration), 1 = the iterate
                // carries the Prop-1 certificate against its OWN rebuilt rows.
                phase_history.push(if vtrue_c > 1e-6 { 0.0 } else { 1.0 });
                // Converged = merit stationary for conv_streak_required consecutive
                // accepted steps AND the accepted iterate carries the hull certificate.
                // vtrue_c is measured against the rows rebuilt AT the candidate, so
                // this gate already asserts the certificate the run reports at the end
                // — no separate re-check is needed. A penalized-stationary-but-
                // uncertified point keeps iterating instead of reporting a false
                // success.
                if rel < tol_f && vtrue_c <= 1e-6 {
                    conv_streak += 1;
                    if conv_streak >= conv_streak_required {
                        converged_scvx = true;
                        stop_reason = 1.0;
                        break;
                    }
                } else {
                    conv_streak = 0;
                }
            } else {
                conv_streak = 0;
                reject_count += 1;
                trust *= 0.5;
                if trace_scvx {
                    eprintln!(
                        "SCVXTRACE,{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},\
{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},\
{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                        it,
                        if pred < pred_floor { "reject_null" } else { "reject" },
                        rho, pred, act, l_p, l_c, t_p, t_c, rel,
                        trust_before, trust, step_norm, vlin_p, vlin_c, vtrue_c, hard_viol_p,
                        0, pred_floor, iter_total_slack, quad_p, quad_c,
                        tres_p - mres_p, tres_c - mres_c, cpviol_c, minrad_c, kozslack_min_p
                    );
                }
                if trust < trust_min {
                    // Trust-radius collapse is the second standard SCvx stopping
                    // criterion. At a reference satisfying the hull certificate
                    // (Prop 1) AND every hard row this is convergence to a
                    // constrained local optimum; otherwise it is a genuine failure
                    // and converged stays false.
                    converged_scvx = vlin_p <= 1e-6 && hard_viol_p <= 1e-9;
                    stop_reason = 2.0;
                    break;
                }
                continue;
            }
        } else {
            // ---- Legacy fixed-point path (no trust region supplied) ----
            let x_result = x_new;
            let delta: f64 = (0..nvars)
                .map(|i| (x_result[i] - p[i]).powi(2))
                .sum::<f64>()
                .sqrt();

            last_delta = delta;
            p = x_result;

            if delta < tol {
                break;
            }
        }
    }

    // Final feasibility check: dense radius probe (diagnostic) plus the exact
    // Prop-1 certificate — half-spaces rebuilt at the final iterate; if every
    // subdivided control point satisfies them, the whole curve clears the KOZ.
    let min_radius = min_radius_of(&p, np1, dim);
    let c_koz_final = vec![0.0; dim];
    let (koz_final, koz_degen_final) = constraints::build_koz_constraints(
        &a_list, &p, np1, dim, r_e, &c_koz_final, koz_degenerate_mode,
    );
    koz_degenerate_max = koz_degenerate_max.max(koz_degen_final);
    let final_hull_violation = koz_row_violation(
        &p,
        &koz_final.a,
        &koz_final.lb,
        0,
        koz_final.n_rows,
        nvars,
    );
    let final_cp_violation = true_cp_violation(&p, &a_list, np1, dim, r_e, &c_koz_final);

    // Compute final cost (always uses fresh linearization at the final p, so
    // both frozen and unfrozen runs report a self-consistent cost).
    let (hf, ff, cf) = build_ctrl_accel_quadratic(
        &p, np1, dim, t, n_lin_seg, &consts, None,
    );
    let mut cost_no_const = 0.0f64;
    for i in 0..nvars {
        cost_no_const += 0.5 * p[i] * {
            let mut hx = 0.0;
            for j in 0..nvars {
                hx += hf[i * nvars + j] * p[j];
            }
            hx
        };
        cost_no_const += ff[i] * p[i];
    }
    let cost_true_energy = cost_no_const + cf;

    // Compute control acceleration metrics
    let n_metrics = 300;
    let t2_inv = 1.0 / (t * t);
    let mut max_ctrl_accel = 0.0f64;
    let mut sum_ctrl_accel = 0.0f64;

    for i in 0..=n_metrics {
        let tau = i as f64 / n_metrics as f64;
        let pt = bezier::evaluate(&p, np1, dim, tau);
        let a_geom = bezier::evaluate_acceleration(&p, np1, dim, tau);
        let a_geom_phys: Vec<f64> = a_geom.iter().map(|x| x * t2_inv).collect();
        let r_km = [pt[0], pt[1], pt[2]];
        let a_grav = gravity::accel_total(&r_km, consts.mu, consts.r_e_km, consts.j2);

        let a_u_km_s2 = [
            a_geom_phys[0] - a_grav[0],
            a_geom_phys[1] - a_grav[1],
            a_geom_phys[2] - a_grav[2],
        ];
        let a_u_m_s2 =
            (a_u_km_s2[0].powi(2) + a_u_km_s2[1].powi(2) + a_u_km_s2[2].powi(2)).sqrt() * 1e3;
        if a_u_m_s2 > max_ctrl_accel {
            max_ctrl_accel = a_u_m_s2;
        }
        sum_ctrl_accel += a_u_m_s2;
    }
    let mean_ctrl_accel = sum_ctrl_accel / (n_metrics + 1) as f64;

    let feasible = min_radius >= r_e - 1e-6;

    let mut info = HashMap::new();
    info.insert("iterations".to_string(), iterations as f64);
    info.insert("feasible".to_string(), if feasible { 1.0 } else { 0.0 });
    info.insert("min_radius".to_string(), min_radius);
    info.insert("cost_true_energy".to_string(), cost_true_energy);
    info.insert("cost_no_const".to_string(), cost_no_const);
    info.insert("cost".to_string(), cost_true_energy);
    info.insert("max_control_accel_ms2".to_string(), max_ctrl_accel);
    info.insert("mean_control_accel_ms2".to_string(), mean_ctrl_accel);
    info.insert("final_delta_norm".to_string(), last_delta);
    info.insert("total_koz_slack".to_string(), last_total_slack);
    info.insert("max_koz_slack".to_string(), last_max_slack);
    // Prop-1 certificate at the final iterate (0 ⇒ curve provably clears the KOZ).
    info.insert("final_hull_violation_km".to_string(), final_hull_violation);
    info.insert("final_cp_violation_km".to_string(), final_cp_violation);
    info.insert(
        "scvx_converged".to_string(),
        if converged_scvx { 1.0 } else { 0.0 },
    );
    // 0 = iteration cap, 1 = K-consecutive merit streak, 2 = trust-region collapse,
    // 3 = QP failure, 4 = model-stationarity (predicted reduction negligible).
    // 1 and 4 are principled stops; 0 and 2 are the loop giving up.
    info.insert("scvx_stop_reason".to_string(), stop_reason);
    info.insert("final_trust_radius".to_string(), trust);
    // QP health. qp_almost_solved > 0 means Clarabel hit its reduced tolerances on
    // that many solves and those solutions still fed the ratio test.
    info.insert("qp_almost_solved".to_string(), qp_almost_solved as f64);
    // TEMPORARY INSTRUMENTATION (2026-08-09): how many iterations took the
    // pred < pred_floor "null step" branch (no ratio test), and how many were
    // rejected by the ratio test (invisible in the accepted-only *_history vectors).
    info.insert("scvx_null_steps".to_string(), null_step_count as f64);
    info.insert("scvx_rejects".to_string(), reject_count as f64);
    // Segments whose supporting-half-space normal was undefined. Under the default
    // Skip mode a nonzero value means the Prop-1 certificate does not cover the
    // whole curve.
    info.insert(
        "koz_degenerate_segments".to_string(),
        koz_degenerate_max as f64,
    );
    // Drift summary stats (full history is on OptResult.jacobian_drift_history).
    if !jacobian_drift_history.is_empty() {
        let mut max_drift = 0.0f64;
        let mut last_max_drift = 0.0f64;
        for (i, row) in jacobian_drift_history.iter().enumerate() {
            for &v in row {
                if v > max_drift {
                    max_drift = v;
                }
                if i == jacobian_drift_history.len() - 1 && v > last_max_drift {
                    last_max_drift = v;
                }
            }
        }
        info.insert("jacobian_drift_max".to_string(), max_drift);
        info.insert("jacobian_drift_last_iter_max".to_string(), last_max_drift);
    }

    OptResult {
        p_opt: p,
        np1,
        dim,
        info,
        feasible,
        iterations,
        jacobian_drift_history,
        rho_history,
        trust_history,
        merit_history,
        step_norm_history,
        slack_history,
        phase_history,
    }
}

/// Generate initial control points (straight line from P_start to P_end).
pub fn generate_initial_control_points(degree: usize, p_start: &[f64], p_end: &[f64]) -> Vec<f64> {
    let dim = p_start.len();
    let np1 = degree + 1;
    let mut pts = vec![0.0; np1 * dim];
    for i in 0..np1 {
        let t = if degree == 1 {
            if i == 0 { 0.0 } else { 1.0 }
        } else {
            i as f64 / degree as f64
        };
        for d in 0..dim {
            pts[i * dim + d] = p_start[d] + t * (p_end[d] - p_start[d]);
        }
    }
    pts
}

#[cfg(test)]
mod objective_tests {
    use super::*;

    /// Build a plausible non-degenerate control polygon in LEO.
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

    /// The Gram-matrix objective must reproduce the integral it claims to compute.
    ///
    /// This is the hinge of the whole SCvx loop: the merit is assembled as
    ///     T = quad_form(H, f, x) + c_const + (tres - mres) + w_s * viol
    /// which collapses to the true merit ONLY IF
    ///     quad_form(H, f, x) + c_const == mres(x)
    /// i.e. only if the closed-form Bernstein-Gram assembly equals the numerically
    /// integrated model residual. If it does not, every rho and therefore every
    /// acceptance decision is computed against the wrong model, silently.
    ///
    /// `eval_residual_terms` is an INDEPENDENT implementation (midpoint quadrature
    /// over the curve, n_q = 1000) sharing only the gravity linearization, so it is
    /// a genuine oracle rather than a self-comparison.
    #[test]
    fn gram_integral_matches_numerical_quadrature() {
        let consts = OrbitalConstants::default();
        let dim = 3;
        let t = 1500.0;

        for &np1 in &[4usize, 6, 8] {
            for &n_lin_seg in &[1usize, 4, 16] {
                let p_ref = sample_polygon(np1);
                let nvars = np1 * dim;
                let lin = compute_segment_lin(&p_ref, np1, dim, t, n_lin_seg, &consts);
                let (h, f, c_const) =
                    build_ctrl_accel_quadratic(&p_ref, np1, dim, t, n_lin_seg, &consts, Some(&lin));

                // The identity must hold at ANY x for the fixed linearization, not
                // just at the reference — the QP moves x away from p_ref.
                for shift in [0.0f64, 15.0, -40.0] {
                    let x: Vec<f64> = p_ref.iter().enumerate()
                        .map(|(i, v)| v + shift * ((i % 7) as f64 - 3.0) / 3.0)
                        .collect();

                    let closed_form = quad_form(&h, &f, &x, nvars) + c_const;
                    let (quadrature, _) = eval_residual_terms(&x, &lin, np1, dim, t, &consts);

                    // Tolerance is set by the ORACLE's accuracy, not the Gram form's:
                    // the closed form is exact, the midpoint rule is O(h^2) and lands
                    // ~2e-7 here at n_q = 1000. Any real assembly bug (a dropped
                    // 1/T^2, a missing transpose in P2, a wrong (1'G1)c'c constant,
                    // the wrong w_seg) shifts the value by order 1 -- six orders above
                    // this bound -- so the loose-looking threshold still catches them.
                    let rel = (closed_form - quadrature).abs() / quadrature.abs().max(1e-30);
                    assert!(
                        rel < 1e-6,
                        "Gram closed form disagrees with quadrature: np1={np1} \
                         n_lin_seg={n_lin_seg} shift={shift} closed={closed_form:.12e} \
                         quad={quadrature:.12e} rel={rel:.3e}"
                    );
                }
            }
        }
    }

    /// The assembled Hessian must be symmetric and positive semi-definite, and the
    /// constant term non-negative -- it is a squared L2 norm.
    #[test]
    fn gram_quadratic_is_symmetric_psd() {
        let consts = OrbitalConstants::default();
        let (np1, dim, t) = (6usize, 3usize, 1500.0);
        let nvars = np1 * dim;
        let p_ref = sample_polygon(np1);
        let (h, _f, c_const) = build_ctrl_accel_quadratic(&p_ref, np1, dim, t, 4, &consts, None);

        assert!(c_const >= 0.0, "c_const must be >= 0, got {c_const}");
        for i in 0..nvars {
            for j in 0..nvars {
                let a = h[i * nvars + j];
                let b = h[j * nvars + i];
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0),
                    "H not symmetric at ({i},{j}): {a} vs {b}"
                );
            }
        }
        // PSD via Rayleigh quotients on deterministic pseudo-random directions.
        for seed in 0..25u64 {
            let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v: Vec<f64> = (0..nvars)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
                })
                .collect();
            let mut q = 0.0;
            for i in 0..nvars {
                for j in 0..nvars {
                    q += v[i] * h[i * nvars + j] * v[j];
                }
            }
            assert!(q >= -1e-9 * nvars as f64, "H not PSD: vHv = {q}");
        }
    }
}
