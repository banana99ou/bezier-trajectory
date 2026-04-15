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
}

/// Build quadratic objective: 0.5 x^T H x + f^T x + c
/// for control acceleration energy with gravity+J2 linearized about P_ref.
fn build_ctrl_accel_quadratic(
    p_ref: &[f64],  // (Np1, dim) row-major
    np1: usize,
    dim: usize,
    t: f64,
    sample_count: usize,
    objective: &str,
    irls_eps: f64,
    geom_reg: f64,
    consts: &OrbitalConstants,
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = np1 - 1;
    let nvars = np1 * dim;

    let mut big_q = vec![0.0; nvars * nvars];
    let mut q_vec = vec![0.0; nvars];
    let mut c_const = 0.0f64;

    // 1) Geometric regularization via G_tilde
    if let Some(g_tilde) = bezier::get_g_tilde(n) {
        let w_geom = if objective == "energy" { 1.0 } else { geom_reg };
        if w_geom != 0.0 {
            let scale = w_geom / (t.powi(4));
            // Q += scale * kron(G_tilde, I_dim)
            for i in 0..np1 {
                for j in 0..np1 {
                    let g_val = g_tilde[i * np1 + j];
                    for d in 0..dim {
                        big_q[(i * dim + d) * nvars + (j * dim + d)] += scale * g_val;
                    }
                }
            }
        }
    }

    // 2) Gravity/J2 linearization via De Casteljau segment centroids
    let n_lin_seg = sample_count.max(1);
    let a_seg_list = de_casteljau::segment_matrices_equal_params(n, n_lin_seg);
    let w_seg = 1.0 / n_lin_seg as f64;

    // Compute L = E @ D @ E @ D (linear mapping for acceleration)
    let sz = np1;
    let d_mat = bezier::get_d_matrix(n);
    let e_mat = bezier::get_e_matrix(n - 1);
    let ed = bezier::matmul(&e_mat, sz, n, &d_mat, sz);
    let l_mat = bezier::matmul(&ed, sz, sz, &ed, sz);

    for a_seg in &a_seg_list {
        // Segment centroid weights: w_row = mean of rows of A_seg
        let mut w_row = vec![0.0; np1];
        for i in 0..np1 {
            for j in 0..np1 {
                w_row[j] += a_seg[i * np1 + j];
            }
        }
        for j in 0..np1 {
            w_row[j] /= np1 as f64;
        }

        // R_i = kron(w_row, I_dim): maps x -> r_i (centroid position)
        // A_i = (1/T^2) * kron(a_row, I_dim): maps x -> geometric accel at centroid
        let mut a_row = vec![0.0; np1];
        for i in 0..np1 {
            for j in 0..np1 {
                a_row[i] += w_row[j] * l_mat[j * np1 + i];
            }
        }
        // Actually: a_row[i] = sum_j w_row[j] * L[j, i] ... let me redo this.
        // In Python: a_row = (w_row @ L), i.e. a_row[i] = sum_j w_row[j] * L[j][i]
        let mut a_row2 = vec![0.0; np1];
        for i in 0..np1 {
            let mut s = 0.0;
            for j in 0..np1 {
                s += w_row[j] * l_mat[j * np1 + i];
            }
            a_row2[i] = s;
        }

        // Reference point r_ref = R_i @ x_ref = sum_j w_row[j] * P_ref[j, :]
        let mut r_ref = [0.0f64; 3];
        for j in 0..np1 {
            for d in 0..dim {
                r_ref[d] += w_row[j] * p_ref[j * dim + d];
            }
        }

        let g_ref = gravity::accel_total(&r_ref, consts.mu, consts.r_e_km, consts.j2);
        let j_i = gravity::jacobian_numeric(&r_ref, consts.mu, consts.r_e_km, consts.j2, 1e-3);

        // c_i = g_ref - J_i @ r_ref
        let mut c_i = [0.0f64; 3];
        for row in 0..3 {
            let mut jr = 0.0;
            for col in 0..3 {
                jr += j_i[row][col] * r_ref[col];
            }
            c_i[row] = g_ref[row] - jr;
        }

        // M_i = A_i - B_i where B_i = J_i @ R_i
        // M_i is (dim, nvars)
        // A_i[d, j*dim+d] = (1/T^2) * a_row2[j]
        // B_i[d, j*dim+dd] = J_i[d][dd_inner] * w_row[j] (when dd_inner matches)
        // Actually B_i = J_i @ R_i, R_i is (3, nvars)
        // R_i[d, j*dim+d] = w_row[j]
        let t2_inv = 1.0 / (t * t);
        let mut m_i = vec![0.0f64; dim * nvars];
        for d in 0..dim {
            for j in 0..np1 {
                // A_i contribution
                m_i[d * nvars + j * dim + d] += t2_inv * a_row2[j];
                // -B_i contribution: -sum_{dd} J_i[d][dd] * w_row[j] * delta(col, j*dim+dd)
                for dd in 0..dim {
                    m_i[d * nvars + j * dim + dd] -= j_i[d][dd] * w_row[j];
                }
            }
        }

        // IRLS weight for dv objective
        let mut w_i = w_seg;
        if objective == "dv" {
            // r_ref_res = M_i @ x_ref - c_i
            let mut r_ref_res = [0.0f64; 3];
            for d in 0..dim {
                let mut mx = 0.0;
                for v in 0..nvars {
                    mx += m_i[d * nvars + v] * p_ref[v]; // p_ref is already flat? No, it's (Np1, dim) row-major
                    // Actually p_ref here is already the flat x_ref
                }
                r_ref_res[d] = mx - c_i[d];
            }
            // Wait, p_ref is (Np1, dim) row-major, which IS the same as x_ref flat.
            let alpha = (r_ref_res[0] * r_ref_res[0]
                + r_ref_res[1] * r_ref_res[1]
                + r_ref_res[2] * r_ref_res[2]
                + irls_eps)
                .sqrt();
            w_i = w_seg / alpha;
        }

        // Q += w_i * M_i^T @ M_i
        // q += w_i * (-M_i^T @ c_i)
        // c_const += w_i * c_i^T c_i
        for v1 in 0..nvars {
            for v2 in 0..nvars {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += m_i[d * nvars + v1] * m_i[d * nvars + v2];
                }
                big_q[v1 * nvars + v2] += w_i * dot;
            }
            let mut dot_c = 0.0;
            for d in 0..dim {
                dot_c += m_i[d * nvars + v1] * c_i[d];
            }
            q_vec[v1] += w_i * (-dot_c);
        }
        c_const += w_i * (c_i[0] * c_i[0] + c_i[1] * c_i[1] + c_i[2] * c_i[2]);
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
) -> Option<Vec<f64>> {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{DefaultSettings, DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus};

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

    let total_rows = a_rows.len();

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

    let settings = DefaultSettingsBuilder::default()
        .max_iter(200)
        .verbose(false)
        .build()
        .unwrap();

    let mut solver = match DefaultSolver::new(&p_csc, f, &a_csc, &b_vals2, &cones2, settings) {
        Ok(s) => s,
        Err(_) => return None,
    };
    solver.solve();

    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            Some(solver.solution.x.clone())
        }
        _ => None,
    }
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
    sample_count: usize,
    objective_mode: &str,
    irls_eps: f64,
    geom_reg: f64,
    scp_prox_weight: f64,
    scp_trust_radius: f64,
    v0: Option<&[f64]>,
    v1: Option<&[f64]>,
    a0: Option<&[f64]>,
    a1: Option<&[f64]>,
    enforce_prograde: bool,
    prograde_n_samples: usize,
    elastic_weight: f64,
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

    for it in 1..=max_iter {
        iterations = it;

        // Build KOZ constraints
        let c_koz = vec![0.0; dim];
        let koz = constraints::build_koz_constraints(&a_list, &p, np1, dim, r_e, &c_koz);

        // Build quadratic objective
        let (mut h_mat, mut f_vec, _c_const) = build_ctrl_accel_quadratic(
            &p,
            np1,
            dim,
            t,
            sample_count,
            objective_mode,
            irls_eps,
            geom_reg,
            &consts,
        );

        // Proximal regularization
        if scp_prox_weight > 0.0 {
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

        // Solve QP -- try hard constraints first; if infeasible and elastic
        // relaxation is enabled, retry with slack variables on KOZ rows.
        let (x_new, iter_total_slack, iter_max_slack);

        let hard_sol = solve_qp(
            &h_mat, &f_vec, &all_a_rows, &all_lb, &all_ub, nvars, total_rows,
        );

        if let Some(x_sol) = hard_sol {
            x_new = x_sol;
            iter_total_slack = 0.0;
            iter_max_slack = 0.0;
        } else if elastic_weight > 0.0 && n_koz > 0 {
            // Hard QP infeasible -- retry with elastic relaxation.
            // Add slack s_k >= 0 to each KOZ row: A_koz x + s >= b_koz,
            // with L1 penalty M * sum(s) in the objective.
            let nvars_ext = nvars + n_koz;
            let ext_nrows = total_rows + n_koz;

            let mut h_ext = vec![0.0; nvars_ext * nvars_ext];
            for i in 0..nvars {
                for j in 0..nvars {
                    h_ext[i * nvars_ext + j] = h_mat[i * nvars + j];
                }
            }

            let mut f_ext = vec![0.0; nvars_ext];
            f_ext[..nvars].copy_from_slice(&f_vec);
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

            let x_full = match solve_qp(
                &h_ext, &f_ext, &a_ext, &lb_ext, &ub_ext, nvars_ext, ext_nrows,
            ) {
                Some(x) => x,
                None => break,
            };

            x_new = x_full[..nvars].to_vec();
            iter_total_slack = x_full[nvars..].iter().sum::<f64>();
            iter_max_slack = x_full[nvars..].iter().cloned().fold(0.0f64, f64::max);
        } else {
            break;
        }

        last_total_slack = iter_total_slack;
        last_max_slack = iter_max_slack;

        // Trust region clipping
        let mut x_result = x_new.clone();
        let step_norm: f64 = (0..nvars)
            .map(|i| (x_result[i] - p[i]).powi(2))
            .sum::<f64>()
            .sqrt();

        if scp_trust_radius > 0.0 && step_norm > scp_trust_radius && step_norm > 1e-15 {
            let alpha = scp_trust_radius / step_norm;
            for i in 0..nvars {
                x_result[i] = p[i] + alpha * (x_result[i] - p[i]);
            }
        }

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

    // Final feasibility check
    let n_check = 1000;
    let mut min_radius = f64::INFINITY;
    for i in 0..=n_check {
        let tau = i as f64 / n_check as f64;
        let pt = bezier::evaluate(&p, np1, dim, tau);
        let r: f64 = pt.iter().map(|x| x * x).sum::<f64>().sqrt();
        if r < min_radius {
            min_radius = r;
        }
    }

    // Compute final cost
    let (hf, ff, cf) = build_ctrl_accel_quadratic(
        &p, np1, dim, t, sample_count, objective_mode, irls_eps, geom_reg, &consts,
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
    let mut a_u_samples = Vec::with_capacity(n_metrics + 1);

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
        a_u_samples.push(a_u_m_s2);
        if a_u_m_s2 > max_ctrl_accel {
            max_ctrl_accel = a_u_m_s2;
        }
        sum_ctrl_accel += a_u_m_s2;
    }
    let mean_ctrl_accel = sum_ctrl_accel / (n_metrics + 1) as f64;

    // Delta-v proxy (trapezoidal)
    let mut dv_proxy = 0.0f64;
    let dt_tau = 1.0 / n_metrics as f64;
    for i in 0..n_metrics {
        dv_proxy += 0.5 * (a_u_samples[i] + a_u_samples[i + 1]) * dt_tau;
    }
    dv_proxy *= t; // scale to physical time

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
    info.insert("dv_proxy_m_s".to_string(), dv_proxy);
    info.insert("final_delta_norm".to_string(), last_delta);
    info.insert("total_koz_slack".to_string(), last_total_slack);
    info.insert("max_koz_slack".to_string(), last_max_slack);

    OptResult {
        p_opt: p,
        np1,
        dim,
        info,
        feasible,
        iterations,
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
