use crate::bezier;
use crate::constraints::LinearConstraint;
use crate::de_casteljau;
use crate::optimizer::{solve_qp, OptResult};
use crate::spacetime_constraints::{self, KozRowData, SpacetimeObstacleData};
use std::collections::HashMap;

fn build_spatial_energy_h(np1: usize, dim: usize) -> Vec<f64> {
    let n = np1 - 1;
    let nvars = np1 * dim;
    let spatial_dim = dim - 1;
    let mut h = vec![0.0; nvars * nvars];

    if let Some(g_tilde) = bezier::get_g_tilde(n) {
        for i in 0..np1 {
            for j in 0..np1 {
                let g_val = g_tilde[i * np1 + j];
                for d in 0..spatial_dim {
                    h[(i * dim + d) * nvars + (j * dim + d)] += g_val;
                }
            }
        }
    }

    h
}

fn compute_min_clearance(
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    n_eval: usize,
) -> f64 {
    if obstacles.n_obs == 0 {
        return f64::INFINITY;
    }

    let spatial_dim = dim - 1;
    let mut worst = f64::INFINITY;
    for i in 0..n_eval {
        let tau = if n_eval <= 1 {
            0.0
        } else {
            i as f64 / (n_eval - 1) as f64
        };
        let pt = bezier::evaluate(p, np1, dim, tau);
        let t_val = pt[dim - 1];

        for obs_idx in 0..obstacles.n_obs {
            if t_val < obstacles.t_start[obs_idx] || t_val > obstacles.t_end[obs_idx] {
                continue;
            }

            let mut dist_sq = 0.0;
            for d in 0..spatial_dim {
                let base = obs_idx * spatial_dim + d;
                let obs_pos = obstacles.pos0[base] + obstacles.vel[base] * t_val;
                let diff = pt[d] - obs_pos;
                dist_sq += diff * diff;
            }
            let clearance = dist_sq.sqrt() - obstacles.radii[obs_idx];
            if clearance < worst {
                worst = clearance;
            }
        }
    }
    worst
}

fn quadratic_cost(h: &[f64], x: &[f64], nvars: usize) -> f64 {
    let mut cost = 0.0;
    for i in 0..nvars {
        let mut hx = 0.0;
        for j in 0..nvars {
            hx += h[i * nvars + j] * x[j];
        }
        cost += 0.5 * x[i] * hx;
    }
    cost
}

fn append_constraint(
    constraint: &LinearConstraint,
    all_a_rows: &mut Vec<f64>,
    all_lb: &mut Vec<f64>,
    all_ub: &mut Vec<f64>,
    total_rows: &mut usize,
) {
    for r in 0..constraint.n_rows {
        all_a_rows.extend_from_slice(
            &constraint.a[r * constraint.n_vars..(r + 1) * constraint.n_vars],
        );
        all_lb.push(constraint.lb[r]);
        all_ub.push(constraint.ub[r]);
        *total_rows += 1;
    }
}

/// Data precomputed once before the SCP loop.
pub struct ScpPrecomputed {
    pub a_list: Vec<Vec<f64>>,
    pub h_energy: Vec<f64>,
    pub boundary: LinearConstraint,
    pub monotonicity: LinearConstraint,
    pub box_constraints: LinearConstraint,
    pub np1: usize,
    pub dim: usize,
}

/// Result of a single SCP iteration.
pub struct ScpStepResult {
    pub p_new: Vec<f64>,
    pub solver_status: String,
    pub delta: f64,
    pub raw_step_norm: f64,
    pub clearance: f64,
    pub total_slack: f64,
    pub max_slack: f64,
    pub converged: bool,
    pub cost: f64,
    pub koz_rows: Vec<KozRowData>,
    pub koz_slack_per_row: Vec<f64>,
}

/// Precompute the data that stays constant across SCP iterations.
pub fn precompute_scp(
    p_init: &[f64],
    np1: usize,
    dim: usize,
    n_seg: usize,
    min_dt: f64,
    coord_lb: f64,
    coord_ub: f64,
    time_lb: f64,
    time_ub: f64,
) -> ScpPrecomputed {
    let n = np1 - 1;
    ScpPrecomputed {
        a_list: de_casteljau::segment_matrices_equal_params(n, n_seg),
        h_energy: build_spatial_energy_h(np1, dim),
        boundary: spacetime_constraints::build_boundary_constraints(
            np1, dim, &p_init[0..dim], &p_init[(np1 - 1) * dim..np1 * dim],
        ),
        monotonicity: spacetime_constraints::build_time_monotonicity(np1, dim, min_dt),
        box_constraints: spacetime_constraints::build_box_constraints(
            p_init, np1, dim, coord_lb, coord_ub, time_lb, time_ub,
        ),
        np1,
        dim,
    }
}

/// Run one SCP iteration: linearize KOZ, build QP, solve, clip, evaluate.
pub fn scp_step(
    p_current: &[f64],
    pre: &ScpPrecomputed,
    obstacles: &SpacetimeObstacleData<'_>,
    scp_prox_weight: f64,
    scp_trust_radius: f64,
    elastic_weight: f64,
    tol: f64,
    capsule_time_scale: f64,
) -> ScpStepResult {
    let np1 = pre.np1;
    let dim = pre.dim;
    let nvars = np1 * dim;

    // Build KOZ constraints with per-row metadata
    let koz_bundle = spacetime_constraints::build_spacetime_koz_constraints(
        &pre.a_list, p_current, np1, dim, obstacles, capsule_time_scale,
    );

    // Build objective: H_energy + proximal
    let mut h = pre.h_energy.clone();
    let mut f = vec![0.0; nvars];
    if scp_prox_weight > 0.0 {
        for i in 0..nvars {
            h[i * nvars + i] += scp_prox_weight;
            f[i] -= scp_prox_weight * p_current[i];
        }
    }

    // Assemble all constraints
    let mut all_a_rows: Vec<f64> = Vec::new();
    let mut all_lb: Vec<f64> = Vec::new();
    let mut all_ub: Vec<f64> = Vec::new();
    let mut total_rows = 0usize;

    append_constraint(&pre.box_constraints, &mut all_a_rows, &mut all_lb, &mut all_ub, &mut total_rows);
    append_constraint(&pre.boundary, &mut all_a_rows, &mut all_lb, &mut all_ub, &mut total_rows);
    append_constraint(&pre.monotonicity, &mut all_a_rows, &mut all_lb, &mut all_ub, &mut total_rows);

    let koz_row_start = total_rows;
    if let Some(ref bundle) = koz_bundle {
        append_constraint(&bundle.constraint, &mut all_a_rows, &mut all_lb, &mut all_ub, &mut total_rows);
    }
    let n_koz = total_rows - koz_row_start;

    // Solve QP — try hard first, elastic fallback if infeasible
    let (x_new, iter_total_slack, iter_max_slack, koz_slack_per_row, solver_status);

    let hard_sol = solve_qp(&h, &f, &all_a_rows, &all_lb, &all_ub, nvars, total_rows);

    if let Some(x_sol) = hard_sol {
        x_new = x_sol;
        iter_total_slack = 0.0;
        iter_max_slack = 0.0;
        koz_slack_per_row = vec![0.0; n_koz];
        solver_status = "Solved".to_string();
    } else if elastic_weight > 0.0 && n_koz > 0 {
        let nvars_ext = nvars + n_koz;
        let ext_nrows = total_rows + n_koz;

        let mut h_ext = vec![0.0; nvars_ext * nvars_ext];
        for i in 0..nvars {
            for j in 0..nvars {
                h_ext[i * nvars_ext + j] = h[i * nvars + j];
            }
        }

        let mut f_ext = vec![0.0; nvars_ext];
        f_ext[..nvars].copy_from_slice(&f);
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

        match solve_qp(&h_ext, &f_ext, &a_ext, &lb_ext, &ub_ext, nvars_ext, ext_nrows) {
            Some(x_full) => {
                x_new = x_full[..nvars].to_vec();
                koz_slack_per_row = x_full[nvars..].to_vec();
                iter_total_slack = koz_slack_per_row.iter().sum::<f64>();
                iter_max_slack = koz_slack_per_row.iter().cloned().fold(0.0f64, f64::max);
                solver_status = "Elastic".to_string();
            }
            None => {
                return ScpStepResult {
                    p_new: p_current.to_vec(),
                    solver_status: "Failed".to_string(),
                    delta: 0.0,
                    raw_step_norm: 0.0,
                    clearance: compute_min_clearance(p_current, np1, dim, obstacles, 1500),
                    total_slack: 0.0,
                    max_slack: 0.0,
                    converged: false,
                    cost: quadratic_cost(&pre.h_energy, p_current, nvars),
                    koz_rows: Vec::new(),
                    koz_slack_per_row: Vec::new(),
                };
            }
        }
    } else {
        return ScpStepResult {
            p_new: p_current.to_vec(),
            solver_status: "Failed".to_string(),
            delta: 0.0,
            raw_step_norm: 0.0,
            clearance: compute_min_clearance(p_current, np1, dim, obstacles, 1500),
            total_slack: 0.0,
            max_slack: 0.0,
            converged: false,
            cost: quadratic_cost(&pre.h_energy, p_current, nvars),
            koz_rows: Vec::new(),
            koz_slack_per_row: Vec::new(),
        };
    }

    // Trust region clipping
    let mut x_result = x_new.clone();
    let raw_step_norm = (0..nvars)
        .map(|i| (x_result[i] - p_current[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    if scp_trust_radius > 0.0 && raw_step_norm > scp_trust_radius && raw_step_norm > 1e-15 {
        let alpha = scp_trust_radius / raw_step_norm;
        for i in 0..nvars {
            x_result[i] = p_current[i] + alpha * (x_result[i] - p_current[i]);
        }
    }

    let delta = (0..nvars)
        .map(|i| (x_result[i] - p_current[i]).powi(2))
        .sum::<f64>()
        .sqrt();

    let clearance = compute_min_clearance(&x_result, np1, dim, obstacles, 1500);
    let cost = quadratic_cost(&pre.h_energy, &x_result, nvars);
    let converged = delta < tol && iter_total_slack < 1e-10;

    // Move per-row metadata out of the bundle
    let koz_rows_out = if let Some(bundle) = koz_bundle {
        bundle.rows
    } else {
        Vec::new()
    };

    ScpStepResult {
        p_new: x_result,
        solver_status,
        delta,
        raw_step_norm,
        clearance,
        total_slack: iter_total_slack,
        max_slack: iter_max_slack,
        converged,
        cost,
        koz_rows: koz_rows_out,
        koz_slack_per_row,
    }
}

/// Full SCP optimization loop. Thin wrapper over precompute_scp + scp_step.
pub fn optimize_spacetime(
    p_init: &[f64],
    np1: usize,
    dim: usize,
    n_seg: usize,
    max_iter: usize,
    tol: f64,
    scp_prox_weight: f64,
    scp_trust_radius: f64,
    min_dt: f64,
    coord_lb: f64,
    coord_ub: f64,
    time_lb: f64,
    time_ub: f64,
    obstacles: &SpacetimeObstacleData<'_>,
    elastic_weight: f64,
    capsule_time_scale: f64,
) -> OptResult {
    let nvars = np1 * dim;
    let pre = precompute_scp(p_init, np1, dim, n_seg, min_dt, coord_lb, coord_ub, time_lb, time_ub);
    let mut p = p_init.to_vec();
    let mut best_p = p.clone();
    let mut best_clearance = compute_min_clearance(&best_p, np1, dim, obstacles, 1500);
    let mut iterations = 0usize;
    let mut last_delta = f64::NAN;
    let mut last_total_slack = 0.0f64;
    let mut last_max_slack = 0.0f64;

    for it in 1..=max_iter {
        iterations = it;
        let step = scp_step(
            &p, &pre, obstacles, scp_prox_weight, scp_trust_radius, elastic_weight, tol,
            capsule_time_scale,
        );

        if step.solver_status == "Failed" {
            break;
        }

        last_delta = step.delta;
        last_total_slack = step.total_slack;
        last_max_slack = step.max_slack;
        p = step.p_new;

        if step.clearance > 0.0 && step.clearance > best_clearance {
            best_clearance = step.clearance;
            best_p = p.clone();
        }
        if step.converged {
            break;
        }
    }

    let mut final_clearance = compute_min_clearance(&p, np1, dim, obstacles, 1500);
    if final_clearance < 0.0 && best_clearance > 0.0 {
        p = best_p;
        final_clearance = best_clearance;
    }

    let feasible = final_clearance > 0.0 || obstacles.n_obs == 0;
    let cost = quadratic_cost(&pre.h_energy, &p, nvars);

    let mut info = HashMap::new();
    info.insert("iterations".to_string(), iterations as f64);
    info.insert("feasible".to_string(), if feasible { 1.0 } else { 0.0 });
    info.insert("min_clearance".to_string(), final_clearance);
    info.insert("cost_true_energy".to_string(), cost);
    info.insert("cost_no_const".to_string(), cost);
    info.insert("cost".to_string(), cost);
    info.insert("max_control_accel_ms2".to_string(), 0.0);
    info.insert("mean_control_accel_ms2".to_string(), 0.0);
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
