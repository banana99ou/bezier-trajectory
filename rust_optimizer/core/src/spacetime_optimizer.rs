use crate::bezier;
use crate::constraints::LinearConstraint;
use crate::de_casteljau;
use crate::optimizer::{solve_qp, OptResult};
use crate::spacetime_constraints::{self, SpacetimeObstacleData};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn control_points_to_rows(p: &[f64], np1: usize, dim: usize) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(np1);
    for i in 0..np1 {
        rows.push(p[i * dim..(i + 1) * dim].to_vec());
    }
    rows
}

fn debug_log(run_id: &str, hypothesis_id: &str, location: &str, message: &str, data: serde_json::Value) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0);
    let payload = serde_json::json!({
        "sessionId": "9abff6",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": timestamp,
    });
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/Volumes/Sandisk/code/bezier-trajectory-merge/.cursor/debug-9abff6.log")
    {
        let _ = writeln!(file, "{payload}");
    }
}

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

pub fn optimize_spacetime(
    p_init: &[f64], // (np1, dim) row-major
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
) -> OptResult {
    let n = np1 - 1;
    let nvars = np1 * dim;
    let mut p = p_init.to_vec();
    let mut best_p = p.clone();
    let a_list = de_casteljau::segment_matrices_equal_params(n, n_seg);
    let h_energy = build_spatial_energy_h(np1, dim);
    let boundary = spacetime_constraints::build_boundary_constraints(
        np1,
        dim,
        &p_init[0..dim],
        &p_init[(np1 - 1) * dim..np1 * dim],
    );
    let monotonicity = spacetime_constraints::build_time_monotonicity(np1, dim, min_dt);
    let box_constraints = spacetime_constraints::build_box_constraints(
        p_init,
        np1,
        dim,
        coord_lb,
        coord_ub,
        time_lb,
        time_ub,
    );

    let mut best_clearance = compute_min_clearance(&best_p, np1, dim, obstacles, 1500);
    let mut iterations = 0usize;
    let mut last_delta = f64::NAN;
    let debug_run_id = format!(
        "scp-np{}_dim{}_seg{}_obs{}_max{}",
        np1, dim, n_seg, obstacles.n_obs, max_iter
    );
    // #region agent log
    debug_log(
        &debug_run_id,
        "H0",
        "rust_optimizer/core/src/spacetime_optimizer.rs:optimize_spacetime:entry",
        "spacetime optimize entry",
        serde_json::json!({
            "buildTag": "scp-debug-20260410a",
            "np1": np1,
            "dim": dim,
            "n_seg": n_seg,
            "max_iter": max_iter,
            "tol": tol,
            "n_obs": obstacles.n_obs,
            "initial_clearance": best_clearance,
            "control_points": control_points_to_rows(&p, np1, dim),
        }),
    );
    // #endregion

    for it in 1..=max_iter {
        iterations = it;
        let koz = spacetime_constraints::build_spacetime_koz_constraints(
            &a_list,
            &p,
            np1,
            dim,
            obstacles,
        );

        let mut h = h_energy.clone();
        let mut f = vec![0.0; nvars];
        if scp_prox_weight > 0.0 {
            for i in 0..nvars {
                h[i * nvars + i] += scp_prox_weight;
                f[i] -= scp_prox_weight * p[i];
            }
        }

        let mut all_a_rows: Vec<f64> = Vec::new();
        let mut all_lb: Vec<f64> = Vec::new();
        let mut all_ub: Vec<f64> = Vec::new();
        let mut total_rows = 0usize;

        append_constraint(&box_constraints, &mut all_a_rows, &mut all_lb, &mut all_ub, &mut total_rows);
        append_constraint(&boundary, &mut all_a_rows, &mut all_lb, &mut all_ub, &mut total_rows);
        append_constraint(
            &monotonicity,
            &mut all_a_rows,
            &mut all_lb,
            &mut all_ub,
            &mut total_rows,
        );
        if let Some(koz_constraint) = &koz {
            append_constraint(
                koz_constraint,
                &mut all_a_rows,
                &mut all_lb,
                &mut all_ub,
                &mut total_rows,
            );
        }

        // #region agent log
        debug_log(
            &debug_run_id,
            "H1_H3",
            "rust_optimizer/core/src/spacetime_optimizer.rs:optimize_spacetime:pre_solve",
            "assembled scp subproblem",
            serde_json::json!({
                "iteration": it,
                "koz_present": koz.is_some(),
                "koz_rows": koz.as_ref().map(|constraint| constraint.n_rows).unwrap_or(0),
                "total_rows": total_rows,
                "best_clearance_before": best_clearance,
                "control_points": control_points_to_rows(&p, np1, dim),
            }),
        );
        // #endregion

        let x_new = match solve_qp(
            &h,
            &f,
            &all_a_rows,
            &all_lb,
            &all_ub,
            nvars,
            total_rows,
        ) {
            Some(x) => x,
            None => {
                // #region agent log
                debug_log(
                    &debug_run_id,
                    "H1",
                    "rust_optimizer/core/src/spacetime_optimizer.rs:optimize_spacetime:solve_failure",
                    "qp solve returned none",
                    serde_json::json!({
                        "iteration": it,
                        "koz_present": koz.is_some(),
                        "koz_rows": koz.as_ref().map(|constraint| constraint.n_rows).unwrap_or(0),
                        "total_rows": total_rows,
                        "control_points": control_points_to_rows(&p, np1, dim),
                    }),
                );
                // #endregion
                break;
            }
        };

        let mut x_result = x_new.clone();
        let step_norm = (0..nvars)
            .map(|i| (x_result[i] - p[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        if scp_trust_radius > 0.0 && step_norm > scp_trust_radius && step_norm > 1e-15 {
            let alpha = scp_trust_radius / step_norm;
            for i in 0..nvars {
                x_result[i] = p[i] + alpha * (x_result[i] - p[i]);
            }
        }

        let delta = (0..nvars)
            .map(|i| (x_result[i] - p[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        last_delta = delta;
        p = x_result;

        let clearance = compute_min_clearance(&p, np1, dim, obstacles, 1500);
        // #region agent log
        debug_log(
            &debug_run_id,
            "H2",
            "rust_optimizer/core/src/spacetime_optimizer.rs:optimize_spacetime:post_solve",
            "qp solve produced candidate",
            serde_json::json!({
                "iteration": it,
                "step_norm": step_norm,
                "delta": delta,
                "clearance_after": clearance,
                "tol_break": delta < tol,
                "accepted_control_points": control_points_to_rows(&p, np1, dim),
            }),
        );
        // #endregion
        if clearance > 0.0 && clearance > best_clearance {
            best_clearance = clearance;
            best_p = p.clone();
        }
        if delta < tol {
            break;
        }
    }

    let mut final_clearance = compute_min_clearance(&p, np1, dim, obstacles, 1500);
    if final_clearance < 0.0 && best_clearance > 0.0 {
        p = best_p;
        final_clearance = best_clearance;
    }

    let feasible = final_clearance > 0.0 || obstacles.n_obs == 0;
    let cost = quadratic_cost(&h_energy, &p, nvars);
    // #region agent log
    debug_log(
        &debug_run_id,
        "H4",
        "rust_optimizer/core/src/spacetime_optimizer.rs:optimize_spacetime:final",
        "spacetime optimize exit",
        serde_json::json!({
            "iterations_reported": iterations,
            "final_clearance": final_clearance,
            "feasible": feasible,
            "final_delta_norm": last_delta,
            "final_control_points": control_points_to_rows(&p, np1, dim),
        }),
    );
    // #endregion

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

    OptResult {
        p_opt: p,
        np1,
        dim,
        info,
        feasible,
        iterations,
    }
}
