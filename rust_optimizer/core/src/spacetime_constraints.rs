use crate::constraints::LinearConstraint;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn debug_log_constraints(run_id: &str, hypothesis_id: &str, location: &str, message: &str, data: serde_json::Value) {
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

pub struct SpacetimeObstacleData<'a> {
    pub pos0: &'a [f64],    // (n_obs, spatial_dim) row-major
    pub vel: &'a [f64],     // (n_obs, spatial_dim) row-major
    pub radii: &'a [f64],   // (n_obs,)
    pub t_start: &'a [f64], // (n_obs,)
    pub t_end: &'a [f64],   // (n_obs,)
    pub n_obs: usize,
    pub spatial_dim: usize,
}

const SPACETIME_CAPSULE_TIME_SCALE: f64 = 0.5;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn scaled_spacetime_capsule_support(
    query_orig: &[f64],
    obstacles: &SpacetimeObstacleData<'_>,
    obs_idx: usize,
    time_start: f64,
    time_end: f64,
    time_scale: f64,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
    let spatial_dim = obstacles.spatial_dim;
    let dim = spatial_dim + 1;
    let eff_t0 = obstacles.t_start[obs_idx].max(time_start);
    let eff_t1 = obstacles.t_end[obs_idx].min(time_end);
    if eff_t1 < eff_t0 {
        return None;
    }

    let mut query_scaled = query_orig.to_vec();
    query_scaled[dim - 1] *= time_scale;

    let mut a_scaled = vec![0.0; dim];
    let mut b_scaled = vec![0.0; dim];
    for d in 0..spatial_dim {
        let base = obs_idx * spatial_dim + d;
        a_scaled[d] = obstacles.pos0[base] + obstacles.vel[base] * eff_t0;
        b_scaled[d] = obstacles.pos0[base] + obstacles.vel[base] * eff_t1;
    }
    a_scaled[dim - 1] = time_scale * eff_t0;
    b_scaled[dim - 1] = time_scale * eff_t1;

    let axis: Vec<f64> = b_scaled.iter().zip(a_scaled.iter()).map(|(b, a)| b - a).collect();
    let denom = dot(&axis, &axis);
    let rel: Vec<f64> = query_scaled
        .iter()
        .zip(a_scaled.iter())
        .map(|(q, a)| q - a)
        .collect();
    let tau = if denom > 1e-12 {
        (dot(&rel, &axis) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let closest_scaled: Vec<f64> = a_scaled
        .iter()
        .zip(axis.iter())
        .map(|(a, axis_val)| a + tau * axis_val)
        .collect();
    let diff_scaled: Vec<f64> = query_scaled
        .iter()
        .zip(closest_scaled.iter())
        .map(|(q, c)| q - c)
        .collect();
    let dist = dot(&diff_scaled, &diff_scaled).sqrt();
    if dist <= 1e-10 {
        return None;
    }

    let n_scaled: Vec<f64> = diff_scaled.iter().map(|value| value / dist).collect();
    let mut n_orig = n_scaled.clone();
    n_orig[dim - 1] *= time_scale;

    let radius = obstacles.radii[obs_idx];
    let support_scaled: Vec<f64> = closest_scaled
        .iter()
        .zip(n_scaled.iter())
        .map(|(c, n)| c + radius * n)
        .collect();
    let mut support_orig = support_scaled.clone();
    support_orig[dim - 1] /= time_scale;

    let mut closest_orig = closest_scaled.clone();
    closest_orig[dim - 1] /= time_scale;
    let lb = dot(&n_orig, &support_orig);

    Some((n_orig, support_orig, closest_orig, n_scaled, lb))
}

pub fn build_spacetime_koz_constraints(
    a_list: &[Vec<f64>],
    p: &[f64], // (np1, dim) row-major
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
) -> Option<LinearConstraint> {
    let n_vars = np1 * dim;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut lbs: Vec<f64> = Vec::new();
    let debug_run_id = format!("koz-np{}_dim{}_seg{}_obs{}", np1, dim, a_list.len(), obstacles.n_obs);
    let plan_t0 = p[dim - 1];
    let plan_t1 = p[(np1 - 1) * dim + (dim - 1)];
    let mut total_active_obstacles = 0usize;
    let mut total_violated_rows = 0usize;
    let mut min_margin = f64::INFINITY;
    let mut worst_segment = usize::MAX;
    let mut worst_obstacle = usize::MAX;
    let mut worst_segment_point: Vec<f64> = Vec::new();
    let mut worst_obstacle_center: Vec<f64> = Vec::new();
    let mut worst_normal: Vec<f64> = Vec::new();
    let mut worst_support_point: Vec<f64> = Vec::new();
    let mut worst_lb = f64::NAN;
    let mut worst_lhs = f64::NAN;

    if obstacles.n_obs == 0 {
        return None;
    }

    for (seg_idx, a_seg) in a_list.iter().enumerate() {
        let mut q = vec![0.0; np1 * dim];
        for i in 0..np1 {
            for d in 0..dim {
                let mut sum = 0.0;
                for j in 0..np1 {
                    sum += a_seg[i * np1 + j] * p[j * dim + d];
                }
                q[i * dim + d] = sum;
            }
        }

        let mut centroid = vec![0.0; dim];
        for i in 0..np1 {
            for d in 0..dim {
                centroid[d] += q[i * dim + d];
            }
        }
        for d in 0..dim {
            centroid[d] /= np1 as f64;
        }

        for k in 0..np1 {
            let query_point: Vec<f64> = (0..dim).map(|d| q[k * dim + d]).collect();
            for obs_idx in 0..obstacles.n_obs {
                let Some((n_orig, support_orig, closest_orig, n_scaled, lb)) =
                    scaled_spacetime_capsule_support(
                        &query_point,
                        obstacles,
                        obs_idx,
                        plan_t0,
                        plan_t1,
                        SPACETIME_CAPSULE_TIME_SCALE,
                    )
                else {
                    continue;
                };
                total_active_obstacles += 1;

                let mut row = vec![0.0; n_vars];
                for j in 0..np1 {
                    let coeff = a_seg[k * np1 + j];
                    for d in 0..dim {
                        row[j * dim + d] += coeff * n_orig[d];
                    }
                }
                let lhs = (0..n_vars).map(|idx| row[idx] * p[idx]).sum::<f64>();
                let margin = lhs - lb;
                if margin < 0.0 {
                    total_violated_rows += 1;
                }
                if margin < min_margin {
                    min_margin = margin;
                    worst_segment = seg_idx;
                    worst_obstacle = obs_idx;
                    worst_segment_point = query_point.clone();
                    worst_obstacle_center = closest_orig;
                    worst_normal = n_scaled;
                    worst_support_point = support_orig;
                    worst_lb = lb;
                    worst_lhs = lhs;
                }
                rows.push(row);
                lbs.push(lb);
            }
        }
    }

    if rows.is_empty() {
        return None;
    }

    let n_rows = rows.len();
    let mut a = vec![0.0; n_rows * n_vars];
    for (i, row) in rows.iter().enumerate() {
        a[i * n_vars..(i + 1) * n_vars].copy_from_slice(row);
    }

    // #region agent log
    debug_log_constraints(
        &debug_run_id,
        "H7_H8_H9_H10",
        "rust_optimizer/core/src/spacetime_constraints.rs:build_spacetime_koz_constraints:summary",
        "built spacetime koz constraints",
        serde_json::json!({
            "n_rows": n_rows,
            "active_obstacles": total_active_obstacles,
            "violated_rows_at_reference": total_violated_rows,
            "min_margin_at_reference": min_margin,
            "worst_segment": if worst_segment == usize::MAX { None } else { Some(worst_segment) },
            "worst_obstacle": if worst_obstacle == usize::MAX { None } else { Some(worst_obstacle) },
            "worst_segment_point": worst_segment_point,
            "worst_obstacle_center": worst_obstacle_center,
            "worst_normal_with_time_coeff": worst_normal,
            "worst_support_point": worst_support_point,
            "worst_lb": worst_lb,
            "worst_lhs": worst_lhs,
        }),
    );
    // #endregion

    Some(LinearConstraint {
        a,
        lb: lbs,
        ub: vec![f64::INFINITY; n_rows],
        n_rows,
        n_vars,
    })
}

pub fn build_boundary_constraints(np1: usize, dim: usize, p_start: &[f64], p_end: &[f64]) -> LinearConstraint {
    let n_vars = np1 * dim;
    let n_rows = 2 * dim;
    let mut a = vec![0.0; n_rows * n_vars];
    let mut lb = vec![0.0; n_rows];
    let mut ub = vec![0.0; n_rows];

    for d in 0..dim {
        a[d * n_vars + d] = 1.0;
        lb[d] = p_start[d];
        ub[d] = p_start[d];

        let row_idx = dim + d;
        a[row_idx * n_vars + (np1 - 1) * dim + d] = 1.0;
        lb[row_idx] = p_end[d];
        ub[row_idx] = p_end[d];
    }

    LinearConstraint {
        a,
        lb,
        ub,
        n_rows,
        n_vars,
    }
}

pub fn build_time_monotonicity(np1: usize, dim: usize, min_dt: f64) -> LinearConstraint {
    let n_vars = np1 * dim;
    let n_rows = np1 - 1;
    let mut a = vec![0.0; n_rows * n_vars];
    let mut lb = vec![min_dt; n_rows];
    let ub = vec![f64::INFINITY; n_rows];
    let t_idx = dim - 1;

    for i in 0..n_rows {
        a[i * n_vars + i * dim + t_idx] = -1.0;
        a[i * n_vars + (i + 1) * dim + t_idx] = 1.0;
        lb[i] = min_dt;
    }

    LinearConstraint {
        a,
        lb,
        ub,
        n_rows,
        n_vars,
    }
}

pub fn build_box_constraints(
    p_init: &[f64],
    np1: usize,
    dim: usize,
    coord_lb: f64,
    coord_ub: f64,
    time_lb: f64,
    time_ub: f64,
) -> LinearConstraint {
    let n_vars = np1 * dim;
    let n_rows = n_vars;
    let mut a = vec![0.0; n_rows * n_vars];
    let mut lb = vec![0.0; n_rows];
    let mut ub = vec![0.0; n_rows];
    let t_idx = dim - 1;

    for i in 0..np1 {
        for d in 0..dim {
            let var_idx = i * dim + d;
            a[var_idx * n_vars + var_idx] = 1.0;

            let is_endpoint = i == 0 || i == np1 - 1;
            if is_endpoint {
                lb[var_idx] = p_init[var_idx];
                ub[var_idx] = p_init[var_idx];
            } else if d == t_idx {
                lb[var_idx] = time_lb;
                ub[var_idx] = time_ub;
            } else {
                lb[var_idx] = coord_lb;
                ub[var_idx] = coord_ub;
            }
        }
    }

    LinearConstraint {
        a,
        lb,
        ub,
        n_rows,
        n_vars,
    }
}
