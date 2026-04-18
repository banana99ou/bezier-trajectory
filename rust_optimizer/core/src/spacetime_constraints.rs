use crate::constraints::LinearConstraint;

pub struct SpacetimeObstacleData<'a> {
    pub pos0: &'a [f64],    // (n_obs, spatial_dim) row-major
    pub vel: &'a [f64],     // (n_obs, spatial_dim) row-major
    pub radii: &'a [f64],   // (n_obs,)
    pub t_start: &'a [f64], // (n_obs,)
    pub t_end: &'a [f64],   // (n_obs,)
    pub n_obs: usize,
    pub spatial_dim: usize,
}

/// Per-row metadata for a single KOZ constraint.
pub struct KozRowData {
    pub segment_idx: usize,
    pub cp_idx: usize,
    pub obstacle_idx: usize,
    pub normal: Vec<f64>,
    pub support_point: Vec<f64>,
    pub closest_center: Vec<f64>,
    pub lower_bound: f64,
    pub lhs: f64,
    pub margin: f64,
}

/// KOZ constraint matrix plus per-row metadata.
pub struct KozConstraintBundle {
    pub constraint: LinearConstraint,
    pub rows: Vec<KozRowData>,
}

pub const DEFAULT_CAPSULE_TIME_SCALE: f64 = 0.5;

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
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
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

    Some((n_orig, support_orig, closest_orig, lb))
}

pub fn build_spacetime_koz_constraints(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    capsule_time_scale: f64,
) -> Option<KozConstraintBundle> {
    let n_vars = np1 * dim;
    let mut constraint_rows: Vec<Vec<f64>> = Vec::new();
    let mut lbs: Vec<f64> = Vec::new();
    let mut row_meta: Vec<KozRowData> = Vec::new();
    let plan_t0 = p[dim - 1];
    let plan_t1 = p[(np1 - 1) * dim + (dim - 1)];

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

        for k in 0..np1 {
            let query_point: Vec<f64> = (0..dim).map(|d| q[k * dim + d]).collect();
            for obs_idx in 0..obstacles.n_obs {
                let Some((n_orig, support_orig, closest_orig, lb)) =
                    scaled_spacetime_capsule_support(
                        &query_point,
                        obstacles,
                        obs_idx,
                        plan_t0,
                        plan_t1,
                        capsule_time_scale,
                    )
                else {
                    continue;
                };

                let mut row = vec![0.0; n_vars];
                for j in 0..np1 {
                    let coeff = a_seg[k * np1 + j];
                    for d in 0..dim {
                        row[j * dim + d] += coeff * n_orig[d];
                    }
                }
                let lhs = (0..n_vars).map(|idx| row[idx] * p[idx]).sum::<f64>();
                let margin = lhs - lb;

                row_meta.push(KozRowData {
                    segment_idx: seg_idx,
                    cp_idx: k,
                    obstacle_idx: obs_idx,
                    normal: n_orig,
                    support_point: support_orig,
                    closest_center: closest_orig,
                    lower_bound: lb,
                    lhs,
                    margin,
                });
                constraint_rows.push(row);
                lbs.push(lb);
            }
        }
    }

    if constraint_rows.is_empty() {
        return None;
    }

    let n_rows = constraint_rows.len();
    let mut a = vec![0.0; n_rows * n_vars];
    for (i, row) in constraint_rows.iter().enumerate() {
        a[i * n_vars..(i + 1) * n_vars].copy_from_slice(row);
    }

    Some(KozConstraintBundle {
        constraint: LinearConstraint {
            a,
            lb: lbs,
            ub: vec![f64::INFINITY; n_rows],
            n_rows,
            n_vars,
        },
        rows: row_meta,
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
    let lb = vec![min_dt; n_rows];
    let ub = vec![f64::INFINITY; n_rows];
    let t_idx = dim - 1;

    for i in 0..n_rows {
        a[i * n_vars + i * dim + t_idx] = -1.0;
        a[i * n_vars + (i + 1) * dim + t_idx] = 1.0;
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
