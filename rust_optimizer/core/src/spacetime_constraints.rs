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

    if obstacles.n_obs == 0 {
        return None;
    }

    for a_seg in a_list {
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
        let t_seg = centroid[dim - 1];

        for obs_idx in 0..obstacles.n_obs {
            if t_seg < obstacles.t_start[obs_idx] || t_seg > obstacles.t_end[obs_idx] {
                continue;
            }

            let mut obs_pos = vec![0.0; obstacles.spatial_dim];
            for d in 0..obstacles.spatial_dim {
                let base = obs_idx * obstacles.spatial_dim + d;
                obs_pos[d] = obstacles.pos0[base] + obstacles.vel[base] * t_seg;
            }

            let mut n_hat = vec![0.0; obstacles.spatial_dim];
            let mut dist_sq = 0.0;
            for d in 0..obstacles.spatial_dim {
                n_hat[d] = centroid[d] - obs_pos[d];
                dist_sq += n_hat[d] * n_hat[d];
            }
            let dist = dist_sq.sqrt();
            if dist <= 1e-10 {
                continue;
            }
            for d in 0..obstacles.spatial_dim {
                n_hat[d] /= dist;
            }

            let mut lb = obstacles.radii[obs_idx];
            for d in 0..obstacles.spatial_dim {
                lb += n_hat[d] * obs_pos[d];
            }

            for k in 0..np1 {
                let mut row = vec![0.0; n_vars];
                for j in 0..np1 {
                    let coeff = a_seg[k * np1 + j];
                    for d in 0..obstacles.spatial_dim {
                        row[j * dim + d] += coeff * n_hat[d];
                    }
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
