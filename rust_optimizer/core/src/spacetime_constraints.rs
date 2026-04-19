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

pub const DEFAULT_CAP_BULGE_RATIO: f64 = 2.0;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

/// Closest-point / supporting-half-space for a moving-obstacle capsule.
///
/// The capsule is modeled as the union of two pieces:
///   - **body**: swept circular cross-section of radius `r` along the
///     path `pos0 + vel·t` for `t ∈ [eff_t0, eff_t1]`. Cross-section at
///     every interior `t` is a circle of radius `r` in xy, independent
///     of any slider.
///   - **caps**: axis-aligned ellipsoid halves at each endpoint with xy
///     semi-axes `r` and `t` semi-axis `k·r`, where `k = cap_bulge_ratio`.
///     The bottom cap lives at `t ≤ eff_t0`, the top at `t ≥ eff_t1`.
///
/// Returns (outward normal, support point on surface, closest axis point,
/// plane offset `lb = n·support`) — the tangent half-space is `n·p ≥ lb`.
fn capsule_surface_support(
    query_orig: &[f64],
    obstacles: &SpacetimeObstacleData<'_>,
    obs_idx: usize,
    time_start: f64,
    time_end: f64,
    cap_bulge_ratio: f64,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
    let spatial_dim = obstacles.spatial_dim;
    let dim = spatial_dim + 1;
    let eff_t0 = obstacles.t_start[obs_idx].max(time_start);
    let eff_t1 = obstacles.t_end[obs_idx].min(time_end);
    if eff_t1 < eff_t0 {
        return None;
    }

    let radius = obstacles.radii[obs_idx];
    let qt = query_orig[dim - 1];

    let pos_at = |t: f64| -> Vec<f64> {
        (0..spatial_dim)
            .map(|d| {
                obstacles.pos0[obs_idx * spatial_dim + d]
                    + obstacles.vel[obs_idx * spatial_dim + d] * t
            })
            .collect()
    };

    // Case 1: cylinder body.
    // Closest surface point sits at the same t with xy direction toward query.
    if qt >= eff_t0 && qt <= eff_t1 {
        let pos = pos_at(qt);
        let diff_xy: Vec<f64> = (0..spatial_dim).map(|d| query_orig[d] - pos[d]).collect();
        let xy_dist_sq: f64 = diff_xy.iter().map(|v| v * v).sum();
        if xy_dist_sq <= 1e-20 {
            return None; // query on the axis, degenerate
        }
        let xy_dist = xy_dist_sq.sqrt();
        let mut n = vec![0.0; dim];
        let mut support = vec![0.0; dim];
        for d in 0..spatial_dim {
            n[d] = diff_xy[d] / xy_dist;
            support[d] = pos[d] + radius * n[d];
        }
        // Normal's t-component is 0 for the cylinder body.
        support[dim - 1] = qt;
        let mut closest = vec![0.0; dim];
        for d in 0..spatial_dim {
            closest[d] = pos[d];
        }
        closest[dim - 1] = qt;
        let lb = dot(&n, &support);
        return Some((n, support, closest, lb));
    }

    // Case 2/3: cap. Pick the nearer endpoint.
    let cap_t = if qt > eff_t1 { eff_t1 } else { eff_t0 };
    let cap_center_xy = pos_at(cap_t);

    // Scale the query's t offset so the ellipsoid becomes a unit sphere of
    // radius r: (x, y, (q_t - cap_t) / k).
    let k = cap_bulge_ratio;
    let mut u_scaled = vec![0.0; dim];
    for d in 0..spatial_dim {
        u_scaled[d] = query_orig[d] - cap_center_xy[d];
    }
    u_scaled[dim - 1] = (qt - cap_t) / k;
    let u_norm_sq: f64 = u_scaled.iter().map(|v| v * v).sum();
    if u_norm_sq <= 1e-20 {
        return None; // query sitting exactly at the cap center
    }
    let u_norm = u_norm_sq.sqrt();

    // Closest point on the ellipsoid: scaled direction scaled by r, then
    // un-scale (multiply t component by k).
    let mut support = vec![0.0; dim];
    for d in 0..spatial_dim {
        support[d] = cap_center_xy[d] + radius * u_scaled[d] / u_norm;
    }
    support[dim - 1] = cap_t + radius * k * u_scaled[dim - 1] / u_norm;

    // Outward normal = grad f / |grad f| where f(p) = (p_xy - c)² + ((p_t - cap_t)/k)² - r².
    // grad f ∝ ((p_xy - c), (p_t - cap_t)/k²).
    let mut n = vec![0.0; dim];
    for d in 0..spatial_dim {
        n[d] = support[d] - cap_center_xy[d];
    }
    n[dim - 1] = (support[dim - 1] - cap_t) / (k * k);
    let n_norm_sq: f64 = n.iter().map(|v| v * v).sum();
    if n_norm_sq <= 1e-20 {
        return None;
    }
    let n_norm = n_norm_sq.sqrt();
    for val in n.iter_mut() {
        *val /= n_norm;
    }

    // "Closest axis point" for diagnostic viz is the cap center itself.
    let mut closest = vec![0.0; dim];
    for d in 0..spatial_dim {
        closest[d] = cap_center_xy[d];
    }
    closest[dim - 1] = cap_t;

    let lb = dot(&n, &support);
    Some((n, support, closest, lb))
}

pub fn build_spacetime_koz_constraints(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    cap_bulge_ratio: f64,
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
                let Some((n_orig, support_orig, closest_orig, lb)) = capsule_surface_support(
                    &query_point,
                    obstacles,
                    obs_idx,
                    plan_t0,
                    plan_t1,
                    cap_bulge_ratio,
                ) else {
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
