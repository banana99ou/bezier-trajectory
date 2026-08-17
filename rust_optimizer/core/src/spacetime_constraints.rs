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
    pub iteration: u32,
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

/// How many spatial units equal one unit of time when measuring distance in the
/// lifted space.
///
/// The tube is a sausage of radius `r` measured PERPENDICULAR to its slanted
/// centreline, and "perpendicular" is only defined once the two axes are
/// commensurable. Pinned at 1.0 and declared as a modelling choice rather than
/// tuned, so no result depends on an unexplained constant.
///
/// Consequence, stated plainly: at this scale the constant-time cross-section is
/// not a circle of radius `r` but an ellipse stretched by `sqrt(1 + speed^2)`
/// along the direction of travel. The forbidden region is therefore LARGER than
/// the true obstacle — conservative, never permissive. Measured on the shipped
/// scenarios: 1.32x on `original`, 1.29x on `diverse`, 1.04x on `wall`.
pub const SPACETIME_AXIS_SCALE: f64 = 1.0;

/// Geometry of the space-time tube relative to one query point.
///
/// The obstacle sweeps a straight but SLANTED centreline in (x, y, t): from where
/// it sits at `eff_t0` to where it sits at `eff_t1`. Wrapping that segment in a
/// capsule of radius `r` gives a convex set, so a plane touching it anywhere has
/// the whole tube on one side — which is what licenses one plane per curve
/// segment. Clamping to the segment ends gives rounded caps, and that is what
/// makes a time-limited obstacle a finite-height tube the curve can wait out.
///
/// The outward direction has a nonzero TIME component precisely because the
/// centreline is slanted. The pre-2026-08-17 builder used an unslanted axis and
/// then deleted that component, telling the solver that arriving earlier or later
/// could not change clearance.
///
/// Everything is in original coordinates: `SPACETIME_AXIS_SCALE` is 1.0, so the
/// scaled and unscaled frames coincide and no transform is carried around.
struct TubeGeometry {
    /// Nearest point on the centreline to the query.
    closest: Vec<f64>,
    /// Unit outward direction, from `closest` toward the query.
    normal: Vec<f64>,
    /// Distance from the centreline to the query.
    u_norm: f64,
    /// Centreline direction (end minus start), not normalized.
    axis: Vec<f64>,
    /// Squared length of `axis`.
    axis_sq: f64,
    /// True when the nearest point is a cap end rather than the tube body. The
    /// nearest point then does NOT slide as the query moves, which changes the
    /// rotation term.
    clamped: bool,
    radius: f64,
}

fn tube_geometry(
    query: &[f64],
    obstacles: &SpacetimeObstacleData<'_>,
    obs_idx: usize,
    time_start: f64,
    time_end: f64,
) -> Option<TubeGeometry> {
    let spatial_dim = obstacles.spatial_dim;
    let dim = spatial_dim + 1;
    let s = SPACETIME_AXIS_SCALE;

    let eff_t0 = obstacles.t_start[obs_idx].max(time_start);
    let eff_t1 = obstacles.t_end[obs_idx].min(time_end);
    if eff_t1 < eff_t0 {
        return None;
    }

    let mut a_end = vec![0.0; dim];
    let mut b_end = vec![0.0; dim];
    for d in 0..spatial_dim {
        let base = obs_idx * spatial_dim + d;
        a_end[d] = obstacles.pos0[base] + obstacles.vel[base] * eff_t0;
        b_end[d] = obstacles.pos0[base] + obstacles.vel[base] * eff_t1;
    }
    a_end[dim - 1] = s * eff_t0;
    b_end[dim - 1] = s * eff_t1;

    let axis: Vec<f64> = (0..dim).map(|d| b_end[d] - a_end[d]).collect();
    let axis_sq = dot(&axis, &axis);
    let rel: Vec<f64> = (0..dim).map(|d| query[d] - a_end[d]).collect();
    let tau_raw = if axis_sq > 1e-12 {
        dot(&rel, &axis) / axis_sq
    } else {
        0.0
    };
    let tau = tau_raw.clamp(0.0, 1.0);
    let clamped = axis_sq <= 1e-12 || tau_raw <= 0.0 || tau_raw >= 1.0;

    let closest: Vec<f64> = (0..dim).map(|d| a_end[d] + tau * axis[d]).collect();
    let u: Vec<f64> = (0..dim).map(|d| query[d] - closest[d]).collect();
    let u_norm = dot(&u, &u).sqrt();
    if u_norm <= 1e-10 {
        return None; // query sits on the centreline; direction undefined
    }

    Some(TubeGeometry {
        normal: u.iter().map(|v| v / u_norm).collect(),
        closest,
        u_norm,
        axis,
        axis_sq,
        clamped,
        radius: obstacles.radii[obs_idx],
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

/// Segment control points and centroid weights for one De Casteljau segment.
///
/// `w[j]` is how much global control point j contributes to this segment's
/// centroid. It is what makes the centroid — and therefore the aiming direction —
/// a function of the optimization variables.
fn segment_points_and_weights(
    a_seg: &[f64],
    p: &[f64],
    np1: usize,
    dim: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
    let mut w = vec![0.0; np1];
    for i in 0..np1 {
        for j in 0..np1 {
            w[j] += a_seg[i * np1 + j];
        }
    }
    for value in w.iter_mut() {
        *value /= np1 as f64;
    }
    let mut centroid = vec![0.0; dim];
    for i in 0..np1 {
        for d in 0..dim {
            centroid[d] += q[i * dim + d];
        }
    }
    for value in centroid.iter_mut() {
        *value /= np1 as f64;
    }
    (q, w, centroid)
}

/// EXACT obstacle rows: one half-space per (segment, obstacle), normal frozen at
/// the reference, applied to every control point of that segment.
///
/// These are the rows the CERTIFICATE is evaluated with. Satisfying them proves
/// the curve segment is outside the tube, because the tube is convex and a
/// Bezier segment lies inside the convex hull of its control points — so one
/// half-space checked at finitely many points bounds the whole segment.
///
/// Per-control-point planes would prove only that each point individually is
/// outside its own plane, which is exactly as strong as sampling the curve, and
/// opposing normals would let the hull wrap around the tube. See
/// `doc/notes/_shared/c3_safe_corridor_refs.md`.
pub fn build_spacetime_koz_constraints(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    // Parameter of the removed ellipsoid-cap geometry. The tube's caps are round
    // by construction now, so there is nothing to tune. Retained so existing
    // callers keep compiling; it has no effect.
    _cap_bulge_ratio: f64,
) -> Option<KozConstraintBundle> {
    build_koz_rows(a_list, p, np1, dim, obstacles, false)
}

/// SELF-CONSISTENT obstacle rows: the same half-spaces, plus the term that
/// accounts for the plane ROTATING as the solver moves the control points.
///
/// The exact rows freeze the aiming direction at the reference. They are sound —
/// satisfying them certifies the curve, whoever aimed them — but they describe a
/// plane that is no longer the one the centroid rule picks once the solver has
/// moved. A step optimized against them lands flush on a plane that then pivots
/// out from under it, and the next iteration grades it against a plane it never
/// saw. Measured before this term existed: 10 to 13 of every ~17 steps rejected,
/// trust region collapsing on every scenario, nothing converging.
///
/// Differentiating the clearance `g_k = n·(q_k − m) − r` through the centroid's
/// effect on the aiming direction gives
///
/// ```text
/// grad g_k[j] = A[k][j]·n + (w_j / |u|) · corr_k
/// ```
///
/// where `u` is the offset from the axis to the centroid and `corr_k` is the part
/// of `d_k = q_k − m` that is sideways:
///
/// ```text
/// cap end (nearest point fixed):  corr_k = d_k − (n·d_k) n
/// tube body (nearest point slides): corr_k = d_k − (n·d_k) n − (e·d_k/|e|²) e
/// ```
///
/// The second case is where this differs from the orbital-docking version, whose
/// obstacle centre is a fixed POINT. Ours is a LINE: when the centroid slides
/// along the tube, so does the nearest point on the axis, and the component of
/// `d_k` along the axis contributes nothing to the rotation.
///
/// NOTE this row is a first-order model and is NOT a conservative restriction: a
/// point satisfying it may have `g_k < 0`. Soundness of the reported guarantee is
/// unaffected, because the certificate is always evaluated with the exact rows
/// from `build_spacetime_koz_constraints`, never with these.
pub fn build_spacetime_koz_constraints_linearized(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
) -> Option<KozConstraintBundle> {
    build_koz_rows(a_list, p, np1, dim, obstacles, true)
}

fn build_koz_rows(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    with_rotation: bool,
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
        let (q, w, centroid) = segment_points_and_weights(a_seg, p, np1, dim);

        for obs_idx in 0..obstacles.n_obs {
            let Some(geom) = tube_geometry(&centroid, obstacles, obs_idx, plan_t0, plan_t1) else {
                continue;
            };
            let n = &geom.normal;
            let support: Vec<f64> = (0..dim)
                .map(|d| geom.closest[d] + geom.radius * n[d])
                .collect();

            for k in 0..np1 {
                let d_k: Vec<f64> = (0..dim).map(|d| q[k * dim + d] - geom.closest[d]).collect();
                // Exact clearance of this control point against the tube.
                let g_k = dot(n, &d_k) - geom.radius;

                let mut row = vec![0.0; n_vars];
                if with_rotation {
                    // Sideways part of d_k: strip the component along the normal,
                    // and — only on the tube body — the component along the axis,
                    // because the nearest point slides that way with the centroid.
                    let s_k = dot(n, &d_k);
                    let along_axis = if geom.clamped || geom.axis_sq <= 1e-12 {
                        0.0
                    } else {
                        dot(&geom.axis, &d_k) / geom.axis_sq
                    };
                    let corr: Vec<f64> = (0..dim)
                        .map(|d| d_k[d] - s_k * n[d] - along_axis * geom.axis[d])
                        .collect();
                    let scale = 1.0 / geom.u_norm;
                    for j in 0..np1 {
                        let a_kj = a_seg[k * np1 + j];
                        for d in 0..dim {
                            row[j * dim + d] += a_kj * n[d] + w[j] * scale * corr[d];
                        }
                    }
                } else {
                    for j in 0..np1 {
                        let a_kj = a_seg[k * np1 + j];
                        for d in 0..dim {
                            row[j * dim + d] += a_kj * n[d];
                        }
                    }
                }

                // Bound chosen so the constraint reproduces g_k(p) >= 0 exactly at
                // x = p, for both row types.
                let grad_dot_p = (0..n_vars).map(|idx| row[idx] * p[idx]).sum::<f64>();
                let lb = grad_dot_p - g_k;

                row_meta.push(KozRowData {
                    segment_idx: seg_idx,
                    cp_idx: k,
                    obstacle_idx: obs_idx,
                    iteration: 0,
                    normal: n.clone(),
                    support_point: support.clone(),
                    closest_center: geom.closest.clone(),
                    lower_bound: lb,
                    lhs: grad_dot_p,
                    margin: g_k,
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
