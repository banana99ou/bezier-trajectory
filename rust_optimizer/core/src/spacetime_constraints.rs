use crate::constraints::LinearConstraint;
use crate::spacetime_generator::Generator;
use crate::spacetime_obstacle::{clip_geometry, rotation_correction, ClipGeometry, ClipOutcome};

pub use crate::spacetime_obstacle::SpacetimeObstacleData;

/// Fixed observation stations the vehicle must keep line of sight to.
///
/// Purely spatial: a station is a point in space, present at every instant, so it
/// maps each instant into itself when the shadow is taken. An empty set is the
/// default and produces no shadow generators at all, so every scenario that
/// predates item B12 solves the identical problem it always did.
pub struct StationData<'a> {
    pub pos: &'a [f64], // (n_stations, spatial_dim) row-major
    pub n_stations: usize,
}

/// Per-row metadata for a single KOZ constraint.
pub struct KozRowData {
    pub segment_idx: usize,
    pub cp_idx: usize,
    pub obstacle_idx: usize,
    /// Which local APPROACH of the obstacle this row's wall was built against:
    /// the band is cut at every interior local maximum of `|gamma(s) - c|` and
    /// this indexes the sub-intervals in parameter order. **The grouping key is
    /// (segment, obstacle, component), not (segment, obstacle)** — one obstacle
    /// can approach one segment twice, and each approach gets its own plane;
    /// grouping without this index folds two different walls together. All
    /// control points of a segment still share one plane per approach; that is
    /// the convex-hull certificate and it is what makes the row say anything
    /// about the curve between the control points. (The name predates the
    /// 2026-08-26 change from connected components to approaches; the wire
    /// format keeps it.)
    pub component_idx: usize,
    /// Which station's shadow this row constrains, or `None` for the obstacle's
    /// own keep-out zone.
    ///
    /// **There is one keep-out zone and one row set.** The field is not a second
    /// code path — it says which generator produced the wall, so the certificate
    /// can be reported as the two numbers the figure gate and the frontend layers
    /// have always read. A run with no station has `None` on every row.
    pub station_idx: Option<usize>,
    pub iteration: u32,
    pub normal: Vec<f64>,
    /// The point of the component where `n . z` attains its maximum — the point
    /// the plane rests on. Was `y* + r_m n`, which is the nearest point pushed
    /// out; the two coincide only for a straight tube, and the difference is
    /// exactly the conservatism a curved one costs.
    pub support_point: Vec<f64>,
    /// The point of the component nearest the segment centroid. The centroid
    /// itself when the centroid is inside the keep-out zone.
    pub closest_center: Vec<f64>,
    pub lower_bound: f64,
    pub lhs: f64,
    pub margin: f64,
    /// Clip radius used for this row, and whether PAPER_1 statement (7) held.
    pub rho: f64,
    pub sound: bool,
}

/// KOZ constraint matrix plus per-row metadata, plus what could NOT be built.
pub struct KozConstraintBundle {
    pub constraint: LinearConstraint,
    pub rows: Vec<KozRowData>,
    /// Approaches in reach whose supporting half-space could not be built.
    /// **Not the same as "no constraint needed."** A row set that is silent about
    /// such an approach sums to zero violation, which turns "cannot be certified"
    /// into "certified 0.0" — a check that cannot fail.
    ///
    /// Counted separately for the obstacle's own zone and for the shadows, so the
    /// two certificates keep the meaning they have always had: a shadow plane
    /// that could not be built must not poison the keep-out certificate, and vice
    /// versa.
    pub dropped_planes: usize,
    pub dropped_shadow_planes: usize,
    /// Walls where statement (7) failed — counted per emitted plane, per
    /// generator, so a clip cut into two approaches counts twice and a shadow
    /// generator counts on its own. The clip radius was smaller than the segment
    /// radius plus the trust box radius, so the rows certify against the clipped
    /// piece and the next iterate can leave it. **This is the hole, counted.**
    /// Zero at every iteration means the run was sound by construction whether
    /// or not `sound_clip` was requested.
    pub unsound_clips: usize,
}

/// How many spatial units equal one unit of time when measuring distance in the
/// lifted space.
///
/// **This is structural, not a tuning knob.** The construction is built from
/// balls, distances and orthogonal projections in the lifted space; none of them
/// is defined until the axes are commensurable. Pinned at 1.0, which is the
/// formal content of PAPER_1's "no axis is privileged": remove it and the clip
/// ball, the projection `y*` and the normal `n` all stop meaning anything.
///
/// Consequence, stated plainly: at this scale the constant-time cross-section of
/// a moving obstacle is not a disc of radius `r` but an ellipse stretched by
/// `sqrt(1 + speed^2)` along the direction of travel. The forbidden region is
/// therefore LARGER than the true obstacle — conservative, never permissive.
pub const SPACETIME_AXIS_SCALE: f64 = 1.0;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

/// Segment control points, centroid weights, centroid, and segment radius.
///
/// `w[j]` is how much global control point j contributes to this segment's
/// centroid. It is what makes the centroid — and therefore the aiming direction —
/// a function of the optimization variables.
///
/// `E = max_i ||Q_i - c||` is the segment radius of PAPER_1 statements (6) and
/// (7). Together with the trust radius it decides whether the clip covers
/// everywhere the next iterate can reach.
fn segment_points_and_weights(
    a_seg: &[f64],
    p: &[f64],
    np1: usize,
    dim: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
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
    let mut seg_radius: f64 = 0.0;
    for i in 0..np1 {
        let mut acc = 0.0;
        for d in 0..dim {
            let delta = q[i * dim + d] - centroid[d];
            acc += delta * delta;
        }
        seg_radius = seg_radius.max(acc.sqrt());
    }
    (q, w, centroid, seg_radius)
}

/// EXACT obstacle rows: one half-space per (segment, generator, approach) — a
/// support of that approach's piece of the clipped keep-out volume, aimed from
/// the piece's nearest point back to the segment centroid — applied to every
/// control point of that segment. The generator is the obstacle's own centreline
/// or its center surface through a station; the rows are built the same way.
///
/// These are the rows the CERTIFICATE is evaluated with. Satisfying them proves
/// the curve segment is outside every approach's piece: a Bezier segment lies in
/// the convex hull of its control points, the rows are linear, and the piece
/// lies entirely on the other side of its plane because the offset is a rigorous
/// ceiling on the piece's support. No hull of the piece is ever taken.
///
/// **What they do NOT prove.** They certify against `K_m ∩ B(c, rho)`, not
/// against `K_m`. The gap closes only when statement (7) holds, which is reported
/// per bundle as `unsound_clips`. Where it fails, the trajectory-wide clearance
/// scan is the only statement made about the obstacle at all.
///
/// Per-control-point planes would destroy statement (5): the step from `Q_i` to a
/// curve point uses the SAME normal for every `i`, and with differing normals the
/// hull can wrap around the tube.
pub fn build_spacetime_koz_constraints(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    stations: &StationData<'_>,
    trust_radius: f64,
    sound_clip: bool,
    with_occlusion: bool,
) -> Option<KozConstraintBundle> {
    build_koz_rows(
        a_list,
        p,
        np1,
        dim,
        obstacles,
        stations,
        trust_radius,
        sound_clip,
        with_occlusion,
        false,
    )
}

/// The generators one obstacle contributes.
///
/// **One keep-out zone, two readings of it.** The obstacle's own zone always
/// exists, with the stretch pinned at one — that generator is the plain tube and
/// its rows are what a scenario without a station has always produced. When a
/// station is present and occlusion is asked for, the same centreline is read a
/// second time through that station, and its generator sweeps `u >= 1`: body at
/// `u = 1`, shadow beyond.
///
/// The two overlap at `u = 1`, and that redundancy is deliberate. It is sound —
/// two valid walls on the same material — and it is what keeps the certificate
/// meaningful as two numbers: a merged set with no body-only rows would report
/// "keep-out certified 0.0" for a run that has no keep-out rows at all, which is
/// exactly the trap `dropped_planes` exists to prevent. Dropping the station
/// generators is also how the baseline run is built: not a second code path, just
/// a shorter list.
fn generators_for<'a>(
    obstacles: &'a SpacetimeObstacleData<'a>,
    obs_idx: usize,
    stations: &'a StationData<'a>,
    with_occlusion: bool,
) -> Vec<Generator<'a>> {
    let base = Generator {
        ctrl: obstacles.ctrl_of(obs_idx),
        n_ctrl: obstacles.n_ctrl,
        dim: obstacles.dim(),
        r_m: obstacles.radii[obs_idx],
        station: None,
        obstacle_idx: obs_idx,
        station_idx: None,
    };
    let mut out = vec![base];
    if with_occlusion {
        let sd = obstacles.spatial_dim;
        for st in 0..stations.n_stations {
            out.push(Generator {
                ctrl: obstacles.ctrl_of(obs_idx),
                n_ctrl: obstacles.n_ctrl,
                dim: obstacles.dim(),
                r_m: obstacles.radii[obs_idx],
                station: Some(&stations.pos[st * sd..(st + 1) * sd]),
                obstacle_idx: obs_idx,
                station_idx: Some(st),
            });
        }
    }
    out
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
/// The correction is `(I - P_F)(I - n n^T) d_k / ||c - y*||`, derived in
/// `spacetime_obstacle::rotation_correction`. It is the SAME formula the straight
/// capsule used, generalized: its two branches are the `P_F = 0` and
/// `P_F = e e^T / |e|^2` cases.
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
    stations: &StationData<'_>,
    trust_radius: f64,
    sound_clip: bool,
    with_occlusion: bool,
) -> Option<KozConstraintBundle> {
    build_koz_rows(
        a_list,
        p,
        np1,
        dim,
        obstacles,
        stations,
        trust_radius,
        sound_clip,
        with_occlusion,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_koz_rows(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    stations: &StationData<'_>,
    trust_radius: f64,
    sound_clip: bool,
    with_occlusion: bool,
    with_rotation: bool,
) -> Option<KozConstraintBundle> {
    let n_vars = np1 * dim;
    let mut constraint_rows: Vec<Vec<f64>> = Vec::new();
    let mut lbs: Vec<f64> = Vec::new();
    let mut row_meta: Vec<KozRowData> = Vec::new();
    let mut dropped_planes = 0usize;
    let mut dropped_shadow_planes = 0usize;
    let mut unsound_clips = 0usize;

    if obstacles.n_obs == 0 {
        return None;
    }

    for (seg_idx, a_seg) in a_list.iter().enumerate() {
        let (q, w, centroid, seg_radius) = segment_points_and_weights(a_seg, p, np1, dim);

        for obs_idx in 0..obstacles.n_obs {
          for gen in generators_for(obstacles, obs_idx, stations, with_occlusion) {
            let components =
                match clip_geometry(&centroid, seg_radius, trust_radius, &gen, sound_clip) {
                    // Nothing in the trust region can touch this keep-out zone.
                    // No row is needed and silence is correct.
                    ClipOutcome::OutOfReach => continue,
                    ClipOutcome::Components(c) => c,
                };
            // A row is needed and cannot be built. Count it: the guarantee has a
            // hole here, it is not satisfied here. Which counter it lands in is
            // which generator failed, so one kind's hole never silences the
            // other kind's certificate.
            if gen.station_idx.is_some() {
                dropped_shadow_planes += components.dropped;
            } else {
                dropped_planes += components.dropped;
            }

            // ONE WALL PER LOCAL APPROACH of the generator to the segment,
            // not one per (segment, obstacle). Each dip of the distance profile
            // is a separate approach and gets its own plane; the gap between two
            // approaches — a corridor, or the mouth of a bend that wraps the
            // centroid — stays usable instead of being swallowed by one fused
            // wall. The row count therefore varies with the geometry, which the
            // rest of this builder already tolerates — rows are pushed, not
            // indexed, and an out-of-reach pair has always emitted none.
            for (comp_idx, geom) in components.planes.iter().enumerate() {
                let geom: &ClipGeometry = geom;
                if !geom.sound {
                    unsound_clips += 1;
                }

                let n = &geom.normal;
                // The point of the component where the support is attained. Not
                // `y* + r_m n`: that is the nearest point pushed out, which coincides
                // with the support point only for a straight tube.
                let support: Vec<f64> = (0..dim)
                    .map(|d| centroid[d] + (geom.offset - dot(n, &centroid)) * n[d])
                    .collect();

                for k in 0..np1 {
                    // The row IS the wall: `n . Q_k >= b` with `b` the support of the
                    // component. The retired construction wrote this as
                    // `n . (Q_k - y*) - r_m` because its offset was `n . y* + r_m` by
                    // definition; that identity does not survive clipping the volume.
                    let g_k = dot(n, &q[k * dim..(k + 1) * dim]) - geom.offset;
                    // Anchor for the rotation term. `db/dn` is the SUPPORT point, so
                    // `d(n . Q_k - b)/dn` is `Q_k - support` — the nearest point `y*`
                    // anchored the old row because the old offset was written through
                    // it, and it is the wrong point for this one.
                    let d_k: Vec<f64> = (0..dim).map(|d| q[k * dim + d] - support[d]).collect();

                    let mut row = vec![0.0; n_vars];
                    if with_rotation {
                        let corr = rotation_correction(&geom, &d_k);
                        for j in 0..np1 {
                            let a_kj = a_seg[k * np1 + j];
                            for d in 0..dim {
                                row[j * dim + d] += a_kj * n[d] + w[j] * corr[d];
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
                        component_idx: comp_idx,
                        station_idx: gen.station_idx,
                        iteration: 0,
                        normal: n.clone(),
                        support_point: support.clone(),
                        closest_center: geom.y_star.clone(),
                        lower_bound: lb,
                        lhs: grad_dot_p,
                        margin: g_k,
                        rho: geom.rho,
                        sound: geom.sound,
                    });
                    constraint_rows.push(row);
                    lbs.push(lb);
                }
            }
          }
        }
    }

    if constraint_rows.is_empty() {
        if dropped_planes == 0 && dropped_shadow_planes == 0 && unsound_clips == 0 {
            return None;
        }
        // Nothing to hand the QP, but the caller still has to hear about the
        // holes. An empty matrix carries them rather than swallowing them.
        return Some(KozConstraintBundle {
            constraint: LinearConstraint {
                a: Vec::new(),
                lb: Vec::new(),
                ub: Vec::new(),
                n_rows: 0,
                n_vars,
            },
            rows: Vec::new(),
            dropped_planes,
            dropped_shadow_planes,
            unsound_clips,
        });
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
        dropped_planes,
        dropped_shadow_planes,
        unsound_clips,
    })
}

/// Endpoint constraints.
///
/// `free_arrival_time` releases exactly one variable: the TIME coordinate of the
/// last control point (item B10). A Bezier passes through its last control
/// point, so that coordinate *is* the arrival time, and freeing it is what turns
/// arrival time into a decision variable. The spatial coordinates of both
/// endpoints, and every coordinate of the start point, stay pinned — the start
/// time in particular, because a free start time would let the plan translate in
/// time for free and the linear time penalty would then measure duration only by
/// accident.
///
/// The freed row is written as `t_last >= p_start[t]` — the START time, with no
/// `min_dt` term in it. (A previous version of this comment described a
/// `t_start + (np1-1)*min_dt` row; no such row is emitted here, and that bound
/// arises from the monotonicity block instead.) It is implied by monotonicity
/// anyway; the real upper bound comes from the box constraint's `time_ub`.
/// Keeping a row here rather than deleting one keeps the row count fixed, so
/// nothing downstream has to branch on the flag.
pub fn build_boundary_constraints(
    np1: usize,
    dim: usize,
    p_start: &[f64],
    p_end: &[f64],
    free_arrival_time: bool,
) -> LinearConstraint {
    let n_vars = np1 * dim;
    let n_rows = 2 * dim;
    let mut a = vec![0.0; n_rows * n_vars];
    let mut lb = vec![0.0; n_rows];
    let mut ub = vec![0.0; n_rows];
    let t_idx = dim - 1;

    for d in 0..dim {
        a[d * n_vars + d] = 1.0;
        lb[d] = p_start[d];
        ub[d] = p_start[d];

        let row_idx = dim + d;
        a[row_idx * n_vars + (np1 - 1) * dim + d] = 1.0;
        if free_arrival_time && d == t_idx {
            lb[row_idx] = p_start[t_idx];
            ub[row_idx] = f64::INFINITY;
        } else {
            lb[row_idx] = p_end[d];
            ub[row_idx] = p_end[d];
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

/// The slant-limit speed cap, one second-order cone per control-polygon leg
/// (item B9, formulation decision 4).
///
/// For every consecutive pair of control points,
///
/// ```text
/// || P[i+1, spatial] - P[i, spatial] ||  <=  v_max * ( P[i+1, t] - P[i, t] )
/// ```
///
/// **Why this bounds the physical speed of the whole curve.** Physical velocity
/// is the spatial parameter-derivative over the time parameter-derivative, so
/// the bound with the denominator cleared reads "spatial gap per parameter, in
/// norm, at most v_max times the time gap per parameter". Both sides are then
/// Beziers of the same degree, and a Bezier is a convex combination of its own
/// control points (Bernstein weights are non-negative and sum to one). By the
/// triangle inequality, if every control point of the left side obeys the bound
/// against the matching control point of the right side, the whole curve does.
/// The degree factor `N` is common to both sides and cancels, which is why this
/// is written on raw gaps.
///
/// **It is sufficient, not necessary** — conservative in the safe direction, the
/// same character as the hull certificate.
///
/// **It depends on time monotonicity.** Clearing the denominator is legal only
/// because the time gap is strictly positive. `build_time_monotonicity` is
/// therefore physics-load-bearing here, not merely a sanity constraint. (The
/// cone itself also forces `Δt >= 0`, so the two agree rather than conflict.)
///
/// Returns an empty vector when `v_max` is not a positive finite number, which
/// is how "no speed cap" is expressed — existing scenarios keep their behaviour.
pub fn build_speed_cap_socs(np1: usize, dim: usize, v_max: f64) -> Vec<crate::optimizer::SocBlock> {
    if !(v_max > 0.0) || !v_max.is_finite() {
        return Vec::new();
    }
    let n_vars = np1 * dim;
    let spatial_dim = dim - 1;
    let t_idx = dim - 1;
    let cone_dim = spatial_dim + 1;

    let mut blocks = Vec::with_capacity(np1 - 1);
    for i in 0..np1 - 1 {
        // Clarabel wants `b - A x` in the cone, and b is zero here, so every
        // entry of A is the NEGATIVE of the quantity being bounded.
        let mut a = vec![0.0; cone_dim * n_vars];
        // Row 0: v_max * (t_{i+1} - t_i)
        a[i * dim + t_idx] = v_max;
        a[(i + 1) * dim + t_idx] = -v_max;
        // Rows 1..: the spatial gap, componentwise.
        for d in 0..spatial_dim {
            let r = d + 1;
            a[r * n_vars + i * dim + d] = 1.0;
            a[r * n_vars + (i + 1) * dim + d] = -1.0;
        }
        blocks.push(crate::optimizer::SocBlock {
            a,
            b: vec![0.0; cone_dim],
            cone_dim,
        });
    }
    blocks
}

/// Total slant-limit violation at `p`, summed over legs. Zero when the cap is
/// off or satisfied.
pub fn speed_cap_violation(socs: &[crate::optimizer::SocBlock], p: &[f64], n_vars: usize) -> f64 {
    socs.iter().map(|b| b.violation(p, n_vars)).sum()
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

/// Per-variable box. Endpoints are pinned to their initial values.
///
/// `free_arrival_time` exempts one variable — the last control point's time —
/// which is then bounded by `[time_lb, time_ub]` like any interior time
/// coordinate. Without this the box would re-pin what
/// `build_boundary_constraints` just released, and freeing the arrival time
/// would silently do nothing (item B10).
/// `coord_lb` / `coord_ub` carry one bound per SPATIAL coordinate (length
/// `dim - 1`), so an altitude band costs nothing but different numbers in one
/// slot. These rows are physics-class: appended before the keep-out block,
/// outside the elastic slack range, so the penalty can never buy its way
/// through a workspace wall. Endpoints stay pinned to `p_init` and are exempt
/// -- a bound that excludes an endpoint is refused Python-side, not here.
pub fn build_box_constraints(
    p_init: &[f64],
    np1: usize,
    dim: usize,
    coord_lb: &[f64],
    coord_ub: &[f64],
    time_lb: f64,
    time_ub: f64,
    free_arrival_time: bool,
) -> LinearConstraint {
    assert_eq!(coord_lb.len(), dim - 1, "one lower bound per spatial coordinate");
    assert_eq!(coord_ub.len(), dim - 1, "one upper bound per spatial coordinate");
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

            let is_endpoint = (i == 0 || i == np1 - 1)
                && !(free_arrival_time && i == np1 - 1 && d == t_idx);
            if is_endpoint {
                lb[var_idx] = p_init[var_idx];
                ub[var_idx] = p_init[var_idx];
            } else if d == t_idx {
                lb[var_idx] = time_lb;
                ub[var_idx] = time_ub;
            } else {
                lb[var_idx] = coord_lb[d];
                ub[var_idx] = coord_ub[d];
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
