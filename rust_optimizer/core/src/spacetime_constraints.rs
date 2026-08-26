use crate::constraints::LinearConstraint;
use crate::spacetime_obstacle::{
    band_over_times, clip_geometry, rotation_correction, spatial_containing_ball, ClipGeometry,
    ClipOutcome,
};

pub use crate::spacetime_obstacle::SpacetimeObstacleData;

/// Fixed observation stations the vehicle must keep line of sight to.
///
/// Purely spatial: a station is a point, not a body, and it does not move. An
/// empty set is the default and produces zero occlusion rows, so every scenario
/// that predates item B12 solves the identical problem it always did.
pub struct StationData<'a> {
    pub pos: &'a [f64], // (n_stations, spatial_dim) row-major
    pub n_stations: usize,
}

/// Per-row metadata for a single KOZ constraint.
pub struct KozRowData {
    pub segment_idx: usize,
    pub cp_idx: usize,
    pub obstacle_idx: usize,
    /// Which connected component of the clipped keep-out volume this row's wall
    /// was built against. **The grouping key is (segment, obstacle, component),
    /// not (segment, obstacle)** — one obstacle can present two separated lumps
    /// to one segment, and each gets its own plane. All control points of a
    /// segment still share one plane per component; that is the convex-hull
    /// certificate and it is what makes the row say anything about the curve
    /// between the control points.
    pub component_idx: usize,
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
    /// (segment, obstacle) pairs in reach whose supporting half-space could not
    /// be built because the centroid was already inside the convexified keep-out
    /// zone. **Not the same as "no constraint needed."** A row set that is silent
    /// about such a pair sums to zero violation, which turns "cannot be
    /// certified" into "certified 0.0" — the same trap the occlusion block
    /// already guards with `dropped_planes`.
    pub dropped_planes: usize,
    /// (segment, obstacle) pairs where statement (7) failed: the clip radius was
    /// smaller than the segment radius plus the trust box radius, so the rows
    /// certify against the clipped piece and the next iterate can leave it.
    /// **This is the hole, counted.** Zero at every iteration means the run was
    /// sound by construction whether or not `sound_clip` was requested.
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

/// EXACT obstacle rows: one half-space per (segment, obstacle), aimed by the
/// projection of the segment centroid onto the convexified clipped centreline,
/// applied to every control point of that segment.
///
/// These are the rows the CERTIFICATE is evaluated with. Satisfying them proves
/// the curve segment is outside the convexified clipped piece `H`, by PAPER_1
/// statement (5): a Bezier segment lies in the convex hull of its control points,
/// the rows are linear, and `H` lies entirely on the other side of the plane by
/// statement (4).
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
    trust_radius: f64,
    sound_clip: bool,
) -> Option<KozConstraintBundle> {
    build_koz_rows(a_list, p, np1, dim, obstacles, trust_radius, sound_clip, false)
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
    trust_radius: f64,
    sound_clip: bool,
) -> Option<KozConstraintBundle> {
    build_koz_rows(a_list, p, np1, dim, obstacles, trust_radius, sound_clip, true)
}

#[allow(clippy::too_many_arguments)]
fn build_koz_rows(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    trust_radius: f64,
    sound_clip: bool,
    with_rotation: bool,
) -> Option<KozConstraintBundle> {
    let n_vars = np1 * dim;
    let mut constraint_rows: Vec<Vec<f64>> = Vec::new();
    let mut lbs: Vec<f64> = Vec::new();
    let mut row_meta: Vec<KozRowData> = Vec::new();
    let mut dropped_planes = 0usize;
    let mut unsound_clips = 0usize;

    if obstacles.n_obs == 0 {
        return None;
    }

    for (seg_idx, a_seg) in a_list.iter().enumerate() {
        let (q, w, centroid, seg_radius) = segment_points_and_weights(a_seg, p, np1, dim);

        for obs_idx in 0..obstacles.n_obs {
            let components = match clip_geometry(
                &centroid,
                seg_radius,
                trust_radius,
                obstacles,
                obs_idx,
                sound_clip,
            ) {
                // Nothing in the trust region can touch this obstacle. No row is
                // needed and silence is correct.
                ClipOutcome::OutOfReach => continue,
                ClipOutcome::Components(c) => c,
            };
            // A row is needed and cannot be built. Count it: the guarantee has a
            // hole here, it is not satisfied here.
            dropped_planes += components.dropped;

            // ONE WALL PER CONNECTED COMPONENT of the clipped keep-out volume,
            // not one per (segment, obstacle). A centreline that leaves the clip
            // ball and re-enters puts two separate lumps of tube inside it, and
            // the gap between them is a corridor the trajectory is entitled to
            // use. The row count therefore varies with the geometry, which the
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

    if constraint_rows.is_empty() {
        if dropped_planes == 0 && unsound_clips == 0 {
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
        unsound_clips,
    })
}

// ===========================================================================
// Line-of-sight occlusion (item B12)
// ===========================================================================

/// Per-row metadata for a single occlusion constraint.
pub struct OcclusionRowData {
    pub segment_idx: usize,
    pub cp_idx: usize,
    pub obstacle_idx: usize,
    pub station_idx: usize,
    /// Unit outward normal. SPATIAL only — see `build_spacetime_occlusion_constraints`
    /// for why the time coefficient is zero here and must not be zero for a KOZ row.
    pub normal: Vec<f64>,
    /// Centre of the piece's convex outer approximation.
    pub body_center: Vec<f64>,
    /// Radius of that approximation, obstacle radius plus the inflation.
    pub body_radius: f64,
    /// Time range of the CLIPPED BAND the approximation was built over — the
    /// same band the keep-out half-space uses, read through its time coordinate.
    /// It is a consequence of the clip, not a window imposed on the clip.
    pub t_lo: f64,
    pub t_hi: f64,
    pub lower_bound: f64,
    pub margin: f64,
}

pub struct OcclusionConstraintBundle {
    pub constraint: LinearConstraint,
    pub rows: Vec<OcclusionRowData>,
}

/// What one occlusion build produced, INCLUDING what it could not produce.
///
/// The second field is the whole point. `shadow_plane` returns `None` when the
/// station sits inside the piece's inflated body: no half-space separates
/// anything then, so no row can be emitted. Emitting nothing is correct for the
/// QP — there is no valid convex constraint to hand it — but silently emitting
/// nothing to the CERTIFICATE turns "cannot be certified" into "certified 0.0",
/// which is a check that cannot fail. The count travels with the bundle so the
/// certificate can refuse rather than pass.
pub struct OcclusionBuildResult {
    /// `None` when no row at all was emitted — either nothing was in range, or
    /// every plane in range was dropped. `dropped_planes` distinguishes them.
    pub bundle: Option<OcclusionConstraintBundle>,
    /// (segment, station, obstacle) windows that were in range and whose
    /// supporting plane could NOT be built. Each one is a piece of the
    /// guarantee that is missing, not a piece that is satisfied.
    pub dropped_planes: usize,
}

/// Some unit vector orthogonal to `e`, chosen by killing the axis `e` leans on
/// least so the subtraction never cancels.
fn any_orthogonal(e: &[f64]) -> Vec<f64> {
    let n = e.len();
    let mut axis = 0usize;
    for d in 1..n {
        if e[d].abs() < e[axis].abs() {
            axis = d;
        }
    }
    let mut v = vec![0.0; n];
    v[axis] = 1.0;
    let proj = dot(&v, e);
    for d in 0..n {
        v[d] -= proj * e[d];
    }
    let norm = dot(&v, &v).sqrt();
    if norm <= 1e-12 {
        let mut fallback = vec![0.0; n];
        fallback[(axis + 1) % n] = 1.0;
        return fallback;
    }
    v.iter().map(|x| x / norm).collect()
}

/// A supporting half-space of the shadow that `ball(center, radius)` casts from
/// `station`, aimed at `query`.
///
/// **Why a half-space of the shadow is available at all.** The shadow of a convex
/// body from a point observer is convex (PAPER_1 Part A §5, derived here). For a
/// direction `n` the shadow's support value is attained on the body itself,
/// because every shadow point is a body point pushed AWAY from the station:
///
/// ```text
/// shadow = { station + λ (y − station) : y ∈ body, λ ≥ 1 }
/// n · (station + λ (y − station)) = n · station + λ n · (y − station)
/// ```
///
/// so as long as `n · (y − station) ≤ 0` for every body point — i.e. the station
/// lies in the free half-space — the value decreases in λ and the maximum is at
/// λ = 1. The tight supporting half-space is therefore
///
/// ```text
/// n · x  ≤  n · center + radius        (the body's own support value)
/// ```
///
/// and the constraint handed to the solver is its complement, `n · x ≥ offset`.
/// **The validity condition is exactly `n · station ≥ offset`**: the station must
/// be on the free side, otherwise the half-space does not contain the shadow and
/// certifies nothing. Both branches below establish it, and the second one
/// establishes it with equality.
///
/// Returns `None` when the station lies inside the inflated body. No plane
/// separates anything then — the sight line starts blocked — and emitting one
/// would be emitting a row that asserts nothing.
///
/// **`None` is not "satisfied".** It is "no certificate exists for this window".
/// The caller counts these as `OcclusionBuildResult::dropped_planes`, and the
/// exported certificate is INFINITE whenever the count is nonzero. Treating a
/// drop as a zero violation is what let a straight flight whose true
/// line-of-sight margin was −0.3999 come back converged, certified 0.0 and
/// figure-grade.
struct ShadowPlane {
    /// Unit outward normal; the free side is `n · x >= offset`.
    normal: Vec<f64>,
    offset: f64,
    /// Everything the rotation term needs, and nothing it does not.
    rot: ShadowRotation,
}

/// How the aiming direction moves when the segment centroid moves. Which of the
/// two branches produced the plane decides the formula, so the branch is carried
/// rather than re-derived.
enum ShadowRotation {
    /// Branch 1, sight line clear. `foot_norm` is the distance from the body
    /// centre to the sight segment, `tau` where the foot sits along it, and
    /// `to_query`/`to_center` the two offsets from the station.
    Clear {
        foot_norm: f64,
        tau: f64,
        to_query: Vec<f64>,
        to_center: Vec<f64>,
        clamped: bool,
    },
    /// Branch 2, plane rolled through the station. `e1` is the station-to-body
    /// axis, `e2` the lateral direction the plane was rolled toward, and
    /// `lat_norm` the length that direction was normalized by.
    Rolled {
        e1: Vec<f64>,
        e2: Vec<f64>,
        lat_norm: f64,
        cos_a: f64,
    },
}

fn shadow_plane(
    station: &[f64],
    query: &[f64],
    center: &[f64],
    radius: f64,
) -> Option<ShadowPlane> {
    let sd = center.len();
    let to_center: Vec<f64> = (0..sd).map(|d| center[d] - station[d]).collect();
    let dist_station = dot(&to_center, &to_center).sqrt();
    if dist_station <= radius * (1.0 + 1e-9) {
        return None;
    }

    // Branch 1 — the sight line is clear. Aim the normal along the shortest
    // offset from the body centre to the sight segment [station, query]. The
    // resulting row reproduces the exact line-of-sight margin at the reference,
    // the same property the KOZ rows have.
    let seg: Vec<f64> = (0..sd).map(|d| query[d] - station[d]).collect();
    let seg_sq = dot(&seg, &seg);
    let tau_raw = if seg_sq > 1e-12 {
        dot(&to_center, &seg) / seg_sq
    } else {
        0.0
    };
    let tau = tau_raw.clamp(0.0, 1.0);
    let clamped = seg_sq <= 1e-12 || tau_raw <= 0.0 || tau_raw >= 1.0;
    let foot: Vec<f64> = (0..sd).map(|d| station[d] + tau * seg[d]).collect();
    let off: Vec<f64> = (0..sd).map(|d| foot[d] - center[d]).collect();
    let off_norm = dot(&off, &off).sqrt();
    if off_norm > 1e-12 {
        let n: Vec<f64> = off.iter().map(|v| v / off_norm).collect();
        let offset = dot(&n, center) + radius;
        if dot(&n, station) >= offset - 1e-12 {
            return Some(ShadowPlane {
                normal: n,
                offset,
                rot: ShadowRotation::Clear {
                    foot_norm: off_norm,
                    tau,
                    to_query: seg,
                    to_center,
                    clamped,
                },
            });
        }
    }

    // Branch 2 — the query is inside the shadow, so branch 1's plane would put
    // the STATION on the forbidden side and certify nothing. Roll the plane
    // around the body until it passes through the station, keeping the query's
    // side. That is the limiting valid member of the family: `n · station`
    // equals the offset exactly, so the validity condition holds with equality
    // and the row still pushes the query out along `n`.
    let e1: Vec<f64> = to_center.iter().map(|v| v / dist_station).collect();
    let mut lateral: Vec<f64> = (0..sd).map(|d| query[d] - station[d]).collect();
    let proj = dot(&lateral, &e1);
    for d in 0..sd {
        lateral[d] -= proj * e1[d];
    }
    let lat_norm = dot(&lateral, &lateral).sqrt();
    let e2: Vec<f64> = if lat_norm > 1e-9 {
        lateral.iter().map(|v| v / lat_norm).collect()
    } else {
        // The query sits on the station-to-body axis: every direction is
        // equally good, so any of them is picked rather than none.
        any_orthogonal(&e1)
    };
    let sin_a = radius / dist_station;
    let cos_a = (1.0 - sin_a * sin_a).max(0.0).sqrt();
    let n: Vec<f64> = (0..sd)
        .map(|d| -sin_a * e1[d] + cos_a * e2[d])
        .collect();
    let offset = dot(&n, center) + radius;
    Some(ShadowPlane {
        normal: n,
        offset,
        rot: ShadowRotation::Rolled {
            e1,
            e2,
            lat_norm: lat_norm.max(1e-9),
            cos_a,
        },
    })
}

/// The part of `d_k` that moves the aiming direction when the segment centroid
/// moves — the occlusion analogue of the KOZ rotation term, and it exists for the
/// same measured reason.
///
/// Freezing the normal at the reference is SOUND: the plane supports the shadow
/// whoever aimed it, so the certificate is unaffected. What freezing costs is
/// convergence. A step optimized against a frozen plane lands flush on a plane
/// that has since pivoted, the next iteration grades it against a plane it never
/// saw, and the ratio test rejects. Measured on `station_fence` before this term
/// existed: 12 to 19 rejections per run, trust region collapsing at every
/// configuration below 16 segments.
///
/// Differentiating `g_k = n(m) · (q_k − centre) − R` through the centroid's
/// effect on `n` gives, with `P = I − n nᵀ` and `d_k = q_k − centre`:
///
/// ```text
/// clear branch:   corr = ( τ · P d_k  +  ((P d_k)·b) (a − 2τ b)/|b|² ) / |u|
/// rolled branch:  corr = cos_a · (I − e1 e1ᵀ − e2 e2ᵀ) d_k / |lateral|
/// ```
///
/// where `b` is station-to-query, `a` station-to-centre, `u` the offset from the
/// centre to the sight segment, and `τ` where the foot sits along the sight
/// segment. A CLAMPED foot does not slide, so its `τ` derivative drops out.
///
/// Like the KOZ rotation row this is a first-order model and is NOT conservative;
/// it is never used for the certificate, which always rebuilds the exact rows.
fn shadow_rotation_correction(rot: &ShadowRotation, normal: &[f64], d_k: &[f64]) -> Vec<f64> {
    let sd = d_k.len();
    match rot {
        ShadowRotation::Clear {
            foot_norm,
            tau,
            to_query,
            to_center,
            clamped,
        } => {
            let s_k = dot(normal, d_k);
            let proj: Vec<f64> = (0..sd).map(|d| d_k[d] - s_k * normal[d]).collect();
            let b_sq = dot(to_query, to_query);
            let mut corr: Vec<f64> = (0..sd).map(|d| tau * proj[d] / foot_norm).collect();
            if !*clamped && b_sq > 1e-12 {
                let scale = dot(&proj, to_query) / (b_sq * foot_norm);
                for d in 0..sd {
                    corr[d] += scale * (to_center[d] - 2.0 * tau * to_query[d]);
                }
            }
            corr
        }
        ShadowRotation::Rolled {
            e1,
            e2,
            lat_norm,
            cos_a,
        } => {
            let c1 = dot(e1, d_k);
            let c2 = dot(e2, d_k);
            (0..sd)
                .map(|d| cos_a * (d_k[d] - c1 * e1[d] - c2 * e2[d]) / lat_norm)
                .collect()
        }
    }
}

/// Line-of-sight occlusion rows: one supporting half-space per
/// (segment, occluder piece, station), applied to every control point of that
/// segment.
///
/// The claim these rows exist to make true: keeping line of sight to a fixed
/// station past a moving occluder costs ONE linearized supporting-half-space row
/// per (segment, occluder-piece, station). The row count below is exactly that,
/// times the control points of a segment — the same shape as the KOZ block,
/// where one plane per (segment, obstacle) is asserted at every control point so
/// the convex-hull property carries it to the whole segment.
///
/// A non-straight occluder path is a CHAIN of straight pieces on adjacent time
/// windows (PAPER_1 §"Occluder geometry"). Each piece is an ordinary obstacle
/// carrying `t_start`/`t_end`, so this builder needs no chain machinery: it
/// iterates obstacles, and a piece whose window misses the segment's time extent
/// emits nothing.
///
/// **The time coefficient is zero here, and that is not defect G1 returning.**
/// G1 was a KOZ plane against a SLANTED tube, where deleting the time component
/// told the solver that arriving earlier or later could not change clearance. The
/// object here is different: the piece's outer approximation is a fixed ball over
/// the window, so its shadow extruded across the window is a PRISM with walls
/// parallel to the time axis, and a supporting plane of a prism genuinely has no
/// time component. Time enters through which pieces are active and through the
/// window the body is built over — both functions of the control points.
///
/// The half-space ignores the window's finite height, so it forbids its side for
/// all time rather than only during the window. Conservative, in the safe
/// direction, and the reason the occlusion demo does not exhibit waiting.
///
/// **The occlusion constraint subsumes collision with the same body**: a sight
/// line that starts inside the occluder is blocked, so a point satisfying these
/// rows is outside the body as well. The keep-out rows for that body are present
/// but not expected to bind.
pub fn build_spacetime_occlusion_constraints(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    stations: &StationData<'_>,
    trust_radius: f64,
    sound_clip: bool,
) -> OcclusionBuildResult {
    build_occlusion_rows(
        a_list, p, np1, dim, obstacles, stations, trust_radius, sound_clip, false,
    )
}

/// The same half-spaces, plus the term that anticipates the plane rotating as the
/// solver moves the control points. See `shadow_rotation_correction`. Handed to
/// the QP; never used for the certificate.
pub fn build_spacetime_occlusion_constraints_linearized(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    stations: &StationData<'_>,
    trust_radius: f64,
    sound_clip: bool,
) -> OcclusionBuildResult {
    build_occlusion_rows(
        a_list, p, np1, dim, obstacles, stations, trust_radius, sound_clip, true,
    )
}

/// No rows and nothing dropped — the honest description of "occlusion is off".
fn no_occlusion_rows() -> OcclusionBuildResult {
    OcclusionBuildResult {
        bundle: None,
        dropped_planes: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_occlusion_rows(
    a_list: &[Vec<f64>],
    p: &[f64],
    np1: usize,
    dim: usize,
    obstacles: &SpacetimeObstacleData<'_>,
    stations: &StationData<'_>,
    trust_radius: f64,
    sound_clip: bool,
    with_rotation: bool,
) -> OcclusionBuildResult {
    if stations.n_stations == 0 || obstacles.n_obs == 0 {
        return no_occlusion_rows();
    }
    let spatial_dim = obstacles.spatial_dim;
    if dim != spatial_dim + 1 {
        return no_occlusion_rows();
    }
    let n_vars = np1 * dim;
    let t_idx = dim - 1;

    let mut constraint_rows: Vec<Vec<f64>> = Vec::new();
    let mut lbs: Vec<f64> = Vec::new();
    let mut row_meta: Vec<OcclusionRowData> = Vec::new();
    let mut dropped_planes = 0usize;

    for (seg_idx, a_seg) in a_list.iter().enumerate() {
        let (q, w, centroid, seg_radius) = segment_points_and_weights(a_seg, p, np1, dim);

        // The times this segment occupies. The curve's time values over the
        // segment lie in the hull of its own control point times, so this
        // bracket contains them — conservative, and a function of the
        // optimization variables like everything else here.
        let mut seg_t_lo = f64::INFINITY;
        let mut seg_t_hi = f64::NEG_INFINITY;
        for k in 0..np1 {
            let t_val = q[k * dim + t_idx];
            seg_t_lo = seg_t_lo.min(t_val);
            seg_t_hi = seg_t_hi.max(t_val);
        }
        let _ = (seg_radius, trust_radius, sound_clip);

        for obs_idx in 0..obstacles.n_obs {
            // Occluder material COEXISTING with the segment, convexified exactly
            // by subdividing the obstacle's own Bezier. Deliberately not the
            // keep-out clip ball — see `band_over_times` for why proximity is the
            // wrong localization for a shadow.
            let Some(band) = band_over_times(obstacles, obs_idx, seg_t_lo, seg_t_hi) else {
                continue;
            };
            let n_ctrl = obstacles.n_ctrl;
            let t_lo = band.g_tilde[t_idx];
            let t_hi = band.g_tilde[(n_ctrl - 1) * dim + t_idx];
            let (center, body_radius) =
                spatial_containing_ball(&band, spatial_dim, obstacles.radii[obs_idx]);

            for st_idx in 0..stations.n_stations {
                let station = &stations.pos[st_idx * spatial_dim..(st_idx + 1) * spatial_dim];
                // A window that IS in range but whose plane cannot be built is
                // counted, not skipped quietly. The QP still gets nothing —
                // there is no valid convex constraint to give it — but the
                // caller now knows the guarantee has a hole in it.
                let Some(plane) =
                    shadow_plane(station, &centroid[..spatial_dim], &center, body_radius)
                else {
                    dropped_planes += 1;
                    continue;
                };
                let n = &plane.normal;
                let offset = plane.offset;

                for k in 0..np1 {
                    let mut lhs = 0.0;
                    for d in 0..spatial_dim {
                        lhs += n[d] * q[k * dim + d];
                    }
                    let margin = lhs - offset;

                    let mut row = vec![0.0; n_vars];
                    let corr = if with_rotation {
                        let d_k: Vec<f64> =
                            (0..spatial_dim).map(|d| q[k * dim + d] - center[d]).collect();
                        Some(shadow_rotation_correction(&plane.rot, n, &d_k))
                    } else {
                        None
                    };
                    for j in 0..np1 {
                        let a_kj = a_seg[k * np1 + j];
                        for d in 0..spatial_dim {
                            row[j * dim + d] += a_kj * n[d];
                            if let Some(ref c) = corr {
                                row[j * dim + d] += w[j] * c[d];
                            }
                        }
                    }

                    // Bound chosen so the row reproduces `margin` exactly at
                    // x = p, for both row types. With no rotation term this is
                    // just `offset`.
                    let grad_dot_p = (0..n_vars).map(|idx| row[idx] * p[idx]).sum::<f64>();
                    let lb = grad_dot_p - margin;

                    row_meta.push(OcclusionRowData {
                        segment_idx: seg_idx,
                        cp_idx: k,
                        obstacle_idx: obs_idx,
                        station_idx: st_idx,
                        normal: n.clone(),
                        body_center: center.clone(),
                        body_radius,
                        t_lo,
                        t_hi,
                        lower_bound: lb,
                        margin,
                    });
                    constraint_rows.push(row);
                    lbs.push(lb);
                }
            }
        }
    }

    if constraint_rows.is_empty() {
        return OcclusionBuildResult {
            bundle: None,
            dropped_planes,
        };
    }

    let n_rows = constraint_rows.len();
    let mut a = vec![0.0; n_rows * n_vars];
    for (i, row) in constraint_rows.iter().enumerate() {
        a[i * n_vars..(i + 1) * n_vars].copy_from_slice(row);
    }

    OcclusionBuildResult {
        bundle: Some(OcclusionConstraintBundle {
            constraint: LinearConstraint {
                a,
                lb: lbs,
                ub: vec![f64::INFINITY; n_rows],
                n_rows,
                n_vars,
            },
            rows: row_meta,
        }),
        dropped_planes,
    }
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
pub fn build_box_constraints(
    p_init: &[f64],
    np1: usize,
    dim: usize,
    coord_lb: f64,
    coord_ub: f64,
    time_lb: f64,
    time_ub: f64,
    free_arrival_time: bool,
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

            let is_endpoint = (i == 0 || i == np1 - 1)
                && !(free_arrival_time && i == np1 - 1 && d == t_idx);
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
