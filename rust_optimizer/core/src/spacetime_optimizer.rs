use crate::bezier;
use crate::constraints::LinearConstraint;
use crate::de_casteljau;
use crate::optimizer::{solve_qp_with_socs, OptResult, SocBlock};
use crate::spacetime_constraints::{self, KozRowData, SpacetimeObstacleData};
use std::collections::HashMap;

/// Trust radius used when a caller supplies none.
///
/// There is one solver, and it is a trust-region method — so there is always a
/// trust region. A zero radius is not "the old behaviour", it is a degenerate
/// region that pins every variable to its current value, so it is treated as
/// "unspecified" and replaced by this.
pub const DEFAULT_TRUST_RADIUS: f64 = 0.5;

/// Step is accepted when the actual/predicted merit reduction exceeds this.
const ETA_ACCEPT: f64 = 0.1;
/// Consecutive qualifying iterations required before convergence is declared. A
/// single quiet iteration is not evidence when the half-spaces re-aim between
/// iterations.
const CONV_STREAK_REQUIRED: usize = 3;

/// Quadratic form of the PARAMETER-DOMAIN smoothness regularizer on the spatial
/// control points.
///
/// It is not spatial acceleration energy, and calling it that was defect-grade
/// wrong (formulation decisions 1-3, item B8). The curve parameter is not time:
/// physical velocity is the spatial parameter-derivative over the time
/// parameter-derivative, a ratio of Beziers, so it is not polynomial in the
/// control points and no quadratic form can equal its energy.
///
/// The term is blind to timing, and provably so: it never writes into the time
/// column, so two polygons with identical spatial control points and any time
/// coordinates whatsoever score the same. Physics is bounded by constraints --
/// the slant-limit speed cap (item B9) -- never by this cost.
fn build_smoothness_regularizer_h(np1: usize, dim: usize) -> Vec<f64> {
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

/// Quadratic cost together with the magnitude of the arithmetic that produced it.
///
/// The value is a signed sum whose terms can cancel almost completely: a straight
/// line has exactly zero smoothness cost, so `0.5 xᵀHx` evaluates to ~1e-12 of
/// cancellation residue rather than to zero. A tolerance stated relative to the
/// VALUE then has nothing to be relative to, and every comparison against it is a
/// comparison against noise.
///
/// `scale` is the sum of absolute term magnitudes — the standard floating-point
/// error bound for a quadratic form. Noise in the value is bounded by roughly
/// `eps · scale`, which stays meaningful when the value itself cancels to zero.
fn quadratic_cost_scaled(h: &[f64], f: &[f64], x: &[f64], nvars: usize) -> (f64, f64) {
    let mut value = 0.0;
    let mut scale = 0.0;
    for i in 0..nvars {
        for j in 0..nvars {
            let term = 0.5 * x[i] * h[i * nvars + j] * x[j];
            value += term;
            scale += term.abs();
        }
        let lin = f[i] * x[i];
        value += lin;
        scale += lin.abs();
    }
    (value, scale)
}

/// The subproblem's objective evaluated at `x`: quadratic smoothness plus the
/// linear time penalty.
///
/// The linear term MUST appear here as well as in the QP. The ratio test grades
/// the step against this function, so a cost term present in the solver but
/// absent from the merit would mean the solver is minimizing something other
/// than what is being measured — and the resulting rho would look like model
/// error when it is bookkeeping error.
fn quadratic_cost(h: &[f64], f: &[f64], x: &[f64], nvars: usize) -> f64 {
    let mut cost = 0.0;
    for i in 0..nvars {
        let mut hx = 0.0;
        for j in 0..nvars {
            hx += h[i * nvars + j] * x[j];
        }
        cost += 0.5 * x[i] * hx + f[i] * x[i];
    }
    cost
}

/// Total lower-bound violation of rows `[row_start, row_start + n_rows)` at `x`.
///
/// This is the penalty term of the merit function. KOZ rows are one-sided
/// (`a·x >= lb`, `ub = +inf`), so only the lower bound can be violated.
fn row_violation(
    x: &[f64],
    a_rows: &[f64],
    lb: &[f64],
    row_start: usize,
    n_rows: usize,
    nvars: usize,
) -> f64 {
    let mut total = 0.0;
    for r in row_start..row_start + n_rows {
        if !lb[r].is_finite() {
            continue;
        }
        let mut ax = 0.0;
        for j in 0..nvars {
            ax += a_rows[r * nvars + j] * x[j];
        }
        total += (lb[r] - ax).max(0.0);
    }
    total
}

/// Two-sided violation of every row OUTSIDE the KOZ block: box, boundary,
/// monotonicity, trust. These carry no slack variable, so `pred >= 0` only holds
/// when the reference satisfies them — see the bootstrap branch in the SCvx loop.
fn hard_row_violation(
    x: &[f64],
    a_rows: &[f64],
    lb: &[f64],
    ub: &[f64],
    total_rows: usize,
    koz_row_start: usize,
    n_koz: usize,
    nvars: usize,
) -> f64 {
    let mut total = 0.0;
    for r in 0..total_rows {
        if r >= koz_row_start && r < koz_row_start + n_koz {
            continue;
        }
        let mut ax = 0.0;
        for j in 0..nvars {
            ax += a_rows[r * nvars + j] * x[j];
        }
        if lb[r].is_finite() {
            total += (lb[r] - ax).max(0.0);
        }
        if ub[r].is_finite() {
            total += (ax - ub[r]).max(0.0);
        }
    }
    total
}

/// Violation of KOZ rows rebuilt AT `x` — the "true" merit's penalty term.
///
/// This is the whole point of the ratio test here. The QP optimizes against
/// half-spaces built at the reference; this rebuilds them at the candidate and
/// asks whether the improvement survived the walls moving. Because the
/// smoothness regularizer is EXACTLY quadratic (no linearization anywhere), the gap
/// between predicted and actual reduction is attributable to this term and
/// nothing else.
fn koz_violation_rebuilt_at(
    x: &[f64],
    pre: &ScpPrecomputed,
    obstacles: &SpacetimeObstacleData<'_>,
    cap_bulge_ratio: f64,
) -> f64 {
    let nvars = pre.np1 * pre.dim;
    match spacetime_constraints::build_spacetime_koz_constraints(
        &pre.a_list,
        x,
        pre.np1,
        pre.dim,
        obstacles,
        cap_bulge_ratio,
    ) {
        Some(bundle) => row_violation(
            x,
            &bundle.constraint.a,
            &bundle.constraint.lb,
            0,
            bundle.constraint.n_rows,
            nvars,
        ),
        None => 0.0,
    }
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
    /// Linear cost. All zeros unless a time penalty is set, in which case the
    /// single nonzero entry is `time_weight` on the last control point's time
    /// coordinate — which IS the arrival time, because a Bezier passes through
    /// its last control point (item B10).
    pub f_linear: Vec<f64>,
    pub boundary: LinearConstraint,
    pub monotonicity: LinearConstraint,
    pub box_constraints: LinearConstraint,
    /// Slant-limit speed cap, one cone per control-polygon leg. Empty when no
    /// cap was requested.
    pub speed_cap: Vec<SocBlock>,
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

    // ---- SCvx quantities. Populated only when `use_scvx` is true. ----
    //
    // On the SCvx path `p_new` is the RAW CANDIDATE, not an accepted iterate:
    // scp_step solves one convex subproblem and the outer loop decides whether to
    // take it. On the legacy path `p_new` is the (unconditionally accepted) next
    // iterate, exactly as before. These are different concepts and the debug trace
    // must not conflate them — hence `is_candidate`.
    pub is_candidate: bool,
    /// Convex (predicted) merit at the reference: objective + weight · linearized violation.
    pub l_p: f64,
    /// Convex (predicted) merit at the candidate, against the SAME rows.
    pub l_c: f64,
    /// Linearized KOZ violation at the reference. Equals the true violation there:
    /// each row's support point is the exact closest surface point to its query
    /// point, so `a·q - lb` reproduces the exact clearance at the reference.
    pub vlin_p: f64,
    /// Linearized KOZ violation at the candidate, against the reference's rows.
    pub vlin_c: f64,
    /// Violation of the non-KOZ rows at the reference. Must be ~0 for the
    /// predicted reduction to be meaningful.
    pub hard_viol_p: f64,
}

/// Precompute the data that stays constant across SCP iterations.
///
/// `v_max <= 0` or non-finite means no speed cap; `time_weight == 0.0` plus
/// `free_arrival_time == false` is the pre-B9/B10 problem exactly.
#[allow(clippy::too_many_arguments)]
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
    v_max: f64,
    time_weight: f64,
    free_arrival_time: bool,
) -> ScpPrecomputed {
    let n = np1 - 1;
    let nvars = np1 * dim;
    let mut f_linear = vec![0.0; nvars];
    if time_weight != 0.0 {
        f_linear[(np1 - 1) * dim + (dim - 1)] = time_weight;
    }
    ScpPrecomputed {
        a_list: de_casteljau::segment_matrices_equal_params(n, n_seg),
        h_energy: build_smoothness_regularizer_h(np1, dim),
        f_linear,
        boundary: spacetime_constraints::build_boundary_constraints(
            np1,
            dim,
            &p_init[0..dim],
            &p_init[(np1 - 1) * dim..np1 * dim],
            free_arrival_time,
        ),
        monotonicity: spacetime_constraints::build_time_monotonicity(np1, dim, min_dt),
        box_constraints: spacetime_constraints::build_box_constraints(
            p_init, np1, dim, coord_lb, coord_ub, time_lb, time_ub, free_arrival_time,
        ),
        speed_cap: spacetime_constraints::build_speed_cap_socs(np1, dim, v_max),
        np1,
        dim,
    }
}

/// The step result for a subproblem the solver could not solve at all.
///
/// `converged: false` matters: a failed solve is the loop giving up, and the two
/// must never be reported as the same outcome.
fn failed_step(
    p_current: &[f64],
    pre: &ScpPrecomputed,
    obstacles: &SpacetimeObstacleData<'_>,
) -> ScpStepResult {
    let nvars = pre.np1 * pre.dim;
    ScpStepResult {
        p_new: p_current.to_vec(),
        solver_status: "Failed".to_string(),
        delta: 0.0,
        raw_step_norm: 0.0,
        clearance: compute_min_clearance(p_current, pre.np1, pre.dim, obstacles, 1500),
        total_slack: 0.0,
        max_slack: 0.0,
        converged: false,
        cost: quadratic_cost(&pre.h_energy, &pre.f_linear, p_current, nvars),
        koz_rows: Vec::new(),
        koz_slack_per_row: Vec::new(),
        is_candidate: false,
        l_p: f64::NAN,
        l_c: f64::NAN,
        vlin_p: f64::NAN,
        vlin_c: f64::NAN,
        hard_viol_p: f64::NAN,
    }
}

/// Run one SCP iteration: linearize KOZ, build QP, solve, evaluate.
///
/// On the legacy path (`use_scvx == false`) the solved step is clipped to the
/// trust radius and returned as the next iterate — unchanged behaviour.
///
/// On the SCvx path the trust region is a constraint instead of a clip, and the
/// returned point is a CANDIDATE: this function does not decide whether to take
/// it. The caller runs the ratio test.
pub fn scp_step(
    p_current: &[f64],
    pre: &ScpPrecomputed,
    obstacles: &SpacetimeObstacleData<'_>,
    scp_prox_weight: f64,
    scp_trust_radius: f64,
    elastic_weight: f64,
    tol: f64,
    cap_bulge_ratio: f64,
    iteration: u32,
) -> ScpStepResult {
    let np1 = pre.np1;
    let dim = pre.dim;
    let nvars = np1 * dim;
    // A zero or negative radius means "unspecified", not "no trust region": a
    // degenerate box pins every variable to its current value and would report
    // instant convergence at the initial guess.
    let trust_radius = if scp_trust_radius > 0.0 {
        scp_trust_radius
    } else {
        DEFAULT_TRUST_RADIUS
    };

    // Build KOZ constraints with per-row metadata
    // The QP gets the SELF-CONSISTENT rows: same half-spaces, plus the term that
    // anticipates the plane rotating as the control points move. Without it the
    // step lands flush on a plane that pivots out from under it and the next
    // iteration rejects it — measured at 10-13 rejections per ~17 iterations.
    //
    // The certificate is never evaluated with these; `koz_violation_rebuilt_at`
    // uses the exact rows. That separation is what keeps the reported guarantee
    // sound while letting the subproblem model the pivot.
    let _ = cap_bulge_ratio;
    let mut koz_bundle = spacetime_constraints::build_spacetime_koz_constraints_linearized(
        &pre.a_list, p_current, np1, dim, obstacles,
    );
    if let Some(ref mut bundle) = koz_bundle {
        for row in &mut bundle.rows {
            row.iteration = iteration;
        }
    }

    // Objective: the parameter-domain smoothness regularizer on the spatial
    // control points, and nothing else. NOT acceleration energy -- see
    // `build_smoothness_regularizer_h`.
    //
    // There is deliberately no proximal term. The trust region already bounds the
    // step, and a proximal weight present in the QP but absent from the merit
    // would break the argument that the predicted reduction is nonnegative — the
    // solver would be minimizing a different function from the one being graded.
    // `scp_prox_weight` is retained in the signature only so existing callers
    // keep compiling; it is not applied.
    let _ = scp_prox_weight;
    let h = pre.h_energy.clone();
    // Linear time penalty (item B10). Zero-filled unless a time weight was set,
    // so a default run solves the identical QP it always did.
    let f = pre.f_linear.clone();

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

    // SCvx trust region, enforced as CONSTRAINTS in the subproblem rather than by
    // clipping the solved step afterwards.
    //
    // Clipping (the legacy behaviour, below) solves an unbounded-step QP and then
    // scales the answer back. The scaled point is not the optimum of anything: the
    // QP never saw the bound, so the returned direction is optimal for a region the
    // step is not allowed to reach. That makes the convex subproblem an unfaithful
    // local model, which is exactly what the ratio test assumes it is not.
    // Appended AFTER the KOZ block so the elastic extension's row indexing (which
    // keys off `koz_row_start .. koz_row_start + n_koz`) is unaffected, and so the
    // trust rows are carried as hard rows there.
    for i in 0..nvars {
        let mut row = vec![0.0; nvars];
        row[i] = 1.0;
        all_a_rows.extend_from_slice(&row);
        all_lb.push(p_current[i] - trust_radius);
        all_ub.push(p_current[i] + trust_radius);
        total_rows += 1;
    }

    // Solve QP.
    //
    // Legacy path: hard first, elastic fallback if infeasible.
    // SCvx path: elastic UNCONDITIONALLY. The subproblem then minimizes exactly the
    // penalized convex merit the ratio test measures, which makes the predicted
    // reduction nonnegative by construction — x = p with slack set to the
    // reference's own violation is always elastic-feasible. Solving the hard form
    // first would minimize a different function than the one being graded, and a
    // negative predicted reduction is then not a model defect but a bookkeeping
    // error that is indistinguishable from one.
    let (x_new, iter_total_slack, iter_max_slack, koz_slack_per_row, solver_status);

    //
    // The "unconditionally elastic" rule has one exception: with no KOZ rows there
    // is nothing to relax, and the elastic branch below is guarded on `n_koz > 0`.
    // Forcing the hard solve to be skipped there sends every such subproblem to the
    // failure return — so an obstacle-free problem, or one where every obstacle's
    // time window misses the plan horizon, would report a QP failure on iteration 1.
    // Caught by the no-obstacle invariant check, which requires the ratio to be
    // exactly 1 when the model is exact and instead found no ratios at all.
    let elastic_available = elastic_weight > 0.0 && n_koz > 0;
    let hard_sol = if elastic_available {
        None
    } else {
        solve_qp_with_socs(
            &h,
            &f,
            &all_a_rows,
            &all_lb,
            &all_ub,
            nvars,
            total_rows,
            &pre.speed_cap,
        )
    };

    if let Some(x_sol) = hard_sol {
        x_new = x_sol;
        iter_total_slack = 0.0;
        iter_max_slack = 0.0;
        koz_slack_per_row = vec![0.0; n_koz];
        solver_status = "Solved".to_string();
    } else if elastic_available {
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

        // The speed cap carries NO slack variable: it is a hard convex
        // constraint, not something the elastic penalty may buy its way out of.
        // Its columns are widened with zeros so the cone sees only the original
        // variables.
        let socs_ext: Vec<SocBlock> = pre
            .speed_cap
            .iter()
            .map(|b| b.widened(nvars, n_koz))
            .collect();

        match solve_qp_with_socs(
            &h_ext, &f_ext, &a_ext, &lb_ext, &ub_ext, nvars_ext, ext_nrows, &socs_ext,
        ) {
            Some(x_full) => {
                x_new = x_full[..nvars].to_vec();
                koz_slack_per_row = x_full[nvars..].to_vec();
                iter_total_slack = koz_slack_per_row.iter().sum::<f64>();
                iter_max_slack = koz_slack_per_row.iter().cloned().fold(0.0f64, f64::max);
                solver_status = "Elastic".to_string();
            }
            None => return failed_step(p_current, pre, obstacles),
        }
    } else {
        return failed_step(p_current, pre, obstacles);
    }

    // The trust box was imposed as rows above, so the solution is inside it by
    // construction. Nothing is clipped: clipping would return a point that is the
    // optimum of a problem the solver never saw.
    let x_result = x_new.clone();
    let raw_step_norm = (0..nvars)
        .map(|i| (x_result[i] - p_current[i]).powi(2))
        .sum::<f64>()
        .sqrt();

    // SCvx merit pieces. Both merits use the UNREGULARIZED objective (`pre.h_energy`,
    // which carries no linear term) so the ratio measures model error and not the
    // solver's own regularization. `vlin_p == vtrue_p` for this builder: each row's
    // support point is the exact closest point on the tube surface to its query
    // point and the normal is the unit vector along that offset, so `a·q - lb`
    // reproduces the exact clearance at the reference. Only the candidate needs
    // rebuilt walls, and the caller does that.
    let (l_p, l_c, vlin_p, vlin_c, hard_viol_p) = {
        let w_s = elastic_weight.max(0.0);
        let vp = row_violation(p_current, &all_a_rows, &all_lb, koz_row_start, n_koz, nvars);
        let vc = row_violation(&x_result, &all_a_rows, &all_lb, koz_row_start, n_koz, nvars);
        let qp_ = quadratic_cost(&pre.h_energy, &pre.f_linear, p_current, nvars);
        let qc_ = quadratic_cost(&pre.h_energy, &pre.f_linear, &x_result, nvars);
        // The speed-cap cones carry no slack, so they belong with the hard rows:
        // if the reference violates them, `x = p` is not feasible for the
        // subproblem and the predicted reduction is not guaranteed nonnegative.
        // The bootstrap branch in `scp_iterate` keys off exactly this.
        let hv = hard_row_violation(
            p_current,
            &all_a_rows,
            &all_lb,
            &all_ub,
            total_rows,
            koz_row_start,
            n_koz,
            nvars,
        ) + spacetime_constraints::speed_cap_violation(&pre.speed_cap, p_current, nvars);
        (qp_ + w_s * vp, qc_ + w_s * vc, vp, vc, hv)
    };

    let delta = (0..nvars)
        .map(|i| (x_result[i] - p_current[i]).powi(2))
        .sum::<f64>()
        .sqrt();

    let clearance = compute_min_clearance(&x_result, np1, dim, obstacles, 1500);
    let cost = quadratic_cost(&pre.h_energy, &pre.f_linear, &x_result, nvars);
    // `scp_step` solves one convex subproblem. It does not decide anything: a small
    // step may well be a rejected one. The caller grades the candidate and owns the
    // convergence decision.
    let _ = tol;
    let converged = false;

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
        is_candidate: true,
        l_p,
        l_c,
        vlin_p,
        vlin_c,
        hard_viol_p,
    }
}

/// Why the loop stopped. Recorded in `info["stop_reason"]`.
///
/// "Hit the iteration cap" and "reached a stationary point" are opposite
/// outcomes and used to be indistinguishable — the loop simply broke out and the
/// caller saw an iteration count.
///
/// Only `MERIT_STREAK` and `STATIONARY` are claims of success. `ITERATION_CAP`,
/// `TRUST_COLLAPSE` and `QP_FAILURE` all mean the solver gave up.
pub mod stop_reason {
    /// Ran out of iterations. Not convergence.
    pub const ITERATION_CAP: f64 = 0.0;
    /// K consecutive accepted steps with negligible relative merit change, each
    /// carrying the hull certificate against its own rebuilt half-spaces.
    pub const MERIT_STREAK: f64 = 1.0;
    /// Trust radius shrank below its floor — the ratio test stopped accepting
    /// anything. Convergence only if the reference is certified.
    pub const TRUST_COLLAPSE: f64 = 2.0;
    /// The convex subproblem itself could not be solved.
    pub const QP_FAILURE: f64 = 3.0;
    /// K consecutive iterations where the model predicts no achievable progress.
    pub const STATIONARY: f64 = 4.0;
    /// Still running.
    pub const RUNNING: f64 = -1.0;
}

/// Everything that changes from one iteration to the next.
///
/// The trust radius and the streak counters are state of the ALGORITHM, not of
/// any single subproblem. A caller stepping one iteration at a time has to carry
/// them; a caller that re-derives them per step is running a different algorithm
/// from the batch loop, which is exactly the "two solvers" failure this type
/// exists to prevent.
pub struct ScpState {
    pub p: Vec<f64>,
    pub trust: f64,
    pub trust_min: f64,
    pub trust_max: f64,
    pub iteration: u32,
    pub conv_streak: usize,
    pub stat_streak: usize,
    pub converged: bool,
    pub stop: f64,
    pub best_p: Vec<f64>,
    pub best_clearance: f64,
    /// Elastic slack of the subproblem whose solution is the CURRENT iterate.
    ///
    /// `last_total_slack` is the slack of the most recent subproblem, accepted or
    /// not. On a rejected step those are different numbers describing different
    /// trajectories, and the feasibility gate (item B7) has to grade the point it
    /// actually returns.
    pub accepted_total_slack: f64,
    /// The same quantity for `best_p`.
    pub best_total_slack: f64,
    pub accept_count: usize,
    pub reject_count: usize,
    pub null_step_count: usize,
    pub bootstrap_count: usize,
    pub rho_sum: f64,
    pub rho_n: usize,
    pub rho_min: f64,
    pub rho_max: f64,
    pub last_rho: f64,
    pub last_delta: f64,
    pub last_total_slack: f64,
    pub last_max_slack: f64,
    pub last_vtrue_p: f64,
    pub last_vtrue_c: f64,
}

impl ScpState {
    pub fn new(p_init: &[f64], trust_radius: f64) -> Self {
        let trust = if trust_radius > 0.0 {
            trust_radius
        } else {
            DEFAULT_TRUST_RADIUS
        };
        Self {
            p: p_init.to_vec(),
            trust,
            trust_min: trust * 1e-3,
            trust_max: trust * 4.0,
            iteration: 0,
            conv_streak: 0,
            stat_streak: 0,
            converged: false,
            stop: stop_reason::RUNNING,
            best_p: p_init.to_vec(),
            best_clearance: f64::NEG_INFINITY,
            accepted_total_slack: f64::NAN,
            best_total_slack: f64::NAN,
            accept_count: 0,
            reject_count: 0,
            null_step_count: 0,
            bootstrap_count: 0,
            rho_sum: 0.0,
            rho_n: 0,
            rho_min: f64::INFINITY,
            rho_max: f64::NEG_INFINITY,
            last_rho: f64::NAN,
            last_delta: f64::NAN,
            last_total_slack: 0.0,
            last_max_slack: 0.0,
            last_vtrue_p: f64::NAN,
            last_vtrue_c: f64::NAN,
        }
    }

    pub fn running(&self) -> bool {
        self.stop == stop_reason::RUNNING
    }
}

/// One graded iteration: what the subproblem returned, and what was done with it.
pub struct ScpIteration {
    pub step: ScpStepResult,
    /// accept / accept_null / reject / reject_null / bootstrap / failed
    pub outcome: &'static str,
    pub accepted: bool,
    pub rho: f64,
    pub pred: f64,
    pub act: f64,
    /// Violation of the half-spaces rebuilt AT the candidate — the hull
    /// certificate for the point the step would move to.
    pub vtrue_c: f64,
    pub trust_before: f64,
    pub trust_after: f64,
}

/// The canonical SCP iteration: solve one convex subproblem, grade it against
/// half-spaces rebuilt at the candidate, accept or reject, adapt the trust
/// radius, and update the stopping state.
///
/// This is the ONLY place a step is accepted. The batch loop and the step
/// debugger both call it, so a debug session cannot diverge from a batch run.
pub fn scp_iterate(
    state: &mut ScpState,
    pre: &ScpPrecomputed,
    obstacles: &SpacetimeObstacleData<'_>,
    elastic_weight: f64,
    tol: f64,
    cap_bulge_ratio: f64,
) -> ScpIteration {
    let nvars = pre.np1 * pre.dim;
    state.iteration += 1;
    let trust_before = state.trust;

    let step = scp_step(
        &state.p,
        pre,
        obstacles,
        0.0,
        state.trust,
        elastic_weight,
        tol,
        cap_bulge_ratio,
        state.iteration,
    );

    if step.solver_status == "Failed" {
        state.stop = stop_reason::QP_FAILURE;
        return ScpIteration {
            step,
            outcome: "failed",
            accepted: false,
            rho: f64::NAN,
            pred: f64::NAN,
            act: f64::NAN,
            vtrue_c: f64::NAN,
            trust_before,
            trust_after: state.trust,
        };
    }

    state.last_delta = step.delta;
    state.last_total_slack = step.total_slack;
    state.last_max_slack = step.max_slack;

    // Convex merit  L(x) = J(x) + w · (violation of the rows the QP was given)
    // True merit    T(x) = J(x) + w · (violation of rows REBUILT at x)
    //
    // J is exactly quadratic — no linearization anywhere in the objective — so the
    // two can differ for exactly one reason: the half-spaces moved. rho is a pure
    // measurement of that.
    let cand = step.p_new.clone();
    let w_s = elastic_weight.max(0.0);
    let (quad_p, scale_p) = quadratic_cost_scaled(&pre.h_energy, &pre.f_linear, &state.p, nvars);
    let (quad_c, scale_c) = quadratic_cost_scaled(&pre.h_energy, &pre.f_linear, &cand, nvars);
    let vtrue_p = step.vlin_p; // rows were built at p; exact there
    let vtrue_c = koz_violation_rebuilt_at(&cand, pre, obstacles, cap_bulge_ratio);
    let t_p = quad_p + w_s * vtrue_p;
    let t_c = quad_c + w_s * vtrue_c;
    let pred = step.l_p - step.l_c;
    let act = t_p - t_c;

    // The magnitude every tolerance below is stated against. `|t_p|` alone is not
    // usable: the quadratic form is a signed sum whose terms cancel, and where the
    // optimum has near-zero smoothness cost the merit is cancellation residue rather
    // than a value.
    let cancel_floor = f64::EPSILON * scale_p.max(scale_c);
    let merit_scale = t_p.abs().max(cancel_floor / 1e-9);
    let merit_eps = 1e-9 * merit_scale;
    let rel = act.abs() / merit_scale;
    let tol_f = if tol > 0.0 { tol.max(1e-8) } else { 1e-8 };

    state.last_vtrue_p = vtrue_p;
    state.last_vtrue_c = vtrue_c;

    // `pred >= 0` holds only if x = p is feasible for the elastic subproblem, which
    // needs the reference to satisfy every row carrying no slack. The straight-line
    // initial guess generally violates the box/monotonicity rows, so the first
    // iteration is a repair step: the candidate does satisfy them, and grading a
    // repair with an undefined merit comparison would reject it forever.
    if step.hard_viol_p > 1e-9 {
        state.bootstrap_count += 1;
        state.p = cand;
        state.conv_streak = 0;
        state.stat_streak = 0;
        update_best(state, &step);
        return ScpIteration {
            step,
            outcome: "bootstrap",
            accepted: true,
            rho: f64::NAN,
            pred,
            act,
            vtrue_c,
            trust_before,
            trust_after: state.trust,
        };
    }

    // Model stationarity: the subproblem predicts no achievable progress from this
    // reference. Compared on |pred| — near a stationary point pred is a difference
    // of two merits agreeing to near machine precision, so its SIGN is noise and a
    // `pred >= 0` guard would reset the streak forever. Guarded on the reference
    // being certified so a stationary-but-penetrating point cannot report success.
    if pred.abs() < tol_f * merit_scale && vtrue_p <= 1e-6 {
        state.stat_streak += 1;
        if state.stat_streak >= CONV_STREAK_REQUIRED {
            state.converged = true;
            state.stop = stop_reason::STATIONARY;
        }
    } else {
        state.stat_streak = 0;
    }

    let pred_floor = merit_eps;
    let rho = if pred < pred_floor {
        state.null_step_count += 1;
        // The model sees no improvement to predict. Treat as a null step: accept if
        // it does not make things worse. Never divide by a non-positive prediction.
        if act >= -merit_eps {
            f64::INFINITY
        } else {
            -1.0
        }
    } else {
        act / pred
    };
    state.last_rho = rho;
    if rho.is_finite() {
        state.rho_sum += rho;
        state.rho_n += 1;
        state.rho_min = state.rho_min.min(rho);
        state.rho_max = state.rho_max.max(rho);
    }

    let outcome;
    let accepted;
    if rho > ETA_ACCEPT {
        accepted = true;
        state.accept_count += 1;
        outcome = if pred < pred_floor { "accept_null" } else { "accept" };
        state.p = cand;
        if rho > 0.9 && rho.is_finite() {
            state.trust = (state.trust * 2.0).min(state.trust_max);
        }
        update_best(state, &step);
        // Converged = merit stationary for K consecutive accepted steps AND the
        // accepted iterate carries the hull certificate against its OWN rebuilt
        // half-spaces. The second half is what stops a penetrating iterate being
        // reported as a success.
        if rel < tol_f && vtrue_c <= 1e-6 {
            state.conv_streak += 1;
            if state.conv_streak >= CONV_STREAK_REQUIRED && state.running() {
                state.converged = true;
                state.stop = stop_reason::MERIT_STREAK;
            }
        } else {
            state.conv_streak = 0;
        }
    } else {
        accepted = false;
        state.reject_count += 1;
        state.conv_streak = 0;
        outcome = if pred < pred_floor { "reject_null" } else { "reject" };
        state.trust *= 0.5;
        if state.trust < state.trust_min && state.running() {
            // The ratio test stopped accepting anything. Convergence only if the
            // reference is actually certified; otherwise a genuine failure, and it
            // must not be dressed up as one.
            state.converged = vtrue_p <= 1e-6 && step.hard_viol_p <= 1e-9;
            state.stop = stop_reason::TRUST_COLLAPSE;
        }
    }

    ScpIteration {
        step,
        outcome,
        accepted,
        rho,
        pred,
        act,
        vtrue_c,
        trust_before,
        trust_after: state.trust,
    }
}

/// Track the best FEASIBLE iterate seen. Kept separate from the current iterate:
/// they are different points, and reporting one alongside the other's convergence
/// claim would be quoting two different trajectories.
fn update_best(state: &mut ScpState, step: &ScpStepResult) {
    // `state.p` is the just-accepted candidate, so the slack that produced it is
    // this step's slack. Recorded together with the point, never separately.
    state.accepted_total_slack = step.total_slack;
    if step.clearance > 0.0 && step.clearance > state.best_clearance {
        state.best_clearance = step.clearance;
        state.best_p = state.p.clone();
        state.best_total_slack = step.total_slack;
    }
}

/// Full optimization loop: repeated `scp_iterate` until the state says stop.
#[allow(clippy::too_many_arguments)]
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
    cap_bulge_ratio: f64,
    v_max: f64,
    time_weight: f64,
    free_arrival_time: bool,
) -> OptResult {
    let _ = scp_prox_weight; // see scp_step: the trust region does this job
    let nvars = np1 * dim;
    let pre = precompute_scp(
        p_init, np1, dim, n_seg, min_dt, coord_lb, coord_ub, time_lb, time_ub, v_max,
        time_weight, free_arrival_time,
    );
    let mut state = ScpState::new(p_init, scp_trust_radius);
    // `best` deliberately does NOT start at the initial guess. Seeding it there
    // lets the fallback return the solver's own input when every iterate it
    // produced is worse — reporting a result the optimizer did not compute. It
    // also diverged from the stepping entry point, which never seeded it: two
    // callers of one solver disagreeing about the answer.

    // Per-iteration trace of EVERY step, accepted or rejected. Summary counters
    // alone hide the rejected steps that collapse the trust radius, which is
    // precisely the population worth looking at.
    let trace = std::env::var("SPACETIME_SCVX_TRACE").is_ok();
    if trace {
        eprintln!(
            "STTRACE,it,outcome,rho,pred,act,step_norm,trust_before,trust_after,\
vlin_p,vlin_c,vtrue_c,hard_viol_p,clearance,total_slack,conv_streak,stat_streak"
        );
    }

    while state.running() && (state.iteration as usize) < max_iter {
        let r = scp_iterate(&mut state, &pre, obstacles, elastic_weight, tol, cap_bulge_ratio);
        if trace {
            let cols = [
                r.rho,
                r.pred,
                r.act,
                r.step.delta,
                r.trust_before,
                r.trust_after,
                r.step.vlin_p,
                r.step.vlin_c,
                r.vtrue_c,
                r.step.hard_viol_p,
                r.step.clearance,
                r.step.total_slack,
            ];
            let nums: Vec<String> = cols.iter().map(|v| format!("{v:.9e}")).collect();
            eprintln!(
                "STTRACE,{},{},{},{},{}",
                state.iteration,
                r.outcome,
                nums.join(","),
                state.conv_streak,
                state.stat_streak
            );
        }
    }
    if state.running() {
        state.stop = stop_reason::ITERATION_CAP;
    }

    let mut p = state.p.clone();
    let mut final_clearance = compute_min_clearance(&p, np1, dim, obstacles, 1500);
    // Fall back to the best feasible iterate seen. This is a DIFFERENT point from
    // the one the loop stopped on, so it is flagged rather than silently swapped.
    let mut returned_best = false;
    if final_clearance < 0.0 && state.best_clearance > 0.0 {
        p = state.best_p.clone();
        final_clearance = state.best_clearance;
        returned_best = true;
    }
    // Slack belonging to the point being RETURNED, which is what the feasibility
    // gate must grade (item B7). NaN only if no step was ever accepted, i.e. the
    // returned point is the initial guess -- and NaN fails every comparison,
    // which is the correct outcome for "no evidence".
    let returned_total_slack = if returned_best {
        state.best_total_slack
    } else {
        state.accepted_total_slack
    };
    let speed_cap_viol =
        spacetime_constraints::speed_cap_violation(&pre.speed_cap, &p, nvars);

    let feasible = final_clearance > 0.0 || obstacles.n_obs == 0;
    let cost = quadratic_cost(&pre.h_energy, &pre.f_linear, &p, nvars);

    let mut info = HashMap::new();
    info.insert("iterations".to_string(), state.iteration as f64);
    info.insert("feasible".to_string(), if feasible { 1.0 } else { 0.0 });
    info.insert("min_clearance".to_string(), final_clearance);
    info.insert("cost_true_energy".to_string(), cost);
    info.insert("cost_no_const".to_string(), cost);
    info.insert("cost".to_string(), cost);
    info.insert("max_control_accel_ms2".to_string(), 0.0);
    info.insert("mean_control_accel_ms2".to_string(), 0.0);
    info.insert("final_delta_norm".to_string(), state.last_delta);
    info.insert("total_koz_slack".to_string(), state.last_total_slack);
    info.insert("max_koz_slack".to_string(), state.last_max_slack);
    info.insert(
        "total_koz_slack_returned".to_string(),
        returned_total_slack,
    );
    // Arrival time IS the last control point's time coordinate: a Bezier passes
    // through its last control point. Exported so a sweep over `time_weight` can
    // be read straight off the info dict.
    info.insert("arrival_time".to_string(), p[(np1 - 1) * dim + (dim - 1)]);
    info.insert("speed_cap_violation".to_string(), speed_cap_viol);
    info.insert("stop_reason".to_string(), state.stop);
    info.insert(
        "converged".to_string(),
        if state.converged { 1.0 } else { 0.0 },
    );
    info.insert(
        "returned_best_iterate".to_string(),
        if returned_best { 1.0 } else { 0.0 },
    );
    info.insert("accept_count".to_string(), state.accept_count as f64);
    info.insert("reject_count".to_string(), state.reject_count as f64);
    info.insert("null_step_count".to_string(), state.null_step_count as f64);
    info.insert("bootstrap_count".to_string(), state.bootstrap_count as f64);
    info.insert("final_trust".to_string(), state.trust);
    info.insert("rho_last".to_string(), state.last_rho);
    info.insert("rho_samples".to_string(), state.rho_n as f64);
    info.insert(
        "rho_mean".to_string(),
        if state.rho_n > 0 {
            state.rho_sum / state.rho_n as f64
        } else {
            f64::NAN
        },
    );
    info.insert(
        "rho_min".to_string(),
        if state.rho_n > 0 { state.rho_min } else { f64::NAN },
    );
    info.insert(
        "rho_max".to_string(),
        if state.rho_n > 0 { state.rho_max } else { f64::NAN },
    );
    // The hull certificate: > 0 means the curve does not satisfy the half-spaces
    // its own control points generate.
    //
    // Rebuilt at `p` -- the point this call actually RETURNS. It used to export
    // `state.last_vtrue_p`, the violation of the loop's final reference iterate.
    // Those are the same point only when the loop ended on a stationary or
    // rejected step; when the best-feasible fallback fires (`returned_best`),
    // `p` is `state.best_p` and the exported number described a trajectory
    // nobody received. Measured on `diverse` N8_seg4: reported 1.616 against a
    // true 2.264 at the returned control points. `certified` in optimize.py is
    // derived from this key, so the guarantee has to be evaluated where the
    // curve is.
    info.insert(
        "koz_violation_reference".to_string(),
        koz_violation_rebuilt_at(&p, &pre, obstacles, cap_bulge_ratio),
    );
    info.insert("koz_violation_candidate".to_string(), state.last_vtrue_c);
    // Kept for provenance: what the old key would have said.
    info.insert(
        "koz_violation_last_reference".to_string(),
        state.last_vtrue_p,
    );

    OptResult {
        p_opt: p,
        np1,
        dim,
        info,
        feasible,
        iterations: state.iteration as usize,
    }
}
