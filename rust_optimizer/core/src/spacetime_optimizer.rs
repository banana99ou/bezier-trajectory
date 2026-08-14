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

/// Quadratic cost together with the magnitude of the arithmetic that produced it.
///
/// The value is a signed sum whose terms can cancel almost completely: a straight
/// line has exactly zero bending energy, so `0.5 xᵀHx` evaluates to ~1e-12 of
/// cancellation residue rather than to zero. A tolerance stated relative to the
/// VALUE then has nothing to be relative to, and every comparison against it is a
/// comparison against noise.
///
/// `scale` is the sum of absolute term magnitudes — the standard floating-point
/// error bound for a quadratic form. Noise in the value is bounded by roughly
/// `eps · scale`, which stays meaningful when the value itself cancels to zero.
fn quadratic_cost_scaled(h: &[f64], x: &[f64], nvars: usize) -> (f64, f64) {
    let mut value = 0.0;
    let mut scale = 0.0;
    for i in 0..nvars {
        for j in 0..nvars {
            let term = 0.5 * x[i] * h[i * nvars + j] * x[j];
            value += term;
            scale += term.abs();
        }
    }
    (value, scale)
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
/// asks whether the improvement survived the walls moving. Because the spatial
/// energy objective is EXACTLY quadratic (no linearization anywhere), the gap
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
        cost: quadratic_cost(&pre.h_energy, p_current, nvars),
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
    use_scvx: bool,
) -> ScpStepResult {
    let np1 = pre.np1;
    let dim = pre.dim;
    let nvars = np1 * dim;
    // SCvx needs a positive radius to build the trust box; a caller asking for
    // SCvx with a zero radius would otherwise pin every variable to its current
    // value and report instant "convergence" at the initial guess.
    let scvx = use_scvx && scp_trust_radius > 0.0;

    // Build KOZ constraints with per-row metadata
    let mut koz_bundle = spacetime_constraints::build_spacetime_koz_constraints(
        &pre.a_list, p_current, np1, dim, obstacles, cap_bulge_ratio,
    );
    if let Some(ref mut bundle) = koz_bundle {
        for row in &mut bundle.rows {
            row.iteration = iteration;
        }
    }

    // Build objective: H_energy + proximal.
    //
    // The proximal term is SKIPPED on the SCvx path. The trust region already
    // bounds the step, and a proximal weight added on top would be graded by the
    // ratio test as if it were part of the objective — the merit would then
    // measure "how far did we move from the anchor" rather than "did the walls
    // hold". Both jobs at once also shrinks every step to a crawl.
    let mut h = pre.h_energy.clone();
    let mut f = vec![0.0; nvars];
    if scp_prox_weight > 0.0 && !scvx {
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
    if scvx {
        for i in 0..nvars {
            let mut row = vec![0.0; nvars];
            row[i] = 1.0;
            all_a_rows.extend_from_slice(&row);
            all_lb.push(p_current[i] - scp_trust_radius);
            all_ub.push(p_current[i] + scp_trust_radius);
            total_rows += 1;
        }
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
    let hard_sol = if scvx && elastic_available {
        None
    } else {
        solve_qp(&h, &f, &all_a_rows, &all_lb, &all_ub, nvars, total_rows)
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

        match solve_qp(&h_ext, &f_ext, &a_ext, &lb_ext, &ub_ext, nvars_ext, ext_nrows) {
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

    // Trust region. On the SCvx path the box was already imposed as rows above, so
    // the solution is inside it by construction and clipping would be a no-op that
    // silently masks a solver bug if it ever were not. Clip only on the legacy path.
    let mut x_result = x_new.clone();
    let raw_step_norm = (0..nvars)
        .map(|i| (x_result[i] - p_current[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    if !scvx && scp_trust_radius > 0.0 && raw_step_norm > scp_trust_radius && raw_step_norm > 1e-15
    {
        let alpha = scp_trust_radius / raw_step_norm;
        for i in 0..nvars {
            x_result[i] = p_current[i] + alpha * (x_result[i] - p_current[i]);
        }
    }

    // SCvx merit pieces. Both merits use the UNREGULARIZED objective (`pre.h_energy`,
    // which carries no linear term) so the ratio measures model error and not the
    // solver's own regularization. `vlin_p == vtrue_p` for this builder: each row's
    // support point is the exact closest point on the tube surface to its query
    // point and the normal is the unit vector along that offset, so `a·q - lb`
    // reproduces the exact clearance at the reference. Only the candidate needs
    // rebuilt walls, and the caller does that.
    let (l_p, l_c, vlin_p, vlin_c, hard_viol_p) = if scvx {
        let w_s = elastic_weight.max(0.0);
        let vp = row_violation(p_current, &all_a_rows, &all_lb, koz_row_start, n_koz, nvars);
        let vc = row_violation(&x_result, &all_a_rows, &all_lb, koz_row_start, n_koz, nvars);
        let qp_ = quadratic_cost(&pre.h_energy, p_current, nvars);
        let qc_ = quadratic_cost(&pre.h_energy, &x_result, nvars);
        let hv = hard_row_violation(
            p_current,
            &all_a_rows,
            &all_lb,
            &all_ub,
            total_rows,
            koz_row_start,
            n_koz,
            nvars,
        );
        (qp_ + w_s * vp, qc_ + w_s * vc, vp, vc, hv)
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };

    let delta = (0..nvars)
        .map(|i| (x_result[i] - p_current[i]).powi(2))
        .sum::<f64>()
        .sqrt();

    let clearance = compute_min_clearance(&x_result, np1, dim, obstacles, 1500);
    let cost = quadratic_cost(&pre.h_energy, &x_result, nvars);
    // Legacy convergence flag: "the step got small and no slack was used". That is a
    // statement about the step size, NOT an optimality claim — it is left exactly as
    // it was so the legacy path is bit-identical. On the SCvx path it is meaningless
    // (a small step can be a rejected one), so it is not set here: the outer loop
    // owns the convergence decision and records which criterion fired.
    let converged = !scvx && delta < tol && iter_total_slack < 1e-10;

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
        is_candidate: scvx,
        l_p,
        l_c,
        vlin_p,
        vlin_c,
        hard_viol_p,
    }
}

/// Why the outer loop stopped. Recorded in `info["stop_reason"]`.
///
/// The legacy loop reports every one of these as the same thing — it breaks out
/// and the caller sees an iteration count. That is the defect this enum exists to
/// remove: "hit the iteration cap" and "reached a stationary point" are opposite
/// outcomes and were indistinguishable.
///
/// Only `Stationary` and `MeritStreak` are claims of success. `IterationCap`,
/// `TrustCollapse` and `QpFailure` all mean the loop gave up.
mod stop_reason {
    /// Ran out of iterations. Not convergence.
    pub const ITERATION_CAP: f64 = 0.0;
    /// K consecutive accepted steps with negligible relative merit change, each
    /// carrying the hull certificate against its own rebuilt walls.
    pub const MERIT_STREAK: f64 = 1.0;
    /// Trust radius shrank below its floor — the ratio test stopped accepting
    /// anything. Counts as convergence only if the reference is certified.
    pub const TRUST_COLLAPSE: f64 = 2.0;
    /// The convex subproblem itself could not be solved.
    pub const QP_FAILURE: f64 = 3.0;
    /// K consecutive iterations where the model predicts no achievable progress.
    pub const STATIONARY: f64 = 4.0;
    /// Legacy path only: the step norm fell below `tol` with no slack in use. This
    /// says the iterate stopped moving, which — on a path that accepts every step
    /// unconditionally — is not an optimality claim and gets its own code so it can
    /// never be read as one.
    pub const LEGACY_SMALL_STEP: f64 = 5.0;
}

/// Full SCP optimization loop.
///
/// Two paths, selected by `use_scvx`:
///
/// * **legacy** (`use_scvx == false`, the default) — every step is accepted
///   unconditionally and the loop stops when the step norm drops below `tol`.
///   Unchanged, bit-for-bit.
/// * **SCvx** — each step is a candidate graded by a ratio test against walls
///   rebuilt at the candidate, the trust radius adapts, and the exit criterion is
///   recorded rather than implied.
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
    use_scvx: bool,
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

    // SCvx requires a positive trust radius to have a region to trust.
    let scvx = use_scvx && scp_trust_radius > 0.0;

    // ---- SCvx state (unused on the legacy path) ----
    let mut trust = scp_trust_radius;
    let trust_min = scp_trust_radius * 1e-3;
    let trust_max = scp_trust_radius * 4.0;
    let eta_accept = 0.1_f64;
    // The legacy `tol` is a step-norm threshold that can be absurdly small (the
    // Python entrypoint passes 1e-12), and under a moving linearization a step norm
    // that small may never occur. On the SCvx path it is reinterpreted as a
    // RELATIVE merit-change tolerance, with a floor so a legacy value cannot make
    // the test unsatisfiable.
    let tol_f = if tol > 0.0 { tol.max(1e-8) } else { 1e-8 };
    // A single quiet iteration is not evidence of stationarity when the walls
    // re-aim between iterations — that is exactly the premature stop this guards.
    let conv_streak_required = 3usize;
    let mut conv_streak = 0usize;
    let mut stat_streak = 0usize;
    let mut converged = false;
    let mut reason = stop_reason::ITERATION_CAP;
    let mut accept_count = 0usize;
    let mut reject_count = 0usize;
    let mut null_step_count = 0usize;
    let mut bootstrap_count = 0usize;
    let mut rho_sum = 0.0f64;
    let mut rho_n = 0usize;
    let mut rho_min = f64::INFINITY;
    let mut rho_max = f64::NEG_INFINITY;
    let mut last_rho = f64::NAN;
    let mut last_vtrue_c = f64::NAN;
    // Violation at the REFERENCE, i.e. at the iterate actually being carried. This
    // and `last_vtrue_c` answer different questions and must both be reported: a
    // rejected candidate can be certified while the reference the loop keeps is not,
    // and quoting only the candidate's number would claim a certificate for a point
    // that was thrown away.
    let mut last_vtrue_p = f64::NAN;
    // Per-iteration trace of EVERY step, accepted or rejected. Summary statistics
    // alone would hide the rejected steps that collapse the trust radius, which is
    // precisely the population this port exists to measure.
    let trace = std::env::var("SPACETIME_SCVX_TRACE").is_ok();
    if scvx && trace {
        eprintln!(
            "STTRACE,it,outcome,rho,pred,act,l_p,l_c,t_p,t_c,rel,trust_before,trust_after,\
step_norm,vlin_p,vlin_c,vtrue_c,hard_viol_p,clearance,total_slack,conv_streak,stat_streak"
        );
    }

    for it in 1..=max_iter {
        iterations = it;
        let step = scp_step(
            &p,
            &pre,
            obstacles,
            scp_prox_weight,
            trust,
            elastic_weight,
            tol,
            cap_bulge_ratio,
            it as u32,
            scvx,
        );

        if step.solver_status == "Failed" {
            reason = stop_reason::QP_FAILURE;
            break;
        }

        last_delta = step.delta;
        last_total_slack = step.total_slack;
        last_max_slack = step.max_slack;

        if !scvx {
            // ---- Legacy: accept unconditionally. ----
            p = step.p_new;
            if step.clearance > 0.0 && step.clearance > best_clearance {
                best_clearance = step.clearance;
                best_p = p.clone();
            }
            if step.converged {
                // "The step got small." `converged` stays false: this path accepts
                // every step without grading it, so it has never been in a position
                // to assert optimality and must not appear to.
                reason = stop_reason::LEGACY_SMALL_STEP;
                break;
            }
            continue;
        }

        // ---- SCvx: grade the candidate. ----
        //
        // Convex merit  L(x) = J(x) + w · (violation of the rows the QP was given)
        // True merit    T(x) = J(x) + w · (violation of rows REBUILT at x)
        //
        // J is exactly quadratic here — no linearization anywhere in the objective —
        // so `pred` and `act` can differ for exactly one reason: the supporting
        // half-spaces moved. rho is therefore a pure measurement of wall stability,
        // which is narrower and more interpretable than the orbital-docking version
        // where gravity-linearization error is mixed into the same number.
        let cand = step.p_new.clone();
        let w_s = elastic_weight.max(0.0);
        let (quad_p, scale_p) = quadratic_cost_scaled(&pre.h_energy, &p, nvars);
        let (quad_c, scale_c) = quadratic_cost_scaled(&pre.h_energy, &cand, nvars);
        let vtrue_p = step.vlin_p; // rows were built at p; exact there
        let vtrue_c = koz_violation_rebuilt_at(&cand, &pre, obstacles, cap_bulge_ratio);
        let t_p = quad_p + w_s * vtrue_p;
        let t_c = quad_c + w_s * vtrue_c;
        let pred = step.l_p - step.l_c;
        let act = t_p - t_c;

        // The magnitude every tolerance below is stated against.
        //
        // `|t_p|` alone is not usable: the quadratic form is a signed sum whose terms
        // cancel, and where the optimum has near-zero bending energy the merit is
        // cancellation residue rather than a value. `cancel_floor` is the standard
        // floating-point bound on that residue, and `merit_scale` is lifted to
        // whichever of the two is larger — so a relative tolerance is never applied to
        // a number that is entirely noise.
        let cancel_floor = f64::EPSILON * scale_p.max(scale_c);
        let merit_scale = t_p.abs().max(cancel_floor / 1e-9);
        // Resolution of the merit: the QP solver returns solutions accurate to roughly
        // 1e-9 relative, and both merits are evaluated at that solution.
        let merit_eps = 1e-9 * merit_scale;
        let rel = act.abs() / merit_scale;
        let trust_before = trust;
        last_vtrue_c = vtrue_c;
        last_vtrue_p = vtrue_p;

        // `pred >= 0` holds only if x = p is feasible for the elastic subproblem,
        // which needs the reference to satisfy every row that carries no slack. The
        // straight-line initial guess generally violates the box/monotonicity rows,
        // so iteration 1 is a repair step: the candidate does satisfy them, and
        // grading a repair with a merit comparison that is undefined would reject it
        // forever. Accept it and skip the convergence tests.
        // Emit one trace row. Built by joining a slice rather than by a format string
        // with 19 positional holes — the arity of those two must agree with the header
        // and with each other, and a miscount silently shifts every column.
        let emit = |outcome: &str, rho: f64, trust_after: f64, cs: usize, ss: usize| {
            let cols = [
                rho,
                pred,
                act,
                step.l_p,
                step.l_c,
                t_p,
                t_c,
                rel,
                trust_before,
                trust_after,
                step.delta,
                step.vlin_p,
                step.vlin_c,
                vtrue_c,
                step.hard_viol_p,
                step.clearance,
                step.total_slack,
            ];
            let nums: Vec<String> = cols.iter().map(|v| format!("{v:.9e}")).collect();
            eprintln!("STTRACE,{it},{outcome},{},{cs},{ss}", nums.join(","));
        };

        if step.hard_viol_p > 1e-9 {
            bootstrap_count += 1;
            if trace {
                emit("bootstrap", f64::NAN, trust, 0, 0);
            }
            p = cand;
            conv_streak = 0;
            stat_streak = 0;
            if step.clearance > 0.0 && step.clearance > best_clearance {
                best_clearance = step.clearance;
                best_p = p.clone();
            }
            continue;
        }

        // Model stationarity: the subproblem itself predicts no achievable progress
        // from this reference. Compared on |pred|, not pred — near a stationary point
        // pred is a difference of two merits that agree to near machine precision, so
        // its SIGN is noise and a `pred >= 0` guard would reset the streak forever.
        // Guarded on the reference being certified so a stationary-but-penetrating
        // point cannot report success.
        // Measured motivation for `merit_scale`: with no obstacles the straight-line
        // optimum has exactly zero bending energy, so every merit is ~1e-12 of
        // cancellation residue. Against `|t_p|` the floor was ~1e-21, and 13 of 16
        // steps were rejected on a problem whose FIRST step already landed on the
        // exact optimum — with `pred` and `act` agreeing to all 17 digits. The model
        // was perfect and the loop rejected it anyway. Caught by invariant A.
        let pred_floor = merit_eps;
        if pred.abs() < tol_f * merit_scale && vtrue_p <= 1e-6 {
            stat_streak += 1;
            if stat_streak >= conv_streak_required {
                converged = true;
                reason = stop_reason::STATIONARY;
                break;
            }
        } else {
            stat_streak = 0;
        }

        let rho = if pred < pred_floor {
            null_step_count += 1;
            // The model sees no improvement to predict. Treat as a null step: accept
            // if it does not make things worse, so the tests above can fire. Never
            // divide by a non-positive prediction.
            if act >= -merit_eps {
                f64::INFINITY
            } else {
                -1.0
            }
        } else {
            act / pred
        };
        last_rho = rho;
        if rho.is_finite() {
            rho_sum += rho;
            rho_n += 1;
            rho_min = rho_min.min(rho);
            rho_max = rho_max.max(rho);
        }

        let outcome;
        if rho > eta_accept {
            accept_count += 1;
            outcome = if pred < pred_floor { "accept_null" } else { "accept" };
            p = cand;
            if rho > 0.9 && rho.is_finite() {
                trust = (trust * 2.0).min(trust_max); // model held — be bolder
            }
            if step.clearance > 0.0 && step.clearance > best_clearance {
                best_clearance = step.clearance;
                best_p = p.clone();
            }
            // Converged = merit stationary for K consecutive accepted steps AND the
            // accepted iterate carries the hull certificate against its OWN rebuilt
            // walls. The second half is what stops a penetrating iterate from being
            // reported as a success — the failure mode the paper cannot afford.
            if rel < tol_f && vtrue_c <= 1e-6 {
                conv_streak += 1;
                if conv_streak >= conv_streak_required {
                    converged = true;
                    reason = stop_reason::MERIT_STREAK;
                    if trace {
                        eprintln!("STTRACE,{it},{outcome},converged_merit_streak");
                    }
                    break;
                }
            } else {
                conv_streak = 0;
            }
        } else {
            reject_count += 1;
            conv_streak = 0;
            outcome = if pred < pred_floor { "reject_null" } else { "reject" };
            trust *= 0.5;
        }

        if trace {
            emit(outcome, rho, trust, conv_streak, stat_streak);
        }

        if trust < trust_min {
            // The ratio test stopped accepting anything. That is convergence to a
            // constrained local optimum only if the reference is actually certified;
            // otherwise it is a genuine failure and must not be dressed up as one.
            converged = vtrue_p <= 1e-6 && step.hard_viol_p <= 1e-9;
            reason = stop_reason::TRUST_COLLAPSE;
            break;
        }
    }

    let mut final_clearance = compute_min_clearance(&p, np1, dim, obstacles, 1500);
    // Fall back to the best feasible iterate seen. This is a DIFFERENT point from the
    // one the loop converged on, and callers that quote "converged" alongside this
    // trajectory are quoting two different iterates — hence `returned_best_iterate`.
    let mut returned_best = false;
    if final_clearance < 0.0 && best_clearance > 0.0 {
        p = best_p;
        final_clearance = best_clearance;
        returned_best = true;
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
    info.insert("scvx".to_string(), if scvx { 1.0 } else { 0.0 });
    info.insert("stop_reason".to_string(), reason);
    info.insert("converged".to_string(), if converged { 1.0 } else { 0.0 });
    info.insert(
        "returned_best_iterate".to_string(),
        if returned_best { 1.0 } else { 0.0 },
    );

    if scvx {
        info.insert("scvx_accept_count".to_string(), accept_count as f64);
        info.insert("scvx_reject_count".to_string(), reject_count as f64);
        info.insert("scvx_null_step_count".to_string(), null_step_count as f64);
        info.insert("scvx_bootstrap_count".to_string(), bootstrap_count as f64);
        info.insert("scvx_final_trust".to_string(), trust);
        info.insert("scvx_rho_last".to_string(), last_rho);
        info.insert("scvx_rho_samples".to_string(), rho_n as f64);
        info.insert(
            "scvx_rho_mean".to_string(),
            if rho_n > 0 { rho_sum / rho_n as f64 } else { f64::NAN },
        );
        info.insert(
            "scvx_rho_min".to_string(),
            if rho_n > 0 { rho_min } else { f64::NAN },
        );
        info.insert(
            "scvx_rho_max".to_string(),
            if rho_n > 0 { rho_max } else { f64::NAN },
        );
        // Violation of the walls rebuilt at the last graded candidate. This is the
        // hull certificate: > 0 means the returned curve does not satisfy the
        // half-spaces its own control points generate.
        info.insert("scvx_koz_violation_final".to_string(), last_vtrue_c);
        info.insert("scvx_koz_violation_reference".to_string(), last_vtrue_p);
    }

    OptResult {
        p_opt: p,
        np1,
        dim,
        info,
        feasible,
        iterations,
    }
}
