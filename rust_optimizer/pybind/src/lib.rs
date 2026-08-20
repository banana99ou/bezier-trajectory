use bezier_opt_core::{optimizer, spacetime_constraints::SpacetimeObstacleData, spacetime_optimizer};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (
    p_init,
    n_seg = 8,
    r_e = None,
    max_iter = 20,
    tol = 1e-6,
    v0 = None,
    v1 = None,
    a0 = None,
    a1 = None,
    sample_count = 100,
    objective_mode = "energy",
    dv_irls_eps = 1e-9,
    dv_geom_reg = 0.0,
    scp_prox_weight = 0.0,
    scp_trust_radius = 0.0,
    enforce_prograde = false,
    prograde_n_samples = 16,
))]
fn optimize_orbital_docking<'py>(
    py: Python<'py>,
    p_init: PyReadonlyArray2<'py, f64>,
    n_seg: usize,
    r_e: Option<f64>,
    max_iter: usize,
    tol: f64,
    v0: Option<PyReadonlyArray1<'py, f64>>,
    v1: Option<PyReadonlyArray1<'py, f64>>,
    a0: Option<PyReadonlyArray1<'py, f64>>,
    a1: Option<PyReadonlyArray1<'py, f64>>,
    sample_count: usize,
    objective_mode: &str,
    dv_irls_eps: f64,
    dv_geom_reg: f64,
    scp_prox_weight: f64,
    scp_trust_radius: f64,
    enforce_prograde: bool,
    prograde_n_samples: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let p_arr = p_init.as_array();
    let np1 = p_arr.shape()[0];
    let dim = p_arr.shape()[1];

    let p_flat: Vec<f64> = p_arr.iter().copied().collect();
    let r_e_val = r_e.unwrap_or(6471.0);

    let v0_vec: Option<Vec<f64>> = v0.map(|a| a.as_array().iter().copied().collect());
    let v1_vec: Option<Vec<f64>> = v1.map(|a| a.as_array().iter().copied().collect());
    let a0_vec: Option<Vec<f64>> = a0.map(|a| a.as_array().iter().copied().collect());
    let a1_vec: Option<Vec<f64>> = a1.map(|a| a.as_array().iter().copied().collect());

    let result = optimizer::optimize_orbital_docking(
        &p_flat,
        np1,
        dim,
        n_seg,
        r_e_val,
        max_iter,
        tol,
        1500.0,
        sample_count,
        objective_mode,
        dv_irls_eps,
        dv_geom_reg,
        scp_prox_weight,
        scp_trust_radius,
        v0_vec.as_deref(),
        v1_vec.as_deref(),
        a0_vec.as_deref(),
        a1_vec.as_deref(),
        enforce_prograde,
        prograde_n_samples,
    );

    let p_opt = PyArray2::from_vec2(py, &{
        let mut rows = Vec::with_capacity(np1);
        for i in 0..np1 {
            rows.push(result.p_opt[i * dim..(i + 1) * dim].to_vec());
        }
        rows
    }).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

    let info = PyDict::new(py);
    for (k, v) in &result.info {
        info.set_item(k, *v)?;
    }
    info.set_item("feasible", result.feasible)?;
    info.set_item("iterations", result.iterations)?;

    Ok((p_opt, info))
}

#[pyfunction]
#[pyo3(signature = (
    p_init,
    obstacle_pos0,
    obstacle_vel,
    obstacle_r,
    obstacle_t_start = None,
    obstacle_t_end = None,
    n_seg = 8,
    max_iter = 30,
    tol = 1e-6,
    scp_prox_weight = 0.5,
    scp_trust_radius = 0.0,
    min_dt = 0.1,
    coord_lb = -20.0,
    coord_ub = 20.0,
    time_lb = 0.0,
    time_ub = 15.0,
    elastic_weight = 100.0,
    cap_bulge_ratio = 2.0,
    v_max = None,
    time_weight = 0.0,
    free_arrival_time = false,
))]
fn optimize_spacetime_bezier<'py>(
    py: Python<'py>,
    p_init: PyReadonlyArray2<'py, f64>,
    obstacle_pos0: PyReadonlyArray2<'py, f64>,
    obstacle_vel: PyReadonlyArray2<'py, f64>,
    obstacle_r: PyReadonlyArray1<'py, f64>,
    obstacle_t_start: Option<PyReadonlyArray1<'py, f64>>,
    obstacle_t_end: Option<PyReadonlyArray1<'py, f64>>,
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
    elastic_weight: f64,
    cap_bulge_ratio: f64,
    // Slant-limit speed cap (item B9). `None` means no cap, which is the
    // default, so every pre-existing scenario reproduces exactly.
    v_max: Option<f64>,
    // Linear arrival-time penalty (item B10).
    time_weight: f64,
    free_arrival_time: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let p_arr = p_init.as_array();
    let np1 = p_arr.shape()[0];
    let dim = p_arr.shape()[1];
    if dim < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Space-time Bezier requires dim >= 2",
        ));
    }
    let spatial_dim = dim - 1;
    let p_flat: Vec<f64> = p_arr.iter().copied().collect();

    let pos0_arr = obstacle_pos0.as_array();
    let vel_arr = obstacle_vel.as_array();
    if pos0_arr.shape().len() != 2 || vel_arr.shape().len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Obstacle position and velocity arrays must be 2D",
        ));
    }
    if pos0_arr.shape()[1] != spatial_dim || vel_arr.shape()[1] != spatial_dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Obstacle arrays must have spatial_dim={}, got pos0={:?}, vel={:?}",
            spatial_dim,
            pos0_arr.shape(),
            vel_arr.shape()
        )));
    }
    if pos0_arr.shape() != vel_arr.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Obstacle position and velocity arrays must have matching shape",
        ));
    }

    let n_obs = pos0_arr.shape()[0];
    let radii_vec: Vec<f64> = obstacle_r.as_array().iter().copied().collect();
    if radii_vec.len() != n_obs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} obstacle radii, got {}",
            n_obs,
            radii_vec.len()
        )));
    }

    let pos0_vec: Vec<f64> = pos0_arr.iter().copied().collect();
    let vel_vec: Vec<f64> = vel_arr.iter().copied().collect();
    let t_start_vec: Vec<f64> = if let Some(arr) = obstacle_t_start {
        let vals: Vec<f64> = arr.as_array().iter().copied().collect();
        if vals.len() != n_obs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} obstacle start times, got {}",
                n_obs,
                vals.len()
            )));
        }
        vals
    } else {
        vec![f64::NEG_INFINITY; n_obs]
    };
    let t_end_vec: Vec<f64> = if let Some(arr) = obstacle_t_end {
        let vals: Vec<f64> = arr.as_array().iter().copied().collect();
        if vals.len() != n_obs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} obstacle end times, got {}",
                n_obs,
                vals.len()
            )));
        }
        vals
    } else {
        vec![f64::INFINITY; n_obs]
    };

    let obstacles = SpacetimeObstacleData {
        pos0: &pos0_vec,
        vel: &vel_vec,
        radii: &radii_vec,
        t_start: &t_start_vec,
        t_end: &t_end_vec,
        n_obs,
        spatial_dim,
    };

    let result = spacetime_optimizer::optimize_spacetime(
        &p_flat,
        np1,
        dim,
        n_seg,
        max_iter,
        tol,
        scp_prox_weight,
        scp_trust_radius,
        min_dt,
        coord_lb,
        coord_ub,
        time_lb,
        time_ub,
        &obstacles,
        elastic_weight,
        cap_bulge_ratio,
        v_max.unwrap_or(f64::NAN),
        time_weight,
        free_arrival_time,
    );

    let p_opt = PyArray2::from_vec2(py, &{
        let mut rows = Vec::with_capacity(np1);
        for i in 0..np1 {
            rows.push(result.p_opt[i * dim..(i + 1) * dim].to_vec());
        }
        rows
    })
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

    let info = PyDict::new(py);
    for (k, v) in &result.info {
        info.set_item(k, *v)?;
    }
    info.set_item("feasible", result.feasible)?;
    info.set_item("iterations", result.iterations)?;

    Ok((p_opt, info))
}

/// The EXACT obstacle half-spaces at a given control polygon.
///
/// These are the rows the certificate is evaluated with -- one plane per
/// (segment, obstacle), normal frozen at the segment centroid, shared by every
/// control point of that segment. Exposed so tests can assert the two properties
/// that make the guarantee work, neither of which holds for the self-consistent
/// rows the QP is given (those carry a rotation term and are explicitly not
/// conservative).
///
/// Returns (normals, lower_bounds, segment_idx, cp_idx, obstacle_idx).
#[pyfunction]
#[pyo3(signature = (
    p, obstacle_pos0, obstacle_vel, obstacle_r,
    obstacle_t_start = None, obstacle_t_end = None, n_seg = 8,
))]
fn spacetime_koz_rows_exact<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    obstacle_pos0: PyReadonlyArray2<'py, f64>,
    obstacle_vel: PyReadonlyArray2<'py, f64>,
    obstacle_r: PyReadonlyArray1<'py, f64>,
    obstacle_t_start: Option<PyReadonlyArray1<'py, f64>>,
    obstacle_t_end: Option<PyReadonlyArray1<'py, f64>>,
    n_seg: usize,
) -> PyResult<PyObject> {
    let p_arr = p.as_array();
    let np1 = p_arr.shape()[0];
    let dim = p_arr.shape()[1];
    let spatial_dim = dim - 1;
    let p_flat: Vec<f64> = p_arr.iter().copied().collect();

    let n_obs = obstacle_pos0.as_array().shape()[0];
    let pos0: Vec<f64> = obstacle_pos0.as_array().iter().copied().collect();
    let vel: Vec<f64> = obstacle_vel.as_array().iter().copied().collect();
    let radii: Vec<f64> = obstacle_r.as_array().iter().copied().collect();
    let t_start: Vec<f64> = obstacle_t_start
        .map(|a| a.as_array().iter().copied().collect())
        .unwrap_or_else(|| vec![f64::NEG_INFINITY; n_obs]);
    let t_end: Vec<f64> = obstacle_t_end
        .map(|a| a.as_array().iter().copied().collect())
        .unwrap_or_else(|| vec![f64::INFINITY; n_obs]);

    let obstacles = SpacetimeObstacleData {
        pos0: &pos0,
        vel: &vel,
        radii: &radii,
        t_start: &t_start,
        t_end: &t_end,
        n_obs,
        spatial_dim,
    };
    let a_list = bezier_opt_core::de_casteljau::segment_matrices_equal_params(np1 - 1, n_seg);

    let bundle = bezier_opt_core::spacetime_constraints::build_spacetime_koz_constraints(
        &a_list, &p_flat, np1, dim, &obstacles, 2.0,
    );
    let rows = match bundle {
        Some(b) => b.rows,
        None => Vec::new(),
    };

    let normals: Vec<Vec<f64>> = rows.iter().map(|r| r.normal.clone()).collect();
    let lbs: Vec<f64> = rows.iter().map(|r| r.lower_bound).collect();
    let seg: Vec<i32> = rows.iter().map(|r| r.segment_idx as i32).collect();
    let cp: Vec<i32> = rows.iter().map(|r| r.cp_idx as i32).collect();
    let obs: Vec<i32> = rows.iter().map(|r| r.obstacle_idx as i32).collect();

    let normals_arr = if normals.is_empty() {
        PyArray2::from_vec2(py, &vec![vec![0.0; dim]; 0])
    } else {
        PyArray2::from_vec2(py, &normals)
    }
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

    Ok((
        normals_arr,
        PyArray1::from_vec(py, lbs),
        PyArray1::from_vec(py, seg),
        PyArray1::from_vec(py, cp),
        PyArray1::from_vec(py, obs),
    )
        .into_pyobject(py)?
        .into())
}

/// Opaque handle holding precomputed SCP data and obstacle arrays.
/// Exposes a `step()` method that runs one SCP iteration.
#[pyclass]
struct SpacetimeScpContext {
    precomputed: spacetime_optimizer::ScpPrecomputed,
    pos0: Vec<f64>,
    vel: Vec<f64>,
    radii: Vec<f64>,
    t_start: Vec<f64>,
    t_end: Vec<f64>,
    n_obs: usize,
    spatial_dim: usize,
    elastic_weight: f64,
    tol: f64,
    cap_bulge_ratio: f64,
    /// Trust radius and streak counters live here, not in Python. A stepper that
    /// re-derived them per call would be running a different algorithm from the
    /// batch loop.
    state: spacetime_optimizer::ScpState,
}

#[pymethods]
impl SpacetimeScpContext {
    #[new]
    #[pyo3(signature = (
        p_init,
        obstacle_pos0,
        obstacle_vel,
        obstacle_r,
        obstacle_t_start = None,
        obstacle_t_end = None,
        n_seg = 8,
        min_dt = 0.1,
        coord_lb = -20.0,
        coord_ub = 20.0,
        time_lb = 0.0,
        time_ub = 15.0,
        scp_prox_weight = 0.5,
        scp_trust_radius = 0.0,
        elastic_weight = 100.0,
        tol = 1e-6,
        cap_bulge_ratio = 2.0,
        v_max = None,
        time_weight = 0.0,
        free_arrival_time = false,
    ))]
    fn new(
        p_init: PyReadonlyArray2<'_, f64>,
        obstacle_pos0: PyReadonlyArray2<'_, f64>,
        obstacle_vel: PyReadonlyArray2<'_, f64>,
        obstacle_r: PyReadonlyArray1<'_, f64>,
        obstacle_t_start: Option<PyReadonlyArray1<'_, f64>>,
        obstacle_t_end: Option<PyReadonlyArray1<'_, f64>>,
        n_seg: usize,
        min_dt: f64,
        coord_lb: f64,
        coord_ub: f64,
        time_lb: f64,
        time_ub: f64,
        scp_prox_weight: f64,
        scp_trust_radius: f64,
        elastic_weight: f64,
        tol: f64,
        cap_bulge_ratio: f64,
        v_max: Option<f64>,
        time_weight: f64,
        free_arrival_time: bool,
    ) -> PyResult<Self> {
        let p_arr = p_init.as_array();
        let np1 = p_arr.shape()[0];
        let dim = p_arr.shape()[1];
        if dim < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("dim >= 2 required"));
        }
        let spatial_dim = dim - 1;
        let p_flat: Vec<f64> = p_arr.iter().copied().collect();

        let n_obs = obstacle_pos0.as_array().shape()[0];
        let pos0: Vec<f64> = obstacle_pos0.as_array().iter().copied().collect();
        let vel: Vec<f64> = obstacle_vel.as_array().iter().copied().collect();
        let radii: Vec<f64> = obstacle_r.as_array().iter().copied().collect();
        let t_start = obstacle_t_start
            .map(|a| a.as_array().iter().copied().collect())
            .unwrap_or_else(|| vec![f64::NEG_INFINITY; n_obs]);
        let t_end = obstacle_t_end
            .map(|a| a.as_array().iter().copied().collect())
            .unwrap_or_else(|| vec![f64::INFINITY; n_obs]);

        let pre = spacetime_optimizer::precompute_scp(
            &p_flat, np1, dim, n_seg, min_dt, coord_lb, coord_ub, time_lb, time_ub,
            v_max.unwrap_or(f64::NAN), time_weight, free_arrival_time,
        );

        let _ = scp_prox_weight; // the trust region does this job; see scp_step
        let state = spacetime_optimizer::ScpState::new(&p_flat, scp_trust_radius);

        Ok(Self {
            precomputed: pre,
            pos0,
            vel,
            radii,
            t_start,
            t_end,
            n_obs,
            spatial_dim,
            elastic_weight,
            tol,
            cap_bulge_ratio,
            state,
        })
    }

    /// The best feasible iterate seen so far, as (np1, dim).
    fn best_control_points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let np1 = self.precomputed.np1;
        let dim = self.precomputed.dim;
        let mut rows = Vec::with_capacity(np1);
        for i in 0..np1 {
            rows.push(self.state.best_p[i * dim..(i + 1) * dim].to_vec());
        }
        PyArray2::from_vec2(py, &rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Run one canonical SCP iteration and return what happened.
    ///
    /// Takes no control points: the iterate lives in the context, because the
    /// accept/reject decision is what determines it. A caller that passed its own
    /// point in would be able to advance along a path the solver never chose.
    ///
    /// Returns (p_iterate, info, koz_* arrays) where `p_iterate` is the ACCEPTED
    /// iterate after the decision -- unchanged when the step was rejected. The raw
    /// candidate is in `info["p_candidate"]`.
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        let obstacles = SpacetimeObstacleData {
            pos0: &self.pos0,
            vel: &self.vel,
            radii: &self.radii,
            t_start: &self.t_start,
            t_end: &self.t_end,
            n_obs: self.n_obs,
            spatial_dim: self.spatial_dim,
        };

        let iter_out = spacetime_optimizer::scp_iterate(
            &mut self.state,
            &self.precomputed,
            &obstacles,
            self.elastic_weight,
            self.tol,
            self.cap_bulge_ratio,
        );
        let outcome = iter_out.outcome;
        let accepted = iter_out.accepted;
        let rho = iter_out.rho;
        let pred = iter_out.pred;
        let act = iter_out.act;
        let vtrue_c = iter_out.vtrue_c;
        let trust_before = iter_out.trust_before;
        let trust_after = iter_out.trust_after;
        let result = iter_out.step;

        let np1 = self.precomputed.np1;
        let dim = self.precomputed.dim;

        let as_rows = |flat: &[f64]| {
            let mut rows = Vec::with_capacity(np1);
            for i in 0..np1 {
                rows.push(flat[i * dim..(i + 1) * dim].to_vec());
            }
            rows
        };
        // The ACCEPTED iterate. On a rejected step this is the previous point --
        // which is the whole reason the two are returned separately.
        let p_new = PyArray2::from_vec2(py, &as_rows(&self.state.p))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let p_candidate = PyArray2::from_vec2(py, &as_rows(&result.p_new))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

        // Pack info dict
        let info = PyDict::new(py);
        info.set_item("solver_status", &result.solver_status)?;
        info.set_item("delta", result.delta)?;
        info.set_item("raw_step_norm", result.raw_step_norm)?;
        info.set_item("clearance", result.clearance)?;
        info.set_item("total_slack", result.total_slack)?;
        info.set_item("max_slack", result.max_slack)?;
        info.set_item("converged", result.converged)?;
        info.set_item("cost", result.cost)?;
        info.set_item("koz_row_count", result.koz_rows.len())?;
        // The accept/reject decision. `converged` above is always false -- a single
        // subproblem never decides that; the state does.
        info.set_item("p_candidate", p_candidate)?;
        info.set_item("outcome", outcome)?;
        info.set_item("accepted", accepted)?;
        info.set_item("rho", rho)?;
        info.set_item("pred_reduction", pred)?;
        info.set_item("actual_reduction", act)?;
        info.set_item("koz_violation_candidate", vtrue_c)?;
        info.set_item("koz_violation_reference", result.vlin_p)?;
        info.set_item("trust_before", trust_before)?;
        info.set_item("trust_after", trust_after)?;
        info.set_item("iteration", self.state.iteration)?;
        info.set_item("state_converged", self.state.converged)?;
        info.set_item("stop_reason", self.state.stop)?;
        info.set_item("running", self.state.running())?;
        info.set_item("accept_count", self.state.accept_count)?;
        info.set_item("reject_count", self.state.reject_count)?;
        // Best FEASIBLE iterate seen, tracked by the solver. Exposed so a stepping
        // caller does not have to re-derive it and risk tracking a different point
        // from the batch loop.
        info.set_item("best_clearance", self.state.best_clearance)?;

        // Pack per-row KOZ data as parallel flat arrays
        let n_koz = result.koz_rows.len();
        let seg_idx: Vec<i32> = result.koz_rows.iter().map(|r| r.segment_idx as i32).collect();
        let cp_idx: Vec<i32> = result.koz_rows.iter().map(|r| r.cp_idx as i32).collect();
        let obs_idx: Vec<i32> = result.koz_rows.iter().map(|r| r.obstacle_idx as i32).collect();
        let iter_idx: Vec<i32> = result.koz_rows.iter().map(|r| r.iteration as i32).collect();

        let mut normals_flat = Vec::with_capacity(n_koz * dim);
        let mut support_flat = Vec::with_capacity(n_koz * dim);
        let mut closest_flat = Vec::with_capacity(n_koz * dim);
        let mut lbs = Vec::with_capacity(n_koz);
        let mut margins = Vec::with_capacity(n_koz);

        for row in &result.koz_rows {
            normals_flat.extend_from_slice(&row.normal);
            support_flat.extend_from_slice(&row.support_point);
            closest_flat.extend_from_slice(&row.closest_center);
            lbs.push(row.lower_bound);
            margins.push(row.margin);
        }

        let koz_seg = PyArray1::from_vec(py, seg_idx);
        let koz_cp = PyArray1::from_vec(py, cp_idx);
        let koz_obs = PyArray1::from_vec(py, obs_idx);
        let koz_iter = PyArray1::from_vec(py, iter_idx);
        let koz_normals = PyArray2::from_vec2(py, &{
            result.koz_rows.iter().map(|r| r.normal.clone()).collect::<Vec<_>>()
        }).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let koz_supports = PyArray2::from_vec2(py, &{
            result.koz_rows.iter().map(|r| r.support_point.clone()).collect::<Vec<_>>()
        }).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let koz_centers = PyArray2::from_vec2(py, &{
            result.koz_rows.iter().map(|r| r.closest_center.clone()).collect::<Vec<_>>()
        }).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let koz_lbs = PyArray1::from_vec(py, lbs);
        let koz_margins = PyArray1::from_vec(py, margins);
        let koz_slack = PyArray1::from_vec(py, result.koz_slack_per_row);

        Ok((
            p_new, info,
            koz_seg, koz_cp, koz_obs, koz_iter,
            koz_normals, koz_supports, koz_centers,
            koz_lbs, koz_margins, koz_slack,
        ).into_pyobject(py)?.into_any().unbind())
    }
}

#[pymodule]
fn bezier_opt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_orbital_docking, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_spacetime_bezier, m)?)?;
    m.add_function(wrap_pyfunction!(spacetime_koz_rows_exact, m)?)?;
    m.add_class::<SpacetimeScpContext>()?;
    Ok(())
}
