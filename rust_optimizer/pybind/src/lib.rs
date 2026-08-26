use bezier_opt_core::{
    optimizer,
    spacetime_constraints::{SpacetimeObstacleData, StationData},
    spacetime_optimizer,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
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

/// Flatten an optional `(n_stations, spatial_dim)` array, rejecting a width that
/// does not match the problem's spatial dimension. A silently-transposed or
/// short station array would place the observer somewhere nobody asked for, and
/// every occlusion row would then certify the wrong geometry.
fn station_arrays(
    stations: Option<PyReadonlyArray2<'_, f64>>,
    spatial_dim: usize,
) -> PyResult<(Vec<f64>, usize)> {
    match stations {
        None => Ok((Vec::new(), 0)),
        Some(arr) => {
            let a = arr.as_array();
            if a.shape()[0] > 0 && a.shape()[1] != spatial_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Station array must have spatial_dim={}, got {:?}",
                    spatial_dim,
                    a.shape()
                )));
            }
            let n = a.shape()[0];
            Ok((a.iter().copied().collect(), n))
        }
    }
}


/// Read the obstacle wire format: lifted Bezier control points.
///
/// `obstacle_ctrl` is `(n_obs, n_ctrl, dim)` where the last coordinate of each
/// control point is TIME. There is no separate `t_start` / `t_end`: the active
/// window is the first and last control point's time coordinate, so the two can
/// never disagree. Obstacles of differing degree are degree-elevated on the
/// Python side before they get here, which is exact.
fn obstacle_arrays<'py>(
    obstacle_ctrl: PyReadonlyArray3<'py, f64>,
    obstacle_r: PyReadonlyArray1<'py, f64>,
    dim: usize,
) -> PyResult<(Vec<f64>, usize, Vec<f64>, usize)> {
    let arr = obstacle_ctrl.as_array();
    if arr.shape().len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obstacle_ctrl must be 3-D: (n_obs, n_ctrl, spatial_dim + 1)",
        ));
    }
    let (n_obs, n_ctrl, ctrl_dim) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    if n_obs > 0 && ctrl_dim != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "obstacle_ctrl last axis must be {dim} (spatial_dim + 1), got {ctrl_dim}"
        )));
    }
    if n_obs > 0 && n_ctrl < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "each obstacle needs at least 2 control points",
        ));
    }
    let radii: Vec<f64> = obstacle_r.as_array().iter().copied().collect();
    if radii.len() != n_obs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} obstacle radii, got {}",
            n_obs,
            radii.len()
        )));
    }
    // Time must not run backwards along an obstacle: `param_at_time` inverts the
    // time coordinate affinely and a reversed window makes that inversion
    // meaningless rather than merely wrong.
    for m in 0..n_obs {
        let t0 = arr[[m, 0, dim - 1]];
        let t1 = arr[[m, n_ctrl - 1, dim - 1]];
        if !(t1 >= t0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "obstacle {m}: control point times run backwards ({t0} to {t1})"
            )));
        }
    }
    Ok((
        arr.iter().copied().collect(),
        n_ctrl.max(2usize),
        radii,
        n_obs,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    p_init,
    obstacle_ctrl,
    obstacle_r,
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
    sound_clip = false,
    v_max = None,
    time_weight = 0.0,
    free_arrival_time = false,
    stations = None,
))]
fn optimize_spacetime_bezier<'py>(
    py: Python<'py>,
    p_init: PyReadonlyArray2<'py, f64>,
    obstacle_ctrl: PyReadonlyArray3<'py, f64>,
    obstacle_r: PyReadonlyArray1<'py, f64>,
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
    // PAPER_1 statement (8): clamp the clip radius from below so statement (7)
    // holds unconditionally and the construction is sound by construction, at the
    // cost of conservatism where the row binds. Off by default; PAPER_1 calls the
    // choice an open experimental question.
    sound_clip: bool,
    // Slant-limit speed cap (item B9). `None` means no cap, which is the
    // default, so every pre-existing scenario reproduces exactly.
    v_max: Option<f64>,
    // Linear arrival-time penalty (item B10).
    time_weight: f64,
    free_arrival_time: bool,
    // Fixed stations the vehicle must keep line of sight to (item B12). `None`
    // means no occlusion rows at all, which is the default, so every scenario
    // that predates B12 solves the identical problem.
    stations: Option<PyReadonlyArray2<'py, f64>>,
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

    let (ctrl_vec, n_ctrl, radii_vec, n_obs) = obstacle_arrays(obstacle_ctrl, obstacle_r, dim)?;

    let obstacles = SpacetimeObstacleData {
        ctrl: &ctrl_vec,
        n_ctrl,
        radii: &radii_vec,
        n_obs,
        spatial_dim,
    };

    let (station_vec, n_stations) = station_arrays(stations, spatial_dim)?;
    let station_data = StationData {
        pos: &station_vec,
        n_stations,
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
        &station_data,
        elastic_weight,
        sound_clip,
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
/// Returns (normals, lower_bounds, segment_idx, cp_idx, obstacle_idx,
/// component_idx, rho, sound, dropped_planes, unsound_clips). The last three are
/// the holes: `sound` is per row (statement 7), `dropped_planes` counts
/// components needing a row that admits none, `unsound_clips` counts pairs where
/// statement (7) failed.
///
/// **The grouping key is (segment, obstacle, component_idx), not (segment,
/// obstacle).** One obstacle can present two separated lumps of tube to one
/// segment, and each lump gets its own plane; grouping without the component
/// index mixes two different walls together.
#[pyfunction]
#[pyo3(signature = (
    p, obstacle_ctrl, obstacle_r, n_seg = 8, trust_radius = 0.5, sound_clip = false,
))]
fn spacetime_koz_rows_exact<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    obstacle_ctrl: PyReadonlyArray3<'py, f64>,
    obstacle_r: PyReadonlyArray1<'py, f64>,
    n_seg: usize,
    trust_radius: f64,
    sound_clip: bool,
) -> PyResult<PyObject> {
    let p_arr = p.as_array();
    let np1 = p_arr.shape()[0];
    let dim = p_arr.shape()[1];
    let spatial_dim = dim - 1;
    let p_flat: Vec<f64> = p_arr.iter().copied().collect();

    let (ctrl, n_ctrl, radii, n_obs) = obstacle_arrays(obstacle_ctrl, obstacle_r, dim)?;

    let obstacles = SpacetimeObstacleData {
        ctrl: &ctrl,
        n_ctrl,
        radii: &radii,
        n_obs,
        spatial_dim,
    };
    let a_list = bezier_opt_core::de_casteljau::segment_matrices_equal_params(np1 - 1, n_seg);

    let bundle = bezier_opt_core::spacetime_constraints::build_spacetime_koz_constraints(
        &a_list, &p_flat, np1, dim, &obstacles, trust_radius, sound_clip,
    );
    let (rows, dropped, unsound) = match bundle {
        Some(b) => (b.rows, b.dropped_planes, b.unsound_clips),
        None => (Vec::new(), 0usize, 0usize),
    };

    let normals: Vec<Vec<f64>> = rows.iter().map(|r| r.normal.clone()).collect();
    let lbs: Vec<f64> = rows.iter().map(|r| r.lower_bound).collect();
    let seg: Vec<i32> = rows.iter().map(|r| r.segment_idx as i32).collect();
    let cp: Vec<i32> = rows.iter().map(|r| r.cp_idx as i32).collect();
    let obs: Vec<i32> = rows.iter().map(|r| r.obstacle_idx as i32).collect();
    let comp: Vec<i32> = rows.iter().map(|r| r.component_idx as i32).collect();
    let rho: Vec<f64> = rows.iter().map(|r| r.rho).collect();
    let sound: Vec<bool> = rows.iter().map(|r| r.sound).collect();

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
        PyArray1::from_vec(py, comp),
        PyArray1::from_vec(py, rho),
        sound,
        dropped,
        unsound,
    )
        .into_pyobject(py)?
        .into())
}

/// The EXACT occlusion half-spaces at a given control polygon (item B12).
///
/// One plane per (segment, occluder piece, station), shared by every control
/// point of that segment -- the same rows the certificate is evaluated with.
/// Exposed so a test can check the two properties the guarantee rests on: that
/// the piece body CONTAINS the true occluder across the window it was built for,
/// and that the half-space contains the shadow that body casts.
///
/// Returns (normals, lower_bounds, segment_idx, cp_idx, obstacle_idx,
/// station_idx, body_centers, body_radii, window_lo, window_hi, margins).
#[pyfunction]
#[pyo3(signature = (
    p, obstacle_ctrl, obstacle_r, stations, n_seg = 8,
    trust_radius = 0.5, sound_clip = false,
))]
#[allow(clippy::too_many_arguments)]
fn spacetime_occlusion_rows_exact<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    obstacle_ctrl: PyReadonlyArray3<'py, f64>,
    obstacle_r: PyReadonlyArray1<'py, f64>,
    stations: PyReadonlyArray2<'py, f64>,
    n_seg: usize,
    trust_radius: f64,
    sound_clip: bool,
) -> PyResult<PyObject> {
    let p_arr = p.as_array();
    let np1 = p_arr.shape()[0];
    let dim = p_arr.shape()[1];
    let spatial_dim = dim - 1;
    let p_flat: Vec<f64> = p_arr.iter().copied().collect();

    let (ctrl, n_ctrl, radii, n_obs) = obstacle_arrays(obstacle_ctrl, obstacle_r, dim)?;

    let obstacles = SpacetimeObstacleData {
        ctrl: &ctrl,
        n_ctrl,
        radii: &radii,
        n_obs,
        spatial_dim,
    };
    let (station_vec, n_stations) = station_arrays(Some(stations), spatial_dim)?;
    let station_data = StationData {
        pos: &station_vec,
        n_stations,
    };
    let a_list = bezier_opt_core::de_casteljau::segment_matrices_equal_params(np1 - 1, n_seg);

    let rows = match bezier_opt_core::spacetime_constraints::build_spacetime_occlusion_constraints(
        &a_list,
        &p_flat,
        np1,
        dim,
        &obstacles,
        &station_data,
        trust_radius,
        sound_clip,
    )
    .bundle
    {
        Some(b) => b.rows,
        None => Vec::new(),
    };

    let empty2 = |width: usize| PyArray2::from_vec2(py, &vec![vec![0.0; width]; 0]);
    let normals_arr = if rows.is_empty() {
        empty2(spatial_dim)
    } else {
        PyArray2::from_vec2(py, &rows.iter().map(|r| r.normal.clone()).collect::<Vec<_>>())
    }
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let centers_arr = if rows.is_empty() {
        empty2(spatial_dim)
    } else {
        PyArray2::from_vec2(
            py,
            &rows.iter().map(|r| r.body_center.clone()).collect::<Vec<_>>(),
        )
    }
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

    Ok((
        normals_arr,
        PyArray1::from_vec(py, rows.iter().map(|r| r.lower_bound).collect::<Vec<_>>()),
        PyArray1::from_vec(py, rows.iter().map(|r| r.segment_idx as i32).collect::<Vec<_>>()),
        PyArray1::from_vec(py, rows.iter().map(|r| r.cp_idx as i32).collect::<Vec<_>>()),
        PyArray1::from_vec(py, rows.iter().map(|r| r.obstacle_idx as i32).collect::<Vec<_>>()),
        PyArray1::from_vec(py, rows.iter().map(|r| r.station_idx as i32).collect::<Vec<_>>()),
        centers_arr,
        PyArray1::from_vec(py, rows.iter().map(|r| r.body_radius).collect::<Vec<_>>()),
        PyArray1::from_vec(py, rows.iter().map(|r| r.t_lo).collect::<Vec<_>>()),
        PyArray1::from_vec(py, rows.iter().map(|r| r.t_hi).collect::<Vec<_>>()),
        PyArray1::from_vec(py, rows.iter().map(|r| r.margin).collect::<Vec<_>>()),
    )
        .into_pyobject(py)?
        .into())
}

/// Opaque handle holding precomputed SCP data and obstacle arrays.
/// Exposes a `step()` method that runs one SCP iteration.
#[pyclass]
struct SpacetimeScpContext {
    precomputed: spacetime_optimizer::ScpPrecomputed,
    ctrl: Vec<f64>,
    n_ctrl: usize,
    radii: Vec<f64>,
    n_obs: usize,
    spatial_dim: usize,
    stations: Vec<f64>,
    n_stations: usize,
    elastic_weight: f64,
    tol: f64,
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
        obstacle_ctrl,
        obstacle_r,
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
        sound_clip = false,
        v_max = None,
        time_weight = 0.0,
        free_arrival_time = false,
        stations = None,
    ))]
    fn new(
        p_init: PyReadonlyArray2<'_, f64>,
        obstacle_ctrl: PyReadonlyArray3<'_, f64>,
        obstacle_r: PyReadonlyArray1<'_, f64>,
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
        sound_clip: bool,
        v_max: Option<f64>,
        time_weight: f64,
        free_arrival_time: bool,
        stations: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        let p_arr = p_init.as_array();
        let np1 = p_arr.shape()[0];
        let dim = p_arr.shape()[1];
        if dim < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("dim >= 2 required"));
        }
        let spatial_dim = dim - 1;
        let p_flat: Vec<f64> = p_arr.iter().copied().collect();

        let (ctrl, n_ctrl, radii, n_obs) = obstacle_arrays(obstacle_ctrl, obstacle_r, dim)?;

        let pre = spacetime_optimizer::precompute_scp(
            &p_flat, np1, dim, n_seg, min_dt, coord_lb, coord_ub, time_lb, time_ub,
            v_max.unwrap_or(f64::NAN), time_weight, free_arrival_time, sound_clip,
        );

        let _ = scp_prox_weight; // the trust region does this job; see scp_step
        let state = spacetime_optimizer::ScpState::new(&p_flat, scp_trust_radius);
        let (station_vec, n_stations) = station_arrays(stations, spatial_dim)?;

        Ok(Self {
            precomputed: pre,
            ctrl,
            n_ctrl,
            radii,
            n_obs,
            spatial_dim,
            stations: station_vec,
            n_stations,
            elastic_weight,
            tol,
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
    ///
    /// The koz arrays are (seg, cp, obs, iter, normals, supports, centers, lbs,
    /// margins, slack), plus `info["koz_component"]` — a per-row array that rides
    /// in the dict only because pyo3 stops implementing IntoPyObject past a
    /// 12-tuple. **`component` joins the key**: one obstacle can present two
    /// separated lumps of tube to one segment and each gets its own plane, so
    /// rows group by (segment, obstacle, component), never (segment, obstacle).
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        let obstacles = SpacetimeObstacleData {
            ctrl: &self.ctrl,
            n_ctrl: self.n_ctrl,
            radii: &self.radii,
            n_obs: self.n_obs,
            spatial_dim: self.spatial_dim,
        };

        let station_data = StationData {
            pos: &self.stations,
            n_stations: self.n_stations,
        };

        let iter_out = spacetime_optimizer::scp_iterate(
            &mut self.state,
            &self.precomputed,
            &obstacles,
            &station_data,
            self.elastic_weight,
            self.tol,
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
        info.set_item("occlusion_row_count", result.occlusion_rows.len())?;
        // Windows in range whose supporting plane could not be built at this
        // reference. The QP got nothing for them, so the row count above does
        // not distinguish "nothing to constrain" from "could not constrain it".
        info.set_item("occlusion_planes_dropped", result.occlusion_planes_dropped)?;
        info.set_item(
            "occlusion_total_slack",
            result.occlusion_slack_per_row.iter().sum::<f64>(),
        )?;
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
        let comp_idx: Vec<i32> =
            result.koz_rows.iter().map(|r| r.component_idx as i32).collect();
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
        // Rides in `info` rather than the tuple: pyo3 stops implementing
        // IntoPyObject at 12 elements and the tuple is already there. It is a
        // per-row array like the rest, and it is part of the grouping key.
        info.set_item("koz_component", PyArray1::from_vec(py, comp_idx))?;
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
    m.add_function(wrap_pyfunction!(spacetime_occlusion_rows_exact, m)?)?;
    m.add_class::<SpacetimeScpContext>()?;
    Ok(())
}
