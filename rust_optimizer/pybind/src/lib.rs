use bezier_opt_core::{optimizer, spacetime_constraints::SpacetimeObstacleData, spacetime_optimizer};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

#[pymodule]
fn bezier_opt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_orbital_docking, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_spacetime_bezier, m)?)?;
    Ok(())
}
