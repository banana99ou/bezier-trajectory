use bezier_opt_core::constraints;
use bezier_opt_core::optimizer;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
    n_lin_seg = 100,
    scp_prox_weight = 0.0,
    scp_trust_radius = 0.0,
    enforce_prograde = false,
    prograde_n_samples = 16,
    elastic_weight = 1e-2,
    transfer_time = 1500.0,
    strict_koz_normals = false,
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
    n_lin_seg: usize,
    scp_prox_weight: f64,
    scp_trust_radius: f64,
    enforce_prograde: bool,
    prograde_n_samples: usize,
    elastic_weight: f64,
    transfer_time: f64,
    // false = Skip (historical): a segment whose centroid sits on the KOZ centre
    // emits no rows and is invisible to the certificate.
    // true  = Fallback: deterministic replacement normal, counted in
    //         info["koz_degenerate_segments"].
    strict_koz_normals: bool,
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
        transfer_time,
        n_lin_seg,
        scp_prox_weight,
        scp_trust_radius,
        v0_vec.as_deref(),
        v1_vec.as_deref(),
        a0_vec.as_deref(),
        a1_vec.as_deref(),
        enforce_prograde,
        prograde_n_samples,
        elastic_weight,
        if strict_koz_normals {
            constraints::DegenerateNormal::Fallback
        } else {
            constraints::DegenerateNormal::Skip
        },
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

    // Surface jacobian drift trace as a list-of-lists (outer = SCP iter, inner = segment).
    if !result.jacobian_drift_history.is_empty() {
        let outer = PyList::empty(py);
        for row in &result.jacobian_drift_history {
            let inner = PyList::new(py, row)?;
            outer.append(inner)?;
        }
        info.set_item("jacobian_drift_history", outer)?;
    }

    // Per-accepted-step SCvx diagnostics (flat lists, one entry per accepted step).
    for (key, hist) in [
        ("rho_history", &result.rho_history),
        ("trust_history", &result.trust_history),
        ("merit_history", &result.merit_history),
        ("step_norm_history", &result.step_norm_history),
        ("slack_history", &result.slack_history),
        ("phase_history", &result.phase_history),
    ] {
        if !hist.is_empty() {
            info.set_item(key, PyList::new(py, hist)?)?;
        }
    }

    Ok((p_opt, info))
}

#[pymodule]
fn bezier_opt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_orbital_docking, m)?)?;
    Ok(())
}
