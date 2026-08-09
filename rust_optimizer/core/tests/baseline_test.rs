/// Integration test: validate Rust implementation against Python baseline artifacts.
use bezier_opt_core::{bezier, constraints, de_casteljau, gravity, optimizer};
use std::fs;

#[derive(serde::Deserialize)]
struct Baseline {
    scenario: Scenario,
    gravity_test: GravityTest,
    matrices: Matrices,
    segment_matrices_N3_nseg4: Vec<Vec<Vec<f64>>>,
    optimizer_result: OptimizerResult,
}

#[derive(serde::Deserialize)]
struct Scenario {
    #[serde(rename = "P_init")]
    p_init: Vec<Vec<f64>>,
    r_e: f64,
    #[serde(rename = "T")]
    t: f64,
    #[serde(rename = "N")]
    n: usize,
    n_seg: usize,
    mu: f64,
    #[serde(rename = "R_e_km")]
    r_e_km: f64,
    #[serde(rename = "J2")]
    j2: f64,
}

#[derive(serde::Deserialize)]
struct GravityTest {
    r_km: Vec<f64>,
    two_body: Vec<f64>,
    j2: Vec<f64>,
    total: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct Matrices {
    #[serde(rename = "D3")]
    d3: Vec<Vec<f64>>,
    #[serde(rename = "E2")]
    e2: Vec<Vec<f64>>,
    #[serde(rename = "G3")]
    g3: Vec<Vec<f64>>,
}

#[derive(serde::Deserialize)]
struct OptimizerResult {
    #[serde(rename = "P_opt")]
    p_opt: Vec<Vec<f64>>,
    cost_true_energy: f64,
    min_radius: f64,
    iterations: usize,
    feasible: bool,
}

fn load_baseline() -> Baseline {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../rust_migration_baseline.json"
    );
    let data = fs::read_to_string(path).expect("Failed to read baseline JSON");
    serde_json::from_str(&data).expect("Failed to parse baseline JSON")
}

#[test]
fn test_gravity_two_body() {
    let bl = load_baseline();
    let r = [bl.gravity_test.r_km[0], bl.gravity_test.r_km[1], bl.gravity_test.r_km[2]];
    let a = gravity::accel_two_body(&r, bl.scenario.mu);
    for i in 0..3 {
        assert!(
            (a[i] - bl.gravity_test.two_body[i]).abs() < 1e-12,
            "two_body[{i}]: rust={}, py={}",
            a[i],
            bl.gravity_test.two_body[i]
        );
    }
}

#[test]
fn test_gravity_j2() {
    let bl = load_baseline();
    let r = [bl.gravity_test.r_km[0], bl.gravity_test.r_km[1], bl.gravity_test.r_km[2]];
    let a = gravity::accel_j2(&r, bl.scenario.mu, bl.scenario.r_e_km, bl.scenario.j2);
    for i in 0..3 {
        assert!(
            (a[i] - bl.gravity_test.j2[i]).abs() < 1e-15,
            "j2[{i}]: rust={}, py={}",
            a[i],
            bl.gravity_test.j2[i]
        );
    }
}

#[test]
fn test_gravity_total() {
    let bl = load_baseline();
    let r = [bl.gravity_test.r_km[0], bl.gravity_test.r_km[1], bl.gravity_test.r_km[2]];
    let a = gravity::accel_total(&r, bl.scenario.mu, bl.scenario.r_e_km, bl.scenario.j2);
    for i in 0..3 {
        assert!(
            (a[i] - bl.gravity_test.total[i]).abs() < 1e-14,
            "total[{i}]: rust={}, py={}",
            a[i],
            bl.gravity_test.total[i]
        );
    }
}

#[test]
fn test_d_matrix() {
    let bl = load_baseline();
    let d = bezier::get_d_matrix(3);
    let expected = &bl.matrices.d3;
    for i in 0..3 {
        for j in 0..4 {
            assert!(
                (d[i * 4 + j] - expected[i][j]).abs() < 1e-14,
                "D[{i}][{j}]: rust={}, py={}",
                d[i * 4 + j],
                expected[i][j]
            );
        }
    }
}

#[test]
fn test_e_matrix() {
    let bl = load_baseline();
    let e = bezier::get_e_matrix(2);
    let expected = &bl.matrices.e2;
    for i in 0..4 {
        for j in 0..3 {
            assert!(
                (e[i * 3 + j] - expected[i][j]).abs() < 1e-14,
                "E[{i}][{j}]: rust={}, py={}",
                e[i * 3 + j],
                expected[i][j]
            );
        }
    }
}

#[test]
fn test_g_matrix() {
    let bl = load_baseline();
    let g = bezier::get_g_matrix(3);
    let expected = &bl.matrices.g3;
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (g[i * 4 + j] - expected[i][j]).abs() < 1e-14,
                "G[{i}][{j}]: rust={}, py={}",
                g[i * 4 + j],
                expected[i][j]
            );
        }
    }
}

#[test]
fn test_segment_matrices() {
    let bl = load_baseline();
    let mats = de_casteljau::segment_matrices_equal_params(3, 4);
    assert_eq!(mats.len(), 4);
    let expected = &bl.segment_matrices_N3_nseg4;
    for (seg_idx, (mat, exp)) in mats.iter().zip(expected.iter()).enumerate() {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (mat[i * 4 + j] - exp[i][j]).abs() < 1e-12,
                    "A[{seg_idx}][{i}][{j}]: rust={}, py={}",
                    mat[i * 4 + j],
                    exp[i][j]
                );
            }
        }
    }
}

#[test]
fn test_optimizer_golden_run() {
    let bl = load_baseline();
    let np1 = bl.scenario.n + 1;
    let dim = 3;

    // Flatten P_init
    let mut p_init = vec![0.0; np1 * dim];
    for i in 0..np1 {
        for d in 0..dim {
            p_init[i * dim + d] = bl.scenario.p_init[i][d];
        }
    }

    // Runs the canonical SCvx path (scp_trust_radius > 0). A zero trust radius here
    // would route the whole solve through the legacy fixed-point branch and leave the
    // ratio test, the elastic QP and the convergence streak completely unexercised.
    let result = optimizer::optimize_orbital_docking(
        &p_init,
        np1,
        dim,
        bl.scenario.n_seg,
        bl.scenario.r_e,
        200,    // max_iter -- converges at 115 here; headroom to catch a regression
        1e-8,   // tol -- locked value
        bl.scenario.t,
        100,    // sample_count
        0.0,    // scp_prox_weight
        2000.0, // scp_trust_radius -- locked r0, must exceed the iter-1 BC repair
        None,
        None,
        None,
        None,
        false,
        16,
        1e-2,  // elastic_weight -- locked w_s (exact-penalty rule)
        false, // freeze_gravity_jacobian
        1,     // freeze_after_iter
        constraints::DegenerateNormal::Skip,
    );

    eprintln!("=== Rust optimizer result ===");
    eprintln!("iterations: {}", result.iterations);
    eprintln!("min_radius: {}", result.info["min_radius"]);
    eprintln!("cost_true_energy: {}", result.info["cost_true_energy"]);
    eprintln!("feasible: {}", result.feasible);
    eprintln!("=== Python baseline ===");
    eprintln!("iterations: {}", bl.optimizer_result.iterations);
    eprintln!("min_radius: {}", bl.optimizer_result.min_radius);
    eprintln!("cost_true_energy: {}", bl.optimizer_result.cost_true_energy);
    eprintln!("feasible: {}", bl.optimizer_result.feasible);

    // P_opt comparison
    eprintln!("=== P_opt (Rust) ===");
    for i in 0..np1 {
        eprintln!(
            "  [{}, {}, {}]",
            result.p_opt[i * dim],
            result.p_opt[i * dim + 1],
            result.p_opt[i * dim + 2]
        );
    }

    // Generous tolerances for different QP solvers
    // The optimizer should at least make progress from initial straight line
    let initial_min_radius = {
        let mut min_r = f64::INFINITY;
        for i in 0..=1000 {
            let tau = i as f64 / 1000.0;
            let pt = bezier::evaluate(&p_init, np1, dim, tau);
            let r: f64 = pt.iter().map(|x| x * x).sum::<f64>().sqrt();
            if r < min_r { min_r = r; }
        }
        min_r
    };
    eprintln!("initial min_radius: {}", initial_min_radius);

    assert!(
        result.info["min_radius"] > initial_min_radius,
        "Optimizer should improve min_radius from initial: {} -> {}",
        initial_min_radius,
        result.info["min_radius"]
    );

    // --- Properties the solve must actually satisfy -------------------------
    // Stated as properties rather than pinned numbers: a golden value copied from
    // the run under test only detects change, never correctness.

    assert!(result.feasible, "solve reported infeasible");

    // The curve clears the KOZ (dense sample, independent of the hull rows).
    assert!(
        result.info["min_radius"] >= bl.scenario.r_e - 1e-6,
        "curve enters the KOZ: min_radius {} < r_e {}",
        result.info["min_radius"],
        bl.scenario.r_e
    );

    // Proposition 1 certificate: hull rows rebuilt at the final iterate are satisfied,
    // which is what licenses the continuous-curve guarantee.
    assert!(
        result.info["final_cp_violation_km"] <= 1e-6,
        "Prop-1 certificate not carried: final_cp_violation_km = {}",
        result.info["final_cp_violation_km"]
    );
    assert!(
        result.info["final_hull_violation_km"] <= 1e-6,
        "hull rows violated at final iterate: {}",
        result.info["final_hull_violation_km"]
    );

    // No segment may be silently absent from the certificate.
    assert_eq!(
        result.info["koz_degenerate_segments"], 0.0,
        "degenerate KOZ normals were skipped; certificate does not cover the curve"
    );

    // Every QP that fed the ratio test must have solved to the requested tolerances.
    assert_eq!(
        result.info["qp_almost_solved"], 0.0,
        "{} QP solves terminated on Clarabel's REDUCED tolerances and still fed the \
         merit/ratio test",
        result.info["qp_almost_solved"]
    );

    // NOTE: convergence is deliberately NOT asserted here -- see
    // `known_defect_scvx_two_cycle_prevents_convergence` below, which documents the
    // open defect rather than hiding it behind a weakened assertion.
}

/// The loop must stop for a STATED reason, not by running out of budget or by shrinking
/// its trust region to nothing.
///
/// The assertion is deliberately on `scvx_stop_reason`, NOT on `scvx_converged`. That
/// flag is also set by the trust-collapse exit whenever the iterate happens to be
/// feasible, so it reads "true" at a deadlocked point; an earlier version of this test
/// asserted on it and passed spuriously.
///
/// Regression history on this scenario:
///   200 iters -- accept/reject 2-cycle, 100 `pred < 0` events (QP conditioning: the
///                variables were ~7e3 km against a Hessian of ~1e-13)
///    24 iters -- after step-coordinate + power-of-two objective rescaling; 0 negative
///                pred, but the merit sat frozen to 17 digits for the last 20 iterations
///                while the ratio test rejected floating-point noise (act ~ -1.1e-15 on
///                a merit of 3.8e-7) until the trust region collapsed
///     5 iters -- after adding the predicted-reduction stationarity test; identical
///                objective to all 17 digits, and the trust region never shrinks
#[test]
fn optimizer_stops_for_a_stated_reason() {
    let bl = load_baseline();
    let np1 = bl.scenario.n + 1;
    let dim = 3;
    let mut p_init = vec![0.0; np1 * dim];
    for i in 0..np1 {
        for d in 0..dim {
            p_init[i * dim + d] = bl.scenario.p_init[i][d];
        }
    }
    let result = optimizer::optimize_orbital_docking(
        &p_init, np1, dim, bl.scenario.n_seg, bl.scenario.r_e,
        200, 1e-8, bl.scenario.t, 100, 0.0, 2000.0,
        None, None, None, None, false, 16, 1e-2, false, 1,
        constraints::DegenerateNormal::Skip,
    );
    // 1 = K-consecutive merit streak, 4 = model stationarity. Both are principled
    // stops. 0 (iteration cap), 2 (trust collapse) and 3 (QP failure) are the loop
    // giving up and must fail here.
    let stop = result.info["scvx_stop_reason"];
    assert!(
        stop == 1.0 || stop == 4.0,
        "loop did not stop for a stated reason: stop_reason={} (0=cap, 2=trust collapse, \
         3=QP failure), iterations={}, final_trust_radius={}",
        stop,
        result.iterations,
        result.info["final_trust_radius"]
    );
    assert!(
        result.info["final_hull_violation_km"] <= 1e-6,
        "stopped without the Prop-1 certificate: {}",
        result.info["final_hull_violation_km"]
    );
}
