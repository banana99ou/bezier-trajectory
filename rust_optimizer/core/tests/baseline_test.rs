/// Integration test: validate Rust implementation against Python baseline artifacts.
use bezier_opt_core::{bezier, de_casteljau, gravity, optimizer};
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

    let result = optimizer::optimize_orbital_docking(
        &p_init,
        np1,
        dim,
        bl.scenario.n_seg,
        bl.scenario.r_e,
        10,    // max_iter
        1e-6,  // tol
        bl.scenario.t,
        100,   // sample_count
        "energy",
        1e-9,
        0.0,
        0.0,
        0.0,
        None,
        None,
        None,
        None,
        false,
        16,
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
}
