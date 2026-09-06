/// Gravitational acceleration models (two-body + J2).

/// Two-body gravitational acceleration in km/s^2.
pub fn accel_two_body(r_km: &[f64; 3], mu_km3_s2: f64) -> [f64; 3] {
    let rn_sq = r_km[0] * r_km[0] + r_km[1] * r_km[1] + r_km[2] * r_km[2];
    let rn = rn_sq.sqrt();
    let factor = -mu_km3_s2 / (rn * rn * rn);
    [factor * r_km[0], factor * r_km[1], factor * r_km[2]]
}

/// J2 perturbation acceleration in km/s^2.
pub fn accel_j2(r_km: &[f64; 3], mu_km3_s2: f64, r_e_km: f64, j2: f64) -> [f64; 3] {
    let (x, y, z) = (r_km[0], r_km[1], r_km[2]);
    let r2 = x * x + y * y + z * z;
    let rn = r2.sqrt();
    if rn < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    let z2 = z * z;
    let r5 = rn.powi(5);
    let factor = 1.5 * j2 * mu_km3_s2 * (r_e_km * r_e_km) / r5;
    let k = 5.0 * z2 / r2;
    [
        factor * x * (k - 1.0),
        factor * y * (k - 1.0),
        factor * z * (k - 3.0),
    ]
}

/// Two-body + J2 total gravitational acceleration in km/s^2.
pub fn accel_total(r_km: &[f64; 3], mu_km3_s2: f64, r_e_km: f64, j2: f64) -> [f64; 3] {
    let tb = accel_two_body(r_km, mu_km3_s2);
    let j2a = accel_j2(r_km, mu_km3_s2, r_e_km, j2);
    [tb[0] + j2a[0], tb[1] + j2a[1], tb[2] + j2a[2]]
}

/// Analytic Jacobian of `accel_total` at r, in 1/s^2.
///
/// The paper writes this operator as the partial derivative it is; the solver
/// used to approximate it by central differences with a 1 km step, so the
/// method as implemented was one finite difference away from the method as
/// described. Both terms are gradients of a harmonic potential outside the
/// body, which gives two invariants the tests check: the matrix is symmetric,
/// and its trace vanishes.
pub fn jacobian_analytic(r: &[f64; 3], mu: f64, r_e: f64, j2: f64) -> [[f64; 3]; 3] {
    let (x, y, z) = (r[0], r[1], r[2]);
    let r2 = x * x + y * y + z * z;
    let rn = r2.sqrt();
    if rn < 1e-12 {
        return [[0.0; 3]; 3];
    }
    let r3 = rn * r2;
    let r5 = r3 * r2;
    let r7 = r5 * r2;
    let r9 = r7 * r2;

    // Two-body: -mu (I/r^3 - 3 r r^T / r^5)
    let mut jac = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            let kron = if i == k { 1.0 } else { 0.0 };
            jac[i][k] = -mu * (kron / r3 - 3.0 * r[i] * r[k] / r5);
        }
    }

    // J2, differentiating g_x = c(5 x z^2/r^7 - x/r^5) and its siblings.
    let c = 1.5 * j2 * mu * r_e * r_e;
    let z2 = z * z;
    jac[0][0] += c * (5.0 * z2 / r7 - 35.0 * x * x * z2 / r9 - 1.0 / r5 + 5.0 * x * x / r7);
    jac[1][1] += c * (5.0 * z2 / r7 - 35.0 * y * y * z2 / r9 - 1.0 / r5 + 5.0 * y * y / r7);
    jac[2][2] += c * (30.0 * z2 / r7 - 35.0 * z2 * z2 / r9 - 3.0 / r5);

    let off_xy = c * (5.0 * x * y / r7 - 35.0 * x * y * z2 / r9);
    let off_xz = c * (15.0 * x * z / r7 - 35.0 * x * z * z2 / r9);
    let off_yz = c * (15.0 * y * z / r7 - 35.0 * y * z * z2 / r9);
    jac[0][1] += off_xy;
    jac[1][0] += off_xy;
    jac[0][2] += off_xz;
    jac[2][0] += off_xz;
    jac[1][2] += off_yz;
    jac[2][1] += off_yz;
    jac
}

/// Central-difference Jacobian of accel_total at r0, h in km.
pub fn jacobian_numeric(
    r0: &[f64; 3],
    mu: f64,
    r_e: f64,
    j2: f64,
    h: f64,
) -> [[f64; 3]; 3] {
    let mut jac = [[0.0f64; 3]; 3];
    for i in 0..3 {
        let mut rp = *r0;
        let mut rm = *r0;
        rp[i] += h;
        rm[i] -= h;
        let fp = accel_total(&rp, mu, r_e, j2);
        let fm = accel_total(&rm, mu, r_e, j2);
        for j in 0..3 {
            jac[j][i] = (fp[j] - fm[j]) / (2.0 * h);
        }
    }
    jac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobian_analytic_matches_central_difference() {
        let mu = 398600.4418;
        let r_e = 6371.0;
        let j2 = 1.08262668e-3;
        for r in [
            [6616.0, 0.0, 0.0],
            [0.0, 6771.0, 0.0],
            [3000.0, -4000.0, 5000.0],
            [-4200.0, 1500.0, -5100.0],
        ] {
            let a = jacobian_analytic(&r, mu, r_e, j2);
            let n = jacobian_numeric(&r, mu, r_e, j2, 1e-3);
            let scale = a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            for i in 0..3 {
                for k in 0..3 {
                    assert!(
                        (a[i][k] - n[i][k]).abs() < 1e-6 * scale,
                        "r={r:?} [{i}][{k}]: analytic {} vs numeric {}",
                        a[i][k],
                        n[i][k]
                    );
                }
            }
        }
    }

    #[test]
    fn test_jacobian_analytic_is_symmetric_and_traceless() {
        // Gravity outside the body is the gradient of a harmonic potential, so
        // the Jacobian must be symmetric and its trace must vanish. Neither can
        // hold by accident: perturbing any single coefficient breaks one of them.
        let mu = 398600.4418;
        let r_e = 6371.0;
        let j2 = 1.08262668e-3;
        for r in [
            [6616.0, 0.0, 0.0],
            [3000.0, -4000.0, 5000.0],
            [-1200.0, 6300.0, 2400.0],
        ] {
            let a = jacobian_analytic(&r, mu, r_e, j2);
            let scale = a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            for (i, k) in [(0, 1), (0, 2), (1, 2)] {
                assert!((a[i][k] - a[k][i]).abs() < 1e-14 * scale, "asymmetric at {i},{k}");
            }
            let trace = a[0][0] + a[1][1] + a[2][2];
            assert!(trace.abs() < 1e-13 * scale, "trace {trace} not zero for r={r:?}");
        }
    }

    #[test]
    fn test_two_body_x_axis() {
        let r = [6771.0, 0.0, 0.0];
        let mu = 398600.4418;
        let a = accel_two_body(&r, mu);
        assert!((a[0] - (-0.008694250482823736)).abs() < 1e-12);
        assert!(a[1].abs() < 1e-15);
        assert!(a[2].abs() < 1e-15);
    }

    #[test]
    fn test_j2_x_axis() {
        let r = [6771.0, 0.0, 0.0];
        let mu = 398600.4418;
        let r_e = 6371.0;
        let j2 = 0.00108262668;
        let a = accel_j2(&r, mu, r_e, j2);
        assert!((a[0] - (-1.2500048995892425e-05)).abs() < 1e-15);
        assert!(a[1].abs() < 1e-15);
        assert!(a[2].abs() < 1e-15);
    }

    #[test]
    fn test_total_x_axis() {
        let r = [6771.0, 0.0, 0.0];
        let mu = 398600.4418;
        let r_e = 6371.0;
        let j2 = 0.00108262668;
        let a = accel_total(&r, mu, r_e, j2);
        assert!((a[0] - (-0.00870675053181963)).abs() < 1e-14);
    }
}
