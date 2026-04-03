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
