"""
Helpers for validating the repo's J2 gravity logic against an independent
reference constructed from normalized C20 gravity coefficients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_EGM2008_C20_URL = (
    "https://raw.githubusercontent.com/CelesTrak/fundamentals-of-astrodynamics/"
    "main/datalib/EGM-08norm100.txt"
)


@dataclass(frozen=True)
class J2ValidationCase:
    """Deterministic sample position definition for baseline validation."""

    sample_id: str
    altitude_km: float
    latitude_deg: float
    longitude_deg: float


def j2_from_c20(c20_normalized: float) -> float:
    """
    Convert fully normalized C20 into the classical J2 coefficient.

    For fully normalized zonal coefficients:
        J2 = -sqrt(5) * Cbar20
    """
    return float(-np.sqrt(5.0) * float(c20_normalized))


def normalized_p20(sin_lat: float) -> float:
    """Fully normalized degree-2 zonal Legendre basis value."""
    s = float(sin_lat)
    return float(0.5 * np.sqrt(5.0) * (3.0 * s * s - 1.0))


def total_potential_from_c20(
    r_km: np.ndarray,
    mu_km3_s2: float,
    r_e_km: float,
    c20_normalized: float,
) -> float:
    """
    Two-body + degree-2 zonal gravity potential in km^2/s^2.

    This uses the normalized C20 representation directly:

        U(r, phi) = mu / r * [1 + (Re / r)^2 * Cbar20 * Pbar20(sin(phi))]

    The production code uses the equivalent closed-form J2 acceleration.
    This scalar-potential form is kept separate to provide an independent
    comparison path for validation.
    """
    r = np.asarray(r_km, dtype=float)
    rn = float(np.linalg.norm(r))
    if rn < 1e-12:
        return 0.0
    sin_lat = float(r[2] / rn)
    zonal = (float(r_e_km) / rn) ** 2 * float(c20_normalized) * normalized_p20(sin_lat)
    return float(mu_km3_s2 / rn * (1.0 + zonal))


def accel_total_from_c20_numeric_gradient(
    r_km: np.ndarray,
    mu_km3_s2: float,
    r_e_km: float,
    c20_normalized: float,
    h_km: float = 1e-3,
) -> np.ndarray:
    """
    Reference gravity acceleration from a scalar potential via central differences.

    The returned vector is in km/s^2.
    """
    r0 = np.asarray(r_km, dtype=float)
    grad = np.zeros(3, dtype=float)
    h = float(h_km)
    for i in range(3):
        dr = np.zeros(3, dtype=float)
        dr[i] = h
        up = total_potential_from_c20(r0 + dr, mu_km3_s2, r_e_km, c20_normalized)
        um = total_potential_from_c20(r0 - dr, mu_km3_s2, r_e_km, c20_normalized)
        grad[i] = (up - um) / (2.0 * h)
    return grad


def accel_two_body_reference(r_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    """Independent two-body acceleration helper in km/s^2."""
    r = np.asarray(r_km, dtype=float)
    rn = float(np.linalg.norm(r))
    if rn < 1e-12:
        return np.zeros(3, dtype=float)
    return (-float(mu_km3_s2) / (rn**3)) * r


def spherical_to_cartesian_km(radius_km: float, latitude_deg: float, longitude_deg: float) -> np.ndarray:
    """Convert spherical latitude/longitude to an Earth-centered Cartesian position."""
    lat = np.deg2rad(float(latitude_deg))
    lon = np.deg2rad(float(longitude_deg))
    clat = float(np.cos(lat))
    return np.array(
        [
            float(radius_km) * clat * float(np.cos(lon)),
            float(radius_km) * clat * float(np.sin(lon)),
            float(radius_km) * float(np.sin(lat)),
        ],
        dtype=float,
    )


def default_validation_cases() -> list[J2ValidationCase]:
    """Deterministic sample states spanning equatorial, inclined, and polar cases."""
    return [
        J2ValidationCase("leo_equator_x", 245.0, 0.0, 0.0),
        J2ValidationCase("leo_equator_y", 400.0, 0.0, 90.0),
        J2ValidationCase("iss_like_plane_1", 400.0, 51.64, 0.0),
        J2ValidationCase("iss_like_plane_2", 400.0, 51.64, 120.0),
        J2ValidationCase("polar_high_lat", 400.0, 80.0, 15.0),
        J2ValidationCase("mid_lat_mto", 800.0, 20.0, 45.0),
        J2ValidationCase("southern_transfer", 1500.0, -35.0, 180.0),
        J2ValidationCase("high_altitude", 20000.0, 10.0, -60.0),
    ]


def build_reference_dataset(
    *,
    mu_km3_s2: float,
    r_e_km: float,
    c20_normalized: float,
    source_url: str,
    source_label: str,
    gradient_step_km: float = 1e-3,
    cases: list[J2ValidationCase] | None = None,
) -> dict:
    """Create a normalized J2 validation dataset dictionary."""
    cases = default_validation_cases() if cases is None else list(cases)
    samples = []
    for case in cases:
        radius_km = float(r_e_km) + float(case.altitude_km)
        r_km = spherical_to_cartesian_km(radius_km, case.latitude_deg, case.longitude_deg)
        a_two = accel_two_body_reference(r_km, mu_km3_s2)
        a_total = accel_total_from_c20_numeric_gradient(
            r_km,
            mu_km3_s2,
            r_e_km,
            c20_normalized,
            h_km=gradient_step_km,
        )
        a_j2 = a_total - a_two
        samples.append(
            {
                "sample_id": case.sample_id,
                "altitude_km": float(case.altitude_km),
                "latitude_deg": float(case.latitude_deg),
                "longitude_deg": float(case.longitude_deg),
                "r_km": [float(v) for v in r_km],
                "a_two_body_km_s2": [float(v) for v in a_two],
                "a_j2_km_s2": [float(v) for v in a_j2],
                "a_total_km_s2": [float(v) for v in a_total],
            }
        )

    return {
        "schema_version": 1,
        "source": {
            "label": str(source_label),
            "url": str(source_url),
            "model": "degree2_zonal_from_normalized_c20",
        },
        "constants": {
            "mu_km3_s2": float(mu_km3_s2),
            "r_e_km": float(r_e_km),
            "c20_normalized": float(c20_normalized),
            "j2": float(j2_from_c20(c20_normalized)),
            "gradient_step_km": float(gradient_step_km),
            "axis_assumption": "Earth symmetry axis aligned with inertial z",
        },
        "samples": samples,
    }


def parse_c20_from_egm_text(text: str) -> float:
    """Extract the fully normalized C20 coefficient from an EGM coefficient table."""
    for raw_line in str(text).splitlines():
        parts = raw_line.split()
        if len(parts) < 4:
            continue
        if parts[0] == "2" and parts[1] == "0":
            return float(parts[2])
    raise ValueError("Could not find degree-2 order-0 coefficient in the provided text.")


def load_reference_dataset(path: str | Path) -> dict:
    """Load a normalized J2 reference dataset from JSON."""
    with Path(path).open() as f:
        return json.load(f)


def save_reference_dataset(path: str | Path, dataset: dict) -> None:
    """Write a normalized J2 reference dataset to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump(dataset, f, indent=2, sort_keys=True)
        f.write("\n")


def error_summary(actual: np.ndarray, reference: np.ndarray) -> dict:
    """Return basic vector error metrics."""
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    diff = actual - reference
    ref_norm = float(np.linalg.norm(reference))
    abs_norm = float(np.linalg.norm(diff))
    rel_norm = abs_norm / max(ref_norm, 1e-18)
    return {
        "abs_norm": abs_norm,
        "rel_norm": rel_norm,
        "max_abs_component": float(np.max(np.abs(diff))),
    }
