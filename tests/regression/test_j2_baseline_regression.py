"""
Regression test for the normalized J2 reference dataset.

This stays offline by loading the committed JSON fixture under tests/data/.
Refresh that fixture intentionally with tools/fetch_j2_reference_data.py.
"""

from pathlib import Path

import numpy as np

from orbital_docking import constants
from orbital_docking.gravity import _accel_total
from orbital_docking.j2_validation import load_reference_dataset
from orbital_docking.visualization import accel_gravity_total_km_s2


ABS_TOL_TOTAL = 2e-11  # km/s^2
REL_TOL_TOTAL = 5e-9
ABS_TOL_CROSS = 1e-15  # km/s^2


def _dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "j2_reference" / "egm2008_degree2_samples.json"


def test_j2_reference_dataset_matches_gravity_helper_and_visualization():
    dataset_path = _dataset_path()
    assert dataset_path.exists(), f"Missing J2 reference dataset: {dataset_path}"
    dataset = load_reference_dataset(dataset_path)

    for sample in dataset["samples"]:
        r_km = np.array(sample["r_km"], dtype=float)
        a_ref = np.array(sample["a_total_km_s2"], dtype=float)
        a_opt = _accel_total(
            r_km,
            constants.EARTH_MU_SCALED,
            constants.EARTH_RADIUS_KM,
            constants.EARTH_J2,
        )
        a_viz = accel_gravity_total_km_s2(r_km)

        np.testing.assert_allclose(
            a_opt,
            a_ref,
            rtol=REL_TOL_TOTAL,
            atol=ABS_TOL_TOTAL,
            err_msg=f"gravity helper mismatch for sample {sample['sample_id']}",
        )
        np.testing.assert_allclose(
            a_viz,
            a_ref,
            rtol=REL_TOL_TOTAL,
            atol=ABS_TOL_TOTAL,
            err_msg=f"visualization mismatch for sample {sample['sample_id']}",
        )
        np.testing.assert_allclose(
            a_opt,
            a_viz,
            rtol=0.0,
            atol=ABS_TOL_CROSS,
            err_msg=f"gravity-helper-vs-visualization mismatch for sample {sample['sample_id']}",
        )
