import numpy as np

from orbital_docking import constants
from orbital_docking.dymos_t6 import (
    DymosT6Config,
    OrbitalTransferODE,
    build_experiment_spec,
    dymos_library_versions,
    make_initial_guess_bundle,
    make_demo_problem_for_time,
    warm_start_contract,
)
from orbital_docking.optimization import _accel_total


def test_dymos_versions_available():
    versions = dymos_library_versions()
    assert "dymos" in versions
    assert "openmdao" in versions


def test_orbital_transfer_ode_matches_repo_gravity():
    r = np.array(
        [
            [constants.CHASER_RADIUS, 0.0, 0.0],
            [0.0, constants.ISS_RADIUS, 20.0],
        ],
        dtype=float,
    )
    v = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=float,
    )
    u = np.array(
        [
            [1e-4, -2e-4, 3e-4],
            [-4e-4, 5e-4, -6e-4],
        ],
        dtype=float,
    )

    comp = OrbitalTransferODE(num_nodes=2)
    comp.setup()
    outputs = {
        "r_dot": np.zeros_like(r),
        "v_dot": np.zeros_like(r),
        "koz_margin": np.zeros(2),
        "control_power": np.zeros((2, 1)),
    }
    comp.compute({"r": r, "v": v, "u": u}, outputs)

    expected_g = np.array(
        [
            _accel_total(
                r_i,
                constants.EARTH_MU_SCALED,
                constants.EARTH_RADIUS_KM,
                constants.EARTH_J2,
            )
            for r_i in r
        ],
        dtype=float,
    )
    assert np.allclose(outputs["r_dot"], v)
    assert np.allclose(outputs["v_dot"], u + expected_g)
    assert np.allclose(outputs["koz_margin"], np.linalg.norm(r, axis=1) - constants.KOZ_RADIUS)


def test_initial_guess_bundle_has_clean_naive_and_warm_checks():
    bundle = make_initial_guess_bundle()

    naive = bundle["naive"]["sanity"]
    warm = bundle["warm_start"]["sanity"]
    assert naive["finite"]
    assert naive["endpoint_position_error_km"]["start"] <= 1e-9
    assert naive["endpoint_velocity_error_km_s"]["start"] <= 1e-9
    assert warm["finite"]
    assert warm["endpoint_velocity_error_km_s"]["start"] <= 1e-6
    assert warm["endpoint_velocity_error_km_s"]["end"] <= 1e-6


def test_warm_start_contract_matches_locked_selection():
    contract = warm_start_contract()
    assert contract["degree"] == 7
    assert contract["n_seg"] == 16
    assert contract["raw_endpoint_velocity_error_km_s"]["start"] <= 1e-6
    assert contract["raw_endpoint_velocity_error_km_s"]["end"] <= 1e-6


def test_experiment_spec_freezes_only_initial_guess_difference():
    spec = build_experiment_spec(DymosT6Config())
    assert spec["framework"]["transcription"] == "Radau"
    assert spec["framework"]["optimizer"] == "SLSQP"
    assert spec["comparison_contract"]["different_between_runs"] == ["initial guess only"]


def test_transfer_time_override_propagates_to_problem_and_guesses():
    transfer_time_s = 1800.0
    problem = make_demo_problem_for_time(transfer_time_s)
    bundle = make_initial_guess_bundle(transfer_time_s=transfer_time_s)
    contract = warm_start_contract(transfer_time_s=transfer_time_s)

    assert problem.transfer_time_s == transfer_time_s
    assert bundle["time_s"][-1] == transfer_time_s
    assert bundle["naive"]["time_s"][-1] == transfer_time_s
    assert bundle["warm_start"]["time_s"][-1] == transfer_time_s
    assert contract["transfer_time_s"] == transfer_time_s
    assert contract["raw_endpoint_velocity_error_km_s"]["start"] <= 1e-6
    assert contract["raw_endpoint_velocity_error_km_s"]["end"] <= 1e-6
