"""
Regression test: one fixed optimizer run compared to stored golden values.

Scenario: N=3, n_seg=4, P_init from conftest, no BC, small max_iter. Compares
cost_true_energy and min_radius to tests/data/golden_run.json.

To update the golden file when intended behavior changes, set UPDATE_GOLDEN=1
and run this test; see tests/data/README.md.
"""

import json
import os
from pathlib import Path

import pytest

from orbital_docking import optimize_orbital_docking


# Fixed scenario (keep run small)
GOLDEN_N_SEG = 4
GOLDEN_MAX_ITER = 10
GOLDEN_TOL = 1e-6
GOLDEN_RTOL = 1e-5
GOLDEN_ATOL_COST = 1e-12
GOLDEN_ATOL_RADIUS = 1e-6  # km


def _golden_path():
    return Path(__file__).resolve().parent.parent / "data" / "golden_run.json"


def test_golden_run_matches_stored(P_init, r_e):
    """
    Run optimize_orbital_docking once with fixed scenario; assert cost_true_energy
    and min_radius match stored golden values within tolerance.
    """
    golden_path = _golden_path()
    if os.environ.get("UPDATE_GOLDEN") == "1":
        P_opt, info = optimize_orbital_docking(
            P_init,
            n_seg=GOLDEN_N_SEG,
            r_e=r_e,
            max_iter=GOLDEN_MAX_ITER,
            tol=GOLDEN_TOL,
            v0=None,
            v1=None,
            use_cache=False,
            verbose=False,
        )
        data = {
            "cost_true_energy": float(info["cost_true_energy"]),
            "min_radius": float(info["min_radius"]),
            "_comment": "Fixed scenario: N=3, n_seg=4, P_init from conftest, no BC, max_iter=10, tol=1e-6. Update with UPDATE_GOLDEN=1 (see tests/data/README.md).",
        }
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        with open(golden_path, "w") as f:
            json.dump(data, f, indent=2)
        pytest.skip("UPDATE_GOLDEN=1: golden file updated; run again without it to assert.")
        return

    assert golden_path.exists(), (
        f"Golden file not found: {golden_path}. Run with UPDATE_GOLDEN=1 to create it (see tests/data/README.md)."
    )
    with open(golden_path) as f:
        golden = json.load(f)

    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=GOLDEN_N_SEG,
        r_e=r_e,
        max_iter=GOLDEN_MAX_ITER,
        tol=GOLDEN_TOL,
        v0=None,
        v1=None,
        use_cache=False,
        verbose=False,
    )

    assert info["cost_true_energy"] == pytest.approx(
        golden["cost_true_energy"],
        rel=GOLDEN_RTOL,
        abs=GOLDEN_ATOL_COST,
    ), (
        f"cost_true_energy regression: got {info['cost_true_energy']}, golden {golden['cost_true_energy']}"
    )
    assert info["min_radius"] == pytest.approx(
        golden["min_radius"],
        rel=GOLDEN_RTOL,
        abs=GOLDEN_ATOL_RADIUS,
    ), (
        f"min_radius regression: got {info['min_radius']}, golden {golden['min_radius']}"
    )
