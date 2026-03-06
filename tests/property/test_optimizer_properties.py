"""
Property tests for the orbital docking optimizer: reproducibility, info contract, cache.
"""

import numpy as np
import pytest

from orbital_docking import optimize_orbital_docking
from orbital_docking.cache import get_cache_key, get_cache_path, clear_cache
import time


# Small scenario for fast property tests
PROP_N_SEG = 4
PROP_MAX_ITER = 5
PROP_TOL = 1e-6
COST_TOL = 1e-9
P_OPT_REL_TOL = 1e-8
# Cache hit timing can vary across machines/CI.
# Keep a loose absolute bound and a stronger relative check when the cold run is non-trivial.
CACHE_FAST_SEC = 2.0  # second run with cache hit should be under this


# ---- Reproducibility ----

def test_optimizer_reproducibility_same_cost(P_init, r_e):
    """Two calls with same args (use_cache=False) yield the same cost_true_energy within tolerance."""
    kwargs = dict(
        P_init=P_init,
        n_seg=PROP_N_SEG,
        r_e=r_e,
        max_iter=PROP_MAX_ITER,
        tol=PROP_TOL,
        v0=None,
        v1=None,
        use_cache=False,
        verbose=False,
    )
    P1, info1 = optimize_orbital_docking(**kwargs)
    P2, info2 = optimize_orbital_docking(**kwargs)
    assert info1["cost_true_energy"] == pytest.approx(info2["cost_true_energy"], abs=COST_TOL, rel=COST_TOL), (
        "cost_true_energy should be identical across runs with same args"
    )


def test_optimizer_reproducibility_same_P_opt(P_init, r_e):
    """Two calls with same args (use_cache=False) yield the same P_opt within tolerance."""
    kwargs = dict(
        P_init=P_init,
        n_seg=PROP_N_SEG,
        r_e=r_e,
        max_iter=PROP_MAX_ITER,
        tol=PROP_TOL,
        v0=None,
        v1=None,
        use_cache=False,
        verbose=False,
    )
    P1, _ = optimize_orbital_docking(**kwargs)
    P2, _ = optimize_orbital_docking(**kwargs)
    np.testing.assert_allclose(P1, P2, rtol=P_OPT_REL_TOL, atol=1e-12, err_msg="P_opt should be identical across runs")


# ---- Info contract ----

REQUIRED_INFO_KEYS = (
    "iterations",
    "feasible",
    "min_radius",
    "cost",
    "cost_true_energy",
    "max_control_accel_ms2",
    "elapsed_time",
)

INFO_KEY_TYPES = {
    "iterations": int,
    "feasible": bool,
    "min_radius": (int, float),
    "cost": (int, float),
    "cost_true_energy": (int, float),
    "max_control_accel_ms2": (int, float),
    "elapsed_time": (int, float),
}


def test_info_has_required_keys(P_init, r_e):
    """Returned info has keys: iterations, feasible, min_radius, cost, cost_true_energy, max_control_accel_ms2, elapsed_time."""
    _, info = optimize_orbital_docking(
        P_init,
        n_seg=PROP_N_SEG,
        r_e=r_e,
        max_iter=PROP_MAX_ITER,
        tol=PROP_TOL,
        use_cache=False,
        verbose=False,
    )
    for key in REQUIRED_INFO_KEYS:
        assert key in info, f"info missing required key: {key}"


def test_info_key_types(P_init, r_e):
    """Required info keys have expected types (int, bool, float)."""
    _, info = optimize_orbital_docking(
        P_init,
        n_seg=PROP_N_SEG,
        r_e=r_e,
        max_iter=PROP_MAX_ITER,
        tol=PROP_TOL,
        use_cache=False,
        verbose=False,
    )
    for key, expected_type in INFO_KEY_TYPES.items():
        if key not in info:
            continue
        val = info[key]
        if isinstance(expected_type, tuple):
            assert isinstance(val, expected_type), (
                f"info[{key!r}] should be one of {expected_type}, got {type(val)}"
            )
        else:
            assert isinstance(val, expected_type), (
                f"info[{key!r}] should be {expected_type}, got {type(val)}"
            )


# ---- Cache: second call returns quickly ----

def test_cache_second_call_fast(P_init, r_e):
    """With use_cache=True, second call for same params returns immediately (elapsed_time < threshold)."""
    kwargs = dict(
        P_init=P_init,
        n_seg=PROP_N_SEG,
        r_e=r_e,
        max_iter=PROP_MAX_ITER,
        tol=PROP_TOL,
        v0=None,
        v1=None,
        use_cache=True,
        verbose=False,
    )
    # Ensure no stale cache for this exact key
    cache_key = get_cache_key(
        P_init, PROP_N_SEG, r_e, PROP_MAX_ITER, PROP_TOL, 100, None, None, None, None
    )
    clear_cache(cache_key=cache_key)

    t_start = time.perf_counter()
    _, info1 = optimize_orbital_docking(**kwargs)
    wall_1 = time.perf_counter() - t_start

    t_start = time.perf_counter()
    _, info2 = optimize_orbital_docking(**kwargs)
    wall_2 = time.perf_counter() - t_start

    assert wall_2 < CACHE_FAST_SEC, (
        f"Second call with cache hit should be fast (wall={wall_2:.3f}s), expected < {CACHE_FAST_SEC}s"
    )
    # Stronger check when the cold run is meaningfully large.
    if wall_1 > 1.0:
        assert wall_2 < 0.5 * wall_1, (
            f"Cache hit should be faster than cold run: wall_2={wall_2:.3f}s, wall_1={wall_1:.3f}s"
        )
