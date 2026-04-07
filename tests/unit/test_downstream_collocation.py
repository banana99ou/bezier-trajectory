import numpy as np

from orbital_docking.downstream_collocation import (
    build_bezier_warm_start,
    build_naive_initial_guess,
    make_demo_problem,
)
from orbital_docking.optimization import generate_initial_control_points


def test_naive_initial_guess_matches_boundary_states():
    problem = make_demo_problem(n_intervals=6)
    x0 = build_naive_initial_guess(problem)
    n_block = problem.n_nodes * 3

    r = x0[:n_block].reshape(problem.n_nodes, 3)
    v = x0[n_block : 2 * n_block].reshape(problem.n_nodes, 3)
    u = x0[2 * n_block :].reshape(problem.n_nodes, 3)

    assert np.allclose(r[0], problem.r0)
    assert np.allclose(r[-1], problem.rf)
    assert np.allclose(v[0], problem.v0)
    assert np.allclose(v[-1], problem.vf)
    assert np.all(np.isfinite(u))


def test_bezier_warm_start_overrides_boundaries_and_populates_controls():
    problem = make_demo_problem(n_intervals=6)
    control_points = generate_initial_control_points(4, problem.r0, problem.rf)
    x0 = build_bezier_warm_start(problem, control_points)
    n_block = problem.n_nodes * 3

    r = x0[:n_block].reshape(problem.n_nodes, 3)
    v = x0[n_block : 2 * n_block].reshape(problem.n_nodes, 3)
    u = x0[2 * n_block :].reshape(problem.n_nodes, 3)

    assert np.allclose(r[0], problem.r0)
    assert np.allclose(r[-1], problem.rf)
    assert np.allclose(v[0], problem.v0)
    assert np.allclose(v[-1], problem.vf)
    assert np.all(np.isfinite(u))
