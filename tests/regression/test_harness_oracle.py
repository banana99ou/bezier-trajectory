"""The verification harness's own oracles must agree with each other.

Two defects of the same shape have now been found in this harness:

  2026-08-09  J_true was a uniform-mean Riemann sum, O(1/n), ~1.6e-3 relative
              error at its default -- LARGER than the A/B objective gap it was
              being used to resolve. Replaced with Gauss-Legendre.
  2026-08-11  the same fix had never reached J_true_and_grad, which was still a
              uniform mean. Pillar 1 optimizes with it and scores with J_true,
              so the NLP minimized one function and was graded on another; at
              n_seg=64 the oracle's own error (0.207%) exceeded the gap it
              reported (0.165%).

The second defect existed for two days behind a green suite because nothing
compared the two oracles. These tests do.

Run:  .venv/bin/python -m pytest tests/regression/test_harness_oracle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.verify import harness_common as H

T = 1500.0


def _net(seed=0):
    """A representative orbital-scale control net (km). No solver needed."""
    rng = np.random.default_rng(seed)
    N = 7
    ang = np.linspace(0.0, 1.2, N + 1)
    P = np.zeros((N + 1, 3))
    P[:, 0] = 6800.0 * np.cos(ang)
    P[:, 1] = 6800.0 * np.sin(ang)
    P[:, 2] = rng.normal(size=N + 1) * 50.0
    return P


def test_the_two_objective_oracles_agree():
    """J_true and J_true_and_grad must integrate the same thing.

    Fails if either drifts to a different quadrature -- the 2026-08-11 defect.
    A uniform mean over 600 points differs from this by ~2e-3, four orders
    above the tolerance here, so this assertion is not vacuous.
    """
    P = _net()
    j_scalar = H.J_true(P, T)
    j_grad_fn, _ = H.J_true_and_grad(P, T)
    assert j_scalar > 0.0
    assert abs(j_grad_fn - j_scalar) / j_scalar < 1e-12


def test_analytic_gradient_matches_central_differences():
    """The analytic gradient must differentiate the objective it ships with.

    Central differences on J_true, which is the quantity Pillar 1 scores with.
    Before the 2026-08-11 fix this stood at 4.147e-03 -- and, tellingly, was
    INDEPENDENT of the step size, which is the signature of two different
    integrands rather than finite-difference noise.
    """
    P = _net()
    _, g = H.J_true_and_grad(P, T)

    step = 1e-4
    g_fd = np.zeros_like(P)
    for i in range(P.shape[0]):
        for d in range(P.shape[1]):
            h = np.zeros_like(P)
            h[i, d] = step
            g_fd[i, d] = (H.J_true(P + h, T) - H.J_true(P - h, T)) / (2 * step)

    rel = np.linalg.norm(g - g_fd.ravel()) / np.linalg.norm(g_fd)
    assert rel < 1e-6, f"analytic gradient disagrees with finite differences: {rel:.3e}"


def test_gradient_check_can_detect_a_wrong_gradient():
    """The check above must be able to fail. Scale the gradient by 0.1% and
    confirm the comparison catches it -- otherwise the test is decoration."""
    P = _net()
    _, g = H.J_true_and_grad(P, T)

    step = 1e-4
    g_fd = np.zeros_like(P)
    for i in range(P.shape[0]):
        for d in range(P.shape[1]):
            h = np.zeros_like(P)
            h[i, d] = step
            g_fd[i, d] = (H.J_true(P + h, T) - H.J_true(P - h, T)) / (2 * step)

    rel_bad = np.linalg.norm(g * 1.001 - g_fd.ravel()) / np.linalg.norm(g_fd)
    assert rel_bad > 1e-6


@pytest.mark.parametrize("n_nodes", [48, 96, 192])
def test_quadrature_is_converged_not_merely_fine(n_nodes):
    """Gauss-Legendre must have CONVERGED, not just be using many points.

    A Riemann sum also "improves" with more points, at O(1/n); that is what let
    the old form look adequate. Here the value must be node-count invariant to
    f64, which a Riemann sum cannot achieve at these counts.
    """
    P = _net()
    ref = H.J_true(P, T, n_nodes=192)
    got = H.J_true(P, T, n_nodes=n_nodes)
    assert abs(got - ref) / ref < 1e-12
