"""
Objective construction helpers for the space-time Bezier optimizer.

Two things live here, and the distinction is the point of item B8:

* the **cost**, which is a smoothness regularizer in the *curve parameter*
  domain and knows nothing about physical time; and
* the **derivative operators**, which are exact linear maps from control points
  to the control points of the parameter-domain derivative curves. They are the
  building blocks for the physical bounds, which are constraints (item B9), not
  cost terms.

Why the cost is not "acceleration energy" (frozen formulation decisions 1-3,
derived in ``idea/spacetime.md`` sec. "The derivation behind the decisions"): the curve
parameter is not time. Physical velocity is the spatial parameter-derivative
divided by the time parameter-derivative -- a ratio of two Beziers, hence not a
polynomial -- and physical acceleration has that ratio's derivative, with a cubic
denominator. No quadratic form in the control points can equal either. So no
"acceleration energy matrix" exists to be derived, and the one that does exist
measures something else.
"""

from __future__ import annotations

import numpy as np

from orbital_docking.bezier import BezierCurve, get_D_matrix


def build_smoothness_regularizer(N: int, dim: int) -> np.ndarray:
    """
    Build the quadratic form of the parameter-domain smoothness regularizer.

    This is ``integral over the curve parameter of |d^2 r / d(parameter)^2|^2``,
    restricted to the spatial coordinates. It is **not** spatial acceleration
    energy: the parameter is not time, and the two coincide only when timing is
    uniform -- the case this method exists to escape. A plan that waits and then
    dashes scores exactly the same as one that cruises, because the time column
    is never written into (that is asserted by
    ``tests/unit/test_objective_operators.py``).

    Physics is bounded by constraints instead: the slant-limit speed cap
    (item B9) and the linearized acceleration cap.

    The time coordinate is left unpenalized, deliberately: penalizing it would
    bias the arrival time through the smoothness term rather than through the
    declared time penalty (item B10).

    **This is the TESTED MIRROR of the derivation, not the matrix production
    uses.** Rust is the sole optimizer backend and builds its own H in
    ``build_smoothness_regularizer_h`` (``spacetime_optimizer.rs``); nothing in
    the solve path calls this function. It exists so the derivation can be
    written out in a readable form and checked -- which is only worth anything if
    the two are pinned to each other, so they are:
    ``tests/unit/test_objective_matches_rust.py`` asserts
    ``0.5 * p^T H_python p`` against the Rust cost oracle over degrees 2 to 12 in
    three and four dimensions. They agree to 4.9e-16 relative. If the two ever
    diverge, that test is what says so; without it this file could drift into
    documenting a matrix the solver does not use.
    """
    bc = BezierCurve(np.zeros((N + 1, dim), dtype=float))
    if bc.G_tilde is None:
        raise ValueError("The smoothness regularizer requires Bezier degree >= 2.")

    spatial_dim = dim - 1
    n_cp = N + 1
    H = np.zeros((n_cp * dim, n_cp * dim), dtype=float)

    for coord_idx in range(spatial_dim):
        for i in range(n_cp):
            for j in range(n_cp):
                H[i * dim + coord_idx, j * dim + coord_idx] += bc.G_tilde[i, j]

    return 0.5 * (H + H.T)


def build_energy_objective(N: int, dim: int) -> np.ndarray:
    """Deprecated alias for :func:`build_smoothness_regularizer`.

    The old name asserted the term measures acceleration energy, which is false
    of it (see the module docstring). Kept so nothing breaks on import; it emits
    a DeprecationWarning.
    """
    import warnings

    warnings.warn(
        "build_energy_objective measures a parameter-domain smoothness "
        "regularizer, not acceleration energy. Use build_smoothness_regularizer.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_smoothness_regularizer(N, dim)


# ---------------------------------------------------------------------------
# Control-point difference operators (item B8)
#
# These are the D-matrix pattern from orbital_docking.bezier, reused rather than
# rewritten: get_D_matrix(N) is exactly N times the first forward difference, so
# every operator below is built from it.
#
# The relation each one satisfies is EXACT and polynomial -- it is a statement
# about the curve parameter, never about physical time -- which is why it can be
# unit-tested against numerical differentiation to near machine precision.
# ---------------------------------------------------------------------------


def first_difference_matrix(N: int) -> np.ndarray:
    """Forward first differences of control points: row i is ``P[i+1] - P[i]``.

    Shape ``(N, N+1)``. This is ``get_D_matrix(N) / N`` -- the raw gap, with the
    degree factor stripped off, because the slant limit (item B9) is written on
    gaps and not on derivatives.
    """
    N = int(N)
    if N < 1:
        raise ValueError("first_difference_matrix requires degree >= 1.")
    return get_D_matrix(N) / float(N)


def second_difference_matrix(N: int) -> np.ndarray:
    """Forward second differences: row i is ``P[i+2] - 2 P[i+1] + P[i]``.

    Shape ``(N-1, N+1)``. Composed from two first-difference matrices, so it
    inherits the D-matrix convention rather than restating the stencil.
    """
    N = int(N)
    if N < 2:
        raise ValueError("second_difference_matrix requires degree >= 2.")
    return first_difference_matrix(N - 1) @ first_difference_matrix(N)


def derivative_control_point_matrix(N: int) -> np.ndarray:
    """Map control points to the control points of the parameter derivative.

    The derivative of a degree-``N`` Bezier is the degree-``N-1`` Bezier whose
    control points are ``N (P[i+1] - P[i])``. Shape ``(N, N+1)``; identical to
    ``get_D_matrix(N)`` and re-exported here so callers of this module do not
    have to know that.
    """
    N = int(N)
    if N < 1:
        raise ValueError("derivative_control_point_matrix requires degree >= 1.")
    return get_D_matrix(N)


def second_derivative_control_point_matrix(N: int) -> np.ndarray:
    """Map control points to the control points of the second parameter derivative.

    Degree ``N-2``, control points ``N (N-1) (P[i+2] - 2 P[i+1] + P[i])``.
    Shape ``(N-1, N+1)``.
    """
    N = int(N)
    if N < 2:
        raise ValueError("second_derivative_control_point_matrix requires degree >= 2.")
    return get_D_matrix(N - 1) @ get_D_matrix(N)


def lift_operator(M: np.ndarray, dim: int) -> np.ndarray:
    """Apply a control-point operator coordinate-wise to a flattened polygon.

    The optimizer's variable vector is the control polygon flattened row-major,
    ``x[i*dim + d]``. This returns ``kron(M, I_dim)``, the same operator acting
    on that vector and producing a flattened ``(rows, dim)`` result -- which is
    the shape the Rust constraint rows are written in.
    """
    M = np.asarray(M, dtype=float)
    return np.kron(M, np.eye(int(dim), dtype=float))


def build_initial_guess(p_start, p_end, n_cp: int, init_curve: dict | None = None) -> np.ndarray:
    """
    Build the initial control polygon for SCP.

    The default is a straight line in space-time. For 2D spatial demos we can
    optionally add a quadratic-looking lateral bow so the initial curve swings
    toward a corner instead of passing through the middle of the workspace.
    """
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    s_vals = np.linspace(0.0, 1.0, int(n_cp))
    P = np.array([(1.0 - s) * p_start + s * p_end for s in s_vals], dtype=float)

    if not init_curve or init_curve.get("mode", "straight") == "straight":
        return P

    if init_curve.get("mode") != "quadratic_bow":
        raise ValueError(f"Unknown init_curve mode: {init_curve.get('mode')}")

    spatial_dim = p_start.size - 1
    if spatial_dim < 2:
        return P

    bow = float(init_curve.get("bow", 0.0))
    if bow <= 0.0:
        return P

    chord = p_end[:2] - p_start[:2]
    chord_norm = np.linalg.norm(chord)
    if chord_norm < 1e-12:
        return P

    normal = np.array([-chord[1], chord[0]], dtype=float) / chord_norm
    workspace_center = init_curve.get("workspace_center")
    if workspace_center is not None:
        workspace_center = np.asarray(workspace_center, dtype=float)
        midpoint = 0.5 * (p_start[:2] + p_end[:2])
        if np.dot(workspace_center - midpoint, normal) > 0.0:
            normal *= -1.0
    normal *= float(init_curve.get("side", 1.0))

    bump = bow * (4.0 * s_vals * (1.0 - s_vals))
    P[:, :2] += bump[:, None] * normal
    return P
