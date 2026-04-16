"""BLADE 이중 적분기 문제 정의 및 QP/SOCP 풀이.

이론: efc005 (BLADE: Bernstein Local Adaptive Degree Elements)
"""

from __future__ import annotations

import dataclasses

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from bezier_orbit.bezier.basis import gram_matrix, bernstein

from .basis import (
    blade_gram,
    boundary_matrices,
    continuity_matrices,
    forward_propagate,
)


@dataclasses.dataclass
class BLADEResult:
    """BLADE 풀이 결과."""

    p_segments: list[NDArray]   # 각 세그먼트 최적 제어점
    cost: float                 # 최적 비용
    v_nodes: NDArray            # (K+1, ...) 분할점 속도
    r_nodes: NDArray            # (K+1, ...) 분할점 위치
    coasting_segments: list[int]  # ||p_k|| < ε인 세그먼트 인덱스
    status: str                 # 솔버 상태


@dataclasses.dataclass
class BLADEProblem:
    """BLADE 이중 적분기 문제 정의."""

    r0: float | NDArray         # 초기 위치
    v0: float | NDArray         # 초기 속도
    rf: float | NDArray         # 종말 위치
    vf: float | NDArray         # 종말 속도
    t_f: float                  # 비행시간
    K: int                      # 세그먼트 수
    n: int                      # 세그먼트 차수 (균일)
    deltas: NDArray | None = None  # (K,) 세그먼트 길이
    u_max: float | None = None     # 추력 제약 (None → QP)
    continuity: int = -1           # 연속성 차수 (-1: 없음, 0: C⁰, 1: C¹)
    l1_lambda: float = 0.0         # ℓ₁ 정규화 강도 (0 → 없음)

    def __post_init__(self):
        if self.deltas is None:
            self.deltas = np.full(self.K, 1.0 / self.K)
        self.deltas = np.asarray(self.deltas, dtype=float)


def solve_blade(prob: BLADEProblem) -> BLADEResult:
    """BLADE QP/SOCP 풀이.

    이중 적분기 ẍ = u에 대해, 에너지 최소화 + 경계조건 + 제약.

    Parameters
    ----------
    prob : BLADEProblem

    Returns
    -------
    result : BLADEResult
    """
    K, n = prob.K, prob.n
    dim = K * (n + 1)
    t_f = prob.t_f
    deltas = prob.deltas

    # ── 결정변수 ──
    z = cp.Variable(dim)

    # ── 목적함수: (1/t_f) Σ_k Δ_k p_k^T G_n p_k ──
    G = blade_gram(K, n, deltas)
    cost_energy = (1.0 / t_f) * cp.quad_form(z, G)

    # ℓ₁ 정규화 (코스팅 유도)
    cost_l1 = 0.0
    if prob.l1_lambda > 0:
        for k in range(K):
            s = k * (n + 1)
            cost_l1 = cost_l1 + cp.norm(z[s:s + n + 1], 1)
        cost_l1 = prob.l1_lambda * cost_l1

    objective = cp.Minimize(cost_energy + cost_l1)

    # ── 경계조건 (등식) ──
    A_v, A_r = boundary_matrices(K, n, deltas, t_f)

    r0 = np.asarray(prob.r0, dtype=float).ravel()
    v0 = np.asarray(prob.v0, dtype=float).ravel()
    rf = np.asarray(prob.rf, dtype=float).ravel()
    vf = np.asarray(prob.vf, dtype=float).ravel()

    b_v = (vf - v0).ravel()
    b_r = (rf - r0 - v0 * t_f).ravel()

    constraints = []

    # 스칼라(1D) vs 벡터(d-D) 처리
    if r0.size == 1:
        # 스칼라
        constraints.append(A_v @ z == b_v[0])
        constraints.append(A_r @ z == b_r[0])
    else:
        # 다차원: 축별 독립
        d = r0.size
        dim_d = K * (n + 1)
        for ax in range(d):
            constraints.append(A_v @ z == b_v[ax])  # TODO: 축별 분리 필요
            constraints.append(A_r @ z == b_r[ax])

    # ── 연속성 제약 ──
    if prob.continuity >= 0:
        A_cont = continuity_matrices(K, n, deltas, order=prob.continuity)
        if A_cont.shape[0] > 0:
            constraints.append(A_cont @ z == 0)

    # ── 추력 상자 제약 ──
    if prob.u_max is not None:
        constraints.append(z <= prob.u_max)
        constraints.append(z >= -prob.u_max)

    # ── 풀이 ──
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return BLADEResult(
            p_segments=[np.zeros(n + 1) for _ in range(K)],
            cost=float("inf"),
            v_nodes=np.zeros(K + 1),
            r_nodes=np.zeros(K + 1),
            coasting_segments=[],
            status=problem.status,
        )

    z_opt = z.value

    # ── 결과 추출 ──
    p_segments = []
    for k in range(K):
        s = k * (n + 1)
        p_segments.append(z_opt[s:s + n + 1].copy())

    # 순차 전파
    v_nodes, r_nodes = forward_propagate(
        p_segments, n, deltas, t_f,
        float(prob.r0) if np.asarray(prob.r0).size == 1 else prob.r0,
        float(prob.v0) if np.asarray(prob.v0).size == 1 else prob.v0,
    )

    # 코스팅 세그먼트 식별
    eps_coast = 1e-6
    coasting = [k for k in range(K) if np.linalg.norm(p_segments[k]) < eps_coast]

    return BLADEResult(
        p_segments=p_segments,
        cost=problem.value,
        v_nodes=v_nodes,
        r_nodes=r_nodes,
        coasting_segments=coasting,
        status=problem.status,
    )
