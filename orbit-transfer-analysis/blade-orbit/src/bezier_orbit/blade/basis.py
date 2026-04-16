"""BLADE 대수적 구조: Gram 행렬, 적분, 순차 전파, 연속성.

이론: efc005 (BLADE: Bernstein Local Adaptive Degree Elements)

핵심 구조:
- 블록 대각 Gram: G_BLADE = blkdiag(Δ₁ G_{n₁}, ..., Δ_K G_{n_K})
- 세그먼트 순차 적분: Δv_k, Δr_k 누적
- 선택적 연속성: C⁰, C¹ 등 세그먼트 경계 제약
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la

from bezier_orbit.bezier.basis import (
    gram_matrix,
    int_matrix,
    double_int_matrix,
)


# ── 블록 대각 Gram 행렬 ──────────────────────────────────────────

def blade_gram(K: int, n: int, deltas: NDArray | None = None) -> NDArray:
    """블록 대각 Gram 행렬.

    G_BLADE = blkdiag(Δ₁ G_n, Δ₂ G_n, ..., Δ_K G_n)

    Parameters
    ----------
    K : 세그먼트 수.
    n : 세그먼트 차수 (균일).
    deltas : (K,) 세그먼트 길이. None이면 균일 Δ=1/K.

    Returns
    -------
    G : (K(n+1), K(n+1))
    """
    if deltas is None:
        deltas = np.full(K, 1.0 / K)
    G_n = gram_matrix(n)
    dim = (n + 1) * K
    G = np.zeros((dim, dim))
    for k in range(K):
        s = k * (n + 1)
        e = s + (n + 1)
        G[s:e, s:e] = deltas[k] * G_n
    return G


# ── 세그먼트 속도·위치 변화 ───────────────────────────────────────

def segment_delta_v(p_k: NDArray, n: int, delta_k: float, t_f: float) -> NDArray:
    """세그먼트 k의 속도 변화.

    Δv_k = t_f · Δ_k · e_{n+1}^T · I_n · p_k

    Parameters
    ----------
    p_k : (n+1,) 또는 (n+1, d) 제어점.
    n : 세그먼트 차수.
    delta_k : 세그먼트 길이.
    t_f : 비행시간.

    Returns
    -------
    dv : 스칼라 또는 (d,)
    """
    I_n = int_matrix(n)
    e_last = I_n[-1, :]  # (n+1,)
    return t_f * delta_k * (e_last @ p_k)


def segment_delta_r(
    p_k: NDArray, v_entry: NDArray, n: int, delta_k: float, t_f: float,
) -> NDArray:
    """세그먼트 k의 위치 변화.

    Δr_k = v_entry · t_f · Δ_k + t_f² · Δ_k² · e_{n+2}^T · Ī_n · p_k

    Parameters
    ----------
    p_k : (n+1,) 또는 (n+1, d) 제어점.
    v_entry : 세그먼트 진입 속도.
    n : 세그먼트 차수.
    delta_k : 세그먼트 길이.
    t_f : 비행시간.
    """
    Ibar_n = double_int_matrix(n)
    e_last = Ibar_n[-1, :]  # (n+1,)
    return v_entry * t_f * delta_k + t_f**2 * delta_k**2 * (e_last @ p_k)


# ── 순차 전파 ────────────────────────────────────────────────────

def forward_propagate(
    p_segments: list[NDArray],
    n: int,
    deltas: NDArray,
    t_f: float,
    r0: float | NDArray,
    v0: float | NDArray,
) -> tuple[NDArray, NDArray]:
    """세그먼트 순차 전파로 분할점 상태 계산.

    Parameters
    ----------
    p_segments : 길이 K 리스트, 각 원소 (n+1,) 또는 (n+1, d).
    n : 세그먼트 차수.
    deltas : (K,) 세그먼트 길이.
    t_f : 비행시간.
    r0, v0 : 초기 상태.

    Returns
    -------
    v_nodes : (K+1, ...) 분할점 속도.
    r_nodes : (K+1, ...) 분할점 위치.
    """
    K = len(p_segments)
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    shape = r0.shape
    v_nodes = np.zeros((K + 1, *shape))
    r_nodes = np.zeros((K + 1, *shape))
    v_nodes[0] = v0
    r_nodes[0] = r0

    for k in range(K):
        dv = segment_delta_v(p_segments[k], n, deltas[k], t_f)
        dr = segment_delta_r(p_segments[k], v_nodes[k], n, deltas[k], t_f)
        v_nodes[k + 1] = v_nodes[k] + dv
        r_nodes[k + 1] = r_nodes[k] + dr

    return v_nodes, r_nodes


# ── 경계조건 행렬 ────────────────────────────────────────────────

def boundary_matrices(
    K: int, n: int, deltas: NDArray, t_f: float,
) -> tuple[NDArray, NDArray]:
    """종말 경계조건 행렬 (속도, 위치).

    속도: Σ_k t_f Δ_k e_I^T p_k = v_f - v_0
    위치: Σ_k [v(t_{k-1}) Δ_k + t_f² Δ_k² e_Ī^T p_k] = r_f - r_0

    위치 경계조건은 p에 대해 비선형(v(t_{k-1})이 이전 세그먼트의 p에 의존)이지만,
    이중 적분기에서는 명시적으로 전개할 수 있다.

    Returns
    -------
    A_v : (1, K(n+1)) — 속도 경계조건 행
    A_r : (1, K(n+1)) — 위치 경계조건 행
    """
    dim = K * (n + 1)
    I_n = int_matrix(n)
    Ibar_n = double_int_matrix(n)
    e_I = I_n[-1, :]        # (n+1,)
    e_Ibar = Ibar_n[-1, :]  # (n+1,)

    # 속도: Σ_k t_f Δ_k e_I^T p_k
    A_v = np.zeros(dim)
    for k in range(K):
        s = k * (n + 1)
        A_v[s:s + n + 1] = t_f * deltas[k] * e_I

    # 위치: 순차 누적 전개
    # Δr_k = v(t_{k-1}) · t_f · Δ_k + t_f² · Δ_k² · e_Ī^T · p_k
    # v(t_{k-1}) = v_0 + Σ_{j<k} t_f Δ_j e_I^T p_j
    # → r(1) - r_0 - v_0·t_f = Σ_k [(Σ_{j<k} t_f Δ_j e_I^T p_j)·t_f·Δ_k + t_f² Δ_k² e_Ī^T p_k]
    # = Σ_j t_f² Δ_j (Σ_{k>j} Δ_k) e_I^T p_j + Σ_k t_f² Δ_k² e_Ī^T p_k
    # 세그먼트 j의 계수: t_f² Δ_j (Σ_{k>j} Δ_k) e_I + t_f² Δ_j² e_Ī
    A_r = np.zeros(dim)
    suffix_sum = np.array([np.sum(deltas[j + 1:]) for j in range(K)])

    for j in range(K):
        s = j * (n + 1)
        A_r[s:s + n + 1] = (
            t_f**2 * deltas[j] * suffix_sum[j] * e_I
            + t_f**2 * deltas[j]**2 * e_Ibar
        )

    return A_v.reshape(1, -1), A_r.reshape(1, -1)


# ── 연속성 제약 ──────────────────────────────────────────────────

def continuity_matrices(
    K: int, n: int, deltas: NDArray | None = None, order: int = 0,
) -> NDArray:
    """세그먼트 간 연속성 제약 행렬.

    order = 0 (C⁰): p_{k,n} = p_{k+1,0}  → (K-1)개 등식
    order = 1 (C¹): + n/Δ_k(p_{k,n}-p_{k,n-1}) = n/Δ_{k+1}(p_{k+1,1}-p_{k+1,0})

    Parameters
    ----------
    K : 세그먼트 수.
    n : 세그먼트 차수.
    deltas : (K,) 세그먼트 길이.
    order : 연속성 차수 (0 또는 1).

    Returns
    -------
    A_cont : (n_eq, K(n+1)) 연속성 행렬.
        A_cont @ z = 0 형태.
    """
    if n < 1 and order >= 0:
        # n=0이면 C⁰ 연속성은 q_k = q_{k+1}: 값 제약
        dim = K * (n + 1)
        n_eq = K - 1
        A = np.zeros((n_eq, dim))
        for j in range(K - 1):
            A[j, j] = 1.0       # p_{k,0} = q_k
            A[j, j + 1] = -1.0  # -p_{k+1,0} = -q_{k+1}
        return A

    if deltas is None:
        deltas = np.full(K, 1.0 / K)

    dim = K * (n + 1)
    rows = []

    # C⁰: p_{k,n} = p_{k+1,0}
    if order >= 0:
        for j in range(K - 1):
            row = np.zeros(dim)
            row[j * (n + 1) + n] = 1.0         # p_{k, n}
            row[(j + 1) * (n + 1)] = -1.0       # -p_{k+1, 0}
            rows.append(row)

    # C¹: n/Δ_k (p_{k,n} - p_{k,n-1}) = n/Δ_{k+1} (p_{k+1,1} - p_{k+1,0})
    if order >= 1 and n >= 2:
        for j in range(K - 1):
            row = np.zeros(dim)
            coeff_left = n / deltas[j]
            coeff_right = n / deltas[j + 1]
            sk = j * (n + 1)
            sk1 = (j + 1) * (n + 1)
            row[sk + n] = coeff_left       # p_{k,n}
            row[sk + n - 1] = -coeff_left  # -p_{k,n-1}
            row[sk1 + 1] = -coeff_right    # -p_{k+1,1}
            row[sk1] = coeff_right          # p_{k+1,0}
            rows.append(row)

    return np.array(rows) if rows else np.zeros((0, dim))
