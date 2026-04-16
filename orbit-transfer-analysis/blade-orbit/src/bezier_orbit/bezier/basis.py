"""Bernstein 기저함수 및 적분/미분/Gram 행렬.

이론: docs/reports/004_bezier_basis/

주요 행렬:
- B_N(τ)  : (N+1,) Bernstein 기저 벡터
- D_N     : (N, N+1) 미분 행렬
- I_N     : (N+2, N+1) 적분 행렬
- Ibar_N  : (N+3, N+1) 이중 적분 행렬 = I_{N+1} · I_N
- G_N     : (N+1, N+1) Gram 행렬 (대칭 양정치)
"""

from __future__ import annotations

import functools
import math

import numpy as np
from numpy.typing import NDArray


# ── Bernstein 기저 벡터 ─────────────────────────────────────────
def bernstein(N: int, tau: float | NDArray) -> NDArray:
    """N차 Bernstein 기저 벡터 B_N(τ).

    Parameters
    ----------
    N : int
        베지어 차수.
    tau : float or array-like
        정규화 시간 τ ∈ [0, 1].  스칼라이면 (N+1,) 반환,
        배열이면 (..., N+1) 반환.

    Returns
    -------
    B : ndarray
        Bernstein 기저 값.
    """
    tau = np.asarray(tau, dtype=float)
    scalar = tau.ndim == 0
    tau = np.atleast_1d(tau)

    i = np.arange(N + 1)
    # log-space로 계산하여 큰 N에서도 안정적
    log_binom = _log_binom_vec(N, i)

    # 끝점 마스크 (log(0) 경고 방지)
    interior = (tau > 0.0) & (tau < 1.0)
    tau_safe = np.where(interior, tau, 0.5)  # 끝점은 나중에 덮어쓸 것

    # shape: (len(tau), N+1)
    log_B = (
        log_binom[np.newaxis, :]
        + i[np.newaxis, :] * np.log(tau_safe[:, np.newaxis])
        + (N - i)[np.newaxis, :] * np.log(1.0 - tau_safe[:, np.newaxis])
    )
    B = np.exp(log_B)

    # Subnormal 값 제거 (matmul에서 spurious 경고 방지)
    B[B < 1e-300] = 0.0

    # 끝점 보간 정확성 보장
    B[tau == 0.0] = 0.0
    B[tau == 0.0, 0] = 1.0
    B[tau == 1.0] = 0.0
    B[tau == 1.0, N] = 1.0

    if scalar:
        return B[0]
    return B


def _log_binom_vec(n: int, k: NDArray) -> NDArray:
    """log(C(n, k)) 벡터 계산."""
    return np.array([math.lgamma(n + 1) - math.lgamma(ki + 1) - math.lgamma(n - ki + 1) for ki in k])


def bernstein_eval(N: int, P: NDArray, tau: float | NDArray) -> NDArray:
    """베지어 곡선 값 계산: q(τ) = B_N(τ)^T · P.

    Parameters
    ----------
    N : int
        베지어 차수.
    P : ndarray, shape (N+1,) or (N+1, d)
        제어점 행렬.
    tau : float or array
        정규화 시간.

    Returns
    -------
    q : ndarray
        곡선 값. 스칼라 τ이면 (d,) 또는 스칼라, 배열이면 (..., d).
    """
    B = bernstein(N, tau)  # (..., N+1)
    return np.dot(B, P)


# ── 미분 행렬 D_N ───────────────────────────────────────────────
@functools.lru_cache(maxsize=32)
def diff_matrix(N: int) -> NDArray:
    """미분 행렬 D_N = N · Δ_N.

    D_N ∈ R^{N × (N+1)}, 전향 차분 행렬에 N을 곱한 것.
    q'(τ) = B_{N-1}(τ)^T · D_N · p
    """
    D = np.zeros((N, N + 1))
    for i in range(N):
        D[i, i] = -N
        D[i, i + 1] = N
    return D


# ── 적분 행렬 I_N ───────────────────────────────────────────────
@functools.lru_cache(maxsize=32)
def int_matrix(N: int) -> NDArray:
    """적분 행렬 I_N ∈ R^{(N+2) × (N+1)}.

    ∫₀^τ q(s) ds = B_{N+1}(τ)^T · I_N · p

    [I_N]_{k,i} = 1/(N+1)  if k > i,  else 0
    """
    I = np.zeros((N + 2, N + 1))
    for k in range(N + 2):
        for i in range(N + 1):
            if k > i:
                I[k, i] = 1.0 / (N + 1)
    return I


# ── 이중 적분 행렬 Ī_N ──────────────────────────────────────────
@functools.lru_cache(maxsize=32)
def double_int_matrix(N: int) -> NDArray:
    """이중 적분 행렬 Ī_N = I_{N+1} · I_N ∈ R^{(N+3) × (N+1)}.

    ∫₀^τ ∫₀^s q(σ) dσ ds = B_{N+2}(τ)^T · Ī_N · p
    """
    return int_matrix(N + 1) @ int_matrix(N)


# ── Gram 행렬 G_N ───────────────────────────────────────────────
@functools.lru_cache(maxsize=32)
def gram_matrix(N: int) -> NDArray:
    """Gram 행렬 G_N ∈ R^{(N+1) × (N+1)}.

    [G_N]_{i,j} = C(N,i)·C(N,j) / [C(2N, i+j)·(2N+1)]

    대칭 양정치 행렬. 목적함수 J = t_f · p^T G_N p 의 핵심.
    """
    G = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i, N + 1):
            val = math.comb(N, i) * math.comb(N, j) / (math.comb(2 * N, i + j) * (2 * N + 1))
            G[i, j] = val
            G[j, i] = val
    return G


# ── 블록 대각 Hessian ───────────────────────────────────────────
@functools.lru_cache(maxsize=32)
def block_hessian(N: int, d: int = 3) -> NDArray:
    """블록 대각 Hessian H = blkdiag(G_N, ..., G_N).

    J = p^T H p,  p = vec(P_u) ∈ R^{d(N+1)}.

    Parameters
    ----------
    N : int
        베지어 차수.
    d : int
        공간 차원 (기본 3).
    """
    G = gram_matrix(N)
    return np.kron(np.eye(d), G)


# ── 유틸리티: 정적분 (제어점 산술평균) ────────────────────────────
def definite_integral(P: NDArray) -> NDArray:
    """∫₀¹ q(τ) dτ = mean(P), i.e. 제어점의 산술평균.

    Parameters
    ----------
    P : ndarray, shape (N+1,) or (N+1, d)
        제어점.
    """
    return np.mean(P, axis=0)
