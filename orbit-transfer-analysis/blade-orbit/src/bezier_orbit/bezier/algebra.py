"""Bernstein 대수 연산 모듈.

Bernstein 기저 위에서의 곱(product), 합성(composition), 차수 올림/축소,
비선형 함수 근사를 제어점 벡터공간 내 대수적 연산으로 제공한다.

이론: docs/reports/010_bernstein_algebra/

주요 연산:
- bern_product    : f∈B_N × g∈B_M → (f·g)∈B_{N+M}
- bern_compose    : h∈B_K ∘ g∈B_M → (h∘g)∈B_{KM}
- degree_elevate  : B_N → B_R (R ≥ N, 무손실)
- degree_reduce   : B_Q → B_R (R < Q, L² 최적)
- chebyshev_bernstein_approx : 스칼라 함수의 K차 Bernstein 근사

성능 노트:
  모든 핵심 연산이 numpy 벡터 연산으로 구현되어 있다.
  가중치 행렬·Toeplitz 인덱스·Chebyshev 근사는 LRU 캐싱한다.
"""

from __future__ import annotations

import functools
import math
import warnings
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import comb as _scipy_comb

from .basis import bernstein, gram_matrix


# ── 캐싱된 가중치 행렬 및 인덱스 ──────────────────────────────


@functools.lru_cache(maxsize=256)
def _product_weights(N: int, M: int) -> NDArray:
    """Bernstein 곱 가중치 행렬 W (캐싱).

    W[k, j] = C(N,j)·C(M,k-j) / C(N+M,k)

    Returns (N+M+1, N+1).
    """
    K_out = N + M
    js = np.arange(N + 1)
    ks = np.arange(K_out + 1)[:, np.newaxis]
    ls = ks - js

    valid = (ls >= 0) & (ls <= M)

    cN = _scipy_comb(N, js, exact=False)
    cM = _scipy_comb(M, np.clip(ls, 0, M), exact=False)
    cNM = _scipy_comb(K_out, ks.ravel(), exact=False)[:, np.newaxis]

    return np.where(valid, cN * cM / cNM, 0.0)


@functools.lru_cache(maxsize=256)
def _toeplitz_indices(N: int, M: int) -> tuple[NDArray, NDArray]:
    """Bernstein 곱의 Toeplitz 인덱스 (캐싱).

    Q_shifted 구성에 필요한 (valid_mask, clip_indices)를 반환.
    """
    ks = np.arange(N + M + 1)[:, np.newaxis]
    js = np.arange(N + 1)[np.newaxis, :]
    ls = ks - js
    valid = (ls >= 0) & (ls <= M)
    clip = np.clip(ls, 0, M)
    return valid, clip


@functools.lru_cache(maxsize=64)
def _gram_cholesky(R: int) -> tuple:
    """정규화된 Gram 행렬의 Cholesky 분해 (캐싱).

    G_R + λI 를 Cholesky 분해하여 반환.
    λ = eps_machine · trace(G_R) / (R+1) 으로 최소 고유값을 바닥에서 들어올림.
    """
    G_R = gram_matrix(R)
    lam = np.finfo(float).eps * np.trace(G_R) / (R + 1)
    G_reg = G_R + lam * np.eye(R + 1)
    return cho_factor(G_reg)


@functools.lru_cache(maxsize=64)
def _cross_gram(R: int, Q: int) -> NDArray:
    """교차 Gram 행렬 C_{Q→R} (캐싱).

    Returns (R+1, Q+1).
    """
    is_ = np.arange(R + 1)[:, np.newaxis]
    js = np.arange(Q + 1)[np.newaxis, :]

    cR = _scipy_comb(R, is_, exact=False)
    cQ = _scipy_comb(Q, js, exact=False)
    cRQ = _scipy_comb(R + Q, is_ + js, exact=False)

    return cR * cQ / (cRQ * (R + Q + 1))


@functools.lru_cache(maxsize=64)
def _chebyshev_approx_cached(K: int, a_round: float, b_round: float) -> NDArray:
    """(a + (b-a)·s)^{-3/2} 근사의 캐싱 버전 (반올림된 [a,b] 구간)."""
    M_sample = max(3 * K, 30)
    nodes_01 = 0.5 * (1.0 - np.cos(np.pi * np.arange(M_sample + 1) / M_sample))
    s_nodes = a_round + (b_round - a_round) * nodes_01
    # a_round > 0 보장이므로 안전
    f_vals = s_nodes ** (-1.5)
    B = bernstein(K, nodes_01)
    coeffs, _, _, _ = np.linalg.lstsq(B, f_vals, rcond=None)
    return coeffs


# ── Bernstein 곱 내부 헬퍼 ────────────────────────────────────


def _weighted_p(N: int, M: int, p: NDArray) -> NDArray:
    """W * p[newaxis,:] 융합 — 3-way broadcast 대신 2-way einsum 가능."""
    W = _product_weights(N, M)
    return W * p[np.newaxis, :]


def _build_Q_shifted(q: NDArray, valid: NDArray, clip: NDArray) -> NDArray:
    """q의 Toeplitz-shifted 행렬 구성 (np.append 제거)."""
    M = len(q) - 1
    Q_raw = q[np.minimum(clip, M)]  # 새 배열 (advanced indexing → 복사)
    Q_raw[~valid] = 0.0
    return Q_raw


# ── Bernstein 곱 ─────────────────────────────────────────────


def bern_product(p: NDArray, q: NDArray) -> NDArray:
    """두 Bernstein 다항식의 곱의 제어점 (벡터화).

    r_k = Σ_{j} W[k,j] · p_j · q_{k-j}

    Parameters
    ----------
    p : (N+1,) — f의 제어점
    q : (M+1,) — g의 제어점

    Returns
    -------
    r : (N+M+1,) — (f·g)의 제어점
    """
    N = len(p) - 1
    M = len(q) - 1
    Wp = _product_weights(N, M) * p[np.newaxis, :]  # (N+M+1, N+1)

    # W가 invalid 위치에서 0이므로 Q_shifted 복사/마스킹 불필요
    _, clip = _toeplitz_indices(N, M)
    return np.einsum("kj,kj->k", Wp, q[clip])


def bern_product_with_jacobian(
    p: NDArray, q: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Bernstein 곱 + Jacobian (벡터화).

    r = p ⊗_B q 와 동시에 ∂r/∂p, ∂r/∂q를 반환한다.
    Jacobian은 곱 계산의 부산물이므로 추가 비용이 거의 없다.

    Returns
    -------
    r : (N+M+1,)
    Jr_p : (N+M+1, N+1)  — ∂r/∂p
    Jr_q : (N+M+1, M+1)  — ∂r/∂q
    """
    N = len(p) - 1
    M = len(q) - 1
    W_pq = _product_weights(N, M)
    valid_pq, clip_pq = _toeplitz_indices(N, M)

    Q_shifted = _build_Q_shifted(q, valid_pq, clip_pq)  # (N+M+1, N+1)
    r = np.einsum("kj,kj->k", W_pq * p[np.newaxis, :], Q_shifted)

    # ∂r_k/∂p_j = W[k,j] · q[k-j]
    Jr_p = W_pq * Q_shifted  # (N+M+1, N+1)

    # ∂r_k/∂q_l = W_qp[k,l] · p[k-l]  (역방향 Toeplitz)
    W_qp = _product_weights(M, N)
    valid_qp, clip_qp = _toeplitz_indices(M, N)
    P_shifted = _build_Q_shifted(p, valid_qp, clip_qp)  # (N+M+1, M+1)
    Jr_q = W_qp * P_shifted  # (N+M+1, M+1)

    return r, Jr_p, Jr_q


# ── Bernstein 합성 ───────────────────────────────────────────


def bern_compose(h: NDArray, g: NDArray) -> NDArray:
    """Bernstein 합성 (h ∘ g)(τ) = h(g(τ))  (copy-free 최적화).

    h(g) = Σ_{i=0}^{K} h_i · C(K,i) · g^i · (1-g)^{K-i}

    g^i · (1-g)^{K-i}의 차수는 항상 K·M이므로 degree_elevate 불필요.
    W 행렬이 invalid 위치에 0을 가지므로 Q_shifted 복사/마스킹 생략.

    Parameters
    ----------
    h : (K+1,) — 외부 함수의 제어점
    g : (M+1,) — 내부 함수의 제어점

    Returns
    -------
    r : (KM+1,) — (h∘g)의 제어점
    """
    K_h = len(h) - 1
    M = len(g) - 1

    if K_h == 0:
        return np.array([h[0]])

    target_deg = K_h * M
    one_minus_g = np.ones(M + 1) - g

    # g^i, (1-g)^i 순차 곱 — copy-free: W가 invalid를 0으로 처리
    g_pow = [np.ones(1)]
    omg_pow = [np.ones(1)]

    for i in range(1, K_h + 1):
        # g_pow[i] = g_pow[i-1] ⊗ g
        p_prev = g_pow[-1]
        Np = len(p_prev) - 1
        Wp = _product_weights(Np, M) * p_prev[np.newaxis, :]
        _, clip = _toeplitz_indices(Np, M)
        g_pow.append(np.einsum("kj,kj->k", Wp, g[clip]))

        # omg_pow[i] = omg_pow[i-1] ⊗ (1-g)
        p_prev = omg_pow[-1]
        Np = len(p_prev) - 1
        Wp = _product_weights(Np, M) * p_prev[np.newaxis, :]
        _, clip = _toeplitz_indices(Np, M)
        omg_pow.append(np.einsum("kj,kj->k", Wp, one_minus_g[clip]))

    # 이항계수 사전 계산
    binom_K = np.array([math.comb(K_h, i) for i in range(K_h + 1)], dtype=float)

    # 합산 — g^i·(1-g)^{K-i}는 항상 차수 K·M
    r = np.zeros(target_deg + 1)
    for i in range(K_h + 1):
        p_g = g_pow[i]
        p_o = omg_pow[K_h - i]
        Ng = len(p_g) - 1
        No = len(p_o) - 1
        Wp = _product_weights(Ng, No) * p_g[np.newaxis, :]
        _, clip = _toeplitz_indices(Ng, No)
        term = np.einsum("kj,kj->k", Wp, p_o[clip])
        r += (h[i] * binom_K[i]) * term

    return r


# ── 차수 올림 ────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def _elevation_matrix(N: int, R: int) -> NDArray:
    """B_N → B_R 차수 올림 행렬 (closed-form, 캐싱).

    E[k, j] = C(N, j) · C(R-N, k-j) / C(R, k)

    Returns (R+1, N+1).
    """
    ks = np.arange(R + 1)[:, np.newaxis]
    js = np.arange(N + 1)[np.newaxis, :]
    d = ks - js  # (R+1, N+1)

    valid = (d >= 0) & (d <= R - N)
    cN = _scipy_comb(N, js, exact=False)
    cD = _scipy_comb(R - N, np.clip(d, 0, R - N), exact=False)
    cR = _scipy_comb(R, ks.ravel(), exact=False)[:, np.newaxis]

    return np.where(valid, cN * cD / cR, 0.0)


def degree_elevate(p: NDArray, target_deg: int) -> NDArray:
    """Bernstein 차수 올림 (무손실): B_N → B_{target_deg}.

    Closed-form 이항계수로 O(1) 행렬-벡터 곱.
    행렬은 (N, target_deg) 쌍별 캐싱.

    Parameters
    ----------
    p : (N+1,)
    target_deg : int ≥ N

    Returns
    -------
    q : (target_deg+1,)
    """
    N = len(p) - 1
    if N == target_deg:
        return p.copy()
    if N > target_deg:
        raise ValueError(f"Cannot elevate from degree {N} to {target_deg}")

    E = _elevation_matrix(N, target_deg)
    return E @ p


# ── 차수 축소 (L² 최적) ─────────────────────────────────────


def degree_reduce(p: NDArray, R: int) -> NDArray:
    """L² 최적 차수 축소 (벡터화): B_Q → B_R.

    Parameters
    ----------
    p : (Q+1,) 또는 (Q+1, d)
    R : int — 목표 차수

    Returns
    -------
    q : (R+1,) 또는 (R+1, d)
    """
    Q = len(p) - 1
    if R >= Q:
        if R == Q:
            return p.copy()
        # degree_elevate는 1D만 지원 → 2D는 열별 적용
        if p.ndim == 1:
            return degree_elevate(p, R)
        return np.column_stack([
            degree_elevate(p[:, k], R) for k in range(p.shape[1])
        ])

    cho = _gram_cholesky(R)  # 정규화 + Cholesky (캐싱)
    C = _cross_gram(R, Q)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return cho_solve(cho, C @ p)


# ── 비선형 함수의 Bernstein 근사 ─────────────────────────────


def chebyshev_bernstein_approx(
    func: Callable[[float], float],
    K: int,
    a: float = 0.0,
    b: float = 1.0,
) -> NDArray:
    """스칼라 함수를 [a,b] 위의 K차 Bernstein 다항식으로 근사.

    Parameters
    ----------
    func : callable — f: R → R
    K : int — Bernstein 근사 차수
    a, b : float — 근사 구간

    Returns
    -------
    h_K : (K+1,) — B_K의 제어점
    """
    M_sample = max(3 * K, 30)
    nodes_01 = 0.5 * (1.0 - np.cos(np.pi * np.arange(M_sample + 1) / M_sample))

    s_nodes = a + (b - a) * nodes_01
    f_vals = np.array([func(s) for s in s_nodes])

    B = bernstein(K, nodes_01)
    coeffs, _, _, _ = np.linalg.lstsq(B, f_vals, rcond=None)
    return coeffs


# ── 다축 배치 연산 ────────────────────────────────────────────


def degree_elevate_multi(p: NDArray, target_deg: int) -> NDArray:
    """다축 Bernstein 차수 올림: (N+1, d) → (target_deg+1, d).

    단일 행렬-행렬 곱으로 모든 축을 한 번에 처리한다.
    """
    N = p.shape[0] - 1
    if N == target_deg:
        return p.copy()
    E = _elevation_matrix(N, target_deg)
    return E @ p  # (target_deg+1, d)


def degree_reduce_multi(p: NDArray, R: int) -> NDArray:
    """다축 Bernstein 차수 축소: (Q+1, d) → (R+1, d).

    단일 Cholesky solve로 모든 축을 한 번에 처리한다.
    """
    Q = p.shape[0] - 1
    if R >= Q:
        if R == Q:
            return p.copy()
        return degree_elevate_multi(p, R)

    cho = _gram_cholesky(R)
    C = _cross_gram(R, Q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return cho_solve(cho, C @ p)  # (R+1, d)


def bern_product_multi(p: NDArray, q: NDArray) -> NDArray:
    """다축 Bernstein 곱: (N+1, d) × (M+1, d) → (N+M+1, d).

    동일 차수 구조의 d개 독립 곱을 한 번의 einsum으로 처리한다.

    Parameters
    ----------
    p : (N+1, d) — d개 다항식의 제어점
    q : (M+1, d) — d개 다항식의 제어점

    Returns
    -------
    r : (N+M+1, d)
    """
    N = p.shape[0] - 1
    M = q.shape[0] - 1
    W = _product_weights(N, M)        # (N+M+1, N+1)
    valid, clip = _toeplitz_indices(N, M)

    # W가 invalid 위치에서 0이므로 마스킹 불필요 (copy-free)
    # Wp: (N+M+1, N+1, d)
    Wp = W[:, :, np.newaxis] * p[np.newaxis, :, :]

    # einsum으로 모든 축 동시 처리
    return np.einsum("kja,kja->ka", Wp, q[clip])  # (N+M+1, d)


def bern_product_rsq(q_r: NDArray) -> NDArray:
    """r²(τ) = x² + y² + z²를 한 번에 계산.

    Parameters
    ----------
    q_r : (P+1, 3) — 위치 제어점

    Returns
    -------
    q_r2 : (2P+1,) — r² 제어점
    """
    # 3축의 자기곱을 배치로 계산한 뒤 합산
    r = bern_product_multi(q_r, q_r)  # (2P+1, 3)
    return r[:, 0] + r[:, 1] + r[:, 2]


def bern_product_broadcast(p_scalar: NDArray, q_multi: NDArray) -> NDArray:
    """스칼라 × 다축 Bernstein 곱: (R+1,) × (R+1, d) → (2R+1, d).

    q_rinv3 같은 스칼라 다항식을 3축 위치에 각각 곱할 때 사용.
    내부적으로 q_multi의 각 축을 독립적으로 곱한다.

    Parameters
    ----------
    p_scalar : (N+1,) — 스칼라 다항식
    q_multi : (M+1, d) — d축 다항식

    Returns
    -------
    r : (N+M+1, d)
    """
    N = len(p_scalar) - 1
    M = q_multi.shape[0] - 1
    d = q_multi.shape[1]
    W = _product_weights(N, M)        # (N+M+1, N+1)
    valid, clip = _toeplitz_indices(N, M)

    # p_scalar의 weighted 버전: (N+M+1, N+1)
    Wp = W * p_scalar[np.newaxis, :]

    # W가 invalid 위치에서 0이므로 마스킹 불필요 (copy-free)

    # einsum: Wp(k,j) * q_multi[clip](k,j,a) → (k, a) — copy-free
    return np.einsum("kj,kja->ka", Wp, q_multi[clip])  # (N+M+1, d)


# ── 중력 합성 파이프라인 ─────────────────────────────────────


def gravity_composition_pipeline(
    q_r: NDArray,
    K: int = 12,
    R: int = 20,
) -> NDArray:
    """위치 제어점으로부터 중력 가속도의 Bernstein 제어점을 계산.

    파이프라인:
    1. r²(τ) = x² + y² + z²
    2. s̃(τ) = (r² - a) / (b - a) → [0,1]
    3. h(s) = (a + (b-a)s)^{-3/2} → K차 Bernstein 근사 (캐싱)
    4. r⁻³(τ) = h(s̃(τ))
    5. a_grav = -r · r⁻³
    6. 차수 축소 → R차

    Parameters
    ----------
    q_r : (P+1, 3) — 위치 궤적의 Bernstein 제어점
    K : int — h(s) 근사 차수
    R : int — 최종 출력 차수

    Returns
    -------
    q_agrav : (R+1, 3) — 중력 가속도의 Bernstein 제어점
    """
    P = len(q_r) - 1

    # Step 1: r²(τ) — 3축 배치 곱
    q_r2 = bern_product_rsq(q_r)

    # Step 2: 아핀 재매개변수화
    a_val = float(np.min(q_r2))
    b_val = float(np.max(q_r2))
    if b_val - a_val < 1e-14:
        r_const = np.sqrt(a_val)
        agrav_const = -q_r / r_const**3
        return degree_reduce(agrav_const, R) if P > R else agrav_const

    q_stilde = (q_r2 - a_val) / (b_val - a_val)

    # Step 3: h(s) 근사 — 반올림 [a,b]로 캐싱
    if np.isfinite(a_val) and a_val > 0 and np.isfinite(b_val):
        sig = max(3, int(-np.log10(a_val)))
        a_r = round(a_val, sig)
        b_r = round(b_val, sig)
        if a_r <= 0:
            a_r = a_val
        if b_r <= a_r:
            b_r = b_val
        try:
            h_K = _chebyshev_approx_cached(K, a_r, b_r)
        except Exception:
            h_K = chebyshev_bernstein_approx(
                lambda s: (a_val + (b_val - a_val) * s) ** (-1.5), K,
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            h_K = chebyshev_bernstein_approx(
                lambda s: (a_val + (b_val - a_val) * s) ** (-1.5), K,
            )

    # Step 4: 합성
    q_rinv3 = bern_compose(h_K, q_stilde)
    q_rinv3 = degree_reduce(q_rinv3, R)

    # Step 5: a_grav = -r · r⁻³ — 3축 배치 곱
    q_r_elev = degree_elevate_multi(q_r, R)  # (R+1, 3)
    q_agrav = -bern_product_broadcast(q_rinv3, q_r_elev)  # (2R+1, 3)

    # Step 6: 최종 차수 축소
    q_agrav = degree_reduce_multi(q_agrav, R)

    return q_agrav


def gravity_pipeline_jacobian(
    q_r: NDArray,
    K: int = 12,
    R: int = 20,
) -> tuple[NDArray, NDArray]:
    """중력 파이프라인의 Jacobian ∂vec(q_agrav)/∂vec(q_r).

    gravity_composition_pipeline과 동일한 파이프라인을 실행하면서,
    bern_product_with_jacobian으로 각 단계의 Jacobian을 추적하여
    체인 룰로 전체 Jacobian을 조립한다.

    Parameters
    ----------
    q_r : (P+1, 3) — 위치 Bernstein 제어점
    K, R : int — 파이프라인 파라미터

    Returns
    -------
    q_agrav : (R+1, 3) — 중력 가속도 제어점
    J : (3(R+1), 3(P+1)) — ∂vec(q_agrav)/∂vec(q_r)
    """
    P = len(q_r) - 1
    dim_in = 3 * (P + 1)
    dim_out = 3 * (R + 1)

    # ── Step 1: r²(τ) = x² + y² + z² ──
    # q_r2 = Σ_k p_k ⊗ p_k  where p_k = q_r[:, k]
    # ∂q_r2/∂p_k = 2 * (W · P_shifted)  (자기 곱의 미분)
    q_r2 = np.zeros(2 * P + 1)
    # Jacobian of q_r2 w.r.t. vec(q_r): (2P+1, 3(P+1))
    J_r2 = np.zeros((2 * P + 1, dim_in))

    for k in range(3):
        pk = q_r[:, k]
        rk, Jr_p, Jr_q = bern_product_with_jacobian(pk, pk)
        q_r2 += rk
        # 자기 곱: ∂(p⊗p)/∂p = Jr_p + Jr_q (= 2 * Jr_p for symmetric)
        J_self = Jr_p + Jr_q  # (2P+1, P+1)
        # vec(q_r)에서 k번째 성분 블록: 열 k*(P+1) ~ (k+1)*(P+1)
        J_r2[:, k * (P + 1):(k + 1) * (P + 1)] = J_self

    # ── Step 2: q̃ = (q_r2 - a) / (b - a) ──
    a_val = float(np.min(q_r2))
    b_val = float(np.max(q_r2))
    if b_val - a_val < 1e-14:
        # 원궤도: 단순 처리
        r_const = np.sqrt(a_val)
        agrav_const = -q_r / r_const**3
        q_agrav = degree_reduce(agrav_const, R) if P > R else agrav_const
        # Jacobian: -(1/r³) * I (근사)
        J_simple = np.zeros((dim_out, dim_in))
        for k in range(3):
            for i in range(R + 1):
                if i < P + 1:
                    J_simple[k * (R + 1) + i, k * (P + 1) + i] = -1.0 / r_const**3
        return q_agrav, J_simple

    scale = 1.0 / (b_val - a_val)
    q_stilde = (q_r2 - a_val) * scale
    # ∂q̃/∂q_r = scale * ∂q_r2/∂q_r  (a, b 고정)
    J_stilde = scale * J_r2  # (2P+1, 3(P+1))

    # ── Step 3: h_K 근사 (상수, 미분 불필요) ──
    if np.isfinite(a_val) and a_val > 0 and np.isfinite(b_val):
        sig = max(3, int(-np.log10(a_val)))
        a_r = round(a_val, sig)
        b_r = round(b_val, sig)
        if a_r <= 0:
            a_r = a_val
        if b_r <= a_r:
            b_r = b_val
        try:
            h_K_arr = _chebyshev_approx_cached(K, a_r, b_r)
        except Exception:
            h_K_arr = chebyshev_bernstein_approx(
                lambda s: (a_val + (b_val - a_val) * s) ** (-1.5), K,
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            h_K_arr = chebyshev_bernstein_approx(
                lambda s: (a_val + (b_val - a_val) * s) ** (-1.5), K,
            )

    # ── Step 4: 합성 h(q̃) — Jacobian을 체인 룰로 추적 ──
    M = len(q_stilde) - 1  # 2P
    target_deg = K * M
    one_minus_g = np.ones(M + 1) - q_stilde

    # g^i, (1-g)^i 및 각각의 ∂/∂g Jacobian
    g_pow = [np.ones(1)]
    omg_pow = [np.ones(1)]
    Jg_pow = [np.zeros((1, M + 1))]   # ∂g^i/∂g
    Jomg_pow = [np.zeros((1, M + 1))]  # ∂(1-g)^i/∂g

    I_g = np.eye(M + 1)  # ∂g/∂g = I
    I_omg = -np.eye(M + 1)  # ∂(1-g)/∂g = -I

    for i in range(1, K + 1):
        # g^i = g^{i-1} ⊗ g
        gi, Jr_prev, Jr_g = bern_product_with_jacobian(g_pow[-1], q_stilde)
        # ∂g^i/∂g = Jr_prev · ∂g^{i-1}/∂g + Jr_g · I
        Jgi = Jr_prev @ Jg_pow[-1] + Jr_g
        g_pow.append(gi)
        Jg_pow.append(Jgi)

        # (1-g)^i = (1-g)^{i-1} ⊗ (1-g)
        oi, Jr_prev_o, Jr_omg = bern_product_with_jacobian(omg_pow[-1], one_minus_g)
        # ∂(1-g)^i/∂g = Jr_prev_o · ∂(1-g)^{i-1}/∂g + Jr_omg · (-I)
        Joi = Jr_prev_o @ Jomg_pow[-1] + Jr_omg @ I_omg
        omg_pow.append(oi)
        Jomg_pow.append(Joi)

    # 합산: q_rinv3 = Σ h_i C(K,i) · (g^i ⊗ (1-g)^{K-i})
    q_rinv3_full = np.zeros(target_deg + 1)
    J_rinv3_g = np.zeros((target_deg + 1, M + 1))  # ∂q_rinv3/∂q̃

    for i in range(K + 1):
        # term = g^i ⊗ (1-g)^{K-i}
        term, Jr_gi, Jr_oi = bern_product_with_jacobian(g_pow[i], omg_pow[K - i])
        # ∂term/∂g = Jr_gi · ∂g^i/∂g + Jr_oi · ∂(1-g)^{K-i}/∂g
        J_term = Jr_gi @ Jg_pow[i] + Jr_oi @ Jomg_pow[K - i]

        coeff = h_K_arr[i] * math.comb(K, i)
        # term은 이미 차수 K*M = target_deg
        q_rinv3_full += coeff * term
        J_rinv3_g += coeff * J_term

    # degree_reduce: target_deg → R
    cho_R = _gram_cholesky(R)
    C_red = _cross_gram(R, target_deg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q_rinv3 = cho_solve(cho_R, C_red @ q_rinv3_full)
    # ∂reduce(x)/∂x = (G_R+λI)^{-1} C_red (선형)
    J_reduce = cho_solve(cho_R, C_red)  # (R+1, target_deg+1)

    # ∂q_rinv3/∂q̃ = J_reduce · J_rinv3_g
    J_rinv3_stilde = J_reduce @ J_rinv3_g  # (R+1, 2P+1)

    # ∂q_rinv3/∂vec(q_r) = J_rinv3_stilde · J_stilde
    J_rinv3_qr = J_rinv3_stilde @ J_stilde  # (R+1, 3(P+1))

    # ── Step 5: q_agrav = -elev(q_r[:,k], R) ⊗ q_rinv3(R) ──
    # 결과 차수: R + R = 2R → degree_reduce → R
    C_final = _cross_gram(R, 2 * R)
    J_reduce_final = cho_solve(cho_R, C_final)  # (R+1, 2R+1)

    J_full = np.zeros((dim_out, dim_in))
    q_agrav_cols = []

    for k in range(3):
        pk_elev = degree_elevate(q_r[:, k], R)  # (R+1,)

        # -elev(p_k) ⊗ q_rinv3
        prod, Jr_pk, Jr_ri = bern_product_with_jacobian(pk_elev, q_rinv3)
        prod_reduced = cho_solve(cho_R, C_final @ (-prod))

        # degree_elevate Jacobian: E (R+1, P+1) bidiagonal
        q_temp = q_r[:, k].copy()
        E = np.zeros((R + 1, P + 1))
        for j in range(P + 1):
            e_j = np.zeros(P + 1)
            e_j[j] = 1.0
            E[:, j] = degree_elevate(e_j, R)

        # ∂prod_reduced/∂q_r[:,k]:
        #   = J_reduce_final · (-Jr_pk · E)  (through position)
        #   + J_reduce_final · (-Jr_ri) · J_rinv3_qr  (through q_rinv3 dependency on ALL q_r)
        #   But q_rinv3's dependency on q_r is via r², which involves all 3 components.
        #   The Jr_ri part contributes to ALL columns of J_full.

        # Position contribution (only to k-th block)
        J_pos_k = J_reduce_final @ (-Jr_pk @ E)  # (R+1, P+1)
        row_start = k * (R + 1)
        col_start = k * (P + 1)
        J_full[row_start:row_start + R + 1,
               col_start:col_start + P + 1] += J_pos_k

        # q_rinv3 contribution (to all blocks, via r²→q̃→compose)
        J_via_rinv3 = J_reduce_final @ (-Jr_ri) @ J_rinv3_qr  # (R+1, 3(P+1))
        J_full[row_start:row_start + R + 1, :] += J_via_rinv3

        q_agrav_cols.append(prod_reduced)

    q_agrav = np.column_stack(q_agrav_cols)
    return q_agrav, J_full
