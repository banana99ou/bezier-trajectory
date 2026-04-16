"""Legendre-Gauss-Lobatto (LGL) 노드, 가중치, 미분행렬 계산 모듈."""

import numpy as np


def legendre_poly(N, x):
    """Legendre 다항식 P_N(x) 계산 (Bonnet 재귀)."""
    x = np.asarray(x, dtype=float)
    if N == 0:
        return np.ones_like(x)
    if N == 1:
        return x.copy()
    P_prev = np.ones_like(x)
    P_curr = x.copy()
    for n in range(1, N):
        P_next = ((2 * n + 1) * x * P_curr - n * P_prev) / (n + 1)
        P_prev = P_curr
        P_curr = P_next
    return P_curr


def legendre_poly_derivative(N, x):
    """Legendre 다항식 미분 P'_N(x) 계산."""
    x = np.asarray(x, dtype=float)
    if N == 0:
        return np.zeros_like(x)
    P, dP, _ = _legendre_recurrence(N, x)
    tol = 1e-14
    at_p1 = np.abs(x - 1.0) < tol
    at_m1 = np.abs(x + 1.0) < tol
    dP[at_p1] = N * (N + 1) / 2.0
    dP[at_m1] = (-1.0) ** (N + 1) * N * (N + 1) / 2.0
    return dP


def _legendre_recurrence(N, x):
    """Bonnet 재귀로 P_N, P'_N, P''_N 동시 계산."""
    x = np.asarray(x, dtype=float)
    P0 = np.ones_like(x)
    P1 = x.copy()
    dP0 = np.zeros_like(x)
    dP1 = np.ones_like(x)
    ddP0 = np.zeros_like(x)
    ddP1 = np.zeros_like(x)
    if N == 0:
        return P0, dP0, ddP0
    if N == 1:
        return P1, dP1, ddP1
    for n in range(1, N):
        c1 = (2 * n + 1) / (n + 1)
        c2 = n / (n + 1)
        P2 = c1 * x * P1 - c2 * P0
        dP2 = c1 * (P1 + x * dP1) - c2 * dP0
        ddP2 = c1 * (2.0 * dP1 + x * ddP1) - c2 * ddP0
        P0, P1 = P1, P2
        dP0, dP1 = dP1, dP2
        ddP0, ddP1 = ddP1, ddP2
    return P1, dP1, ddP1


def compute_lgl_nodes(N):
    """LGL 노드 계산 (N+1개 점).

    대칭성을 이용하여 절반만 계산하고 반사한다.
    Newton iteration으로 P'_N(tau)=0의 근을 구한다.
    """
    if N == 1:
        return np.array([-1.0, 1.0])

    tau = np.zeros(N + 1)
    tau[0] = -1.0
    tau[N] = 1.0

    # 대칭성 이용: 절반만 계산
    n_half = (N - 1) // 2

    for j in range(n_half):
        # 초기값: Chebyshev-Gauss-Lobatto 기반
        idx = j + 1  # 1-indexed from left
        x = -np.cos(np.pi * idx / N)

        # Newton iteration: P'_N(x) = 0
        for _ in range(200):
            _, dP, ddP = _legendre_recurrence(N, np.array([x]))
            if abs(ddP[0]) < 1e-30:
                break
            delta = dP[0] / ddP[0]
            x -= delta
            if abs(delta) < 1e-15:
                break

        tau[idx] = x
        tau[N - idx] = -x  # 대칭

    # 홀수 N이면 가운데 노드 = 0
    if (N - 1) % 2 == 0:
        tau[(N) // 2] = 0.0  # N odd => N+1 even, but N-1 interior, middle one is 0

    # N이 짝수이면 내부 노드 N-1개 중 가운데 노드가 0
    if N % 2 == 0:
        tau[N // 2] = 0.0

    return tau


def compute_lgl_weights(N, tau):
    """LGL 적분 가중치: w_j = 2 / (N*(N+1) * [P_N(tau_j)]^2)."""
    P_N = legendre_poly(N, tau)
    return 2.0 / (N * (N + 1) * P_N**2)


def compute_differentiation_matrix(N, tau):
    """LGL 미분 행렬.

    D_{ij} = P_N(tau_i) / (P_N(tau_j) * (tau_i - tau_j))  for i != j
    D_{00} = -N*(N+1)/4,  D_{NN} = N*(N+1)/4
    내부 대각: 행 합 = 0 조건으로 결정
    """
    n_pts = N + 1
    P_N = legendre_poly(N, tau)
    D = np.zeros((n_pts, n_pts))

    for i in range(n_pts):
        for j in range(n_pts):
            if i != j:
                D[i, j] = P_N[i] / (P_N[j] * (tau[i] - tau[j]))

    # 대각 원소: 끝점은 정확한 값, 내부는 행 합 = 0
    D[0, 0] = -N * (N + 1) / 4.0
    D[N, N] = N * (N + 1) / 4.0
    for i in range(1, N):
        D[i, i] = -np.sum(D[i, :i]) - np.sum(D[i, i + 1:])

    return D
