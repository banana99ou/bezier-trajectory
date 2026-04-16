"""베지어 제약조건 처리: 볼록껍질, 극값 정리, SOCP.

이론: docs/reports/005_bezier_constraints/

- 볼록껍질 (conservative): 제어점 p_i ≤ c → q(τ) ≤ c  (보수적, 선형)
- 극값 정리 (lossless): 극값점에서만 제약 → 정확, SCP 반복마다 갱신
- SOCP 추력 제약: ‖u(τ_j)‖ ≤ u_max → 이산 격자 SOCP
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .basis import bernstein, bernstein_eval, diff_matrix, double_int_matrix


# ═══════════════════════════════════════════════════════════════
#  볼록껍질 (보수적 제약)
# ═══════════════════════════════════════════════════════════════
def convex_hull_bound(
    P: NDArray,
    upper: float | None = None,
    lower: float | None = None,
) -> tuple[NDArray, NDArray]:
    """볼록껍질 성질 기반 제어점 부등식 제약.

    모든 제어점 p_i ≤ upper (또는 p_i ≥ lower)이면
    곡선 q(τ) ≤ upper (또는 q(τ) ≥ lower) ∀τ∈[0,1].

    Parameters
    ----------
    P : (N+1,) or (N+1, d) 제어점.
    upper : 상한. None이면 무한대.
    lower : 하한. None이면 -무한대.

    Returns
    -------
    (A_ub, b_ub) : 부등식 제약 A_ub @ p ≤ b_ub.
        p = vec(P) if d > 1, else p = P.
    """
    P = np.asarray(P)
    n = P.shape[0]

    rows_A = []
    rows_b = []

    if P.ndim == 1:
        if upper is not None:
            rows_A.append(np.eye(n))
            rows_b.append(np.full(n, upper))
        if lower is not None:
            rows_A.append(-np.eye(n))
            rows_b.append(np.full(n, -lower))
    else:
        # 각 축(열)별로 독립 적용
        d = P.shape[1]
        nd = n * d
        for k in range(d):
            # p의 k번째 축 성분에 대한 제약
            # vec(P) = [P[:,0]; P[:,1]; P[:,2]] (column-major)
            sel = np.zeros((n, nd))
            sel[:, k * n:(k + 1) * n] = np.eye(n)
            if upper is not None:
                rows_A.append(sel)
                rows_b.append(np.full(n, upper))
            if lower is not None:
                rows_A.append(-sel)
                rows_b.append(np.full(n, -lower))

    if not rows_A:
        return np.zeros((0, n if P.ndim == 1 else n * P.shape[1])), np.zeros(0)

    return np.vstack(rows_A), np.concatenate(rows_b)


# ═══════════════════════════════════════════════════════════════
#  극값 정리 (무손실 제약)
# ═══════════════════════════════════════════════════════════════
def find_extrema(P: NDArray, N: int) -> NDArray:
    """스칼라 베지어 곡선 q(τ)의 극값점 {τ*} 집합 계산.

    q'(τ) = B_{N-1}(τ)^T · (D_N · P) = 0 의 실수근 중 (0,1) 구간 내의 것.
    동반 행렬(companion matrix)의 고유값으로 안정적 계산.

    Parameters
    ----------
    P : (N+1,) 제어점 (스칼라).
    N : 베지어 차수.

    Returns
    -------
    tau_extrema : (m,) 극값점 배열 (끝점 0, 1은 미포함).
    """
    D = diff_matrix(N)
    dp = D @ P  # (N,) — 미분 곡선의 제어점

    if N <= 1:
        return np.array([])

    # Bernstein → power basis 변환 후 companion matrix
    # 또는 직접 Bernstein 근 찾기: Bezier clipping 또는 subdivision
    # 여기서는 power basis 변환 + numpy.roots 사용
    coeffs = _bernstein_to_power(dp, N - 1)

    if np.allclose(coeffs, 0.0):
        return np.array([])

    roots = np.roots(coeffs[::-1])  # np.roots는 내림차순 계수

    # 실수근만, (0, 1) 구간 내
    real_roots = []
    for root in roots:
        if np.abs(root.imag) < 1e-10:
            t = root.real
            if 0.0 < t < 1.0:
                real_roots.append(t)

    return np.sort(np.array(real_roots))


def _bernstein_to_power(bp: NDArray, N: int) -> NDArray:
    """Bernstein 기저 계수 → power 기저 계수 변환.

    q(τ) = Σ bp_i B_i^N(τ) = Σ ap_k τ^k

    Returns (N+1,) power basis coefficients [a_0, a_1, ..., a_N].
    """
    import math
    n = N
    ap = np.zeros(n + 1)
    for k in range(n + 1):
        s = 0.0
        for i in range(k + 1):
            s += (-1)**(k - i) * math.comb(n, k) * math.comb(k, i) * bp[i]
        ap[k] = s
    return ap


def extremum_constraint_matrix(
    P_ref: NDArray,
    N: int,
) -> NDArray:
    """극값 정리 기반 제약 행렬 C 구성.

    C @ p ≤ c·1 형태의 제약에서 C를 반환.
    극값점 + 끝점(0, 1)에서의 Bernstein 기저 평가.

    Parameters
    ----------
    P_ref : (N+1,) 참조 제어점 (현재 SCP 반복의 참조해).
    N : 베지어 차수.

    Returns
    -------
    C : (m, N+1) 제약 행렬. m = |{0, 1} ∪ {극값점}|.
    tau_eval : (m,) 평가 시점.
    """
    tau_ext = find_extrema(P_ref, N)
    tau_eval = np.concatenate([[0.0], tau_ext, [1.0]])

    B = bernstein(N, tau_eval)  # (m, N+1)
    return B, tau_eval


# ═══════════════════════════════════════════════════════════════
#  SOCP 추력 크기 제약
# ═══════════════════════════════════════════════════════════════
def thrust_constraint_grid(
    N: int,
    M: int | None = None,
) -> NDArray:
    """추력 크기 제약용 이산 격자점.

    τ_j = j/M,  j = 0, ..., M.
    기본 M = 2N.

    Returns
    -------
    tau_grid : (M+1,)
    """
    if M is None:
        M = 2 * N
    return np.linspace(0.0, 1.0, M + 1)


def thrust_socp_matrices(
    N: int,
    M: int | None = None,
) -> tuple[NDArray, NDArray]:
    """SOCP 추력 제약에 필요한 Bernstein 기저 행렬.

    각 τ_j에서:  ‖P_u^T B_N(τ_j)‖ ≤ u_max

    Parameters
    ----------
    N : 베지어 차수.
    M : 이산화 격자 수 (기본 2N).

    Returns
    -------
    tau_grid : (M+1,)
    B_grid : (M+1, N+1) — 각 행이 B_N(τ_j).
    """
    tau_grid = thrust_constraint_grid(N, M)
    B_grid = bernstein(N, tau_grid)  # (M+1, N+1)
    return tau_grid, B_grid


def thrust_norm_at_grid(
    P_u: NDArray,
    B_grid: NDArray,
) -> NDArray:
    """격자점에서의 추력 노름 ‖u(τ_j)‖ = ‖P_u^T B_N(τ_j)‖.

    Parameters
    ----------
    P_u : (N+1, 3) 제어점 행렬.
    B_grid : (M+1, N+1) Bernstein 기저.

    Returns
    -------
    norms : (M+1,) 각 격자점에서의 추력 크기.
    """
    # u(τ_j) = P_u^T @ B_N(τ_j)  →  (3,)  for each j
    # B_grid @ P_u → (M+1, 3)
    u_grid = np.dot(B_grid, P_u)  # (M+1, 3)
    return np.linalg.norm(u_grid, axis=1)


def check_thrust_feasibility(
    P_u: NDArray,
    u_max: float,
    N: int,
    M: int | None = None,
) -> tuple[bool, float]:
    """추력 제약 위반 확인.

    Returns
    -------
    (feasible, max_violation) : 위반 여부와 최대 위반량.
    """
    tau_grid, B_grid = thrust_socp_matrices(N, M)
    norms = thrust_norm_at_grid(P_u, B_grid)
    max_norm = np.max(norms)
    return bool(max_norm <= u_max + 1e-8), float(max_norm - u_max)


# ═══════════════════════════════════════════════════════════════
#  경로 제약 (위치 부등식)
# ═══════════════════════════════════════════════════════════════

def _subdivision_matrix(n: int, a: float, b: float) -> NDArray:
    """드 카스텔조 알고리즘으로 [a, b] 구간 추출 행렬 계산.

    차수 n 베지어 곡선의 제어점 P에 대해,
    M @ P = [a, b] 구간에서의 부분 곡선 제어점.

    Parameters
    ----------
    n : 베지어 차수.
    a, b : 구간 경계 (0 ≤ a < b ≤ 1).

    Returns
    -------
    M : (n+1, n+1) 추출 행렬.
    """
    if abs(b - a) < 1e-15:
        return np.eye(n + 1)

    # Step 1: t=b에서 분할 → 왼쪽 부분 [0, b]
    work = np.eye(n + 1)
    left_rows = [work[0].copy()]
    for r in range(1, n + 1):
        work = (1.0 - b) * work[:-1] + b * work[1:]
        left_rows.append(work[0].copy())
    left = np.array(left_rows)  # (n+1, n+1)

    if abs(a) < 1e-15:
        return left  # [0, b] 구간

    # Step 2: 왼쪽 곡선을 t2=a/b에서 분할 → 오른쪽 부분이 원래 [a, b]
    t2 = a / b
    levels = [left.copy()]
    for r in range(1, n + 1):
        prev = levels[-1]
        new_level = (1.0 - t2) * prev[:-1] + t2 * prev[1:]
        levels.append(new_level)

    # 오른쪽 부분 제어점: R_r = levels[n-r][r]
    right_rows = []
    for r in range(n + 1):
        right_rows.append(levels[n - r][r].copy())

    return np.array(right_rows)


def path_constraint_matrices(
    P_u_ref: NDArray,
    r0: NDArray,
    v0: NDArray,
    t_f: float,
    N: int,
    *,
    r_min: float | None = None,
    r_max: float | None = None,
    K_subdiv: int = 8,
    gravity_drift: NDArray | None = None,
    **_kwargs,
) -> tuple[NDArray, NDArray]:
    """드 카스텔조 세분화 + 볼록껍질 기반 경로 제약.

    [0,1]을 K개 구간으로 드 카스텔조 세분화하여 부분 제어점을 생성하고,
    볼록껍질 성질로 구간 전체에서의 제약을 보장한다.

    위치 곡선 (차수 N+2):
      q_r[:, k] = q_fixed[:, k] + Ī_N @ Z[:, k]
    여기서 q_fixed = r₀ + t_f·v₀·ℓ + gravity_drift (Z 무관)

    구간 [τ_k, τ_{k+1}]의 부분 제어점:
      q_sub = M_k @ q_r   (M_k: 드 카스텔조 추출 행렬)

    선형화 제약: r̂_mid · q_sub_i ≤ r_max  ∀i
      → 볼록껍질 성질로 r̂_mid · r(τ) ≤ r_max, ∀τ ∈ [τ_k, τ_{k+1}]

    Parameters
    ----------
    P_u_ref : (N+1, 3) — 현재 반복의 참조 제어점
    r0, v0 : (3,) — 초기 위치/속도 (정규화)
    t_f : float — 비행시간
    N : int — 베지어 차수
    r_min, r_max : float | None — 궤도 반경 하한/상한
    K_subdiv : int — 드 카스텔조 세분화 구간 수 (기본 8)
    gravity_drift : (N+3, 3) | None — 중력 드리프트 Φ 제어점

    Returns
    -------
    A_ub : (n_constraints, 3(N+1))
    b_ub : (n_constraints,)
    """
    if r_min is None and r_max is None:
        dim = 3 * (N + 1)
        return np.zeros((0, dim)), np.zeros(0)

    Ibar_N = double_int_matrix(N)  # (N+3, N+1)
    dim = 3 * (N + 1)
    deg = N + 2  # 위치 곡선 차수

    # Z 무관 고정 부분
    ell = np.arange(N + 3) / (N + 2)
    q_fixed = np.empty((N + 3, 3))
    for k in range(3):
        q_fixed[:, k] = r0[k] + t_f * v0[k] * ell
    if gravity_drift is not None:
        q_fixed = q_fixed + gravity_drift

    # 참조 위치 제어점 (선형화 방향 계산용)
    q_r_ref = np.empty((N + 3, 3))
    for k in range(3):
        q_r_ref[:, k] = q_fixed[:, k] + Ibar_N @ (t_f * P_u_ref[:, k])

    # 구간 경계
    tau_breaks = np.linspace(0.0, 1.0, K_subdiv + 1)

    rows_A = []
    rows_b = []

    for seg in range(K_subdiv):
        tau_lo = tau_breaks[seg]
        tau_hi = tau_breaks[seg + 1]
        tau_mid = 0.5 * (tau_lo + tau_hi)

        # 구간 중점에서 선형화 방향
        r_mid = bernstein_eval(deg, q_r_ref, tau_mid)  # (3,)
        r_mid_mag = np.linalg.norm(r_mid)
        if r_mid_mag < 1e-15:
            continue
        r_hat_mid = r_mid / r_mid_mag

        # 드 카스텔조 추출 행렬
        M_s = _subdivision_matrix(deg, tau_lo, tau_hi)  # (N+3, N+3)

        # 부분 제어점의 Z 의존 행렬: W = M_s @ Ī_N
        W = M_s @ Ibar_N  # (N+3, N+1)

        # 부분 제어점의 고정 부분: M_s @ q_fixed
        q_fixed_sub = M_s @ q_fixed  # (N+3, 3)

        # 각 부분 제어점에 대해 제약
        for i in range(deg + 1):
            # r̂_mid · q_sub_i = r̂_mid · q_fixed_sub_i + Σ_k r̂_mid[k] · W[i,:] · Z[:,k]
            row = np.zeros(dim)
            for k in range(3):
                row[k * (N + 1):(k + 1) * (N + 1)] = r_hat_mid[k] * W[i, :]

            rhat_dot_qfixed = np.dot(r_hat_mid, q_fixed_sub[i])

            if r_max is not None:
                # r̂_mid · q_sub_i ≤ r_max  →  row · vec(Z) ≤ r_max - r̂_mid · q_fixed_sub_i
                rows_A.append(row.copy())
                rows_b.append(r_max - rhat_dot_qfixed)

            if r_min is not None:
                # r̂_mid · q_sub_i ≥ r_min  →  -row · vec(Z) ≤ -(r_min - r̂_mid · q_fixed_sub_i)
                rows_A.append(-row.copy())
                rows_b.append(-(r_min - rhat_dot_qfixed))

    if not rows_A:
        return np.zeros((0, dim)), np.zeros(0)

    return np.vstack(rows_A), np.array(rows_b)
