"""N4: JAX 자동 미분으로 대수적 커플링 행렬 ∂c_v/∂P_u 검증.

보고서 012의 Bernstein 체인 룰을 JAX로 자동 미분하여:
1. 유한차분과 비교 (정확도 검증)
2. 기존 수치적 커플링 행렬과 비교
3. SCP 수렴 효과 측정 (커플링 off / 수치적 / JAX 대수적)
"""

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

warnings.filterwarnings("ignore")

# JAX 64비트 활성화
jax.config.update("jax_enable_x64", True)

from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.drift import (
    DriftConfig, position_control_points, compute_drift,
    compute_coupling_matrices, _gravity_accel_vec,
)
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop
from bezier_orbit.bezier.basis import double_int_matrix, int_matrix


# ══════════════════════════════════════════════════════════════
# JAX 버전 Bernstein 연산 (순수 JAX로 재구현)
# ══════════════════════════════════════════════════════════════

def _jax_binom(n, k):
    """이항계수 (JAX-compatible)."""
    from scipy.special import comb as sp_comb
    return sp_comb(n, k, exact=False)


def _jax_product_weights(N, M):
    """Bernstein 곱 가중치 행렬 (numpy로 사전 계산)."""
    from scipy.special import comb as sp_comb
    K_out = N + M
    js = np.arange(N + 1)
    ks = np.arange(K_out + 1)[:, np.newaxis]
    ls = ks - js
    valid = (ls >= 0) & (ls <= M)
    cN = sp_comb(N, js, exact=False)
    cM = sp_comb(M, np.clip(ls, 0, M), exact=False)
    cNM = sp_comb(K_out, ks.ravel(), exact=False)[:, np.newaxis]
    W = np.where(valid, cN * cM / cNM, 0.0)
    return jnp.array(W), jnp.array(valid), jnp.array(np.clip(ls, 0, M))


def jax_bern_product(p, q, W, valid, clip):
    """Bernstein 곱 (JAX differentiable)."""
    q_padded = jnp.append(q, 0.0)
    Q_shifted = jnp.where(valid, q_padded[clip], 0.0)
    return jnp.sum(W * p[None, :] * Q_shifted, axis=1)


def jax_bern_compose(h, g, compose_data):
    """Bernstein 합성 (JAX differentiable)."""
    K = len(h) - 1
    M = len(g) - 1
    target_deg = K * M
    one_minus_g = jnp.ones(M + 1) - g

    # 순차 곱으로 g^i, (1-g)^i 계산
    g_pow = [jnp.ones(1)]
    omg_pow = [jnp.ones(1)]
    for i in range(1, K + 1):
        Wg = compose_data[('g_pow', i)]
        validg = compose_data[('g_pow_v', i)]
        clipg = compose_data[('g_pow_c', i)]
        g_pow.append(jax_bern_product(g_pow[-1], g, Wg, validg, clipg))

        Wo = compose_data[('omg_pow', i)]
        valido = compose_data[('omg_pow_v', i)]
        clipo = compose_data[('omg_pow_c', i)]
        omg_pow.append(jax_bern_product(omg_pow[-1], one_minus_g, Wo, valido, clipo))

    # 합산
    r = jnp.zeros(target_deg + 1)
    for i in range(K + 1):
        Wt = compose_data[('term', i)]
        validt = compose_data[('term_v', i)]
        clipt = compose_data[('term_c', i)]
        term = jax_bern_product(g_pow[i], omg_pow[K - i], Wt, validt, clipt)

        # degree_elevate (JAX version)
        deg_term = len(term) - 1
        t = term
        for _ in range(target_deg - deg_term):
            n = len(t) - 1
            alpha = jnp.arange(1, n + 1) / (n + 1)
            new_t = jnp.zeros(n + 2)
            new_t = new_t.at[0].set(t[0])
            new_t = new_t.at[-1].set(t[-1])
            new_t = new_t.at[1:-1].set(alpha * t[:-1] + (1.0 - alpha) * t[1:])
            t = new_t

        r = r + h[i] * float(_jax_binom(K, i)) * t

    return r


def jax_degree_reduce(p, R, G_R_inv, C_mat):
    """차수 축소 (JAX differentiable)."""
    return G_R_inv @ (C_mat @ p)


def jax_gravity_pipeline(q_r, h_K, R, pipeline_data, a_val, b_val):
    """중력 파이프라인 (JAX differentiable).

    a_val, b_val은 상수로 취급 (미분 시 고정).
    """
    P = len(q_r) - 1

    # Step 1: r²
    W_self = pipeline_data['W_self']
    v_self = pipeline_data['v_self']
    c_self = pipeline_data['c_self']
    q_r2 = (
        jax_bern_product(q_r[:, 0], q_r[:, 0], W_self, v_self, c_self)
        + jax_bern_product(q_r[:, 1], q_r[:, 1], W_self, v_self, c_self)
        + jax_bern_product(q_r[:, 2], q_r[:, 2], W_self, v_self, c_self)
    )

    # Step 2: affine rescaling (a_val, b_val 고정)
    q_stilde = (q_r2 - a_val) / (b_val - a_val + 1e-30)

    # Step 3: h_K is pre-computed (not differentiable w.r.t. P_u)

    # Step 4: compose
    q_rinv3 = jax_bern_compose(h_K, q_stilde, pipeline_data['compose_data'])
    q_rinv3 = jax_degree_reduce(q_rinv3, R,
                                 pipeline_data['G_R_inv'],
                                 pipeline_data['C_compose_to_R'])

    # Step 5: -r * r^{-3}
    W_comp = pipeline_data['W_comp']
    v_comp = pipeline_data['v_comp']
    c_comp = pipeline_data['c_comp']

    q_agrav_cols = []
    for k in range(3):
        # degree_elevate q_r[:, k] to R
        t = q_r[:, k]
        for _ in range(R - P):
            n = len(t) - 1
            alpha = jnp.arange(1, n + 1) / (n + 1)
            new_t = jnp.zeros(n + 2)
            new_t = new_t.at[0].set(t[0])
            new_t = new_t.at[-1].set(t[-1])
            new_t = new_t.at[1:-1].set(alpha * t[:-1] + (1.0 - alpha) * t[1:])
            t = new_t
        comp = -jax_bern_product(t, q_rinv3, W_comp, v_comp, c_comp)
        q_agrav_cols.append(comp)

    q_agrav = jnp.column_stack(q_agrav_cols)

    # Step 6: degree_reduce
    q_agrav_reduced = pipeline_data['G_R_inv'] @ (pipeline_data['C_final_to_R'] @ q_agrav)

    return q_agrav_reduced


def jax_cv_from_Pu(P_u_flat, r0, v0, t_f, N, K, R, h_K, pipeline_data,
                    a_val, b_val, Ibar_N_j, ell_j):
    """P_u (flat) → c_v (JAX differentiable end-to-end).

    a_val, b_val, Ibar_N_j, ell_j는 사전 계산된 상수.
    """
    P_u = P_u_flat.reshape(N + 1, 3)

    # position control points — jnp.stack으로 구성
    cols = []
    for k in range(3):
        cols.append(r0[k] + t_f * v0[k] * ell_j + t_f**2 * (Ibar_N_j @ P_u[:, k]))
    q_r = jnp.stack(cols, axis=1)  # (N+3, 3)

    # gravity pipeline
    q_agrav = jax_gravity_pipeline(q_r, h_K, R, pipeline_data, a_val, b_val)

    # c_v = mean
    c_v = jnp.mean(q_agrav, axis=0)
    return c_v


# ══════════════════════════════════════════════════════════════
# 사전 계산 (가중치 행렬 등)
# ══════════════════════════════════════════════════════════════

def precompute_pipeline_data(N, K, R):
    """JAX 파이프라인에 필요한 가중치/행렬을 사전 계산."""
    from bezier_orbit.bezier.basis import gram_matrix
    from bezier_orbit.bezier.algebra import _cross_gram
    from scipy.special import comb as sp_comb

    P = N + 2
    data = {}

    # self-product weights (deg P × P → 2P)
    W, v, c = _jax_product_weights(P, P)
    data['W_self'] = W
    data['v_self'] = v
    data['c_self'] = c

    # compose data
    M = 2 * P  # degree of q_stilde
    compose_data = {}
    for i in range(1, K + 1):
        # g_pow[i] = g_pow[i-1] (deg (i-1)*M) × g (deg M)
        Wg, vg, cg = _jax_product_weights((i - 1) * M, M)
        compose_data[('g_pow', i)] = Wg
        compose_data[('g_pow_v', i)] = vg
        compose_data[('g_pow_c', i)] = cg
        # omg_pow same structure
        compose_data[('omg_pow', i)] = Wg
        compose_data[('omg_pow_v', i)] = vg
        compose_data[('omg_pow_c', i)] = cg

    for i in range(K + 1):
        # term = g_pow[i] (deg i*M) × omg_pow[K-i] (deg (K-i)*M) = deg K*M
        Wt, vt, ct = _jax_product_weights(i * M, (K - i) * M)
        compose_data[('term', i)] = Wt
        compose_data[('term_v', i)] = vt
        compose_data[('term_c', i)] = ct

    data['compose_data'] = compose_data

    # degree_reduce: compose output (deg K*2P) → R
    G_R = gram_matrix(R)
    G_R_inv = jnp.array(np.linalg.inv(G_R))
    data['G_R_inv'] = G_R_inv

    C_compose = np.array(_cross_gram(R, K * M))
    data['C_compose_to_R'] = jnp.array(C_compose)

    # component product weights (deg R × R → 2R)
    # Actually: deg_elevate(P→R) × R → P+R
    W_comp, v_comp, c_comp = _jax_product_weights(R, R)
    data['W_comp'] = W_comp
    data['v_comp'] = v_comp
    data['c_comp'] = c_comp

    # final degree_reduce: 2R → R (after product of deg R × deg R)
    C_final = np.array(_cross_gram(R, 2 * R))
    data['C_final_to_R'] = jnp.array(C_final)

    return data


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main():
    MU_KM = 398600.4418
    A0_KM = 6678.137

    r0_km, v0_km = keplerian_to_cartesian(A0_KM, 0.0, 0.0, 0.0, 0.0, 0.0, MU_KM)
    rf_km, vf_km = keplerian_to_cartesian(A0_KM * 1.2, 0.0, 0.0, 0.0, 0.0, np.pi, MU_KM)

    # 정규화 단위 (mu* = 1)
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    N = 8
    K = 4  # 보정용 K (낮은 값)
    R = 20
    t_f = 4.0

    # 수렴된 P_u 확보
    prob_rk4 = SCPProblem(
        r0=r0, v0=v0, rf=rf_km/A0_KM, vf=vf_km * np.sqrt(A0_KM / MU_KM),
        t_f=t_f, N=N, drift_config=DriftConfig(method="rk4"),
    )
    res = solve_inner_loop(prob_rk4)
    P_u = res.P_u_opt

    print("=" * 60)
    print("  검증 1: JAX 자동 미분 vs 유한차분")
    print("=" * 60)

    # 사전 계산
    print("  파이프라인 데이터 사전 계산 중...")
    pipeline_data = precompute_pipeline_data(N, K, R)

    # h_K 사전 계산 (from gravity pipeline)
    from bezier_orbit.bezier.algebra import chebyshev_bernstein_approx
    q_r_np = position_control_points(P_u, r0, v0, t_f, N)
    from bezier_orbit.bezier.algebra import bern_product
    q_r2_np = (bern_product(q_r_np[:, 0], q_r_np[:, 0])
               + bern_product(q_r_np[:, 1], q_r_np[:, 1])
               + bern_product(q_r_np[:, 2], q_r_np[:, 2]))
    a_val = float(np.min(q_r2_np))
    b_val = float(np.max(q_r2_np))
    h_K_np = chebyshev_bernstein_approx(
        lambda s: (a_val + (b_val - a_val) * s) ** (-1.5), K,
    )
    h_K = jnp.array(h_K_np)

    r0_j = jnp.array(r0)
    v0_j = jnp.array(v0)
    P_u_flat = jnp.array(P_u.ravel())

    # 사전 계산 상수 (JAX traced 영역 밖)
    a_val_f = float(a_val)
    b_val_f = float(b_val)
    Ibar_N_j = jnp.array(double_int_matrix(N))
    ell_j = jnp.arange(N + 3) / (N + 2)

    # JAX forward pass
    cv_jax = jax_cv_from_Pu(P_u_flat, r0_j, v0_j, t_f, N, K, R, h_K, pipeline_data,
                             a_val_f, b_val_f, Ibar_N_j, ell_j)
    print(f"  c_v (JAX):  {np.array(cv_jax)}")

    # JAX Jacobian
    print("  JAX Jacobian 계산 중...")
    t0 = time.perf_counter()
    jac_fn = jax.jacobian(jax_cv_from_Pu, argnums=0)
    M_v_jax = np.array(jac_fn(P_u_flat, r0_j, v0_j, t_f, N, K, R, h_K, pipeline_data,
                                a_val_f, b_val_f, Ibar_N_j, ell_j))
    t_jax = (time.perf_counter() - t0) * 1000
    print(f"  JAX Jacobian shape: {M_v_jax.shape}, 시간: {t_jax:.0f}ms")

    # 유한차분
    print("  유한차분 Jacobian 계산 중...")
    eps = 1e-7
    n_params = len(P_u_flat)
    M_v_fd = np.zeros((3, n_params))
    cv_base = np.array(cv_jax)
    t0 = time.perf_counter()
    for j in range(n_params):
        P_u_pert = P_u_flat.at[j].set(P_u_flat[j] + eps)
        cv_pert = jax_cv_from_Pu(P_u_pert, r0_j, v0_j, t_f, N, K, R, h_K, pipeline_data,
                                  a_val_f, b_val_f, Ibar_N_j, ell_j)
        M_v_fd[:, j] = (np.array(cv_pert) - cv_base) / eps
    t_fd = (time.perf_counter() - t0) * 1000
    print(f"  유한차분 시간: {t_fd:.0f}ms")

    # 비교
    diff = np.linalg.norm(M_v_jax - M_v_fd)
    scale = np.linalg.norm(M_v_fd)
    print(f"\n  ‖M_v_jax - M_v_fd‖ = {diff:.2e}")
    print(f"  ‖M_v_fd‖ = {scale:.2e}")
    print(f"  상대 오차: {diff/scale:.2e}")

    print("\n" + "=" * 60)
    print("  검증 2: JAX vs 기존 수치적 커플링 행렬")
    print("=" * 60)

    # 기존 수치적 커플링 행렬
    prob_bern = SCPProblem(
        r0=r0, v0=v0, rf=rf_km/A0_KM, vf=vf_km * np.sqrt(A0_KM / MU_KM),
        t_f=t_f, N=N, drift_config=DriftConfig(method="bernstein"),
    )
    M_v_num, M_r_num = compute_coupling_matrices(prob_bern, P_u, 1.0)

    diff_num = np.linalg.norm(M_v_jax - M_v_num)
    print(f"  ‖M_v_jax - M_v_num‖ = {diff_num:.2e}")
    print(f"  ‖M_v_jax‖ = {np.linalg.norm(M_v_jax):.2e}")
    print(f"  ‖M_v_num‖ = {np.linalg.norm(M_v_num):.2e}")
    print(f"  상대 차이: {diff_num/np.linalg.norm(M_v_jax):.2e}")


if __name__ == "__main__":
    main()
