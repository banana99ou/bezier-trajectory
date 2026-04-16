"""BLADE 수치 검증 시각화.

보고서 016용 그림 생성:
1. 3D 궤적 (출발/도착 궤도 + 전이 궤적)
2. 상태 프로파일 (Position, Orbital radius, Thrust comp., Thrust mag.)
3. SCP 수렴 이력
4. ℓ₁ 코스팅 추력 프로파일
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from bezier_orbit.normalize import from_orbit
from bezier_orbit.bezier.basis import bernstein
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.blade.orbit import (
    OrbitBC, BLADEOrbitProblem, solve_blade_scp,
    _propagate_blade_reference,
)
from bezier_orbit.blade.problem import BLADEProblem, solve_blade

rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIGDIR = "docs/reports/016_blade_verification/figures"

import os
os.makedirs(FIGDIR, exist_ok=True)

A0 = 7000.0
CU = from_orbit(A0)


# ── 유틸리티 ──

def _keplerian_orbit(a, e, inc, raan, aop, n_pts=200, mu=1.0):
    """케플러 궤도 전체 곡선 (정규화 단위)."""
    tas = np.linspace(0, 2 * np.pi, n_pts)
    pts = np.array([keplerian_to_cartesian(a, e, inc, raan, aop, ta, mu)[0] for ta in tas])
    return pts


def _extract_trajectory(result, prob):
    """최적 궤적 + 추력 프로파일 추출."""
    t_f = prob.t_f
    K, n = prob.K, prob.n
    deltas = np.full(K, 1.0 / K)

    t_f_phys = CU.dim_time(t_f)
    r0, v0 = prob.dep.at_time(0.0)
    x0 = np.concatenate([CU.nondim_pos(r0), CU.nondim_vel(v0)])

    seg_trajs = _propagate_blade_reference(
        result.p_segments, n, K, deltas, t_f, x0, CU.R_earth_star, 50,
    )

    # 전체 궤적 + 시간 조립
    x_all = np.vstack(seg_trajs)
    n_per_seg = seg_trajs[0].shape[0]
    tau_all = []
    for k in range(K):
        tau_seg = np.linspace(k * deltas[k], (k + 1) * deltas[k], n_per_seg)
        # 정규화 시간 [0, 1]
        tau_seg = np.linspace(sum(deltas[:k]), sum(deltas[:k+1]), n_per_seg)
        tau_all.append(tau_seg)
    tau_all = np.concatenate(tau_all)

    # 추력 프로파일
    u_all = np.zeros((len(tau_all), 3))
    idx = 0
    for k in range(K):
        pk = result.p_segments[k]
        for j in range(n_per_seg):
            tau_local = j / (n_per_seg - 1)
            B = bernstein(n, tau_local)
            u_all[idx] = B @ pk
            idx += 1

    return tau_all, x_all, u_all


# ── 시나리오 정의 ──

SCENARIOS = {
    "V1": ("$a \\to 1.1a$ (공면)", dict(
        dep=OrbitBC(a=A0, e=0, inc=0, raan=0, aop=0, ta=0),
        arr=OrbitBC(a=A0*1.1, e=0, inc=0, raan=0, aop=0, ta=0),
        t_f=4.0, K=10, n=2)),
    "V3": ("$\\Delta i = 28.5°$", dict(
        dep=OrbitBC(a=A0, e=0, inc=0, raan=0, aop=0, ta=0),
        arr=OrbitBC(a=A0, e=0, inc=math.radians(28.5), raan=0, aop=0, ta=0),
        t_f=4.0, K=10, n=2)),
    "V4": ("$a \\to 1.05a + \\Delta i = 15°$", dict(
        dep=OrbitBC(a=A0, e=0, inc=0, raan=0, aop=0, ta=0),
        arr=OrbitBC(a=A0*1.05, e=0, inc=math.radians(15), raan=0, aop=0, ta=0),
        t_f=4.0, K=10, n=2)),
}


def run_scenario(name, label, kwargs):
    """시나리오 실행 + 데이터 추출."""
    prob = BLADEOrbitProblem(
        canonical_units=CU, max_iter=50, tol_bc=1e-2,
        n_steps_per_seg=50, relax_alpha=0.3, trust_region=5.0,
        **kwargs)
    result = solve_blade_scp(prob)
    tau, x, u = _extract_trajectory(result, prob)
    return prob, result, tau, x, u


# ── 그림 1: 3D 궤적 ──

def plot_3d_trajectories():
    fig = plt.figure(figsize=(15, 5))

    for i, (name, (label, kwargs)) in enumerate(SCENARIOS.items()):
        prob, result, tau, x, u = run_scenario(name, label, kwargs)
        t_f_phys = CU.dim_time(prob.t_f)

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        # 지구
        u_s = np.linspace(0, 2*np.pi, 30)
        v_s = np.linspace(0, np.pi, 20)
        Re = CU.R_earth_star
        xs = Re * np.outer(np.cos(u_s), np.sin(v_s))
        ys = Re * np.outer(np.sin(u_s), np.sin(v_s))
        zs = Re * np.outer(np.ones_like(u_s), np.cos(v_s))
        ax.plot_surface(xs, ys, zs, alpha=0.15, color="dodgerblue")

        # 출발 궤도
        dep = kwargs["dep"]
        orb_dep = _keplerian_orbit(
            dep.a/CU.DU, dep.e, dep.inc, dep.raan, dep.aop)
        ax.plot(orb_dep[:,0], orb_dep[:,1], orb_dep[:,2],
                "b--", alpha=0.4, lw=1, label="Dep. orbit")

        # 도착 궤도
        arr = kwargs["arr"]
        r_arr, v_arr = arr.at_time(t_f_phys)
        from bezier_orbit.orbit.elements import cartesian_to_keplerian
        a_f, e_f, i_f, raan_f, aop_f, _ = cartesian_to_keplerian(
            CU.nondim_pos(r_arr), CU.nondim_vel(v_arr))
        orb_arr = _keplerian_orbit(a_f, e_f, i_f, raan_f, aop_f)
        ax.plot(orb_arr[:,0], orb_arr[:,1], orb_arr[:,2],
                "r--", alpha=0.4, lw=1, label="Arr. orbit")

        # 전이 궤적
        ax.plot(x[:,0], x[:,1], x[:,2], "g-", lw=2, label="Transfer")
        ax.scatter(*x[0,:3], c="blue", s=40, zorder=5)
        ax.scatter(*x[-1,:3], c="red", s=40, zorder=5)

        ax.set_title(f"{name}: {label}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("BLADE-SCP 3D Trajectories", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIGDIR}/blade_3d_trajectories.pdf", dpi=150)
    fig.savefig(f"{FIGDIR}/blade_3d_trajectories.png", dpi=150)
    print("Saved: blade_3d_trajectories")


# ── 그림 2: 상태/추력 프로파일 ──

def plot_profiles():
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for i, (name, (label, kwargs)) in enumerate(SCENARIOS.items()):
        prob, result, tau, x, u = run_scenario(name, label, kwargs)
        t_phys = tau * CU.dim_time(prob.t_f) / 60  # 분

        # Position
        ax = axes[i, 0]
        for k, c in enumerate(["r", "g", "b"]):
            ax.plot(t_phys, x[:, k], c, lw=1, label=f"{'xyz'[k]}")
        ax.set_ylabel("Position [DU]")
        ax.set_title(f"{name}: Position")
        ax.legend(fontsize=7)

        # Orbital radius
        ax = axes[i, 1]
        r_mag = np.linalg.norm(x[:, :3], axis=1)
        ax.plot(t_phys, r_mag, "k-", lw=1.5)
        ax.axhline(1.0, ls=":", c="blue", alpha=0.5, label="dep a")
        ax.set_ylabel("$\\|r\\|$ [DU]")
        ax.set_title(f"{name}: Orbital radius")

        # Thrust comp.
        ax = axes[i, 2]
        for k, c in enumerate(["r", "g", "b"]):
            ax.plot(t_phys, u[:, k], c, lw=0.8, label=f"$u_{'xyz'[k]}$")
        ax.set_ylabel("Thrust [AU]")
        ax.set_title(f"{name}: Thrust comp.")
        ax.legend(fontsize=7)

        # Thrust mag.
        ax = axes[i, 3]
        u_mag = np.linalg.norm(u, axis=1)
        ax.plot(t_phys, u_mag, "k-", lw=1.5)
        ax.set_ylabel("$\\|u\\|$ [AU]")
        ax.set_title(f"{name}: Thrust mag.")

        # Segment 경계 표시
        K = prob.K
        for ax_j in axes[i]:
            for seg in range(1, K):
                t_seg = seg / K * CU.dim_time(prob.t_f) / 60
                ax_j.axvline(t_seg, ls=":", c="gray", alpha=0.3, lw=0.5)

    for ax in axes[-1]:
        ax.set_xlabel("Time [min]")

    fig.suptitle("BLADE-SCP State & Thrust Profiles", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIGDIR}/blade_profiles.pdf", dpi=150)
    fig.savefig(f"{FIGDIR}/blade_profiles.png", dpi=150)
    print("Saved: blade_profiles")


# ── 그림 3: SCP 수렴 이력 ──

def plot_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, (label, kwargs) in SCENARIOS.items():
        prob = BLADEOrbitProblem(
            canonical_units=CU, max_iter=50, tol_bc=1e-2,
            n_steps_per_seg=30, relax_alpha=0.3, trust_region=5.0,
            **kwargs)
        result = solve_blade_scp(prob)

        iters = range(1, len(result.bc_history) + 1)
        ax1.semilogy(iters, result.bc_history, "o-", ms=3, lw=1.2, label=f"{name}: {label}")
        ax2.plot(iters, result.cost_history, "o-", ms=3, lw=1.2, label=f"{name}: {label}")

    ax1.axhline(1e-2, ls="--", c="red", alpha=0.5, label="tol=$10^{-2}$")
    ax1.set_xlabel("SCP iteration"); ax1.set_ylabel("BC violation")
    ax1.set_title("BC Violation Convergence"); ax1.legend(fontsize=7)

    ax2.set_xlabel("SCP iteration"); ax2.set_ylabel("Cost $J$")
    ax2.set_title("Cost Convergence"); ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/blade_convergence.pdf", dpi=150)
    fig.savefig(f"{FIGDIR}/blade_convergence.png", dpi=150)
    print("Saved: blade_convergence")


# ── 그림 4: ℓ₁ 코스팅 (이중 적분기) ──

def plot_l1_coasting():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    lambdas = [0, 0.5, 1.0]
    K, n = 20, 1

    for i, lam in enumerate(lambdas):
        prob = BLADEProblem(r0=0, v0=0, rf=1, vf=0, t_f=1, K=K, n=n,
                            u_max=5, l1_lambda=lam)
        result = solve_blade(prob)

        # 추력 프로파일
        tau_pts = np.linspace(0, 1, 500)
        u_vals = np.zeros(500)
        deltas = np.full(K, 1.0/K)
        for j, tau in enumerate(tau_pts):
            seg = min(int(tau * K), K - 1)
            tau_local = (tau - seg * deltas[0]) / deltas[0]
            tau_local = np.clip(tau_local, 0, 1)
            B = bernstein(n, tau_local)
            u_vals[j] = B @ result.p_segments[seg]

        ax = axes[0, i]
        ax.plot(tau_pts, u_vals, "k-", lw=1.5)
        ax.axhline(0, ls="-", c="gray", alpha=0.3)
        ax.axhline(5, ls="--", c="red", alpha=0.4)
        ax.axhline(-5, ls="--", c="red", alpha=0.4)
        norms = [np.linalg.norm(pk) for pk in result.p_segments]
        n_coast = sum(1 for nm in norms if nm < 0.01)
        ax.set_title(f"$\\lambda = {lam}$, coast={n_coast}/{K}")
        ax.set_ylabel("$u(\\tau)$")
        ax.set_ylim(-6.5, 6.5)

        # Segment 노름 (코스팅 식별)
        ax = axes[1, i]
        ax.bar(range(K), norms, color=["green" if nm > 0.01 else "lightgray" for nm in norms])
        ax.set_xlabel("Segment $k$")
        ax.set_ylabel("$\\|\\mathbf{p}_k\\|$")
        ax.set_title(f"Segment ctrl-point norm")

    fig.suptitle("$\\ell_1$ Automatic coasting via regularization (K=20, n=1, $u_{\\max}=5$)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIGDIR}/blade_l1_coasting.pdf", dpi=150)
    fig.savefig(f"{FIGDIR}/blade_l1_coasting.png", dpi=150)
    print("Saved: blade_l1_coasting")


# ── 실행 ──

if __name__ == "__main__":
    print("Generating BLADE verification figures...")
    plot_3d_trajectories()
    plot_profiles()
    plot_convergence()
    plot_l1_coasting()
    print("Done!")
