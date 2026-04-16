"""ℓ₁ 코스팅의 궤도전이 적용 검증 + 시각화.

보고서 016 업데이트용 그림 생성.
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
from bezier_orbit.db.store import SimulationStore

rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

FIGDIR = "docs/reports/016_blade_verification/figures"
RESULTS_DIR = "scripts/results"
DB_PATH = f"{RESULTS_DIR}/simulations.duckdb"
A0 = 7000.0
CU = from_orbit(A0)


def _extract(result, prob):
    """Reconstruct trajectory and thrust from BLADE result."""
    K, n = prob.K, prob.n
    deltas = np.full(K, 1.0 / K)
    t_f_phys = CU.dim_time(prob.t_f)
    r0, v0 = prob.dep.at_time(0.0)
    x0 = np.concatenate([CU.nondim_pos(r0), CU.nondim_vel(v0)])

    seg_trajs = _propagate_blade_reference(
        result.p_segments, n, K, deltas, prob.t_f, x0, CU.R_earth_star, 50)

    n_per = seg_trajs[0].shape[0]
    x_all = np.vstack(seg_trajs)
    tau_all = np.concatenate([
        np.linspace(sum(deltas[:k]), sum(deltas[:k+1]), n_per)
        for k in range(K)])

    u_all = np.zeros((len(tau_all), 3))
    idx = 0
    for k in range(K):
        pk = result.p_segments[k]
        for j in range(n_per):
            B = bernstein(n, j / (n_per - 1))
            u_all[idx] = B @ pk
            idx += 1

    return tau_all, x_all, u_all


def run_sweep():
    """Lambda sweep and collect results."""
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    store = SimulationStore(DB_PATH)

    dep = OrbitBC(a=A0, e=0, inc=0, raan=0, aop=0, ta=0)
    arr = OrbitBC(a=A0*1.1, e=0, inc=0, raan=0, aop=0, ta=0)

    lambdas = [0, 0.05, 0.1, 0.3, 0.5, 1.0]
    results = {}

    for lam in lambdas:
        prob = BLADEOrbitProblem(
            dep=dep, arr=arr, t_f=4.0, K=12, n=1,
            u_max=3.0, l1_lambda=lam,
            canonical_units=CU, max_iter=50, tol_bc=5e-2,
            n_steps_per_seg=30, relax_alpha=0.3, trust_region=5.0)
        result = solve_blade_scp(prob)
        tau, x, u = _extract(result, prob)
        norms = [np.linalg.norm(pk) for pk in result.p_segments]
        n_coast = sum(1 for nm in norms if nm < 0.01)

        # DB 저장
        blade_id = store.save_blade_simulation(prob, result, CU)

        results[lam] = dict(
            result=result, prob=prob, tau=tau, x=x, u=u,
            norms=norms, n_coast=n_coast)

        print(f"  lambda={lam:.2f}: coast={n_coast}/12, "
              f"bc={result.bc_violation:.4f}, cost={result.cost:.4f} [blade_id={blade_id}]")

    store.close()
    print(f"  DB 저장: {DB_PATH}")
    return lambdas, results


def plot_thrust_profiles(lambdas, results):
    """Fig A: thrust magnitude profiles per lambda."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.ravel()

    for i, lam in enumerate(lambdas):
        d = results[lam]
        tau, u = d["tau"], d["u"]
        t_min = tau * CU.dim_time(d["prob"].t_f) / 60
        u_mag = np.linalg.norm(u, axis=1)

        ax = axes[i]
        ax.plot(t_min, u_mag, "k-", lw=1.2)

        # Shade coasting segments
        K = d["prob"].K
        deltas = np.full(K, 1.0 / K)
        for k in range(K):
            if d["norms"][k] < 0.01:
                t0 = sum(deltas[:k]) * CU.dim_time(d["prob"].t_f) / 60
                t1 = sum(deltas[:k+1]) * CU.dim_time(d["prob"].t_f) / 60
                ax.axvspan(t0, t1, alpha=0.15, color="blue")

        ax.axhline(d["prob"].u_max, ls="--", c="red", alpha=0.4)
        ax.set_title(f"$\\lambda = {lam}$, coast={d['n_coast']}/12")
        ax.set_ylabel("$\\|u\\|$ [AU]")
        ax.set_ylim(-0.1, d["prob"].u_max * 1.2)

    for ax in axes[-3:]:
        ax.set_xlabel("Time [min]")

    fig.suptitle("Thrust magnitude profiles ($a \\to 1.1a$, $u_{\\max}=3$)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIGDIR}/blade_l1_orbit_thrust.pdf", dpi=150)
    fig.savefig(f"{FIGDIR}/blade_l1_orbit_thrust.png", dpi=150)
    print("Saved: blade_l1_orbit_thrust")


def plot_segment_norms(lambdas, results):
    """Fig B: segment control-point norms."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    axes = axes.ravel()

    for i, lam in enumerate(lambdas):
        d = results[lam]
        K = d["prob"].K
        colors = ["green" if nm > 0.01 else "lightgray" for nm in d["norms"]]
        axes[i].bar(range(K), d["norms"], color=colors, edgecolor="gray", lw=0.5)
        axes[i].set_title(f"$\\lambda = {lam}$")
        axes[i].set_ylabel("$\\|\\mathbf{{p}}_k\\|$")
        axes[i].set_xlabel("Segment $k$")

    fig.suptitle("Segment control-point norms (green=active, gray=coasting)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIGDIR}/blade_l1_orbit_norms.pdf", dpi=150)
    fig.savefig(f"{FIGDIR}/blade_l1_orbit_norms.png", dpi=150)
    print("Saved: blade_l1_orbit_norms")


def plot_summary(lambdas, results):
    """Fig C: lambda vs coasting count / cost / BC violation."""
    n_coasts = [results[l]["n_coast"] for l in lambdas]
    costs = [results[l]["result"].cost for l in lambdas]
    bc_viols = [results[l]["result"].bc_violation for l in lambdas]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

    ax1.plot(lambdas, n_coasts, "o-", c="blue", lw=1.5)
    ax1.set_xlabel("$\\lambda$"); ax1.set_ylabel("Coasting segments")
    ax1.set_title("Coasting count vs $\\lambda$")
    ax1.set_ylim(-0.5, 13)

    ax2.plot(lambdas, costs, "s-", c="red", lw=1.5)
    ax2.set_xlabel("$\\lambda$"); ax2.set_ylabel("Cost $J$")
    ax2.set_title("Cost vs $\\lambda$")

    ax3.semilogy(lambdas[1:], bc_viols[1:], "^-", c="green", lw=1.5)
    if bc_viols[0] > 0:
        ax3.semilogy(lambdas[0], bc_viols[0], "^", c="green", ms=8)
    ax3.axhline(5e-2, ls="--", c="red", alpha=0.5, label="tol")
    ax3.set_xlabel("$\\lambda$"); ax3.set_ylabel("BC violation")
    ax3.set_title("BC violation vs $\\lambda$")
    ax3.legend()

    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/blade_l1_orbit_summary.pdf", dpi=150)
    fig.savefig(f"{FIGDIR}/blade_l1_orbit_summary.png", dpi=150)
    print("Saved: blade_l1_orbit_summary")


def plot_3d_coasting(lambdas, results):
    """Fig D: 3D trajectory with coasting highlighted."""
    fig = plt.figure(figsize=(14, 5))

    for i, lam in enumerate([0, 0.1, 1.0]):
        d = results[lam]
        prob = d["prob"]
        x = d["x"]
        K = prob.K
        deltas = np.full(K, 1.0 / K)
        n_per = len(d["tau"]) // K

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        # Earth
        Re = CU.R_earth_star
        us = np.linspace(0, 2*np.pi, 20)
        vs = np.linspace(0, np.pi, 15)
        ax.plot_surface(
            Re*np.outer(np.cos(us), np.sin(vs)),
            Re*np.outer(np.sin(us), np.sin(vs)),
            Re*np.outer(np.ones_like(us), np.cos(vs)),
            alpha=0.1, color="dodgerblue")

        # Trajectory segments
        for k in range(K):
            s = k * n_per
            e = s + n_per
            seg_x = x[s:e]
            if d["norms"][k] < 0.01:
                ax.plot(seg_x[:,0], seg_x[:,1], seg_x[:,2],
                        "b:", lw=1.5, alpha=0.6)
            else:
                ax.plot(seg_x[:,0], seg_x[:,1], seg_x[:,2],
                        "g-", lw=2)

        ax.scatter(*x[0,:3], c="blue", s=40, zorder=5)
        ax.scatter(*x[-1,:3], c="red", s=40, zorder=5)
        ax.set_title(f"$\\lambda = {lam}$, coast={d['n_coast']}/12")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    fig.suptitle("3D Trajectories (green=thrust, blue dotted=coasting)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIGDIR}/blade_l1_orbit_3d.pdf", dpi=150)
    fig.savefig(f"{FIGDIR}/blade_l1_orbit_3d.png", dpi=150)
    print("Saved: blade_l1_orbit_3d")


if __name__ == "__main__":
    print("Running orbital coasting lambda sweep...")
    lambdas, results = run_sweep()
    plot_thrust_profiles(lambdas, results)
    plot_segment_norms(lambdas, results)
    plot_summary(lambdas, results)
    plot_3d_coasting(lambdas, results)
    print("Done!")
