"""비공면 궤도전이 시각화: 3D 궤적, 추력/상태 프로파일.

보고서 014에 포함할 그림 생성.
출력: docs/reports/014_noncoplanar_verification/figures/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop
from bezier_orbit.scp.drift import DriftConfig
from bezier_orbit.bezier.basis import bernstein_eval, double_int_matrix, int_matrix

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

_RK4 = DriftConfig(method="rk4")
FIGDIR = Path(__file__).resolve().parent.parent / "docs/reports/014_noncoplanar_verification/figures"
FIGDIR.mkdir(exist_ok=True)


def compute_profiles(r0, v0, t_f, N, P_u_opt, n_steps=500):
    """RK4 전파로 실제 궤적 프로파일 계산."""
    from bezier_orbit.scp.inner_loop import _propagate_reference

    # RK4 전파 (중력 + 추력 포함한 실제 궤적)
    # _propagate_reference는 (n_steps+1, 6) 반환: [r, v]
    from bezier_orbit.scp.problem import SCPProblem
    prob = SCPProblem(r0=r0, v0=v0, rf=r0, vf=v0, t_f=t_f, N=N)
    traj = _propagate_reference(prob, P_u_opt, 1.0, n_steps=n_steps)

    r_traj = traj[:, :3]  # (n_steps+1, 3)
    v_traj = traj[:, 3:]  # (n_steps+1, 3)
    tau = np.linspace(0, 1, n_steps + 1)

    # 추력 프로파일 (베지어 평가)
    u_traj = np.array([bernstein_eval(N, P_u_opt, t) for t in tau])

    r_norms = np.linalg.norm(r_traj, axis=1)
    v_norms = np.linalg.norm(v_traj, axis=1)
    u_norms = np.linalg.norm(u_traj, axis=1)

    return tau, r_traj, v_traj, u_traj, r_norms, v_norms, u_norms


def solve_scenario(name, r0, v0, rf, vf, t_f, N=12, u_max=None, r_min=None):
    """시나리오 풀기."""
    prob = SCPProblem(
        r0=r0, v0=v0, rf=rf, vf=vf,
        t_f=t_f, N=N, u_max=u_max,
        perturbation_level=0,
        max_iter=50, tol_ctrl=1e-6, tol_bc=1e-3,
        r_min=r_min, r_max=None,
        path_K_subdiv=8,
        drift_config=_RK4,
    )
    res = solve_inner_loop(prob)
    tau, r_traj, v_traj, u_traj, r_norms, v_norms, u_norms = compute_profiles(
        r0, v0, t_f, N, res.P_u_opt
    )
    return {
        "res": res, "tau": tau,
        "r_traj": r_traj, "v_traj": v_traj, "u_traj": u_traj,
        "r_norms": r_norms, "v_norms": v_norms, "u_norms": u_norms,
        "r0": r0, "v0": v0, "rf": rf, "vf": vf, "t_f": t_f,
        "u_max": u_max, "r_min": r_min,
    }


def _keplerian_orbit_curve(r_vec, v_vec, mu=1.0, n_pts=200):
    """Cartesian 상태에서 케플러 궤도 전체를 계산."""
    from bezier_orbit.orbit.elements import cartesian_to_keplerian
    a, e, inc, raan, aop, _ = cartesian_to_keplerian(r_vec, v_vec, mu)
    p = a * (1.0 - e**2)

    ta_arr = np.linspace(0, 2 * np.pi, n_pts)

    # PQW → ECI 회전행렬
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(aop), np.sin(aop)
    R = np.array([
        [co*cw - so*ci*sw, -co*sw - so*ci*cw,  so*si],
        [so*cw + co*ci*sw, -so*sw + co*ci*cw, -co*si],
        [si*sw,             si*cw,              ci   ],
    ])

    pts = np.empty((n_pts, 3))
    for j, ta in enumerate(ta_arr):
        r_mag = p / (1.0 + e * np.cos(ta))
        r_pqw = np.array([r_mag * np.cos(ta), r_mag * np.sin(ta), 0.0])
        pts[j] = R @ r_pqw
    return pts


def _draw_earth(ax, R_e=0.15):
    """반투명 지구 구 + 대략적 대륙 윤곽선."""
    n_lon, n_lat = 40, 25
    u_sp = np.linspace(0, 2 * np.pi, n_lon)
    v_sp = np.linspace(0, np.pi, n_lat)
    xs = R_e * np.outer(np.cos(u_sp), np.sin(v_sp))
    ys = R_e * np.outer(np.sin(u_sp), np.sin(v_sp))
    zs = R_e * np.outer(np.ones(n_lon), np.cos(v_sp))

    # 바다색 반투명 구
    ax.plot_surface(xs, ys, zs, color="#4488cc", alpha=0.25,
                    linewidth=0, antialiased=True)

    # 대략적 대륙 윤곽 (경도, 위도 in degrees — 매우 간략화)
    continents = [
        # 아프리카 (간략)
        [(0, 35), (10, 35), (20, 30), (30, 10), (40, -5), (35, -15),
         (30, -35), (20, -35), (15, -20), (10, 5), (0, 5), (-5, 10),
         (-15, 15), (-15, 30), (0, 35)],
        # 유럽 (간략)
        [(0, 40), (5, 45), (10, 50), (20, 55), (30, 60), (40, 55),
         (30, 45), (25, 40), (15, 38), (5, 38), (0, 40)],
        # 아시아 (간략)
        [(40, 55), (60, 55), (80, 50), (100, 40), (120, 35),
         (130, 40), (140, 45), (140, 55), (130, 60), (100, 65),
         (80, 60), (60, 55)],
        # 북미 (간략)
        [(-130, 50), (-120, 55), (-100, 60), (-80, 55), (-70, 45),
         (-80, 30), (-90, 25), (-100, 30), (-105, 35), (-120, 40),
         (-130, 50)],
        # 남미 (간략)
        [(-80, 10), (-70, 5), (-60, -5), (-50, -15), (-45, -25),
         (-55, -35), (-65, -50), (-70, -45), (-75, -30), (-80, -5),
         (-80, 10)],
        # 호주 (간략)
        [(115, -15), (130, -15), (145, -20), (150, -30), (145, -40),
         (135, -35), (120, -30), (115, -20), (115, -15)],
    ]

    for cont in continents:
        lons = np.array([p[0] for p in cont]) * np.pi / 180
        lats = np.array([p[1] for p in cont]) * np.pi / 180
        cx = R_e * np.cos(lats) * np.cos(lons)
        cy = R_e * np.cos(lats) * np.sin(lons)
        cz = R_e * np.sin(lats)
        ax.plot(cx, cy, cz, color="#228B22", lw=0.7, alpha=0.6)

    # 적도선
    eq_lon = np.linspace(0, 2 * np.pi, 100)
    ax.plot(R_e * np.cos(eq_lon), R_e * np.sin(eq_lon),
            np.zeros_like(eq_lon), color="gray", lw=0.3, alpha=0.4)


def plot_3d_trajectory(data, title, filename, R_earth=0.15):
    """3D 궤적 시각화 — 출발/도착 궤도 전체 + 지구."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    r = data["r_traj"]
    r0 = data["r0"]
    v0 = data["v0"]
    rf = data["rf"]
    vf = data["vf"]

    # ── 출발 궤도 (전체) ──
    orbit0 = _keplerian_orbit_curve(r0, v0)
    ax.plot(orbit0[:, 0], orbit0[:, 1], orbit0[:, 2],
            color="#22aa22", ls="--", lw=1.2, alpha=0.6, label="Initial orbit")

    # ── 도착 궤도 (전체) ──
    orbitf = _keplerian_orbit_curve(rf, vf)
    ax.plot(orbitf[:, 0], orbitf[:, 1], orbitf[:, 2],
            color="#cc3333", ls="--", lw=1.2, alpha=0.6, label="Final orbit")

    # ── 전이 궤적 ──
    ax.plot(r[:, 0], r[:, 1], r[:, 2], "b-", lw=2.0, label="Transfer", zorder=5)

    # ── 출발/도착 마커 ──
    ax.scatter(*r[0], color="#22aa22", s=100, marker="o", edgecolors="k",
               linewidths=0.8, zorder=10, label="Departure")
    ax.scatter(*r[-1], color="#cc3333", s=100, marker="s", edgecolors="k",
               linewidths=0.8, zorder=10, label="Arrival")

    # 출발/도착점과 궤도의 연결 확인용 — 작은 화살표
    # 출발 속도 방향
    v0_hat = v0 / np.linalg.norm(v0) * 0.15
    ax.quiver(r[0, 0], r[0, 1], r[0, 2], v0_hat[0], v0_hat[1], v0_hat[2],
              color="#22aa22", arrow_length_ratio=0.3, lw=1.5)
    vf_hat = vf / np.linalg.norm(vf) * 0.15
    ax.quiver(r[-1, 0], r[-1, 1], r[-1, 2], vf_hat[0], vf_hat[1], vf_hat[2],
              color="#cc3333", arrow_length_ratio=0.3, lw=1.5)

    # ── 지구 ──
    _draw_earth(ax, R_e=R_earth)

    ax.set_xlabel("$x$ [DU]")
    ax.set_ylabel("$y$ [DU]")
    ax.set_zlabel("$z$ [DU]")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    # 축 비율 맞추기 — 모든 궤도 포함
    all_pts = np.vstack([orbit0, orbitf, r])
    max_range = max(np.ptp(all_pts[:, i]) for i in range(3)) / 2 * 1.1
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    fig.savefig(FIGDIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_state_profiles(data, title_prefix, filename_prefix, u_max=None, r_min=None):
    """상태변수 + 추력 프로파일 (2x2 subplot)."""
    tau = data["tau"]
    r_traj = data["r_traj"]
    v_traj = data["v_traj"]
    u_traj = data["u_traj"]
    t_f = data["t_f"]
    t_phys = tau * t_f  # 정규화 시간

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # (a) 위치 성분
    ax = axes[0, 0]
    ax.plot(t_phys, r_traj[:, 0], label="$r_x$")
    ax.plot(t_phys, r_traj[:, 1], label="$r_y$")
    ax.plot(t_phys, r_traj[:, 2], label="$r_z$")
    ax.set_ylabel("Position [DU]")
    ax.set_title(f"(a) Position components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) 궤도 반경
    ax = axes[0, 1]
    ax.plot(t_phys, data["r_norms"], "b-", lw=1.5)
    if r_min is not None:
        ax.axhline(r_min, color="r", ls="--", lw=1, label=f"$r_{{\\min}}={r_min}$")
        ax.legend()
    ax.set_ylabel("$\\|\\mathbf{r}\\|$ [DU]")
    ax.set_title("(b) Orbital radius")
    ax.grid(True, alpha=0.3)

    # (c) 추력 성분
    ax = axes[1, 0]
    ax.plot(t_phys, u_traj[:, 0], label="$u_x$")
    ax.plot(t_phys, u_traj[:, 1], label="$u_y$")
    ax.plot(t_phys, u_traj[:, 2], label="$u_z$")
    ax.set_xlabel(f"Time $t^*$ [TU]")
    ax.set_ylabel("Thrust accel. [AU]")
    ax.set_title("(c) Thrust components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) 추력 크기
    ax = axes[1, 1]
    ax.plot(t_phys, data["u_norms"], "b-", lw=1.5)
    if u_max is not None:
        ax.axhline(u_max, color="r", ls="--", lw=1, label=f"$u_{{\\max}}={u_max}$")
        ax.legend()
    ax.set_xlabel(f"Time $t^*$ [TU]")
    ax.set_ylabel("$\\|\\mathbf{u}\\|$ [AU]")
    ax.set_title("(d) Thrust magnitude")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGDIR / f"{filename_prefix}_profiles.pdf")
    fig.savefig(FIGDIR / f"{filename_prefix}_profiles.png")
    plt.close(fig)
    print(f"  Saved: {filename_prefix}_profiles.pdf/png")


def plot_convergence(results_dict, filename):
    """BC 위반 수렴 곡선 (semilog)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for key, data in results_dict.items():
        res = data["res"]
        # BC 위반 이력은 직접 없으므로 ctrl_change로 대체 표시
        iters = range(1, len(res.cost_history) + 1)
        costs = [c if c < 1e10 else np.nan for c in res.cost_history]
        ax.semilogy(iters, res.ctrl_change_history, "-o", ms=3, label=key)

    ax.set_xlabel("SCP Iteration")
    ax.set_ylabel("Control point change $\\|\\Delta Z\\|_F$")
    ax.set_title("SCP Convergence History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / filename)
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_path_comparison(data_with, data_without, filename):
    """경로 제약 유무 비교: ||r(τ)|| 프로파일."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(data_without["tau"] * data_without["t_f"], data_without["r_norms"],
            "b--", lw=1.2, label="Without path constraint")
    ax.plot(data_with["tau"] * data_with["t_f"], data_with["r_norms"],
            "r-", lw=1.5, label="With $r_{\\min}=0.9$")

    if data_with["r_min"] is not None:
        ax.axhline(data_with["r_min"], color="k", ls=":", lw=1,
                    label=f"$r_{{\\min}}={data_with['r_min']}$")

    ax.set_xlabel("Time $t^*$ [TU]")
    ax.set_ylabel("$\\|\\mathbf{r}\\|$ [DU]")
    ax.set_title("Path Constraint Effect on Orbital Radius")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / filename)
    plt.close(fig)
    print(f"  Saved: {filename}")


def main():
    print("Solving scenarios...")

    # V4: 경사각 변경
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.0, 0.0, np.radians(28.5), 0.0, 0.0, np.pi, mu=1.0)
    print("  V4: Inclination change...")
    d4 = solve_scenario("V4", r0, v0, rf, vf, t_f=4.0)

    # V5: 이심률+경사각
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, np.radians(15), 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.3, 0.3, np.radians(15), 0.0, np.radians(90), np.pi, mu=1.0)
    print("  V5: Eccentric+inclined...")
    d5 = solve_scenario("V5", r0, v0, rf, vf, t_f=5.0)

    # V6: RAAN + r_min
    r0, v0 = keplerian_to_cartesian(1.0, 0.01, np.radians(10), 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.01, np.radians(30), np.radians(20), 0.0, np.pi, mu=1.0)
    print("  V6: RAAN+path...")
    d6 = solve_scenario("V6", r0, v0, rf, vf, t_f=5.0, r_min=0.9)
    print("  V6 (no path)...")
    d6n = solve_scenario("V6_nopath", r0, v0, rf, vf, t_f=5.0)

    # V7: SOCP + 비공면 + r_min
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.0, np.radians(20), 0.0, 0.0, np.pi, mu=1.0)
    print("  V7: SOCP+noncoplanar+path...")
    d7 = solve_scenario("V7", r0, v0, rf, vf, t_f=5.0, u_max=3.0, r_min=0.9)

    print("\nGenerating figures...")

    # 3D 궤적 — 4개 시나리오 전부
    for fmt in ["pdf", "png"]:
        plot_3d_trajectory(d4,
            "V4: Inclination Change ($0^\\circ \\to 28.5^\\circ$)",
            f"v4_3d.{fmt}")
        plot_3d_trajectory(d5,
            "V5: Circular $\\to$ Elliptic ($e{=}0.3$, $i{=}15^\\circ$)",
            f"v5_3d.{fmt}")
        plot_3d_trajectory(d6,
            "V6: RAAN Change + Path Constraint ($r_{\\min}{=}0.9$)",
            f"v6_3d.{fmt}")
        plot_3d_trajectory(d7,
            "V7: SOCP + Noncoplanar + Path Constraint",
            f"v7_3d.{fmt}")

    # 상태 프로파일
    plot_state_profiles(d4, "V4: Inclination Change", "v4", r_min=None)
    plot_state_profiles(d5, "V5: Eccentric + Inclined Transfer", "v5", r_min=None)
    plot_state_profiles(d6, "V6: RAAN Change + Path Constraint", "v6", r_min=0.9)
    plot_state_profiles(d7, "V7: SOCP + Noncoplanar + Path", "v7",
                        u_max=3.0, r_min=0.9)

    # 수렴 곡선
    plot_convergence({"V4": d4, "V5": d5, "V6": d6, "V7": d7},
                     "convergence.pdf")
    plot_convergence({"V4": d4, "V5": d5, "V6": d6, "V7": d7},
                     "convergence.png")

    # 경로 제약 비교
    plot_path_comparison(d6, d6n, "v6_path_comparison.pdf")
    plot_path_comparison(d6, d6n, "v6_path_comparison.png")

    print("\nDone! Figures saved to:", FIGDIR)


if __name__ == "__main__":
    main()
