"""
Diagnostic plot for boundary-velocity constraint cases.

This script constructs Bézier curves directly from endpoint positions and
selected endpoint velocity constraints (without running optimization), then
draws all requested cases and reports basic feasibility diagnostics.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from orbital_docking import BezierCurve, constants, generate_initial_control_points


def ccw_equatorial_tangent(pos_xyz: np.ndarray) -> np.ndarray:
    """Unit tangent in xy-plane, counter-clockwise, tangent to Earth-centric sphere."""
    x, y, _ = pos_xyz
    r_xy = np.hypot(x, y)
    if r_xy < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([-y / r_xy, x / r_xy, 0.0], dtype=float)


def current_velocity(pos_xyz: np.ndarray) -> np.ndarray:
    """
    Current project velocity convention:
    tangent direction with magnitude based on KOZ circular speed.
    """
    v_mag = np.sqrt(constants.EARTH_MU_SCALED / constants.KOZ_RADIUS)
    return v_mag * ccw_equatorial_tangent(pos_xyz)


def tes_velocity(pos_xyz: np.ndarray) -> np.ndarray:
    """
    Tangent-to-Earth-spheroid (TES) style velocity:
    tangent direction with local circular speed at |r|.
    """
    r_norm = np.linalg.norm(pos_xyz)
    if r_norm < 1e-12:
        return np.zeros(3, dtype=float)
    v_mag = np.sqrt(constants.EARTH_MU_SCALED / r_norm)
    return v_mag * ccw_equatorial_tangent(pos_xyz)


def control_points_from_endpoint_velocities(
    degree: int,
    p_start: np.ndarray,
    p_end: np.ndarray,
    v0: np.ndarray | None,
    v1: np.ndarray | None,
) -> np.ndarray:
    """
    Construct control points satisfying endpoint positions and optional endpoint velocities.
    For degree N:
      v(0) = N (P1 - P0), v(1) = N (PN - P{N-1}).
    """
    p = generate_initial_control_points(degree, p_start, p_end)
    n = degree

    if v0 is not None:
        p[1] = p[0] + (np.asarray(v0, dtype=float) / n)
    if v1 is not None:
        p[-2] = p[-1] - (np.asarray(v1, dtype=float) / n)

    return p


def evaluate_case(case_name: str, degree: int, p_start: np.ndarray, p_end: np.ndarray, v0, v1):
    """Build and sample a case, returning diagnostics."""
    out = {
        "name": case_name,
        "degree": degree,
        "ok": False,
        "error": "",
        "control_points": None,
        "curve_xyz": None,
        "min_radius_km": np.nan,
        "koz_ok": False,
    }

    try:
        p = control_points_from_endpoint_velocities(degree, p_start, p_end, v0, v1)
        curve = BezierCurve(p)
        ts = np.linspace(0.0, 1.0, 500)
        xyz = np.array([curve.point(t) for t in ts])

        if not np.all(np.isfinite(xyz)):
            raise ValueError("Sampled curve contains non-finite values.")

        radii = np.linalg.norm(xyz, axis=1)
        min_radius = float(np.min(radii))

        out["ok"] = True
        out["control_points"] = p
        out["curve_xyz"] = xyz
        out["min_radius_km"] = min_radius
        out["koz_ok"] = min_radius >= constants.KOZ_RADIUS
    except Exception as exc:  # broad by design for diagnostics
        out["error"] = str(exc)

    return out


def main():
    parser = argparse.ArgumentParser(description="Diagnose boundary velocity cases by plotting Bézier curves.")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window.")
    args = parser.parse_args()

    # Match endpoint setup used in Orbital_Docking_Optimizer.py
    theta_start = -np.pi / 4.0
    theta_end = np.pi / 4.0
    p_start = np.array(
        [
            constants.CHASER_RADIUS * np.cos(theta_start),
            constants.CHASER_RADIUS * np.sin(theta_start),
            0.0,
        ],
        dtype=float,
    )
    p_end = np.array(
        [
            constants.ISS_RADIUS * np.cos(theta_end),
            constants.ISS_RADIUS * np.sin(theta_end),
            0.0,
        ],
        dtype=float,
    )

    v0_current = current_velocity(p_start)
    v1_current = current_velocity(p_end)
    v0_tes = tes_velocity(p_start)
    v1_tes = tes_velocity(p_end)

    cases = [
        ("N=3 current v0+v1", 3, v0_current, v1_current),
        ("N=3 TES v0+v1", 3, v0_tes, v1_tes),
        ("N=3 TES v0 only", 3, v0_tes, None),
        ("N=3 TES v1 only", 3, None, v1_tes),
        ("N=4 TES v0+v1", 4, v0_tes, v1_tes),
    ]

    results = [evaluate_case(name, deg, p_start, p_end, v0, v1) for (name, deg, v0, v1) in cases]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange"]

    # Draw Earth and KOZ circles in xy projection for quick readability
    ang = np.linspace(0.0, 2.0 * np.pi, 600)
    earth_xy = constants.EARTH_RADIUS_KM * np.column_stack([np.cos(ang), np.sin(ang)])
    koz_xy = constants.KOZ_RADIUS * np.column_stack([np.cos(ang), np.sin(ang)])
    ax.plot(earth_xy[:, 0], earth_xy[:, 1], color="black", lw=1.0, alpha=0.7, label="Earth radius")
    ax.plot(koz_xy[:, 0], koz_xy[:, 1], color="black", lw=1.5, ls="--", alpha=0.8, label="KOZ radius")

    ax.scatter([p_start[0]], [p_start[1]], s=80, c="limegreen", edgecolor="black", label="Start")
    ax.scatter([p_end[0]], [p_end[1]], s=80, c="gold", edgecolor="black", label="End")

    for c, res in zip(colors, results):
        if not res["ok"]:
            print(f"[FAIL DRAW] {res['name']}: {res['error']}")
            continue

        xyz = res["curve_xyz"]
        cp = res["control_points"]
        status = "KOZ OK" if res["koz_ok"] else "KOZ VIOLATION"
        label = f"{res['name']} | min_r={res['min_radius_km']:.2f} km | {status}"

        ax.plot(xyz[:, 0], xyz[:, 1], color=c, lw=2.0, label=label)
        ax.plot(cp[:, 0], cp[:, 1], color=c, lw=1.0, alpha=0.45, marker="o", ms=4)

    ax.set_title("Boundary-velocity diagnostic (xy projection)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    # Print compact table to terminal
    print("=" * 100)
    print(f"{'Case':34s} | {'Drawn':5s} | {'Min radius (km)':15s} | {'KOZ pass':8s}")
    print("-" * 100)
    for res in results:
        drawn = "yes" if res["ok"] else "no"
        min_r = f"{res['min_radius_km']:.3f}" if res["ok"] else "-"
        koz_pass = str(bool(res["koz_ok"])) if res["ok"] else "-"
        print(f"{res['name'][:34]:34s} | {drawn:5s} | {min_r:15s} | {koz_pass:8s}")
    print("=" * 100)
    print(f"KOZ radius: {constants.KOZ_RADIUS:.3f} km")

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boundary_velocity_diagnostic.png"
    fig.savefig(out_path, dpi=220)
    print(f"Saved figure: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
