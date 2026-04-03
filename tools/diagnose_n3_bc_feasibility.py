"""
Run N=3 optimizer feasibility diagnostics across boundary-velocity BC modes.

Modes tested:
  - none: no velocity BC
  - v0_only: initial velocity BC only
  - v1_only: final velocity BC only
  - both: both initial and final velocity BC

This script runs the actual optimizer and summarizes feasibility vs n_seg.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from orbital_docking import constants, optimize_all_segment_counts, generate_initial_control_points


def ccw_equatorial_tangent(pos_xyz: np.ndarray) -> np.ndarray:
    """Unit tangent in xy-plane, counter-clockwise."""
    x, y, _ = pos_xyz
    r_xy = np.hypot(x, y)
    if r_xy < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([-y / r_xy, x / r_xy, 0.0], dtype=float)


def velocity_current_model(pos_xyz: np.ndarray) -> np.ndarray:
    """Current project convention: KOZ circular-speed magnitude + tangent direction."""
    v_mag = np.sqrt(constants.EARTH_MU_SCALED / constants.KOZ_RADIUS)
    return v_mag * ccw_equatorial_tangent(pos_xyz)


def velocity_tes_model(pos_xyz: np.ndarray) -> np.ndarray:
    """TES convention: local circular-speed magnitude + tangent direction."""
    r_norm = np.linalg.norm(pos_xyz)
    if r_norm < 1e-12:
        return np.zeros(3, dtype=float)
    v_mag = np.sqrt(constants.EARTH_MU_SCALED / r_norm)
    return v_mag * ccw_equatorial_tangent(pos_xyz)


def make_endpoints():
    """Simplified coplanar projection of the Progress-to-ISS inspired scenario."""
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
    return p_start, p_end


def run_mode(
    mode_name: str,
    p_init_n3: np.ndarray,
    v0: np.ndarray | None,
    v1: np.ndarray | None,
    segment_counts: list[int],
    max_iter: int,
    tol: float,
    use_cache: bool,
    debug: bool,
):
    results = optimize_all_segment_counts(
        p_init_n3,
        r_e=constants.KOZ_RADIUS,
        segment_counts=segment_counts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
        debug=debug,
        use_cache=use_cache,
        v0=v0,
        v1=v1,
    )

    rows = []
    for n_seg, _p_opt, info in results:
        rows.append(
            {
                "mode": mode_name,
                "n_seg": int(n_seg),
                "feasible": bool(info.get("feasible", False)),
                "min_radius_km": float(info.get("min_radius", np.nan)),
                "max_control_accel_ms2": float(info.get("max_control_accel_ms2", np.nan)),
                "iterations": int(info.get("iterations", -1)),
                "elapsed_time_s": float(info.get("elapsed_time", np.nan)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Diagnose N=3 feasibility under boundary velocity BC modes.")
    parser.add_argument(
        "--velocity-model",
        choices=["current", "tes"],
        default="current",
        help="Velocity model for v0/v1 boundary vectors.",
    )
    parser.add_argument(
        "--segment-counts",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32, 64],
        help="Segment counts to test.",
    )
    parser.add_argument("--max-iter", type=int, default=120, help="Max SCP iterations per n_seg.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance.")
    parser.add_argument("--use-cache", action="store_true", help="Use cache (default: disabled).")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints inside optimizer.")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window.")
    args = parser.parse_args()

    p_start, p_end = make_endpoints()
    p_init_n3 = generate_initial_control_points(3, p_start, p_end)

    vel_fn = velocity_current_model if args.velocity_model == "current" else velocity_tes_model
    v0 = vel_fn(p_start)
    v1 = vel_fn(p_end)

    modes = [
        ("none", None, None),
        ("v0_only", v0, None),
        ("v1_only", None, v1),
        ("both", v0, v1),
    ]

    all_rows = []
    for mode_name, mode_v0, mode_v1 in modes:
        print(f"[RUN] mode={mode_name} model={args.velocity_model}")
        rows = run_mode(
            mode_name=mode_name,
            p_init_n3=p_init_n3,
            v0=mode_v0,
            v1=mode_v1,
            segment_counts=args.segment_counts,
            max_iter=args.max_iter,
            tol=args.tol,
            use_cache=args.use_cache,
            debug=args.debug,
        )
        all_rows.extend(rows)

    # Terminal summary table
    print("\n" + "=" * 110)
    print(f"{'Mode':10s} | {'n_seg':5s} | {'feasible':8s} | {'min_radius_km':13s} | {'KOZ_margin_km':13s} | {'max_u(m/s²)':11s}")
    print("-" * 110)
    for r in all_rows:
        margin = r["min_radius_km"] - constants.KOZ_RADIUS
        print(
            f"{r['mode']:10s} | {r['n_seg']:5d} | {str(r['feasible']):8s} | "
            f"{r['min_radius_km']:13.3f} | {margin:13.3f} | {r['max_control_accel_ms2']:11.3f}"
        )
    print("=" * 110)

    # Plot min radius vs n_seg per mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    color_map = {"none": "tab:green", "v0_only": "tab:blue", "v1_only": "tab:purple", "both": "tab:red"}

    for mode_name, _, _ in modes:
        mode_rows = sorted([r for r in all_rows if r["mode"] == mode_name], key=lambda x: x["n_seg"])
        x = [r["n_seg"] for r in mode_rows]
        y_minr = [r["min_radius_km"] for r in mode_rows]
        y_u = [r["max_control_accel_ms2"] for r in mode_rows]
        ax1.plot(x, y_minr, marker="o", lw=2, color=color_map[mode_name], label=mode_name)
        ax2.plot(x, y_u, marker="o", lw=2, color=color_map[mode_name], label=mode_name)

    ax1.axhline(constants.KOZ_RADIUS, color="black", ls="--", lw=1.5, label="KOZ radius")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("n_seg")
    ax1.set_ylabel("min radius (km)")
    ax1.set_title(f"N=3 feasibility diagnostic (velocity model: {args.velocity_model})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("n_seg")
    ax2.set_ylabel("max control accel (m/s²)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"n3_bc_feasibility_{args.velocity_model}.png"
    fig.savefig(out_path, dpi=220)
    print(f"Saved figure: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
