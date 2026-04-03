"""
Diagnose why trajectory shape and acceleration-profile trends can diverge.

This script runs targeted ablations to evaluate four hypotheses:
  H1) KOZ linearization conservatism increases with n_seg.
  H2) Lack of trust-region can cause drift/sensitivity (proxy via init perturbation).
  H3) Boundary-velocity scaling mismatch affects geometry and effort.
  H4) Surrogate objective vs dense profile integral mismatch.

Outputs are printed as compact tables plus heuristic verdicts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from orbital_docking import BezierCurve, constants, generate_initial_control_points, optimize_orbital_docking
from orbital_docking.visualization import accel_gravity_total_km_s2


@dataclass
class RunMetrics:
    tag: str
    n_seg: int
    feasible: bool
    iterations: int
    r_min_km: float
    r_max_km: float
    path_len_km: float
    u_max_m_s2: float
    u_mean_m_s2: float
    u_apex_m_s2: float
    j_surrogate_tau: float
    j_dense_tau: float
    j_plot_tau: float
    bc_v0_rel_err: float | None
    bc_v1_rel_err: float | None


def ccw_equatorial_tangent(pos_xyz: np.ndarray) -> np.ndarray:
    x, y, _ = pos_xyz
    r_xy = np.hypot(x, y)
    if r_xy < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([-y / r_xy, x / r_xy, 0.0], dtype=float)


def make_endpoints() -> tuple[np.ndarray, np.ndarray]:
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


def default_boundary_velocities(p_start: np.ndarray, p_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Match current project convention (KOZ circular speed + tangent direction).
    v_mag = np.sqrt(constants.EARTH_MU_SCALED / constants.KOZ_RADIUS)
    v0 = v_mag * ccw_equatorial_tangent(p_start)
    v1 = v_mag * ccw_equatorial_tangent(p_end)
    return v0, v1


def _trapz_mean_square(norm_vals_km_s2: np.ndarray, ts: np.ndarray) -> float:
    # J = integral_0^1 ||u||^2 d tau (units: (km/s^2)^2).
    return float(np.trapezoid(norm_vals_km_s2**2, ts))


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return np.nan
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_metrics(
    p_opt: np.ndarray,
    info: dict,
    n_seg: int,
    tag: str,
    target_v0_km_s: np.ndarray | None,
    target_v1_km_s: np.ndarray | None,
    dense_samples: int,
    plot_samples: int,
) -> RunMetrics:
    curve = BezierCurve(p_opt)
    ts_dense = np.linspace(0.0, 1.0, dense_samples)
    ts_plot = np.linspace(0.0, 1.0, plot_samples)
    t_transfer = float(info.get("T_transfer_s", constants.TRANSFER_TIME_S))
    degree = p_opt.shape[0] - 1

    xyz = np.array([curve.point(t) for t in ts_dense])
    radii = np.linalg.norm(xyz, axis=1)
    r_min = float(np.min(radii))
    apex_idx = int(np.argmax(radii))
    r_max = float(np.max(radii))
    path_len = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))

    a_geom_dense = np.array([curve.acceleration(t) for t in ts_dense]) / (t_transfer**2)
    a_grav_dense = np.array([accel_gravity_total_km_s2(curve.point(t)) for t in ts_dense])
    u_dense_km_s2 = np.linalg.norm(a_geom_dense - a_grav_dense, axis=1)

    a_geom_plot = np.array([curve.acceleration(t) for t in ts_plot]) / (t_transfer**2)
    a_grav_plot = np.array([accel_gravity_total_km_s2(curve.point(t)) for t in ts_plot])
    u_plot_km_s2 = np.linalg.norm(a_geom_plot - a_grav_plot, axis=1)

    j_surrogate = float(info.get("cost_true_energy", np.nan))
    j_dense = _trapz_mean_square(u_dense_km_s2, ts_dense)
    j_plot = _trapz_mean_square(u_plot_km_s2, ts_plot)

    u_m_s2 = u_dense_km_s2 * 1e3
    u_apex_m_s2 = float(u_m_s2[apex_idx])

    v0_err_rel = None
    v1_err_rel = None
    if target_v0_km_s is not None:
        v0_phys = degree * (p_opt[1] - p_opt[0]) / t_transfer
        den = max(np.linalg.norm(target_v0_km_s), 1e-12)
        v0_err_rel = float(np.linalg.norm(v0_phys - target_v0_km_s) / den)
    if target_v1_km_s is not None:
        v1_phys = degree * (p_opt[-1] - p_opt[-2]) / t_transfer
        den = max(np.linalg.norm(target_v1_km_s), 1e-12)
        v1_err_rel = float(np.linalg.norm(v1_phys - target_v1_km_s) / den)

    return RunMetrics(
        tag=tag,
        n_seg=n_seg,
        feasible=bool(info.get("feasible", False)),
        iterations=int(info.get("iterations", -1)),
        r_min_km=r_min,
        r_max_km=r_max,
        path_len_km=path_len,
        u_max_m_s2=float(np.max(u_m_s2)),
        u_mean_m_s2=float(np.mean(u_m_s2)),
        u_apex_m_s2=u_apex_m_s2,
        j_surrogate_tau=j_surrogate,
        j_dense_tau=j_dense,
        j_plot_tau=j_plot,
        bc_v0_rel_err=v0_err_rel,
        bc_v1_rel_err=v1_err_rel,
    )


def run_single(
    p_init: np.ndarray,
    n_seg: int,
    r_e_km: float,
    v0_for_constraint: np.ndarray | None,
    v1_for_constraint: np.ndarray | None,
    target_v0_physical: np.ndarray | None,
    target_v1_physical: np.ndarray | None,
    max_iter: int,
    tol: float,
    sample_count: int,
    use_cache: bool,
    dense_samples: int,
    plot_samples: int,
    tag: str,
) -> RunMetrics:
    p_opt, info = optimize_orbital_docking(
        p_init,
        n_seg=n_seg,
        r_e=r_e_km,
        max_iter=max_iter,
        tol=tol,
        v0=v0_for_constraint,
        v1=v1_for_constraint,
        sample_count=sample_count,
        verbose=False,
        debug=False,
        use_cache=use_cache,
    )
    return compute_metrics(
        p_opt=p_opt,
        info=info,
        n_seg=n_seg,
        tag=tag,
        target_v0_km_s=target_v0_physical,
        target_v1_km_s=target_v1_physical,
        dense_samples=dense_samples,
        plot_samples=plot_samples,
    )


def print_table(rows: Iterable[RunMetrics], title: str) -> None:
    rows = list(rows)
    print("\n" + "=" * 150)
    print(title)
    print("-" * 150)
    print(
        f"{'tag':24s} | {'n_seg':5s} | {'feas':4s} | {'r_min':8s} | {'r_max':8s} | {'u_apex':8s} | "
        f"{'u_max':8s} | {'J_sur':10s} | {'J_dense':10s} | {'J_plot':10s} | {'v0_err':8s} | {'v1_err':8s}"
    )
    print("-" * 150)
    for r in rows:
        v0e = "-" if r.bc_v0_rel_err is None else f"{r.bc_v0_rel_err:8.3f}"
        v1e = "-" if r.bc_v1_rel_err is None else f"{r.bc_v1_rel_err:8.3f}"
        print(
            f"{r.tag[:24]:24s} | {r.n_seg:5d} | {str(r.feasible):4s} | {r.r_min_km:8.2f} | {r.r_max_km:8.2f} | "
            f"{r.u_apex_m_s2:8.3f} | {r.u_max_m_s2:8.3f} | {r.j_surrogate_tau:10.3e} | {r.j_dense_tau:10.3e} | "
            f"{r.j_plot_tau:10.3e} | {v0e} | {v1e}"
        )
    print("=" * 150)


def verdict_h1(rows_base: list[RunMetrics], rows_relaxed: list[RunMetrics]) -> str:
    base_sorted = sorted(rows_base, key=lambda r: r.n_seg)
    relaxed_sorted = sorted(rows_relaxed, key=lambda r: r.n_seg)
    if len(base_sorted) < 2 or len(relaxed_sorted) < 2:
        return "inconclusive (not enough points)"
    growth = base_sorted[-1].r_max_km - base_sorted[0].r_max_km
    med_base = float(np.median([r.r_max_km for r in base_sorted]))
    med_relaxed = float(np.median([r.r_max_km for r in relaxed_sorted]))
    drop = med_base - med_relaxed
    if growth > 1.0 and drop > 0.5:
        return f"supported (r_max growth={growth:.2f} km, relaxation drop={drop:.2f} km)"
    return f"weak/inconclusive (r_max growth={growth:.2f} km, relaxation drop={drop:.2f} km)"


def verdict_h2(rows_perturbed: list[RunMetrics]) -> str:
    if len(rows_perturbed) < 3:
        return "inconclusive (not enough perturbation runs)"
    rmax_vals = np.array([r.r_max_km for r in rows_perturbed], dtype=float)
    jd_vals = np.array([r.j_dense_tau for r in rows_perturbed], dtype=float)
    r_cv = float(np.std(rmax_vals) / max(np.mean(rmax_vals), 1e-12))
    j_cv = float(np.std(jd_vals) / max(np.mean(jd_vals), 1e-12))
    if r_cv > 0.01 or j_cv > 0.05:
        return f"supported by proxy (init sensitivity: CV_rmax={r_cv:.3f}, CV_Jdense={j_cv:.3f})"
    return f"weak/inconclusive by proxy (CV_rmax={r_cv:.3f}, CV_Jdense={j_cv:.3f})"


def verdict_h3(rows_as_is: list[RunMetrics], rows_scaled: list[RunMetrics]) -> str:
    if not rows_as_is or not rows_scaled:
        return "inconclusive (missing A/B runs)"
    err_as = np.array(
        [np.nanmean([r.bc_v0_rel_err, r.bc_v1_rel_err]) for r in rows_as_is],
        dtype=float,
    )
    err_sc = np.array(
        [np.nanmean([r.bc_v0_rel_err, r.bc_v1_rel_err]) for r in rows_scaled],
        dtype=float,
    )
    med_as = float(np.nanmedian(err_as))
    med_sc = float(np.nanmedian(err_sc))
    best = min(med_as, med_sc)
    worst = max(med_as, med_sc)
    if best < 0.1 and worst > 0.5:
        better = "as-is" if med_as < med_sc else "scaled-by-T"
        return (
            "supported "
            f"(best endpoint-velocity error is {better}: as-is={med_as:.3f}, scaled={med_sc:.3f})"
        )
    return (
        "weak/inconclusive "
        f"(endpoint velocity relative error: as-is={med_as:.3f}, scaled={med_sc:.3f})"
    )


def verdict_h4(rows: list[RunMetrics]) -> str:
    if len(rows) < 3:
        return "inconclusive (not enough points)"
    js = np.array([r.j_surrogate_tau for r in rows], dtype=float)
    jd = np.array([r.j_dense_tau for r in rows], dtype=float)
    jp = np.array([r.j_plot_tau for r in rows], dtype=float)
    if np.any(~np.isfinite(js)) or np.any(~np.isfinite(jd)) or np.any(~np.isfinite(jp)):
        return "inconclusive (non-finite values)"
    rel_gap = np.abs(js - jd) / np.maximum(np.abs(jd), 1e-12)
    corr_rank = _rank_corr(js, jd)
    corr_plot = _rank_corr(jd, jp)
    if np.nanmedian(rel_gap) > 0.2 or (np.isfinite(corr_rank) and corr_rank < 0.7):
        return (
            f"supported (median rel gap surrogate-dense={np.nanmedian(rel_gap):.3f}, "
            f"rank corr surrogate-dense={corr_rank:.3f}, rank corr dense-plot={corr_plot:.3f})"
        )
    return (
        f"weak/inconclusive (median rel gap surrogate-dense={np.nanmedian(rel_gap):.3f}, "
        f"rank corr surrogate-dense={corr_rank:.3f}, rank corr dense-plot={corr_plot:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation diagnostics for objective/acceleration-profile mismatch.")
    parser.add_argument("--degree", type=int, default=3, choices=[2, 3, 4], help="Bezier degree to test.")
    parser.add_argument(
        "--segment-counts",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32, 64],
        help="n_seg values for sweeps.",
    )
    parser.add_argument("--max-iter", type=int, default=120, help="Max SCP iterations.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Convergence tolerance.")
    parser.add_argument("--sample-count", type=int, default=100, help="Linearization sample_count in optimizer.")
    parser.add_argument("--dense-samples", type=int, default=1200, help="Dense samples for integral metrics.")
    parser.add_argument("--plot-samples", type=int, default=300, help="Samples that emulate plotted profile resolution.")
    parser.add_argument(
        "--koz-offsets-km",
        nargs="+",
        type=float,
        default=[0.0, -2.0],
        help="Offsets applied to KOZ radius for H1 (first should be 0).",
    )
    parser.add_argument(
        "--perturb-sigma-km",
        type=float,
        default=3.0,
        help="Stddev for interior control-point perturbation (H2 proxy).",
    )
    parser.add_argument("--perturb-runs", type=int, default=5, help="Number of perturbation runs for H2 proxy.")
    parser.add_argument("--perturb-n-seg", type=int, default=32, help="n_seg used for H2 perturbation study.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--use-cache", action="store_true", help="Use optimizer cache (default: false).")
    args = parser.parse_args()

    if not args.koz_offsets_km:
        raise ValueError("Provide at least one --koz-offsets-km value (first should be 0.0).")

    p_start, p_end = make_endpoints()
    p_init = generate_initial_control_points(args.degree, p_start, p_end)
    v0_phys, v1_phys = default_boundary_velocities(p_start, p_end)
    t_transfer = constants.TRANSFER_TIME_S

    # H1: KOZ conservatism with n_seg and KOZ relaxation.
    h1_rows_by_offset: dict[float, list[RunMetrics]] = {}
    for off in args.koz_offsets_km:
        r_e_now = constants.KOZ_RADIUS + float(off)
        rows = []
        for n_seg in args.segment_counts:
            m = run_single(
                p_init=p_init,
                n_seg=n_seg,
                r_e_km=r_e_now,
                v0_for_constraint=v0_phys,
                v1_for_constraint=v1_phys,
                target_v0_physical=v0_phys,
                target_v1_physical=v1_phys,
                max_iter=args.max_iter,
                tol=args.tol,
                sample_count=args.sample_count,
                use_cache=args.use_cache,
                dense_samples=args.dense_samples,
                plot_samples=args.plot_samples,
                tag=f"H1_off{off:+.1f}",
            )
            rows.append(m)
        h1_rows_by_offset[off] = rows
        print_table(rows, f"H1 sweep: KOZ radius offset {off:+.1f} km")

    # H3: Boundary velocity scaling A/B.
    h3_rows_as_is = []
    h3_rows_scaled = []
    for n_seg in args.segment_counts:
        h3_rows_as_is.append(
            run_single(
                p_init=p_init,
                n_seg=n_seg,
                r_e_km=constants.KOZ_RADIUS,
                v0_for_constraint=v0_phys,
                v1_for_constraint=v1_phys,
                target_v0_physical=v0_phys,
                target_v1_physical=v1_phys,
                max_iter=args.max_iter,
                tol=args.tol,
                sample_count=args.sample_count,
                use_cache=args.use_cache,
                dense_samples=args.dense_samples,
                plot_samples=args.plot_samples,
                tag="H3_as_is",
            )
        )
        h3_rows_scaled.append(
            run_single(
                p_init=p_init,
                n_seg=n_seg,
                r_e_km=constants.KOZ_RADIUS,
                v0_for_constraint=v0_phys * t_transfer,
                v1_for_constraint=v1_phys * t_transfer,
                target_v0_physical=v0_phys,
                target_v1_physical=v1_phys,
                max_iter=args.max_iter,
                tol=args.tol,
                sample_count=args.sample_count,
                use_cache=args.use_cache,
                dense_samples=args.dense_samples,
                plot_samples=args.plot_samples,
                tag="H3_scaled_by_T",
            )
        )
    print_table(h3_rows_as_is + h3_rows_scaled, "H3 A/B: boundary velocity scaling (as-is vs multiplied by T)")

    # H4: Compare surrogate vs dense-vs-plot integral on a consistent run family.
    # Use the "scaled_by_T" family because it is physically consistent for endpoint velocity checks.
    h4_rows = h3_rows_scaled
    print_table(h4_rows, "H4 data source: surrogate vs dense/profile integrals")

    # H2 (proxy): perturb initial interior control points, same n_seg.
    rng = np.random.default_rng(args.seed)
    h2_rows = []
    for k in range(args.perturb_runs):
        p0 = p_init.copy()
        if p0.shape[0] > 2:
            perturb = rng.normal(loc=0.0, scale=args.perturb_sigma_km, size=p0[1:-1].shape)
            p0[1:-1] += perturb
        h2_rows.append(
            run_single(
                p_init=p0,
                n_seg=args.perturb_n_seg,
                r_e_km=constants.KOZ_RADIUS,
                v0_for_constraint=v0_phys * t_transfer,
                v1_for_constraint=v1_phys * t_transfer,
                target_v0_physical=v0_phys,
                target_v1_physical=v1_phys,
                max_iter=args.max_iter,
                tol=args.tol,
                sample_count=args.sample_count,
                use_cache=False,  # force fresh runs for sensitivity study
                dense_samples=args.dense_samples,
                plot_samples=args.plot_samples,
                tag=f"H2_perturb_{k:02d}",
            )
        )
    print_table(h2_rows, f"H2 proxy: initialization sensitivity @ n_seg={args.perturb_n_seg}")

    # Verdicts.
    print("\n" + "#" * 84)
    print("Heuristic verdicts")
    print("#" * 84)
    h1_base = h1_rows_by_offset.get(0.0, [])
    h1_relaxed_key = args.koz_offsets_km[1] if len(args.koz_offsets_km) > 1 else args.koz_offsets_km[0]
    h1_relaxed = h1_rows_by_offset.get(h1_relaxed_key, [])
    print(f"H1 (KOZ conservatism): {verdict_h1(h1_base, h1_relaxed)}")
    print(f"H2 (no trust-region drift proxy): {verdict_h2(h2_rows)}")
    print(f"H3 (BC scaling mismatch): {verdict_h3(h3_rows_as_is, h3_rows_scaled)}")
    print(f"H4 (surrogate/profile mismatch): {verdict_h4(h4_rows)}")
    print("#" * 84)


if __name__ == "__main__":
    main()
