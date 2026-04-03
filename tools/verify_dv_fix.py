"""
Automated verification for the delta-v proxy objective fix.

Goal:
  Compare objective_mode="energy" (legacy L2 surrogate) vs objective_mode="dv"
  under the same scenario/settings and decide PASS/FAIL using explicit thresholds.

Usage examples:
  python verify_dv_fix.py
  python verify_dv_fix.py --degrees 3 4 --nseg 8 32 64 --max-iter 600 --no-cache
  python verify_dv_fix.py --min-dv-improve-pct 10 --min-rmax-reduce-pct 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from orbital_docking import BezierCurve, constants, generate_initial_control_points, optimize_orbital_docking


@dataclass
class CaseResult:
    degree: int
    n_seg: int
    objective: str
    feasible: bool
    min_radius_km: float
    r_max_km: float
    iterations: int
    dv_proxy_m_s: float | None
    cost_true_energy: float
    elapsed_s: float
    drift_slope_km_per_iter: float | None


def _eci_from_circular_elements(radius_km: float, inc_deg: float, raan_deg: float, u_deg: float) -> tuple[np.ndarray, np.ndarray]:
    mu = constants.EARTH_MU_SCALED
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    u = np.deg2rad(u_deg)

    def _rotz(theta_rad: float) -> np.ndarray:
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _rotx(theta_rad: float) -> np.ndarray:
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    r_pqw = np.array([radius_km * np.cos(u), radius_km * np.sin(u), 0.0], dtype=float)
    v_circ = np.sqrt(mu / radius_km)
    v_pqw = v_circ * np.array([-np.sin(u), np.cos(u), 0.0], dtype=float)
    q = _rotz(raan) @ _rotx(inc)
    return q @ r_pqw, q @ v_pqw


def make_scenario_endpoints(chaser_lag_deg: float) -> tuple[np.ndarray, np.ndarray]:
    earth_r = constants.EARTH_RADIUS_KM
    progress_alt = 245.0
    iss_alt = 400.0
    inc_deg = 51.64
    raan_deg = 0.0
    iss_u_deg = 45.0
    progress_u_deg = iss_u_deg - float(chaser_lag_deg)

    p_start, _ = _eci_from_circular_elements(earth_r + progress_alt, inc_deg, raan_deg, progress_u_deg)
    p_end, _ = _eci_from_circular_elements(earth_r + iss_alt, inc_deg, raan_deg, iss_u_deg)
    return p_start, p_end


def _rmax_from_curve(P_opt: np.ndarray, samples: int) -> float:
    curve = BezierCurve(P_opt)
    ts = np.linspace(0.0, 1.0, int(samples))
    pts = np.array([curve.point(t) for t in ts])
    return float(np.max(np.linalg.norm(pts, axis=1)))


def _drift_slope_from_history(info: dict) -> float | None:
    hist = info.get("history")
    if not isinstance(hist, list) or len(hist) < 3:
        return None
    ys = np.array([float(h.get("r_max_km", np.nan)) for h in hist], dtype=float)
    xs = np.arange(1, ys.size + 1, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 3:
        return None
    # Fit slope over latter 70% to detect sustained drift.
    start = int(np.floor(0.3 * xs.size))
    xs = xs[start:]
    ys = ys[start:]
    if xs.size < 3:
        return None
    x0 = xs - np.mean(xs)
    denom = float(np.sum(x0 * x0))
    if denom < 1e-12:
        return None
    slope = float(np.sum(x0 * (ys - np.mean(ys))) / denom)
    return slope


def run_case(
    degree: int,
    n_seg: int,
    objective_mode: str,
    max_iter: int,
    tol: float,
    sample_count: int,
    use_cache: bool,
    debug_history: bool,
    history_samples: int,
    rmax_samples: int,
    p_start: np.ndarray,
    p_end: np.ndarray,
    scp_prox: float,
    scp_trust_radius: float,
) -> CaseResult:
    p_init = generate_initial_control_points(int(degree), p_start, p_end)
    p_opt, info = optimize_orbital_docking(
        p_init,
        n_seg=int(n_seg),
        r_e=constants.KOZ_RADIUS,
        max_iter=int(max_iter),
        tol=float(tol),
        v0=None,
        v1=None,
        sample_count=int(sample_count),
        objective_mode=str(objective_mode),
        scp_prox_weight=float(scp_prox),
        scp_trust_radius=float(scp_trust_radius),
        verbose=True,
        debug=False,
        use_cache=bool(use_cache),
        ignore_existing_cache=not bool(use_cache),
        store_history=bool(debug_history),
        history_samples=int(history_samples),
    )
    return CaseResult(
        degree=int(degree),
        n_seg=int(n_seg),
        objective=str(objective_mode),
        feasible=bool(info.get("feasible", False)),
        min_radius_km=float(info.get("min_radius", np.nan)),
        r_max_km=_rmax_from_curve(p_opt, rmax_samples),
        iterations=int(info.get("iterations", -1)),
        dv_proxy_m_s=None if info.get("dv_proxy_m_s") is None else float(info.get("dv_proxy_m_s")),
        cost_true_energy=float(info.get("cost_true_energy", np.nan)),
        elapsed_s=float(info.get("elapsed_time", np.nan)),
        drift_slope_km_per_iter=_drift_slope_from_history(info),
    )


def pct_improvement(old: float, new: float) -> float:
    den = max(abs(float(old)), 1e-12)
    return 100.0 * (float(old) - float(new)) / den


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify delta-v proxy objective fix.")
    parser.add_argument("--degrees", nargs="+", type=int, default=[3, 4], choices=[2, 3, 4])
    parser.add_argument("--nseg", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--chaser-lag-deg", type=float, default=30.0)
    parser.add_argument("--rmax-samples", type=int, default=600)
    parser.add_argument("--history-samples", type=int, default=200)
    parser.add_argument("--with-history", action="store_true", help="Store SCP history and compute drift slope.")
    parser.add_argument("--use-cache", action="store_true", help="Use cache for speed. Default is fresh runs.")
    parser.add_argument("--scp-prox", type=float, default=0.0, help="Outer SCP proximal weight.")
    parser.add_argument("--scp-trust-radius", type=float, default=0.0, help="Outer SCP trust radius.")

    # Acceptance thresholds
    parser.add_argument("--min-dv-improve-pct", type=float, default=20.0)
    parser.add_argument("--min-rmax-reduce-pct", type=float, default=20.0)
    parser.add_argument("--max-drift-slope-ratio", type=float, default=1.0,
                        help="Require drift_slope(dv) <= ratio * drift_slope(energy) when history is enabled.")

    parser.add_argument("--save-json", type=str, default="", help="Optional path to save full JSON results.")
    args = parser.parse_args()

    p_start, p_end = make_scenario_endpoints(args.chaser_lag_deg)

    pairs = []
    all_rows: list[CaseResult] = []
    print("=" * 92)
    print("A/B verification: objective_mode='energy' vs 'dv'")
    print(f"degrees={args.degrees}, nseg={args.nseg}, max_iter={args.max_iter}, tol={args.tol}")
    print(
        f"use_cache={args.use_cache}, with_history={args.with_history}, "
        f"scp_prox={args.scp_prox}, scp_trust_radius={args.scp_trust_radius}, "
        f"KOZ={constants.KOZ_RADIUS:.2f} km"
    )
    print("=" * 92)

    for degree in args.degrees:
        for n_seg in args.nseg:
            row_energy = run_case(
                degree=degree,
                n_seg=n_seg,
                objective_mode="energy",
                max_iter=args.max_iter,
                tol=args.tol,
                sample_count=args.sample_count,
                use_cache=args.use_cache,
                debug_history=args.with_history,
                history_samples=args.history_samples,
                rmax_samples=args.rmax_samples,
                p_start=p_start,
                p_end=p_end,
                scp_prox=args.scp_prox,
                scp_trust_radius=args.scp_trust_radius,
            )
            print(f"n_seg {n_seg} energy done.")
            row_dv = run_case(
                degree=degree,
                n_seg=n_seg,
                objective_mode="dv",
                max_iter=args.max_iter,
                tol=args.tol,
                sample_count=args.sample_count,
                use_cache=args.use_cache,
                debug_history=args.with_history,
                history_samples=args.history_samples,
                rmax_samples=args.rmax_samples,
                p_start=p_start,
                p_end=p_end,
                scp_prox=args.scp_prox,
                scp_trust_radius=args.scp_trust_radius,
            )
            print(f"n_seg {n_seg} dv done.")
            all_rows.extend([row_energy, row_dv])
            pairs.append((row_energy, row_dv))

    # Evaluate pass/fail per pair
    pair_results = []
    n_pass = 0
    for e, d in pairs:
        dv_ok = (e.dv_proxy_m_s is not None and d.dv_proxy_m_s is not None)
        dv_imp = pct_improvement(e.dv_proxy_m_s, d.dv_proxy_m_s) if dv_ok else float("nan")
        rmax_imp = pct_improvement(e.r_max_km, d.r_max_km)
        feasible_ok = bool(e.feasible and d.feasible)
        minrad_ok = bool(
            e.min_radius_km >= constants.KOZ_RADIUS - 1e-6 and d.min_radius_km >= constants.KOZ_RADIUS - 1e-6
        )

        drift_ok = True
        drift_ratio = None
        if args.with_history:
            se = e.drift_slope_km_per_iter
            sd = d.drift_slope_km_per_iter
            if se is not None and sd is not None and abs(se) > 1e-12:
                drift_ratio = float(sd / se)
                drift_ok = bool(drift_ratio <= args.max_drift_slope_ratio)

        passed = (
            feasible_ok
            and minrad_ok
            and dv_ok
            and (dv_imp >= args.min_dv_improve_pct)
            and (rmax_imp >= args.min_rmax_reduce_pct)
            and drift_ok
        )
        if passed:
            n_pass += 1

        pair_results.append(
            {
                "degree": e.degree,
                "n_seg": e.n_seg,
                "feasible_ok": feasible_ok,
                "min_radius_ok": minrad_ok,
                "dv_improve_pct": dv_imp,
                "rmax_reduce_pct": rmax_imp,
                "drift_ratio_dv_over_energy": drift_ratio,
                "passed": passed,
            }
        )

    # Print compact report
    print("\nPer-case A/B results:")
    print("-" * 92)
    print(" N | n_seg | dv_improve% | rmax_reduce% | feasible | min_radius | pass")
    print("-" * 92)
    for r in pair_results:
        print(
            f"{r['degree']:2d} | {r['n_seg']:5d} | "
            f"{r['dv_improve_pct']:10.2f} | {r['rmax_reduce_pct']:11.2f} | "
            f"{str(r['feasible_ok']):8s} | {str(r['min_radius_ok']):10s} | {str(r['passed'])}"
        )
    print("-" * 92)

    overall_pass = (n_pass == len(pair_results))
    print(
        f"Overall: {'PASS' if overall_pass else 'FAIL'} "
        f"({n_pass}/{len(pair_results)} cases passed)"
    )
    print(
        f"Thresholds: dv_improve>={args.min_dv_improve_pct:.1f}%, "
        f"rmax_reduce>={args.min_rmax_reduce_pct:.1f}%"
    )
    if args.with_history:
        print(f"History threshold: drift_ratio(dv/energy)<={args.max_drift_slope_ratio:.2f}")

    payload = {
        "config": {
            "degrees": args.degrees,
            "nseg": args.nseg,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "sample_count": args.sample_count,
            "chaser_lag_deg": args.chaser_lag_deg,
            "rmax_samples": args.rmax_samples,
            "with_history": args.with_history,
            "use_cache": args.use_cache,
            "scp_prox": args.scp_prox,
            "scp_trust_radius": args.scp_trust_radius,
            "thresholds": {
                "min_dv_improve_pct": args.min_dv_improve_pct,
                "min_rmax_reduce_pct": args.min_rmax_reduce_pct,
                "max_drift_slope_ratio": args.max_drift_slope_ratio,
            },
        },
        "cases": [asdict(x) for x in all_rows],
        "pair_results": pair_results,
        "overall_pass": overall_pass,
        "passed_count": n_pass,
        "total_pairs": len(pair_results),
    }

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON report: {out}")

    # Print single-line JSON-ish for quick copy.
    print("\nSummary payload:")
    print(json.dumps({"overall_pass": overall_pass, "passed_count": n_pass, "total_pairs": len(pair_results)}))

    # CI-friendly exit code.
    raise SystemExit(0 if overall_pass else 2)


if __name__ == "__main__":
    main()

