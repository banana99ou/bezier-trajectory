"""
Pillar 1 -- Independent NLP cross-check.

Solve the SAME transfer problem with an independent solver (SciPy trust-constr)
as a DIRECT NLP on the true nonconvex objective J_true with the TRUE dense-grid
KOZ constraint, then compare to the Rust SCvx result.

Because the Rust solver uses the CONSERVATIVE convex-hull KOZ (control polygon,
per de Casteljau segment), its feasible set is a subset of the true set, so
    J_true(x*_rust(n_seg)) >= J_true(x*_scipy)
and the gap Delta(n_seg) should SHRINK as n_seg grows (polygon -> curve). That
monotone shrink is the positive validation signal.

Run:  .venv/bin/python tools/verify/nlp_crosscheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
from scipy.optimize import Bounds, BFGS, NonlinearConstraint, minimize

from tools.verify import harness_common as H
from orbital_docking.constraints import build_boundary_constraints

N_KOZ = 1000          # dense tau grid for the true KOZ constraint
N_SEGS = [8, 16, 32, 64]
HEADLINE_NSEG = 16
OUT = H.ARTIFACT_ROOT / "pillar1_nlp"


def _bounds(scenario):
    P0, PN = scenario["P_start"], scenario["P_end"]
    N = scenario["N"]
    dim = 3
    nvar = (N + 1) * dim
    lb = np.full(nvar, -np.inf)
    ub = np.full(nvar, np.inf)
    for d in range(dim):
        lb[d] = ub[d] = P0[d]
        lb[N * dim + d] = ub[N * dim + d] = PN[d]
    return Bounds(lb, ub)


def _koz_constraint(scenario):
    N = scenario["N"]
    taus = np.linspace(0.0, 1.0, N_KOZ)
    Bmat = H.bernstein_matrix(N, taus)  # (N_KOZ, N+1)

    def fun(x):
        P = x.reshape(N + 1, 3)
        r = Bmat @ P
        return np.linalg.norm(r, axis=1)

    def jac(x):
        P = x.reshape(N + 1, 3)
        r = Bmat @ P
        rn = np.linalg.norm(r, axis=1)
        unit = r / rn[:, None]
        J = np.zeros((N_KOZ, (N + 1) * 3))
        for d in range(3):
            J[:, d::3] = unit[:, d][:, None] * Bmat
        return J

    return NonlinearConstraint(fun, lb=scenario["r_e"], ub=np.inf, jac=jac)


OBJ_SCALE = 1e6  # condition the ~1e-5 objective so trust-constr tolerances bite


def solve_nlp(scenario, x0):
    N, T = scenario["N"], scenario["T"]
    koz = _koz_constraint(scenario)
    bcs = build_boundary_constraints(scenario["P_init"], v0=scenario["v0"],
                                     v1=scenario["v1"], dim=3, T=T)

    def fun(x):
        J, _ = H.J_true_and_grad(x.reshape(N + 1, 3), T, n_dense=600)
        return OBJ_SCALE * J

    def jac(x):
        _, g = H.J_true_and_grad(x.reshape(N + 1, 3), T, n_dense=600)
        return OBJ_SCALE * g

    res = minimize(
        fun, np.asarray(x0, float).reshape(-1),
        jac=jac, hess=BFGS(),
        method="trust-constr",
        bounds=_bounds(scenario),
        constraints=[koz, *bcs],
        options={"maxiter": 800, "gtol": 1e-8, "xtol": 1e-12, "verbose": 0},
    )
    return res


def run(scenario_name="phase120"):
    sc = H.make_scenario(scenario_name)
    T = sc["T"]

    # Rust SCvx solutions per n_seg (energy).
    rust = {}
    for ns in N_SEGS:
        P, info = H.run_rust(sc, n_seg=ns, objective_mode="energy")
        rust[ns] = dict(P=P, Jtrue=H.J_true(P, T), min_r=H.min_radius(P),
                        iters=int(info["iterations"]),
                        conv=int(info.get("scvx_converged", -1)))

    # The reference optimum is the COLD start: SciPy from the straight-line P_init
    # (deep inside the KOZ), carrying NO Rust information. This is the only
    # Rust-independent optimality signal in the harness, so it is the reference AND is
    # gated below. The WARM start (from the finest Rust solution) is a corroborator: if
    # it lands at the same optimum, the reference is start-independent, not a fluke.
    res_cold = solve_nlp(sc, sc["P_init"])
    P_cold = res_cold.x.reshape(sc["N"] + 1, 3)
    J_cold, minr_cold = H.J_true(P_cold, T), H.min_radius(P_cold)

    res_warm = solve_nlp(sc, rust[64]["P"])
    P_warm = res_warm.x.reshape(sc["N"] + 1, 3)
    J_warm, minr_warm = H.J_true(P_warm, T), H.min_radius(P_warm)

    # Reference = cold (independent of Rust).
    J_scipy, minr_scipy, res_ref = J_cold, minr_cold, res_cold

    # Warm-start relative move off the Rust point (diagnostic, not gated).
    warm_dx = (np.linalg.norm(res_warm.x - rust[64]["P"].reshape(-1))
               / (np.linalg.norm(rust[64]["P"]) + 1e-12))

    rows = []
    for ns in N_SEGS:
        gap = rust[ns]["Jtrue"] - J_scipy
        rows.append(dict(
            scenario=scenario_name, n_seg=ns,
            iters_rust=rust[ns]["iters"], conv_rust=rust[ns]["conv"],
            J_true_rust=f"{rust[ns]['Jtrue']:.6e}",
            J_true_scipy=f"{J_scipy:.6e}",
            gap_abs=f"{gap:.6e}",
            gap_pct=f"{100.0 * gap / J_scipy:.3f}",
            min_r_rust=f"{rust[ns]['min_r']:.3f}",
            min_r_scipy=f"{minr_scipy:.3f}",
        ))
    H.write_csv(OUT / "crosscheck.csv", rows)

    # PASS criteria. gap = J_rust - J_scipy(cold) >= 0 => Rust is conservative ABOVE the true
    # independent optimum (expected: the convex-hull KOZ shrinks Rust's feasible set).
    gap16 = rust[HEADLINE_NSEG]["Jtrue"] - J_scipy
    gap64 = rust[64]["Jtrue"] - J_scipy
    rust_feasible = all(rust[ns]["min_r"] >= sc["r_e"] - 1e-6 for ns in N_SEGS)
    cold_feasible = minr_cold >= sc["r_e"] - 1e-3
    # The cold (Rust-blind) solve must reach a GENUINE first-order optimum: gtol (status 1),
    # not xtol (status 2, which only means the step got small). This is the core certificate.
    cold_converged = (res_cold.status == 1) and (res_cold.optimality < 1e-6)
    # Warm start must land at the SAME optimum -> the reference is start-independent.
    cold_warm_agree = (abs(J_cold - J_warm) / J_cold < 0.02) and (abs(minr_cold - minr_warm) < 1.0)
    # Rust within 10% ABOVE the true optimum. A value BELOW it would mean Rust violates the
    # true KOZ, so the lower bound is gated too (not abs()'d away).
    close_ok = -1e-3 <= gap16 / J_scipy <= 0.10
    shrink_ok = abs(gap64) <= abs(gap16) + 1e-12     # conservatism gap shrinks with n_seg
    passed = (rust_feasible and cold_feasible and cold_converged
              and cold_warm_agree and close_ok and shrink_ok)

    md = [f"# Pillar 1 -- Independent NLP cross-check ({scenario_name})", ""]
    md.append(f"- **Independent reference (COLD start, Rust-blind)**: status={res_cold.status} "
              f"(1=gtol), optimality={res_cold.optimality:.2e}, J_true={J_cold:.6e}, "
              f"min_r={minr_cold:.3f} km, feasible={cold_feasible}")
    md.append(f"- Warm start (from n_seg=64 Rust): status={res_warm.status}, J_true={J_warm:.6e}, "
              f"min_r={minr_warm:.3f} km; cold≈warm agree={cold_warm_agree} "
              f"(|dJ|/J={100*abs(J_cold-J_warm)/J_cold:.3f}%); warm move off Rust={warm_dx:.4f}")
    md.append(f"- gap%(n_seg={HEADLINE_NSEG})={100*gap16/J_scipy:.3f}  "
              f"gap%(64)={100*gap64/J_scipy:.3f}  (Rust conservative above the true optimum; shrinks)")
    md.append("")
    md.append("| n_seg | iters | J_true_rust | J_true_scipy | gap% | min_r_rust | min_r_scipy |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['n_seg']} | {r['iters_rust']} | {r['J_true_rust']} | "
                  f"{r['J_true_scipy']} | {r['gap_pct']} | {r['min_r_rust']} | {r['min_r_scipy']} |")
    md.append("")
    md.append(f"**checks**: rust_feasible={rust_feasible}, cold_feasible={cold_feasible}, "
              f"cold_converged(gtol,opt<1e-6)={cold_converged}, cold≈warm={cold_warm_agree}, "
              f"close(0≤gap≤10%)={close_ok}, gap_shrinks={shrink_ok}")
    md.append("")
    md.append(f"## VERDICT: {'PASS' if passed else 'FAIL'}")
    H.write_text(OUT / "summary.md", "\n".join(md))

    print("\n".join(md))
    return passed


if __name__ == "__main__":
    run("phase120")
