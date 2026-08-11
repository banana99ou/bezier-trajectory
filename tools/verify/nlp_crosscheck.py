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
# Every start that converges at all does so well inside this: phase120 at 310,
# planechange at 459, phase170 at 597. The budget is not what limits phase135 --
# BOTH of its starts return exactly the J they already had at 800 iterations
# (projected 8.319355e-05, straight-line 8.307425e-05) when given 6000, with
# optimality flat at 2.1e-03 and 9.5e-04. It is stalled, not starved, so paying
# 880 s per solve to re-confirm that would buy nothing.
MAXITER = 1500
OPT_TOL = 1e-6   # first-order optimality of the scaled Lagrangian
CV_TOL = 1e-6    # constraint violation, km


def cold_start(scenario, pad=1.02):
    """Rust-blind initial guess: the straight line, pushed clear of the KOZ.

    The straight line between the boundary conditions passes 3400-5900 km INSIDE
    the keep-out sphere on the harder geometries, which is a hostile start for an
    NLP carrying a 1000-point KOZ constraint -- trust-constr spent its whole
    iteration budget restoring feasibility and never reached a first-order point
    on phase135, phase170 or planechange. Projecting the interior control points
    radially out to the KOZ surface costs nothing in independence: it uses the
    KOZ radius and the initial guess, and no quantity produced by the solver
    under test. Measured on phase170 it takes the reference from
    optimality=1.5e-02 (not converged) to 9.7e-09 in 597 iterations, landing on
    the same J as the Rust-warm solve to all seven digits.

    Endpoints are excluded: they are fixed by the boundary conditions.
    """
    P = np.array(scenario["P_init"], float).copy()
    for i in range(1, P.shape[0] - 1):
        n = np.linalg.norm(P[i])
        if n < scenario["r_e"] * pad:
            P[i] *= (scenario["r_e"] * pad) / n
    return P


def straight_start(scenario):
    """The original Rust-blind guess: the straight line between the endpoints."""
    return np.array(scenario["P_init"], float)


# Tried in order; the first one that reaches a first-order point becomes the
# reference. Both are Rust-blind, so which one succeeds changes nothing about the
# independence of the comparison -- only about whether a comparison exists at all.
# Neither start dominates: the projected guess converges on phase120 (310 iters),
# phase170 (597) and planechange (459) where the straight line does not, while on
# phase135 the straight line gets further before stalling.
COLD_STARTS = (("projected", cold_start), ("straight-line", straight_start))


def converged(res):
    """Did this solve reach a first-order point?

    Tested on the CONDITION (optimality and constraint violation), not on the
    exit code. `status == 1` means trust-constr's own gtol fired; a solve can
    satisfy the KKT residual and still exit with status 0 because it hit the
    iteration limit on the same step -- planechange did exactly that, returning
    optimality=1.15e-07 with status=0, and was recorded as a non-reference for
    an exit code rather than for a number.
    """
    return bool(res.optimality < OPT_TOL and res.constr_violation < CV_TOL)


def solve_nlp(scenario, x0):
    N, T = scenario["N"], scenario["T"]
    koz = _koz_constraint(scenario)
    bcs = build_boundary_constraints(scenario["P_init"], v0=scenario["v0"],
                                     v1=scenario["v1"], dim=3, T=T)

    # Defaults (Gauss-Legendre, 192 nodes) deliberately: this must be the SAME
    # objective the gap is later scored with, or the NLP minimizes one function
    # and gets graded on another. The former n_dense=600 uniform mean carried
    # ~0.2% error -- larger than the n_seg=64 gap this pillar reports.
    def fun(x):
        J, _ = H.J_true_and_grad(x.reshape(N + 1, 3), T)
        return OBJ_SCALE * J

    def jac(x):
        _, g = H.J_true_and_grad(x.reshape(N + 1, 3), T)
        return OBJ_SCALE * g

    res = minimize(
        fun, np.asarray(x0, float).reshape(-1),
        jac=jac, hess=BFGS(),
        method="trust-constr",
        bounds=_bounds(scenario),
        constraints=[koz, *bcs],
        options={"maxiter": MAXITER, "gtol": 1e-8, "xtol": 1e-12, "verbose": 0},
    )
    return res


def solve_reference(scenario):
    """Best Rust-blind reference solve. Returns (result, which_start).

    Tries each start in COLD_STARTS and stops at the first that reaches a
    first-order point, so a geometry whose usual start works pays for one solve.
    If none converges, the attempt with the smallest optimality is returned and
    the caller reports NO REFERENCE -- there is no fallback to the warm solve,
    which is seeded from the solver under test and could never be independent.
    """
    attempts = []
    for tag, make in COLD_STARTS:
        res = solve_nlp(scenario, make(scenario))
        attempts.append((tag, res))
        if converged(res):
            return res, tag
    tag, res = min(attempts, key=lambda tr: tr[1].optimality)
    return res, tag


def _run_one(scenario_name):
    """All Pillar 1 quantities for one geometry. Returns (csv_rows, md_lines, passed)."""
    sc = H.make_scenario(scenario_name)
    T = sc["T"]

    # Rust SCvx solutions per n_seg (energy).
    rust = {}
    for ns in N_SEGS:
        P, info = H.run_rust(sc, n_seg=ns)
        rust[ns] = dict(P=P, Jtrue=H.J_true(P, T), min_r=H.min_radius(P),
                        iters=int(info["iterations"]),
                        conv=int(info.get("scvx_converged", -1)))

    # The reference optimum is the COLD start: SciPy from a Rust-blind guess (see
    # `cold_start`), carrying NO Rust information. This is the only Rust-independent
    # optimality signal in the harness, so it is the reference AND is gated below.
    # The WARM start (from the finest Rust solution) is a corroborator: if it lands
    # at the same optimum, the reference is start-independent, not a fluke. It can
    # never BE the reference -- it is seeded from the solver under test.
    res_cold, cold_tag = solve_reference(sc)
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
    # PASS criteria. gap = J_rust - J_scipy(cold) >= 0 => Rust is conservative ABOVE the true
    # independent optimum (expected: the convex-hull KOZ shrinks Rust's feasible set).
    gap16 = rust[HEADLINE_NSEG]["Jtrue"] - J_scipy
    gap64 = rust[64]["Jtrue"] - J_scipy
    rust_feasible = all(rust[ns]["min_r"] >= sc["r_e"] - 1e-6 for ns in N_SEGS)
    cold_feasible = minr_cold >= sc["r_e"] - 1e-3
    # The cold (Rust-blind) solve must reach a GENUINE first-order point, tested on
    # the KKT residual rather than on trust-constr's exit code (see `converged`).
    # Without a converged reference there is nothing to compare against, so this
    # scenario reports NO REFERENCE rather than a verdict on the Rust solver: an
    # oracle that fails to converge says nothing about the code under test, and
    # scoring it as FAIL would be attributing the oracle's limits to the solver.
    cold_converged = converged(res_cold)
    # Warm start must land at the SAME optimum -> the reference is start-independent.
    cold_warm_agree = (abs(J_cold - J_warm) / J_cold < 0.02) and (abs(minr_cold - minr_warm) < 1.0)
    # Rust within 10% ABOVE the true optimum. A value BELOW it would mean Rust violates the
    # true KOZ, so the lower bound is gated too (not abs()'d away).
    close_ok = -1e-3 <= gap16 / J_scipy <= 0.10
    shrink_ok = abs(gap64) <= abs(gap16) + 1e-12     # conservatism gap shrinks with n_seg
    passed = (rust_feasible and cold_feasible and cold_converged
              and cold_warm_agree and close_ok and shrink_ok)
    verdict = "PASS" if passed else ("NO REFERENCE" if not cold_converged else "FAIL")

    md = [f"## {scenario_name}", ""]
    md.append(f"- **Independent reference ({cold_tag} start, Rust-blind)**: converged="
              f"{cold_converged} (optimality={res_cold.optimality:.2e} vs {OPT_TOL:g}, "
              f"constraint violation={res_cold.constr_violation:.2e} vs {CV_TOL:g} km; "
              f"exit status={res_cold.status}, {res_cold.nit} iterations of {MAXITER}), "
              f"J_true={J_cold:.6e}, min_r={minr_cold:.3f} km, feasible={cold_feasible}")
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
              f"cold_converged={cold_converged}, cold≈warm={cold_warm_agree}, "
              f"close(0≤gap≤10%)={close_ok}, gap_shrinks={shrink_ok} "
              f"→ **{verdict}**")
    if verdict == "NO REFERENCE":
        md.append("")
        md.append("  The independent solve did not reach a first-order point on this "
                  "geometry, so there is no reference to compare the Rust result "
                  "against. Every number above is reported; none of them is a verdict "
                  "on the solver.")
    md.append("")
    return rows, md, verdict


def run(scenarios=H.ALL_SCENARIOS):
    all_rows, sections, results = [], [], {}
    for name in scenarios:
        rows, md, verdict = _run_one(name)
        all_rows.extend(rows)
        sections.extend(md)
        results[name] = verdict

    H.write_csv(OUT / "crosscheck.csv", all_rows)
    referenced = [k for k, v in results.items() if v != "NO REFERENCE"]
    failed = [k for k, v in results.items() if v == "FAIL"]
    unreferenced = [k for k, v in results.items() if v == "NO REFERENCE"]
    # PASS requires that every geometry with a reference agrees AND that the
    # baseline geometry is one of them -- a pillar whose only referenced case had
    # dropped out would otherwise report PASS on an empty comparison.
    all_pass = not failed and "phase120" in referenced

    md = [f"# Pillar 1 -- Independent NLP cross-check ({len(scenarios)} geometries)", "",
          "SciPy trust-constr solves the same transfer as a direct NLP on the true",
          "nonconvex objective with the true dense-grid KOZ, from a COLD start that",
          "carries no Rust information. The Rust convex-hull KOZ is a subset of the",
          "true feasible set, so the gap must be non-negative and must shrink with",
          "n_seg -- a NEGATIVE gap would mean Rust cuts below the true optimum, i.e.",
          "violates the true KOZ, and is gated as such.", "",
          "A geometry where the independent solve does not itself converge yields",
          "NO REFERENCE: there is nothing to compare against, and calling that a",
          "solver failure would blame the code under test for the oracle's limits.",
          "It is not a pass either -- the coverage line below says how many",
          "geometries this pillar actually certifies.", "",
          "| scenario | verdict |", "|---|---|"]
    md += [f"| {k} | **{v}** |" for k, v in results.items()]
    md += ["",
           f"- independent reference obtained on {len(referenced)}/{len(scenarios)} "
           f"geometries: {', '.join(referenced) or 'none'}"
           + (f" (no reference: {', '.join(unreferenced)})" if unreferenced else ""),
           f"- every referenced geometry agrees: **{not failed}**"
           + (f" — failing: {', '.join(failed)}" if failed else ""),
           "", f"## VERDICT: {'PASS' if all_pass else 'FAIL'}", "", "---", ""] + sections
    H.write_text(OUT / "summary.md", "\n".join(md) + H.provenance())

    print("\n".join(md))
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
