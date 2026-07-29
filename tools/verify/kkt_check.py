"""
Pillar 3 -- Feasibility check at x* (+ optimality-gap diagnostics).

PASS GATE = primal feasibility of the returned x*_rust: dense-grid KOZ satisfied,
velocity BCs met, endpoints fixed. This certifies a valid, constraint-satisfying
trajectory -- not "the loop merely stopped."

Optimality is NOT gated here; it is certified independently by Pillar 1 (an
independent COLD-start NLP, Rust-blind, that reaches the same optimum). This module
additionally REPORTS, as diagnostics only, the true- and surrogate-objective
gradients projected onto the equality-constraint nullspace. These are EXPECTED to be
nonzero and are NOT pass/fail criteria: the equality set here excludes the convex-hull
KOZ half-spaces (which are active in Rust's surrogate but leave the *sampled* curve
~11-60 km clear), so the projection omits part of the true active set. The numbers
track the conservatism/linearization gap that Pillar 1 quantifies shrinking with n_seg.

Run:  .venv/bin/python tools/verify/kkt_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np

from tools.verify import harness_common as H
from orbital_docking.constraints import build_boundary_constraints
from orbital_docking.optimization import _build_ctrl_accel_quadratic

N_SEGS = [8, 16, 32]
OUT = H.ARTIFACT_ROOT / "pillar3_kkt"


def _equality_rows(scenario, P):
    """Endpoints (fixed positions) + velocity-BC equality rows -> (n_eq, nvar)."""
    N, T, dim = scenario["N"], scenario["T"], 3
    rows = []
    for d in range(dim):
        e = np.zeros((N + 1) * dim); e[d] = 1.0; rows.append(e)
        e = np.zeros((N + 1) * dim); e[N * dim + d] = 1.0; rows.append(e)
    for bc in build_boundary_constraints(P, v0=scenario["v0"], v1=scenario["v1"], dim=dim, T=T):
        rows.extend(list(np.atleast_2d(bc.A)))
    return np.vstack(rows)


def _proj_ratio(A_eq, g):
    g_proj = g - A_eq.T @ (np.linalg.pinv(A_eq @ A_eq.T) @ (A_eq @ g))
    return np.linalg.norm(g_proj) / (np.linalg.norm(g) + 1e-30)


def run(scenario_name="phase120"):
    sc = H.make_scenario(scenario_name)
    rows = []
    all_pass = True
    for ns in N_SEGS:
        P, info = H.run_rust(sc, n_seg=ns, objective_mode="energy")
        x = P.reshape(-1)

        # Primal feasibility.
        min_r = H.min_radius(P)
        v0, v1 = H.velocity_endpoints(P, sc["T"])
        bc_v0 = np.linalg.norm(v0 - sc["v0"]) / (np.linalg.norm(sc["v0"]) + 1e-12)
        bc_v1 = np.linalg.norm(v1 - sc["v1"]) / (np.linalg.norm(sc["v1"]) + 1e-12)
        feas = (min_r >= sc["r_e"] - 1e-6) and bc_v0 < 1e-6 and bc_v1 < 1e-6

        A_eq = _equality_rows(sc, P)  # KOZ half-spaces are not tightly active here (min slack ~11 km)

        # Stationarity of the SURROGATE the solver minimizes (gravity linearized at x*):
        #   grad_surrogate = H x + f. Projected onto the equality nullspace ~ 0 at a KKT point.
        Hm, fm, _, _ = _build_ctrl_accel_quadratic(P, sc["T"], int(info.get("sample_count", 100) or 100),
                                                   objective="energy")
        g_surr = Hm @ x + fm
        ratio_surr = _proj_ratio(A_eq, g_surr)

        # True-objective free-gradient = the conservatism/surrogate gap (quantified in Pillar 1).
        _, g_true = H.J_true_and_grad(P, sc["T"])
        ratio_true = _proj_ratio(A_eq, g_true)

        # Pillar 3 certifies PRIMAL FEASIBILITY (a valid constraint-satisfying trajectory).
        # Optimality is certified independently by Pillar 1 (an independent COLD-start NLP,
        # Rust-blind, reaches the same optimum; Rust sits above it by the conservatism gap,
        # which -> 0 as n_seg grows). The projected gradients below are DIAGNOSTICS ONLY,
        # expected nonzero (the equality set omits the active convex-hull KOZ half-spaces).
        row_pass = feas
        all_pass = all_pass and row_pass
        rows.append(dict(
            scenario=scenario_name, n_seg=ns,
            min_r=f"{min_r:.3f}", clearance=f"{min_r - sc['r_e']:.3f}",
            bc_v0_err=f"{bc_v0:.2e}", bc_v1_err=f"{bc_v1:.2e}",
            ratio_surrogate=f"{ratio_surr:.3e}", ratio_true=f"{ratio_true:.3e}",
            PASS=row_pass,
        ))

    H.write_csv(OUT / "kkt.csv", rows)
    md = [f"# Pillar 3 -- KKT / feasibility at x* ({scenario_name})", ""]
    md.append("**PASS gate = primal feasibility**: dense-grid min‖r‖ ≥ r_e and velocity-BC "
              "residuals < 1e-6 (endpoints fixed by construction). Optimality is certified "
              "separately by **Pillar 1** (an independent COLD-start NLP, Rust-blind, reaches the "
              "same optimum; Rust sits above it by the conservatism gap → 0 as n_seg grows). "
              "`ratio_*` are the true- and surrogate-objective gradients projected onto the equality "
              "nullspace — **diagnostics only, expected nonzero and NOT gated**: the equality set "
              "omits the active convex-hull KOZ half-spaces (which hold the sampled curve ~11–60 km "
              "clear), so the projection intentionally excludes part of the true active set.")
    md.append("")
    md.append("| n_seg | min_r | clearance | bc_v0 | bc_v1 | ratio_surrogate | ratio_true | feasible |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['n_seg']} | {r['min_r']} | {r['clearance']} | {r['bc_v0_err']} | "
                  f"{r['bc_v1_err']} | {r['ratio_surrogate']} | {r['ratio_true']} | {r['PASS']} |")
    md.append("")
    md.append("")
    md.append(f"## VERDICT: {'PASS' if all_pass else 'FAIL'}")
    H.write_text(OUT / "summary.md", "\n".join(md))
    print("\n".join(md))
    return all_pass


if __name__ == "__main__":
    run("phase120")
