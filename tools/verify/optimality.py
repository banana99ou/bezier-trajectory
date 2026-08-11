"""
Pillar 5 -- Optimality of the returned point (external KKT check).

Every other pillar checks that the solver STOPS for a stated reason and that the
answer is FEASIBLE. None of them checks that the answer is OPTIMAL. This one does,
and it does so without asking the solver anything: it takes the returned control
net, rebuilds the true objective gradient and the active constraint set in NumPy,
and asks whether the first-order optimality conditions hold.

Two independent checks:

  (1) KKT stationarity.  At a constrained local minimum there must exist
      multipliers with
            grad J(x*)  =  A_act^T lambda,     lambda >= 0 on inequalities
      i.e. the objective gradient is a nonnegative combination of the active
      constraint normals (equalities unrestricted in sign). We solve that
      nonnegative least-squares problem and report the RELATIVE residual
            ||grad J - A_act^T lambda|| / ||grad J||.
      A point that is merely "where the solver stopped" leaves a large residual;
      a genuine constrained optimum leaves ~0.

  (2) Descent search.  A direct, multiplier-free check: sample many feasible
      directions and confirm none of them improves J while keeping the Prop-1
      certificate. This cannot prove optimality, but it can DISPROVE it, which
      is what a gate is for.

It also measures the quantity design_freeze section 5 records as "estimated":
the KOZ dual scale ||lambda_KOZ||_inf, which the exact-penalty rule requires
w_s to exceed.

Run:  .venv/bin/python tools/verify/optimality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
from scipy.optimize import nnls

from tools.verify import harness_common as H
from orbital_docking.de_casteljau import segment_matrices_equal_params

OUT = H.ARTIFACT_ROOT / "pillar5_optimality"

# Relative stationarity residual accepted as "first-order optimal".
#
# 1e-3, tightened from 5e-2 on 2026-08-11. Measured residuals with the corrected
# rows (see koz_rows) are 3.06e-06 .. 1.131e-05 across all five scenarios, so
# this leaves 88x margin on the worst. The old 5e-2 left 4419x, which is not a
# gate: it passed the pre-2026-08-11 rows, whose residual reached 1.42e-2 at
# n_seg=16 and 3.72e-01 at n_seg=4. 1e-3 would have caught both.
KKT_TOL = 1e-3

# km: a KOZ row within this of its bound counts as active.
#
# Swept 2026-08-11 over 1e-5 .. 1e0 on all five scenarios: the residual is
# IDENTICAL across 1e-5 .. 1e-1 and only softens at 1e0 (which admits inactive
# rows and can only help the fit). So the verdict does not depend on where this
# threshold is drawn -- the open question recorded in session_handoff is closed.
ACTIVE_TOL = 1e-3

N_DIRS = 400        # random feasible directions for the descent search


# ---------------------------------------------------------------------------
# True objective and its gradient (finite differences on H.J_true -- an
# INDEPENDENT implementation from the Rust Gram-matrix closed form)
# ---------------------------------------------------------------------------

def grad_J(P, T, step=1e-4):
    """Central-difference gradient of the true objective wrt the control net."""
    P = np.asarray(P, float)
    g = np.zeros_like(P)
    for i in range(P.shape[0]):
        for d in range(P.shape[1]):
            hp = np.zeros_like(P); hp[i, d] = step
            g[i, d] = (H.J_true(P + hp, T) - H.J_true(P - hp, T)) / (2 * step)
    return g


def koz_rows(P, A_list, r_e, c_koz=None):
    """Active-set data for the KOZ half-spaces rebuilt at P.

    Returns (rows, slacks) with rows[i] the gradient of the i-th clearance wrt
    the flattened control net and slacks[i] its value (0 = active).

    The clearance whose gradient this is:

        gamma^(s)_m(x) = n^(s)(x) . (q^(s)_m(x) - c_KOZ) - r_e

    The witness n^(s) is a FUNCTION OF x -- the centroid rule rebuilds it at
    whatever point it is handed -- so the gradient carries two terms:

        d gamma^(s)_m / d p_i
            = S^(s)_mi n^(s)                                    <- base
            + (w^(s)_i / ||c - c_KOZ||) (I - n n^T)(q_m - c_KOZ) <- witness rotation

    with w^(s)_i = (1/(N+1)) sum_m S^(s)_mi the centroid weights. The projection
    (I - n n^T) means the second term grows with how far the control point sits
    from the tangent point ALONG the plane.

    Until 2026-08-11 this function dropped the rotation term, on the stated
    ground that it is "second order at an active point". That is false wherever
    a sub-arc has appreciable lateral reach, i.e. at every mesh the paper
    reports. Measured on phase120, the stationarity residual with the term
    against without:

        n_seg   with       without     inflation
            4   2.835e-06  3.723e-01     131000x   (reported a FALSE failure)
            8   1.044e-05  3.127e-02       2995x
           16   1.131e-05  1.420e-02       1255x
           32   1.166e-05  6.933e-03        595x
           64   1.168e-05  3.442e-03        295x

    With the term the residual is FLAT across the mesh -- the signature of a
    genuine KKT point -- and the old numbers were dominated by the omission,
    not by the solver. The same gradient is verified against central differences
    to 1.1e-10 in the Rust row builder's test; dropping the rotation term makes
    that check plateau at 4.3e-2 regardless of step size.
    """
    P = np.asarray(P, float)
    np1, dim = P.shape
    c_koz = np.zeros(dim) if c_koz is None else np.asarray(c_koz, float)
    rows, slacks = [], []
    for A in A_list:
        Q = A @ P
        c = Q.mean(axis=0)
        v = c - c_koz
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        n = v / nv
        w_row = A.sum(axis=0) / np1          # centroid weights w^(s)_i
        proj = np.eye(dim) - np.outer(n, n)  # tangential projector
        for k in range(np1):
            tang = proj @ (Q[k] - c_koz)
            row = np.zeros((np1, dim))
            for m in range(np1):
                row[m] += A[k, m] * n + (w_row[m] / nv) * tang
            rows.append(row.ravel())
            slacks.append(float(n @ (Q[k] - c_koz) - r_e))
    return np.array(rows), np.array(slacks)


def bc_rows(P, T, has_v0=True, has_v1=True):
    """Gradients of the (linear) boundary-condition equality rows."""
    np1, dim = P.shape
    N = np1 - 1
    out = []
    for d in range(dim):                       # P0 pinned
        r = np.zeros((np1, dim)); r[0, d] = 1.0; out.append(r.ravel())
    for d in range(dim):                       # PN pinned
        r = np.zeros((np1, dim)); r[-1, d] = 1.0; out.append(r.ravel())
    if has_v0:
        for d in range(dim):
            r = np.zeros((np1, dim)); r[0, d] = -N / T; r[1, d] = N / T; out.append(r.ravel())
    if has_v1:
        for d in range(dim):
            r = np.zeros((np1, dim)); r[-2, d] = -N / T; r[-1, d] = N / T; out.append(r.ravel())
    return np.array(out)


def kkt_residual(P, T, A_list, r_e):
    """Solve  min || grad J - [A_eq^T | A_ineq^T] [mu; lambda] ||,  lambda >= 0.

    Equalities are sign-free, so they enter as a +/- pair for the NNLS solve.
    Returns (relative residual, lambda_koz, n_active).
    """
    g = grad_J(P, T).ravel()
    Aq, slk = koz_rows(P, A_list, r_e)
    act = slk <= ACTIVE_TOL
    Ai = Aq[act] if Aq.size else np.zeros((0, g.size))
    Ae = bc_rows(np.asarray(P, float), T)

    # columns: [ +Ae^T , -Ae^T , Ai^T ]  with all coefficients >= 0
    cols = [Ae.T, -Ae.T] + ([Ai.T] if Ai.shape[0] else [])
    M = np.hstack(cols)
    coef, _ = nnls(M, g)
    resid = float(np.linalg.norm(M @ coef - g))
    rel = resid / max(np.linalg.norm(g), 1e-30)
    n_eq = Ae.shape[0]
    lam = coef[2 * n_eq:] if Ai.shape[0] else np.zeros(0)
    return rel, lam, int(act.sum())


def descent_search(P, T, A_list, r_e, n_dirs=N_DIRS, steps_km=(0.01, 0.1, 1.0, 10.0),
                   seed=0, rel_tol=1e-9):
    """Try to DISPROVE local optimality: any feasible direction that lowers J.

    Endpoints and the velocity-BC control points are held fixed so every trial
    point still satisfies the hard rows exactly; feasibility is then only the
    Prop-1 certificate, which is re-checked exactly.

    `rel_tol` must sit above the oracle's own resolution. That is a live hazard:
    with the pre-2026-08-09 Riemann-sum J_true (O(1/n), ~1.6e-3 error at its
    default) this search reported 562 "improving" directions on phase70 whose
    best step REVERSED to +5.0e-5 once the quadrature converged. J_true is now
    Gauss-Legendre and exact to f64, so 1e-9 is safely above its noise.
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(P, float)
    J0 = H.J_true(P, T)
    np1, dim = P.shape
    free = np.arange(2, np1 - 2)               # interior points not pinned by BCs
    if free.size == 0:
        return 0, J0, 0.0, 0
    best, n_better, n_tested = 0.0, 0, 0
    for _ in range(n_dirs):
        D = np.zeros_like(P)
        D[free] = rng.normal(size=(free.size, dim))
        D /= max(np.linalg.norm(D), 1e-30)
        for s in steps_km:
            X = P + s * D
            _, slk = koz_rows(X, A_list, r_e)
            if slk.size and slk.min() < -1e-6:
                continue                        # left the certified set
            n_tested += 1
            J = H.J_true(X, T)
            if J < J0 * (1 - rel_tol):
                n_better += 1
                best = max(best, (J0 - J) / J0)
    return n_better, J0, best, n_tested


def run(scenarios=("phase70", "phase120", "phase135", "phase170", "planechange"), n_seg=16):
    rows, all_ok = [], True
    for name in scenarios:
        sc = H.make_scenario(name)
        P, info = H.run_rust(sc, n_seg=n_seg)
        A_list = segment_matrices_equal_params(sc["N"], n_seg)
        rel, lam, n_act = kkt_residual(P, sc["T"], A_list, sc["r_e"])
        n_better, J0, best, n_tested = descent_search(P, sc["T"], A_list, sc["r_e"])
        lam_inf = float(np.max(np.abs(lam))) if lam.size else 0.0
        ok = (rel < KKT_TOL) and (n_better == 0)
        all_ok &= ok
        rows.append(dict(
            scenario=name, stop=int(info.get("scvx_stop_reason", -1)),
            iters=int(info["iterations"]), J_true=J0,
            kkt_rel_resid=rel, n_active=n_act, lambda_koz_inf=lam_inf,
            descent_hits=n_better, descent_tested=n_tested,
            best_improvement=best, passed=ok,
        ))

    H.write_csv(OUT / "optimality.csv", [
        {k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows])

    w_s = 1e-2
    lam_max = max((r["lambda_koz_inf"] for r in rows), default=0.0)
    penalty_ok = w_s >= lam_max

    md = [f"# Pillar 5 -- Optimality (external KKT, n_seg={n_seg})", "",
          "Independent of the solver: gradient by finite differences on the dense",
          "true objective, active set and constraint normals rebuilt in NumPy.", "",
          "| scenario | stop | iters | KKT rel resid | active rows | max KOZ dual | "
          "descent hits/tested | verdict |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scenario']} | {r['stop']} | {r['iters']} | {r['kkt_rel_resid']:.3e} | "
                  f"{r['n_active']} | {r['lambda_koz_inf']:.3e} | "
                  f"{r['descent_hits']}/{r['descent_tested']} | "
                  f"{'PASS' if r['passed'] else 'FAIL'} |")
    md += ["",
           f"- KKT stationarity residual below {KKT_TOL:g} on every scenario: "
           f"**{all(r['kkt_rel_resid'] < KKT_TOL for r in rows)}**",
           f"- no feasible descent direction found ({N_DIRS} directions x 4 step sizes "
           f"per scenario, {sum(r['descent_tested'] for r in rows)} certified trials "
           f"total): **{all(r['descent_hits'] == 0 for r in rows)}**",
           f"- exact-penalty rule w_s >= ||lambda_KOZ||_inf "
           f"({w_s:g} >= {lam_max:.3e}): **{penalty_ok}** "
           f"(design_freeze section 5 recorded this as 'estimated ~1e-6'; it is now MEASURED)",
           "",
           f"## VERDICT: {'PASS' if (all_ok and penalty_ok) else 'FAIL'}"]
    H.write_text(OUT / "summary.md", "\n".join(md) + H.provenance())
    print("\n".join(md))
    return all_ok and penalty_ok


if __name__ == "__main__":
    run()
