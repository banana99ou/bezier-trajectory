"""
Pillar 4a -- Per-iteration diagnostics.

A correct SCvx run shows: rho ~ 1 on optimality steps, trust grows-then-settles
(never collapsing to the floor), merit monotone within each phase, slack -> 0.

Run:  .venv/bin/python tools/verify/diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.verify import harness_common as H

OUT = H.ARTIFACT_ROOT / "pillar4_diag"
TRUST_MIN = 1e-2


def run(scenario_name="phase120", n_seg=16):
    sc = H.make_scenario(scenario_name)
    # Paper-baseline config: canonical SCvx. (scvx_freeze has been deleted.)
    P, info = H.run_rust(sc, n_seg=n_seg, scp_trust_radius=2000.0)

    rho = np.array(info.get("rho_history", []), float)
    trust = np.array(info.get("trust_history", []), float)
    merit = np.array(info.get("merit_history", []), float)
    step = np.array(info.get("step_norm_history", []), float)
    slack = np.array(info.get("slack_history", []), float)
    phase = np.array(info.get("phase_history", []), float)  # 0=bootstrap/uncertified, 1=certified
    n = len(trust)

    rows = [dict(step=i + 1, phase=int(phase[i]),
                 rho=(f"{rho[i]:.4f}" if np.isfinite(rho[i]) else "nan"),
                 trust=f"{trust[i]:.3f}", merit=f"{merit[i]:.6e}",
                 step_norm=f"{step[i]:.4e}", slack=f"{slack[i]:.4e}") for i in range(n)]
    H.write_csv(OUT / "iter_trace.csv", rows)

    # Correct-run signature checks.
    opt = phase == 1.0
    # rho ~ 1 on optimality steps means the convex model is faithful. rho = +inf is the
    # convergence signal (QP found the optimum: pred ~ 0, step accepted). Both are healthy;
    # only a finite rho far from 1 would indicate a bad model.
    rho_opt = rho[opt]
    rho_finite = rho_opt[np.isfinite(rho_opt)]
    # Require actual evidence: a solver routing every step through the rho=+inf
    # null-step branch used to pass this gate on ZERO samples.
    # NOT SUFFICIENT ALONE, measured: commit 12b5b06 produced rho = 1.000 on all
    # 139 accepted steps of a run that ratcheted 0.178% away from its own best
    # point. rho only compares within one iteration; pair it with best_ok below.
    rho_ok = bool(rho_finite.size >= 3 and np.all((rho_finite > 0.5) & (rho_finite < 2.0)))

    # merit monotone within each phase (allow tiny epsilon).
    # NOTE: near-tautological — acceptance requires rho > eta with pred > 0, hence
    # act > 0. Kept as an internal-consistency check, NOT as evidence of convergence.
    def monotone(mask):
        m = merit[mask]
        return bool(np.all(np.diff(m) <= 1e-9 * (np.abs(m[:-1]) + 1))) if m.size > 1 else True
    merit_ok = monotone(phase == 0.0) and monotone(phase == 1.0)

    # THE GATE WITH TEETH: the returned iterate must be the best one visited.
    # Neither rho_ok nor merit_ok can catch a run that drifts away from its own
    # optimum, because both only ever compare values WITHIN one iteration — and
    # the merit function itself changes between iterations, since the KOZ rows
    # re-aim. A run can therefore descend, wander, and return a point worse than
    # one it already stood on, with every per-iteration signal healthy.
    # Measured: the 2026-08-09 same-rows variant (commit 12b5b06) did exactly
    # that on phase120 — minimum at accepted step 26, returned step 139, +0.178%
    # worse — with rho = 1.000 on all 139 steps and this file reporting PASS.
    # The bootstrap step is excluded: its merit is pre-BC-repair and ~7 orders
    # larger, so including it would make the gate unfailable.
    post = merit[1:] if n > 1 else merit
    best_ok = bool(post.size > 0 and merit[-1] <= post.min() * (1.0 + 1e-9))
    drift = float((merit[-1] - post.min()) / abs(post.min())) if post.size else 0.0

    slack_ok = bool(slack[-1] < 1e-6) if n else False

    # The loop must have stopped on the merit criterion (K consecutive sub-tol steps
    # + certificate) rather than on the iteration cap or trust-region collapse.
    # The old `trust[-1] > TRUST_MIN` gate could not fail: trust_history is appended
    # only on ACCEPTED steps and the loop breaks as soon as trust < trust_min, so
    # every recorded value was >= the floor by construction.
    # Must be on scvx_stop_reason, NOT scvx_converged: that flag is also set by the
    # trust-collapse exit whenever the iterate happens to be feasible, so it reads
    # True at a deadlocked point. 1 = K-consecutive merit streak, 4 = model
    # stationarity (both principled); 0 = iteration cap, 2 = trust collapse,
    # 3 = QP failure (all "the loop gave up").
    stop_reason = int(info.get("scvx_stop_reason", -1))
    converged_ok = stop_reason in (1, 4)
    feasible_ok = bool(info["feasible"]) and float(info.get("final_cp_violation_km", 1.0)) <= 1e-6
    # Clarabel must not have fallen back to its reduced tolerances on any solve that
    # fed the ratio test.
    qp_clean = int(info.get("qp_almost_solved", -1)) == 0
    # No segment may be silently missing from the Prop-1 certificate.
    no_degenerate = int(info.get("koz_degenerate_segments", -1)) == 0

    passed = (rho_ok and merit_ok and best_ok and slack_ok and converged_ok
              and feasible_ok and qp_clean and no_degenerate)

    # Plot.
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(5, 1, figsize=(7, 10), sharex=True)
    ax[0].plot(x, rho, "o-"); ax[0].axhline(1.0, color="grey", ls=":"); ax[0].set_ylabel("rho")
    ax[1].plot(x, trust, "o-"); ax[1].set_ylabel("trust (km)")
    ax[2].semilogy(x, np.clip(merit, 1e-30, None), "o-"); ax[2].set_ylabel("merit")
    ax[3].semilogy(x, np.clip(step, 1e-30, None), "o-"); ax[3].set_ylabel("step norm")
    ax[4].semilogy(x, np.clip(slack, 1e-30, None), "o-"); ax[4].set_ylabel("slack"); ax[4].set_xlabel("accepted step")
    for a in ax:
        for i in range(n):
            if phase[i] == 0.0:
                a.axvspan(i + 0.5, i + 1.5, color="orange", alpha=0.08)
    fig.suptitle(f"SCvx per-iteration trace ({scenario_name}, n_seg={n_seg}, energy)\n"
                 "orange = bootstrap / uncertified-iterate step")
    fig.tight_layout()
    fig.savefig(OUT / "iter_trace.png", dpi=110)
    plt.close(fig)

    md = [f"# Pillar 4a -- Diagnostics ({scenario_name}, n_seg={n_seg}, energy)", ""]
    md.append(f"- accepted steps: {n} (iterations={int(info['iterations'])})")
    md.append(f"- rho on optimality steps in (0.5,2): **{rho_ok}**  (values: "
              f"{', '.join(f'{v:.3f}' for v in rho[opt]) if opt.any() else '—'})")
    md.append(f"- merit monotone within each phase: **{merit_ok}** "
              f"(near-tautological; consistency check only)")
    md.append(f"- returned iterate is the BEST visited (drift {drift:+.3e}): **{best_ok}** "
              f"— the gate rho and monotonicity cannot provide; catches a run that "
              f"descends then wanders off its own optimum")
    md.append(f"- slack -> 0 (final={slack[-1]:.2e}): **{slack_ok}**")
    _STOP = {0: "iteration cap", 1: "K-consecutive merit streak", 2: "trust-region collapse",
             3: "QP failure", 4: "model stationarity"}
    md.append(f"- stopped for a stated reason (not the cap, not trust collapse): "
              f"**{converged_ok}** (stop_reason={stop_reason} "
              f"[{_STOP.get(stop_reason, 'unknown')}], iterations={int(info['iterations'])}, "
              f"final trust={float(info.get('final_trust_radius', float('nan'))):.4g} km, "
              f"floor={TRUST_MIN})")
    md.append(f"- feasible + Prop-1 certificate at the final iterate "
              f"(final_cp_violation_km={float(info.get('final_cp_violation_km', float('nan'))):.2e}): "
              f"**{feasible_ok}**")
    md.append(f"- every QP hit the requested tolerances "
              f"(qp_almost_solved={int(info.get('qp_almost_solved', -1))}): **{qp_clean}**")
    md.append(f"- no degenerate KOZ normals skipped "
              f"(koz_degenerate_segments={int(info.get('koz_degenerate_segments', -1))}): "
              f"**{no_degenerate}**")
    md.append("")
    md.append("See `iter_trace.png` (orange band = feasibility-restoration).")
    md.append("")
    md.append(f"## VERDICT: {'PASS' if passed else 'FAIL'}")
    H.write_text(OUT / "summary.md", "\n".join(md))
    print("\n".join(md))
    return passed


if __name__ == "__main__":
    run("phase120", 16)
