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
    # Paper-baseline config: canonical SCvx, freeze off (the freeze is a legacy
    # non-canonical mechanism kept only as an ablation cell in pillar 2).
    P, info = H.run_rust(sc, n_seg=n_seg, scp_trust_radius=2000.0,
                         disable_scvx_freeze=True)

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
    rho_ok = bool(np.all((rho_finite > 0.5) & (rho_finite < 2.0))) if rho_finite.size else True
    trust_ok = bool(trust[-1] > TRUST_MIN)
    # merit monotone within each phase (allow tiny epsilon).
    def monotone(mask):
        m = merit[mask]
        return bool(np.all(np.diff(m) <= 1e-9 * (np.abs(m[:-1]) + 1))) if m.size > 1 else True
    merit_ok = monotone(phase == 0.0) and monotone(phase == 1.0)
    slack_ok = bool(slack[-1] < 1e-6) if n else False
    iters_ok = int(info["iterations"]) < 50
    passed = rho_ok and trust_ok and merit_ok and slack_ok and iters_ok

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
    md.append(f"- trust settles above floor (final={trust[-1]:.1f} > {TRUST_MIN}): **{trust_ok}**")
    md.append(f"- merit monotone within each phase: **{merit_ok}**")
    md.append(f"- slack -> 0 (final={slack[-1]:.2e}): **{slack_ok}**")
    md.append(f"- iterations < 50: **{iters_ok}**")
    md.append("")
    md.append("See `iter_trace.png` (orange band = feasibility-restoration).")
    md.append("")
    md.append(f"## VERDICT: {'PASS' if passed else 'FAIL'}")
    H.write_text(OUT / "summary.md", "\n".join(md))
    print("\n".join(md))
    return passed


if __name__ == "__main__":
    run("phase120", 16)
