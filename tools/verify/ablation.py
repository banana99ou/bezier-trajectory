"""
Pillar 2 -- Ablation: which change is the fix?

Cells (energy, phase120, n_seg=16):
  (a)  trust OFF + prox ON             -> reproduces the crawl to the cap (the bug)
  (a0) trust OFF + prox OFF            -> legacy loop, prox removed: still slow, but NOT capped
  (b)  trust ON  + prox OFF + freeze OFF (canonical re-linearize-every-step SCvx) -> converges
  (c)  trust ON  + prox OFF + freeze ON  (shipped config)                          -> converges
  (d)  trust ON  + prox ON  + freeze ON  -> still converges (prox is inert in the trust path)

Conclusion (isolated): (a) vs (a0) separates the two changes that trust_radius=0 flips at once.
The trust region is the PRIMARY fix -- even with the proximal removed, the legacy loop (a0)
never satisfies the SCvx convergence criterion and is >3x slower than the trust path (b/c),
which converges by it. The mis-scaled proximal is a SECONDARY aggravator:
it drives the already-slow legacy loop (a0) all the way to the cap (a). (b) vs (c) reach the
same optimum => freeze is speed-only.

Run:  .venv/bin/python tools/verify/ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np

from tools.verify import harness_common as H

N_SEG = 16
CAP = 2000  # max_iter for the legacy crawl cell (caps well before this)
OUT = H.ARTIFACT_ROOT / "pillar2_ablation"

# The scvx_freeze mechanism was deleted (no literature basis, measurably worse on
# both objective and wall time, and a permanent config-provenance hazard), so the
# former freezeON/freezeOFF cells collapse to a single trust-path configuration.
CELLS = [
    ("(a) trust0 + prox1e-6", dict(scp_trust_radius=0.0, scp_prox_weight=1e-6), CAP),
    ("(a0) trust0 + prox0", dict(scp_trust_radius=0.0, scp_prox_weight=0.0), CAP),
    ("(b) trust2000 + prox0", dict(scp_trust_radius=2000.0, scp_prox_weight=0.0), 1000),
    ("(c) trust2000 + prox1e-6", dict(scp_trust_radius=2000.0, scp_prox_weight=1e-6), 1000),
]


def run(scenario_name="phase120"):
    sc = H.make_scenario(scenario_name)
    rows = []
    for label, cfg, max_iter in CELLS:
        P, info = H.run_rust(sc, n_seg=N_SEG, max_iter=max_iter, **cfg)
        it = int(info["iterations"])
        conv = int(info.get("scvx_converged", -1))
        capped = it >= max_iter
        rows.append(dict(
            cell=label, iterations=it, capped=capped, scvx_converged=conv,
            feasible=bool(info["feasible"]), min_radius=float(info["min_radius"]),
            J_true=H.J_true(P, sc["T"]), cost_true_energy=float(info["cost_true_energy"]),
        ))

    # PASS logic.
    a, a0, b, c = rows
    a_caps = a["capped"] and a["scvx_converged"] != 1
    # Isolation: trust_radius=0 flips TWO things at once (legacy loop ON + proximal ON).
    # (a0) holds the proximal OFF to expose the legacy loop alone. The trust region is the
    # PRIMARY fix: even without the proximal, the legacy loop (a0) is far slower than the
    # trust path (b/c). The mis-scaled proximal is a SECONDARY aggravator: it pushes the
    # already-slow legacy loop (a0) all the way to the cap (a).
    # Primary-fix gate: the trust path must converge by the SCvx criterion while the
    # legacy loop (a0) cannot (it only exits via the step-norm tolerance), and must be
    # at least 3x faster. (The old >10x threshold was calibrated to the two-phase
    # acceptance's 4-5 iter runs; the canonical penalized-merit acceptance takes
    # ~25-30 iters, which changes the ratio but not the isolation conclusion.)
    # The config under test is (b) — the paper baseline. (c) isolates the proximal.
    trust_is_primary = (
        (not b["capped"])
        and b["scvx_converged"] == 1
        and a0["scvx_converged"] == 0
        and (a0["iterations"] > 3 * b["iterations"])
    )
    prox_aggravates = a["iterations"] > a0["iterations"]
    all_conv = all((not r["capped"]) and r["scvx_converged"] == 1 and r["feasible"]
                   for r in [b, c])
    # Prox is inert in the trust path: (b) prox0 and (c) prox1e-6 must be identical.
    # NOTE this gate is structurally unfailable — optimizer.rs skips the proximal
    # entirely when trust_active, so (b) and (c) execute the same code. It documents
    # the intent; it is NOT evidence about the proximal.
    prox_inert = (abs(b["J_true"] - c["J_true"]) / b["J_true"] < 1e-6
                  and abs(b["min_radius"] - c["min_radius"]) < 1e-6)
    passed = a_caps and trust_is_primary and prox_aggravates and all_conv and prox_inert

    H.write_csv(OUT / "ablation.csv", [
        {k: (f"{v:.6e}" if isinstance(v, float) and k in ("J_true", "cost_true_energy")
             else (f"{v:.3f}" if isinstance(v, float) else v)) for k, v in r.items()}
        for r in rows])

    md = [f"# Pillar 2 -- Ablation ({scenario_name}, energy, n_seg={N_SEG})", ""]
    md.append("| cell | iters | capped | converged | feasible | min_r | J_true | cost_true_energy |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['cell']} | {r['iterations']} | {r['capped']} | {r['scvx_converged']} | "
                  f"{r['feasible']} | {r['min_radius']:.3f} | {r['J_true']:.6e} | {r['cost_true_energy']:.6e} |")
    md.append("")
    md.append(f"- (a) legacy loop + mis-scaled prox reproduces the crawl (caps): **{a_caps}** "
              f"({a['iterations']} iters)")
    md.append(f"- trust region is the PRIMARY fix: legacy loop even WITHOUT the prox "
              f"(a0={a0['iterations']} iters, never satisfies the SCvx convergence criterion) is >3x "
              f"slower than the trust path, which does converge "
              f"(b={b['iterations']}, c={c['iterations']} iters): **{trust_is_primary}**")
    md.append(f"- the mis-scaled proximal is a SECONDARY aggravator: it pushes the legacy loop "
              f"from {a0['iterations']} iters (a0) to the cap (a): **{prox_aggravates}**")
    md.append(f"- (b)(c) both converge + feasible: **{all_conv}**")
    md.append(f"- proximal is INERT in the trust path: (b) prox0 == (c) prox1e-6 identical: "
              f"**{prox_inert}** (structurally unfailable — the proximal is skipped when "
              f"`trust_active`; documents intent, not evidence)")
    md.append("")
    md.append("**Isolation finding (corrected)**: Cell (a) alone conflates two changes -- setting "
              "`scp_trust_radius=0` both reverts to the legacy unconditional-accept loop AND re-enables "
              f"the proximal. The added cell (a0) separates them: with the proximal removed the legacy "
              f"loop still takes {a0['iterations']} iters (it exits via the step-norm tolerance, not the "
              f"SCvx criterion) -- >3x the trust path's {b['iterations']}. So the **trust region is the "
              "primary fix**; the mis-scaled proximal is a **secondary aggravator** that drives the "
              f"already-slow legacy loop from {a0['iterations']} iters to the cap. In the trust path the "
              "proximal is inert ((b)==(c)). The `scvx_freeze` cells were removed with the mechanism "
              "itself.")
    md.append("")
    md.append(f"## VERDICT: {'PASS' if passed else 'FAIL'}")
    H.write_text(OUT / "summary.md", "\n".join(md))
    print("\n".join(md))
    return passed


if __name__ == "__main__":
    run("phase120")
