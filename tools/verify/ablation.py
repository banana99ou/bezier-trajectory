"""
Pillar 2 -- Ablation: which change is the fix?

Cells (energy, n_seg=16), run on every geometry in H.ALL_SCENARIOS:
  (a)  trust OFF + prox ON             -> reproduces the crawl to the cap (the bug)
  (a0) trust OFF + prox OFF            -> legacy loop, prox removed: still slow, but NOT capped
  (b)  trust ON  + prox OFF + freeze OFF (canonical re-linearize-every-step SCvx) -> converges
  (c)  trust ON  + prox OFF + freeze ON  (shipped config)                          -> converges
  (d)  trust ON  + prox ON  + freeze ON  -> still converges (prox is inert in the trust path)

The trust cells use the SCENARIO's own r0, not a literal 2000 km: the initial
trust radius must exceed the iteration-1 boundary-condition repair distance,
which is a property of the geometry (design_freeze section 5). phase170's
straight-line guess starts 5420 km inside the KOZ and a 2000 km box cannot
repair it, so a hardcoded 2000 would have made that cell fail for a reason
having nothing to do with the ablation.

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
# max_iter for the legacy cells. It must exceed what (a0) needs to TERMINATE,
# or the aggravation claim is measured between two censored values. At CAP=2000
# that is exactly what happened: on phase135/phase170/planechange both (a) and
# (a0) read 2000 and `a > a0` was false, reporting FAIL for want of headroom.
# Measured with the cap lifted: (a0) converges at 2385 (phase135) and 2339
# (planechange) while (a) still runs past 12000. (a) capping is fine -- it is
# the "worse" direction, so it lower-bounds the aggravation.
CAP = 12000
OUT = H.ARTIFACT_ROOT / "pillar2_ablation"

# The scvx_freeze mechanism was deleted (no literature basis, measurably worse on
# both objective and wall time, and a permanent config-provenance hazard), so the
# former freezeON/freezeOFF cells collapse to a single trust-path configuration.
def _cells(r0):
    return [
        ("(a) trust0 + prox1e-6", dict(scp_trust_radius=0.0, scp_prox_weight=1e-6), CAP),
        ("(a0) trust0 + prox0", dict(scp_trust_radius=0.0, scp_prox_weight=0.0), CAP),
        (f"(b) trust{r0:.0f} + prox0", dict(scp_trust_radius=r0, scp_prox_weight=0.0), 1000),
        (f"(c) trust{r0:.0f} + prox1e-6", dict(scp_trust_radius=r0, scp_prox_weight=1e-6), 1000),
    ]


def _run_one(scenario_name):
    """The four ablation cells on one geometry. Returns (rows, md_lines, passed)."""
    sc = H.make_scenario(scenario_name)
    rows = []
    for label, cfg, max_iter in _cells(sc["r0"]):
        P, info = H.run_rust(sc, n_seg=N_SEG, max_iter=max_iter, **cfg)
        it = int(info["iterations"])
        conv = int(info.get("scvx_converged", -1))
        capped = it >= max_iter
        rows.append(dict(
            scenario=scenario_name, cell=label, iterations=it, capped=capped,
            scvx_converged=conv,
            stop_reason=int(info.get("scvx_stop_reason", -1)),
            feasible=bool(info["feasible"]), min_radius=float(info["min_radius"]),
            J_true=H.J_true(P, sc["T"]), cost_true_energy=float(info["cost_true_energy"]),
        ))

    # PASS logic.
    a, a0, b, c = rows
    # Gate on scvx_stop_reason, not scvx_converged: that flag is ALSO set by the
    # trust-collapse exit whenever the iterate happens to be feasible, so it reads
    # True at a deadlocked point. Principled stops are 1 (K-consecutive merit
    # streak) and 4 (model stationarity); 0/2/3 all mean "the loop gave up".
    # The legacy cells (a)/(a0) never enter the SCvx path, so they have no
    # stop_reason and are judged on the cap alone.
    principled = lambda r: r["stop_reason"] in (1, 4)
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
        and principled(b)
        and a0["scvx_converged"] == 0
        and (a0["iterations"] > 3 * b["iterations"])
    )
    # The comparison only means something when (a0) actually TERMINATES: two cells
    # both sitting at the cap compare nothing, and reading that as "no aggravation"
    # is a measurement artifact, not a result. A censored (a0) therefore yields
    # NOT MEASURED for this geometry rather than a verdict in either direction --
    # the same rule Pillar 1 applies to a reference that will not converge.
    # Measured: (a0) terminates at 69 / 1197 / 2385 / 2339 iterations on phase70,
    # phase120, phase135 and planechange, and does not terminate within CAP on
    # phase170.
    prox_measurable = not a0["capped"]
    prox_aggravates = prox_measurable and a["iterations"] > a0["iterations"]
    all_conv = all((not r["capped"]) and principled(r) and r["feasible"]
                   for r in [b, c])
    # (b) prox0 vs (c) prox1e-6 are BIT-IDENTICAL by construction: optimizer.rs
    # skips the proximal entirely when trust_active, so the two cells execute the
    # same code on the same inputs. A gate that cannot fail is not evidence, so
    # this is reported as an identity CHECK and excluded from `passed`. It would
    # only become informative if the proximal were ever reachable on the trust
    # path -- at which point it should move back into the pass logic.
    prox_identical = (abs(b["J_true"] - c["J_true"]) / b["J_true"] < 1e-12
                      and abs(b["min_radius"] - c["min_radius"]) < 1e-12)
    passed = a_caps and trust_is_primary and prox_aggravates and all_conv
    if passed:
        verdict = "PASS"
    elif not prox_measurable and a_caps and trust_is_primary and all_conv:
        verdict = "NOT MEASURED"
    else:
        verdict = "FAIL"

    md = [f"## {scenario_name} (trust cells at r0 = {sc['r0']:.0f} km)", ""]
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
              f"from {a0['iterations']} iters (a0) to {a['iterations']}"
              f"{' (the cap)' if a['capped'] else ''} (a): **{prox_aggravates}**"
              + ("" if not a0["capped"] else
                 f" — NOT MEASURED: (a0) itself hit the {CAP} cap, so the two cells are "
                 f"both censored and the comparison is empty"))
    md.append(f"- (b)(c) both converge + feasible: **{all_conv}**")
    md.append(f"- (b) prox0 and (c) prox1e-6 are bit-identical: **{prox_identical}** "
              f"(NOT A GATE — the proximal is skipped when `trust_active`, so these two "
              f"cells run the same code; excluded from the verdict because it cannot fail)")
    md.append("")
    md.append("Isolation finding: Cell (a) alone conflates two changes -- setting "
              "`scp_trust_radius=0` both reverts to the legacy unconditional-accept loop AND re-enables "
              f"the proximal. The added cell (a0) separates them: with the proximal removed the legacy "
              f"loop still takes {a0['iterations']} iters (it exits via the step-norm tolerance, not the "
              f"SCvx criterion) -- >3x the trust path's {b['iterations']}. So the **trust region is the "
              "primary fix**; the mis-scaled proximal is a **secondary aggravator** that drives the "
              f"already-slow legacy loop from {a0['iterations']} iters to the cap. In the trust path the "
              "proximal is inert ((b)==(c)). The `scvx_freeze` cells were removed with the mechanism "
              "itself.")
    md.append("")
    return rows, md, verdict, dict(a=a["iterations"], a0=a0["iterations"],
                                   b=b["iterations"], c=c["iterations"],
                                   a0_capped=a0["capped"])


def run(scenarios=H.ALL_SCENARIOS):
    all_rows, sections, results, stats = [], [], {}, {}
    for name in scenarios:
        rows, md, verdict, st = _run_one(name)
        all_rows.extend(rows)
        sections.extend(md)
        results[name] = verdict
        stats[name] = st

    H.write_csv(OUT / "ablation.csv", [
        {k: (f"{v:.6e}" if isinstance(v, float) and k in ("J_true", "cost_true_energy")
             else (f"{v:.3f}" if isinstance(v, float) else v)) for k, v in r.items()}
        for r in all_rows])
    measured = [k for k, v in results.items() if v != "NOT MEASURED"]
    failed = [k for k, v in results.items() if v == "FAIL"]
    unmeasured = [k for k, v in results.items() if v == "NOT MEASURED"]
    all_pass = not failed and "phase120" in measured

    md = [f"# Pillar 2 -- Ablation (energy, n_seg={N_SEG}, {len(scenarios)} geometries)", "",
          "The conclusion under test is an ISOLATION claim, so it has to hold on more",
          "than the geometry it was diagnosed on: the trust region is the primary fix",
          "and the mis-scaled proximal is a secondary aggravator. Trust cells use each",
          "scenario's own r0 (see module docstring).", "",
          "A geometry where (a0) itself runs to the cap yields NOT MEASURED: with both",
          "legacy cells censored at the same ceiling the aggravation comparison is",
          "empty, and reading that as either result would be a measurement artifact.", "",
          "| scenario | (a) iters | (a0) iters | (b) iters | (c) iters | a0/b ratio | verdict |",
          "|---|---|---|---|---|---|---|"]
    for name in results:
        s = stats[name]
        a0_cell = f"{s['a0']}{' (cap)' if s['a0_capped'] else ''}"
        md.append(f"| {name} | {s['a']}{' (cap)' if s['a'] >= CAP else ''} | {a0_cell} | "
                  f"{s['b']} | {s['c']} | "
                  f"{'≥' if s['a0_capped'] else ''}{s['a0'] / max(s['b'], 1):.1f}x | "
                  f"**{results[name]}** |")
    md += ["",
           f"- aggravation measurable on {len(measured)}/{len(scenarios)} geometries: "
           f"{', '.join(measured) or 'none'}"
           + (f" (not measured: {', '.join(unmeasured)} — (a0) did not terminate "
              f"within {CAP})" if unmeasured else ""),
           f"- every measured geometry agrees: **{not failed}**"
           + (f" — failing: {', '.join(failed)}" if failed else "")]
    md += ["", f"## VERDICT: {'PASS' if all_pass else 'FAIL'}", "", "---", ""] + sections
    H.write_text(OUT / "summary.md", "\n".join(md) + H.provenance())
    print("\n".join(md))
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
