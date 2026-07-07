#!/usr/bin/env python3
"""Static PNG visualisation of the Bézier warm-start curves fed to the DCM Pass 2,
viewed perpendicular to the (equatorial) departure plane so all curves are visible.

Produces:
  results/dcm_warmstart_viz/gallery.png         -- 5-10 warm-start curves overlaid
  results/dcm_warmstart_viz/compare_<id>.png    -- warm-start vs H-S Pass1 vs
                                                   baseline Pass2 vs proposed Pass2

Uses the SAME (fixed) harness code path as tools/dcm_downstream_experiment.py.
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# import the (fixed) experiment module to reuse its exact pipeline
spec = importlib.util.spec_from_file_location("dde", str(REPO / "tools" / "dcm_downstream_experiment.py"))
dde = importlib.util.module_from_spec(spec); sys.modules["dde"] = dde; spec.loader.exec_module(dde)

from orbit_transfer.astrodynamics.orbital_elements import oe_to_rv
from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.collocation.hermite_simpson import HermiteSimpsonCollocation
from orbit_transfer.collocation.multiphase_lgl import MultiPhaseLGLCollocation
from orbit_transfer.collocation.interpolation import interpolate_pass1_to_pass2
from orbit_transfer.optimizer.initial_guess import linear_interpolation_guess
from orbit_transfer.classification.peak_detection import detect_peaks
from orbit_transfer.classification.classifier import determine_phase_structure

OUT = REPO / "results" / "dcm_warmstart_viz"; OUT.mkdir(parents=True, exist_ok=True)


def cfg_of(cid):
    return dde.load_cases(dde.DEFAULT_DB, case_id=cid)[0].config


def ref_orbit(a, e, i, n=400):
    pts = np.array([oe_to_rv((a, e, i, 0.0, 0.0, nu), MU_EARTH)[0]
                    for nu in np.linspace(0, 2 * np.pi, n)])
    return pts


def warm_start(cfg):
    """Bézier warm-start positions (3,M) + feasibility, via the fixed harness."""
    t, x, u, info = dde.bezier_warm_start(cfg, degree=6, n_seg=16, n_samples=300)
    return x[:3], bool(info.get("feasible")), info


def hs_pass1(cfg):
    hs = HermiteSimpsonCollocation(cfg)
    tg, xg, ug, n0, nf = linear_interpolation_guess(cfg, hs.N_points)
    r1 = hs.solve(x_guess=xg, u_guess=ug, nu0_guess=n0, nuf_guess=nf)
    return r1


def baseline_pass2(cfg, r1):
    """Drive Pass 2 explicitly from the H-S Pass1 (so we DON'T silently get the
    Pass-1 fallback). Returns (pos(3,N) or None, converged)."""
    try:
        umag = np.linalg.norm(r1.u, axis=0)
        npk, pt, pw = detect_peaks(r1.t, umag, r1.T_f)
        ph = determine_phase_structure(pt, pw, r1.T_f)
        _, xph, uph = interpolate_pass1_to_pass2(r1.t, r1.x, r1.u, ph)
        lgl = MultiPhaseLGLCollocation(cfg, ph, T_fixed=r1.T_f)
        r2 = lgl.solve(x_phases=xph, u_phases=uph, nu0_guess=r1.nu0, nuf_guess=r1.nuf)
        return (r2.x[:3] if r2.converged else None), r2.converged
    except Exception:
        return None, False


def proposed_pass2(cfg):
    res, _, _, binfo = dde.run_proposed(cfg, degree=6, n_seg=16)
    ok = bool(res.converged)
    return (res.x[:3] if ok else None), ok, binfo


def draw_bodies(ax, r_e, lim, proj):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(R_E * np.cos(th), R_E * np.sin(th), color="#6fa8dc", alpha=0.35, zorder=0)
    ax.plot(r_e * np.cos(th), r_e * np.sin(th), "--", color="#C0392B", lw=1.2,
            label=f"KOZ r={r_e:.0f} km", zorder=1)
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(proj[0] + " [km]"); ax.set_ylabel(proj[1] + " [km]")
    ax.grid(alpha=0.25)


# ── Gallery: many warm-start curves, perpendicular to departure plane ───────────
def gallery(case_ids, lim_cap=15000.0):
    fig, (axxy, axxz) = plt.subplots(1, 2, figsize=(15, 7.2))
    lim = 0.0
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(case_ids)))
    for k, cid in enumerate(case_ids):
        cfg = cfg_of(cid)
        try:
            pos, feas, info = warm_start(cfg)
        except Exception as e:
            print(f"  gallery case {cid}: {e}"); continue
        lim = max(lim, np.max(np.abs(pos)) * 1.05)
        col = cmap[k]
        ls = "-" if feas else ":"
        lw = 2.0 if feas else 2.4
        lbl = f"#{cid} T={cfg.T_max_normed:.2f} {'feas' if feas else 'INFEAS'}"
        for ax, (a, b) in ((axxy, (0, 1)), (axxz, (0, 2))):
            ax.plot(pos[a], pos[b], ls, color=col, lw=lw, label=lbl if ax is axxy else None,
                    zorder=3 if feas else 4)
            ax.plot(pos[a, 0], pos[b, 0], "o", color=col, ms=4)
    r_e = R_E + cfg_of(case_ids[0]).h_min
    lim = min(lim, lim_cap)
    draw_bodies(axxy, r_e, lim, ("X", "Y"))
    draw_bodies(axxz, r_e, lim, ("X", "Z"))
    axxy.set_title("Top-down  (look down +Z, ⟂ to departure plane)  [zoomed to near-KOZ]")
    axxz.set_title("Edge-on  (X–Z, shows out-of-plane / inclination)")
    axxy.legend(fontsize=7.5, loc="upper right", ncol=1)
    fig.suptitle("Bézier warm-start curves fed to DCM Pass 2  "
                 "(solid = KOZ-feasible, dotted = infeasible)", fontsize=13)
    fig.tight_layout()
    p = OUT / "gallery.png"; fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


# ── Per-case comparison: warm-start vs H-S Pass1 vs baseline/proposed Pass2 ─────
def compare(cid):
    cfg = cfg_of(cid)
    ws, feas, _ = warm_start(cfg)
    r1 = hs_pass1(cfg)
    p1 = r1.x[:3]
    base2, base_ok = baseline_pass2(cfg, r1)
    prop2, prop_ok, binfo = proposed_pass2(cfg)
    dep = ref_orbit(cfg.a0, cfg.e0, cfg.i0)
    arr = ref_orbit(cfg.af, cfg.ef, cfg.if_)
    r_e = R_E + cfg.h_min

    stacks = [ws, p1, dep, arr]
    if base2 is not None: stacks.append(base2)
    if prop2 is not None: stacks.append(prop2)
    lim = np.max([np.max(np.abs(s)) for s in stacks]) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, (a, b), ttl in ((axes[0], (0, 1), "Top-down (X–Y, ⟂ departure plane)"),
                            (axes[1], (0, 2), "Edge-on (X–Z)")):
        ax.plot(dep[:, a], dep[:, b], color="green", lw=1, ls=":", alpha=0.5, label="departure orbit")
        ax.plot(arr[:, a], arr[:, b], color="orange", lw=1, ls=":", alpha=0.6, label="arrival orbit")
        ax.plot(ws[a], ws[b], color="magenta", lw=2.4, ls="--",
                label=f"Bézier warm-start ({'feas' if feas else 'INFEAS'})")
        ax.plot(p1[a], p1[b], color="0.45", lw=1.8, label="stock H-S Pass 1")
        if base2 is not None:
            ax.plot(base2[a], base2[b], color="blue", lw=2.2, label="baseline DCM Pass 2")
        if prop2 is not None:
            ax.plot(prop2[a], prop2[b], color="red", lw=2.2, ls="-.", label="proposed DCM Pass 2")
        draw_bodies(ax, r_e, lim, ("XYZ"[a], "XYZ"[b]))
        ax.set_title(ttl)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Case {cid}: T_normed={cfg.T_max_normed:.2f}, Δa={cfg.delta_a:.0f} km, "
                 f"Δi={cfg.delta_i:.1f}°  |  baseline Pass2={'OK' if base_ok else 'FAIL'}, "
                 f"proposed Pass2={'OK' if prop_ok else 'FAIL'}", fontsize=12)
    fig.tight_layout()
    p = OUT / f"compare_{cid:03d}.png"; fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p, f"(base_ok={base_ok} prop_ok={prop_ok} ws_feas={feas})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-only", action="store_true")
    a = ap.parse_args()
    # feasible (short+long) and genuinely KOZ-infeasible (long, curve penetrates KOZ)
    GALLERY = [2, 39, 126, 117, 127, 31, 36, 110, 116, 13]
    gallery(GALLERY)
    if not a.gallery_only:
        for cid in (4, 125, 17, 88):
            try:
                compare(cid)
            except Exception as e:
                print(f"compare {cid} failed: {e}")
