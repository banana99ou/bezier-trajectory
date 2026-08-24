"""Why does a run certify at ~0 and still penetrate?

Rebuilds the exact KOZ rows at a RETURNED iterate and reports, per
(segment, obstacle) pair, which of the three outcomes it got: a row, no row
because the obstacle was out of reach, or no row because no half-space exists.
Then locates the actual penetration and asks which pair should have covered it.

    python3 tools/diagnose_clip.py <scenario> <N> <n_seg>
"""

from __future__ import annotations

import sys

import numpy as np

import bezier_opt
from spacetime_bezier.geometry import (
    obstacle_array_bundle,
    obstacle_positions_at,
    normalize_obstacles,
)
from spacetime_bezier.optimize import optimize_scenario
from spacetime_bezier.scenarios import SCENARIO_MAP


def main(argv: list[str]) -> int:
    name = argv[0] if argv else "diverse"
    N = int(argv[1]) if len(argv) > 1 else 8
    n_seg = int(argv[2]) if len(argv) > 2 else 8
    sound = "--sound-clip" in argv

    fn, _ = SCENARIO_MAP[name]
    sc = fn()
    out = optimize_scenario(sc, [(N, n_seg)], verbose=False, sound_clip=sound)
    key = f"N{N}_seg{n_seg}"
    r = out["results"][key]
    P = np.asarray(r["control_points"], float)
    dim = P.shape[1]
    spatial_dim = dim - 1
    trust = float(r.get("trust_radius", 0.5))

    print(f"{name} {key}  sound_clip={sound}")
    print(f"  clearance   {r['min_clearance']:+.6f}")
    print(f"  koz cert    {r['certificate_violation']:.3e}")
    print(f"  slack       {r['total_slack']:.3e}")
    print(f"  trust       {trust}")

    ctrl, radii = obstacle_array_bundle(sc["obstacles"], spatial_dim)
    res = bezier_opt.spacetime_koz_rows_exact(
        p=P, obstacle_ctrl=ctrl, obstacle_r=radii, n_seg=n_seg,
        trust_radius=trust, sound_clip=sound,
    )
    normals, lbs, seg, cp, obs, rho, sound_flags, dropped, unsound = res
    seg = np.asarray(seg)
    obs = np.asarray(obs)
    pairs = {(int(a), int(b)) for a, b in zip(seg, obs)}
    n_obs = ctrl.shape[0]
    print(f"\n  rows emitted for {len(pairs)} of {n_seg * n_obs} (segment, obstacle) pairs")
    print(f"  dropped_planes (row needed, none exists) : {dropped}")
    print(f"  unsound_clips  (statement 7 failed)      : {unsound}")
    if len(sound_flags):
        print(f"  rows with sound=False                    : {sum(1 for f in sound_flags if not f)}")

    # Where does it actually penetrate, and which obstacle?
    from spacetime_bezier.geometry import _eval_at

    taus = np.linspace(0, 1, 4000)
    pts = _eval_at(P, taus)
    norm_obs = normalize_obstacles(sc["obstacles"])
    worst = (np.inf, -1, -1.0)
    for m, o in enumerate(norm_obs):
        cps = np.asarray(o["control_points"], float)
        t0, t1 = float(cps[0, -1]), float(cps[-1, -1])
        tv = pts[:, -1]
        act = (tv >= t0) & (tv <= t1)
        if not act.any():
            continue
        d = np.linalg.norm(pts[act, :spatial_dim] - obstacle_positions_at(cps, tv[act]), axis=1)
        d = d - float(o["radius"])
        j = int(np.argmin(d))
        if d[j] < worst[0]:
            worst = (float(d[j]), m, float(taus[act][j]))

    clr, m_bad, tau_bad = worst
    if m_bad < 0:
        print("\n  no active obstacle anywhere on the curve")
        return 0
    print(f"\n  worst penetration {clr:+.6f} at tau={tau_bad:.4f}, obstacle {m_bad}"
          f" ({norm_obs[m_bad].get('name', '?')})")
    seg_of_tau = min(int(tau_bad * n_seg), n_seg - 1)
    print(f"  that tau lies in segment {seg_of_tau}")
    got = (seg_of_tau, m_bad) in pairs
    print(f"  did segment {seg_of_tau} get a row against obstacle {m_bad}? -> {got}")
    if got:
        sel = (seg == seg_of_tau) & (obs == m_bad)
        print(f"     rho used = {np.asarray(rho)[sel][0]:.4f}"
              f"   sound = {bool(np.asarray(sound_flags)[sel.nonzero()[0][0]])}")
        print("     ROW EXISTS AND IS SATISFIED YET THE CURVE PENETRATES:")
        print("     that is the statement (7) hole -- the penetration is outside B(c, rho).")
    else:
        print("     NO ROW WAS EMITTED. The certificate is silent about this pair,")
        print("     so summing row violations reports 0.0 -- a check that cannot fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
