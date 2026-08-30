#!/usr/bin/env python3
"""Debug/monitoring viewer for the paper-1 solver. Standalone; shares nothing
with sandbox.py / viewer.py / io.py.

Design rule: this tool computes NO geometry the solver already computed. It
renders the solver's own per-iteration trace (STTRACE) and the solver's own
exact constraint rows (via `spacetime_koz_rows_exact`, the same builder the
certificate uses). The single deliberate exception: quantities that SHOULD be
computed twice -- the independent sampled clearance vs the solver's reported
one, and the certificate recomputed from returned control points vs the
reported one -- are always shown as a PAIR with their delta, never one
standing in for the other.

Output is one self-contained HTML file. No server, no ports, no external
assets, so there is nothing to be stale relative to except the compiled
extension -- and that is exactly what the header checks.

Usage:
    python3 tools/trace_viewer.py original            # N8_seg4 default
    python3 tools/trace_viewer.py fence3d --N 8 --seg 2
    python3 tools/trace_viewer.py wall --N 10 --seg 16 --out /tmp/wall.html
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # run from anywhere; the package is not installed

# Columns of an STTRACE data line, matching the header the Rust emits at
# spacetime_optimizer.rs (grep STTRACE). Read the header at parse time and
# refuse to render if it disagrees -- a silently reordered column is exactly
# the kind of lie this tool exists to prevent.
TRACE_HEADER = (
    "it,outcome,rho,pred,act,step_norm,trust_before,trust_after,"
    "vlin_p,vlin_c,vtrue_c,hard_viol_p,clearance,total_slack,conv_streak,stat_streak"
)

CHILD_SCRIPT = r"""
import json, sys
import numpy as np
from spacetime_bezier.scenarios import SCENARIO_MAP, scenario_elastic_weight
from spacetime_bezier.optimize import optimize_spacetime

req = json.loads(sys.argv[1])
fn, _configs = SCENARIO_MAP[req["scenario"]]
sc = fn()
weight = req["elastic_weight"]
if weight is None:
    weight = scenario_elastic_weight(req["scenario"])
P_opt, info = optimize_spacetime(
    N=req["N"], dim=len(sc["start"]), p_start=sc["start"], p_end=sc["end"],
    obstacles=sc["obstacles"], n_seg=req["seg"], max_iter=req["max_iter"],
    tol=req["tol"], scp_trust_radius=req["trust_radius"], min_dt=req["min_dt"],
    elastic_weight=weight, verbose=False, init_curve=sc.get("init_curve"),
    stations=sc.get("stations"),
)
# ---- iteration replay: drive the SAME canonical Rust iteration the solve
# used, via the stepping context, with identical parameters. The replay is the
# solver's own state per step, never a viewer-side re-derivation. Each frame
# records the post-step reference, the QP candidate, whether the reference
# advanced, and the constraint rows built at the pre-step reference (the 24
# tightest by margin, plus every violated one).
import bezier_opt
from spacetime_bezier.geometry import bezier_curve, obstacle_array_bundle

def _capture_replay():
    dim = len(sc["start"])
    spatial_dim = dim - 1
    obstacle_ctrl, obstacle_radii = obstacle_array_bundle(sc["obstacles"], spatial_dim)
    from spacetime_bezier.objective import build_initial_guess
    P0 = build_initial_guess(sc["start"], sc["end"], req["N"] + 1,
                             init_curve=sc.get("init_curve"))
    st = sc.get("stations")
    ctx = bezier_opt.SpacetimeScpContext(
        p_init=np.asarray(P0, float), obstacle_ctrl=obstacle_ctrl,
        obstacle_r=obstacle_radii,
        # These MUST match optimize_spacetime's defaults exactly, or the replay
        # is a different run -- the drift check below is what catches a mismatch.
        n_seg=req["seg"], min_dt=req["min_dt"], coord_lb=-20.0, coord_ub=20.0,
        time_lb=0.0, time_ub=float(P0[-1, -1]) * 1.5,
        scp_prox_weight=0.5, scp_trust_radius=req["trust_radius"],
        elastic_weight=weight, tol=req["tol"], sound_clip=False,
        stations=(np.asarray(st, float) if st else None),
    )
    frames, prev = [], np.asarray(P0, float)
    n_steps = max(1, min(int(info.get("iterations", 1)), req["max_iter"]))
    for it in range(1, n_steps + 1):
        (p_new, sinfo, seg_i, cp_i, obs_i, _it_i,
         _norm, supports, _cent, _lbs, margins, _slk) = ctx.step()
        p_new = np.asarray(p_new, float)
        supports = np.asarray(supports, float).reshape(-1, dim)
        margins = np.asarray(margins, float)
        keep = np.argsort(margins)[: max(24, int((margins < 0).sum()))]
        cand = np.asarray(sinfo["p_candidate"], float) if "p_candidate" in sinfo else p_new
        frames.append({
            "it": it,
            "advanced": bool(not np.allclose(p_new, prev)),
            "status": str(sinfo.get("solver_status", "?")),
            "total_slack": float(sinfo.get("total_slack", float("nan"))),
            "P": p_new.tolist(),
            "cand": np.asarray(cand, float).reshape(p_new.shape).tolist(),
            "curve": bezier_curve(p_new, num_pts=120).tolist(),
            "sup": supports[keep].tolist(),
            "sup_margin": margins[keep].tolist(),
        })
        prev = p_new
        if str(sinfo.get("solver_status", "")) == "Failed":
            break
    # Honesty pair: the replay must land where the solve landed (unless the
    # best-iterate fallback returned a different point -- reported either way).
    drift = float(np.max(np.abs(prev - np.asarray(P_opt, float))))
    return frames, drift

# No such object since the center-surface rewrite. This returned the
# conservative containing ball the old occlusion builder wrapped the occluder
# in, and the page drew a static cone from it. The shadow is now generated by
# the centreline's own shadow surface and its walls ARE the geometry; a cone
# drawn from a ball nothing built would be a picture of a constraint that is
# not there. (No docstring here: this whole block lives inside CHILD_SCRIPT's
# raw triple-quoted string.)
def _shadow_balls():
    return None


replay_frames, replay_drift = _capture_replay()
json.dump({
    "P": np.asarray(P_opt, float).tolist(),
    "info": {k: (float(v) if isinstance(v, (int, float)) else v)
             for k, v in info.items() if isinstance(v, (int, float, str))},
    "obstacles": sc["obstacles"],
    "start": list(map(float, sc["start"])), "end": list(map(float, sc["end"])),
    "elastic_weight": float(weight),
    "stations": sc.get("stations"),
    "replay": replay_frames,
    "replay_drift": replay_drift,
    "shadow_balls": _shadow_balls(),
}, sys.stdout)
"""


def run_solve(args) -> tuple[dict, list[dict], list[str]]:
    """Run one solve in a subprocess with the trace enabled.

    A subprocess, not an in-process call: the Rust trace is eprintln straight to
    fd 2, which a Python-level stderr redirect does not capture.
    """
    req = {
        "scenario": args.scenario, "N": args.N, "seg": args.seg,
        "elastic_weight": args.elastic_weight, "max_iter": args.max_iter,
        "tol": args.tol, "trust_radius": args.trust_radius, "min_dt": args.min_dt,
    }
    env = dict(os.environ, SPACETIME_SCVX_TRACE="1")
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT, json.dumps(req)],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=600,
    )
    if proc.returncode != 0:
        sys.exit(f"solve failed:\n{proc.stderr[-3000:]}")
    payload = json.loads(proc.stdout)

    rows, warnings = [], []
    header_seen = None
    for line in proc.stderr.splitlines():
        if not line.startswith("STTRACE"):
            continue
        if line.startswith("STTRACE,it,"):
            header_seen = line[len("STTRACE,"):].replace("\\", "")
            continue
        parts = line.split(",")[1:]
        cols = TRACE_HEADER.split(",")
        if len(parts) != len(cols):
            warnings.append(f"malformed trace line skipped: {line[:80]}")
            continue
        row = dict(zip(cols, parts))
        for k in cols:
            if k not in ("outcome",):
                row[k] = float(row[k])
        rows.append(row)
    if header_seen is not None and header_seen != TRACE_HEADER:
        sys.exit(
            "trace header mismatch -- the Rust emits different columns than this "
            f"tool expects.\n  emitted:  {header_seen}\n  expected: {TRACE_HEADER}\n"
            "Refusing to render rather than mislabel columns."
        )
    if not rows:
        warnings.append(
            "NO TRACE LINES CAPTURED -- the loaded extension predates the "
            "STTRACE emission or the env var did not reach it. Timeline is empty "
            "for that reason, not because the loop did nothing."
        )
    return payload, rows, warnings


def recompute(payload: dict, n_seg: int) -> dict:
    """The deliberate second computations: exact rows via the solver's own
    builder at the RETURNED control points, and an independent sampled
    clearance. Both are paired with the reported values in the page."""
    import numpy as np
    import bezier_opt
    from spacetime_bezier.geometry import compute_min_clearance
    from orbital_docking.de_casteljau import segment_matrices_equal_params

    P = np.asarray(payload["P"], float)
    obstacles = payload["obstacles"]
    dim = P.shape[1]
    ctrl, r = obstacle_array_bundle(obstacles, P.shape[1] - 1)
    (normals, lbs, seg, cp, obs, _comp, _sta, _rho, _sound,
     _dropped, _dropped_shadow, _unsound) = (
        bezier_opt.spacetime_koz_rows_exact(
            p=P, obstacle_ctrl=ctrl, obstacle_r=r, n_seg=n_seg,
        )
    )
    a_list = [np.asarray(a, float) for a in
              segment_matrices_equal_params(P.shape[0] - 1, n_seg)]
    ledger = []
    for n, lb, s_i, c_i, o_i in zip(
        np.asarray(normals, float), np.asarray(lbs, float),
        np.asarray(seg), np.asarray(cp), np.asarray(obs),
    ):
        q = (a_list[int(s_i)] @ P)[int(c_i)]
        slack = float(n @ q) - float(lb)  # >= 0 means satisfied
        ledger.append({
            "seg": int(s_i), "cp": int(c_i), "obs": int(o_i),
            "n_spatial": [float(x) for x in n[:-1]], "n_time": float(n[-1]),
            "slack": slack,
        })
    cert = sum(max(0.0, -row["slack"]) for row in ledger)
    clearance = float(compute_min_clearance(P, obstacles, dim=dim, n_eval=20001))
    los = None
    if payload.get("stations"):
        from spacetime_bezier.geometry import compute_los_margin
        t_v, m_v = compute_los_margin(P, payload["stations"][0], obstacles,
                                      dim=dim, n_eval=2001)
        los = {"t": [float(x) for x in t_v], "m": [float(x) for x in m_v],
               "min": float(min(m_v))}
    return {"ledger": ledger, "certificate": cert, "clearance": clearance,
            "P": P.tolist(), "dim": dim, "los": los}


def geometry_data(payload: dict, rc: dict) -> dict:
    """Everything the 3D panel draws, computed with the package's own evaluator
    (`bezier_curve` -- the same sampling the clearance check uses, so the curve
    on screen IS the curve that was checked). Obstacle tubes come from the
    scenario parameters, not from any viewer-side re-derivation."""
    import numpy as np
    from spacetime_bezier.geometry import bezier_curve

    P = np.asarray(rc["P"], float)
    dim = P.shape[1]
    pts = bezier_curve(P, num_pts=400)
    t_lo, t_hi = float(P[0, -1]), float(P[-1, -1])

    obstacles = []
    for o in payload["obstacles"]:
        o_t0 = max(float(o.get("t_start", t_lo)), t_lo)
        o_t1 = min(float(o.get("t_end", t_hi)), t_hi)
        if o_t1 < o_t0:
            continue
        obstacles.append({
            "pos0": [float(v) for v in o["pos0"]],
            "vel": [float(v) for v in o["vel"]],
            "r": float(o["r"]), "t0": o_t0, "t1": o_t1,
        })
    sight = None
    st = payload.get("stations")
    if st and dim == 4 and rc.get("los") is not None:
        import numpy as _np
        station = [float(v) for v in st[0]]
        t_arr = _np.asarray(rc["los"]["t"]); m_arr = _np.asarray(rc["los"]["m"])
        lines = []
        for k in range(0, len(pts), max(1, len(pts) // 16)):
            tq = float(pts[k, -1])
            j = int(_np.argmin(_np.abs(t_arr - tq)))
            lines.append({"v": [float(x) for x in pts[k, :3]],
                          "m": float(m_arr[j]), "t": tq})
        sight = {"station": station, "lines": lines}
    return {"curve": pts.tolist(), "cp": P.tolist(), "dim": dim,
            "t_lo": t_lo, "t_hi": t_hi, "obstacles": obstacles,
            "start": payload["start"], "end": payload["end"],
            "sight": sight, "shadow_balls": payload.get("shadow_balls")}


def render_geometry(geo: dict) -> str:
    """First panel: the trajectory you can rotate. dim==3 draws (x, y, t) with
    obstacle tubes as wireframes; dim==4 draws (x, y, z) with TIME AS COLOR on
    both the curve and obstacle snapshots -- same colorscale, so 'same color
    near each other' means 'close at the same moment'. Refuses any other dim."""
    dim = geo["dim"]
    if dim not in (3, 4):
        return f"<h2>Trajectory</h2><p><b>dim={dim} has no honest projection; refusing to draw.</b></p>"

    plotly = (REPO / "spacetime_bezier" / "static" / "plotly.min.js").read_text()
    data = json.dumps(geo)
    if dim == 3:
        js = """
      // (x, y, t): the lifted space itself. Tubes are the obstacles' true
      // swept geometry: circle wireframes at time slices along the slanted axis.
      const traces = [];
      for (const o of G.obstacles) {
        const nSlice = 24, nTh = 28;
        for (let i = 0; i <= nSlice; i++) {
          const t = o.t0 + (o.t1 - o.t0) * i / nSlice;
          const cx = o.pos0[0] + o.vel[0] * t, cy = o.pos0[1] + o.vel[1] * t;
          const xs = [], ys = [], ts = [];
          for (let k = 0; k <= nTh; k++) {
            const th = 2 * Math.PI * k / nTh;
            xs.push(cx + o.r * Math.cos(th)); ys.push(cy + o.r * Math.sin(th)); ts.push(t);
          }
          traces.push({type: "scatter3d", mode: "lines", x: xs, y: ys, z: ts,
            line: {color: "rgba(200,60,60,0.35)", width: 1},
            hoverinfo: "skip", showlegend: false});
        }
      }
      traces.push({type: "scatter3d", mode: "lines", name: "trajectory",
        x: G.curve.map(p => p[0]), y: G.curve.map(p => p[1]), z: G.curve.map(p => p[2]),
        line: {color: "#1155cc", width: 5}});
      traces.push({type: "scatter3d", mode: "lines+markers", name: "control polygon",
        x: G.cp.map(p => p[0]), y: G.cp.map(p => p[1]), z: G.cp.map(p => p[2]),
        line: {color: "#888", width: 2, dash: "dot"}, marker: {size: 3, color: "#555"}});
      const scene = {xaxis: {title: "x"}, yaxis: {title: "y"}, zaxis: {title: "t (time)"}};
"""
        caption = ("(x, y, t) — the lifted space. Red wireframes are the obstacle "
                   "tubes exactly as constrained: slanted = moving, finite height = "
                   "time-limited. A vertical stretch of the blue curve is WAITING.")
    else:
        js = """
      // (x, y, z) with time as color. Obstacle spheres drawn at snapshots
      // sharing the curve's colorscale: same color + near = close at that moment.
      const traces = [];
      const tSpan = G.t_hi - G.t_lo || 1;
      const nSnap = 7;
      for (const o of G.obstacles) {
        for (let i = 0; i <= nSnap; i++) {
          const t = o.t0 + (o.t1 - o.t0) * i / nSnap;
          const c = [o.pos0[0] + o.vel[0] * t, o.pos0[1] + o.vel[1] * t, o.pos0[2] + o.vel[2] * t];
          const xs = [], ys = [], zs = [];
          for (let a = 0; a < 8; a++) for (let b = 0; b <= 8; b++) {
            const u = Math.PI * a / 8, v = 2 * Math.PI * b / 8;
            xs.push(c[0] + o.r * Math.sin(u) * Math.cos(v));
            ys.push(c[1] + o.r * Math.sin(u) * Math.sin(v));
            zs.push(c[2] + o.r * Math.cos(u));
          }
          traces.push({type: "scatter3d", mode: "markers", x: xs, y: ys, z: zs,
            marker: {size: 2, color: (t - G.t_lo) / tSpan, colorscale: "Viridis",
                     cmin: 0, cmax: 1, opacity: 0.45},
            hovertext: "obstacle @ t=" + t.toFixed(2), hoverinfo: "text", showlegend: false});
        }
      }
      traces.push({type: "scatter3d", mode: "markers", name: "trajectory (color = time)",
        x: G.curve.map(p => p[0]), y: G.curve.map(p => p[1]), z: G.curve.map(p => p[2]),
        marker: {size: 3, color: G.curve.map(p => (p[3] - G.t_lo) / tSpan),
                 colorscale: "Viridis", cmin: 0, cmax: 1,
                 colorbar: {title: "t", tickvals: [0, 1],
                            ticktext: [G.t_lo.toFixed(1), G.t_hi.toFixed(1)]}}});
      traces.push({type: "scatter3d", mode: "lines+markers", name: "control polygon",
        x: G.cp.map(p => p[0]), y: G.cp.map(p => p[1]), z: G.cp.map(p => p[2]),
        line: {color: "#888", width: 2, dash: "dot"}, marker: {size: 3, color: "#555"}});
      if (G.sight) {
        const st = G.sight.station;
        traces.push({type: "scatter3d", mode: "markers", name: "station",
          x: [st[0]], y: [st[1]], z: [st[2]],
          marker: {size: 7, symbol: "diamond", color: "#117733"}});
        for (const L of G.sight.lines) {
          traces.push({type: "scatter3d", mode: "lines", showlegend: false,
            x: [st[0], L.v[0]], y: [st[1], L.v[1]], z: [st[2], L.v[2]],
            line: {color: L.m > 0 ? "rgba(20,120,60,0.35)" : "rgba(200,40,40,0.8)",
                   width: L.m > 0 ? 1 : 3},
            hovertext: "sight line t=" + L.t.toFixed(2) + " margin=" + L.m.toFixed(3),
            hoverinfo: "text"});
        }
      }
      if (G.shadow_balls) {
        // The solver's own inflated occlusion balls (per piece window): the
        // bodies the sight line must clear, exported by the exact-rows builder.
        for (const b of G.shadow_balls) {
          const xs = [], ys = [], zs = [];
          for (let a = 0; a < 6; a++) for (let k = 0; k <= 12; k++) {
            const u = Math.PI * a / 6, v = 2 * Math.PI * k / 12;
            xs.push(b.c[0] + b.R * Math.sin(u) * Math.cos(v));
            ys.push(b.c[1] + b.R * Math.sin(u) * Math.sin(v));
            zs.push(b.c[2] + b.R * Math.cos(u));
          }
          traces.push({type: "scatter3d", mode: "markers", x: xs, y: ys, z: zs,
            marker: {size: 1.6, color: "rgba(120,70,160,0.5)"},
            hovertext: "occlusion ball R=" + b.R.toFixed(2) +
                       " window [" + b.t0.toFixed(2) + ", " + b.t1.toFixed(2) + "]",
            hoverinfo: "text", showlegend: false});
        }
      }
      const scene = {xaxis: {title: "x"}, yaxis: {title: "y"}, zaxis: {title: "z (altitude)"},
                     aspectmode: "data"};
"""
        caption = ("(x, y, z), TIME AS COLOR on curve and obstacle snapshots alike — "
                   "same color near each other means close at the same moment. "
                   "Distant colors passing through the same place are NOT a conflict. "
                   "Green diamond = station; sight lines red where the link is lost; "
                   "purple shells are the solver's own inflated occlusion balls "
                   "(per piece window, from the exact-rows builder) — the bodies the "
                   "sight line must clear, i.e. the reason the curve climbs.")
    return f"""
    <h2>Trajectory</h2>
    <p style="font-size:12px">{caption} Drag to rotate; scroll to zoom.</p>
    <div id="geo" style="width:100%;height:640px"></div>
    <script>{plotly}</script>
    <script>
      const G = {data};
      {js}
      Plotly.newPlot("geo", traces, {{scene: scene, margin: {{l:0,r:0,t:0,b:0}},
        showlegend: true, legend: {{x: 0, y: 1}}}});
    </script>"""


def build_identity() -> dict:
    import bezier_opt as rust_optimizer_py
    so = pathlib.Path(rust_optimizer_py.__file__)
    so_m = so.stat().st_mtime
    newest_src = max(
        (p.stat().st_mtime for p in (REPO / "rust_optimizer").rglob("*.rs")),
        default=0.0,
    )
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True, cwd=REPO).stdout.strip())
    return {
        "so": str(so), "so_mtime": so_m, "stale": newest_src > so_m,
        "git": git + ("+dirty" if dirty else ""),
    }


# ---------------------------------------------------------------------------
# Rendering. Inline SVG only; no libraries, no theme, no interactivity beyond
# <details>. The page is a record, not an app.
# ---------------------------------------------------------------------------

def svg_line(series: list[float], w=560, h=90, color="#2266cc", logy=False) -> str:
    import math
    vals = [math.log10(max(v, 1e-16)) if logy else v for v in series]
    if not vals:
        return "<i>empty</i>"
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pts = " ".join(
        f"{4 + i * (w - 8) / max(len(vals) - 1, 1):.1f},"
        f"{h - 4 - (v - lo) / span * (h - 8):.1f}"
        for i, v in enumerate(vals)
    )
    lab = (f"{10**lo:.1e} .. {10**hi:.1e}" if logy else f"{lo:.3g} .. {hi:.3g}")
    return (f'<svg width="{w}" height="{h}" style="background:#f7f7f7">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>'
            f'<text x="6" y="12" font-size="10" fill="#555">{lab}</text></svg>')


def fmt(v, nd=4):
    return f"{v:.{nd}g}" if isinstance(v, float) else html.escape(str(v))


def render_station_view(payload: dict, rc: dict) -> str:
    """Film strip: six stills of what the station sees -- occluder disks and
    the vehicle as angular objects (azimuth/elevation from the station), from
    the TRUE scenario geometry and the returned curve. Small multiples, not an
    animation: everything visible at once. Supplementary intuition; the
    line-of-sight margin strip below remains the proof."""
    import numpy as np
    st = payload.get("stations")
    if not st or rc["dim"] != 4:
        return ""
    station = np.asarray(st[0], float)
    P = np.asarray(rc["P"], float)
    from spacetime_bezier.geometry import bezier_curve
    pts = bezier_curve(P, num_pts=2001)          # (x, y, z, t)
    los = rc.get("los") or {}
    t_lo, t_hi = float(pts[0, -1]), float(pts[-1, -1])
    panels = []
    for tq in [t_lo + (t_hi - t_lo) * k / 5 for k in range(6)]:
        i = int(np.argmin(np.abs(pts[:, -1] - tq)))  # time is monotone; nearest sample
        veh = pts[i, :3]
        d_v = veh - station
        az_v = float(np.degrees(np.arctan2(d_v[1], d_v[0])))
        el_v = float(np.degrees(np.arcsin(d_v[2] / max(np.linalg.norm(d_v), 1e-12))))
        discs = []
        for o in payload["obstacles"]:
            if not (float(o.get("t_start", -1e18)) <= pts[i, -1] <= float(o.get("t_end", 1e18))):
                continue
            c = np.asarray(o["pos0"], float) + np.asarray(o["vel"], float) * pts[i, -1]
            d = c - station
            dist = float(np.linalg.norm(d))
            if dist <= float(o["r"]):
                discs.append((0.0, 0.0, 180.0))   # station inside body: all sky
                continue
            discs.append((float(np.degrees(np.arctan2(d[1], d[0]))),
                          float(np.degrees(np.arcsin(d[2] / dist))),
                          float(np.degrees(np.arcsin(min(1.0, float(o["r"]) / dist))))))
        m_here = None
        if los:
            j = int(np.argmin(np.abs(np.asarray(los["t"]) - pts[i, -1])))
            m_here = float(los["m"][j])
        vis = m_here is None or m_here > 0.0
        # angular window centered between vehicle and occluders
        all_az = [az_v] + [a for a, _, _ in discs]
        all_el = [el_v] + [e for _, e, _ in discs]
        az_c, el_c = sum(all_az) / len(all_az), sum(all_el) / len(all_el)
        span = 30.0
        def sx(a): return 80 + (a - az_c) / span * 74
        def sy(e): return 80 - (e - el_c) / span * 74
        svg = [f'<svg width="160" height="176" style="background:#f2f4f8">']
        for a, e, r in discs:
            svg.append(f'<circle cx="{sx(a):.1f}" cy="{sy(e):.1f}" r="{r/span*74:.1f}" '
                       f'fill="rgba(180,70,70,0.35)" stroke="#b04040"/>')
        color = "#117733" if vis else "#bb2222"
        svg.append(f'<circle cx="{sx(az_v):.1f}" cy="{sy(el_v):.1f}" r="4" fill="{color}"/>')
        label = f"t={pts[i,-1]:.1f}" + ("" if m_here is None else f"  m={m_here:+.2f}")
        svg.append(f'<text x="6" y="172" font-size="11" fill="#333">{label}</text></svg>')
        panels.append("".join(svg))
    return ("<h2>What the station sees (film strip, true geometry)</h2>"
            "<p style='font-size:12px'>Angular view from the station: red discs are the "
            "occluder bodies' angular extent at that instant, the dot is the vehicle "
            "(green = line of sight held, red = lost). Six stills, not an animation — "
            "the margin strip below is the proof; this is the physical meaning.</p>"
            "<div style='display:flex;gap:6px;flex-wrap:wrap'>" + "".join(panels) + "</div>")


def render_replay(payload: dict, rc: dict) -> str:
    """The mechanism panel: one SCP iteration per slider step. The slider is the
    ITERATION INDEX, never time -- each frame is a complete static space-time
    scene, and what changes between frames is the solver's reference. Frames
    carry the solver's own per-step state (stepping context, identical
    parameters): reference polygon, QP candidate as a ghost, and the tightest
    constraint-row support points colored by margin (red = violated)."""
    frames = payload.get("replay") or []
    if not frames:
        return ""
    dim = rc["dim"]
    if dim not in (3, 4):
        return "<h2>Iteration replay</h2><p><b>unsupported dim</b></p>"
    drift = float(payload.get("replay_drift", float("nan")))
    fell_back = bool(payload.get("info", {}).get("returned_best_iterate", 0.0))
    drift_note = (
        f"replay endpoint vs returned control points: max |delta| = {drift:.2e}"
        + (" — the best-iterate fallback fired, so the returned point is an "
           "EARLIER iterate than the replay's last frame (expected mismatch)."
          if fell_back else
          ". A large value here without the fallback means the replay is not "
          "the run — do not trust either.")
    )
    n_rej = sum(1 for f in frames if not f["advanced"])
    data = json.dumps({"frames": frames, "dim": dim,
                       "obstacles": (payload.get("obstacles") or [])})
    js = """
      const R = REPLAY_DATA;
      const dim = R.dim, F = R.frames;
      const col = (f, i) => f.advanced ? "#2255bb" : "#bb2222";
      function frameTraces(f) {
        const t = [];
        const X = c => c.map(q => q[0]), Y = c => c.map(q => q[1]);
        const Z3 = c => c.map(q => q[2]);           // (x,y,t) for dim 3
        const Z4 = c => c.map(q => q[2]);           // (x,y,z) for dim 4
        const zf = dim === 3 ? Z3 : Z4;
        t.push({type:"scatter3d", mode:"lines", name:"reference curve",
          x:X(f.curve), y:Y(f.curve), z:zf(f.curve),
          line:{color:"#2255bb", width:4}});
        t.push({type:"scatter3d", mode:"lines+markers", name:"control polygon",
          x:X(f.P), y:Y(f.P), z:zf(f.P),
          line:{color:"#88aadd", width:2, dash:"dot"}, marker:{size:3}});
        t.push({type:"scatter3d", mode:"lines+markers",
          name:f.advanced ? "candidate (became reference)" : "candidate (REJECTED)",
          x:X(f.cand), y:Y(f.cand), z:zf(f.cand),
          line:{color:col(f), width:2, dash:"dash"},
          marker:{size:2}, opacity:f.advanced ? 0.45 : 0.9});
        if (f.sup.length) {
          t.push({type:"scatter3d", mode:"markers", name:"tightest constraint rows",
            x:X(f.sup), y:Y(f.sup), z:zf(f.sup),
            marker:{size:4, color:f.sup_margin, colorscale:"RdYlGn", cmin:-0.5, cmax:0.5},
            text:f.sup_margin.map(m => "margin " + m.toFixed(4)),
            hoverinfo:"text"});
        }
        return t;
      }
      // static obstacle context, drawn once (same construction as the main scene)
      const staticTraces = [];
      if (dim === 3) {
        for (const o of R.obstacles) {
          const t0 = Math.max(o.t_start ?? 0, 0), t1 = Math.min(o.t_end ?? 10, 10);
          for (let i = 0; i <= 10; i++) {
            const tt = t0 + (t1 - t0) * i / 10;
            const cx = o.pos0[0] + o.vel[0]*tt, cy = o.pos0[1] + o.vel[1]*tt;
            const xs=[], ys=[], zs=[];
            for (let k = 0; k <= 20; k++) {
              const th = 2*Math.PI*k/20;
              xs.push(cx + o.r*Math.cos(th)); ys.push(cy + o.r*Math.sin(th)); zs.push(tt);
            }
            staticTraces.push({type:"scatter3d", mode:"lines", x:xs, y:ys, z:zs,
              line:{color:"rgba(200,60,60,0.25)", width:1}, hoverinfo:"skip",
              showlegend:false});
          }
        }
      }
      const plFrames = F.map((f, i) => ({
        name: String(i),
        data: staticTraces.concat(frameTraces(f)),
      }));
      const steps = F.map((f, i) => ({
        method: "animate", args: [[String(i)],
          {mode:"immediate", frame:{duration:0, redraw:true}, transition:{duration:0}}],
        label: f.it + (f.advanced ? "" : " ✗"),
      }));
      const zTitle = dim === 3 ? "t" : "z";
      Plotly.newPlot("replay", plFrames[0].data, {
        height: 560, margin:{l:0,r:0,t:24,b:0},
        scene:{xaxis:{title:"x"}, yaxis:{title:"y"}, zaxis:{title:zTitle},
               aspectmode:"data"},
        sliders:[{active:0, steps:steps, currentvalue:{prefix:"SCP iteration: "}}],
      }).then(gd => { Plotly.addFrames(gd, plFrames); });
    """
    return f"""
    <h2>Iteration replay — the mechanism, one SCP step per slider notch</h2>
    <p style="font-size:12px">The slider is the <b>iteration index, not time</b>: every frame is a
    complete static space-time scene; what moves between frames is the solver's reference.
    Ghost polygon = the QP candidate (red when the ratio test rejected it, marked ✗ on the
    slider). Dots = the tightest constraint-row support points at that reference, colored by
    margin (red = violated; the elastic phase is the curve being pushed out of the forbidden
    region frame by frame). {n_rej} of {len(frames)} steps rejected.
    {html.escape(drift_note)}</p>
    <div id="replay"></div>
    <script>const REPLAY_DATA = {data};{js}</script>"""


def render(args, payload, trace, warnings, rc, ident, geo_html='', replay_html='', station_html='') -> str:
    info = payload["info"]
    e = html.escape
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- honesty bar ---------------------------------------------------------
    stale_note = (
        '<b style="color:#b00">EXTENSION OLDER THAN RUST SOURCES — rebuild '
        "before trusting anything below</b>"
        if ident["stale"] else "extension newer than sources"
    )
    hdr = f"""
    <div style="border:2px solid #333;padding:8px;font-family:monospace;font-size:12px">
    scenario=<b>{e(args.scenario)}</b> N={args.N} seg={args.seg}
    elastic_weight={fmt(payload['elastic_weight'])} tol={args.tol:g}
    max_iter={args.max_iter} trust_radius={args.trust_radius}
    min_dt={args.min_dt}<br>
    generated {now} · git {e(ident['git'])} · {stale_note}<br>
    stop=<b>{e(str(info.get('stop_label','?')))}</b>
    converged={int(info.get('converged',0))}
    iterations={int(info.get('iterations',-1))}
    returned_best_iterate={int(info.get('returned_best_iterate',0))}
    </div>"""

    # -- the pairs -----------------------------------------------------------
    rep_cert = float(info.get("koz_violation_reference", float("nan")))
    rep_clear = float(info.get("min_clearance", float("nan")))
    occl_pair = ""
    if rc.get("los") is not None:
        rep_occl = float(info.get("occlusion_violation_reference", float("nan")))
        # The occlusion certificate (solver rows) and the sampled LOS margin are
        # DIFFERENT quantities -- one is a sum of row violations, the other a
        # worst-case true margin. Shown side by side, not differenced.
        occl_pair = (
            f"<tr><td>occlusion certificate @ returned P / independent min LOS margin"
            f" (positive = visible)</td><td>{rep_occl:.6g}</td>"
            f"<td>{rc['los']['min']:.6g}</td><td>—</td></tr>"
        )
    pairs = f"""
    <h2>Paired checks (reported vs recomputed — both always shown)</h2>
    <table border=1 cellpadding=4 style="border-collapse:collapse;font-family:monospace">
    <tr><th></th><th>solver reported</th><th>recomputed here</th><th>delta</th></tr>
    <tr><td>certificate violation (exact rows @ returned P)</td>
        <td>{rep_cert:.6g}</td><td>{rc['certificate']:.6g}</td>
        <td>{abs(rep_cert - rc['certificate']):.2e}</td></tr>
    <tr><td>min clearance (solver 1500 samples vs 20001 here)</td>
        <td>{rep_clear:.6g}</td><td>{rc['clearance']:.6g}</td>
        <td>{abs(rep_clear - rc['clearance']):.2e}</td></tr>
    {occl_pair}
    </table>
    <p style="font-size:12px">Honesty note on what each pair can catch. The certificate
    pair calls the SAME Rust builder on both legs, so it can only expose plumbing
    mismatches (row embedding, subdivision matrices) — a wrong geometry moves both legs
    identically and this pair stays at zero delta. The clearance pair is the genuinely
    independent one (separate implementation, separate language), though both legs sample
    uniformly in the curve parameter, so they share that method's blind spots. A clearance
    disagreement is a finding; a certificate agreement proves plumbing, not geometry.</p>"""

    # -- timeline ------------------------------------------------------------
    if trace:
        rej = sum(1 for r in trace if r["outcome"] != "accept")
        spark = "".join(
            f"<div><b>{name}</b><br>{svg_line([r[key] for r in trace], logy=logy)}</div>"
            for name, key, logy in [
                ("trust radius (after)", "trust_after", True),
                ("rho", "rho", False),
                ("true violation @ candidate", "vtrue_c", True),
                ("clearance", "clearance", False),
            ]
        )
        rows_html = "".join(
            f'<tr style="background:{"#ffe8e8" if r["outcome"] != "accept" else "#fff"}">'
            + "".join(f"<td>{fmt(r[c])}</td>" for c in TRACE_HEADER.split(","))
            + "</tr>"
            for r in trace
        )
        head = "".join(f"<th>{c}</th>" for c in TRACE_HEADER.split(","))
        timeline = f"""
        <h2>Iteration timeline — every step, rejected steps highlighted</h2>
        <p style="font-size:12px">{len(trace)} steps, {rej} rejected.</p>
        <div style="display:flex;gap:12px;flex-wrap:wrap">{spark}</div>
        <details><summary>full table</summary>
        <table border=1 cellpadding=3 style="border-collapse:collapse;font-family:monospace;font-size:11px">
        <tr>{head}</tr>{rows_html}</table></details>"""
    else:
        timeline = "<h2>Iteration timeline</h2><p><b>empty — see warnings</b></p>"

    # -- ledger --------------------------------------------------------------
    led = sorted(rc["ledger"], key=lambda r: r["slack"])
    zero_t = sum(1 for r in rc["ledger"] if abs(r["n_time"]) < 1e-12)
    moving = any(any(abs(v) > 0 for v in o["vel"]) for o in payload["obstacles"])
    zt_flag = (
        f'<b style="color:#b00">{zero_t} rows have a ZERO time component while '
        "obstacles are moving — that is the G1 defect signature</b>"
        if zero_t and moving else f"{zero_t} rows with zero time component"
    )
    led_rows = "".join(
        f'<tr style="background:{"#ffe8e8" if r["slack"] < 0 else "#fff"}">'
        f'<td>{r["seg"]}</td><td>{r["cp"]}</td><td>{r["obs"]}</td>'
        f'<td>{", ".join(f"{x:.3f}" for x in r["n_spatial"])}</td>'
        f'<td>{r["n_time"]:.3f}</td><td>{r["slack"]:.5f}</td></tr>'
        for r in led[:200]
    )
    ledger = f"""
    <h2>Constraint ledger — exact rows @ returned iterate</h2>
    <p style="font-size:12px">{len(led)} rows ({len(payload['obstacles'])} obstacles ×
    {args.seg} segments × control points). {zt_flag}. Sorted worst-first,
    first 200 shown; violated rows red.</p>
    <table border=1 cellpadding=3 style="border-collapse:collapse;font-family:monospace;font-size:11px">
    <tr><th>seg</th><th>cp</th><th>obs</th><th>normal (spatial)</th>
    <th>normal (time)</th><th>slack</th></tr>{led_rows}</table>"""

    # -- strip: control-point times ------------------------------------------
    import numpy as np
    P = np.asarray(rc["P"])
    times = P[:, -1]
    gaps = np.diff(times)
    on_floor = int(np.sum(np.abs(gaps - args.min_dt) < 1e-6))
    floor_note = (
        f'<b style="color:#b00">{on_floor}/{len(gaps)} consecutive gaps sit exactly '
        "on the min_dt floor — the timing profile is an objective artifact, do not "
        "read it as a result</b>" if on_floor else "no gaps on the min_dt floor"
    )
    los_html = ""
    if rc.get("los") is not None:
        badge = ("<b style='color:#b00'>LINE OF SIGHT LOST</b>"
                 if rc["los"]["min"] < 0 else "line of sight held everywhere")
        los_html = (f"<h2>Line-of-sight margin vs time (independent sampling)</h2>"
                    f"{svg_line(rc['los']['m'], color='#2255bb')}"
                    f"<p style='font-size:12px'>min = {rc['los']['min']:.4g} — {badge}.</p>")
    strip = f"""
    {los_html}
    <h2>Control-point time coordinates</h2>
    {svg_line([float(t) for t in times], color="#22aa55")}
    <p style="font-size:12px">t per control point, index order. {floor_note}.
    dim={rc['dim']} (row width read from the data, not assumed).</p>"""

    warn_html = "".join(f'<p style="color:#b00"><b>⚠ {e(w)}</b></p>' for w in warnings)
    return f"""<!doctype html><meta charset="utf-8">
    <title>trace {e(args.scenario)} N{args.N}_seg{args.seg}</title>
    <body style="font-family:sans-serif;max-width:1200px;margin:20px auto">
    <h1>Solver trace — {e(args.scenario)} N{args.N}_seg{args.seg}</h1>
    {hdr}{warn_html}{geo_html}{station_html}{replay_html}{pairs}{timeline}{ledger}{strip}</body>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seg", type=int, default=4)
    ap.add_argument("--elastic-weight", type=float, default=None,
                    help="default: the scenario's measured weight")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--trust-radius", type=float, default=0.5)
    ap.add_argument("--min-dt", type=float, default=0.1)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    payload, trace, warnings = run_solve(args)
    rc = recompute(payload, args.seg)
    ident = build_identity()
    page = render(args, payload, trace, warnings, rc, ident,
                  render_geometry(geometry_data(payload, rc)),
                  render_replay(payload, rc),
                  render_station_view(payload, rc))

    out = args.out or REPO / "figures" / "debug" / (
        f"trace_{args.scenario}_N{args.N}_seg{args.seg}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    print(f"  reported cert {payload['info'].get('koz_violation_reference')} "
          f"vs recomputed {rc['certificate']:.6g}")
    print(f"  reported clearance {payload['info'].get('min_clearance')} "
          f"vs independent {rc['clearance']:.6g}")
    if warnings:
        print("  WARNINGS:", *warnings, sep="\n    ")
    if not args.no_open:
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
