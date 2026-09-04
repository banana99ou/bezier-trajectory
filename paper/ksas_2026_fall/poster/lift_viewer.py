#!/usr/bin/env python3
"""Rotate Fig. 1's two scenes in a browser and read off the camera angle.

Fig. 1 is a 3-D figure printed flat, so the one thing that decides whether it
reads is the view direction, and that is not something to guess. This writes one
self-contained HTML file with the same geometry the figure draws, spins under the
mouse, and prints the matplotlib ``elev`` / ``azim`` of whatever you are looking
at. Drag until it reads, copy the line it shows, and put it in
``make_poster_figures.py``.

Nothing here draws the poster. It shares the geometry helpers with
``make_poster_figures.py`` so the two cannot drift apart.

    python3 paper/ksas_2026_fall/poster/lift_viewer.py && open <the file it names>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from make_poster_figures import (  # noqa: E402  (path set above)
    SIDECAR, SLATE, TUBE, BALL, CTRL, BLUE, DARK, RED_PAPER,
    bezier, loiter_shadow_track, tube_surface,
)

PLOTLY = REPO / "spacetime_bezier" / "static" / "plotly.min.js"


def _surf(X, Y, Z, colour, opacity, name):
    return dict(type="surface", x=X.tolist(), y=Y.tolist(), z=Z.tolist(),
                showscale=False, opacity=opacity, name=name, hoverinfo="name",
                colorscale=[[0, colour], [1, colour]], lighting=dict(ambient=0.75, diffuse=0.5))


def _line(P, colour, width, name, dash=None):
    return dict(type="scatter3d", mode="lines", name=name, hoverinfo="name",
                x=P[:, 0].tolist(), y=P[:, 1].tolist(), z=P[:, 2].tolist(),
                line=dict(color=colour, width=width, **({"dash": dash} if dash else {})))


def _slab(y, x0, x1, t0, t1, colour, opacity, name):
    """One vertical wall of the corridor: constant y, spanning x and t."""
    X = np.array([[x0, x1], [x0, x1]], float)
    Y = np.full((2, 2), y, float)
    Z = np.array([[t0, t0], [t1, t1]], float)
    return _surf(X, Y, Z, colour, opacity, name)


def scene_a():
    """(a) a constant-velocity sphere becomes one static tube."""
    r, v, T = 6.0, 2.0, 30.0
    n = 60
    line = np.column_stack([v * np.linspace(0, T, n), np.zeros(n), np.linspace(0, T, n)])
    data = [_surf(*tube_surface(line, r, n_theta=48, ref=(0.0, 1.0, 0.0)), TUBE, 0.55, "lifted KOZ (tube)"),
            _line(line, SLATE, 4, "centreline")]
    # the sphere itself, at four instants, and its shadow on the t = 0 floor
    from make_poster_figures import sphere
    for t in (0.0, 10.0, 20.0, 30.0):
        xs, ys, zs = sphere(np.array([v * t, 0.0, t]), r)
        data.append(_surf(xs, ys, zs, "#8f99a3", 0.95, f"KOZ at t = {t:.0f} s"))
        th = np.linspace(0, 2 * np.pi, 60)
        data.append(_line(np.column_stack([v * t + r * np.cos(th), r * np.sin(th), np.zeros(60)]),
                          "#aab3bb", 3, f"space footprint, t = {t:.0f} s"))
    return data, dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="t [s]",
                      aspect=dict(x=80, y=32, z=34))


def scene_b(sc, side):
    """(b) the loiter scenario at the corridor altitude, with the corridor as a
    slab through time and the two returned runs inside it."""
    t, xs, ys, spot_r = loiter_shadow_track(sc)
    t_max = 52.0
    keep = t <= t_max
    helix = np.column_stack([xs[keep], ys[keep], t[keep]])
    (xlo, xhi), (ylo, yhi), _ = sc["coord_bounds"]
    data = [_surf(*tube_surface(helix, spot_r, n_theta=48), TUBE, 0.55, "shadow spot's tube"),
            _line(helix, SLATE, 3, "shadow centre"),
            _slab(ylo, xlo, xhi, 0.0, t_max, CTRL, 0.10, "corridor wall y-"),
            _slab(yhi, xlo, xhi, 0.0, t_max, CTRL, 0.10, "corridor wall y+")]
    cp = side["control_points"]
    cb = bezier(np.asarray(cp["baseline"])[:, [0, 1, 3]], 300)
    cc = bezier(np.asarray(cp["constrained"])[:, [0, 1, 3]], 300)
    data.append(_line(cb, DARK, 6, f"baseline, {side['baseline']['arrival_time']:.1f} s", dash="dash"))
    data.append(_line(cc, BLUE, 8, f"proposed, {side['constrained']['arrival_time']:.1f} s"))
    lo, hi = side["baseline"]["los_loss_interval"]
    lost = cb[(cb[:, 2] >= lo) & (cb[:, 2] <= hi)]
    data.append(_line(lost, RED_PAPER, 14, f"link lost, {lo:.2f}-{hi:.2f} s"))
    return data, dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="t [s]",
                      aspect=dict(x=200, y=105, z=t_max * 1.15))


PAGE = """<!doctype html><meta charset="utf-8"><title>Fig. 1 — 시점 고르기</title>
<style>
 body{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;
      margin:0;padding:18px 22px;color:#212f3d;background:#fff}
 h1{font-size:20px;margin:0 0 2px} p.lead{margin:0 0 16px;color:#6b7783}
 .row{display:flex;gap:18px;flex-wrap:wrap}
 .cell{flex:1 1 520px;min-width:420px;border:1px solid #e3e7ea;border-radius:8px;padding:10px 12px}
 h2{font-size:16px;margin:0 0 6px}
 .plot{height:460px}
 .cam{margin-top:8px;font-family:ui-monospace,Menlo,monospace;font-size:14px;
      background:#f5f7f8;border-radius:6px;padding:8px 10px;user-select:all;cursor:text}
 .hint{color:#6b7783;font-size:13px;margin-top:4px}
 b{color:#2874a6}
</style>
<h1>Fig. 1 — 시점 고르기</h1>
<p class="lead">마우스로 돌려서 마음에 드는 각도를 찾은 뒤, 아래 회색 줄을 그대로 복사해서 알려주세요.
그 각도를 그림에 그대로 넣습니다. (드래그 = 회전, 스크롤 = 확대, 오른쪽 드래그 = 이동)</p>
<div class="row">
  <div class="cell"><h2>(a) 등속 구 → 하나의 정적인 관</h2>
    <div id="a" class="plot"></div><div id="camA" class="cam"></div>
    <div class="hint">관이 관으로 보이는 각도. 구 네 개가 따로 놀면 실패입니다.</div></div>
  <div class="cell"><h2>(b) loiter 시나리오 — 회랑, 그림자 관, 두 실행</h2>
    <div id="b" class="plot"></div><div id="camB" class="cam"></div>
    <div class="hint">회랑(파란 벽 두 장) 안에 두 궤적이 들어 있고, 그림자 관이 그 앞을 가로지르는 것이
      보이는 각도.</div></div>
</div>
<script>__PLOTLY__</script>
<script>
const SCENES = __SCENES__;
function layout(s){return {margin:{l:0,r:0,t:0,b:0},paper_bgcolor:"#fff",showlegend:false,
  scene:{xaxis:{title:s.xaxis_title,color:"#212f3d",gridcolor:"#e6e6e6",showbackground:false},
         yaxis:{title:s.yaxis_title,color:"#212f3d",gridcolor:"#e6e6e6",showbackground:false},
         zaxis:{title:s.zaxis_title,color:"#212f3d",gridcolor:"#e6e6e6",showbackground:false},
         aspectmode:"manual",aspectratio:s.aspect}};}
function report(div,out,tag){
  const c=div._fullLayout.scene.camera, e=c.eye, a=div._fullLayout.scene.aspectratio;
  // plotly's eye is in units of the (normalised) aspect box; matplotlib's elev/azim
  // are the same direction read in that box, so the conversion is a direct one.
  const x=e.x*a.x, y=e.y*a.y, z=e.z*a.z, r=Math.hypot(x,y,z);
  const elev=Math.asin(z/r)*180/Math.PI, azim=Math.atan2(y,x)*180/Math.PI;
  out.textContent=`${tag}:  ax.view_init(elev=${elev.toFixed(0)}, azim=${azim.toFixed(0)})`
    +`   |  eye=(${e.x.toFixed(2)}, ${e.y.toFixed(2)}, ${e.z.toFixed(2)})`;
}
for(const [id,tag] of [["a","(a)"],["b","(b)"]]){
  const s=SCENES[id], div=document.getElementById(id);
  Plotly.newPlot(div,s.data,layout(s.scene),{displaylogo:false,responsive:true}).then(()=>{
    const out=document.getElementById("cam"+id.toUpperCase());
    report(div,out,tag); div.on("plotly_relayout",()=>report(div,out,tag));
  });
}
</script>
"""


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar", type=pathlib.Path, default=SIDECAR)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "lift_viewer.html")
    args = ap.parse_args(argv)

    from spacetime_bezier.scenarios import scenario_loiter
    sc = scenario_loiter()
    side = json.loads(args.sidecar.read_text())
    if side["scenario"] != sc["name"]:
        sys.exit(f"sidecar scenario {side['scenario']!r} is not {sc['name']!r}")

    da, sa = scene_a()
    db, sb = scene_b(sc, side)
    scenes = {"a": {"data": da, "scene": sa}, "b": {"data": db, "scene": sb}}
    html = (PAGE.replace("__PLOTLY__", PLOTLY.read_text())
                .replace("__SCENES__", json.dumps(scenes)))
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({args.out.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
