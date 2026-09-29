"""
The walls' reach, and what the certificate is allowed to claim. Decided 2026-09-29
(SPEC.md sec. 2.7): every keep-out wall is built at the trust region's CAP.

Two separate facts, tested separately:

* The certificate at a trajectory is sound whenever the clip ball covers the segment
  itself. The reach floor guarantees that at ANY build radius. The floor-off
  construction does not, and three recorded lines prove it -- they are the control
  that shows this file can see the hole at all.
* One-step coverage (condition C) needs the walls' reach to cover the largest step
  the trust region allows. The starting radius does not once the trust region
  grows; the cap does by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

bezier_opt = pytest.importorskip("bezier_opt")

from orbital_docking.de_casteljau import segment_matrices_equal_params  # noqa: E402
from spacetime_bezier.families import family_motion  # noqa: E402
from spacetime_bezier.geometry import (  # noqa: E402
    compute_min_clearance,
    normalize_obstacles,
    obstacle_array_bundle,
)
from spacetime_bezier.optimize import optimize_spacetime  # noqa: E402
from spacetime_bezier.scenarios import SCENARIO_MAP  # noqa: E402

CERT = 1e-9

# Straight lines in (x, y, t), degree 3, found 2026-09-29 by throwing 28,000 random
# lines at the three curved tubes below (seed 1). Each PENETRATES its tube and yet
# satisfies every floor-off wall: the documented hole, caught in the act.
FLOOR_OFF_FALSE_CERTIFICATES = [
    ("curve", [[9.13765699190513, 1.1290460432143847, 1.5642259808120027], [7.102364743426482, 2.468199336906916, 2.382993649962715], [5.067072494947833, 3.8073526305994476, 3.201761319113427], [3.0317802464691845, 5.146505924291979, 4.0205289882641395]]),
    ("motion_circle", [[6.189350559117228, 1.808025567426058, 8.516352409403803], [6.354998499049659, 3.1635472232003146, 9.3142836876256], [6.520646438982089, 4.519068878974571, 10.1122149658474], [6.6862943789145195, 5.874590534748828, 10.910146244069198]]),
    ("motion_circle", [[6.199319067538564, 2.1069021206367387, 8.363093906057324], [6.5175880994927065, 3.325396145573625, 8.842575693545015], [6.835857131446848, 4.543890170510512, 9.322057481032704], [7.15412616340099, 5.762384195447399, 9.801539268520395]]),
]


def _scene(name):
    if name == "curve":
        return SCENARIO_MAP["curve"][0]()
    return family_motion(name.removeprefix("motion_"))


def _violation(P, obstacles, n_seg, trust, sound_clip):
    P = np.asarray(P, dtype=float)
    ctrl, radii = obstacle_array_bundle(obstacles, P.shape[1] - 1)
    res = bezier_opt.spacetime_koz_rows_exact(
        p=P, obstacle_ctrl=ctrl, obstacle_r=radii, n_seg=n_seg,
        trust_radius=float(trust), sound_clip=sound_clip, stations=None)
    normals = np.asarray(res[0], dtype=float).reshape(-1, P.shape[1])
    lbs, seg, cp = np.asarray(res[1], dtype=float), np.asarray(res[2]), np.asarray(res[3])
    a_list = segment_matrices_equal_params(P.shape[0] - 1, n_seg)
    total = 0.0
    for n, lb, s, c in zip(normals, lbs, seg, cp):
        q = (np.asarray(a_list[int(s)]) @ P)[int(c)]
        total += max(0.0, float(lb) - float(n @ q))
    return total


@pytest.mark.parametrize("scene,Q", FLOOR_OFF_FALSE_CERTIFICATES)
def test_the_floor_off_certificate_passes_a_penetrating_line(scene, Q):
    """The CONTROL. FAILS IF the floor-off construction stops being fooled by
    these lines -- then this file no longer demonstrates that it can see the
    hole, and new specimens must be found before the next test means anything."""
    obs = _scene(scene)["obstacles"]
    assert compute_min_clearance(np.asarray(Q), obs, dim=3, n_eval=2000) < -1e-6
    assert _violation(Q, obs, 1, 0.5, sound_clip=False) <= CERT


@pytest.mark.parametrize("trust", [0.0, 2.0])
@pytest.mark.parametrize("scene,Q", FLOOR_OFF_FALSE_CERTIFICATES)
def test_the_reach_floor_refuses_every_penetrating_line(scene, Q, trust):
    """The certificate claim. With the floor on, the clip ball is never smaller
    than the segment, so a line that enters the tube must violate a wall --
    at zero trust and at the cap of the default trust region alike.
    FAILS IF the floor stops covering the segment itself."""
    obs = _scene(scene)["obstacles"]
    assert _violation(Q, obs, 1, trust, sound_clip=True) > CERT


def test_walls_at_the_cap_block_a_step_the_starting_radius_misses():
    """Condition C, built by hand. A hovering trajectory, a ball just outside the
    reach at the starting radius, and a step four times that radius toward it --
    a step the trust region allows once it has grown to its cap.

    FAILS IF walls built at the starting radius stop letting that step through
    (then the reason for the cap is gone), or if walls built at the cap let it
    through (then the cap does not do its job)."""
    N, n_seg, t0, T, r = 8, 8, 0.5, 10.0, 0.5
    P = np.stack([[0.0, 0.0, T * i / N] for i in range(N + 1)])
    D = 2.0 + r
    obs = normalize_obstacles([{"pos0": [D / np.sqrt(2), D / np.sqrt(2)], "vel": [0.0, 0.0], "r": r}], T)
    step = P.copy()
    step[1:-1, :2] += 4 * t0
    assert compute_min_clearance(step, obs, dim=3) < 0.0, "the step must actually penetrate"

    def violation_of_step(build_trust):
        ctrl, radii = obstacle_array_bundle(obs, 2)
        res = bezier_opt.spacetime_koz_rows_exact(
            p=P, obstacle_ctrl=ctrl, obstacle_r=radii, n_seg=n_seg,
            trust_radius=build_trust, sound_clip=True, stations=None)
        normals = np.asarray(res[0], dtype=float).reshape(-1, 3)
        a_list = segment_matrices_equal_params(N, n_seg)
        return sum(max(0.0, float(lb) - float(n @ (np.asarray(a_list[int(s)]) @ step)[int(c)]))
                   for n, lb, s, c in zip(normals, res[1], res[2], res[3]))

    assert violation_of_step(t0) <= CERT, "walls at the starting radius should miss this step"
    assert violation_of_step(4 * t0) > CERT, "walls at the cap must block this step"


@pytest.mark.parametrize("scenario,N,n_seg", [("original", 8, 8), ("curve", 8, 16), ("wall", 8, 16)])
def test_the_trust_region_never_outgrows_the_walls(scenario, N, n_seg):
    """The solver-level guard on the policy. FAILS IF the walls' reach is set
    below the trust region's cap -- e.g. back to the starting radius. All three
    runs grow the trust region to its cap under both policies (measured
    2026-09-29), so with walls at the starting radius each one fails the second
    assertion; the first assertion keeps a run that never grows from passing
    vacuously."""
    fn, _ = SCENARIO_MAP[scenario]
    sc = fn()
    _, info = optimize_spacetime(
        N=N, dim=len(sc["start"]), p_start=sc["start"], p_end=sc["end"],
        obstacles=sc["obstacles"], n_seg=n_seg, max_iter=200, tol=1e-6,
        scp_prox_weight=0.3, scp_trust_radius=float(sc.get("trust_radius", 0.5)),
        min_dt=0.1, verbose=False, init_curve=sc.get("init_curve"),
        stations=sc.get("stations"), coord_bounds=sc.get("coord_bounds"))
    grew, reach, start = float(info["max_trust"]), float(info["wall_reach_trust"]), float(info["trust_init"])
    assert grew > start, f"{scenario}: trust never grew, so this run cannot tell start from cap"
    assert grew <= reach * (1 + 1e-12), f"{scenario}: trust reached {grew} but walls cover {reach}"
