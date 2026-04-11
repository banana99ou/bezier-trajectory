from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import time

import dymos as dm
import openmdao
import openmdao.api as om
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from . import constants
from .downstream_collocation import (
    _eci_from_circular_elements,
    _local_frame,
    build_matched_demo_bezier_warm_start,
    build_naive_initial_guess,
    make_demo_problem,
    sample_bezier_state_profile,
)
from .optimization import _accel_total


DIM = 3
NOMINAL_TRANSFER_TIME_S = float(constants.TRANSFER_TIME_S)


@dataclass(frozen=True)
class DymosT6Config:
    transfer_time_s: float = constants.TRANSFER_TIME_S
    num_segments: int = 12
    transcription_order: int = 3
    compressed: bool = True
    optimizer: str = "SLSQP"
    maxiter: int = 300
    tol: float = 1e-6
    simulate_times_per_seg: int = 30


class OrbitalTransferODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("r", shape=(nn, DIM), units="km")
        self.add_input("v", shape=(nn, DIM), units="km/s")
        self.add_input("u", shape=(nn, DIM), units="km/s**2")

        self.add_output("r_dot", shape=(nn, DIM), units="km/s")
        self.add_output("v_dot", shape=(nn, DIM), units="km/s**2")
        self.add_output("koz_margin", shape=(nn,), units="km")
        self.add_output("control_power", shape=(nn, 1), units="km**2/s**4")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs):
        r = np.asarray(inputs["r"], dtype=float)
        v = np.asarray(inputs["v"], dtype=float)
        u = np.asarray(inputs["u"], dtype=float)

        g = np.array(
            [
                _accel_total(
                    r_i,
                    constants.EARTH_MU_SCALED,
                    constants.EARTH_RADIUS_KM,
                    constants.EARTH_J2,
                )
                for r_i in r
            ],
            dtype=float,
        )
        outputs["r_dot"] = v
        outputs["v_dot"] = u + g
        outputs["koz_margin"] = np.linalg.norm(r, axis=1) - constants.KOZ_RADIUS
        outputs["control_power"] = 0.5 * np.sum(u**2, axis=1, keepdims=True)


def dymos_library_versions() -> dict[str, str]:
    return {
        "dymos": dm.__version__,
        "openmdao": openmdao.__version__,
    }


@contextmanager
def temporary_transfer_time(transfer_time_s: float):
    old_value = float(constants.TRANSFER_TIME_S)
    constants.TRANSFER_TIME_S = float(transfer_time_s)
    try:
        yield
    finally:
        constants.TRANSFER_TIME_S = old_value


def make_demo_problem_for_time(transfer_time_s: float):
    base = make_demo_problem()
    return replace(base, transfer_time_s=float(transfer_time_s))


def _retime_control_points_for_boundary_velocities(
    control_points: np.ndarray,
    *,
    transfer_time_s: float,
    v0: np.ndarray,
    v1: np.ndarray,
) -> np.ndarray:
    points = np.array(control_points, dtype=float, copy=True)
    degree = points.shape[0] - 1
    if degree < 1:
        return points
    points[1] = points[0] + (float(transfer_time_s) / degree) * np.asarray(v0, dtype=float)
    points[-2] = points[-1] - (float(transfer_time_s) / degree) * np.asarray(v1, dtype=float)
    return points


def build_matched_demo_bezier_warm_start_for_time(
    transfer_time_s: float,
    degree: int = 7,
    n_seg: int = 16,
    objective_mode: str = "dv",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    if np.isclose(float(transfer_time_s), NOMINAL_TRANSFER_TIME_S):
        return build_matched_demo_bezier_warm_start(
            degree=degree,
            n_seg=n_seg,
            objective_mode=objective_mode,
            max_iter=max_iter,
            tol=tol,
            use_cache=True,
            ignore_existing_cache=False,
        )
    nominal_points, upstream_info = build_matched_demo_bezier_warm_start(
        degree=degree,
        n_seg=n_seg,
        objective_mode=objective_mode,
        max_iter=max_iter,
        tol=tol,
        use_cache=True,
        ignore_existing_cache=False,
    )
    problem = make_demo_problem_for_time(transfer_time_s)
    retimed_points = _retime_control_points_for_boundary_velocities(
        nominal_points,
        transfer_time_s=transfer_time_s,
        v0=problem.v0,
        v1=problem.vf,
    )
    info = dict(upstream_info)
    info["retimed_from_transfer_time_s"] = NOMINAL_TRANSFER_TIME_S
    info["retimed_to_transfer_time_s"] = float(transfer_time_s)
    return retimed_points, info


def warm_start_contract(transfer_time_s: float = constants.TRANSFER_TIME_S) -> dict:
    problem = make_demo_problem_for_time(transfer_time_s)
    control_points, upstream_info = build_matched_demo_bezier_warm_start_for_time(
        transfer_time_s=transfer_time_s
    )
    r_raw, v_raw, a_raw = sample_bezier_state_profile(problem, control_points)
    return {
        "selection_rule": "Rust-backed representative row N=7, n_seg=16 with endpoint velocity boundary conditions enforced",
        "transfer_time_s": float(transfer_time_s),
        "degree": 7,
        "n_seg": 16,
        "objective_mode": "dv",
        "max_iter": 500,
        "tol": 1e-6,
        "upstream_optimizer_info": {
            "iterations": int(upstream_info.get("iterations", -1)),
            "feasible": bool(upstream_info.get("feasible", False)),
            "elapsed_time_s": float(upstream_info.get("elapsed_time", float("nan"))),
            "retimed_from_transfer_time_s": upstream_info.get("retimed_from_transfer_time_s"),
            "retimed_to_transfer_time_s": upstream_info.get("retimed_to_transfer_time_s"),
        },
        "control_points_km": np.asarray(control_points, dtype=float).tolist(),
        "raw_endpoint_position_error_km": {
            "start": float(np.linalg.norm(r_raw[0] - problem.r0)),
            "end": float(np.linalg.norm(r_raw[-1] - problem.rf)),
        },
        "raw_endpoint_velocity_error_km_s": {
            "start": float(np.linalg.norm(v_raw[0] - problem.v0)),
            "end": float(np.linalg.norm(v_raw[-1] - problem.vf)),
        },
        "raw_sampled_states": {
            "r_km": r_raw.tolist(),
            "v_km_s": v_raw.tolist(),
            "a_km_s2": a_raw.tolist(),
        },
    }


def _unpack_guess(x0: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_block = n_nodes * DIM
    r = x0[:n_block].reshape(n_nodes, DIM)
    v = x0[n_block : 2 * n_block].reshape(n_nodes, DIM)
    u = x0[2 * n_block :].reshape(n_nodes, DIM)
    return r, v, u


def _objective_from_profile(t_s: np.ndarray, u_km_s2: np.ndarray) -> float:
    return float(np.trapezoid(0.5 * np.sum(u_km_s2**2, axis=1), t_s))


def _cumulative_objective_profile(t_s: np.ndarray, u_km_s2: np.ndarray) -> np.ndarray:
    values = 0.5 * np.sum(u_km_s2**2, axis=1)
    cumulative = cumulative_trapezoid(values, t_s, initial=0.0)
    return cumulative[:, None]


def build_experiment_spec(config: DymosT6Config) -> dict:
    problem = make_demo_problem_for_time(config.transfer_time_s)
    r0, v0 = _eci_from_circular_elements(
        constants.EARTH_RADIUS_KM + constants.PROGRESS_START_ALTITUDE_KM,
        51.64,
        0.0,
        15.0,
    )
    rf, vf = _eci_from_circular_elements(
        constants.EARTH_RADIUS_KM + constants.ISS_TARGET_ALTITUDE_KM,
        51.64,
        0.0,
        45.0,
    )
    return {
        "framework": {
            "name": "dymos",
            "versions": dymos_library_versions(),
            "transcription": "Radau",
            "num_segments": int(config.num_segments),
            "transcription_order": int(config.transcription_order),
            "compressed": bool(config.compressed),
            "driver": "ScipyOptimizeDriver",
            "optimizer": config.optimizer,
            "maxiter": int(config.maxiter),
            "tol": float(config.tol),
        },
        "dynamics": {
            "state_equations": ["r_dot = v", "v_dot = u + gravity_J2(r)"],
            "gravity_source": "orbital_docking.optimization._accel_total",
            "units": {"r": "km", "v": "km/s", "u": "km/s^2", "time": "s"},
        },
        "scenario": {
            "transfer_time_s": float(config.transfer_time_s),
            "koz_radius_km": float(problem.koz_radius_km),
            "start_position_km": problem.r0.tolist(),
            "start_velocity_km_s": problem.v0.tolist(),
            "goal_position_km": problem.rf.tolist(),
            "goal_velocity_km_s": problem.vf.tolist(),
            "geometry_check": {
                "r0_matches_direct_formula": bool(np.allclose(problem.r0, r0)),
                "v0_matches_direct_formula": bool(np.allclose(problem.v0, v0)),
                "rf_matches_direct_formula": bool(np.allclose(problem.rf, rf)),
                "vf_matches_direct_formula": bool(np.allclose(problem.vf, vf)),
            },
        },
        "comparison_contract": {
            "exact_question": "Does a Bezier-derived warm start improve solve behavior on one frozen Dymos optimal-control problem relative to an honest naive initialization?",
            "identical_between_runs": [
                "Dymos phase definition",
                "transcription and mesh",
                "driver and optimizer settings",
                "objective definition",
                "boundary conditions",
                "path constraints",
                "physics implementation",
            ],
            "different_between_runs": ["initial guess only"],
        },
        "objective": {
            "definition": "minimize final accumulated integral of 0.5 * ||u||^2 dt",
        },
    }


def make_initial_guess_bundle(
    transfer_time_s: float = constants.TRANSFER_TIME_S,
) -> dict:
    problem = make_demo_problem_for_time(transfer_time_s)
    naive_x0 = build_naive_initial_guess(problem)
    naive_r, naive_v, naive_u = _unpack_guess(naive_x0, problem.n_nodes)

    control_points, _info = build_matched_demo_bezier_warm_start_for_time(
        transfer_time_s=transfer_time_s
    )
    warm_r, warm_v, warm_a = sample_bezier_state_profile(problem, control_points)
    warm_u = np.array(
        [
            a_i
            - _accel_total(
                r_i,
                constants.EARTH_MU_SCALED,
                constants.EARTH_RADIUS_KM,
                constants.EARTH_J2,
            )
            for r_i, a_i in zip(warm_r, warm_a)
        ],
        dtype=float,
    )

    t_s = problem.tau_grid * problem.transfer_time_s

    return {
        "time_s": t_s.tolist(),
        "naive": {
            "time_s": t_s.tolist(),
            "r_km": naive_r.tolist(),
            "v_km_s": naive_v.tolist(),
            "u_km_s2": naive_u.tolist(),
            "sanity": {
                "finite": bool(np.all(np.isfinite(naive_x0))),
                "endpoint_position_error_km": {
                    "start": float(np.linalg.norm(naive_r[0] - problem.r0)),
                    "end": float(np.linalg.norm(naive_r[-1] - problem.rf)),
                },
                "endpoint_velocity_error_km_s": {
                    "start": float(np.linalg.norm(naive_v[0] - problem.v0)),
                    "end": float(np.linalg.norm(naive_v[-1] - problem.vf)),
                },
                "min_node_margin_km": float(np.min(np.linalg.norm(naive_r, axis=1) - problem.koz_radius_km)),
                "initial_objective": _objective_from_profile(t_s, naive_u),
            },
        },
        "warm_start": {
            "time_s": t_s.tolist(),
            "r_km": warm_r.tolist(),
            "v_km_s": warm_v.tolist(),
            "u_km_s2": warm_u.tolist(),
            "sanity": {
                "finite": bool(np.all(np.isfinite(warm_r)) and np.all(np.isfinite(warm_v)) and np.all(np.isfinite(warm_u))),
                "endpoint_position_error_km": {
                    "start": float(np.linalg.norm(warm_r[0] - problem.r0)),
                    "end": float(np.linalg.norm(warm_r[-1] - problem.rf)),
                },
                "endpoint_velocity_error_km_s": {
                    "start": float(np.linalg.norm(warm_v[0] - problem.v0)),
                    "end": float(np.linalg.norm(warm_v[-1] - problem.vf)),
                },
                "min_node_margin_km": float(np.min(np.linalg.norm(warm_r, axis=1) - problem.koz_radius_km)),
                "initial_objective": _objective_from_profile(t_s, warm_u),
            },
        },
    }


def build_problem(config: DymosT6Config, case_recorder_path: Path | None = None):
    prob = om.Problem(model=om.Group())
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = config.optimizer
    prob.driver.options["tol"] = float(config.tol)
    prob.driver.options["maxiter"] = int(config.maxiter)
    prob.driver.options["disp"] = False

    if case_recorder_path is not None:
        recorder = om.SqliteRecorder(str(case_recorder_path))
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options["record_desvars"] = True
        prob.driver.recording_options["record_objectives"] = True
        prob.driver.recording_options["record_constraints"] = True

    traj = dm.Trajectory()
    phase = dm.Phase(
        ode_class=OrbitalTransferODE,
        transcription=dm.Radau(
            num_segments=config.num_segments,
            order=config.transcription_order,
            compressed=config.compressed,
        ),
    )
    traj.add_phase("phase0", phase)
    prob.model.add_subsystem("traj", traj)

    phase.set_time_options(
        initial_val=0.0,
        duration_val=float(config.transfer_time_s),
        fix_initial=True,
        fix_duration=True,
        units="s",
    )
    phase.add_state("r", shape=(DIM,), rate_source="r_dot", units="km", fix_initial=True, fix_final=True)
    phase.add_state("v", shape=(DIM,), rate_source="v_dot", units="km/s", fix_initial=True, fix_final=True)
    phase.add_state(
        "J",
        shape=(1,),
        rate_source="control_power",
        units="km**2/s**3",
        fix_initial=True,
        fix_final=False,
        lower=0.0,
    )
    phase.add_control("u", shape=(DIM,), units="km/s**2", continuity=True, rate_continuity=True)
    phase.add_path_constraint("koz_margin", lower=0.0, units="km")
    phase.add_objective("J", loc="final")
    phase.add_timeseries_output("koz_margin")
    phase.add_timeseries_output("control_power")

    prob.setup()
    return prob, traj, phase


def _set_guess(prob, phase, guess: dict) -> None:
    t_s = np.array(guess["time_s"], dtype=float)
    naive_r = np.array(guess["r_km"], dtype=float)
    naive_v = np.array(guess["v_km_s"], dtype=float)
    naive_u = np.array(guess["u_km_s2"], dtype=float)
    J = _cumulative_objective_profile(t_s, naive_u)

    phase.set_time_val(initial=0.0, duration=float(t_s[-1]))
    phase.set_state_val("r", phase.interp("r", xs=t_s, ys=naive_r))
    phase.set_state_val("v", phase.interp("v", xs=t_s, ys=naive_v))
    phase.set_state_val("J", phase.interp("J", xs=t_s, ys=J))
    phase.set_control_val("u", phase.interp("u", xs=t_s, ys=naive_u))


def _get_val_any(prob_like, names: list[str]) -> np.ndarray:
    for name in names:
        try:
            return np.asarray(prob_like.get_val(name))
        except Exception:
            continue
    raise KeyError(f"None of the variable names were found: {names}")


def _extract_timeseries(prob) -> dict:
    return {
        "time_s": _get_val_any(prob, ["traj.phases.phase0.timeseries.time", "traj.phase0.timeseries.time"]).reshape(-1).tolist(),
        "r_km": _get_val_any(prob, ["traj.phases.phase0.timeseries.r", "traj.phase0.timeseries.r"]).tolist(),
        "v_km_s": _get_val_any(prob, ["traj.phases.phase0.timeseries.v", "traj.phase0.timeseries.v"]).tolist(),
        "u_km_s2": _get_val_any(prob, ["traj.phases.phase0.timeseries.u", "traj.phase0.timeseries.u", "traj.phase0.timeseries.controls:u"]).tolist(),
        "J": _get_val_any(prob, ["traj.phases.phase0.timeseries.J", "traj.phase0.timeseries.J"]).reshape(-1).tolist(),
        "koz_margin_km": _get_val_any(prob, ["traj.phases.phase0.timeseries.koz_margin", "traj.phase0.timeseries.koz_margin"]).reshape(-1).tolist(),
        "control_power": _get_val_any(prob, ["traj.phases.phase0.timeseries.control_power", "traj.phase0.timeseries.control_power"]).reshape(-1).tolist(),
    }


def _solve_ivp_replay(times: np.ndarray, controls: np.ndarray, y0: np.ndarray):
    unique_times, unique_idx = np.unique(times, return_index=True)
    unique_controls = controls[unique_idx]

    def rhs(t, y):
        r = y[:DIM]
        v = y[DIM:]
        u = np.array(
            [np.interp(t, unique_times, unique_controls[:, i]) for i in range(DIM)],
            dtype=float,
        )
        g = _accel_total(
            r,
            constants.EARTH_MU_SCALED,
            constants.EARTH_RADIUS_KM,
            constants.EARTH_J2,
        )
        return np.concatenate([v, u + g])

    return solve_ivp(
        rhs,
        (float(unique_times[0]), float(unique_times[-1])),
        y0,
        t_eval=unique_times,
        rtol=1e-8,
        atol=1e-10,
    )


def _velocity_audit(r: np.ndarray, v: np.ndarray, problem) -> dict:
    plane_normal = np.cross(problem.r0, problem.v0)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    radii = np.linalg.norm(r, axis=1)
    speed = np.linalg.norm(v, axis=1)
    v_circ = np.sqrt(constants.EARTH_MU_SCALED / radii)
    radial = []
    prograde = []
    angle = []
    for r_i, v_i, speed_i in zip(r, v, speed):
        r_hat, t_hat = _local_frame(r_i, plane_normal)
        radial.append(float(v_i @ r_hat))
        prograde.append(float(v_i @ t_hat))
        if speed_i <= 1e-12:
            angle.append(180.0)
        else:
            angle.append(
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(float(np.dot(v_i / speed_i, t_hat)), -1.0, 1.0)
                        )
                    )
                )
            )
    return {
        "speed_km_s": speed.tolist(),
        "circular_speed_km_s": v_circ.tolist(),
        "radial_velocity_km_s": radial,
        "prograde_velocity_km_s": prograde,
        "angle_to_prograde_deg": angle,
        "summary": {
            "min_speed_ratio": float(np.min(speed / v_circ)),
            "retrograde_node_count": int(np.sum(np.array(prograde) < 0.0)),
            "max_angle_to_prograde_deg": float(np.max(angle)),
        },
    }


def _run_single_case(label: str, guess: dict, config: DymosT6Config, artifact_dir: Path) -> dict:
    recorder_path = artifact_dir / f"t6_dymos_{label}.sql"
    prob, traj, phase = build_problem(config=config, case_recorder_path=recorder_path)
    _set_guess(prob, phase, guess)
    t0 = time.time()
    prob.run_driver()
    runtime_s = time.time() - t0

    problem = make_demo_problem_for_time(config.transfer_time_s)
    collocation = _extract_timeseries(prob)
    simulation = traj.simulate(times_per_seg=int(config.simulate_times_per_seg))
    sim_ts = {
        "time_s": _get_val_any(simulation, ["traj.phases.phase0.timeseries.time", "traj.phase0.timeseries.time"]).reshape(-1).tolist(),
        "r_km": _get_val_any(simulation, ["traj.phases.phase0.timeseries.r", "traj.phase0.timeseries.r"]).tolist(),
        "v_km_s": _get_val_any(simulation, ["traj.phases.phase0.timeseries.v", "traj.phase0.timeseries.v"]).tolist(),
        "u_km_s2": _get_val_any(simulation, ["traj.phases.phase0.timeseries.u", "traj.phase0.timeseries.u", "traj.phase0.timeseries.controls:u"]).tolist(),
        "koz_margin_km": _get_val_any(simulation, ["traj.phases.phase0.timeseries.koz_margin", "traj.phase0.timeseries.koz_margin"]).reshape(-1).tolist(),
    }

    t = np.array(collocation["time_s"], dtype=float)
    r = np.array(collocation["r_km"], dtype=float)
    v = np.array(collocation["v_km_s"], dtype=float)
    u = np.array(collocation["u_km_s2"], dtype=float)
    J = np.array(collocation["J"], dtype=float)
    objective_recomputed = _objective_from_profile(t, u)
    objective_recorded = float(J[-1])

    replay = _solve_ivp_replay(t, u, np.concatenate([r[0], v[0]]))
    replay_r = replay.y[:DIM].T
    replay_v = replay.y[DIM:].T
    replay_margin = np.linalg.norm(replay_r, axis=1) - problem.koz_radius_km

    return {
        "label": label,
        "case_recorder": str(recorder_path),
        "driver": {
            "success": bool(prob.driver.result.success),
            "iter_count": int(getattr(prob.driver.result, "iter_count", -1)),
            "runtime_s": float(runtime_s),
            "exit_status": str(getattr(prob.driver.result, "exit_status", "")),
        },
        "boundary_check": {
            "start_position_error_km": float(np.linalg.norm(r[0] - problem.r0)),
            "end_position_error_km": float(np.linalg.norm(r[-1] - problem.rf)),
            "start_velocity_error_km_s": float(np.linalg.norm(v[0] - problem.v0)),
            "end_velocity_error_km_s": float(np.linalg.norm(v[-1] - problem.vf)),
        },
        "objective_check": {
            "recorded_final_J": objective_recorded,
            "recomputed_trapezoid_J": objective_recomputed,
            "abs_difference": float(abs(objective_recorded - objective_recomputed)),
        },
        "path_check": {
            "collocation_min_koz_margin_km": float(np.min(np.array(collocation["koz_margin_km"], dtype=float))),
            "simulate_min_koz_margin_km": float(np.min(np.array(sim_ts["koz_margin_km"], dtype=float))),
            "solve_ivp_min_koz_margin_km": float(np.min(replay_margin)),
        },
        "timeseries": {
            "collocation": collocation,
            "simulate": sim_ts,
            "solve_ivp_replay": {
                "time_s": replay.t.tolist(),
                "r_km": replay_r.tolist(),
                "v_km_s": replay_v.tolist(),
                "koz_margin_km": replay_margin.tolist(),
            },
        },
        "velocity_audit": {
            "collocation": _velocity_audit(r, v, problem),
            "simulate": _velocity_audit(
                np.array(sim_ts["r_km"], dtype=float),
                np.array(sim_ts["v_km_s"], dtype=float),
                problem,
            ),
        },
    }


def _ode_regression_check() -> dict:
    sample_r = np.array(
        [
            [constants.CHASER_RADIUS, 0.0, 0.0],
            [0.0, constants.ISS_RADIUS, 10.0],
            [5000.0, 4200.0, 2100.0],
        ],
        dtype=float,
    )
    sample_v = np.zeros_like(sample_r)
    sample_u = np.array(
        [
            [0.0, 0.0, 0.0],
            [1e-4, -2e-4, 0.0],
            [-3e-4, 0.0, 2e-4],
        ],
        dtype=float,
    )
    comp = OrbitalTransferODE(num_nodes=len(sample_r))
    comp.setup()
    inputs = {"r": sample_r, "v": sample_v, "u": sample_u}
    outputs = {
        "r_dot": np.zeros_like(sample_r),
        "v_dot": np.zeros_like(sample_r),
        "koz_margin": np.zeros(len(sample_r)),
        "control_power": np.zeros((len(sample_r), 1)),
    }
    comp.compute(inputs, outputs)

    expected_g = np.array(
        [
            _accel_total(
                r_i,
                constants.EARTH_MU_SCALED,
                constants.EARTH_RADIUS_KM,
                constants.EARTH_J2,
            )
            for r_i in sample_r
        ],
        dtype=float,
    )
    expected_v_dot = sample_u + expected_g
    return {
        "max_r_dot_error": float(np.max(np.abs(outputs["r_dot"] - sample_v))),
        "max_v_dot_error": float(np.max(np.abs(outputs["v_dot"] - expected_v_dot))),
        "max_koz_margin_error": float(
            np.max(
                np.abs(outputs["koz_margin"] - (np.linalg.norm(sample_r, axis=1) - constants.KOZ_RADIUS))
            )
        ),
    }


def run_dymos_t6_experiment(
    artifact_dir: Path,
    config: DymosT6Config | None = None,
) -> dict:
    if config is None:
        config = DymosT6Config()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    spec = build_experiment_spec(config)
    init_bundle = make_initial_guess_bundle(transfer_time_s=config.transfer_time_s)
    warm_contract = warm_start_contract(transfer_time_s=config.transfer_time_s)

    naive_result = _run_single_case("naive", init_bundle["naive"], config, artifact_dir)
    warm_result = _run_single_case("warm", init_bundle["warm_start"], config, artifact_dir)
    ode_check = _ode_regression_check()

    validity = {
        "framework_gate": True,
        "physics_gate": bool(ode_check["max_v_dot_error"] <= 1e-12 and ode_check["max_r_dot_error"] <= 1e-12),
        "fairness_gate": True,
        "naive_feasible": bool(naive_result["driver"]["success"]),
        "warm_feasible": bool(warm_result["driver"]["success"]),
        "objective_recompute_gate": bool(
            naive_result["objective_check"]["abs_difference"] <= 1e-8
            and warm_result["objective_check"]["abs_difference"] <= 1e-8
        ),
        "dense_koz_gate": bool(
            naive_result["path_check"]["solve_ivp_min_koz_margin_km"] >= -1e-6
            and warm_result["path_check"]["solve_ivp_min_koz_margin_km"] >= -1e-6
        ),
        "warm_export_gate": bool(
            warm_contract["raw_endpoint_velocity_error_km_s"]["start"] <= 1e-6
            and warm_contract["raw_endpoint_velocity_error_km_s"]["end"] <= 1e-6
        ),
        "naive_init_gate": bool(
            init_bundle["naive"]["sanity"]["finite"]
            and init_bundle["naive"]["sanity"]["endpoint_position_error_km"]["start"] <= 1e-9
            and init_bundle["naive"]["sanity"]["endpoint_velocity_error_km_s"]["start"] <= 1e-9
        ),
        "orbital_credibility_gate": bool(
            naive_result["velocity_audit"]["simulate"]["summary"]["retrograde_node_count"] == 0
            and warm_result["velocity_audit"]["simulate"]["summary"]["retrograde_node_count"] == 0
            and naive_result["velocity_audit"]["simulate"]["summary"]["min_speed_ratio"] >= 0.5
            and warm_result["velocity_audit"]["simulate"]["summary"]["min_speed_ratio"] >= 0.5
        ),
    }
    validity["comparison_valid"] = bool(all(validity.values()))

    summary = {
        "naive": {
            "success": naive_result["driver"]["success"],
            "iter_count": naive_result["driver"]["iter_count"],
            "runtime_s": naive_result["driver"]["runtime_s"],
            "objective": naive_result["objective_check"]["recorded_final_J"],
            "min_sim_koz_margin_km": naive_result["path_check"]["solve_ivp_min_koz_margin_km"],
            "min_speed_ratio": naive_result["velocity_audit"]["simulate"]["summary"]["min_speed_ratio"],
            "retrograde_node_count": naive_result["velocity_audit"]["simulate"]["summary"]["retrograde_node_count"],
        },
        "warm_start": {
            "success": warm_result["driver"]["success"],
            "iter_count": warm_result["driver"]["iter_count"],
            "runtime_s": warm_result["driver"]["runtime_s"],
            "objective": warm_result["objective_check"]["recorded_final_J"],
            "min_sim_koz_margin_km": warm_result["path_check"]["solve_ivp_min_koz_margin_km"],
            "min_speed_ratio": warm_result["velocity_audit"]["simulate"]["summary"]["min_speed_ratio"],
            "retrograde_node_count": warm_result["velocity_audit"]["simulate"]["summary"]["retrograde_node_count"],
        },
    }

    result = {
        "experiment_spec": spec,
        "ode_regression_check": ode_check,
        "warm_start_contract": warm_contract,
        "initial_guesses": init_bundle,
        "runs": {
            "naive": naive_result,
            "warm_start": warm_result,
        },
        "validity_gates": validity,
        "summary": summary,
        "robustness_followup": {
            "purpose": "only run if the single-case experiment is valid but inconclusive",
            "varied_inputs": {
                "phase_lag_deg": [20.0, 30.0, 40.0],
                "transfer_time_s": [1200.0, 1500.0, 1800.0],
            },
            "fixed_inputs": [
                "same plane and Earth constants",
                "same KOZ definition",
                "same Dymos transcription and driver settings",
                "same warm-start construction rule",
                "same naive initializer rule",
            ],
            "reported_summary": [
                "solve-success counts",
                "median paired delta in runtime",
                "median paired delta in iterations",
                "median paired delta in objective",
                "win/loss/tie counts",
            ],
        },
    }

    (artifact_dir / "t6_dymos_experiment.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def validity_memo_markdown(result: dict) -> str:
    gates = result["validity_gates"]
    summary = result["summary"]
    lines = [
        "# T6 Dymos Validity Memo",
        "",
        f"- comparison valid: `{gates['comparison_valid']}`",
        f"- framework gate: `{gates['framework_gate']}`",
        f"- physics gate: `{gates['physics_gate']}`",
        f"- fairness gate: `{gates['fairness_gate']}`",
        f"- objective recompute gate: `{gates['objective_recompute_gate']}`",
        f"- dense KOZ gate: `{gates['dense_koz_gate']}`",
        f"- warm export gate: `{gates['warm_export_gate']}`",
        f"- naive init gate: `{gates['naive_init_gate']}`",
        f"- orbital credibility gate: `{gates['orbital_credibility_gate']}`",
        "",
        "## Run Summary",
        "",
        f"- naive success: `{summary['naive']['success']}`, iterations: `{summary['naive']['iter_count']}`, runtime: `{summary['naive']['runtime_s']:.3f} s`, objective: `{summary['naive']['objective']:.6f}`",
        f"- warm success: `{summary['warm_start']['success']}`, iterations: `{summary['warm_start']['iter_count']}`, runtime: `{summary['warm_start']['runtime_s']:.3f} s`, objective: `{summary['warm_start']['objective']:.6f}`",
        "",
        "## Interpretation Boundary",
        "",
        "- Treat the comparison as trustworthy only if all validity gates pass.",
        "- If any gate fails, the run is an engineering diagnostic and not paper-facing T6 evidence.",
    ]
    return "\n".join(lines) + "\n"


def experiment_artifact_paths(artifact_dir: Path) -> dict[str, str]:
    return {
        "experiment_json": str((artifact_dir / "t6_dymos_experiment.json").resolve()),
        "naive_case_recorder": str((artifact_dir / "t6_dymos_naive.sql").resolve()),
        "warm_case_recorder": str((artifact_dir / "t6_dymos_warm.sql").resolve()),
    }


def serialize_config(config: DymosT6Config) -> dict:
    return asdict(config)
