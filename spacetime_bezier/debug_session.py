"""
Stateful live session for stepping through optimizer stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time

from .optimize import create_spacetime_debug_stepper
from .scenarios import SCENARIO_MAP


@dataclass(slots=True)
class SessionConfig:
    """Parameters for starting an optimizer debug session."""

    scenario_name: str
    N: int
    n_seg: int
    max_iter: int = 200
    tol: float = 1e-6
    scp_prox_weight: float = 0.3
    scp_trust_radius: float = 0.0
    min_dt: float = 0.1


class OptimizerDebugSession:
    """Caches live frames so stepping backward is instant."""

    def __init__(self, scenario_map: dict | None = None):
        self.scenario_map = scenario_map or SCENARIO_MAP
        self._catalog = self._build_catalog()
        self._config: SessionConfig | None = None
        self._scenario: dict | None = None
        self._stepper = None
        self._history: list[dict] = []
        self._current_index = -1
        self._last_action_diagnostics: dict = {}

    def _build_catalog(self) -> list[dict]:
        catalog = []
        for name, (scenario_fn, configs) in self.scenario_map.items():
            scenario = scenario_fn()
            catalog.append(
                {
                    "name": name,
                    "title": scenario["title"],
                    "configs": [
                        {
                            "key": f"N{int(N)}_seg{int(n_seg)}",
                            "N": int(N),
                            "n_seg": int(n_seg),
                        }
                        for N, n_seg in configs
                    ],
                }
            )
        return catalog

    def available_scenarios(self) -> dict:
        return {"scenarios": self._catalog}

    def start(self, config: SessionConfig) -> dict:
        started = time.perf_counter()
        scenario_fn, _configs = self.scenario_map[config.scenario_name]
        self._scenario = scenario_fn()
        self._config = config
        self._stepper = create_spacetime_debug_stepper(
            N=config.N,
            dim=len(self._scenario["start"]),
            p_start=self._scenario["start"],
            p_end=self._scenario["end"],
            obstacles=self._scenario["obstacles"],
            n_seg=config.n_seg,
            max_iter=config.max_iter,
            tol=config.tol,
            scp_prox_weight=config.scp_prox_weight,
            scp_trust_radius=config.scp_trust_radius,
            min_dt=config.min_dt,
            init_curve=self._scenario.get("init_curve"),
        )
        self._history = []
        self._current_index = -1
        self._last_action_diagnostics = {
            "action": "start",
            "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "history_reuse": False,
            "frame_compute_ms": 0.0,
            "frame_to_dict_ms": 0.0,
            "frame_json_measure_ms": 0.0,
            "frame_json_bytes": 0,
        }
        return self.get_state()

    def _current_frame(self) -> dict | None:
        if self._current_index < 0 or self._current_index >= len(self._history):
            return None
        return self._history[self._current_index]

    def _history_summary(self) -> list[dict]:
        return [
            {
                "index": idx,
                "frame_id": frame["frame_id"],
                "stage": frame["stage"],
                "label": frame["label"],
                "iteration": frame["iteration"],
            }
            for idx, frame in enumerate(self._history)
        ]

    def get_state(self) -> dict:
        session = {
            "active": self._stepper is not None,
            "frame_index": self._current_index,
            "frame_count": len(self._history),
            "finished": bool(self._stepper.done) if self._stepper is not None else False,
            "can_prev": self._current_index >= 0,
            "can_next": bool(
                self._stepper is not None
                and (self._current_index < len(self._history) - 1 or not self._stepper.done)
            ),
            "history": self._history_summary(),
            "diagnostics": self._last_action_diagnostics,
        }
        if self._config is not None:
            session["config"] = asdict(self._config)
        return {
            "available": self.available_scenarios(),
            "session": session,
            "scenario": None
            if self._scenario is None
            else {
                "name": self._scenario["name"],
                "title": self._scenario["title"],
                "start": self._scenario["start"],
                "end": self._scenario["end"],
                "T": self._scenario["T"],
                "init_curve": self._scenario.get("init_curve"),
                "obstacles": self._scenario["obstacles"],
            },
            "current_frame": self._current_frame(),
        }

    def next(self) -> dict:
        if self._stepper is None:
            raise RuntimeError("Debug session has not been started.")
        started = time.perf_counter()
        if self._current_index < len(self._history) - 1:
            self._current_index += 1
            frame = self._history[self._current_index]
            self._last_action_diagnostics = {
                "action": "next",
                "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "history_reuse": True,
                "frame_compute_ms": 0.0,
                "frame_to_dict_ms": 0.0,
                "frame_json_measure_ms": 0.0,
                "frame_json_bytes": 0,
                "stage": frame["stage"],
            }
            return self.get_state()
        frame_compute_started = time.perf_counter()
        frame = self._stepper.next_frame()
        frame_compute_ms = (time.perf_counter() - frame_compute_started) * 1000.0
        frame_to_dict_ms = 0.0
        frame_json_measure_ms = 0.0
        frame_json_bytes = 0
        if frame is not None:
            frame_to_dict_started = time.perf_counter()
            frame_dict = frame.to_dict()
            frame_to_dict_ms = (time.perf_counter() - frame_to_dict_started) * 1000.0
            json_measure_started = time.perf_counter()
            frame_json_bytes = len(json.dumps(frame_dict).encode("utf-8"))
            frame_json_measure_ms = (time.perf_counter() - json_measure_started) * 1000.0
            self._history.append(frame_dict)
            self._current_index = len(self._history) - 1
            self._last_action_diagnostics = {
                "action": "next",
                "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "history_reuse": False,
                "frame_compute_ms": round(frame_compute_ms, 3),
                "frame_to_dict_ms": round(frame_to_dict_ms, 3),
                "frame_json_measure_ms": round(frame_json_measure_ms, 3),
                "frame_json_bytes": int(frame_json_bytes),
                "stage": frame_dict["stage"],
                "frame_profile": frame_dict.get("profile", {}),
            }
        else:
            self._last_action_diagnostics = {
                "action": "next",
                "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "history_reuse": False,
                "frame_compute_ms": round(frame_compute_ms, 3),
                "frame_to_dict_ms": 0.0,
                "frame_json_measure_ms": 0.0,
                "frame_json_bytes": 0,
                "stage": None,
            }
        return self.get_state()

    def prev(self) -> dict:
        if self._stepper is None:
            raise RuntimeError("Debug session has not been started.")
        started = time.perf_counter()
        self._current_index = max(-1, self._current_index - 1)
        frame = self._current_frame()
        self._last_action_diagnostics = {
            "action": "prev",
            "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "history_reuse": True,
            "frame_compute_ms": 0.0,
            "frame_to_dict_ms": 0.0,
            "frame_json_measure_ms": 0.0,
            "frame_json_bytes": 0,
            "stage": None if frame is None else frame["stage"],
        }
        return self.get_state()

    def next_iteration(self) -> dict:
        if self._stepper is None:
            raise RuntimeError("Debug session has not been started.")
        started = time.perf_counter()
        steps_advanced = 0
        while True:
            state = self.next()
            steps_advanced += 1
            frame = state["current_frame"]
            if frame is None:
                self._last_action_diagnostics = {
                    "action": "next-iteration",
                    "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "steps_advanced": int(steps_advanced),
                    "terminal_stage": None,
                }
                return state
            if frame["stage"] in {"post-eval", "finalize"}:
                self._last_action_diagnostics = {
                    "action": "next-iteration",
                    "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "steps_advanced": int(steps_advanced),
                    "terminal_stage": frame["stage"],
                    "last_step": state["session"]["diagnostics"],
                }
                return state
            if state["session"]["finished"] and not state["session"]["can_next"]:
                self._last_action_diagnostics = {
                    "action": "next-iteration",
                    "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "steps_advanced": int(steps_advanced),
                    "terminal_stage": frame["stage"],
                    "last_step": state["session"]["diagnostics"],
                }
                return state

    def reset(self) -> dict:
        if self._config is None:
            raise RuntimeError("Debug session has not been started.")
        started = time.perf_counter()
        state = self.start(SessionConfig(**asdict(self._config)))
        self._last_action_diagnostics = {
            "action": "reset",
            "action_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "history_reuse": False,
            "frame_compute_ms": 0.0,
            "frame_to_dict_ms": 0.0,
            "frame_json_measure_ms": 0.0,
            "frame_json_bytes": 0,
        }
        return state
