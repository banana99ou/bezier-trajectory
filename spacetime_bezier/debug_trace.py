"""
Shared schema helpers for live optimizer debug frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint


def to_serializable(value):
    """Convert numpy/scipy objects into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return to_serializable(value.tolist())
    if isinstance(value, np.generic):
        return to_serializable(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Bounds):
        return {
            "type": "bounds",
            "lower": to_serializable(np.asarray(value.lb, dtype=float)),
            "upper": to_serializable(np.asarray(value.ub, dtype=float)),
        }
    if isinstance(value, LinearConstraint):
        matrix = np.asarray(value.A, dtype=float)
        return {
            "type": "linear_constraint",
            "matrix": to_serializable(matrix),
            "lower": to_serializable(np.asarray(value.lb, dtype=float)),
            "upper": to_serializable(np.asarray(value.ub, dtype=float)),
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        }
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


@dataclass(slots=True)
class DebugFrame:
    """One steppable optimizer stage."""

    frame_id: int
    stage: str
    label: str
    iteration: int | None
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_id": int(self.frame_id),
            "stage": self.stage,
            "label": self.label,
            "iteration": None if self.iteration is None else int(self.iteration),
            "payload": to_serializable(self.payload),
        }
