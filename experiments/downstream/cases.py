"""The frozen case file: population, thresholds, seeds, provenance.

Design reference: `doc/dcm_downstream_experiment_design.md` section 4.1 (two
strata declared in advance) and P1 (the population is declared from parameters
only, never from outcomes).

The rule this module exists to enforce: **no outcome column of the inherited
database is ever read**. Only the parameter vectors are inherited. Every outcome
is re-measured. `build_stratum_a` therefore selects exactly six parameter
columns and nothing else.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "dcm_baseline" / "data" / "trajectories.duckdb"

# The parameter columns inherited from the database. Deliberately excludes
# `converged`, `cost`, `n_peaks`, `profile_class`, `T_f`, and every other
# outcome column -- see P1.
PARAM_COLUMNS = ("h0", "delta_a", "delta_i", "T_normed", "e0", "ef")

# Constraint values the database does not store (design section 2, last row).
# Declared here so a re-run cannot silently inherit a changed default.
DEFAULT_U_MAX = 0.01    # thrust-acceleration cap [km/s^2]
DEFAULT_H_MIN = 150.0   # altitude floor [km]

# Initial altitude for the designed stratum. Fixed at 400 km to stay comparable
# with the inherited stratum, which has 400 km on every row. The other three
# slices declared in the baseline config are future work (design section 9.3).
STRATUM_B_ALTITUDE_KM = 400.0


@dataclass(frozen=True)
class Thresholds:
    """Grader accept/reject thresholds. Frozen before the first run (design 5.3).

    Every one of these is a number the grader compares against; none of them is
    read from a solver.
    """

    # Terminal boundary-condition error on the propagated trajectory.
    pos_err_km: float = 1.0
    vel_err_km_s: float = 1.0e-3
    # Altitude floor, as a violation allowance below `R_E + h_min`.
    altitude_slack_km: float = 1.0
    # Thrust cap, as a relative overshoot allowance above `u_max`.
    thrust_rel_slack: float = 1.0e-3
    # Resolution self-check: agreement required between the grader run and the
    # same grader run at doubled resolution.
    resolution_rel_tol: float = 1.0e-3
    # Cost-reconstruction check: agreement required between the objective the
    # grader recomputes from its own control interpolant and the value the
    # solver reports. A mismatch means the grader is not reading the same
    # control the solver produced, which voids its other verdicts.
    cost_recon_rel_tol: float = 5.0e-2


@dataclass(frozen=True)
class Case:
    """One transfer problem. Immutable once the case file is written."""

    case_id: str
    stratum: str          # "A" (inherited) or "B" (designed)
    h0: float             # initial altitude [km]
    delta_a: float        # semi-major-axis change [km]
    delta_i: float        # inclination change [deg]
    T_normed: float       # transfer-time bound / initial orbital period
    e0: float
    ef: float
    u_max: float = DEFAULT_U_MAX
    h_min: float = DEFAULT_H_MIN
    # Provenance only. Never used to select or weight a case.
    source_db_ids: tuple = field(default_factory=tuple)

    def to_transfer_config(self):
        """Build the baseline's own config object. Imported lazily so the case
        file can be inspected without CasADi installed."""
        from orbit_transfer.types import TransferConfig

        return TransferConfig(
            h0=self.h0,
            delta_a=self.delta_a,
            delta_i=self.delta_i,
            T_max_normed=self.T_normed,
            e0=self.e0,
            ef=self.ef,
            u_max=self.u_max,
            h_min=self.h_min,
        )


def git_provenance() -> dict:
    """Producing commit and dirty-tree flag (design P6)."""

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": status != "" and status != "unknown",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def build_stratum_a(db_path: Path = DEFAULT_DB) -> list[Case]:
    """All distinct parameter vectors in the inherited database.

    Duplicates are collapsed on the exact six-tuple. The database holds 249 rows
    but fewer distinct vectors; the surviving `source_db_ids` records which rows
    collapsed together, so the collapse is auditable rather than asserted.
    """
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cols = ", ".join(PARAM_COLUMNS)
        rows = con.execute(
            f"SELECT id, {cols} FROM trajectories ORDER BY id"
        ).fetchall()
    finally:
        con.close()

    grouped: dict[tuple, list[int]] = {}
    for row in rows:
        db_id, params = int(row[0]), tuple(float(v) for v in row[1:])
        grouped.setdefault(params, []).append(db_id)

    # Sort by the parameter tuple, not by database id, so the case ordering is a
    # function of the parameters alone and survives a database reordering.
    cases = []
    for k, params in enumerate(sorted(grouped)):
        h0, delta_a, delta_i, t_normed, e0, ef = params
        cases.append(Case(
            case_id=f"A{k:04d}",
            stratum="A",
            h0=h0, delta_a=delta_a, delta_i=delta_i,
            T_normed=t_normed, e0=e0, ef=ef,
            source_db_ids=tuple(grouped[params]),
        ))
    return cases


def build_stratum_b(n: int, seed: int) -> list[Case]:
    """Latin-hypercube sample over the ranges the baseline config declares.

    Sampling the baseline's own declared box is what removes the "you picked the
    battlefield" objection (design 4.1). The ranges are read from the baseline
    config at run time rather than copied, so a change there is visible here.
    """
    from orbit_transfer.config import PARAM_RANGES

    keys = ["T_max_normed", "delta_a", "delta_i", "e0", "ef"]
    rng = np.random.default_rng(seed)

    # Standard LHS: one stratified draw per dimension, independently permuted.
    unit = np.empty((n, len(keys)))
    for j in range(len(keys)):
        strata = (np.arange(n) + rng.random(n)) / n
        unit[:, j] = rng.permutation(strata)

    cases = []
    for k in range(n):
        vals = {}
        for j, key in enumerate(keys):
            lo, hi = PARAM_RANGES[key]
            vals[key] = lo + unit[k, j] * (hi - lo)
        cases.append(Case(
            case_id=f"B{k:04d}",
            stratum="B",
            h0=STRATUM_B_ALTITUDE_KM,
            delta_a=vals["delta_a"],
            delta_i=vals["delta_i"],
            T_normed=vals["T_max_normed"],
            e0=vals["e0"],
            ef=vals["ef"],
        ))
    return cases


def freeze(
    out_path: Path,
    n_stratum_b: int = 100,
    seed: int = 20260812,
    db_path: Path = DEFAULT_DB,
    thresholds: Thresholds | None = None,
) -> dict:
    """Write the case file. This is the act that closes the population."""
    thresholds = thresholds or Thresholds()
    a = build_stratum_a(db_path)
    b = build_stratum_b(n_stratum_b, seed)

    payload = {
        "design_doc": "doc/dcm_downstream_experiment_design.md",
        "frozen_by": git_provenance(),
        "seed": seed,
        "thresholds": asdict(thresholds),
        "constraints_declared_here": {
            "u_max_km_s2": DEFAULT_U_MAX,
            "h_min_km": DEFAULT_H_MIN,
            "note": "absent from the database; declared so re-runs cannot "
                    "silently inherit a changed default",
        },
        "stratum_a": {
            "source": str(db_path.relative_to(REPO_ROOT)),
            "columns_read": list(PARAM_COLUMNS),
            "outcome_columns_read": [],
            "n_rows": sum(len(c.source_db_ids) for c in a),
            "n_distinct": len(a),
        },
        "stratum_b": {
            "n": len(b),
            "altitude_km": STRATUM_B_ALTITUDE_KM,
            "sampler": "latin hypercube, one stratified draw per dimension",
        },
        "cases": [asdict(c) for c in (a + b)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
    return payload


def load(path: Path) -> tuple[list[Case], Thresholds, dict]:
    """Read a frozen case file back."""
    payload = json.loads(Path(path).read_text())
    cases = [
        Case(**{**c, "source_db_ids": tuple(c.get("source_db_ids", ()))})
        for c in payload["cases"]
    ]
    return cases, Thresholds(**payload["thresholds"]), payload
