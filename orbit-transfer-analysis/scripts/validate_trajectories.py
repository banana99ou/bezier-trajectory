#!/usr/bin/env python3
"""DB 궤적 물리적 정합성 검증 스크립트.

모든 수렴 궤적에 대해 7가지 물리적 정합성을 자동 검증한다:
1. 시간 단조성
2. 추력 상한
3. 최소 고도
4. 초기 궤도 매칭
5. 최종 궤도 매칭
6. 전이시간 범위
7. 동역학 잔차

Usage:
    python scripts/validate_trajectories.py
    python scripts/validate_trajectories.py --db_path data/h400/trajectories.duckdb
    python scripts/validate_trajectories.py --output validation_results.csv --verbose
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbit_transfer.astrodynamics import rv_to_oe
from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.database import TrajectoryDatabase
from orbit_transfer.dynamics import spacecraft_eom_numpy
from orbit_transfer.types import TransferConfig

# 검증 항목 이름 (순서 고정)
CHECK_NAMES = [
    "time_mono",
    "thrust_bound",
    "altitude_bound",
    "init_orbit",
    "final_orbit",
    "transfer_time",
    "dynamics_residual",
]

CHECK_LABELS = {
    "time_mono": "Time monotonicity",
    "thrust_bound": "Thrust bound",
    "altitude_bound": "Altitude bound",
    "init_orbit": "Initial orbit match",
    "final_orbit": "Final orbit match",
    "transfer_time": "Transfer time range",
    "dynamics_residual": "Dynamics residual",
}

# CSV 컬럼 순서
CSV_COLUMNS = [
    "id", "h0", "delta_a", "delta_i", "T_max_normed", "e0", "ef",
    "time_mono_ok", "thrust_bound_ok", "altitude_bound_ok",
    "init_orbit_ok", "final_orbit_ok", "transfer_time_ok", "dynamics_residual_ok",
    "max_thrust", "min_altitude",
    "init_a_err", "init_e_err", "init_i_err",
    "final_a_err", "final_e_err", "final_i_err",
    "max_dyn_residual",
]


@dataclass
class CheckResult:
    """단일 검증 항목 결과."""
    passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """궤적 1개에 대한 전체 검증 결과."""
    record_id: int
    checks: dict  # {check_name: CheckResult}

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks.values())


class TrajectoryValidator:
    """궤적 물리적 정합성 검증기."""

    # 허용 오차
    THRUST_EPS = 1e-6       # 추력 상한 여유 [km/s^2]
    ALTITUDE_EPS = 1.0      # 최소 고도 여유 [km]
    A_TOL = 1.0             # 장반경 허용 오차 [km]
    E_TOL = 1e-3            # 이심률 허용 오차
    I_TOL_DEG = 0.1         # 경사각 허용 오차 [deg]
    TIME_TOL = 1.0          # 시간 일치 허용 오차 [s]
    DYN_RESIDUAL_TOL = 1e-3  # 동역학 잔차 허용 [km/s^2]

    def check_time_monotonicity(self, t: np.ndarray) -> CheckResult:
        """시간 배열 엄격 단조증가 확인."""
        dt = np.diff(t)
        n_violations = int(np.sum(dt <= 0))
        passed = n_violations == 0
        return CheckResult(
            passed=passed,
            details={
                "n_violations": n_violations,
                "min_dt": float(np.min(dt)) if len(dt) > 0 else 0.0,
            },
        )

    def check_thrust_bound(self, u: np.ndarray, u_max: float) -> CheckResult:
        """추력 크기 상한 확인. u: (3, N)."""
        u_norms = np.linalg.norm(u, axis=0)
        max_thrust = float(np.max(u_norms))
        n_violations = int(np.sum(u_norms > u_max + self.THRUST_EPS))
        passed = n_violations == 0
        return CheckResult(
            passed=passed,
            details={
                "max_thrust": max_thrust,
                "u_max": u_max,
                "n_violations": n_violations,
                "violation_ratio": n_violations / len(u_norms),
            },
        )

    def check_altitude_bound(self, x: np.ndarray, h_min: float) -> CheckResult:
        """최소 고도 확인. x: (6, N)."""
        r_norms = np.linalg.norm(x[:3], axis=0)
        altitudes = r_norms - R_E
        min_alt = float(np.min(altitudes))
        n_violations = int(np.sum(altitudes < h_min - self.ALTITUDE_EPS))
        passed = n_violations == 0
        return CheckResult(
            passed=passed,
            details={
                "min_altitude": min_alt,
                "h_min": h_min,
                "n_violations": n_violations,
                "violation_ratio": n_violations / len(altitudes),
            },
        )

    def check_initial_orbit(
        self, x: np.ndarray, config: TransferConfig
    ) -> CheckResult:
        """초기 상태의 궤도요소가 config와 일치하는지 확인."""
        r0 = x[:3, 0]
        v0 = x[3:, 0]
        a, e, i, *_ = rv_to_oe(r0, v0, MU_EARTH)

        a_err = abs(a - config.a0)
        e_err = abs(e - config.e0)
        i_err = abs(np.degrees(i) - np.degrees(config.i0))

        passed = (
            a_err < self.A_TOL
            and e_err < self.E_TOL
            and i_err < self.I_TOL_DEG
        )
        return CheckResult(
            passed=passed,
            details={
                "a_err": a_err,
                "e_err": e_err,
                "i_err_deg": i_err,
                "a_actual": a,
                "a_target": config.a0,
                "e_actual": e,
                "e_target": config.e0,
            },
        )

    def check_final_orbit(
        self, x: np.ndarray, config: TransferConfig
    ) -> CheckResult:
        """최종 상태의 궤도요소가 config와 일치하는지 확인."""
        rf = x[:3, -1]
        vf = x[3:, -1]
        a, e, i, *_ = rv_to_oe(rf, vf, MU_EARTH)

        a_err = abs(a - config.af)
        e_err = abs(e - config.ef)
        i_err = abs(np.degrees(i) - np.degrees(config.if_))

        passed = (
            a_err < self.A_TOL
            and e_err < self.E_TOL
            and i_err < self.I_TOL_DEG
        )
        return CheckResult(
            passed=passed,
            details={
                "a_err": a_err,
                "e_err": e_err,
                "i_err_deg": i_err,
                "a_actual": a,
                "a_target": config.af,
                "e_actual": e,
                "e_target": config.ef,
            },
        )

    def check_transfer_time(
        self, t: np.ndarray, T_f: float, config: TransferConfig
    ) -> CheckResult:
        """전이시간 범위 및 t[-1] ≈ T_f 확인."""
        t_final = float(t[-1])
        in_range = config.T_min <= T_f <= config.T_max
        t_match = abs(t_final - T_f) < self.TIME_TOL
        passed = in_range and t_match
        return CheckResult(
            passed=passed,
            details={
                "T_f": T_f,
                "t_final": t_final,
                "T_min": config.T_min,
                "T_max": config.T_max,
                "in_range": in_range,
                "t_match": t_match,
                "t_diff": abs(t_final - T_f),
            },
        )

    def check_dynamics_residual(
        self, t: np.ndarray, x: np.ndarray, u: np.ndarray
    ) -> CheckResult:
        """수치 미분과 운동방정식 비교로 동역학 잔차 계산."""
        N = len(t)
        if N < 3:
            return CheckResult(passed=True, details={"max_residual": 0.0, "mean_residual": 0.0})

        # 벡터화: 내부 노드 중앙차분
        dt_central = t[2:] - t[:-2]  # (N-2,)
        dx_central = x[:, 2:] - x[:, :-2]  # (6, N-2)
        # dt=0 방어
        dt_safe = np.where(dt_central > 0, dt_central, 1.0)
        dx_dt_inner = dx_central / dt_safe[np.newaxis, :]  # (6, N-2)

        # EoM 평가 (내부 노드만)
        xdot_inner = np.column_stack(
            [spacecraft_eom_numpy(x[:, k], u[:, k]) for k in range(1, N - 1)]
        )  # (6, N-2)

        diff = dx_dt_inner - xdot_inner
        residuals = np.linalg.norm(diff, axis=0)  # (N-2,)
        pos_residuals = np.linalg.norm(diff[:3], axis=0)
        vel_residuals = np.linalg.norm(diff[3:], axis=0)

        max_res = float(np.max(residuals))
        mean_res = float(np.mean(residuals))

        passed = max_res < self.DYN_RESIDUAL_TOL
        return CheckResult(
            passed=passed,
            details={
                "max_residual": max_res,
                "mean_residual": mean_res,
                "max_pos_residual": float(np.max(pos_residuals)),
                "max_vel_residual": float(np.max(vel_residuals)),
            },
        )

    def validate(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        config: TransferConfig,
        T_f: float,
        record_id: int = 0,
    ) -> ValidationResult:
        """궤적에 대한 전체 검증 수행."""
        checks = {
            "time_mono": self.check_time_monotonicity(t),
            "thrust_bound": self.check_thrust_bound(u, config.u_max),
            "altitude_bound": self.check_altitude_bound(x, config.h_min),
            "init_orbit": self.check_initial_orbit(x, config),
            "final_orbit": self.check_final_orbit(x, config),
            "transfer_time": self.check_transfer_time(t, T_f, config),
            "dynamics_residual": self.check_dynamics_residual(t, x, u),
        }
        return ValidationResult(record_id=record_id, checks=checks)


def validate_database(
    db_path: str,
    npz_base: str | None = None,
    converged_only: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """DB 전체 궤적에 대해 검증 수행하고 결과 리스트 반환."""
    db = TrajectoryDatabase(db_path=db_path, npz_dir=npz_base or "data/trajectories")

    # 전체/수렴 레코드 수 조회
    total = db.conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    converged_count = db.conn.execute(
        "SELECT COUNT(*) FROM trajectories WHERE converged = true"
    ).fetchone()[0]

    if converged_only:
        rows = db.get_results(converged=True)
    else:
        rows = db.get_results()

    n = len(rows)
    print(f"\n{'=' * 40}")
    print(f" DB Trajectory Validation")
    print(f"{'=' * 40}")
    print(f"DB: {db_path}")
    print(f"Total records: {total} (converged: {converged_count})")
    print(f"Validating {n} trajectories...\n")

    if n == 0:
        db.close()
        return []

    validator = TrajectoryValidator()
    results = []

    for idx, row in enumerate(rows):
        # 진행률 표시
        if (idx + 1) % 50 == 0 or idx + 1 == n:
            pct = (idx + 1) / n * 100
            bar_len = 30
            filled = int(bar_len * (idx + 1) / n)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r  [{idx + 1:>{len(str(n))}}/{n}] [{bar}] {pct:5.1f}%", end="", flush=True)

        rid = row["id"]

        # NPZ 로드
        try:
            traj = db.get_trajectory(rid)
        except (ValueError, FileNotFoundError, KeyError) as e:
            if verbose:
                print(f"\n  [WARN] ID {rid}: NPZ load failed - {e}")
            entry = _make_fail_entry(row)
            results.append(entry)
            continue

        t = traj["t"]
        x = traj["x"]
        u = traj["u"]

        # Config 재구성
        config = TransferConfig(
            h0=row["h0"],
            delta_a=row["delta_a"],
            delta_i=row["delta_i"],
            T_max_normed=row["T_max_normed"],
            e0=row.get("e0", 0.0),
            ef=row.get("ef", 0.0),
        )

        T_f = row.get("T_f", t[-1])

        # 검증 실행
        vr = validator.validate(t, x, u, config, T_f, record_id=rid)

        # 결과 레코드 구성
        entry = _make_entry(row, vr)
        results.append(entry)

        if verbose and not vr.all_passed:
            failed = [cn for cn in CHECK_NAMES if not vr.checks[cn].passed]
            print(f"\n  [FAIL] ID {rid}: {', '.join(failed)}")
            for cn in failed:
                d = vr.checks[cn].details
                print(f"         {cn}: {d}")

    print()  # 진행률 줄바꿈
    db.close()
    return results


def _make_fail_entry(row: dict) -> dict:
    """NPZ 로드 실패 시 모든 항목 실패로 기록."""
    entry = {
        "id": row["id"],
        "h0": row["h0"],
        "delta_a": row["delta_a"],
        "delta_i": row["delta_i"],
        "T_max_normed": row["T_max_normed"],
        "e0": row.get("e0", 0.0),
        "ef": row.get("ef", 0.0),
    }
    for cn in CHECK_NAMES:
        entry[f"{cn}_ok"] = False
    for col in ["max_thrust", "min_altitude", "init_a_err", "init_e_err",
                "init_i_err", "final_a_err", "final_e_err", "final_i_err",
                "max_dyn_residual"]:
        entry[col] = float("nan")
    return entry


def _make_entry(row: dict, vr: ValidationResult) -> dict:
    """검증 결과를 CSV용 dict로 변환."""
    entry = {
        "id": row["id"],
        "h0": row["h0"],
        "delta_a": row["delta_a"],
        "delta_i": row["delta_i"],
        "T_max_normed": row["T_max_normed"],
        "e0": row.get("e0", 0.0),
        "ef": row.get("ef", 0.0),
    }
    for cn in CHECK_NAMES:
        entry[f"{cn}_ok"] = vr.checks[cn].passed
    entry["max_thrust"] = vr.checks["thrust_bound"].details["max_thrust"]
    entry["min_altitude"] = vr.checks["altitude_bound"].details["min_altitude"]
    entry["init_a_err"] = vr.checks["init_orbit"].details["a_err"]
    entry["init_e_err"] = vr.checks["init_orbit"].details["e_err"]
    entry["init_i_err"] = vr.checks["init_orbit"].details["i_err_deg"]
    entry["final_a_err"] = vr.checks["final_orbit"].details["a_err"]
    entry["final_e_err"] = vr.checks["final_orbit"].details["e_err"]
    entry["final_i_err"] = vr.checks["final_orbit"].details["i_err_deg"]
    entry["max_dyn_residual"] = vr.checks["dynamics_residual"].details.get(
        "max_residual", 0.0
    )
    return entry


def print_summary(results: list[dict]) -> None:
    """검증 결과 요약 출력."""
    n = len(results)
    if n == 0:
        print("\nNo trajectories to validate.")
        return

    print(f"\n{'=' * 50}")
    print(f" Results (N={n})")
    print(f"{'=' * 50}")
    print(f"{'Check':<24s} {'Pass':>6s} {'Fail':>6s} {'Rate':>8s}")
    print("-" * 50)

    ok_cols = [f"{cn}_ok" for cn in CHECK_NAMES]

    for cn in CHECK_NAMES:
        col = f"{cn}_ok"
        n_pass = sum(1 for r in results if r.get(col, False))
        n_fail = n - n_pass
        rate = n_pass / n * 100 if n > 0 else 0.0
        print(f"{CHECK_LABELS[cn]:<24s} {n_pass:>6d} {n_fail:>6d} {rate:>7.1f}%")

    # 전체 통과율
    all_pass = sum(1 for r in results if all(r.get(c, False) for c in ok_cols))
    print("-" * 50)
    print(f"{'All checks passed':<24s} {all_pass:>6d} {n - all_pass:>6d} {all_pass / n * 100:>7.1f}%")

    # 실패 궤적 수치 요약
    n_fail = n - all_pass
    if n_fail > 0:
        fail_entries = [r for r in results if not all(r.get(c, False) for c in ok_cols)]
        numeric_cols = [
            "max_thrust", "min_altitude",
            "init_a_err", "init_e_err", "init_i_err",
            "final_a_err", "final_e_err", "final_i_err",
            "max_dyn_residual",
        ]
        print(f"\n--- Failed trajectory statistics ({n_fail} cases) ---")
        print(f"{'Metric':<20s} {'Min':>12s} {'Max':>12s} {'Mean':>12s}")
        print("-" * 58)
        for col in numeric_cols:
            vals = [r[col] for r in fail_entries if col in r and not np.isnan(r[col])]
            if vals:
                print(
                    f"{col:<20s} {min(vals):>12.6f} {max(vals):>12.6f} "
                    f"{sum(vals)/len(vals):>12.6f}"
                )


def write_csv(results: list[dict], path: str) -> None:
    """결과를 CSV로 저장."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="DB 궤적 물리적 정합성 검증"
    )
    parser.add_argument(
        "--db_path",
        default="data/trajectories_all.duckdb",
        help="DuckDB 파일 경로 (기본: data/trajectories_all.duckdb)",
    )
    parser.add_argument(
        "--npz_base",
        default=None,
        help="NPZ 파일 기본 디렉토리 (기본: DB 내 trajectory_file 경로 사용)",
    )
    parser.add_argument(
        "--no_converged_only",
        action="store_true",
        help="수렴하지 않은 궤적도 포함",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 CSV 출력 경로",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="실패 궤적 상세 출력",
    )
    args = parser.parse_args()

    db_path = args.db_path
    if not Path(db_path).exists():
        print(f"[ERROR] DB file not found: {db_path}")
        sys.exit(1)

    results = validate_database(
        db_path=db_path,
        npz_base=args.npz_base,
        converged_only=not args.no_converged_only,
        verbose=args.verbose,
    )

    print_summary(results)

    ok_cols = [f"{cn}_ok" for cn in CHECK_NAMES]

    if args.output:
        write_csv(results, args.output)
        print(f"\nResults saved to: {args.output}")
    else:
        # 실패 케이스만 자동 저장
        fail_entries = [r for r in results if not all(r.get(c, False) for c in ok_cols)]
        if fail_entries:
            out_path = "validation_failures.csv"
            write_csv(fail_entries, out_path)
            print(f"\nFailed trajectory details saved to: {out_path}")


if __name__ == "__main__":
    main()
