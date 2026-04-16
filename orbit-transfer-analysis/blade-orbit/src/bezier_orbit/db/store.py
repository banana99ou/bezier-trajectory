"""DuckDB 저장/로드 인터페이스.

시뮬레이션 결과를 DuckDB에 체계적으로 관리.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
from numpy.typing import NDArray

from ..normalize import CanonicalUnits
from ..scp.problem import SCPProblem
from ..scp.inner_loop import SCPResult
from .schema import SCHEMA_SQL


class SimulationStore:
    """DuckDB 기반 시뮬레이션 결과 저장소."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.con = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        """테이블 스키마 초기화."""
        self.con.execute(SCHEMA_SQL)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── 저장 ────────────────────────────────────────────────────
    def save_simulation(
        self,
        prob: SCPProblem,
        result: SCPResult,
        cu: CanonicalUnits,
    ) -> int:
        """시뮬레이션 결과 저장.

        Returns sim_id.
        """
        sim_id = self._next_id("seq_sim_id")

        # DriftConfig 추출
        dc = prob.drift_config
        drift_method = dc.method
        drift_K = dc.bernstein_K if drift_method == "bernstein" else dc.affine_K
        drift_R = dc.bernstein_R if drift_method == "bernstein" else None

        self.con.execute(
            """
            INSERT INTO simulations (
                sim_id, a0_km,
                r0_x, r0_y, r0_z, v0_x, v0_y, v0_z,
                rf_x, rf_y, rf_z, vf_x, vf_y, vf_z,
                t_f, bezier_N, u_max, pert_level, max_iter,
                DU_km, TU_s, VU_kms, mu_km3s2,
                cost, converged, n_iter, bc_violation,
                drift_method, drift_K, drift_R, n_gravity_iter, use_coupling,
                solver_used, convergence_reason, solve_time_s,
                tol_ctrl, tol_bc, trust_region, relax_alpha,
                thrust_grid_M,
                r_min, r_max, path_K_subdiv,
                max_jn_degree, Cd_A_over_m, Cr_A_over_m,
                mu_sun_star, mu_moon_star
            ) VALUES (
                ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?
            )
            """,
            [
                sim_id, cu.a0,
                *prob.r0.tolist(), *prob.v0.tolist(),
                *prob.rf.tolist(), *prob.vf.tolist(),
                prob.t_f, prob.N,
                prob.u_max, prob.perturbation_level, prob.max_iter,
                cu.DU, cu.TU, cu.VU, cu.mu,
                result.cost, result.converged, result.n_iter, result.bc_violation,
                drift_method, drift_K, drift_R, dc.n_gravity_iter, dc.use_coupling,
                result.solver_used, result.convergence_reason, result.solve_time_s,
                prob.tol_ctrl, prob.tol_bc, prob.trust_region, prob.relax_alpha,
                prob.thrust_grid_M,
                prob.r_min, prob.r_max, prob.path_K_subdiv,
                prob.max_jn_degree, prob.Cd_A_over_m, prob.Cr_A_over_m,
                prob.mu_sun_star, prob.mu_moon_star,
            ],
        )

        # SCP 반복 이력
        for i in range(len(result.cost_history)):
            cost_i = result.cost_history[i]
            ctrl_i = result.ctrl_change_history[i] if i < len(result.ctrl_change_history) else None
            bc_i = result.bc_violation_history[i] if i < len(result.bc_violation_history) else None
            tr_i = result.trust_radius_history[i] if i < len(result.trust_radius_history) else None
            self.con.execute(
                """INSERT INTO scp_iterations
                   (sim_id, iteration, cost, ctrl_change, bc_violation, trust_radius)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [sim_id, i + 1, cost_i, ctrl_i, bc_i, tr_i],
            )

        return sim_id

    def save_trajectory(
        self,
        sim_id: int,
        tau: NDArray,
        states: NDArray,
        u_arr: NDArray | None = None,
        Z: NDArray | None = None,
        P_u: NDArray | None = None,
    ) -> int:
        """궤적 데이터 저장.

        Parameters
        ----------
        sim_id : int
        tau : (n,)
        states : (n, 6)
        u_arr : (n, 3), optional
        Z : (N+1, 3), optional
        P_u : (N+1, 3), optional
        """
        traj_id = self._next_id("seq_traj_id")

        self.con.execute(
            """
            INSERT INTO trajectories (
                traj_id, sim_id, tau,
                rx, ry, rz, vx, vy, vz,
                ux, uy, uz,
                Z_flat, P_u_flat
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                traj_id, sim_id,
                tau.tolist(),
                states[:, 0].tolist(), states[:, 1].tolist(), states[:, 2].tolist(),
                states[:, 3].tolist(), states[:, 4].tolist(), states[:, 5].tolist(),
                u_arr[:, 0].tolist() if u_arr is not None else None,
                u_arr[:, 1].tolist() if u_arr is not None else None,
                u_arr[:, 2].tolist() if u_arr is not None else None,
                Z.flatten().tolist() if Z is not None else None,
                P_u.flatten().tolist() if P_u is not None else None,
            ],
        )
        return traj_id

    def save_outer_loop(
        self,
        sim_id: int,
        search_method: str,
        t_f_bounds: tuple[float, float],
        outer_result,
    ) -> int:
        """Outer loop t_f 탐색 결과 저장.

        Parameters
        ----------
        sim_id : int
            최적 t_f에 해당하는 simulation ID.
        search_method : str
            "grid" | "golden_section" | "free_time"
        t_f_bounds : (t_f_min, t_f_max)
        outer_result : OuterLoopResult
        """
        sweep_id = self._next_id("seq_outer_id")

        converged_flags = [r.converged for r in outer_result.all_results]
        n_iters = [float(r.n_iter) for r in outer_result.all_results]
        bc_violations = [r.bc_violation for r in outer_result.all_results]

        self.con.execute(
            """
            INSERT INTO outer_loop_sweeps (
                sweep_id, sim_id, search_method, t_f_min, t_f_max,
                n_evaluations, t_f_values, costs,
                converged_flags, n_iters, bc_violations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                sweep_id, sim_id, search_method,
                t_f_bounds[0], t_f_bounds[1],
                len(outer_result.t_f_history),
                outer_result.t_f_history,
                outer_result.cost_history,
                converged_flags,
                n_iters,
                bc_violations,
            ],
        )
        return sweep_id

    def save_param_sweep(
        self,
        sim_id: int,
        param_name: str,
        param_value: float,
        result: SCPResult,
    ) -> int:
        """파라미터 스윕 단건 결과 저장.

        Parameters
        ----------
        sim_id : int
            해당 시뮬레이션 ID.
        param_name : str
            스윕 대상 파라미터 이름 (예: "N", "u_max", "t_f").
        param_value : float
            파라미터 값.
        result : SCPResult
        """
        sweep_id = self._next_id("seq_sweep_id")

        self.con.execute(
            """
            INSERT INTO param_sweep (
                sweep_id, sim_id, param_name, param_value,
                cost, converged, n_iter, bc_violation, solve_time_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                sweep_id, sim_id, param_name, param_value,
                result.cost, result.converged, result.n_iter,
                result.bc_violation, result.solve_time_s,
            ],
        )
        return sweep_id

    def save_blade_simulation(
        self, prob, result, cu: CanonicalUnits | None = None,
        batch_tag: str | None = None,
    ) -> int:
        """BLADE 궤도전이 SCP 결과 저장.

        Parameters
        ----------
        prob : BLADEOrbitProblem
        result : BLADESCPResult
        cu : CanonicalUnits, optional
        batch_tag : str, optional
            확장 배치 식별 태그 (예: "altitude_grid_v1", "omega_ecc_v1").

        Returns blade_id.
        """
        blade_id = self._next_id("seq_blade_id")

        self.con.execute(
            """
            INSERT INTO blade_simulations (
                blade_id,
                dep_a, dep_e, dep_inc, dep_raan, dep_aop, dep_ta,
                arr_a, arr_e, arr_inc, arr_raan, arr_aop, arr_ta,
                t_f, K, n, u_max, max_iter, tol_bc,
                relax_alpha, trust_region, l1_lambda,
                n_steps_per_seg, coupling_order,
                algebraic_drift, gc_K, gc_R,
                DU_km, TU_s, VU_kms,
                cost, converged, n_iter, bc_violation,
                bc_violation_r, bc_violation_v, thrust_violation,
                status, ta_opt,
                cost_history, bc_history,
                batch_tag
            ) VALUES (
                ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?
            )
            """,
            [
                blade_id,
                prob.dep.a, prob.dep.e, prob.dep.inc,
                prob.dep.raan, prob.dep.aop, prob.dep.ta,
                prob.arr.a, prob.arr.e, prob.arr.inc,
                prob.arr.raan, prob.arr.aop, prob.arr.ta,
                prob.t_f, prob.K, prob.n, prob.u_max,
                prob.max_iter, prob.tol_bc,
                prob.relax_alpha, prob.trust_region, prob.l1_lambda,
                prob.n_steps_per_seg, prob.coupling_order,
                bool(prob.algebraic_drift), prob.gc_K, prob.gc_R,
                float(cu.DU) if cu else None, float(cu.TU) if cu else None, float(cu.VU) if cu else None,
                float(result.cost), bool(result.converged), int(result.n_iter), float(result.bc_violation),
                float(result.bc_violation_r) if result.bc_violation_r is not None else None,
                float(result.bc_violation_v) if result.bc_violation_v is not None else None,
                float(result.thrust_violation) if result.thrust_violation is not None else None,
                result.status,
                float(result.ta_opt) if result.ta_opt is not None else None,
                [float(x) for x in result.cost_history] if result.cost_history else [],
                [float(x) for x in result.bc_history] if result.bc_history else [],
                batch_tag,
            ],
        )

        # 사후검증 리포트
        if result.validation is not None:
            self.save_blade_validation(blade_id, result.validation)

        return blade_id

    def save_blade_validation(self, blade_id: int, val) -> int:
        """BLADE 사후검증 리포트 저장.

        Parameters
        ----------
        blade_id : int
        val : BLADEValidation
        """
        val_id = self._next_id("seq_val_id")

        self.con.execute(
            """
            INSERT INTO blade_validations (
                validation_id, blade_id,
                bc_violation_rk4, bc_violation_r, bc_violation_v,
                max_thrust_norm, thrust_violation, energy_error, passed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                val_id, blade_id,
                float(val.bc_violation_rk4), float(val.bc_violation_r), float(val.bc_violation_v),
                float(val.max_thrust_norm), float(val.thrust_violation), float(val.energy_error),
                bool(val.passed),
            ],
        )
        return val_id

    # ── 로드 ────────────────────────────────────────────────────
    def load_simulation(self, sim_id: int) -> dict:
        """시뮬레이션 메타데이터 로드."""
        result = self.con.execute(
            "SELECT * FROM simulations WHERE sim_id = ?", [sim_id]
        ).fetchone()
        if result is None:
            raise ValueError(f"Simulation {sim_id} not found")
        columns = [desc[0] for desc in self.con.description]
        return dict(zip(columns, result))

    def load_trajectory(self, sim_id: int) -> dict | None:
        """궤적 데이터 로드."""
        result = self.con.execute(
            "SELECT * FROM trajectories WHERE sim_id = ?", [sim_id]
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self.con.description]
        data = dict(zip(columns, result))

        # 리스트 → numpy 배열
        for key in ("tau", "rx", "ry", "rz", "vx", "vy", "vz", "ux", "uy", "uz"):
            if data.get(key) is not None:
                data[key] = np.array(data[key])

        for key in ("Z_flat", "P_u_flat"):
            if data.get(key) is not None:
                data[key] = np.array(data[key])

        return data

    def load_scp_history(self, sim_id: int) -> list[dict]:
        """SCP 반복 이력 로드."""
        results = self.con.execute(
            "SELECT * FROM scp_iterations WHERE sim_id = ? ORDER BY iteration",
            [sim_id],
        ).fetchall()
        columns = [desc[0] for desc in self.con.description]
        return [dict(zip(columns, row)) for row in results]

    def load_outer_loop(self, sim_id: int) -> dict | None:
        """Outer loop 탐색 이력 로드."""
        result = self.con.execute(
            "SELECT * FROM outer_loop_sweeps WHERE sim_id = ?", [sim_id]
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self.con.description]
        return dict(zip(columns, result))

    def load_blade_simulation(self, blade_id: int) -> dict:
        """BLADE 시뮬레이션 결과 로드."""
        result = self.con.execute(
            "SELECT * FROM blade_simulations WHERE blade_id = ?", [blade_id]
        ).fetchone()
        if result is None:
            raise ValueError(f"BLADE simulation {blade_id} not found")
        columns = [desc[0] for desc in self.con.description]
        return dict(zip(columns, result))

    def load_blade_validation(self, blade_id: int) -> dict | None:
        """BLADE 사후검증 리포트 로드."""
        result = self.con.execute(
            "SELECT * FROM blade_validations WHERE blade_id = ?", [blade_id]
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self.con.description]
        return dict(zip(columns, result))

    def list_simulations(self) -> list[dict]:
        """전체 시뮬레이션 목록."""
        results = self.con.execute(
            "SELECT sim_id, created_at, a0_km, t_f, cost, converged, n_iter, "
            "drift_method, solver_used, solve_time_s "
            "FROM simulations ORDER BY sim_id"
        ).fetchall()
        columns = [desc[0] for desc in self.con.description]
        return [dict(zip(columns, row)) for row in results]

    def list_blade_simulations(self) -> list[dict]:
        """전체 BLADE 시뮬레이션 목록."""
        results = self.con.execute(
            "SELECT blade_id, created_at, t_f, K, n, cost, converged, n_iter, status "
            "FROM blade_simulations ORDER BY blade_id"
        ).fetchall()
        columns = [desc[0] for desc in self.con.description]
        return [dict(zip(columns, row)) for row in results]

    def get_or_create_config(
        self,
        name: str,
        *,
        K: int = 12,
        n: int = 2,
        max_iter: int = 50,
        tol_bc: float = 1e-3,
        relax_alpha: float = 0.5,
        trust_region: float = 5.0,
        l1_lambda: float = 0.0,
        coupling_order: int = 1,
        ta_free: bool = True,
        algebraic_drift: bool = True,
        n_steps_per_seg: int = 30,
        gc_K: int = 8,
        gc_R: int = 12,
        u_max_phys: float = 0.01,
        description: str = "",
    ) -> int:
        """실행 설정 조회 또는 생성. 이름이 같으면 기존 ID 반환.

        Returns config_id.
        """
        row = self.con.execute(
            "SELECT config_id FROM run_configs WHERE name = ?", [name]
        ).fetchone()
        if row is not None:
            return row[0]

        config_id = self._next_id("seq_cfg_id")
        self.con.execute(
            """
            INSERT INTO run_configs (
                config_id, name, description,
                K, n, max_iter, tol_bc, relax_alpha, trust_region,
                l1_lambda, coupling_order, ta_free, algebraic_drift,
                n_steps_per_seg, gc_K, gc_R, u_max_phys
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                config_id, name, description,
                K, n, max_iter, tol_bc, relax_alpha, trust_region,
                l1_lambda, coupling_order, bool(ta_free), bool(algebraic_drift),
                n_steps_per_seg, gc_K, gc_R, u_max_phys,
            ],
        )
        return config_id

    def load_config(self, config_id: int) -> dict:
        """실행 설정 로드."""
        row = self.con.execute(
            "SELECT * FROM run_configs WHERE config_id = ?", [config_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"Config {config_id} not found")
        columns = [desc[0] for desc in self.con.description]
        return dict(zip(columns, row))

    def list_configs(self) -> list[dict]:
        """전체 실행 설정 목록."""
        rows = self.con.execute(
            "SELECT config_id, name, K, n, max_iter, tol_bc FROM run_configs ORDER BY config_id"
        ).fetchall()
        columns = [desc[0] for desc in self.con.description]
        return [dict(zip(columns, row)) for row in rows]

    def save_comparison(
        self,
        colloc_row: dict,
        blade_id: int,
        blade_result,
        colloc_db_path: str = "",
        blade_n_peaks: int | None = None,
        blade_profile_class: int | None = None,
        config_id: int | None = None,
    ) -> int:
        """Collocation vs BLADE 비교 결과 저장.

        Parameters
        ----------
        colloc_row : dict
            collocation DB에서 읽은 원본 행 (id, h0, delta_a, ...).
        blade_id : int
            blade_simulations 테이블의 ID.
        blade_result : BLADESCPResult
        colloc_db_path : str
        blade_n_peaks : int, optional
        blade_profile_class : int, optional
        config_id : int, optional
            run_configs 테이블의 설정 ID.
        """
        cmp_id = self._next_id("seq_cmp_id")

        self.con.execute(
            """
            INSERT INTO comparison_runs (
                cmp_id, config_id, colloc_id, colloc_db_path,
                h0, delta_a, delta_i, T_max_normed, e0, ef,
                colloc_converged, colloc_cost, colloc_n_peaks,
                colloc_profile_class, colloc_solve_time,
                blade_id, blade_converged, blade_cost,
                blade_n_iter, blade_bc_violation, blade_solve_time,
                blade_n_peaks, blade_profile_class, blade_status
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            """,
            [
                cmp_id, config_id, int(colloc_row["id"]), colloc_db_path,
                float(colloc_row["h0"]), float(colloc_row["delta_a"]),
                float(colloc_row["delta_i"]), float(colloc_row["T_max_normed"]),
                float(colloc_row["e0"]), float(colloc_row["ef"]),
                bool(colloc_row["converged"]),
                float(colloc_row["cost"]) if colloc_row["cost"] is not None else None,
                int(colloc_row["n_peaks"]) if colloc_row["n_peaks"] is not None else None,
                int(colloc_row["profile_class"]) if colloc_row["profile_class"] is not None else None,
                float(colloc_row["solve_time"]) if colloc_row["solve_time"] is not None else None,
                blade_id, bool(blade_result.converged), float(blade_result.cost),
                int(blade_result.n_iter), float(blade_result.bc_violation),
                0.0,  # blade_solve_time — 호출자가 측정
                int(blade_n_peaks) if blade_n_peaks is not None else None,
                int(blade_profile_class) if blade_profile_class is not None else None,
                blade_result.status,
            ],
        )
        return cmp_id

    def update_comparison_solve_time(self, cmp_id: int, solve_time: float):
        """비교 결과의 BLADE 풀이 시간 업데이트."""
        self.con.execute(
            "UPDATE comparison_runs SET blade_solve_time = ? WHERE cmp_id = ?",
            [solve_time, cmp_id],
        )

    def list_comparisons(self, h0: float | None = None) -> list[dict]:
        """비교 결과 목록."""
        query = "SELECT * FROM comparison_runs"
        params = []
        if h0 is not None:
            query += " WHERE h0 = ?"
            params.append(h0)
        query += " ORDER BY cmp_id"
        results = self.con.execute(query, params).fetchall()
        columns = [desc[0] for desc in self.con.description]
        return [dict(zip(columns, row)) for row in results]

    def get_completed_colloc_keys(self, config_id: int | None = None) -> set[tuple[int, float]]:
        """이미 비교 완료된 (colloc_id, h0) 집합.

        Parameters
        ----------
        config_id : int, optional
            특정 설정에 대해서만 조회. None이면 전체.
        """
        if config_id is not None:
            rows = self.con.execute(
                "SELECT colloc_id, h0 FROM comparison_runs WHERE config_id = ?",
                [config_id],
            ).fetchall()
        else:
            rows = self.con.execute(
                "SELECT colloc_id, h0 FROM comparison_runs"
            ).fetchall()
        return {(r[0], r[1]) for r in rows}

    # ── 유틸리티 ────────────────────────────────────────────────
    def _next_id(self, seq_name: str) -> int:
        return self.con.execute(f"SELECT nextval('{seq_name}')").fetchone()[0]
