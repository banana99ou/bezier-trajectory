"""DuckDB 기반 궤적 데이터베이스 CRUD."""

import os
from pathlib import Path

import duckdb
import numpy as np

from ..types import TransferConfig, TrajectoryResult
from .schema import CREATE_INDEX_DDL, MIGRATE_ADD_RUN_COLUMNS, TRAJECTORY_TABLE_DDL


class TrajectoryDatabase:
    """DuckDB 기반 궤적 데이터베이스 CRUD."""

    def __init__(
        self,
        db_path: str = "data/trajectories.duckdb",
        npz_dir: str = "data/trajectories",
    ):
        """
        Args:
            db_path: DuckDB 파일 경로
            npz_dir: npz 궤적 파일 저장 디렉토리
        """
        self.db_path = db_path
        self.npz_dir = npz_dir
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        os.makedirs(npz_dir, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """테이블 및 인덱스 생성 (기존 DB 마이그레이션 포함)."""
        self.conn.execute(TRAJECTORY_TABLE_DDL)
        # 기존 DB에 누락된 컬럼 추가 (마이그레이션)
        for statement in MIGRATE_ADD_RUN_COLUMNS.strip().split(";"):
            statement = statement.strip()
            if statement:
                try:
                    self.conn.execute(statement)
                except duckdb.CatalogException:
                    pass
        # DuckDB에서 CREATE INDEX는 한 번에 하나씩 실행
        for statement in CREATE_INDEX_DDL.strip().split(";"):
            statement = statement.strip()
            if statement:
                self.conn.execute(statement)

    def _next_id(self) -> int:
        """다음 ID 반환."""
        result = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM trajectories").fetchone()
        return result[0]

    def insert_result(
        self,
        config: TransferConfig,
        result: TrajectoryResult,
        solve_time: float = 0.0,
        run_id: str = None,
        param_config: str = None,
    ) -> int:
        """결과 삽입 및 궤적 npz 저장. 삽입된 row ID 반환."""
        row_id = self._next_id()

        # npz 파일 저장
        trajectory_file = self.save_trajectory_npz(row_id, result)

        self.conn.execute(
            """
            INSERT INTO trajectories (
                id, h0, delta_a, delta_i, T_max_normed, e0, ef,
                converged, cost, pass1_cost, T_f, nu0, nuf,
                n_peaks, profile_class, solve_time, trajectory_file,
                run_id, param_config
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row_id,
                config.h0,
                config.delta_a,
                config.delta_i,
                config.T_max_normed,
                config.e0,
                config.ef,
                result.converged,
                result.cost,
                result.pass1_cost,
                result.T_f,
                result.nu0,
                result.nuf,
                result.n_peaks,
                result.profile_class,
                solve_time,
                trajectory_file,
                run_id,
                param_config,
            ],
        )
        return row_id

    def save_trajectory_npz(self, result_id: int, result: TrajectoryResult) -> str:
        """궤적 데이터를 npz로 저장하고 상대 경로 반환.

        저장 전 시간 배열의 엄격한 단조증가를 검증하고,
        비단조 구간이 있으면 수정 후 저장한다.
        """
        t = result.t.copy()
        x = result.x.copy()
        u = result.u.copy()

        # 시간 단조성 검증 및 수정
        if len(t) > 1:
            dt = np.diff(t)
            if np.any(dt <= 0):
                # 정렬 + 중복 제거 + 미세 보정
                sort_idx = np.argsort(t, kind='stable')
                t = t[sort_idx]
                x = x[:, sort_idx]
                u = u[:, sort_idx]
                _, unique_idx = np.unique(t, return_index=True)
                t = t[unique_idx]
                x = x[:, unique_idx]
                u = u[:, unique_idx]
                eps = max(np.finfo(float).eps * np.abs(t[-1] - t[0]), 1e-14)
                for i in range(1, len(t)):
                    if t[i] <= t[i - 1]:
                        t[i] = t[i - 1] + eps

        filename = f"traj_{result_id:06d}.npz"
        filepath = os.path.join(self.npz_dir, filename)
        np.savez_compressed(filepath, t=t, x=x, u=u)
        return filepath

    def get_trajectory(self, result_id: int) -> dict:
        """npz에서 궤적 로드. t, x, u를 포함한 dict 반환."""
        row = self.conn.execute(
            "SELECT trajectory_file FROM trajectories WHERE id = ?",
            [result_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"ID {result_id}에 해당하는 결과 없음")
        filepath = row[0]
        data = np.load(filepath)
        return {"t": data["t"], "x": data["x"], "u": data["u"]}

    def get_results(
        self,
        h0: float = None,
        converged: bool = None,
        profile_class: int = None,
        run_id: str = None,
    ) -> list[dict]:
        """필터 조건으로 결과 조회."""
        query = "SELECT * FROM trajectories WHERE 1=1"
        params = []

        if h0 is not None:
            query += " AND h0 = ?"
            params.append(h0)
        if converged is not None:
            query += " AND converged = ?"
            params.append(converged)
        if profile_class is not None:
            query += " AND profile_class = ?"
            params.append(profile_class)
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)

        query += " ORDER BY id"

        result = self.conn.execute(query, params)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def count_by_class(self, h0: float = None) -> dict:
        """프로파일 클래스별 카운트. {class_id: count} 형태 반환."""
        query = "SELECT profile_class, COUNT(*) as cnt FROM trajectories"
        params = []

        if h0 is not None:
            query += " WHERE h0 = ?"
            params.append(h0)

        query += " GROUP BY profile_class ORDER BY profile_class"

        rows = self.conn.execute(query, params).fetchall()
        return {row[0]: row[1] for row in rows}

    def close(self):
        """DB 연결 종료."""
        self.conn.close()
