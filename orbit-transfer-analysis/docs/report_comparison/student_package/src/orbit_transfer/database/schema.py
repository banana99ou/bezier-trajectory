"""DuckDB 스키마 정의."""

TRAJECTORY_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY,
    -- 구성 파라미터
    h0 DOUBLE NOT NULL,               -- 초기 고도 [km]
    delta_a DOUBLE NOT NULL,          -- 장반경 변화 [km]
    delta_i DOUBLE NOT NULL,          -- 경사각 변화 [deg]
    T_max_normed DOUBLE NOT NULL,     -- 전이시간 상한/T0
    e0 DOUBLE NOT NULL DEFAULT 0.0,   -- 출발 이심률
    ef DOUBLE NOT NULL DEFAULT 0.0,   -- 도착 이심률
    -- 결과
    converged BOOLEAN NOT NULL,
    cost DOUBLE,                      -- 최적 비용 (L2)
    pass1_cost DOUBLE,                -- Pass 1 비용
    T_f DOUBLE,                       -- 최적화된 전이시간 [s]
    nu0 DOUBLE,                       -- 출발 true anomaly [rad]
    nuf DOUBLE,                       -- 도착 true anomaly [rad]
    n_peaks INTEGER,                  -- 피크 개수
    profile_class INTEGER,            -- 분류 (0/1/2)
    -- 메타데이터
    solve_time DOUBLE,                -- 솔버 소요 시간 [s]
    trajectory_file VARCHAR,          -- npz 파일 경로
    run_id VARCHAR,                   -- 샘플링 실행 ID (예: run_20260215_001)
    param_config VARCHAR,             -- 파라미터 설정 ID (예: T015_5D)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_h0 ON trajectories(h0);
CREATE INDEX IF NOT EXISTS idx_class ON trajectories(profile_class);
CREATE INDEX IF NOT EXISTS idx_converged ON trajectories(converged);
CREATE INDEX IF NOT EXISTS idx_run_id ON trajectories(run_id);
"""

MIGRATE_ADD_RUN_COLUMNS = """
ALTER TABLE trajectories ADD COLUMN IF NOT EXISTS run_id VARCHAR;
ALTER TABLE trajectories ADD COLUMN IF NOT EXISTS param_config VARCHAR;
"""
