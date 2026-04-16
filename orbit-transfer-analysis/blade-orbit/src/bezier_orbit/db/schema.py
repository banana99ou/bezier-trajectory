"""DuckDB 테이블 스키마 정의.

테이블:
- simulations: 실행 메타데이터
- trajectories: 궤적 데이터
- scp_iterations: SCP 수렴 이력
- outer_loop_sweeps: Outer loop t_f 탐색 이력
- param_sweep: 파라미터 스윕 결과
- blade_simulations: BLADE 궤도전이 SCP 결과
- blade_validations: BLADE 사후검증 리포트
"""

from __future__ import annotations

SCHEMA_SQL = """
-- 시뮬레이션 메타데이터
CREATE TABLE IF NOT EXISTS simulations (
    sim_id      INTEGER PRIMARY KEY,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 궤도 파라미터 (정규화)
    a0_km       DOUBLE NOT NULL,
    r0_x        DOUBLE, r0_y    DOUBLE, r0_z    DOUBLE,
    v0_x        DOUBLE, v0_y    DOUBLE, v0_z    DOUBLE,
    rf_x        DOUBLE, rf_y    DOUBLE, rf_z    DOUBLE,
    vf_x        DOUBLE, vf_y    DOUBLE, vf_z    DOUBLE,

    -- 비행시간
    t_f         DOUBLE,

    -- Solver 설정
    bezier_N    INTEGER DEFAULT 12,
    u_max       DOUBLE,
    pert_level  INTEGER DEFAULT 0,
    max_iter    INTEGER DEFAULT 20,

    -- 정규화 기준량 (물리 단위 복원용)
    DU_km       DOUBLE NOT NULL,
    TU_s        DOUBLE NOT NULL,
    VU_kms      DOUBLE NOT NULL,
    mu_km3s2    DOUBLE DEFAULT 398600.4418,

    -- 결과 요약
    cost        DOUBLE,
    converged   BOOLEAN,
    n_iter      INTEGER,
    bc_violation DOUBLE,

    -- 드리프트 설정
    drift_method    VARCHAR,          -- "rk4" | "affine" | "bernstein"
    drift_K         INTEGER,          -- bernstein_K 또는 affine_K
    drift_R         INTEGER,          -- bernstein_R (차수 축소 목표)
    n_gravity_iter  INTEGER,          -- 자기일관 보정 반복
    use_coupling    BOOLEAN,          -- 커플링 행렬 사용 여부

    -- 솔버 상세
    solver_used     VARCHAR,          -- "SCS" | "CLARABEL"
    convergence_reason VARCHAR,       -- "ctrl" | "cost" | "max_iter"
    solve_time_s    DOUBLE,           -- 총 소요 시간 [초]

    -- SCP 파라미터
    tol_ctrl        DOUBLE,
    tol_bc          DOUBLE,
    trust_region    DOUBLE,           -- 초기 trust region
    relax_alpha     DOUBLE,

    -- 추력 제약 설정
    thrust_grid_M   INTEGER,

    -- 경로 제약
    r_min           DOUBLE,
    r_max           DOUBLE,
    path_K_subdiv   INTEGER,

    -- Level 1/2 섭동 파라미터
    max_jn_degree   INTEGER,
    Cd_A_over_m     DOUBLE,
    Cr_A_over_m     DOUBLE,
    mu_sun_star     DOUBLE,
    mu_moon_star    DOUBLE
);

-- 궤적 데이터
CREATE TABLE IF NOT EXISTS trajectories (
    traj_id     INTEGER PRIMARY KEY,
    sim_id      INTEGER REFERENCES simulations(sim_id),

    -- 정규화 시간 τ 및 상태
    tau         DOUBLE[],
    rx          DOUBLE[], ry    DOUBLE[], rz    DOUBLE[],
    vx          DOUBLE[], vy    DOUBLE[], vz    DOUBLE[],

    -- 추진 가속도
    ux          DOUBLE[], uy    DOUBLE[], uz    DOUBLE[],

    -- 베지어 제어점 Z = t_f · P_u
    Z_flat      DOUBLE[],     -- vec(Z), length = 3*(N+1)
    P_u_flat    DOUBLE[]      -- vec(P_u), length = 3*(N+1)
);

-- SCP 수렴 이력
CREATE TABLE IF NOT EXISTS scp_iterations (
    sim_id      INTEGER REFERENCES simulations(sim_id),
    iteration   INTEGER,
    cost        DOUBLE,
    ctrl_change DOUBLE,
    bc_violation DOUBLE,
    trust_radius DOUBLE,

    PRIMARY KEY (sim_id, iteration)
);

-- Outer loop t_f 탐색 이력
CREATE TABLE IF NOT EXISTS outer_loop_sweeps (
    sweep_id    INTEGER PRIMARY KEY,
    sim_id      INTEGER REFERENCES simulations(sim_id),

    search_method   VARCHAR,          -- "grid" | "golden_section" | "free_time"
    t_f_min         DOUBLE,
    t_f_max         DOUBLE,
    n_evaluations   INTEGER,

    -- 각 t_f 평가 결과 (배열)
    t_f_values      DOUBLE[],
    costs           DOUBLE[],
    converged_flags BOOLEAN[],
    n_iters         DOUBLE[],
    bc_violations   DOUBLE[]
);

-- 파라미터 스윕 결과
CREATE TABLE IF NOT EXISTS param_sweep (
    sweep_id    INTEGER PRIMARY KEY,
    sim_id      INTEGER REFERENCES simulations(sim_id),

    param_name  VARCHAR,
    param_value DOUBLE,
    cost        DOUBLE,
    converged   BOOLEAN,
    n_iter      INTEGER,
    bc_violation DOUBLE,
    solve_time_s DOUBLE
);

-- BLADE 궤도전이 SCP 결과
CREATE TABLE IF NOT EXISTS blade_simulations (
    blade_id    INTEGER PRIMARY KEY,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 궤도 파라미터 (물리 단위)
    dep_a       DOUBLE, dep_e   DOUBLE, dep_inc DOUBLE,
    dep_raan    DOUBLE, dep_aop DOUBLE, dep_ta  DOUBLE,
    arr_a       DOUBLE, arr_e   DOUBLE, arr_inc DOUBLE,
    arr_raan    DOUBLE, arr_aop DOUBLE, arr_ta  DOUBLE,

    -- 비행시간 & 설정
    t_f         DOUBLE,
    K           INTEGER,
    n           INTEGER,
    u_max       DOUBLE,
    max_iter    INTEGER,
    tol_bc      DOUBLE,
    relax_alpha DOUBLE,
    trust_region DOUBLE,
    l1_lambda   DOUBLE,
    n_steps_per_seg INTEGER,
    coupling_order  INTEGER,
    algebraic_drift BOOLEAN,
    gc_K        INTEGER,
    gc_R        INTEGER,

    -- 정규화 기준량
    DU_km       DOUBLE,
    TU_s        DOUBLE,
    VU_kms      DOUBLE,

    -- 결과 요약
    cost        DOUBLE,
    converged   BOOLEAN,
    n_iter      INTEGER,
    bc_violation DOUBLE,
    bc_violation_r  DOUBLE,
    bc_violation_v  DOUBLE,
    thrust_violation DOUBLE,
    status      VARCHAR,
    ta_opt      DOUBLE,

    -- 수렴 이력 (배열)
    cost_history    DOUBLE[],
    bc_history      DOUBLE[],

    -- 확장 배치 태그 (격자 확장 세트 식별용)
    batch_tag       VARCHAR
);

-- BLADE 사후검증 리포트
CREATE TABLE IF NOT EXISTS blade_validations (
    validation_id   INTEGER PRIMARY KEY,
    blade_id        INTEGER REFERENCES blade_simulations(blade_id),

    bc_violation_rk4 DOUBLE,
    bc_violation_r   DOUBLE,
    bc_violation_v   DOUBLE,
    max_thrust_norm  DOUBLE,
    thrust_violation DOUBLE,
    energy_error     DOUBLE,
    passed           BOOLEAN
);

-- BLADE 실행 설정 (run config)
-- 같은 collocation 케이스를 다른 BLADE 설정으로 반복 실행할 때
-- 각 설정을 고유하게 식별한다.
CREATE TABLE IF NOT EXISTS run_configs (
    config_id   INTEGER PRIMARY KEY,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    name        VARCHAR NOT NULL,           -- 사람이 읽을 수 있는 이름 (예: "baseline_K12n2")
    description VARCHAR,                    -- 설정 설명

    -- BLADE 파라미터
    K           INTEGER NOT NULL,
    n           INTEGER NOT NULL,
    max_iter    INTEGER DEFAULT 50,
    tol_bc      DOUBLE DEFAULT 1e-3,
    relax_alpha DOUBLE DEFAULT 0.5,
    trust_region DOUBLE DEFAULT 5.0,
    l1_lambda   DOUBLE DEFAULT 0.0,
    coupling_order INTEGER DEFAULT 1,
    ta_free     BOOLEAN DEFAULT TRUE,
    algebraic_drift BOOLEAN DEFAULT TRUE,
    n_steps_per_seg INTEGER DEFAULT 30,
    gc_K        INTEGER DEFAULT 8,
    gc_R        INTEGER DEFAULT 12,
    u_max_phys  DOUBLE DEFAULT 0.01,        -- [km/s²]

    UNIQUE(name)
);

-- Collocation 대비 BLADE 비교 결과
CREATE TABLE IF NOT EXISTS comparison_runs (
    cmp_id          INTEGER PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 실행 설정 참조
    config_id       INTEGER REFERENCES run_configs(config_id),

    -- collocation 원본 참조
    colloc_id       INTEGER NOT NULL,       -- orbit-transfer-analysis trajectories.id
    colloc_db_path  VARCHAR,                -- 원본 DB 경로

    -- 5D 구성 공간 (collocation과 동일)
    h0              DOUBLE,                 -- 초기 고도 [km]
    delta_a         DOUBLE,                 -- 장반경 변화 [km]
    delta_i         DOUBLE,                 -- 경사각 변화 [deg]
    T_max_normed    DOUBLE,                 -- 전이시간 / T0
    e0              DOUBLE,                 -- 초기 이심률
    ef              DOUBLE,                 -- 최종 이심률

    -- collocation 원본 결과
    colloc_converged    BOOLEAN,
    colloc_cost         DOUBLE,
    colloc_n_peaks      INTEGER,
    colloc_profile_class INTEGER,           -- 0=unimodal, 1=bimodal, 2=multimodal
    colloc_solve_time   DOUBLE,

    -- BLADE 결과
    blade_id        INTEGER REFERENCES blade_simulations(blade_id),
    blade_converged BOOLEAN,
    blade_cost      DOUBLE,
    blade_n_iter    INTEGER,
    blade_bc_violation  DOUBLE,
    blade_solve_time    DOUBLE,
    blade_n_peaks       INTEGER,
    blade_profile_class INTEGER,
    blade_status        VARCHAR,

    -- (config_id, colloc_id, h0) 조합은 고유
    UNIQUE(config_id, colloc_id, h0)
);

-- 자동 증가 시퀀스
CREATE SEQUENCE IF NOT EXISTS seq_sim_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_traj_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_sweep_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_outer_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_blade_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_val_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_cmp_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_cfg_id START 1;
"""
