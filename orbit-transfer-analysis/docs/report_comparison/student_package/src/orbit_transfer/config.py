"""파라미터 범위 및 솔버 설정."""

# --- 전이시간 하한 비율 ---
T_MIN_FACTOR = 0.15   # T_min = 0.15 * T0 (0.1~0.15 구간은 수렴률 14.5%로 비효율)

# --- 파라미터 범위 (Table 1) ---
PARAM_RANGES = {
    "T_max_normed": (0.15, 1.2),   # 전이시간 상한 / T0 (하한 0.15: sub-15% 구간 제거)
    "delta_a": (-500.0, 2000.0),   # 장반경 변화 [km]
    "delta_i": (0.0, 15.0),        # 경사각 변화 [deg]
    "e0": (0.0, 0.1),              # 출발 이심률
    "ef": (0.0, 0.1),              # 도착 이심률
}
# sorted keys: ['T_max_normed', 'delta_a', 'delta_i', 'e0', 'ef']

# 고도 슬라이스
H0_SLICES = [400.0, 600.0, 800.0, 1000.0]  # [km]

# --- Pass 1: Hermite-Simpson ---
HS_NUM_SEGMENTS = 30        # 균일 구간 수 (M=30)

# --- Pass 2: Multi-Phase LGL ---
LGL_NODES_PEAK = 15         # 피크 구간 노드 수
LGL_NODES_COAST = 8         # coasting 구간 노드 수
MIN_PHASE_FRACTION = 0.05   # 최소 phase 길이 비율

# --- IPOPT 옵션 ---
IPOPT_OPTIONS_PASS1 = {
    "ipopt.tol": 1e-4,
    "ipopt.constr_viol_tol": 1e-4,
    "ipopt.max_iter": 500,
    "ipopt.linear_solver": "mumps",
    "ipopt.mu_strategy": "adaptive",
    "ipopt.print_level": 0,
}

IPOPT_OPTIONS_PASS2 = {
    "ipopt.tol": 1e-6,
    "ipopt.constr_viol_tol": 1e-6,
    "ipopt.max_iter": 1000,
    "ipopt.linear_solver": "mumps",
    "ipopt.mu_strategy": "adaptive",
    "ipopt.warm_start_init_point": "yes",
    "ipopt.warm_start_bound_push": 1e-6,
    "ipopt.print_level": 0,
}

# --- 피크 탐지 (Topological Persistence) ---
PEAK_PERSISTENCE_RATIO = 0.10  # 최대값 대비 persistence 임계값
PEAK_INTERP_POINTS = 200  # 피크 탐지용 보간 포인트 수

# --- GP 샘플링 ---
GP_N_INIT = 150             # 초기 LHS 샘플 수 (5D용)
GP_BATCH_SIZE = 15           # 배치 크기 (k)
GP_D_MIN = 0.1              # 최소 분리 거리 (정규화 좌표)
GP_N_MAX = 800              # 최대 샘플 수 (5D용)
GP_ENTROPY_THRESHOLD = 0.3  # 엔트로피 수렴 임계값

# --- NLP 수렴 실패 복구 ---
MAX_NU_RETRIES = 3           # true anomaly 변경 최대 시도 횟수
TOL_RELAXATION_FACTOR = 10.0  # 허용 오차 완화 배수
