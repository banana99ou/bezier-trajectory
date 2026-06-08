# DCM DB Toolchain Guide

## Status

> **Updated after the `orbit-transfer-analysis/` extraction.** The `orbit_transfer` package and its
> case database now live in **`dcm_baseline/`** (install with `pip install -e dcm_baseline`). The
> pieces this guide originally also covered — the `blade-orbit` inline-array DB, the
> `run_dcm_db_experiment.py` single-script driver, and the per-row `.npz` trajectory files — were
> **not** promoted; they are preserved in `~/code/orbit-transfer-archive/` and will leave the repo
> when `orbit-transfer-analysis/` is deleted. This guide now documents only what remains usable in
> the repo.

The goal is unchanged: don't rediscover the DB layout by reading source again — remember what is
stored in DuckDB versus external files, and which entry points support a DB-seeded DCM experiment.

## Main `orbit_transfer` DB

Paths:

- `dcm_baseline/src/orbit_transfer/database/schema.py`
- `dcm_baseline/src/orbit_transfer/database/storage.py`

The case database is `dcm_baseline/data/trajectories.duckdb`. The main `trajectories` table stores
**scalar case parameters and solver-result metadata** (one row per result) plus a `trajectory_file`
path. It does **not** store the full curve inline in the row.

Schema excerpt:

```13:27:dcm_baseline/src/orbit_transfer/database/schema.py
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
```

### Caveat: the `.npz` trajectory files were not promoted

The `trajectory_file` column points at per-row `.npz` files (`t`, `x`, `u` arrays) that lived under
`orbit-transfer-analysis/data/trajectories/`. Those `.npz` files were **not** copied into
`dcm_baseline/` — the live DCM tools read only scalar case rows, not stored curves. Full-curve
loading via `storage.get_trajectory()` is therefore unavailable from `dcm_baseline` unless the
`.npz` set is restored from the archive.

## What the main DB rows mean

The table stores result-oriented columns (`converged`, `cost`, `T_f`, `nu0`, `nuf`, `n_peaks`,
`profile_class`), so each row is primarily a **result row**. Any matched-case experiment drawing
from it should be described as **matched-case / matched-seed-proxy / matched-downstream-solver** —
not as replaying the exact raw pre-optimization seed.

## Seeding the DCM solver externally

`TwoPassOptimizer` accepts external initial conditions, which is what makes a seeded matched-case
experiment possible:

```33:46:dcm_baseline/src/orbit_transfer/optimizer/two_pass.py
    def solve(
        self,
        x_init: np.ndarray | None = None,
        u_init: np.ndarray | None = None,
        t_init: np.ndarray | None = None,
    ) -> TrajectoryResult:
        ...
        x_init : ndarray, optional  외부 초기치 상태 (6, M). None이면 선형 보간 사용
        u_init : ndarray, optional  외부 초기치 제어 (3, M). None이면 zero
        t_init : ndarray, optional  외부 초기치 시간 (M,). 리샘플링에 사용
```

The benchmark wrapper threads those seeds through:

```252:267:dcm_baseline/src/orbit_transfer/benchmark/solvers.py
class CollocationSolver:
    ...
    def __init__(self, x_init=None, u_init=None, t_init=None, l1_lambda=0.0):
        self.x_init = x_init
        self.u_init = u_init
        self.t_init = t_init
        self.l1_lambda = l1_lambda
```

```280:288:dcm_baseline/src/orbit_transfer/benchmark/solvers.py
        opt = TwoPassOptimizer(config, l1_lambda=self.l1_lambda)
        res = opt.solve(
            x_init=self.x_init,
            u_init=self.u_init,
            t_init=self.t_init,
        )
```

The live consumers of this DB today are `tools/dcm_visualize_case.py` and
`tools/dcm_downstream_experiment.py` (design: `doc/dcm_downstream_experiment_design.md`).
