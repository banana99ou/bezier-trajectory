# DCM DB Toolchain Guide

## Status

This note records the DB and trajectory-file toolchain that already exists in
this repository.

The goal is simple:

- do not rediscover the DB layout by reading source code again
- remember what is actually stored in DuckDB versus external files
- remember which repo paths already support a DB-seeded DCM-style experiment

## Reasonability Check

Documenting this is reasonable and necessary.

- Logically, the toolchain is non-obvious because the main DB does not store the
  full curve inline.
- Practically, future work will waste time unless the storage pattern and entry
  points are written down.
- Scientifically, the distinction between a raw initial guess and a saved solved
  trajectory matters for how claims must be phrased.

## Main Finding

There are two different DB patterns in `orbit-transfer-analysis`.

### 1. Main `orbit_transfer` DB

Path:

- `orbit-transfer-analysis/src/orbit_transfer/database/schema.py`
- `orbit-transfer-analysis/src/orbit_transfer/database/storage.py`

This is the DB pattern relevant to the DCM-DB experiment discussion.

The main `trajectories` table stores:

- case parameters
- solver result metadata
- a `trajectory_file` path

It does **not** store the full curve inline in the row.

Schema excerpt:

```13:27:orbit-transfer-analysis/src/orbit_transfer/database/schema.py
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

The actual trajectory arrays are saved to a separate `.npz` file:

```214:242:orbit-transfer-analysis/src/orbit_transfer/database/storage.py
    def _save_trajectory_npz_raw(
        self, result_id: int, t, x, u,
    ) -> str:
        """t, x, u 배열을 npz로 저장하고 상대 경로 반환."""
        ...
        filename = f"traj_{result_id:06d}.npz"
        filepath = os.path.join(self.npz_dir, filename)
        np.savez_compressed(filepath, t=t, x=x, u=u)
        return filepath
```

And the load path is explicit:

```248:258:orbit-transfer-analysis/src/orbit_transfer/database/storage.py
    def get_trajectory(self, result_id: int) -> dict:
        """npz에서 궤적 로드. t, x, u를 포함한 dict 반환."""
        row = self.conn.execute(
            "SELECT trajectory_file FROM trajectories WHERE id = ?",
            [result_id],
        ).fetchone()
        ...
        data = np.load(filepath)
        return {"t": data["t"], "x": data["x"], "u": data["u"]}
```

So the storage pattern is:

- one DB row per trajectory result
- one external `.npz` file per row
- the row points to that file through `trajectory_file`

### 2. `blade-orbit` DB

Path:

- `orbit-transfer-analysis/blade-orbit/src/bezier_orbit/db/schema.py`
- `orbit-transfer-analysis/blade-orbit/src/bezier_orbit/db/store.py`

This is a different pattern.

In `blade-orbit`, the trajectory arrays are stored inline in DuckDB array
columns:

```83:99:orbit-transfer-analysis/blade-orbit/src/bezier_orbit/db/schema.py
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
    Z_flat      DOUBLE[],
    P_u_flat    DOUBLE[]
);
```

This is useful to know, but it is not the storage pattern driving the main
DCM-DB question.

## What The Main DB Probably Means

The main table stores result-oriented columns such as:

- `converged`
- `cost`
- `T_f`
- `nu0`
- `nuf`
- `n_peaks`
- `profile_class`

Because of that, the natural interpretation is:

- each row is primarily a **result row**
- `trajectory_file` is most likely the saved trajectory associated with that
  result
- it is **not automatically proven** to be the raw original initial guess

That is the key caveat.

## What We Can Trust

We can trust the following:

- the row and its `trajectory_file` belong to the same case/result
- the `.npz` file stores trajectory arrays `t`, `x`, and `u`
- the main repo already has code to load the row and then load the `.npz`
- this is sufficient to run a matched-case seeded experiment

We cannot yet strictly claim:

- that `trajectory_file` is the raw DB initial guess
- that the stored `.npz` is exactly the original pre-optimization seed artifact

So any experiment using it should be described as:

- matched-case
- matched-seed-proxy
- matched-downstream-solver

unless later evidence proves that `trajectory_file` is the actual raw seed.

## Existing Toolchain We Can Use

### A. DB read + `trajectory_file` load

The repo already supports:

- reading the main `trajectories` table
- resolving `trajectory_file`
- loading `t`, `x`, and `u` from `.npz`

Relevant paths:

- `orbit-transfer-analysis/src/orbit_transfer/database/storage.py`
- `orbit-transfer-analysis/src/orbit_transfer/database/schema.py`

### B. Downstream DCM with an external seed

The repo's collocation wrapper already accepts external trajectory seeds:

```252:267:orbit-transfer-analysis/src/orbit_transfer/benchmark/solvers.py
class CollocationSolver:
    ...
    def __init__(self, x_init=None, u_init=None, t_init=None, l1_lambda=0.0):
        self.x_init = x_init
        self.u_init = u_init
        self.t_init = t_init
        self.l1_lambda = l1_lambda
```

And it passes them into `TwoPassOptimizer`:

```280:288:orbit-transfer-analysis/src/orbit_transfer/benchmark/solvers.py
        opt = TwoPassOptimizer(config, l1_lambda=self.l1_lambda)
        res = opt.solve(
            x_init=self.x_init,
            u_init=self.u_init,
            t_init=self.t_init,
        )
```

And `TwoPassOptimizer` really does use external initial conditions:

```33:46:orbit-transfer-analysis/src/orbit_transfer/optimizer/two_pass.py
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

This is the core reason the DB-seeded DCM experiment is possible at all.

### C. Single-script experiment driver

There is now a single script for this workflow:

- `orbit-transfer-analysis/scripts/run_dcm_db_experiment.py`

This script:

1. selects a converged DB row from `data/trajectories.duckdb`
2. loads the saved trajectory from `trajectory_file`
3. runs a baseline seeded collocation branch
4. runs a proposed branch
5. writes comparison outputs to `results/dcm_db_experiment/`

## Supported Experiment Modes

### Mode 1. `seeded_collocation` (default)

This is the strictest repo-native mode.

Workflow:

- `DB stored trajectory -> seeded collocation upstream -> seeded collocation DCM`

Why this mode exists:

- it preserves the same seed proxy for both branches
- it uses existing repo code only
- it is the closest honest implementation of the professor-note workflow that
  the repo can currently support

### Mode 2. `blade`

Workflow:

- `same case params -> BLADE -> seeded collocation DCM`

This mode is useful, but it is exploratory with respect to the original note,
because the current BLADE wrapper does not accept the external DB seed directly.

## How To Use The Toolchain

### 1. Run one seeded DCM-style experiment

From repo root:

```bash
python orbit-transfer-analysis/scripts/run_dcm_db_experiment.py --case-id 1
```

This writes outputs under:

- `orbit-transfer-analysis/results/dcm_db_experiment/case_000001/`

Expected outputs:

- `summary.json`
- `stage_metrics.csv`
- `db_seed_trajectory.csv`
- `baseline_dcm_trajectory.csv`
- `upstream_optimizer_trajectory.csv`
- `proposed_dcm_trajectory.csv` if upstream converges

### 2. Use deterministic auto-selection instead of explicit case ID

```bash
python orbit-transfer-analysis/scripts/run_dcm_db_experiment.py \
  --selection first-converged
```

or:

```bash
python orbit-transfer-analysis/scripts/run_dcm_db_experiment.py \
  --selection lowest-cost-converged
```

### 3. Run the exploratory BLADE branch

```bash
python orbit-transfer-analysis/scripts/run_dcm_db_experiment.py \
  --case-id 1 \
  --upstream-mode blade
```

## What The Saved Trajectory Files Look Like

The saved `.npz` trajectory files contain:

- `t`
- `x`
- `u`

Observed example in this workspace:

- `orbit-transfer-analysis/data/trajectories/traj_000001.npz`

Observed array shapes:

- `t`: `(15,)`
- `x`: `(6, 15)`
- `u`: `(3, 15)`

That means the file really is storing a concrete sampled trajectory, not just
metadata.

## Safe Interpretation To Reuse Later

If this toolchain is used in analysis or writing, the safe wording is:

The repository contains a DB-plus-trajectory-file toolchain in which the main
DuckDB table stores case/result metadata and a path to an external saved
trajectory file. That saved trajectory can be reused as a seed proxy for matched
downstream collocation reruns. This supports a reproducible matched-case
comparison, but it should not be described as a validated raw-initial-guess
experiment unless the provenance of `trajectory_file` is separately confirmed.

## Practical Bottom Line

The toolchain already exists.

The shortest path to use it is:

1. select a row from `data/trajectories.duckdb`
2. follow `trajectory_file`
3. load `t`, `x`, `u`
4. pass them into the collocation solver as `x_init`, `u_init`, `t_init`
5. compare baseline versus proposed reruns with the same downstream settings

That is the current repo-native way to do the DCM-DB experiment without
rediscovering the storage and seed path from scratch.
