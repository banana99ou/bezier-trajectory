"""N3: 드리프트 방법별 SCP 전체 풀이 시간 벤치마크.

시나리오 B (LEO→1.2a₀)에서 각 방법의 SCP 총 시간·비용·수렴을 비교한다.

출력
----
- 콘솔 요약 테이블
- results/n3_scp_benchmark.csv + DuckDB
"""

import time
import warnings
import csv
import os

import numpy as np

from bezier_orbit.normalize import CanonicalUnits
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.drift import DriftConfig
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop
from bezier_orbit.db.store import SimulationStore

warnings.filterwarnings("ignore")

# ── 시나리오 B: LEO → 1.2a₀ ──────────────────────────────────

MU_KM = 398600.4418
A0_KM = 6678.137
CU = CanonicalUnits(a0=A0_KM, mu=MU_KM)
N = 12
T_F = 3.62

r0, v0 = keplerian_to_cartesian(A0_KM, 0.0, 0.0, 0.0, 0.0, 0.0, MU_KM)
rf, vf = keplerian_to_cartesian(A0_KM * 1.2, 0.0, 0.0, 0.0, 0.0, np.pi, MU_KM)

RESULTS_DIR = "scripts/results"
DB_PATH = os.path.join(RESULTS_DIR, "simulations.duckdb")


def make(cfg, **kw):
    defaults = dict(
        r0=r0 / CU.DU, v0=v0 / CU.VU,
        rf=rf / CU.DU, vf=vf / CU.VU,
        t_f=T_F, N=N, canonical_units=CU,
        max_iter=50, tol_ctrl=1e-3, tol_bc=0.05,
        drift_config=cfg,
    )
    defaults.update(kw)
    return SCPProblem(**defaults)


CONFIGS = {
    "RK4": DriftConfig(method="rk4"),
    "Affine(K=16,alg)": DriftConfig(method="affine", affine_K=16,
                                     gravity_correction="algebraic"),
    "Affine(K=16,num)": DriftConfig(method="affine", affine_K=16,
                                     gravity_correction="numerical"),
    "Bern(K=8,alg)": DriftConfig(method="bernstein", bernstein_K=8,
                                  gravity_correction="algebraic"),
    "Bern(K=8,num)": DriftConfig(method="bernstein", bernstein_K=8,
                                  gravity_correction="numerical"),
}

# ── 실행 ──────────────────────────────────────────────────────

print("=" * 72)
print("  SCP 전체 풀이 벤치마크  (시나리오 B: LEO → 1.2a₀, N=12)")
print("=" * 72)

os.makedirs(RESULTS_DIR, exist_ok=True)
store = SimulationStore(DB_PATH)

results = []
for name, cfg in CONFIGS.items():
    prob = make(cfg)

    # 워밍업
    solve_inner_loop(prob)

    # 측정 (3회)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = solve_inner_loop(prob)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    # DB 저장 (마지막 실행)
    sim_id = store.save_simulation(prob, res, CU)

    row = dict(
        method=name,
        time_ms=np.median(times),
        cost=res.cost,
        n_iter=res.n_iter,
        converged=res.converged,
        bc_violation=res.bc_violation,
        sim_id=sim_id,
    )
    results.append(row)
    print(f"  {name:<22s}  완료 ({np.median(times):.0f} ms) [sim_id={sim_id}]")

# ── 결과 출력 ─────────────────────────────────────────────────

rk4_time = results[0]["time_ms"]

print("\n" + "-" * 72)
print(f"  {'방법':<22s}  {'시간(ms)':>10s}  {'배속':>7s}  "
      f"{'비용':>10s}  {'반복':>4s}  {'수렴':>4s}  {'BC위반':>10s}")
print("-" * 72)
for r in results:
    sp = rk4_time / r["time_ms"] if r["time_ms"] > 0 else 0
    cv = "✓" if r["converged"] else "✗"
    print(f"  {r['method']:<22s}  {r['time_ms']:10.1f}  {sp:6.2f}×  "
          f"{r['cost']:10.6f}  {r['n_iter']:4d}  {cv:>4s}  "
          f"{r['bc_violation']:10.2e}")

with open(os.path.join(RESULTS_DIR, "n3_scp_benchmark.csv"), "w", newline="") as f:
    fieldnames = [k for k in results[0].keys() if k != "sim_id"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in results:
        w.writerow({k: r[k] for k in fieldnames})

store.close()
print(f"\n  → {RESULTS_DIR}/n3_scp_benchmark.csv 저장 완료")
print(f"  → DB 저장: {DB_PATH}")
