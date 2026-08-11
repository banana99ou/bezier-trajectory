"""
Generate the paper's result tables T2/T3/T4 from the solver, with provenance.

Every number is read from the solver's `info` dict on a fresh cache-off run --
nothing is re-derived here, and nothing is read from the cache. Output goes to
`doc/results/`, which is COMMITTED, and carries the producing commit SHA. That
is deliberate: `artifacts/` is gitignored, so numbers written there cannot be
traced to the code that produced them from a clean checkout.

Run:  .venv/bin/python tools/build_tables.py

Definitions used here, matching paper section 4.1:

  성공 여부   the Proposition 1 certificate holds at the returned iterate
              (final_hull_violation_km <= 1e-6) AND the loop stopped for a
              principled reason (scvx_stop_reason in {1, 4}). This is the
              paper's actual guarantee. The dense min-radius probe
              (`info['feasible']`) is a separate diagnostic and is emitted to
              the CSV as `dense_probe_ok`; where the two disagree the run is
              flagged, because that disagreement is exactly the case the
              certificate exists to adjudicate.
  안전 여유   min_radius - R_KOZ (km)
  제어 비용   mean_control_accel_ms2 (m/s^2), per section 4.1's definition
  계산 시간   MINIMUM of REPEATS solves (s). A single solve is ~0.1 s, so any
              one sample is dominated by scheduler noise. The minimum is the
              standard robust estimator for timings: noise only ever ADDS
              time, so the floor is the reproducible quantity. The median was
              tried first and was NOT stable -- consecutive generations
              disagreed in the second decimal, which would churn the paper on
              every rebuild.

No Delta-v proxy appears anywhere: the solver no longer emits one.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verify import harness_common as H

OUT_DIR = Path(__file__).resolve().parents[1] / "doc" / "results"
SCENARIO = "phase120"
REPEATS = 15

T3_SEGS = (2, 4, 8, 16, 32, 64)     # section 4.2, first experiment (N = 7)
T24_DEGREES = (6, 7, 8)             # section 4.2, second experiment (n_seg = 16)
T24_NSEG = 16
T3_DEGREE = 7

PRINCIPLED_STOPS = (1, 4)
CERT_TOL = 1e-6                     # km, aggregate hull violation
_STOP_NAME = {0: "반복 한도", 1: "merit 감소 연속", 2: "신뢰 구간 붕괴",
              3: "QP 실패", 4: "모형 정지점"}


def _git(*args, default="unknown"):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True, cwd=OUT_DIR.parent.parent).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def measure(degree, n_seg):
    """One configuration. Runtime is a median; everything else comes from info."""
    sc = H.make_scenario(SCENARIO, N=degree)
    times = []
    info = None
    for _ in range(REPEATS):
        _, info = H.run_rust(sc, n_seg=n_seg)
        times.append(float(info["elapsed_time"]))

    hull = float(info["final_hull_violation_km"])
    stop = int(info["scvx_stop_reason"])
    certified = hull <= CERT_TOL and stop in PRINCIPLED_STOPS
    dense_ok = bool(info["feasible"])
    return dict(
        degree=degree, n_ctrl=degree + 1, n_seg=n_seg,
        certified=certified, dense_probe_ok=dense_ok,
        disagree=(certified != dense_ok),
        margin_km=float(info["min_radius"]) - sc["r_e"],
        ctrl_cost_ms2=float(info["mean_control_accel_ms2"]),
        objective=float(info["cost"]),
        runtime_s=min(times),
        iters=int(info["iterations"]),
        stop=stop, stop_name=_STOP_NAME.get(stop, "?"),
        hull_violation_km=hull,
    )


def _ok(r):
    return "성공" if r["certified"] else "실패"


def _rows_t2(rs):
    for r in rs:
        yield (f"| {r['degree']} | {r['n_ctrl']} | {r['n_seg']} | {_ok(r)} | "
               f"{r['margin_km']:.2f} | {r['ctrl_cost_ms2']:.3f} | "
               f"{r['runtime_s']:.3f} | {r['iters']} |")


def _rows_t3(rs):
    for r in rs:
        yield (f"| {r['n_seg']} | {_ok(r)} | {r['margin_km']:.2f} | "
               f"{r['ctrl_cost_ms2']:.3f} | {r['runtime_s']:.3f} | {r['iters']} |")


def _rows_t4(rs):
    for r in rs:
        yield (f"| {r['degree']} | {r['n_ctrl']} | {r['n_seg']} | {_ok(r)} | "
               f"{r['margin_km']:.2f} | {r['ctrl_cost_ms2']:.3f} | {r['runtime_s']:.3f} |")


def main():
    t3 = [measure(T3_DEGREE, n) for n in T3_SEGS]
    t24 = [measure(d, T24_NSEG) for d in T24_DEGREES]
    rows = t3 + t24

    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", default=""))
    sc = H.make_scenario(SCENARIO)

    md = [
        "# 논문 결과 표 (T2 / T3 / T4) — 생성 결과", "",
        "이 파일은 `tools/build_tables.py`가 생성한다. 손으로 고치지 말 것.",
        "표의 모든 수치는 solver가 반환한 값을 그대로 옮긴 것이며, 이 파일에서 다시",
        "계산하는 양은 없다.", "",
        "## 생성 출처", "",
        f"- commit: `{sha}`{'  **(uncommitted changes present)**' if dirty else ''}",
        f"- 시나리오: `{SCENARIO}` · 차수 {T3_DEGREE} 기준 · "
        f"$R_{{\\mathrm{{KOZ}}}}$ = {sc['r_e']:.0f} km · $T$ = {sc['T']:.0f} s",
        f"- 신뢰 구간 초기 크기 {sc['r0']:.0f} km · 허용오차 1e-8 · "
        f"$n_{{\\mathrm{{conv}}}}$ = 3 · $\\mu$ = 1e-2",
        f"- 계산 시간: {REPEATS}회 실행의 최소값 · {platform.machine()} / "
        f"Python {platform.python_version()}",
        "- solver는 결정적이다. 안전 여유·제어 비용·목적함수·반복 횟수·성공 여부는",
        "  재실행해도 모든 자리가 동일하다. **계산 시간은 예외이며 재생성 시 달라진다.**",
        "  잡음은 시간을 늘리기만 하므로 최소값을 쓰지만, 그래도 실행 묶음 사이에서",
        "  약 ±5% 변동한다(측정값). 따라서 재현되는 것은 자릿수가 아니라 경향이며,",
        "  본문이 근거로 삼는 것도 분할 수와 차수에 대한 단조 증가 경향이다.",
        f"- 성공 여부 = 명제 1 인증($h \\le$ {CERT_TOL:g} km) **그리고** "
        f"원칙적 종료(stop_reason ∈ {{1, 4}})", "",
        "## 표 2 [T2]. 대표 설정에서의 결과 요약", "",
        "| 차수 | 제어점 수 | $n_{\\mathrm{seg}}$ | 성공 여부 | 안전 여유 (km) | "
        "제어 비용 (m/s²) | 계산 시간 (s) | 반복 횟수 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_rows_t2(t24), "",
        f"## 표 3 [T3]. 분할 수에 대한 비교 실험 결과 ($N={T3_DEGREE}$)", "",
        "| $n_{\\mathrm{seg}}$ | 성공 여부 | 안전 여유 (km) | 제어 비용 (m/s²) | "
        "계산 시간 (s) | 반복 횟수 |",
        "|---:|---:|---:|---:|---:|---:|",
        *_rows_t3(t3), "",
        f"## 표 4 [T4]. 차수에 대한 비교 실험 결과 ($n_{{\\mathrm{{seg}}}}={T24_NSEG}$)", "",
        "| 차수 | 제어점 수 | $n_{\\mathrm{seg}}$ | 성공 여부 | 안전 여유 (km) | "
        "제어 비용 (m/s²) | 계산 시간 (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *_rows_t4(t24), "",
        "## 표에 싣지 않은 진단값", "",
        "| 차수 | $n_{\\mathrm{seg}}$ | 목적함수 $J$ | 종료 사유 | "
        "$h$ (km) | 조밀 표본 판정 |", "|---:|---:|---:|:--|---:|:--|",
    ]
    for r in rows:
        flag = "  ⚠ 인증과 불일치" if r["disagree"] else ""
        md.append(f"| {r['degree']} | {r['n_seg']} | {r['objective']:.6e} | "
                  f"{r['stop_name']} ({r['stop']}) | {r['hull_violation_km']:.2e} | "
                  f"{'통과' if r['dense_probe_ok'] else '실패'}{flag} |")

    disagreements = [r for r in rows if r["disagree"]]
    md += ["", "### 인증과 조밀 표본 판정이 갈리는 사례", ""]
    if disagreements:
        for r in disagreements:
            md.append(
                f"- 차수 {r['degree']}, $n_{{\\mathrm{{seg}}}}$ = {r['n_seg']}: "
                f"조밀 표본으로는 KOZ를 침범하지 않으나 명제 1의 인증은 성립하지 "
                f"않는다($h$ = {r['hull_violation_km']:.2e} km, 종료 사유 "
                f"{r['stop_name']}). 본 논문이 보장하는 것은 인증이므로 실패로 적는다.")
    else:
        md.append("- 없음.")
    md.append("")

    H.write_text(OUT_DIR / "paper_tables.md", "\n".join(md))
    H.write_csv(OUT_DIR / "paper_tables.csv", [
        {k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()}
        for r in rows])
    print("\n".join(md))
    print(f"\nwrote {OUT_DIR/'paper_tables.md'} and .csv")
    return rows


if __name__ == "__main__":
    main()
