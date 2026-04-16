"""SCP Outer Loop: t_f 탐색 및 보조변수 승강.

이론: docs/reports/007_free_time_lifting/

이중 루프 알고리즘:
- Outer loop: t_f를 업데이트 (격자 탐색 / 황금분할)
- Inner loop: t_f 고정, Z = t_f · P_u에 대한 SCP

복원: P_u* = Z* / t_f
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from .problem import SCPProblem
from .inner_loop import SCPResult, solve_inner_loop


@dataclasses.dataclass
class OuterLoopResult:
    """Outer loop 결과."""

    t_f_opt: float              # 최적 비행시간
    inner_result: SCPResult     # 최적 t_f에서의 inner loop 결과
    t_f_history: list[float]    # 탐색된 t_f 값들
    cost_history: list[float]   # 각 t_f에서의 비용
    all_results: list[SCPResult]  # 모든 inner loop 결과


def grid_search(
    prob: SCPProblem,
    t_f_grid: NDArray,
    *,
    Z_init: NDArray | None = None,
) -> OuterLoopResult:
    """격자 탐색으로 최적 t_f 결정.

    Parameters
    ----------
    prob : SCPProblem
        문제 정의 (prob.t_f는 무시, 격자에서 설정).
    t_f_grid : (M,)
        탐색할 t_f 값 배열.
    Z_init : (N+1, 3), optional
        초기 Z 추정.

    Returns
    -------
    result : OuterLoopResult
    """
    best_cost = float("inf")
    best_idx = 0
    results = []
    cost_list = []

    for i, t_f in enumerate(t_f_grid):
        prob_i = dataclasses.replace(prob, t_f=float(t_f))
        result = solve_inner_loop(prob_i, Z_init=Z_init)
        results.append(result)
        cost_list.append(result.cost)

        if result.converged and result.cost < best_cost:
            best_cost = result.cost
            best_idx = i

    return OuterLoopResult(
        t_f_opt=float(t_f_grid[best_idx]),
        inner_result=results[best_idx],
        t_f_history=list(t_f_grid),
        cost_history=cost_list,
        all_results=results,
    )


def golden_section_search(
    prob: SCPProblem,
    t_f_bounds: tuple[float, float],
    *,
    tol: float = 0.01,
    max_eval: int = 20,
    Z_init: NDArray | None = None,
) -> OuterLoopResult:
    """황금분할 탐색으로 최적 t_f 결정.

    J(t_f)가 단봉(unimodal)이라는 가정 하에 효율적 탐색.

    Parameters
    ----------
    prob : SCPProblem
    t_f_bounds : (t_f_min, t_f_max)
    tol : t_f 수렴 허용치.
    max_eval : 최대 inner loop 호출 수.
    Z_init : 초기 Z.

    Returns
    -------
    result : OuterLoopResult
    """
    gr = (np.sqrt(5.0) + 1.0) / 2.0  # 황금비
    a, b = t_f_bounds

    t_f_history = []
    cost_history = []
    all_results = []

    def evaluate(t_f: float) -> tuple[float, SCPResult]:
        prob_i = dataclasses.replace(prob, t_f=t_f)
        result = solve_inner_loop(prob_i, Z_init=Z_init)
        t_f_history.append(t_f)
        cost_history.append(result.cost)
        all_results.append(result)
        return result.cost, result

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    fc, rc = evaluate(c)
    fd, rd = evaluate(d)

    for _ in range(max_eval - 2):
        if abs(b - a) < tol:
            break

        if fc < fd:
            b = d
            d = c
            fd = fc
            rd = rc
            c = b - (b - a) / gr
            fc, rc = evaluate(c)
        else:
            a = c
            c = d
            fc = fd
            rc = rd
            d = a + (b - a) / gr
            fd, rd = evaluate(d)

    # 최적점 선택
    best_idx = int(np.argmin(cost_history))

    return OuterLoopResult(
        t_f_opt=t_f_history[best_idx],
        inner_result=all_results[best_idx],
        t_f_history=t_f_history,
        cost_history=cost_history,
        all_results=all_results,
    )


def solve_free_time(
    prob: SCPProblem,
    t_f_bounds: tuple[float, float],
    *,
    n_grid: int = 5,
    refine: bool = True,
    refine_tol: float = 0.01,
    Z_init: NDArray | None = None,
) -> OuterLoopResult:
    """자유 종말시간 최적화 (격자 탐색 → 황금분할 정밀화).

    1단계: 격자 탐색으로 전체 형상 파악
    2단계 (선택): 황금분할로 정밀화

    Parameters
    ----------
    prob : SCPProblem
    t_f_bounds : (t_f_min, t_f_max)
    n_grid : 초기 격자 수.
    refine : True이면 황금분할 정밀화 수행.
    refine_tol : 정밀화 수렴 허용치.
    Z_init : 초기 Z.
    """
    # 1단계: 격자 탐색
    t_f_grid = np.linspace(t_f_bounds[0], t_f_bounds[1], n_grid)
    grid_result = grid_search(prob, t_f_grid, Z_init=Z_init)

    if not refine:
        return grid_result

    # 2단계: 황금분할 정밀화
    # 최적 격자점 주변 구간으로 좁혀서 탐색
    best_idx = np.argmin(grid_result.cost_history)
    dt = t_f_grid[1] - t_f_grid[0] if len(t_f_grid) > 1 else 1.0

    a = max(t_f_bounds[0], t_f_grid[best_idx] - dt)
    b = min(t_f_bounds[1], t_f_grid[best_idx] + dt)

    # 격자 최적해를 warm start로 사용
    Z_warm = grid_result.inner_result.Z_opt

    gs_result = golden_section_search(
        prob, (a, b), tol=refine_tol, Z_init=Z_warm,
    )

    # 격자 결과와 황금분할 결과 중 더 좋은 것 선택
    if gs_result.inner_result.cost < grid_result.inner_result.cost:
        return gs_result
    return grid_result
