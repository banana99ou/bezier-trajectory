"""고급 쿼리 함수."""

from .storage import TrajectoryDatabase


def get_statistics(db: TrajectoryDatabase, h0: float = None) -> dict:
    """통계 조회: 수렴율, 클래스 분포, 평균 비용 등.

    Returns:
        dict: {
            "total": 총 케이스 수,
            "converged": 수렴 케이스 수,
            "convergence_rate": 수렴율 (0~1),
            "class_distribution": {class_id: count},
            "mean_cost": 수렴 케이스 평균 비용,
        }
    """
    where_clause = ""
    params = []
    if h0 is not None:
        where_clause = " WHERE h0 = ?"
        params.append(h0)

    # 전체 개수
    total = db.conn.execute(
        f"SELECT COUNT(*) FROM trajectories{where_clause}", params
    ).fetchone()[0]

    if total == 0:
        return {
            "total": 0,
            "converged": 0,
            "convergence_rate": 0.0,
            "class_distribution": {},
            "mean_cost": None,
        }

    # 수렴 개수
    converged_where = " WHERE converged = TRUE" + (" AND h0 = ?" if h0 is not None else "")
    converged_params = [h0] if h0 is not None else []
    converged = db.conn.execute(
        f"SELECT COUNT(*) FROM trajectories{converged_where}", converged_params
    ).fetchone()[0]

    # 수렴율
    convergence_rate = converged / total if total > 0 else 0.0

    # 클래스 분포
    class_distribution = db.count_by_class(h0=h0)

    # 평균 비용 (수렴 케이스)
    mean_cost_row = db.conn.execute(
        f"SELECT AVG(cost) FROM trajectories{converged_where}", converged_params
    ).fetchone()
    mean_cost = mean_cost_row[0] if mean_cost_row[0] is not None else None

    return {
        "total": total,
        "converged": converged,
        "convergence_rate": convergence_rate,
        "class_distribution": class_distribution,
        "mean_cost": mean_cost,
    }


def get_boundary_cases(db: TrajectoryDatabase, h0: float) -> list[dict]:
    """결정 경계 근처 케이스 조회 (인접한 다른 클래스 존재).

    5D 매개변수 공간 (delta_a, delta_i, T_max_normed, e0, ef)에서
    유클리드 거리 기준으로 가장 가까운 이웃이 다른 클래스인 케이스를 반환한다.

    Args:
        db: TrajectoryDatabase 인스턴스
        h0: 초기 고도 필터 [km]

    Returns:
        인접 클래스가 다른 케이스 목록 (dict 리스트)
    """
    # 수렴 케이스만 대상
    rows = db.get_results(h0=h0, converged=True)
    if len(rows) < 2:
        return []

    boundary = []
    for i, row_i in enumerate(rows):
        min_dist = float("inf")
        nearest_class = row_i["profile_class"]
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            dist = (
                (row_i["delta_a"] - row_j["delta_a"]) ** 2
                + (row_i["delta_i"] - row_j["delta_i"]) ** 2
                + (row_i["T_max_normed"] - row_j["T_max_normed"]) ** 2
                + (row_i["e0"] - row_j["e0"]) ** 2
                + (row_i["ef"] - row_j["ef"]) ** 2
            ) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_class = row_j["profile_class"]
        if nearest_class != row_i["profile_class"]:
            boundary.append(row_i)

    return boundary
