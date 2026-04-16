"""DuckDB 데이터베이스 모듈 테스트."""

import numpy as np
import pytest

from orbit_transfer.database import TrajectoryDatabase, get_statistics
from orbit_transfer.types import TransferConfig, TrajectoryResult


# ============================================================
# 더미 데이터 생성 헬퍼
# ============================================================

def make_dummy_config(h0=400.0, delta_a=200.0, delta_i=0.0, T_max_normed=2.0):
    return TransferConfig(
        h0=h0, delta_a=delta_a, delta_i=delta_i, T_max_normed=T_max_normed
    )


def make_dummy_result(converged=True, n_peaks=1):
    N = 61
    return TrajectoryResult(
        converged=converged,
        cost=0.001 if converged else float("inf"),
        t=np.linspace(0, 5400, N),
        x=np.random.randn(6, N),
        u=np.random.randn(3, N) * 0.001,
        nu0=0.0,
        nuf=3.14,
        n_peaks=n_peaks,
        profile_class=0 if n_peaks <= 1 else (1 if n_peaks == 2 else 2),
    )


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def db(tmp_path):
    """임시 디렉토리에 DB 생성."""
    db_path = str(tmp_path / "test.duckdb")
    npz_dir = str(tmp_path / "trajectories")
    database = TrajectoryDatabase(db_path=db_path, npz_dir=npz_dir)
    yield database
    database.close()


# ============================================================
# 1. CRUD round-trip
# ============================================================

class TestCRUDRoundTrip:
    """삽입 -> 조회 -> 값 일치 확인."""

    def test_insert_and_query(self, db):
        config = make_dummy_config()
        result = make_dummy_result(converged=True, n_peaks=2)

        row_id = db.insert_result(config, result, solve_time=1.5)
        assert row_id == 1

        rows = db.get_results()
        assert len(rows) == 1

        row = rows[0]
        assert row["id"] == 1
        assert row["h0"] == config.h0
        assert row["delta_a"] == config.delta_a
        assert row["delta_i"] == config.delta_i
        assert row["T_max_normed"] == config.T_max_normed
        assert row["converged"] is True
        assert row["cost"] == pytest.approx(result.cost)
        assert row["nu0"] == pytest.approx(result.nu0)
        assert row["nuf"] == pytest.approx(result.nuf)
        assert row["n_peaks"] == result.n_peaks
        assert row["profile_class"] == result.profile_class
        assert row["solve_time"] == pytest.approx(1.5)

    def test_insert_multiple(self, db):
        for i in range(5):
            config = make_dummy_config(delta_a=100.0 * (i + 1))
            result = make_dummy_result(n_peaks=i % 3 + 1)
            db.insert_result(config, result)

        rows = db.get_results()
        assert len(rows) == 5

    def test_auto_increment_id(self, db):
        config = make_dummy_config()
        id1 = db.insert_result(config, make_dummy_result())
        id2 = db.insert_result(config, make_dummy_result())
        id3 = db.insert_result(config, make_dummy_result())
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_pass1_cost_none(self, db):
        config = make_dummy_config()
        result = make_dummy_result()
        assert result.pass1_cost is None

        db.insert_result(config, result)
        rows = db.get_results()
        assert rows[0]["pass1_cost"] is None

    def test_diverged_result(self, db):
        config = make_dummy_config()
        result = make_dummy_result(converged=False)

        db.insert_result(config, result)
        rows = db.get_results()
        assert rows[0]["converged"] is False


# ============================================================
# 2. npz 저장/로드
# ============================================================

class TestNpzStorage:
    """궤적 데이터 정확 복원 확인."""

    def test_trajectory_roundtrip(self, db):
        config = make_dummy_config()
        result = make_dummy_result()
        row_id = db.insert_result(config, result)

        loaded = db.get_trajectory(row_id)
        np.testing.assert_array_almost_equal(loaded["t"], result.t)
        np.testing.assert_array_almost_equal(loaded["x"], result.x)
        np.testing.assert_array_almost_equal(loaded["u"], result.u)

    def test_trajectory_not_found(self, db):
        with pytest.raises(ValueError, match="ID 999"):
            db.get_trajectory(999)

    def test_multiple_trajectories(self, db):
        results = []
        for i in range(3):
            config = make_dummy_config(delta_a=100.0 * (i + 1))
            result = make_dummy_result()
            row_id = db.insert_result(config, result)
            results.append((row_id, result))

        for row_id, original in results:
            loaded = db.get_trajectory(row_id)
            np.testing.assert_array_almost_equal(loaded["t"], original.t)
            np.testing.assert_array_almost_equal(loaded["x"], original.x)
            np.testing.assert_array_almost_equal(loaded["u"], original.u)


# ============================================================
# 3. 필터 쿼리
# ============================================================

class TestFilterQuery:
    """h0, converged, profile_class 각각 필터링 정확."""

    def _insert_varied_data(self, db):
        """다양한 파라미터 조합으로 데이터 삽입."""
        # h0=400, converged, class 0
        db.insert_result(
            make_dummy_config(h0=400.0),
            make_dummy_result(converged=True, n_peaks=1),
        )
        # h0=400, converged, class 1
        db.insert_result(
            make_dummy_config(h0=400.0),
            make_dummy_result(converged=True, n_peaks=2),
        )
        # h0=500, converged, class 2
        db.insert_result(
            make_dummy_config(h0=500.0),
            make_dummy_result(converged=True, n_peaks=3),
        )
        # h0=500, diverged, class 0
        db.insert_result(
            make_dummy_config(h0=500.0),
            make_dummy_result(converged=False, n_peaks=1),
        )

    def test_filter_by_h0(self, db):
        self._insert_varied_data(db)
        rows = db.get_results(h0=400.0)
        assert len(rows) == 2
        assert all(r["h0"] == 400.0 for r in rows)

    def test_filter_by_converged(self, db):
        self._insert_varied_data(db)
        rows = db.get_results(converged=True)
        assert len(rows) == 3
        assert all(r["converged"] is True for r in rows)

    def test_filter_by_profile_class(self, db):
        self._insert_varied_data(db)
        rows = db.get_results(profile_class=2)
        assert len(rows) == 1
        assert rows[0]["profile_class"] == 2

    def test_filter_combined(self, db):
        self._insert_varied_data(db)
        rows = db.get_results(h0=400.0, converged=True)
        assert len(rows) == 2

    def test_filter_no_match(self, db):
        self._insert_varied_data(db)
        rows = db.get_results(h0=999.0)
        assert len(rows) == 0


# ============================================================
# 4. count_by_class
# ============================================================

class TestCountByClass:
    """더미 데이터 삽입 후 집계 정확."""

    def test_count_all(self, db):
        # class 0: 3개, class 1: 2개, class 2: 1개
        for _ in range(3):
            db.insert_result(make_dummy_config(), make_dummy_result(n_peaks=1))
        for _ in range(2):
            db.insert_result(make_dummy_config(), make_dummy_result(n_peaks=2))
        db.insert_result(make_dummy_config(), make_dummy_result(n_peaks=3))

        counts = db.count_by_class()
        assert counts[0] == 3
        assert counts[1] == 2
        assert counts[2] == 1

    def test_count_by_h0(self, db):
        db.insert_result(
            make_dummy_config(h0=400.0), make_dummy_result(n_peaks=1)
        )
        db.insert_result(
            make_dummy_config(h0=400.0), make_dummy_result(n_peaks=2)
        )
        db.insert_result(
            make_dummy_config(h0=500.0), make_dummy_result(n_peaks=1)
        )

        counts_400 = db.count_by_class(h0=400.0)
        assert counts_400[0] == 1
        assert counts_400[1] == 1

        counts_500 = db.count_by_class(h0=500.0)
        assert counts_500[0] == 1
        assert 1 not in counts_500

    def test_count_empty(self, db):
        counts = db.count_by_class()
        assert counts == {}


# ============================================================
# 5. 빈 DB 조회
# ============================================================

class TestEmptyDatabase:
    """에러 없이 빈 결과 반환."""

    def test_get_results_empty(self, db):
        rows = db.get_results()
        assert rows == []

    def test_get_results_with_filter_empty(self, db):
        rows = db.get_results(h0=400.0, converged=True, profile_class=0)
        assert rows == []

    def test_count_by_class_empty(self, db):
        counts = db.count_by_class()
        assert counts == {}

    def test_statistics_empty(self, db):
        stats = get_statistics(db)
        assert stats["total"] == 0
        assert stats["converged"] == 0
        assert stats["convergence_rate"] == 0.0
        assert stats["class_distribution"] == {}
        assert stats["mean_cost"] is None
