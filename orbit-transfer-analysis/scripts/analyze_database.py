"""데이터베이스 분석.

Usage:
    python scripts/analyze_database.py --db data/trajectories.duckdb
"""
import argparse
import sys

sys.path.insert(0, 'src')

from orbit_transfer.database.storage import TrajectoryDatabase
from orbit_transfer.database.query import get_statistics


def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory database')
    parser.add_argument('--db', type=str, default='data/trajectories.duckdb')
    args = parser.parse_args()

    db = TrajectoryDatabase(db_path=args.db)
    stats = get_statistics(db)

    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    from orbit_transfer.config import H0_SLICES
    for h0 in H0_SLICES:
        counts = db.count_by_class(h0=h0)
        if sum(counts.values()) > 0:
            print(f"\n  h0={h0}km: {counts}")

    db.close()


if __name__ == '__main__':
    main()
