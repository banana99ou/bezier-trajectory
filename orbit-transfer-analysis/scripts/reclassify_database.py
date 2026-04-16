"""기존 DB를 새 Topological Persistence 피크 탐지기로 재분류.

모든 수렴 궤적의 npz를 로드하여 새 피크 탐지기로 n_peaks/profile_class를 갱신한다.

Usage:
    python scripts/reclassify_database.py
    python scripts/reclassify_database.py --db_path data/trajectories.duckdb
    python scripts/reclassify_database.py --all
"""
import argparse
import sys
import time

sys.path.insert(0, 'src')

import duckdb
import numpy as np

from orbit_transfer.classification.peak_detection import detect_peaks
from orbit_transfer.classification.classifier import classify_profile


def reclassify_db(db_path, dry_run=False):
    """단일 DB 재분류."""
    conn = duckdb.connect(db_path)

    # 수렴 케이스 조회
    rows = conn.execute(
        "SELECT id, trajectory_file, T_f, n_peaks, profile_class "
        "FROM trajectories WHERE converged = TRUE"
    ).fetchall()

    print(f"\n[{db_path}] Converged cases: {len(rows)}")

    # 기존 분포
    old_dist = conn.execute(
        "SELECT profile_class, COUNT(*) FROM trajectories "
        "WHERE converged = TRUE GROUP BY profile_class ORDER BY profile_class"
    ).fetchall()
    print(f"  Old distribution: {dict(old_dist)}")

    changed = 0
    errors = 0
    new_classes = {0: 0, 1: 0, 2: 0}

    for row_id, traj_file, T_f, old_npeaks, old_class in rows:
        try:
            data = np.load(traj_file)
            t, x, u = data['t'], data['x'], data['u']
            u_mag = np.linalg.norm(u, axis=0)

            n_peaks, _, _ = detect_peaks(t, u_mag, T_f)
            new_class = classify_profile(n_peaks)
            new_classes[new_class] = new_classes.get(new_class, 0) + 1

            if n_peaks != old_npeaks or new_class != old_class:
                changed += 1
                if not dry_run:
                    conn.execute(
                        "UPDATE trajectories SET n_peaks = ?, profile_class = ? WHERE id = ?",
                        [n_peaks, new_class, row_id],
                    )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR id={row_id}: {e}")

    if not dry_run:
        # 비수렴 케이스도 재분류 (npz가 있는 경우)
        non_conv_rows = conn.execute(
            "SELECT id, trajectory_file, T_f FROM trajectories "
            "WHERE converged = FALSE AND trajectory_file IS NOT NULL"
        ).fetchall()

        for row_id, traj_file, T_f in non_conv_rows:
            try:
                data = np.load(traj_file)
                t, u = data['t'], data['u']
                u_mag = np.linalg.norm(u, axis=0)
                n_peaks, _, _ = detect_peaks(t, u_mag, T_f or 1.0)
                new_class = classify_profile(n_peaks)
                conn.execute(
                    "UPDATE trajectories SET n_peaks = ?, profile_class = ? WHERE id = ?",
                    [n_peaks, new_class, row_id],
                )
            except Exception:
                pass

    # 새 분포
    new_dist = conn.execute(
        "SELECT profile_class, COUNT(*) FROM trajectories "
        "WHERE converged = TRUE GROUP BY profile_class ORDER BY profile_class"
    ).fetchall()
    print(f"  New distribution: {dict(new_dist)}")
    print(f"  Changed: {changed}/{len(rows)} ({100*changed/max(len(rows),1):.1f}%)")
    if errors > 0:
        print(f"  Errors: {errors}")

    conn.close()
    return changed, len(rows), new_classes


def main():
    parser = argparse.ArgumentParser(description='Reclassify DB with new peak detector')
    parser.add_argument('--db_path', type=str, default=None, help='Single DB path')
    parser.add_argument('--all', action='store_true', help='Reclassify all DBs')
    parser.add_argument('--dry_run', action='store_true', help='Dry run (no writes)')
    args = parser.parse_args()

    t0 = time.time()

    if args.all:
        import os
        db_paths = [
            'data/trajectories.duckdb',
        ]
        # 고도별 DB
        for d in ['h600', 'h800', 'h1000']:
            p = f'data/{d}/trajectories.duckdb'
            if os.path.exists(p):
                db_paths.append(p)

        total_changed = 0
        total_rows = 0
        for db_path in db_paths:
            c, r, _ = reclassify_db(db_path, dry_run=args.dry_run)
            total_changed += c
            total_rows += r

        print(f"\n{'='*60}")
        print(f"Total: {total_changed}/{total_rows} changed")

        # 병합 DB 재생성
        if not args.dry_run:
            from run_database import merge_databases
            merge_databases(db_paths, 'data/trajectories_all.duckdb')

    elif args.db_path:
        reclassify_db(args.db_path, dry_run=args.dry_run)
    else:
        # 기본: 메인 DB + 고도별 DB 모두
        import os
        db_paths = ['data/trajectories.duckdb']
        for d in ['h600', 'h800', 'h1000']:
            p = f'data/{d}/trajectories.duckdb'
            if os.path.exists(p):
                db_paths.append(p)

        total_changed = 0
        total_rows = 0
        for db_path in db_paths:
            c, r, _ = reclassify_db(db_path, dry_run=args.dry_run)
            total_changed += c
            total_rows += r

        print(f"\n{'='*60}")
        print(f"Total: {total_changed}/{total_rows} changed")
        print(f"Time: {time.time() - t0:.1f}s")

        # 병합 DB 재생성
        if not args.dry_run:
            from run_database import merge_databases
            merge_databases(db_paths, 'data/trajectories_all.duckdb')


if __name__ == '__main__':
    main()
