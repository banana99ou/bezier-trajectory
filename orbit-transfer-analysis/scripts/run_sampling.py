import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

from orbit_transfer.pipeline.evaluate import create_database, make_sampler_evaluate_fn
from orbit_transfer.sampling.adaptive_sampler import AdaptiveSampler
from orbit_transfer.database.query import get_statistics


def main():
    # 먼저 아주 작게 테스트
    h0 = 400.0

    # 고정 슬라이스
    e0 = 0.0
    ef = 0.0

    db = create_database(
        db_path="data/trajectories_circular.duckdb",
        npz_dir="data/trajectories_circular",
    )

    evaluate_fn = make_sampler_evaluate_fn(h0=h0, e0=e0, ef=ef, db=db)

    sampler = AdaptiveSampler(
        h0=h0,
        evaluate_fn=evaluate_fn,
        n_init=30,
        n_max=120,
        batch_size=2,
        seed=42,
    )

    X, y, gpc = sampler.run()

    stats = get_statistics(db, h0=h0)

    print("샘플링 완료")
    print("총 샘플 수:", len(y))
    print("통계:", stats)
    print("slice: e0 =", e0, ", ef =", ef)


if __name__ == "__main__":
    main()