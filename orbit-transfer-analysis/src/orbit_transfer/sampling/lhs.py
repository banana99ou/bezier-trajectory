"""Latin Hypercube Sampling 및 좌표 변환."""

import numpy as np

from ..config import PARAM_RANGES


def latin_hypercube_sample(n_samples, param_ranges=None, seed=None):
    """Latin Hypercube Sampling.

    Args:
        n_samples: 샘플 수
        param_ranges: dict 또는 None (기본: PARAM_RANGES)
            각 key: (min, max) 튜플
        seed: 랜덤 시드

    Returns:
        samples: (n_samples, n_dims) ndarray, 물리 좌표
        samples_normed: (n_samples, n_dims) ndarray, [0,1] 정규화 좌표
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    rng = np.random.default_rng(seed)
    n_dims = len(param_ranges)
    keys = sorted(param_ranges.keys())

    # LHS: 각 차원을 n_samples개 구간으로 나누고, 각 구간에서 균일 랜덤
    samples_normed = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples_normed[perm[i], d] = (i + rng.uniform()) / n_samples

    # 물리 좌표 변환
    samples = np.zeros_like(samples_normed)
    for d, key in enumerate(keys):
        lo, hi = param_ranges[key]
        samples[:, d] = lo + (hi - lo) * samples_normed[:, d]

    return samples, samples_normed


def stratified_latin_hypercube_sample(
    n_samples, strata=None, param_ranges=None, seed=None
):
    """계층화 Latin Hypercube Sampling.

    T_max_normed 차원을 구간별로 나누어 할당 비율에 따라 샘플링.
    나머지 차원은 전체 범위에서 LHS 수행.

    Args:
        n_samples: 총 샘플 수
        strata: 계층 정의 리스트. 각 항목은 (lo_frac, hi_frac, weight) 튜플.
            lo_frac, hi_frac: T_max_normed 범위 내 정규화 비율 [0, 1]
            weight: 할당 비중 (합이 1일 필요 없음, 자동 정규화)
            기본값: [(0.0, 1/3, 0.4), (1/3, 2/3, 0.3), (2/3, 1.0, 0.3)]
            → T_normed [0.15, 0.50]: 40%, [0.50, 0.85]: 30%, [0.85, 1.2]: 30%
        param_ranges: dict 또는 None (기본: PARAM_RANGES)
        seed: 랜덤 시드

    Returns:
        samples: (n_samples, n_dims) ndarray, 물리 좌표
        samples_normed: (n_samples, n_dims) ndarray, [0,1] 정규화 좌표
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    if strata is None:
        strata = [
            (0.0, 1 / 3, 0.4),   # T_normed 하위 1/3 (단봉형 유망)
            (1 / 3, 2 / 3, 0.3),  # 중간
            (2 / 3, 1.0, 0.3),    # 상위
        ]

    rng = np.random.default_rng(seed)
    n_dims = len(param_ranges)
    keys = sorted(param_ranges.keys())
    t_dim = keys.index("T_max_normed")

    # 가중치 정규화 및 샘플 수 배분
    weights = np.array([s[2] for s in strata])
    weights = weights / weights.sum()
    counts = np.round(weights * n_samples).astype(int)
    # 반올림 오차 보정: 마지막 계층에서 조정
    counts[-1] = n_samples - counts[:-1].sum()

    all_normed = []
    for (lo_frac, hi_frac, _), n_stratum in zip(strata, counts):
        if n_stratum <= 0:
            continue
        # 각 계층에 대해 독립 LHS
        block = np.zeros((n_stratum, n_dims))
        for d in range(n_dims):
            perm = rng.permutation(n_stratum)
            for i in range(n_stratum):
                block[perm[i], d] = (i + rng.uniform()) / n_stratum

        # T_max_normed 차원: 계층 범위로 리스케일
        block[:, t_dim] = lo_frac + (hi_frac - lo_frac) * block[:, t_dim]
        all_normed.append(block)

    samples_normed = np.vstack(all_normed)
    # 셔플 (계층 순서 제거)
    rng.shuffle(samples_normed)

    # 물리 좌표 변환
    samples = np.zeros_like(samples_normed)
    for d, key in enumerate(keys):
        lo, hi = param_ranges[key]
        samples[:, d] = lo + (hi - lo) * samples_normed[:, d]

    return samples, samples_normed


def normalize_params(params, param_ranges=None):
    """물리 좌표 -> [0,1] 정규화."""
    if param_ranges is None:
        param_ranges = PARAM_RANGES
    keys = sorted(param_ranges.keys())
    normed = np.zeros_like(params)
    for d, key in enumerate(keys):
        lo, hi = param_ranges[key]
        normed[..., d] = (params[..., d] - lo) / (hi - lo)
    return normed


def denormalize_params(normed, param_ranges=None):
    """[0,1] 정규화 -> 물리 좌표."""
    if param_ranges is None:
        param_ranges = PARAM_RANGES
    keys = sorted(param_ranges.keys())
    params = np.zeros_like(normed)
    for d, key in enumerate(keys):
        lo, hi = param_ranges[key]
        params[..., d] = lo + (hi - lo) * normed[..., d]
    return params
