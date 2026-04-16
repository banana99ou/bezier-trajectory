"""획득 함수 및 배치 선택."""

import numpy as np


def predictive_entropy(proba):
    """예측 엔트로피 계산.

    H = -sum_c p_c * log(p_c)

    Args:
        proba: (N, C) 클래스 확률
    Returns:
        entropy: (N,) 엔트로피 배열
    """
    # log(0) 방지
    proba_safe = np.clip(proba, 1e-15, 1.0)
    return -np.sum(proba * np.log(proba_safe), axis=1)


def greedy_batch_selection(candidates, entropy, k, d_min):
    """Greedy 배치 선택 (엔트로피 + 다양성).

    Args:
        candidates: (N_cand, d) 후보 점 (정규화 좌표)
        entropy: (N_cand,) 엔트로피
        k: 배치 크기
        d_min: 최소 분리 거리

    Returns:
        selected_indices: (k,) 선택된 인덱스
    """
    selected = []
    available = np.ones(len(candidates), dtype=bool)

    for _ in range(k):
        if not np.any(available):
            break

        # 가용 후보 중 최대 엔트로피 선택
        masked_entropy = np.where(available, entropy, -np.inf)
        idx = np.argmax(masked_entropy)
        selected.append(idx)

        # 선택된 점 주변 d_min 이내 후보 제외
        dists = np.linalg.norm(candidates - candidates[idx], axis=1)
        available[dists < d_min] = False

    return np.array(selected)
