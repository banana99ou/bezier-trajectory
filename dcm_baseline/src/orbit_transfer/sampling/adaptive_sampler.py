"""GP 기반 적응적 샘플링 루프."""

import numpy as np

from ..config import (
    GP_BATCH_SIZE,
    GP_D_MIN,
    GP_ENTROPY_THRESHOLD,
    GP_N_INIT,
    GP_N_MAX,
    PARAM_RANGES,
)
from .acquisition import greedy_batch_selection, predictive_entropy
from .gp_classifier import GPClassifierWrapper
from .lhs import denormalize_params, latin_hypercube_sample, stratified_latin_hypercube_sample


class AdaptiveSampler:
    """GP 기반 적응적 샘플링 루프.

    1단계: LHS 초기 샘플링 (N_init)
    2단계: GP 훈련 -> 엔트로피 계산 -> 배치 선택 -> 최적화 -> 반복
    수렴 조건: max_entropy < threshold OR n_samples >= N_max OR 3회 연속 경계 안정
    """

    def __init__(
        self,
        h0,
        evaluate_fn,
        n_init=GP_N_INIT,
        n_max=GP_N_MAX,
        batch_size=GP_BATCH_SIZE,
        d_min=GP_D_MIN,
        entropy_threshold=GP_ENTROPY_THRESHOLD,
        seed=42,
        stratified=False,
        strata=None,
    ):
        """
        Args:
            h0: 초기 고도 [km]
            evaluate_fn: callable(**kwargs) -> int (class label)
                궤적 최적화 + 분류를 수행하는 함수.
                kwargs 키: sorted(PARAM_RANGES.keys())
            n_init: 초기 LHS 샘플 수
            n_max: 최대 샘플 수
            batch_size: 배치 크기
            d_min: 최소 분리 거리 (정규화)
            entropy_threshold: 수렴 엔트로피 임계값
            seed: 랜덤 시드
            stratified: True이면 계층화 LHS 사용
            strata: 계층 정의 (stratified=True일 때만 사용)
        """
        self.h0 = h0
        self.evaluate_fn = evaluate_fn
        self.n_init = n_init
        self.n_max = n_max
        self.batch_size = batch_size
        self.d_min = d_min
        self.entropy_threshold = entropy_threshold
        self.seed = seed
        self.stratified = stratified
        self.strata = strata

        # 정렬된 키 순서 → evaluate_fn 인자 매핑
        self._sorted_keys = sorted(PARAM_RANGES.keys())
        self._key_to_dim = {k: i for i, k in enumerate(self._sorted_keys)}
        n_dims = len(PARAM_RANGES)

        self.X_train = []  # 정규화 좌표
        self.y_train = []  # 클래스 라벨
        self.gpc = GPClassifierWrapper(n_dims=n_dims)

        # 5D 랜덤 후보 (50,000)
        rng = np.random.default_rng(self.seed)
        self.candidates = rng.uniform(0, 1, size=(50_000, n_dims))

    def _call_evaluate(self, phys):
        """물리 좌표 배열에서 evaluate_fn(**kwargs) 호출 (차원 독립적)."""
        kwargs = {}
        for key in self._sorted_keys:
            kwargs[key] = float(phys[self._key_to_dim[key]])
        return self.evaluate_fn(**kwargs)

    def run(self):
        """적응적 샘플링 루프 실행.

        Returns:
            X_train: (N, n_dims) 정규화 좌표
            y_train: (N,) 클래스 라벨
            gpc: 훈련된 GP 분류기
        """
        # 1단계: 초기 LHS (계층화 또는 균일)
        if self.stratified:
            samples_phys, samples_norm = stratified_latin_hypercube_sample(
                self.n_init, strata=self.strata, seed=self.seed
            )
        else:
            samples_phys, samples_norm = latin_hypercube_sample(
                self.n_init, seed=self.seed
            )

        for i in range(self.n_init):
            label = self._call_evaluate(samples_phys[i])
            self.X_train.append(samples_norm[i])
            self.y_train.append(label)

        X = np.array(self.X_train)
        y = np.array(self.y_train)

        # 2단계: 적응적 루프
        prev_predictions = None
        stable_count = 0

        while len(y) < self.n_max:
            # GP 훈련
            # 최소 2개 클래스 필요
            if len(np.unique(y)) < 2:
                break

            self.gpc.fit(X, y)

            # 후보에서 엔트로피 계산
            proba = self.gpc.predict_proba(self.candidates)
            entropy = predictive_entropy(proba)

            # 수렴 확인: max entropy < threshold
            max_entropy = np.max(entropy)
            if max_entropy < self.entropy_threshold:
                break

            # 경계 안정성 확인
            predictions = self.gpc.predict(self.candidates)
            if prev_predictions is not None and np.array_equal(
                predictions, prev_predictions
            ):
                stable_count += 1
                if stable_count >= 3:
                    break
            else:
                stable_count = 0
            prev_predictions = predictions.copy()

            # 배치 선택
            selected = greedy_batch_selection(
                self.candidates, entropy, self.batch_size, self.d_min
            )

            if len(selected) == 0:
                break

            # 선택된 점에서 평가
            for idx in selected:
                phys = denormalize_params(self.candidates[idx : idx + 1])[0]
                label = self._call_evaluate(phys)
                self.X_train.append(self.candidates[idx])
                self.y_train.append(label)

            X = np.array(self.X_train)
            y = np.array(self.y_train)

        return X, y, self.gpc
