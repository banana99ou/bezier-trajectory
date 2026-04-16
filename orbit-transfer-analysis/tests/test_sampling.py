"""GP 기반 적응적 샘플링 모듈 테스트."""

import numpy as np
import pytest

from orbit_transfer.sampling.lhs import (
    latin_hypercube_sample,
    stratified_latin_hypercube_sample,
    normalize_params,
    denormalize_params,
)
from orbit_transfer.config import PARAM_RANGES
from orbit_transfer.sampling.gp_classifier import GPClassifierWrapper
from orbit_transfer.sampling.acquisition import (
    predictive_entropy,
    greedy_batch_selection,
)
from orbit_transfer.sampling.adaptive_sampler import AdaptiveSampler


class TestLHS:
    def test_shape(self):
        samples, normed = latin_hypercube_sample(50, seed=42)
        assert samples.shape == (50, 5)
        assert normed.shape == (50, 5)

    def test_range(self):
        _, normed = latin_hypercube_sample(100, seed=42)
        assert np.all(normed >= 0) and np.all(normed <= 1)

    def test_uniformity(self):
        """각 차원의 marginal 분포가 균일에 가까운지 확인."""
        _, normed = latin_hypercube_sample(1000, seed=42)
        for d in range(5):
            hist, _ = np.histogram(normed[:, d], bins=10)
            # 각 bin에 대략 100개씩 (+-30%)
            assert np.all(hist > 70)
            assert np.all(hist < 130)

    def test_normalize_denormalize_roundtrip(self):
        samples, normed = latin_hypercube_sample(20, seed=0)
        recovered = denormalize_params(normalize_params(samples))
        np.testing.assert_allclose(recovered, samples, atol=1e-12)


class TestStratifiedLHS:
    def test_shape(self):
        samples, normed = stratified_latin_hypercube_sample(100, seed=42)
        assert samples.shape == (100, 5)
        assert normed.shape == (100, 5)

    def test_range(self):
        """정규화 좌표가 [0, 1] 범위."""
        _, normed = stratified_latin_hypercube_sample(100, seed=42)
        assert np.all(normed >= 0) and np.all(normed <= 1)

    def test_physical_range(self):
        """물리 좌표가 PARAM_RANGES 범위 내."""
        samples, _ = stratified_latin_hypercube_sample(200, seed=42)
        keys = sorted(PARAM_RANGES.keys())
        for d, key in enumerate(keys):
            lo, hi = PARAM_RANGES[key]
            assert np.all(samples[:, d] >= lo - 1e-10)
            assert np.all(samples[:, d] <= hi + 1e-10)

    def test_stratum_allocation(self):
        """계층별 샘플 비율이 대략 지정된 가중치를 따르는지."""
        strata = [
            (0.0, 0.5, 0.6),  # 60%
            (0.5, 1.0, 0.4),  # 40%
        ]
        _, normed = stratified_latin_hypercube_sample(
            200, strata=strata, seed=42
        )
        keys = sorted(PARAM_RANGES.keys())
        t_dim = keys.index("T_max_normed")
        n_lower = np.sum(normed[:, t_dim] < 0.5)
        # 60% ± 5%
        assert 100 < n_lower < 140, f"하위 계층 샘플: {n_lower}/200"

    def test_custom_strata(self):
        """사용자 정의 계층이 올바르게 적용."""
        strata = [
            (0.0, 1 / 3, 0.5),
            (1 / 3, 2 / 3, 0.25),
            (2 / 3, 1.0, 0.25),
        ]
        samples, normed = stratified_latin_hypercube_sample(
            120, strata=strata, seed=42
        )
        assert samples.shape == (120, 5)

    def test_roundtrip(self):
        """stratified LHS도 normalize/denormalize round-trip 성립."""
        samples, normed = stratified_latin_hypercube_sample(50, seed=42)
        recovered = denormalize_params(normalize_params(samples))
        np.testing.assert_allclose(recovered, samples, atol=1e-12)


class TestGPClassifier:
    def test_synthetic_3class(self):
        """합성 3-class 데이터에서 정확도 > 80%."""
        rng = np.random.default_rng(42)
        N = 200
        X = rng.uniform(0, 1, (N, 5))
        # 간단한 규칙: class = 0 if x0<0.33, 1 if 0.33<x0<0.66, 2 otherwise
        y = np.where(X[:, 0] < 0.33, 0, np.where(X[:, 0] < 0.66, 1, 2))

        gpc = GPClassifierWrapper(n_dims=5)
        # 80% train, 20% test
        n_train = 160
        gpc.fit(X[:n_train], y[:n_train])
        pred = gpc.predict(X[n_train:])
        accuracy = np.mean(pred == y[n_train:])
        assert accuracy > 0.80, f"accuracy {accuracy:.2f} < 0.80"


class TestAcquisition:
    def test_entropy_uniform(self):
        """균일 확률 -> 최대 엔트로피 log(3)."""
        proba = np.array([[1 / 3, 1 / 3, 1 / 3]])
        h = predictive_entropy(proba)
        np.testing.assert_allclose(h, np.log(3), atol=1e-10)

    def test_entropy_certain(self):
        """확정 확률 -> 엔트로피 0."""
        proba = np.array([[1.0, 0.0, 0.0]])
        h = predictive_entropy(proba)
        assert h[0] < 1e-10

    def test_boundary_higher(self):
        """경계 근처 > 내부 엔트로피."""
        proba_boundary = np.array([[0.4, 0.4, 0.2]])
        proba_interior = np.array([[0.9, 0.05, 0.05]])
        h_b = predictive_entropy(proba_boundary)[0]
        h_i = predictive_entropy(proba_interior)[0]
        assert h_b > h_i

    def test_batch_diversity(self):
        """배치 내 모든 쌍 거리 >= d_min."""
        rng = np.random.default_rng(42)
        candidates = rng.uniform(0, 1, (100, 5))
        entropy = rng.uniform(0, 1, 100)
        selected = greedy_batch_selection(candidates, entropy, k=5, d_min=0.15)
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                dist = np.linalg.norm(
                    candidates[selected[i]] - candidates[selected[j]]
                )
                assert dist >= 0.15 - 1e-10


class TestAdaptiveSampler:
    def test_small_convergence(self):
        """소규모 수렴 테스트 (합성 evaluate_fn)."""

        def mock_evaluate(T_max_normed, delta_a, delta_i, e0, ef):
            # 간단 규칙
            if delta_a < 500:
                return 0
            elif delta_a < 1200:
                return 1
            else:
                return 2

        sampler = AdaptiveSampler(
            h0=400,
            evaluate_fn=mock_evaluate,
            n_init=20,
            n_max=50,
            batch_size=5,
            d_min=0.05,
            entropy_threshold=0.3,
            seed=42,
        )
        X, y, gpc = sampler.run()
        assert len(y) >= 20  # 최소 초기 샘플
        assert len(y) <= 50  # 최대 제한
        assert len(np.unique(y)) >= 2  # 최소 2 클래스
