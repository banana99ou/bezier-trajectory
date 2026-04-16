"""5D 샘플링 모듈 테스트."""

import numpy as np
import pytest

from orbit_transfer.config import PARAM_RANGES
from orbit_transfer.sampling.adaptive_sampler import AdaptiveSampler
from orbit_transfer.sampling.lhs import (
    latin_hypercube_sample,
    normalize_params,
    denormalize_params,
)


class Test5DSampler:
    def test_sorted_keys_order(self):
        """키 순서: ['T_max_normed', 'delta_a', 'delta_i', 'e0', 'ef']"""
        expected = ['T_max_normed', 'delta_a', 'delta_i', 'e0', 'ef']
        assert sorted(PARAM_RANGES.keys()) == expected

    def test_param_ranges_5d(self):
        """PARAM_RANGES가 5개 차원."""
        assert len(PARAM_RANGES) == 5

    def test_candidates_shape(self):
        """후보 격자 shape = (50000, 5)"""
        def dummy_eval(T_max_normed, delta_a, delta_i, e0, ef):
            return 0

        sampler = AdaptiveSampler(
            h0=400, evaluate_fn=dummy_eval,
            n_init=10, n_max=20, seed=42,
        )
        assert sampler.candidates.shape == (50_000, 5)

    def test_evaluate_kwargs(self):
        """_call_evaluate가 올바른 kwargs로 호출하는지."""
        received = {}

        def capture_eval(T_max_normed, delta_a, delta_i, e0, ef):
            received['T_max_normed'] = T_max_normed
            received['delta_a'] = delta_a
            received['delta_i'] = delta_i
            received['e0'] = e0
            received['ef'] = ef
            return 0

        sampler = AdaptiveSampler(
            h0=400, evaluate_fn=capture_eval,
            n_init=10, n_max=20, seed=42,
        )
        # 물리 좌표 (sorted order)
        phys = np.array([0.5, 500.0, 5.0, 0.03, 0.04])
        sampler._call_evaluate(phys)
        assert received['T_max_normed'] == pytest.approx(0.5)
        assert received['delta_a'] == pytest.approx(500.0)
        assert received['delta_i'] == pytest.approx(5.0)
        assert received['e0'] == pytest.approx(0.03)
        assert received['ef'] == pytest.approx(0.04)

    def test_normalize_denormalize_roundtrip_5d(self):
        """5D 정규화 <-> 역정규화 round-trip."""
        samples, normed = latin_hypercube_sample(20, seed=0)
        recovered = denormalize_params(normalize_params(samples))
        np.testing.assert_allclose(recovered, samples, atol=1e-12)

    def test_lhs_5d_shape(self):
        """LHS 5D 출력 shape."""
        samples, normed = latin_hypercube_sample(50, seed=42)
        assert samples.shape == (50, 5)
        assert normed.shape == (50, 5)

    def test_lhs_5d_range(self):
        """LHS 5D 정규화 범위 [0, 1]."""
        _, normed = latin_hypercube_sample(100, seed=42)
        assert np.all(normed >= 0) and np.all(normed <= 1)

    def test_small_convergence_5d(self):
        """5D 소규모 수렴 테스트."""
        def mock_evaluate(T_max_normed, delta_a, delta_i, e0, ef):
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
        assert len(y) >= 20
        assert len(y) <= 50
        assert X.shape[1] == 5

    def test_stratified_sampler(self):
        """stratified=True일 때 정상 동작."""
        def mock_evaluate(T_max_normed, delta_a, delta_i, e0, ef):
            if T_max_normed < 0.5:
                return 0  # unimodal
            elif delta_a < 1000:
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
            stratified=True,
        )
        X, y, gpc = sampler.run()
        assert len(y) >= 20
        assert len(y) <= 50
        assert X.shape[1] == 5
        # stratified이면 unimodal(0)이 더 많이 나와야 함
        n_unimodal = np.sum(y == 0)
        assert n_unimodal >= 3, f"unimodal 수 {n_unimodal} < 3"
