"""sklearn GaussianProcessClassifier 래퍼."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


class GPClassifierWrapper:
    """GP 분류기 래퍼 (ARD RBF, Laplace 근사).

    3-class 분류 (0: unimodal, 1: bimodal, 2: multimodal)
    sklearn의 GaussianProcessClassifier 사용.
    One-vs-Rest (OvR) 전략.
    """

    def __init__(self, n_dims=3):
        # ARD RBF 커널: 차원별 길이 척도
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_dims))
        self.gpc = GaussianProcessClassifier(
            kernel=kernel,
            multi_class='one_vs_rest',
            max_iter_predict=100,
            n_restarts_optimizer=3,
        )
        self.is_fitted = False

    def fit(self, X, y):
        """훈련. X: (N, d), y: (N,) integer labels."""
        self.gpc.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X):
        """클래스 확률 예측. Returns: (N, C) ndarray."""
        return self.gpc.predict_proba(X)

    def predict(self, X):
        """클래스 예측. Returns: (N,) ndarray."""
        return self.gpc.predict(X)

    def get_length_scales(self):
        """학습된 ARD 길이 척도 반환.

        OvR 다중 클래스의 경우 각 이진 분류기의 길이 척도를 평균.
        """
        if not self.is_fitted:
            return None

        def _extract_rbf_ls(kernel):
            """Product kernel (Constant * RBF) 에서 RBF 길이 척도 추출."""
            if hasattr(kernel, 'k2'):
                return kernel.k2.length_scale
            if hasattr(kernel, 'length_scale'):
                return kernel.length_scale
            raise AttributeError(f"Cannot extract length_scale from {type(kernel)}")

        kernel = self.gpc.kernel_
        # OvR CompoundKernel: 여러 이진 분류기 커널 포함
        if hasattr(kernel, 'kernels'):
            ls_list = [np.array(_extract_rbf_ls(k)) for k in kernel.kernels]
            return np.mean(ls_list, axis=0)
        else:
            return np.array(_extract_rbf_ls(kernel))
