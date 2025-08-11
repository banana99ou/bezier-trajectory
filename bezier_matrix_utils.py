"""
베지어 곡선 행렬 계산 유틸리티 모듈

더미 베지어 곡선 객체를 생성하지 않고 직접적으로 적분/미분 행렬을 계산하는 유틸리티 함수들을 제공합니다.
캐싱을 통해 성능을 최적화하고, 명시적인 인터페이스로 코드 가독성을 향상시킵니다.
"""

import numpy as np
from typing import Dict, Tuple
from functools import lru_cache


# 캐시 저장소 (전역 캐시)
_MATRIX_CACHE: Dict[Tuple[str, int], np.ndarray] = {}


def _generate_cache_key(matrix_type: str, degree: int) -> Tuple[str, int]:
    """캐시 키 생성"""
    return (matrix_type, degree)


@lru_cache(maxsize=128)
def _compute_binomial_coefficient(n: int, k: int) -> float:
    """이항계수 계산 (캐시 적용)"""
    if k > n or k < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    
    # C(n,k) = n! / (k! * (n-k)!)
    result = 1.0
    for i in range(min(k, n - k)):
        result = result * (n - i) / (i + 1)
    return result


def get_affine_integral_matrix(degree: int) -> np.ndarray:
    """
    베지어 곡선의 아핀 적분 행렬 계산
    
    아핀 적분 연산자: Ĩ = I - 1_{n+2} * e_{n+2,1}^T * I
    여기서 I는 일반 적분 행렬, e_{n+2,1}은 첫 번째 단위벡터
    
    Args:
        degree (int): 베지어 곡선의 차수
        
    Returns:
        np.ndarray: 아핀 적분 행렬, shape ((degree+2), (degree+1))
    """
    cache_key = _generate_cache_key("affine_integral", degree)
    
    # 캐시에서 확인
    if cache_key in _MATRIX_CACHE:
        return _MATRIX_CACHE[cache_key]
    
    # 새로 계산
    n = degree
    matrix_shape = (n + 2, n + 1)
    
    # 일반 적분 행렬 I 계산 (bezier.py와 동일한 공식 사용)
    # 공식: Q_0 = 0, Q_j = (1/(N+1)) * sum_{i=0}^{j-1} P_i for j=1~N+1
    I = np.zeros(matrix_shape)
    for j in range(1, n + 2):
        I[j, :j] = 1.0 / (n + 1)
    
    # 아핀 적분 행렬 계산: Ĩ = I - 1_{n+2} * e_{n+2,1}^T * I
    # e_{n+2,1} = [1, 0, 0, ..., 0]^T (첫 번째 표준순서기저)
    e_first = np.zeros(n + 2)
    e_first[0] = 1
    
    # 1_{n+2} = [1, 1, 1, ..., 1]^T (모든 성분이 1인 벡터)
    ones_vector = np.ones(n + 2)
    
    # Ĩ = I - 1_{n+2} * e_{n+2,1}^T * I
    affine_integral_matrix = I - np.outer(ones_vector, e_first @ I)
    
    # 캐시에 저장
    _MATRIX_CACHE[cache_key] = affine_integral_matrix
    
    return affine_integral_matrix


def get_derivative_matrix(degree: int, order: int = 1) -> np.ndarray:
    """
    베지어 곡선의 미분 행렬 계산
    
    Args:
        degree (int): 베지어 곡선의 차수
        order (int): 미분 차수 (기본값: 1)
        
    Returns:
        np.ndarray: 미분 행렬, shape ((degree-order+1), (degree+1))
    """
    cache_key = _generate_cache_key(f"derivative_{order}", degree)
    
    # 캐시에서 확인
    if cache_key in _MATRIX_CACHE:
        return _MATRIX_CACHE[cache_key]
    
    if order > degree:
        # 차수보다 높은 미분은 0
        result = np.zeros((1, degree + 1))
        _MATRIX_CACHE[cache_key] = result
        return result
    
    # 미분 행렬 계산
    n = degree
    matrix_shape = (n - order + 1, n + 1)
    D = np.zeros(matrix_shape)
    
    for i in range(n - order + 1):
        for j in range(n + 1):
            if j >= i and j <= i + order:
                # 미분 계수 계산
                coeff = 1.0
                for k in range(order):
                    coeff *= (n - k)
                
                # 이항계수 비율
                if i + order <= n:
                    numerator = _compute_binomial_coefficient(n - order, i)
                    denominator = _compute_binomial_coefficient(n, j)
                    if denominator != 0:
                        D[i, j] = coeff * numerator / denominator * (1 if (j - i) % 2 == 0 else -1)
    
    # 캐시에 저장
    _MATRIX_CACHE[cache_key] = D
    
    return D


def get_elevation_matrix(from_degree: int, to_degree: int) -> np.ndarray:
    """
    차수 승격 행렬 계산
    
    Args:
        from_degree (int): 원래 차수
        to_degree (int): 목표 차수
        
    Returns:
        np.ndarray: 차수 승격 행렬, shape ((to_degree+1), (from_degree+1))
    """
    if to_degree < from_degree:
        raise ValueError("목표 차수는 원래 차수보다 크거나 같아야 합니다.")
    
    cache_key = _generate_cache_key(f"elevation_{from_degree}_to_{to_degree}", 0)
    
    # 캐시에서 확인
    if cache_key in _MATRIX_CACHE:
        return _MATRIX_CACHE[cache_key]
    
    n = from_degree
    m = to_degree
    matrix_shape = (m + 1, n + 1)
    E = np.zeros(matrix_shape)
    
    for i in range(m + 1):
        for j in range(n + 1):
            if j <= i <= j + (m - n):
                # 차수 승격 계수
                numerator = _compute_binomial_coefficient(n, j) * _compute_binomial_coefficient(m - n, i - j)
                denominator = _compute_binomial_coefficient(m, i)
                E[i, j] = numerator / denominator if denominator != 0 else 0.0
    
    # 캐시에 저장
    _MATRIX_CACHE[cache_key] = E
    
    return E


def clear_matrix_cache():
    """행렬 캐시 초기화"""
    global _MATRIX_CACHE
    _MATRIX_CACHE.clear()


def get_cache_info() -> dict:
    """캐시 정보 반환"""
    return {
        'cached_matrices': len(_MATRIX_CACHE),
        'cache_keys': list(_MATRIX_CACHE.keys())
    }


def get_bernstein_at_one(degree: int) -> np.ndarray:
    """
    베르누슈타인 기저함수를 τ=1에서 계산
    
    Args:
        degree (int): 베지어 곡선의 차수
        
    Returns:
        np.ndarray: B_N(1) 벡터, shape (degree+1,)
    """
    cache_key = _generate_cache_key("bernstein_at_one", degree)
    
    # 캐시에서 확인
    if cache_key in _MATRIX_CACHE:
        return _MATRIX_CACHE[cache_key]
    
    # 베르누슈타인 기저함수: B_{i,N}(τ) = C(N,i) * τ^i * (1-τ)^(N-i)
    # τ=1에서: B_{i,N}(1) = C(N,i) * 1^i * 0^(N-i) = 0 (i < N), 1 (i = N)
    bernstein = np.zeros(degree + 1)
    bernstein[degree] = 1.0  # B_{N,N}(1) = 1, 나머지는 0
    
    # 캐시에 저장
    _MATRIX_CACHE[cache_key] = bernstein
    
    return bernstein


def precompute_common_matrices(max_degree: int = 10):
    """
    자주 사용되는 행렬들을 미리 계산하여 캐시에 저장
    
    Args:
        max_degree (int): 최대 차수 (기본값: 10)
    """
    print(f"베지어 행렬 캐시 사전 계산 중... (최대 차수: {max_degree})")
    
    for degree in range(max_degree + 1):
        # 아핀 적분 행렬
        get_affine_integral_matrix(degree)
        
        # 베르누슈타인 기저함수
        get_bernstein_at_one(degree)
        
        # 1차, 2차 미분 행렬
        if degree >= 1:
            get_derivative_matrix(degree, 1)
        if degree >= 2:
            get_derivative_matrix(degree, 2)
    
    print(f"사전 계산 완료: {len(_MATRIX_CACHE)}개 행렬 캐시됨")