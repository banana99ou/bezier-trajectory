"""
베지어 곡선 라이브러리

이 라이브러리는 베지어 곡선의 생성, 평가, 미분, 적분 등의 
수학적 연산을 제공합니다.

주요 기능:
- N차 베지어 곡선 생성 및 평가
- 베지어 곡선의 미분 계산 (임의 차수)
- 베지어 곡선의 적분 계산 (아핀 적분 포함)
- 차수 승격 연산
- 행렬 기반 고성능 계산

사용 예제:
    >>> import numpy as np
    >>> from bezier import BezierCurve
    >>> 
    >>> # 2차 베지어 곡선 생성
    >>> control_points = np.array([[0, 0], [1, 2], [2, 0]])
    >>> curve = BezierCurve(control_points)
    >>> 
    >>> # 곡선 평가
    >>> t = np.linspace(0, 1, 100)
    >>> points = curve.evaluate(t)
    >>> 
    >>> # 미분 계산
    >>> derivatives = curve.derivative(t, order=1)
"""

__version__ = "1.0.0"
__author__ = "베지어 곡선 라이브러리"

# 주요 클래스들을 패키지 레벨에서 import 가능하도록 설정
from .bezier import BezierCurve
from .bezier_matrix_utils import (
    get_affine_integral_matrix,
    get_differentiation_matrix,
    get_elevation_matrix,
    get_bernstein_at_one
)

__all__ = [
    'BezierCurve',
    'get_affine_integral_matrix',
    'get_differentiation_matrix', 
    'get_elevation_matrix',
    'get_bernstein_at_one'
]