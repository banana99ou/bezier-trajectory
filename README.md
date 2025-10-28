# 베지어 곡선 계산 라이브러리

베지어 곡선의 수치 계산을 위한 Python 라이브러리입니다.

## 설치

```bash
pip install numpy scipy plotly
```

## 기본 사용법

```python
import numpy as np
from bezier import BezierCurve

# 베지어 곡선 생성
control_points = np.array([[0, 0], [1, 2], [2, 0]])
curve = BezierCurve(control_points)

# 곡선 평가
t = np.linspace(0, 1, 100)
points = curve.evaluate(t)

# 미분 계산
derivatives = curve.derivative(t, order=1)

# 적분 계산
integrated = curve.integrate()
```

## 주요 기능

### 곡선 연산
- **평가**: `curve.evaluate(t)` - 매개변수 t에서 곡선 좌표
- **미분**: `curve.derivative(t, order=n)` - n차 미분 계산
- **적분**: `curve.integrate()` - 아핀 적분 연산

### 곡선 변환
- **차수 승격**: `curve.elevate_degree()` - 곡선 차수 증가
- **제어점 접근**: `curve.get_control_points()` - 제어점 행렬 반환

### 행렬 연산
```python
from bezier_matrix_utils import get_affine_integral_matrix, get_differentiation_matrix

# 적분 행렬 계산
integral_matrix = get_affine_integral_matrix(degree=3)

# 미분 행렬 계산  
diff_matrix = get_differentiation_matrix(degree=3)
```

## 예제 실행

```bash
python examples/basic_usage.py      # 기본 연산 예제
python examples/integration_demo.py # 적분 연산 데모
```

## Version History

### v1.0-chorok (October 2025)
**Stable version used for 초록 (abstract) figure generation**

- **Commit**: `a877a33` (a877a33355c1404a99daaad8af1af617c11c1a5d)
- **File**: `examples/Bezier_Curve_Optimizer.py` (600 lines)
- **Purpose**: Generated figures for conference abstract submission
- **Date**: October 26-27, 2025
- **Key features**:
  - Bézier curve optimization with sphere avoidance
  - Segment-based linearization
  - Half-space constraint formulation
  - SLSQP-based iterative optimization
  - 3D visualization utilities

## 수학적 배경

- **베른슈타인 기저 함수**: 베지어 곡선의 수학적 기반
- **아핀 적분**: 적분 상수 항을 제거하는 적분 연산
- **차수 승격**: 곡선 형태 보존하며 제어점 개수 증가
