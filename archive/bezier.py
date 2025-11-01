import numpy as np
from scipy.special import comb
from typing import Union, List, Tuple, Optional


class BezierCurve:
    """
    일반적인 n차 베지어 곡선 클래스
    
    베지어 곡선의 수치 계산, 미분, 차수 승격 등의 선형 연산을 지원합니다.
    """
    
    def __init__(self, control_points: np.ndarray, degree: Optional[int] = None):
        """
        베지어 곡선 초기화
        
        Args:
            control_points: 제어점 배열 (N+1) x D 형태
                          N은 곡선의 차수, D는 차원
            degree: 곡선의 차수 (None이면 자동 계산)
        """
        self.control_points = np.array(control_points, dtype=float)
        
        if self.control_points.ndim == 1:
            self.control_points = self.control_points.reshape(-1, 1)
        
        if degree is None:
            self.degree = self.control_points.shape[0] - 1
        else:
            self.degree = degree
            if self.control_points.shape[0] != degree + 1:
                raise ValueError(f"제어점 개수({self.control_points.shape[0]})가 차수({degree})+1과 일치하지 않습니다.")
        
        self.dimension = self.control_points.shape[1]
        self._compute_matrices()
    
    def _compute_matrices(self):
        """미분 행렬과 차수 승격 행렬들을 미리 계산"""
        self._diff_matrix = self._compute_diff_matrix()
        self._elevation_matrix = self._compute_elevation_matrix()
        self._integral_matrix = self._compute_integral_matrix()
        self._affine_integral_matrix = self._compute_affine_integral_matrix()
    
    def _compute_diff_matrix(self) -> np.ndarray:
        """
        미분 행렬 계산
        
        Returns:
            N x (N+1) 크기의 미분 행렬
        """
        N = self.degree
        D = np.zeros((N, N + 1))
        
        for i in range(N):
            D[i, i] = -N
            D[i, i + 1] = N
        
        return D
    
    def _compute_elevation_matrix(self) -> np.ndarray:
        """
        차수 승격 행렬 계산 (N차 → N+1차)
        
        공식: Q_j = (j/(N+1)) * P_{j-1} + ((N+1-j)/(N+1)) * P_j
        
        Returns:
            (N+2) x (N+1) 크기의 차수 승격 행렬
        """
        N = self.degree
        E = np.zeros((N + 2, N + 1))
        
        # 경계 조건
        E[0, 0] = 1  # Q_0 = P_0
        E[N + 1, N] = 1  # Q_{N+1} = P_N
        
        # 내부 가중치: Q_j = (j/(N+1)) * P_{j-1} + ((N+1-j)/(N+1)) * P_j
        for j in range(1, N + 1):
            E[j, j - 1] = j / (N + 1)  # P_{j-1}의 계수
            E[j, j] = (N + 1 - j) / (N + 1)  # P_j의 계수
        
        return E
    
    def _compute_integral_matrix(self) -> np.ndarray:
        """
        올바른 적분 행렬 계산 (N차 → N+1차)
        
        공식: Q_0 = 0, Q_j = (1/(N+1)) * sum_{i=0}^{j-1} P_i for j=1~N+1
        
        Returns:
            (N+2) x (N+1) 크기의 적분 행렬
        """
        N = self.degree
        I = np.zeros((N + 2, N + 1))
        for j in range(1, N + 2):
            I[j, :j] = 1.0 / (N + 1)
        return I
    
    def _compute_affine_integral_matrix(self) -> np.ndarray:
        """
        아핀 적분 행렬 계산 (상수항 조정 포함)
        
        공식: Ĩ = I - 1_{n+2} * e_{n+2,1}^T * I
        
        Returns:
            (N+2) x (N+1) 크기의 아핀 적분 행렬
        """
        N = self.degree
        I = self._integral_matrix
        
        # e_{n+2,1} = [1, 0, 0, ..., 0]^T (첫 번째 표준순서기저)
        e_first = np.zeros(N + 2)
        e_first[0] = 1
        
        # 1_{n+2} = [1, 1, 1, ..., 1]^T (모든 성분이 1인 벡터)
        ones_vector = np.ones(N + 2)
        
        # Ĩ = I - 1_{n+2} * e_{n+2,1}^T * I
        # 연산 검토: np.outer(1_{n+2}, e_{n+2,1}^T * I) 부분이 맞는지 확인
        # e_first.T @ I는 (1, N+1) 크기의 벡터가 되어야 하며, ones_vector와 외적을 통해 (N+2, N+1) 크기의 행렬 생성
        affine_I = I - np.outer(ones_vector, e_first @ I)
        return affine_I
    
    def evaluate(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        베지어 곡선의 점 계산
        
        Args:
            t: 매개변수 (0 ≤ t ≤ 1)
            
        Returns:
            곡선 위의 점들
        """
        if isinstance(t, (int, float)):
            t = np.array([t])
        
        t = np.array(t)
        if np.any((t < 0) | (t > 1)):
            raise ValueError("매개변수 t는 0과 1 사이의 값이어야 합니다.")
        
        # 베르누슈타인 기저함수 계산
        N = self.degree
        result = np.zeros((len(t), self.dimension))
        
        for i in range(N + 1):
            # 베르누슈타인 다항식: B_{i,N}(t) = C(N,i) * t^i * (1-t)^(N-i)
            bernstein = comb(N, i) * (t ** i) * ((1 - t) ** (N - i))
            result += bernstein.reshape(-1, 1) * self.control_points[i]
        
        return result
    
    def evaluate_basis(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        베르누슈타인 기저 함수들 계산
        
        Args:
            t: 매개변수 (0 ≤ t ≤ 1)
            
        Returns:
            기저 함수 값들 (num_points x (degree+1))
        """
        if isinstance(t, (int, float)):
            t = np.array([t])
        
        t = np.array(t)
        if np.any((t < 0) | (t > 1)):
            raise ValueError("매개변수 t는 0과 1 사이의 값이어야 합니다.")
        
        N = self.degree
        num_points = len(t)
        basis_values = np.zeros((num_points, N + 1))
        
        for i in range(N + 1):
            # 베르누슈타인 다항식: B_{i,N}(t) = C(N,i) * t^i * (1-t)^(N-i)
            bernstein = comb(N, i) * (t ** i) * ((1 - t) ** (N - i))
            basis_values[:, i] = bernstein
        
        return basis_values
    
    def derivative(self, t: Union[float, np.ndarray], order: int = 1, elevate: bool = False, return_curve: bool = False):
        """
        베지어 곡선의 미분 계산
        
        베지어 곡선의 미분 공식:
        B'(t) = n * Σ(i=0 to n-1) (P_{i+1} - P_i) * B_{i,n-1}(t)
        
        Args:
            t: 매개변수 (0 ≤ t ≤ 1)
            order: 미분 차수 (기본값: 1)
            elevate: 차수 승격 포함 여부 (기본값: False)
            return_curve: True일 경우 (미분된 곡선의 점, 미분된 BezierCurve 객체) 튜플 반환
        Returns:
            미분된 곡선의 점들 또는 (미분된 곡선의 점들, 미분된 BezierCurve 객체)
        """
        if order == 0:
            if return_curve:
                return self.evaluate(t), self
            return self.evaluate(t)
        
        if order > self.degree:
            # 차수보다 높은 미분은 0
            if isinstance(t, (int, float)):
                zeros = np.zeros(self.dimension)
            else:
                zeros = np.zeros((len(t), self.dimension))
            if return_curve:
                return zeros, None
            return zeros
        
        # order번 미분을 위해 미분 행렬을 order번 적용
        current_points = self.control_points.copy()
        current_degree = self.degree
        
        for _ in range(order):
            if current_degree == 0:
                # 0차 곡선의 미분은 0
                if isinstance(t, (int, float)):
                    zeros = np.zeros(self.dimension)
                else:
                    zeros = np.zeros((len(t), self.dimension))
                if return_curve:
                    return zeros, None
                return zeros
            # 미분 행렬 D 계산
            D = np.zeros((current_degree, current_degree + 1))
            for i in range(current_degree):
                D[i, i] = -current_degree
                D[i, i + 1] = current_degree
            # 미분된 제어점 계산
            current_points = D @ current_points
            current_degree -= 1
        
        if elevate and order == 1:
            # 1차 미분에 대해서만 차수 승격 적용
            E = np.zeros((current_degree + 2, current_degree + 1))
            E[0, 0] = 1
            E[current_degree + 1, current_degree] = 1
            for j in range(1, current_degree + 1):
                E[j, j - 1] = j / (current_degree + 1)
                E[j, j] = (current_degree + 1 - j) / (current_degree + 1)
            current_points = E @ current_points
            current_degree += 1
        
        # 미분된 곡선 생성
        diff_curve = BezierCurve(current_points, current_degree)
        result = diff_curve.evaluate(t)
        if return_curve:
            return result, diff_curve
        return result
    
    def elevate_degree(self) -> 'BezierCurve':
        """
        차수 승격 (N차 → N+1차)
        
        Returns:
            승격된 베지어 곡선
        """
        new_points = self._elevation_matrix @ self.control_points
        return BezierCurve(new_points, self.degree + 1)
    
    def elevate_degree_by(self, steps: int) -> 'BezierCurve':
        """
        여러 단계 차수 승격
        
        Args:
            steps: 승격할 단계 수
            
        Returns:
            승격된 베지어 곡선
        """
        if steps < 0:
            raise ValueError("승격 단계는 음수가 될 수 없습니다.")
        
        result = self
        for _ in range(steps):
            result = result.elevate_degree()
        
        return result
    
    def get_control_points(self) -> np.ndarray:
        """제어점 반환"""
        return self.control_points.copy()
    
    def get_degree(self) -> int:
        """곡선의 차수 반환"""
        return self.degree
    
    def get_dimension(self) -> int:
        """곡선의 차원 반환"""
        return self.dimension
    
    def __repr__(self) -> str:
        return f"BezierCurve(degree={self.degree}, dimension={self.dimension}, control_points={self.control_points.shape})"

    def integrate_with_constant(self, constant: float = 0.0) -> 'BezierCurve':
        """
        상수항을 지정하면서 베지어 곡선 적분
        
        아핀 적분 연산을 사용하여 상수항을 원하는 값으로 지정하면서 적분합니다.
        공식: q_final = Ĩ * p + c * 1_{n+2}
        
        Args:
            constant: 적분 상수 (기본값: 0.0)
            
        Returns:
            적분된 베지어 곡선 (차수가 1 증가)
        """
        # 아핀 적분 행렬을 사용하여 제어점 변환
        new_points = self._affine_integral_matrix @ self.control_points
        
        # 상수항 조정: c * 1_{n+2} 더하기
        constant_vector = constant * np.ones(self.degree + 2)
        new_points += constant_vector.reshape(-1, 1)
        
        return BezierCurve(new_points, self.degree + 1)
    
    def integrate(self) -> 'BezierCurve':
        """
        베지어 곡선 적분 (상수항 = 0)
        
        Returns:
            적분된 베지어 곡선 (차수가 1 증가)
        """
        return self.integrate_with_constant(0.0)


def create_bezier_curve(control_points: Union[List, np.ndarray], degree: Optional[int] = None) -> BezierCurve:
    """
    베지어 곡선 생성 함수
    
    Args:
        control_points: 제어점들
        degree: 곡선의 차수 (None이면 자동 계산)
        
    Returns:
        베지어 곡선 객체
    """
    return BezierCurve(control_points, degree)


def bezier_curve_evaluate(t: Union[float, np.ndarray], control_points: np.ndarray) -> np.ndarray:
    """
    베지어 곡선의 점 계산 (함수형 인터페이스)
    
    Args:
        t: 매개변수 (0 ≤ t ≤ 1)
        control_points: 제어점 배열
        
    Returns:
        곡선 위의 점들
    """
    curve = BezierCurve(control_points)
    return curve.evaluate(t)


def bezier_curve_derivative(t: Union[float, np.ndarray], control_points: np.ndarray, order: int = 1) -> np.ndarray:
    """
    베지어 곡선의 미분 계산 (함수형 인터페이스)
    
    Args:
        t: 매개변수 (0 ≤ t ≤ 1)
        control_points: 제어점 배열
        order: 미분 차수
        
    Returns:
        미분된 곡선의 점들
    """
    curve = BezierCurve(control_points)
    return curve.derivative(t, order)


def bezier_curve_integrate(control_points: np.ndarray, constant: float = 0.0) -> np.ndarray:
    """
    베지어 곡선 적분 (함수형 인터페이스)
    
    Args:
        control_points: 제어점 배열
        constant: 적분 상수 (기본값: 0.0)
        
    Returns:
        적분된 곡선의 제어점들
    """
    curve = BezierCurve(control_points)
    integrated_curve = curve.integrate_with_constant(constant)
    return integrated_curve.get_control_points()


def bezier_curve_integrate_evaluate(t: Union[float, np.ndarray], control_points: np.ndarray, constant: float = 0.0) -> np.ndarray:
    """
    베지어 곡선 적분 후 점 계산 (함수형 인터페이스)
    
    Args:
        t: 매개변수 (0 ≤ t ≤ 1)
        control_points: 제어점 배열
        constant: 적분 상수 (기본값: 0.0)
        
    Returns:
        적분된 곡선 위의 점들
    """
    curve = BezierCurve(control_points)
    integrated_curve = curve.integrate_with_constant(constant)
    return integrated_curve.evaluate(t)
