#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
베지어 곡선 기본 사용법 예제
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 패키지 루트를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bezier import BezierCurve


def basic_2d_example():
    """기본 2D 베지어 곡선 예제"""
    print("=== 기본 2D 베지어 곡선 예제 ===")
    
    # 2차 베지어 곡선의 제어점
    control_points = np.array([
        [0, 0],    # P0 (시작점)
        [2, 3],    # P1 (제어점)
        [4, 0],     # P2 
        [5, 1]      # P3 
    ])
    
    # 베지어 곡선 생성
    curve = BezierCurve(control_points)
    print(f"곡선 차수: {curve.get_degree()}")
    print(f"곡선 차원: {curve.get_dimension()}")
    
    # 곡선을 100개 점으로 평가
    t = np.linspace(0, 1, 100)
    curve_points = curve.evaluate(t)
    
    # 시각화
    fig = go.Figure()
    
    # 베지어 곡선
    fig.add_trace(go.Scatter(
        x=curve_points[:, 0], 
        y=curve_points[:, 1],
        mode='lines',
        name='베지어 곡선',
        line=dict(color='blue', width=3)
    ))
    
    # 제어점
    fig.add_trace(go.Scatter(
        x=control_points[:, 0], 
        y=control_points[:, 1],
        mode='markers+lines',
        name='제어점',
        line=dict(color='red', dash='dash'),
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title="2차 베지어 곡선",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
        width=600,
        height=500
    )
    
    fig.show()


def derivative_example():
    """베지어 곡선 미분 예제"""
    print("\n=== 베지어 곡선 미분 예제 ===")
    
    # 3차 베지어 곡선의 제어점
    control_points = np.array([
        [0, 0],
        [1, 2], 
        [3, 2],
        [4, 0],
        [5, 2]
    ])
    
    curve = BezierCurve(control_points)
    t = np.linspace(0, 1, 100)
    
    # 원곡선과 1차, 2차 미분 계산
    original = curve.evaluate(t)
    first_deriv = curve.derivative(t, order=1)
    second_deriv = curve.derivative(t, order=2)
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('원곡선 (x-y)', '1차 미분 X성분', '1차 미분 Y성분',
                       '원곡선 성분별', '2차 미분 X성분', '2차 미분 Y성분')
    )
    
    # 원곡선 (x-y 평면)
    fig.add_trace(go.Scatter(
        x=original[:, 0], y=original[:, 1],
        mode='lines', name='곡선',
        line=dict(color='blue', width=3),
        showlegend=False
    ), row=1, col=1)
    
    # 1차 미분 X성분 (t vs dx/dt)
    fig.add_trace(go.Scatter(
        x=t, y=first_deriv[:, 0],
        mode='lines', name='dx/dt',
        line=dict(color='green', width=2),
        showlegend=False
    ), row=1, col=2)
    
    # 1차 미분 Y성분 (t vs dy/dt)  
    fig.add_trace(go.Scatter(
        x=t, y=first_deriv[:, 1],
        mode='lines', name='dy/dt',
        line=dict(color='green', width=2),
        showlegend=False
    ), row=1, col=3)
    
    # 원곡선 X성분 (t vs x)
    fig.add_trace(go.Scatter(
        x=t, y=original[:, 0],
        mode='lines', name='x(t)',
        line=dict(color='blue', width=2),
        showlegend=False
    ), row=2, col=1)
    
    # 원곡선 Y성분 (t vs y)
    fig.add_trace(go.Scatter(
        x=t, y=original[:, 1],
        mode='lines', name='y(t)',
        line=dict(color='cyan', width=2),
        showlegend=False
    ), row=2, col=1)
    
    # 2차 미분 X성분 (t vs d²x/dt²)
    fig.add_trace(go.Scatter(
        x=t, y=second_deriv[:, 0],
        mode='lines', name='d²x/dt²',
        line=dict(color='red', width=2),
        showlegend=False
    ), row=2, col=2)
    
    # 2차 미분 Y성분 (t vs d²y/dt²)
    fig.add_trace(go.Scatter(
        x=t, y=second_deriv[:, 1],
        mode='lines', name='d²y/dt²',
        line=dict(color='red', width=2),
        showlegend=False
    ), row=2, col=3)
    
    # 축 레이블 업데이트
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="t", row=1, col=2)
    fig.update_yaxes(title_text="dx/dt", row=1, col=2)
    fig.update_xaxes(title_text="t", row=1, col=3)
    fig.update_yaxes(title_text="dy/dt", row=1, col=3)
    fig.update_xaxes(title_text="t", row=2, col=1)
    fig.update_yaxes(title_text="값", row=2, col=1)
    fig.update_xaxes(title_text="t", row=2, col=2)
    fig.update_yaxes(title_text="d²x/dt²", row=2, col=2)
    fig.update_xaxes(title_text="t", row=2, col=3)
    fig.update_yaxes(title_text="d²y/dt²", row=2, col=3)
    
    fig.update_layout(
        title="베지어 곡선과 미분들",
        showlegend=True,
        height=600
    )
    
    fig.show()


def integration_example():
    """베지어 곡선 적분 예제"""
    print("\n=== 베지어 곡선 적분 예제 ===")
    
    # 간단한 2차 베지어 곡선
    control_points = np.array([[1, 1], [2, 3], [3, 1]])
    curve = BezierCurve(control_points)
    
    # 다양한 적분 상수로 적분
    integrated_c0 = curve.integrate(c0=0.0)
    integrated_c1 = curve.integrate(c0=1.0)
    integrated_c2 = curve.integrate(c0=-0.5)
    
    print(f"원본 곡선 차수: {curve.get_degree()}")
    print(f"적분된 곡선 차수: {integrated_c0.get_degree()}")
    
    # 평가점
    t = np.linspace(0, 1, 100)
    
    # 시각화
    fig = go.Figure()
    
    # 원본 곡선
    original_points = curve.evaluate(t)
    fig.add_trace(go.Scatter(
        x=original_points[:, 0], y=original_points[:, 1],
        mode='lines', name='원본 곡선',
        line=dict(color='blue', width=3)
    ))
    
    # 적분된 곡선들
    colors = ['red', 'green', 'purple']
    constants = [0.0, 1.0, -0.5]
    curves = [integrated_c0, integrated_c1, integrated_c2]
    
    for i, (c, color, const) in enumerate(zip(curves, colors, constants)):
        points = c.evaluate(t)
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines', name=f'적분 (c₀={const})',
            line=dict(color=color, width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="베지어 곡선 적분 (다양한 적분상수)",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
        width=700,
        height=500
    )
    
    fig.show()


if __name__ == "__main__":
    basic_2d_example()
    derivative_example()
    integration_example()