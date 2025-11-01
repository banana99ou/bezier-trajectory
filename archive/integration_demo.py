#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
베지어 곡선 적분 연산 데모
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 패키지 루트를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bezier import BezierCurve


def integration_demo():
    """베지어 곡선 적분 기능 데모"""
    print("=== 베지어 곡선 적분 데모 ===")
    
    # 2차 베지어 곡선 생성
    control_points = np.array([
        [0, 1],
        [1, 2], 
        [2, 0]
    ])
    
    curve = BezierCurve(control_points)
    print(f"원본 곡선 차수: {curve.get_degree()}")
    print(f"원본 제어점:\n{curve.get_control_points()}")
    
    # 다양한 적분 상수로 적분
    constants = [0.0, 1.0, -0.5]
    integrated_curves = []
    
    for c in constants:
        integrated = curve.integrate_with_constant(c)
        integrated_curves.append({
            'constant': c,
            'curve': integrated
        })
        print(f"\n적분 상수 c={c}인 경우:")
        print(f"적분된 곡선 차수: {integrated.get_degree()}")
        print(f"적분된 제어점:\n{integrated.get_control_points()}")
    
    # 시각화
    t = np.linspace(0, 1, 100)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '원본 곡선 (x-y)', 
            '적분된 곡선들 (x-y)',
            '원본 곡선 성분별 (t vs x,y)',
            '적분된 곡선들 성분별'
        )
    )
    
    # 원본 곡선 (x-y 평면)
    original_points = curve.evaluate(t)
    fig.add_trace(go.Scatter(
        x=original_points[:, 0], 
        y=original_points[:, 1],
        mode='lines+markers',
        name='원본 곡선',
        line=dict(color='blue', width=3),
        marker=dict(size=3),
        showlegend=True
    ), row=1, col=1)
    
    # 제어점 표시
    fig.add_trace(go.Scatter(
        x=control_points[:, 0], 
        y=control_points[:, 1],
        mode='markers+lines',
        name='원본 제어점',
        line=dict(color='blue', dash='dash'),
        marker=dict(color='blue', size=8),
        showlegend=False
    ), row=1, col=1)
    
    # 적분된 곡선들 (x-y 평면)
    colors = ['red', 'green', 'purple']
    for i, data in enumerate(integrated_curves):
        points = data['curve'].evaluate(t)
        c = data['constant']
        
        fig.add_trace(go.Scatter(
            x=points[:, 0], 
            y=points[:, 1],
            mode='lines',
            name=f'적분 (c={c})',
            line=dict(color=colors[i], width=2),
            showlegend=True
        ), row=1, col=2)
    
    # 원본 곡선 성분별 (t vs x, y)
    fig.add_trace(go.Scatter(
        x=t, y=original_points[:, 0],
        mode='lines', name='원본 x(t)',
        line=dict(color='blue', width=2),
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=t, y=original_points[:, 1],
        mode='lines', name='원본 y(t)',
        line=dict(color='cyan', width=2),
        showlegend=False
    ), row=2, col=1)
    
    # 적분된 곡선들 성분별
    for i, data in enumerate(integrated_curves):
        points = data['curve'].evaluate(t)
        c = data['constant']
        color = colors[i]
        
        # X 성분
        fig.add_trace(go.Scatter(
            x=t, y=points[:, 0],
            mode='lines', name=f'적분 x(t) c={c}',
            line=dict(color=color, width=2),
            showlegend=False
        ), row=2, col=2)
        
        # Y 성분 (점선으로 구분)
        fig.add_trace(go.Scatter(
            x=t, y=points[:, 1],
            mode='lines', name=f'적분 y(t) c={c}',
            line=dict(color=color, width=2, dash='dash'),
            showlegend=False
        ), row=2, col=2)
    
    # 축 레이블 설정
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Y", row=1, col=2)
    fig.update_xaxes(title_text="t", row=2, col=1)
    fig.update_yaxes(title_text="값", row=2, col=1)
    fig.update_xaxes(title_text="t", row=2, col=2)
    fig.update_yaxes(title_text="값", row=2, col=2)
    
    fig.update_layout(
        title="베지어 곡선 적분 데모",
        height=700,
        showlegend=True
    )
    
    fig.show()


def affine_integration_demo():
    """아핀 적분 연산 데모"""
    print("\n=== 아핀 적분 연산 데모 ===")
    
    # 1차 베지어 곡선 (직선)
    control_points = np.array([
        [0, 0],
        [2, 1]
    ])
    
    line = BezierCurve(control_points)
    print(f"직선 제어점: {line.get_control_points()}")
    
    # 일반 적분과 아핀 적분 비교
    normal_integrated = line.integrate_with_constant(0.0)
    affine_integrated = line.integrate()  # 아핀 적분 (상수항 자동 조정)
    
    print(f"일반 적분 제어점:\n{normal_integrated.get_control_points()}")
    print(f"아핀 적분 제어점:\n{affine_integrated.get_control_points()}")
    
    # t=0에서의 값 비교
    t_test = np.array([0.0, 0.5, 1.0])
    
    print(f"\n[t=0, 0.5, 1에서의 값 비교]")
    print(f"원본 직선: {line.evaluate(t_test)}")
    print(f"일반 적분: {normal_integrated.evaluate(t_test)}")  
    print(f"아핀 적분: {affine_integrated.evaluate(t_test)}")


if __name__ == "__main__":
    integration_demo()
    affine_integration_demo()