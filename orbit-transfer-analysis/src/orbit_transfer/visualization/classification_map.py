"""분류 결과 시각화 (2D/3D scatter)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_classification_2d(X, y, dim1=0, dim2=1, dim_labels=None, ax=None):
    """2D 분류 지도 (scatter).

    Args:
        X: (N, d) 파라미터 (정규화 또는 물리)
        y: (N,) 클래스 라벨
        dim1, dim2: 표시할 차원 인덱스
        dim_labels: 축 라벨 리스트
    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    cmap = ListedColormap(['#2196F3', '#FF9800', '#F44336'])
    class_names = ['Unimodal', 'Bimodal', 'Multimodal']

    for c in range(3):
        mask = y == c
        if np.any(mask):
            ax.scatter(X[mask, dim1], X[mask, dim2], c=[cmap(c)],
                       label=class_names[c], alpha=0.7, s=20)

    if dim_labels:
        ax.set_xlabel(dim_labels[dim1])
        ax.set_ylabel(dim_labels[dim2])
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_classification_3d(X, y, dim_labels=None, ax=None):
    """3D 분류 지도.

    Args:
        X: (N, d) 파라미터 (최소 3차원)
        y: (N,) 클래스 라벨
        dim_labels: 축 라벨 리스트
    Returns:
        fig, ax
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    cmap = ListedColormap(['#2196F3', '#FF9800', '#F44336'])
    class_names = ['Unimodal', 'Bimodal', 'Multimodal']

    for c in range(3):
        mask = y == c
        if np.any(mask):
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                       c=[cmap(c)], label=class_names[c], alpha=0.7, s=20)

    if dim_labels:
        ax.set_xlabel(dim_labels[0])
        ax.set_ylabel(dim_labels[1])
        ax.set_zlabel(dim_labels[2])
    ax.legend()
    return fig, ax
