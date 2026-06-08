"""통계 시각화 (분포, 수렴 이력, ARD 길이 척도)."""

import numpy as np
import matplotlib.pyplot as plt


def plot_class_distribution(y, ax=None):
    """클래스 분포 막대 그래프.

    Args:
        y: (N,) 클래스 라벨 배열
    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    classes = [0, 1, 2]
    counts = [np.sum(y == c) for c in classes]
    labels = ['Unimodal', 'Bimodal', 'Multimodal']
    colors = ['#2196F3', '#FF9800', '#F44336']
    ax.bar(labels, counts, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Profile Classification Distribution')
    return fig, ax


def plot_convergence_history(entropies, ax=None):
    """수렴 이력 (최대 엔트로피 vs 반복).

    Args:
        entropies: 반복별 최대 엔트로피 리스트/배열
    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.plot(entropies, 'o-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Entropy')
    ax.set_title('Adaptive Sampling Convergence')
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_ard_lengthscales(length_scales, param_names=None, ax=None):
    """ARD 길이 척도 막대 그래프.

    Args:
        length_scales: 길이 척도 배열
        param_names: 파라미터 이름 리스트
    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    if param_names is None:
        param_names = [f'dim {i}' for i in range(len(length_scales))]

    ax.bar(param_names, length_scales)
    ax.set_ylabel('Length Scale')
    ax.set_title('ARD Length Scales')
    return fig, ax
