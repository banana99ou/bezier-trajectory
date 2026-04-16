"""IPOPT 솔버 설정 유틸리티."""

from ..config import IPOPT_OPTIONS_PASS1, IPOPT_OPTIONS_PASS2


def get_pass1_options(**overrides):
    """Pass 1 (Hermite-Simpson) IPOPT 옵션 반환.

    Args:
        **overrides: 기본값을 덮어쓸 옵션

    Returns:
        dict: IPOPT 옵션
    """
    opts = dict(IPOPT_OPTIONS_PASS1)
    opts.update(overrides)
    return opts


def get_pass2_options(**overrides):
    """Pass 2 (Multi-Phase LGL) IPOPT 옵션 반환.

    Args:
        **overrides: 기본값을 덮어쓸 옵션

    Returns:
        dict: IPOPT 옵션
    """
    opts = dict(IPOPT_OPTIONS_PASS2)
    opts.update(overrides)
    return opts
