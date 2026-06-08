"""궤도전이 기법 비교 벤치마크 패키지.

다른 프로젝트에서 사용 예:

    from orbit_transfer.benchmark import TransferBenchmark
    from orbit_transfer.types import TransferConfig

    config = TransferConfig(h0=400, delta_a=500, delta_i=5.0, T_max_normed=0.8)
    bench = TransferBenchmark(config)
    bench.run_all()
    bench.print_summary()
    bench.export_csv("results/comparison.csv")
    bench.export_trajectory_csv("results/")
    bench.export_figures("results/")
"""

from .benchmark import TransferBenchmark
from .result import BenchmarkResult
from .metrics import compute_metrics, add_metric
from .solvers import HohmannSolver, LambertSolver, CollocationSolver
from .cases import ComparisonCase, get_case, get_group, list_cases, CASE_REGISTRY

try:
    from .solvers import BladeSolver
except ImportError:
    BladeSolver = None  # type: ignore

try:
    from .solvers import BladeCollocationSolver
except ImportError:
    BladeCollocationSolver = None  # type: ignore

__all__ = [
    "TransferBenchmark",
    "BenchmarkResult",
    "compute_metrics",
    "add_metric",
    "HohmannSolver",
    "LambertSolver",
    "CollocationSolver",
    "BladeSolver",
    "ComparisonCase",
    "get_case",
    "get_group",
    "list_cases",
    "BladeCollocationSolver",
    "CASE_REGISTRY",
]
