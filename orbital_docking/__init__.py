"""
Orbital Docking Optimizer using Bézier Curves

This package implements an optimization framework for designing orbital docking trajectories
using Bézier curves. The optimizer finds trajectories that minimize the difference between
geometric acceleration and gravitational acceleration while satisfying Keep Out Zone (KOZ)
constraints.
"""

from .bezier import BezierCurve, get_D_matrix, get_E_matrix
from .de_casteljau import (
    de_casteljau_split_1d,
    de_casteljau_split_matrices,
    segment_matrices_equal_params
)
from .constraints import build_koz_constraints, build_boundary_constraints
from .optimization import (
    optimize_orbital_docking,
    optimize_all_segment_counts,
    generate_initial_control_points
)
from .visualization import (
    add_wire_sphere,
    add_earth_sphere,
    set_axes_equal_around,
    set_isometric,
    beautify_3d_axes,
    plot_segments_gradient,
    create_trajectory_comparison_figure,
    create_performance_figure,
    create_acceleration_figure,
    create_time_vs_order_figure,
    compute_profile_ylims
)
from .cache import (
    get_cache_key,
    get_cache_path,
    load_from_cache,
    save_to_cache,
    clear_cache
)
from .utils import configure_custom_font, format_number
from . import constants

# Configure font on import
configure_custom_font()

__all__ = [
    # Core classes
    'BezierCurve',
    
    # Matrix functions
    'get_D_matrix',
    'get_E_matrix',
    
    # De Casteljau functions
    'de_casteljau_split_1d',
    'de_casteljau_split_matrices',
    'segment_matrices_equal_params',
    
    # Constraint functions
    'build_koz_constraints',
    'build_boundary_constraints',
    
    # Optimization functions
    'optimize_orbital_docking',
    'optimize_all_segment_counts',
    'generate_initial_control_points',
    
    # Visualization functions
    'add_wire_sphere',
    'add_earth_sphere',
    'set_axes_equal_around',
    'set_isometric',
    'beautify_3d_axes',
    'plot_segments_gradient',
    'create_trajectory_comparison_figure',
    'create_performance_figure',
    'create_acceleration_figure',
    'create_time_vs_order_figure',
    'compute_profile_ylims',
    
    # Cache functions
    'get_cache_key',
    'get_cache_path',
    'load_from_cache',
    'save_to_cache',
    'clear_cache',
    
    # Utility functions
    'configure_custom_font',
    'format_number',
    
    # Constants module
    'constants',
]

__version__ = "1.0.0"

