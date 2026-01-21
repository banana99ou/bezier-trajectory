"""
Utility functions for font configuration and number formatting.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


def configure_custom_font(font_filename="NanumSquareR.otf"):
    """
    Configure Matplotlib to use a custom font if available.
    Uses fallback font for math symbols to fix negative sign rendering.
    """
    font_path = Path(__file__).resolve().parent.parent / font_filename
    if not font_path.is_file():
        return

    try:
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        
        # Use font with fallback to DejaVu Sans for better symbol rendering
        # This ensures minus signs and other symbols render correctly
        plt.rcParams["font.family"] = [font_name, "DejaVu Sans", "sans-serif"]
        
        # Use DejaVu Sans for math symbols to fix negative sign rendering
        plt.rcParams["mathtext.fontset"] = "dejavusans"
        plt.rcParams["mathtext.default"] = "regular"
        
        # Enable Unicode minus sign for better rendering in tick labels
        # This makes matplotlib use U+2212 (proper minus) instead of U+002D (hyphen-minus)
        plt.rcParams["axes.unicode_minus"] = True
        
        # The format_number() helper function ensures all manually formatted numbers
        # also use the proper Unicode minus sign (U+2212)
    except Exception:
        # Silently ignore font configuration errors to avoid breaking plots
        pass


def format_number(value, format_spec='.1f'):
    """
    Format a number with proper Unicode minus sign for better rendering.
    
    Args:
        value: Numeric value to format
        format_spec: Format specification (e.g., '.1f', '.2f')
    
    Returns:
        str: Formatted string with proper minus sign
    """
    if isinstance(value, (int, float)):
        if value < 0:
            # Use Unicode minus sign (U+2212) instead of hyphen-minus (U+002D)
            return '−' + format(abs(value), format_spec)
        else:
            return format(value, format_spec)
    return str(value)

