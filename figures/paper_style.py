"""Manuscript typography shared by every paper figure.

The export places each figure 6.8 in wide (papers/exports/*/export.lua) beside
10 pt body text set in Helvetica Neue, with NanumBarunGothic for Hangul and
Latin Modern for math (tools/export_professor_paper.py, export header). A figure
of width W is scaled by 6.8/W on the page, so text drawn at pt * W/6.8 prints at
pt -- the same face and size as the body, whatever size the figure is drawn at.

Call apply(width_in) after every import: orbital_docking.utils rewrites the
mathtext settings when it is imported.

Author decision, 2026-10-06: the LaTeX manuscript keeps these same fonts, so
figures and body text stay matched when the draft moves from the Pandoc export
to LaTeX. Change the fonts here and in the LaTeX preamble together.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

PAGE_WIDTH_IN = 6.8
TEXT_PT = 9.0       # labels and titles, one step under the body like the captions
SMALL_PT = 8.0      # tick labels


def apply(width_in=PAGE_WIDTH_IN):
    """Set the manuscript fonts; return the factor that turns printed pt into drawn pt."""
    font_manager.fontManager.addfont("/System/Library/Fonts/HelveticaNeue.ttc")
    font_manager.fontManager.addfont(str(Path.home() / "Library/Fonts/NanumBarunGothic.ttf"))
    k = width_in / PAGE_WIDTH_IN
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "NanumBarunGothic"],
        "mathtext.fontset": "cm",
        "mathtext.default": "it",
        "font.size": TEXT_PT * k,
        "axes.titlesize": TEXT_PT * k,
        "axes.labelsize": TEXT_PT * k,
        "xtick.labelsize": SMALL_PT * k,
        "ytick.labelsize": SMALL_PT * k,
        "legend.fontsize": TEXT_PT * k,
        "legend.title_fontsize": TEXT_PT * k,
    })
    return k
