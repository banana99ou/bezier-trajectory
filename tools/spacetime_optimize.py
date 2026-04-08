"""
Compatibility CLI wrapper for the repo-root `spacetime_optimize.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spacetime_bezier.io import main


if __name__ == "__main__":
    main()
