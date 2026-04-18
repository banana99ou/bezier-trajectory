"""Launch the interactive sandbox when the package is invoked directly.

``python3 -m spacetime_bezier`` → same as ``python3 -m spacetime_bezier.sandbox``.
"""

from .sandbox import main

if __name__ == "__main__":
    main()
