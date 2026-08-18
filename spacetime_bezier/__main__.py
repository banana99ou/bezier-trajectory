"""The one entrypoint: ``python3 -m spacetime_bezier``.

Launches the sandbox and solves nothing up front; ``--bake`` re-solves every
configuration instead. There used to be three ways in (this module,
``spacetime_bezier.io``, ``spacetime_bezier.sandbox``) landing on two different
default ports, which is how two sandboxes ended up live at once -- one of them
four days stale on a different interpreter.
"""

import sys

from .io import main

if __name__ == "__main__":
    sys.exit(main())
