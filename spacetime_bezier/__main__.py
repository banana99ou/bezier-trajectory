"""The one entrypoint: ``python3 -m spacetime_bezier``.

Serves the frontend -- one page, port 8767, config panel and diagnostics drawer
over the same scene. It replaced the sandbox here on 2026-08-21.

``sandbox.py``, ``viewer.py`` and ``io.py`` are still on disk and still runnable
by module path; deleting them is a separate decision that has not been taken.
What changed is only which one this entrypoint starts. All of them bind 8767, so
the shared port keeps enforcing that exactly one is live -- there used to be
three ways in landing on two different default ports, which is how two sandboxes
ended up running at once, one of them four days stale on a different interpreter.
"""

import sys

from .frontend import main

if __name__ == "__main__":
    sys.exit(main())
