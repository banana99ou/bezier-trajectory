# Rust QA Strategy

> **Superseded.** This document described a Python/Rust two-backend comparison, with
> Python (`trust-constr`) as the ground-truth oracle. **The Python optimizer has been
> removed.** There is no second backend to cross-validate against, and
> `spacetime_bezier/constraints.py` is not a reference implementation — see
> [`CLAUDE.md`](../CLAUDE.md) §Architecture.
>
> Do not build a verification harness against a Python solver. It does not exist.

## What verification means now

With one backend, correctness cannot come from agreement between two implementations.
It has to come from checks that could actually fail:

- **Geometry, against the true obstacle.** Sample the returned curve densely and measure
  clearance against the analytic tube `‖p_xy − p₀ − v·p_t‖ ≥ r`, independent of whatever
  half-spaces the QP was given. This is the only check that does not inherit the
  constraint builder's own assumptions.
- **Slack must gate.** A run that ends with `total_slack > 0` solved a relaxed problem.
  It is not a feasible result and must not produce a figure. See workstream **B7**.
- **Tests must be able to fail.** The current KOZ suite cannot: every case uses
  `vel = [0, 0]`, where the defect it would catch has exactly zero error, and
  `tests/unit/test_spacetime_constraints.py:65` asserts the buggy behavior outright.
  Any new constraint test needs a **moving** obstacle.
- **The Rust constraint builder has no tests at all.** `spacetime_constraints.rs` is the
  code that runs in production; the Python tests exercise a builder nothing uses.

## Still valid from the original document

Per-component parity against closed-form values (not against another solver):

- D/E/G matrices for degrees 2–6
- Segment matrices across N / n_seg combinations
- Bernstein basis and derivative weights at several tau values
- Gravity at a spread of positions

Edge cases worth covering: `N=2` (minimal degree for acceleration), `n_seg=1` (no
subdivision), velocity/acceleration boundary conditions.
