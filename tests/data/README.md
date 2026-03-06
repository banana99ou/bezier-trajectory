# Test data

## Golden run (regression)

`golden_run.json` stores reference values for a single fixed optimizer run used by
`tests/regression/test_golden_run.py`. Scenario: N=3, n_seg=4, `P_init` from conftest,
no velocity/acceleration BC, max_iter=10, tol=1e-6.

### Updating the golden file

When the **intended** behavior of the optimizer changes (e.g. formulation, constants, or
tolerances), update the golden values so the regression test reflects the new baseline:

1. Set the environment variable **`UPDATE_GOLDEN=1`**.
2. Run the golden regression test (or the full test suite):
   ```bash
   UPDATE_GOLDEN=1 python -m pytest tests/regression/test_golden_run.py -v
   ```
3. The test will run the optimizer once and **overwrite** `golden_run.json` with the new
   `cost_true_energy` and `min_radius`.
4. Unset `UPDATE_GOLDEN` and commit the updated `golden_run.json` if the change was intentional.

Do **not** update the golden file to make a failing test pass unless you have explicitly
changed the optimizer or scenario and accept the new baseline.
