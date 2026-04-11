# T6 Dymos Protocol

## Frozen Framework

- Dymos version: `1.15.1`
- OpenMDAO version: `3.43.0`
- transcription: `Radau`, segments `12`, order `3`
- driver: `ScipyOptimizeDriver` with optimizer `SLSQP`

## Fairness Rule

- The Dymos phase, mesh, objective, path constraint, boundary conditions, and optimizer settings are identical between runs.
- Only the initial guess differs: naive Hermite vs Rust-backed matched Bezier warm start.

## Saved Audit Artifacts

- `artifacts/paper_artifacts/t6_dymos_experiment.json`
- `artifacts/paper_artifacts/t6_dymos_naive.sql`
- `artifacts/paper_artifacts/t6_dymos_warm.sql`
- `figures/t6_dymos_paths.html`
- `figures/t6_dymos_velocity.html`
- `doc/t6_dymos_validity_memo.md`

## Robustness Follow-up

- Predeclared grid only if the single-case experiment is valid but inconclusive.
- Vary phase lag over `20, 30, 40 deg` and transfer time over `1200, 1500, 1800 s`.
- Report paired success counts, median deltas, win/loss/tie counts, and bootstrap intervals if sample count is large enough.
