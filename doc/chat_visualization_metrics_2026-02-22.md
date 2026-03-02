# Chat Record: Visualization Metric Alignment

Date: 2026-02-22

## User Request

- Review docs and `orbital_docking/visualization.py` and identify the obvious issue.
- Then implement a fix so plotted metrics align with the optimizer objective.

## Key Discussion Points

1. Initial obvious issue in visualization:
   - In `create_time_vs_order_figure`, text info was computed but not rendered.
2. More important issue (user-identified and confirmed):
   - Plots and titles emphasized max/mean control acceleration.
   - Optimizer objective is control-effort (least-squares energy style), so reporting should align with that objective.
3. Unit-consistent physical reporting:
   - Use transfer-time scaling (`TRANSFER_TIME_S`) and convert to physical units.

## Agreed Metric Direction

- Use objective-consistent metrics derived from `cost_true_energy`.
- Report in physical units:
  - RMS control acceleration: `m/s^2`
  - L2 control effort over time: `m^2/s^3`

Conversions used:

- Let `J_tau = cost_true_energy` (tau-domain objective).
- `RMS[m/s^2] = sqrt(J_tau) * 1e3`
- `L2_effort[m^2/s^3] = J_tau * T * 1e6`, where `T = T_transfer_s` (fallback `TRANSFER_TIME_S`).

## Implemented Changes

File updated: `orbital_docking/visualization.py`

- Added helper:
  - `control_effort_metrics(info)` to compute RMS and L2 effort from optimizer info.
- Updated `create_trajectory_comparison_figure`:
  - Replaced max-control-accel title metric with RMS control acceleration.
- Updated `create_performance_figure`:
  - Replaced curve data from sampled max accel to RMS from objective.
  - Updated labels/title to RMS wording.
- Updated `create_acceleration_figure`:
  - Replaced max/mean summary with RMS + L2 effort summary.
  - Updated title accordingly.
- Updated `create_time_vs_order_figure`:
  - Replaced max accel text with RMS/L2 text from `info`.
  - Actually renders the text onto the figure.

## Validation

- Lint check on `orbital_docking/visualization.py`: no linter errors.

## Outcome

- Visualization now reports metrics consistent with the optimization objective and physical time scaling.
