"""Downstream two-pass comparison experiment.

Implements `doc/dcm_downstream_experiment_design.md`. Nothing in this package
modifies `dcm_baseline/` — the baseline is imported read-only and driven stage
by stage, which is how the protocol symmetry of design section 4.4 is obtained
without editing `TwoPassOptimizer`.
"""
