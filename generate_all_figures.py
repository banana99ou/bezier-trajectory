import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(description: str, *cmd: str) -> None:
    """Run a single step and print a short description."""
    print(f"\n=== {description} ===")
    print(">>", " ".join(cmd))
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def main() -> None:
    """
    One-click script to regenerate all main figures for the project.

    This is a thin orchestrator that simply calls the existing scripts:
    - Orbital_Docking_Optimizer.py                (main optimization & figures)
    - figure/scnario_figure.py                    (scenario / expectation figure)
    - figure/constraint_linearization_figures.py  (KOZ / constraint linearization demo)
    - archive/Bezier_Curve_Optimizer_legacy.py    (legacy sphere-avoidance figure)
    """

    # Main orbital docking optimization & figures (saved under figure/figures/)
    # To disable caching, you can add "--no-cache" below.
    run_step("Orbital docking optimizer (main figures)",
             "Orbital_Docking_Optimizer.py")

    # Scenario / expectation figure: orbital_docking_expectation.png
    run_step("Orbital docking scenario / expectation figure",
             "figure/scnario_figure.py")

    # KOZ / constraint linearization illustrations (3D + 2D).
    # Currently these are shown interactively; saving is handled inside
    # figure/constraint_linearization_figures.py if desired.
    run_step("Constraint linearization / KOZ illustration",
             "figure/constraint_linearization_figures.py")

    # Legacy Bézier optimizer used for initial paper figures.
    # This opens a 2×3 subplot window; saving is controlled inside
    # archive/Bezier_Curve_Optimizer_legacy.py.
    run_step("Legacy Bézier curve optimizer (initial paper figure)",
             "archive/Bezier_Curve_Optimizer_legacy.py")


if __name__ == "__main__":
    main()



