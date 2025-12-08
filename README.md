## Bézier Trajectory – Orbital Docking Optimizer

An orbital docking trajectory optimizer based on Bézier curves. The code finds fuel‑efficient paths for a chaser satellite to rendezvous with the ISS while respecting a spherical Keep‑Out Zone (KOZ), and generates all figures used in the paper and slides.

## Requirements

- **Python**: 3.7 or newer  
- **Dependencies**: install via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Run all commands from the project root directory (`bezier-trajectory`).

## One‑click: generate all figures

- **Script**: `generate_all_figures.py`  
- **What it does**: sequentially calls the existing scripts to regenerate the main figures.

```bash
python generate_all_figures.py
```

This will:
- **Run** `Orbital_Docking_Optimizer.py` to:
  - Optimize docking trajectories for different segment counts and curve orders
  - Save main figures under `figure/figures/`, including:
    - `comparison_N2.png`, `comparison_N3.png`, `comparison_N4.png`
    - `performance_N2.png`, `performance_N3.png`, `performance_N4.png`
    - `accel_profiles_N2_seg{2,4,8,16,32,64}.png`
    - `time_vs_order.png`
- **Run** `figure/scnario_figure.py` to create the orbital docking scenario / expectation figure:
  - `orbital_docking_expectation.png`
- **Run** `figure/constraint_linearization_figures.py` to show KOZ / constraint‑linearization demo figures (3D + 2D).
- **Run** `archive/Bezier_Curve_Optimizer_legacy.py` to reproduce the legacy sphere‑avoidance optimization figure (used in the initial paper).

> Note: some of the illustration scripts are primarily interactive and call `plt.show()`. To save additional static images, uncomment or add `plt.savefig(...)` lines inside those modules if needed.

## Main scripts and what they do

- **`Orbital_Docking_Optimizer.py`**  
  - Full orbital docking optimizer using Bézier curves (N=2,3,4).  
  - Minimizes the difference between geometric and gravitational acceleration with KOZ constraints.  
  - Saves the main optimization, performance, and profile figures into `figure/figures/`.  
  - Usage:
    - **Default (cache enabled)**:
      ```bash
      python Orbital_Docking_Optimizer.py
      ```
    - **Force recomputation (no cache)**:
      ```bash
      python Orbital_Docking_Optimizer.py --no-cache
      ```

- **`figure/scnario_figure.py`**  
  - Generates an “expectation” figure showing Earth, KOZ, chaser, and ISS positions (3D + 2D layout).  
  - Saves `orbital_docking_expectation.png` (in the current working directory).  
  - Usage:
    ```bash
    python figure/scnario_figure.py
    ```

- **`figure/constraint_linearization_figures.py`**  
  - Visualizes how nonlinear KOZ constraints are turned into supporting half‑spaces using Bézier segment subdivision.  
  - Provides:
    - 3D illustrations of a curve and sphere with segment‑wise constraint planes  
    - 2D KOZ linearization plots showing violating segments and corrected segments  
  - When run as a script, it currently focuses on the 2D KOZ linearization figure and shows it interactively.  
  - Usage:
    ```bash
    python figure/constraint_linearization_figures.py
    ```

- **`archive/Bezier_Curve_Optimizer_legacy.py`**  
  - Legacy implementation used to generate figures for the initial paper (sphere‑avoidance with a single Bézier curve).  
  - Shows a 2×3 grid of optimized curves with different segment counts.  
  - Contains a commented `plt.savefig("bezier_outside_sphere_2x3.png", ...)` line you can uncomment to save the legacy figure.  
  - Usage:
    ```bash
    python archive/Bezier_Curve_Optimizer_legacy.py
    ```

- **`archive/basic_usage.py`, `archive/bezier.py`, `archive/bezier_matrix_utils.py`, `archive/integration_demo.py`**  
  - Earlier, more didactic utilities and demos for Bézier curves and integration.  
  - Kept for reference; not required for the main orbital docking results.

- **`generate_all_figures.py`**  
  - Thin “orchestrator” that runs all major scripts in sequence for one‑click reproduction of figures.  
  - Recommended entry point for reproducing all results.

## Directories

- **`figure/figures/`**  
  - Output directory for all generated figures (PNG/PDF) from the main optimizer and some helper scripts.

- **`cache/`**  
  - Stores pickled optimization results used by `Orbital_Docking_Optimizer.py` for faster repeated runs.

- **`archive/`**  
  - Legacy code and figure generators corresponding to the initial paper versions.  


