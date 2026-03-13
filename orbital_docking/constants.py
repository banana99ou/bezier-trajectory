"""
Orbital parameters and constants for the rendezvous optimizer.

Scenario: Progress-to-ISS fast-rendezvous inspired (simplified single-arc).
See Project_Spec.md for sourced mission values and modeling choices.
"""

# Orbital parameters (Progress-to-ISS inspired, see Project_Spec.md)
PROGRESS_START_ALTITUDE_KM = 245.0   # Circularized Progress-like parking orbit [km AMSL]
ISS_TARGET_ALTITUDE_KM = 400.0       # ISS-like target orbit [km AMSL]
KOZ_ALTITUDE_KM = 100.0              # Keep Out Zone altitude [km AMSL]

EARTH_RADIUS_KM = 6371.0  # Earth radius [km]
KOZ_RADIUS = EARTH_RADIUS_KM + KOZ_ALTITUDE_KM
ISS_RADIUS = EARTH_RADIUS_KM + ISS_TARGET_ALTITUDE_KM
CHASER_RADIUS = EARTH_RADIUS_KM + PROGRESS_START_ALTITUDE_KM

# Gravitational parameters
EARTH_MU = 3.986004418e14  # m³/s²

# Earth gravity field (z-axis aligned with ECI Z for this simplified demo)
EARTH_J2 = 1.08262668e-3  # [-]

# Scaling: use km as base unit
SCALE_FACTOR = 1e3  # 1 unit = 1 km
EARTH_MU_SCALED = EARTH_MU / (SCALE_FACTOR**3)  # km³/s²

# Fixed transfer time for time-scaling (seconds).
# Chosen so that endpoint-velocity tangent handles remain on the same order
# as the endpoint separation (~3500 km) for this Progress-inspired geometry.
# 25 min = 1500 s is a reasonable reduced-order single-arc transfer time.
TRANSFER_TIME_S = 1500.0