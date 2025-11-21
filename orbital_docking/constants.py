"""
Orbital parameters and constants for the docking optimizer.
"""

# Orbital parameters (from Project_Spec.md)
ISS_ALTITUDE_KM = 423  # ISS orbit altitude in km AMSL
CHASER_ALTITUDE_KM = 300  # Chaser altitude in km AMSL
KOZ_ALTITUDE_KM = 100  # Keep Out Zone altitude in km AMSL

EARTH_RADIUS_KM = 6371.0  # Earth radius in km
KOZ_RADIUS = EARTH_RADIUS_KM + KOZ_ALTITUDE_KM  # KOZ radius from Earth center
ISS_RADIUS = EARTH_RADIUS_KM + ISS_ALTITUDE_KM
CHASER_RADIUS = EARTH_RADIUS_KM + CHASER_ALTITUDE_KM

# Gravitational parameters
EARTH_MU = 3.986004418e14  # m³/s²

# Scaling: use km as base unit
SCALE_FACTOR = 1e3  # 1 unit = 1 km
EARTH_MU_SCALED = EARTH_MU / (SCALE_FACTOR**3)  # Scaled for km

