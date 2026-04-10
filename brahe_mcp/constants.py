import brahe
from loguru import logger

from brahe_mcp.server import mcp

CONSTANTS: dict[str, dict] = {
    # Physical constants
    "R_EARTH": {"value": brahe.R_EARTH, "unit": "m", "description": "Earth equatorial radius", "category": "physical"},
    "R_MOON": {"value": brahe.R_MOON, "unit": "m", "description": "Moon mean radius", "category": "physical"},
    "R_SUN": {"value": brahe.R_SUN, "unit": "m", "description": "Sun mean radius", "category": "physical"},
    "GM_EARTH": {"value": brahe.GM_EARTH, "unit": "m^3/s^2", "description": "Earth gravitational parameter", "category": "physical"},
    "GM_SUN": {"value": brahe.GM_SUN, "unit": "m^3/s^2", "description": "Sun gravitational parameter", "category": "physical"},
    "GM_MOON": {"value": brahe.GM_MOON, "unit": "m^3/s^2", "description": "Moon gravitational parameter", "category": "physical"},
    "GM_MERCURY": {"value": brahe.GM_MERCURY, "unit": "m^3/s^2", "description": "Mercury gravitational parameter", "category": "physical"},
    "GM_VENUS": {"value": brahe.GM_VENUS, "unit": "m^3/s^2", "description": "Venus gravitational parameter", "category": "physical"},
    "GM_MARS": {"value": brahe.GM_MARS, "unit": "m^3/s^2", "description": "Mars gravitational parameter", "category": "physical"},
    "GM_JUPITER": {"value": brahe.GM_JUPITER, "unit": "m^3/s^2", "description": "Jupiter gravitational parameter", "category": "physical"},
    "GM_SATURN": {"value": brahe.GM_SATURN, "unit": "m^3/s^2", "description": "Saturn gravitational parameter", "category": "physical"},
    "GM_URANUS": {"value": brahe.GM_URANUS, "unit": "m^3/s^2", "description": "Uranus gravitational parameter", "category": "physical"},
    "GM_NEPTUNE": {"value": brahe.GM_NEPTUNE, "unit": "m^3/s^2", "description": "Neptune gravitational parameter", "category": "physical"},
    "GM_PLUTO": {"value": brahe.GM_PLUTO, "unit": "m^3/s^2", "description": "Pluto gravitational parameter", "category": "physical"},
    "WGS84_A": {"value": brahe.WGS84_A, "unit": "m", "description": "WGS84 semi-major axis", "category": "physical"},
    "WGS84_F": {"value": brahe.WGS84_F, "unit": "", "description": "WGS84 flattening", "category": "physical"},
    "J2_EARTH": {"value": brahe.J2_EARTH, "unit": "", "description": "Earth J2 oblateness coefficient", "category": "physical"},
    "ECC_EARTH": {"value": brahe.ECC_EARTH, "unit": "", "description": "Earth orbital eccentricity", "category": "physical"},
    "OMEGA_EARTH": {"value": brahe.OMEGA_EARTH, "unit": "rad/s", "description": "Earth rotation rate", "category": "physical"},
    "AU": {"value": brahe.AU, "unit": "m", "description": "Astronomical unit", "category": "physical"},
    "C_LIGHT": {"value": brahe.C_LIGHT, "unit": "m/s", "description": "Speed of light in vacuum", "category": "physical"},
    "P_SUN": {"value": brahe.P_SUN, "unit": "N/m^2", "description": "Solar radiation pressure at 1 AU", "category": "physical"},
    # Math constants
    "DEG2RAD": {"value": brahe.DEG2RAD, "unit": "rad/deg", "description": "Degrees to radians conversion factor", "category": "math"},
    "RAD2DEG": {"value": brahe.RAD2DEG, "unit": "deg/rad", "description": "Radians to degrees conversion factor", "category": "math"},
    "AS2RAD": {"value": brahe.AS2RAD, "unit": "rad/arcsec", "description": "Arcseconds to radians conversion factor", "category": "math"},
    "RAD2AS": {"value": brahe.RAD2AS, "unit": "arcsec/rad", "description": "Radians to arcseconds conversion factor", "category": "math"},
    # Time constants
    "MJD_ZERO": {"value": brahe.MJD_ZERO, "unit": "days", "description": "Modified Julian Date zero point (JD offset)", "category": "time"},
    "MJD_J2000": {"value": brahe.MJD_J2000, "unit": "days", "description": "Modified Julian Date of J2000 epoch", "category": "time"},
    "JD_J2000": {"value": brahe.JD_J2000, "unit": "days", "description": "Julian Date of J2000 epoch", "category": "time"},
    "GPS_ZERO": {"value": brahe.GPS_ZERO, "unit": "days", "description": "MJD of GPS epoch (Jan 6, 1980)", "category": "time"},
    "GPS_TAI": {"value": brahe.GPS_TAI, "unit": "s", "description": "GPS to TAI offset", "category": "time"},
    "TAI_GPS": {"value": brahe.TAI_GPS, "unit": "s", "description": "TAI to GPS offset", "category": "time"},
    "TT_TAI": {"value": brahe.TT_TAI, "unit": "s", "description": "TT to TAI offset", "category": "time"},
    "TAI_TT": {"value": brahe.TAI_TT, "unit": "s", "description": "TAI to TT offset", "category": "time"},
    "GPS_TT": {"value": brahe.GPS_TT, "unit": "s", "description": "GPS to TT offset", "category": "time"},
    "TT_GPS": {"value": brahe.TT_GPS, "unit": "s", "description": "TT to GPS offset", "category": "time"},
}

_CONSTANTS_UPPER = {k.upper(): k for k in CONSTANTS}


@mcp.tool()
def list_constants() -> dict[str, list[dict]]:
    """List all available brahe astrodynamics constants, organized by category."""
    logger.debug("Listing all constants")
    by_category: dict[str, list[dict]] = {}
    for name, info in CONSTANTS.items():
        entry = {"name": name, "value": info["value"], "unit": info["unit"], "description": info["description"]}
        by_category.setdefault(info["category"], []).append(entry)
    return by_category


@mcp.tool()
def get_constant(name: str) -> dict:
    """Get a single brahe constant by name (case-insensitive).

    Args:
        name: The constant name, e.g. "R_EARTH" or "gm_earth".
    """
    key = _CONSTANTS_UPPER.get(name.upper())
    if key is None:
        logger.warning("Unknown constant requested: {}", name)
        return {
            "error": f"Unknown constant: {name!r}",
            "valid_names": sorted(CONSTANTS.keys()),
        }
    info = CONSTANTS[key]
    logger.debug("Retrieved constant: {}", key)
    return {"name": key, "value": info["value"], "unit": info["unit"], "description": info["description"]}
