import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp

VALID_ANGLE_FORMATS = {
    "degrees": brahe.AngleFormat.DEGREES,
    "radians": brahe.AngleFormat.RADIANS,
}

ORBITAL_PROPERTIES = {
    "orbital_period": {
        "desc": "Orbital period (s)",
        "required": ["a"],
        "optional": ["gm"],
        "unit": "s",
    },
    "orbital_period_from_state": {
        "desc": "Orbital period from ECI state vector (s)",
        "required": ["state_eci"],
        "optional": ["gm"],
        "unit": "s",
    },
    "mean_motion": {
        "desc": "Mean motion (deg/s or rad/s)",
        "required": ["a"],
        "optional": ["gm", "angle_format"],
        "unit": "deg/s",
    },
    "semimajor_axis": {
        "desc": "Semi-major axis from mean motion (m)",
        "required": ["n"],
        "optional": ["gm", "angle_format"],
        "unit": "m",
    },
    "semimajor_axis_from_period": {
        "desc": "Semi-major axis from orbital period (m)",
        "required": ["period"],
        "optional": ["gm"],
        "unit": "m",
    },
    "periapsis_velocity": {
        "desc": "Velocity at periapsis (m/s)",
        "required": ["a", "e"],
        "optional": ["gm"],
        "unit": "m/s",
    },
    "apoapsis_velocity": {
        "desc": "Velocity at apoapsis (m/s)",
        "required": ["a", "e"],
        "optional": ["gm"],
        "unit": "m/s",
    },
    "periapsis_distance": {
        "desc": "Distance from body center at periapsis (m)",
        "required": ["a", "e"],
        "optional": [],
        "unit": "m",
    },
    "apoapsis_distance": {
        "desc": "Distance from body center at apoapsis (m)",
        "required": ["a", "e"],
        "optional": [],
        "unit": "m",
    },
    "periapsis_altitude": {
        "desc": "Altitude above surface at periapsis (m)",
        "required": ["a", "e"],
        "optional": ["r_body"],
        "unit": "m",
    },
    "apoapsis_altitude": {
        "desc": "Altitude above surface at apoapsis (m)",
        "required": ["a", "e"],
        "optional": ["r_body"],
        "unit": "m",
    },
    "sun_synchronous_inclination": {
        "desc": "Inclination for sun-synchronous orbit (deg or rad)",
        "required": ["a", "e"],
        "optional": ["angle_format"],
        "unit": "deg",
    },
    "geo_sma": {
        "desc": "Geostationary orbit semi-major axis (m)",
        "required": [],
        "optional": [],
        "unit": "m",
    },
}

ANOMALY_CONVERSIONS = {
    "eccentric_to_mean": {
        "desc": "Eccentric anomaly to mean anomaly",
        "func": brahe.anomaly_eccentric_to_mean,
    },
    "mean_to_eccentric": {
        "desc": "Mean anomaly to eccentric anomaly",
        "func": brahe.anomaly_mean_to_eccentric,
    },
    "true_to_eccentric": {
        "desc": "True anomaly to eccentric anomaly",
        "func": brahe.anomaly_true_to_eccentric,
    },
    "eccentric_to_true": {
        "desc": "Eccentric anomaly to true anomaly",
        "func": brahe.anomaly_eccentric_to_true,
    },
    "true_to_mean": {
        "desc": "True anomaly to mean anomaly",
        "func": brahe.anomaly_true_to_mean,
    },
    "mean_to_true": {
        "desc": "Mean anomaly to true anomaly",
        "func": brahe.anomaly_mean_to_true,
    },
}

_PROPERTIES_LOWER = {k.lower(): k for k in ORBITAL_PROPERTIES}
_CONVERSIONS_LOWER = {k.lower(): k for k in ANOMALY_CONVERSIONS}


def _error_response(message: str) -> dict:
    """Return an error dict with valid options for discoverability."""
    return {
        "error": message,
        "valid_computations": sorted(ORBITAL_PROPERTIES.keys()),
        "valid_conversions": sorted(ANOMALY_CONVERSIONS.keys()),
        "valid_angle_formats": sorted(VALID_ANGLE_FORMATS.keys()),
    }


def _dispatch_computation(
    name: str,
    *,
    a: float | None,
    e: float | None,
    n: float | None,
    period: float | None,
    state_eci: str | None,
    gm: float | None,
    r_body: float | None,
    angle_format: brahe.AngleFormat,
) -> float:
    """Dispatch to the appropriate brahe function."""
    if name == "orbital_period":
        if gm is not None:
            return brahe.orbital_period_general(a, gm)
        return brahe.orbital_period(a)

    elif name == "orbital_period_from_state":
        parts = [float(x.strip()) for x in state_eci.split(",")]
        state = np.array(parts)
        return brahe.orbital_period_from_state(
            state, gm if gm is not None else brahe.GM_EARTH
        )

    elif name == "mean_motion":
        if gm is not None:
            return brahe.mean_motion_general(a, gm, angle_format)
        return brahe.mean_motion(a, angle_format)

    elif name == "semimajor_axis":
        if gm is not None:
            return brahe.semimajor_axis_general(n, gm, angle_format)
        return brahe.semimajor_axis(n, angle_format)

    elif name == "semimajor_axis_from_period":
        if gm is not None:
            return brahe.semimajor_axis_from_orbital_period_general(period, gm)
        return brahe.semimajor_axis_from_orbital_period(period)

    elif name == "periapsis_velocity":
        return brahe.periapsis_velocity(
            a, e, gm=gm if gm is not None else brahe.GM_EARTH
        )

    elif name == "apoapsis_velocity":
        return brahe.apoapsis_velocity(
            a, e, gm=gm if gm is not None else brahe.GM_EARTH
        )

    elif name == "periapsis_distance":
        return brahe.periapsis_distance(a, e)

    elif name == "apoapsis_distance":
        return brahe.apoapsis_distance(a, e)

    elif name == "periapsis_altitude":
        return brahe.periapsis_altitude(
            a, e, r_body=r_body if r_body is not None else brahe.R_EARTH
        )

    elif name == "apoapsis_altitude":
        return brahe.apoapsis_altitude(
            a, e, r_body=r_body if r_body is not None else brahe.R_EARTH
        )

    elif name == "sun_synchronous_inclination":
        return brahe.sun_synchronous_inclination(a, e, angle_format=angle_format)

    elif name == "geo_sma":
        return brahe.geo_sma()


@mcp.tool()
def list_orbital_computations() -> dict:
    """List all available orbital property computations and anomaly conversions.

    Returns names, descriptions, and parameter requirements for each computation.
    """
    logger.debug("Listing orbital computations")
    return {
        "orbital_properties": [
            {
                "name": name,
                "description": info["desc"],
                "required_params": info["required"],
                "optional_params": info["optional"],
            }
            for name, info in ORBITAL_PROPERTIES.items()
        ],
        "anomaly_conversions": [
            {"name": name, "description": info["desc"]}
            for name, info in ANOMALY_CONVERSIONS.items()
        ],
    }


@mcp.tool()
def compute_orbital_property(
    computation: str,
    a: float | None = None,
    e: float | None = None,
    n: float | None = None,
    period: float | None = None,
    state_eci: str | None = None,
    gm: float | None = None,
    r_body: float | None = None,
    angle_format: str = "degrees",
) -> dict:
    """Compute an orbital property using brahe astrodynamics functions.

    Use list_orbital_computations() to see all available computations and their parameters.

    Args:
        computation: Name of the computation (case-insensitive), e.g. "orbital_period".
        a: Semi-major axis in meters.
        e: Eccentricity (dimensionless).
        n: Mean motion (deg/s or rad/s depending on angle_format).
        period: Orbital period in seconds.
        state_eci: ECI state vector as comma-separated string "x,y,z,vx,vy,vz" (m, m/s).
        gm: Gravitational parameter (m^3/s^2). Defaults to Earth if omitted.
        r_body: Body radius (m). Defaults to Earth equatorial radius if omitted.
        angle_format: Angle unit - "degrees" (default) or "radians".
    """
    # Validate computation name
    key = _PROPERTIES_LOWER.get(computation.lower())
    if key is None:
        logger.warning("Unknown computation: {}", computation)
        return _error_response(f"Unknown computation: {computation!r}")

    # Validate angle_format
    fmt_lower = angle_format.lower()
    if fmt_lower not in VALID_ANGLE_FORMATS:
        logger.warning("Invalid angle_format: {}", angle_format)
        return _error_response(f"Invalid angle_format: {angle_format!r}")
    angle_fmt = VALID_ANGLE_FORMATS[fmt_lower]

    # Check required params
    prop = ORBITAL_PROPERTIES[key]
    param_map = {
        "a": a, "e": e, "n": n, "period": period, "state_eci": state_eci,
    }
    missing = [p for p in prop["required"] if param_map.get(p) is None]
    if missing:
        logger.warning("Missing required params for {}: {}", key, missing)
        return _error_response(
            f"Missing required parameter(s) for {key!r}: {missing}. "
            f"Required: {prop['required']}, Optional: {prop['optional']}"
        )

    # Validate state_eci format
    if key == "orbital_period_from_state":
        parts = state_eci.split(",")
        if len(parts) != 6:
            return _error_response(
                "state_eci must have exactly 6 comma-separated values: x,y,z,vx,vy,vz"
            )
        try:
            [float(x.strip()) for x in parts]
        except ValueError:
            return _error_response("state_eci values must be numeric")

    # Dispatch
    try:
        result = _dispatch_computation(
            key,
            a=a, e=e, n=n, period=period, state_eci=state_eci,
            gm=gm, r_body=r_body, angle_format=angle_fmt,
        )
    except Exception as exc:
        logger.error("Error computing {}: {}", key, exc)
        return _error_response(f"Error computing {key!r}: {exc}")

    # Build input summary
    inputs = {}
    for p in prop["required"] + prop["optional"]:
        val = param_map.get(p)
        if val is not None:
            inputs[p] = val
    if gm is not None:
        inputs["gm"] = gm
    if r_body is not None:
        inputs["r_body"] = r_body
    if "angle_format" in prop["optional"]:
        inputs["angle_format"] = fmt_lower

    # Determine unit
    unit = prop["unit"]
    if unit in ("deg/s", "deg") and fmt_lower == "radians":
        unit = unit.replace("deg", "rad")

    logger.debug("Computed {}: {}", key, result)
    return {
        "computation": key,
        "result": {"value": result, "unit": unit},
        "inputs": inputs,
    }


@mcp.tool()
def convert_anomaly(
    conversion: str,
    anomaly: float,
    e: float,
    angle_format: str = "degrees",
) -> dict:
    """Convert between orbital anomaly types (mean, eccentric, true).

    Use list_orbital_computations() to see all available conversions.

    Args:
        conversion: Conversion name (case-insensitive), e.g. "eccentric_to_mean".
        anomaly: Input anomaly value in the specified angle_format.
        e: Orbital eccentricity (dimensionless).
        angle_format: Angle unit - "degrees" (default) or "radians".
    """
    # Validate conversion name
    key = _CONVERSIONS_LOWER.get(conversion.lower())
    if key is None:
        logger.warning("Unknown anomaly conversion: {}", conversion)
        return _error_response(f"Unknown anomaly conversion: {conversion!r}")

    # Validate angle_format
    fmt_lower = angle_format.lower()
    if fmt_lower not in VALID_ANGLE_FORMATS:
        logger.warning("Invalid angle_format: {}", angle_format)
        return _error_response(f"Invalid angle_format: {angle_format!r}")
    angle_fmt = VALID_ANGLE_FORMATS[fmt_lower]

    # Call brahe
    try:
        result = ANOMALY_CONVERSIONS[key]["func"](
            anomaly, e, angle_format=angle_fmt
        )
    except Exception as exc:
        logger.error("Error in anomaly conversion {}: {}", key, exc)
        return _error_response(f"Error in conversion {key!r}: {exc}")

    logger.debug("Anomaly conversion {}: {} -> {}", key, anomaly, result)
    return {
        "conversion": key,
        "input": {"anomaly": anomaly, "eccentricity": e, "angle_format": fmt_lower},
        "output": {"anomaly": result, "angle_format": fmt_lower},
    }
