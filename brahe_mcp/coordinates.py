"""Coordinate and reference frame conversion MCP tools."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch, resolve_angle_format

# Initialize EOP data for epoch-dependent frame transforms
brahe.initialize_eop()

VALID_STATION_TYPES = {
    "geodetic": brahe.EllipsoidalConversionType.GEODETIC,
    "geocentric": brahe.EllipsoidalConversionType.GEOCENTRIC,
}

# Frames supported by each tool
POSITION_FRAMES = {"ECEF", "GEODETIC", "GEOCENTRIC", "ECI", "GCRF", "ITRF", "EME2000"}
STATE_FRAMES = {"ECI", "KOE", "ECEF", "GCRF", "ITRF", "EME2000"}
RELATIVE_FROM_FRAMES = {"ECEF", "ENZ", "SEZ"}
RELATIVE_TO_FRAMES = {"ECEF", "ENZ", "SEZ", "AZEL"}

# Conversions requiring an epoch
EPOCH_REQUIRED = {
    ("ECI", "ECEF"), ("ECEF", "ECI"),
    ("GCRF", "ITRF"), ("ITRF", "GCRF"),
}

# Position component labels by frame
_POSITION_LABELS = {
    "ECEF": ("x_m", "y_m", "z_m"),
    "ECI": ("x_m", "y_m", "z_m"),
    "GCRF": ("x_m", "y_m", "z_m"),
    "ITRF": ("x_m", "y_m", "z_m"),
    "EME2000": ("x_m", "y_m", "z_m"),
    "GEODETIC": ("lon", "lat", "alt_m"),
    "GEOCENTRIC": ("lon", "lat", "radius_m"),
    "ENZ": ("east_m", "north_m", "zenith_m"),
    "SEZ": ("south_m", "east_m", "zenith_m"),
    "AZEL": ("az", "el", "range_m"),
}

# State component labels by frame
_STATE_LABELS = {
    "ECI": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "ECEF": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "GCRF": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "ITRF": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "EME2000": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "KOE": ("a_m", "e", "i", "RAAN", "omega", "M"),
}


def _label_components(vector: list[float], frame: str, labels: dict) -> dict:
    """Build a labeled dict from a vector and frame-specific labels."""
    keys = labels.get(frame, tuple(f"c{i}" for i in range(len(vector))))
    return dict(zip(keys, vector))


def _resolve_epoch(epoch_str: str, from_frame: str, to_frame: str) -> brahe.Epoch | None:
    """Parse epoch if required for the conversion, or return None."""
    pair = (from_frame, to_frame)
    if pair in EPOCH_REQUIRED:
        if not epoch_str:
            raise ValueError(
                f"epoch is required for {from_frame} -> {to_frame} conversion"
            )
        return parse_epoch(epoch_str)
    return None


def _coord_error(message: str, **extra) -> dict:
    """Build a coordinates-specific error response."""
    return error_response(message, **extra)


# ---------------------------------------------------------------------------
# Tool 1: list_coordinate_systems
# ---------------------------------------------------------------------------

@mcp.tool()
def list_coordinate_systems() -> dict:
    """List all supported coordinate frames, their components, and valid conversions.

    Returns groups for position frames, state frames, and relative frames,
    along with allowed conversion pairs for each tool.
    """
    logger.debug("Listing coordinate systems")

    position_conversions = [
        "GEODETIC <-> ECEF", "GEOCENTRIC <-> ECEF", "GEODETIC <-> GEOCENTRIC",
        "ECI <-> ECEF (requires epoch)", "GCRF <-> ITRF (requires epoch)",
        "GCRF <-> EME2000",
    ]
    state_conversions = [
        "KOE <-> ECI", "ECI <-> ECEF (requires epoch)",
        "GCRF <-> ITRF (requires epoch)", "GCRF <-> EME2000",
    ]
    relative_conversions = [
        "ECEF <-> ENZ", "ECEF <-> SEZ", "ENZ -> AZEL", "SEZ -> AZEL",
    ]

    def _frame_info(name: str, labels: dict) -> dict:
        return {"name": name, "components": list(labels.get(name, []))}

    return {
        "position_frames": [
            _frame_info(f, _POSITION_LABELS)
            for f in sorted(POSITION_FRAMES)
        ],
        "state_frames": [
            _frame_info(f, _STATE_LABELS)
            for f in sorted(STATE_FRAMES)
        ],
        "relative_frames": [
            _frame_info(f, _POSITION_LABELS)
            for f in sorted(RELATIVE_FROM_FRAMES | RELATIVE_TO_FRAMES)
        ],
        "position_conversions": position_conversions,
        "state_conversions": state_conversions,
        "relative_conversions": relative_conversions,
    }


# ---------------------------------------------------------------------------
# Tool 2: convert_position
# ---------------------------------------------------------------------------

def _convert_position_vector(
    vec: np.ndarray,
    from_frame: str,
    to_frame: str,
    angle_fmt: brahe.AngleFormat,
    epoch: brahe.Epoch | None,
) -> np.ndarray:
    """Dispatch to the appropriate brahe position conversion function."""
    key = (from_frame, to_frame)

    dispatch = {
        ("GEODETIC", "ECEF"): lambda: brahe.position_geodetic_to_ecef(vec, angle_fmt),
        ("ECEF", "GEODETIC"): lambda: brahe.position_ecef_to_geodetic(vec, angle_fmt),
        ("GEOCENTRIC", "ECEF"): lambda: brahe.position_geocentric_to_ecef(vec, angle_fmt),
        ("ECEF", "GEOCENTRIC"): lambda: brahe.position_ecef_to_geocentric(vec, angle_fmt),
        ("GEODETIC", "GEOCENTRIC"): lambda: brahe.position_ecef_to_geocentric(
            brahe.position_geodetic_to_ecef(vec, angle_fmt), angle_fmt
        ),
        ("GEOCENTRIC", "GEODETIC"): lambda: brahe.position_ecef_to_geodetic(
            brahe.position_geocentric_to_ecef(vec, angle_fmt), angle_fmt
        ),
        ("ECI", "ECEF"): lambda: brahe.position_eci_to_ecef(epoch, vec),
        ("ECEF", "ECI"): lambda: brahe.position_ecef_to_eci(epoch, vec),
        ("GCRF", "ITRF"): lambda: brahe.position_gcrf_to_itrf(epoch, vec),
        ("ITRF", "GCRF"): lambda: brahe.position_itrf_to_gcrf(epoch, vec),
        ("GCRF", "EME2000"): lambda: brahe.position_gcrf_to_eme2000(vec),
        ("EME2000", "GCRF"): lambda: brahe.position_eme2000_to_gcrf(vec),
    }

    if key not in dispatch:
        raise ValueError(
            f"Unsupported position conversion: {from_frame} -> {to_frame}"
        )
    return dispatch[key]()


@mcp.tool()
def convert_position(
    vector: list[float],
    from_frame: str,
    to_frame: str,
    angle_format: str = "degrees",
    epoch: str = "",
) -> dict:
    """Convert a 3-element position vector between coordinate frames.

    Supported conversions: GEODETIC<->ECEF, GEOCENTRIC<->ECEF, GEODETIC<->GEOCENTRIC,
    ECI<->ECEF (requires epoch), GCRF<->ITRF (requires epoch), GCRF<->EME2000.

    Args:
        vector: 3-element position vector [x, y, z] or [lon, lat, alt] etc.
        from_frame: Source frame (ECEF, GEODETIC, GEOCENTRIC, ECI, GCRF, ITRF, EME2000).
        to_frame: Target frame (same set).
        angle_format: "degrees" (default) or "radians" for geodetic/geocentric angles.
        epoch: ISO epoch string, required for ECI<->ECEF and GCRF<->ITRF.
    """
    from_upper = from_frame.upper()
    to_upper = to_frame.upper()

    # Validate frames
    if from_upper not in POSITION_FRAMES:
        return _coord_error(
            f"Unknown from_frame: {from_frame!r}",
            valid_frames=sorted(POSITION_FRAMES),
        )
    if to_upper not in POSITION_FRAMES:
        return _coord_error(
            f"Unknown to_frame: {to_frame!r}",
            valid_frames=sorted(POSITION_FRAMES),
        )

    # Validate vector length
    if len(vector) != 3:
        return _coord_error(
            f"Position vector must have exactly 3 elements, got {len(vector)}"
        )

    # Identity conversion
    if from_upper == to_upper:
        return {
            "input": {"vector": vector, "frame": from_upper, "angle_format": angle_format},
            "output": {
                "vector": vector,
                "frame": to_upper,
                "components": _label_components(vector, to_upper, _POSITION_LABELS),
            },
        }

    # Validate angle_format
    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return _coord_error(str(e))

    # Resolve epoch
    try:
        epc = _resolve_epoch(epoch, from_upper, to_upper)
    except ValueError as e:
        return _coord_error(str(e))

    # Convert
    try:
        result = _convert_position_vector(
            np.array(vector, dtype=float), from_upper, to_upper, angle_fmt, epc
        )
    except ValueError as e:
        return _coord_error(str(e))
    except Exception as e:
        logger.error("Error in position conversion: {}", e)
        return _coord_error(f"Conversion error: {e}")

    out_list = result.tolist()
    logger.debug("Position {} {} -> {} {}", vector, from_upper, out_list, to_upper)
    return {
        "input": {"vector": vector, "frame": from_upper, "angle_format": angle_format},
        "output": {
            "vector": out_list,
            "frame": to_upper,
            "components": _label_components(out_list, to_upper, _POSITION_LABELS),
        },
    }


# ---------------------------------------------------------------------------
# Tool 3: convert_state
# ---------------------------------------------------------------------------

def _convert_state_vector(
    vec: np.ndarray,
    from_frame: str,
    to_frame: str,
    angle_fmt: brahe.AngleFormat,
    epoch: brahe.Epoch | None,
) -> np.ndarray:
    """Dispatch to the appropriate brahe state conversion function."""
    key = (from_frame, to_frame)

    dispatch = {
        ("KOE", "ECI"): lambda: brahe.state_koe_to_eci(vec, angle_fmt),
        ("ECI", "KOE"): lambda: brahe.state_eci_to_koe(vec, angle_fmt),
        ("ECI", "ECEF"): lambda: brahe.state_eci_to_ecef(epoch, vec),
        ("ECEF", "ECI"): lambda: brahe.state_ecef_to_eci(epoch, vec),
        ("GCRF", "ITRF"): lambda: brahe.state_gcrf_to_itrf(epoch, vec),
        ("ITRF", "GCRF"): lambda: brahe.state_itrf_to_gcrf(epoch, vec),
        ("GCRF", "EME2000"): lambda: brahe.state_gcrf_to_eme2000(vec),
        ("EME2000", "GCRF"): lambda: brahe.state_eme2000_to_gcrf(vec),
    }

    if key not in dispatch:
        raise ValueError(
            f"Unsupported state conversion: {from_frame} -> {to_frame}"
        )
    return dispatch[key]()


@mcp.tool()
def convert_state(
    vector: list[float],
    from_frame: str,
    to_frame: str,
    angle_format: str = "degrees",
    epoch: str = "",
) -> dict:
    """Convert a 6-element state vector between coordinate frames.

    Supported conversions: KOE<->ECI, ECI<->ECEF (requires epoch),
    GCRF<->ITRF (requires epoch), GCRF<->EME2000.

    Args:
        vector: 6-element state vector [x,y,z,vx,vy,vz] or [a,e,i,RAAN,omega,M].
        from_frame: Source frame (ECI, KOE, ECEF, GCRF, ITRF, EME2000).
        to_frame: Target frame (same set).
        angle_format: "degrees" (default) or "radians" for KOE angles.
        epoch: ISO epoch string, required for ECI<->ECEF and GCRF<->ITRF.
    """
    from_upper = from_frame.upper()
    to_upper = to_frame.upper()

    if from_upper not in STATE_FRAMES:
        return _coord_error(
            f"Unknown from_frame: {from_frame!r}",
            valid_frames=sorted(STATE_FRAMES),
        )
    if to_upper not in STATE_FRAMES:
        return _coord_error(
            f"Unknown to_frame: {to_frame!r}",
            valid_frames=sorted(STATE_FRAMES),
        )

    if len(vector) != 6:
        return _coord_error(
            f"State vector must have exactly 6 elements, got {len(vector)}"
        )

    # Identity conversion
    if from_upper == to_upper:
        return {
            "input": {"vector": vector, "frame": from_upper, "angle_format": angle_format},
            "output": {
                "vector": vector,
                "frame": to_upper,
                "components": _label_components(vector, to_upper, _STATE_LABELS),
            },
        }

    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return _coord_error(str(e))

    try:
        epc = _resolve_epoch(epoch, from_upper, to_upper)
    except ValueError as e:
        return _coord_error(str(e))

    try:
        result = _convert_state_vector(
            np.array(vector, dtype=float), from_upper, to_upper, angle_fmt, epc
        )
    except ValueError as e:
        return _coord_error(str(e))
    except Exception as e:
        logger.error("Error in state conversion: {}", e)
        return _coord_error(f"Conversion error: {e}")

    out_list = result.tolist()
    logger.debug("State {} {} -> {} {}", vector, from_upper, out_list, to_upper)
    return {
        "input": {"vector": vector, "frame": from_upper, "angle_format": angle_format},
        "output": {
            "vector": out_list,
            "frame": to_upper,
            "components": _label_components(out_list, to_upper, _STATE_LABELS),
        },
    }


# ---------------------------------------------------------------------------
# Tool 4: convert_relative_position
# ---------------------------------------------------------------------------

def _resolve_station_ecef(
    station: np.ndarray,
    station_type: str,
    angle_fmt: brahe.AngleFormat,
) -> np.ndarray:
    """Convert station to ECEF if given as geodetic/geocentric."""
    st_lower = station_type.lower()
    if st_lower == "ecef":
        return station
    if st_lower not in VALID_STATION_TYPES:
        raise ValueError(
            f"Invalid station_type: {station_type!r}. "
            f"Must be one of: ecef, {', '.join(sorted(VALID_STATION_TYPES.keys()))}"
        )
    # Convert geodetic/geocentric station to ECEF
    conv_type = VALID_STATION_TYPES[st_lower]
    if conv_type == brahe.EllipsoidalConversionType.GEODETIC:
        return brahe.position_geodetic_to_ecef(station, angle_fmt)
    else:
        return brahe.position_geocentric_to_ecef(station, angle_fmt)


@mcp.tool()
def convert_relative_position(
    station: list[float],
    vector: list[float],
    from_frame: str,
    to_frame: str,
    station_type: str = "geodetic",
    angle_format: str = "degrees",
) -> dict:
    """Convert a position relative to a ground station between ECEF and topocentric frames.

    Supported conversions: ECEF<->ENZ, ECEF<->SEZ, ENZ->AZEL, SEZ->AZEL.

    Args:
        station: 3-element station position (geodetic [lon,lat,alt] or ECEF [x,y,z]).
        vector: 3-element position to convert.
        from_frame: Source frame (ECEF, ENZ, SEZ).
        to_frame: Target frame (ECEF, ENZ, SEZ, AZEL).
        station_type: How station is specified: "geodetic" (default), "geocentric", or "ecef".
        angle_format: "degrees" (default) or "radians" for AZEL output and station angles.
    """
    from_upper = from_frame.upper()
    to_upper = to_frame.upper()

    if from_upper not in RELATIVE_FROM_FRAMES:
        return _coord_error(
            f"Unknown from_frame: {from_frame!r}",
            valid_from_frames=sorted(RELATIVE_FROM_FRAMES),
            valid_to_frames=sorted(RELATIVE_TO_FRAMES),
        )
    if to_upper not in RELATIVE_TO_FRAMES:
        return _coord_error(
            f"Unknown to_frame: {to_frame!r}",
            valid_from_frames=sorted(RELATIVE_FROM_FRAMES),
            valid_to_frames=sorted(RELATIVE_TO_FRAMES),
        )

    if len(station) != 3:
        return _coord_error(
            f"Station vector must have exactly 3 elements, got {len(station)}"
        )
    if len(vector) != 3:
        return _coord_error(
            f"Position vector must have exactly 3 elements, got {len(vector)}"
        )

    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return _coord_error(str(e))

    # Resolve station to ECEF
    try:
        station_ecef = _resolve_station_ecef(
            np.array(station, dtype=float), station_type, angle_fmt
        )
    except ValueError as e:
        return _coord_error(str(e))

    # Determine the EllipsoidalConversionType for brahe calls
    # If station was given as ecef, default to geodetic conversion type
    if station_type.lower() == "ecef":
        conv_type = brahe.EllipsoidalConversionType.GEODETIC
    else:
        conv_type = VALID_STATION_TYPES[station_type.lower()]

    vec = np.array(vector, dtype=float)

    try:
        if from_upper == "ECEF" and to_upper == "ENZ":
            result = brahe.relative_position_ecef_to_enz(station_ecef, vec, conv_type)
        elif from_upper == "ECEF" and to_upper == "SEZ":
            result = brahe.relative_position_ecef_to_sez(station_ecef, vec, conv_type)
        elif from_upper == "ENZ" and to_upper == "ECEF":
            result = brahe.relative_position_enz_to_ecef(station_ecef, vec, conv_type)
        elif from_upper == "SEZ" and to_upper == "ECEF":
            result = brahe.relative_position_sez_to_ecef(station_ecef, vec, conv_type)
        elif from_upper == "ENZ" and to_upper == "AZEL":
            result = brahe.position_enz_to_azel(vec, angle_fmt)
        elif from_upper == "SEZ" and to_upper == "AZEL":
            result = brahe.position_sez_to_azel(vec, angle_fmt)
        else:
            return _coord_error(
                f"Unsupported relative conversion: {from_upper} -> {to_upper}",
                valid_conversions=[
                    "ECEF->ENZ", "ECEF->SEZ", "ENZ->ECEF", "SEZ->ECEF",
                    "ENZ->AZEL", "SEZ->AZEL",
                ],
            )
    except Exception as e:
        logger.error("Error in relative position conversion: {}", e)
        return _coord_error(f"Conversion error: {e}")

    out_list = result.tolist()
    logger.debug("Relative {} {} -> {} {}", vector, from_upper, out_list, to_upper)
    return {
        "input": {
            "station": station,
            "station_type": station_type,
            "vector": vector,
            "frame": from_upper,
            "angle_format": angle_format,
        },
        "output": {
            "vector": out_list,
            "frame": to_upper,
            "components": _label_components(out_list, to_upper, _POSITION_LABELS),
        },
    }
