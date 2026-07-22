"""Relative motion (RTN / ROE) MCP tools."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, resolve_angle_format

RTN_DIRECTIONS = {"eci_to_rtn", "rtn_to_eci"}
ROE_DIRECTIONS = {"eci_to_roe", "roe_to_eci", "oe_to_roe", "roe_to_oe"}

# Quasi-nonsingular relative orbital elements.
ROE_LABELS = ("da", "dlambda", "dex", "dey", "dix", "diy")

_ECI_LABELS = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
_RTN_LABELS = ("r_m", "t_m", "n_m", "vr_m_s", "vt_m_s", "vn_m_s")
_KOE_LABELS = ("a_m", "e", "i", "RAAN", "omega", "M")

_ROE_UNITS = {
    "da": "dimensionless",
    "dlambda": "degrees or radians (see angle_format)",
    "dex": "dimensionless",
    "dey": "dimensionless",
    "dix": "degrees or radians (see angle_format)",
    "diy": "degrees or radians (see angle_format)",
}


def _check_six(name: str, vector: list[float]) -> str | None:
    """Return an error message if the vector is not 6 elements, else None."""
    if len(vector) != 6:
        return f"{name} must have exactly 6 elements, got {len(vector)}"
    return None


@mcp.tool()
def list_relative_motion_options() -> dict:
    """List relative motion conversions, ROE element meanings, and units."""
    logger.debug("Listing relative motion options")
    return {
        "rtn_directions": sorted(RTN_DIRECTIONS),
        "roe_directions": sorted(ROE_DIRECTIONS),
        "roe_components": list(ROE_LABELS),
        "eci_components": list(_ECI_LABELS),
        "rtn_components": list(_RTN_LABELS),
        "koe_components": list(_KOE_LABELS),
        "chief_types": ["eci", "koe"],
        "units": _ROE_UNITS,
        "notes": [
            "eci_to_rtn takes the deputy's ABSOLUTE ECI state and returns the "
            "RELATIVE RTN state; rtn_to_eci takes a RELATIVE RTN state and "
            "returns the deputy's ABSOLUTE ECI state.",
            "ROE is the quasi-nonsingular formulation, valid for circular and "
            "near-circular orbits.",
            "chief_type must be 'koe' for oe_to_roe and roe_to_oe, and 'eci' "
            "for eci_to_roe and roe_to_eci.",
        ],
    }


@mcp.tool()
def convert_rtn_state(
    chief_state_eci: list[float],
    vector: list[float],
    direction: str,
) -> dict:
    """Convert a deputy state between ECI and the chief's RTN frame.

    Args:
        chief_state_eci: Chief 6-element ECI state [x,y,z,vx,vy,vz] (m, m/s).
        vector: For "eci_to_rtn", the deputy's ABSOLUTE ECI state. For
            "rtn_to_eci", the RELATIVE RTN state.
        direction: "eci_to_rtn" or "rtn_to_eci".
    """
    key = direction.lower()
    if key not in RTN_DIRECTIONS:
        return error_response(
            f"Unknown direction: {direction!r}",
            valid_directions=sorted(RTN_DIRECTIONS),
        )

    for name, vec in (("chief_state_eci", chief_state_eci), ("vector", vector)):
        msg = _check_six(name, vec)
        if msg:
            return error_response(msg)

    try:
        chief = np.array(chief_state_eci, dtype=float)
        vec = np.array(vector, dtype=float)
        if key == "eci_to_rtn":
            out = brahe.state_eci_to_rtn(chief, vec)
            labels = _RTN_LABELS
        else:
            out = brahe.state_rtn_to_eci(chief, vec)
            labels = _ECI_LABELS
    except Exception as e:
        logger.error("RTN conversion error: {}", e)
        return error_response(f"Conversion error: {e}")

    out_list = np.array(out, dtype=float).tolist()
    logger.debug("RTN {}: {} -> {}", key, vector, out_list)
    return {
        "direction": key,
        "input": {"chief_state_eci": chief_state_eci, "vector": vector},
        "output": {"state": out_list, "components": dict(zip(labels, out_list))},
    }


@mcp.tool()
def compute_rtn_rotation(chief_state_eci: list[float], direction: str) -> dict:
    """Compute the 3x3 rotation between ECI and the chief's RTN frame.

    Args:
        chief_state_eci: Chief 6-element ECI state [x,y,z,vx,vy,vz] (m, m/s).
        direction: "eci_to_rtn" or "rtn_to_eci".
    """
    key = direction.lower()
    if key not in RTN_DIRECTIONS:
        return error_response(
            f"Unknown direction: {direction!r}",
            valid_directions=sorted(RTN_DIRECTIONS),
        )

    msg = _check_six("chief_state_eci", chief_state_eci)
    if msg:
        return error_response(msg)

    try:
        chief = np.array(chief_state_eci, dtype=float)
        if key == "eci_to_rtn":
            mat = brahe.rotation_eci_to_rtn(chief)
        else:
            mat = brahe.rotation_rtn_to_eci(chief)
    except Exception as e:
        logger.error("RTN rotation error: {}", e)
        return error_response(f"Rotation error: {e}")

    return {
        "direction": key,
        "input": {"chief_state_eci": chief_state_eci},
        "output": {"matrix": np.array(mat, dtype=float).tolist()},
    }


# Which chief_type each ROE direction requires, and the output labels.
_ROE_DISPATCH = {
    "eci_to_roe": ("eci", ROE_LABELS),
    "roe_to_eci": ("eci", _ECI_LABELS),
    "oe_to_roe": ("koe", ROE_LABELS),
    "roe_to_oe": ("koe", _KOE_LABELS),
}


@mcp.tool()
def convert_roe_state(
    chief: list[float],
    vector: list[float],
    direction: str,
    chief_type: str = "eci",
    angle_format: str = "degrees",
) -> dict:
    """Convert between absolute states and quasi-nonsingular relative orbital elements.

    ROE is [da, dlambda, dex, dey, dix, diy]. da, dex, and dey are
    dimensionless; dlambda, dix, and diy are angular.

    Args:
        chief: Chief 6-element state. ECI [x,y,z,vx,vy,vz] when chief_type is
            "eci", or Keplerian [a,e,i,RAAN,omega,M] when chief_type is "koe".
        vector: The deputy state or the ROE vector, per direction.
        direction: "eci_to_roe", "roe_to_eci", "oe_to_roe", or "roe_to_oe".
        chief_type: "eci" (default) or "koe". Must match the direction.
        angle_format: "degrees" (default) or "radians".
    """
    key = direction.lower()
    if key not in ROE_DIRECTIONS:
        return error_response(
            f"Unknown direction: {direction!r}",
            valid_directions=sorted(ROE_DIRECTIONS),
        )

    required_type, labels = _ROE_DISPATCH[key]
    given_type = chief_type.lower()
    if given_type != required_type:
        return error_response(
            f"direction {key!r} requires chief_type={required_type!r}, "
            f"got {chief_type!r}",
            valid_chief_types=["eci", "koe"],
        )

    for name, vec in (("chief", chief), ("vector", vector)):
        msg = _check_six(name, vec)
        if msg:
            return error_response(msg)

    try:
        fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    try:
        chief_arr = np.array(chief, dtype=float)
        vec = np.array(vector, dtype=float)
        if key == "eci_to_roe":
            out = brahe.state_eci_to_roe(chief_arr, vec, fmt)
        elif key == "roe_to_eci":
            out = brahe.state_roe_to_eci(chief_arr, vec, fmt)
        elif key == "oe_to_roe":
            out = brahe.state_oe_to_roe(chief_arr, vec, fmt)
        else:
            out = brahe.state_roe_to_oe(chief_arr, vec, fmt)
    except Exception as e:
        logger.error("ROE conversion error: {}", e)
        return error_response(f"Conversion error: {e}")

    out_list = np.array(out, dtype=float).tolist()
    if not np.all(np.isfinite(out_list)):
        return error_response(
            "Conversion produced non-finite output (chief orbit may be singular "
            "for this transform, e.g. near-zero inclination or eccentricity)",
            direction=key,
        )
    logger.debug("ROE {}: {} -> {}", key, vector, out_list)
    return {
        "direction": key,
        "input": {
            "chief": chief,
            "chief_type": given_type,
            "vector": vector,
            "angle_format": angle_format.lower(),
        },
        "output": {
            "state": out_list,
            "components": dict(zip(labels, out_list)),
        },
    }
