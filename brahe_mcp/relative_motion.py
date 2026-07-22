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
