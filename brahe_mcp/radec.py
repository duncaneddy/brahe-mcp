"""Right ascension / declination coordinate MCP tools."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch, resolve_angle_format

# brahe's RA/Dec conversions are frame-agnostic spherical <-> Cartesian, so ECI
# and GCRF are equivalent inputs here and neither needs an epoch.
INERTIAL_FRAMES = {"ECI", "GCRF"}
RADEC_FRAMES = {"RADEC", "AZEL"} | INERTIAL_FRAMES

_SITE_TYPES = {"geodetic", "geocentric", "ecef"}

_POSITION_PAIRS = {
    ("RADEC", "ECI"), ("ECI", "RADEC"),
    ("RADEC", "GCRF"), ("GCRF", "RADEC"),
    ("RADEC", "AZEL"), ("AZEL", "RADEC"),
}

# AZEL is position-only in brahe, so it is absent here.
_STATE_PAIRS = {
    ("RADEC", "ECI"), ("ECI", "RADEC"),
    ("RADEC", "GCRF"), ("GCRF", "RADEC"),
}

_POSITION_LABELS = {
    "RADEC": ("ra", "dec", "range_m"),
    "AZEL": ("az", "el", "range_m"),
    "ECI": ("x_m", "y_m", "z_m"),
    "GCRF": ("x_m", "y_m", "z_m"),
}

_STATE_LABELS = {
    "RADEC": ("ra", "dec", "range_m", "ra_rate", "dec_rate", "range_rate_m_s"),
    "ECI": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "GCRF": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
}


def _label(vector: list[float], frame: str, labels: dict) -> dict:
    """Build a labeled dict from a vector and frame-specific labels."""
    keys = labels.get(frame, tuple(f"c{i}" for i in range(len(vector))))
    return dict(zip(keys, vector))


def _resolve_site_geodetic(
    site: list[float], site_type: str, angle_fmt: brahe.AngleFormat
) -> np.ndarray:
    """Resolve a site to geodetic [lon, lat, alt], which brahe's AZEL calls expect."""
    st = site_type.lower()
    if st not in _SITE_TYPES:
        raise ValueError(
            f"Invalid site_type: {site_type!r}. Must be one of: {sorted(_SITE_TYPES)}"
        )
    arr = np.array(site, dtype=float)
    if st == "geodetic":
        return arr
    if st == "geocentric":
        arr = brahe.position_geocentric_to_ecef(arr, angle_fmt)
    return brahe.position_ecef_to_geodetic(arr, angle_fmt)


@mcp.tool()
def list_radec_options() -> dict:
    """List RA/Dec frames, vector components, and conversion requirements.

    Use before convert_radec to discover valid frame pairs and which
    conversions require a site and epoch.
    """
    logger.debug("Listing radec options")
    return {
        "frames": sorted(RADEC_FRAMES),
        "position_conversions": sorted(f"{a} -> {b}" for a, b in _POSITION_PAIRS),
        "state_conversions": sorted(f"{a} -> {b}" for a, b in _STATE_PAIRS),
        "position_components": {f: list(v) for f, v in _POSITION_LABELS.items()},
        "state_components": {f: list(v) for f, v in _STATE_LABELS.items()},
        "site_types": sorted(_SITE_TYPES),
        "notes": [
            "RADEC <-> ECI/GCRF is a frame-agnostic spherical <-> Cartesian "
            "conversion; it needs no epoch and ECI and GCRF are equivalent.",
            "RADEC <-> AZEL is position-only and requires both site and epoch.",
            "EME2000 is not supported here. Use transform_frame to convert "
            "GCRF <-> EME2000 first; they differ by a ~23 mas frame bias.",
        ],
    }


@mcp.tool()
def convert_radec(
    vector: list[float],
    from_frame: str,
    to_frame: str,
    angle_format: str = "degrees",
    site: list[float] | None = None,
    site_type: str = "geodetic",
    epoch: str = "",
) -> dict:
    """Convert a position or state between RA/Dec and inertial or topocentric frames.

    A 3-element vector is treated as a position, a 6-element vector as a state.
    Use list_radec_options() to discover valid frame pairs.

    Args:
        vector: 3 elements for a position, 6 for a state.
        from_frame: Source frame (RADEC, ECI, GCRF, AZEL).
        to_frame: Target frame (same set).
        angle_format: "degrees" (default) or "radians".
        site: Ground site [lon, lat, alt] — required for RADEC <-> AZEL only.
        site_type: How site is given: "geodetic" (default), "geocentric", "ecef".
        epoch: ISO epoch string — required for RADEC <-> AZEL only.
    """
    src = from_frame.upper()
    dst = to_frame.upper()

    if src not in RADEC_FRAMES:
        return error_response(
            f"Unknown from_frame: {from_frame!r}", valid_frames=sorted(RADEC_FRAMES)
        )
    if dst not in RADEC_FRAMES:
        return error_response(
            f"Unknown to_frame: {to_frame!r}", valid_frames=sorted(RADEC_FRAMES)
        )

    n = len(vector)
    if n not in (3, 6):
        return error_response(
            f"vector must have 3 (position) or 6 (state) elements, got {n}"
        )

    pairs = _POSITION_PAIRS if n == 3 else _STATE_PAIRS
    kind = "position" if n == 3 else "state"
    if (src, dst) not in pairs:
        return error_response(
            f"Unsupported {kind} conversion: {src} -> {dst}",
            valid_conversions=sorted(f"{a} -> {b}" for a, b in pairs),
        )

    try:
        fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    site_geod = None
    epc = None
    if "AZEL" in (src, dst):
        if site is None:
            return error_response(
                f"{src} -> {dst} requires a site (3-element [lon, lat, alt])"
            )
        if len(site) != 3:
            return error_response(
                f"site must have exactly 3 elements, got {len(site)}"
            )
        if not epoch:
            return error_response(f"{src} -> {dst} requires an epoch")
        try:
            site_geod = _resolve_site_geodetic(site, site_type, fmt)
        except ValueError as e:
            return error_response(str(e))
        try:
            epc = parse_epoch(epoch)
        except ValueError as e:
            return error_response(f"Invalid epoch: {e}")

    try:
        vec = np.array(vector, dtype=float)
        if dst == "RADEC" and src in INERTIAL_FRAMES:
            fn = brahe.position_inertial_to_radec if n == 3 else brahe.state_inertial_to_radec
            out = fn(vec, fmt)
        elif src == "RADEC" and dst in INERTIAL_FRAMES:
            fn = brahe.position_radec_to_inertial if n == 3 else brahe.state_radec_to_inertial
            out = fn(vec, fmt)
        elif src == "RADEC" and dst == "AZEL":
            out = brahe.position_radec_to_azel(vec, site_geod, epc, fmt)
        else:
            out = brahe.position_azel_to_radec(vec, site_geod, epc, fmt)
    except Exception as e:
        logger.error("RA/Dec conversion error: {}", e)
        return error_response(f"Conversion error: {e}")

    out_list = np.array(out, dtype=float).tolist()
    labels = _POSITION_LABELS if n == 3 else _STATE_LABELS
    inputs = {
        "vector": vector,
        "frame": src,
        "angle_format": angle_format.lower(),
    }
    if site is not None:
        inputs["site"] = site
        inputs["site_type"] = site_type.lower()
    if epoch:
        inputs["epoch"] = epoch

    logger.debug("RA/Dec {} {} -> {} {}", vector, src, out_list, dst)
    return {
        "input": inputs,
        "output": {
            "vector": out_list,
            "frame": dst,
            "components": _label(out_list, dst, labels),
        },
    }
