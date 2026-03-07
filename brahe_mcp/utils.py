"""Shared utilities for brahe-mcp tool modules."""

import brahe

def serialize_gp_record(record: brahe.GPRecord) -> dict:
    """Extract key fields from a GPRecord into a plain dict.

    Args:
        record: A brahe GPRecord from CelesTrak or SpaceTrack queries.

    Returns:
        Dict with orbital elements and TLE data.
    """
    return {
        "object_name": record.object_name,
        "norad_cat_id": record.norad_cat_id,
        "object_id": record.object_id,
        "epoch": record.epoch,
        "inclination": record.inclination,
        "eccentricity": record.eccentricity,
        "ra_of_asc_node": record.ra_of_asc_node,
        "arg_of_pericenter": record.arg_of_pericenter,
        "mean_anomaly": record.mean_anomaly,
        "mean_motion": record.mean_motion,
        "bstar": record.bstar,
        "semimajor_axis": record.semimajor_axis,
        "period": record.period,
        "apoapsis": record.apoapsis,
        "periapsis": record.periapsis,
        "classification_type": record.classification_type,
        "object_type": record.object_type,
        "country_code": record.country_code,
        "launch_date": record.launch_date,
        "decay_date": record.decay_date,
        "tle_line0": record.tle_line0,
        "tle_line1": record.tle_line1,
        "tle_line2": record.tle_line2,
    }


VALID_ANGLE_FORMATS = {
    "degrees": brahe.AngleFormat.DEGREES,
    "radians": brahe.AngleFormat.RADIANS,
}


def error_response(message: str, **context) -> dict:
    """Build an error response dict with optional context for discoverability.

    Args:
        message: Human-readable error description.
        **context: Additional key-value pairs (e.g. valid_formats, valid_frames).

    Returns:
        Dict with "error" key and any extra context.
    """
    return {"error": message, **context}


def parse_epoch(value: str) -> brahe.Epoch:
    """Parse an ISO-format epoch string into a brahe.Epoch.

    Handles formats like "2024-01-01T12:00:00Z", "2024-01-01 12:00:00 UTC",
    and bare datetimes (assumed UTC).

    Args:
        value: ISO epoch string.

    Returns:
        Parsed brahe.Epoch.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    stripped = value.strip()
    try:
        return brahe.Epoch(stripped)
    except ValueError:
        pass
    normalized = stripped.replace("T", " ")
    return brahe.Epoch(f"{normalized} UTC")


def resolve_angle_format(s: str) -> brahe.AngleFormat:
    """Validate and resolve an angle format string to a brahe enum.

    Args:
        s: "degrees" or "radians" (case-insensitive).

    Returns:
        Corresponding brahe.AngleFormat.

    Raises:
        ValueError: If the string is not a valid angle format.
    """
    key = s.lower()
    if key not in VALID_ANGLE_FORMATS:
        raise ValueError(
            f"Invalid angle_format: {s!r}. Must be one of: {sorted(VALID_ANGLE_FORMATS.keys())}"
        )
    return VALID_ANGLE_FORMATS[key]
