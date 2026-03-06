"""Shared utilities for brahe-mcp tool modules."""

import brahe

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
