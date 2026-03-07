"""Shared utilities for brahe-mcp tool modules."""

import re
from datetime import datetime, timedelta

import brahe


_INTERVAL_PATTERN = re.compile(r"^(\d+)\s*([smhdw])$", re.IGNORECASE)

_INTERVAL_UNITS: dict[str, str] = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def parse_epoch_datetime(epoch_str: str) -> datetime:
    """Convert an epoch string to a naive datetime.

    Handles ISO-format strings with optional ``Z`` or ``UTC`` suffixes.

    Args:
        epoch_str: Epoch string, e.g. ``"2024-01-01T00:00:00Z"``.

    Returns:
        Naive ``datetime`` (assumed UTC).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    cleaned = epoch_str.strip()
    for suffix in (" UTC", "Z"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    cleaned = cleaned.replace("T", " ", 1)
    return datetime.fromisoformat(cleaned)


def parse_decimation_interval(interval: str) -> timedelta:
    """Parse a human-readable interval string into a ``timedelta``.

    Supported formats: ``"30s"``, ``"5m"``, ``"12h"``, ``"1d"``, ``"1w"``.

    Args:
        interval: Interval string with a numeric value and unit suffix.

    Returns:
        Corresponding ``timedelta``.

    Raises:
        ValueError: If the format is invalid.
    """
    match = _INTERVAL_PATTERN.match(interval.strip())
    if not match:
        raise ValueError(
            f"Invalid decimation interval: {interval!r}. "
            "Use a number followed by s/m/h/d/w (e.g. '1d', '12h', '30m')."
        )
    value = int(match.group(1))
    unit_key = match.group(2).lower()
    return timedelta(**{_INTERVAL_UNITS[unit_key]: value})


def decimate_records(
    records: list[dict],
    interval: timedelta,
    epoch_key: str = "epoch",
    group_key: str = "norad_cat_id",
) -> list[dict]:
    """Thin a list of records to at most one per time interval.

    Records are grouped by ``group_key`` (satellite ID), sorted by epoch
    within each group, and thinned so consecutive kept records are at least
    ``interval`` apart.  The first and last record of each group are always
    preserved.

    Args:
        records: List of record dicts with epoch strings.
        interval: Minimum time between kept records.
        epoch_key: Dict key containing the epoch string.
        group_key: Dict key to group records by (e.g. satellite ID).

    Returns:
        Decimated list of records, preserving original dict ordering within
        groups.
    """
    if len(records) <= 2:
        return list(records)

    # Group by satellite
    groups: dict[str, list[tuple[datetime, dict]]] = {}
    for rec in records:
        key = rec.get(group_key, "")
        try:
            dt = parse_epoch_datetime(rec.get(epoch_key, ""))
        except (ValueError, TypeError):
            # Keep records with unparseable epochs
            dt = datetime.min
        groups.setdefault(key, []).append((dt, rec))

    result: list[dict] = []
    for _key, items in groups.items():
        items.sort(key=lambda x: x[0])
        if len(items) <= 2:
            result.extend(rec for _, rec in items)
            continue

        # Always keep first
        kept = [items[0]]
        last_kept_dt = items[0][0]

        for item in items[1:-1]:
            if item[0] - last_kept_dt >= interval:
                kept.append(item)
                last_kept_dt = item[0]

        # Always keep last
        kept.append(items[-1])
        result.extend(rec for _, rec in kept)

    return result

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
