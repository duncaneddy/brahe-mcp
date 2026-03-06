import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch

VALID_TIME_SYSTEMS = {
    "UTC": brahe.TimeSystem.UTC,
    "GPS": brahe.TimeSystem.GPS,
    "TAI": brahe.TimeSystem.TAI,
    "TT": brahe.TimeSystem.TT,
    "UT1": brahe.TimeSystem.UT1,
}

TIME_SYSTEM_DESCRIPTIONS = {
    "UTC": "Coordinated Universal Time - civil time standard with leap seconds",
    "GPS": "GPS Time - continuous time scale, no leap seconds, epoch Jan 6 1980",
    "TAI": "International Atomic Time - continuous time scale, basis for UTC",
    "TT": "Terrestrial Time - used in solar system ephemerides",
    "UT1": "Universal Time corrected for polar motion (requires EOP data)",
}

VALID_INPUT_FORMATS = {"iso", "mjd", "jd", "gps_seconds", "gps_nanoseconds", "gps_date"}
VALID_OUTPUT_FORMATS = {"iso", "iso_precise", "string", "mjd", "jd", "gps_seconds", "gps_nanoseconds", "gps_date"}


def _error_response(message: str) -> dict:
    return error_response(
        message,
        valid_input_formats=sorted(VALID_INPUT_FORMATS),
        valid_output_formats=sorted(VALID_OUTPUT_FORMATS),
        valid_time_systems=sorted(VALID_TIME_SYSTEMS.keys()),
    )


def _parse_epoch(value: str, input_format: str, time_system: brahe.TimeSystem) -> brahe.Epoch:
    """Parse a string value into a brahe.Epoch based on the input format."""
    if input_format == "iso":
        return parse_epoch(value)
    elif input_format == "mjd":
        return brahe.Epoch.from_mjd(float(value), time_system)
    elif input_format == "jd":
        return brahe.Epoch.from_jd(float(value), time_system)
    elif input_format == "gps_seconds":
        return brahe.Epoch.from_gps_seconds(float(value))
    elif input_format == "gps_nanoseconds":
        return brahe.Epoch.from_gps_nanoseconds(int(value))
    elif input_format == "gps_date":
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError("gps_date value must be 'week,seconds' (e.g. '2295,129618.0')")
        return brahe.Epoch.from_gps_date(int(parts[0].strip()), float(parts[1].strip()))
    else:
        raise ValueError(f"Unknown input_format: {input_format!r}")


def _format_output(epoch: brahe.Epoch, output_format: str, out_ts: brahe.TimeSystem) -> object:
    """Format an Epoch into the requested output representation."""
    if output_format in ("iso", "string"):
        return epoch.to_string_as_time_system(out_ts)
    elif output_format == "iso_precise":
        precise_epoch = brahe.Epoch.from_mjd(epoch.mjd_as_time_system(out_ts), out_ts)
        return precise_epoch.isostring_with_decimals(9)
    elif output_format == "mjd":
        return epoch.mjd_as_time_system(out_ts)
    elif output_format == "jd":
        return epoch.jd_as_time_system(out_ts)
    elif output_format == "gps_seconds":
        return epoch.gps_seconds()
    elif output_format == "gps_nanoseconds":
        return epoch.gps_nanoseconds()
    elif output_format == "gps_date":
        week, seconds = epoch.gps_date()
        return {"week": week, "seconds": seconds}
    else:
        raise ValueError(f"Unknown output_format: {output_format!r}")


@mcp.tool()
def list_time_systems() -> dict:
    """List all supported time systems for epoch conversion."""
    logger.debug("Listing time systems")
    return {
        "time_systems": [
            {"name": name, "description": TIME_SYSTEM_DESCRIPTIONS[name]}
            for name in sorted(VALID_TIME_SYSTEMS.keys())
        ]
    }


@mcp.tool()
def convert_epoch(
    value: str,
    input_format: str = "iso",
    input_time_system: str = "UTC",
    output_format: str = "iso",
    output_time_system: str = "",
) -> dict:
    """Convert an epoch between time representations and time systems.

    Supports ISO strings, MJD, JD, GPS seconds, GPS nanoseconds, and GPS date (week,seconds).
    Time systems: UTC, GPS, TAI, TT, UT1.

    Args:
        value: The epoch value as a string. For ISO: '2024-01-01T12:00:00Z'. For MJD/JD/GPS: numeric string. For gps_date: 'week,seconds'.
        input_format: Input format - one of: iso, mjd, jd, gps_seconds, gps_nanoseconds, gps_date.
        input_time_system: Time system of input (ignored for iso and gps_* formats). One of: UTC, GPS, TAI, TT, UT1.
        output_format: Output format - one of: iso, iso_precise, string, mjd, jd, gps_seconds, gps_nanoseconds, gps_date.
        output_time_system: Time system for output. Empty string means same as input (or GPS for gps_* inputs).
    """
    # Validate input_format
    if input_format not in VALID_INPUT_FORMATS:
        logger.warning("Invalid input_format: {}", input_format)
        return _error_response(f"Invalid input_format: {input_format!r}")

    # Validate output_format
    if output_format not in VALID_OUTPUT_FORMATS:
        logger.warning("Invalid output_format: {}", output_format)
        return _error_response(f"Invalid output_format: {output_format!r}")

    # Validate and resolve input time system
    input_ts_upper = input_time_system.upper()
    if input_ts_upper not in VALID_TIME_SYSTEMS:
        logger.warning("Invalid input_time_system: {}", input_time_system)
        return _error_response(f"Invalid input_time_system: {input_time_system!r}")
    in_ts = VALID_TIME_SYSTEMS[input_ts_upper]

    # Parse the epoch
    try:
        epoch = _parse_epoch(value, input_format, in_ts)
    except (ValueError, TypeError) as e:
        logger.warning("Failed to parse epoch value {!r}: {}", value, e)
        return _error_response(f"Failed to parse value {value!r} as {input_format}: {e}")
    except Exception as e:
        logger.error("Unexpected error parsing epoch: {}", e)
        return _error_response(f"Error parsing epoch: {e}")

    # Resolve output time system
    if output_time_system:
        out_ts_upper = output_time_system.upper()
        if out_ts_upper not in VALID_TIME_SYSTEMS:
            logger.warning("Invalid output_time_system: {}", output_time_system)
            return _error_response(f"Invalid output_time_system: {output_time_system!r}")
        out_ts = VALID_TIME_SYSTEMS[out_ts_upper]
    else:
        out_ts = epoch.time_system

    # Determine the output time system name
    out_ts_name = next(k for k, v in VALID_TIME_SYSTEMS.items() if v == out_ts)
    in_ts_name = next(k for k, v in VALID_TIME_SYSTEMS.items() if v == epoch.time_system)

    # Format output
    try:
        result_value = _format_output(epoch, output_format, out_ts)
    except Exception as e:
        logger.error("Failed to format epoch output: {}", e)
        return _error_response(f"Error formatting output as {output_format}: {e}")

    logger.debug("Converted epoch: {} {} -> {} {}", value, input_format, result_value, output_format)
    return {
        "input": {"value": value, "format": input_format, "time_system": in_ts_name},
        "output": {"value": result_value, "format": output_format, "time_system": out_ts_name},
    }
