"""Groundstation location query MCP tools."""

import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response


def _serialize_station(station) -> dict:
    """Extract key fields from a PointLocation into a plain dict."""
    props = dict(station.properties)
    return {
        "name": station.get_name(),
        "lon": station.lon,
        "lat": station.lat,
        "altitude_m": station.alt,
        "properties": props,
    }


_PROVIDER_MAP = {
    "atlas": "atlas",
    "aws": "aws",
    "ksat": "ksat",
    "leaf": "leaf",
    "nasa dsn": "nasa dsn",
    "nasa nen": "nasa nen",
    "ssc": "ssc",
    "viasat": "viasat",
}


# ---------------------------------------------------------------------------
# Tool 1: list_groundstation_options
# ---------------------------------------------------------------------------


@mcp.tool()
def list_groundstation_options() -> dict:
    """List available groundstation providers and query options.

    Returns discovery information for groundstation tools including
    available providers and field descriptions.
    """
    logger.debug("Listing groundstation options")
    return {
        "providers": brahe.datasets.groundstations.list_providers(),
        "lookup_methods": {
            "get_groundstations": "Load stations by provider or all providers",
            "query_groundstations": "Advanced query with geographic and attribute filtering",
        },
        "station_fields": ["name", "lon", "lat", "altitude_m", "properties"],
        "property_fields": ["provider", "frequency_bands"],
        "query_filters": {
            "provider": "Provider name (case-insensitive)",
            "name": "Station name substring search (case-insensitive)",
            "lat_min/lat_max": "Latitude range in degrees",
            "lon_min/lon_max": "Longitude range in degrees",
            "frequency_band": "Required frequency band (e.g. 'S', 'X', 'Ka')",
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: get_groundstations
# ---------------------------------------------------------------------------


@mcp.tool()
def get_groundstations(
    provider: str | None = None,
    limit: int | None = None,
) -> dict:
    """Load groundstation locations by provider or all providers.

    Args:
        provider: Provider name (case-insensitive). Use list_groundstation_options()
            to see available providers. If omitted, loads all stations.
        limit: Maximum number of stations to return.
    """
    logger.info("Loading groundstations, provider={}", provider)

    try:
        if provider is not None:
            provider_lower = provider.lower()
            if provider_lower not in _PROVIDER_MAP:
                return error_response(
                    f"Unknown provider: {provider!r}",
                    valid_providers=brahe.datasets.groundstations.list_providers(),
                )
            stations = brahe.datasets.groundstations.load(_PROVIDER_MAP[provider_lower])
        else:
            stations = brahe.datasets.groundstations.load_all()
    except Exception as exc:
        logger.error("Groundstations error: {}", exc)
        return error_response(f"Groundstation load failed: {exc}")

    serialized = [_serialize_station(s) for s in stations]
    if limit is not None and limit > 0:
        serialized = serialized[:limit]

    return {
        "provider": provider.lower() if provider else "all",
        "count": len(serialized),
        "stations": serialized,
    }


# ---------------------------------------------------------------------------
# Tool 3: query_groundstations
# ---------------------------------------------------------------------------


@mcp.tool()
def query_groundstations(
    provider: str | None = None,
    name: str | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    frequency_band: str | None = None,
    limit: int | None = None,
) -> dict:
    """Query groundstations with geographic and attribute filters.

    At least one filter parameter must be provided. All filters are
    combined (AND logic).

    Args:
        provider: Provider name (case-insensitive).
        name: Station name substring search (case-insensitive).
        lat_min: Minimum latitude in degrees.
        lat_max: Maximum latitude in degrees.
        lon_min: Minimum longitude in degrees.
        lon_max: Maximum longitude in degrees.
        frequency_band: Required frequency band (e.g. "S", "X", "Ka").
        limit: Maximum number of stations to return.
    """
    if all(v is None for v in [provider, name, lat_min, lat_max, lon_min, lon_max, frequency_band]):
        return error_response(
            "At least one filter parameter is required",
        )

    filters_applied = []
    logger.info("Querying groundstations with filters")

    try:
        # Load stations based on provider filter
        if provider is not None:
            provider_lower = provider.lower()
            if provider_lower not in _PROVIDER_MAP:
                return error_response(
                    f"Unknown provider: {provider!r}",
                    valid_providers=brahe.datasets.groundstations.list_providers(),
                )
            stations = brahe.datasets.groundstations.load(_PROVIDER_MAP[provider_lower])
            filters_applied.append(f"provider={provider_lower!r}")
        else:
            stations = brahe.datasets.groundstations.load_all()

        # Apply client-side filters
        if name is not None:
            name_lower = name.lower()
            stations = [s for s in stations if name_lower in s.get_name().lower()]
            filters_applied.append(f"name={name!r}")

        if lat_min is not None:
            stations = [s for s in stations if s.lat >= lat_min]
            filters_applied.append(f"lat_min={lat_min}")
        if lat_max is not None:
            stations = [s for s in stations if s.lat <= lat_max]
            filters_applied.append(f"lat_max={lat_max}")

        if lon_min is not None:
            stations = [s for s in stations if s.lon >= lon_min]
            filters_applied.append(f"lon_min={lon_min}")
        if lon_max is not None:
            stations = [s for s in stations if s.lon <= lon_max]
            filters_applied.append(f"lon_max={lon_max}")

        if frequency_band is not None:
            band_upper = frequency_band.upper()
            stations = [
                s for s in stations
                if band_upper in [b.upper() for b in dict(s.properties).get("frequency_bands", [])]
            ]
            filters_applied.append(f"frequency_band={frequency_band!r}")

    except Exception as exc:
        logger.error("Groundstations query error: {}", exc)
        return error_response(f"Groundstation query failed: {exc}")

    serialized = [_serialize_station(s) for s in stations]
    if limit is not None and limit > 0:
        serialized = serialized[:limit]

    return {
        "filters": filters_applied,
        "count": len(serialized),
        "stations": serialized,
    }
