"""CelesTrak GP, supplemental GP, and SATCAT query MCP tools."""

import brahe
from brahe.celestrak import CelestrakClient, CelestrakQuery, SupGPSource
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response

_client = CelestrakClient()

_SUP_GP_SOURCES = {
    "spacex": SupGPSource.SPACEX,
    "spacex_sup": SupGPSource.SPACEX_SUP,
    "planet": SupGPSource.PLANET,
    "oneweb": SupGPSource.ONEWEB,
    "starlink": SupGPSource.STARLINK,
    "starlink_sup": SupGPSource.STARLINK_SUP,
    "geo": SupGPSource.GEO,
    "gps": SupGPSource.GPS,
    "glonass": SupGPSource.GLONASS,
    "meteosat": SupGPSource.METEOSAT,
    "intelsat": SupGPSource.INTELSAT,
    "ses": SupGPSource.SES,
    "iridium": SupGPSource.IRIDIUM,
    "iridium_next": SupGPSource.IRIDIUM_NEXT,
    "orbcomm": SupGPSource.ORBCOMM,
    "globalstar": SupGPSource.GLOBALSTAR,
    "swarm_technologies": SupGPSource.SWARM_TECHNOLOGIES,
    "amateur": SupGPSource.AMATEUR,
    "celestrak": SupGPSource.CELESTRAK,
    "kuiper": SupGPSource.KUIPER,
}

_GP_GROUPS = [
    "active", "analyst", "stations", "visual", "last-30-days",
    "starlink", "oneweb", "planet", "gnss", "gps-ops", "glonass-ops",
    "galileo", "beidou", "geo", "weather", "noaa", "resource",
    "sarsat", "dmc", "tdrss", "argos", "intelsat", "ses",
    "iridium", "iridium-NEXT", "orbcomm", "globalstar", "amateur",
    "x-comm", "other-comm", "satnogs", "gorizont", "raduga",
    "molniya", "science", "geodetic", "engineering", "education",
    "military", "radar", "cubesat", "other", "supplemental",
]

_FILTER_OPERATORS = {
    ">": "greater than (e.g. >50)",
    "<": "less than (e.g. <0.01)",
    "<>": "not equal (e.g. <>DEBRIS)",
    "--": "range (e.g. 25544--25600)",
    "~~": "contains, case-insensitive (e.g. ~~STARLINK)",
    "^": "starts with, case-insensitive (e.g. ^NOAA)",
}

_GP_RECORD_FIELDS = [
    "OBJECT_NAME", "NORAD_CAT_ID", "OBJECT_ID", "EPOCH",
    "INCLINATION", "ECCENTRICITY", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "MEAN_MOTION",
    "BSTAR", "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS",
    "CLASSIFICATION_TYPE", "OBJECT_TYPE", "COUNTRY_CODE",
    "LAUNCH_DATE", "DECAY_DATE", "RCS_SIZE",
]


def _serialize_gp_record(record: brahe.GPRecord) -> dict:
    """Extract key fields from a GPRecord into a plain dict."""
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


def _serialize_satcat_record(record) -> dict:
    """Extract key fields from a CelestrakSATCATRecord into a plain dict."""
    return {
        "object_name": record.object_name,
        "object_id": record.object_id,
        "norad_cat_id": record.norad_cat_id,
        "object_type": record.object_type,
        "ops_status_code": record.ops_status_code,
        "owner": record.owner,
        "launch_date": record.launch_date,
        "launch_site": record.launch_site,
        "decay_date": record.decay_date,
        "period": record.period,
        "inclination": record.inclination,
        "apogee": record.apogee,
        "perigee": record.perigee,
        "rcs": record.rcs,
        "orbit_center": record.orbit_center,
        "orbit_type": record.orbit_type,
    }


# ---------------------------------------------------------------------------
# Tool 1: list_celestrak_options
# ---------------------------------------------------------------------------


@mcp.tool()
def list_celestrak_options() -> dict:
    """List available CelesTrak lookup methods, groups, supplemental sources, and filter options.

    Returns discovery information for all CelesTrak tools including valid
    group names, supplemental GP sources, SATCAT query options, filter
    operators, and GP record field names.
    """
    logger.debug("Listing CelesTrak options")
    return {
        "lookup_methods": {
            "get_celestrak_gp": "Look up GP records by catnr, group, name, or intdes",
            "get_celestrak_sup_gp": "Look up supplemental GP records by source",
            "get_celestrak_satcat": "Look up SATCAT records",
            "query_celestrak": "Advanced query builder with filtering and sorting",
        },
        "gp_groups": _GP_GROUPS,
        "supplemental_sources": sorted(_SUP_GP_SOURCES.keys()),
        "satcat_options": {
            "catnr": "NORAD catalog number (int)",
            "active": "Filter to active objects (bool)",
            "payloads": "Filter to payloads only (bool)",
            "on_orbit": "Filter to on-orbit objects (bool)",
        },
        "filter_operators": _FILTER_OPERATORS,
        "gp_record_fields": _GP_RECORD_FIELDS,
    }


# ---------------------------------------------------------------------------
# Tool 2: get_celestrak_gp
# ---------------------------------------------------------------------------


@mcp.tool()
def get_celestrak_gp(
    catnr: int | None = None,
    group: str | None = None,
    name: str | None = None,
    intdes: str | None = None,
    limit: int | None = None,
) -> dict:
    """Look up GP (General Perturbations) ephemeris records from CelesTrak.

    Provide exactly one identifier to query by. Use list_celestrak_options()
    to see available groups and other options.

    Args:
        catnr: NORAD catalog number (e.g. 25544 for ISS).
        group: Satellite group name (e.g. "stations", "active", "starlink").
        name: Satellite name substring search.
        intdes: International designator (e.g. "1998-067A").
        limit: Maximum number of records to return.
    """
    identifiers = {
        "catnr": catnr,
        "group": group,
        "name": name,
        "intdes": intdes,
    }
    provided = {k: v for k, v in identifiers.items() if v is not None}

    if len(provided) == 0:
        return error_response(
            "Provide exactly one identifier: catnr, group, name, or intdes",
            valid_groups=_GP_GROUPS,
        )
    if len(provided) > 1:
        return error_response(
            f"Provide exactly one identifier, got {len(provided)}: {list(provided.keys())}",
        )

    method, identifier = next(iter(provided.items()))
    logger.info("CelesTrak GP lookup: {}={}", method, identifier)

    try:
        records = _client.get_gp(**provided)
    except Exception as exc:
        logger.error("CelesTrak GP error: {}", exc)
        return error_response(f"CelesTrak GP lookup failed: {exc}")

    serialized = [_serialize_gp_record(r) for r in records]
    if limit is not None and limit > 0:
        serialized = serialized[:limit]

    return {
        "method": method,
        "identifier": identifier,
        "count": len(serialized),
        "records": serialized,
    }


# ---------------------------------------------------------------------------
# Tool 3: get_celestrak_sup_gp
# ---------------------------------------------------------------------------


@mcp.tool()
def get_celestrak_sup_gp(
    source: str,
    limit: int | None = None,
) -> dict:
    """Look up supplemental GP records from CelesTrak by source.

    Supplemental GP data provides operator-provided ephemerides that may
    be more accurate than standard GP data for certain constellations.

    Args:
        source: Supplemental source name (case-insensitive), e.g. "starlink", "spacex".
        limit: Maximum number of records to return.
    """
    source_lower = source.lower()
    if source_lower not in _SUP_GP_SOURCES:
        return error_response(
            f"Unknown supplemental GP source: {source!r}",
            valid_sources=sorted(_SUP_GP_SOURCES.keys()),
        )

    sup_source = _SUP_GP_SOURCES[source_lower]
    logger.info("CelesTrak supplemental GP lookup: {}", source)

    try:
        records = _client.get_sup_gp(sup_source)
    except Exception as exc:
        logger.error("CelesTrak supplemental GP error: {}", exc)
        return error_response(f"CelesTrak supplemental GP lookup failed: {exc}")

    serialized = [_serialize_gp_record(r) for r in records]
    if limit is not None and limit > 0:
        serialized = serialized[:limit]

    return {
        "source": source_lower,
        "count": len(serialized),
        "records": serialized,
    }


# ---------------------------------------------------------------------------
# Tool 4: get_celestrak_satcat
# ---------------------------------------------------------------------------


@mcp.tool()
def get_celestrak_satcat(
    catnr: int | None = None,
    active: bool | None = None,
    payloads: bool | None = None,
    on_orbit: bool | None = None,
    limit: int | None = None,
) -> dict:
    """Look up satellite catalog (SATCAT) records from CelesTrak.

    At least one filter parameter must be provided.

    Args:
        catnr: NORAD catalog number.
        active: Filter to active objects only.
        payloads: Filter to payloads only.
        on_orbit: Filter to on-orbit objects only.
        limit: Maximum number of records to return.
    """
    kwargs = {}
    if catnr is not None:
        kwargs["catnr"] = catnr
    if active is not None:
        kwargs["active"] = active
    if payloads is not None:
        kwargs["payloads"] = payloads
    if on_orbit is not None:
        kwargs["on_orbit"] = on_orbit

    if not kwargs:
        return error_response(
            "At least one parameter required: catnr, active, payloads, or on_orbit",
        )

    logger.info("CelesTrak SATCAT lookup: {}", kwargs)

    try:
        records = _client.get_satcat(**kwargs)
    except Exception as exc:
        logger.error("CelesTrak SATCAT error: {}", exc)
        return error_response(f"CelesTrak SATCAT lookup failed: {exc}")

    serialized = [_serialize_satcat_record(r) for r in records]
    if limit is not None and limit > 0:
        serialized = serialized[:limit]

    return {
        "filters": kwargs,
        "count": len(serialized),
        "records": serialized,
    }


# ---------------------------------------------------------------------------
# Tool 5: query_celestrak
# ---------------------------------------------------------------------------


@mcp.tool()
def query_celestrak(
    query_type: str,
    group: str | None = None,
    name: str | None = None,
    catnr: int | None = None,
    intdes: str | None = None,
    source: str | None = None,
    active: bool | None = None,
    payloads: bool | None = None,
    on_orbit: bool | None = None,
    filters: list[dict] | None = None,
    order_by: str | None = None,
    order_ascending: bool = True,
    limit: int | None = None,
) -> dict:
    """Build and execute an advanced CelesTrak query with filtering and sorting.

    This tool provides full access to the CelesTrak query builder, including
    client-side filtering and ordering that the simpler tools don't expose.

    Args:
        query_type: One of "gp", "sup_gp", or "satcat" (case-insensitive).
        group: Satellite group name (GP queries).
        name: Satellite name search (GP queries).
        catnr: NORAD catalog number (GP/SATCAT queries).
        intdes: International designator (GP queries).
        source: Supplemental source name (sup_gp queries).
        active: Filter to active objects (SATCAT queries).
        payloads: Filter to payloads only (SATCAT queries).
        on_orbit: Filter to on-orbit objects (SATCAT queries).
        filters: List of filter dicts with "field" and "value" keys,
            e.g. [{"field": "INCLINATION", "value": ">50"}].
        order_by: Field name to sort results by.
        order_ascending: Sort ascending (True) or descending (False).
        limit: Maximum number of records to return.
    """
    qt = query_type.lower()
    valid_types = ["gp", "sup_gp", "satcat"]
    if qt not in valid_types:
        return error_response(
            f"Unknown query_type: {query_type!r}",
            valid_types=valid_types,
        )

    try:
        if qt == "gp":
            query = CelestrakQuery.gp
            if group is not None:
                query = query.group(group)
            if name is not None:
                query = query.name_search(name)
            if catnr is not None:
                query = query.catnr(catnr)
            if intdes is not None:
                query = query.intdes(intdes)

        elif qt == "sup_gp":
            query = CelestrakQuery.sup_gp
            if source is None:
                return error_response(
                    "source is required for sup_gp queries",
                    valid_sources=sorted(_SUP_GP_SOURCES.keys()),
                )
            source_lower = source.lower()
            if source_lower not in _SUP_GP_SOURCES:
                return error_response(
                    f"Unknown supplemental GP source: {source!r}",
                    valid_sources=sorted(_SUP_GP_SOURCES.keys()),
                )
            query = query.source(_SUP_GP_SOURCES[source_lower])

        else:  # satcat
            query = CelestrakQuery.satcat
            if catnr is not None:
                query = query.catnr(catnr)
            if active is not None:
                query = query.active(active)
            if payloads is not None:
                query = query.payloads(payloads)
            if on_orbit is not None:
                query = query.on_orbit(on_orbit)

        # Apply filters
        if filters:
            for f in filters:
                field = f.get("field", "")
                value = f.get("value", "")
                query = query.filter(field, value)

        # Apply ordering
        if order_by is not None:
            query = query.order_by(order_by, order_ascending)

        # Apply limit
        if limit is not None and limit > 0:
            query = query.limit(limit)

        logger.info("CelesTrak query: {}", query)
        records = _client.query(query)

    except Exception as exc:
        logger.error("CelesTrak query error: {}", exc)
        return error_response(f"CelesTrak query failed: {exc}")

    if qt == "satcat":
        serialized = [_serialize_satcat_record(r) for r in records]
    else:
        serialized = [_serialize_gp_record(r) for r in records]

    return {
        "query_type": qt,
        "count": len(serialized),
        "records": serialized,
    }
