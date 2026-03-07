"""SpaceTrack GP, CDM, Decay, and SATCAT query MCP tools."""

import os

from brahe import (
    RequestClass,
    SATCATRecord,
    SortOrder,
    SpaceTrackClient,
    SpaceTrackQuery,
)
from brahe.spacetrack import operators
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, serialize_gp_record

# ---------------------------------------------------------------------------
# Lazy client initialization (requires env vars)
# ---------------------------------------------------------------------------

_client: SpaceTrackClient | None = None


def _get_client() -> SpaceTrackClient:
    """Return a shared SpaceTrackClient, creating it on first call."""
    global _client
    if _client is None:
        user = os.environ.get("SPACETRACK_USER")
        password = os.environ.get("SPACETRACK_PASS")
        if not user or not password:
            raise RuntimeError(
                "Set SPACETRACK_USER and SPACETRACK_PASS environment variables."
            )
        _client = SpaceTrackClient(identity=user, password=password)
    return _client


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUEST_CLASSES = {
    "gp": RequestClass.GP,
    "gp_history": RequestClass.GP_HISTORY,
    "satcat": RequestClass.SATCAT,
    "cdm_public": RequestClass.CDM_PUBLIC,
    "decay": RequestClass.DECAY,
}

_GP_FIELDS = [
    "CCSDS_OMM_VERS", "COMMENT", "CREATION_DATE", "ORIGINATOR",
    "OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME",
    "TIME_SYSTEM", "MEAN_ELEMENT_THEORY", "EPOCH", "MEAN_MOTION",
    "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER",
    "MEAN_ANOMALY", "EPHEMERIS_TYPE", "CLASSIFICATION_TYPE", "NORAD_CAT_ID",
    "ELEMENT_SET_NO", "REV_AT_EPOCH", "BSTAR", "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT", "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS",
    "PERIAPSIS", "OBJECT_TYPE", "RCS_SIZE", "COUNTRY_CODE",
    "LAUNCH_DATE", "SITE", "DECAY_DATE", "FILE", "GP_ID",
    "TLE_LINE0", "TLE_LINE1", "TLE_LINE2",
]

_CDM_FIELDS = [
    "CDM_ID", "CREATED", "EMERGENCY_REPORTABLE", "TCA", "MIN_RNG",
    "PC", "SAT_1_ID", "SAT_1_NAME", "SAT1_OBJECT_TYPE",
    "SAT1_RCS", "SAT_1_EXCL_VOL", "SAT_2_ID", "SAT_2_NAME",
    "SAT2_OBJECT_TYPE", "SAT2_RCS", "SAT_2_EXCL_VOL",
]

_DECAY_FIELDS = [
    "NORAD_CAT_ID", "OBJECT_NUMBER", "OBJECT_NAME", "INTLDES",
    "OBJECT_ID", "RCS", "RCS_SIZE", "COUNTRY", "MSG_EPOCH",
    "DECAY_EPOCH", "SOURCE", "MSG_TYPE", "PRECEDENCE",
]

_SATCAT_FIELDS = [
    "INTLDES", "NORAD_CAT_ID", "OBJECT_TYPE", "SATNAME", "COUNTRY",
    "LAUNCH", "SITE", "DECAY", "PERIOD", "INCLINATION", "APOGEE",
    "PERIGEE", "COMMENT", "COMMENTCODE", "RCSVALUE", "RCS_SIZE",
    "FILE", "LAUNCH_YEAR", "LAUNCH_NUM", "LAUNCH_PIECE",
    "CURRENT", "OBJECT_NAME", "OBJECT_ID", "OBJECT_NUMBER",
]

_OPERATORS = {
    "greater_than": "Greater than comparison (e.g. operators.greater_than(50))",
    "less_than": "Less than comparison (e.g. operators.less_than(0.01))",
    "not_equal": "Not equal (e.g. operators.not_equal('DEBRIS'))",
    "inclusive_range": "Range filter (e.g. operators.inclusive_range('2024-01-01', '2024-12-31'))",
    "like": "Contains substring, case-insensitive (e.g. operators.like('STARLINK'))",
    "startswith": "Starts with, case-insensitive (e.g. operators.startswith('NOAA'))",
    "or_list": "Match any in list (e.g. operators.or_list(25544, 25545))",
    "now": "Current UTC datetime placeholder",
    "now_offset": "Offset from current UTC (e.g. operators.now_offset(days=-30))",
    "null_val": "Match null/empty values",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_satcat_record(record: SATCATRecord) -> dict:
    """Extract key fields from a SpaceTrack SATCATRecord into a plain dict."""
    return {
        "intldes": record.intldes,
        "norad_cat_id": record.norad_cat_id,
        "object_type": record.object_type,
        "satname": record.satname,
        "country": record.country,
        "launch": record.launch,
        "site": record.site,
        "decay": record.decay,
        "period": record.period,
        "inclination": record.inclination,
        "apogee": record.apogee,
        "perigee": record.perigee,
        "rcsvalue": record.rcsvalue,
        "rcs_size": record.rcs_size,
        "object_name": record.object_name,
        "object_id": record.object_id,
        "object_number": record.object_number,
    }


def _dispatch_query(request_class_str: str, query: SpaceTrackQuery) -> list[dict]:
    """Execute a query and serialize results based on request class.

    Args:
        request_class_str: Lowercase request class key (e.g. "gp", "satcat").
        query: Built SpaceTrackQuery to execute.

    Returns:
        List of serialized record dicts.
    """
    client = _get_client()
    if request_class_str in ("gp", "gp_history"):
        records = client.query_gp(query)
        return [serialize_gp_record(r) for r in records]
    elif request_class_str == "satcat":
        records = client.query_satcat(query)
        return [_serialize_satcat_record(r) for r in records]
    else:
        return client.query_json(query)


# ---------------------------------------------------------------------------
# Tool 1: list_spacetrack_options
# ---------------------------------------------------------------------------


@mcp.tool()
def list_spacetrack_options() -> dict:
    """List available SpaceTrack tools, request classes, filter fields, and operators.

    Returns discovery information for all SpaceTrack tools. No authentication
    required for this tool — it returns static reference data.
    """
    logger.debug("Listing SpaceTrack options")
    return {
        "tools": {
            "get_spacetrack_gp": "Query GP records by NORAD ID, name, or epoch range",
            "get_spacetrack_gp_history": "Query historical GP records for orbital evolution over time",
            "get_spacetrack_satcat": "Query SATCAT records by NORAD ID, name, country, or type",
            "get_spacetrack_cdm": "Query conjunction data messages (CDM) by NORAD ID or epoch",
            "get_spacetrack_decay": "Query decay predictions by NORAD ID or epoch",
            "query_spacetrack": "Advanced query builder with full filter/sort/pagination support",
        },
        "request_classes": sorted(_REQUEST_CLASSES.keys()),
        "fields": {
            "gp": _GP_FIELDS,
            "gp_history": _GP_FIELDS,
            "cdm_public": _CDM_FIELDS,
            "decay": _DECAY_FIELDS,
            "satcat": _SATCAT_FIELDS,
        },
        "operators": _OPERATORS,
        "sort_orders": ["asc", "desc"],
    }


# ---------------------------------------------------------------------------
# Tool 2: get_spacetrack_gp
# ---------------------------------------------------------------------------


@mcp.tool()
def get_spacetrack_gp(
    norad_cat_id: int | None = None,
    name: str | None = None,
    epoch_range: str | None = None,
    limit: int | None = None,
) -> dict:
    """Query GP (General Perturbations) records from SpaceTrack.

    At least one filter parameter must be provided. Requires SPACETRACK_USER
    and SPACETRACK_PASS environment variables.

    Args:
        norad_cat_id: NORAD catalog number (e.g. 25544 for ISS).
        name: Satellite name substring search (case-insensitive).
        epoch_range: ISO datetime range string "start--end" to filter by EPOCH
            (e.g. "2024-01-01--2024-01-31").
        limit: Maximum number of records to return.
    """
    if norad_cat_id is None and name is None and epoch_range is None:
        return error_response(
            "At least one filter required: norad_cat_id, name, or epoch_range",
        )

    try:
        query = SpaceTrackQuery(RequestClass.GP)

        if norad_cat_id is not None:
            query = query.filter("NORAD_CAT_ID", str(norad_cat_id))
        if name is not None:
            query = query.filter("OBJECT_NAME", operators.like(name))
        if epoch_range is not None:
            query = query.filter("EPOCH", epoch_range)
        if limit is not None and limit > 0:
            query = query.limit(limit)

        logger.info("SpaceTrack GP query: norad_cat_id={}, name={}, epoch_range={}",
                     norad_cat_id, name, epoch_range)
        records = _dispatch_query("gp", query)

    except RuntimeError as exc:
        logger.error("SpaceTrack auth error: {}", exc)
        return error_response(str(exc))
    except Exception as exc:
        logger.error("SpaceTrack GP error: {}", exc)
        return error_response(f"SpaceTrack GP query failed: {exc}")

    return {
        "request_class": "gp",
        "count": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Tool 3: get_spacetrack_gp_history
# ---------------------------------------------------------------------------


@mcp.tool()
def get_spacetrack_gp_history(
    norad_cat_id: int | None = None,
    name: str | None = None,
    epoch_range: str | None = None,
    limit: int | None = None,
) -> dict:
    """Query historical GP records from SpaceTrack.

    Unlike get_spacetrack_gp which returns only the latest element set,
    this tool returns all historical element sets for matching objects.
    Useful for tracking orbital evolution over time.

    At least one filter parameter must be provided. An epoch_range is
    strongly recommended to avoid very large result sets. Requires
    SPACETRACK_USER and SPACETRACK_PASS environment variables.

    Args:
        norad_cat_id: NORAD catalog number (e.g. 25544 for ISS).
        name: Satellite name substring search (case-insensitive).
        epoch_range: ISO datetime range string "start--end" to filter by EPOCH
            (e.g. "2024-01-01--2024-01-31").
        limit: Maximum number of records to return.
    """
    if norad_cat_id is None and name is None and epoch_range is None:
        return error_response(
            "At least one filter required: norad_cat_id, name, or epoch_range",
        )

    try:
        query = SpaceTrackQuery(RequestClass.GP_HISTORY)

        if norad_cat_id is not None:
            query = query.filter("NORAD_CAT_ID", str(norad_cat_id))
        if name is not None:
            query = query.filter("OBJECT_NAME", operators.like(name))
        if epoch_range is not None:
            query = query.filter("EPOCH", epoch_range)
        if limit is not None and limit > 0:
            query = query.limit(limit)

        logger.info("SpaceTrack GP History query: norad_cat_id={}, name={}, epoch_range={}",
                     norad_cat_id, name, epoch_range)
        records = _dispatch_query("gp_history", query)

    except RuntimeError as exc:
        logger.error("SpaceTrack auth error: {}", exc)
        return error_response(str(exc))
    except Exception as exc:
        logger.error("SpaceTrack GP History error: {}", exc)
        return error_response(f"SpaceTrack GP History query failed: {exc}")

    return {
        "request_class": "gp_history",
        "count": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Tool 4: get_spacetrack_satcat
# ---------------------------------------------------------------------------


@mcp.tool()
def get_spacetrack_satcat(
    norad_cat_id: int | None = None,
    name: str | None = None,
    country: str | None = None,
    object_type: str | None = None,
    limit: int | None = None,
) -> dict:
    """Query satellite catalog (SATCAT) records from SpaceTrack.

    At least one filter parameter must be provided. Requires SPACETRACK_USER
    and SPACETRACK_PASS environment variables.

    Args:
        norad_cat_id: NORAD catalog number.
        name: Satellite name substring search (case-insensitive).
        country: Country code filter (e.g. "US", "CIS").
        object_type: Object type filter (e.g. "PAYLOAD", "ROCKET BODY", "DEBRIS").
        limit: Maximum number of records to return.
    """
    if norad_cat_id is None and name is None and country is None and object_type is None:
        return error_response(
            "At least one filter required: norad_cat_id, name, country, or object_type",
        )

    try:
        query = SpaceTrackQuery(RequestClass.SATCAT)

        if norad_cat_id is not None:
            query = query.filter("NORAD_CAT_ID", str(norad_cat_id))
        if name is not None:
            query = query.filter("SATNAME", operators.like(name))
        if country is not None:
            query = query.filter("COUNTRY", country)
        if object_type is not None:
            query = query.filter("OBJECT_TYPE", object_type)
        if limit is not None and limit > 0:
            query = query.limit(limit)

        logger.info("SpaceTrack SATCAT query: norad_cat_id={}, name={}, country={}, object_type={}",
                     norad_cat_id, name, country, object_type)
        records = _dispatch_query("satcat", query)

    except RuntimeError as exc:
        logger.error("SpaceTrack auth error: {}", exc)
        return error_response(str(exc))
    except Exception as exc:
        logger.error("SpaceTrack SATCAT error: {}", exc)
        return error_response(f"SpaceTrack SATCAT query failed: {exc}")

    return {
        "request_class": "satcat",
        "count": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Tool 4: get_spacetrack_cdm
# ---------------------------------------------------------------------------


@mcp.tool()
def get_spacetrack_cdm(
    norad_cat_id: int | None = None,
    epoch_range: str | None = None,
    limit: int | None = None,
) -> dict:
    """Query conjunction data messages (CDM) from SpaceTrack.

    At least one filter parameter must be provided. Returns raw CDM records
    as dicts. Requires SPACETRACK_USER and SPACETRACK_PASS environment variables.

    Args:
        norad_cat_id: NORAD catalog number for SAT_1_ID or SAT_2_ID.
        epoch_range: ISO datetime range string "start--end" to filter by TCA
            (e.g. "2024-01-01--2024-01-31").
        limit: Maximum number of records to return.
    """
    if norad_cat_id is None and epoch_range is None:
        return error_response(
            "At least one filter required: norad_cat_id or epoch_range",
        )

    try:
        query = SpaceTrackQuery(RequestClass.CDM_PUBLIC)

        if norad_cat_id is not None:
            query = query.filter("SAT_1_ID", str(norad_cat_id))
        if epoch_range is not None:
            query = query.filter("TCA", epoch_range)
        if limit is not None and limit > 0:
            query = query.limit(limit)

        logger.info("SpaceTrack CDM query: norad_cat_id={}, epoch_range={}",
                     norad_cat_id, epoch_range)
        records = _dispatch_query("cdm_public", query)

    except RuntimeError as exc:
        logger.error("SpaceTrack auth error: {}", exc)
        return error_response(str(exc))
    except Exception as exc:
        logger.error("SpaceTrack CDM error: {}", exc)
        return error_response(f"SpaceTrack CDM query failed: {exc}")

    return {
        "request_class": "cdm_public",
        "count": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Tool 5: get_spacetrack_decay
# ---------------------------------------------------------------------------


@mcp.tool()
def get_spacetrack_decay(
    norad_cat_id: int | None = None,
    epoch_range: str | None = None,
    limit: int | None = None,
) -> dict:
    """Query decay predictions from SpaceTrack.

    At least one filter parameter must be provided. Returns raw decay records
    as dicts. Requires SPACETRACK_USER and SPACETRACK_PASS environment variables.

    Args:
        norad_cat_id: NORAD catalog number.
        epoch_range: ISO datetime range string "start--end" to filter by MSG_EPOCH
            (e.g. "2024-01-01--2024-01-31").
        limit: Maximum number of records to return.
    """
    if norad_cat_id is None and epoch_range is None:
        return error_response(
            "At least one filter required: norad_cat_id or epoch_range",
        )

    try:
        query = SpaceTrackQuery(RequestClass.DECAY)

        if norad_cat_id is not None:
            query = query.filter("NORAD_CAT_ID", str(norad_cat_id))
        if epoch_range is not None:
            query = query.filter("MSG_EPOCH", epoch_range)
        if limit is not None and limit > 0:
            query = query.limit(limit)

        logger.info("SpaceTrack Decay query: norad_cat_id={}, epoch_range={}",
                     norad_cat_id, epoch_range)
        records = _dispatch_query("decay", query)

    except RuntimeError as exc:
        logger.error("SpaceTrack auth error: {}", exc)
        return error_response(str(exc))
    except Exception as exc:
        logger.error("SpaceTrack Decay error: {}", exc)
        return error_response(f"SpaceTrack Decay query failed: {exc}")

    return {
        "request_class": "decay",
        "count": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Tool 6: query_spacetrack
# ---------------------------------------------------------------------------


@mcp.tool()
def query_spacetrack(
    request_class: str,
    filters: list[dict] | None = None,
    order_by: str | None = None,
    order_ascending: bool = True,
    limit: int | None = None,
    offset: int | None = None,
) -> dict:
    """Build and execute an advanced SpaceTrack query with full control.

    Provides direct access to the SpaceTrack query builder with arbitrary
    filters, sorting, and pagination. Requires SPACETRACK_USER and
    SPACETRACK_PASS environment variables.

    Args:
        request_class: One of "gp", "gp_history", "satcat", "cdm_public", "decay" (case-insensitive).
        filters: List of filter dicts with "field" and "value" keys,
            e.g. [{"field": "NORAD_CAT_ID", "value": "25544"}].
        order_by: Field name to sort results by.
        order_ascending: Sort ascending (True) or descending (False).
        limit: Maximum number of records to return.
        offset: Number of records to skip (requires limit).
    """
    rc_key = request_class.lower()
    if rc_key not in _REQUEST_CLASSES:
        return error_response(
            f"Unknown request_class: {request_class!r}",
            valid_classes=sorted(_REQUEST_CLASSES.keys()),
        )

    try:
        query = SpaceTrackQuery(_REQUEST_CLASSES[rc_key])

        if filters:
            for f in filters:
                field = f.get("field", "")
                value = f.get("value", "")
                query = query.filter(field, value)

        if order_by is not None:
            sort_order = SortOrder.ASC if order_ascending else SortOrder.DESC
            query = query.order_by(order_by, sort_order)

        if limit is not None and limit > 0:
            if offset is not None and offset > 0:
                query = query.limit_offset(limit, offset)
            else:
                query = query.limit(limit)

        logger.info("SpaceTrack query: class={}, filters={}", rc_key, filters)
        records = _dispatch_query(rc_key, query)

    except RuntimeError as exc:
        logger.error("SpaceTrack auth error: {}", exc)
        return error_response(str(exc))
    except Exception as exc:
        logger.error("SpaceTrack query error: {}", exc)
        return error_response(f"SpaceTrack query failed: {exc}")

    return {
        "request_class": rc_key,
        "count": len(records),
        "records": records,
    }
