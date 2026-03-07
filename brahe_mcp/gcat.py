"""GCAT (General Catalog of Artificial Space Objects) MCP tools.

Wraps Jonathan McDowell's General Catalog, providing SATCAT (~60k orbital
objects) and PSATCAT (payload mission) queries via brahe's datasets API.
"""

import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response


def _serialize_satcat_record(record) -> dict:
    """Extract key fields from a GCATSatcatRecord into a plain dict."""
    return {
        "jcat": record.jcat,
        "satcat": record.satcat,
        "piece": record.piece,
        "object_type": record.object_type,
        "name": record.name,
        "pl_name": record.pl_name,
        "ldate": record.ldate,
        "ddate": record.ddate,
        "status": record.status,
        "dest": record.dest,
        "owner": record.owner,
        "state": record.state,
        "manufacturer": record.manufacturer,
        "mass": record.mass,
        "dry_mass": record.dry_mass,
        "perigee": record.perigee,
        "apogee": record.apogee,
        "inc": record.inc,
        "op_orbit": record.op_orbit,
        "alt_names": record.alt_names,
    }


def _serialize_psatcat_record(record) -> dict:
    """Extract all fields from a GCATPsatcatRecord into a plain dict."""
    return {
        "jcat": record.jcat,
        "name": record.name,
        "piece": record.piece,
        "ldate": record.ldate,
        "category": record.category,
        "class_": record.class_,
        "program": record.program,
        "result": record.result,
        "comment": record.comment,
        "tdate": record.tdate,
        "top": record.top,
        "tlast": record.tlast,
        "att": record.att,
        "control": record.control,
        "discipline": record.discipline,
        "mvr": record.mvr,
        "plane": record.plane,
        "tf": record.tf,
        "disp_epoch": record.disp_epoch,
        "disp_peri": record.disp_peri,
        "disp_apo": record.disp_apo,
        "disp_inc": record.disp_inc,
        "un_reg": record.un_reg,
        "un_state": record.un_state,
        "un_period": record.un_period,
        "un_perigee": record.un_perigee,
        "un_apogee": record.un_apogee,
        "un_inc": record.un_inc,
    }


# ---------------------------------------------------------------------------
# Field descriptions and enumeration reference data
# ---------------------------------------------------------------------------

_SATCAT_FIELD_DESCRIPTIONS = {
    "jcat": "JCAT identifier — McDowell's unique catalog ID (e.g. 'S00001')",
    "satcat": "NORAD/USSPACECOM SATCAT number as string (e.g. '00001', '25544')",
    "piece": "International designation / piece ID (e.g. '1957 ALP 1', '1998-067A')",
    "object_type": (
        "Compound object type code. See satcat_object_types for grouped "
        "primary types (P=Payload, R1-R4=Rocket stages, D=Debris, C=Craft) "
        "and common suffixes. Examples: 'P', 'R2', 'PP-H', 'D  J', 'C  A'"
    ),
    "name": "Object name as cataloged",
    "pl_name": "Payload/launch vehicle name (may differ from object name)",
    "ldate": "Launch date string (e.g. '1957 Oct 4')",
    "ddate": "Decay/reentry date string, or None if still in orbit",
    "status": (
        "Orbital status code. See satcat_status_codes for grouped values: "
        "in_orbit (O, C, OX), no_longer_in_orbit (R, D, L), "
        "attached (AO, AR, DK), deep_space (DSA, DSO), etc."
    ),
    "dest": "Destination body or orbit (e.g. 'Moon', 'Mars'), None for Earth orbit",
    "owner": "Owner/operator organization code (e.g. 'NASA', 'OKB1', 'SpX')",
    "state": "Responsible state/country code (e.g. 'US', 'SU', 'CN', 'IN', 'JP')",
    "manufacturer": "Manufacturer/builder code",
    "mass": "Total mass in kg at launch (None if unknown)",
    "dry_mass": "Dry mass in kg (None if unknown)",
    "perigee": "Perigee altitude in km (None if unknown or not applicable)",
    "apogee": "Apogee altitude in km (None if unknown or not applicable)",
    "inc": "Orbital inclination in degrees (None if unknown)",
    "op_orbit": (
        "Operational orbit type code. See satcat_op_orbit_codes for grouped values: "
        "low_earth_orbit (LEO/*, LLEO/*), medium_earth_orbit (MEO), "
        "geosynchronous (GEO/*, GTO), highly_elliptical (HEO, EEO, VHEO), "
        "beyond_earth (SO, CLO, DSO, TA)"
    ),
    "alt_names": "Alternative names list (None if no alternates)",
}

_SATCAT_STATUS_CODES = {
    "in_orbit": {
        "O": "Operational",
        "C": "Non-functional",
        "OX": "Status unknown",
    },
    "no_longer_in_orbit": {
        "R": "Reentered (confirmed)",
        "R?": "Reentered (uncertain)",
        "D": "Deorbited (controlled reentry)",
        "L": "Landed on surface",
    },
    "escaped": {
        "E": "Escaped Earth orbit",
    },
    "attached": {
        "AO": "Attached to operational object in orbit",
        "AR": "Attached to object, later reentered",
        "DK": "Docked or berthed to another object",
    },
    "deep_space": {
        "DSA": "Arrived at destination",
        "DSO": "In orbit at destination",
    },
    "other": {
        "N": "No orbit achieved / not yet launched",
        "ERR": "Catalog error or reassigned entry",
    },
}

_SATCAT_OBJECT_TYPES = {
    "payloads": {
        "_description": "Satellites, spacecraft, and probes. Primary prefix 'P'.",
        "P": "Payload (basic)",
        "PP": "Piloted payload",
        "PH": "Human-tended payload",
        "PX": "Experimental payload",
    },
    "rocket_bodies": {
        "_description": "Spent launch vehicle stages. Primary prefix 'R'.",
        "R1": "1st stage",
        "R2": "2nd stage",
        "R3": "3rd stage",
        "R4": "4th stage",
    },
    "debris": {
        "_description": "Fragments, jettisoned hardware, and mission-related debris. Primary prefix 'D'.",
        "D": "Debris (general)",
    },
    "craft": {
        "_description": "Sub-satellites, attached modules, and deployed sub-components. Primary prefix 'C'.",
        "C": "Craft / sub-satellite",
    },
    "other": {
        "_description": "Error entries or unclassified objects.",
        "Z": "Error or unknown classification",
    },
    "common_suffixes": {
        "_description": (
            "Object type codes are compound strings where suffixes modify the primary type. "
            "Examples: 'P  H'=payload human-related, 'PP-H'=piloted human-tended, "
            "'D  J'=jettisoned debris, 'C  A'=attached craft, 'R2  D'=2nd stage derelict."
        ),
        "H": "Human-related or human-tended",
        "A": "Attached to parent object",
        "D": "Derelict / decommissioned",
        "X": "Experimental",
        "V": "Crewed vehicle",
        "N": "Navigation-related",
    },
}

_SATCAT_OP_ORBIT_CODES = {
    "low_earth_orbit": {
        "_description": "LEO: ~200-2000km altitude. LLEO: below ~300km (very short-lived).",
        "LEO/E": "Equatorial",
        "LEO/I": "Inclined",
        "LEO/P": "Polar",
        "LEO/R": "Retrograde",
        "LEO/S": "Sun-synchronous",
        "LLEO/E": "Very low LEO, equatorial",
        "LLEO/I": "Very low LEO, inclined",
        "LLEO/P": "Very low LEO, polar",
        "LLEO/R": "Very low LEO, retrograde",
        "LLEO/S": "Very low LEO, sun-synchronous",
    },
    "medium_earth_orbit": {
        "_description": "MEO: ~2000-35786km altitude. Includes navigation constellations (GPS, Galileo).",
        "MEO": "Medium Earth orbit",
    },
    "geosynchronous": {
        "_description": "GEO/GTO: ~35786km altitude (24h period). Includes geostationary and transfer orbits.",
        "GEO/S": "Geostationary (near-zero inclination)",
        "GEO/I": "Geosynchronous, inclined",
        "GEO/D": "GEO drift orbit",
        "GEO/ID": "GEO inclined drift",
        "GEO/NS": "Near-synchronous",
        "GEO/T": "Transfer to GEO",
        "GTO": "Geostationary transfer orbit",
    },
    "highly_elliptical": {
        "_description": "HEO/EEO/VHEO: orbits with high eccentricity and apogees well above GEO.",
        "HEO": "Highly elliptical orbit",
        "HEO/M": "Molniya orbit (~12h period, 63.4 deg inc)",
        "EEO": "Extended elliptical orbit",
        "VHEO": "Very high elliptical orbit",
    },
    "beyond_earth": {
        "_description": "Orbits beyond Earth's gravitational sphere of influence.",
        "SO": "Sub-orbital",
        "CLO": "Cislunar orbit",
        "DSO": "Deep space orbit",
        "TA": "Heliocentric / trans-asteroid",
    },
}

_PSATCAT_FIELD_DESCRIPTIONS = {
    "jcat": "JCAT identifier — McDowell's unique catalog ID (e.g. 'S00049')",
    "name": "Payload/mission name",
    "piece": "International designation / piece ID",
    "ldate": "Launch date string (e.g. '1960 Aug 12')",
    "category": (
        "Mission category code. See psatcat_category_codes for grouped values: "
        "communications_and_navigation (COM, NAV), earth_observation (IMG, MET, EOSCI), "
        "science_and_astronomy (SCI, AST, PLAN), defense_and_intelligence (SIG, EW), "
        "other (TECH, SS, RV). Compound codes like 'COM/MET' also exist."
    ),
    "class_": (
        "Object class: "
        "A=Astronomy/deep space, B=Biological, C=Communications/utility, "
        "D=Defense/military"
    ),
    "program": "Mission program name (e.g. 'Echo', 'Starlink', 'GPS')",
    "result": (
        "Mission result code: "
        "S=Successful, F=Failed, U=Unknown/uncertain, N=Not yet determined"
    ),
    "comment": "Additional comment text (often None)",
    "tdate": "Date of first tracking or operation",
    "top": "Date operations began",
    "tlast": "Date of last known operation/contact",
    "att": "Attitude control type",
    "control": "Control/operations organization",
    "discipline": "Scientific discipline",
    "mvr": "Maneuver capability",
    "plane": "Orbital plane designation",
    "tf": "Transfer stage information",
    "disp_epoch": "Disposal/final orbit epoch date string",
    "disp_peri": "Disposal orbit perigee altitude in km",
    "disp_apo": "Disposal orbit apogee altitude in km",
    "disp_inc": "Disposal orbit inclination in degrees",
    "un_reg": "UN registration document reference",
    "un_state": "UN registering state code",
    "un_period": "UN-registered orbital period in minutes",
    "un_perigee": "UN-registered perigee altitude in km",
    "un_apogee": "UN-registered apogee altitude in km",
    "un_inc": "UN-registered inclination in degrees",
}

_PSATCAT_CATEGORY_CODES = {
    "communications_and_navigation": {
        "_description": "Satellites providing communications, navigation, or data relay services.",
        "COM": "Communications",
        "NAV": "Navigation (GPS, GLONASS, Galileo, etc.)",
        "NAV/COM": "Navigation with communications capability",
    },
    "earth_observation": {
        "_description": "Satellites observing Earth's surface, atmosphere, or oceans.",
        "IMG": "Optical imaging / Earth observation",
        "IMG-R": "Radar imaging (SAR)",
        "MET": "Meteorology / weather",
        "EOSCI": "Earth and ocean science",
        "GEOD": "Geodesy",
    },
    "science_and_astronomy": {
        "_description": "Space science, astronomy, and research missions.",
        "SCI": "Science / research",
        "AST": "Astronomy",
        "PLAN": "Planetary mission",
        "BIO": "Biological / life science",
        "MGRAV": "Microgravity research",
    },
    "defense_and_intelligence": {
        "_description": "Military, intelligence, and early warning missions.",
        "SIG": "Signals intelligence",
        "EW": "Early warning",
        "WEAPON": "Weapon system / ASAT test",
    },
    "other": {
        "_description": "Technology demonstrations, crewed platforms, and support missions.",
        "TECH": "Technology demonstration",
        "SS": "Space station / crewed platform",
        "CAL": "Calibration target",
        "RV": "Reentry vehicle / capsule",
        "TARG": "Target (for tracking or ASAT tests)",
    },
    "compound_codes": {
        "_description": (
            "Categories can be combined with '/' for dual-purpose missions. "
            "Examples: 'COM/MET', 'IMG/SCI', 'SCI/TECH', 'SS/BIO'."
        ),
    },
}

_PSATCAT_RESULT_CODES = {
    "S": "Successful mission",
    "F": "Failed mission",
    "U": "Unknown or uncertain outcome",
    "N": "Not yet determined",
}

_PSATCAT_CLASS_CODES = {
    "A": "Astronomy / deep space",
    "B": "Biological / life science",
    "C": "Communications / utility",
    "D": "Defense / military",
}

_SATCAT_FILTERS = {
    "name": "Substring search (case-insensitive)",
    "object_type": "Object type prefix code — see object_type_prefixes for values",
    "owner": "Owner/operator organization code (e.g. 'NASA', 'SpX')",
    "state": "State/country code (e.g. 'US', 'CN', 'JP')",
    "status": "Orbital status code — see status_codes for values",
    "perigee_min/perigee_max": "Perigee altitude range in km",
    "apogee_min/apogee_max": "Apogee altitude range in km",
    "inc_min/inc_max": "Inclination range in degrees",
}

_PSATCAT_FILTERS = {
    "name": "Substring search (case-insensitive)",
    "category": "Mission category code — see category in psatcat_field_descriptions",
    "object_class": "Object class code: A, B, C, or D — see class_codes",
    "result_code": "Mission result code: S, F, U, or N — see result_codes",
    "active_only": "Filter to active payloads only (bool)",
}


# ---------------------------------------------------------------------------
# Tool 1: list_gcat_options
# ---------------------------------------------------------------------------


@mcp.tool()
def list_gcat_options() -> dict:
    """List available GCAT catalog types, field descriptions, enumeration values, and filter options.

    Returns comprehensive discovery information for all GCAT tools including
    catalog types, detailed field descriptions with enumeration code meanings,
    and available filters. Call this first to understand what fields and codes
    mean before querying GCAT data.
    """
    logger.debug("Listing GCAT options")
    return {
        "catalog_types": {
            "satcat": "Satellite catalog (~60k records) — all tracked orbital objects with physical/orbital data",
            "psatcat": "Payload satellite catalog — payload missions with program, result, and disposal orbit data",
        },
        "lookup_methods": {
            "get_gcat_satcat": "Look up SATCAT records by jcat, satcat number, or name",
            "query_gcat_satcat": "Advanced filtered query on SATCAT records",
            "get_gcat_psatcat": "Look up PSATCAT records by jcat or name",
            "query_gcat_psatcat": "Advanced filtered query on PSATCAT records",
        },
        "satcat_field_descriptions": _SATCAT_FIELD_DESCRIPTIONS,
        "satcat_status_codes": _SATCAT_STATUS_CODES,
        "satcat_object_types": _SATCAT_OBJECT_TYPES,
        "satcat_op_orbit_codes": _SATCAT_OP_ORBIT_CODES,
        "satcat_filters": _SATCAT_FILTERS,
        "psatcat_field_descriptions": _PSATCAT_FIELD_DESCRIPTIONS,
        "psatcat_category_codes": _PSATCAT_CATEGORY_CODES,
        "psatcat_result_codes": _PSATCAT_RESULT_CODES,
        "psatcat_class_codes": _PSATCAT_CLASS_CODES,
        "psatcat_filters": _PSATCAT_FILTERS,
    }


# ---------------------------------------------------------------------------
# Tool 2: get_gcat_satcat
# ---------------------------------------------------------------------------


@mcp.tool()
def get_gcat_satcat(
    jcat: str | None = None,
    satcat_num: str | None = None,
    name: str | None = None,
    limit: int = 100,
) -> dict:
    """Look up GCAT SATCAT records by identifier.

    Provide exactly one of jcat, satcat_num, or name to query by.
    The GCAT SATCAT contains ~60k records of orbital objects from
    Jonathan McDowell's General Catalog.

    Use list_gcat_options() to see field descriptions and enumeration
    code meanings (status, object_type, op_orbit, etc.).

    Args:
        jcat: JCAT identifier (e.g. "S00001").
        satcat_num: SATCAT/NORAD catalog number as string (e.g. "00001").
        name: Name substring search (case-insensitive).
        limit: Maximum number of records to return (default 100).
    """
    identifiers = {
        "jcat": jcat,
        "satcat_num": satcat_num,
        "name": name,
    }
    provided = {k: v for k, v in identifiers.items() if v is not None}

    if len(provided) == 0:
        return error_response(
            "Provide exactly one identifier: jcat, satcat_num, or name",
        )
    if len(provided) > 1:
        return error_response(
            f"Provide exactly one identifier, got {len(provided)}: {list(provided.keys())}",
        )

    method, identifier = next(iter(provided.items()))
    logger.info("GCAT SATCAT lookup: {}={}", method, identifier)

    try:
        container = brahe.datasets.gcat.get_satcat()

        if method == "jcat":
            record = container.get_by_jcat(identifier)
            records = [_serialize_satcat_record(record)]
        elif method == "satcat_num":
            record = container.get_by_satcat(identifier)
            records = [_serialize_satcat_record(record)]
        else:  # name
            results = container.search_by_name(identifier)
            records = [_serialize_satcat_record(r) for r in results.records()]
            if limit > 0:
                records = records[:limit]
    except Exception as exc:
        logger.error("GCAT SATCAT error: {}", exc)
        return error_response(f"GCAT SATCAT lookup failed: {exc}")

    return {
        "method": method,
        "identifier": identifier,
        "count": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Tool 3: query_gcat_satcat
# ---------------------------------------------------------------------------


@mcp.tool()
def query_gcat_satcat(
    name: str | None = None,
    object_type: str | None = None,
    owner: str | None = None,
    state: str | None = None,
    status: str | None = None,
    perigee_min: float | None = None,
    perigee_max: float | None = None,
    apogee_min: float | None = None,
    apogee_max: float | None = None,
    inc_min: float | None = None,
    inc_max: float | None = None,
    limit: int = 100,
) -> dict:
    """Query GCAT SATCAT with advanced chainable filters.

    At least one filter parameter must be provided. Filters are applied
    sequentially to narrow results. Use list_gcat_options() to see
    valid enumeration codes for object_type, status, etc.

    Args:
        name: Name substring search (case-insensitive).
        object_type: Object type prefix — P=Payload, R1-R4=Rocket stages,
            D=Debris, C=Craft. Use list_gcat_options() for full codes.
        owner: Owner/operator organization code (e.g. "NASA", "SpX").
        state: State/country code (e.g. "US", "CN", "JP", "SU").
        status: Orbital status code — O=In orbit, R=Reentered, D=Deorbited,
            E=Escaped, C=Non-functional. Use list_gcat_options() for full codes.
        perigee_min: Minimum perigee altitude in km.
        perigee_max: Maximum perigee altitude in km.
        apogee_min: Minimum apogee altitude in km.
        apogee_max: Maximum apogee altitude in km.
        inc_min: Minimum inclination in degrees.
        inc_max: Maximum inclination in degrees.
        limit: Maximum number of records to return (default 100).
    """
    filters_applied = []

    if all(v is None for v in [
        name, object_type, owner, state, status,
        perigee_min, perigee_max, apogee_min, apogee_max,
        inc_min, inc_max,
    ]):
        return error_response(
            "At least one filter parameter is required",
            available_filters=_SATCAT_FILTERS,
        )

    logger.info("GCAT SATCAT query with filters")

    try:
        container = brahe.datasets.gcat.get_satcat()

        if name is not None:
            container = container.search_by_name(name)
            filters_applied.append(f"name={name!r}")
        if object_type is not None:
            container = container.filter_by_type(object_type)
            filters_applied.append(f"object_type={object_type!r}")
        if owner is not None:
            container = container.filter_by_owner(owner)
            filters_applied.append(f"owner={owner!r}")
        if state is not None:
            container = container.filter_by_state(state)
            filters_applied.append(f"state={state!r}")
        if status is not None:
            container = container.filter_by_status(status)
            filters_applied.append(f"status={status!r}")
        if perigee_min is not None or perigee_max is not None:
            p_min = perigee_min if perigee_min is not None else 0.0
            p_max = perigee_max if perigee_max is not None else 1e9
            container = container.filter_by_perigee_range(p_min, p_max)
            filters_applied.append(f"perigee=[{p_min}, {p_max}]")
        if apogee_min is not None or apogee_max is not None:
            a_min = apogee_min if apogee_min is not None else 0.0
            a_max = apogee_max if apogee_max is not None else 1e9
            container = container.filter_by_apogee_range(a_min, a_max)
            filters_applied.append(f"apogee=[{a_min}, {a_max}]")
        if inc_min is not None or inc_max is not None:
            i_min = inc_min if inc_min is not None else 0.0
            i_max = inc_max if inc_max is not None else 180.0
            container = container.filter_by_inc_range(i_min, i_max)
            filters_applied.append(f"inc=[{i_min}, {i_max}]")

        records = [_serialize_satcat_record(r) for r in container.records()]
        if limit > 0:
            records = records[:limit]

    except Exception as exc:
        logger.error("GCAT SATCAT query error: {}", exc)
        return error_response(f"GCAT SATCAT query failed: {exc}")

    return {
        "filters": filters_applied,
        "count": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Tool 4: get_gcat_psatcat
# ---------------------------------------------------------------------------


@mcp.tool()
def get_gcat_psatcat(
    jcat: str | None = None,
    name: str | None = None,
    limit: int = 100,
) -> dict:
    """Look up GCAT PSATCAT (payload satellite catalog) records by identifier.

    Provide exactly one of jcat or name to query by. The PSATCAT contains
    mission-level data including program, result, disposal orbit, and
    UN registration information.

    Use list_gcat_options() to see field descriptions and enumeration
    code meanings (category, class_, result, etc.).

    Args:
        jcat: JCAT identifier (e.g. "S00049").
        name: Name substring search (case-insensitive).
        limit: Maximum number of records to return (default 100).
    """
    identifiers = {"jcat": jcat, "name": name}
    provided = {k: v for k, v in identifiers.items() if v is not None}

    if len(provided) == 0:
        return error_response(
            "Provide exactly one identifier: jcat or name",
        )
    if len(provided) > 1:
        return error_response(
            f"Provide exactly one identifier, got {len(provided)}: {list(provided.keys())}",
        )

    method, identifier = next(iter(provided.items()))
    logger.info("GCAT PSATCAT lookup: {}={}", method, identifier)

    try:
        container = brahe.datasets.gcat.get_psatcat()

        if method == "jcat":
            record = container.get_by_jcat(identifier)
            records = [_serialize_psatcat_record(record)]
        else:  # name
            results = container.search_by_name(identifier)
            records = [_serialize_psatcat_record(r) for r in results.records()]
            if limit > 0:
                records = records[:limit]
    except Exception as exc:
        logger.error("GCAT PSATCAT error: {}", exc)
        return error_response(f"GCAT PSATCAT lookup failed: {exc}")

    return {
        "method": method,
        "identifier": identifier,
        "count": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Tool 5: query_gcat_psatcat
# ---------------------------------------------------------------------------


@mcp.tool()
def query_gcat_psatcat(
    name: str | None = None,
    category: str | None = None,
    object_class: str | None = None,
    result_code: str | None = None,
    active_only: bool = False,
    limit: int = 100,
) -> dict:
    """Query GCAT PSATCAT with advanced chainable filters.

    At least one filter parameter must be provided (active_only=True counts).
    Filters are applied sequentially to narrow results. Use list_gcat_options()
    for full enumeration code descriptions.

    Args:
        name: Name substring search (case-insensitive).
        category: Mission category code — COM=Communications, IMG=Imaging,
            NAV=Navigation, SCI=Science, SIG=Signals intelligence, MET=Meteorology,
            TECH=Technology demo. Compound codes like "COM/MET" also exist.
            Use list_gcat_options() for full list.
        object_class: Object class code — A=Astronomy/deep space, B=Biological,
            C=Communications/utility, D=Defense/military.
        result_code: Mission result code — S=Successful, F=Failed,
            U=Unknown, N=Not yet determined.
        active_only: Filter to active payloads only.
        limit: Maximum number of records to return (default 100).
    """
    if not active_only and all(v is None for v in [name, category, object_class, result_code]):
        return error_response(
            "At least one filter parameter is required",
            available_filters=_PSATCAT_FILTERS,
        )

    filters_applied = []
    logger.info("GCAT PSATCAT query with filters")

    try:
        container = brahe.datasets.gcat.get_psatcat()

        if name is not None:
            container = container.search_by_name(name)
            filters_applied.append(f"name={name!r}")
        if category is not None:
            container = container.filter_by_category(category)
            filters_applied.append(f"category={category!r}")
        if object_class is not None:
            container = container.filter_by_class(object_class)
            filters_applied.append(f"object_class={object_class!r}")
        if result_code is not None:
            container = container.filter_by_result(result_code)
            filters_applied.append(f"result_code={result_code!r}")
        if active_only:
            container = container.filter_active()
            filters_applied.append("active_only=True")

        records = [_serialize_psatcat_record(r) for r in container.records()]
        if limit > 0:
            records = records[:limit]

    except Exception as exc:
        logger.error("GCAT PSATCAT query error: {}", exc)
        return error_response(f"GCAT PSATCAT query failed: {exc}")

    return {
        "filters": filters_applied,
        "count": len(records),
        "records": records,
    }
