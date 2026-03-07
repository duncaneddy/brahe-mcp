"""Access computation MCP tools using brahe location_accesses."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch
from brahe_mcp.propagation import (
    _sgp4_from_gp,
    _eci_state_from_gp,
    _build_force_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CONSTRAINT_TYPES = {
    "elevation", "elevation_mask", "off_nadir", "local_time",
    "local_time_hours", "look_direction", "asc_dsc",
}
VALID_SATELLITE_SOURCES = {"gp_record", "tle", "state"}
VALID_PROPAGATOR_TYPES = {"sgp4", "keplerian", "numerical"}

LOOK_DIRECTION_MAP = {
    "left": brahe.LookDirection.LEFT,
    "right": brahe.LookDirection.RIGHT,
    "either": brahe.LookDirection.EITHER,
}

ASC_DSC_MAP = {
    "ascending": brahe.AscDsc.ASCENDING,
    "descending": brahe.AscDsc.DESCENDING,
    "either": brahe.AscDsc.EITHER,
}

VALID_PROPERTY_TYPES = {"range", "range_rate", "doppler"}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_constraint(spec: dict):
    """Parse a single constraint dict into a brahe constraint object.

    Args:
        spec: Constraint specification dict with "type" key and parameters.

    Returns:
        A brahe constraint object.

    Raises:
        ValueError: If the constraint type or parameters are invalid.
    """
    ctype = spec.get("type", "").lower()
    if ctype not in VALID_CONSTRAINT_TYPES:
        raise ValueError(
            f"Unknown constraint type: {spec.get('type')!r}. "
            f"Valid types: {sorted(VALID_CONSTRAINT_TYPES)}"
        )

    if ctype == "elevation":
        min_deg = spec.get("min_deg", 5.0)
        max_deg = spec.get("max_deg")
        if max_deg is not None:
            return brahe.ElevationConstraint(
                min_elevation_deg=min_deg, max_elevation_deg=max_deg
            )
        return brahe.ElevationConstraint(min_deg)

    if ctype == "elevation_mask":
        mask = spec.get("mask")
        if not mask:
            raise ValueError("elevation_mask constraint requires 'mask' parameter")
        return brahe.ElevationMaskConstraint(
            [(float(az), float(el)) for az, el in mask]
        )

    if ctype == "off_nadir":
        max_deg = spec.get("max_deg")
        if max_deg is None:
            raise ValueError("off_nadir constraint requires 'max_deg' parameter")
        return brahe.OffNadirConstraint(max_deg)

    if ctype == "local_time":
        windows = spec.get("windows")
        if not windows:
            raise ValueError("local_time constraint requires 'windows' parameter (HHMM pairs)")
        return brahe.LocalTimeConstraint(
            [(int(start), int(end)) for start, end in windows]
        )

    if ctype == "local_time_hours":
        windows = spec.get("windows")
        if not windows:
            raise ValueError("local_time_hours constraint requires 'windows' parameter (decimal hour pairs)")
        # Convert decimal hours to HHMM integer format
        hhmm_windows = []
        for start_h, end_h in windows:
            start_hhmm = int(start_h) * 100 + int((start_h % 1) * 60)
            end_hhmm = int(end_h) * 100 + int((end_h % 1) * 60)
            hhmm_windows.append((start_hhmm, end_hhmm))
        return brahe.LocalTimeConstraint(hhmm_windows)

    if ctype == "look_direction":
        allowed = spec.get("allowed", "either").lower()
        if allowed not in LOOK_DIRECTION_MAP:
            raise ValueError(
                f"Invalid look direction: {spec.get('allowed')!r}. "
                f"Valid: {sorted(LOOK_DIRECTION_MAP.keys())}"
            )
        return brahe.LookDirectionConstraint(LOOK_DIRECTION_MAP[allowed])

    if ctype == "asc_dsc":
        allowed = spec.get("allowed", "either").lower()
        if allowed not in ASC_DSC_MAP:
            raise ValueError(
                f"Invalid asc/dsc value: {spec.get('allowed')!r}. "
                f"Valid: {sorted(ASC_DSC_MAP.keys())}"
            )
        return brahe.AscDscConstraint(ASC_DSC_MAP[allowed])

    raise ValueError(f"Unhandled constraint type: {ctype}")


def _build_constraints(specs: list[dict] | None, logic: str, min_elevation_deg: float | None):
    """Combine constraint specs into a single brahe constraint.

    Args:
        specs: List of constraint spec dicts, or None.
        logic: "all" (AND) or "any" (OR).
        min_elevation_deg: Convenience shortcut for elevation constraint.

    Returns:
        A brahe constraint object.

    Raises:
        ValueError: If inputs are invalid.
    """
    constraints = []

    if specs:
        for spec in specs:
            constraints.append(_build_constraint(spec))

    if min_elevation_deg is not None:
        constraints.append(brahe.ElevationConstraint(min_elevation_deg))

    # Default: 5 degree elevation if nothing specified
    if not constraints:
        constraints.append(brahe.ElevationConstraint(5.0))

    if len(constraints) == 1:
        return constraints[0]

    logic_lower = logic.lower()
    if logic_lower == "all":
        return brahe.ConstraintAll(constraints)
    elif logic_lower == "any":
        return brahe.ConstraintAny(constraints)
    else:
        raise ValueError(f"Invalid constraint_logic: {logic!r}. Must be 'all' or 'any'.")


def _build_location(loc_dict: dict) -> brahe.PointLocation:
    """Convert a location dict to a brahe PointLocation.

    Accepts both groundstation-style dicts (from _serialize_station) and
    user-provided dicts with lon/lat/alt keys.

    Args:
        loc_dict: Dict with lon, lat, and optional alt/altitude_m/name keys.

    Returns:
        brahe.PointLocation instance.

    Raises:
        ValueError: If required fields are missing.
    """
    lon = loc_dict.get("lon")
    lat = loc_dict.get("lat")
    if lon is None or lat is None:
        raise ValueError("Location dict must have 'lon' and 'lat' keys.")

    alt = loc_dict.get("altitude_m", loc_dict.get("alt", 0.0))

    location = brahe.PointLocation(float(lon), float(lat), float(alt))

    name = loc_dict.get("name")
    if name:
        location = location.with_name(str(name))

    return location


def _build_propagator(sat_dict: dict):
    """Build a propagator from a satellite specification dict.

    Args:
        sat_dict: Satellite spec with "source" key and source-specific parameters.

    Returns:
        A brahe propagator instance (SGP4, Keplerian, or NumericalOrbit).

    Raises:
        ValueError: If the spec is invalid.
    """
    source = sat_dict.get("source", "").lower()
    if source not in VALID_SATELLITE_SOURCES:
        raise ValueError(
            f"Unknown satellite source: {sat_dict.get('source')!r}. "
            f"Valid: {sorted(VALID_SATELLITE_SOURCES)}"
        )

    if source == "tle":
        line1 = sat_dict.get("tle_line1")
        line2 = sat_dict.get("tle_line2")
        if not line1 or not line2:
            raise ValueError("TLE source requires 'tle_line1' and 'tle_line2'.")
        return brahe.SGPPropagator.from_tle(line1, line2, step_size=60.0)

    if source == "gp_record":
        gp = sat_dict.get("gp_record")
        if not gp:
            raise ValueError("gp_record source requires 'gp_record' dict.")
        prop_type = sat_dict.get("propagator_type", "sgp4").lower()
        if prop_type == "sgp4":
            return _sgp4_from_gp(gp)
        elif prop_type == "keplerian":
            state_eci, epoch = _eci_state_from_gp(gp)
            return brahe.KeplerianPropagator.from_eci(epoch, state_eci, step_size=60.0)
        elif prop_type == "numerical":
            state_eci, epoch = _eci_state_from_gp(gp)
            force_model = sat_dict.get("force_model", "default")
            spacecraft_params = sat_dict.get("spacecraft_params")
            force_config = _build_force_config(force_model)
            params = (
                np.array(spacecraft_params, dtype=float)
                if spacecraft_params is not None
                else None
            )
            if force_config.requires_params() and params is None:
                raise ValueError(
                    f"Force model '{force_model}' requires spacecraft_params: "
                    "[mass_kg, drag_area_m2, Cd, srp_area_m2, Cr]."
                )
            prop_config = brahe.NumericalPropagationConfig.default()
            return brahe.NumericalOrbitPropagator(
                epoch, state_eci, prop_config, force_config, params
            )
        else:
            raise ValueError(
                f"Unknown propagator_type: {prop_type!r}. "
                f"Valid: {sorted(VALID_PROPAGATOR_TYPES)}"
            )

    if source == "state":
        epoch_str = sat_dict.get("epoch")
        state_eci = sat_dict.get("state_eci")
        if not epoch_str or state_eci is None:
            raise ValueError("State source requires 'epoch' and 'state_eci'.")
        if len(state_eci) != 6:
            raise ValueError(
                f"state_eci must have 6 elements [x,y,z,vx,vy,vz], got {len(state_eci)}"
            )
        prop_type = sat_dict.get("propagator_type", "keplerian").lower()
        epoch = parse_epoch(epoch_str)
        cart = np.array(state_eci, dtype=float)
        if prop_type == "keplerian":
            return brahe.KeplerianPropagator.from_eci(epoch, cart, step_size=60.0)
        elif prop_type == "sgp4":
            raise ValueError(
                "SGP4 propagation from state is not supported. "
                "Use 'keplerian' or provide TLE/GP data for SGP4."
            )
        elif prop_type == "numerical":
            force_model = sat_dict.get("force_model", "default")
            spacecraft_params = sat_dict.get("spacecraft_params")
            force_config = _build_force_config(force_model)
            params = (
                np.array(spacecraft_params, dtype=float)
                if spacecraft_params is not None
                else None
            )
            if force_config.requires_params() and params is None:
                raise ValueError(
                    f"Force model '{force_model}' requires spacecraft_params: "
                    "[mass_kg, drag_area_m2, Cd, srp_area_m2, Cr]."
                )
            prop_config = brahe.NumericalPropagationConfig.default()
            return brahe.NumericalOrbitPropagator(
                epoch, cart, prop_config, force_config, params
            )
        else:
            raise ValueError(
                f"Unknown propagator_type: {prop_type!r}. "
                f"Valid for state source: ['keplerian', 'numerical']"
            )

    raise ValueError(f"Unhandled satellite source: {source}")


def _build_search_config(config_dict: dict | None) -> brahe.AccessSearchConfig | None:
    """Build an AccessSearchConfig from an optional config dict.

    Args:
        config_dict: Optional dict with config parameters.

    Returns:
        AccessSearchConfig instance or None.
    """
    if not config_dict:
        return None

    kwargs = {}
    if "initial_time_step" in config_dict:
        kwargs["initial_time_step"] = float(config_dict["initial_time_step"])
    if "adaptive_step" in config_dict:
        kwargs["adaptive_step"] = bool(config_dict["adaptive_step"])
    if "time_tolerance" in config_dict:
        kwargs["time_tolerance"] = float(config_dict["time_tolerance"])
    if "subdivisions" in config_dict:
        kwargs["subdivisions"] = int(config_dict["subdivisions"])
    if "num_threads" in config_dict:
        kwargs["num_threads"] = int(config_dict["num_threads"])

    return brahe.AccessSearchConfig(**kwargs)


def _build_property_computers(specs: list[dict] | None) -> list | None:
    """Build property computer instances from spec dicts.

    Args:
        specs: List of property computer specs, or None.

    Returns:
        List of brahe property computer instances, or None.

    Raises:
        ValueError: If a spec type is unknown.
    """
    if not specs:
        return None

    computers = []
    for spec in specs:
        ptype = spec.get("type", "").lower()
        sampling = brahe.SamplingConfig.midpoint()

        if ptype == "range":
            computers.append(brahe.RangeComputer(sampling))
        elif ptype == "range_rate":
            computers.append(brahe.RangeRateComputer(sampling))
        elif ptype == "doppler":
            uplink = spec.get("uplink_frequency", 2.2e9)
            downlink = spec.get("downlink_frequency", 2.2e9)
            computers.append(brahe.DopplerComputer(uplink, downlink, sampling))
        else:
            raise ValueError(
                f"Unknown property computer type: {spec.get('type')!r}. "
                f"Valid: {sorted(VALID_PROPERTY_TYPES)}"
            )

    return computers if computers else None


def _serialize_access_window(window) -> dict:
    """Serialize an AccessWindow to a plain dict.

    Args:
        window: A brahe AccessWindow instance.

    Returns:
        Dict with all access window fields.
    """
    result = {
        "window_open": str(window.window_open),
        "window_close": str(window.window_close),
        "duration_seconds": window.duration,
        "elevation_max_deg": window.elevation_max,
        "elevation_open_deg": window.elevation_open,
        "elevation_close_deg": window.elevation_close,
        "azimuth_open_deg": window.azimuth_open,
        "azimuth_close_deg": window.azimuth_close,
        "off_nadir_min_deg": window.off_nadir_min,
        "off_nadir_max_deg": window.off_nadir_max,
        "look_direction": str(window.look_direction),
        "asc_dsc": str(window.asc_dsc),
        "location_name": window.location_name,
        "satellite_name": window.satellite_name,
    }

    # Include additional properties from property computers
    if window.properties and window.properties.additional:
        additional = dict(window.properties.additional)
        if additional:
            result["additional_properties"] = additional

    return result


def _describe_constraint(spec: dict) -> str:
    """Create a human-readable description of a constraint spec."""
    ctype = spec.get("type", "unknown")
    if ctype == "elevation":
        parts = [f"min={spec.get('min_deg', 5.0)}deg"]
        if spec.get("max_deg"):
            parts.append(f"max={spec['max_deg']}deg")
        return f"elevation({', '.join(parts)})"
    if ctype == "off_nadir":
        return f"off_nadir(max={spec.get('max_deg')}deg)"
    if ctype in ("local_time", "local_time_hours"):
        return f"{ctype}(windows={spec.get('windows')})"
    if ctype == "look_direction":
        return f"look_direction({spec.get('allowed', 'either')})"
    if ctype == "asc_dsc":
        return f"asc_dsc({spec.get('allowed', 'either')})"
    if ctype == "elevation_mask":
        return f"elevation_mask({len(spec.get('mask', []))} points)"
    return ctype


def _get_propagator_type_name(propagator) -> str:
    """Get a string name for the propagator type."""
    cls_name = type(propagator).__name__
    if "SGP" in cls_name:
        return "sgp4"
    if "Keplerian" in cls_name:
        return "keplerian"
    if "Numerical" in cls_name:
        return "numerical"
    return cls_name.lower()


def _get_satellite_name(sat_dict: dict) -> str | None:
    """Extract satellite name from a satellite spec dict."""
    if sat_dict.get("source") == "gp_record":
        gp = sat_dict.get("gp_record", {})
        return gp.get("object_name")
    return sat_dict.get("name")


# ---------------------------------------------------------------------------
# Tool 1: list_access_options
# ---------------------------------------------------------------------------


@mcp.tool()
def list_access_options() -> dict:
    """List available access computation options, constraint types, and input formats.

    Use this to discover how to call compute_access() and compute_access_from_gp().
    """
    logger.debug("Listing access options")
    return {
        "constraint_types": {
            "elevation": {
                "description": "Minimum (and optional maximum) elevation angle in degrees",
                "params": {"min_deg": "float (default 5.0)", "max_deg": "float (optional)"},
            },
            "elevation_mask": {
                "description": "Azimuth-dependent minimum elevation mask",
                "params": {"mask": "list of [azimuth_deg, min_elevation_deg] pairs"},
            },
            "off_nadir": {
                "description": "Maximum off-nadir angle constraint",
                "params": {"max_deg": "float"},
            },
            "local_time": {
                "description": "Local solar time window constraint (HHMM integer pairs)",
                "params": {"windows": "list of [start_HHMM, end_HHMM] pairs"},
            },
            "local_time_hours": {
                "description": "Local solar time window constraint (decimal hours)",
                "params": {"windows": "list of [start_hours, end_hours] pairs"},
            },
            "look_direction": {
                "description": "Satellite look direction constraint",
                "params": {"allowed": "'left', 'right', or 'either'"},
            },
            "asc_dsc": {
                "description": "Ascending/descending pass constraint",
                "params": {"allowed": "'ascending', 'descending', or 'either'"},
            },
        },
        "constraint_logic": {
            "all": "All constraints must be satisfied (AND logic, default)",
            "any": "Any constraint being satisfied is sufficient (OR logic)",
        },
        "property_computers": {
            "range": "Compute slant range in meters at pass midpoint",
            "range_rate": "Compute range rate in m/s at pass midpoint",
            "doppler": "Compute Doppler shift (requires uplink_frequency and downlink_frequency in Hz)",
        },
        "location_format": {
            "lon": "Longitude in degrees (required)",
            "lat": "Latitude in degrees (required)",
            "altitude_m": "Altitude in meters (default 0.0, also accepts 'alt')",
            "name": "Location name (optional, shown in results)",
        },
        "satellite_sources": {
            "tle": {"fields": ["tle_line1", "tle_line2"], "propagator": "SGP4"},
            "gp_record": {
                "fields": ["gp_record", "propagator_type"],
                "propagator_types": ["sgp4 (default)", "keplerian", "numerical"],
                "note": "GP record dict from celestrak/spacetrack tools. Numerical requires force_model and possibly spacecraft_params.",
            },
            "state": {
                "fields": ["epoch", "state_eci", "propagator_type"],
                "propagator_types": ["keplerian", "numerical"],
                "note": "ECI state [x,y,z,vx,vy,vz] in meters and m/s. Numerical requires force_model and possibly spacecraft_params.",
            },
        },
        "config_options": {
            "initial_time_step": "float - Initial search step in seconds (default ~60s)",
            "adaptive_step": "bool - Use adaptive time stepping",
            "time_tolerance": "float - Time tolerance in seconds for window boundary refinement",
            "subdivisions": "int - Number of subdivisions for boundary search",
        },
        "defaults": {
            "constraint": "ElevationConstraint(min=5.0deg) when no constraints specified",
            "constraint_logic": "all",
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: compute_access
# ---------------------------------------------------------------------------


@mcp.tool()
def compute_access(
    location: dict,
    satellite: dict,
    search_start: str,
    search_end: str,
    constraints: list[dict] | None = None,
    constraint_logic: str = "all",
    min_elevation_deg: float | None = None,
    property_computers: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    """Compute access windows (visibility periods) between a satellite and a ground location.

    Finds time intervals when a satellite is visible from a location, subject to
    geometric constraints (elevation, off-nadir, look direction, etc.).

    Use list_access_options() to see constraint types, satellite input formats, and
    configuration options.

    Args:
        location: Ground location dict with lon, lat, and optional altitude_m/name.
        satellite: Satellite spec dict. Must include "source" key ("tle", "gp_record", or "state").
        search_start: Start of search window (ISO epoch string).
        search_end: End of search window (ISO epoch string).
        constraints: List of constraint spec dicts. Each needs a "type" key.
        constraint_logic: How to combine constraints: "all" (AND) or "any" (OR).
        min_elevation_deg: Convenience shortcut to add an elevation constraint.
        property_computers: List of property computer specs (e.g. [{"type": "range"}]).
        config: Optional AccessSearchConfig overrides dict.
    """
    # Build location
    try:
        loc = _build_location(location)
    except ValueError as e:
        return error_response(str(e))

    # Build propagator
    try:
        propagator = _build_propagator(satellite)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Failed to create propagator: {e}")

    # Parse search window
    try:
        start = parse_epoch(search_start)
        end = parse_epoch(search_end)
    except ValueError as e:
        return error_response(f"Invalid epoch: {e}")

    if float(end - start) <= 0:
        return error_response("search_end must be after search_start.")

    # Build constraints
    try:
        constraint = _build_constraints(constraints, constraint_logic, min_elevation_deg)
    except ValueError as e:
        return error_response(str(e))

    # Build property computers
    try:
        computers = _build_property_computers(property_computers)
    except ValueError as e:
        return error_response(str(e))

    # Build search config
    try:
        search_config = _build_search_config(config)
    except Exception as e:
        return error_response(f"Invalid config: {e}")

    # Describe applied constraints
    constraints_applied = []
    if constraints:
        for spec in constraints:
            constraints_applied.append(_describe_constraint(spec))
    if min_elevation_deg is not None:
        constraints_applied.append(f"elevation(min={min_elevation_deg}deg)")
    if not constraints_applied:
        constraints_applied.append("elevation(min=5.0deg) [default]")

    # For numerical propagators, propagate to cover the full search window.
    # The integrator must step through the trajectory before access search
    # can query states. Propagate backwards first (if needed), then forward.
    if isinstance(propagator, brahe.NumericalOrbitPropagator):
        try:
            propagator.propagate_to(start)
            propagator.propagate_to(end)
        except Exception as e:
            logger.error("Numerical propagation error: {}", e)
            return error_response(f"Numerical propagation failed: {e}")

    # Compute accesses
    try:
        windows = brahe.location_accesses(
            loc, propagator, start, end, constraint,
            property_computers=computers,
            config=search_config,
        )
    except Exception as e:
        logger.error("Access computation error: {}", e)
        return error_response(f"Access computation failed: {e}")

    # Serialize results
    serialized = [_serialize_access_window(w) for w in windows]

    sat_name = _get_satellite_name(satellite)
    prop_type = _get_propagator_type_name(propagator)

    logger.debug(
        "Access computation found {} windows for {} over {}",
        len(serialized), sat_name or "satellite", location.get("name", "location"),
    )

    return {
        "search_start": str(start),
        "search_end": str(end),
        "location": {
            "name": location.get("name"),
            "lon": location.get("lon"),
            "lat": location.get("lat"),
            "altitude_m": location.get("altitude_m", location.get("alt", 0.0)),
        },
        "satellite_name": sat_name,
        "propagator_type": prop_type,
        "constraints_applied": constraints_applied,
        "count": len(serialized),
        "windows": serialized,
    }


# ---------------------------------------------------------------------------
# Tool 3: compute_access_from_gp
# ---------------------------------------------------------------------------


@mcp.tool()
def compute_access_from_gp(
    gp_record: dict,
    location: dict,
    search_start: str,
    search_end: str,
    propagator_type: str = "sgp4",
    constraints: list[dict] | None = None,
    constraint_logic: str = "all",
    min_elevation_deg: float | None = None,
    force_model: str | None = None,
    spacecraft_params: list[float] | None = None,
    property_computers: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    """Compute access windows from a GP record (from celestrak/spacetrack tools).

    Convenience wrapper that creates a satellite spec from a GP record dict and
    delegates to compute_access(). This is the most common workflow: fetch GP data
    with get_celestrak_gp() or query_spacetrack_gp(), then compute accesses.

    Args:
        gp_record: GP record dict from get_celestrak_gp or query_spacetrack_gp tools.
        location: Ground location dict with lon, lat, and optional altitude_m/name.
        search_start: Start of search window (ISO epoch string).
        search_end: End of search window (ISO epoch string).
        propagator_type: Propagator to use: "sgp4" (default), "keplerian", or "numerical".
        constraints: List of constraint spec dicts. Each needs a "type" key.
        constraint_logic: How to combine constraints: "all" (AND) or "any" (OR).
        min_elevation_deg: Convenience shortcut to add an elevation constraint.
        force_model: Force model preset for numerical propagation (e.g. "two_body", "leo_default").
        spacecraft_params: [mass_kg, drag_area_m2, Cd, srp_area_m2, Cr] for numerical propagation.
        property_computers: List of property computer specs (e.g. [{"type": "range"}]).
        config: Optional AccessSearchConfig overrides dict.
    """
    pt = propagator_type.lower()
    if pt not in VALID_PROPAGATOR_TYPES:
        return error_response(
            f"Unknown propagator_type: {propagator_type!r}.",
            valid_types=sorted(VALID_PROPAGATOR_TYPES),
        )

    satellite = {
        "source": "gp_record",
        "gp_record": gp_record,
        "propagator_type": pt,
    }
    if pt == "numerical":
        if force_model is not None:
            satellite["force_model"] = force_model
        if spacecraft_params is not None:
            satellite["spacecraft_params"] = spacecraft_params

    return compute_access(
        location=location,
        satellite=satellite,
        search_start=search_start,
        search_end=search_end,
        constraints=constraints,
        constraint_logic=constraint_logic,
        min_elevation_deg=min_elevation_deg,
        property_computers=property_computers,
        config=config,
    )
