"""Plotting MCP tools for satellite visualization using brahe and matplotlib."""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: F401, E402
import brahe  # noqa: E402
from loguru import logger  # noqa: E402
from mcp.types import ImageContent, TextContent  # noqa: E402

from brahe_mcp.server import mcp  # noqa: E402
from brahe_mcp.utils import error_response, parse_epoch  # noqa: E402

# NOTE: imports from accesses and propagation are deferred to function bodies
# to avoid circular imports (server -> plotting -> accesses -> server).

# ---------------------------------------------------------------------------
# Style context manager
# ---------------------------------------------------------------------------

_STYLE = ["science", "no-latex"]


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


def _figure_to_image(fig) -> ImageContent:
    """Render matplotlib Figure to MCP ImageContent (base64 PNG).

    Args:
        fig: A matplotlib Figure instance.

    Returns:
        ImageContent with base64-encoded PNG data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return ImageContent(
        type="image",
        data=base64.b64encode(buf.read()).decode(),
        mimeType="image/png",
    )


def _propagator_from_gp(gp_record: dict, propagator_type: str = "sgp4"):
    """Build a propagator from a GP record dict.

    Args:
        gp_record: GP record dict from celestrak/spacetrack tools.
        propagator_type: "sgp4" or "keplerian".

    Returns:
        A brahe propagator instance.
    """
    from brahe_mcp.propagation import _sgp4_from_gp, _eci_state_from_gp

    if propagator_type == "sgp4":
        return _sgp4_from_gp(gp_record)
    state_eci, epoch = _eci_state_from_gp(gp_record)
    return brahe.KeplerianPropagator.from_eci(epoch, state_eci, step_size=60.0)


def _propagate_trajectory(propagator, start_epoch, end_epoch):
    """Propagate and return the trajectory.

    Args:
        propagator: A brahe propagator instance.
        start_epoch: Start epoch.
        end_epoch: End epoch.

    Returns:
        OrbitTrajectory from the propagator.
    """
    propagator.propagate_to(start_epoch)
    propagator.propagate_to(end_epoch)
    return propagator.trajectory


# ---------------------------------------------------------------------------
# SpaceTrack GP history fetch (reuse spacetrack client pattern)
# ---------------------------------------------------------------------------

def _fetch_gp_history(norad_cat_id: int, limit: int | None = None) -> list[dict]:
    """Fetch GP history records from SpaceTrack for a given NORAD ID.

    Args:
        norad_cat_id: NORAD catalog number.
        limit: Maximum records to return.

    Returns:
        List of serialized GP record dicts.

    Raises:
        RuntimeError: If SpaceTrack credentials are not set.
    """
    from brahe_mcp.spacetrack import _get_client
    from brahe_mcp.utils import serialize_gp_record

    client = _get_client()
    query = brahe.SpaceTrackQuery(brahe.RequestClass.GP_HISTORY)
    query = query.filter("NORAD_CAT_ID", str(norad_cat_id))
    query = query.order_by("EPOCH", brahe.SortOrder.ASC)
    if limit is not None and limit > 0:
        query = query.limit(limit)
    records = client.query_gp(query)
    return [serialize_gp_record(r) for r in records]


# ---------------------------------------------------------------------------
# GP element labels
# ---------------------------------------------------------------------------

_ELEMENT_LABELS = {
    "semimajor_axis": ("Semi-major Axis", "km"),
    "eccentricity": ("Eccentricity", ""),
    "inclination": ("Inclination", "deg"),
    "ra_of_asc_node": ("RAAN", "deg"),
    "arg_of_pericenter": ("Arg. of Perigee", "deg"),
    "mean_anomaly": ("Mean Anomaly", "deg"),
    "mean_motion": ("Mean Motion", "rev/day"),
    "bstar": ("B*", "1/R_E"),
    "period": ("Period", "min"),
    "apoapsis": ("Apoapsis", "km"),
    "periapsis": ("Periapsis", "km"),
}

_CLASSICAL_ELEMENTS = [
    "semimajor_axis", "eccentricity", "inclination",
    "ra_of_asc_node", "arg_of_pericenter", "mean_anomaly",
]


# ---------------------------------------------------------------------------
# Tool 1: list_plotting_options
# ---------------------------------------------------------------------------


@mcp.tool()
def list_plotting_options() -> dict:
    """List available plotting tools, plot types, and their parameters.

    Use this to discover which plotting tool to use and what inputs are required.
    """
    logger.debug("Listing plotting options")
    return {
        "plot_types": {
            "plot_gp_history_elements": {
                "description": "Plot orbital element trends from GP history records",
                "inputs": "gp_records (list) OR norad_cat_id (int)",
                "elements": list(_ELEMENT_LABELS.keys()),
                "default_elements": _CLASSICAL_ELEMENTS,
            },
            "plot_ground_track": {
                "description": "Plot satellite ground track on a world map",
                "inputs": "satellite dict (same format as compute_access), start/end epochs",
            },
            "plot_ground_track_from_gp": {
                "description": "Plot ground track from a GP record",
                "inputs": "gp_record dict, start/end epochs",
            },
            "plot_orbit_elements": {
                "description": "Plot Keplerian element evolution during propagation",
                "inputs": "satellite dict, start/end epochs",
            },
            "plot_orbit_elements_from_gp": {
                "description": "Plot Keplerian element evolution from a GP record",
                "inputs": "gp_record dict, start/end epochs",
            },
            "plot_access_geometry": {
                "description": "Plot access geometry (polar, elevation, or elevation-azimuth)",
                "inputs": "location dict, satellite dict, search start/end",
                "plot_types": ["polar", "elevation", "elevation_azimuth"],
            },
            "plot_access_geometry_from_gp": {
                "description": "Plot access geometry from a GP record",
                "inputs": "gp_record dict, location dict, search start/end",
            },
            "plot_gabbard_diagram": {
                "description": "Plot Gabbard diagram (apogee/perigee vs period) for multiple objects",
                "inputs": "gp_records list",
            },
        },
        "satellite_sources": {
            "tle": "Provide tle_line1, tle_line2",
            "gp_record": "Provide gp_record dict, optional propagator_type",
            "state": "Provide epoch, state_eci, optional propagator_type",
        },
        "location_format": {
            "lon": "Longitude in degrees (required)",
            "lat": "Latitude in degrees (required)",
            "altitude_m": "Altitude in meters (default 0.0)",
            "name": "Location name (optional)",
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: plot_gp_history_elements
# ---------------------------------------------------------------------------


@mcp.tool()
def plot_gp_history_elements(
    gp_records: list[dict] | None = None,
    norad_cat_id: int | None = None,
    elements: list[str] | None = None,
    title: str | None = None,
) -> list | dict:
    """Plot orbital element trends from GP history records.

    Visualizes how orbital elements change over time from a series of GP records.
    Provide either pre-fetched gp_records or a norad_cat_id to auto-fetch history
    from SpaceTrack.

    Args:
        gp_records: List of GP record dicts (from spacetrack/celestrak tools).
        norad_cat_id: NORAD catalog ID to fetch GP history from SpaceTrack.
        elements: Element names to plot. None plots all 6 classical elements.
            Valid: semimajor_axis, eccentricity, inclination, ra_of_asc_node,
            arg_of_pericenter, mean_anomaly, mean_motion, bstar, period,
            apoapsis, periapsis.
        title: Optional plot title.
    """
    if gp_records is None and norad_cat_id is None:
        return error_response("Provide either gp_records or norad_cat_id.")

    # Fetch from SpaceTrack if needed
    if gp_records is None:
        try:
            gp_records = _fetch_gp_history(norad_cat_id)
        except RuntimeError as e:
            return error_response(str(e))
        except Exception as e:
            return error_response(f"Failed to fetch GP history: {e}")

    if not gp_records:
        return error_response("No GP records provided or found.")

    # Resolve elements
    plot_elements = elements if elements else _CLASSICAL_ELEMENTS
    invalid = [e for e in plot_elements if e not in _ELEMENT_LABELS]
    if invalid:
        return error_response(
            f"Unknown elements: {invalid}",
            valid_elements=list(_ELEMENT_LABELS.keys()),
        )

    # Extract epochs and values
    from datetime import datetime

    epochs = []
    for rec in gp_records:
        epoch_str = rec.get("epoch", "")
        # Parse epoch string to datetime for plotting
        cleaned = epoch_str.strip()
        for suffix in (" UTC", "Z"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        cleaned = cleaned.replace("T", " ", 1)
        try:
            epochs.append(datetime.fromisoformat(cleaned))
        except ValueError:
            epochs.append(None)

    values = {}
    for elem in plot_elements:
        values[elem] = [rec.get(elem) for rec in gp_records]

    # Filter out records with None epochs
    valid_mask = [e is not None for e in epochs]
    epochs = [e for e, v in zip(epochs, valid_mask) if v]
    for elem in plot_elements:
        values[elem] = [val for val, v in zip(values[elem], valid_mask) if v]

    if not epochs:
        return error_response("No valid epoch data in GP records.")

    # Plot
    n_plots = len(plot_elements)
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for ax, elem in zip(axes, plot_elements):
            label, unit = _ELEMENT_LABELS[elem]
            ax.plot(epochs, values[elem], ".", markersize=2)
            ylabel = f"{label} [{unit}]" if unit else label
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Epoch")
        fig.autofmt_xdate()

        obj_name = gp_records[0].get("object_name", "")
        if title:
            fig.suptitle(title)
        elif obj_name:
            fig.suptitle(f"{obj_name} - Orbital Element History")

        fig.tight_layout()

    summary = (
        f"Plotted {len(plot_elements)} element(s) for "
        f"{obj_name or 'object'} over {len(epochs)} GP records "
        f"({epochs[0]:%Y-%m-%d} to {epochs[-1]:%Y-%m-%d})."
    )
    logger.debug(summary)

    return [
        TextContent(type="text", text=summary),
        _figure_to_image(fig),
    ]


# ---------------------------------------------------------------------------
# Tool 3: plot_ground_track
# ---------------------------------------------------------------------------


@mcp.tool()
def plot_ground_track(
    satellite: dict,
    start_epoch: str,
    end_epoch: str,
    step_seconds: float = 60.0,
    ground_stations: list[dict] | None = None,
    show_grid: bool = False,
    show_legend: bool = False,
) -> list | dict:
    """Plot satellite ground track on a world map.

    Propagates the satellite and plots its sub-satellite point trace.

    Args:
        satellite: Satellite spec dict (same format as compute_access).
            Must include "source" key ("tle", "gp_record", or "state").
        start_epoch: Start of ground track (ISO epoch string).
        end_epoch: End of ground track (ISO epoch string).
        step_seconds: Propagation step size in seconds (default 60).
        ground_stations: Optional list of location dicts to plot on map.
        show_grid: Show latitude/longitude grid lines.
        show_legend: Show plot legend.
    """
    from brahe_mcp.accesses import _build_propagator, _build_location

    try:
        propagator = _build_propagator(satellite)
    except (ValueError, Exception) as e:
        return error_response(f"Failed to create propagator: {e}")

    try:
        start = parse_epoch(start_epoch)
        end = parse_epoch(end_epoch)
    except ValueError as e:
        return error_response(f"Invalid epoch: {e}")

    if float(end - start) <= 0:
        return error_response("end_epoch must be after start_epoch.")

    # Propagate
    try:
        traj = _propagate_trajectory(propagator, start, end)
    except Exception as e:
        return error_response(f"Propagation failed: {e}")

    # Build ground station locations
    gs_locations = None
    if ground_stations:
        try:
            gs_locations = [_build_location(gs) for gs in ground_stations]
        except ValueError as e:
            return error_response(f"Invalid ground station: {e}")

    # Plot
    try:
        with plt.style.context(_STYLE):
            fig = brahe.plot_groundtrack(
                trajectories=[traj],
                ground_stations=gs_locations,
                show_grid=show_grid,
                show_legend=show_legend,
                backend="matplotlib",
            )
    except Exception as e:
        return error_response(f"Ground track plot failed: {e}")

    duration_h = float(end - start) / 3600.0
    summary = f"Ground track plotted for {duration_h:.1f} hours ({start} to {end})."
    logger.debug(summary)

    return [
        TextContent(type="text", text=summary),
        _figure_to_image(fig),
    ]


# ---------------------------------------------------------------------------
# Tool 4: plot_ground_track_from_gp
# ---------------------------------------------------------------------------


@mcp.tool()
def plot_ground_track_from_gp(
    gp_record: dict,
    start_epoch: str,
    end_epoch: str,
    propagator_type: str = "sgp4",
    step_seconds: float = 60.0,
    ground_stations: list[dict] | None = None,
    show_grid: bool = False,
    show_legend: bool = False,
) -> list | dict:
    """Plot satellite ground track from a GP record.

    Convenience wrapper that creates a satellite spec from a GP record dict
    and delegates to plot_ground_track().

    Args:
        gp_record: GP record dict from celestrak/spacetrack tools.
        start_epoch: Start of ground track (ISO epoch string).
        end_epoch: End of ground track (ISO epoch string).
        propagator_type: "sgp4" (default) or "keplerian".
        step_seconds: Propagation step size in seconds (default 60).
        ground_stations: Optional list of location dicts to plot on map.
        show_grid: Show latitude/longitude grid lines.
        show_legend: Show plot legend.
    """
    satellite = {
        "source": "gp_record",
        "gp_record": gp_record,
        "propagator_type": propagator_type,
    }
    return plot_ground_track(
        satellite=satellite,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        step_seconds=step_seconds,
        ground_stations=ground_stations,
        show_grid=show_grid,
        show_legend=show_legend,
    )


# ---------------------------------------------------------------------------
# Tool 5: plot_orbit_elements
# ---------------------------------------------------------------------------


@mcp.tool()
def plot_orbit_elements(
    satellite: dict,
    start_epoch: str,
    end_epoch: str,
    step_seconds: float = 60.0,
) -> list | dict:
    """Plot Keplerian orbital element evolution over a propagation arc.

    Uses brahe's plot_keplerian_trajectory to show how each of the 6 classical
    orbital elements changes over time.

    Args:
        satellite: Satellite spec dict (same format as compute_access).
        start_epoch: Start epoch (ISO string).
        end_epoch: End epoch (ISO string).
        step_seconds: Propagation step size in seconds (default 60).
    """
    from brahe_mcp.accesses import _build_propagator

    try:
        propagator = _build_propagator(satellite)
    except (ValueError, Exception) as e:
        return error_response(f"Failed to create propagator: {e}")

    try:
        start = parse_epoch(start_epoch)
        end = parse_epoch(end_epoch)
    except ValueError as e:
        return error_response(f"Invalid epoch: {e}")

    if float(end - start) <= 0:
        return error_response("end_epoch must be after start_epoch.")

    try:
        traj = _propagate_trajectory(propagator, start, end)
    except Exception as e:
        return error_response(f"Propagation failed: {e}")

    try:
        with plt.style.context(_STYLE):
            fig = brahe.plot_keplerian_trajectory(
                trajectories=[traj],
                backend="matplotlib",
            )
    except Exception as e:
        return error_response(f"Orbit element plot failed: {e}")

    duration_h = float(end - start) / 3600.0
    summary = f"Orbital elements plotted for {duration_h:.1f} hours ({start} to {end})."
    logger.debug(summary)

    return [
        TextContent(type="text", text=summary),
        _figure_to_image(fig),
    ]


# ---------------------------------------------------------------------------
# Tool 6: plot_orbit_elements_from_gp
# ---------------------------------------------------------------------------


@mcp.tool()
def plot_orbit_elements_from_gp(
    gp_record: dict,
    start_epoch: str,
    end_epoch: str,
    propagator_type: str = "sgp4",
    step_seconds: float = 60.0,
) -> list | dict:
    """Plot Keplerian orbital element evolution from a GP record.

    Args:
        gp_record: GP record dict from celestrak/spacetrack tools.
        start_epoch: Start epoch (ISO string).
        end_epoch: End epoch (ISO string).
        propagator_type: "sgp4" (default) or "keplerian".
        step_seconds: Propagation step size in seconds (default 60).
    """
    satellite = {
        "source": "gp_record",
        "gp_record": gp_record,
        "propagator_type": propagator_type,
    }
    return plot_orbit_elements(
        satellite=satellite,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        step_seconds=step_seconds,
    )


# ---------------------------------------------------------------------------
# Tool 7: plot_access_geometry
# ---------------------------------------------------------------------------

VALID_ACCESS_PLOT_TYPES = {"polar", "elevation", "elevation_azimuth"}


@mcp.tool()
def plot_access_geometry(
    location: dict,
    satellite: dict,
    search_start: str,
    search_end: str,
    plot_type: str = "polar",
    constraints: list[dict] | None = None,
    constraint_logic: str = "all",
    min_elevation_deg: float | None = None,
) -> list | dict:
    """Plot access geometry between a satellite and ground location.

    Computes access windows and plots the satellite pass geometry.

    Args:
        location: Ground location dict with lon, lat, optional altitude_m/name.
        satellite: Satellite spec dict (same format as compute_access).
        search_start: Start of search window (ISO epoch string).
        search_end: End of search window (ISO epoch string).
        plot_type: "polar" (default), "elevation", or "elevation_azimuth".
        constraints: List of constraint spec dicts.
        constraint_logic: "all" (AND) or "any" (OR).
        min_elevation_deg: Convenience shortcut for elevation constraint.
    """
    from brahe_mcp.accesses import _build_propagator, _build_location, _build_constraints

    pt = plot_type.lower()
    if pt not in VALID_ACCESS_PLOT_TYPES:
        return error_response(
            f"Unknown plot_type: {plot_type!r}",
            valid_types=sorted(VALID_ACCESS_PLOT_TYPES),
        )

    try:
        loc = _build_location(location)
    except ValueError as e:
        return error_response(str(e))

    try:
        propagator = _build_propagator(satellite)
    except (ValueError, Exception) as e:
        return error_response(f"Failed to create propagator: {e}")

    try:
        start = parse_epoch(search_start)
        end = parse_epoch(search_end)
    except ValueError as e:
        return error_response(f"Invalid epoch: {e}")

    if float(end - start) <= 0:
        return error_response("search_end must be after search_start.")

    try:
        constraint = _build_constraints(constraints, constraint_logic, min_elevation_deg)
    except ValueError as e:
        return error_response(str(e))

    # For numerical propagators, pre-propagate
    if isinstance(propagator, brahe.NumericalOrbitPropagator):
        try:
            propagator.propagate_to(start)
            propagator.propagate_to(end)
        except Exception as e:
            return error_response(f"Numerical propagation failed: {e}")

    # Compute access windows
    try:
        windows = brahe.location_accesses(loc, propagator, start, end, constraint)
    except Exception as e:
        return error_response(f"Access computation failed: {e}")

    if not windows:
        return error_response(
            "No access windows found for the given parameters. "
            "Try a longer time window or lower elevation constraint."
        )

    # Plot
    plot_fn = {
        "polar": brahe.plot_access_polar,
        "elevation": brahe.plot_access_elevation,
        "elevation_azimuth": brahe.plot_access_elevation_azimuth,
    }[pt]

    try:
        with plt.style.context(_STYLE):
            fig = plot_fn(windows, propagator, backend="matplotlib")
    except Exception as e:
        return error_response(f"Access geometry plot failed: {e}")

    loc_name = location.get("name", "location")
    summary = (
        f"{pt.replace('_', ' ').title()} plot for {len(windows)} access window(s) "
        f"at {loc_name} ({search_start} to {search_end})."
    )
    logger.debug(summary)

    return [
        TextContent(type="text", text=summary),
        _figure_to_image(fig),
    ]


# ---------------------------------------------------------------------------
# Tool 8: plot_access_geometry_from_gp
# ---------------------------------------------------------------------------


@mcp.tool()
def plot_access_geometry_from_gp(
    gp_record: dict,
    location: dict,
    search_start: str,
    search_end: str,
    propagator_type: str = "sgp4",
    plot_type: str = "polar",
    constraints: list[dict] | None = None,
    constraint_logic: str = "all",
    min_elevation_deg: float | None = None,
) -> list | dict:
    """Plot access geometry from a GP record.

    Convenience wrapper that creates a satellite spec from a GP record dict
    and delegates to plot_access_geometry().

    Args:
        gp_record: GP record dict from celestrak/spacetrack tools.
        location: Ground location dict with lon, lat, optional altitude_m/name.
        search_start: Start of search window (ISO epoch string).
        search_end: End of search window (ISO epoch string).
        propagator_type: "sgp4" (default) or "keplerian".
        plot_type: "polar" (default), "elevation", or "elevation_azimuth".
        constraints: List of constraint spec dicts.
        constraint_logic: "all" (AND) or "any" (OR).
        min_elevation_deg: Convenience shortcut for elevation constraint.
    """
    satellite = {
        "source": "gp_record",
        "gp_record": gp_record,
        "propagator_type": propagator_type,
    }
    return plot_access_geometry(
        location=location,
        satellite=satellite,
        search_start=search_start,
        search_end=search_end,
        plot_type=plot_type,
        constraints=constraints,
        constraint_logic=constraint_logic,
        min_elevation_deg=min_elevation_deg,
    )


# ---------------------------------------------------------------------------
# Tool 9: plot_gabbard_diagram
# ---------------------------------------------------------------------------


@mcp.tool()
def plot_gabbard_diagram(
    gp_records: list[dict],
    altitude_units: str = "km",
    period_units: str = "min",
    title: str | None = None,
) -> list | dict:
    """Plot a Gabbard diagram (apogee/perigee altitude vs orbital period).

    Useful for visualizing debris clouds or constellation distributions.
    Each GP record becomes a point on the diagram.

    Args:
        gp_records: List of GP record dicts from celestrak/spacetrack tools.
        altitude_units: Units for altitude axis ("km" or "m").
        period_units: Units for period axis ("min", "hr", or "s").
        title: Optional plot title.
    """
    from brahe_mcp.propagation import _sgp4_from_gp

    if not gp_records:
        return error_response("gp_records must be a non-empty list.")

    # Build SGP4 propagators from each GP record
    propagators = []
    errors = []
    for i, gp in enumerate(gp_records):
        try:
            prop = _sgp4_from_gp(gp)
            propagators.append(prop)
        except Exception as e:
            errors.append(f"Record {i} ({gp.get('object_name', '?')}): {e}")

    if not propagators:
        return error_response(
            f"Could not create any propagators. Errors: {errors}"
        )

    try:
        with plt.style.context(_STYLE):
            fig = brahe.plot_gabbard_diagram(
                propagators,
                altitude_units=altitude_units,
                period_units=period_units,
                backend="matplotlib",
            )
            if title:
                fig.suptitle(title)
    except Exception as e:
        return error_response(f"Gabbard diagram plot failed: {e}")

    summary = (
        f"Gabbard diagram plotted for {len(propagators)} object(s)"
        f"{f' ({len(errors)} failed)' if errors else ''}."
    )
    logger.debug(summary)

    return [
        TextContent(type="text", text=summary),
        _figure_to_image(fig),
    ]
