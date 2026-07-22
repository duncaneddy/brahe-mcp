import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch, resolve_angle_format

VALID_ANGLE_FORMATS = {
    "degrees": brahe.AngleFormat.DEGREES,
    "radians": brahe.AngleFormat.RADIANS,
}

PARAM_UNITS = {
    "a": "m",
    "e": "dimensionless",
    "n": "deg/s or rad/s (see angle_format)",
    "period": "s",
    "state_eci": "m, m/s (comma-separated: x,y,z,vx,vy,vz)",
    "gm": "m^3/s^2",
    "r_body": "m",
    "angle_format": "degrees or radians",
}

ORBITAL_PROPERTIES = {
    "orbital_period": {
        "desc": "Compute orbital period from semi-major axis. Input: a (m). Output: period (s).",
        "required": ["a"],
        "optional": ["gm"],
        "output_unit": "s",
    },
    "orbital_period_from_state": {
        "desc": "Compute orbital period from ECI state vector. Input: state_eci (m, m/s). Output: period (s).",
        "required": ["state_eci"],
        "optional": ["gm"],
        "output_unit": "s",
    },
    "mean_motion": {
        "desc": "Compute mean motion from semi-major axis. Input: a (m). Output: mean motion (deg/s or rad/s per angle_format).",
        "required": ["a"],
        "optional": ["gm", "angle_format"],
        "output_unit": "deg/s",
    },
    "semimajor_axis": {
        "desc": "Compute semi-major axis from mean motion. Input: n (deg/s or rad/s per angle_format). Output: semi-major axis (m).",
        "required": ["n"],
        "optional": ["gm", "angle_format"],
        "output_unit": "m",
    },
    "semimajor_axis_from_period": {
        "desc": "Compute semi-major axis from orbital period. Input: period (s). Output: semi-major axis (m).",
        "required": ["period"],
        "optional": ["gm"],
        "output_unit": "m",
    },
    "periapsis_velocity": {
        "desc": "Compute velocity at periapsis. Input: a (m), e (dimensionless). Output: velocity (m/s).",
        "required": ["a", "e"],
        "optional": ["gm"],
        "output_unit": "m/s",
    },
    "apoapsis_velocity": {
        "desc": "Compute velocity at apoapsis. Input: a (m), e (dimensionless). Output: velocity (m/s).",
        "required": ["a", "e"],
        "optional": ["gm"],
        "output_unit": "m/s",
    },
    "periapsis_distance": {
        "desc": "Compute distance from body center at periapsis. Input: a (m), e (dimensionless). Output: distance (m).",
        "required": ["a", "e"],
        "optional": [],
        "output_unit": "m",
    },
    "apoapsis_distance": {
        "desc": "Compute distance from body center at apoapsis. Input: a (m), e (dimensionless). Output: distance (m).",
        "required": ["a", "e"],
        "optional": [],
        "output_unit": "m",
    },
    "periapsis_altitude": {
        "desc": "Compute altitude above surface at periapsis. Input: a (m), e (dimensionless). Output: altitude (m).",
        "required": ["a", "e"],
        "optional": ["r_body"],
        "output_unit": "m",
    },
    "apoapsis_altitude": {
        "desc": "Compute altitude above surface at apoapsis. Input: a (m), e (dimensionless). Output: altitude (m).",
        "required": ["a", "e"],
        "optional": ["r_body"],
        "output_unit": "m",
    },
    "sun_synchronous_inclination": {
        "desc": "Compute inclination for sun-synchronous orbit. Input: a (m), e (dimensionless). Output: inclination (deg or rad per angle_format).",
        "required": ["a", "e"],
        "optional": ["angle_format"],
        "output_unit": "deg",
    },
    "geo_sma": {
        "desc": "Compute geostationary orbit semi-major axis. No inputs required. Output: semi-major axis (m).",
        "required": [],
        "optional": [],
        "output_unit": "m",
    },
}

ANOMALY_CONVERSIONS = {
    "eccentric_to_mean": {
        "desc": "Eccentric anomaly to mean anomaly",
        "func": brahe.anomaly_eccentric_to_mean,
    },
    "mean_to_eccentric": {
        "desc": "Mean anomaly to eccentric anomaly",
        "func": brahe.anomaly_mean_to_eccentric,
    },
    "true_to_eccentric": {
        "desc": "True anomaly to eccentric anomaly",
        "func": brahe.anomaly_true_to_eccentric,
    },
    "eccentric_to_true": {
        "desc": "Eccentric anomaly to true anomaly",
        "func": brahe.anomaly_eccentric_to_true,
    },
    "true_to_mean": {
        "desc": "True anomaly to mean anomaly",
        "func": brahe.anomaly_true_to_mean,
    },
    "mean_to_true": {
        "desc": "Mean anomaly to true anomaly",
        "func": brahe.anomaly_mean_to_true,
    },
}

_PROPERTIES_LOWER = {k.lower(): k for k in ORBITAL_PROPERTIES}
_CONVERSIONS_LOWER = {k.lower(): k for k in ANOMALY_CONVERSIONS}


def _error_response(message: str) -> dict:
    """Return an error dict with valid options for discoverability."""
    return {
        "error": message,
        "valid_computations": sorted(ORBITAL_PROPERTIES.keys()),
        "valid_conversions": sorted(ANOMALY_CONVERSIONS.keys()),
        "valid_angle_formats": sorted(VALID_ANGLE_FORMATS.keys()),
    }


def _dispatch_computation(
    name: str,
    *,
    a: float | None,
    e: float | None,
    n: float | None,
    period: float | None,
    state_eci: str | None,
    gm: float | None,
    r_body: float | None,
    angle_format: brahe.AngleFormat,
) -> float:
    """Dispatch to the appropriate brahe function."""
    if name == "orbital_period":
        if gm is not None:
            return brahe.orbital_period_general(a, gm)
        return brahe.orbital_period(a)

    elif name == "orbital_period_from_state":
        parts = [float(x.strip()) for x in state_eci.split(",")]
        state = np.array(parts)
        return brahe.orbital_period_from_state(
            state, gm if gm is not None else brahe.GM_EARTH
        )

    elif name == "mean_motion":
        if gm is not None:
            return brahe.mean_motion_general(a, gm, angle_format)
        return brahe.mean_motion(a, angle_format)

    elif name == "semimajor_axis":
        if gm is not None:
            return brahe.semimajor_axis_general(n, gm, angle_format)
        return brahe.semimajor_axis(n, angle_format)

    elif name == "semimajor_axis_from_period":
        if gm is not None:
            return brahe.semimajor_axis_from_orbital_period_general(period, gm)
        return brahe.semimajor_axis_from_orbital_period(period)

    elif name == "periapsis_velocity":
        return brahe.periapsis_velocity(
            a, e, gm=gm if gm is not None else brahe.GM_EARTH
        )

    elif name == "apoapsis_velocity":
        return brahe.apoapsis_velocity(
            a, e, gm=gm if gm is not None else brahe.GM_EARTH
        )

    elif name == "periapsis_distance":
        return brahe.periapsis_distance(a, e)

    elif name == "apoapsis_distance":
        return brahe.apoapsis_distance(a, e)

    elif name == "periapsis_altitude":
        return brahe.periapsis_altitude(
            a, e, r_body=r_body if r_body is not None else brahe.R_EARTH
        )

    elif name == "apoapsis_altitude":
        return brahe.apoapsis_altitude(
            a, e, r_body=r_body if r_body is not None else brahe.R_EARTH
        )

    elif name == "sun_synchronous_inclination":
        return brahe.sun_synchronous_inclination(a, e, angle_format=angle_format)

    elif name == "geo_sma":
        return brahe.geo_sma()


@mcp.tool()
def list_orbital_computations() -> dict:
    """List all available orbital property computations and anomaly conversions.

    Returns names, descriptions, and parameter requirements for each computation.
    """
    logger.debug("Listing orbital computations")
    return {
        "orbital_properties": [
            {
                "name": name,
                "description": info["desc"],
                "required_params": info["required"],
                "optional_params": info["optional"],
                "input_units": {p: PARAM_UNITS[p] for p in info["required"] + info["optional"] if p in PARAM_UNITS},
                "output_unit": info["output_unit"],
            }
            for name, info in ORBITAL_PROPERTIES.items()
        ],
        "anomaly_conversions": [
            {
                "name": name,
                "description": info["desc"],
                "input_units": {"anomaly": "deg or rad (see angle_format)", "e": "dimensionless"},
                "output_unit": "deg or rad (matches angle_format)",
            }
            for name, info in ANOMALY_CONVERSIONS.items()
        ],
        "mean_element_conversions": {
            "tools": [
                "convert_equinoctial",
                "convert_mean_osculating",
                "convert_mean_osculating_batch",
            ],
            "directions": sorted(MEAN_OSC_DIRECTIONS),
            "methods": sorted(MEAN_ELEMENT_METHODS),
            "equinoctial_components": list(_EQN_LABELS),
            "keplerian_components": list(_KOE_LABELS),
            "notes": [
                "The numerical method is batch-only.",
                "Numerical mean_to_osc requires force_config.",
                "Numerical osc_to_mean with edge='truncate' returns fewer "
                "states than it receives; read n_output.",
                "Brouwer-Lyddane is a first-order theory; mean -> osc -> mean "
                "is not exact.",
            ],
        },
    }


@mcp.tool()
def compute_orbital_property(
    computation: str,
    a: float | None = None,
    e: float | None = None,
    n: float | None = None,
    period: float | None = None,
    state_eci: str | None = None,
    gm: float | None = None,
    r_body: float | None = None,
    angle_format: str = "degrees",
) -> dict:
    """Compute an orbital property using brahe astrodynamics functions.

    Use list_orbital_computations() to see all available computations and their parameters.

    Args:
        computation: Name of the computation (case-insensitive), e.g. "orbital_period".
        a: Semi-major axis in meters.
        e: Eccentricity (dimensionless).
        n: Mean motion (deg/s or rad/s depending on angle_format).
        period: Orbital period in seconds.
        state_eci: ECI state vector as comma-separated string "x,y,z,vx,vy,vz" (m, m/s).
        gm: Gravitational parameter (m^3/s^2). Defaults to Earth if omitted.
        r_body: Body radius (m). Defaults to Earth equatorial radius if omitted.
        angle_format: Angle unit - "degrees" (default) or "radians".
    """
    # Validate computation name
    key = _PROPERTIES_LOWER.get(computation.lower())
    if key is None:
        logger.warning("Unknown computation: {}", computation)
        return _error_response(f"Unknown computation: {computation!r}")

    # Validate angle_format
    fmt_lower = angle_format.lower()
    if fmt_lower not in VALID_ANGLE_FORMATS:
        logger.warning("Invalid angle_format: {}", angle_format)
        return _error_response(f"Invalid angle_format: {angle_format!r}")
    angle_fmt = VALID_ANGLE_FORMATS[fmt_lower]

    # Check required params
    prop = ORBITAL_PROPERTIES[key]
    param_map = {
        "a": a, "e": e, "n": n, "period": period, "state_eci": state_eci,
    }
    missing = [p for p in prop["required"] if param_map.get(p) is None]
    if missing:
        logger.warning("Missing required params for {}: {}", key, missing)
        return _error_response(
            f"Missing required parameter(s) for {key!r}: {missing}. "
            f"Required: {prop['required']}, Optional: {prop['optional']}"
        )

    # Validate state_eci format
    if key == "orbital_period_from_state":
        parts = state_eci.split(",")
        if len(parts) != 6:
            return _error_response(
                "state_eci must have exactly 6 comma-separated values: x,y,z,vx,vy,vz"
            )
        try:
            [float(x.strip()) for x in parts]
        except ValueError:
            return _error_response("state_eci values must be numeric")

    # Dispatch
    try:
        result = _dispatch_computation(
            key,
            a=a, e=e, n=n, period=period, state_eci=state_eci,
            gm=gm, r_body=r_body, angle_format=angle_fmt,
        )
    except Exception as exc:
        logger.error("Error computing {}: {}", key, exc)
        return _error_response(f"Error computing {key!r}: {exc}")

    # Build input summary
    inputs = {}
    for p in prop["required"] + prop["optional"]:
        val = param_map.get(p)
        if val is not None:
            inputs[p] = val
    if gm is not None:
        inputs["gm"] = gm
    if r_body is not None:
        inputs["r_body"] = r_body
    if "angle_format" in prop["optional"]:
        inputs["angle_format"] = fmt_lower

    # Determine output unit
    output_unit = prop["output_unit"]
    if output_unit in ("deg/s", "deg") and fmt_lower == "radians":
        output_unit = output_unit.replace("deg", "rad")

    # Build input units for the params actually used
    input_units = {p: PARAM_UNITS[p] for p in inputs if p in PARAM_UNITS}

    logger.debug("Computed {}: {}", key, result)
    return {
        "computation": key,
        "result": {"value": result, "output_unit": output_unit},
        "inputs": inputs,
        "input_units": input_units,
    }


@mcp.tool()
def convert_anomaly(
    conversion: str,
    anomaly: float,
    e: float,
    angle_format: str = "degrees",
) -> dict:
    """Convert between orbital anomaly types (mean, eccentric, true).

    Use list_orbital_computations() to see all available conversions.

    Args:
        conversion: Conversion name (case-insensitive), e.g. "eccentric_to_mean".
        anomaly: Input anomaly value in the specified angle_format.
        e: Orbital eccentricity (dimensionless).
        angle_format: Angle unit - "degrees" (default) or "radians".
    """
    # Validate conversion name
    key = _CONVERSIONS_LOWER.get(conversion.lower())
    if key is None:
        logger.warning("Unknown anomaly conversion: {}", conversion)
        return _error_response(f"Unknown anomaly conversion: {conversion!r}")

    # Validate angle_format
    fmt_lower = angle_format.lower()
    if fmt_lower not in VALID_ANGLE_FORMATS:
        logger.warning("Invalid angle_format: {}", angle_format)
        return _error_response(f"Invalid angle_format: {angle_format!r}")
    angle_fmt = VALID_ANGLE_FORMATS[fmt_lower]

    # Call brahe
    try:
        result = ANOMALY_CONVERSIONS[key]["func"](
            anomaly, e, angle_format=angle_fmt
        )
    except Exception as exc:
        logger.error("Error in anomaly conversion {}: {}", key, exc)
        return _error_response(f"Error in conversion {key!r}: {exc}")

    logger.debug("Anomaly conversion {}: {} -> {}", key, anomaly, result)
    return {
        "conversion": key,
        "input": {"anomaly": anomaly, "eccentricity": e, "angle_format": fmt_lower},
        "output": {"anomaly": result, "angle_format": fmt_lower},
    }


EQUINOCTIAL_DIRECTIONS = {"koe_to_equinoctial", "equinoctial_to_koe"}

# Keplerian [a, e, i, RAAN, omega, M]; equinoctial [a, h, k, p, q, l]
# (Vallado 2-99). Only `l` is angular in the equinoctial set.
_KOE_LABELS = ("a_m", "e", "i", "RAAN", "omega", "M")
_EQN_LABELS = ("a_m", "h", "k", "p", "q", "l")


@mcp.tool()
def convert_equinoctial(
    state: list[float],
    direction: str,
    fr: int = 1,
    angle_format: str = "degrees",
) -> dict:
    """Convert between Keplerian and equinoctial orbital elements.

    Equinoctial elements are nonsingular for circular and equatorial orbits,
    where the Keplerian argument of perigee and RAAN become undefined.

    Args:
        state: 6-element state. Keplerian [a, e, i, RAAN, omega, M] with a in
            meters, or equinoctial [a, h, k, p, q, l] with a in meters.
        direction: "koe_to_equinoctial" or "equinoctial_to_koe".
        fr: Retrograde factor, +1 (default) for direct orbits or -1 for
            near-retrograde orbits where the standard set is singular at i=180.
        angle_format: "degrees" (default) or "radians".
    """
    key = direction.lower()
    if key not in EQUINOCTIAL_DIRECTIONS:
        return error_response(
            f"Unknown direction: {direction!r}",
            valid_directions=sorted(EQUINOCTIAL_DIRECTIONS),
        )

    if fr not in (1, -1):
        return error_response(f"Invalid fr: {fr!r}. Must be +1 or -1.")

    if len(state) != 6:
        return error_response(
            f"state must have exactly 6 elements, got {len(state)}"
        )

    fmt_lower = angle_format.lower()
    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    try:
        vec = np.array(state, dtype=float)
        if key == "koe_to_equinoctial":
            out = brahe.state_koe_to_equinoctial(vec, angle_fmt, fr)
            labels = _EQN_LABELS
        else:
            out = brahe.state_equinoctial_to_koe(vec, angle_fmt, fr)
            labels = _KOE_LABELS
    except Exception as e:
        logger.error("Equinoctial conversion error: {}", e)
        return error_response(f"Conversion error: {e}")

    out_list = np.array(out, dtype=float).tolist()
    if not np.all(np.isfinite(out_list)):
        return error_response(
            "Conversion produced non-finite output (state may be singular "
            "for this transform, e.g. near-zero eccentricity or equatorial "
            "inclination)",
            direction=key,
        )
    logger.debug("Equinoctial {}: {} -> {}", key, state, out_list)
    return {
        "direction": key,
        "input": {"state": state, "fr": fr, "angle_format": fmt_lower},
        "output": {
            "state": out_list,
            "components": dict(zip(labels, out_list)),
        },
    }


MEAN_OSC_DIRECTIONS = {"mean_to_osc", "osc_to_mean"}
MEAN_ELEMENT_METHODS = {"brouwer_lyddane", "numerical"}


@mcp.tool()
def convert_mean_osculating(
    state: list[float],
    direction: str,
    method: str = "brouwer_lyddane",
    angle_format: str = "degrees",
) -> dict:
    """Convert a single Keplerian state between mean and osculating elements.

    Uses the Brouwer-Lyddane analytical theory. The numerical method is
    batch-only; use convert_mean_osculating_batch for it.

    Brouwer-Lyddane is a first-order theory, so mean -> osc -> mean does not
    return the input exactly. Residuals grow for near-circular orbits where
    omega and M are ill-conditioned.

    Args:
        state: 6-element Keplerian [a, e, i, RAAN, omega, M], a in meters.
        direction: "mean_to_osc" or "osc_to_mean".
        method: "brouwer_lyddane" (default). "numerical" is rejected here.
        angle_format: "degrees" (default) or "radians".
    """
    key = direction.lower()
    if key not in MEAN_OSC_DIRECTIONS:
        return error_response(
            f"Unknown direction: {direction!r}",
            valid_directions=sorted(MEAN_OSC_DIRECTIONS),
        )

    method_key = method.lower()
    if method_key == "numerical":
        return error_response(
            "The numerical mean-element method is batch-only. Use "
            "convert_mean_osculating_batch with a series of epochs and states.",
            valid_methods=["brouwer_lyddane"],
        )
    if method_key not in MEAN_ELEMENT_METHODS:
        return error_response(
            f"Unknown method: {method!r}",
            valid_methods=sorted(MEAN_ELEMENT_METHODS),
        )

    if len(state) != 6:
        return error_response(
            f"state must have exactly 6 elements, got {len(state)}"
        )

    fmt_lower = angle_format.lower()
    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    bl = brahe.MeanElementMethod.BROUWER_LYDDANE
    try:
        vec = np.array(state, dtype=float)
        if key == "mean_to_osc":
            out = brahe.state_koe_mean_to_osc(vec, bl, angle_fmt)
        else:
            out = brahe.state_koe_osc_to_mean(vec, bl, angle_fmt)
    except Exception as e:
        logger.error("Mean/osculating conversion error: {}", e)
        return error_response(f"Conversion error: {e}")

    out_list = np.array(out, dtype=float).tolist()
    if not np.all(np.isfinite(out_list)):
        return error_response(
            "Conversion produced non-finite output (state may be singular "
            "for this transform, e.g. near-zero eccentricity or equatorial "
            "inclination)",
            direction=key,
        )
    logger.debug("Mean/osc {}: {} -> {}", key, state, out_list)
    return {
        "direction": key,
        "input": {"state": state, "angle_format": fmt_lower},
        "output": {
            "state": out_list,
            "method": "brouwer_lyddane",
            "components": dict(zip(_KOE_LABELS, out_list)),
        },
    }


_WINDOW_ALIGNMENTS = {
    "centered": brahe.WindowAlignment.CENTERED,
    "leading": brahe.WindowAlignment.LEADING,
    "trailing": brahe.WindowAlignment.TRAILING,
}

_WINDOW_EDGES = {
    "truncate": brahe.WindowEdgeHandling.TRUNCATE,
    "preserve_window": brahe.WindowEdgeHandling.PRESERVE_WINDOW,
}


@mcp.tool()
def convert_mean_osculating_batch(
    epochs: list[str],
    states: list[list[float]],
    direction: str,
    method: str = "brouwer_lyddane",
    window_seconds: float = 5400.0,
    alignment: str = "centered",
    edge: str = "truncate",
    force_model: str = "earth_gravity",
    force_config: dict | None = None,
    integrator: dict | None = None,
    tolerance: float = 1.0,
    max_iterations: int = 50,
    angle_format: str = "degrees",
) -> dict:
    """Convert a series of Keplerian states between mean and osculating elements.

    Two methods are available. "brouwer_lyddane" maps each state independently
    and always preserves series length. "numerical" applies windowed averaging;
    with edge="truncate" it returns FEWER states than it receives, because the
    averaging window consumes the series edges. Always read n_output rather
    than assuming it equals n_input.

    Numerical mean_to_osc inverts the averaging by differential correction and
    therefore requires force_config. Numerical osc_to_mean does not.

    Args:
        epochs: ISO epoch strings, one per row of states.
        states: Rows of Keplerian [a, e, i, RAAN, omega, M], a in meters.
        direction: "mean_to_osc" or "osc_to_mean".
        method: "brouwer_lyddane" (default) or "numerical".
        window_seconds: Averaging window length. Numerical only.
        alignment: "centered" (default), "leading", or "trailing".
        edge: "truncate" (default) or "preserve_window".
        force_model: Force model preset for numerical mean_to_osc.
            "earth_gravity" (default) uses 20x20 EGM2008 with no drag or SRP,
            which captures the secular effects mean elements average over and
            needs no spacecraft parameters. See list_propagation_options() for
            other presets. Not used by osc_to_mean or by brouwer_lyddane.
        force_config: Force model override dict for numerical mean_to_osc.
            Same keys as propagate_numerical; see list_propagation_options().
            Required for numerical mean_to_osc.
        integrator: Integrator dict, optional for numerical mean_to_osc.
            Same keys as propagate_numerical.
        tolerance: Convergence tolerance on the mean-element residual.
        max_iterations: Maximum differential-correction iterations.
        angle_format: "degrees" (default) or "radians".
    """
    from brahe_mcp.propagation import (
        _build_force_config,
        _build_propagation_config,
    )

    key = direction.lower()
    if key not in MEAN_OSC_DIRECTIONS:
        return error_response(
            f"Unknown direction: {direction!r}",
            valid_directions=sorted(MEAN_OSC_DIRECTIONS),
        )

    method_key = method.lower()
    if method_key not in MEAN_ELEMENT_METHODS:
        return error_response(
            f"Unknown method: {method!r}",
            valid_methods=sorted(MEAN_ELEMENT_METHODS),
        )

    if len(epochs) != len(states):
        return error_response(
            f"epochs and states must be the same length, got "
            f"{len(epochs)} epochs and {len(states)} states"
        )
    if not epochs:
        return error_response("epochs and states must not be empty")

    for i, row in enumerate(states):
        if len(row) != 6:
            return error_response(
                f"states[{i}] must have exactly 6 elements, got {len(row)}"
            )

    fmt_lower = angle_format.lower()
    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    try:
        epoch_objs = [parse_epoch(e) for e in epochs]
    except Exception as e:
        return error_response(f"Invalid epoch: {e}")

    if method_key == "brouwer_lyddane":
        mean_method = brahe.MeanElementMethod.BROUWER_LYDDANE
    else:
        align_key = alignment.lower()
        if align_key not in _WINDOW_ALIGNMENTS:
            return error_response(
                f"Unknown alignment: {alignment!r}",
                valid_alignments=sorted(_WINDOW_ALIGNMENTS),
            )
        edge_key = edge.lower()
        if edge_key not in _WINDOW_EDGES:
            return error_response(
                f"Unknown edge: {edge!r}", valid_edges=sorted(_WINDOW_EDGES)
            )

        inverse_cfg = None
        if key == "mean_to_osc":
            if force_config is None:
                return error_response(
                    "Numerical mean_to_osc inverts the windowed average by "
                    "differential correction and requires force_config. See "
                    "list_propagation_options() for valid keys, or use "
                    "method='brouwer_lyddane'."
                )
            try:
                # The "central_body" preset accepts only moon/mars/emb, so it
                # must NOT be used here. Non-central_body presets ignore the
                # body argument entirely, so "earth" is inert but explicit.
                fc = _build_force_config(force_model, "earth", force_config)
                pc = _build_propagation_config(integrator)
                inverse_cfg = brahe.MeanElementInverseConfig(
                    fc, pc, float(tolerance), int(max_iterations)
                )
            except (ValueError, KeyError, AttributeError, TypeError, OverflowError) as e:
                return error_response(
                    f"Invalid force_model, force_config, integrator, tolerance, "
                    f"or max_iterations: {e}"
                )

        num_cfg = brahe.MeanElementNumericalMethodConfig(
            float(window_seconds),
            _WINDOW_ALIGNMENTS[align_key],
            _WINDOW_EDGES[edge_key],
            inverse_cfg,
        )
        mean_method = brahe.MeanElementMethod.numerical(num_cfg)

    try:
        arr = np.array(states, dtype=float)
        if key == "mean_to_osc":
            out_epochs, out_states = brahe.batch_state_koe_mean_to_osc(
                epoch_objs, arr, mean_method, angle_fmt
            )
        else:
            out_epochs, out_states = brahe.batch_state_koe_osc_to_mean(
                epoch_objs, arr, mean_method, angle_fmt
            )
    except Exception as e:
        logger.error("Batch mean/osculating error: {}", e)
        return error_response(f"Conversion error: {e}")

    out_arr = np.array(out_states, dtype=float)
    out_list = out_arr.tolist()
    n_in = len(states)
    n_out = len(out_list)

    # One non-finite row invalidates the whole response: partial success
    # (returning the good rows while silently dropping the bad ones) would
    # decouple output length from n_output/dropped_by_edge_handling, which
    # callers rely on to detect edge-truncation. Reject and name the rows
    # instead, matching the single-state tool's error-not-NaN behavior.
    non_finite_rows = [
        i for i in range(n_out) if not np.all(np.isfinite(out_arr[i]))
    ]
    if non_finite_rows:
        return error_response(
            "Conversion produced non-finite output in one or more rows "
            "(state may be singular for this transform, e.g. near-zero "
            "eccentricity or equatorial inclination)",
            direction=key,
            non_finite_rows=non_finite_rows,
        )

    logger.debug("Batch mean/osc {} {}: {} -> {}", key, method_key, n_in, n_out)
    return {
        "direction": key,
        "input": {
            "n_states": n_in,
            "method": method_key,
            "angle_format": fmt_lower,
        },
        "output": {
            "epochs": [str(e) for e in out_epochs],
            "states": out_list,
            "n_input": n_in,
            "n_output": n_out,
            "dropped_by_edge_handling": n_in - n_out,
            "method": method_key,
            "component_names": list(_KOE_LABELS),
        },
    }
