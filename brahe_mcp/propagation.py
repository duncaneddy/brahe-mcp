"""Orbit propagation MCP tools using brahe propagators."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch, resolve_angle_format
from brahe_mcp._gp import _sgp4_from_gp, _eci_state_from_gp  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_OUTPUT_FRAMES = {
    "eci", "ecef", "gcrf", "itrf", "eme2000", "koe_osc", "koe_mean", "bci", "bcbf",
}

FORCE_MODEL_PRESETS = {
    "default": "Default: 20x20 EGM2008 gravity, Harris-Priester drag, SRP with conical eclipse, Sun/Moon third-body. Requires spacecraft_params.",
    "two_body": "Point-mass gravity only. No spacecraft_params needed.",
    "earth_gravity": "20x20 EGM2008 gravity only. No drag/SRP/third-body. No spacecraft_params needed.",
    "leo_default": "Optimized for LEO orbits. Requires spacecraft_params.",
    "geo_default": "Optimized for GEO orbits. Requires spacecraft_params.",
    "high_fidelity": "High-fidelity force model. Requires spacecraft_params.",
    "conservative_forces": "Gravity + third-body + relativity. No drag/SRP. No spacecraft_params needed.",
    "cislunar_default": "Earth-Moon barycenter (EMB) cislunar propagation. Requires spacecraft_params.",
    "lunar_default": "Propagation about the Moon (50x50 lunar gravity). Requires spacecraft_params.",
    "mars_default": "Propagation about Mars (50x50 Mars gravity + drag). Requires spacecraft_params.",
    "central_body": "Generic body default force model; set central_body to moon/mars/emb.",
}

# Output component labels by frame
OUTPUT_LABELS = {
    "eci": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "ecef": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "gcrf": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "itrf": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "eme2000": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "koe_osc": ("a_m", "e", "i", "RAAN", "omega", "M"),
    "koe_mean": ("a_m", "e", "i", "RAAN", "omega", "M"),
    "bci": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "bcbf": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
}

MAX_EPOCHS = 100_000

_CENTRAL_BODIES = {
    "earth": brahe.CentralBody.Earth,
    "moon": brahe.CentralBody.Moon,
    "mars": brahe.CentralBody.Mars,
    "emb": brahe.CentralBody.EMB,
    "ssb": brahe.CentralBody.SSB,
}

_BODY_PRESETS = {
    "moon": brahe.ForceModelConfig.lunar_default,
    "mars": brahe.ForceModelConfig.mars_default,
    "emb": brahe.ForceModelConfig.cislunar_default,
}

_INTEGRATION_METHODS = {
    "dp54": brahe.IntegrationMethod.DP54,
    "rk4": brahe.IntegrationMethod.RK4,
    "rkf45": brahe.IntegrationMethod.RKF45,
    "rkf78": brahe.IntegrationMethod.RKF78,
    "rkn1210": brahe.IntegrationMethod.RKN1210,
}

_ATMOSPHERIC_MODELS = {
    "harris_priester": brahe.AtmosphericModel.HARRIS_PRIESTER,
    "nrlmsise00": brahe.AtmosphericModel.NRLMSISE00,
}

_ECLIPSE_MODELS = {
    "conical": brahe.EclipseModel.CONICAL,
    "cylindrical": brahe.EclipseModel.CYLINDRICAL,
    "none": brahe.EclipseModel.NONE,
}

_THIRD_BODIES = {
    "sun": brahe.ThirdBody.SUN, "moon": brahe.ThirdBody.MOON,
    "earth": brahe.ThirdBody.EARTH, "venus": brahe.ThirdBody.VENUS,
    "mars": brahe.ThirdBody.MARS, "jupiter": brahe.ThirdBody.JUPITER,
    "saturn": brahe.ThirdBody.SATURN,
}

_EPHEMERIS_SOURCES = {
    "low_precision": brahe.EphemerisSource.LowPrecision,
    "de440s": brahe.EphemerisSource.DE440s,
    "de440": brahe.EphemerisSource.DE440,
}


def _resolve_central_body(name: str):
    """Resolve a central-body name or NAIF id string to a CentralBody."""
    key = str(name).lower()
    if key in _CENTRAL_BODIES:
        return _CENTRAL_BODIES[key]
    try:
        return brahe.CentralBody.from_naif_id(int(name))
    except (ValueError, TypeError):
        raise ValueError(
            f"Unknown central_body: {name!r}. Valid: {sorted(_CENTRAL_BODIES)} or a NAIF id."
        )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prop_error(message: str, **context) -> dict:
    """Build a propagation-specific error response."""
    return error_response(message, **context)


def _label_components(vector: list[float], frame: str) -> dict:
    """Build a labeled dict from a vector and frame-specific labels."""
    keys = OUTPUT_LABELS.get(frame, tuple(f"c{i}" for i in range(len(vector))))
    return dict(zip(keys, vector))


def _build_epoch_list(
    target_epoch: str | None,
    start_epoch: str | None,
    end_epoch: str | None,
    step_seconds: float,
) -> list[brahe.Epoch]:
    """Build a list of epochs from either a single target or a time range.

    Args:
        target_epoch: Single target epoch (ISO string).
        start_epoch: Range start (ISO string).
        end_epoch: Range end (ISO string).
        step_seconds: Step size in seconds for range propagation.

    Returns:
        List of brahe.Epoch objects.

    Raises:
        ValueError: If inputs are invalid or range too large.
    """
    if target_epoch is not None:
        return [parse_epoch(target_epoch)]

    if start_epoch is not None and end_epoch is not None:
        start = parse_epoch(start_epoch)
        end = parse_epoch(end_epoch)
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")

        # Calculate number of steps
        duration = float(end - start)
        if duration < 0:
            raise ValueError("end_epoch must be after start_epoch")

        n_steps = round(duration / step_seconds) + 1
        if n_steps > MAX_EPOCHS:
            raise ValueError(
                f"Time range would produce {n_steps} epochs, exceeding maximum of {MAX_EPOCHS}. "
                f"Increase step_seconds or reduce time range."
            )

        return [start + i * step_seconds for i in range(n_steps)]

    raise ValueError(
        "Provide either target_epoch for a single point, or both start_epoch and end_epoch for a range."
    )


def _get_state(
    propagator,
    epoch: brahe.Epoch,
    output_frame: str,
    angle_format: brahe.AngleFormat,
) -> np.ndarray:
    """Get state from a propagator in the requested output frame.

    Args:
        propagator: A brahe propagator (SGP, Keplerian, or NumericalOrbit).
        epoch: Target epoch.
        output_frame: Output coordinate frame (lowercase).
        angle_format: Angle format for KOE output.

    Returns:
        State vector as numpy array.
    """
    dispatch = {
        "eci": lambda: propagator.state_eci(epoch),
        "ecef": lambda: propagator.state_ecef(epoch),
        "gcrf": lambda: propagator.state_gcrf(epoch),
        "itrf": lambda: propagator.state_itrf(epoch),
        "eme2000": lambda: propagator.state_eme2000(epoch),
        "koe_osc": lambda: propagator.state_koe_osc(epoch, angle_format),
        "koe_mean": lambda: propagator.state_koe_mean(epoch, angle_format),
        "bci": lambda: propagator.state_bci(epoch),
        "bcbf": lambda: propagator.state_bcbf(epoch),
    }
    return dispatch[output_frame]()


def _format_state(vector: list[float], epoch_str: str, frame: str) -> dict:
    """Format a state vector with epoch and labeled components."""
    return {
        "epoch": epoch_str,
        "vector": vector,
        "components": _label_components(vector, frame),
    }


def _propagate_and_collect(
    propagator,
    epochs: list[brahe.Epoch],
    output_frame: str,
    angle_format: brahe.AngleFormat,
    needs_propagate_to: bool = False,
) -> list[dict]:
    """Propagate to each epoch and collect formatted state dicts.

    Args:
        propagator: A brahe propagator instance.
        epochs: List of target epochs.
        output_frame: Output coordinate frame (lowercase).
        angle_format: Angle format for KOE output.
        needs_propagate_to: If True, call propagate_to() for the last epoch
            before querying states (required for NumericalOrbitPropagator).
    """
    if needs_propagate_to and epochs:
        propagator.propagate_to(epochs[-1])

    results = []
    for epc in epochs:
        state = _get_state(propagator, epc, output_frame, angle_format)
        vec = state.tolist()
        results.append(_format_state(vec, str(epc), output_frame))
    return results


def _build_result(
    propagator_type: str,
    initial_epoch_str: str,
    output_frame: str,
    states: list[dict],
) -> dict:
    """Build the common return dict for propagation results."""
    base = {
        "propagator_type": propagator_type,
        "initial_epoch": initial_epoch_str,
        "output_frame": output_frame,
    }
    if len(states) == 1:
        base["state"] = states[0]
    else:
        base["count"] = len(states)
        base["states"] = states
    return base


def _validate_output_frame(frame: str) -> str | None:
    """Validate and normalize output frame. Returns lowercase or None on error."""
    lower = frame.lower()
    if lower not in VALID_OUTPUT_FRAMES:
        return None
    return lower


def _build_force_config(preset: str, central_body: str, force_config: dict | None):
    """Build a validated ForceModelConfig from a preset, central body, and overrides."""
    preset_lower = preset.lower()
    cb_lower = str(central_body).lower()

    base_presets = {
        "default": brahe.ForceModelConfig.default,
        "two_body": brahe.ForceModelConfig.two_body,
        "earth_gravity": brahe.ForceModelConfig.earth_gravity,
        "leo_default": brahe.ForceModelConfig.leo_default,
        "geo_default": brahe.ForceModelConfig.geo_default,
        "high_fidelity": brahe.ForceModelConfig.high_fidelity,
        "conservative_forces": brahe.ForceModelConfig.conservative_forces,
        "cislunar_default": brahe.ForceModelConfig.cislunar_default,
        "lunar_default": brahe.ForceModelConfig.lunar_default,
        "mars_default": brahe.ForceModelConfig.mars_default,
    }

    # central_body preset routes to the body's default force model.
    if preset_lower == "central_body":
        if cb_lower not in _BODY_PRESETS:
            raise ValueError(
                f"central_body preset needs central_body in {sorted(_BODY_PRESETS)}; "
                f"got {central_body!r}."
            )
        config = _BODY_PRESETS[cb_lower]()
    elif preset_lower in base_presets:
        config = base_presets[preset_lower]()
    else:
        raise ValueError(
            f"Unknown force_model preset: {preset!r}. Valid: {sorted(base_presets) + ['central_body']}"
        )

    fc = force_config or {}
    if not fc:
        _validate_force_config(config)
        return config

    gravity = _override_gravity(config.gravity, fc.get("gravity"))
    drag = _override_drag(config.drag, fc.get("drag"))
    srp = _override_srp(config.srp, fc.get("srp"))
    third_body = _override_third_body(config.third_body, fc.get("third_body"))
    relativity = fc.get("relativity", config.relativity)
    tides = _override_tides(config.tides, fc.get("tides"))
    frame_transform = _override_frame_transform(config.frame_transform, fc.get("frame_transform"))

    mass = config.mass
    if (drag is not None or srp is not None) and mass is None:
        mass = brahe.ParameterSource.parameter_index(0)

    new_config = brahe.ForceModelConfig(
        gravity=gravity, drag=drag, srp=srp, third_body=third_body,
        relativity=relativity, mass=mass, frame_transform=frame_transform,
    )
    if tides is not None:
        new_config = brahe.ForceModelConfig(
            gravity=gravity, drag=drag, srp=srp, third_body=third_body,
            relativity=relativity, mass=mass, frame_transform=frame_transform,
            tides=tides,
        )
    _validate_force_config(new_config)
    return new_config


def _validate_force_config(config) -> None:
    try:
        config.validate()
    except Exception as e:
        raise ValueError(f"Invalid force model configuration: {e}")


def _override_gravity(current, g: dict | None):
    if not g:
        return current
    if g.get("degree") is None:
        return brahe.GravityConfiguration.point_mass()
    kwargs = {"degree": g["degree"], "order": g.get("order", g["degree"])}
    mt = g.get("model_type")
    if mt is not None:
        kwargs["model_type"] = getattr(brahe.GravityModelType, mt.upper())
    if g.get("use_global") is not None:
        kwargs["use_global"] = bool(g["use_global"])
    return brahe.GravityConfiguration(**kwargs)


def _override_drag(current, d: dict | None):
    if d is None:
        return current
    model_name = str(d.get("model", "")).lower()
    if model_name in ("none", ""):
        return None
    if model_name == "exponential":
        model = brahe.AtmosphericModel.exponential()
    elif model_name in _ATMOSPHERIC_MODELS:
        model = _ATMOSPHERIC_MODELS[model_name]
    else:
        raise ValueError(f"Unknown drag model: {d.get('model')!r}")
    kwargs = dict(
        model=model,
        area=brahe.ParameterSource.parameter_index(1),
        cd=brahe.ParameterSource.parameter_index(2),
    )
    if d.get("body") is not None:
        kwargs["body"] = _resolve_central_body(d["body"])
    return brahe.DragConfiguration(**kwargs)


def _override_srp(current, s: dict | None):
    if s is None:
        return current
    if s.get("enable") is False:
        return None
    eclipse = _ECLIPSE_MODELS[str(s.get("eclipse_model", "conical")).lower()]
    kwargs = dict(
        area=brahe.ParameterSource.parameter_index(3),
        cr=brahe.ParameterSource.parameter_index(4),
        eclipse_model=eclipse,
    )
    bodies = s.get("occulting_bodies")
    if bodies:
        kwargs["occulting_bodies"] = [getattr(brahe.OccultingBody, b.capitalize()) for b in bodies]
    return brahe.SolarRadiationPressureConfiguration(**kwargs)


def _override_third_body(current, t: dict | None):
    if t is None:
        return current
    bodies = t.get("bodies")
    if not bodies:
        return None
    source = _EPHEMERIS_SOURCES[str(t.get("ephemeris_source", "de440s")).lower()]
    entries = []
    for b in bodies:
        if isinstance(b, dict):
            tb = brahe.ThirdBody.Custom(name=b["name"], naif_id=b["naif_id"], gm=b["gm"])
        else:
            tb = _THIRD_BODIES[str(b).lower()]
        entries.append(brahe.ThirdBodyConfiguration(body=tb, ephemeris_source=source))
    return entries


def _override_tides(current, t: dict | None):
    if t is None:
        return current
    if not (t.get("solid") or t.get("ocean") or t.get("permanent")):
        return None
    kwargs = {}
    if t.get("solid"):
        kwargs["solid"] = brahe.SolidTideConfig()
    if t.get("ocean"):
        kwargs["ocean"] = brahe.OceanTideConfig()
    if t.get("permanent"):
        kwargs["permanent"] = brahe.PermanentTideConfig()
    return brahe.TidesConfiguration(**kwargs)


def _override_frame_transform(current, name: str | None):
    if name is None:
        return current
    return {
        "full": brahe.FrameTransformationModel.FULL_EARTH_ROTATION,
        "earth_rotation_only": brahe.FrameTransformationModel.EARTH_ROTATION_ONLY,
    }[name.lower()]


def _build_propagation_config(integrator: dict | None):
    """Build a NumericalPropagationConfig from an integrator options dict."""
    cfg_dict = integrator or {}
    preset = str(cfg_dict.get("preset", "default")).lower()
    if preset == "high_precision":
        config = brahe.NumericalPropagationConfig.high_precision()
    elif preset == "default":
        config = brahe.NumericalPropagationConfig.default()
    else:
        raise ValueError(f"Unknown integrator preset: {preset!r}. Valid: default, high_precision")

    method = cfg_dict.get("method")
    if method is not None:
        key = method.lower()
        if key not in _INTEGRATION_METHODS:
            raise ValueError(
                f"Unknown integrator method: {method!r}. Valid: {sorted(_INTEGRATION_METHODS)}"
            )
        config = config.with_method(_INTEGRATION_METHODS[key])
    if cfg_dict.get("abs_tol") is not None:
        config = config.with_abs_tol(float(cfg_dict["abs_tol"]))
    if cfg_dict.get("rel_tol") is not None:
        config = config.with_rel_tol(float(cfg_dict["rel_tol"]))
    if cfg_dict.get("initial_step") is not None:
        config = config.with_initial_step(float(cfg_dict["initial_step"]))
    if cfg_dict.get("max_step") is not None:
        config = config.with_max_step(float(cfg_dict["max_step"]))
    if cfg_dict.get("store_accelerations") is not None:
        config = config.with_store_accelerations(bool(cfg_dict["store_accelerations"]))
    return config


# ---------------------------------------------------------------------------
# Tool 1: list_propagation_options
# ---------------------------------------------------------------------------

@mcp.tool()
def list_propagation_options() -> dict:
    """List available propagator types, output frames, force model presets, and their options.

    Use this to discover which propagation tool to use and what parameters are available.
    """
    logger.debug("Listing propagation options")
    return {
        "propagator_types": [
            {
                "type": "sgp4",
                "tool": "propagate_sgp4",
                "description": "SGP4/SDP4 propagation from TLE data. Standard model for tracking Earth-orbiting objects.",
                "required_inputs": "tle_line1, tle_line2",
            },
            {
                "type": "keplerian",
                "tool": "propagate_keplerian",
                "description": "Two-body analytical propagation. Fast, no perturbations. Accepts ECI state or Keplerian elements.",
                "required_inputs": "epoch + (state_eci OR elements_koe)",
            },
            {
                "type": "numerical",
                "tool": "propagate_numerical",
                "description": "High-fidelity numerical propagation with configurable force models (gravity, drag, SRP, third-body, relativity).",
                "required_inputs": "epoch + state_eci",
            },
        ],
        "convenience_tools": [
            {
                "tool": "propagate_from_gp_record",
                "description": "Propagate directly from a GP record dict (from celestrak/spacetrack tools). Supports all propagator types.",
            },
        ],
        "output_frames": sorted(VALID_OUTPUT_FRAMES),
        "output_frame_details": {
            "eci": "Earth-Centered Inertial [x,y,z,vx,vy,vz] (m, m/s)",
            "ecef": "Earth-Centered Earth-Fixed [x,y,z,vx,vy,vz] (m, m/s)",
            "gcrf": "Geocentric Celestial Reference Frame [x,y,z,vx,vy,vz] (m, m/s)",
            "itrf": "International Terrestrial Reference Frame [x,y,z,vx,vy,vz] (m, m/s)",
            "eme2000": "Earth Mean Equator J2000 [x,y,z,vx,vy,vz] (m, m/s)",
            "koe_osc": "Osculating Keplerian elements [a,e,i,RAAN,omega,M]",
            "koe_mean": "Mean Keplerian elements [a,e,i,RAAN,omega,M]",
        },
        "central_bodies": ["earth", "moon", "mars", "emb", "ssb", "<naif id>"],
        "output_frame_details_body": {
            "bci": "Body-centered inertial [x,y,z,vx,vy,vz] (non-Earth central bodies)",
            "bcbf": "Body-centered body-fixed [x,y,z,vx,vy,vz] (non-Earth central bodies)",
        },
        "force_model_presets": FORCE_MODEL_PRESETS,
        "force_config_keys": {
            "gravity": "{degree, order, model_type: EGM2008_120|GGM05S|JGM3, use_global}",
            "drag": "{model: harris_priester|nrlmsise00|exponential|none, body}",
            "srp": "{enable, eclipse_model: conical|cylindrical|none, occulting_bodies: [earth,moon,mars]}",
            "third_body": "{bodies: [sun,moon,earth,...|{name,naif_id,gm}], ephemeris_source: low_precision|de440s|de440}",
            "tides": "{solid, ocean, permanent}",
            "relativity": "bool",
            "frame_transform": "full|earth_rotation_only",
        },
        "integrator_keys": {
            "preset": "default|high_precision",
            "method": "dp54|rk4|rkf45|rkf78|rkn1210 (rk4 = fixed step)",
            "abs_tol": "float", "rel_tol": "float",
            "initial_step": "seconds", "max_step": "seconds",
            "store_accelerations": "bool",
        },
        "spacecraft_params_format": "[mass_kg, drag_area_m2, Cd, srp_area_m2, Cr]",
    }


# ---------------------------------------------------------------------------
# Tool 2: propagate_sgp4
# ---------------------------------------------------------------------------

@mcp.tool()
def propagate_sgp4(
    tle_line1: str,
    tle_line2: str,
    target_epoch: str | None = None,
    start_epoch: str | None = None,
    end_epoch: str | None = None,
    step_seconds: float = 60.0,
    output_frame: str = "eci",
    angle_format: str = "degrees",
) -> dict:
    """Propagate a satellite orbit using the SGP4/SDP4 model from TLE data.

    Provide either target_epoch for a single point, or start_epoch + end_epoch for a range.

    Args:
        tle_line1: TLE line 1.
        tle_line2: TLE line 2.
        target_epoch: Single target epoch (ISO string, e.g. "2024-01-01T12:00:00Z").
        start_epoch: Range start epoch (ISO string).
        end_epoch: Range end epoch (ISO string).
        step_seconds: Step size in seconds for range propagation (default 60).
        output_frame: Output coordinate frame (eci, ecef, gcrf, itrf, eme2000, koe_osc, koe_mean).
        angle_format: Angle format for KOE output ("degrees" or "radians").
    """
    # Validate output frame
    frame = _validate_output_frame(output_frame)
    if frame is None:
        return _prop_error(
            f"Unknown output_frame: {output_frame!r}",
            valid_frames=sorted(VALID_OUTPUT_FRAMES),
        )

    # Validate angle format
    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return _prop_error(str(e))

    # Build epoch list
    try:
        epochs = _build_epoch_list(target_epoch, start_epoch, end_epoch, step_seconds)
    except ValueError as e:
        return _prop_error(str(e))

    # Create propagator
    try:
        prop = brahe.SGPPropagator.from_tle(tle_line1, tle_line2, step_size=step_seconds)
    except Exception as e:
        return _prop_error(f"Failed to create SGP4 propagator: {e}")

    # Propagate
    try:
        states = _propagate_and_collect(prop, epochs, frame, angle_fmt)
    except Exception as e:
        logger.error("SGP4 propagation error: {}", e)
        return _prop_error(f"Propagation error: {e}")

    initial_epoch_str = str(prop.epoch)
    logger.debug("SGP4 propagated {} epochs from {}", len(states), initial_epoch_str)
    return _build_result("sgp4", initial_epoch_str, frame, states)


# ---------------------------------------------------------------------------
# Tool 3: propagate_keplerian
# ---------------------------------------------------------------------------

@mcp.tool()
def propagate_keplerian(
    epoch: str,
    state_eci: list[float] | None = None,
    elements_koe: list[float] | None = None,
    input_angle_format: str = "degrees",
    target_epoch: str | None = None,
    start_epoch: str | None = None,
    end_epoch: str | None = None,
    step_seconds: float = 60.0,
    output_frame: str = "eci",
    angle_format: str = "degrees",
) -> dict:
    """Propagate a satellite orbit using two-body (Keplerian) analytical dynamics.

    Initialize with either an ECI Cartesian state vector or Keplerian elements.
    Provide either target_epoch for a single point, or start_epoch + end_epoch for a range.

    Args:
        epoch: Initial epoch (ISO string).
        state_eci: ECI Cartesian state [x,y,z,vx,vy,vz] in meters and m/s.
        elements_koe: Keplerian elements [a,e,i,RAAN,omega,M] (a in meters, angles in input_angle_format).
        input_angle_format: Angle format for input KOE elements ("degrees" or "radians").
        target_epoch: Single target epoch (ISO string).
        start_epoch: Range start epoch (ISO string).
        end_epoch: Range end epoch (ISO string).
        step_seconds: Step size in seconds for range propagation (default 60).
        output_frame: Output coordinate frame (eci, ecef, gcrf, itrf, eme2000, koe_osc, koe_mean).
        angle_format: Angle format for KOE output ("degrees" or "radians").
    """
    # Validate inputs
    if state_eci is None and elements_koe is None:
        return _prop_error(
            "Provide either state_eci (ECI Cartesian [x,y,z,vx,vy,vz]) "
            "or elements_koe (Keplerian [a,e,i,RAAN,omega,M])."
        )

    frame = _validate_output_frame(output_frame)
    if frame is None:
        return _prop_error(
            f"Unknown output_frame: {output_frame!r}",
            valid_frames=sorted(VALID_OUTPUT_FRAMES),
        )

    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return _prop_error(str(e))

    try:
        epochs = _build_epoch_list(target_epoch, start_epoch, end_epoch, step_seconds)
    except ValueError as e:
        return _prop_error(str(e))

    # Parse initial epoch
    try:
        epc0 = parse_epoch(epoch)
    except ValueError as e:
        return _prop_error(f"Invalid epoch: {e}")

    # Create propagator
    try:
        if elements_koe is not None:
            if len(elements_koe) != 6:
                return _prop_error(
                    f"elements_koe must have exactly 6 elements [a,e,i,RAAN,omega,M], got {len(elements_koe)}"
                )
            input_fmt = resolve_angle_format(input_angle_format)
            oe = np.array(elements_koe, dtype=float)
            prop = brahe.KeplerianPropagator.from_keplerian(
                epc0, oe, input_fmt, step_size=step_seconds
            )
        else:
            if len(state_eci) != 6:
                return _prop_error(
                    f"state_eci must have exactly 6 elements [x,y,z,vx,vy,vz], got {len(state_eci)}"
                )
            cart = np.array(state_eci, dtype=float)
            prop = brahe.KeplerianPropagator.from_eci(
                epc0, cart, step_size=step_seconds
            )
    except ValueError as e:
        return _prop_error(str(e))
    except Exception as e:
        return _prop_error(f"Failed to create Keplerian propagator: {e}")

    # Propagate
    try:
        states = _propagate_and_collect(prop, epochs, frame, angle_fmt)
    except Exception as e:
        logger.error("Keplerian propagation error: {}", e)
        return _prop_error(f"Propagation error: {e}")

    initial_epoch_str = str(prop.initial_epoch)
    logger.debug("Keplerian propagated {} epochs from {}", len(states), initial_epoch_str)
    return _build_result("keplerian", initial_epoch_str, frame, states)


# ---------------------------------------------------------------------------
# Tool 4: propagate_numerical
# ---------------------------------------------------------------------------

@mcp.tool()
def propagate_numerical(
    epoch: str,
    state_eci: list[float],
    target_epoch: str | None = None,
    start_epoch: str | None = None,
    end_epoch: str | None = None,
    step_seconds: float = 60.0,
    output_frame: str = "eci",
    angle_format: str = "degrees",
    force_model: str = "default",
    central_body: str = "earth",
    spacecraft_params: list[float] | None = None,
    force_config: dict | None = None,
    integrator: dict | None = None,
) -> dict:
    """Propagate a satellite orbit using high-fidelity numerical integration.

    Use list_propagation_options() for presets, central bodies, output frames,
    and the force_config / integrator dict keys.

    Args:
        epoch: Initial epoch (ISO string).
        state_eci: Cartesian state [x,y,z,vx,vy,vz] (m, m/s) in the central body's inertial frame.
        target_epoch: Single target epoch (ISO string).
        start_epoch: Range start epoch (ISO string).
        end_epoch: Range end epoch (ISO string).
        step_seconds: Step size in seconds for range propagation (default 60).
        output_frame: eci/ecef/gcrf/itrf/eme2000/koe_osc/koe_mean for Earth; bci/bcbf for other bodies.
        angle_format: "degrees" or "radians" for KOE output.
        force_model: Preset (default, two_body, earth_gravity, leo_default, geo_default,
            high_fidelity, conservative_forces, cislunar_default, lunar_default, mars_default, central_body).
        central_body: "earth" (default), "moon", "mars", "emb", "ssb", or a NAIF id.
        spacecraft_params: [mass_kg, drag_area_m2, Cd, srp_area_m2, Cr].
        force_config: Optional overrides dict (gravity/drag/srp/third_body/tides/relativity/frame_transform).
        integrator: Optional integrator dict (preset/method/abs_tol/rel_tol/initial_step/max_step/store_accelerations).
    """
    if len(state_eci) != 6:
        return _prop_error(f"state_eci must have exactly 6 elements, got {len(state_eci)}")

    frame = _validate_output_frame(output_frame)
    if frame is None:
        return _prop_error(f"Unknown output_frame: {output_frame!r}", valid_frames=sorted(VALID_OUTPUT_FRAMES))

    try:
        angle_fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return _prop_error(str(e))

    try:
        epochs = _build_epoch_list(target_epoch, start_epoch, end_epoch, step_seconds)
    except ValueError as e:
        return _prop_error(str(e))

    try:
        epc0 = parse_epoch(epoch)
    except ValueError as e:
        return _prop_error(f"Invalid epoch: {e}")

    try:
        force_cfg = _build_force_config(force_model, central_body, force_config)
    except ValueError as e:
        return _prop_error(str(e), valid_presets=sorted(FORCE_MODEL_PRESETS.keys()))

    try:
        prop_config = _build_propagation_config(integrator)
    except ValueError as e:
        return _prop_error(str(e))

    if force_cfg.requires_params() and spacecraft_params is None:
        return _prop_error(
            f"Force model '{force_model}' requires spacecraft_params: [mass_kg, drag_area_m2, Cd, srp_area_m2, Cr].",
            spacecraft_params_format="[mass_kg, drag_area_m2, Cd, srp_area_m2, Cr]",
        )

    try:
        cart = np.array(state_eci, dtype=float)
        params = np.array(spacecraft_params, dtype=float) if spacecraft_params is not None else None
        prop = brahe.NumericalOrbitPropagator(epc0, cart, prop_config, force_cfg, params)
    except Exception as e:
        return _prop_error(f"Failed to create numerical propagator: {e}")

    try:
        states = _propagate_and_collect(prop, epochs, frame, angle_fmt, needs_propagate_to=True)
    except Exception as e:
        logger.error("Numerical propagation error: {}", e)
        return _prop_error(f"Propagation error: {e}")

    initial_epoch_str = str(prop.initial_epoch)
    logger.debug("Numerical propagated {} epochs from {}", len(states), initial_epoch_str)
    return _build_result("numerical", initial_epoch_str, frame, states)


# ---------------------------------------------------------------------------
# Tool 5: propagate_from_gp_record
# ---------------------------------------------------------------------------

VALID_PROPAGATOR_TYPES = {"sgp4", "keplerian", "numerical"}


@mcp.tool()
def propagate_from_gp_record(
    gp_record: dict,
    propagator_type: str = "sgp4",
    target_epoch: str | None = None,
    start_epoch: str | None = None,
    end_epoch: str | None = None,
    step_seconds: float = 60.0,
    output_frame: str = "eci",
    angle_format: str = "degrees",
    force_model: str = "default",
    central_body: str = "earth",
    spacecraft_params: list[float] | None = None,
    force_config: dict | None = None,
    integrator: dict | None = None,
) -> dict:
    """Propagate from a GP record dict (from celestrak/spacetrack query tools).

    Bridges GP data queries directly into propagation. For SGP4, uses the OMM
    elements directly (avoiding TLE precision loss). For Keplerian/Numerical,
    converts OMM elements to an ECI state.

    Args:
        gp_record: Dict from get_celestrak_gp or query_spacetrack_gp tools.
        propagator_type: "sgp4" (default), "keplerian", or "numerical".
        target_epoch: Single target epoch (ISO string).
        start_epoch: Range start epoch (ISO string).
        end_epoch: Range end epoch (ISO string).
        step_seconds: Step size in seconds for range propagation (default 60).
        output_frame: Output coordinate frame (eci, ecef, gcrf, itrf, eme2000, koe_osc, koe_mean).
        angle_format: Angle format for KOE output ("degrees" or "radians").
        force_model: Force model preset for numerical propagation (default "default").
        central_body: Central body for numerical propagation (numerical only).
        spacecraft_params: [mass_kg, drag_area_m2, Cd, srp_area_m2, Cr] for numerical propagation.
        force_config: Force model overrides dict (numerical only).
        integrator: Integrator options dict (numerical only).
    """
    pt = propagator_type.lower()
    if pt not in VALID_PROPAGATOR_TYPES:
        return _prop_error(
            f"Unknown propagator_type: {propagator_type!r}",
            valid_types=sorted(VALID_PROPAGATOR_TYPES),
        )

    if pt == "sgp4":
        # Use OMM elements directly to avoid TLE precision loss
        frame = _validate_output_frame(output_frame)
        if frame is None:
            return _prop_error(
                f"Unknown output_frame: {output_frame!r}",
                valid_frames=sorted(VALID_OUTPUT_FRAMES),
            )

        try:
            angle_fmt = resolve_angle_format(angle_format)
        except ValueError as e:
            return _prop_error(str(e))

        try:
            epochs = _build_epoch_list(target_epoch, start_epoch, end_epoch, step_seconds)
        except ValueError as e:
            return _prop_error(str(e))

        try:
            prop = _sgp4_from_gp(gp_record, step_size=step_seconds)
        except ValueError as e:
            return _prop_error(str(e))
        except Exception as e:
            return _prop_error(f"Failed to create SGP4 propagator from GP record: {e}")

        try:
            states = _propagate_and_collect(prop, epochs, frame, angle_fmt)
        except Exception as e:
            logger.error("SGP4 propagation error: {}", e)
            return _prop_error(f"Propagation error: {e}")

        initial_epoch_str = str(prop.epoch)
        return _build_result("sgp4", initial_epoch_str, frame, states)

    # For keplerian and numerical, convert GP elements to ECI state
    try:
        state_eci, gp_epoch = _eci_state_from_gp(gp_record)
    except ValueError as e:
        return _prop_error(str(e))

    epoch_str = str(gp_epoch)

    if pt == "keplerian":
        return propagate_keplerian(
            epoch=epoch_str,
            state_eci=state_eci.tolist(),
            target_epoch=target_epoch,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            step_seconds=step_seconds,
            output_frame=output_frame,
            angle_format=angle_format,
        )

    # numerical
    return propagate_numerical(
        epoch=epoch_str,
        state_eci=state_eci.tolist(),
        target_epoch=target_epoch,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        step_seconds=step_seconds,
        output_frame=output_frame,
        angle_format=angle_format,
        force_model=force_model,
        central_body=central_body,
        spacecraft_params=spacecraft_params,
        force_config=force_config,
        integrator=integrator,
    )
