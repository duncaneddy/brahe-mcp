"""Orbit propagation MCP tools using brahe propagators."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch, resolve_angle_format

# Ensure EOP data is loaded for frame transforms
brahe.initialize_eop()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_OUTPUT_FRAMES = {"eci", "ecef", "gcrf", "itrf", "eme2000", "koe_osc", "koe_mean"}

FORCE_MODEL_PRESETS = {
    "default": "Default: 20x20 EGM2008 gravity, Harris-Priester drag, SRP with conical eclipse, Sun/Moon third-body. Requires spacecraft_params.",
    "two_body": "Point-mass gravity only. No spacecraft_params needed.",
    "earth_gravity": "20x20 EGM2008 gravity only. No drag/SRP/third-body. No spacecraft_params needed.",
    "leo_default": "Optimized for LEO orbits. Requires spacecraft_params.",
    "geo_default": "Optimized for GEO orbits. Requires spacecraft_params.",
    "high_fidelity": "High-fidelity force model. Requires spacecraft_params.",
    "conservative_forces": "Gravity + third-body + relativity. No drag/SRP. No spacecraft_params needed.",
}

VALID_DRAG_MODELS = {"harris_priester", "nrlmsise00", "none"}

# Output component labels by frame
OUTPUT_LABELS = {
    "eci": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "ecef": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "gcrf": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "itrf": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "eme2000": ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
    "koe_osc": ("a_m", "e", "i", "RAAN", "omega", "M"),
    "koe_mean": ("a_m", "e", "i", "RAAN", "omega", "M"),
}

MAX_EPOCHS = 100_000

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


def _build_force_config(
    preset: str,
    gravity_degree: int | None = None,
    gravity_order: int | None = None,
    drag_model: str | None = None,
    enable_srp: bool | None = None,
    enable_third_body: bool | None = None,
    enable_relativity: bool | None = None,
) -> brahe.ForceModelConfig:
    """Build a ForceModelConfig from a preset name with optional granular overrides.

    Args:
        preset: Preset name (e.g. "default", "two_body").
        gravity_degree: Override gravity degree.
        gravity_order: Override gravity order.
        drag_model: Override drag model ("harris_priester", "nrlmsise00", "none").
        enable_srp: Override SRP toggle.
        enable_third_body: Override third-body toggle.
        enable_relativity: Override relativity toggle.

    Returns:
        ForceModelConfig instance.

    Raises:
        ValueError: If preset or drag_model is invalid.
    """
    preset_lower = preset.lower()

    # Get base config from preset
    preset_dispatch = {
        "default": brahe.ForceModelConfig.default,
        "two_body": brahe.ForceModelConfig.two_body,
        "earth_gravity": brahe.ForceModelConfig.earth_gravity,
        "leo_default": brahe.ForceModelConfig.leo_default,
        "geo_default": brahe.ForceModelConfig.geo_default,
        "high_fidelity": brahe.ForceModelConfig.high_fidelity,
        "conservative_forces": brahe.ForceModelConfig.conservative_forces,
    }

    if preset_lower not in preset_dispatch:
        raise ValueError(
            f"Unknown force_model preset: {preset!r}. "
            f"Valid presets: {sorted(preset_dispatch.keys())}"
        )

    config = preset_dispatch[preset_lower]()

    # Check if any granular overrides were provided
    has_overrides = any(
        v is not None
        for v in [gravity_degree, gravity_order, drag_model, enable_srp,
                  enable_third_body, enable_relativity]
    )

    if not has_overrides:
        return config

    # Build custom config with overrides
    # Gravity
    gravity = config.gravity
    if gravity_degree is not None or gravity_order is not None:
        deg = gravity_degree if gravity_degree is not None else 20
        order = gravity_order if gravity_order is not None else deg
        gravity = brahe.GravityConfiguration(degree=deg, order=order)

    # Drag
    drag = config.drag
    if drag_model is not None:
        dm = drag_model.lower()
        if dm not in VALID_DRAG_MODELS:
            raise ValueError(
                f"Unknown drag_model: {drag_model!r}. Valid: {sorted(VALID_DRAG_MODELS)}"
            )
        if dm == "none":
            drag = None
        else:
            atm_model = {
                "harris_priester": brahe.AtmosphericModel.HARRIS_PRIESTER,
                "nrlmsise00": brahe.AtmosphericModel.NRLMSISE00,
            }[dm]
            drag = brahe.DragConfiguration(
                model=atm_model,
                area=brahe.ParameterSource.parameter_index(1),
                cd=brahe.ParameterSource.parameter_index(2),
            )

    # SRP
    srp = config.srp
    if enable_srp is not None:
        if not enable_srp:
            srp = None
        elif srp is None:
            srp = brahe.SolarRadiationPressureConfiguration(
                area=brahe.ParameterSource.parameter_index(3),
                cr=brahe.ParameterSource.parameter_index(4),
            )

    # Third body
    third_body = config.third_body
    if enable_third_body is not None:
        if not enable_third_body:
            third_body = None
        elif third_body is None:
            third_body = brahe.ThirdBodyConfiguration()

    # Relativity
    relativity = config.relativity
    if enable_relativity is not None:
        relativity = enable_relativity

    # Include mass parameter source when drag or SRP requires it
    mass = config.mass if hasattr(config, 'mass') else None
    if (drag is not None or srp is not None) and mass is None:
        mass = brahe.ParameterSource.parameter_index(0)

    return brahe.ForceModelConfig(
        gravity=gravity,
        drag=drag,
        srp=srp,
        third_body=third_body,
        relativity=relativity,
        mass=mass,
    )


def _sgp4_from_gp(gp_record: dict, step_size: float = 60.0) -> brahe.SGPPropagator:
    """Create an SGP propagator from a GP record's OMM elements.

    Uses ``SGPPropagator.from_omm_elements()`` to avoid the precision loss
    inherent in TLE fixed-width formatting.  The GP record already carries
    full-precision OMM fields from the CelesTrak / SpaceTrack API.

    Args:
        gp_record: Dict from serialize_gp_record().
        step_size: Propagation step size in seconds.

    Returns:
        SGPPropagator instance.

    Raises:
        ValueError: If required OMM fields are missing from the GP record.
    """
    required = [
        "epoch", "mean_motion", "eccentricity", "inclination",
        "ra_of_asc_node", "arg_of_pericenter", "mean_anomaly", "norad_cat_id",
    ]
    missing = [k for k in required if gp_record.get(k) is None]
    if missing:
        raise ValueError(f"GP record missing required OMM fields: {missing}")

    # from_omm_elements expects bare ISO 8601 (e.g. "2024-01-01T12:00:00")
    # GP records may have " UTC", "Z", or space separators
    epoch_str = gp_record["epoch"].strip()
    for suffix in (" UTC", "Z"):
        if epoch_str.endswith(suffix):
            epoch_str = epoch_str[: -len(suffix)]
    epoch_str = epoch_str.replace(" ", "T", 1)

    return brahe.SGPPropagator.from_omm_elements(
        epoch=epoch_str,
        mean_motion=gp_record["mean_motion"],
        eccentricity=gp_record["eccentricity"],
        inclination=gp_record["inclination"],
        raan=gp_record["ra_of_asc_node"],
        arg_of_pericenter=gp_record["arg_of_pericenter"],
        mean_anomaly=gp_record["mean_anomaly"],
        norad_id=gp_record["norad_cat_id"],
        step_size=step_size,
        object_name=gp_record.get("object_name"),
        object_id=gp_record.get("object_id"),
        classification=gp_record.get("classification_type"),
        bstar=gp_record.get("bstar"),
    )


def _eci_state_from_gp(gp_record: dict) -> tuple[np.ndarray, brahe.Epoch]:
    """Convert GP record to an osculating ECI state vector via SGP4.

    GP records contain mean elements fit to the SGP4 theory, so converting
    them directly with ``state_koe_to_eci()`` produces a theory-inconsistent
    state.  Instead, we initialize an SGP4 propagator with the OMM elements
    and evaluate ``state_eci()`` at the TLE epoch to obtain a physically
    consistent osculating Cartesian state.

    Args:
        gp_record: Dict from serialize_gp_record().

    Returns:
        Tuple of (state_eci, epoch).

    Raises:
        ValueError: If required OMM fields are missing.
    """
    prop = _sgp4_from_gp(gp_record)
    epoch = prop.epoch
    state_eci = prop.state_eci(epoch)
    return state_eci, epoch


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
        "force_model_presets": FORCE_MODEL_PRESETS,
        "force_model_overrides": {
            "gravity_degree": "int - Override spherical harmonic degree",
            "gravity_order": "int - Override spherical harmonic order",
            "drag_model": "str - 'harris_priester', 'nrlmsise00', or 'none'",
            "enable_srp": "bool - Toggle solar radiation pressure",
            "enable_third_body": "bool - Toggle Sun/Moon perturbations",
            "enable_relativity": "bool - Toggle relativistic corrections",
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
    spacecraft_params: list[float] | None = None,
    gravity_degree: int | None = None,
    gravity_order: int | None = None,
    drag_model: str | None = None,
    enable_srp: bool | None = None,
    enable_third_body: bool | None = None,
    enable_relativity: bool | None = None,
) -> dict:
    """Propagate a satellite orbit using high-fidelity numerical integration.

    Uses configurable force models: gravity, atmospheric drag, solar radiation pressure,
    third-body perturbations, and relativistic corrections.

    Use list_propagation_options() to see available force model presets and overrides.

    Args:
        epoch: Initial epoch (ISO string).
        state_eci: ECI Cartesian state [x,y,z,vx,vy,vz] in meters and m/s.
        target_epoch: Single target epoch (ISO string).
        start_epoch: Range start epoch (ISO string).
        end_epoch: Range end epoch (ISO string).
        step_seconds: Step size in seconds for range propagation (default 60).
        output_frame: Output coordinate frame (eci, ecef, gcrf, itrf, eme2000, koe_osc, koe_mean).
        angle_format: Angle format for KOE output ("degrees" or "radians").
        force_model: Force model preset name (default "default").
        spacecraft_params: [mass_kg, drag_area_m2, Cd, srp_area_m2, Cr]. Required for most presets.
        gravity_degree: Override gravity spherical harmonic degree.
        gravity_order: Override gravity spherical harmonic order.
        drag_model: Override drag model ("harris_priester", "nrlmsise00", "none").
        enable_srp: Override solar radiation pressure toggle.
        enable_third_body: Override third-body perturbations toggle.
        enable_relativity: Override relativistic corrections toggle.
    """
    # Validate state
    if len(state_eci) != 6:
        return _prop_error(
            f"state_eci must have exactly 6 elements [x,y,z,vx,vy,vz], got {len(state_eci)}"
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

    try:
        epc0 = parse_epoch(epoch)
    except ValueError as e:
        return _prop_error(f"Invalid epoch: {e}")

    # Build force model config
    try:
        force_config = _build_force_config(
            force_model,
            gravity_degree=gravity_degree,
            gravity_order=gravity_order,
            drag_model=drag_model,
            enable_srp=enable_srp,
            enable_third_body=enable_third_body,
            enable_relativity=enable_relativity,
        )
    except ValueError as e:
        return _prop_error(str(e), valid_presets=sorted(FORCE_MODEL_PRESETS.keys()))

    # Check if params are required
    if force_config.requires_params() and spacecraft_params is None:
        return _prop_error(
            f"Force model '{force_model}' requires spacecraft_params: [mass_kg, drag_area_m2, Cd, srp_area_m2, Cr].",
            spacecraft_params_format="[mass_kg, drag_area_m2, Cd, srp_area_m2, Cr]",
        )

    # Create propagator
    try:
        cart = np.array(state_eci, dtype=float)
        params = np.array(spacecraft_params, dtype=float) if spacecraft_params is not None else None
        prop_config = brahe.NumericalPropagationConfig.default()
        prop = brahe.NumericalOrbitPropagator(
            epc0, cart, prop_config, force_config, params
        )
    except Exception as e:
        return _prop_error(f"Failed to create numerical propagator: {e}")

    # Propagate (numerical propagator requires propagate_to before state queries)
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
    spacecraft_params: list[float] | None = None,
    gravity_degree: int | None = None,
    gravity_order: int | None = None,
    drag_model: str | None = None,
    enable_srp: bool | None = None,
    enable_third_body: bool | None = None,
    enable_relativity: bool | None = None,
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
        spacecraft_params: [mass_kg, drag_area_m2, Cd, srp_area_m2, Cr] for numerical propagation.
        gravity_degree: Override gravity degree (numerical only).
        gravity_order: Override gravity order (numerical only).
        drag_model: Override drag model (numerical only).
        enable_srp: Override SRP toggle (numerical only).
        enable_third_body: Override third-body toggle (numerical only).
        enable_relativity: Override relativity toggle (numerical only).
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
        spacecraft_params=spacecraft_params,
        gravity_degree=gravity_degree,
        gravity_order=gravity_order,
        drag_model=drag_model,
        enable_srp=enable_srp,
        enable_third_body=enable_third_body,
        enable_relativity=enable_relativity,
    )
