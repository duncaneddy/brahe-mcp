"""Shared GP-record → propagator helpers (kept separate to avoid a
propagation.py <-> accesses.py circular import)."""

import numpy as np
import brahe

from brahe_mcp.utils import parse_epoch


def _sgp4_from_gp(gp_record: dict, step_size: float = 60.0) -> brahe.SGPPropagator:
    """Create an SGP propagator from a GP record's OMM elements.

    Uses ``SGPPropagator.from_omm_elements()`` to avoid the precision loss
    inherent in TLE fixed-width formatting. In brahe 1.7.0 the ``epoch``
    argument must be a ``brahe.Epoch`` (previously a string).
    """
    required = [
        "epoch", "mean_motion", "eccentricity", "inclination",
        "ra_of_asc_node", "arg_of_pericenter", "mean_anomaly", "norad_cat_id",
    ]
    missing = [k for k in required if gp_record.get(k) is None]
    if missing:
        raise ValueError(f"GP record missing required OMM fields: {missing}")

    epoch = parse_epoch(gp_record["epoch"])

    return brahe.SGPPropagator.from_omm_elements(
        epoch=epoch,
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
    """Convert a GP record to an osculating ECI state via SGP4 at the TLE epoch."""
    prop = _sgp4_from_gp(gp_record)
    epoch = prop.epoch
    state_eci = prop.state_eci(epoch)
    return state_eci, epoch
