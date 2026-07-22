import brahe
import numpy as np
from brahe_mcp._gp import _sgp4_from_gp, _eci_state_from_gp

FAKE_GP_RECORD = {
    "object_name": "ISS (ZARYA)",
    "norad_cat_id": 25544,
    "object_id": "1998-067A",
    "epoch": "2024-01-01 12:00:00 UTC",
    "mean_motion": 15.5,
    "eccentricity": 0.0007,
    "inclination": 51.6,
    "ra_of_asc_node": 100.0,
    "arg_of_pericenter": 130.0,
    "mean_anomaly": 230.0,
    "bstar": 0.0001,
    "classification_type": "U",
}


def test_sgp4_from_gp_returns_propagator():
    prop = _sgp4_from_gp(FAKE_GP_RECORD)
    assert isinstance(prop, brahe.SGPPropagator)
    assert prop.norad_id == 25544


def test_eci_state_from_gp_returns_state_and_epoch():
    state, epoch = _eci_state_from_gp(FAKE_GP_RECORD)
    assert isinstance(state, np.ndarray)
    assert state.shape == (6,)
    assert isinstance(epoch, brahe.Epoch)
