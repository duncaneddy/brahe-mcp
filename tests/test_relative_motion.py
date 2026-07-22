import numpy as np
import brahe

from brahe_mcp.relative_motion import (
    compute_rtn_rotation,
    convert_rtn_state,
    list_relative_motion_options,
)

CHIEF = [6878136.3, 0.0, 0.0, 0.0, 7613.0, 0.0]
DEPUTY = [6878136.3, 100.0, 50.0, 0.0, 7613.0, 1.0]


def test_list_options_documents_roe_elements():
    opts = list_relative_motion_options()
    assert "eci_to_rtn" in opts["rtn_directions"]
    assert opts["roe_components"][0] == "da"
    assert "dlambda" in opts["roe_components"]
    assert "units" in opts


def test_eci_to_rtn_matches_brahe():
    res = convert_rtn_state(CHIEF, DEPUTY, "eci_to_rtn")
    assert "error" not in res
    expected = brahe.state_eci_to_rtn(np.array(CHIEF), np.array(DEPUTY))
    assert np.allclose(res["output"]["state"], expected)


def test_rtn_roundtrip():
    fwd = convert_rtn_state(CHIEF, DEPUTY, "eci_to_rtn")
    back = convert_rtn_state(CHIEF, fwd["output"]["state"], "rtn_to_eci")
    assert np.allclose(back["output"]["state"], DEPUTY, rtol=1e-9, atol=1e-6)


def test_rtn_rotation_is_orthonormal():
    res = compute_rtn_rotation(CHIEF, "eci_to_rtn")
    assert "error" not in res
    m = np.array(res["output"]["matrix"])
    assert m.shape == (3, 3)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)


def test_rtn_rotation_directions_are_transposes():
    a = np.array(compute_rtn_rotation(CHIEF, "eci_to_rtn")["output"]["matrix"])
    b = np.array(compute_rtn_rotation(CHIEF, "rtn_to_eci")["output"]["matrix"])
    assert np.allclose(a, b.T, atol=1e-12)


def test_rtn_invalid_direction():
    res = convert_rtn_state(CHIEF, DEPUTY, "sideways")
    assert "error" in res
    assert "valid_directions" in res


def test_rtn_bad_chief_length():
    res = convert_rtn_state([1.0, 2.0], DEPUTY, "eci_to_rtn")
    assert "error" in res


def test_rtn_bad_vector_length():
    res = convert_rtn_state(CHIEF, [1.0, 2.0], "eci_to_rtn")
    assert "error" in res
