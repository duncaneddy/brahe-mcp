import numpy as np
import brahe

from brahe_mcp.relative_motion import (
    compute_rtn_rotation,
    convert_roe_state,
    convert_rtn_state,
    list_relative_motion_options,
)

CHIEF = [6878136.3, 0.0, 0.0, 0.0, 7613.0, 0.0]
DEPUTY = [6878136.3, 100.0, 50.0, 0.0, 7613.0, 1.0]

CHIEF_KOE = [6878136.3, 0.001, 51.6, 30.0, 45.0, 10.0]
DEPUTY_KOE = [6878136.3, 0.001, 51.601, 30.002, 45.0, 10.001]

# CHIEF/DEPUTY above are exactly equatorial (i=0), which is fine for
# eci_to_roe (a forward multiply by sin(i)) but roe_to_eci must divide by
# sin(i) to recover RAAN, which is singular at i=0 and returns NaN. Use an
# inclined pair (CHIEF_KOE converted to ECI, deputy offset the same way as
# DEPUTY is offset from CHIEF) for the ECI ROE roundtrip test.
CHIEF_ECI_INCLINED = [
    1662979.016176783,
    4998224.560295982,
    4412242.022992976,
    -6763.505652730542,
    -770.8402564930082,
    3424.4499458966766,
]
DEPUTY_ECI_INCLINED = [
    1662979.016176783,
    4998324.560295982,
    4412292.022992976,
    -6763.505652730542,
    -770.8402564930082,
    3425.4499458966766,
]

# Equatorial (i=0) Keplerian chief/deputy pair, for exercising the same
# roe_to_oe singularity as CHIEF/CHIEF_ECI_INCLINED above but on the koe path.
CHIEF_KOE_EQUATORIAL = [6878136.3, 0.001, 0.0, 30.0, 45.0, 10.0]
DEPUTY_KOE_EQUATORIAL = [6878136.3, 0.001, 0.0, 30.002, 45.0, 10.001]


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


def test_eci_to_roe_matches_brahe():
    res = convert_roe_state(CHIEF, DEPUTY, "eci_to_roe")
    assert "error" not in res
    expected = brahe.state_eci_to_roe(
        np.array(CHIEF), np.array(DEPUTY), brahe.AngleFormat.DEGREES
    )
    assert np.allclose(res["output"]["state"], expected)
    assert list(res["output"]["components"]) == list(
        ("da", "dlambda", "dex", "dey", "dix", "diy")
    )


def test_eci_roe_roundtrip():
    fwd = convert_roe_state(CHIEF_ECI_INCLINED, DEPUTY_ECI_INCLINED, "eci_to_roe")
    back = convert_roe_state(
        CHIEF_ECI_INCLINED, fwd["output"]["state"], "roe_to_eci"
    )
    assert np.allclose(
        back["output"]["state"], DEPUTY_ECI_INCLINED, rtol=1e-6, atol=1e-3
    )


def test_oe_roe_roundtrip():
    fwd = convert_roe_state(
        CHIEF_KOE, DEPUTY_KOE, "oe_to_roe", chief_type="koe"
    )
    back = convert_roe_state(
        CHIEF_KOE, fwd["output"]["state"], "roe_to_oe", chief_type="koe"
    )
    assert np.allclose(back["output"]["state"], DEPUTY_KOE, rtol=1e-6, atol=1e-6)


def test_roe_to_eci_equatorial_chief_errors():
    # RAAN is undefined for an equatorial chief, so state_roe_to_eci returns
    # NaNs; convert_roe_state must surface that as an error, not a "success"
    # envelope containing a non-finite state.
    fwd = convert_roe_state(CHIEF, DEPUTY, "eci_to_roe")
    assert "error" not in fwd
    res = convert_roe_state(CHIEF, fwd["output"]["state"], "roe_to_eci")
    assert "error" in res
    assert "output" not in res


def test_roe_to_oe_equatorial_chief_errors():
    # Same singularity as above, but on the koe (state_roe_to_oe) path.
    fwd = convert_roe_state(
        CHIEF_KOE_EQUATORIAL,
        DEPUTY_KOE_EQUATORIAL,
        "oe_to_roe",
        chief_type="koe",
    )
    assert "error" not in fwd
    res = convert_roe_state(
        CHIEF_KOE_EQUATORIAL,
        fwd["output"]["state"],
        "roe_to_oe",
        chief_type="koe",
    )
    assert "error" in res
    assert "output" not in res


def test_roe_chief_type_mismatch_errors():
    res = convert_roe_state(CHIEF, DEPUTY, "eci_to_roe", chief_type="koe")
    assert "error" in res
    assert "chief_type" in res["error"]


def test_roe_oe_direction_requires_koe_chief():
    res = convert_roe_state(
        CHIEF_KOE, DEPUTY_KOE, "oe_to_roe", chief_type="eci"
    )
    assert "error" in res
    assert "chief_type" in res["error"]


def test_roe_invalid_direction():
    res = convert_roe_state(CHIEF, DEPUTY, "sideways")
    assert "error" in res
    assert "valid_directions" in res


def test_roe_radians_matches_degrees():
    deg = convert_roe_state(CHIEF, DEPUTY, "eci_to_roe")
    rad = convert_roe_state(CHIEF, DEPUTY, "eci_to_roe", angle_format="radians")
    d = np.array(deg["output"]["state"])
    r = np.array(rad["output"]["state"])
    # da, dex, dey (indices 0, 2, 3) are dimensionless; the rest are angular.
    assert np.allclose(d[[0, 2, 3]], r[[0, 2, 3]])
    assert np.allclose(np.radians(d[[1, 4, 5]]), r[[1, 4, 5]])


def test_roe_bad_length_errors():
    res = convert_roe_state(CHIEF, [1.0, 2.0], "eci_to_roe")
    assert "error" in res


def test_roe_non_numeric_input_errors():
    res = convert_roe_state(["a", "b", "c", "d", "e", "f"], DEPUTY, "eci_to_roe")
    assert "error" in res
