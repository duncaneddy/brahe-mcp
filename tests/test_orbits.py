import math

import brahe
import pytest

from brahe_mcp.orbits import (
    list_orbital_computations,
    compute_orbital_property,
    convert_anomaly,
)


# --- list_orbital_computations ---


def test_list_orbital_computations():
    result = list_orbital_computations()
    assert "orbital_properties" in result
    assert "anomaly_conversions" in result
    assert len(result["orbital_properties"]) > 0
    assert len(result["anomaly_conversions"]) > 0
    # Check structure
    prop = result["orbital_properties"][0]
    assert "name" in prop
    assert "description" in prop
    assert "required_params" in prop
    assert "optional_params" in prop
    assert "input_units" in prop
    assert "output_unit" in prop
    # Check anomaly conversion structure
    conv = result["anomaly_conversions"][0]
    assert "input_units" in conv
    assert "output_unit" in conv


# --- compute_orbital_property ---


def test_orbital_period_earth():
    result = compute_orbital_property("orbital_period", a=7000e3)
    assert result["computation"] == "orbital_period"
    assert abs(result["result"]["value"] - 5828.5) < 1.0
    assert result["result"]["output_unit"] == "s"
    assert "input_units" in result
    assert result["input_units"]["a"] == "m"


def test_orbital_period_general():
    result = compute_orbital_property("orbital_period", a=7000e3, gm=brahe.GM_EARTH)
    assert abs(result["result"]["value"] - 5828.5) < 1.0


def test_orbital_period_from_state():
    a = 7000e3
    v = math.sqrt(brahe.GM_EARTH / a)
    state_str = f"{a},0,0,0,{v},0"
    result = compute_orbital_property("orbital_period_from_state", state_eci=state_str)
    assert abs(result["result"]["value"] - 5828.5) < 1.0


def test_mean_motion_degrees():
    result = compute_orbital_property("mean_motion", a=7000e3)
    assert abs(result["result"]["value"] - 0.0618) < 0.001
    assert result["result"]["output_unit"] == "deg/s"


def test_mean_motion_radians():
    result = compute_orbital_property("mean_motion", a=7000e3, angle_format="radians")
    assert abs(result["result"]["value"] - 0.00108) < 0.0001
    assert result["result"]["output_unit"] == "rad/s"


def test_semimajor_axis_from_mean_motion():
    n = compute_orbital_property("mean_motion", a=7000e3)["result"]["value"]
    result = compute_orbital_property("semimajor_axis", n=n)
    assert abs(result["result"]["value"] - 7000e3) < 1.0


def test_semimajor_axis_from_period():
    period = compute_orbital_property("orbital_period", a=7000e3)["result"]["value"]
    result = compute_orbital_property("semimajor_axis_from_period", period=period)
    assert abs(result["result"]["value"] - 7000e3) < 1.0


def test_periapsis_velocity():
    result = compute_orbital_property("periapsis_velocity", a=7000e3, e=0.001)
    assert result["result"]["value"] > 7000  # m/s, reasonable LEO velocity
    assert result["result"]["output_unit"] == "m/s"


def test_apoapsis_velocity():
    result = compute_orbital_property("apoapsis_velocity", a=7000e3, e=0.001)
    assert result["result"]["value"] > 7000


def test_periapsis_distance():
    result = compute_orbital_property("periapsis_distance", a=7000e3, e=0.001)
    assert result["result"]["value"] == pytest.approx(6993000.0, abs=1.0)


def test_apoapsis_distance():
    result = compute_orbital_property("apoapsis_distance", a=7000e3, e=0.001)
    assert result["result"]["value"] == pytest.approx(7007000.0, abs=1.0)


def test_periapsis_altitude_earth():
    result = compute_orbital_property("periapsis_altitude", a=7000e3, e=0.001)
    # 6993 km - R_EARTH (~6378 km) ≈ 615 km
    assert abs(result["result"]["value"] - 614864) < 100


def test_apoapsis_altitude_earth():
    result = compute_orbital_property("apoapsis_altitude", a=7000e3, e=0.001)
    # 7007 km - R_EARTH (~6378 km) ≈ 629 km
    assert abs(result["result"]["value"] - 628864) < 100


def test_altitude_custom_r_body():
    result = compute_orbital_property(
        "periapsis_altitude", a=7000e3, e=0.001, r_body=brahe.R_EARTH
    )
    earth_result = compute_orbital_property("periapsis_altitude", a=7000e3, e=0.001)
    assert result["result"]["value"] == pytest.approx(
        earth_result["result"]["value"], abs=0.001
    )


def test_sun_synchronous_inclination():
    result = compute_orbital_property(
        "sun_synchronous_inclination", a=7000e3, e=0.001
    )
    assert abs(result["result"]["value"] - 97.87) < 0.1


def test_geo_sma():
    result = compute_orbital_property("geo_sma")
    assert abs(result["result"]["value"] - 42164172) < 10


# --- Error cases ---


def test_invalid_computation():
    result = compute_orbital_property("not_a_computation", a=1.0)
    assert "error" in result
    assert "valid_computations" in result


def test_missing_required_param():
    result = compute_orbital_property("orbital_period")
    assert "error" in result
    assert "Missing required" in result["error"]


def test_invalid_angle_format():
    result = compute_orbital_property("mean_motion", a=7000e3, angle_format="gradians")
    assert "error" in result
    assert "valid_angle_formats" in result


def test_computation_case_insensitive():
    result = compute_orbital_property("Orbital_Period", a=7000e3)
    assert result["computation"] == "orbital_period"
    assert abs(result["result"]["value"] - 5828.5) < 1.0


# --- convert_anomaly ---


def test_eccentric_to_mean():
    result = convert_anomaly("eccentric_to_mean", anomaly=45.0, e=0.1)
    assert abs(result["output"]["anomaly"] - 40.95) < 0.1


def test_mean_to_eccentric():
    # Inverse of eccentric_to_mean
    fwd = convert_anomaly("eccentric_to_mean", anomaly=45.0, e=0.1)
    rev = convert_anomaly(
        "mean_to_eccentric", anomaly=fwd["output"]["anomaly"], e=0.1
    )
    assert abs(rev["output"]["anomaly"] - 45.0) < 0.01


def test_true_to_eccentric():
    result = convert_anomaly("true_to_eccentric", anomaly=60.0, e=0.1)
    assert "output" in result
    assert isinstance(result["output"]["anomaly"], float)


def test_eccentric_to_true():
    fwd = convert_anomaly("true_to_eccentric", anomaly=60.0, e=0.1)
    rev = convert_anomaly(
        "eccentric_to_true", anomaly=fwd["output"]["anomaly"], e=0.1
    )
    assert abs(rev["output"]["anomaly"] - 60.0) < 0.01


def test_true_to_mean():
    result = convert_anomaly("true_to_mean", anomaly=60.0, e=0.1)
    assert "output" in result
    assert isinstance(result["output"]["anomaly"], float)


def test_mean_to_true():
    fwd = convert_anomaly("true_to_mean", anomaly=60.0, e=0.1)
    rev = convert_anomaly("mean_to_true", anomaly=fwd["output"]["anomaly"], e=0.1)
    assert abs(rev["output"]["anomaly"] - 60.0) < 0.01


def test_anomaly_radians():
    result = convert_anomaly(
        "eccentric_to_mean", anomaly=math.radians(45.0), e=0.1, angle_format="radians"
    )
    assert result["output"]["angle_format"] == "radians"
    expected = math.radians(40.95)
    assert abs(result["output"]["anomaly"] - expected) < 0.01


def test_invalid_conversion():
    result = convert_anomaly("bad_conversion", anomaly=45.0, e=0.1)
    assert "error" in result
    assert "valid_conversions" in result


def test_anomaly_roundtrip():
    """true -> eccentric -> mean -> true should return to original."""
    true_anom = 75.0
    e = 0.15

    r1 = convert_anomaly("true_to_eccentric", anomaly=true_anom, e=e)
    ecc_anom = r1["output"]["anomaly"]

    r2 = convert_anomaly("eccentric_to_mean", anomaly=ecc_anom, e=e)
    mean_anom = r2["output"]["anomaly"]

    r3 = convert_anomaly("mean_to_true", anomaly=mean_anom, e=e)
    assert abs(r3["output"]["anomaly"] - true_anom) < 0.01


# --- convert_equinoctial ---

import numpy as np
from brahe_mcp.orbits import convert_equinoctial

KOE = [brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0]


def test_koe_to_equinoctial_matches_brahe():
    res = convert_equinoctial(KOE, "koe_to_equinoctial")
    assert "error" not in res
    expected = brahe.state_koe_to_equinoctial(
        np.array(KOE), brahe.AngleFormat.DEGREES, 1
    )
    assert np.allclose(res["output"]["state"], expected)
    assert list(res["output"]["components"]) == ["a_m", "h", "k", "p", "q", "l"]


def test_equinoctial_roundtrip():
    fwd = convert_equinoctial(KOE, "koe_to_equinoctial")
    back = convert_equinoctial(fwd["output"]["state"], "equinoctial_to_koe")
    assert np.allclose(back["output"]["state"], KOE, rtol=1e-9)


def test_equinoctial_retrograde_roundtrip():
    retro = [brahe.R_EARTH + 500e3, 0.01, 175.0, 30.0, 60.0, 90.0]
    fwd = convert_equinoctial(retro, "koe_to_equinoctial", fr=-1)
    back = convert_equinoctial(fwd["output"]["state"], "equinoctial_to_koe", fr=-1)
    assert np.allclose(back["output"]["state"], retro, rtol=1e-9)


def test_equinoctial_invalid_fr():
    res = convert_equinoctial(KOE, "koe_to_equinoctial", fr=0)
    assert "error" in res
    assert "fr" in res["error"]


def test_equinoctial_invalid_direction():
    res = convert_equinoctial(KOE, "sideways")
    assert "error" in res
    assert "valid_directions" in res


def test_equinoctial_bad_length():
    res = convert_equinoctial([1.0, 2.0, 3.0], "koe_to_equinoctial")
    assert "error" in res


def test_equinoctial_radians_matches_degrees():
    koe_rad = [KOE[0], KOE[1]] + [np.radians(v) for v in KOE[2:]]
    deg = convert_equinoctial(KOE, "koe_to_equinoctial")
    rad = convert_equinoctial(koe_rad, "koe_to_equinoctial", angle_format="radians")
    # Only l (index 5) is angular.
    assert np.allclose(deg["output"]["state"][:5], rad["output"]["state"][:5])
    assert np.isclose(np.radians(deg["output"]["state"][5]), rad["output"]["state"][5])


# --- convert_mean_osculating ---

from brahe_mcp.orbits import convert_mean_osculating


def test_mean_to_osc_matches_brahe():
    res = convert_mean_osculating(KOE, "mean_to_osc")
    assert "error" not in res
    expected = brahe.state_koe_mean_to_osc(
        np.array(KOE), brahe.MeanElementMethod.BROUWER_LYDDANE,
        brahe.AngleFormat.DEGREES,
    )
    assert np.allclose(res["output"]["state"], expected)
    assert res["output"]["method"] == "brouwer_lyddane"


def test_osc_to_mean_matches_brahe():
    res = convert_mean_osculating(KOE, "osc_to_mean")
    expected = brahe.state_koe_osc_to_mean(
        np.array(KOE), brahe.MeanElementMethod.BROUWER_LYDDANE,
        brahe.AngleFormat.DEGREES,
    )
    assert np.allclose(res["output"]["state"], expected)


def test_mean_osc_is_not_a_noop():
    res = convert_mean_osculating(KOE, "mean_to_osc")
    assert not np.allclose(res["output"]["state"], KOE)


def test_mean_osc_roundtrip_within_bl_theory_error():
    # Brouwer-Lyddane is a first-order theory, so mean -> osc -> mean is NOT
    # exact. Tolerances below are sized from measured worst-case residuals
    # across three test orbits (see spec section 6). Do not tighten these:
    # correct code fails a strict np.allclose here.
    fwd = convert_mean_osculating(KOE, "mean_to_osc")
    back = convert_mean_osculating(fwd["output"]["state"], "osc_to_mean")
    got = np.array(back["output"]["state"])
    ref = np.array(KOE)
    assert abs(got[0] - ref[0]) < 50.0        # a, meters
    assert abs(got[1] - ref[1]) < 1e-5        # e
    d_ang = (got[2:] - ref[2:] + 180.0) % 360.0 - 180.0
    assert np.all(np.abs(d_ang) < 1.0)        # i, RAAN, omega, M in degrees


def test_mean_osc_numerical_rejected_with_pointer_to_batch():
    res = convert_mean_osculating(KOE, "mean_to_osc", method="numerical")
    assert "error" in res
    assert "batch" in res["error"].lower()


def test_mean_osc_invalid_direction():
    res = convert_mean_osculating(KOE, "sideways")
    assert "error" in res
    assert "valid_directions" in res


def test_mean_osc_bad_length():
    res = convert_mean_osculating([1.0, 2.0], "mean_to_osc")
    assert "error" in res


# --- convert_mean_osculating_batch ---

from brahe_mcp.orbits import convert_mean_osculating_batch

E0 = "2024-01-01T00:00:00Z"


def _series(n=100, step=60.0):
    e0 = brahe.Epoch(E0)
    epochs = [str(e0 + step * i) for i in range(n)]
    states = [list(KOE) for _ in range(n)]
    return epochs, states


def test_batch_brouwer_lyddane_preserves_length():
    epochs, states = _series(10)
    res = convert_mean_osculating_batch(epochs, states, "mean_to_osc")
    assert "error" not in res
    assert res["output"]["n_input"] == 10
    assert res["output"]["n_output"] == 10
    assert res["output"]["dropped_by_edge_handling"] == 0
    assert len(res["output"]["states"]) == 10


def test_batch_bl_matches_single_state_tool():
    epochs, states = _series(3)
    batch = convert_mean_osculating_batch(epochs, states, "mean_to_osc")
    single = convert_mean_osculating(KOE, "mean_to_osc")
    assert np.allclose(batch["output"]["states"][0], single["output"]["state"])


def test_batch_numerical_osc_to_mean_shortens_series():
    epochs, states = _series(100)
    res = convert_mean_osculating_batch(
        epochs, states, "osc_to_mean", method="numerical", window_seconds=5400.0
    )
    assert "error" not in res
    assert res["output"]["n_input"] == 100
    # Windowed averaging with truncation consumes the series edges.
    assert res["output"]["n_output"] < 100
    assert res["output"]["dropped_by_edge_handling"] == (
        100 - res["output"]["n_output"]
    )
    assert len(res["output"]["epochs"]) == res["output"]["n_output"]


def test_batch_numerical_mean_to_osc_requires_force_config():
    epochs, states = _series(20)
    res = convert_mean_osculating_batch(
        epochs, states, "mean_to_osc", method="numerical"
    )
    assert "error" in res
    assert "force_config" in res["error"]


def test_batch_numerical_mean_to_osc_succeeds_with_force_config():
    """Happy path for the heaviest code path. Verified to run in ~0.2s."""
    epochs, states = _series(20)
    res = convert_mean_osculating_batch(
        epochs, states, "mean_to_osc",
        method="numerical", force_config={}, force_model="earth_gravity",
    )
    assert "error" not in res
    # mean_to_osc preserves length; only osc_to_mean truncates.
    assert res["output"]["n_output"] == 20
    assert res["output"]["dropped_by_edge_handling"] == 0
    assert not np.allclose(res["output"]["states"][0], KOE)


def test_batch_length_mismatch_errors():
    epochs, states = _series(5)
    res = convert_mean_osculating_batch(epochs[:3], states, "mean_to_osc")
    assert "error" in res
    assert "3" in res["error"] and "5" in res["error"]


def test_batch_bad_row_length_errors():
    epochs, states = _series(3)
    states[1] = [1.0, 2.0]
    res = convert_mean_osculating_batch(epochs, states, "mean_to_osc")
    assert "error" in res
    assert "1" in res["error"]


def test_batch_invalid_alignment_errors():
    epochs, states = _series(3)
    res = convert_mean_osculating_batch(
        epochs, states, "osc_to_mean", method="numerical", alignment="sideways"
    )
    assert "error" in res
    assert "valid_alignments" in res


def test_batch_invalid_edge_errors():
    epochs, states = _series(3)
    res = convert_mean_osculating_batch(
        epochs, states, "osc_to_mean", method="numerical", edge="nope"
    )
    assert "error" in res
    assert "valid_edges" in res


def test_batch_empty_input_errors():
    res = convert_mean_osculating_batch([], [], "mean_to_osc")
    assert "error" in res


def test_list_orbital_computations_includes_mean_elements():
    opts = list_orbital_computations()
    mec = opts["mean_element_conversions"]
    assert "convert_mean_osculating_batch" in mec["tools"]
    assert "numerical" in mec["methods"]
    assert mec["equinoctial_components"] == ["a_m", "h", "k", "p", "q", "l"]
