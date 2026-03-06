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
