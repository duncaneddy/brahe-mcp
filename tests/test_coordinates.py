import pytest

from brahe_mcp.coordinates import (
    list_coordinate_systems,
    convert_position,
    convert_state,
    convert_relative_position,
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_list_coordinate_systems():
    result = list_coordinate_systems()
    assert "position_frames" in result
    assert "state_frames" in result
    assert "relative_frames" in result
    assert "position_conversions" in result
    pos_names = [f["name"] for f in result["position_frames"]]
    for name in ["ECEF", "GEODETIC", "GEOCENTRIC", "ECI", "GCRF", "ITRF", "EME2000"]:
        assert name in pos_names


# ---------------------------------------------------------------------------
# Position conversions — geographic
# ---------------------------------------------------------------------------

def test_geodetic_to_ecef():
    result = convert_position([-122.4, 37.8, 0.0], "GEODETIC", "ECEF")
    assert "output" in result
    vec = result["output"]["vector"]
    assert len(vec) == 3
    assert result["output"]["frame"] == "ECEF"
    # San Francisco area: x < 0, y < 0, z > 0
    assert vec[0] < 0
    assert vec[1] < 0
    assert vec[2] > 0


def test_ecef_to_geodetic():
    result = convert_position([-2703817.25, -4260534.25, 3887927.17], "ECEF", "GEODETIC")
    assert "output" in result
    vec = result["output"]["vector"]
    assert abs(vec[0] - (-122.4)) < 0.01  # lon
    assert abs(vec[1] - 37.8) < 0.01      # lat


def test_geodetic_ecef_roundtrip():
    geod = [-122.4, 37.8, 100.0]
    r1 = convert_position(geod, "GEODETIC", "ECEF")
    r2 = convert_position(r1["output"]["vector"], "ECEF", "GEODETIC")
    for a, b in zip(geod, r2["output"]["vector"]):
        assert abs(a - b) < 1e-6


def test_geocentric_to_ecef():
    result = convert_position([-122.4, 37.6, 6371000.0], "GEOCENTRIC", "ECEF")
    assert "output" in result
    assert result["output"]["frame"] == "ECEF"
    assert len(result["output"]["vector"]) == 3


def test_ecef_to_geocentric():
    result = convert_position([-2703817.25, -4260534.25, 3887927.17], "ECEF", "GEOCENTRIC")
    assert "output" in result
    vec = result["output"]["vector"]
    assert abs(vec[0] - (-122.4)) < 0.01  # lon


def test_geodetic_to_geocentric():
    result = convert_position([-122.4, 37.8, 0.0], "GEODETIC", "GEOCENTRIC")
    assert "output" in result
    assert result["output"]["frame"] == "GEOCENTRIC"
    # Geocentric lat should be slightly less than geodetic lat
    assert result["output"]["vector"][1] < 37.8


# ---------------------------------------------------------------------------
# Position conversions — frame transforms
# ---------------------------------------------------------------------------

def test_eci_to_ecef_requires_epoch():
    result = convert_position([6778137.0, 0.0, 0.0], "ECI", "ECEF")
    assert "error" in result
    assert "epoch" in result["error"].lower()


def test_eci_to_ecef():
    result = convert_position(
        [6778137.0, 0.0, 0.0], "ECI", "ECEF",
        epoch="2024-01-01 12:00:00 UTC",
    )
    assert "output" in result
    assert result["output"]["frame"] == "ECEF"
    assert len(result["output"]["vector"]) == 3


def test_ecef_to_eci():
    # Convert to ECEF first, then back
    r1 = convert_position(
        [6778137.0, 0.0, 0.0], "ECI", "ECEF",
        epoch="2024-01-01 12:00:00 UTC",
    )
    r2 = convert_position(
        r1["output"]["vector"], "ECEF", "ECI",
        epoch="2024-01-01 12:00:00 UTC",
    )
    assert abs(r2["output"]["vector"][0] - 6778137.0) < 1.0
    assert abs(r2["output"]["vector"][1]) < 1.0
    assert abs(r2["output"]["vector"][2]) < 1.0


def test_gcrf_to_itrf():
    result = convert_position(
        [6778137.0, 0.0, 0.0], "GCRF", "ITRF",
        epoch="2024-01-01 12:00:00 UTC",
    )
    assert "output" in result
    assert result["output"]["frame"] == "ITRF"


def test_itrf_to_gcrf():
    r1 = convert_position(
        [6778137.0, 0.0, 0.0], "GCRF", "ITRF",
        epoch="2024-01-01 12:00:00 UTC",
    )
    r2 = convert_position(
        r1["output"]["vector"], "ITRF", "GCRF",
        epoch="2024-01-01 12:00:00 UTC",
    )
    assert abs(r2["output"]["vector"][0] - 6778137.0) < 1.0


def test_gcrf_to_eme2000():
    result = convert_position([6778137.0, 0.0, 0.0], "GCRF", "EME2000")
    assert "output" in result
    assert result["output"]["frame"] == "EME2000"


def test_eme2000_to_gcrf():
    r1 = convert_position([6778137.0, 0.0, 0.0], "GCRF", "EME2000")
    r2 = convert_position(r1["output"]["vector"], "EME2000", "GCRF")
    assert abs(r2["output"]["vector"][0] - 6778137.0) < 1.0


# ---------------------------------------------------------------------------
# State conversions
# ---------------------------------------------------------------------------

def test_koe_to_eci():
    # a=7000km, e=0.001, i=98deg, RAAN=45, omega=0, M=0
    koe = [7000000.0, 0.001, 98.0, 45.0, 0.0, 0.0]
    result = convert_state(koe, "KOE", "ECI")
    assert "output" in result
    assert result["output"]["frame"] == "ECI"
    assert len(result["output"]["vector"]) == 6


def test_eci_to_koe():
    state = [6778137.0, 0.0, 0.0, 0.0, 7672.0, 0.0]
    result = convert_state(state, "ECI", "KOE")
    assert "output" in result
    assert result["output"]["frame"] == "KOE"
    # Semi-major axis should be close to input radius
    assert abs(result["output"]["vector"][0] - 6778137.0) < 100000.0


def test_koe_eci_roundtrip():
    koe = [7000000.0, 0.001, 98.0, 45.0, 30.0, 60.0]
    r1 = convert_state(koe, "KOE", "ECI")
    r2 = convert_state(r1["output"]["vector"], "ECI", "KOE")
    for a, b in zip(koe, r2["output"]["vector"]):
        assert abs(a - b) < 0.01, f"KOE roundtrip failed: {a} vs {b}"


def test_state_eci_to_ecef():
    state = [6778137.0, 0.0, 0.0, 0.0, 7672.0, 0.0]
    result = convert_state(
        state, "ECI", "ECEF",
        epoch="2024-01-01 12:00:00 UTC",
    )
    assert "output" in result
    assert result["output"]["frame"] == "ECEF"
    assert len(result["output"]["vector"]) == 6


def test_state_gcrf_to_itrf():
    state = [6778137.0, 0.0, 0.0, 0.0, 7672.0, 0.0]
    result = convert_state(
        state, "GCRF", "ITRF",
        epoch="2024-01-01 12:00:00 UTC",
    )
    assert "output" in result
    assert result["output"]["frame"] == "ITRF"


def test_state_gcrf_to_eme2000():
    state = [6778137.0, 0.0, 0.0, 0.0, 7672.0, 0.0]
    result = convert_state(state, "GCRF", "EME2000")
    assert "output" in result
    assert result["output"]["frame"] == "EME2000"


# ---------------------------------------------------------------------------
# Relative position
# ---------------------------------------------------------------------------

def test_ecef_to_enz_geodetic_station():
    station_geod = [-122.4, 37.8, 0.0]
    # Get station ECEF and offset target
    ecef_result = convert_position(station_geod, "GEODETIC", "ECEF")
    station_ecef = ecef_result["output"]["vector"]
    target_ecef = [station_ecef[0] + 1000, station_ecef[1] + 2000, station_ecef[2] + 3000]

    result = convert_relative_position(
        station_geod, target_ecef, "ECEF", "ENZ",
        station_type="geodetic",
    )
    assert "output" in result
    assert result["output"]["frame"] == "ENZ"
    assert len(result["output"]["vector"]) == 3


def test_ecef_to_sez():
    station_geod = [-122.4, 37.8, 0.0]
    ecef_result = convert_position(station_geod, "GEODETIC", "ECEF")
    station_ecef = ecef_result["output"]["vector"]
    target_ecef = [station_ecef[0] + 1000, station_ecef[1] + 2000, station_ecef[2] + 3000]

    result = convert_relative_position(
        station_geod, target_ecef, "ECEF", "SEZ",
        station_type="geodetic",
    )
    assert "output" in result
    assert result["output"]["frame"] == "SEZ"


def test_enz_to_azel():
    enz = [1000.0, 2000.0, 500.0]
    result = convert_relative_position(
        [0.0, 0.0, 0.0], enz, "ENZ", "AZEL",
        station_type="ecef",
    )
    assert "output" in result
    assert result["output"]["frame"] == "AZEL"
    vec = result["output"]["vector"]
    assert len(vec) == 3  # az, el, range
    # Range should be magnitude of ENZ vector
    import math
    expected_range = math.sqrt(1000**2 + 2000**2 + 500**2)
    assert abs(vec[2] - expected_range) < 1.0


def test_sez_to_azel():
    sez = [-2000.0, 1000.0, 500.0]
    result = convert_relative_position(
        [0.0, 0.0, 0.0], sez, "SEZ", "AZEL",
        station_type="ecef",
    )
    assert "output" in result
    assert result["output"]["frame"] == "AZEL"


def test_enz_to_ecef_roundtrip():
    station_geod = [-122.4, 37.8, 0.0]
    ecef_result = convert_position(station_geod, "GEODETIC", "ECEF")
    station_ecef = ecef_result["output"]["vector"]
    target_ecef = [station_ecef[0] + 1000, station_ecef[1] + 2000, station_ecef[2] + 3000]

    # ECEF -> ENZ
    r1 = convert_relative_position(
        station_geod, target_ecef, "ECEF", "ENZ",
        station_type="geodetic",
    )
    # ENZ -> ECEF
    r2 = convert_relative_position(
        station_geod, r1["output"]["vector"], "ENZ", "ECEF",
        station_type="geodetic",
    )
    for a, b in zip(target_ecef, r2["output"]["vector"]):
        assert abs(a - b) < 1.0, f"ENZ roundtrip failed: {a} vs {b}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_frame():
    result = convert_position([1, 2, 3], "INVALID", "ECEF")
    assert "error" in result
    assert "valid_frames" in result


def test_wrong_vector_length():
    result = convert_position([1, 2, 3, 4], "GEODETIC", "ECEF")
    assert "error" in result
    assert "3 elements" in result["error"]


def test_wrong_state_vector_length():
    result = convert_state([1, 2, 3], "ECI", "KOE")
    assert "error" in result
    assert "6 elements" in result["error"]


def test_invalid_angle_format():
    result = convert_position([-122.4, 37.8, 0.0], "GEODETIC", "ECEF", angle_format="grads")
    assert "error" in result


def test_missing_epoch_for_frame_conversion():
    result = convert_state([6778137.0, 0.0, 0.0, 0.0, 7672.0, 0.0], "ECI", "ECEF")
    assert "error" in result
    assert "epoch" in result["error"].lower()


def test_same_frame_returns_identity():
    vec = [6778137.0, 0.0, 0.0]
    result = convert_position(vec, "ECEF", "ECEF")
    assert "output" in result
    assert result["output"]["vector"] == vec


def test_same_state_frame_returns_identity():
    vec = [6778137.0, 0.0, 0.0, 0.0, 7672.0, 0.0]
    result = convert_state(vec, "ECI", "ECI")
    assert "output" in result
    assert result["output"]["vector"] == vec


# ---------------------------------------------------------------------------
# Angle format
# ---------------------------------------------------------------------------

def test_radians_angle_format():
    import math
    lon_rad = math.radians(-122.4)
    lat_rad = math.radians(37.8)
    result = convert_position(
        [lon_rad, lat_rad, 0.0], "GEODETIC", "ECEF",
        angle_format="radians",
    )
    assert "output" in result
    # Should produce same ECEF as degrees version
    result_deg = convert_position([-122.4, 37.8, 0.0], "GEODETIC", "ECEF")
    for a, b in zip(result["output"]["vector"], result_deg["output"]["vector"]):
        assert abs(a - b) < 1.0


def test_unsupported_position_conversion():
    result = convert_position([1, 2, 3], "ECI", "GEODETIC")
    assert "error" in result


def test_unsupported_state_conversion():
    result = convert_state([1, 2, 3, 4, 5, 6], "KOE", "ECEF")
    assert "error" in result


def test_invalid_relative_from_frame():
    result = convert_relative_position([0, 0, 0], [1, 2, 3], "INVALID", "ENZ")
    assert "error" in result


def test_invalid_relative_to_frame():
    result = convert_relative_position([0, 0, 0], [1, 2, 3], "ECEF", "INVALID")
    assert "error" in result


def test_components_labeled():
    result = convert_position([-122.4, 37.8, 0.0], "GEODETIC", "ECEF")
    comps = result["output"]["components"]
    assert "x_m" in comps
    assert "y_m" in comps
    assert "z_m" in comps
