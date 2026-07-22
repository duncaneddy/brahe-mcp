import numpy as np
import brahe

from brahe_mcp.radec import convert_radec, list_radec_options


def test_list_radec_options_structure():
    opts = list_radec_options()
    assert "RADEC" in opts["frames"]
    assert "AZEL" in opts["frames"]
    assert opts["position_components"]["RADEC"] == ["ra", "dec", "range_m"]
    assert "site_types" in opts


def test_position_inertial_to_radec_matches_brahe():
    vec = [7000e3, 1000e3, 2000e3]
    res = convert_radec(vec, "ECI", "RADEC")
    assert "error" not in res
    expected = brahe.position_inertial_to_radec(
        np.array(vec), brahe.AngleFormat.DEGREES
    )
    assert np.allclose(res["output"]["vector"], expected)
    assert res["output"]["components"]["ra"] == res["output"]["vector"][0]


def test_position_radec_roundtrip():
    vec = [7000e3, 1000e3, 2000e3]
    fwd = convert_radec(vec, "GCRF", "RADEC")
    back = convert_radec(fwd["output"]["vector"], "RADEC", "GCRF")
    assert np.allclose(back["output"]["vector"], vec, rtol=1e-9)


def test_state_radec_roundtrip():
    state = [7000e3, 1000e3, 2000e3, -1000.0, 7000.0, 100.0]
    fwd = convert_radec(state, "ECI", "RADEC")
    assert len(fwd["output"]["vector"]) == 6
    assert "ra_rate" in fwd["output"]["components"]
    back = convert_radec(fwd["output"]["vector"], "RADEC", "ECI")
    assert np.allclose(back["output"]["vector"], state, rtol=1e-7)


def test_radians_matches_degrees():
    vec = [7000e3, 1000e3, 2000e3]
    deg = convert_radec(vec, "ECI", "RADEC", angle_format="degrees")
    rad = convert_radec(vec, "ECI", "RADEC", angle_format="radians")
    assert np.isclose(np.radians(deg["output"]["vector"][0]),
                      rad["output"]["vector"][0])
    assert np.isclose(deg["output"]["vector"][2], rad["output"]["vector"][2])


def test_unknown_frame_returns_valid_frames():
    res = convert_radec([1.0, 2.0, 3.0], "NOPE", "RADEC")
    assert "error" in res
    assert "RADEC" in res["valid_frames"]


def test_unsupported_pair_errors():
    res = convert_radec([1.0, 2.0, 3.0], "ECI", "GCRF")
    assert "error" in res
    assert "valid_conversions" in res


def test_bad_vector_length_errors():
    res = convert_radec([1.0, 2.0], "ECI", "RADEC")
    assert "error" in res


def test_state_to_azel_unsupported():
    res = convert_radec([1.0] * 6, "RADEC", "AZEL")
    assert "error" in res


SITE = [-122.4, 37.8, 100.0]
EPOCH = "2024-01-01T00:00:00Z"


def test_radec_to_azel_matches_brahe():
    radec = [45.0, 20.0, 1.0e9]
    res = convert_radec(radec, "RADEC", "AZEL", site=SITE, epoch=EPOCH)
    assert "error" not in res
    expected = brahe.position_radec_to_azel(
        np.array(radec), np.array(SITE), brahe.Epoch(EPOCH),
        brahe.AngleFormat.DEGREES,
    )
    assert np.allclose(res["output"]["vector"], expected)
    assert set(res["output"]["components"]) == {"az", "el", "range_m"}


def test_radec_azel_roundtrip():
    radec = [45.0, 20.0, 1.0e9]
    fwd = convert_radec(radec, "RADEC", "AZEL", site=SITE, epoch=EPOCH)
    back = convert_radec(
        fwd["output"]["vector"], "AZEL", "RADEC", site=SITE, epoch=EPOCH
    )
    assert np.allclose(back["output"]["vector"], radec, rtol=1e-8)


def test_azel_requires_site():
    res = convert_radec([45.0, 20.0, 1.0e9], "RADEC", "AZEL", epoch=EPOCH)
    assert "error" in res
    assert "site" in res["error"]


def test_azel_requires_epoch():
    res = convert_radec([45.0, 20.0, 1.0e9], "RADEC", "AZEL", site=SITE)
    assert "error" in res
    assert "epoch" in res["error"]


def test_azel_bad_site_length():
    res = convert_radec(
        [45.0, 20.0, 1.0e9], "RADEC", "AZEL", site=[1.0, 2.0], epoch=EPOCH
    )
    assert "error" in res


def test_azel_invalid_site_type():
    res = convert_radec(
        [45.0, 20.0, 1.0e9], "RADEC", "AZEL",
        site=SITE, site_type="lunar", epoch=EPOCH,
    )
    assert "error" in res
    assert "site_type" in res["error"]


def test_azel_ecef_site_matches_geodetic_site():
    radec = [45.0, 20.0, 1.0e9]
    ecef = brahe.position_geodetic_to_ecef(
        np.array(SITE), brahe.AngleFormat.DEGREES
    ).tolist()
    a = convert_radec(radec, "RADEC", "AZEL", site=SITE, epoch=EPOCH)
    b = convert_radec(
        radec, "RADEC", "AZEL", site=ecef, site_type="ecef", epoch=EPOCH
    )
    assert np.allclose(a["output"]["vector"], b["output"]["vector"], rtol=1e-6)


def test_azel_invalid_epoch():
    res = convert_radec(
        [45.0, 20.0, 1.0e9], "RADEC", "AZEL", site=SITE, epoch="not-a-date"
    )
    assert "error" in res
