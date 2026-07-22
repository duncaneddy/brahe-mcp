import numpy as np
import brahe

from brahe_mcp.radec import apply_proper_motion, convert_radec, list_radec_options


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


def test_zero_position_errors_not_nan():
    # dec is undefined for a zero-length position vector, so brahe returns
    # NaN. This must surface as an error dict, never a success envelope
    # containing non-finite values.
    res = convert_radec([0.0, 0.0, 0.0], "ECI", "RADEC")
    assert "error" in res
    assert "output" not in res


def test_convert_radec_non_numeric_input_errors():
    """Regression test for the non-numeric guard added in 0093f99, which
    shipped without a test and could be silently removed."""
    res = convert_radec(["a", "b", "c"], "ECI", "RADEC")
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


def test_radec_to_azel_radians_matches_degrees():
    # angle_format also governs the units of `site`'s lon/lat, not just the
    # RA/Dec angles (see convert_radec's docstring). Passing site in
    # radians alongside angle_format="radians" must agree with the all-
    # degrees path.
    radec = [45.0, 20.0, 1.0e9]
    deg = convert_radec(radec, "RADEC", "AZEL", site=SITE, epoch=EPOCH)
    assert "error" not in deg

    radec_rad = [np.radians(radec[0]), np.radians(radec[1]), radec[2]]
    site_rad = [np.radians(SITE[0]), np.radians(SITE[1]), SITE[2]]
    rad = convert_radec(
        radec_rad, "RADEC", "AZEL", site=site_rad, epoch=EPOCH,
        angle_format="radians",
    )
    assert "error" not in rad
    assert np.isclose(
        np.radians(deg["output"]["vector"][0]), rad["output"]["vector"][0]
    )
    assert np.isclose(
        np.radians(deg["output"]["vector"][1]), rad["output"]["vector"][1]
    )
    assert np.isclose(deg["output"]["vector"][2], rad["output"]["vector"][2])


def test_azel_invalid_epoch():
    res = convert_radec(
        [45.0, 20.0, 1.0e9], "RADEC", "AZEL", site=SITE, epoch="not-a-date"
    )
    assert "error" in res


E_FROM = "2000-01-01T12:00:00Z"
E_TO = "2030-01-01T00:00:00Z"


def test_apply_proper_motion_matches_brahe():
    res = apply_proper_motion(
        269.45, 4.69, -798.0, 10330.0, E_FROM, E_TO,
        parallax=547.0, radial_velocity=-110.6,
    )
    assert "error" not in res
    expected = brahe.apply_proper_motion(
        269.45, 4.69, -798.0, 10330.0, 547.0, -110.6,
        brahe.Epoch(E_FROM), brahe.Epoch(E_TO), brahe.AngleFormat.DEGREES,
    )
    assert np.isclose(res["output"]["ra"], expected[0])
    assert np.isclose(res["output"]["dec"], expected[1])


def test_zero_proper_motion_is_near_identity():
    res = apply_proper_motion(269.45, 4.69, 0.0, 0.0, E_FROM, E_TO)
    assert np.isclose(res["output"]["ra"], 269.45, atol=1e-9)
    assert np.isclose(res["output"]["dec"], 4.69, atol=1e-9)


def test_proper_motion_moves_position():
    res = apply_proper_motion(269.45, 4.69, -798.0, 10330.0, E_FROM, E_TO)
    # 10330 mas/yr over ~30 yr is ~0.086 deg of declination drift.
    assert abs(res["output"]["dec"] - 4.69) > 1e-3


def test_proper_motion_omitted_parallax_defaults_to_none():
    res = apply_proper_motion(269.45, 4.69, -798.0, 10330.0, E_FROM, E_TO)
    assert "error" not in res
    assert res["input"]["parallax"] is None


def test_proper_motion_radians_matches_degrees():
    deg = apply_proper_motion(269.45, 4.69, -798.0, 10330.0, E_FROM, E_TO)
    rad = apply_proper_motion(
        np.radians(269.45), np.radians(4.69), -798.0, 10330.0,
        E_FROM, E_TO, angle_format="radians",
    )
    assert np.isclose(np.radians(deg["output"]["ra"]), rad["output"]["ra"], atol=1e-12)
    assert np.isclose(np.radians(deg["output"]["dec"]), rad["output"]["dec"], atol=1e-12)


def test_proper_motion_invalid_epoch_errors():
    res = apply_proper_motion(269.45, 4.69, 0.0, 0.0, "nope", E_TO)
    assert "error" in res


def test_proper_motion_reports_units():
    res = apply_proper_motion(269.45, 4.69, 0.0, 0.0, E_FROM, E_TO)
    assert res["units"]["pm_ra"] == "mas/yr"
    assert res["units"]["parallax"] == "mas"
    assert res["units"]["radial_velocity"] == "km/s"
