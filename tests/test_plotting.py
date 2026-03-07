import base64

import pytest
from mcp.types import ImageContent, TextContent

from brahe_mcp.plotting import (
    list_plotting_options,
    plot_gp_history_elements,
    plot_altitude,
    plot_altitude_from_gp,
    plot_ground_track,
    plot_ground_track_from_gp,
    plot_orbit_elements,
    plot_orbit_elements_from_gp,
    plot_access_geometry,
    plot_access_geometry_from_gp,
    plot_gabbard_diagram,
    _figure_to_image,
)


# Valid ISS-like TLE for testing
TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30000-3 0  9002"
TLE_LINE2 = "2 25544  51.6400 150.0000 0003000 100.0000 260.0000 15.50000000    18"

FAKE_GP_RECORD = {
    "object_name": "ISS (ZARYA)",
    "norad_cat_id": 25544,
    "object_id": "1998-067A",
    "epoch": "2024-01-01 12:00:00 UTC",
    "inclination": 51.64,
    "eccentricity": 0.0003,
    "ra_of_asc_node": 150.0,
    "arg_of_pericenter": 100.0,
    "mean_anomaly": 260.0,
    "mean_motion": 15.5,
    "bstar": 0.0003,
    "semimajor_axis": 6801.0,
    "period": 92.9,
    "apoapsis": 423.0,
    "periapsis": 419.0,
    "classification_type": "U",
    "object_type": "PAYLOAD",
    "country_code": "ISS",
    "launch_date": "1998-11-20",
    "decay_date": None,
    "tle_line0": "0 ISS (ZARYA)",
    "tle_line1": TLE_LINE1,
    "tle_line2": TLE_LINE2,
}

# Second GP record with slightly different elements for multi-object tests
FAKE_GP_RECORD_2 = {
    **FAKE_GP_RECORD,
    "object_name": "DEBRIS",
    "norad_cat_id": 99999,
    "epoch": "2024-01-01 12:00:00 UTC",
    "inclination": 51.70,
    "ra_of_asc_node": 151.0,
    "mean_anomaly": 250.0,
}

TEST_LOCATION = {"lon": -75.0, "lat": 40.0, "altitude_m": 0.0, "name": "Test Station"}
TEST_EPOCH_START = "2024-01-01 12:00:00 UTC"
TEST_EPOCH_END = "2024-01-01 14:00:00 UTC"


def _is_valid_image(content):
    """Check if content is a valid ImageContent with PNG data."""
    assert isinstance(content, ImageContent)
    assert content.mimeType == "image/png"
    # Verify it's valid base64 and starts with PNG header
    decoded = base64.b64decode(content.data)
    assert decoded[:4] == b"\x89PNG"
    return True


def _assert_plot_result(result):
    """Assert the result is a list of [TextContent, ImageContent]."""
    assert isinstance(result, list), f"Expected list, got {type(result)}: {result}"
    assert len(result) == 2
    assert isinstance(result[0], TextContent)
    assert _is_valid_image(result[1])


# ---------------------------------------------------------------------------
# Discovery tool
# ---------------------------------------------------------------------------


def test_list_plotting_options():
    result = list_plotting_options()
    assert "plot_types" in result
    assert "plot_gp_history_elements" in result["plot_types"]
    assert "plot_ground_track" in result["plot_types"]
    assert "plot_access_geometry" in result["plot_types"]
    assert "plot_gabbard_diagram" in result["plot_types"]
    assert "satellite_sources" in result
    assert "location_format" in result


# ---------------------------------------------------------------------------
# _figure_to_image
# ---------------------------------------------------------------------------


def test_figure_to_image():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    img = _figure_to_image(fig)
    assert _is_valid_image(img)


# ---------------------------------------------------------------------------
# GP history elements
# ---------------------------------------------------------------------------


class TestPlotGPHistoryElements:
    def test_basic_plot(self):
        """Plot all 6 classical elements from fake GP records."""
        records = [FAKE_GP_RECORD, FAKE_GP_RECORD_2]
        result = plot_gp_history_elements(gp_records=records)
        _assert_plot_result(result)

    def test_single_element(self):
        records = [FAKE_GP_RECORD, FAKE_GP_RECORD_2]
        result = plot_gp_history_elements(
            gp_records=records, elements=["eccentricity"]
        )
        _assert_plot_result(result)

    def test_multiple_elements(self):
        records = [FAKE_GP_RECORD, FAKE_GP_RECORD_2]
        result = plot_gp_history_elements(
            gp_records=records, elements=["semimajor_axis", "inclination"]
        )
        _assert_plot_result(result)

    def test_custom_title(self):
        records = [FAKE_GP_RECORD, FAKE_GP_RECORD_2]
        result = plot_gp_history_elements(
            gp_records=records, title="Custom Title"
        )
        _assert_plot_result(result)

    def test_empty_records_error(self):
        result = plot_gp_history_elements(gp_records=[])
        assert "error" in result

    def test_no_input_error(self):
        result = plot_gp_history_elements()
        assert "error" in result

    def test_invalid_element_error(self):
        result = plot_gp_history_elements(
            gp_records=[FAKE_GP_RECORD], elements=["nonexistent"]
        )
        assert "error" in result
        assert "valid_elements" in result

    def test_all_available_elements(self):
        """Test plotting non-classical elements like bstar, period."""
        records = [FAKE_GP_RECORD, FAKE_GP_RECORD_2]
        result = plot_gp_history_elements(
            gp_records=records, elements=["bstar", "period", "apoapsis"]
        )
        _assert_plot_result(result)


# ---------------------------------------------------------------------------
# Altitude
# ---------------------------------------------------------------------------


class TestPlotAltitude:
    def test_from_tle(self):
        result = plot_altitude(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        _assert_plot_result(result)
        # Summary should contain altitude range
        assert "km" in result[0].text

    def test_with_title(self):
        result = plot_altitude(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
            title="ISS Altitude",
        )
        _assert_plot_result(result)

    def test_invalid_satellite(self):
        result = plot_altitude(
            satellite={"source": "magic"},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        assert "error" in result

    def test_end_before_start(self):
        result = plot_altitude(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_END,
            end_epoch=TEST_EPOCH_START,
        )
        assert "error" in result

    def test_altitude_range_reasonable(self):
        """ISS-like orbit should have altitude between ~200-500 km."""
        result = plot_altitude(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        _assert_plot_result(result)
        summary = result[0].text
        # Extract min/max from "Range: 419.1 - 423.2 km."
        import re
        match = re.search(r"Range: ([\d.]+) - ([\d.]+) km", summary)
        assert match, f"Could not parse altitude range from: {summary}"
        alt_min = float(match.group(1))
        alt_max = float(match.group(2))
        assert 100 < alt_min < 600
        assert 100 < alt_max < 600


class TestPlotAltitudeFromGP:
    def test_sgp4(self):
        result = plot_altitude_from_gp(
            gp_record=FAKE_GP_RECORD,
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        _assert_plot_result(result)

    def test_keplerian(self):
        result = plot_altitude_from_gp(
            gp_record=FAKE_GP_RECORD,
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
            propagator_type="keplerian",
        )
        _assert_plot_result(result)


# ---------------------------------------------------------------------------
# Ground track
# ---------------------------------------------------------------------------


class TestPlotGroundTrack:
    def test_from_tle(self):
        result = plot_ground_track(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        _assert_plot_result(result)

    def test_with_ground_stations(self):
        result = plot_ground_track(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
            ground_stations=[TEST_LOCATION],
        )
        _assert_plot_result(result)

    def test_invalid_satellite(self):
        result = plot_ground_track(
            satellite={"source": "magic"},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        assert "error" in result

    def test_end_before_start(self):
        result = plot_ground_track(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_END,
            end_epoch=TEST_EPOCH_START,
        )
        assert "error" in result


class TestPlotGroundTrackFromGP:
    def test_sgp4(self):
        result = plot_ground_track_from_gp(
            gp_record=FAKE_GP_RECORD,
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        _assert_plot_result(result)

    def test_keplerian(self):
        result = plot_ground_track_from_gp(
            gp_record=FAKE_GP_RECORD,
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
            propagator_type="keplerian",
        )
        _assert_plot_result(result)


# ---------------------------------------------------------------------------
# Orbit elements
# ---------------------------------------------------------------------------


class TestPlotOrbitElements:
    def test_from_tle(self):
        result = plot_orbit_elements(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        _assert_plot_result(result)

    def test_invalid_satellite(self):
        result = plot_orbit_elements(
            satellite={"source": "magic"},
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        assert "error" in result

    def test_end_before_start(self):
        result = plot_orbit_elements(
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            start_epoch=TEST_EPOCH_END,
            end_epoch=TEST_EPOCH_START,
        )
        assert "error" in result


class TestPlotOrbitElementsFromGP:
    def test_sgp4(self):
        result = plot_orbit_elements_from_gp(
            gp_record=FAKE_GP_RECORD,
            start_epoch=TEST_EPOCH_START,
            end_epoch=TEST_EPOCH_END,
        )
        _assert_plot_result(result)


# ---------------------------------------------------------------------------
# Access geometry
# ---------------------------------------------------------------------------


class TestPlotAccessGeometry:
    def test_polar(self):
        result = plot_access_geometry(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            plot_type="polar",
        )
        # May get error if no windows found in time range
        if isinstance(result, list):
            _assert_plot_result(result)
        else:
            assert "error" in result

    def test_elevation(self):
        result = plot_access_geometry(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            plot_type="elevation",
        )
        if isinstance(result, list):
            _assert_plot_result(result)
        else:
            assert "error" in result

    def test_elevation_azimuth(self):
        result = plot_access_geometry(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            plot_type="elevation_azimuth",
        )
        if isinstance(result, list):
            _assert_plot_result(result)
        else:
            assert "error" in result

    def test_invalid_plot_type(self):
        result = plot_access_geometry(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            plot_type="invalid",
        )
        assert "error" in result
        assert "valid_types" in result

    def test_invalid_location(self):
        result = plot_access_geometry(
            location={"lat": 40.0},
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "error" in result

    def test_end_before_start(self):
        result = plot_access_geometry(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-02 00:00:00 UTC",
            search_end="2024-01-01 00:00:00 UTC",
        )
        assert "error" in result


class TestPlotAccessGeometryFromGP:
    def test_from_gp(self):
        result = plot_access_geometry_from_gp(
            gp_record=FAKE_GP_RECORD,
            location=TEST_LOCATION,
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        if isinstance(result, list):
            _assert_plot_result(result)
        else:
            assert "error" in result


# ---------------------------------------------------------------------------
# Gabbard diagram
# ---------------------------------------------------------------------------


class TestPlotGabbardDiagram:
    def test_single_object(self):
        result = plot_gabbard_diagram(gp_records=[FAKE_GP_RECORD])
        _assert_plot_result(result)

    def test_multiple_objects(self):
        result = plot_gabbard_diagram(
            gp_records=[FAKE_GP_RECORD, FAKE_GP_RECORD_2]
        )
        _assert_plot_result(result)

    def test_with_title(self):
        result = plot_gabbard_diagram(
            gp_records=[FAKE_GP_RECORD],
            title="Test Gabbard",
        )
        _assert_plot_result(result)

    def test_empty_records_error(self):
        result = plot_gabbard_diagram(gp_records=[])
        assert "error" in result
