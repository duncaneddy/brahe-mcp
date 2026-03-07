import pytest

from brahe_mcp.accesses import (
    list_access_options,
    compute_access,
    compute_access_from_gp,
    _build_constraint,
    _build_constraints,
    _build_location,
    _build_propagator,
    _build_search_config,
    _build_property_computers,
    _serialize_access_window,
)


# Valid ISS-like TLE for testing (same as test_propagation.py)
TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30000-3 0  9002"
TLE_LINE2 = "2 25544  51.6400 150.0000 0003000 100.0000 260.0000 15.50000000    18"

# Fake GP record matching serialize_gp_record() output shape
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

TEST_LOCATION = {"lon": -75.0, "lat": 40.0, "altitude_m": 0.0, "name": "Test Station"}
TEST_STATE_ECI = [7000000.0, 0.0, 0.0, 0.0, 7546.0, 0.0]
TEST_EPOCH = "2024-01-01 12:00:00 UTC"


# ---------------------------------------------------------------------------
# Discovery tool
# ---------------------------------------------------------------------------


def test_list_access_options():
    result = list_access_options()
    assert "constraint_types" in result
    assert "property_computers" in result
    assert "location_format" in result
    assert "satellite_sources" in result
    assert "config_options" in result
    assert "defaults" in result
    assert "elevation" in result["constraint_types"]
    assert "range" in result["property_computers"]


# ---------------------------------------------------------------------------
# Build constraint tests
# ---------------------------------------------------------------------------


class TestBuildConstraint:
    def test_elevation_min_only(self):
        c = _build_constraint({"type": "elevation", "min_deg": 10.0})
        assert c is not None

    def test_elevation_min_max(self):
        c = _build_constraint({"type": "elevation", "min_deg": 5.0, "max_deg": 85.0})
        assert c is not None

    def test_elevation_default_min(self):
        c = _build_constraint({"type": "elevation"})
        assert c is not None

    def test_off_nadir(self):
        c = _build_constraint({"type": "off_nadir", "max_deg": 45.0})
        assert c is not None

    def test_off_nadir_missing_param(self):
        with pytest.raises(ValueError, match="max_deg"):
            _build_constraint({"type": "off_nadir"})

    def test_local_time(self):
        c = _build_constraint({"type": "local_time", "windows": [[600, 1800]]})
        assert c is not None

    def test_local_time_hours(self):
        c = _build_constraint({"type": "local_time_hours", "windows": [[6.0, 18.0]]})
        assert c is not None

    def test_look_direction(self):
        c = _build_constraint({"type": "look_direction", "allowed": "right"})
        assert c is not None

    def test_asc_dsc(self):
        c = _build_constraint({"type": "asc_dsc", "allowed": "ascending"})
        assert c is not None

    def test_elevation_mask(self):
        c = _build_constraint({
            "type": "elevation_mask",
            "mask": [[0, 15], [90, 5], [180, 5], [270, 5]],
        })
        assert c is not None

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown constraint type"):
            _build_constraint({"type": "nonexistent"})

    def test_invalid_look_direction(self):
        with pytest.raises(ValueError, match="Invalid look direction"):
            _build_constraint({"type": "look_direction", "allowed": "sideways"})

    def test_invalid_asc_dsc(self):
        with pytest.raises(ValueError, match="Invalid asc/dsc"):
            _build_constraint({"type": "asc_dsc", "allowed": "sideways"})


# ---------------------------------------------------------------------------
# Build location tests
# ---------------------------------------------------------------------------


class TestBuildLocation:
    def test_basic_location(self):
        loc = _build_location({"lon": -75.0, "lat": 40.0})
        assert loc.lon == -75.0
        assert loc.lat == 40.0

    def test_with_altitude_m(self):
        loc = _build_location({"lon": 0.0, "lat": 0.0, "altitude_m": 100.0})
        assert loc.alt == 100.0

    def test_with_alt_key(self):
        loc = _build_location({"lon": 0.0, "lat": 0.0, "alt": 50.0})
        assert loc.alt == 50.0

    def test_with_name(self):
        loc = _build_location({"lon": 0.0, "lat": 0.0, "name": "My Station"})
        assert loc.get_name() == "My Station"

    def test_station_dict_format(self):
        """Test compatibility with _serialize_station output format."""
        station_dict = {
            "name": "Canberra",
            "lon": 148.98,
            "lat": -35.40,
            "altitude_m": 550.0,
            "properties": {"provider": "nasa dsn"},
        }
        loc = _build_location(station_dict)
        assert loc.lon == 148.98
        assert loc.lat == -35.40
        assert loc.alt == 550.0
        assert loc.get_name() == "Canberra"

    def test_missing_lon(self):
        with pytest.raises(ValueError, match="lon"):
            _build_location({"lat": 40.0})

    def test_missing_lat(self):
        with pytest.raises(ValueError, match="lat"):
            _build_location({"lon": -75.0})


# ---------------------------------------------------------------------------
# Build propagator tests
# ---------------------------------------------------------------------------


class TestBuildPropagator:
    def test_from_tle(self):
        prop = _build_propagator({
            "source": "tle",
            "tle_line1": TLE_LINE1,
            "tle_line2": TLE_LINE2,
        })
        assert prop is not None

    def test_from_gp_sgp4(self):
        prop = _build_propagator({
            "source": "gp_record",
            "gp_record": FAKE_GP_RECORD,
            "propagator_type": "sgp4",
        })
        assert prop is not None

    def test_from_gp_keplerian(self):
        prop = _build_propagator({
            "source": "gp_record",
            "gp_record": FAKE_GP_RECORD,
            "propagator_type": "keplerian",
        })
        assert prop is not None

    def test_from_state_keplerian(self):
        prop = _build_propagator({
            "source": "state",
            "epoch": TEST_EPOCH,
            "state_eci": TEST_STATE_ECI,
            "propagator_type": "keplerian",
        })
        assert prop is not None

    def test_from_gp_numerical(self):
        prop = _build_propagator({
            "source": "gp_record",
            "gp_record": FAKE_GP_RECORD,
            "propagator_type": "numerical",
            "force_model": "two_body",
        })
        assert prop is not None

    def test_from_gp_numerical_missing_params(self):
        with pytest.raises(ValueError, match="spacecraft_params"):
            _build_propagator({
                "source": "gp_record",
                "gp_record": FAKE_GP_RECORD,
                "propagator_type": "numerical",
                "force_model": "default",
            })

    def test_from_state_numerical(self):
        prop = _build_propagator({
            "source": "state",
            "epoch": TEST_EPOCH,
            "state_eci": TEST_STATE_ECI,
            "propagator_type": "numerical",
            "force_model": "two_body",
        })
        assert prop is not None

    def test_from_state_sgp4_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            _build_propagator({
                "source": "state",
                "epoch": TEST_EPOCH,
                "state_eci": TEST_STATE_ECI,
                "propagator_type": "sgp4",
            })

    def test_invalid_source(self):
        with pytest.raises(ValueError, match="Unknown satellite source"):
            _build_propagator({"source": "magic"})

    def test_tle_missing_lines(self):
        with pytest.raises(ValueError, match="tle_line1"):
            _build_propagator({"source": "tle"})

    def test_state_missing_epoch(self):
        with pytest.raises(ValueError, match="epoch"):
            _build_propagator({"source": "state", "state_eci": TEST_STATE_ECI})

    def test_state_wrong_length(self):
        with pytest.raises(ValueError, match="6 elements"):
            _build_propagator({
                "source": "state",
                "epoch": TEST_EPOCH,
                "state_eci": [1, 2, 3],
            })


# ---------------------------------------------------------------------------
# Build search config tests
# ---------------------------------------------------------------------------


class TestBuildSearchConfig:
    def test_none_input(self):
        assert _build_search_config(None) is None

    def test_empty_dict(self):
        assert _build_search_config({}) is None

    def test_with_params(self):
        cfg = _build_search_config({
            "initial_time_step": 30.0,
            "adaptive_step": True,
            "time_tolerance": 0.01,
        })
        assert cfg is not None
        assert cfg.initial_time_step == 30.0
        assert cfg.adaptive_step is True
        assert cfg.time_tolerance == 0.01


# ---------------------------------------------------------------------------
# Build property computers tests
# ---------------------------------------------------------------------------


class TestBuildPropertyComputers:
    def test_none_input(self):
        assert _build_property_computers(None) is None

    def test_empty_list(self):
        assert _build_property_computers([]) is None

    def test_range_computer(self):
        computers = _build_property_computers([{"type": "range"}])
        assert computers is not None
        assert len(computers) == 1

    def test_range_rate_computer(self):
        computers = _build_property_computers([{"type": "range_rate"}])
        assert computers is not None
        assert len(computers) == 1

    def test_doppler_computer(self):
        computers = _build_property_computers([{
            "type": "doppler",
            "uplink_frequency": 2.2e9,
            "downlink_frequency": 2.2e9,
        }])
        assert computers is not None

    def test_multiple_computers(self):
        computers = _build_property_computers([
            {"type": "range"},
            {"type": "range_rate"},
        ])
        assert len(computers) == 2

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown property computer"):
            _build_property_computers([{"type": "invalid"}])


# ---------------------------------------------------------------------------
# Compute access end-to-end tests
# ---------------------------------------------------------------------------


class TestComputeAccess:
    def test_basic_elevation_access(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "error" not in result
        assert result["count"] >= 0
        assert "windows" in result
        assert result["propagator_type"] == "sgp4"
        assert result["location"]["name"] == "Test Station"
        # Default constraint should be applied
        assert "elevation" in result["constraints_applied"][0]

    def test_min_elevation_shortcut(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            min_elevation_deg=10.0,
        )
        assert "error" not in result
        assert "elevation(min=10.0deg)" in result["constraints_applied"]

    def test_default_constraint(self):
        """When no constraints are given, defaults to elevation(5deg)."""
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "default" in result["constraints_applied"][0]

    def test_with_explicit_constraints(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            constraints=[{"type": "elevation", "min_deg": 15.0}],
        )
        assert "error" not in result
        assert "elevation(min=15.0deg)" in result["constraints_applied"]

    def test_no_windows_case(self):
        """Very high elevation should produce no windows."""
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-01 01:00:00 UTC",
            min_elevation_deg=89.0,
        )
        assert "error" not in result
        assert result["count"] == 0
        assert result["windows"] == []

    def test_invalid_location(self):
        result = compute_access(
            location={"lat": 40.0},  # missing lon
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "error" in result

    def test_invalid_satellite(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "magic"},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "error" in result

    def test_end_before_start(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-02 00:00:00 UTC",
            search_end="2024-01-01 00:00:00 UTC",
        )
        assert "error" in result

    def test_with_config(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            config={"initial_time_step": 30.0, "time_tolerance": 0.01},
        )
        assert "error" not in result

    def test_with_property_computers(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            property_computers=[{"type": "range"}],
        )
        assert "error" not in result
        if result["count"] > 0:
            w = result["windows"][0]
            assert "additional_properties" in w
            assert "range" in w["additional_properties"]

    def test_station_dict_compatibility(self):
        """Station dicts from groundstation tools should work as locations."""
        station_dict = {
            "name": "Canberra",
            "lon": 148.98,
            "lat": -35.40,
            "altitude_m": 550.0,
            "properties": {"provider": "nasa dsn"},
        }
        result = compute_access(
            location=station_dict,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "error" not in result
        assert result["location"]["name"] == "Canberra"

    def test_from_state_source(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={
                "source": "state",
                "epoch": TEST_EPOCH,
                "state_eci": TEST_STATE_ECI,
                "propagator_type": "keplerian",
            },
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "error" not in result
        assert result["propagator_type"] == "keplerian"

    def test_numerical_propagator(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={
                "source": "state",
                "epoch": TEST_EPOCH,
                "state_eci": TEST_STATE_ECI,
                "propagator_type": "numerical",
                "force_model": "two_body",
            },
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "error" not in result
        assert result["propagator_type"] == "numerical"

    def test_constraint_logic_any(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            constraints=[
                {"type": "elevation", "min_deg": 10.0},
                {"type": "asc_dsc", "allowed": "ascending"},
            ],
            constraint_logic="any",
        )
        assert "error" not in result


# ---------------------------------------------------------------------------
# Compute access from GP record tests
# ---------------------------------------------------------------------------


class TestComputeAccessFromGP:
    def test_sgp4_from_gp(self):
        result = compute_access_from_gp(
            gp_record=FAKE_GP_RECORD,
            location=TEST_LOCATION,
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        assert "error" not in result
        assert result["propagator_type"] == "sgp4"
        assert result["satellite_name"] == "ISS (ZARYA)"

    def test_keplerian_from_gp(self):
        result = compute_access_from_gp(
            gp_record=FAKE_GP_RECORD,
            location=TEST_LOCATION,
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            propagator_type="keplerian",
        )
        assert "error" not in result
        assert result["propagator_type"] == "keplerian"

    def test_numerical_from_gp(self):
        result = compute_access_from_gp(
            gp_record=FAKE_GP_RECORD,
            location=TEST_LOCATION,
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            propagator_type="numerical",
            force_model="two_body",
        )
        assert "error" not in result
        assert result["propagator_type"] == "numerical"

    def test_invalid_propagator_type(self):
        result = compute_access_from_gp(
            gp_record=FAKE_GP_RECORD,
            location=TEST_LOCATION,
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            propagator_type="invalid",
        )
        assert "error" in result
        assert "valid_types" in result

    def test_with_constraints(self):
        result = compute_access_from_gp(
            gp_record=FAKE_GP_RECORD,
            location=TEST_LOCATION,
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            constraints=[{"type": "elevation", "min_deg": 20.0}],
        )
        assert "error" not in result

    def test_with_min_elevation(self):
        result = compute_access_from_gp(
            gp_record=FAKE_GP_RECORD,
            location=TEST_LOCATION,
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
            min_elevation_deg=15.0,
        )
        assert "error" not in result
        assert "elevation(min=15.0deg)" in result["constraints_applied"]


# ---------------------------------------------------------------------------
# Serialize access window tests
# ---------------------------------------------------------------------------


class TestSerializeAccessWindow:
    """Test window serialization using real computed windows."""

    @pytest.fixture
    def sample_windows(self):
        result = compute_access(
            location=TEST_LOCATION,
            satellite={"source": "tle", "tle_line1": TLE_LINE1, "tle_line2": TLE_LINE2},
            search_start="2024-01-01 00:00:00 UTC",
            search_end="2024-01-02 00:00:00 UTC",
        )
        return result["windows"]

    def test_expected_keys(self, sample_windows):
        if not sample_windows:
            pytest.skip("No access windows found for test data")
        w = sample_windows[0]
        expected_keys = {
            "window_open", "window_close", "duration_seconds",
            "elevation_max_deg", "elevation_open_deg", "elevation_close_deg",
            "azimuth_open_deg", "azimuth_close_deg",
            "off_nadir_min_deg", "off_nadir_max_deg",
            "look_direction", "asc_dsc",
            "location_name", "satellite_name",
        }
        assert expected_keys.issubset(set(w.keys()))

    def test_values_reasonable(self, sample_windows):
        if not sample_windows:
            pytest.skip("No access windows found for test data")
        w = sample_windows[0]
        assert w["duration_seconds"] > 0
        assert 0 <= w["elevation_max_deg"] <= 90
        assert 0 <= w["azimuth_open_deg"] <= 360
        assert 0 <= w["azimuth_close_deg"] <= 360
        assert w["look_direction"] in ("Left", "Right", "Either")
        assert w["asc_dsc"] in ("Ascending", "Descending", "Either")
