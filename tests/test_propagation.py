import pytest
import math

from brahe_mcp.propagation import (
    list_propagation_options,
    propagate_sgp4,
    propagate_keplerian,
    propagate_numerical,
    propagate_from_gp_record,
)


# Valid ISS-like TLE for testing
TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30000-3 0  9002"
TLE_LINE2 = "2 25544  51.6400 150.0000 0003000 100.0000 260.0000 15.50000000    18"

# Common test state and epoch
TEST_EPOCH = "2024-01-01 12:00:00 UTC"
TEST_STATE_ECI = [7000000.0, 0.0, 0.0, 0.0, 7546.0, 0.0]
TEST_KOE = [7000000.0, 0.001, 98.0, 45.0, 0.0, 0.0]

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
    "semimajor_axis": 6801.0,  # km
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


# ---------------------------------------------------------------------------
# Discovery tool
# ---------------------------------------------------------------------------

def test_list_propagation_options():
    result = list_propagation_options()
    assert "propagator_types" in result
    assert "output_frames" in result
    assert "force_model_presets" in result
    assert "convenience_tools" in result
    types = [t["type"] for t in result["propagator_types"]]
    assert "sgp4" in types
    assert "keplerian" in types
    assert "numerical" in types
    assert "eci" in result["output_frames"]
    assert "koe_osc" in result["output_frames"]
    assert "spacecraft_params_format" in result


# ---------------------------------------------------------------------------
# SGP4 propagation
# ---------------------------------------------------------------------------

class TestSGP4:
    def test_single_epoch(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            target_epoch="2024-01-02 12:00:00 UTC",
        )
        assert result["propagator_type"] == "sgp4"
        assert "state" in result
        assert len(result["state"]["vector"]) == 6
        assert result["output_frame"] == "eci"

    def test_time_range(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            start_epoch="2024-01-01 12:00:00 UTC",
            end_epoch="2024-01-01 12:05:00 UTC",
            step_seconds=60.0,
        )
        assert "states" in result
        assert result["count"] == 6  # 0, 60, 120, 180, 240, 300
        assert len(result["states"][0]["vector"]) == 6

    def test_ecef_output(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            target_epoch="2024-01-02 12:00:00 UTC",
            output_frame="ecef",
        )
        assert result["output_frame"] == "ecef"
        assert "x_m" in result["state"]["components"]

    def test_koe_output(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            target_epoch="2024-01-02 12:00:00 UTC",
            output_frame="koe_osc",
        )
        assert result["output_frame"] == "koe_osc"
        assert "a_m" in result["state"]["components"]
        assert "e" in result["state"]["components"]

    def test_invalid_tle(self):
        result = propagate_sgp4("bad line 1", "bad line 2", target_epoch="2024-01-02T00:00:00Z")
        assert "error" in result

    def test_invalid_output_frame(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            target_epoch="2024-01-02T00:00:00Z",
            output_frame="invalid",
        )
        assert "error" in result
        assert "valid_frames" in result


# ---------------------------------------------------------------------------
# Keplerian propagation
# ---------------------------------------------------------------------------

class TestKeplerian:
    def test_from_eci_state(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
        )
        assert result["propagator_type"] == "keplerian"
        assert "state" in result
        assert len(result["state"]["vector"]) == 6

    def test_from_koe_elements(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            elements_koe=TEST_KOE,
            target_epoch="2024-01-01 13:00:00 UTC",
        )
        assert result["propagator_type"] == "keplerian"
        assert "state" in result

    def test_koe_output_frame(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            elements_koe=TEST_KOE,
            target_epoch="2024-01-01 13:00:00 UTC",
            output_frame="koe_osc",
        )
        assert result["output_frame"] == "koe_osc"
        comps = result["state"]["components"]
        # Semi-major axis should be ~7000km
        assert abs(comps["a_m"] - 7000000.0) < 100.0

    def test_time_range(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            start_epoch="2024-01-01 12:00:00 UTC",
            end_epoch="2024-01-01 12:03:00 UTC",
            step_seconds=60.0,
        )
        assert result["count"] == 4

    def test_missing_state_input(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            target_epoch="2024-01-01 13:00:00 UTC",
        )
        assert "error" in result

    def test_wrong_state_length(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            state_eci=[1, 2, 3],
            target_epoch="2024-01-01 13:00:00 UTC",
        )
        assert "error" in result
        assert "6 elements" in result["error"]

    def test_wrong_koe_length(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            elements_koe=[7000000.0, 0.001],
            target_epoch="2024-01-01 13:00:00 UTC",
        )
        assert "error" in result
        assert "6 elements" in result["error"]

    def test_radians_input(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            elements_koe=[7000000.0, 0.001, math.radians(98.0), math.radians(45.0), 0.0, 0.0],
            input_angle_format="radians",
            target_epoch="2024-01-01 13:00:00 UTC",
            output_frame="koe_osc",
        )
        assert "state" in result
        # i should be ~98 degrees
        assert abs(result["state"]["components"]["i"] - 98.0) < 0.01


# ---------------------------------------------------------------------------
# Numerical propagation
# ---------------------------------------------------------------------------

class TestNumerical:
    def test_two_body(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="two_body",
        )
        assert result["propagator_type"] == "numerical"
        assert "state" in result
        assert len(result["state"]["vector"]) == 6

    def test_earth_gravity(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="earth_gravity",
        )
        assert "state" in result

    def test_two_body_vs_keplerian_consistency(self):
        """Numerical two-body should produce similar orbit shape to Keplerian."""
        # Use a non-planar orbit to avoid sign ambiguity in the planar case
        state = [6778137.0, 0.0, 0.0, 0.0, 5400.0, 5400.0]
        r_kep = propagate_keplerian(
            epoch=TEST_EPOCH,
            state_eci=state,
            target_epoch="2024-01-01 12:10:00 UTC",
            output_frame="koe_osc",
        )
        r_num = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=state,
            target_epoch="2024-01-01 12:10:00 UTC",
            force_model="two_body",
            output_frame="koe_osc",
        )
        kep_koe = r_kep["state"]["vector"]
        num_koe = r_num["state"]["vector"]
        # Semi-major axis should match closely
        assert abs(kep_koe[0] - num_koe[0]) < 100.0  # Within 100m
        # Eccentricity should match
        assert abs(kep_koe[1] - num_koe[1]) < 0.001

    def test_missing_params_for_default(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="default",
        )
        assert "error" in result
        assert "spacecraft_params" in result["error"]

    def test_invalid_force_model(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="nonexistent",
        )
        assert "error" in result
        assert "valid_presets" in result

    def test_granular_overrides(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="default",
            spacecraft_params=[1000.0, 10.0, 2.2, 10.0, 1.3],
            drag_model="harris_priester",
            gravity_degree=10,
            gravity_order=10,
        )
        assert "state" in result

    def test_disable_drag(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="default",
            spacecraft_params=[1000.0, 10.0, 2.2, 10.0, 1.3],
            drag_model="none",
        )
        assert "state" in result

    def test_wrong_state_length(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=[1, 2, 3],
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="two_body",
        )
        assert "error" in result
        assert "6 elements" in result["error"]

    def test_time_range(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            start_epoch="2024-01-01 12:00:00 UTC",
            end_epoch="2024-01-01 12:03:00 UTC",
            step_seconds=60.0,
            force_model="two_body",
        )
        assert result["count"] == 4

    def test_conservative_forces(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="conservative_forces",
        )
        assert "state" in result

    def test_invalid_drag_model(self):
        result = propagate_numerical(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            force_model="two_body",
            drag_model="invalid_model",
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# Propagate from GP record
# ---------------------------------------------------------------------------

class TestFromGPRecord:
    def test_sgp4_from_gp(self):
        result = propagate_from_gp_record(
            FAKE_GP_RECORD,
            propagator_type="sgp4",
            target_epoch="2024-01-02 12:00:00 UTC",
        )
        assert result["propagator_type"] == "sgp4"
        assert "state" in result

    def test_keplerian_from_gp(self):
        result = propagate_from_gp_record(
            FAKE_GP_RECORD,
            propagator_type="keplerian",
            target_epoch="2024-01-02 12:00:00 UTC",
        )
        assert result["propagator_type"] == "keplerian"
        assert "state" in result

    def test_numerical_from_gp(self):
        result = propagate_from_gp_record(
            FAKE_GP_RECORD,
            propagator_type="numerical",
            target_epoch="2024-01-02 12:00:00 UTC",
            force_model="two_body",
        )
        assert result["propagator_type"] == "numerical"
        assert "state" in result

    def test_invalid_propagator_type(self):
        result = propagate_from_gp_record(
            FAKE_GP_RECORD,
            propagator_type="invalid",
        )
        assert "error" in result
        assert "valid_types" in result

    def test_missing_omm_fields_for_sgp4(self):
        record = {**FAKE_GP_RECORD, "mean_motion": None, "eccentricity": None}
        result = propagate_from_gp_record(
            record,
            propagator_type="sgp4",
            target_epoch="2024-01-02 12:00:00 UTC",
        )
        assert "error" in result
        assert "OMM" in result["error"]

    def test_gp_record_time_range(self):
        result = propagate_from_gp_record(
            FAKE_GP_RECORD,
            propagator_type="sgp4",
            start_epoch="2024-01-01 12:00:00 UTC",
            end_epoch="2024-01-01 12:05:00 UTC",
            step_seconds=60.0,
        )
        assert result["count"] == 6


# ---------------------------------------------------------------------------
# Epoch building edge cases
# ---------------------------------------------------------------------------

class TestEpochBuilding:
    def test_no_target_or_range(self):
        result = propagate_sgp4(TLE_LINE1, TLE_LINE2)
        assert "error" in result

    def test_only_start_no_end(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            start_epoch="2024-01-01 12:00:00 UTC",
        )
        assert "error" in result

    def test_invalid_angle_format(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            target_epoch="2024-01-02T00:00:00Z",
            angle_format="grads",
        )
        assert "error" in result

    def test_end_before_start(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            start_epoch="2024-01-02 12:00:00 UTC",
            end_epoch="2024-01-01 12:00:00 UTC",
        )
        assert "error" in result

    def test_zero_step_seconds(self):
        result = propagate_sgp4(
            TLE_LINE1, TLE_LINE2,
            start_epoch="2024-01-01 12:00:00 UTC",
            end_epoch="2024-01-01 12:05:00 UTC",
            step_seconds=0.0,
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# Output frame tests
# ---------------------------------------------------------------------------

class TestOutputFrames:
    @pytest.mark.parametrize("frame", ["eci", "ecef", "gcrf", "itrf", "eme2000", "koe_osc", "koe_mean"])
    def test_all_output_frames(self, frame):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            output_frame=frame,
        )
        assert "state" in result
        assert result["output_frame"] == frame
        assert len(result["state"]["vector"]) == 6


# ---------------------------------------------------------------------------
# Component labels
# ---------------------------------------------------------------------------

class TestComponentLabels:
    def test_eci_labels(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            output_frame="eci",
        )
        comps = result["state"]["components"]
        for key in ["x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"]:
            assert key in comps

    def test_koe_labels(self):
        result = propagate_keplerian(
            epoch=TEST_EPOCH,
            state_eci=TEST_STATE_ECI,
            target_epoch="2024-01-01 13:00:00 UTC",
            output_frame="koe_osc",
        )
        comps = result["state"]["components"]
        for key in ["a_m", "e", "i", "RAAN", "omega", "M"]:
            assert key in comps
