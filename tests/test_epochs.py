from brahe_mcp.epochs import list_time_systems, convert_epoch


def test_list_time_systems():
    result = list_time_systems()
    assert "time_systems" in result
    names = [ts["name"] for ts in result["time_systems"]]
    for expected in ["UTC", "GPS", "TAI", "TT", "UT1"]:
        assert expected in names
    for ts in result["time_systems"]:
        assert "description" in ts


def test_iso_to_mjd():
    result = convert_epoch("2024-01-01T12:00:00Z", input_format="iso", output_format="mjd")
    assert "output" in result
    assert abs(result["output"]["value"] - 60310.5) < 0.001


def test_mjd_to_iso():
    result = convert_epoch("60310.5", input_format="mjd", output_format="iso")
    assert "output" in result
    assert "2024-01-01" in result["output"]["value"]


def test_jd_to_mjd():
    result = convert_epoch("2460311.0", input_format="jd", output_format="mjd")
    assert "output" in result
    assert abs(result["output"]["value"] - 60310.5) < 0.001


def test_mjd_to_gps_seconds():
    result = convert_epoch("60310.5", input_format="mjd", output_format="gps_seconds")
    assert "output" in result
    assert isinstance(result["output"]["value"], float)


def test_time_system_utc_to_gps():
    result = convert_epoch(
        "2024-01-01T12:00:00Z",
        input_format="iso",
        input_time_system="UTC",
        output_format="string",
        output_time_system="GPS",
    )
    assert "output" in result
    assert "GPS" in result["output"]["value"]


def test_iso_without_tz_defaults_to_utc():
    result = convert_epoch("2024-01-01 12:00:00", input_format="iso", output_format="mjd")
    assert "output" in result
    assert result["output"]["time_system"] == "UTC"
    assert abs(result["output"]["value"] - 60310.5) < 0.001


def test_iso_with_gps_suffix():
    result = convert_epoch("2024-01-01 12:00:00 GPS", input_format="iso", output_format="mjd")
    assert "output" in result
    assert result["output"]["time_system"] == "GPS"


def test_gps_seconds_to_iso():
    result = convert_epoch("1388145618.0", input_format="gps_seconds", output_format="iso")
    assert "output" in result
    assert "2024" in result["output"]["value"]


def test_gps_date_roundtrip():
    # First convert to gps_date
    result1 = convert_epoch("2024-01-01T12:00:00Z", input_format="iso", output_format="gps_date")
    assert "output" in result1
    gps_date_val = result1["output"]["value"]
    week = gps_date_val["week"]
    seconds = gps_date_val["seconds"]

    # Convert back
    result2 = convert_epoch(f"{week},{seconds}", input_format="gps_date", output_format="gps_date")
    assert "output" in result2
    assert result2["output"]["value"]["week"] == week
    assert abs(result2["output"]["value"]["seconds"] - seconds) < 0.01


def test_iso_precise_output():
    result = convert_epoch("2024-01-01T12:00:00Z", input_format="iso", output_format="iso_precise")
    assert "output" in result
    # Should have decimal seconds (more than 3 decimal places)
    assert "." in result["output"]["value"]


def test_output_ts_defaults_to_input():
    result = convert_epoch("60310.5", input_format="mjd", input_time_system="GPS", output_format="mjd")
    assert "output" in result
    assert result["output"]["time_system"] == "GPS"


def test_invalid_input_format():
    result = convert_epoch("2024-01-01T12:00:00Z", input_format="bad_format")
    assert "error" in result
    assert "valid_input_formats" in result


def test_invalid_time_system():
    result = convert_epoch("2024-01-01T12:00:00Z", input_time_system="INVALID")
    assert "error" in result
    assert "valid_time_systems" in result


def test_invalid_value():
    result = convert_epoch("abc", input_format="mjd")
    assert "error" in result
