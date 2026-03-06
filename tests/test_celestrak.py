"""Tests for CelesTrak MCP tools."""

from unittest.mock import MagicMock

import pytest

import brahe_mcp.celestrak as mod
from brahe_mcp.celestrak import (
    get_celestrak_gp,
    get_celestrak_satcat,
    get_celestrak_sup_gp,
    list_celestrak_options,
    query_celestrak,
)


def _make_gp_record(**overrides):
    """Create a mock GPRecord with default field values."""
    defaults = {
        "object_name": "TEST SAT",
        "norad_cat_id": 99999,
        "object_id": "2024-001A",
        "epoch": "2024-01-01T00:00:00",
        "inclination": 51.6,
        "eccentricity": 0.001,
        "ra_of_asc_node": 200.0,
        "arg_of_pericenter": 100.0,
        "mean_anomaly": 50.0,
        "mean_motion": 15.5,
        "bstar": 0.0001,
        "semimajor_axis": 6800.0,
        "period": 92.0,
        "apoapsis": 420.0,
        "periapsis": 410.0,
        "classification_type": "U",
        "object_type": "PAYLOAD",
        "country_code": "US",
        "launch_date": "2024-01-01",
        "decay_date": None,
        "tle_line0": "0 TEST SAT",
        "tle_line1": "1 99999U ...",
        "tle_line2": "2 99999 ...",
    }
    defaults.update(overrides)
    rec = MagicMock()
    for k, v in defaults.items():
        setattr(rec, k, v)
    return rec


def _make_satcat_record(**overrides):
    """Create a mock CelestrakSATCATRecord with default field values."""
    defaults = {
        "object_name": "TEST SAT",
        "object_id": "2024-001A",
        "norad_cat_id": 99999,
        "object_type": "PAY",
        "ops_status_code": "+",
        "owner": "US",
        "launch_date": "2024-01-01",
        "launch_site": "AFETR",
        "decay_date": None,
        "period": "92.0",
        "inclination": "51.6",
        "apogee": "420",
        "perigee": "410",
        "rcs": "LARGE",
        "orbit_center": "EA",
        "orbit_type": "ORB",
    }
    defaults.update(overrides)
    rec = MagicMock()
    for k, v in defaults.items():
        setattr(rec, k, v)
    return rec


def _mock_client(**method_returns):
    """Create a mock CelestrakClient replacing the module-level singleton."""
    client = MagicMock()
    for method_name, return_value in method_returns.items():
        getattr(client, method_name).return_value = return_value
    return client


# --- list_celestrak_options ---


def test_list_celestrak_options():
    result = list_celestrak_options()
    assert "lookup_methods" in result
    assert "gp_groups" in result
    assert "supplemental_sources" in result
    assert "satcat_options" in result
    assert "filter_operators" in result
    assert "gp_record_fields" in result
    assert len(result["supplemental_sources"]) == 20
    assert "starlink" in result["supplemental_sources"]


# --- get_celestrak_gp ---


def test_get_gp_no_identifier():
    result = get_celestrak_gp()
    assert "error" in result
    assert "exactly one identifier" in result["error"]
    assert "valid_groups" in result


def test_get_gp_multiple_identifiers():
    result = get_celestrak_gp(catnr=25544, name="ISS")
    assert "error" in result
    assert "exactly one identifier" in result["error"]


def test_get_gp_by_catnr(monkeypatch):
    mock_records = [_make_gp_record(norad_cat_id=25544, object_name="ISS (ZARYA)")]
    monkeypatch.setattr(mod, "_client", _mock_client(get_gp=mock_records))

    result = get_celestrak_gp(catnr=25544)
    assert result["method"] == "catnr"
    assert result["identifier"] == 25544
    assert result["count"] == 1
    assert result["records"][0]["object_name"] == "ISS (ZARYA)"
    assert result["records"][0]["norad_cat_id"] == 25544


def test_get_gp_by_group(monkeypatch):
    mock_records = [_make_gp_record(), _make_gp_record(object_name="SAT 2")]
    monkeypatch.setattr(mod, "_client", _mock_client(get_gp=mock_records))

    result = get_celestrak_gp(group="stations")
    assert result["method"] == "group"
    assert result["identifier"] == "stations"
    assert result["count"] == 2


def test_get_gp_by_name(monkeypatch):
    mock_records = [_make_gp_record(object_name="ISS (ZARYA)")]
    monkeypatch.setattr(mod, "_client", _mock_client(get_gp=mock_records))

    result = get_celestrak_gp(name="ISS")
    assert result["method"] == "name"
    assert result["identifier"] == "ISS"
    assert result["count"] == 1


def test_get_gp_with_limit(monkeypatch):
    mock_records = [_make_gp_record() for _ in range(10)]
    monkeypatch.setattr(mod, "_client", _mock_client(get_gp=mock_records))

    result = get_celestrak_gp(group="active", limit=3)
    assert result["count"] == 3


# --- get_celestrak_sup_gp ---


def test_get_sup_gp_invalid_source():
    result = get_celestrak_sup_gp(source="nonexistent")
    assert "error" in result
    assert "valid_sources" in result
    assert "starlink" in result["valid_sources"]


def test_get_sup_gp_valid(monkeypatch):
    mock_records = [_make_gp_record(object_name="STARLINK-1000")]
    monkeypatch.setattr(mod, "_client", _mock_client(get_sup_gp=mock_records))

    result = get_celestrak_sup_gp(source="starlink")
    assert result["source"] == "starlink"
    assert result["count"] == 1
    assert result["records"][0]["object_name"] == "STARLINK-1000"


def test_get_sup_gp_case_insensitive(monkeypatch):
    monkeypatch.setattr(mod, "_client", _mock_client(get_sup_gp=[]))

    result = get_celestrak_sup_gp(source="STARLINK")
    assert result["source"] == "starlink"
    assert "error" not in result


# --- get_celestrak_satcat ---


def test_get_satcat_no_filters():
    result = get_celestrak_satcat()
    assert "error" in result
    assert "At least one parameter" in result["error"]


def test_get_satcat_by_catnr(monkeypatch):
    mock_records = [_make_satcat_record(norad_cat_id=25544, object_name="ISS (ZARYA)")]
    monkeypatch.setattr(mod, "_client", _mock_client(get_satcat=mock_records))

    result = get_celestrak_satcat(catnr=25544)
    assert result["count"] == 1
    assert result["records"][0]["object_name"] == "ISS (ZARYA)"
    assert result["filters"] == {"catnr": 25544}


def test_get_satcat_multiple_filters(monkeypatch):
    mock_records = [_make_satcat_record()]
    monkeypatch.setattr(mod, "_client", _mock_client(get_satcat=mock_records))

    result = get_celestrak_satcat(active=True, payloads=True)
    assert result["filters"] == {"active": True, "payloads": True}


# --- query_celestrak ---


def test_query_invalid_type():
    result = query_celestrak(query_type="invalid")
    assert "error" in result
    assert "valid_types" in result
    assert "gp" in result["valid_types"]


def test_query_gp_with_filters(monkeypatch):
    mock_records = [_make_gp_record(inclination=55.0)]
    monkeypatch.setattr(mod, "_client", _mock_client(query=mock_records))

    result = query_celestrak(
        query_type="gp",
        group="active",
        filters=[{"field": "INCLINATION", "value": ">50"}],
        limit=5,
    )
    assert result["query_type"] == "gp"
    assert result["count"] == 1
    assert result["records"][0]["inclination"] == 55.0


def test_query_sup_gp_missing_source():
    result = query_celestrak(query_type="sup_gp")
    assert "error" in result
    assert "source is required" in result["error"]


def test_query_sup_gp_invalid_source():
    result = query_celestrak(query_type="sup_gp", source="fake")
    assert "error" in result
    assert "valid_sources" in result


def test_query_sup_gp_valid(monkeypatch):
    mock_records = [_make_gp_record()]
    monkeypatch.setattr(mod, "_client", _mock_client(query=mock_records))

    result = query_celestrak(query_type="sup_gp", source="spacex")
    assert result["query_type"] == "sup_gp"
    assert result["count"] == 1


def test_query_satcat(monkeypatch):
    mock_records = [_make_satcat_record()]
    monkeypatch.setattr(mod, "_client", _mock_client(query=mock_records))

    result = query_celestrak(query_type="satcat", active=True)
    assert result["query_type"] == "satcat"
    assert result["count"] == 1
    assert result["records"][0]["object_type"] == "PAY"


def test_query_with_ordering(monkeypatch):
    mock_records = [_make_gp_record()]
    monkeypatch.setattr(mod, "_client", _mock_client(query=mock_records))

    result = query_celestrak(
        query_type="gp",
        group="active",
        order_by="INCLINATION",
        order_ascending=False,
    )
    assert result["query_type"] == "gp"
    assert "error" not in result
