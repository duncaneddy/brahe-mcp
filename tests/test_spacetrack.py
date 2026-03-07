"""Tests for SpaceTrack MCP tools."""

from unittest.mock import MagicMock

import pytest

import brahe_mcp.spacetrack as mod
from brahe_mcp.spacetrack import (
    get_spacetrack_cdm,
    get_spacetrack_decay,
    get_spacetrack_gp,
    get_spacetrack_gp_history,
    get_spacetrack_satcat,
    list_spacetrack_options,
    query_spacetrack,
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
    """Create a mock SpaceTrack SATCATRecord with default field values."""
    defaults = {
        "intldes": "2024-001A",
        "norad_cat_id": 99999,
        "object_type": "PAY",
        "satname": "TEST SAT",
        "country": "US",
        "launch": "2024-01-01",
        "site": "AFETR",
        "decay": None,
        "period": 92.0,
        "inclination": 51.6,
        "apogee": 420,
        "perigee": 410,
        "rcsvalue": 1.5,
        "rcs_size": "LARGE",
        "object_name": "TEST SAT",
        "object_id": "2024-001A",
        "object_number": 99999,
    }
    defaults.update(overrides)
    rec = MagicMock()
    for k, v in defaults.items():
        setattr(rec, k, v)
    return rec


def _mock_client(**method_returns):
    """Create a mock SpaceTrackClient."""
    client = MagicMock()
    for method_name, return_value in method_returns.items():
        getattr(client, method_name).return_value = return_value
    return client


# --- list_spacetrack_options ---


def test_list_spacetrack_options():
    result = list_spacetrack_options()
    assert "tools" in result
    assert "request_classes" in result
    assert "fields" in result
    assert "operators" in result
    assert "sort_orders" in result
    assert "gp" in result["request_classes"]
    assert "gp_history" in result["request_classes"]
    assert "satcat" in result["request_classes"]
    assert "cdm_public" in result["request_classes"]
    assert "decay" in result["request_classes"]


# --- get_spacetrack_gp ---


def test_get_gp_no_filters():
    result = get_spacetrack_gp()
    assert "error" in result
    assert "At least one filter" in result["error"]


def test_get_gp_by_norad_cat_id(monkeypatch):
    mock = _mock_client(query_gp=[_make_gp_record(norad_cat_id=25544, object_name="ISS")])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = get_spacetrack_gp(norad_cat_id=25544)
    assert result["request_class"] == "gp"
    assert result["count"] == 1
    assert result["records"][0]["norad_cat_id"] == 25544
    assert result["records"][0]["object_name"] == "ISS"


def test_get_gp_by_name(monkeypatch):
    mock = _mock_client(query_gp=[_make_gp_record(object_name="STARLINK-1000")])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = get_spacetrack_gp(name="STARLINK")
    assert result["count"] == 1
    assert result["records"][0]["object_name"] == "STARLINK-1000"


def test_get_gp_with_limit(monkeypatch):
    mock = _mock_client(query_gp=[_make_gp_record() for _ in range(5)])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = get_spacetrack_gp(norad_cat_id=25544, limit=3)
    # Limit is applied at the query level, mock returns 5 but query should have .limit()
    assert result["request_class"] == "gp"
    assert "error" not in result


def test_get_gp_no_credentials(monkeypatch):
    monkeypatch.setattr(mod, "_get_client", lambda: (_ for _ in ()).throw(
        RuntimeError("Set SPACETRACK_USER and SPACETRACK_PASS environment variables.")
    ))

    result = get_spacetrack_gp(norad_cat_id=25544)
    assert "error" in result
    assert "SPACETRACK_USER" in result["error"]


# --- get_spacetrack_gp_history ---


def test_get_gp_history_no_filters():
    result = get_spacetrack_gp_history()
    assert "error" in result
    assert "At least one filter" in result["error"]


def test_get_gp_history_by_norad_cat_id(monkeypatch):
    mock = _mock_client(query_gp=[
        _make_gp_record(norad_cat_id=25544, epoch="2024-01-01T00:00:00"),
        _make_gp_record(norad_cat_id=25544, epoch="2024-01-02T00:00:00"),
    ])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = get_spacetrack_gp_history(norad_cat_id=25544)
    assert result["request_class"] == "gp_history"
    assert result["count"] == 2


def test_get_gp_history_with_epoch_range(monkeypatch):
    mock = _mock_client(query_gp=[_make_gp_record()])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = get_spacetrack_gp_history(
        norad_cat_id=25544,
        epoch_range="2024-01-01--2024-01-31",
    )
    assert result["request_class"] == "gp_history"
    assert "error" not in result


# --- get_spacetrack_satcat ---


def test_get_satcat_no_filters():
    result = get_spacetrack_satcat()
    assert "error" in result
    assert "At least one filter" in result["error"]


def test_get_satcat_by_norad_cat_id(monkeypatch):
    mock = _mock_client(query_satcat=[_make_satcat_record(norad_cat_id=25544)])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = get_spacetrack_satcat(norad_cat_id=25544)
    assert result["request_class"] == "satcat"
    assert result["count"] == 1
    assert result["records"][0]["norad_cat_id"] == 25544
    assert result["records"][0]["satname"] == "TEST SAT"


# --- get_spacetrack_cdm ---


def test_get_cdm_no_filters():
    result = get_spacetrack_cdm()
    assert "error" in result
    assert "At least one filter" in result["error"]


def test_get_cdm_by_norad_cat_id(monkeypatch):
    cdm_record = {"CDM_ID": "123", "SAT_1_ID": "25544", "TCA": "2024-06-15T12:00:00"}
    mock = _mock_client(query_json=[cdm_record])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = get_spacetrack_cdm(norad_cat_id=25544)
    assert result["request_class"] == "cdm_public"
    assert result["count"] == 1
    assert result["records"][0]["CDM_ID"] == "123"


# --- get_spacetrack_decay ---


def test_get_decay_no_filters():
    result = get_spacetrack_decay()
    assert "error" in result
    assert "At least one filter" in result["error"]


def test_get_decay_by_norad_cat_id(monkeypatch):
    decay_record = {"NORAD_CAT_ID": "25544", "DECAY_EPOCH": "2024-06-15", "SOURCE": "AFSPC"}
    mock = _mock_client(query_json=[decay_record])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = get_spacetrack_decay(norad_cat_id=25544)
    assert result["request_class"] == "decay"
    assert result["count"] == 1
    assert result["records"][0]["NORAD_CAT_ID"] == "25544"


# --- query_spacetrack ---


def test_query_invalid_request_class():
    result = query_spacetrack(request_class="invalid")
    assert "error" in result
    assert "valid_classes" in result
    assert "gp" in result["valid_classes"]


def test_query_gp_with_filters(monkeypatch):
    mock = _mock_client(query_gp=[_make_gp_record(inclination=55.0)])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = query_spacetrack(
        request_class="gp",
        filters=[{"field": "NORAD_CAT_ID", "value": "25544"}],
    )
    assert result["request_class"] == "gp"
    assert result["count"] == 1
    assert result["records"][0]["inclination"] == 55.0


def test_query_gp_history(monkeypatch):
    mock = _mock_client(query_gp=[_make_gp_record(), _make_gp_record()])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = query_spacetrack(
        request_class="gp_history",
        filters=[{"field": "NORAD_CAT_ID", "value": "25544"}],
    )
    assert result["request_class"] == "gp_history"
    assert result["count"] == 2


def test_query_satcat(monkeypatch):
    mock = _mock_client(query_satcat=[_make_satcat_record()])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = query_spacetrack(
        request_class="satcat",
        filters=[{"field": "COUNTRY", "value": "US"}],
    )
    assert result["request_class"] == "satcat"
    assert result["count"] == 1
    assert result["records"][0]["object_type"] == "PAY"


def test_query_cdm(monkeypatch):
    cdm_record = {"CDM_ID": "456", "SAT_1_ID": "25544"}
    mock = _mock_client(query_json=[cdm_record])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = query_spacetrack(
        request_class="cdm_public",
        filters=[{"field": "SAT_1_ID", "value": "25544"}],
    )
    assert result["request_class"] == "cdm_public"
    assert result["count"] == 1
    assert result["records"][0]["CDM_ID"] == "456"


def test_query_with_ordering(monkeypatch):
    mock = _mock_client(query_gp=[_make_gp_record()])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = query_spacetrack(
        request_class="gp",
        filters=[{"field": "NORAD_CAT_ID", "value": "25544"}],
        order_by="EPOCH",
        order_ascending=False,
    )
    assert result["request_class"] == "gp"
    assert "error" not in result


def test_query_with_limit_offset(monkeypatch):
    mock = _mock_client(query_gp=[_make_gp_record()])
    monkeypatch.setattr(mod, "_get_client", lambda: mock)

    result = query_spacetrack(
        request_class="gp",
        filters=[{"field": "NORAD_CAT_ID", "value": "25544"}],
        limit=10,
        offset=5,
    )
    assert result["request_class"] == "gp"
    assert "error" not in result


def test_query_no_credentials(monkeypatch):
    monkeypatch.setattr(mod, "_get_client", lambda: (_ for _ in ()).throw(
        RuntimeError("Set SPACETRACK_USER and SPACETRACK_PASS environment variables.")
    ))

    result = query_spacetrack(
        request_class="gp",
        filters=[{"field": "NORAD_CAT_ID", "value": "25544"}],
    )
    assert "error" in result
    assert "SPACETRACK_USER" in result["error"]
