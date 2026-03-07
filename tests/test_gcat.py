"""Tests for GCAT MCP tools."""

from unittest.mock import MagicMock

import pytest

import brahe_mcp.gcat as mod
from brahe_mcp.gcat import (
    get_gcat_psatcat,
    get_gcat_satcat,
    list_gcat_options,
    query_gcat_psatcat,
    query_gcat_satcat,
)


def _make_satcat_record(**overrides):
    """Create a mock GCATSatcatRecord with default field values."""
    defaults = {
        "jcat": "S00001",
        "satcat": "00001",
        "piece": "1957 ALP 1",
        "object_type": "R2",
        "name": "8K71PS Stage 2",
        "pl_name": "8K71A M1-10",
        "ldate": "1957 Oct 4",
        "ddate": "1957 Dec 1",
        "status": "R",
        "dest": None,
        "owner": "OKB1",
        "state": "SU",
        "manufacturer": "OKB1",
        "mass": 7790.0,
        "dry_mass": 7790.0,
        "perigee": 214.0,
        "apogee": 938.0,
        "inc": 65.1,
        "op_orbit": "LLEO/I",
        "alt_names": None,
    }
    defaults.update(overrides)
    rec = MagicMock()
    for k, v in defaults.items():
        setattr(rec, k, v)
    return rec


def _make_psatcat_record(**overrides):
    """Create a mock GCATPsatcatRecord with default field values."""
    defaults = {
        "jcat": "S00049",
        "name": "Echo",
        "piece": "1960 IOT 1",
        "ldate": "1960 Aug 12",
        "category": "COM",
        "class_": "C",
        "program": "Echo",
        "result": "S",
        "comment": None,
        "tdate": "1960?",
        "top": "1960?",
        "tlast": "1960?",
        "att": None,
        "control": None,
        "discipline": None,
        "mvr": None,
        "plane": None,
        "tf": None,
        "disp_epoch": "1960 Aug 16",
        "disp_peri": 1517.0,
        "disp_apo": 1692.0,
        "disp_inc": 47.2,
        "un_reg": "A/AC.105/INF.001",
        "un_state": "US",
        "un_period": 116.2,
        "un_perigee": 1059.0,
        "un_apogee": 1966.0,
        "un_inc": 47.3,
    }
    defaults.update(overrides)
    rec = MagicMock()
    for k, v in defaults.items():
        setattr(rec, k, v)
    return rec


def _make_satcat_container(records, **filter_returns):
    """Create a mock GCATSatcat container with chainable filter methods."""
    container = MagicMock()
    container.records.return_value = records
    container.get_by_jcat = MagicMock(side_effect=lambda j: next(r for r in records if r.jcat == j))
    container.get_by_satcat = MagicMock(side_effect=lambda s: next(r for r in records if r.satcat == s))

    # search_by_name and filter methods return new containers
    def _make_sub(recs):
        sub = MagicMock()
        sub.records.return_value = recs
        sub.search_by_name = MagicMock(return_value=sub)
        sub.filter_by_type = MagicMock(return_value=sub)
        sub.filter_by_owner = MagicMock(return_value=sub)
        sub.filter_by_state = MagicMock(return_value=sub)
        sub.filter_by_status = MagicMock(return_value=sub)
        sub.filter_by_perigee_range = MagicMock(return_value=sub)
        sub.filter_by_apogee_range = MagicMock(return_value=sub)
        sub.filter_by_inc_range = MagicMock(return_value=sub)
        return sub

    sub = _make_sub(filter_returns.get("filtered", records))
    container.search_by_name = MagicMock(return_value=sub)
    container.filter_by_type = MagicMock(return_value=sub)
    container.filter_by_owner = MagicMock(return_value=sub)
    container.filter_by_state = MagicMock(return_value=sub)
    container.filter_by_status = MagicMock(return_value=sub)
    container.filter_by_perigee_range = MagicMock(return_value=sub)
    container.filter_by_apogee_range = MagicMock(return_value=sub)
    container.filter_by_inc_range = MagicMock(return_value=sub)
    return container


def _make_psatcat_container(records, **filter_returns):
    """Create a mock GCATPsatcat container with chainable filter methods."""
    container = MagicMock()
    container.records.return_value = records
    container.get_by_jcat = MagicMock(side_effect=lambda j: next(r for r in records if r.jcat == j))

    sub = MagicMock()
    sub.records.return_value = filter_returns.get("filtered", records)
    sub.search_by_name = MagicMock(return_value=sub)
    sub.filter_by_category = MagicMock(return_value=sub)
    sub.filter_by_class = MagicMock(return_value=sub)
    sub.filter_by_result = MagicMock(return_value=sub)
    sub.filter_active = MagicMock(return_value=sub)

    container.search_by_name = MagicMock(return_value=sub)
    container.filter_by_category = MagicMock(return_value=sub)
    container.filter_by_class = MagicMock(return_value=sub)
    container.filter_by_result = MagicMock(return_value=sub)
    container.filter_active = MagicMock(return_value=sub)
    return container


# --- list_gcat_options ---


def test_list_gcat_options():
    result = list_gcat_options()
    assert "catalog_types" in result
    assert "lookup_methods" in result
    assert "satcat_field_descriptions" in result
    assert "psatcat_field_descriptions" in result
    assert "satcat_filters" in result
    assert "psatcat_filters" in result
    assert "satcat_status_codes" in result
    assert "satcat_object_types" in result
    assert "satcat_op_orbit_codes" in result
    assert "psatcat_category_codes" in result
    assert "psatcat_result_codes" in result
    assert "psatcat_class_codes" in result
    assert "satcat" in result["catalog_types"]
    assert "psatcat" in result["catalog_types"]
    # Verify hierarchical grouping
    assert "in_orbit" in result["satcat_status_codes"]
    assert result["satcat_status_codes"]["in_orbit"]["O"] == "Operational"
    assert "payloads" in result["satcat_object_types"]
    assert "rocket_bodies" in result["satcat_object_types"]
    assert "common_suffixes" in result["satcat_object_types"]
    assert "low_earth_orbit" in result["satcat_op_orbit_codes"]
    assert "geosynchronous" in result["satcat_op_orbit_codes"]
    assert result["psatcat_result_codes"]["S"] == "Successful mission"
    assert "communications_and_navigation" in result["psatcat_category_codes"]
    assert "earth_observation" in result["psatcat_category_codes"]


# --- get_gcat_satcat ---


def test_get_satcat_no_identifier():
    result = get_gcat_satcat()
    assert "error" in result
    assert "exactly one identifier" in result["error"]


def test_get_satcat_multiple_identifiers():
    result = get_gcat_satcat(jcat="S00001", name="test")
    assert "error" in result
    assert "exactly one identifier" in result["error"]


def test_get_satcat_by_jcat(monkeypatch):
    records = [_make_satcat_record(jcat="S00001")]
    container = _make_satcat_container(records)
    monkeypatch.setattr("brahe.datasets.gcat.get_satcat", lambda: container)

    result = get_gcat_satcat(jcat="S00001")
    assert result["method"] == "jcat"
    assert result["identifier"] == "S00001"
    assert result["count"] == 1
    assert result["records"][0]["jcat"] == "S00001"


def test_get_satcat_by_satcat_num(monkeypatch):
    records = [_make_satcat_record(satcat="00001")]
    container = _make_satcat_container(records)
    monkeypatch.setattr("brahe.datasets.gcat.get_satcat", lambda: container)

    result = get_gcat_satcat(satcat_num="00001")
    assert result["method"] == "satcat_num"
    assert result["count"] == 1
    assert result["records"][0]["satcat"] == "00001"


def test_get_satcat_by_name(monkeypatch):
    records = [_make_satcat_record(name="ISS"), _make_satcat_record(name="ISS Module")]
    container = _make_satcat_container(records)
    monkeypatch.setattr("brahe.datasets.gcat.get_satcat", lambda: container)

    result = get_gcat_satcat(name="ISS")
    assert result["method"] == "name"
    assert result["count"] == 2


def test_get_satcat_name_with_limit(monkeypatch):
    records = [_make_satcat_record(name=f"SAT-{i}") for i in range(10)]
    container = _make_satcat_container(records)
    monkeypatch.setattr("brahe.datasets.gcat.get_satcat", lambda: container)

    result = get_gcat_satcat(name="SAT", limit=3)
    assert result["count"] == 3


# --- query_gcat_satcat ---


def test_query_satcat_no_filters():
    result = query_gcat_satcat()
    assert "error" in result
    assert "At least one filter" in result["error"]
    assert "available_filters" in result


def test_query_satcat_by_type(monkeypatch):
    filtered = [_make_satcat_record(object_type="P")]
    container = _make_satcat_container([], filtered=filtered)
    monkeypatch.setattr("brahe.datasets.gcat.get_satcat", lambda: container)

    result = query_gcat_satcat(object_type="P")
    assert result["count"] == 1
    assert "object_type='P'" in result["filters"][0]
    container.filter_by_type.assert_called_once_with("P")


def test_query_satcat_multiple_filters(monkeypatch):
    filtered = [_make_satcat_record(owner="US", state="US")]
    container = _make_satcat_container([], filtered=filtered)
    monkeypatch.setattr("brahe.datasets.gcat.get_satcat", lambda: container)

    result = query_gcat_satcat(owner="US", state="US")
    assert result["count"] == 1
    assert len(result["filters"]) == 2


def test_query_satcat_with_limit(monkeypatch):
    filtered = [_make_satcat_record(name=f"SAT-{i}") for i in range(10)]
    container = _make_satcat_container([], filtered=filtered)
    monkeypatch.setattr("brahe.datasets.gcat.get_satcat", lambda: container)

    result = query_gcat_satcat(name="SAT", limit=3)
    assert result["count"] == 3


def test_query_satcat_perigee_range(monkeypatch):
    filtered = [_make_satcat_record(perigee=400.0)]
    container = _make_satcat_container([], filtered=filtered)
    monkeypatch.setattr("brahe.datasets.gcat.get_satcat", lambda: container)

    result = query_gcat_satcat(perigee_min=300.0, perigee_max=500.0)
    assert result["count"] == 1
    container.filter_by_perigee_range.assert_called_once_with(300.0, 500.0)


# --- get_gcat_psatcat ---


def test_get_psatcat_no_identifier():
    result = get_gcat_psatcat()
    assert "error" in result
    assert "exactly one identifier" in result["error"]


def test_get_psatcat_multiple_identifiers():
    result = get_gcat_psatcat(jcat="S00049", name="Echo")
    assert "error" in result


def test_get_psatcat_by_jcat(monkeypatch):
    records = [_make_psatcat_record(jcat="S00049")]
    container = _make_psatcat_container(records)
    monkeypatch.setattr("brahe.datasets.gcat.get_psatcat", lambda: container)

    result = get_gcat_psatcat(jcat="S00049")
    assert result["method"] == "jcat"
    assert result["count"] == 1
    assert result["records"][0]["jcat"] == "S00049"


def test_get_psatcat_by_name(monkeypatch):
    records = [_make_psatcat_record(name="Echo")]
    container = _make_psatcat_container(records)
    monkeypatch.setattr("brahe.datasets.gcat.get_psatcat", lambda: container)

    result = get_gcat_psatcat(name="Echo")
    assert result["method"] == "name"
    assert result["count"] == 1


# --- query_gcat_psatcat ---


def test_query_psatcat_no_filters():
    result = query_gcat_psatcat()
    assert "error" in result
    assert "At least one filter" in result["error"]


def test_query_psatcat_by_category(monkeypatch):
    filtered = [_make_psatcat_record(category="COM")]
    container = _make_psatcat_container([], filtered=filtered)
    monkeypatch.setattr("brahe.datasets.gcat.get_psatcat", lambda: container)

    result = query_gcat_psatcat(category="COM")
    assert result["count"] == 1
    container.filter_by_category.assert_called_once_with("COM")


def test_query_psatcat_active_only(monkeypatch):
    filtered = [_make_psatcat_record()]
    container = _make_psatcat_container([], filtered=filtered)
    monkeypatch.setattr("brahe.datasets.gcat.get_psatcat", lambda: container)

    result = query_gcat_psatcat(active_only=True)
    assert result["count"] == 1
    assert "active_only=True" in result["filters"]


def test_query_psatcat_multiple_filters(monkeypatch):
    filtered = [_make_psatcat_record()]
    container = _make_psatcat_container([], filtered=filtered)
    monkeypatch.setattr("brahe.datasets.gcat.get_psatcat", lambda: container)

    result = query_gcat_psatcat(category="COM", result_code="S")
    assert result["count"] == 1
    assert len(result["filters"]) == 2
