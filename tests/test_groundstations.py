"""Tests for Groundstations MCP tools."""

from unittest.mock import MagicMock

import pytest

import brahe_mcp.groundstations as mod
from brahe_mcp.groundstations import (
    get_groundstations,
    list_groundstation_options,
    query_groundstations,
)


def _make_station(name="Alaska", lon=-149.9, lat=71.29, alt=0.0,
                  provider="Aws", frequency_bands=None):
    """Create a mock PointLocation station."""
    if frequency_bands is None:
        frequency_bands = ["S", "X"]
    station = MagicMock()
    station.get_name.return_value = name
    station.lon = lon
    station.lat = lat
    station.alt = alt
    station.properties = {
        "provider": provider,
        "frequency_bands": frequency_bands,
    }
    return station


# --- list_groundstation_options ---


def test_list_groundstation_options():
    result = list_groundstation_options()
    assert "providers" in result
    assert "lookup_methods" in result
    assert "station_fields" in result
    assert "query_filters" in result
    assert len(result["providers"]) == 8


# --- get_groundstations ---


def test_get_groundstations_all(monkeypatch):
    stations = [_make_station("A"), _make_station("B")]
    monkeypatch.setattr("brahe.datasets.groundstations.load_all", lambda: stations)

    result = get_groundstations()
    assert result["provider"] == "all"
    assert result["count"] == 2
    assert result["stations"][0]["name"] == "A"


def test_get_groundstations_by_provider(monkeypatch):
    stations = [_make_station("Alaska", provider="Aws")]
    monkeypatch.setattr("brahe.datasets.groundstations.load", lambda p: stations)

    result = get_groundstations(provider="aws")
    assert result["provider"] == "aws"
    assert result["count"] == 1


def test_get_groundstations_invalid_provider():
    result = get_groundstations(provider="nonexistent")
    assert "error" in result
    assert "valid_providers" in result


def test_get_groundstations_case_insensitive(monkeypatch):
    stations = [_make_station()]
    monkeypatch.setattr("brahe.datasets.groundstations.load", lambda p: stations)

    result = get_groundstations(provider="AWS")
    assert "error" not in result
    assert result["provider"] == "aws"


def test_get_groundstations_with_limit(monkeypatch):
    stations = [_make_station(f"S{i}") for i in range(10)]
    monkeypatch.setattr("brahe.datasets.groundstations.load_all", lambda: stations)

    result = get_groundstations(limit=3)
    assert result["count"] == 3


# --- query_groundstations ---


def test_query_groundstations_no_filters():
    result = query_groundstations()
    assert "error" in result
    assert "At least one filter" in result["error"]


def test_query_groundstations_by_name(monkeypatch):
    stations = [_make_station("Alaska"), _make_station("Hawaii")]
    monkeypatch.setattr("brahe.datasets.groundstations.load_all", lambda: stations)

    result = query_groundstations(name="alaska")
    assert result["count"] == 1
    assert result["stations"][0]["name"] == "Alaska"


def test_query_groundstations_by_provider(monkeypatch):
    stations = [_make_station("Alaska", provider="Aws")]
    monkeypatch.setattr("brahe.datasets.groundstations.load", lambda p: stations)

    result = query_groundstations(provider="aws")
    assert result["count"] == 1
    assert "provider='aws'" in result["filters"][0]


def test_query_groundstations_invalid_provider():
    result = query_groundstations(provider="fake")
    assert "error" in result
    assert "valid_providers" in result


def test_query_groundstations_geographic_filter(monkeypatch):
    stations = [
        _make_station("North", lat=70.0, lon=0.0),
        _make_station("South", lat=-30.0, lon=0.0),
    ]
    monkeypatch.setattr("brahe.datasets.groundstations.load_all", lambda: stations)

    result = query_groundstations(lat_min=50.0)
    assert result["count"] == 1
    assert result["stations"][0]["name"] == "North"


def test_query_groundstations_lon_filter(monkeypatch):
    stations = [
        _make_station("East", lat=0.0, lon=50.0),
        _make_station("West", lat=0.0, lon=-100.0),
    ]
    monkeypatch.setattr("brahe.datasets.groundstations.load_all", lambda: stations)

    result = query_groundstations(lon_min=0.0, lon_max=100.0)
    assert result["count"] == 1
    assert result["stations"][0]["name"] == "East"


def test_query_groundstations_frequency_band(monkeypatch):
    stations = [
        _make_station("HasKa", frequency_bands=["S", "X", "Ka"]),
        _make_station("NoKa", frequency_bands=["S", "X"]),
    ]
    monkeypatch.setattr("brahe.datasets.groundstations.load_all", lambda: stations)

    result = query_groundstations(frequency_band="Ka")
    assert result["count"] == 1
    assert result["stations"][0]["name"] == "HasKa"


def test_query_groundstations_combined_filters(monkeypatch):
    stations = [
        _make_station("A", lat=70.0, lon=10.0, frequency_bands=["S", "Ka"]),
        _make_station("B", lat=70.0, lon=10.0, frequency_bands=["S"]),
        _make_station("C", lat=-10.0, lon=10.0, frequency_bands=["S", "Ka"]),
    ]
    monkeypatch.setattr("brahe.datasets.groundstations.load_all", lambda: stations)

    result = query_groundstations(lat_min=50.0, frequency_band="Ka")
    assert result["count"] == 1
    assert result["stations"][0]["name"] == "A"


def test_query_groundstations_with_limit(monkeypatch):
    stations = [_make_station(f"S{i}") for i in range(10)]
    monkeypatch.setattr("brahe.datasets.groundstations.load_all", lambda: stations)

    result = query_groundstations(name="S", limit=3)
    assert result["count"] == 3
