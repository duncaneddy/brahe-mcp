"""Tests for brahe_mcp.utils utility functions."""

from datetime import datetime, timedelta

import pytest

from brahe_mcp.utils import decimate_records, parse_decimation_interval, parse_epoch_datetime


# --- parse_epoch_datetime ---


def test_parse_epoch_datetime_standard():
    dt = parse_epoch_datetime("2024-01-15T12:30:00")
    assert dt == datetime(2024, 1, 15, 12, 30, 0)


def test_parse_epoch_datetime_z_suffix():
    dt = parse_epoch_datetime("2024-01-15T12:30:00Z")
    assert dt == datetime(2024, 1, 15, 12, 30, 0)


def test_parse_epoch_datetime_utc_suffix():
    dt = parse_epoch_datetime("2024-01-15 12:30:00 UTC")
    assert dt == datetime(2024, 1, 15, 12, 30, 0)


def test_parse_epoch_datetime_invalid():
    with pytest.raises(ValueError):
        parse_epoch_datetime("not-a-date")


# --- parse_decimation_interval ---


@pytest.mark.parametrize("input_str,expected", [
    ("1d", timedelta(days=1)),
    ("12h", timedelta(hours=12)),
    ("30m", timedelta(minutes=30)),
    ("90s", timedelta(seconds=90)),
    ("1w", timedelta(weeks=1)),
    ("2d", timedelta(days=2)),
])
def test_parse_decimation_interval_valid(input_str, expected):
    assert parse_decimation_interval(input_str) == expected


@pytest.mark.parametrize("input_str", [
    "abc", "1x", "d1", "", "1.5d", "1 day",
])
def test_parse_decimation_interval_invalid(input_str):
    with pytest.raises(ValueError, match="Invalid decimation interval"):
        parse_decimation_interval(input_str)


# --- decimate_records ---


def _make_records(epochs, norad_cat_id=25544):
    """Helper to create GP-like record dicts."""
    return [
        {"epoch": e, "norad_cat_id": norad_cat_id, "mean_motion": 15.5}
        for e in epochs
    ]


def test_decimate_empty():
    assert decimate_records([], timedelta(days=1)) == []


def test_decimate_single_record():
    records = _make_records(["2024-01-01T00:00:00"])
    result = decimate_records(records, timedelta(days=1))
    assert len(result) == 1


def test_decimate_two_records():
    records = _make_records(["2024-01-01T00:00:00", "2024-01-01T01:00:00"])
    result = decimate_records(records, timedelta(days=1))
    assert len(result) == 2


def test_decimate_preserves_first_and_last():
    epochs = [f"2024-01-{d:02d}T00:00:00" for d in range(1, 11)]
    records = _make_records(epochs)
    result = decimate_records(records, timedelta(days=3))
    # First and last must be present
    assert result[0]["epoch"] == "2024-01-01T00:00:00"
    assert result[-1]["epoch"] == "2024-01-10T00:00:00"
    # Should be thinned: 1st, 4th, 7th, 10th = 4 records
    assert len(result) == 4


def test_decimate_daily_from_hourly():
    # 24 records, one per hour on Jan 1
    epochs = [f"2024-01-01T{h:02d}:00:00" for h in range(24)]
    records = _make_records(epochs)
    result = decimate_records(records, timedelta(days=1))
    # Only first and last (interval too large for anything in between)
    assert len(result) == 2
    assert result[0]["epoch"] == "2024-01-01T00:00:00"
    assert result[-1]["epoch"] == "2024-01-01T23:00:00"


def test_decimate_multi_satellite():
    records_a = _make_records(
        [f"2024-01-{d:02d}T00:00:00" for d in range(1, 6)],
        norad_cat_id=25544,
    )
    records_b = _make_records(
        [f"2024-01-{d:02d}T00:00:00" for d in range(1, 6)],
        norad_cat_id=43013,
    )
    combined = records_a + records_b
    result = decimate_records(combined, timedelta(days=2))
    # Each satellite: first, 3rd, 5th = 3 records -> 6 total
    ids = {r["norad_cat_id"] for r in result}
    assert ids == {25544, 43013}
    assert len(result) == 6
