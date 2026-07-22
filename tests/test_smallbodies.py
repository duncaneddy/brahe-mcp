import socket
import pytest
from brahe_mcp.smallbodies import list_smallbody_options, lookup_small_body, get_small_body_ephemeris


def _online() -> bool:
    try:
        socket.create_connection(("ssd-api.jpl.nasa.gov", 443), timeout=3).close()
        return True
    except OSError:
        return False


def test_list_smallbody_options():
    opts = list_smallbody_options()
    assert "tools" in opts


@pytest.mark.skipif(not _online(), reason="requires JPL SBDB network access")
def test_lookup_ceres():
    res = lookup_small_body("Ceres")
    assert "error" not in res
    assert res["naif_id"] == 20000001


@pytest.mark.skipif(not _online(), reason="requires JPL Horizons network access")
def test_ceres_ephemeris():
    res = get_small_body_ephemeris(
        designation="Ceres",
        start="2024-01-01T00:00:00Z",
        stop="2024-01-02T00:00:00Z",
        step_seconds=43200,
    )
    assert "error" not in res
    assert res["count"] >= 2
    assert len(res["states"][0]["vector"]) == 6
