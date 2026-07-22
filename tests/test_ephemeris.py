import socket
import pytest
from brahe_mcp.ephemeris import list_ephemeris_options, get_body_state, list_spice_kernels


def _online() -> bool:
    try:
        socket.create_connection(("naif.jpl.nasa.gov", 443), timeout=3).close()
        return True
    except OSError:
        return False


def test_list_ephemeris_options_documents_bodies():
    opts = list_ephemeris_options()
    assert "bodies" in opts
    assert "moon" in opts["bodies"]


def test_list_spice_kernels_returns_list():
    res = list_spice_kernels()
    assert "kernels" in res
    assert isinstance(res["kernels"], list)


def test_bad_body_errors():
    res = get_body_state(target="notabody", center="earth", epoch="2024-01-01T00:00:00Z")
    assert "error" in res


@pytest.mark.skipif(not _online(), reason="requires network for DE440s auto-load")
def test_get_moon_state_from_earth():
    res = get_body_state(target="moon", center="earth", epoch="2024-01-01T00:00:00Z")
    assert "error" not in res
    assert len(res["output"]["vector"]) == 6
    dist = sum(c * c for c in res["output"]["vector"][:3]) ** 0.5
    assert 3.5e8 < dist < 4.1e8  # Earth-Moon distance range (m)
