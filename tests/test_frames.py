import numpy as np
from brahe_mcp.frames import transform_frame, list_frame_options


def test_list_frame_options_lists_named_frames():
    opts = list_frame_options()
    assert "GCRF" in opts["named_frames"]
    assert "ITRF" in opts["named_frames"]
    assert "modes" in opts


def test_state_gcrf_itrf_roundtrip():
    state = [6871000.0, 0.0, 0.0, 0.0, 7612.0, 0.0]
    epc = "2024-01-01 00:00:00 UTC"
    fwd = transform_frame("state", "GCRF", "ITRF", epc, vector=state)
    assert "error" not in fwd
    back = transform_frame("state", "ITRF", "GCRF", epc, vector=fwd["output"]["vector"])
    assert np.allclose(back["output"]["vector"], state, atol=1e-3)


def test_rotation_mode_returns_matrix():
    res = transform_frame("rotation", "GCRF", "ITRF", "2024-01-01 00:00:00 UTC")
    assert "error" not in res
    assert np.array(res["output"]["matrix"]).shape == (3, 3)


def test_synodic_emr_position():
    res = transform_frame(
        "position", "GCRF", "Synodic", "2024-03-01 00:00:00 UTC",
        vector=[6871000.0, 0.0, 0.0],
        synodic_primary=399, synodic_secondary=301, synodic_origin="barycenter",
    )
    assert "error" not in res
    assert len(res["output"]["vector"]) == 3


def test_unknown_frame_errors():
    res = transform_frame("state", "NOPE", "GCRF", "2024-01-01 00:00:00 UTC", vector=[0]*6)
    assert "error" in res
