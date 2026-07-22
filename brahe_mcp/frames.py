"""Arbitrary reference-frame transformation MCP tools (brahe 1.7.0 router)."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch

# Named frames the router accepts via ReferenceFrame.from_string (plus ECI/ECEF aliases).
NAMED_FRAMES = [
    "GCRF", "ITRF", "ECI", "ECEF", "EME2000", "EMBI", "SSBI",
    "GSE", "EMR", "SER", "LCI", "MCI", "LFME", "LFPA", "MCMF",
]

_SYNODIC_ORIGINS = {
    "primary": brahe.SynodicOrigin.Primary,
    "secondary": brahe.SynodicOrigin.Secondary,
    "barycenter": brahe.SynodicOrigin.Barycenter,
}

_MODES = {"position", "state", "rotation"}


def _resolve_frame(
    name: str,
    synodic_primary=None,
    synodic_secondary=None,
    synodic_origin="barycenter",
    body_naif_id=None,
    pck_center=None,
    pck_frame=None,
) -> brahe.ReferenceFrame:
    """Resolve a frame name (incl. parameterized frames) to a ReferenceFrame."""
    upper = name.upper()
    if upper == "SYNODIC":
        if synodic_primary is None or synodic_secondary is None:
            raise ValueError(
                "Synodic frame requires synodic_primary and synodic_secondary NAIF ids"
            )
        origin = _SYNODIC_ORIGINS.get(synodic_origin.lower())
        if origin is None:
            raise ValueError(
                f"Unknown synodic_origin: {synodic_origin!r}. "
                f"Valid: {sorted(_SYNODIC_ORIGINS)}"
            )
        return brahe.ReferenceFrame.Synodic(origin, int(synodic_primary), int(synodic_secondary))
    if upper == "BODYCENTEREDICRF":
        if body_naif_id is None:
            raise ValueError("BodyCenteredICRF requires body_naif_id")
        return brahe.ReferenceFrame.BodyCenteredICRF(int(body_naif_id))
    if upper == "BODYFIXEDIAU":
        if body_naif_id is None:
            raise ValueError("BodyFixedIAU requires body_naif_id")
        return brahe.ReferenceFrame.BodyFixedIAU(int(body_naif_id))
    if upper == "BODYFIXEDPCK":
        if pck_center is None or pck_frame is None:
            raise ValueError("BodyFixedPCK requires pck_center and pck_frame")
        return brahe.ReferenceFrame.BodyFixedPCK(int(pck_center), int(pck_frame))
    return brahe.ReferenceFrame.from_string(name)


@mcp.tool()
def list_frame_options() -> dict:
    """List reference frames and modes for the transform_frame tool."""
    logger.debug("Listing frame options")
    return {
        "modes": ["position (3-vector)", "state (6-vector)", "rotation (3x3 matrix, no vector)"],
        "named_frames": NAMED_FRAMES,
        "parameterized_frames": {
            "Synodic": "rotating two-body frame; set synodic_primary, synodic_secondary (NAIF ids), synodic_origin (primary|secondary|barycenter)",
            "BodyCenteredICRF": "ICRF axes centered on a body; set body_naif_id",
            "BodyFixedIAU": "IAU body-fixed frame; set body_naif_id",
            "BodyFixedPCK": "SPICE PCK body-fixed frame; set pck_center and pck_frame (frame id)",
        },
        "synodic_origins": sorted(_SYNODIC_ORIGINS),
        "notes": "EMR/SER/GSE and body frames auto-load the DE440s SPICE kernel (~33 MB) on first use.",
    }


@mcp.tool()
def transform_frame(
    mode: str,
    from_frame: str,
    to_frame: str,
    epoch: str,
    vector: list[float] | None = None,
    synodic_primary: int | None = None,
    synodic_secondary: int | None = None,
    synodic_origin: str = "barycenter",
    body_naif_id: int | None = None,
    pck_center: int | None = None,
    pck_frame: int | None = None,
) -> dict:
    """Transform a position/state between any two brahe reference frames, or get the rotation.

    Covers all frames in the brahe frame router: GCRF, ITRF, EME2000, GSE, EMR,
    SER, EMBI, SSBI, lunar (LCI/LFME/LFPA), Mars (MCI/MCMF), the Synodic
    rotating frame, and generic body frames. Use list_frame_options() to discover
    names and the parameterized-frame arguments.

    Args:
        mode: "position" (3-vector), "state" (6-vector), or "rotation" (3x3 matrix; no vector).
        from_frame: Source frame name.
        to_frame: Target frame name.
        epoch: ISO epoch string (e.g. "2024-01-01T00:00:00Z").
        vector: 3 elements for position, 6 for state; omit for rotation.
        synodic_primary: NAIF id of the Synodic frame primary body.
        synodic_secondary: NAIF id of the Synodic frame secondary body.
        synodic_origin: "primary", "secondary", or "barycenter" (default).
        body_naif_id: NAIF id for BodyCenteredICRF / BodyFixedIAU.
        pck_center: NAIF center id for BodyFixedPCK.
        pck_frame: SPICE frame id for BodyFixedPCK.
    """
    m = mode.lower()
    if m not in _MODES:
        return error_response(f"Unknown mode: {mode!r}", valid_modes=sorted(_MODES))

    try:
        epc = parse_epoch(epoch)
    except ValueError as e:
        return error_response(f"Invalid epoch: {e}")

    fp = dict(
        synodic_primary=synodic_primary, synodic_secondary=synodic_secondary,
        synodic_origin=synodic_origin, body_naif_id=body_naif_id,
        pck_center=pck_center, pck_frame=pck_frame,
    )
    try:
        src = _resolve_frame(from_frame, **fp)
        dst = _resolve_frame(to_frame, **fp)
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Unknown frame: {e}")

    try:
        if m == "rotation":
            mat = brahe.rotation_frame_to_frame(src, dst, epc)
            return {
                "input": {"mode": m, "from_frame": from_frame, "to_frame": to_frame, "epoch": epoch},
                "output": {"matrix": np.array(mat).tolist()},
            }
        if vector is None:
            return error_response(f"mode {m!r} requires a vector")
        expected = 3 if m == "position" else 6
        if len(vector) != expected:
            return error_response(f"mode {m!r} requires a {expected}-element vector, got {len(vector)}")
        vec = np.array(vector, dtype=float)
        if m == "position":
            out = brahe.position_frame_to_frame(src, dst, epc, vec)
        else:
            out = brahe.state_frame_to_frame(src, dst, epc, vec)
    except Exception as e:
        logger.error("Frame transform error: {}", e)
        return error_response(f"Transform error: {e}")

    return {
        "input": {"mode": m, "from_frame": from_frame, "to_frame": to_frame, "epoch": epoch, "vector": vector},
        "output": {"vector": np.array(out).tolist(), "frame": to_frame},
    }
