"""SPICE kernel management and celestial-body ephemeris MCP tools."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch

# Common body-name → NAIF id (SPICE convention: barycenters 1..9, bodies x99).
_BODY_NAIF = {
    "sun": 10, "mercury": 199, "venus": 299, "earth": 399, "moon": 301,
    "mars": 4, "jupiter": 5, "saturn": 6, "uranus": 7, "neptune": 8,
    "ssb": 0, "emb": 3,
}

_MODES = {"state", "position", "velocity"}


def _resolve_naif(name) -> int:
    key = str(name).lower()
    if key in _BODY_NAIF:
        return _BODY_NAIF[key]
    try:
        return int(name)
    except (ValueError, TypeError):
        raise ValueError(
            f"Unknown body: {name!r}. Use one of {sorted(_BODY_NAIF)} or a NAIF id."
        )


@mcp.tool()
def list_ephemeris_options() -> dict:
    """List SPICE kernel and body-ephemeris query options."""
    return {
        "bodies": sorted(_BODY_NAIF),
        "modes": sorted(_MODES),
        "common_kernels": ["de440s (planets, ~33 MB)", "moon_pa_de440 (lunar orientation)"],
        "notes": "get_body_state auto-loads DE440s on first use. States are ICRF (J2000 axes).",
    }


@mcp.tool()
def list_spice_kernels() -> dict:
    """List currently loaded SPICE kernels."""
    return {"kernels": list(brahe.loaded_spice_kernels())}


@mcp.tool()
def load_spice_kernel(name: str) -> dict:
    """Download (cached) and load a NAIF SPICE kernel by name (e.g. "de440s")."""
    try:
        brahe.load_spice_kernel(name)
    except Exception as e:
        return error_response(f"Failed to load kernel {name!r}: {e}")
    return {"loaded": name, "kernels": list(brahe.loaded_spice_kernels())}


@mcp.tool()
def load_common_spice_kernels() -> dict:
    """Load the common kernels (de440s planetary ephemeris + moon_pa_de440 orientation)."""
    try:
        brahe.load_common_spice_kernels()
    except Exception as e:
        return error_response(f"Failed to load common kernels: {e}")
    return {"kernels": list(brahe.loaded_spice_kernels())}


@mcp.tool()
def unload_spice_kernel(name: str) -> dict:
    """Unload a previously loaded SPICE kernel by name."""
    try:
        brahe.unload_spice_kernel(name)
    except Exception as e:
        return error_response(f"Failed to unload kernel {name!r}: {e}")
    return {"unloaded": name, "kernels": list(brahe.loaded_spice_kernels())}


@mcp.tool()
def get_body_state(target: str, center: str, epoch: str, mode: str = "state") -> dict:
    """Get a celestial body's state/position/velocity relative to a center body from SPICE.

    Args:
        target: Body of interest (name like "moon"/"mars" or a NAIF id).
        center: Reference center body (name or NAIF id).
        epoch: ISO epoch string.
        mode: "state" (6-vector), "position" (3-vector), or "velocity" (3-vector).
    """
    m = mode.lower()
    if m not in _MODES:
        return error_response(f"Unknown mode: {mode!r}", valid_modes=sorted(_MODES))
    try:
        tgt = _resolve_naif(target)
        ctr = _resolve_naif(center)
        epc = parse_epoch(epoch)
    except ValueError as e:
        return error_response(str(e))
    try:
        if m == "state":
            out = brahe.spk_state(tgt, ctr, epc)
        elif m == "position":
            out = brahe.spk_position(tgt, ctr, epc)
        else:
            out = brahe.spk_velocity(tgt, ctr, epc)
    except Exception as e:
        logger.error("SPICE query error: {}", e)
        return error_response(f"SPICE query failed: {e}")
    return {
        "input": {"target": target, "center": center, "epoch": epoch, "mode": m},
        "output": {"vector": np.array(out).tolist(), "frame": "ICRF (J2000 axes)"},
    }
