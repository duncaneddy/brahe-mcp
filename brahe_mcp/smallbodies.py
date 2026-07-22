"""JPL Horizons Small-Body Database (SBDB) and ephemeris MCP tools."""

import numpy as np
import brahe
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, parse_epoch

MAX_EPHEMERIS_POINTS = 5000

# Horizons-generated SPKs sometimes end coverage slightly before the
# requested stop epoch (Chebyshev fit windowing). Pad the SPK request past
# the requested stop so the exact stop epoch is always sampleable.
_SPK_COVERAGE_PAD_SECONDS = 21600.0


@mcp.tool()
def list_smallbody_options() -> dict:
    """List small-body lookup and ephemeris tools and their network characteristics."""
    return {
        "tools": {
            "lookup_small_body": "Resolve a small body by name/designation via JPL SBDB → NAIF id, GM, radius, NEO flag.",
            "get_small_body_ephemeris": "Generate a JPL Horizons SPK for a small body and sample its state over a time span.",
        },
        "notes": "Both tools perform live JPL network requests; get_small_body_ephemeris triggers a Horizons SPK generation (multi-second latency).",
        "center_default": "ssb (solar system barycenter); also accepts 'sun' or a NAIF id.",
    }


@mcp.tool()
def lookup_small_body(name: str) -> dict:
    """Look up a small body (asteroid/comet) in the JPL Small-Body Database.

    Args:
        name: Name or designation, e.g. "Ceres", "433", "2000 SG344".
    """
    try:
        client = brahe.sbdb.SBDBClient()
        obj = client.lookup(name)
    except Exception as e:
        logger.error("SBDB lookup error: {}", e)
        return error_response(f"SBDB lookup failed for {name!r}: {e}")
    return {
        "spkid": obj.spkid,
        "naif_id": obj.naif_id(),
        "full_name": obj.full_name,
        "des": obj.des,
        "shortname": obj.shortname,
        "kind": obj.kind,
        "neo": obj.neo,
        "gm": obj.gm,
        "radius": obj.radius,
    }


@mcp.tool()
def get_small_body_ephemeris(
    designation: str,
    start: str,
    stop: str,
    step_seconds: float = 3600.0,
    center: str = "ssb",
) -> dict:
    """Generate a Horizons SPK for a small body and sample its state over a time span.

    Resolves the body via SBDB, requests an SPK from JPL Horizons, loads it, and
    samples spk_state at each step. Live JPL network calls; keep the span modest.

    Args:
        designation: Small-body name/designation (resolved via SBDB).
        start: ISO start epoch.
        stop: ISO stop epoch.
        step_seconds: Sampling step in seconds (default 3600).
        center: Reference center ("ssb" default, "sun", or a NAIF id).
    """
    if step_seconds <= 0:
        return error_response("step_seconds must be positive")
    try:
        t0 = parse_epoch(start)
        t1 = parse_epoch(stop)
    except ValueError as e:
        return error_response(f"Invalid epoch: {e}")
    duration = float(t1 - t0)
    if duration <= 0:
        return error_response("stop must be after start")
    n_regular = int(duration // step_seconds) + 1
    # Include the exact stop epoch as a final sample when the span isn't an
    # even multiple of step_seconds, so the requested stop is always covered.
    include_stop = (duration - (n_regular - 1) * step_seconds) > 1e-6
    n = n_regular + (1 if include_stop else 0)
    if n > MAX_EPHEMERIS_POINTS:
        return error_response(
            f"Span would produce {n} points, exceeding {MAX_EPHEMERIS_POINTS}. Increase step_seconds."
        )

    center_naif = {"ssb": 0, "sun": 10}.get(str(center).lower())
    if center_naif is None:
        try:
            center_naif = int(center)
        except (ValueError, TypeError):
            return error_response(f"Unknown center: {center!r}. Use 'ssb', 'sun', or a NAIF id.")

    try:
        obj = brahe.sbdb.SBDBClient().lookup(designation)
        spkid = obj.spkid
    except Exception as e:
        return error_response(f"SBDB lookup failed for {designation!r}: {e}")

    try:
        req = brahe.horizons.HorizonsSPKRequest.for_spkid(
            spkid, t0, t1 + _SPK_COVERAGE_PAD_SECONDS
        )
        resp = brahe.horizons.HorizonsClient().get_spk(req)
        resp.load()
    except Exception as e:
        logger.error("Horizons SPK error: {}", e)
        return error_response(f"Horizons SPK generation failed: {e}")

    # The small-body SPK only has a direct segment to the Sun (NAIF 10); the
    # planetary kernels provide the Sun -> SSB (NAIF 0) chain needed for other
    # centers (e.g. the "ssb" default).
    try:
        brahe.load_common_spice_kernels()
    except Exception as e:
        logger.warning("Failed to load common SPICE kernels: {}", e)

    states = []
    try:
        for i in range(n_regular):
            epc = t0 + i * step_seconds
            vec = brahe.spk_state(spkid, center_naif, epc)
            states.append({"epoch": str(epc), "vector": np.array(vec).tolist()})
        if include_stop:
            vec = brahe.spk_state(spkid, center_naif, t1)
            states.append({"epoch": str(t1), "vector": np.array(vec).tolist()})
    except Exception as e:
        return error_response(f"Ephemeris sampling failed: {e}")

    return {
        "designation": designation,
        "spkid": spkid,
        "naif_id": obj.naif_id(),
        "center": center,
        "frame": "ICRF (J2000 axes)",
        "count": len(states),
        "states": states,
    }
