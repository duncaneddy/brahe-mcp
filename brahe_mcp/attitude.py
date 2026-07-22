"""Attitude representation and rotation matrix MCP tools."""

import numpy as np
import brahe
from brahe.attitude import (
    EulerAngle,
    EulerAngleOrder,
    EulerAxis,
    Quaternion,
    RotationMatrix,
)
from loguru import logger

from brahe_mcp.server import mcp
from brahe_mcp.utils import error_response, resolve_angle_format

REPRESENTATIONS = {"quaternion", "euler_axis", "euler_angle", "rotation_matrix"}

EULER_ORDERS = [
    "XYX", "XYZ", "XZX", "XZY", "YXY", "YXZ",
    "YZX", "YZY", "ZXY", "ZXZ", "ZYX", "ZYZ",
]


def _resolve_euler_order(order: str) -> EulerAngleOrder:
    """Resolve an Euler order string to the brahe enum."""
    key = str(order).upper()
    if key not in EULER_ORDERS:
        raise ValueError(
            f"Invalid Euler order: {order!r}. Must be one of: {EULER_ORDERS}"
        )
    return getattr(EulerAngleOrder, key)


def _angles_from_radians(values, angle_fmt: brahe.AngleFormat) -> list[float]:
    """Convert brahe attitude angle properties, which are always radians.

    brahe stores attitude internally in radians and EulerAngle's phi/theta/psi
    getters expose that directly, ignoring the angle_format passed to the
    constructor. EulerAngle has no format-aware accessor, so it is the ONLY
    representation needing manual conversion — every other one uses a brahe
    accessor that honors angle_format. Pinned by
    test_euler_angle_properties_are_radians.
    """
    if angle_fmt == brahe.AngleFormat.DEGREES:
        return [float(np.degrees(v)) for v in values]
    return [float(v) for v in values]


def _parse_attitude(
    repr_name: str,
    value,
    angle_fmt: brahe.AngleFormat,
    euler_order: str,
    scalar_first: bool,
):
    """Build a brahe attitude object from the MCP value encoding."""
    if repr_name == "quaternion":
        arr = np.array(value, dtype=float)
        if arr.shape != (4,):
            raise ValueError(
                f"quaternion requires 4 elements, got shape {arr.shape}"
            )
        return Quaternion.from_vector(arr, scalar_first)

    if repr_name == "euler_axis":
        if not isinstance(value, dict):
            raise ValueError(
                'euler_axis requires {"axis": [x, y, z], "angle": float}'
            )
        axis = np.array(value["axis"], dtype=float)
        if axis.shape != (3,):
            raise ValueError("euler_axis 'axis' requires exactly 3 elements")
        return EulerAxis(axis, float(value["angle"]), angle_fmt)

    if repr_name == "euler_angle":
        if not isinstance(value, dict):
            raise ValueError(
                'euler_angle requires {"angles": [phi, theta, psi], "order": "ZYX"}'
            )
        angles = value["angles"]
        if len(angles) != 3:
            raise ValueError("euler_angle 'angles' requires exactly 3 elements")
        order = _resolve_euler_order(value.get("order") or euler_order)
        return EulerAngle(
            order, float(angles[0]), float(angles[1]), float(angles[2]), angle_fmt
        )

    matrix = np.array(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"rotation_matrix requires a 3x3 nested list, got shape {matrix.shape}"
        )
    return RotationMatrix.from_matrix(matrix)


def _serialize_attitude(
    obj,
    repr_name: str,
    angle_fmt: brahe.AngleFormat,
    euler_order: str,
    scalar_first: bool,
):
    """Serialize a brahe attitude object into the MCP value encoding."""
    if repr_name == "quaternion":
        return np.array(
            obj.to_quaternion().to_vector(scalar_first), dtype=float
        ).tolist()

    if repr_name == "euler_axis":
        # to_vector honors angle_format, so no manual conversion here.
        vec = np.array(
            obj.to_euler_axis().to_vector(angle_fmt, True), dtype=float
        )
        return {"axis": vec[:3].tolist(), "angle": float(vec[3])}

    if repr_name == "euler_angle":
        order = _resolve_euler_order(euler_order)
        ea = obj.to_euler_angle(order)
        return {
            "angles": _angles_from_radians([ea.phi, ea.theta, ea.psi], angle_fmt),
            "order": str(euler_order).upper(),
        }

    return np.array(
        obj.to_rotation_matrix().to_matrix(), dtype=float
    ).tolist()


def _all_finite(value) -> bool:
    """Recursively check that a serialized attitude value has no NaN/Inf.

    Degenerate inputs (e.g. a zero-norm quaternion) are accepted without
    error by brahe's constructors but silently propagate NaN through the
    converters. Bare NaN/Inf is not valid JSON (RFC 8259), so this must be
    checked before returning a success envelope.
    """
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(v) for v in value)
    if isinstance(value, str):
        return True
    return bool(np.isfinite(value))


@mcp.tool()
def list_attitude_options() -> dict:
    """List attitude representations, Euler orders, and value encodings."""
    logger.debug("Listing attitude options")
    return {
        "representations": sorted(REPRESENTATIONS),
        "euler_orders": EULER_ORDERS,
        "quaternion_default_ordering": "scalar_first",
        "value_encodings": {
            "quaternion": "[w, x, y, z], or [x, y, z, w] if scalar_first=false",
            "euler_axis": '{"axis": [x, y, z], "angle": float}',
            "euler_angle": '{"angles": [phi, theta, psi], "order": "ZYX"}',
            "rotation_matrix": "[[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]]",
        },
        "composition_order": (
            "compose_rotations applies rotations[0] first, then rotations[1], "
            "and so on: R_total = R_n @ ... @ R_1."
        ),
        "notes": [
            "All angular values honor angle_format on both input and output.",
            "scalar_first is echoed in every response containing a quaternion.",
        ],
    }


@mcp.tool()
def convert_attitude(
    from_repr: str,
    value,
    to_repr: str,
    euler_order_in: str = "ZYX",
    euler_order_out: str = "ZYX",
    scalar_first: bool = True,
    angle_format: str = "degrees",
) -> dict:
    """Convert an attitude between quaternion, Euler axis, Euler angle, and matrix.

    Use list_attitude_options() to see the value encoding for each
    representation.

    Args:
        from_repr: Source representation.
        value: The attitude, encoded per from_repr.
        to_repr: Target representation.
        euler_order_in: Euler order for euler_angle input. An "order" key in
            value takes precedence.
        euler_order_out: Euler order for euler_angle output.
        scalar_first: Quaternion component ordering. True (default) is
            [w, x, y, z]; False is [x, y, z, w].
        angle_format: "degrees" (default) or "radians".
    """
    src = from_repr.lower()
    dst = to_repr.lower()
    for name, rep in (("from_repr", src), ("to_repr", dst)):
        if rep not in REPRESENTATIONS:
            return error_response(
                f"Unknown {name}: {rep!r}",
                valid_representations=sorted(REPRESENTATIONS),
            )

    try:
        fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    try:
        obj = _parse_attitude(src, value, fmt, euler_order_in, scalar_first)
    except (ValueError, KeyError, TypeError) as e:
        return error_response(f"Invalid {src} value: {e}")
    except Exception as e:
        # brahe raises brahe.BraheError (not a ValueError/KeyError/TypeError)
        # for e.g. a well-shaped but non-orthogonal rotation matrix.
        logger.error("Attitude parse error: {}", e)
        return error_response(f"Invalid {src} value: {e}")

    try:
        out = _serialize_attitude(obj, dst, fmt, euler_order_out, scalar_first)
    except (ValueError, KeyError, TypeError) as e:
        return error_response(f"Cannot produce {dst}: {e}")
    except Exception as e:
        logger.error("Attitude conversion error: {}", e)
        return error_response(f"Conversion error: {e}")

    if not _all_finite(out):
        return error_response(
            "Conversion produced non-finite output (input attitude may be "
            "degenerate, e.g. a zero-norm quaternion)",
            from_repr=src,
            to_repr=dst,
        )

    logger.debug("Attitude {} -> {}", src, dst)
    return {
        "input": {
            "representation": src,
            "value": value,
            "angle_format": angle_format.lower(),
        },
        "output": {
            "representation": dst,
            "value": out,
            "angle_format": angle_format.lower(),
            "scalar_first": scalar_first,
        },
    }


_AXES = {"x": RotationMatrix.Rx, "y": RotationMatrix.Ry, "z": RotationMatrix.Rz}


@mcp.tool()
def axis_rotation_matrix(
    axis: str,
    angle: float,
    angle_format: str = "degrees",
) -> dict:
    """Compute a principal-axis rotation matrix (Rx, Ry, or Rz).

    Args:
        axis: "x", "y", or "z".
        angle: Rotation angle in the given angle_format.
        angle_format: "degrees" (default) or "radians".
    """
    key = str(axis).lower()
    if key not in _AXES:
        return error_response(
            f"Unknown axis: {axis!r}", valid_axes=sorted(_AXES)
        )

    try:
        fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    try:
        # RotationMatrix.Rx/Ry/Rz return RotationMatrix objects that support
        # __mul__. The module-level brahe.attitude.Rx/Ry/Rz return bare numpy
        # arrays instead, which compose_rotations could not use.
        rot = _AXES[key](float(angle), fmt)
    except Exception as e:
        logger.error("Axis rotation error: {}", e)
        return error_response(f"Rotation error: {e}")

    matrix = np.array(rot.to_matrix(), dtype=float).tolist()
    if not _all_finite(matrix):
        return error_response("Rotation produced non-finite output", axis=key)

    return {
        "input": {
            "axis": key,
            "angle": angle,
            "angle_format": angle_format.lower(),
        },
        "output": {
            "matrix": matrix
        },
    }


@mcp.tool()
def compose_rotations(
    rotations: list[dict],
    output_repr: str = "rotation_matrix",
    scalar_first: bool = True,
    angle_format: str = "degrees",
) -> dict:
    """Compose a sequence of rotations into a single rotation.

    Rotations are applied in list order: rotations[0] is applied first, then
    rotations[1], and so on. The result is R_total = R_n @ ... @ R_1.

    Args:
        rotations: List of {"repr": ..., "value": ...} entries. "repr" is any
            representation from list_attitude_options(); "value" uses that
            representation's encoding. An optional "order" key sets the Euler
            order for euler_angle entries.
        output_repr: Representation for the composed result.
        scalar_first: Quaternion ordering for both input and output.
        angle_format: "degrees" (default) or "radians".

    Note:
        When output_repr is "euler_angle", the output always uses ZYX order
        regardless of any per-entry "order" (which only governs parsing of
        euler_angle input entries). The returned value self-describes via its
        "order" field.
    """
    dst = output_repr.lower()
    if dst not in REPRESENTATIONS:
        return error_response(
            f"Unknown output_repr: {output_repr!r}",
            valid_representations=sorted(REPRESENTATIONS),
        )

    if not rotations:
        return error_response("rotations must contain at least one entry")

    try:
        fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    composed = None
    for i, entry in enumerate(rotations):
        if not isinstance(entry, dict) or "repr" not in entry or "value" not in entry:
            return error_response(
                f"rotations[{i}] must be an object with 'repr' and 'value' keys",
                valid_representations=sorted(REPRESENTATIONS),
            )
        rep = str(entry["repr"]).lower()
        if rep not in REPRESENTATIONS:
            return error_response(
                f"rotations[{i}] has unknown repr: {entry['repr']!r}",
                valid_representations=sorted(REPRESENTATIONS),
            )
        try:
            obj = _parse_attitude(
                rep, entry["value"], fmt, entry.get("order", "ZYX"), scalar_first
            )
        except (ValueError, KeyError, TypeError) as e:
            return error_response(f"Invalid rotations[{i}] value: {e}")
        except Exception as e:
            logger.error("Compose parse error: {}", e)
            return error_response(f"Invalid rotations[{i}] value: {e}")

        rot = obj.to_rotation_matrix()
        # rotations[0] applied first => later entries multiply on the left.
        composed = rot if composed is None else rot * composed

    try:
        out = _serialize_attitude(composed, dst, fmt, "ZYX", scalar_first)
    except Exception as e:
        logger.error("Compose error: {}", e)
        return error_response(f"Composition error: {e}")

    if not _all_finite(out):
        return error_response(
            "Composition produced non-finite output",
            output_repr=dst,
        )

    logger.debug("Composed {} rotations -> {}", len(rotations), dst)
    return {
        "input": {
            "n_rotations": len(rotations),
            "angle_format": angle_format.lower(),
        },
        "output": {
            "representation": dst,
            "value": out,
            "angle_format": angle_format.lower(),
            "scalar_first": scalar_first,
            "composition_order": "rotations[0] applied first",
        },
    }


@mcp.tool()
def quaternion_slerp(
    q1: list[float],
    q2: list[float],
    t: float,
    scalar_first: bool = True,
    output_repr: str = "quaternion",
    angle_format: str = "degrees",
) -> dict:
    """Spherically interpolate between two quaternions.

    Args:
        q1: Start quaternion, [w, x, y, z] unless scalar_first is False.
        q2: End quaternion, same ordering.
        t: Interpolation parameter in [0, 1]. 0 returns q1, 1 returns q2.
        scalar_first: Quaternion component ordering for input and output.
        output_repr: Representation for the interpolated result.
        angle_format: "degrees" (default) or "radians", for non-quaternion output.
    """
    dst = output_repr.lower()
    if dst not in REPRESENTATIONS:
        return error_response(
            f"Unknown output_repr: {output_repr!r}",
            valid_representations=sorted(REPRESENTATIONS),
        )

    try:
        t = float(t)
    except (TypeError, ValueError):
        return error_response(f"t must be a number, got {t!r}")

    if not 0.0 <= t <= 1.0:
        return error_response(f"t must be in [0, 1], got {t}")

    try:
        fmt = resolve_angle_format(angle_format)
    except ValueError as e:
        return error_response(str(e))

    try:
        a = _parse_attitude("quaternion", q1, fmt, "ZYX", scalar_first)
        b = _parse_attitude("quaternion", q2, fmt, "ZYX", scalar_first)
    except (ValueError, KeyError, TypeError) as e:
        return error_response(f"Invalid quaternion: {e}")
    except Exception as e:
        logger.error("Slerp parse error: {}", e)
        return error_response(f"Invalid quaternion: {e}")

    try:
        result = a.slerp(b, float(t))
        out = _serialize_attitude(result, dst, fmt, "ZYX", scalar_first)
    except Exception as e:
        logger.error("Slerp error: {}", e)
        return error_response(f"Slerp error: {e}")

    if not _all_finite(out):
        return error_response(
            "Slerp produced non-finite output (input quaternion may be "
            "degenerate, e.g. a zero-norm quaternion)",
            output_repr=dst,
        )

    logger.debug("Slerp t={} -> {}", t, dst)
    return {
        "input": {"q1": q1, "q2": q2, "t": t, "scalar_first": scalar_first},
        "output": {
            "representation": dst,
            "value": out,
            "angle_format": angle_format.lower(),
            "scalar_first": scalar_first,
        },
    }
