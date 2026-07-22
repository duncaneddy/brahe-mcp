import numpy as np
import brahe
from brahe.attitude import EulerAngle, EulerAngleOrder, EulerAxis

from brahe_mcp.attitude import (
    axis_rotation_matrix,
    compose_rotations,
    convert_attitude,
    list_attitude_options,
    quaternion_slerp,
)


def test_euler_angle_properties_are_radians():
    """Assumption-pinning test: this exercises brahe directly, not brahe-mcp.

    brahe stores attitude internally in radians and the bare EulerAngle
    property getters expose that, ignoring the constructor's angle_format.
    brahe_mcp.attitude._angles_from_radians compensates. If a future brahe
    version makes these getters format-aware, this test fails loudly instead
    of brahe-mcp silently double-converting.
    """
    ea = EulerAngle(EulerAngleOrder.ZYX, 10.0, 20.0, 30.0, brahe.AngleFormat.DEGREES)
    assert np.isclose(ea.phi, np.radians(10.0))
    assert np.isclose(ea.theta, np.radians(20.0))
    assert np.isclose(ea.psi, np.radians(30.0))


def test_euler_axis_to_vector_honors_angle_format():
    """Assumption-pinning test: EulerAxis DOES have a format-aware accessor."""
    ax = EulerAxis(np.array([0.0, 0.0, 1.0]), 90.0, brahe.AngleFormat.DEGREES)
    v = ax.to_vector(brahe.AngleFormat.DEGREES, True)
    assert np.isclose(v[3], 90.0)


def test_list_attitude_options():
    opts = list_attitude_options()
    assert set(opts["representations"]) == {
        "quaternion", "euler_axis", "euler_angle", "rotation_matrix"
    }
    assert len(opts["euler_orders"]) == 12
    assert "ZYX" in opts["euler_orders"]
    assert opts["quaternion_default_ordering"] == "scalar_first"


def test_quaternion_to_rotation_matrix_identity():
    res = convert_attitude("quaternion", [1.0, 0.0, 0.0, 0.0], "rotation_matrix")
    assert "error" not in res
    assert np.allclose(res["output"]["value"], np.eye(3), atol=1e-15)


def test_euler_axis_to_quaternion_z90():
    res = convert_attitude(
        "euler_axis", {"axis": [0.0, 0.0, 1.0], "angle": 90.0}, "quaternion"
    )
    q = np.array(res["output"]["value"])
    assert np.isclose(q[0], np.cos(np.pi / 4))
    assert np.isclose(q[3], np.sin(np.pi / 4))


def test_euler_angle_output_is_degrees_not_radians():
    """Guards the highest-risk trap: brahe returns radians from .phi/.theta/.psi."""
    res = convert_attitude(
        "euler_angle",
        {"angles": [10.0, 20.0, 30.0], "order": "ZYX"},
        "euler_angle",
        euler_order_out="ZYX",
    )
    assert np.allclose(res["output"]["value"]["angles"], [10.0, 20.0, 30.0], atol=1e-9)


def test_euler_axis_output_is_degrees():
    res = convert_attitude(
        "euler_axis", {"axis": [0.0, 0.0, 1.0], "angle": 90.0}, "euler_axis"
    )
    assert np.isclose(res["output"]["value"]["angle"], 90.0)


def test_radians_angle_format():
    res = convert_attitude(
        "euler_axis",
        {"axis": [0.0, 0.0, 1.0], "angle": np.pi / 2},
        "euler_axis",
        angle_format="radians",
    )
    assert np.isclose(res["output"]["value"]["angle"], np.pi / 2)


def test_scalar_first_false_reverses_ordering():
    a = convert_attitude("euler_axis", {"axis": [0.0, 0.0, 1.0], "angle": 90.0},
                         "quaternion", scalar_first=True)
    b = convert_attitude("euler_axis", {"axis": [0.0, 0.0, 1.0], "angle": 90.0},
                         "quaternion", scalar_first=False)
    qa = np.array(a["output"]["value"])
    qb = np.array(b["output"]["value"])
    assert np.allclose(qa, np.roll(qb, 1))
    assert b["output"]["scalar_first"] is False


def test_all_representation_pairs_roundtrip():
    reps = ["quaternion", "euler_axis", "euler_angle", "rotation_matrix"]
    start = {"axis": [0.0, 0.0, 1.0], "angle": 90.0}
    for target in reps:
        fwd = convert_attitude("euler_axis", start, target)
        assert "error" not in fwd, target
        back = convert_attitude(target, fwd["output"]["value"], "euler_axis")
        assert "error" not in back, target
        assert np.isclose(back["output"]["value"]["angle"], 90.0, atol=1e-9), target
        assert np.allclose(back["output"]["value"]["axis"], [0, 0, 1], atol=1e-9), target


def test_invalid_representation():
    res = convert_attitude("tensor", [1.0, 0.0, 0.0, 0.0], "quaternion")
    assert "error" in res
    assert "valid_representations" in res


def test_invalid_euler_order():
    res = convert_attitude(
        "euler_angle", {"angles": [1.0, 2.0, 3.0], "order": "ABC"}, "quaternion"
    )
    assert "error" in res


def test_bad_quaternion_length():
    res = convert_attitude("quaternion", [1.0, 0.0], "rotation_matrix")
    assert "error" in res


def test_bad_rotation_matrix_shape():
    res = convert_attitude("rotation_matrix", [[1.0, 0.0], [0.0, 1.0]], "quaternion")
    assert "error" in res


def test_non_orthogonal_rotation_matrix_returns_error_not_raise():
    """brahe's RotationMatrix.from_matrix raises brahe.BraheError (not a
    ValueError/KeyError/TypeError) for a 3x3 matrix that fails the proper
    rotation check. Confirmed directly: RotationMatrix.from_matrix on
    [[2,0,0],[0,1,0],[0,0,1]] raises BraheError with message "Matrix is not
    a proper rotation matrix. Determinant: 2, Orthogonal: false". This must
    surface as an error dict, never propagate as a raised exception.
    """
    res = convert_attitude(
        "rotation_matrix",
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "quaternion",
    )
    assert "error" in res


def test_euler_angle_output_is_radians_when_requested():
    """Covers the radians branch of _angles_from_radians.

    test_radians_angle_format only exercises euler_axis, which uses brahe's
    format-aware to_vector and never reaches this helper. Without this test,
    a helper that always converted to degrees would pass the whole suite.
    """
    res = convert_attitude(
        "euler_angle", {"angles": list(np.radians([10.0, 20.0, 30.0])), "order": "ZYX"},
        "euler_angle", angle_format="radians")
    assert np.allclose(
        res["output"]["value"]["angles"], np.radians([10.0, 20.0, 30.0]), atol=1e-9
    )


def test_euler_order_out_produces_correct_non_default_result():
    """euler_order_out is otherwise only exercised at its default value."""
    expected = EulerAngle(
        EulerAngleOrder.ZYX, 10.0, 20.0, 30.0, brahe.AngleFormat.DEGREES
    ).to_euler_angle(EulerAngleOrder.XYZ)
    expected_angles = [
        np.degrees(expected.phi),
        np.degrees(expected.theta),
        np.degrees(expected.psi),
    ]

    res = convert_attitude(
        "euler_angle", {"angles": [10.0, 20.0, 30.0], "order": "ZYX"},
        "euler_angle", euler_order_out="xyz",
    )
    assert "error" not in res
    assert res["output"]["value"]["order"] == "XYZ"
    assert np.allclose(res["output"]["value"]["angles"], expected_angles, atol=1e-9)
    assert not np.allclose(
        res["output"]["value"]["angles"], [10.0, 20.0, 30.0], atol=1e-6
    )


def test_zero_quaternion_produces_finite_error_not_nan_success():
    """Quaternion.from_vector silently normalizes [0,0,0,0] without raising,
    and every downstream converter (to_rotation_matrix, to_euler_axis,
    to_euler_angle) then emits NaN. Confirmed directly. Bare NaN is not
    valid JSON (RFC 8259), so this must be surfaced as an error dict rather
    than a success envelope containing non-finite values.
    """
    res = convert_attitude("quaternion", [0.0, 0.0, 0.0, 0.0], "rotation_matrix")
    assert "error" in res


def test_axis_rotation_matrix_matches_brahe_all_axes():
    for axis, fn in (("x", brahe.attitude.Rx),
                     ("y", brahe.attitude.Ry),
                     ("z", brahe.attitude.Rz)):
        res = axis_rotation_matrix(axis, 37.0)
        assert "error" not in res, axis
        assert np.allclose(res["output"]["matrix"],
                           np.array(fn(37.0, brahe.AngleFormat.DEGREES))), axis


def test_axis_rotation_matrix_all_axes_orthonormal():
    for ax in ("x", "y", "z"):
        m = np.array(axis_rotation_matrix(ax, 37.0)["output"]["matrix"])
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12), ax


def test_axis_rotation_radians():
    a = np.array(axis_rotation_matrix("x", 90.0)["output"]["matrix"])
    b = np.array(
        axis_rotation_matrix("x", np.pi / 2, angle_format="radians")["output"]["matrix"]
    )
    assert np.allclose(a, b)


def test_axis_rotation_invalid_axis():
    res = axis_rotation_matrix("w", 90.0)
    assert "error" in res
    assert "valid_axes" in res


def test_axis_rotation_non_finite_angle_returns_error_not_nan_success():
    """1e400 parses to inf; the matrix then contains NaN/Inf, which is not
    valid JSON (RFC 8259). This must surface as an error dict, never a
    success envelope containing non-finite values.
    """
    res = axis_rotation_matrix("z", 1e400)
    assert "error" in res
    assert "output" not in res


def test_compose_rotations_applies_first_element_first():
    """Pins the composition order: rotations[0] is applied first."""
    rots = [
        {"repr": "rotation_matrix",
         "value": np.array(brahe.attitude.Rz(90.0, brahe.AngleFormat.DEGREES)).tolist()},
        {"repr": "rotation_matrix",
         "value": np.array(brahe.attitude.Rx(90.0, brahe.AngleFormat.DEGREES)).tolist()},
    ]
    res = compose_rotations(rots)
    assert "error" not in res
    rz = np.array(brahe.attitude.Rz(90.0, brahe.AngleFormat.DEGREES))
    rx = np.array(brahe.attitude.Rx(90.0, brahe.AngleFormat.DEGREES))
    assert np.allclose(res["output"]["value"], rx @ rz, atol=1e-12)
    # Rx and Rz do not commute, so this genuinely discriminates the order.
    assert not np.allclose(rx @ rz, rz @ rx)


def test_compose_rotations_mixed_representations():
    rots = [
        {"repr": "quaternion", "value": [1.0, 0.0, 0.0, 0.0]},
        {"repr": "euler_axis", "value": {"axis": [0.0, 0.0, 1.0], "angle": 90.0}},
    ]
    res = compose_rotations(rots, output_repr="euler_axis")
    assert "error" not in res
    assert np.isclose(res["output"]["value"]["angle"], 90.0, atol=1e-9)


def test_compose_rotations_single_is_identity_passthrough():
    rots = [{"repr": "euler_axis",
             "value": {"axis": [0.0, 1.0, 0.0], "angle": 30.0}}]
    res = compose_rotations(rots, output_repr="euler_axis")
    assert np.isclose(res["output"]["value"]["angle"], 30.0, atol=1e-9)


def test_compose_rotations_empty_errors():
    res = compose_rotations([])
    assert "error" in res


def test_compose_rotations_bad_entry_errors():
    res = compose_rotations([{"value": [1.0, 0.0, 0.0, 0.0]}])
    assert "error" in res


def test_slerp_endpoints():
    q1 = [1.0, 0.0, 0.0, 0.0]
    q2 = np.array(
        convert_attitude("euler_axis",
                         {"axis": [0.0, 0.0, 1.0], "angle": 90.0},
                         "quaternion")["output"]["value"]
    ).tolist()
    at0 = quaternion_slerp(q1, q2, 0.0)
    at1 = quaternion_slerp(q1, q2, 1.0)
    assert np.allclose(at0["output"]["value"], q1, atol=1e-12)
    assert np.allclose(at1["output"]["value"], q2, atol=1e-12)


def test_slerp_midpoint_is_half_angle():
    q1 = [1.0, 0.0, 0.0, 0.0]
    q2 = np.array(
        convert_attitude("euler_axis",
                         {"axis": [0.0, 0.0, 1.0], "angle": 90.0},
                         "quaternion")["output"]["value"]
    ).tolist()
    mid = quaternion_slerp(q1, q2, 0.5, output_repr="euler_axis")
    assert np.isclose(mid["output"]["value"]["angle"], 45.0, atol=1e-9)


def test_slerp_t_out_of_range_errors():
    res = quaternion_slerp([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 1.5)
    assert "error" in res
    assert "t" in res["error"]


def test_slerp_bad_quaternion_errors():
    res = quaternion_slerp([1.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.5)
    assert "error" in res


def test_slerp_non_numeric_t_returns_error_not_raise():
    """A non-numeric t must return an error dict, never raise TypeError."""
    res = quaternion_slerp([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], "not-a-number")
    assert "error" in res
