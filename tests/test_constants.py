import brahe

from brahe_mcp.constants import list_constants, get_constant


def test_list_constants_returns_all_categories():
    result = list_constants()
    assert "physical" in result
    assert "math" in result
    assert "time" in result


def test_list_constants_contains_known_entry():
    result = list_constants()
    names = [entry["name"] for entry in result["physical"]]
    assert "R_EARTH" in names


def test_get_constant_known():
    result = get_constant("R_EARTH")
    assert result["name"] == "R_EARTH"
    assert result["value"] == brahe.R_EARTH
    assert result["unit"] == "m"


def test_get_constant_case_insensitive():
    result = get_constant("gm_earth")
    assert result["name"] == "GM_EARTH"
    assert result["value"] == brahe.GM_EARTH


def test_get_constant_unknown():
    result = get_constant("NOT_A_CONSTANT")
    assert "error" in result
    assert "valid_names" in result
