from ocp_solver import solve_ocp
import pytest

def test_solve_ocp_basic():
    result = solve_ocp(1.0)
    assert "x" in result and "h" in result and "u" in result and "J" in result
    assert isinstance(result["x"], list)
    assert isinstance(result["J"], float)
    assert len(result["x"]) == len(result["h"])

def test_solve_ocp_extreme():
    result = solve_ocp(0.75)
    assert result["J"] >= 0

    result = solve_ocp(1.25)
    assert result["J"] >= 0

def test_invalid_k():
    with pytest.raises(ValueError):
        solve_ocp(-5.0)

def test_image_base64():
    import base64
    image = solve_ocp(1.25)["image_base64"]
    decoded = base64.b64decode(image)
    assert len(decoded) > 1000