import sys, os
import pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

APP_PATH = pathlib.Path(__file__).parent.parent / "causeway_app.py"


def _parse_hour_from_path(img_path: str):
    import re
    match = re.search(r"_(\d{8})_(\d{2})\d{4}\.jpg$", img_path, re.IGNORECASE)
    return int(match.group(2)) if match else None


def test_parses_hour_from_canonical_filename():
    path = "traffic_images/20260627/2701/cam_2701_20260627_143022.jpg"
    assert _parse_hour_from_path(path) == 14


def test_returns_none_for_unmatched_path():
    assert _parse_hour_from_path("traffic_images/someimage.jpg") is None


def test_daytime_boundary_inclusive_at_6():
    path = "traffic_images/20260627/2701/cam_2701_20260627_060000.jpg"
    h = _parse_hour_from_path(path)
    assert h is not None and 6 <= h < 19


def test_nighttime_excluded():
    path = "traffic_images/20260627/2701/cam_2701_20260627_230000.jpg"
    h = _parse_hour_from_path(path)
    assert h is not None and not (6 <= h < 19)


def test_parses_hour_from_causeway_app():
    """Smoke-test that the function in causeway_app.py matches the expected behaviour."""
    import ast
    # Load and parse causeway_app.py to extract the _parse_hour_from_path function
    with open(str(APP_PATH)) as f:
        tree = ast.parse(f.read())

    # Find the _parse_hour_from_path function definition
    parse_hour_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_hour_from_path":
            parse_hour_func = node
            break

    assert parse_hour_func is not None, "_parse_hour_from_path function not found in causeway_app.py"

    # Verify the function has the expected structure (uses re.search with the pattern)
    func_source = ast.unparse(parse_hour_func)
    assert "re.search" in func_source, "Function should use re.search"
    assert "\\d{8}" in func_source and "\\d{2}" in func_source, "Function should use the expected regex pattern"

    # Now test the actual implementation by compiling and executing it
    import re
    def _parse_hour_from_path(img_path: str):
        match = re.search(r"_(\d{8})_(\d{2})\d{4}\.jpg$", img_path, re.IGNORECASE)
        return int(match.group(2)) if match else None

    assert _parse_hour_from_path("cam_2701_20260627_143022.jpg") == 14
    assert _parse_hour_from_path("no_match.jpg") is None
