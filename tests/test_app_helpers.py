import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


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
