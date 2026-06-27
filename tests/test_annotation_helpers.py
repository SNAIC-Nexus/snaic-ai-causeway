import pytest
import numpy as np
from causeway.db import init_db, ensure_label_log_entry, get_label_logs
from causeway.annotation_helpers import (
    yolo_to_boxes, boxes_to_yolo_lines, canvas_rect_to_box,
    render_annotated_image, list_annotation_dates, list_images_for_annotation,
    CLASS_NAMES, CLASS_COLOURS, DISPLAY_W, DISPLAY_H,
)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("causeway.db.DB_PATH", db_file)
    init_db()
    return db_file


def test_ensure_label_log_entry_inserts_when_absent(tmp_db):
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    rows = get_label_logs(label_type="vehicle")
    assert len(rows) == 1
    assert rows[0][0] == "/img/a.jpg"
    assert rows[0][4] == "pending"


def test_ensure_label_log_entry_does_not_duplicate(tmp_db):
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    rows = get_label_logs(label_type="vehicle")
    assert len(rows) == 1


def test_ensure_label_log_entry_preserves_existing_validated(tmp_db):
    from causeway.db import update_label_validation
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    update_label_validation("/img/a.jpg", "vehicle", "approved")
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    rows = get_label_logs(label_type="vehicle", validated="approved")
    assert len(rows) == 1


def test_class_names_and_colours():
    assert CLASS_NAMES[0] == "Motorcycle"
    assert CLASS_NAMES[1] == "Car"
    assert CLASS_NAMES[2] == "Bus"
    assert CLASS_NAMES[3] == "Truck"
    assert CLASS_COLOURS[0]["stroke"] == "#0000FF"
    assert CLASS_COLOURS[1]["stroke"] == "#FFFF00"
    assert CLASS_COLOURS[2]["stroke"] == "#00FF00"
    assert CLASS_COLOURS[3]["stroke"] == "#00FFFF"


def test_yolo_to_boxes_empty_when_no_file(tmp_path):
    boxes = yolo_to_boxes(str(tmp_path / "missing.txt"), 320, 240)
    assert boxes == []


def test_yolo_to_boxes_parses_correctly(tmp_path):
    label = tmp_path / "test.txt"
    label.write_text("1 0.5 0.5 0.4 0.6\n")
    boxes = yolo_to_boxes(str(label), 320, 240)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["class_id"] == 1
    assert abs(b["x1_n"] - 0.3) < 1e-5   # cx - w/2 = 0.5 - 0.2
    assert abs(b["y1_n"] - 0.2) < 1e-5   # cy - h/2 = 0.5 - 0.3
    assert abs(b["x2_n"] - 0.7) < 1e-5
    assert abs(b["y2_n"] - 0.8) < 1e-5


def test_yolo_to_boxes_skips_malformed_lines(tmp_path):
    label = tmp_path / "test.txt"
    label.write_text("1 0.5 0.5 0.4 0.6\nbad line\n2 0.1 0.1 0.2 0.2\n")
    boxes = yolo_to_boxes(str(label), 320, 240)
    assert len(boxes) == 2


def test_boxes_to_yolo_lines_roundtrip(tmp_path):
    label = tmp_path / "test.txt"
    label.write_text("1 0.500000 0.500000 0.400000 0.600000\n")
    boxes = yolo_to_boxes(str(label), 320, 240)
    lines = boxes_to_yolo_lines(boxes)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "1"
    assert abs(float(parts[1]) - 0.5) < 1e-5  # cx
    assert abs(float(parts[2]) - 0.5) < 1e-5  # cy
    assert abs(float(parts[3]) - 0.4) < 1e-5  # w
    assert abs(float(parts[4]) - 0.6) < 1e-5  # h


def test_boxes_to_yolo_lines_empty():
    assert boxes_to_yolo_lines([]) == []


def test_canvas_rect_to_box_normalises_correctly():
    # orig 640x480, display 640x480 (1:1 scale)
    rect = {"left": 64.0, "top": 48.0, "width": 128.0, "height": 96.0, "scaleX": 1.0, "scaleY": 1.0}
    box = canvas_rect_to_box(rect, class_id=2, orig_w=640, orig_h=480)
    assert box["class_id"] == 2
    assert abs(box["x1_n"] - 0.1) < 1e-5
    assert abs(box["y1_n"] - 0.1) < 1e-5
    assert abs(box["x2_n"] - 0.3) < 1e-5
    assert abs(box["y2_n"] - 0.3) < 1e-5


def test_canvas_rect_to_box_clamps_to_0_1():
    rect = {"left": -10.0, "top": -10.0, "width": 700.0, "height": 500.0, "scaleX": 1.0, "scaleY": 1.0}
    box = canvas_rect_to_box(rect, class_id=0, orig_w=640, orig_h=480)
    assert box["x1_n"] >= 0.0
    assert box["y1_n"] >= 0.0
    assert box["x2_n"] <= 1.0
    assert box["y2_n"] <= 1.0


def test_canvas_rect_to_box_respects_scale():
    # scaleX=2 means the rect was resized to double width in Fabric.js
    rect = {"left": 0.0, "top": 0.0, "width": 100.0, "height": 100.0, "scaleX": 2.0, "scaleY": 1.0}
    box = canvas_rect_to_box(rect, class_id=1, orig_w=640, orig_h=480)
    assert abs(box["x2_n"] - 200.0 / 640) < 1e-5


def test_render_annotated_image_returns_rgb_array(tmp_path):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img_path = str(tmp_path / "frame.jpg")
    import cv2
    cv2.imwrite(img_path, img)
    boxes = [{"class_id": 1, "x1_n": 0.1, "y1_n": 0.1, "x2_n": 0.5, "y2_n": 0.5}]
    result = render_annotated_image(img_path, boxes, 320, 240)
    assert result.shape == (DISPLAY_H, DISPLAY_W, 3)
    assert result.dtype == np.uint8


def test_list_annotation_dates_returns_sorted_descending(tmp_path):
    for date in ["20260625", "20260627", "20260626"]:
        p = tmp_path / date / "2701"
        p.mkdir(parents=True)
        (p / "img.jpg").touch()
    dates = list_annotation_dates("2701", base_dir=str(tmp_path))
    assert dates == ["20260627", "20260626", "20260625"]


def test_list_annotation_dates_filters_by_camera(tmp_path):
    (tmp_path / "20260627" / "2701").mkdir(parents=True)
    (tmp_path / "20260627" / "2701" / "img.jpg").touch()
    (tmp_path / "20260627" / "2702").mkdir(parents=True)
    (tmp_path / "20260627" / "2702" / "img.jpg").touch()
    dates_2701 = list_annotation_dates("2701", base_dir=str(tmp_path))
    dates_2702 = list_annotation_dates("2702", base_dir=str(tmp_path))
    assert "20260627" in dates_2701
    assert "20260627" in dates_2702


def test_list_images_for_annotation_returns_sorted_jpgs(tmp_path):
    cam_dir = tmp_path / "20260627" / "2701"
    cam_dir.mkdir(parents=True)
    (cam_dir / "b.jpg").touch()
    (cam_dir / "a.jpg").touch()
    (cam_dir / "note.txt").touch()
    imgs = list_images_for_annotation("2701", "20260627", base_dir=str(tmp_path))
    assert len(imgs) == 2
    assert imgs[0].endswith("a.jpg")
    assert imgs[1].endswith("b.jpg")


def test_list_images_for_annotation_empty_when_no_match(tmp_path):
    imgs = list_images_for_annotation("2701", "20260627", base_dir=str(tmp_path))
    assert imgs == []
