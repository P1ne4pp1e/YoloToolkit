import numpy as np

from yolo_toolkit.dataset.pose_augmentation import (
    KEYPOINT_ORDER,
    apply_white_balance,
    all_points_inside,
    letterbox_to_square,
    polygon_intersects_rectangle,
    translate_points,
    yolo_pose_line,
)


def test_translate_points_preserves_named_corner_order():
    points = np.array([[1, 2], [5, 2], [5, 6], [1, 6]], dtype=np.float32)

    translated = translate_points(points, 3, -1)

    assert KEYPOINT_ORDER == ("left_upper", "right_upper", "right_lower", "left_lower")
    assert translated.tolist() == [[4.0, 1.0], [8.0, 1.0], [8.0, 5.0], [4.0, 5.0]]


def test_all_points_inside_rejects_any_out_of_frame_corner():
    points = np.array([[0, 0], [9, 0], [9, 9], [-0.1, 9]], dtype=np.float32)

    assert not all_points_inside(points, width=10, height=10)


def test_polygon_intersects_rectangle_detects_overlap_without_corner_containment():
    polygon = np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=np.float32)

    assert polygon_intersects_rectangle(polygon, 0, 4, 10, 6)


def test_white_balance_changes_channel_balance():
    image = np.full((2, 2, 3), 100, dtype=np.uint8)

    shifted = apply_white_balance(image, __import__("random").Random(1))

    assert not np.array_equal(shifted[:, :, 0], shifted[:, :, 2])


def test_letterbox_and_yolo_label_preserve_complete_target_geometry():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    points = np.array([[20, 10], [180, 10], [180, 90], [20, 90]], dtype=np.float32)

    resized, transformed = letterbox_to_square(image, points, size=640)

    assert resized.shape == (640, 640, 3)
    assert transformed.tolist() == [[64.0, 192.0], [576.0, 192.0], [576.0, 448.0], [64.0, 448.0]]
    assert len(yolo_pose_line(transformed).split()) == 17
