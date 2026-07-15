import json

import cv2
import numpy as np

from yolo_toolkit.dataset.motion_blur_filter import blur_score, filter_dataset, rectangle_rois


def test_rectangle_rois_clips_points_to_image_bounds():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    annotation = {"shapes": [{"shape_type": "rectangle", "points": [[-2, 2], [6, 8]]}]}

    rois = rectangle_rois(image, annotation)

    assert len(rois) == 1
    assert rois[0].shape == (6, 6, 3)


def test_blur_score_is_higher_for_a_blurred_roi():
    sharp = np.zeros((100, 100), dtype=np.uint8)
    sharp[:, ::4] = 255
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)

    assert blur_score(blurred) > blur_score(sharp)


def test_filter_dataset_keeps_unannotated_and_clear_annotated_images(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    clear = np.zeros((100, 100, 3), dtype=np.uint8)
    clear[:, ::4] = 255
    blurry = cv2.GaussianBlur(clear, (0, 0), 4)
    annotation = {"shapes": [{"shape_type": "rectangle", "points": [[0, 0], [100, 100]]}]}

    cv2.imwrite(str(input_dir / "clear.png"), clear)
    cv2.imwrite(str(input_dir / "blurry.png"), blurry)
    cv2.imwrite(str(input_dir / "empty.png"), clear)
    (input_dir / "clear.json").write_text(json.dumps(annotation), encoding="utf-8")
    (input_dir / "blurry.json").write_text(json.dumps(annotation), encoding="utf-8")

    scores = filter_dataset(input_dir, output_dir, threshold=0.01)

    assert scores["blurry.png"] > 0.01
    assert (output_dir / "with_target" / "clear.png").is_file()
    assert (output_dir / "with_target" / "clear.json").is_file()
    assert not (output_dir / "with_target" / "blurry.png").exists()
    assert not (output_dir / "with_target" / "blurry.json").exists()
    assert (output_dir / "without_target" / "empty.png").is_file()
