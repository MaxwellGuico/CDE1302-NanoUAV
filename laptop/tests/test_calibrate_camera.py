"""Exercise detection and calibration on rendered views with known intrinsics."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_camera import (DICTIONARIES, calibrate_images, image_paths, main,
                              make_board)


pytestmark = pytest.mark.skipif(
    not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"),
    reason="ChArUco calibration requires OpenCV 4.8+",
)


@pytest.fixture
def rendered_views(tmp_path):
    board = make_board()
    texture = board.generateImage((1200, 1650))
    matrix = np.array([[950., 0., 480.], [0., 930., 360.], [0., 0., 1.]])
    outline = np.float32([[0, 0, 0], [.12, 0, 0], [.12, .165, 0], [0, .165, 0]])
    source = np.float32([[0, 0], [1200, 0], [1200, 1650], [0, 1650]])
    rng = np.random.default_rng(1302)
    paths = []
    for index in range(15):
        rvec = rng.uniform([-.6, -.6, -.25], [.6, .6, .25])
        rotation, _ = cv2.Rodrigues(rvec)
        center = rng.uniform([-.055, -.035, .38], [.055, .035, .52])
        tvec = center - rotation @ np.array([.06, .0825, 0])
        target, _ = cv2.projectPoints(outline, rvec, tvec, matrix, np.zeros(5))
        transform = cv2.getPerspectiveTransform(source, target.reshape(4, 2))
        image = cv2.warpPerspective(texture, transform, (960, 720), borderValue=255)
        path = tmp_path / f"view_{index:02d}.png"
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return board, matrix, paths


def test_recovers_intrinsics_from_images_and_saves_diagnostics(rendered_views, tmp_path):
    _, expected, paths = rendered_views
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((720, 960), 255, dtype=np.uint8))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    output = tmp_path / "results" / "intrinsics.json"
    debug = tmp_path / "detections"
    assert main(["--images", str(tmp_path), "--output", str(output),
                 "--debug-dir", str(debug)]) == 0
    result = json.loads(output.read_text())
    actual = np.array(result["camera_matrix"])
    np.testing.assert_allclose(actual.diagonal()[:2], expected.diagonal()[:2], rtol=.02)
    np.testing.assert_allclose(actual[:2, 2], expected[:2, 2], atol=5)
    assert result["rms_reprojection_error_px"] < .6
    assert result["num_views"] == len(paths)
    assert len(result["skipped"]) == 2
    assert len(result["dist_coeffs"]) == 5
    assert result["board"]["square_length_mm"] == 15
    assert len(list(debug.glob("*.png"))) == len(paths) + 1
    assert all(v["rms_reprojection_error_px"] < .6 for v in result["views"])


@pytest.mark.parametrize("dictionary", DICTIONARIES)
def test_default_detector_recognizes_each_standard_6x6_dictionary(dictionary):
    printed_board = make_board(dictionary=dictionary)
    image = printed_board.generateImage((1000, 1300), marginSize=40)
    _, ids, _, _ = cv2.aruco.CharucoDetector(make_board()).detectBoard(image)
    assert len(ids) == 70  # (8 - 1) * (11 - 1) internal corners


def test_mixed_resolutions_fail(tmp_path):
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path, shape in zip(paths, [(720, 960), (480, 640)]):
        cv2.imwrite(str(path), np.full(shape, 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="Mixed image resolutions"):
        calibrate_images(paths, make_board())


def test_complete_checkerboard_calibration_recovers_intrinsics(rendered_views):
    board, expected, paths = rendered_views
    result = calibrate_images(paths, board, detector_type="chessboard")
    actual = np.array(result["camera_matrix"])
    np.testing.assert_allclose(actual.diagonal()[:2], expected.diagonal()[:2], rtol=.02)
    np.testing.assert_allclose(actual[:2, 2], expected[:2, 2], atol=5)
    assert result["feature_detector"] == "chessboard"
    assert result["num_views"] == len(paths)
    assert all(view["corners"] == 70 for view in result["views"])
    assert result["rms_reprojection_error_px"] < .6


def test_insufficient_views_does_not_write_output(rendered_views, tmp_path):
    _, _, paths = rendered_views
    output = tmp_path / "intrinsics.json"
    assert main(["--images", str(paths[0]), "--output", str(output)]) == 1
    assert not output.exists()


def test_input_expansion_deduplicates_and_reports_missing(tmp_path):
    image = tmp_path / "frame.JPG"
    image.touch()
    assert image_paths([str(tmp_path), str(image), str(tmp_path / "*.JPG")]) == [image]
    with pytest.raises(ValueError, match="No images found"):
        image_paths([str(tmp_path / "missing")])


@pytest.mark.parametrize("kwargs", [
    {"marker_mm": 15}, {"square_mm": float("nan")}, {"squares_x": 1},
    {"squares_x": 20, "dictionary": "DICT_6X6_50"},
])
def test_invalid_board_configuration(kwargs):
    with pytest.raises(ValueError):
        make_board(**kwargs)
