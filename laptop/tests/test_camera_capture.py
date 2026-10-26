"""Check automatic capture timing and the viewer without a camera or GUI."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camera_stream
from camera_stream import Frame, FrameSaver, _parse_args
from protocol import build_camera_stream_command


def test_interval_and_original_pixels(tmp_path):
    image = np.arange(240 * 320, dtype=np.uint8).reshape(240, 320)
    saver = FrameSaver(tmp_path, interval=4)
    assert saver.save_due(image, 10) is None
    assert saver.save_due(image, 13.99) is None
    first = saver.save_due(image, 14)
    assert first is not None
    np.testing.assert_array_equal(cv2.imread(str(first), cv2.IMREAD_GRAYSCALE), image)
    assert saver.save_due(image, 14.01) is None
    # A long interruption produces one current photo, without a catch-up burst.
    second = saver.save_due(image, 100)
    assert second != first
    assert saver.save_due(image, 100) is None
    assert saver.save_due(image, 103.99) is None
    assert saver.save_due(image, 104) is not None
    assert saver.count == 3


def test_manual_capture_with_automatic_capture_disabled(tmp_path):
    image = np.zeros((24, 32), dtype=np.uint8)
    saver = FrameSaver(tmp_path)
    assert saver.save_due(image, 100) is None
    first, second = saver.save(image), saver.save(image)
    assert first != second
    assert first.exists() and second.exists()
    assert saver.count == 2


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_intervals_are_rejected(value):
    with pytest.raises(SystemExit) as exc:
        _parse_args(["--esp-ip", "127.0.0.1", f"--save-interval={value}"])
    assert exc.value.code == 2


def test_capture_default_and_project_folder_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _parse_args(["--esp-ip", "127.0.0.1"], default_save_interval=4)
    assert args.save_interval == 4
    assert args.save_dir == Path(__file__).resolve().parents[2] / "calibration_images"


def test_viewer_only_saves_new_frames_and_stops_stream_on_exit(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(camera_stream.time, "monotonic", lambda: clock[0])
    image = np.arange(32 * 24, dtype=np.uint8).reshape(24, 32)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()
    frames = iter([Frame(1, jpeg, 10, 0, 0), Frame(2, jpeg, 10, 0, 4),
                   None, Frame(3, jpeg, 10, 0, 12)])
    monkeypatch.setattr(camera_stream, "drain_socket",
                        lambda *args: (next(frames), 0, None))
    monkeypatch.setattr(camera_stream.select, "select", lambda *args: ([], [], []))

    sockets = []

    class FakeSocket:
        def __init__(self, *args):
            self.sent = []
            self.closed = False
            sockets.append(self)

        def setsockopt(self, *args): pass
        def bind(self, *args): pass
        def setblocking(self, *args): pass

        def sendto(self, data, target):
            self.sent.append((data, target))

        def close(self):
            self.closed = True

    monkeypatch.setattr(camera_stream.socket, "socket", FakeSocket)
    for name in ("namedWindow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    previews = []
    monkeypatch.setattr(cv2, "imshow", lambda name, pixels: previews.append(pixels.copy()))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    keys = iter([-1, -1, -1, ord("q")])

    def wait_key(*args):
        clock[0] += 4
        return next(keys)

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    assert camera_stream.main(["--esp-ip", "127.0.0.1", "--save-dir", str(tmp_path)],
                              default_save_interval=4) == 0
    # At 4 s and 12 s; no photo of the frozen preview at 8 s.
    photos = list(tmp_path.glob("*.png"))
    assert len(photos) == 2
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    for photo in photos:
        np.testing.assert_array_equal(cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE), decoded)
    assert previews[-1].shape == (480, 640, 3)
    assert sockets[1].sent[-1][0] == build_camera_stream_command(False)
    assert all(sock.closed for sock in sockets)
