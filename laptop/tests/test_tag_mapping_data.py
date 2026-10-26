"""Precision, provenance, synchronization and replay fixtures."""

from pathlib import Path
import csv
import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tag_mapping.data import RunWriter, file_hash, load_events
from tag_mapping.config import load_config
from tag_mapping.geometry import ground_corners, inverse, mount, project, transform
from tag_mapping.graph import build_graph
from tag_mapping.synchronization import ClockMap, interpolate_pose, synchronize
from test_tag_mapping_config import complete_config, save_config


def synthetic_run(tmp_path, noise=0.12):
    """Noise is seeded; telemetry drifts while camera pixels remain in fixed NED."""
    c = complete_config()
    origin = np.array([4.0, -2.0, 1.0])
    truth = {
        ("tag16h5", 0): np.array([0.85, 0.05, 0.2]),
        ("tag16h5", 1): np.array([1.0, 0.22, -0.35]),
    }
    rng = np.random.default_rng(1302)

    def pose_at(t):
        r = Rotation.from_euler(
            "xyz", [0.04 + 0.03 * t, -0.03 + 0.02 * t, 0.2 + 0.25 * t]
        ).as_matrix()
        return transform(r, [0.18 * t, 0.04 * np.sin(3 * t), -0.7 - 0.025 * t])

    first = pose_at(0)
    start = dict(
        fc_time_ns=1_000_000_000,
        host_receive_ns=10_000_000_000,
        segment_id=0,
        native_position_m=(first[:3, 3] + origin).tolist(),
        quaternion_xyzw=Rotation.from_matrix(first[:3, :3]).as_quat().tolist(),
        initial_body_height_above_floor_m=0.7,
        initial_body_height_sigma_m=0.01,
        coordinate_convention="NED",
        datum_source="synthetic_measured_floor",
    )
    run = tmp_path / "run"
    with RunWriter(run, c, start) as w:
        for k in range(36):
            t = k * 0.02
            true = pose_at(t)
            measured = true.copy()
            measured[:3, 3] += np.array([0.09 * t, -0.04 * t, 0.015 * t])
            native = measured[:3, 3] + origin
            q = Rotation.from_matrix(measured[:3, :3]).as_quat()
            if k % 2:
                q = -q
            envelope = dict(
                segment_id=0,
                fc_time_ns=1_000_000_000 + k * 20_000_000,
                host_receive_ns=10_000_000_000 + k * 20_000_000,
                sysid=1,
                compid=1,
                sequence=k,
            )
            w.append(
                "position",
                **envelope,
                **dict(zip(("x", "y", "z"), native)),
                vx=0.18,
                vy=0.0,
                vz=0.0,
            )
            w.append(
                "attitude",
                **envelope,
                qx=float(q[0]),
                qy=float(q[1]),
                qz=float(q[2]),
                qw=float(q[3]),
                rollspeed=0.0,
                pitchspeed=0.0,
                yawspeed=0.25,
            )
            if k % 5 == 0 and k > 0:
                camera = true @ mount(c)
                dets = []
                for key, tag in truth.items():
                    pc = (ground_corners(tag, 0.12) - camera[:3, 3]) @ camera[:3, :3]
                    uv = project(pc, c, "apriltag_pixel_edges") + rng.normal(
                        0, noise, (4, 2)
                    )
                    dets.append(
                        dict(
                            family_or_dictionary=key[0],
                            tag_id=key[1],
                            corners=uv.tolist(),
                            hamming=0,
                            margin=100.0,
                        )
                    )
                # One detection-free frame still contributes motion connectivity.
                if k == 15:
                    dets = []
                w.append(
                    "frame",
                    segment_id=0,
                    camera_id="camera0",
                    boot_nonce=7,
                    frame_id=k,
                    capture_time_ns=envelope["fc_time_ns"],
                    clock_domain="fc_boot",
                    capture_timestamp_reference="exposure_midpoint",
                    exposure_verified=True,
                    host_receive_ns=envelope["host_receive_ns"],
                    width=320,
                    height=240,
                    calibration_id=c["camera"]["calibration_id"],
                    pixel_convention="apriltag_pixel_edges",
                    image_path=None,
                    detections=dets,
                    status="synthetic",
                )
    path = save_config(tmp_path, c)
    return load_config(path), run, path, truth


def make_graph(tmp_path, noise=0.12):
    c, run, path, truth = synthetic_run(tmp_path, noise)
    events = load_events(run)
    poses, rows, context = synchronize(events, run, c)
    return build_graph(poses, context, c), rows, context, (run, path, truth)


def test_roundtrip_csv_and_unknowns(tmp_path):
    c, run, path, truth = synthetic_run(tmp_path)
    events = load_events(run)
    frames = [e for e in events if e["kind"] == "frame"]
    with (run / "csv/detections.csv").open() as h:
        rows = list(csv.DictReader(h))
    original = frames[0]["detections"][0]["corners"][0][0]
    assert float(rows[0]["c0_u"]) == original
    with (run / "csv/frames.csv").open() as h:
        assert any(r["detection_count"] == "0" for r in csv.DictReader(h))
    assert json.loads((run / "hashes.json").read_text())["events.jsonl"] == file_hash(
        run / "events.jsonl"
    )
    with pytest.raises(FileExistsError):
        RunWriter(run, c)


def test_missing_exposure_is_rejected_without_zero_fallback(tmp_path):
    c, run, _, _ = synthetic_run(tmp_path)
    events = load_events(run)
    for e in events:
        if e["kind"] == "frame":
            e["exposure_verified"] = False
            e["capture_time_ns"] = None
    poses, rows, _ = synchronize(events, run, c)
    assert not poses and all(
        r["status"] == "rejected" and r["fc_time_ns"] is None for r in rows
    )
    c["legacy_replay"]["allow_assumed_pose"] = True
    poses, rows, _ = synchronize(events, run, c)
    assert poses and all(r["status"] == "approximate" for r in rows)


def test_clock_fit_centres_epoch_and_preserves_nanoseconds():
    x0 = 1_800_000_000_000_000_000
    y0 = 1_234_567_890
    samples = [
        dict(
            camera_time_ns=x0 + i * 1_000_000_000,
            fc_time_ns=y0 + i * 1_000_020_000,
            sigma_ns=10,
        )
        for i in range(5)
    ]
    clock = ClockMap.fit(samples)
    assert clock.map(x0 + 2_000_000_123) == y0 + 2_000_040_123
    samples[-1]["fc_time_ns"] += 1_000_000
    with pytest.raises(ValueError, match="inconsistent"):
        ClockMap.fit(samples)
    assert ClockMap(1.0, -x0 + y0).map(x0 + 123) == y0 + 123


def test_interpolation_out_of_order_signs_and_gaps(tmp_path):
    c, run, _, _ = synthetic_run(tmp_path)
    events = load_events(run)
    p = [e for e in events if e["kind"] == "position"]
    a = [e for e in events if e["kind"] == "attitude"]
    r = interpolate_pose(p[::-1], a[::-1], 1_010_000_000, 50_000_000)
    assert np.isfinite(r).all() and np.linalg.det(r[:3, :3]) == pytest.approx(1)
    with pytest.raises(ValueError, match="gap"):
        interpolate_pose(p[::5], a[::5], 1_010_000_000, 50_000_000)
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_pose(p, a, 0, 50_000_000)


def test_verified_metadata_exact_identity_and_camera_clock_reboot(tmp_path):
    c, run, _, _ = synthetic_run(tmp_path)
    c["timing"].update(
        scale=1.0,
        offset_ns=1_000_000,
        capture_timestamp_offset_ns=0,
        sync_sigma_ns=1000,
    )
    events = load_events(run)
    metadata = []
    for e in events:
        if e["kind"] == "frame":
            metadata.append(
                dict(
                    run_id=e["run_id"],
                    camera_id=e["camera_id"],
                    boot_nonce=e["boot_nonce"],
                    frame_id=e["frame_id"],
                    exposure_verified=True,
                    capture_timestamp_reference="exposure_midpoint",
                    clock_domain="camera_boot",
                    capture_time_ns=e["capture_time_ns"] - 1_000_000,
                )
            )
            e["exposure_verified"] = False
            e["capture_time_ns"] = None
    poses, rows, _ = synchronize(events, run, c, metadata)
    assert len(poses) == len(metadata) and all(
        r["status"] == "synchronized" for r in rows
    )
    next(e for e in events if e["kind"] == "frame")["camera_clock_reset"] = True
    poses, rows, _ = synchronize(events, run, c, metadata)
    assert rows[0]["status"] == "rejected" and "reboot" in rows[0]["reason"]
    metadata[0]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="run ID"):
        synchronize(events, run, c, metadata)


def test_reset_segment_is_not_connected_to_start(tmp_path):
    c, run, _, _ = synthetic_run(tmp_path)
    events = load_events(run)
    for e in events:
        if (
            e.get("fc_time_ns", 0) > 1_400_000_000
            or e.get("capture_time_ns", 0) > 1_400_000_000
        ):
            e["segment_id"] = 1
    poses, rows, context = synchronize(events, run, c)
    graph = build_graph(poses, context, c)
    assert all(r["segment"] == 0 for r in graph.pose_records)
    assert any("reset_segment" in d.get("reason", "") for d in graph.diagnostics)


def test_recorder_preserves_wire_bytes_and_each_completed_frame(tmp_path, monkeypatch):
    import struct
    import cv2
    from pymavlink.dialects.v20 import common as mavlink
    from protocol import (
        CAMERA_HEADER_FMT,
        CAMERA_MAGIC,
        CAMERA_VERSION,
        CAMERA_FLAG_END,
    )
    import tag_mapping.recording as recording

    c = complete_config()
    _, encoded = cv2.imencode(".jpg", np.full((240, 320), 255, np.uint8))
    jpeg = encoded.tobytes()
    packets = []
    for frame_id in (1, 2):
        packets.append(
            struct.pack(
                CAMERA_HEADER_FMT,
                CAMERA_MAGIC,
                CAMERA_VERSION,
                0,
                22,
                7,
                frame_id,
                0,
                0,
                320,
                240,
                len(jpeg),
                90,
                0,
            )
            + jpeg
        )
        packets.append(
            struct.pack(
                CAMERA_HEADER_FMT,
                CAMERA_MAGIC,
                CAMERA_VERSION,
                CAMERA_FLAG_END,
                22,
                7,
                frame_id,
                len(jpeg),
                len(jpeg),
                320,
                240,
                0,
                90,
                0,
            )
        )
    mav = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
    position = mav.local_position_ned_encode(1000, 1.0, 2.0, -0.5, 0.1, 0.0, 0.0).pack(
        mav
    )
    attitude = mav.attitude_quaternion_encode(
        1000, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ).pack(mav)

    class FakeSocket:
        def __init__(self, queue):
            self.queue = list(queue)
            self.sent = []

        def bind(self, *args):
            pass

        def setblocking(self, *args):
            pass

        def close(self):
            pass

        def recvfrom(self, *args):
            return self.queue.pop(0), ("127.0.0.1", 8888)

        def sendto(self, data, addr):
            self.sent.append((data, addr))

    cam = FakeSocket(packets)
    telem = FakeSocket([position, attitude])
    tx = FakeSocket([])
    sockets = iter([cam, telem, tx])
    monkeypatch.setattr(recording.socket, "socket", lambda *args: next(sockets))

    def select_ready(*args):
        ready = [s for s in (cam, telem) if s.queue]
        if not ready:
            raise KeyboardInterrupt
        return ready, [], []

    monkeypatch.setattr(recording.select, "select", select_ready)

    class EmptyDetector:
        truncated = 0

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect(self, image):
            return []

    monkeypatch.setattr(recording, "Detector", EmptyDetector)
    monkeypatch.setattr(recording, "library_path", lambda: Path(__file__))
    run = tmp_path / "recorded"
    recording.record(run, c, "127.0.0.1")
    assert (run / "raw/mavlink.bin").read_bytes() == position + attitude
    assert (run / "raw/camera.bin").read_bytes() == b"".join(packets)
    events = load_events(run)
    frames = [e for e in events if e["kind"] == "frame"]
    assert len(frames) == 2 and all(
        not e["detections"] and e["capture_time_ns"] is None for e in frames
    )
    assert all((run / e["image_path"]).read_bytes() == jpeg for e in frames)
    assert any(e["kind"] == "attitude" and e["qw"] == 1 for e in events)
    with (run / "csv/frames.csv").open() as h:
        assert all(row["capture_time_ns"] == "" for row in csv.DictReader(h))
    assert (
        tx.sent and tx.sent[-1][0][2] == 0
    )  # Preview disabled on exit; no flight command.
