import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_common import (
    AprilTagReceiver,
    circular_mean_rad,
    max_consecutive,
    wrap_angle_deg,
)
from protocol import AtDebugPacket, TelemetryPacket, parse_telemetry
from speed_detection_baseline import generate_trajectory


def test_wrapped_yaw_and_circular_mean_cross_zero() -> None:
    assert wrap_angle_deg(358.0) == pytest.approx(-2.0)
    mean = circular_mean_rad([math.radians(359), math.radians(1)])
    assert abs(math.degrees(mean)) < 1e-6


def test_apriltag_receiver_deduplicates_frame_ms_without_socket() -> None:
    receiver = AprilTagReceiver(2)
    packet = AtDebugPacket(2, 1234, 250, -1, 0, [])
    assert receiver.ingest(packet, monotonic_s=1.0)
    assert not receiver.ingest(packet, monotonic_s=1.1)
    assert len(receiver.between(0, 2)) == 1


def test_max_consecutive() -> None:
    assert max_consecutive([False, True, True, False, True]) == 2
    assert max_consecutive([]) == 0


def test_extended_telemetry_parser_reads_z_velocity_and_status() -> None:
    raw = struct.pack(
        "<BBfffBbf32sBHHHHBBBffff",
        0x01, 2, 1.0, 2.0, 0.3, 5, -1, 0.5, b"\x00" * 32, 0,
        0xFFFF, 20, 30, 40, 1, 1, 6, -1.2, 0.1, 0.2, -0.3,
    )
    packet = parse_telemetry(raw)
    assert packet is not None
    assert packet.px4_link_ok is True
    assert packet.px4_pos_age_ms == 30
    assert packet.ned_z == pytest.approx(-1.2)
    assert packet.ned_vy == pytest.approx(0.2)


def test_old_telemetry_remains_compatible() -> None:
    raw = struct.pack("<BBfffBbf32sBH", 0x01, 2, 1.0, 2.0, 0.3, 0,
                      -1, 0.5, b"\x00" * 32, 0, 0xFFFF)
    packet = parse_telemetry(raw)
    assert packet is not None
    assert packet.ned_z is None
    assert packet.px4_link_ok is None


def test_out_and_back_trajectory_is_continuous_and_returns_to_origin() -> None:
    points = generate_trajectory(0.3, 1.0, 0.5, 1.5, 0.45, 1.05,
                                 0.4, math.radians(30), 0.05, 0.15)
    assert points[0].along_path_m == pytest.approx(0.0)
    assert points[-1].along_path_m == pytest.approx(0.0, abs=1e-6)
    assert points[0].commanded_speed_mps == pytest.approx(0.0)
    assert points[-1].commanded_speed_mps == pytest.approx(0.0)
    assert max(abs(b.along_path_m - a.along_path_m)
               for a, b in zip(points, points[1:])) < 0.03


def test_measurement_phase_is_constant_speed_and_rotated_by_heading() -> None:
    heading = math.radians(90)
    points = generate_trajectory(0.4, 1.0, 0.5, 1.5, 0.45, 1.05,
                                 0.4, heading, 0.05, 0.15)
    measurement = [point for point in points if point.phase == "outbound_measurement"]
    assert measurement
    assert all(point.commanded_speed_mps == pytest.approx(0.4) for point in measurement)
    assert max(abs(point.x_m) for point in measurement) < 1e-9
    assert all(point.y_m > 0 for point in measurement)


def test_too_short_path_or_bad_measurement_corridor_is_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        generate_trajectory(0.5, 1.0, 0.5, 0.2, 0.05, 0.15, 0.2, 0.0)
    with pytest.raises(ValueError, match="constant-speed"):
        generate_trajectory(0.4, 1.0, 0.5, 1.5, 0.05, 1.0, 0.4, 0.0)


def test_trajectory_respects_firmware_point_limit() -> None:
    with pytest.raises(ValueError, match="firmware limit"):
        generate_trajectory(0.01, 1.0, 0.5, 20.0, 0.01, 19.99,
                            0.4, 0.0, 0.001, 0.01)
