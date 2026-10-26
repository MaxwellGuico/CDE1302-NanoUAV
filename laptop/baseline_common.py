#!/usr/bin/env python3
"""Shared collection, statistics, and CSV helpers for CDE1302 baselines."""

from __future__ import annotations

import csv
import math
import socket
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from protocol import AT_DEBUG_PORT, AtDebugPacket, TelemetryPacket, parse_at_debug


LOCALIZATION_SUMMARY_FIELDS = [
    "run_id", "timestamp", "test_type", "repeat", "point_id",
    "target_x_m", "target_y_m", "target_z_m", "target_yaw_deg",
    "mean_px4_x_m", "mean_px4_y_m", "mean_px4_z_m", "mean_px4_yaw_deg",
    "std_px4_x_m", "std_px4_y_m", "std_px4_z_m",
    "error_x_m", "error_y_m", "error_z_m", "horizontal_error_m",
    "error_3d_m", "yaw_error_deg", "sample_count",
    "telemetry_drop_count", "valid", "notes",
]

HEIGHT_SUMMARY_FIELDS = [
    "run_id", "timestamp", "repeat", "requested_height_m", "actual_height_m",
    "tag_id", "observation_s", "unique_frames", "raw_detection_frames",
    "accepted_detection_frames", "detection_rate", "acceptance_rate",
    "mean_margin", "min_margin", "mean_pose_error", "max_pose_error",
    "mean_range_m", "range_error_m", "mean_processing_ms", "effective_fps",
    "exposure_ms", "gain", "image_region", "valid", "notes",
]

SPEED_SUMMARY_FIELDS = [
    "run_id", "timestamp", "repeat", "target_speed_mps", "mean_speed_mps",
    "max_speed_mps", "target_height_m", "mean_height_m", "tag_id",
    "path_length_m", "evaluation_start_m", "evaluation_end_m",
    "unique_frames", "accepted_detection_frames", "detection_rate",
    "max_consecutive_detections", "mean_margin", "min_margin",
    "mean_pose_error", "first_detection_x_m", "last_detection_x_m",
    "estimated_tag_x_m", "estimated_tag_y_m", "tag_error_m",
    "trial_result", "abort_reason", "notes",
]


def timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().astimezone():%Y%m%d-%H%M%S}"


def make_run_dir(output_root: str | Path, run_id: str) -> Path:
    path = Path(output_root).expanduser().resolve() / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    """Write a stable-schema CSV atomically enough for a single local process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(clean) if clean else None


def stdev_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.stdev(clean) if len(clean) >= 2 else (0.0 if len(clean) == 1 else None)


def rmse(values: Iterable[float]) -> Optional[float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return math.sqrt(statistics.fmean(v * v for v in clean)) if clean else None


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def circular_mean_rad(values: Iterable[float]) -> Optional[float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return None
    return math.atan2(statistics.fmean(math.sin(v) for v in clean),
                      statistics.fmean(math.cos(v) for v in clean))


def circular_stdev_rad(values: Iterable[float]) -> Optional[float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if len(clean) < 2:
        return 0.0 if clean else None
    r = math.hypot(statistics.fmean(math.cos(v) for v in clean),
                   statistics.fmean(math.sin(v) for v in clean))
    return math.sqrt(max(0.0, -2.0 * math.log(max(r, 1e-12))))


@dataclass(frozen=True)
class TimedTelemetry:
    monotonic_s: float
    wall_time: str
    packet: TelemetryPacket


class TelemetryBuffer:
    """Thread-safe buffer suitable for use as a CommsNode callback."""

    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self._samples: list[TimedTelemetry] = []
        self._condition = threading.Condition()

    def callback(self, packet: TelemetryPacket, _ip: str) -> None:
        if packet.drone_id != self.drone_id:
            return
        sample = TimedTelemetry(time.monotonic(), timestamp_now(), packet)
        with self._condition:
            self._samples.append(sample)
            self._condition.notify_all()

    def latest(self) -> Optional[TimedTelemetry]:
        with self._condition:
            return self._samples[-1] if self._samples else None

    def wait_for_first(self, timeout_s: float = 30.0) -> TimedTelemetry:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._samples:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no telemetry from drone {self.drone_id}")
                self._condition.wait(min(remaining, 0.5))
            return self._samples[-1]

    def collect(self, duration_s: float) -> list[TimedTelemetry]:
        start = time.monotonic()
        end = start + duration_s
        with self._condition:
            start_i = len(self._samples)
            while True:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return list(self._samples[start_i:])
                self._condition.wait(min(remaining, 0.25))

    def between(self, start_s: float, end_s: float) -> list[TimedTelemetry]:
        with self._condition:
            return [s for s in self._samples if start_s <= s.monotonic_s <= end_s]


@dataclass(frozen=True)
class TimedAtFrame:
    monotonic_s: float
    wall_time: str
    packet: AtDebugPacket


class AprilTagReceiver:
    """Receive unique ESP detector frames from UDP, deduplicated by frame_ms."""

    def __init__(self, drone_id: int, port: int = AT_DEBUG_PORT,
                 on_frame: Optional[Callable[[TimedAtFrame], None]] = None):
        self.drone_id = drone_id
        self.port = port
        self.on_frame = on_frame
        self._frames: list[TimedAtFrame] = []
        self._last_frame_ms: Optional[int] = None
        self._condition = threading.Condition()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", self.port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="baseline-apriltag",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()
        self._thread = None
        self._sock = None

    def _loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                data, _source = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            packet = parse_at_debug(data)
            if packet is None or packet.drone_id != self.drone_id or packet.frame_ms == 0:
                continue
            self.ingest(packet)

    def ingest(self, packet: AtDebugPacket, monotonic_s: Optional[float] = None) -> bool:
        """Store one packet if its frame is new. Returns True when accepted."""
        with self._condition:
            if packet.frame_ms == self._last_frame_ms:
                return False
            self._last_frame_ms = packet.frame_ms
            frame = TimedAtFrame(monotonic_s or time.monotonic(), timestamp_now(), packet)
            self._frames.append(frame)
            self._condition.notify_all()
        if self.on_frame:
            self.on_frame(frame)
        return True

    def collect(self, duration_s: float) -> list[TimedAtFrame]:
        start = time.monotonic()
        end = start + duration_s
        with self._condition:
            start_i = len(self._frames)
            while True:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return list(self._frames[start_i:])
                self._condition.wait(min(remaining, 0.25))

    def between(self, start_s: float, end_s: float) -> list[TimedAtFrame]:
        with self._condition:
            return [f for f in self._frames if start_s <= f.monotonic_s <= end_s]


def accepted_detection(frame: TimedAtFrame, tag_id: int) -> Optional[object]:
    candidates = [d for d in frame.packet.detections
                  if d.has_pose and (tag_id < 0 or d.id == tag_id)]
    return max(candidates, key=lambda d: d.margin) if candidates else None


def max_consecutive(flags: Sequence[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def detector_summary(frames: Sequence[TimedAtFrame], tag_id: int,
                     actual_height_m: Optional[float] = None) -> dict:
    accepted = [accepted_detection(frame, tag_id) for frame in frames]
    good = [det for det in accepted if det is not None]
    raw_frames = sum(1 for frame in frames if frame.packet.raw_count > 0)
    span = frames[-1].monotonic_s - frames[0].monotonic_s if len(frames) >= 2 else 0.0
    ranges = [det.range_m for det in good]
    mean_range = mean_or_none(ranges)
    return {
        "unique_frames": len(frames),
        "raw_detection_frames": raw_frames,
        "accepted_detection_frames": len(good),
        "detection_rate": len(good) / len(frames) if frames else None,
        "acceptance_rate": len(good) / raw_frames if raw_frames else None,
        "mean_margin": mean_or_none(det.margin for det in good),
        "min_margin": min((det.margin for det in good), default=None),
        "mean_pose_error": mean_or_none(det.pose_err for det in good),
        "max_pose_error": max((det.pose_err for det in good), default=None),
        "mean_range_m": mean_range,
        "range_error_m": (mean_range - actual_height_m)
                         if mean_range is not None and actual_height_m is not None else None,
        "mean_processing_ms": mean_or_none(frame.packet.proc_ms for frame in frames),
        "effective_fps": (len(frames) - 1) / span if span > 0 else None,
        "max_consecutive_detections": max_consecutive([d is not None for d in accepted]),
    }


def telemetry_to_dict(sample: TimedTelemetry) -> dict:
    packet = sample.packet
    data = asdict(packet)
    data.pop("vfh_blocked", None)
    return {"host_timestamp": sample.wall_time, "host_monotonic_s": sample.monotonic_s,
            **data}


def camera_to_body(tx: float, ty: float, tz: float,
                   pitch_deg: float = 45.0, forward_offset_m: float = 0.02) -> tuple[float, float]:
    pitch = math.radians(pitch_deg)
    forward = math.cos(pitch) * tz + math.sin(pitch) * ty + forward_offset_m
    return forward, tx


def nearest_telemetry(samples: Sequence[TimedTelemetry], when_s: float) -> Optional[TimedTelemetry]:
    return min(samples, key=lambda s: abs(s.monotonic_s - when_s)) if samples else None


def estimate_xy_speeds(samples: Sequence[TimedTelemetry]) -> list[float]:
    direct = [math.hypot(s.packet.ned_vx, s.packet.ned_vy) for s in samples
              if s.packet.ned_vx is not None and s.packet.ned_vy is not None]
    if len(direct) == len(samples) and direct:
        return direct
    speeds: list[float] = []
    for left, right in zip(samples, samples[1:]):
        dt = right.monotonic_s - left.monotonic_s
        if dt > 0:
            speeds.append(math.hypot(right.packet.ned_x - left.packet.ned_x,
                                     right.packet.ned_y - left.packet.ned_y) / dt)
    return speeds
