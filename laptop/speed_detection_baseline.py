#!/usr/bin/env python3
"""Fly one confirmed straight out-and-back AprilTag speed-baseline trial."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from baseline_common import (
    SPEED_SUMMARY_FIELDS,
    AprilTagReceiver,
    TelemetryBuffer,
    accepted_detection,
    camera_to_body,
    detector_summary,
    estimate_xy_speeds,
    make_run_dir,
    mean_or_none,
    nearest_telemetry,
    new_run_id,
    timestamp_now,
    write_csv,
)
from comms import CommsNode
from protocol import (
    CMD_LAND,
    CMD_START,
    NAV_TRAJ,
    TRAJ_MAX_PTS,
    CommandPacket,
    build_traj_packets,
    build_traj_start,
)


log = logging.getLogger("speed_detection_baseline")
UPLOAD_GAP_S = 0.025
SPEED_CAP_MPS = 0.50

TRAJECTORY_FIELDS = ["t_s", "x_m", "y_m", "z_m", "along_path_m",
                     "commanded_speed_mps", "phase"]
RAW_FIELDS = [
    "run_id", "timestamp", "repeat", "elapsed_s", "trajectory_phase",
    "event_source", "target_speed_mps", "target_height_m", "along_path_m",
    "ned_x", "ned_y", "ned_z", "ned_vx", "ned_vy", "ned_vz", "heading_rad",
    "nav_state", "px4_link_ok", "frame_ms", "proc_ms", "raw_count", "det_id",
    "hamming", "margin", "cx", "cy", "tx", "ty", "tz", "range_m",
    "pose_err", "accepted",
]


@dataclass(frozen=True)
class TrajectoryPoint:
    t_s: float
    x_m: float
    y_m: float
    z_m: float
    along_path_m: float
    commanded_speed_mps: float
    phase: str


@dataclass(frozen=True)
class Segment:
    name: str
    start_t: float
    duration: float
    s0: float
    z0: float
    position: object
    velocity: object
    altitude: object


def _smooth_position(u: float) -> float:
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _smooth_velocity_integral(u: float) -> float:
    return u**3 - 0.5 * u**4


def _smooth_velocity(u: float) -> float:
    return 3 * u**2 - 2 * u**3


def generate_trajectory(speed_mps: float, height_m: float, takeoff_height_m: float,
                        path_length_m: float, measurement_start_m: float,
                        measurement_end_m: float, acceleration_mps2: float,
                        heading_rad: float, dt_s: float = 0.05,
                        return_speed_mps: float = 0.15,
                        settle_s: float = 2.0) -> list[TrajectoryPoint]:
    """Generate a smooth forward measurement pass and slower backward return."""
    if not 0 < speed_mps <= SPEED_CAP_MPS:
        raise ValueError(f"speed must be in (0, {SPEED_CAP_MPS}] m/s")
    if height_m <= 0 or takeoff_height_m <= 0 or path_length_m <= 0:
        raise ValueError("height and path length must be positive")
    if not 0 < return_speed_mps <= SPEED_CAP_MPS or acceleration_mps2 <= 0 or dt_s <= 0:
        raise ValueError("return speed, acceleration, and dt must be positive")
    if not 0 <= measurement_start_m < measurement_end_m <= path_length_m:
        raise ValueError("measurement corridor must lie inside the path")

    outbound_ramp_t = 1.5 * speed_mps / acceleration_mps2
    outbound_ramp_d = 0.5 * speed_mps * outbound_ramp_t
    plateau_start = outbound_ramp_d
    plateau_end = path_length_m - outbound_ramp_d
    if plateau_start >= plateau_end:
        raise ValueError("path is too short to reach the requested speed")
    if measurement_start_m < plateau_start - 1e-9 or measurement_end_m > plateau_end + 1e-9:
        raise ValueError(
            f"measurement corridor must be within constant-speed interval "
            f"[{plateau_start:.3f}, {plateau_end:.3f}] m"
        )

    return_ramp_t = 1.5 * return_speed_mps / acceleration_mps2
    return_ramp_d = 0.5 * return_speed_mps * return_ramp_t
    if 2 * return_ramp_d >= path_length_m:
        raise ValueError("path is too short for the backward return profile")

    segments: list[Segment] = []
    cursor_t = 0.0
    z_target = -(height_m - takeoff_height_m)

    def add(name, duration, s0, z0, position, velocity, altitude):
        nonlocal cursor_t
        if duration <= 1e-12:
            return
        segments.append(Segment(name, cursor_t, duration, s0, z0,
                                position, velocity, altitude))
        cursor_t += duration

    climb_duration = max(3.0, abs(height_m - takeoff_height_m) / 0.15)
    add("climb", climb_duration, 0.0, 0.0,
        lambda _u: 0.0, lambda _u: 0.0,
        lambda u: z_target * _smooth_position(u))
    add("settle_before", settle_s, 0.0, z_target,
        lambda _u: 0.0, lambda _u: 0.0, lambda _u: z_target)

    add("outbound_acceleration", outbound_ramp_t, 0.0, z_target,
        lambda u, v=speed_mps, T=outbound_ramp_t: v * T * _smooth_velocity_integral(u),
        lambda u, v=speed_mps: v * _smooth_velocity(u), lambda _u: z_target)

    def add_constant(name: str, s0: float, s1: float, velocity: float) -> None:
        distance = s1 - s0
        add(name, abs(distance / velocity), s0, z_target,
            lambda u, a=s0, d=distance: a + d * u,
            lambda _u, v=velocity: v, lambda _u: z_target)

    add_constant("outbound_pre_measurement", plateau_start, measurement_start_m, speed_mps)
    add_constant("outbound_measurement", measurement_start_m, measurement_end_m, speed_mps)
    add_constant("outbound_post_measurement", measurement_end_m, plateau_end, speed_mps)
    add("outbound_deceleration", outbound_ramp_t, plateau_end, z_target,
        lambda u, a=plateau_end, v=speed_mps, T=outbound_ramp_t:
            a + v * T * (u - _smooth_velocity_integral(u)),
        lambda u, v=speed_mps: v * (1.0 - _smooth_velocity(u)), lambda _u: z_target)
    add("turnaround_hold", settle_s, path_length_m, z_target,
        lambda _u, s=path_length_m: s, lambda _u: 0.0, lambda _u: z_target)
    add("return_acceleration", return_ramp_t, path_length_m, z_target,
        lambda u, a=path_length_m, v=return_speed_mps, T=return_ramp_t:
            a - v * T * _smooth_velocity_integral(u),
        lambda u, v=return_speed_mps: -v * _smooth_velocity(u), lambda _u: z_target)
    add_constant("return_cruise", path_length_m - return_ramp_d,
                 return_ramp_d, -return_speed_mps)
    add("return_deceleration", return_ramp_t, return_ramp_d, z_target,
        lambda u, a=return_ramp_d, v=return_speed_mps, T=return_ramp_t:
            a - v * T * (u - _smooth_velocity_integral(u)),
        lambda u, v=return_speed_mps: -v * (1.0 - _smooth_velocity(u)),
        lambda _u: z_target)
    add("landing_hold", settle_s, 0.0, z_target,
        lambda _u: 0.0, lambda _u: 0.0, lambda _u: z_target)

    total = cursor_t
    sample_count = math.ceil(total / dt_s) + 1
    if sample_count > TRAJ_MAX_PTS:
        raise ValueError(f"trajectory needs {sample_count} points; firmware limit is {TRAJ_MAX_PTS}")
    cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)
    points: list[TrajectoryPoint] = []
    segment_index = 0
    for index in range(sample_count):
        t_s = min(index * dt_s, total)
        while (segment_index + 1 < len(segments)
               and t_s >= segments[segment_index + 1].start_t - 1e-12):
            segment_index += 1
        segment = segments[segment_index]
        local = min(max(t_s - segment.start_t, 0.0), segment.duration)
        u = local / segment.duration
        along = float(segment.position(u))
        speed = float(segment.velocity(u))
        z = float(segment.altitude(u))
        points.append(TrajectoryPoint(t_s, along * cos_h, along * sin_h, z,
                                      along, speed, segment.name))
    return points


def phase_at(points: list[TrajectoryPoint], elapsed_s: float) -> TrajectoryPoint:
    if elapsed_s <= 0:
        return points[0]
    index = min(int(elapsed_s / (points[1].t_s - points[0].t_s)), len(points) - 1)
    return points[index]


def save_trajectory(path: Path, points: list[TrajectoryPoint]) -> None:
    write_csv(path, TRAJECTORY_FIELDS, [
        {"t_s": p.t_s, "x_m": p.x_m, "y_m": p.y_m, "z_m": p.z_m,
         "along_path_m": p.along_path_m, "commanded_speed_mps": p.commanded_speed_mps,
         "phase": p.phase} for p in points
    ])


def send_command(comms: CommsNode, drone_id: int, command: int, name: str,
                 repeat: int = 1) -> None:
    for _ in range(repeat):
        if not comms.send_command(drone_id, CommandPacket(command)):
            raise RuntimeError(f"{name} not sent: drone IP unknown")
        time.sleep(0.15)


def upload_and_start(comms: CommsNode, telemetry: TelemetryBuffer, drone_id: int,
                     points: list[TrajectoryPoint], dt_s: float) -> float:
    xyz = [(p.x_m, p.y_m, p.z_m) for p in points]
    trajectory_id = random.randint(1, 255)
    packets = build_traj_packets(trajectory_id, xyz)
    start_packet = build_traj_start(trajectory_id, round(dt_s * 1000))
    for attempt in range(1, 6):
        for packet in packets:
            if not comms.send_raw(drone_id, packet):
                raise RuntimeError("trajectory upload failed: drone IP unknown")
            time.sleep(UPLOAD_GAP_S)
        started_at = time.monotonic()
        comms.send_raw(drone_id, start_packet)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            latest = telemetry.latest()
            if latest and latest.packet.nav_state == NAV_TRAJ:
                return started_at
            time.sleep(0.05)
        log.warning("trajectory start attempt %d did not enter NAV_TRAJ", attempt)
    raise RuntimeError("drone refused the uploaded trajectory")


def build_raw_rows(run_id: str, repeat: int, speed: float, height: float,
                   points: list[TrajectoryPoint], start_s: float,
                   telemetry_samples, frames, origin_x: float, origin_y: float,
                   heading_rad: float, tag_id: int) -> list[dict]:
    rows = []
    cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)
    for sample in telemetry_samples:
        elapsed = sample.monotonic_s - start_s
        point = phase_at(points, elapsed)
        packet = sample.packet
        along = (packet.ned_x - origin_x) * cos_h + (packet.ned_y - origin_y) * sin_h
        rows.append({
            "run_id": run_id, "timestamp": sample.wall_time, "repeat": repeat,
            "elapsed_s": elapsed, "trajectory_phase": point.phase,
            "event_source": "telemetry", "target_speed_mps": speed,
            "target_height_m": height, "along_path_m": along,
            "ned_x": packet.ned_x, "ned_y": packet.ned_y, "ned_z": packet.ned_z,
            "ned_vx": packet.ned_vx, "ned_vy": packet.ned_vy, "ned_vz": packet.ned_vz,
            "heading_rad": packet.heading_rad, "nav_state": packet.nav_state,
            "px4_link_ok": packet.px4_link_ok,
        })
    for frame in frames:
        elapsed = frame.monotonic_s - start_s
        point = phase_at(points, elapsed)
        nearest = nearest_telemetry(telemetry_samples, frame.monotonic_s)
        packet = nearest.packet if nearest else None
        detections = frame.packet.detections or [None]
        chosen = accepted_detection(frame, tag_id)
        for det in detections:
            rows.append({
                "run_id": run_id, "timestamp": frame.wall_time, "repeat": repeat,
                "elapsed_s": elapsed, "trajectory_phase": point.phase,
                "event_source": "apriltag", "target_speed_mps": speed,
                "target_height_m": height, "along_path_m": point.along_path_m,
                "ned_x": packet.ned_x if packet else None,
                "ned_y": packet.ned_y if packet else None,
                "ned_z": packet.ned_z if packet else None,
                "ned_vx": packet.ned_vx if packet else None,
                "ned_vy": packet.ned_vy if packet else None,
                "ned_vz": packet.ned_vz if packet else None,
                "heading_rad": packet.heading_rad if packet else None,
                "nav_state": packet.nav_state if packet else None,
                "px4_link_ok": packet.px4_link_ok if packet else None,
                "frame_ms": frame.packet.frame_ms, "proc_ms": frame.packet.proc_ms,
                "raw_count": frame.packet.raw_count,
                "det_id": det.id if det else None, "hamming": det.hamming if det else None,
                "margin": det.margin if det else None, "cx": det.cx if det else None,
                "cy": det.cy if det else None, "tx": det.tx if det else None,
                "ty": det.ty if det else None, "tz": det.tz if det else None,
                "range_m": det.range_m if det else None,
                "pose_err": det.pose_err if det else None,
                "accepted": det is not None and det is chosen,
            })
    return sorted(rows, key=lambda row: (row["elapsed_s"], row["event_source"]))


def summarize_trial(run_id: str, args, points, start_s, telemetry_samples, frames,
                    origin_x: float, origin_y: float, heading_rad: float,
                    abort_reason: str) -> dict:
    measurement_frames = [f for f in frames
                          if phase_at(points, f.monotonic_s - start_s).phase == "outbound_measurement"]
    measurement_telem = [s for s in telemetry_samples
                         if phase_at(points, s.monotonic_s - start_s).phase == "outbound_measurement"]
    stats = detector_summary(measurement_frames, args.tag_id)
    accepted = [(frame, accepted_detection(frame, args.tag_id)) for frame in measurement_frames]
    accepted = [(frame, det) for frame, det in accepted if det is not None]
    speeds = estimate_xy_speeds(measurement_telem)
    heights = [-s.packet.ned_z for s in measurement_telem if s.packet.ned_z is not None]
    cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)
    detection_along = []
    estimates = []
    for frame, det in accepted:
        nearest = nearest_telemetry(telemetry_samples, frame.monotonic_s)
        if nearest is None:
            continue
        packet = nearest.packet
        along = (packet.ned_x - origin_x) * cos_h + (packet.ned_y - origin_y) * sin_h
        detection_along.append(along)
        forward, right = camera_to_body(det.tx, det.ty, det.tz)
        estimates.append((packet.ned_x + math.cos(packet.heading_rad) * forward
                          - math.sin(packet.heading_rad) * right,
                          packet.ned_y + math.sin(packet.heading_rad) * forward
                          + math.cos(packet.heading_rad) * right))
    est_x = mean_or_none(x for x, _ in estimates)
    est_y = mean_or_none(y for _, y in estimates)
    true_x = origin_x + args.tag_distance * cos_h
    true_y = origin_y + args.tag_distance * sin_h
    tag_error = math.hypot(est_x - true_x, est_y - true_y) \
        if est_x is not None and est_y is not None else None
    passed = not abort_reason and stats["accepted_detection_frames"] >= 3
    return {
        "run_id": run_id, "timestamp": timestamp_now(), "repeat": args.repeat,
        "target_speed_mps": args.speed, "mean_speed_mps": mean_or_none(speeds),
        "max_speed_mps": max(speeds, default=None), "target_height_m": args.height,
        "mean_height_m": mean_or_none(heights), "tag_id": args.tag_id,
        "path_length_m": args.path_length,
        "evaluation_start_m": args.measurement_start,
        "evaluation_end_m": args.measurement_end,
        "unique_frames": stats["unique_frames"],
        "accepted_detection_frames": stats["accepted_detection_frames"],
        "detection_rate": stats["detection_rate"],
        "max_consecutive_detections": stats["max_consecutive_detections"],
        "mean_margin": stats["mean_margin"], "min_margin": stats["min_margin"],
        "mean_pose_error": stats["mean_pose_error"],
        "first_detection_x_m": min(detection_along, default=None),
        "last_detection_x_m": max(detection_along, default=None),
        "estimated_tag_x_m": est_x, "estimated_tag_y_m": est_y,
        "tag_error_m": tag_error, "trial_result": "PASS" if passed else "FAIL",
        "abort_reason": abort_reason, "notes": "Outbound measurement only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-id", type=int, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--takeoff-height", type=float, default=0.5)
    parser.add_argument("--path-length", type=float, default=1.5)
    parser.add_argument("--max-path-length", type=float, default=3.0)
    parser.add_argument("--tag-distance", type=float, default=0.75)
    parser.add_argument("--tag-id", type=int, default=-1)
    parser.add_argument("--measurement-start", type=float, default=0.45)
    parser.add_argument("--measurement-end", type=float, default=1.05)
    parser.add_argument("--acceleration", type=float, default=0.40)
    parser.add_argument("--return-speed", type=float, default=0.15)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--takeoff-wait", type=float, default=8.0)
    parser.add_argument("--heading-deg", type=float, default=0.0,
                        help="Heading used only with --generate-only")
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--telem-port", type=int, default=5005)
    parser.add_argument("--cmd-port", type=int, default=5006)
    parser.add_argument("--at-port", type=int, default=5008)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.path_length > args.max_path_length:
        parser.error("path-length exceeds max-path-length")
    if not 0 < args.tag_distance < args.path_length:
        parser.error("tag-distance must lie inside the path")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_id = new_run_id("speed")
    run_dir = make_run_dir(args.output_dir, run_id)
    heading = math.radians(args.heading_deg)
    points = generate_trajectory(args.speed, args.height, args.takeoff_height,
                                 args.path_length, args.measurement_start,
                                 args.measurement_end, args.acceleration, heading,
                                 args.dt, args.return_speed)
    trajectory_path = run_dir / "speed_trajectory.csv"
    save_trajectory(trajectory_path, points)
    if args.generate_only:
        print(f"Generated {len(points)} points ({points[-1].t_s:.1f} s): {trajectory_path}")
        return

    telemetry = TelemetryBuffer(args.drone_id)
    comms = CommsNode(listen_port=args.telem_port, cmd_port=args.cmd_port)
    comms.on_telemetry(telemetry.callback)
    receiver = AprilTagReceiver(args.drone_id, port=args.at_port)
    start_s = end_s = None
    origin_x = origin_y = 0.0
    abort_reason = ""
    flying = False
    comms.start()
    receiver.start()
    try:
        first = telemetry.wait_for_first()
        if first.packet.px4_link_ok is False:
            raise RuntimeError("ESP reports stale PX4 link")
        token = f"SPEED-{args.drone_id}-{args.speed:.2f}"
        answer = input(f"Clear the flight area. Type {token} to arm and fly: ").strip()
        if answer != token:
            raise RuntimeError("confirmation did not match")
        send_command(comms, args.drone_id, CMD_START, "CMD_START")
        flying = True
        time.sleep(args.takeoff_wait)
        latest = telemetry.latest()
        if latest is None or time.monotonic() - latest.monotonic_s > 0.5:
            raise RuntimeError("telemetry is stale after takeoff")
        heading = latest.packet.heading_rad
        origin_x, origin_y = latest.packet.ned_x, latest.packet.ned_y
        points = generate_trajectory(args.speed, args.height, args.takeoff_height,
                                     args.path_length, args.measurement_start,
                                     args.measurement_end, args.acceleration, heading,
                                     args.dt, args.return_speed)
        save_trajectory(trajectory_path, points)
        start_s = upload_and_start(comms, telemetry, args.drone_id, points, args.dt)
        deadline = start_s + points[-1].t_s + 10.0
        while time.monotonic() < deadline:
            latest = telemetry.latest()
            if latest and latest.monotonic_s > start_s + 0.5 and latest.packet.nav_state != NAV_TRAJ:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("trajectory did not finish before timeout")
        end_s = time.monotonic()
        send_command(comms, args.drone_id, CMD_LAND, "CMD_LAND", repeat=3)
        flying = False
    except (Exception, KeyboardInterrupt) as exc:
        abort_reason = str(exc) or exc.__class__.__name__
        log.error("trial aborted: %s", abort_reason)
        if flying:
            try:
                send_command(comms, args.drone_id, CMD_LAND, "fallback CMD_LAND", repeat=3)
            except Exception:
                log.exception("fallback landing command failed")
    finally:
        if end_s is None:
            end_s = time.monotonic()
        time.sleep(0.5)
        receiver.stop()
        comms.stop()
        summaries = []
        raw_rows = []
        if start_s is not None:
            telemetry_samples = telemetry.between(start_s, end_s)
            frames = receiver.between(start_s, end_s)
            raw_rows = build_raw_rows(run_id, args.repeat, args.speed, args.height,
                                      points, start_s, telemetry_samples, frames,
                                      origin_x, origin_y, heading, args.tag_id)
            summaries.append(summarize_trial(run_id, args, points, start_s,
                                             telemetry_samples, frames, origin_x,
                                             origin_y, heading, abort_reason))
        write_csv(run_dir / "speed_raw.csv", RAW_FIELDS, raw_rows)
        write_csv(run_dir / "speed_summary.csv", SPEED_SUMMARY_FIELDS, summaries)
        print(f"Saved trial files to {run_dir}")
    if abort_reason:
        raise SystemExit(abort_reason)


if __name__ == "__main__":
    main()
