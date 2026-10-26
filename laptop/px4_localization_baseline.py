#!/usr/bin/env python3
"""Interactive, props-off PX4 local-position baseline collector."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from baseline_common import (
    LOCALIZATION_SUMMARY_FIELDS,
    TelemetryBuffer,
    circular_mean_rad,
    circular_stdev_rad,
    make_run_dir,
    mean_or_none,
    new_run_id,
    stdev_or_none,
    telemetry_to_dict,
    timestamp_now,
    wrap_angle_deg,
    write_csv,
)
from comms import CommsNode


log = logging.getLogger("px4_localization_baseline")

RAW_FIELDS = [
    "run_id", "test_type", "repeat", "point_id", "target_x_m", "target_y_m",
    "target_z_m", "target_yaw_deg", "sample_index", "host_timestamp",
    "host_monotonic_s", "drone_id", "ned_x", "ned_y", "ned_z", "ned_vx",
    "ned_vy", "ned_vz", "heading_rad", "relative_x_m", "relative_y_m",
    "relative_z_m", "relative_yaw_deg", "nav_state", "px4_link_ok",
    "px4_pos_age_ms", "px4_att_age_ms",
]


def build_grid(spacing_m: float) -> list[tuple[str, float, float]]:
    return [(f"G{row + 1}{col + 1}", row * spacing_m, col * spacing_m)
            for row in range(3) for col in range(3)]


def summarize(samples, origin, initial_yaw, run_id: str, test_type: str,
              repeat: int, point_id: str, target_x: float, target_y: float,
              target_z: float, target_yaw_deg: float, duration_s: float) -> tuple[dict, list[dict]]:
    rel_x = [s.packet.ned_x - origin[0] for s in samples]
    rel_y = [s.packet.ned_y - origin[1] for s in samples]
    rel_z = [(s.packet.ned_z - origin[2]) if s.packet.ned_z is not None and origin[2] is not None else None
             for s in samples]
    rel_yaw = [math.degrees(((s.packet.heading_rad - initial_yaw + math.pi) % (2 * math.pi)) - math.pi)
               for s in samples]
    mean_x, mean_y, mean_z = mean_or_none(rel_x), mean_or_none(rel_y), mean_or_none(rel_z)
    mean_yaw_rad = circular_mean_rad(math.radians(v) for v in rel_yaw)
    mean_yaw = math.degrees(mean_yaw_rad) if mean_yaw_rad is not None else None
    error_x = mean_x - target_x if mean_x is not None else None
    error_y = mean_y - target_y if mean_y is not None else None
    error_z = mean_z - target_z if mean_z is not None else None
    horizontal = math.hypot(error_x, error_y) if error_x is not None and error_y is not None else None
    error_3d = math.sqrt(error_x**2 + error_y**2 + error_z**2) \
        if error_x is not None and error_y is not None and error_z is not None else None
    yaw_error = wrap_angle_deg(mean_yaw - target_yaw_deg) if mean_yaw is not None else None
    expected = round(duration_s * 10.0)
    valid = bool(samples) and mean_x is not None and mean_y is not None
    summary = {
        "run_id": run_id,
        "timestamp": timestamp_now(),
        "test_type": test_type,
        "repeat": repeat,
        "point_id": point_id,
        "target_x_m": target_x,
        "target_y_m": target_y,
        "target_z_m": target_z,
        "target_yaw_deg": target_yaw_deg,
        "mean_px4_x_m": mean_x,
        "mean_px4_y_m": mean_y,
        "mean_px4_z_m": mean_z,
        "mean_px4_yaw_deg": mean_yaw,
        "std_px4_x_m": stdev_or_none(rel_x),
        "std_px4_y_m": stdev_or_none(rel_y),
        "std_px4_z_m": stdev_or_none(rel_z),
        "error_x_m": error_x,
        "error_y_m": error_y,
        "error_z_m": error_z,
        "horizontal_error_m": horizontal,
        "error_3d_m": error_3d,
        "yaw_error_deg": yaw_error,
        "sample_count": len(samples),
        "telemetry_drop_count": max(0, expected - len(samples)),
        "valid": valid,
        "notes": "" if valid else "No usable PX4 position samples",
    }
    raw_rows = []
    for index, sample in enumerate(samples, 1):
        packet = sample.packet
        raw_rows.append({
            "run_id": run_id, "test_type": test_type, "repeat": repeat,
            "point_id": point_id, "target_x_m": target_x,
            "target_y_m": target_y, "target_z_m": target_z,
            "target_yaw_deg": target_yaw_deg, "sample_index": index,
            **telemetry_to_dict(sample),
            "relative_x_m": packet.ned_x - origin[0],
            "relative_y_m": packet.ned_y - origin[1],
            "relative_z_m": (packet.ned_z - origin[2])
                              if packet.ned_z is not None and origin[2] is not None else None,
            "relative_yaw_deg": math.degrees(((packet.heading_rad - initial_yaw + math.pi)
                                               % (2 * math.pi)) - math.pi),
        })
    return summary, raw_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-id", type=int, required=True)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=5.0)
    parser.add_argument("--target-z", type=float, default=0.0,
                        help="Physical vertical displacement from the captured start (m)")
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--telem-port", type=int, default=5005)
    parser.add_argument("--cmd-port", type=int, default=5006)
    args = parser.parse_args()
    if args.spacing <= 0 or args.repeats < 1 or args.sample_duration <= 0:
        parser.error("spacing, repeats, and sample-duration must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("\nSAFETY: remove every propeller. This tool only receives telemetry and never arms.\n")
    run_id = new_run_id("localization")
    run_dir = make_run_dir(args.output_dir, run_id)
    telemetry = TelemetryBuffer(args.drone_id)
    comms = CommsNode(listen_port=args.telem_port, cmd_port=args.cmd_port)
    comms.on_telemetry(telemetry.callback)
    summaries: list[dict] = []
    raw_rows: list[dict] = []
    comms.start()
    try:
        telemetry.wait_for_first()
        input("Place the vehicle at the physical origin, level and aligned to 0°. Press Enter: ")
        origin_samples = telemetry.collect(args.sample_duration)
        if not origin_samples:
            raise RuntimeError("no telemetry while capturing the origin")
        origin_x = mean_or_none(s.packet.ned_x for s in origin_samples)
        origin_y = mean_or_none(s.packet.ned_y for s in origin_samples)
        origin_z = mean_or_none(s.packet.ned_z for s in origin_samples)
        initial_yaw = circular_mean_rad(s.packet.heading_rad for s in origin_samples)
        assert origin_x is not None and origin_y is not None and initial_yaw is not None
        origin = (origin_x, origin_y, origin_z)
        if origin_z is None:
            print("WARNING: PX4 Z is unavailable. Reflash the updated ESP firmware for altitude results.")

        grid = build_grid(args.spacing)
        for repeat in range(1, args.repeats + 1):
            for point_id, target_x, target_y in grid:
                input(f"Repeat {repeat}/{args.repeats}: move to {point_id} "
                      f"({target_x:.2f}, {target_y:.2f}) m, keep yaw 0°, then press Enter: ")
                samples = telemetry.collect(args.sample_duration)
                summary, raw = summarize(samples, origin, initial_yaw, run_id, "grid",
                                         repeat, point_id, target_x, target_y,
                                         args.target_z, 0.0, args.sample_duration)
                summaries.append(summary)
                raw_rows.extend(raw)

        for repeat in range(1, args.repeats + 1):
            for target_yaw in (0.0, 90.0, 180.0, 270.0):
                point_id = f"Y{int(target_yaw):03d}"
                input(f"Yaw repeat {repeat}/{args.repeats}: return to origin, align to "
                      f"{target_yaw:.0f}° relative to start, then press Enter: ")
                samples = telemetry.collect(args.sample_duration)
                summary, raw = summarize(samples, origin, initial_yaw, run_id, "yaw",
                                         repeat, point_id, 0.0, 0.0,
                                         args.target_z, target_yaw, args.sample_duration)
                summaries.append(summary)
                raw_rows.extend(raw)
    except KeyboardInterrupt:
        log.warning("collection interrupted; saving completed observations")
    finally:
        comms.stop()
        write_csv(run_dir / "localization_summary.csv", LOCALIZATION_SUMMARY_FIELDS, summaries)
        write_csv(run_dir / "localization_raw.csv", RAW_FIELDS, raw_rows)
        print(f"Saved {len(summaries)} completed conditions to {run_dir}")


if __name__ == "__main__":
    main()
