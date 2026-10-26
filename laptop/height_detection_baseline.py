#!/usr/bin/env python3
"""Interactive, props-off AprilTag detection-versus-height collector."""

from __future__ import annotations

import argparse
import logging

from baseline_common import (
    HEIGHT_SUMMARY_FIELDS,
    AprilTagReceiver,
    accepted_detection,
    detector_summary,
    make_run_dir,
    new_run_id,
    timestamp_now,
    write_csv,
)


log = logging.getLogger("height_detection_baseline")

RAW_FIELDS = [
    "run_id", "timestamp", "repeat", "requested_height_m", "actual_height_m",
    "image_region", "frame_index", "host_timestamp", "host_monotonic_s",
    "frame_ms", "proc_ms", "raw_count", "latched_id", "detection_index",
    "det_id", "hamming", "margin", "cx", "cy", "tx", "ty", "tz",
    "range_m", "pose_err", "accepted",
]


def parse_heights(text: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("heights must be comma-separated numbers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("heights must contain positive values")
    return values


def raw_frame_rows(run_id: str, repeat: int, requested: float, actual: float,
                   image_region: str, frames, tag_id: int) -> list[dict]:
    rows = []
    for frame_i, frame in enumerate(frames, 1):
        packet = frame.packet
        detections = packet.detections or [None]
        chosen = accepted_detection(frame, tag_id)
        for detection_i, det in enumerate(detections, 1):
            rows.append({
                "run_id": run_id, "timestamp": timestamp_now(), "repeat": repeat,
                "requested_height_m": requested, "actual_height_m": actual,
                "image_region": image_region, "frame_index": frame_i,
                "host_timestamp": frame.wall_time, "host_monotonic_s": frame.monotonic_s,
                "frame_ms": packet.frame_ms, "proc_ms": packet.proc_ms,
                "raw_count": packet.raw_count, "latched_id": packet.latched_id,
                "detection_index": detection_i if det is not None else 0,
                "det_id": det.id if det else None,
                "hamming": det.hamming if det else None,
                "margin": det.margin if det else None,
                "cx": det.cx if det else None, "cy": det.cy if det else None,
                "tx": det.tx if det else None, "ty": det.ty if det else None,
                "tz": det.tz if det else None, "range_m": det.range_m if det else None,
                "pose_err": det.pose_err if det else None,
                "accepted": det is not None and det is chosen,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-id", type=int, required=True)
    parser.add_argument("--tag-id", type=int, default=-1,
                        help="Expected tag ID; -1 accepts the strongest tag")
    parser.add_argument("--heights", type=parse_heights,
                        default=parse_heights("0.6,0.8,1.0,1.2,1.4"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--observation-duration", type=float, default=10.0)
    parser.add_argument("--image-region", choices=["near", "centre", "far", "left", "right"],
                        default="centre")
    parser.add_argument("--exposure-ms", type=float)
    parser.add_argument("--gain", type=float)
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--at-port", type=int, default=5008)
    args = parser.parse_args()
    if args.repeats < 1 or args.observation_duration <= 0:
        parser.error("repeats and observation-duration must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("\nSAFETY: remove every propeller. This tool only receives detector packets and never arms.\n")
    run_id = new_run_id("height")
    run_dir = make_run_dir(args.output_dir, run_id)
    receiver = AprilTagReceiver(args.drone_id, port=args.at_port)
    summaries: list[dict] = []
    raw_rows: list[dict] = []
    receiver.start()
    try:
        for requested in args.heights:
            for repeat in range(1, args.repeats + 1):
                entered = input(
                    f"Set camera lens to {requested:.2f} m, tag at {args.image_region}; "
                    "enter measured height in metres or press Enter to accept: "
                ).strip()
                actual = float(entered) if entered else requested
                if actual <= 0:
                    raise ValueError("actual height must be positive")
                input(f"Ready for repeat {repeat}/{args.repeats}. Press Enter to record "
                      f"{args.observation_duration:.1f} s: ")
                frames = receiver.collect(args.observation_duration)
                stats = detector_summary(frames, args.tag_id, actual)
                valid = bool(frames)
                summaries.append({
                    "run_id": run_id, "timestamp": timestamp_now(), "repeat": repeat,
                    "requested_height_m": requested, "actual_height_m": actual,
                    "tag_id": args.tag_id, "observation_s": args.observation_duration,
                    **stats, "exposure_ms": args.exposure_ms, "gain": args.gain,
                    "image_region": args.image_region, "valid": valid,
                    "notes": "" if valid else "No unique detector frames received",
                })
                raw_rows.extend(raw_frame_rows(run_id, repeat, requested, actual,
                                               args.image_region, frames, args.tag_id))
                rate = stats["detection_rate"]
                print(f"  {len(frames)} unique frames, detection rate "
                      f"{rate:.1%}" if rate is not None else "  no frames received")
    except (KeyboardInterrupt, EOFError):
        log.warning("collection interrupted; saving completed observations")
    finally:
        receiver.stop()
        write_csv(run_dir / "height_summary.csv", HEIGHT_SUMMARY_FIELDS, summaries)
        write_csv(run_dir / "height_raw.csv", RAW_FIELDS, raw_rows)
        print(f"Saved {len(summaries)} completed conditions to {run_dir}")


if __name__ == "__main__":
    main()
