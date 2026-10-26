#!/usr/bin/env python3
"""Live view of a drone's ESP32 camera over WiFi.

    python3 laptop/camera_stream.py --esp-ip 192.168.1.222 [--fps 10] [--quality 60]

The drone streams only while this viewer sends keepalives (wire format:
main/camera_stream.c).  Keys: q/Esc quit, s save the current frame.
"""

from __future__ import annotations

import argparse
import math
import select
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from protocol import (
    CAMERA_STREAM_MAX_FPS,
    CAMERA_STREAM_MAX_QUALITY,
    CAMERA_STREAM_MIN_QUALITY,
    CAMERA_STREAM_PORT,
    CAMERA_VERSION,
    CameraChunk,
    build_camera_stream_command,
    camera_packet_version,
    parse_camera_chunk,
)

COMMAND_PORT = 5006
KEEPALIVE_S = 1.0
FRAME_TIMEOUT_S = 0.75      # drop a frame still missing datagrams after this
MAX_JPEG_BYTES = 256 * 1024
MAX_PENDING = 16
REBOOT_BACKSTEP = 32        # an id this far below the last shown one = reboot
STATS_S = 2.0
DEFAULT_SAVE_DIR = Path(__file__).resolve().parents[1] / "calibration_images"


class FrameSaver:
    """Save original-size PNGs, scheduling automatic saves from fresh frames only."""

    def __init__(self, directory: Path, interval: float = 0.0):
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("save interval must be finite and nonnegative")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.next_save: float | None = None
        self.count = 0

    def save(self, image) -> Path:
        import cv2

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError("Could not encode camera image as PNG")
        name = datetime.now().strftime("esp32_camera_%Y%m%d_%H%M%S_%f.png")
        path = self.directory / name
        # Never overwrite an earlier photograph.
        with path.open("xb") as output:
            output.write(encoded.tobytes())
        self.count += 1
        return path

    def save_due(self, image, now: float) -> Path | None:
        if self.interval == 0:
            return None
        if self.next_save is None:
            self.next_save = now + self.interval
        if now < self.next_save:
            return None
        path = self.save(image)
        # After an interruption, save one fresh frame without a catch-up burst.
        self.next_save = now + self.interval
        return path


@dataclass
class Frame:
    frame_id: int
    jpeg: bytes
    age_ms: int             # capture -> END sent, measured on the ESP32
    esp_drops: int
    first_rx: float         # when this frame's first datagram arrived


@dataclass
class _Pending:
    first_rx: float
    pieces: dict[int, bytes] = field(default_factory=dict)
    size: int | None = None


class FrameAssembler:
    """Reassembles one drone's JPEG frames from out-of-order datagrams.

    Returns only frames newer than the last one returned.  `dropped` counts
    frame ids skipped between returned frames (lost, incomplete or never
    sent).  A new boot_nonce, or a large step back in frame_id, means the
    drone rebooted and restarts the sequence.
    """

    def __init__(self) -> None:
        self.pending: dict[int, _Pending] = {}
        self.nonce: int | None = None
        self.last_id: int | None = None
        self.dropped = 0

    def add(self, c: CameraChunk, now: float) -> Frame | None:
        rebooted = (self.last_id is not None
                    and c.frame_id + REBOOT_BACKSTEP < self.last_id)
        if c.boot_nonce != self.nonce or rebooted:
            self.nonce, self.last_id = c.boot_nonce, None
            self.pending.clear()
        if self.last_id is not None and c.frame_id <= self.last_id:
            return None                             # late or duplicate

        p = self.pending.get(c.frame_id)
        if p is None:
            if len(self.pending) >= MAX_PENDING:
                del self.pending[min(self.pending)]
            p = self.pending[c.frame_id] = _Pending(now)
        if c.is_end:
            p.size = c.frame_size
        elif c.offset + len(c.payload) <= MAX_JPEG_BYTES:
            p.pieces[c.offset] = c.payload
        if p.size is None:
            return None
        parts, end = [], 0
        for offset in sorted(p.pieces):
            if offset != end:
                return None                         # a datagram is still missing
            parts.append(p.pieces[offset])
            end += len(p.pieces[offset])
        if end != p.size:
            return None
        if self.last_id is not None:
            self.dropped += c.frame_id - self.last_id - 1
        self.last_id = c.frame_id
        self.pending = {k: v for k, v in self.pending.items() if k > c.frame_id}
        return Frame(c.frame_id, b"".join(parts), c.age_ms, c.esp_drops, p.first_rx)

    def expire(self, now: float) -> None:
        for k in [k for k, p in self.pending.items()
                  if now - p.first_rx > FRAME_TIMEOUT_S]:
            del self.pending[k]


def drain_socket(sock: socket.socket, asm: FrameAssembler, esp_ip: str,
                 max_datagrams: int = 2000):
    """Read every queued datagram from esp_ip.

    Returns (newest frame or None, datagrams read, foreign ECAM version or
    None).  Frames completed earlier in the same pass are counted as dropped.
    """
    newest, count, bad_version = None, 0, None
    while count < max_datagrams:
        try:
            data, (ip, _) = sock.recvfrom(2048)
        except (BlockingIOError, InterruptedError):
            break
        count += 1
        if ip != esp_ip:
            continue
        chunk = parse_camera_chunk(data)
        if chunk is None:
            version = camera_packet_version(data)
            if version not in (None, CAMERA_VERSION):
                bad_version = version
            continue
        frame = asm.add(chunk, time.monotonic())
        if frame is not None:
            if newest is not None:
                asm.dropped += 1
            newest = frame
    return newest, count, bad_version


def _parse_args(argv=None, *, default_save_interval=0.0) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--esp-ip", required=True,
                        help="drone IP, 192.168.1.(200 + drone id)")
    parser.add_argument("--fps", type=int, default=0,
                        help=f"1-{CAMERA_STREAM_MAX_FPS} (default: firmware, 10)")
    parser.add_argument("--quality", type=int, default=0,
                        help=f"JPEG quality {CAMERA_STREAM_MIN_QUALITY}-"
                             f"{CAMERA_STREAM_MAX_QUALITY} (default: firmware, 60)")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="window scale (default: 2)")
    parser.add_argument("--save-interval", type=float, default=default_save_interval,
                        help="seconds between automatic photos; 0 disables "
                             "(default: %(default)s)")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR,
                        help="photo folder (default: project/calibration_images)")
    args = parser.parse_args(argv)
    try:
        args.esp_ip = socket.gethostbyname(args.esp_ip)  # datagrams carry numeric IPs
    except OSError:
        parser.error(f"cannot resolve --esp-ip {args.esp_ip}")
    try:
        build_camera_stream_command(True, args.fps, args.quality)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive")
    if not math.isfinite(args.save_interval) or args.save_interval < 0:
        parser.error("--save-interval must be finite and nonnegative")
    args.save_dir = args.save_dir.expanduser().resolve()
    return args


def main(argv=None, *, default_save_interval=0.0) -> int:
    args = _parse_args(argv, default_save_interval=default_save_interval)
    try:
        import cv2
        import numpy as np
    except ImportError:
        sys.exit("Missing dependencies: python3 -m pip install -r laptop/requirements.txt")

    try:
        saver = FrameSaver(args.save_dir, args.save_interval)
    except OSError as exc:
        print(f"Cannot create photo folder: {exc}", file=sys.stderr)
        return 1

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        rx.bind(("0.0.0.0", CAMERA_STREAM_PORT))
    except OSError as exc:
        rx.close()
        sys.exit(f"Cannot listen on UDP {CAMERA_STREAM_PORT}: {exc}")
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.esp_ip, COMMAND_PORT)
    keepalive = build_camera_stream_command(True, args.fps, args.quality)

    asm = FrameAssembler()
    window = "ESP32 camera"
    size = (round(320 * args.scale), round(240 * args.scale))
    print(f"Requesting {args.esp_ip}; q/Esc quits, s saves a frame.")
    print(f"Photos: {args.save_dir}")
    if args.save_interval:
        print(f"Automatic capture every {args.save_interval:g} seconds, "
              "starting after the first live frame.")

    next_keepalive = stats_start = time.monotonic()
    last_warning = float("-inf")
    warned_version = False
    shown = ages = latencies = 0
    image = None
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        waiting = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.putText(waiting, "Waiting for ESP32 camera...", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(window, waiting)
        while True:
            now = time.monotonic()
            if now >= next_keepalive:
                next_keepalive = now + KEEPALIVE_S
                try:
                    tx.sendto(keepalive, target)
                except OSError as exc:        # e.g. drone off: keep trying
                    if now - last_warning > 5:
                        print(f"keepalive failed: {exc}", file=sys.stderr)
                        last_warning = now

            select.select([rx], [], [], 0.005)
            frame, _, bad_version = drain_socket(rx, asm, args.esp_ip)
            if bad_version is not None and not warned_version:
                warned_version = True
                print(f"drone sends ECAM v{bad_version}, viewer expects "
                      f"v{CAMERA_VERSION}: reflash the drone", file=sys.stderr)
            if frame is not None:
                gray = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    asm.dropped += 1
                else:
                    image = gray
                    saved_path = saver.save_due(image, time.monotonic())
                    if saved_path is not None:
                        print(f"saved photo {saver.count}: {saved_path}")
                    view = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                                      size, interpolation=cv2.INTER_NEAREST)
                    cv2.putText(view, f"#{frame.frame_id}  esp {frame.age_ms} ms  "
                                f"drops {asm.dropped}/{frame.esp_drops}",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1, cv2.LINE_AA)
                    if saver.next_save is not None:
                        remaining = max(0.0, saver.next_save - time.monotonic())
                        cv2.putText(view, f"Photos {saver.count} | next {remaining:.1f}s",
                                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow(window, view)
                    shown += 1
                    ages += frame.age_ms
                    latencies += (time.monotonic() - frame.first_rx) * 1000

            now = time.monotonic()
            asm.expire(now)
            if now - stats_start >= STATS_S:
                n = max(shown, 1)
                print(f"{shown / (now - stats_start):4.1f} fps  "
                      f"esp age {ages / n:.0f} ms  laptop {latencies / n:.0f} ms  "
                      f"drops {asm.dropped}")
                stats_start, shown, ages, latencies = now, 0, 0, 0

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s") and image is not None:
                print(f"saved {saver.save(image)}")
    except KeyboardInterrupt:
        pass
    except (OSError, cv2.error) as exc:
        print(f"Camera viewer failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            tx.sendto(build_camera_stream_command(False), target)
        except OSError:
            pass
        rx.close()
        tx.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
