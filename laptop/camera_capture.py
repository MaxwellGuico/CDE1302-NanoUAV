#!/usr/bin/env python3
"""Preview the ESP32 camera and save a raw photograph every four seconds.

    python3 laptop/camera_capture.py --esp-ip 192.168.1.222

Photos default to calibration_images/ in the project root. --save-interval and
--save-dir override the interval and folder. q/Esc quits; s saves an extra photo.
"""

from camera_stream import main


if __name__ == "__main__":
    raise SystemExit(main(default_save_interval=4.0))
