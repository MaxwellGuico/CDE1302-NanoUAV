# CDE1302 Baseline Data-Collection Toolkit

This guide explains how to collect the localization, AprilTag height, and
AprilTag speed baselines. The tools save timestamped CSV files that match the
tables in `CDE1302_Baseline_Lab_Workbook.xlsx`.

## Files

The Python tools live in `/home/maxwell-guico/cde1302/laptop/`:

- `baseline_common.py` — shared telemetry, AprilTag, statistics, and CSV code.
- `px4_localization_baseline.py` — props-off PX4 position and yaw accuracy test.
- `height_detection_baseline.py` — props-off detection-versus-height test.
- `speed_detection_baseline.py` — one confirmed straight out-and-back flight.

The workbook is located at:

```text
/home/maxwell-guico/Documents/ChatGPT/CDE1302 Mini Project/CDE1302_Baseline_Lab_Workbook.xlsx
```

## Before collecting data

1. Use Python 3 in the same environment as the existing laptop tools.
2. Build and flash the updated ESP32 firmware. The added telemetry fields are
   required for PX4 Z position and XYZ velocity results.
3. Put the laptop and ESP32 on the same network.
4. Confirm the correct drone ID and that telemetry is arriving on UDP port
   `5005`.
5. Keep UDP port `5008` free for the AprilTag debug stream. Do not run
   `tag_debug.py`, `tag_stream.py`, or another collector that binds the same
   ports during a baseline run.
6. Measure the physical black-square size of the printed tag. The firmware
   currently uses `TAG_SIZE = 0.12` m in `main/at_detect.c`; change and reflash
   it if the measured size is different.
7. Calibrate the camera intrinsics before treating pose or range error as an
   absolute measurement.

### AprilTag family warning

The baseline plan assumes `tag36h11`, but the checked-in embedded detector is
currently created with `tag16h5_create()` in `main/at_detect.c`. The collectors
only receive detector packets, so they do not select the family themselves.
Use tags from the family compiled into the firmware, or deliberately switch the
firmware and any related host/map configuration to `tag36h11` before testing.

## Output folders

By default, run the tools from the laptop directory:

```bash
cd /home/maxwell-guico/cde1302/laptop
```

Each invocation creates a new folder under `results/baselines/`:

```text
results/baselines/localization-YYYYMMDD-HHMMSS/
results/baselines/height-YYYYMMDD-HHMMSS/
results/baselines/speed-YYYYMMDD-HHMMSS/
```

`--output-dir` changes the parent directory. Relative paths are resolved from
the directory where the command is run. An existing run directory is never
overwritten.

Pressing Ctrl+C stops collection and saves conditions that were already
completed. During a live speed trial, Ctrl+C also triggers the fallback landing
sequence.

## 1. PX4 localization baseline

This tool never arms the vehicle and never sends flight commands.

### Safety and setup

1. Remove every propeller.
2. Mark a 3 by 3 grid on the floor. The default spacing is 0.5 m.
3. Align the grid axes with the PX4 local NED X and Y axes. The target
   coordinates are relative to the captured starting point.
4. Keep the vehicle level and preserve yaw while moving it between grid points.
5. Use a tape measure and an independent angle reference; do not use the PX4
   display to decide where the vehicle should be placed.

### Run it

```bash
python3 px4_localization_baseline.py --drone-id 1
```

The script waits for telemetry, asks you to place the vehicle at the origin,
and averages the starting pose. It then prompts for:

- Five repeats of all nine grid points.
- Five repeats of 0, 90, 180, and 270 degree yaw at the origin.
- Five seconds of telemetry at each condition by default.

Do not move the vehicle during a sampling interval.

### Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--drone-id` | required | Drone ID whose telemetry will be recorded. |
| `--spacing` | `0.5` | Grid spacing in metres. |
| `--repeats` | `5` | Number of repeats of the grid and yaw sequences. |
| `--sample-duration` | `5.0` | Seconds recorded after each prompt. |
| `--target-z` | `0.0` | Independently measured vertical displacement from the captured origin, in metres. |
| `--output-dir` | `results/baselines` | Parent directory for the timestamped run folder. |
| `--telem-port` | `5005` | Laptop UDP telemetry port. |
| `--cmd-port` | `5006` | ESP command port used when constructing the communications node. This script sends no flight commands. |

Example with a 0.4 m grid and three repeats:

```bash
python3 px4_localization_baseline.py \
  --drone-id 1 \
  --spacing 0.4 \
  --repeats 3 \
  --sample-duration 5
```

### Files and interpretation

- `localization_raw.csv` contains every timestamped PX4 sample and its pose
  relative to the captured origin.
- `localization_summary.csv` contains one row per completed grid or yaw
  condition.

Important summary values are axis error, horizontal error, 3D error, wrapped
yaw error, standard deviation, sample count, and estimated dropped telemetry.
Use `valid` and `notes` before including a row in an aggregate result.

## 2. AprilTag height and image-position baseline

This tool only listens for AprilTag debug packets. It never arms the vehicle.

### Safety and setup

1. Remove every propeller.
2. Place a flat tag on the floor under bright, even lighting.
3. Measure from the camera lens to the tag plane, not from the frame, motors,
   or optical-flow sensor.
4. Keep the camera orientation fixed between conditions.
5. For image-position tests, move the tag to the requested footprint region
   while keeping the lens-to-tag-plane height fixed.

### Run the default height series

```bash
python3 height_detection_baseline.py --drone-id 1 --tag-id 3
```

The default requested heights are 0.6, 0.8, 1.0, 1.2, and 1.4 m. There are
five repeats per height and each interval lasts ten seconds. At every prompt,
enter the measured height or press Enter to accept the requested height.

To test a camera-footprint region separately:

```bash
python3 height_detection_baseline.py \
  --drone-id 1 \
  --tag-id 3 \
  --heights 1.0 \
  --image-region left
```

Run separate invocations for `near`, `centre`, `far`, `left`, and `right`.

### Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--drone-id` | required | Drone ID in the AprilTag debug packets. |
| `--tag-id` | `-1` | Expected tag ID. `-1` accepts the strongest pose-bearing detection in each frame. |
| `--heights` | `0.6,0.8,1.0,1.2,1.4` | Comma-separated requested heights in metres. |
| `--repeats` | `5` | Repeats at every requested height. |
| `--observation-duration` | `10.0` | Seconds recorded per condition. |
| `--image-region` | `centre` | Label: `near`, `centre`, `far`, `left`, or `right`. |
| `--exposure-ms` | blank | Optional exposure metadata written to the summary. It does not configure the camera. |
| `--gain` | blank | Optional gain metadata written to the summary. It does not configure the camera. |
| `--output-dir` | `results/baselines` | Parent directory for the timestamped run folder. |
| `--at-port` | `5008` | Laptop UDP port for the ESP AprilTag debug stream. |

Example with custom heights:

```bash
python3 height_detection_baseline.py \
  --drone-id 1 \
  --tag-id 3 \
  --heights 0.5,0.75,1.0,1.25 \
  --repeats 5 \
  --observation-duration 10 \
  --exposure-ms 4.0 \
  --gain 2.0
```

### Files and interpretation

- `height_raw.csv` contains every unique processed frame. A frame with no
  detection is retained, and repeated UDP reports are deduplicated using
  `frame_ms`.
- `height_summary.csv` contains one row per height/repeat condition.

The firmware quality gate currently requires Hamming distance at most 1 and
decision margin above 55 before pose is computed. The summary distinguishes:

- `unique_frames`: unique camera frames received.
- `raw_detection_frames`: frames containing any raw detection.
- `accepted_detection_frames`: frames containing the requested or strongest
  detection with a computed pose.
- `detection_rate`: accepted frames divided by unique frames.
- `acceptance_rate`: accepted frames divided by raw-detection frames.
- `effective_fps`: unique processed frames per second.
- `range_error_m`: mean camera-to-tag range minus measured height.

`range_error_m` is most meaningful with the tag near the optical axis. For an
off-centre tag, straight-line range is naturally larger than vertical height.

## 3. AprilTag speed baseline

This is the only baseline tool that arms and flies the vehicle. It performs one
trial per invocation.

### Flight path

1. Firmware takeoff to the configured takeoff height.
2. Smooth climb to the requested test height.
3. Two-second settle.
4. Jerk-limited forward acceleration.
5. Constant-speed outbound observation corridor.
6. Jerk-limited stop and two-second turnaround hover.
7. Backward return along the same heading at the configured return speed.
8. Stop over the start, hover, and send repeated landing commands.

The outbound direction is calculated from the yaw measured after takeoff. The
drone therefore flies in the direction its nose points. The camera and return
path must both be clear before confirmation.

Only frames labelled `outbound_measurement` are included in the primary speed
summary. Return detections remain in `speed_raw.csv` for inspection.

### Always generate and inspect first

`--generate-only` performs no network setup, arming, or flight command:

```bash
python3 speed_detection_baseline.py \
  --drone-id 1 \
  --speed 0.20 \
  --height 1.0 \
  --generate-only
```

Inspect `speed_trajectory.csv` before flying. In generate-only mode,
`--heading-deg` rotates the preview. During a real flight, the captured takeoff
yaw is always used instead.

### Run one live trial

```bash
python3 speed_detection_baseline.py \
  --drone-id 1 \
  --speed 0.20 \
  --height 1.0 \
  --repeat 1 \
  --tag-id 3
```

The script waits for fresh telemetry and then displays a token such as:

```text
SPEED-1-0.20
```

The token must be typed exactly before any arming command is sent. A mismatch
aborts without arming. Run the command again for each repeat and speed, changing
`--repeat` so the summary row has the correct repeat number.

Recommended series:

```text
0.10 m/s: repeats 1 to 5
0.20 m/s: repeats 1 to 5
0.30 m/s: repeats 1 to 5
0.40 m/s: repeats 1 to 5
```

### Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--drone-id` | required | Drone ID to arm and monitor. |
| `--speed` | required | Outbound test speed in m/s. Must be greater than 0 and no more than 0.50 m/s. |
| `--height` | `1.0` | Test height above the takeoff origin, in metres. |
| `--takeoff-height` | `0.5` | Height reached by the firmware takeoff routine before trajectory playback. |
| `--path-length` | `1.5` | One-way horizontal path length in metres. |
| `--max-path-length` | `3.0` | Local safety ceiling; the requested path is rejected above this value. Set it to the measured safe-space limit. |
| `--tag-distance` | `0.75` | Surveyed distance from the start to the tag centre along the outbound path. Used for tag-coordinate error. |
| `--tag-id` | `-1` | Expected tag ID. `-1` accepts the strongest pose-bearing detection. |
| `--measurement-start` | `0.45` | Start of the evaluated outbound corridor, in metres from the start. |
| `--measurement-end` | `1.05` | End of the evaluated outbound corridor, in metres from the start. |
| `--acceleration` | `0.40` | Maximum profile acceleration in m/s². |
| `--return-speed` | `0.15` | Backward recovery speed in m/s. |
| `--repeat` | `1` | Metadata identifying this repeat. It does not cause automatic repeated flights. |
| `--dt` | `0.05` | Trajectory sample interval in seconds. |
| `--takeoff-wait` | `8.0` | Seconds allowed for firmware takeoff before freshness is checked and the trajectory is uploaded. |
| `--heading-deg` | `0.0` | Preview heading used only with `--generate-only`. |
| `--output-dir` | `results/baselines` | Parent directory for the timestamped run folder. |
| `--telem-port` | `5005` | Laptop UDP telemetry port. |
| `--cmd-port` | `5006` | ESP32 command port. |
| `--at-port` | `5008` | Laptop UDP AprilTag debug port. |
| `--generate-only` | off | Generate and save the path without opening network receivers or sending commands. |

### Constant-speed corridor constraint

The observation corridor must fit entirely inside the constant-speed portion
of the outbound path. The outbound ramp distance at each end is:

```text
ramp_distance = 0.75 * speed^2 / acceleration
```

Therefore:

```text
ramp_distance <= measurement_start < measurement_end
measurement_end <= path_length - ramp_distance
```

The script rejects an unsafe or impossible combination before flight. It also
rejects excessive speed, non-positive parameters, a path above
`--max-path-length`, and trajectories longer than the firmware limit of 2,400
points.

### Live-flight safety checklist

1. Use a charged battery and verify the PX4/ESP link on the bench.
2. Select an open indoor area longer and wider than the full out-and-back path.
3. Mark the start, path endpoint, observation corridor, and tag position.
4. Point the drone's nose exactly down the marked path before starting.
5. Verify that both the forward and backward paths are free of people and
   obstacles.
6. Use a spotter and keep the transmitter's manual abort/kill control ready.
7. Start with 0.10 m/s and use `--generate-only` before every changed geometry.
8. Do not rely on the laptop script as the only safety mechanism. PX4 failsafes
   and the operator remain authoritative.

On completion, or after a recoverable exception once flight has started, the
script sends the landing command three times. If communications are lost, use
the transmitter and normal PX4 failsafe procedure.

### Files and interpretation

- `speed_trajectory.csv` is the exact commanded path with timestamps, XYZ
  offsets, along-path position, speed, and phase.
- `speed_raw.csv` combines timestamped telemetry and AprilTag observations.
  `event_source` distinguishes them and `trajectory_phase` identifies where
  they occurred.
- `speed_summary.csv` contains the outbound measurement result for the one
  trial.

A trial is marked `PASS` when it completes without an abort and contains at
least three accepted AprilTag frames in `outbound_measurement`. This is a
collection-quality threshold, not proof that localization accuracy is good.
Inspect speed error, height, decision margin, pose error, tag error, and the
raw record before accepting a trial.

## Using the Excel workbook

1. Open `CDE1302_Baseline_Lab_Workbook.xlsx`.
2. Read `Overview` and set the shared project fields.
3. Copy rows from each summary CSV into the matching sheet without changing
   column order:
   - `localization_summary.csv` to `PX4 Localization`.
   - `height_summary.csv` to `Height Detection` or `Image Position`.
   - `speed_summary.csv` to `Speed Detection`.
4. Use `Static Drift` for a separate five-minute stationary PX4 recording.
5. Use `Return to Start` for measured landing/closure error after speed trials.
6. Use `Tag Localization` to compare surveyed tag coordinates against the
   estimates in the speed summary.
7. Exclude rows marked invalid or aborted before calculating final conclusions.

The workbook's yellow cells are operator inputs. Green cells contain formulas.
Warnings flag incomplete or invalid trials, and the charts update from the data
tables.

## Troubleshooting

### No telemetry from the drone

- Confirm `--drone-id` matches the firmware configuration.
- Confirm the laptop and ESP32 are on the same network.
- Check that UDP `5005` is allowed and no incompatible process owns it.
- Confirm the ESP32 is sending to the laptop's current IP address.

### No unique AprilTag frames

- Stop any other process using UDP `5008`.
- Confirm the detector firmware is running and the tag family matches.
- Increase lighting, reduce motion, and move the tag closer.
- Check the camera stream and review `proc_ms` in raw data.

### Raw detections exist but accepted detections are zero

- The tag may fail the firmware quality gate: Hamming distance above 1,
  decision margin at or below 55, or no valid pose estimate.
- Verify the correct `--tag-id`; try `--tag-id -1` only during diagnosis.
- Confirm the physical tag size and camera intrinsics.

### Speed trajectory is rejected

- Run with `--generate-only` and read the reported validation error.
- Increase path length, move the measurement corridor inward, reduce speed, or
  increase acceleration within the drone's safely tested limits.
- A smaller `--dt` creates more points and can exceed the 2,400-point limit.

### Z position or velocity is blank

Rebuild and flash the updated ESP32 firmware. The Python parser remains
compatible with older telemetry packets, but older firmware does not transmit
NED Z or velocity.

## Recommended test order

1. Camera intrinsic calibration and physical tag-size measurement.
2. PX4 stationary drift test.
3. Props-off PX4 grid and yaw localization test.
4. Props-off height test with the tag centred.
5. Props-off near/centre/far/left/right footprint tests.
6. Generate-only review of every speed trajectory.
7. Live speed trials from 0.10 m/s upward.
8. Return-to-start measurements and surveyed tag-coordinate comparison.

Do not proceed to faster live trials until localization, height holding, tag
detection, and return-path behaviour are acceptable at the preceding stage.
