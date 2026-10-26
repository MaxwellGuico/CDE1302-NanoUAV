# Ground AprilTag mapping

The Python mapper records received camera and MAVLink data, calculates floor
intersections, and optimizes an offline graph of drone poses and planar tags.
Every tag centre and corner has exactly world Down = 0. Drone altitude, roll,
pitch, and yaw remain variable. The nonlinear algorithm is explicit
Gauss–Newton, with undamped LSMR, Huber block weighting, and backtracking.
Mapping results are saved locally. No map or pose correction is sent to PX4.

## Install and record

Run from the repository root with Python 3.10 or later:

```bash
python3 -m pip install -r laptop/requirements.txt
cp laptop/mapper_config.example.yaml laptop/mapper.local.yaml
cp laptop/anchors.example.yaml laptop/anchors.local.yaml
python3 laptop/tag_mapper.py --config laptop/mapper.local.yaml \
  --mode live --operation record --esp-ip 192.168.1.222 \
  --run mapping_runs/session01 --duration 60
```

The ESP32 must forward raw MAVLink to the configured `udpin` address (default
UDP 14550), and stream camera data on UDP 5009. The recorder owns those receive
ports; close other local viewers/listeners or route separate copies to them.
It sends camera-preview enable/disable keepalives to UDP 5006. MAVLink reception
is passive: arrange `LOCAL_POSITION_NED` and `ATTITUDE_QUATERNION` publication
at the configured requested rates in PX4 or your existing bridge setup. Those
rates are expectations, not message-rate commands sent by the recorder.

Without `--duration`, Ctrl+C stops collection and finishes the CSV exports.
Each `--run` must be a new directory. Recording works before survey, mount, and
timing commissioning are complete; incomplete inputs remain visibly unknown.
The host AprilTag detector needs a C compiler on its first use, as described
by `apriltag_host.py`. Camera JPEGs are the original received JPEGs; they are
not uncompressed sensor RAW images. Original UDP payloads are also saved.

## What is saved

Each run contains:

- `manifest.json`: run UUID, effective configuration and hash, software hash,
  Python/dependency versions, and timing status.
- `events.jsonl`: append-only, versioned events with original timestamps,
  receive times, message/frame identities, full-precision detections and quality.
- `raw/mavlink.bin` and `raw/camera.bin`: exact concatenated received UDP
  payloads. Datagram events in `events.jsonl` supply byte offsets, lengths,
  source addresses and receive timestamps, including malformed payloads.
- `images/`: each completed received camera JPEG, including no-tag frames.
- `start_event.json`: supplied ground reference, or an explicit missing-reference
  placeholder. It is never inferred from the first visible tag.
- `hashes.json`: SHA-256 hashes of the saved source files and CSVs.
- `csv/`: editable analysis copies described below.

The recorder processes each frame returned by the assembler, rather than
keeping only the newest completed frame in a receive pass. UDP loss,
incomplete/late frames, detector truncation, and boot changes remain possible;
frame IDs, drop counters, raw payloads, and diagnostics expose these limits.

| CSV | Readings |
|---|---|
| `position.csv` | Native x/y/z, vx/vy/vz, FC and host timestamps, sysid/compid/sequence |
| `attitude.csv` | Original w/x/y/z quaternion, angular rates, timestamps and message identity |
| `odometry.csv` | Optional original pose, frame IDs, packed covariance, quality/reset counters |
| `frames.csv` | Run/segment/camera/boot/frame IDs, image path, timing, geometry, calibration, detection count |
| `detections.csv` | Family, tag ID, `c0_u,c0_v` through `c3_u,c3_v`, Hamming distance and margin |
| `clock_samples.csv` | Measured camera/FC clock pairs and timing uncertainty |

Floats use round-trip precision, without display rounding. Missing CSV values
are empty, with status/reason fields identifying unavailable data. Corner
order is the detector's original `p[0..3]`, never sorted by image position.
Native AprilTag pixels use pixel-edge coordinates; projection/intersection
converts the half-pixel convention once. The original readings remain intact.
Times are integer nanoseconds: import these CSV columns as text or 64-bit
integers in your analysis program so it does not round long timestamps.

Work on copies when manipulating CSVs. Replay consumes `events.jsonl`, not
edited CSVs. To export fresh CSVs from a recording without fitting anything:

```bash
python3 laptop/tag_mapper.py --config laptop/mapper.local.yaml \
  --mode replay --operation record --run mapping_runs/session01 \
  --output mapping_results/csv_copy01
```

## Enter known tag positions in YAML

A known tag is identified by **the exact family and decoded ID**. A
`tag16h5` tag with ID 0 matches the ID 0 anchor, regardless of its image
location. The mapper does not guess which tag is the reference from proximity.

Edit `anchors.local.yaml`, using the supplied disabled template:

```yaml
known_tags:
  - enabled: false  # Enable only after replacing the nulls with measurements.
    family_or_dictionary: tag16h5
    tag_id: 0       # Change to your surveyed tag's printed/decoded ID.
    frame: ground_start_NED
    prior_type: position_xy
    position_m: [null, null, 0.0]  # North, East, Down; surveyed tag CENTRE.
    position_sigma_xy_m: [null, null]
    surveyed_yaw_rad: null
    surveyed_yaw_sigma_rad: null
    measurement_notes: "Survey date, method, and datum"
```

For each measured tag, enter North/East in metres and positive measurement
standard deviations, then set `enabled: true`. Yaw stays null unless measured;
a position-only anchor adds no yaw constraint. Add more list entries for more
anchors. Ground Z must be zero within the configured survey tolerance.
IDs must also be in `detector.allowed_tag_ids` (the compiled family supports
IDs 0–21). Duplicate IDs within a family, unknown YAML keys, and duplicate YAML
keys are rejected. Off-plane/disabled/unseen tags are not operational anchors.

Pass `--anchors laptop/anchors.local.yaml`. Alternatively fill `known_tags` in
the main config. Enabled inline anchors and an external anchor file conflict;
the program rejects the combination rather than applying hidden precedence.

## Establish the floor origin and measurements

The world axes are North, East, Down. Physical body axes are Forward, Right,
Down; optical camera axes are right, down, forward. The fixed mounting matrix
maps optical vectors into the physical body. Do not apply its descriptive
45-degree tilt a second time.

The example's intrinsics and five distortion coefficients match the supplied
320×240 calibration. Mount rotation/lever arm and measurement-verification
flags are provisional. Verify NED heading alignment, all mounting axes,
calibration/image geometry, corner correspondence, actual black-square side
length, and measurement uncertainties. Tag size defaults to 0.12 m and supports
per-family/ID overrides. It sets the rigid square geometry, not ray depth.

Supply a measured start event with these fields (nulls below must be filled):

```json
{
  "run_id": null,
  "segment_id": 0,
  "fc_time_ns": null,
  "native_position_m": [null, null, null],
  "quaternion_xyzw": [null, null, null, null],
  "initial_body_height_above_floor_m": null,
  "initial_body_height_sigma_m": null,
  "coordinate_convention": "NED",
  "datum_source": "Describe the measured marked floor reference"
}
```

By default the marked floor point is directly below the start body origin.
Its native position is `native_start_position + [0,0,measured_height]`. The
initial world drone Z is therefore negative height. Alternatively provide
`native_ground_origin_position_m` and set the corresponding start configuration
for a surveyed floor origin. No initial-heading subtraction is applied.
For already-rebased logs, explicitly set
`frames.start.input_positions_already_ground_start_relative: true`.

Pass `--start-event measured_start.json` when recording, or supply it as an
external replay override afterward. Recording assigns its own run UUID;
an external replay override must contain that recording's UUID. This lets you
complete measured reference metadata without editing the original run. An
approximate arrival-time replay also needs `host_receive_ns` at the start.

## Exposure timestamps and synchronization

Current ECAM v2 sends frame age but no absolute exposure timestamp. Production
mapping therefore rejects these frames until verified metadata is supplied.
Network arrival and detector completion are not exposure times.

`--frame-metadata exposure_metadata.jsonl` accepts one object per frame with:
`run_id`, `camera_id`, `boot_nonce`, `frame_id`, `segment_id`,
`capture_time_ns`, `clock_domain`, `capture_timestamp_reference`, and
`exposure_verified: true`. Identities must match the source run exactly. The
timestamp reference must match the configured exposure midpoint convention.

For verified FC-domain timestamps (`fc_boot`), use integer nanoseconds directly.
For camera-domain timestamps, supply measured `timing.scale`, integer
`timing.offset_ns`, integer `timing.capture_timestamp_offset_ns` and positive
`timing.sync_sigma_ns`, or measured clock pairs. Paired samples use
`camera_time_ns`, `fc_time_ns`, and positive `sigma_ns`; at least three pairs
with a time span are required to fit drift. `--clock-samples pairs.jsonl` stores
them during recording. Optional `boot_nonce` scopes samples to a camera boot;
after a reboot a fresh boot-specific mapping is required. Epoch values are
centred before fitting to preserve precision.

Translation and full attitude are independently bracketed, using linear
interpolation and shortest-arc SLERP. Default maximum gap is 50 ms;
extrapolation is unsupported. Quaternion sign flips are harmless. Unknown
clocks, inconsistent clock pairs, invalid quaternions, and missing brackets
produce rejection reasons instead of zeros.

FC boot-time rollback and ODOMETRY reset counters create segments. V1 uses only
the segment with the recorded ground start reference and reports exclusions;
start a separate referenced run for a new estimator segment. Unexplained
coordinate resets without reset metadata must be marked as separate segments
in a newly prepared replay dataset. No motion edge crosses a known reset or
telemetry gap. `ODOMETRY` may be selected explicitly, but requires verified
LOCAL_NED/BODY_FRD frames; duplicate PX4 publications are not independent factors.

For explicitly approximate replay, set `allow_unverified_measurements: true`
and `legacy_replay.allow_assumed_pose: true`. Missing exposure metadata then
uses host receive times, and every result is approximate. Full quaternion
telemetry is still preferred. A prepared `legacy_pose` event can instead hold
`x`, `y`, `yaw_rad`, and explicit timestamps; absent Z and roll/pitch use the
configured legacy assumptions only when coordinates are explicitly confirmed
ground-relative. Neither missing yaw nor a missing start reference becomes zero.

## Initialize and optimize

```bash
python3 laptop/tag_mapper.py --config laptop/mapper.local.yaml \
  --anchors laptop/anchors.local.yaml --mode replay --operation initialize \
  --run mapping_runs/session01 --start-event measured_start.json \
  --frame-metadata exposure_metadata.jsonl --output mapping_results/initial01

python3 laptop/tag_mapper.py --config laptop/mapper.local.yaml \
  --anchors laptop/anchors.local.yaml --mode replay --operation optimize \
  --run mapping_runs/session01 --start-event measured_start.json \
  --frame-metadata exposure_metadata.jsonl --output mapping_results/optimized01
```

Initialization can run without an anchor and does not claim anchor correction.
Optimization requires at least one enabled observed anchor by default. A
deliberate anchor-free diagnostic run requires
`optimization.allow_anchor_free_debug_run: true` and is labeled accordingly.
Live initialization first records a run and then initializes it; live
optimization is rejected because v1 optimization is offline batch.

Output directories must be new and outside the immutable input run. The default
is a timestamped child of `output.directory` (`mapping_results` in the template).
Exit status is 0 for initialization or convergence, 2 for diagnosed failures.
Inspect `failure.json` or the solver status; partial results are not success.

Results include `results.json`, `tags.csv`, `trajectory.csv`,
`synchronized_poses.csv`, `residuals.csv`, `diagnostics.csv`, `iterations.csv`,
`evaluation.csv`, and `map.png`. Raw CSVs are also copied to the result's `csv/`
directory. Diagnostics retain initial ground intersections and inferred edge
lengths. The graph retains actual observation-time poses and telemetry support
through detection-free intervals, including the start reference.

Start XY are fixed through reduced coordinates; height and full attitude have
finite priors. Motion measurements are immutable original telemetry relative
transforms. Motion noise uses conservative configurable heuristic weights
scaled with elapsed time; fused PX4 differences remain correlated. The graph
does not independently reweight PnP estimates or ground intersections from the
same pixels. Solver trial steps preserve planar geometry by construction.

Rank/conditioning are checked before accepting a solution. Small graphs use
dense singular values; larger graphs use sparse extreme-singular-value checks.
Failed rank verification is diagnosed rather than treated as observability.
For converged graphs with at most 600 free variables, exported covariance is
local IRLS curvature, not an independently calibrated accuracy guarantee.
Larger graphs leave covariance unavailable. Zero tag-height variance describes
the assumed exact plane, not an independent height measurement.

## Independent evaluation and checks

Optionally point `evaluation.held_out_survey_file` to an evaluation-only YAML
file (relative paths resolve beside the mapper config):

```yaml
tags:
  - family_or_dictionary: tag16h5
    tag_id: 1
    frame: ground_start_NED
    position_m: [null, null, 0.0]  # Replace with independently measured values.
```

These coordinates are opened only after optimization and never initialize or
constrain tags. Keep these IDs separate from anchors. Detections may remain
in the graph as unknown landmarks. Evaluation reports initial/final horizontal
error, RMSE, median, and maximum, without fitting an extra alignment. Missing
truth yields `independent_accuracy_not_evaluated`. Anchor fit, low pixel error,
and convergence alone do not establish physical accuracy.

Run the acceptance and regression suite with:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q laptop/tests
```

Tests include the supplied numerical intersection, manifold Jacobians,
whitening, exact floor geometry, drifted synthetic recovery, raw/CSV packet
correspondence, synchronization/reset failures, and evaluation isolation.
Hardware accuracy still requires measured surveys and verified exposure timing.
