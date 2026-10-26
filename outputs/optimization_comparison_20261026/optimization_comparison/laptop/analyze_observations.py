#!/usr/bin/env python3
"""Adapt a legacy observations ZIP to the existing ground-tag graph and solver.

This is an offline analysis driver, not a replacement optimisation backend.
All camera factors, ground initialisation, SE(3) updates, robust costs and
Gauss--Newton solves are provided by tag_mapping's existing Python modules.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from tag_mapping.config import load_config
from tag_mapping.comparison import export_comparison
from tag_mapping.data import dump_json, file_hash, software_hash, write_csv
from tag_mapping.graph import build_graph
from tag_mapping.reporting import evaluate, export_results
from tag_mapping.solver import solve
from tag_mapping.synchronization import interpolate_pose, synchronize


def ns(value):
    return int(Decimal(str(value)) * 1_000_000_000)


def read_csv(archive, name):
    return list(csv.DictReader(io.TextIOWrapper(archive.open('observations/' + name))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anchor', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--anchor-sigma-m', type=float, default=0.01)
    parser.add_argument('--keyframe-interval-s', type=float, default=1.0)
    parser.add_argument('--odometry-weight-scale', type=float, default=1.0,
                        help='Multiply default odometry sigmas, for sensitivity checks')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    wall_start = time.perf_counter()
    with zipfile.ZipFile(args.archive) as archive:
        recorded_config = yaml.safe_load(archive.read('observations/config_used.yaml'))
        legacy_events = [json.loads(line) for line in archive.read('observations/events.jsonl').decode().splitlines()]
        origin_event = next(e for e in legacy_events if e['event'] == 'ORIGIN_SET')
        telemetry = read_csv(archive, 'px4_telemetry.csv')
        raw = read_csv(archive, 'raw_detections.csv')
        legacy_observations = read_csv(archive, 'observations.csv')
        legacy_summary = json.loads(archive.read('observations/summary.json'))

    # Rebase both positions AND full attitudes before passing data to the mapper.
    # Its NED-labelled mathematical axes are used as forward/right/down here.
    rz = Rotation.from_euler('z', -origin_event['origin_yaw_rad']).as_matrix()
    origin = np.array([origin_event['origin_x'], origin_event['origin_y'], 0.0])
    run_id = 'legacy-' + file_hash(args.archive)[:16]
    # Several UDP batches share a host timestamp while the cached attitude (and
    # occasionally position) changes. Retain the final snapshot for that host
    # instant rather than inventing distinct capture times for those rows.
    duplicates = len(telemetry) - len({r['t_epoch_sample'] for r in telemetry})
    telemetry = list({r['t_epoch_sample']: r for r in telemetry}.values())
    event_stream = []
    for k, row in enumerate(telemetry):
        xyz = rz @ (np.array([float(row[f'px4_{a}_m']) for a in 'xyz']) - origin)
        q = Rotation.from_matrix(rz @ Rotation.from_quat(
            [float(row['px4_q' + a]) for a in 'xyzw']).as_matrix()).as_quat()
        envelope = dict(run_id=run_id, kind='position', segment_id=0,
                        host_receive_ns=ns(row['t_epoch_sample']),
                        fc_time_ns=ns(row['t_epoch_sample']), sequence=k)
        # The sample epoch is a host-derived approximate lookup time, not a
        # verified FC/exposure timestamp. synchronize uses host_receive_ns.
        event_stream.append(dict(envelope, **dict(zip('xyz', map(float, xyz)))))
        event_stream.append(dict(envelope, kind='attitude',
                                 **dict(zip(('qx', 'qy', 'qz', 'qw'), map(float, q)))))

    c = yaml.safe_load((Path(__file__).parent / 'mapper_config.example.yaml').read_text())
    c['config_id'] = 'legacy_observations_approximate_ground_tags'
    c['allow_unverified_measurements'] = True
    c['legacy_replay']['allow_assumed_pose'] = True
    c['frames']['start']['input_positions_already_ground_start_relative'] = True
    pitch = np.deg2rad(recorded_config['camera']['mount']['pitch_deg'])
    c['camera']['extrinsic_T_B_C']['rotation_matrix'] = [
        [0.0, -float(np.sin(pitch)), float(np.cos(pitch))],
        [1.0, 0.0, 0.0],
        [0.0, float(np.cos(pitch)), float(np.sin(pitch))],
    ]
    mount = recorded_config['camera']['mount']
    c['camera']['extrinsic_T_B_C']['translation_m'] = [
        mount['offset_fwd_m'], mount['offset_right_m'], mount['offset_down_m']]
    c['camera']['extrinsic_T_B_C']['rotation_description'] = 'Optical right/down/forward to FRD; recorded nose-down pitch'
    c['camera']['extrinsic_T_B_C']['measurement_notes'] = 'From archived config_used.yaml, not independently verified'
    c['tag_geometry']['default_size_m'] = recorded_config['tag']['size_m']
    # The original pose lookup already accepted these rows. The longest logged
    # telemetry reception gap is 1.0283 s; keep it explicitly approximate.
    c['timing']['max_bracket_gap_ms'] = 1100.0
    c['optimization']['keyframes']['minimum_interval_s'] = args.keyframe_interval_s
    for key in ('fallback_odometry_rotation_sigma_rad', 'fallback_odometry_translation_sigma_m'):
        c['uncertainty'][key] = [v * args.odometry_weight_scale for v in c['uncertainty'][key]]
    c['known_tags'] = []
    config_path = out / 'config.yaml'
    config_path.write_text(yaml.safe_dump(c, sort_keys=False))
    c = load_config(config_path)
    start_ns = ns(origin_event['t_epoch'])
    positions = [e for e in event_stream if e['kind'] == 'position']
    attitudes = [e for e in event_stream if e['kind'] == 'attitude']
    start_pose = interpolate_pose(positions, attitudes, start_ns, 1_100_000_000, 'host_receive_ns')
    # The marked start XY is from ORIGIN_SET, not interpolated drift.
    start_pose[:2, 3] = 0.0
    start = dict(run_id=run_id, segment_id=0, fc_time_ns=start_ns,
                 host_receive_ns=start_ns, coordinate_convention='NED',
                 native_position_m=start_pose[:3, 3].tolist(),
                 quaternion_xyzw=Rotation.from_matrix(start_pose[:3, :3]).as_quat().tolist(),
                 initial_body_height_above_floor_m=float(-start_pose[2, 3]),
                 initial_body_height_sigma_m=0.05,
                 datum_source='APPROXIMATE: ORIGIN_SET XY; assume native PX4 z=0 is floor; height is telemetry-derived')
    dump_json(out / 'start_event.json', start)
    frames = {}
    accepted_raw = [r for r in raw if r['accepted'] == '1']
    for row in accepted_raw:
        key = int(row['frame_id'])
        if key not in frames:
            frames[key] = dict(kind='frame', run_id=run_id, segment_id=0,
                               camera_id='camera0', boot_nonce=None, frame_id=key,
                               host_receive_ns=ns(row['camera_capture_timestamp']),
                               capture_time_ns=None, exposure_verified=False,
                               width=320, height=240,
                               calibration_id=c['camera']['calibration_id'],
                               pixel_convention='apriltag_pixel_edges', detections=[])
        frames[key]['detections'].append(dict(
            family_or_dictionary='tag16h5', tag_id=int(row['tag_id']),
            corners=[[float(row[f'corner_{i}_{a}']) for a in 'uv'] for i in range(4)],
            hamming=int(row['hamming_distance']), margin=float(row['decision_margin'])))
    event_stream.extend(frames.values())
    converted = out / 'converted_events.jsonl'
    converted.write_text(''.join(json.dumps(e) + '\n' for e in event_stream))
    prep_end = time.perf_counter()

    # Use the existing synchronizer's explicit legacy path. Never label these
    # timestamps, floor reference, calibration or survey uncertainties verified.
    sync_start = time.perf_counter()
    synchronized, sync_rows, context = synchronize(event_stream, out, c, start_event=start)
    context['warnings'].extend([
        'Axes are start-forward/right/down after rotating native XY and full attitude by minus initial heading.',
        'Legacy host-derived camera and telemetry epochs; exposure timing not verified; telemetry max bracket gap 1.1 s.',
        'Assume native PX4 z=0 is floor; start height is a telemetry proxy, not an independent measurement; sigma=0.05 m.',
        'Calibration is the existing repository 320x240 checkerboard calibration; recording contains no independent calibration identifier.',
        'Ground truth constrains tag %d XY only; anchor sigma=%.6g m is assumed, not surveyed uncertainty.' % (args.anchor, args.anchor_sigma_m),
        '%d duplicate host-time telemetry snapshots coalesced by retaining the last logged snapshot at each timestamp.' % duplicates,
        'Old recording acceptance gates preserved; tag 2 has only two accepted frames and remains tentative.',
    ])
    sync_end = time.perf_counter()
    initial_start = time.perf_counter()
    initial_graph = build_graph(synchronized, context, c, require_anchor=False)
    initial_end = time.perf_counter()
    before_tags = {k: v.copy() for k, v in initial_graph.initial.tags.items()}

    # Only the selected anchor enters the fit. Load held-out XY truth later.
    anchor_positions = {0: [1.2, 0.65, 0.0], 1: [1.2, -0.7, 0.0], 2: [2.78, -0.7, 0.0]}
    ca = deepcopy(c)
    ca['known_tags'] = [dict(enabled=True, family_or_dictionary='tag16h5',
                             tag_id=args.anchor, frame='ground_start_NED',
                             prior_type='position_xy', position_m=anchor_positions[args.anchor],
                             position_sigma_xy_m=[args.anchor_sigma_m] * 2,
                             surveyed_yaw_rad=None, surveyed_yaw_sigma_rad=None,
                             measurement_notes='User-provided picture; uncertainty assumed for this approximate offline calculation')]
    config_path.write_text(yaml.safe_dump(ca, sort_keys=False))
    ca = load_config(config_path)
    anchored_start = time.perf_counter()
    graph = build_graph(synchronized, context, ca, require_anchor=True)
    anchored_end = time.perf_counter()
    print(json.dumps(dict(stage='graph_ready', poses=len(graph.initial.poses),
                          tags=len(graph.initial.tags), dimension=graph.dimension,
                          factors=dict(Counter(f.kind for f in graph.factors)),
                          before_tags={str(k): v.tolist() for k, v in before_tags.items()})), flush=True)
    solve_start = time.perf_counter()
    result = solve(graph)
    solve_end = time.perf_counter()
    print(json.dumps(dict(stage='solved', status=result.status,
                          seconds=solve_end-solve_start, conditioning=result.conditioning)), flush=True)

    truth = {0: [1.2, .65, 0.], 1: [1.2, -.7, 0.], 2: [2.78, -.7, 0.], 3: [2.7, .52, 0.]}
    survey_path = out / 'held_out_survey.yaml'
    survey_path.write_text(yaml.safe_dump(dict(tags=[dict(
        family_or_dictionary='tag16h5', tag_id=k, position_m=p)
        for k, p in truth.items() if k != args.anchor])))
    evaluation = evaluate(before_tags, result.state.tags, survey_path, {('tag16h5', args.anchor)})
    provenance = dict(archive_sha256=file_hash(args.archive), software_sha256=software_hash(),
                      config_sha256=file_hash(config_path), effective_config=ca,
                      calibration_sha256=file_hash(Path(__file__).parent.parent / 'camera_intrinsics.json'))
    export_results(out, graph, result, context, sync_rows, evaluation, provenance)
    counts = Counter(int(r['tag_id']) for r in accepted_raw)
    rows = []
    for tag_id, gt in truth.items():
        key = ('tag16h5', tag_id)
        row = dict(tag_id=tag_id, anchor=tag_id == args.anchor,
                   accepted_frames=counts[tag_id], ground_truth_x_m=gt[0], ground_truth_y_m=gt[1])
        if key not in result.state.tags:
            row['status'] = 'unobserved'
        else:
            row['status'] = 'tentative_two_frames' if counts[tag_id] < 3 else result.status
            for stage, state in [('before', before_tags), ('after', result.state.tags)]:
                xy = state[key][:2]
                delta = xy - np.array(gt[:2])
                row.update({stage+'_x_m': float(xy[0]), stage+'_y_m': float(xy[1]),
                            stage+'_error_x_m': float(delta[0]), stage+'_error_y_m': float(delta[1]),
                            stage+'_error_m': float(np.linalg.norm(delta))})
            old = [r for r in legacy_observations if r['accepted'] == '1' and int(r['tag_id']) == tag_id]
            old_mean = np.mean([[float(r['north_m']), float(r['east_m'])] for r in old], axis=0)
            row.update(recorded_mean_x_m=float(old_mean[0]), recorded_mean_y_m=float(old_mean[1]),
                       recorded_mean_error_m=float(np.linalg.norm(old_mean-np.array(gt[:2]))),
                       archived_summary_x_m=legacy_summary['tags'][str(tag_id)]['north_m'],
                       archived_summary_y_m=legacy_summary['tags'][str(tag_id)]['east_m'])
        rows.append(row)
    timing = dict(zip_load_and_adapter_s=prep_end-wall_start,
                  synchronization_s=sync_end-sync_start,
                  before_initialization_s=initial_end-initial_start,
                  anchored_graph_build_s=anchored_end-anchored_start,
                  optimization_s=solve_end-solve_start,
                  before_calculation_s=initial_end-initial_start,
                  after_calculation_s=(anchored_end-anchored_start)+(solve_end-solve_start),
                  baseline_pipeline_s=(prep_end-wall_start)+(sync_end-sync_start)+(initial_end-initial_start),
                  optimized_pipeline_excluding_export_s=(prep_end-wall_start)+(sync_end-sync_start)+(anchored_end-anchored_start)+(solve_end-solve_start),
                  driver_through_mapper_export_s=time.perf_counter()-wall_start,
                  clock='time.perf_counter, wall seconds; excludes Python startup/imports; one run')
    for row in rows:
        row['before_batch_calculation_time_s'] = timing['before_calculation_s']
        row['after_batch_calculation_time_s'] = timing['after_calculation_s']
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(out / 'comparison.csv', rows, fields)
    dump_json(out / 'comparison.json', dict(rows=rows, timing=timing, independent_evaluation=evaluation,
               solver_status=result.status, source=args.archive.as_posix(), assumptions=context['warnings'],
               baseline_definition='Existing mapper ray-floor initialization and medoid, before applying any anchor',
               backend='Existing tag_mapping.graph.build_graph and tag_mapping.solver.solve, with rank-check and linear-column-scaling fixes'))
    # The standard exporter seeds the fitting anchor at truth and consequently
    # calls it initial. Keep the genuine pre-anchor values explicitly alongside.
    dump_json(out / 'unanchored_initial_tags.json', {str(k): v.tolist() for k, v in before_tags.items()})
    print(export_comparison(out, rows, timing, result.status), flush=True)
    return 0 if result.status.startswith('converged') else 2


if __name__ == '__main__':
    raise SystemExit(main())
