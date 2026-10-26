"""Regression geometry and analytic factor derivatives on the manifold."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tag_mapping.geometry import (
    ground_corners,
    ground_intersections,
    inverse,
    pose_from_quaternion,
    project,
    se3_exp,
    se3_log,
    transform,
)
from tag_mapping.synchronization import ground_reference
from test_tag_mapping_config import complete_config
from test_tag_mapping_data import make_graph


def test_supplied_intersection_and_explicit_pixel_convention():
    c = complete_config()
    pose = transform(
        Rotation.from_euler("z", 5.51864).as_matrix(), [0.2237, -0.3076, -0.5]
    )
    pixels = np.array(
        [[112.267, 142.736], [137.211, 147.631], [147.283, 134.110], [125.060, 130.243]]
    )
    points = ground_intersections(pixels, pose, c, "opencv_pixel_centres")
    assert points[0] == pytest.approx([0.536082639, -0.512756191, 0], abs=1e-9)
    assert points.mean(axis=0) == pytest.approx(
        [0.581239162, -0.566574756, 0], abs=1e-9
    )
    assert np.array_equal(points[:, 2], np.zeros(4))
    corrected = ground_intersections(pixels + 0.5, pose, c, "apriltag_pixel_edges")
    assert corrected == pytest.approx(points)
    with pytest.raises(ValueError, match="horizon|upward"):
        ground_intersections(
            pixels, transform(np.diag([-1.0, 1.0, -1.0]), [0, 0, -0.5]), c
        )


def test_frames_and_nonzero_origin():
    r = Rotation.from_euler("xyz", [0.2, -0.3, np.pi / 2]).as_matrix()
    assert Rotation.from_euler("z", np.pi / 2).apply([1, 0, 0]) == pytest.approx(
        [0, 1, 0], abs=1e-15
    )
    t = transform(r, [3, 4, -0.5])
    assert inverse(t) @ t == pytest.approx(np.eye(4))
    q = Rotation.from_matrix(r).as_quat()
    assert pose_from_quaternion([3, 4, -0.5], q) == pytest.approx(t)
    start = dict(
        coordinate_convention="NED",
        datum_source="measured",
        fc_time_ns=1,
        native_position_m=[3, 4, 2],
        quaternion_xyzw=q.tolist(),
        initial_body_height_above_floor_m=0.5,
        initial_body_height_sigma_m=0.01,
    )
    origin, world, _ = ground_reference(start, complete_config())
    assert origin == pytest.approx([3, 4, 2.5])
    assert world[:3, 3] == pytest.approx([0, 0, -0.5])
    assert world[:3, :3] == pytest.approx(r)
    c = complete_config()
    c["frames"]["start"]["input_positions_already_ground_start_relative"] = True
    start["native_position_m"] = [0, 0, -0.5]
    assert ground_reference(start, c)[1][:3, 3] == pytest.approx([0, 0, -0.5])


@pytest.mark.parametrize("scale", [1e-9, 0.1, 1.0])
def test_se3_exp_log_translation_coupling(scale):
    v = np.array([0.2, -0.3, 0.1, 0.5, -0.1, 0.2]) * scale
    assert se3_log(se3_exp(v)) == pytest.approx(v, abs=1e-12)


def test_analytic_jacobians_whitening_and_exact_tag_plane(tmp_path):
    graph, _, _, _ = make_graph(tmp_path)
    rng = np.random.default_rng(4)
    state = graph.retract(graph.initial, rng.normal(0, 0.002, graph.dimension))
    _, r, j = graph.linearize(state, robust=False)
    numerical = np.zeros(j.shape)
    h = 1e-6
    for col in range(graph.dimension):
        d = np.zeros(graph.dimension)
        d[col] = h
        plus = graph.linearize(graph.retract(state, d), robust=False)[1]
        minus = graph.linearize(graph.retract(state, -d), robust=False)[1]
        numerical[:, col] = (plus - minus) / (2 * h)
    np.testing.assert_allclose(j.toarray(), numerical, rtol=1e-5, atol=2e-6)
    for alpha in (1.0, 0.5, 0.01, -100.0):
        trial = graph.retract(state, alpha * rng.normal(0, 0.1, graph.dimension))
        for t in trial.tags.values():
            assert len(t) == 3
            assert np.array_equal(ground_corners(t, 0.12)[:, 2], np.zeros(4))
    assert state.poses[0][:2, 3] == pytest.approx(graph.initial.poses[0][:2, 3])
    assert state.poses[0][2, 3] != graph.initial.poses[0][2, 3]


def test_sensitivity_to_mount_height_size_timing_and_weights(tmp_path):
    from copy import deepcopy
    from tag_mapping.data import load_events
    from tag_mapping.geometry import mount
    from tag_mapping.synchronization import synchronize

    graph, _, _, (run, _, _) = make_graph(tmp_path)
    c = graph.config
    pixel = next(f for f in graph.factors if f.kind == "pixel")
    i, key = pixel.keys
    pose = graph.initial.poses[i]
    p = ground_intersections(pixel.measurement, pose, c, pixel.metadata["convention"])
    higher = pose.copy()
    higher[2, 3] -= 0.05
    assert (
        np.linalg.norm(
            ground_intersections(
                pixel.measurement, higher, c, pixel.metadata["convention"]
            )
            - p
        )
        > 0.01
    )
    changed = deepcopy(c)
    r = Rotation.from_euler("y", np.deg2rad(2)).as_matrix() @ mount(c)[:3, :3]
    changed["camera"]["extrinsic_T_B_C"]["rotation_matrix"] = r.tolist()
    assert (
        np.linalg.norm(
            ground_intersections(
                pixel.measurement, pose, changed, pixel.metadata["convention"]
            )
            - p
        )
        > 0.01
    )
    changed = deepcopy(c)
    changed["tag_geometry"]["default_size_m"] *= 1.1
    np.testing.assert_array_equal(
        ground_intersections(
            pixel.measurement, pose, changed, pixel.metadata["convention"]
        ),
        p,
    )
    assert (
        np.linalg.norm(
            pixel.evaluate(graph.initial, c)[0]
            - pixel.evaluate(graph.initial, changed)[0]
        )
        > 1
    )
    events = load_events(run)
    for e in events:
        if e["kind"] == "frame":
            e["capture_time_ns"] -= 10_000_000
    shifted, _, _ = synchronize(events, run, c)
    assert np.linalg.norm(shifted[0]["pose"] - pose) > 1e-3
    d = np.zeros(graph.dimension)
    d[graph.slices["pose", 1].start + 3] = 0.02
    state = graph.retract(graph.initial, d)
    cost = graph.linearize(state)[0]
    for f in graph.factors:
        if f.kind == "motion":
            f.covariance *= 4
    assert graph.linearize(state)[0] < cost
