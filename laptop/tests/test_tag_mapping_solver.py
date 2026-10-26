"""Meaningful optimization and diagnosed failures, never a solver fallback."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tag_mapping.geometry import se3_log
from tag_mapping.solver import rank_diagnostics, solve
from test_tag_mapping_data import make_graph


@pytest.mark.parametrize('smallest,full_rank', [(1e-3, True), (0.0, False)])
def test_moderate_graph_rank_does_not_depend_on_arpack_convergence(
    monkeypatch, smallest, full_rank
):
    from scipy import sparse
    import tag_mapping.solver as solver

    def unavailable(*args, **kwargs):
        raise RuntimeError('iterative SVD did not converge')

    monkeypatch.setattr(solver, 'svds', unavailable)
    diagonal = np.ones(601)
    diagonal[-1] = smallest
    diagnostics = rank_diagnostics(sparse.diags(diagonal, format='csr'))
    assert diagnostics['full_rank'] is full_rank
    assert diagnostics['smallest_singular_value'] == pytest.approx(smallest)


def test_synthetic_recovery_decreases_cost_without_mutating_motion(tmp_path):
    graph, _, _, (_, _, truth) = make_graph(tmp_path)
    measurements = [f.measurement.copy() for f in graph.factors if f.kind == "motion"]
    for f in graph.factors:
        if f.kind == "motion":
            assert np.linalg.norm(f.evaluate(graph.initial, graph.config)[0]) < 1e-12
    result = solve(graph)
    assert result.status.startswith("converged"), result
    assert result.solver == "explicit_gauss_newton_lsmr_damp_0"
    costs = [i["cost"] for i in result.history if i["accepted"]]
    assert all(a > b for a, b in zip(costs, costs[1:]))
    key = ("tag16h5", 1)
    initial = np.linalg.norm(graph.initial.tags[key][:2] - truth[key][:2])
    final = np.linalg.norm(result.state.tags[key][:2] - truth[key][:2])
    assert final < initial and final < 0.02
    assert result.covariance is not None
    after = [f for f in graph.factors if f.kind == "motion"]
    assert any(
        np.linalg.norm(f.evaluate(result.state, graph.config)[0]) > 1e-5 for f in after
    )
    assert all(
        np.array_equal(before, f.measurement) and not f.measurement.flags.writeable
        for before, f in zip(measurements, after)
    )


def test_position_anchor_has_no_yaw_measurement_and_rank_is_checked(tmp_path):
    graph, _, _, _ = make_graph(tmp_path)
    anchor = next(f for f in graph.factors if f.kind == "anchor")
    _, j = anchor.evaluate(graph.initial, graph.config)
    assert np.array_equal(j["tag", anchor.keys[0]][:, 2], np.zeros(2))
    assert not any(f.kind == "yaw" for f in graph.factors)
    graph.factors = [
        f for f in graph.factors if f.kind not in ("height", "attitude", "anchor")
    ]
    assert solve(graph).status == "unobservable_rank_deficient"


def test_robust_objective_and_correlated_covariance_whitening(tmp_path):
    graph, _, _, _ = make_graph(tmp_path)
    f = next(f for f in graph.factors if f.kind == "pixel")
    a = np.eye(8)
    a[0, 1] = 0.3
    f.covariance = a @ a.T
    f.measurement = f.measurement + 10
    robust = graph.linearize(graph.initial)[0]
    quadratic = graph.linearize(graph.initial, robust=False)[0]
    assert robust < quadratic
    # Whitening must be L^-1, including non-diagonal covariance.
    r, _ = f.evaluate(graph.initial, graph.config)
    norm2 = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(f.covariance), r)) ** 2
    assert norm2 == pytest.approx(r @ np.linalg.solve(f.covariance, r))


def test_optional_survey_yaw_outliers_and_backtracking(tmp_path):
    from tag_mapping.graph import Factor

    graph, _, _, _ = make_graph(tmp_path)
    key = ("tag16h5", 0)
    yaw = Factor("yaw", (key,), 0.2, np.array([[0.02**2]]), {})
    graph.factors.append(yaw)
    np.testing.assert_array_equal(
        yaw.evaluate(graph.initial, graph.config)[1]["tag", key], [[0, 0, 1]]
    )
    first = next(f for f in graph.factors if f.kind == "pixel")
    first.measurement = first.measurement.copy()
    first.measurement[0] += [18, -14]
    result = solve(graph)
    assert result.status.startswith("converged")
    assert np.linalg.norm(result.state.tags[("tag16h5", 1)][:2] - [1.0, 0.22]) < 0.03


def test_invalid_depth_trials_backtrack_without_changing_tag_parameterization(
    tmp_path, monkeypatch
):
    graph, _, _, _ = make_graph(tmp_path)
    original = graph.retract

    def require_smaller_step(state, delta):
        trial = original(state, delta)
        if np.max(np.abs(delta)) > 0.02:
            # Force an observed corner behind the camera on large steps.
            trial.poses[1][2, 3] = 10.0
        return trial

    monkeypatch.setattr(graph, "retract", require_smaller_step)
    result = solve(graph)
    assert any(
        not h["accepted"] and "depth" in h.get("reason", "") for h in result.history
    )
    assert any(h["accepted"] and 0 < h["alpha"] < 1 for h in result.history)
