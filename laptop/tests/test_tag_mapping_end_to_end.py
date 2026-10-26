"""CLI artifacts, independent surveys, and immutable replay sources."""

from pathlib import Path
import json
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tag_mapper import main
from tag_mapping.data import file_hash
from tag_mapping.reporting import evaluate
from test_tag_mapping_data import synthetic_run


def test_cli_optimizes_exports_and_evaluates_after_fit(tmp_path, monkeypatch):
    import tag_mapper

    original_solve, original_evaluate = tag_mapper.solve, tag_mapper.evaluate
    finished = []

    def solve_then_mark(graph):
        result = original_solve(graph)
        finished.append(True)
        return result

    def evaluate_after_fit(*args):
        assert finished
        return original_evaluate(*args)

    monkeypatch.setattr(tag_mapper, "solve", solve_then_mark)
    monkeypatch.setattr(tag_mapper, "evaluate", evaluate_after_fit)
    c, run, config, truth = synthetic_run(tmp_path)
    survey = tmp_path / "held_out_tag_survey.yaml"
    key = ("tag16h5", 1)
    survey.write_text(
        yaml.safe_dump(
            dict(
                tags=[
                    dict(
                        family_or_dictionary=key[0],
                        tag_id=key[1],
                        position_m=[*map(float, truth[key][:2]), 0.0],
                    )
                ]
            )
        )
    )
    original = {p: file_hash(p) for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "results"
    assert (
        main(
            [
                "--config",
                str(config),
                "--operation",
                "optimize",
                "--run",
                str(run),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / "results.json").read_text())
    assert result["evaluation"]["status"] == "evaluated"
    assert (
        result["evaluation"]["optimized"]["horizontal_rmse_m"]
        < result["evaluation"]["initial"]["horizontal_rmse_m"]
    )
    assert result["anchor_status"] == "anchor_corrected"
    for name in [
        "map.png",
        "tags.csv",
        "trajectory.csv",
        "residuals.csv",
        "synchronized_poses.csv",
        "csv/position.csv",
    ]:
        assert (output / name).stat().st_size > 0
    assert all(file_hash(p) == h for p, h in original.items())
    assert set(original) == {p for p in run.rglob("*") if p.is_file()}
    assert all(t["z_m"] == 0 for t in result["tags"])
    # Reusing an output directory or putting output inside a run changes nothing.
    before = {p: file_hash(p) for p in output.rglob("*") if p.is_file()}
    assert (
        main(
            [
                "--config",
                str(config),
                "--operation",
                "optimize",
                "--run",
                str(run),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert before == {p: file_hash(p) for p in output.rglob("*") if p.is_file()}
    assert (
        main(
            [
                "--config",
                str(config),
                "--operation",
                "initialize",
                "--run",
                str(run),
                "--output",
                str(run / "bad"),
            ]
        )
        == 2
    )
    assert not (run / "bad").exists()


def test_no_truth_status_and_overlap_rejection(tmp_path):
    assert (
        evaluate({}, {}, tmp_path / "absent.yaml", set())["status"]
        == "independent_accuracy_not_evaluated"
    )
    p = tmp_path / "truth.yaml"
    p.write_text(
        "tags:\n- family_or_dictionary: tag16h5\n  tag_id: 0\n  position_m: [1, 2, 0]\n"
    )
    with pytest.raises(ValueError, match="overlaps"):
        evaluate({}, {}, p, {("tag16h5", 0)})


def test_initialization_without_survey_and_external_start_override(tmp_path):
    c, run, config, _ = synthetic_run(tmp_path)
    c["known_tags"] = []
    config.write_text(yaml.safe_dump(c))
    override = tmp_path / "start.json"
    override.write_bytes((run / "start_event.json").read_bytes())
    output = tmp_path / "initial"
    assert (
        main(
            [
                "--config",
                str(config),
                "--operation",
                "initialize",
                "--run",
                str(run),
                "--start-event",
                str(override),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / "results.json").read_text())
    assert result["anchor_status"] == "not_anchor_corrected"
    assert result["evaluation"]["status"] == "independent_accuracy_not_evaluated"
