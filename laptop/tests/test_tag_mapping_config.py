"""Configuration errors must not quietly alter the estimator's contract."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tag_mapping.config import ConfigError, TEMPLATE, load_config, read_yaml


def complete_config():
    c = read_yaml(TEMPLATE)
    c["frames"]["navigation_to_true_ned_verified"] = True
    c["camera"]["extrinsic_T_B_C"]["measurement_verified"] = True
    c["camera"]["extrinsic_T_B_C"]["translation_sigma_m"] = [0.001] * 3
    c["camera"]["extrinsic_T_B_C"]["rotation_sigma_rad"] = [0.01] * 3
    c["camera"]["held_out_calibration_validation_complete"] = True
    c["detector"]["preserves_native_corner_order_verified"] = True
    c["tag_geometry"]["default_size_measured"] = True
    c["tag_geometry"]["default_size_sigma_m"] = 0.001
    c["known_tags"] = [
        dict(
            enabled=True,
            family_or_dictionary="tag16h5",
            tag_id=0,
            position_m=[0.85, 0.05, 0.0],
            position_sigma_xy_m=[0.005, 0.005],
        )
    ]
    return c


def save_config(tmp_path, c):
    p = tmp_path / "mapper.yaml"
    p.write_text(yaml.safe_dump(c))
    return p


def test_template_is_valid_but_survey_disabled():
    c = load_config(TEMPLATE)
    assert not any(a["enabled"] for a in c["known_tags"])


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.update(surprise=True),
        lambda c: c["ground"].update(optimize_tag_z=True),
        lambda c: c["optimization"].update(solver="lm"),
        lambda c: c["optimization"].update(linear_damping=0.01),
        lambda c: c["camera"].update(distortion=[0] * 4),
        lambda c: c["camera"]["extrinsic_T_B_C"].update(
            rotation_matrix=np.zeros((3, 3)).tolist()
        ),
        lambda c: c["known_tags"][0].update(position_m=[1, 2, 0.1]),
        lambda c: c["known_tags"][0].update(position_sigma_xy_m=[0, 0.1]),
        lambda c: c["known_tags"][0].update(tag_id=99),
        lambda c: c["known_tags"].append(deepcopy(c["known_tags"][0])),
        lambda c: c["tag_geometry"].update(default_size_m=float("nan")),
    ],
)
def test_invalid_schema_and_model_rejected(tmp_path, edit):
    c = complete_config()
    edit(c)
    with pytest.raises(ConfigError):
        load_config(save_config(tmp_path, c))


def test_duplicate_yaml_key_and_anchor_conflict(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("schema_version: 2\nschema_version: 2\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(p)
    p = save_config(tmp_path, complete_config())
    with pytest.raises(ConfigError, match="conflict"):
        load_config(p, Path(TEMPLATE).with_name("anchors.example.yaml"))


def test_external_survey_matches_id_and_yaw_is_optional(tmp_path):
    c = complete_config()
    anchors = c["known_tags"]
    c["known_tags"] = []
    external = tmp_path / "anchors.yaml"
    external.write_text(yaml.safe_dump(dict(known_tags=anchors)))
    actual = load_config(save_config(tmp_path, c), external)
    assert actual["known_tags"][0]["tag_id"] == 0
    assert actual["known_tags"][0]["surveyed_yaw_rad"] is None
