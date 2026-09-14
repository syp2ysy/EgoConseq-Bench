import json

import pytest
import torch

import train
from train import configure_model_for_training, validate_zero3_config
from vln_baseline.model import trainable_parameter_names


@pytest.mark.parametrize("key", ["depth_enable", "geometry_enable", "learning_ratte"])
def test_load_config_rejects_unsupported_options(tmp_path, key):
    config = tmp_path / "train.yaml"
    config.write_text(f"{key}: true\n")
    with pytest.raises(ValueError, match="Unsupported training config"):
        train.load_config(str(config))


class _FakeVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.merger = torch.nn.Linear(4, 4)


class _FakeQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _FakeVisual()


class _FakeNestedQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = _FakeVisual()


def test_configure_model_for_training_keeps_only_visual_merger_trainable_when_vision_frozen():
    model = _FakeQwen()

    allowed_prefixes = configure_model_for_training(model, {"freeze_vision_encoder": True})

    trainable = trainable_parameter_names(model)
    assert allowed_prefixes is None
    assert trainable == ["visual.merger.weight", "visual.merger.bias"]


def test_configure_model_for_training_freezes_nested_vision_encoder():
    model = _FakeNestedQwen()

    configure_model_for_training(model, {"freeze_vision_encoder": True})

    trainable = trainable_parameter_names(model)
    assert trainable == ["model.visual.merger.weight", "model.visual.merger.bias"]


def test_validate_zero3_config_rejects_non_zero3_config(tmp_path):
    path = tmp_path / "zero2.json"
    path.write_text(json.dumps({"zero_optimization": {"stage": 2}}))

    try:
        validate_zero3_config(str(path))
    except ValueError as exc:
        assert "ZeRO-3" in str(exc)
    else:
        raise AssertionError("Expected non-ZeRO-3 config to raise")
