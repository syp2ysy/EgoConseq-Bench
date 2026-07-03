import json
from pathlib import Path

import pytest
import torch

import train
from train import configure_model_for_training, validate_zero3_config
from vln_baseline.depth_head import DepthReadoutHead
from vln_baseline.model import (
    _clear_zero_active_sub_modules,
    load_depth_head_state,
    save_depth_head_state,
    trainable_parameter_names,
)


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


def test_stage1_configuration_trains_only_depth_head():
    model = _FakeQwen()
    model.depth_head = DepthReadoutHead(input_dim=4, hidden_dim=2)

    allowed_prefixes = configure_model_for_training(
        model,
        {
            "depth_stage": 1,
            "freeze_vision_encoder": True,
            "allowed_trainable_prefixes": ["depth_head."],
        },
    )

    trainable = trainable_parameter_names(model)
    assert allowed_prefixes == ["depth_head."]
    assert trainable
    assert all(name.startswith("depth_head.") for name in trainable)


def test_load_depth_head_state_strictly_loads_prefixed_checkpoint(tmp_path: Path):
    model = _FakeQwen()
    model.depth_head = DepthReadoutHead(input_dim=4, hidden_dim=2)
    expected = {
        f"depth_head.{name}": torch.full_like(param, 0.25)
        for name, param in model.depth_head.state_dict().items()
    }
    torch.save(expected, tmp_path / "pytorch_model.bin")

    loaded_keys = load_depth_head_state(model, str(tmp_path))

    assert loaded_keys == sorted(expected)
    for value in model.depth_head.state_dict().values():
        assert torch.allclose(value, torch.full_like(value, 0.25))


def test_save_depth_head_state_writes_only_depth_head_weights(tmp_path: Path):
    model = _FakeQwen()
    model.depth_head = DepthReadoutHead(input_dim=4, hidden_dim=2)

    saved_path = save_depth_head_state(model, str(tmp_path))

    assert saved_path is not None
    assert saved_path.exists()
    reloaded = _FakeQwen()
    reloaded.depth_head = DepthReadoutHead(input_dim=4, hidden_dim=2)
    loaded_keys = load_depth_head_state(reloaded, str(tmp_path))
    assert loaded_keys == sorted(f"depth_head.{key}" for key in model.depth_head.state_dict())


def test_clear_zero_active_sub_modules_clears_deepspeed_param_markers():
    param = torch.nn.Parameter(torch.ones(2))
    param.ds_active_sub_modules = {123}

    _clear_zero_active_sub_modules([param])

    assert param.ds_active_sub_modules == set()


def test_validate_zero3_config_rejects_non_zero3_config(tmp_path):
    path = tmp_path / "zero2.json"
    path.write_text(json.dumps({"zero_optimization": {"stage": 2}}))

    try:
        validate_zero3_config(str(path))
    except ValueError as exc:
        assert "ZeRO-3" in str(exc)
    else:
        raise AssertionError("Expected non-ZeRO-3 config to raise")


def test_assert_finite_scalar_rejects_nan_loss():
    assert hasattr(train, "assert_finite_scalar")
    with pytest.raises(RuntimeError, match="loss_action"):
        train.assert_finite_scalar(torch.tensor(float("nan")), "loss_action")
