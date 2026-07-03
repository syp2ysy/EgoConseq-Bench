from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval import _episode_instruction
from eval import SingleTurnNavigator


def test_episode_instruction_accepts_object_instruction():
    episode = SimpleNamespace(instruction=SimpleNamespace(instruction_text="go to the room"))

    assert _episode_instruction(episode) == "go to the room"


def test_episode_instruction_accepts_dict_instruction():
    episode = SimpleNamespace(instruction={"instruction_text": "turn left"})

    assert _episode_instruction(episode) == "turn left"


def test_navigator_append_observation_respects_max_action_history():
    navigator = object.__new__(SingleTurnNavigator)
    navigator.image_size = (2, 2)
    navigator.rgb_list = []
    navigator.max_action_history = 3

    for value in range(5):
        rgb = np.full((2, 2, 3), value, dtype=np.uint8)
        navigator._append_observation(rgb)

    assert len(navigator.rgb_list) == 3
    assert list(navigator.rgb_list[0].getdata())[0] == (2, 2, 2)


def test_navigator_generation_kwargs_uses_configured_max_new_tokens():
    navigator = object.__new__(SingleTurnNavigator)
    navigator.generation_max_new_tokens = 512
    navigator.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

    assert navigator._generation_kwargs()["max_new_tokens"] == 512


def test_navigator_generation_kwargs_supports_greedy_decode():
    navigator = object.__new__(SingleTurnNavigator)
    navigator.generation_max_new_tokens = 64
    navigator.decode_strategy = "greedy"
    navigator.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

    kwargs = navigator._generation_kwargs()

    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert kwargs["max_new_tokens"] == 64


def test_navigator_action_policy_can_keep_single_primitive():
    navigator = object.__new__(SingleTurnNavigator)
    navigator.action_execution_policy = "primitive1"

    assert navigator._apply_action_execution_policy([1, 1, 2]) == [1]


def test_navigator_adaptive_turn_policy_limits_turns_more_than_forward():
    navigator = object.__new__(SingleTurnNavigator)
    navigator.action_execution_policy = "adaptive_turn_safe"

    assert navigator._apply_action_execution_policy([2, 2, 1]) == [2]
    assert navigator._apply_action_execution_policy([1, 1, 1]) == [1, 1]
    assert navigator._apply_action_execution_policy([0]) == [0]


def test_navigator_loads_baseline_model(monkeypatch):
    calls = {}

    class FakeModel:
        device = "cpu"

        def cuda(self):
            calls["cuda"] = True
            return self

        def eval(self):
            calls["eval"] = True
            return self

    fake_processor = SimpleNamespace(
        image_processor=SimpleNamespace(max_pixels=None),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )

    def fake_create_model_and_processor(**kwargs):
        calls["loader_kwargs"] = kwargs
        return FakeModel(), fake_processor

    monkeypatch.setattr("eval.create_model_and_processor", fake_create_model_and_processor)

    SingleTurnNavigator(
        model_path="baseline",
        image_size=(2, 2),
    )

    assert calls["loader_kwargs"]["model_name"] == "baseline"
    assert set(calls["loader_kwargs"]) == {
        "model_name",
        "torch_dtype",
        "attn_implementation",
        "gradient_checkpointing",
    }


def test_navigator_history_prompt_uses_uniform_sample_indices():
    images = []
    for value in range(5):
        image = np.full((2, 2, 3), value, dtype=np.uint8)
        images.append(image)

    navigator = object.__new__(SingleTurnNavigator)
    navigator.image_size = (2, 2)
    navigator.rgb_list = []
    navigator.max_action_history = 10
    navigator.history_images = 2
    navigator.last_history_indices = []

    for rgb in images[:4]:
        navigator._append_observation(rgb)
    navigator._append_observation(images[4])

    history = navigator._history_for_prompt()

    assert len(history) == 2
    assert navigator.last_history_indices == [0, 3]
    assert list(history[0].getdata())[0] == (0, 0, 0)
    assert list(history[1].getdata())[0] == (3, 3, 3)


def test_eval_path_has_no_extra_visual_preprocessing_branch():
    source = (Path(__file__).resolve().parents[1] / "eval.py").read_text().lower()
    forbidden_terms = [
        "clip",
        "dino",
        "hashlib",
        "rgb_diff",
        "motion",
        "bev",
        "preprocess",
        "selector",
        "image_feature",
        "vision_feature",
        "frame_score",
        "similarity",
        "get_image_features",
    ]

    assert [term for term in forbidden_terms if term in source] == []
