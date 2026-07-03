import torch

from train import (
    DepthAuxTrainerMixin,
    DepthPlateauStopCallback,
    distributed_mean_scalar,
    extract_visual_feature_maps,
)
from vln_baseline.depth_head import DepthReadoutHead
from vln_baseline.losses import SILogLoss


def test_depth_readout_head_predicts_positive_depth_map():
    head = DepthReadoutHead(input_dim=4, hidden_dim=3, max_depth=10.0)
    features = torch.randn(2, 4, 3, 2)

    depth = head(features)

    assert depth.shape == (2, 1, 3, 2)
    assert torch.all(depth > 0)
    assert torch.all(depth <= 10.0 + head.eps)


def test_silog_loss_is_near_zero_for_identical_depths():
    loss_fn = SILogLoss()
    depth = torch.full((2, 1, 4, 4), 2.0)
    mask = torch.ones_like(depth, dtype=torch.bool)

    loss = loss_fn(depth, depth, mask)

    assert loss.item() < 1e-3


def test_silog_loss_promotes_bfloat16_inputs_to_float32():
    loss_fn = SILogLoss()
    pred = torch.full((1, 1, 4, 4), 2.0, dtype=torch.bfloat16)
    target = torch.full((1, 1, 4, 4), 1.5, dtype=torch.bfloat16)
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = loss_fn(pred, target, mask)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_extract_visual_feature_maps_uses_flat_spans_and_grid_order():
    hidden = torch.arange(2 * 8 * 3, dtype=torch.float32).view(2, 8, 3)
    spans = torch.tensor([[0, 1, 5], [1, 2, 6]])
    image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]])

    features = extract_visual_feature_maps(
        hidden,
        image_token_spans=spans,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
    )

    assert features.shape == (2, 3, 2, 2)
    assert torch.equal(features[0], hidden[0, 1:5].view(2, 2, 3).permute(2, 0, 1))
    assert torch.equal(features[1], hidden[1, 2:6].view(2, 2, 3).permute(2, 0, 1))


class _ToyLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _ToyQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([_ToyLayer()])
        self.depth_head = DepthReadoutHead(input_dim=2, hidden_dim=3)

    def forward(self, input_ids, image_grid_thw, labels=None):
        hidden = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
        self.model.language_model.layers[0](hidden)
        return type("Output", (), {"loss": torch.tensor(999.0)})()


class _ToyQwenTwoImages(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([_ToyLayer()])
        self.depth_head = DepthReadoutHead(input_dim=2, hidden_dim=3)

    def forward(self, input_ids, image_grid_thw, labels=None):
        hidden = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
        self.model.language_model.layers[0](hidden)
        return type("Output", (), {"loss": torch.tensor(999.0)})()


class _RecordingDepthLoss:
    def __init__(self):
        self.pred = None
        self.target = None
        self.mask = None

    def __call__(self, pred, target, mask):
        self.pred = pred.detach().clone()
        self.target = target.detach().clone()
        self.mask = mask.detach().clone()
        return pred.float().sum() * 0.0 + target.float().sum() * 0.0 + 1.0


class _ToyDepthTrainer(DepthAuxTrainerMixin):
    def __init__(self):
        self.depth_enable = True
        self.depth_stage = 1
        self.depth_weight = 0.5
        self.depth_warmup_steps = 0
        self.depth_tap_layer = 0
        self.depth_spatial_merge_size = 1
        self.depth_supervise_frames = "all"
        self.depth_loss_fn = SILogLoss()
        self.state = type("State", (), {"global_step": 0})()
        self.logged = []

    def log(self, metrics):
        self.logged.append(metrics)


def test_stage1_compute_loss_uses_depth_only_not_action_loss():
    model = _ToyQwen()
    trainer = _ToyDepthTrainer()
    inputs = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "image_token_spans": torch.tensor([[0, 0, 4]]),
        "image_batch_indices": torch.tensor([0]),
        "current_image_indices": torch.tensor([0]),
        "depth_gt": torch.full((1, 1, 2, 2), 2.0),
        "depth_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
    }

    loss = trainer.compute_loss(model, inputs)

    assert torch.isfinite(loss)
    assert loss.item() < 999.0
    assert "loss_action" not in trainer.logged[-1]
    assert "loss_depth" in trainer.logged[-1]


def test_compute_loss_can_supervise_current_frame_only():
    model = _ToyQwenTwoImages()
    trainer = _ToyDepthTrainer()
    trainer.depth_supervise_frames = "current"
    recording_loss = _RecordingDepthLoss()
    trainer.depth_loss_fn = recording_loss
    inputs = {
        "input_ids": torch.ones(1, 8, dtype=torch.long),
        "labels": torch.ones(1, 8, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        "image_token_spans": torch.tensor([[0, 0, 4], [0, 4, 8]]),
        "image_batch_indices": torch.tensor([0, 0]),
        "current_image_indices": torch.tensor([1]),
        "depth_gt": torch.stack(
            [
                torch.full((1, 2, 2), 1.0),
                torch.full((1, 2, 2), 3.0),
            ],
            dim=0,
        ),
        "depth_mask": torch.ones(2, 1, 2, 2, dtype=torch.bool),
    }

    loss = trainer.compute_loss(model, inputs)

    assert torch.isfinite(loss)
    assert recording_loss.target.shape == (1, 1, 2, 2)
    assert torch.equal(recording_loss.target, torch.full((1, 1, 2, 2), 3.0))
    assert recording_loss.pred.shape[0] == 1


def test_depth_weight_plateau_cosine_decay_holds_then_decays_to_final_weight():
    trainer = _ToyDepthTrainer()
    trainer.depth_weight = 0.5
    trainer.depth_weight_schedule = "plateau_cosine_decay"
    trainer.depth_weight_final = 0.1
    trainer.depth_weight_hold_ratio = 0.3
    trainer.depth_warmup_steps = 0
    trainer.state = type("State", (), {"global_step": 0, "max_steps": 100})()

    trainer.state.global_step = 0
    assert trainer._depth_weight_for_step() == 0.5

    trainer.state.global_step = 29
    assert trainer._depth_weight_for_step() == 0.5

    trainer.state.global_step = 30
    weight_after_hold = trainer._depth_weight_for_step()
    assert 0.1 < weight_after_hold < 0.5

    trainer.state.global_step = 99
    assert trainer._depth_weight_for_step() == 0.1


def test_depth_plateau_callback_has_transformers_callback_events():
    callback = DepthPlateauStopCallback()

    assert hasattr(callback, "on_init_end")
    assert hasattr(callback, "on_log")


def test_depth_plateau_callback_counts_patience_only_after_min_steps():
    callback = DepthPlateauStopCallback(ema_beta=0.0, patience=2, min_delta=0.01, min_steps=500)
    control = type("Control", (), {"should_training_stop": False})()
    state = type("State", (), {"global_step": 100})()

    callback.on_log(None, state, control, logs={"loss_depth": 1.0})
    callback.on_log(None, state, control, logs={"loss_depth": 1.0})

    assert callback.bad_logs == 0
    assert not control.should_training_stop

    state.global_step = 500
    callback.on_log(None, state, control, logs={"loss_depth": 1.0})
    assert callback.bad_logs == 1
    assert not control.should_training_stop

    callback.on_log(None, state, control, logs={"loss_depth": 1.0})
    assert callback.bad_logs == 2
    assert control.should_training_stop


def test_distributed_mean_scalar_returns_local_value_without_process_group():
    value = torch.tensor(3.25)

    assert distributed_mean_scalar(value) == 3.25
