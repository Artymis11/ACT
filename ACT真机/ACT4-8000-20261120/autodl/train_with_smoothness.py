"""Cloud ACT training with a padding-aware action acceleration penalty.

The installed ACT forward still computes L1 and KL. A public module hook reuses
that same forward's predicted actions, so the penalty adds no second model pass.
Validation retains the original ACT loss. Each checkpoint records the objective.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys


METADATA_FILENAME = "training_regularization.json"

# DataLoader spawn imports the entry module before it starts reading videos.
if os.environ.get("ACT_VIDEO_CACHE_MANIFEST"):
    import train_with_video_cache


def masked_acceleration_l1(actions, action_is_pad):
    """Mean absolute second difference over valid consecutive action triples."""
    import torch

    if actions.ndim != 3 or action_is_pad.shape != actions.shape[:2]:
        raise ValueError("Expected actions [B,T,D] and action_is_pad [B,T].")
    if action_is_pad.dtype != torch.bool:
        raise ValueError("action_is_pad must be boolean.")
    if actions.shape[1] < 3:
        return actions.sum() * 0.0
    acceleration = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    valid = ~(action_is_pad[:, 2:] | action_is_pad[:, 1:-1] | action_is_pad[:, :-2])
    denominator = (valid.sum() * actions.shape[-1]).clamp_min(1)
    return (acceleration.abs() * valid.unsqueeze(-1)).sum() / denominator


def install_smoothness_loss(policy, weight):
    """Wrap this policy's forward without replacing parameters or state keys."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("The smoothness weight must be finite and nonnegative.")
    if policy.config.type != "act":
        raise ValueError("This objective requires ACT.")
    if hasattr(policy, "_act_smoothness_cleanup"):
        raise ValueError("The ACT smoothness objective is already installed.")
    original_forward = policy.forward
    original_keys = tuple(policy.state_dict())
    collecting = False
    prediction = None

    def capture_prediction(module, inputs, output):
        nonlocal prediction
        if collecting:
            prediction = output[0]

    handle = policy.model.register_forward_hook(capture_prediction)

    def forward(batch, *args, **kwargs):
        nonlocal collecting, prediction
        if not policy.training or weight == 0:
            return original_forward(batch, *args, **kwargs)
        collecting = True
        prediction = None
        try:
            loss, metrics = original_forward(batch, *args, **kwargs)
            if prediction is None:
                raise RuntimeError("The ACT forward did not expose predicted actions.")
            acceleration_loss = masked_acceleration_l1(prediction, batch["action_is_pad"])
            metrics = dict(metrics)
            metrics["acceleration_loss"] = float(acceleration_loss.detach())
            metrics["weighted_acceleration_loss"] = float((weight * acceleration_loss).detach())
            return loss + weight * acceleration_loss, metrics
        finally:
            collecting = False
            prediction = None

    def cleanup():
        handle.remove()
        policy.forward = original_forward
        delattr(policy, "_act_smoothness_cleanup")

    policy.forward = forward
    policy._act_smoothness_cleanup = cleanup
    if tuple(policy.state_dict()) != original_keys:
        cleanup()
        raise RuntimeError("Installing the objective changed checkpoint state_dict keys.")
    return cleanup


def write_metadata(path, metadata):
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != metadata:
        raise ValueError(f"Existing regularization metadata differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def self_test():
    import torch
    from types import SimpleNamespace

    ramp = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    padding = torch.zeros((1, 6), dtype=torch.bool)
    assert masked_acceleration_l1(ramp, padding).item() == 0
    values = torch.tensor([[[0.0], [1.0], [4.0], [1000.0], [-1000.0]]], requires_grad=True)
    mask = torch.tensor([[False, False, False, True, True]])
    loss = masked_acceleration_l1(values, mask)
    assert loss.item() == 2
    loss.backward()
    assert torch.equal(values.grad[0, :3, 0], torch.tensor([1.0, -2.0, 1.0]))
    assert torch.equal(values.grad[0, 3:, 0], torch.zeros(2))
    assert masked_acceleration_l1(values, torch.ones_like(mask)).item() == 0
    assert masked_acceleration_l1(values[:, :2], mask[:, :2]).item() == 0

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actions = torch.nn.Parameter(torch.tensor([[[0.0], [1.0], [4.0]]]))
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            return self.actions, (None, None)

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(type="act")
            self.model = FakeModel()

        def forward(self, batch):
            predicted, _ = self.model(batch)
            base = (predicted - batch["action"]).abs().mean()
            return base, {"l1_loss": float(base.detach())}

    policy = FakePolicy()
    batch = {"action": torch.zeros((1, 3, 1)), "action_is_pad": torch.zeros((1, 3), dtype=torch.bool)}
    baseline, _ = policy(batch)
    keys = tuple(policy.state_dict())
    restore = install_smoothness_loss(policy, 0.05)
    before_calls = policy.model.calls
    trained, metrics = policy(batch)
    assert policy.model.calls == before_calls + 1
    assert torch.allclose(trained, baseline + 0.1)
    assert metrics["acceleration_loss"] == 2
    trained.backward()
    assert torch.isfinite(policy.model.actions.grad).all()
    policy.eval()
    evaluated, evaluation_metrics = policy(batch)
    assert torch.equal(evaluated, baseline) and "acceleration_loss" not in evaluation_metrics
    assert tuple(policy.state_dict()) == keys
    restore()
    restored, _ = policy(batch)
    assert torch.equal(restored, baseline)
    print(json.dumps({"status": "passed", "scope": "Synthetic tensors and fake policy only",
                      "real_model_forward_calls": 0, "optimizer_steps": 0,
                      "checks": ["linear motion", "padding triples", "gradient and padded gradient",
                                 "short chunk", "fully padded", "one forward only", "original eval loss",
                                 "state_dict keys", "cleanup"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--compiled-video-cache", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args, training_args = parser.parse_known_args()
    if args.self_test:
        self_test()
        return
    if not math.isfinite(args.smoothness_weight) or args.smoothness_weight < 0:
        raise ValueError("smoothness-weight must be finite and nonnegative.")
    import torch
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Actual ACT training runs only on the AutoDL Linux CUDA runtime.")
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise ValueError("This wrapper targets the deployed LeRobot 0.6.1.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This ACT2 objective wrapper supports the configured single-GPU run.")
    from lerobot.scripts import lerobot_train
    metadata = {"schema_version": 1, "acceleration_weight": args.smoothness_weight,
                "formula": "L1 + kl_weight * KL + acceleration_weight * valid_mean(abs(a[t+2]-2*a[t+1]+a[t]))",
                "coordinates": "Saved processor normalized action coordinates, all seven channels",
                "action_fps": 30, "difference_order": 2, "padding": "All three actions must be valid",
                "applies_to": "Training forward only. Original evaluation L1 is preserved."}
    original_factory = lerobot_train.make_policy
    original_datasets = lerobot_train.make_train_eval_datasets
    original_save = lerobot_train.save_checkpoint
    cleanups = []

    def make_datasets(cfg, *positional, **kwargs):
        if cfg.policy.type != "act" or cfg.policy.chunk_size != 100:
            raise ValueError("Expected the ACT2 100-action model.")
        derivation = Path(cfg.dataset.root) / "action_smoothing_manifest.json"
        metadata["action_label_derivation"] = None
        if derivation.exists():
            with derivation.open("rb") as stream:
                metadata["action_label_derivation"] = {
                    "manifest": derivation.name,
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                }
        if cfg.resume:
            saved = Path(cfg.checkpoint_path) / "pretrained_model" / METADATA_FILENAME
            if not saved.exists() or json.loads(saved.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Resume must retain the checkpoint's recorded regularization objective.")
        result = original_datasets(cfg, *positional, **kwargs)
        if result[0].meta.fps != 30:
            raise ValueError("This objective's action timebase requires the recorded 30 FPS dataset.")
        write_metadata(Path(cfg.output_dir) / METADATA_FILENAME, metadata)
        return result

    def make_policy(*positional, **kwargs):
        policy = original_factory(*positional, **kwargs)
        cleanups.append(install_smoothness_loss(policy, args.smoothness_weight))
        return policy

    def save_checkpoint(*positional, **kwargs):
        result = original_save(*positional, **kwargs)
        write_metadata(Path(kwargs["checkpoint_dir"]) / "pretrained_model" / METADATA_FILENAME, metadata)
        return result

    lerobot_train.make_policy = make_policy
    lerobot_train.make_train_eval_datasets = make_datasets
    lerobot_train.save_checkpoint = save_checkpoint
    previous_argv = sys.argv
    sys.argv = [sys.argv[0], *training_args]
    try:
        if args.compiled_video_cache:
            from train_with_compiled_video_cache import main as train_main
        elif os.environ.get("ACT_VIDEO_CACHE_MANIFEST"):
            from train_with_video_cache import main as train_main
        else:
            from train_with_image_contract import main as train_main
        train_main()
    finally:
        sys.argv = previous_argv
        lerobot_train.make_policy = original_factory
        lerobot_train.make_train_eval_datasets = original_datasets
        lerobot_train.save_checkpoint = original_save
        for cleanup in cleanups:
            cleanup()


if __name__ == "__main__":
    main()
