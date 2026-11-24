"""Train a new ACT4 policy with explicitly selected visual initialization."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys


AUTODL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUTODL_ROOT))
METADATA_FILENAME = "random_initialization.json"
CAMERAS = {"observation.images.front", "observation.images.gripper"}
IMAGENET_BACKBONE = "ResNet18_Weights.IMAGENET1K_V1"
IMAGENET_FILENAME = "resnet18-f37072fd.pth"
IMAGENET_URL = "https://download.pytorch.org/models/" + IMAGENET_FILENAME
JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_yaw.pos", "wrist_roll.pos", "gripper.pos"]

# Spawned DataLoader workers execute the entry module before decoding videos.
if os.environ.get("ACT_VIDEO_CACHE_MANIFEST"):
    import train_with_video_cache


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_split(split, expected_episodes=60):
    train, validation = split["train_episode_indices"], split["eval_episode_indices"]
    if (len(train) != 54 or len(validation) != 6
            or any(type(value) is not int for value in [*train, *validation])
            or len(set(train)) != len(train) or len(set(validation)) != len(validation)
            or set(train) & set(validation)
            or set(train) | set(validation) != set(range(expected_episodes))):
        raise ValueError("ACT4 requires an explicit disjoint 54/6 split covering all 60 episodes.")
    return train, validation


def vision_backbone_metadata():
    """Verify the locally provisioned official checkpoint before any model factory."""
    import torch

    path = Path(torch.hub.get_dir()) / "checkpoints" / IMAGENET_FILENAME
    if not path.is_file():
        raise ValueError(f"Provision the official ResNet18 file in the local Torch Hub cache: {path}")
    digest = file_hash(path)
    if not digest.startswith("f37072fd"):
        raise ValueError("The cached ResNet18 file differs from its official published hash prefix.")
    return {"enum": IMAGENET_BACKBONE, "url": IMAGENET_URL,
            "cache_filename": IMAGENET_FILENAME, "sha256": digest}


def validate_vision_initialization(cfg, vision_init):
    if vision_init not in {"random", "imagenet"}:
        raise ValueError("Choose random or imagenet vision initialization.")
    expected = IMAGENET_BACKBONE if vision_init == "imagenet" else None
    if cfg.policy.pretrained_backbone_weights != expected or cfg.policy.use_peft:
        raise ValueError("Backbone configuration differs from the explicit vision initialization.")
    if bool(cfg.dataset.use_imagenet_stats) != (vision_init == "imagenet"):
        raise ValueError("Visual normalization must match the selected vision initialization.")
    if vision_init == "imagenet":
        vision_backbone_metadata()


def scratch_contract_for_training(cfg, vision_init="random"):
    """Allow fresh ACT construction and the same run's verified resume checkpoints."""
    from image_preprocessing import load_contract, validate_model_contract

    root = Path(cfg.dataset.root)
    contract = load_contract(root)
    policy = cfg.policy
    if (policy is None or policy.type != "act" or policy.chunk_size != 100
            or policy.vision_backbone != "resnet18"):
        raise ValueError("ACT4 requires an ACT policy with 100 predicted actions.")
    validate_vision_initialization(cfg, vision_init)
    if cfg.resume:
        checkpoint = Path(policy.pretrained_path)
        metadata = read_json(checkpoint / METADATA_FILENAME)
        expected = ("new_act_with_imagenet_resnet18" if vision_init == "imagenet"
                    else "random_act_and_resnet18")
        if metadata.get("initialization") != expected:
            raise ValueError("Resume requires this scratch run's recorded initialization.")
        if validate_model_contract(checkpoint) != contract:
            raise ValueError("Resume checkpoint and dataset image contracts differ.")
    elif policy.pretrained_path is not None:
        raise ValueError("A new ACT4 run cannot load an initialization checkpoint.")
    if cfg.dataset.image_transforms.enable:
        raise ValueError("Use the fixed local image preprocessing contract.")
    info = read_json(root / "meta/info.json")
    if info["total_episodes"] != 60 or info["fps"] != 30:
        raise ValueError("Expected ACT4's 60 episodes at 30 FPS.")
    dataset_cameras = {key for key, value in info["features"].items()
                       if value.get("dtype") in {"image", "video"}}
    configured_cameras = {key for key, value in policy.input_features.items()
                          if value.type.value == "VISUAL"}
    if dataset_cameras != CAMERAS or configured_cameras != CAMERAS:
        raise ValueError("ACT4 requires the recorded front and gripper views.")
    for key in CAMERAS:
        if (list(policy.input_features[key].shape) != [3, 224, 224]
                or info["features"][key]["shape"] != [224, 224, 3]):
            raise ValueError(f"Invalid 224 RGB shape for {key}.")
    for key in ("observation.state", "action"):
        if (info["features"][key]["shape"] != [7]
                or info["features"][key]["names"] != JOINT_NAMES):
            raise ValueError(f"ACT4 joint channels differ for {key}.")
    return contract


def make_explicit_datasets(cfg, split, *, dataset_class=None, metadata_class=None):
    """Use the normal dataset implementation with the persisted episode split."""
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.utils.utils import unflatten_dict

    train, validation = validate_split(split)
    dataset_class = dataset_class or LeRobotDataset
    metadata_class = metadata_class or LeRobotDatasetMetadata
    root = Path(cfg.dataset.root)
    meta = metadata_class(cfg.dataset.repo_id, root=root, revision=cfg.dataset.revision)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, meta)
    rows = [row for path in sorted((root / "meta/episodes").rglob("*.parquet"))
            for row in pq.read_table(path).to_pylist()]
    by_episode = {row["episode_index"]: row for row in rows}
    if len(rows) != 60 or set(by_episode) != set(range(60)):
        raise ValueError("Episode metadata must contain each ACT4 episode exactly once.")
    selected_stats = [unflatten_dict({key.removeprefix("stats/"): np.asarray(value)
                                     for key, value in by_episode[episode].items()
                                     if key.startswith("stats/") and value is not None})
                      for episode in train]
    training_stats = aggregate_stats(selected_stats)
    train_frames = sum(by_episode[episode]["length"] for episode in train)
    validation_frames = sum(by_episode[episode]["length"] for episode in validation)
    if int(training_stats["action"]["count"][0]) != train_frames:
        raise ValueError("Training normalization statistics have the wrong frame count.")
    persisted_stats = read_json(root / "meta/stats.json")
    for feature in ("observation.state", "action", *sorted(CAMERAS)):
        for field in ("mean", "std", "count", "min", "max"):
            if not np.allclose(persisted_stats[feature][field], training_stats[feature][field],
                               atol=1e-7, rtol=1e-6):
                raise ValueError(f"Persisted statistics differ from the 54 training episodes: {feature}/{field}")
    datasets = []
    for episodes in (train, validation):
        dataset = dataset_class(cfg.dataset.repo_id, root=root, episodes=episodes,
                                delta_timestamps=delta_timestamps, image_transforms=None,
                                revision=cfg.dataset.revision, video_backend=cfg.dataset.video_backend,
                                return_uint8=True, tolerance_s=cfg.tolerance_s)
        dataset.meta.stats = deepcopy(training_stats)
        if cfg.dataset.use_imagenet_stats:
            import torch
            from lerobot.datasets.factory import IMAGENET_STATS
            for camera in dataset.meta.camera_keys:
                for field, values in IMAGENET_STATS.items():
                    dataset.meta.stats[camera][field] = torch.tensor(values, dtype=torch.float32)
        datasets.append(dataset)
    if datasets[0].num_frames != train_frames or datasets[1].num_frames != validation_frames:
        raise ValueError("Constructed dataset counts differ from the explicit episode split.")
    return tuple(datasets)


def initialization_metadata(cfg, split_path):
    from image_preprocessing import contract_digest

    metadata = {"schema_version": 1, "initialization": "random_act_and_resnet18",
            "external_act_checkpoint": None, "external_backbone_weights": None,
            "pretrained_processors": None, "initial_optimizer_state": None,
            "vision_backbone": cfg.policy.vision_backbone, "seed": cfg.seed,
            "image_contract_sha256": contract_digest(),
            "split_manifest_sha256": file_hash(split_path),
            "training_statistics_sha256": file_hash(Path(cfg.dataset.root) / "meta/stats.json")}
    if cfg.policy.pretrained_backbone_weights == IMAGENET_BACKBONE:
        metadata["initialization"] = "new_act_with_imagenet_resnet18"
        metadata["external_backbone_weights"] = vision_backbone_metadata()
        metadata["normalization"] = {
            "VISUAL": {"mode": "MEAN_STD", "source": "IMAGENET_STATS",
                       "mean": [.485, .456, .406], "std": [.229, .224, .225]},
            "STATE": {"source": "54 ACT4 training episodes"},
            "ACTION": {"source": "54 ACT4 training episodes"}}
    return metadata


def compile_training_forward(policy, batch_size):
    """Compile only the chosen full training batch, retaining eager evaluation."""
    import torch

    model = policy.model
    if policy.config.type != "act" or next(model.parameters()).device.type != "cuda":
        raise ValueError("The compiled scratch dispatcher requires a cloud CUDA ACT model.")
    original_keys = tuple(policy.state_dict())
    eager_forward = model.forward
    compiled_forward = torch.compile(eager_forward, mode="reduce-overhead", fullgraph=True)

    def dispatch(batch):
        if model.training and batch["observation.state"].shape[0] == batch_size:
            torch.compiler.cudagraph_mark_step_begin()
            return compiled_forward(batch)
        return eager_forward(batch)

    model.forward = dispatch
    if tuple(policy.state_dict()) != original_keys:
        raise RuntimeError("Compilation changed the original ACT checkpoint keys.")
    return eager_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--vision-init", choices=["random", "imagenet"], default="random")
    parser.add_argument("--compile-batch-size", type=int)
    args, training_args = parser.parse_known_args()
    split = read_json(args.split_manifest)
    validate_split(split)
    if args.compile_batch_size is not None and args.compile_batch_size <= 0:
        raise ValueError("compile-batch-size must be positive.")
    if "--compiled-video-cache" in training_args and args.compile_batch_size is None:
        raise ValueError("Pass --compile-batch-size matching the chosen ACT4 full training batch.")
    from lerobot.scripts import lerobot_train
    import train_with_image_contract as image_trainer
    import train_with_smoothness as objective

    original_datasets = lerobot_train.make_train_eval_datasets
    original_save = lerobot_train.save_checkpoint
    original_contract = image_trainer.contract_for_training
    previous_argv = sys.argv
    compiled_module = None
    original_compile = None

    def make_datasets(cfg, *positional, **kwargs):
        if args.compile_batch_size is not None and cfg.batch_size != args.compile_batch_size:
            raise ValueError("Configured training batch differs from the compiled batch size.")
        metadata = initialization_metadata(cfg, args.split_manifest)
        if cfg.resume:
            saved = read_json(Path(cfg.checkpoint_path) / "pretrained_model" / METADATA_FILENAME)
            if saved != metadata:
                raise ValueError("Resume changed scratch initialization, split or training statistics.")
        result = make_explicit_datasets(cfg, split)
        objective.write_metadata(Path(cfg.output_dir) / METADATA_FILENAME, metadata)
        return result

    def save_checkpoint(*positional, **kwargs):
        result = original_save(*positional, **kwargs)
        objective.write_metadata(Path(kwargs["checkpoint_dir"]) / "pretrained_model" / METADATA_FILENAME,
                                 initialization_metadata(kwargs["cfg"], args.split_manifest))
        return result

    lerobot_train.make_train_eval_datasets = make_datasets
    lerobot_train.save_checkpoint = save_checkpoint
    image_trainer.contract_for_training = lambda cfg: scratch_contract_for_training(cfg, args.vision_init)
    try:
        if args.compile_batch_size is not None:
            import train_with_compiled_video_cache as compiled_module
            original_compile = compiled_module.compile_training_forward
            compiled_module.compile_training_forward = lambda policy: compile_training_forward(
                policy, args.compile_batch_size)
            if "--compiled-video-cache" not in training_args:
                training_args.append("--compiled-video-cache")
        sys.argv = [sys.argv[0], *training_args]
        objective.main()
    finally:
        sys.argv = previous_argv
        image_trainer.contract_for_training = original_contract
        lerobot_train.make_train_eval_datasets = original_datasets
        lerobot_train.save_checkpoint = original_save
        if compiled_module is not None:
            compiled_module.compile_training_forward = original_compile


if __name__ == "__main__":
    main()
