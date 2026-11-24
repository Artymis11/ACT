"""Run LeRobot's normal training CLI and persist local image preprocessing.

Every 224 checkpoint includes the same contract as its dataset and initialization
model. The hooks run after LeRobot has resolved resume paths and before publishing
the last-checkpoint link. Legacy datasets without this contract use normal LeRobot.
"""

import json
from pathlib import Path

from image_preprocessing import CONTRACT_FILENAME, load_contract, validate_model_contract


def contract_for_training(cfg):
    root = Path(cfg.dataset.root) if cfg.dataset.root else None
    if root is None or not (root / CONTRACT_FILENAME).exists():
        if cfg.policy and any(
            feature.type.value == "VISUAL" and list(feature.shape) == [3, 224, 224]
            for feature in cfg.policy.input_features.values()
        ):
            raise ValueError("The 224 training dataset is missing its image preprocessing contract.")
        return None
    contract = load_contract(root)
    if cfg.policy is None or cfg.policy.type != "act" or not cfg.policy.pretrained_path:
        raise ValueError("The image contract requires ACT with a local initialization checkpoint.")
    if validate_model_contract(cfg.policy.pretrained_path) != contract:
        raise ValueError("Training model and dataset preprocessing differ.")
    if cfg.dataset.image_transforms.enable:
        raise ValueError("This image contract requires dataset.image_transforms.enable=false.")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    model_config = json.loads(
        (Path(cfg.policy.pretrained_path) / "config.json").read_text(encoding="utf-8")
    )
    dataset_cameras = {
        key for key, feature in info["features"].items()
        if feature.get("dtype") in {"image", "video"}
    }
    configured_cameras = {
        key for key, feature in cfg.policy.input_features.items()
        if feature.type.value == "VISUAL"
    }
    model_cameras = {
        key for key, feature in model_config["input_features"].items()
        if feature["type"] == "VISUAL"
    }
    if not configured_cameras or configured_cameras != dataset_cameras or configured_cameras != model_cameras:
        raise ValueError("Training dataset, policy config and initialization checkpoint must declare the same cameras.")
    for key, feature in cfg.policy.input_features.items():
        if feature.type.value == "VISUAL":
            if list(feature.shape) != [3, 224, 224] or info["features"][key]["shape"] != [224, 224, 3]:
                raise ValueError(f"Training image shape differs from its contract: {key}")
    return contract


def save_contract(checkpoint_dir, contract):
    if contract is None:
        return
    destination = Path(checkpoint_dir) / "pretrained_model" / CONTRACT_FILENAME
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    validate_model_contract(destination.parent)


def main():
    from lerobot.scripts import lerobot_train

    original_make_datasets = lerobot_train.make_train_eval_datasets
    original_save_checkpoint = lerobot_train.save_checkpoint

    def make_datasets(cfg, *args, **kwargs):
        contract_for_training(cfg)
        return original_make_datasets(cfg, *args, **kwargs)

    def save_checkpoint(*args, **kwargs):
        contract = contract_for_training(kwargs["cfg"])
        result = original_save_checkpoint(*args, **kwargs)
        save_contract(kwargs["checkpoint_dir"], contract)
        return result

    lerobot_train.make_train_eval_datasets = make_datasets
    lerobot_train.save_checkpoint = save_checkpoint
    try:
        lerobot_train.main()
    finally:
        lerobot_train.make_train_eval_datasets = original_make_datasets
        lerobot_train.save_checkpoint = original_save_checkpoint


if __name__ == "__main__":
    main()
