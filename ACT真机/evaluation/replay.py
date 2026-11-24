"""Repeat the archived two-camera RPC input locally without robot or network I/O."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evaluation/rpc_fixture"
sys.path[:0] = [str(ROOT / "autodl"), str(ROOT / "inference")]


def replay(model, device, tolerance):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import numpy as np
    from PIL import Image
    import torch
    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from compare_rpc_replay import compare_actions
    from image_policy_server import observation_without_resize
    from image_preprocessing import contract_digest, validate_model_contract
    from verify_model import verify_model

    if importlib.metadata.version("lerobot") != "0.6.1":
        raise ValueError("Use the cloud model environment's LeRobot 0.6.1.")
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    checkpoint = verify_model(model)
    fixture = json.loads((FIXTURE / "reference.json").read_text(encoding="utf-8"))
    if (fixture["checkpoint_weights_sha256"] != checkpoint["source_weights_sha256"]
            or fixture["image_contract_sha256"] != contract_digest(validate_model_contract(model))):
        raise ValueError("Fixture and checkpoint refer to different contracts.")

    state = fixture["state"]
    raw = dict(zip(fixture["joint_names"], state, strict=True))
    raw["task"] = fixture["task"]
    for camera in ("front", "gripper"):
        with Image.open(FIXTURE / f"{camera}.png") as image:
            pixels = np.array(image.convert("RGB"), dtype=np.uint8)
        if hashlib.sha256(pixels.tobytes()).hexdigest() != fixture["camera_rgb_sha256"][
                f"observation.images.{camera}"]:
            raise ValueError(f"RGB pixels differ from the recorded payload: {camera}")
        raw[camera] = pixels

    config = ACTConfig.from_pretrained(model, local_files_only=True)
    config.device = device
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(model, config=config, local_files_only=True, strict=True).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(model),
        preprocessor_overrides={"device_processor": {"device": device},
                                "rename_observations_processor": {"rename_map": {}}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    observation = TimedObservation(0.0, fixture["dataset_index"], raw, must_go=True)
    batch = observation_without_resize(observation, fixture["observation_features"])
    count, axes = fixture["actions_shape"]
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(preprocessor(batch))[:, :count, :]
        actual = torch.stack([postprocessor(normalized[:, step, :]) for step in range(count)], dim=1)
    result = compare_actions(actual.squeeze(0).cpu().numpy(), fixture["actions"], tolerance)
    if result["shape"] != [count, axes]:
        raise ValueError("Action shape differs from the archived response.")
    print(json.dumps({"device": device, "input_rgb_exact": True, **result}, indent=2))
    return 0 if result["within_absolute_tolerance"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--atol", type=float, default=2e-3)
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative.")
    raise SystemExit(replay(args.model.resolve(strict=True), args.device, args.atol))
