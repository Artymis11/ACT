"""Compare saved ACT RPC actions with same-frame direct inference on AutoDL.

Only --self-test is intended for local execution. Real comparison requires
Linux and CUDA, uses no robot or network, and leaves checkpoint files intact.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_actions(actual, expected, atol):
    actual, expected = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 7:
        raise ValueError("RPC and direct action shapes must match and have seven channels.")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Actions contain non-finite values.")
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    maximum = np.unravel_index(int(absolute.argmax()), absolute.shape)
    return {
        "shape": list(actual.shape), "exact_equal": bool(np.array_equal(actual, expected)),
        "within_absolute_tolerance": bool(np.all(absolute <= atol)),
        "absolute_tolerance": atol, "maximum_absolute_difference": float(absolute.max()),
        "mean_absolute_difference": float(absolute.mean()),
        "maximum_absolute_difference_per_channel": absolute.max(axis=0).tolist(),
        "differing_elements_exact": int(np.count_nonzero(actual != expected)),
        "elements_exceeding_tolerance": int(np.count_nonzero(absolute > atol)),
        "maximum_difference_location": {"action_offset": int(maximum[0]), "channel": int(maximum[1])},
    }


def require_no_action_labels(values):
    # The saved processor's transition converter includes action=None for observations.
    if values.get("action") is not None:
        raise ValueError("Action labels must not enter the model comparison.")


def compare(args, report):
    if sys.platform != "linux":
        raise RuntimeError("Actual model comparison must run on AutoDL Linux with CUDA.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the cloud-only model comparison.")
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise ValueError("This comparison is pinned to the deployed LeRobot0.6.1 server.")
    from image_preprocessing import contract_digest, preprocess_rgb, validate_model_contract
    from image_policy_server import observation_without_resize
    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    replay = json.loads(args.replay_report.read_text(encoding="utf-8-sig"))
    if replay.get("status") != "passed" or replay.get("mode") == "wire_check_only":
        raise ValueError("Use a passed real RPC replay report containing cloud actions.")
    model = args.model_path.resolve(strict=True)
    if model != Path(replay["remote_model_path"]).resolve():
        raise ValueError("Use the exact serving model path used in the RPC handshake.")
    count = replay["actions_per_chunk"]
    if not isinstance(count, int) or not 1 <= count <= 100:
        raise ValueError("Invalid recorded actions_per_chunk.")
    if not replay["frames"] or [frame["dataset_index"] for frame in replay["frames"]] != replay["frame_indices"]:
        raise ValueError("Replay frame indices differ from the saved frame records.")
    contract = validate_model_contract(model)
    if contract_digest(contract) != replay["contract_sha256"]:
        raise ValueError("RPC and direct preprocessing contracts differ.")
    before = {path.name: sha256_file(path) for path in sorted(model.iterdir()) if path.is_file()}
    serving_manifest_path = model / ".act_serving_manifest.json"
    source_hashes = before
    if serving_manifest_path.exists():
        provenance = json.loads(serving_manifest_path.read_text(encoding="utf-8"))
        source_hashes = {entry["path"]: entry["sha256"] for entry in provenance["source_files"]}
        expected_serving = {entry["path"]: entry["sha256"] for entry in provenance["serving_files"]}
        if any(before.get(name) != digest for name, digest in expected_serving.items()):
            raise ValueError("Serving model files differ from their audited provenance.")
    if replay["local_config_sha256"] != source_hashes["config.json"]:
        raise ValueError("Client metadata config differs from the server's source checkpoint.")
    for name, digest in replay.get("local_model_files_sha256", {}).items():
        if source_hashes.get(name) != digest:
            raise ValueError(f"Client checkpoint backup differs from server source file: {name}")

    info = json.loads((args.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    joints = info["features"]["observation.state"]["names"]
    cameras = sorted(key for key in info["features"] if key.startswith("observation.images."))
    if joints != replay["joint_names"] or cameras != sorted(replay["cameras"]):
        raise ValueError("Dataset channels or cameras differ from the replay.")
    features = {"observation.state": info["features"]["observation.state"]}
    features.update({key: {"dtype": "image", "shape": [224, 224, 3],
                           "names": ["height", "width", "channels"]} for key in cameras})
    dataset = LeRobotDataset("local/act2_compare", root=args.dataset_root,
                            video_backend="pyav", return_uint8=True)
    policy = ACTPolicy.from_pretrained(model, local_files_only=True, strict=True).to("cuda").eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=str(model),
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}},
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    report.update(
        replay_report=str(args.replay_report), replay_report_sha256=sha256_file(args.replay_report),
        model_path=str(model), model_files_sha256=before, dataset_root=str(args.dataset_root),
        actions_per_chunk=count, joint_names=joints, cameras=cameras,
        client_weight_hash_present="model.safetensors" in replay.get("local_model_files_sha256", {}),
        gpu=torch.cuda.get_device_name(0),
        versions={name: importlib.metadata.version(name) for name in ("torch", "lerobot", "pillow", "av")},
        numeric_settings={"float32_matmul_precision": torch.get_float32_matmul_precision(),
                          "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        mode="FP32 saved checkpoint, policy.eval zero latent, same server input conversion and processors, no autocast",
    )
    with torch.inference_mode():
        for frame in replay["frames"]:
            index = frame["dataset_index"]
            if frame.get("actions_shape") != [count, 7] or not frame.get("actions_finite"):
                raise ValueError(f"Missing finite {count}-by-7 RPC actions for frame {index}.")
            recorded = dataset[index]
            state = recorded["observation.state"].cpu().numpy()
            if not np.array_equal(state, np.asarray(frame["state"], dtype=np.float32)):
                raise ValueError(f"Dataset and transmitted state differ at frame {index}.")
            raw = {name: float(state[channel]) for channel, name in enumerate(joints)}
            hashes = {}
            for key in cameras:
                image = recorded[key].permute(1, 2, 0).cpu().numpy()
                raw[key.removeprefix("observation.images.")] = preprocess_rgb(image)
                hashes[key] = hashlib.sha256(raw[key.removeprefix("observation.images.")].tobytes()).hexdigest()
            if hashes != frame["image_rgb_uint8_sha256"]:
                raise ValueError(f"Dataset and transmitted RGB pixels differ at frame {index}.")
            raw["task"] = recorded["task"]
            if raw["task"] != frame["task"]:
                raise ValueError(f"Dataset and transmitted task differ at frame {index}.")
            require_no_action_labels(raw)
            timed = TimedObservation(frame["observation_timestamp"], index, raw, must_go=True)
            observations = observation_without_resize(timed, features)
            require_no_action_labels(observations)
            batch = preprocess(observations)
            require_no_action_labels(batch)
            normalized = policy.predict_action_chunk(batch)[:, :count, :]
            # Match ImagePolicyServer's per-action unnormalization and device placement.
            processed = [postprocess(normalized[:, offset, :]) for offset in range(count)]
            direct = torch.stack(processed, dim=1).squeeze(0).detach().cpu().numpy()
            metrics = compare_actions(frame["actions"], direct, args.atol)
            report["frames"].append({"dataset_index": index, "state_exact_equal": True,
                                     "rgb_byte_hashes_equal": True, "image_rgb_uint8_sha256": hashes,
                                     **metrics, "direct_actions": direct.tolist()})
    if {path.name: sha256_file(path) for path in model.iterdir() if path.is_file()} != before:
        raise ValueError("Checkpoint files changed during comparison.")
    report["checkpoint_files_unchanged"] = True
    report["all_actions_exact_equal"] = all(frame["exact_equal"] for frame in report["frames"])
    report["status"] = "passed" if all(frame["within_absolute_tolerance"] for frame in report["frames"]) else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=0.0,
                        help="Default requires exact float32 equality. Any relaxed tolerance remains explicit.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        values = np.arange(350, dtype=np.float32).reshape(50, 7)
        assert compare_actions(values.tolist(), values, 0.0)["exact_equal"]
        changed = values.copy()
        changed[3, 6] += 0.5
        metrics = compare_actions(values, changed, 0.0)
        assert metrics["differing_elements_exact"] == 1 and metrics["maximum_difference_location"] == {"action_offset": 3, "channel": 6}
        assert not metrics["within_absolute_tolerance"]
        require_no_action_labels({"observation.state": [0] * 7})
        require_no_action_labels({"observation.state": [0] * 7, "action": None})
        try:
            require_no_action_labels({"action": np.zeros((1, 50, 7))})
        except ValueError:
            pass
        else:
            raise AssertionError("Non-None action labels were accepted.")
        print("Pure numpy 50-by-7 comparison self-test passed. Model forward calls=0.")
        return 0
    if not all((args.model_path, args.dataset_root, args.replay_report, args.output)):
        parser.error("model-path, dataset-root, replay-report and output are required.")
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error("atol must be finite and nonnegative.")
    if args.output.exists():
        raise FileExistsError(f"Preserve the existing report and choose a new output: {args.output}")
    report = {"status": "failed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Recorded-frame cloud CUDA direct-versus-RPC comparison; no robot or network",
              "hardware_commands_sent": 0, "frames": []}
    try:
        compare(args, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
