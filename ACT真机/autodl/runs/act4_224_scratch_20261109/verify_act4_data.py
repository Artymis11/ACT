"""Check ACT4 label derivation and official LeRobot dataset entry points."""

import argparse
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import aggregate_stats
from smooth_action_dataset import episode_statistics, inventory, read_dataset, serializable_stats, write_json
from image_preprocessing import load_contract


RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[2]


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "local/data")
    args = parser.parse_args()
    raw_root = args.data_root / "act4_224"
    smooth_root = args.data_root / "act4_224_action_smoothed_v1"
    split = json.loads((RUN / "split_manifest.json").read_text(encoding="utf-8"))
    raw_info, _, _, raw, raw_rows = read_dataset(raw_root)
    info, _, _, data, rows = read_dataset(smooth_root)
    checks = {}
    checks["metadata_geometry_and_contract"] = (
        info == raw_info and load_contract(raw_root) == load_contract(smooth_root)
        and all(info["features"][key]["shape"] == [224, 224, 3]
                for key in ("observation.images.front", "observation.images.gripper")))
    checks["schema_and_non_action_columns"] = raw.schema.equals(data.schema, check_metadata=True) and all(
        raw[name].equals(data[name]) for name in raw.column_names if name != "action")
    original = np.asarray(raw["action"].to_pylist(), dtype=np.float32)
    filtered = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    reference = np.empty_like(original)
    for row in raw_rows:
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        for index in range(start, stop):
            reference[index] = original[max(start, index - 1):min(stop, index + 2)].mean(axis=0, dtype=np.float64)
    checks["independent_episode_bounded_mean_all_labels"] = np.array_equal(reference, filtered)
    checks["episode_metadata_only_action_statistics_changed"] = all(
        {key: value for key, value in before.items() if not key.startswith("stats/action/")}
        == {key: value for key, value in after.items() if not key.startswith("stats/action/")}
        for before, after in zip(raw_rows, rows))
    selected_frames = np.concatenate([np.arange(rows[index]["dataset_from_index"], rows[index]["dataset_to_index"])
                                      for index in split["train_episode_indices"]])
    stats = json.loads((smooth_root / "meta/stats.json").read_text(encoding="utf-8"))
    per_episode = episode_statistics(rows)
    aggregated = aggregate_stats([per_episode[index] for index in split["train_episode_indices"]])
    checks["train_stats_exact_episode_aggregate54"] = stats == serializable_stats(aggregated)
    selected_actions = filtered[selected_frames].astype(np.float64)
    checks["train_action_count_and_direct_mean_std"] = (
        stats["action"]["count"] == [29094]
        and np.allclose(stats["action"]["mean"], selected_actions.mean(axis=0), atol=1e-4, rtol=1e-5)
        and np.allclose(stats["action"]["std"], selected_actions.std(axis=0), atol=1e-4, rtol=1e-5))
    checks["separate_split_is_complete_and_disjoint"] = (
        len(split["train_episode_indices"]) == 54 and len(split["eval_episode_indices"]) == 6
        and not set(split["train_episode_indices"]) & set(split["eval_episode_indices"])
        and sorted(split["train_episode_indices"] + split["eval_episode_indices"]) == list(range(60)))
    manifest = json.loads((smooth_root / "action_smoothing_manifest.json").read_text(encoding="utf-8"))
    checks["image_source_payload_hashes_unchanged"] = inventory(raw_root) == manifest["source_files_before"]
    checks["video_bytes_inherited"] = all(
        file_digest(raw_root / relative) == file_digest(smooth_root / relative)
        for relative in manifest["source_files_before"] if relative.startswith("videos/"))
    train = LeRobotDataset("local/act4_224_action_smoothed_v1", root=smooth_root,
                          episodes=split["train_episode_indices"], video_backend="pyav",
                          delta_timestamps={"action": [step / 30 for step in range(100)]})
    validation = LeRobotDataset("local/act4_224_action_smoothed_v1", root=smooth_root,
                               episodes=split["eval_episode_indices"], video_backend="pyav",
                               delta_timestamps={"action": [step / 30 for step in range(100)]})
    checks["official_dataset54_6_frame_counts"] = len(train) == 29094 and len(validation) == 3189
    checks["official_train_statistics"] = int(train.meta.stats["action"]["count"][0]) == 29094
    validation.meta.stats = train.meta.stats
    checks["validation_explicitly_uses_train_statistics"] = (
        validation.meta.stats is train.meta.stats and int(validation.meta.stats["action"]["count"][0]) == 29094)
    probes = []
    for role, dataset in (("train", train), ("validation", validation)):
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[index]
            valid = sample["action"].shape == (100, 7) and sample["action_is_pad"].shape == (100,)
            valid = valid and bool(torch.isfinite(sample["action"]).all())
            for camera in ("observation.images.front", "observation.images.gripper"):
                valid = valid and sample[camera].shape == (3, 224, 224)
                valid = valid and bool(torch.isfinite(sample[camera]).all())
            probes.append({"split": role, "selected_index": index, "global_index": int(sample["index"]),
                           "episode_index": int(sample["episode_index"]), "valid": bool(valid)})
    checks["official_decoding_and_action100_padding"] = all(probe["valid"] for probe in probes)
    if not all(checks.values()):
        raise AssertionError(checks)
    report = {"status": "passed", "checks": checks, "probes": probes,
              "train_frames": len(train), "eval_frames": len(validation),
              "normalization": "Training split statistics explicitly assigned to validation.",
              "model_loads": 0, "optimizer_steps": 0, "hardware_connections": 0}
    write_json(RUN / "official_dataset_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
