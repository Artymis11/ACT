"""Prepare the completed ACT4 recordings for a fixed 54/6 scratch-training run."""

from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import json
import os
import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

import prepare_image_dataset as image_dataset
from image_preprocessing import contract_digest, preprocess_rgb
from smooth_action_dataset import (
    centered_mean, episode_statistics, inventory, read_dataset, serializable_stats, write_json,
)
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats


RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[2]
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
          "wrist_yaw.pos", "wrist_roll.pos", "gripper.pos"]
DIRECTIONS = {name.removesuffix(".pos"): scale for name, scale in zip(JOINTS, [1, 1, -1, -1, -1, 1, 6])}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_source():
    info, data_tables, meta_tables, table, rows = read_dataset(SOURCE)
    dataset_state = json.loads((SOURCE / "state.json").read_text(encoding="utf-8"))
    recording_state = json.loads((RECORDING / "state.json").read_text(encoding="utf-8"))
    if dataset_state["dataset_name"] != "ACT 4" or dataset_state["status"] != "ready":
        raise ValueError("The selected source must be the finalized ACT 4 dataset.")
    if (recording_state["status"] != "completed" or recording_state["is_writing_dataset"]
            or len(rows) != 60 or table.num_rows != 32283 or info["fps"] != 30):
        raise ValueError("ACT4 completion, episode count or nominal timebase differs from the authorized run.")
    for key in ("action", "observation.state"):
        feature = info["features"][key]
        if feature["shape"] != [7] or feature["names"] != JOINTS or feature["dtype"] != "float32":
            raise ValueError(f"Unexpected numerical observation/action contract: {key}")
    camera_keys = {key for key, feature in info["features"].items() if feature["dtype"] == "video"}
    if camera_keys != {"observation.images.front", "observation.images.gripper"}:
        raise ValueError("ACT4 must retain exactly the recorded two camera channels.")
    for key in camera_keys:
        if info["features"][key]["shape"] != [480, 640, 3]:
            raise ValueError(f"Unexpected source image geometry: {key}")
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Non-finite ACT4 state/action data.")
    times = table["timestamp"].to_numpy()
    for row in rows:
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        if not np.allclose(times[start:stop], np.arange(row["length"]) / 30, rtol=0, atol=2e-6):
            raise ValueError(f"Invalid nominal timestamps in episode {row['episode_index']}.")
    log = (RECORDING / "lerobot_record.log").read_text(encoding="utf-8", errors="replace")
    start = log.index("'joint_directions':")
    start = log.index("{", start)
    directions = ast.literal_eval(log[start:log.index("}", start) + 1])
    if directions != DIRECTIONS:
        raise ValueError("ACT4 collection motor direction/scaling differs from the retained deployment mapping.")
    generator = np.random.default_rng(42)
    validation = sorted(int(generator.choice(np.arange(first, first + 10))) for first in range(0, 60, 10))
    training = [index for index in range(60) if index not in validation]
    lengths = np.asarray([row["length"] for row in rows])
    split = {
        "schema_version": 1, "seed": 42,
        "method": "One validation episode sampled from each consecutive block of ten collected episodes.",
        "reason": "Spread the six held-out episodes across the acquisition session while retaining 90 percent for training.",
        "dataset_id": dataset_state["dataset_id"], "recording_id": recording_state["recording_id"],
        "source_dataset": str(SOURCE), "image_dataset": str(IMAGE_OUTPUT), "smoothed_dataset": str(SMOOTH_OUTPUT),
        "train_episode_indices": training, "eval_episode_indices": validation,
        "train_frames": int(lengths[training].sum()), "eval_frames": int(lengths[validation].sum()),
        "total_episodes": 60, "total_frames": 32283, "fps": 30,
        "normalization": "Aggregate per-episode statistics from the selected 54 training episodes only.",
        "limit": "A single recording session and a single task label do not provide an independent scene/session test set.",
    }
    source_hashes = inventory(SOURCE)
    report = {
        "status": "passed_metadata_and_numeric_audit", "source": str(SOURCE),
        "recording_status": recording_state["status"], "dataset_status": dataset_state["status"],
        "episodes": len(rows), "frames": table.num_rows, "fps": info["fps"],
        "camera_features": {key: info["features"][key] for key in sorted(camera_keys)},
        "camera_binding": {key: recording_state["setup"][key] for key in
                           ("camera_front_index", "camera_wrist_index", "camera_wrist_role")},
        "joint_order": JOINTS, "motor_send_mapping_from_recording": directions,
        "state_coordinate": "Native follower motor degrees from collection.",
        "action_coordinate": "Collected leader action degrees, mapped once by the recorded motor directions/scaling.",
        "action_min": action.min(axis=0).tolist(), "action_max": action.max(axis=0).tolist(),
        "state_min": state.min(axis=0).tolist(), "state_max": state.max(axis=0).tolist(),
        "finite_state_action": True, "nominal_timestamps_continuous": True,
        "nominal_timestamp_limit": "Frame-index-generated timestamps cannot establish real capture latency.",
        "episode_lengths": lengths.tolist(), "source_files": source_hashes,
        "recording_product_warning": recording_state.get("warning"),
        "warning_reconciliation": "The retained product warning says partial_saved, but planned/saved/finalized episode counts are all 60 and all frame intervals are complete.",
        "split": split, "image_contract_sha256": contract_digest(),
        "source_warning_preserved": True, "physical_task_success_labels_available": False,
    }
    write_json(RUN / "split_manifest.json", split)
    write_json(RUN / "source_dataset_audit.json", report)
    return split, report


def fit_split_statistics(rows, split):
    per_episode = episode_statistics(rows)
    selected = [per_episode[index] for index in split["train_episode_indices"]]
    train_stats = aggregate_stats(selected)
    all_stats = aggregate_stats(per_episode)
    if int(train_stats["action"]["count"][0]) != split["train_frames"]:
        raise ValueError("Training normalization count differs from the fixed split.")
    return train_stats, all_stats


def prepare_images(split):
    if IMAGE_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {IMAGE_OUTPUT}")
    IMAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".act4_224.split-", dir=IMAGE_OUTPUT.parent) as temporary:
        staged = Path(temporary) / "ready"
        image_dataset.prepare(SimpleNamespace(dataset_root=SOURCE, output_dir=staged))
        info, _, _, _, rows = read_dataset(staged)
        train_stats, all_stats = fit_split_statistics(rows, split)
        write_json(staged / "meta/stats.json", serializable_stats(train_stats))
        write_json(staged / "train_stats54.json", serializable_stats(train_stats))
        write_json(staged / "all_episode_stats60.json", serializable_stats(all_stats))
        write_json(staged / "split_manifest.json", split)
        manifest = json.loads((staged / image_dataset.MANIFEST_FILENAME).read_text(encoding="utf-8"))
        manifest["output_dataset"] = str(IMAGE_OUTPUT)
        manifest["statistics"].update(train_episode_indices=split["train_episode_indices"],
                                      eval_episode_indices=split["eval_episode_indices"])
        manifest["statistics"]["split_manifest_sha256"] = digest(RUN / "split_manifest.json")
        hashes = inventory(staged)
        hashes.pop(image_dataset.MANIFEST_FILENAME)
        manifest["output_files_excluding_manifest"] = hashes
        manifest["total_frames"] = info["total_frames"]
        write_json(staged / image_dataset.MANIFEST_FILENAME, manifest)
        staged.rename(IMAGE_OUTPUT)
    return manifest


def prepare_smoothed(split):
    if SMOOTH_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {SMOOTH_OUTPUT}")
    before = inventory(IMAGE_OUTPUT)
    info, data_tables, meta_tables, table, rows = read_dataset(IMAGE_OUTPUT)
    original = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    filtered = original.copy()
    curvature_original, curvature_smoothed = [], []
    for row in rows:
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        filtered[start:stop] = centered_mean(original[start:stop], 3)
        curvature_original.append(np.diff(original[start:stop], n=2, axis=0))
        curvature_smoothed.append(np.diff(filtered[start:stop], n=2, axis=0))
        for name, value in get_feature_stats(filtered[start:stop], axis=0, keepdims=False).items():
            row[f"stats/action/{name}"] = value.tolist()
    train_stats, all_stats = fit_split_statistics(rows, split)
    changed = {path.as_posix() for path in (*data_tables, *meta_tables)} | {
        "meta/stats.json", "train_stats54.json", "all_episode_stats60.json"}
    with tempfile.TemporaryDirectory(prefix=".act4_224_action_smoothed_v1.prepare-", dir=SMOOTH_OUTPUT.parent) as temporary:
        staged = Path(temporary) / "ready"
        staged.mkdir()
        for name in before:
            if name in changed:
                continue
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.startswith("videos/"):
                try:
                    os.link(IMAGE_OUTPUT / name, target)
                except OSError:
                    shutil.copy2(IMAGE_OUTPUT / name, target)
            else:
                shutil.copy2(IMAGE_OUTPUT / name, target)
        offset = 0
        for relative, item in data_tables.items():
            count = item.num_rows
            column = pa.array(filtered[offset:offset + count].tolist(), type=item.schema.field("action").type)
            updated = item.set_column(item.schema.get_field_index("action"), item.schema.field("action"), column)
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(updated, target)
            check = pq.read_table(target)
            if not check.schema.equals(item.schema, check_metadata=True):
                raise ValueError("Smoothing changed the data parquet schema.")
            for name in item.column_names:
                if name != "action" and not item[name].equals(check[name]):
                    raise ValueError(f"Smoothing changed a non-action column: {name}")
            offset += count
        by_episode = {row["episode_index"]: row for row in rows}
        for relative, item in meta_tables.items():
            updated = [by_episode[row["episode_index"]] for row in item.to_pylist()]
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(updated, schema=item.schema), target)
        write_json(staged / "meta/stats.json", serializable_stats(train_stats))
        write_json(staged / "train_stats54.json", serializable_stats(train_stats))
        write_json(staged / "all_episode_stats60.json", serializable_stats(all_stats))
        if inventory(IMAGE_OUTPUT) != before:
            raise ValueError("The image dataset changed during label derivation.")
        output_hashes = inventory(staged)
        for name, expected in before.items():
            if name not in changed and output_hashes[name] != expected:
                raise ValueError(f"Smoothing changed an inherited payload: {name}")
        original_rms = np.sqrt(np.mean(np.vstack(curvature_original).astype(np.float64) ** 2, axis=0))
        smoothed_rms = np.sqrt(np.mean(np.vstack(curvature_smoothed).astype(np.float64) ** 2, axis=0))
        manifest = {
            "schema_version": 1, "status": "verified", "source_dataset": str(IMAGE_OUTPUT),
            "output_dataset": str(SMOOTH_OUTPUT), "operation": "centered_episode_local_arithmetic_mean_of_action_labels",
            "window_frames": 3, "edge_rule": "Truncate at each episode edge and renormalize by available frames.",
            "accumulation_dtype": "float64", "output_dtype": "float32", "fps": info["fps"],
            "train_episode_indices": split["train_episode_indices"], "eval_episode_indices": split["eval_episode_indices"],
            "train_frames": split["train_frames"], "eval_frames": split["eval_frames"],
            "split_manifest_sha256": digest(RUN / "split_manifest.json"),
            "normalization_source": "meta/stats.json equals train_stats54.json and aggregates only the selected 54 training episodes.",
            "all_episode_statistics": "all_episode_stats60.json is for audit only.",
            "unchanged": ["observation.state", "images", "timestamps", "indices", "episode split", "image preprocessing contract"],
            "image_dataset_manifest_note": "Inherited image_dataset_manifest.json describes the preceding image conversion.",
            "future_context_frames": 1,
            "limitation": "Centered labels use one nearby future frame. Smoothed-label validation metrics require a separate original-label evaluation for comparison.",
            "changed_files": sorted(changed), "source_files_before": before,
            "source_hashes_unchanged": True, "output_files_before_manifest": output_hashes,
            "action_delta_abs_p99": np.quantile(np.abs(filtered - original), 0.99, axis=0).tolist(),
            "second_difference_rms_original": original_rms.tolist(),
            "second_difference_rms_smoothed": smoothed_rms.tolist(),
            "second_difference_rms_reduction_percent": ((1 - smoothed_rms / original_rms) * 100).tolist(),
        }
        write_json(staged / "action_smoothing_manifest.json", manifest)
        staged.rename(SMOOTH_OUTPUT)
    return manifest


def sample_contact_sheets():
    info, _, _, _, rows = read_dataset(IMAGE_OUTPUT)
    selections = [rows[index] for index in (0, 17, 34, 58)]
    paths = []
    for camera in ("front", "gripper"):
        key = "observation.images." + camera
        sheet = Image.new("RGB", (224 * 3, 252 * len(selections)), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for row_number, row in enumerate(selections):
            relative = info["video_path"].format(video_key=key,
                chunk_index=row[f"videos/{key}/chunk_index"], file_index=row[f"videos/{key}/file_index"])
            first = round(row[f"videos/{key}/from_timestamp"] * info["fps"])
            samples = [0, row["length"] // 2, row["length"] - 1]
            with av.open(str(IMAGE_OUTPUT / relative)) as container:
                stream = container.streams.video[0]
                for column, local_frame in enumerate(samples):
                    target_time = Fraction(first + local_frame, info["fps"])
                    container.seek(int(target_time / stream.time_base), stream=stream, backward=True)
                    frame = next(item for item in container.decode(video=0) if item.pts * item.time_base >= target_time)
                    pixels = frame.to_ndarray(format="rgb24")
                    if pixels.shape != (224, 224, 3) or not np.array_equal(pixels, preprocess_rgb(pixels)):
                        raise ValueError("Prepared sample does not retain the shared RGB224 contract.")
                    sheet.paste(Image.fromarray(pixels), (column * 224, row_number * 252 + 28))
                    draw.text((column * 224 + 5, row_number * 252 + 7),
                              f"{camera} ep{row['episode_index']:02d} frame{local_frame} t={local_frame / 30:.2f}s", fill="black")
        path = RUN / f"samples_{camera}_224.png"
        sheet.save(path)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT / "local/data")
    args = parser.parse_args()
    global SOURCE, RECORDING, IMAGE_OUTPUT, SMOOTH_OUTPUT
    SOURCE = args.source.resolve(strict=True)
    RECORDING = args.recording.resolve(strict=True)
    IMAGE_OUTPUT = args.output_root.resolve() / "act4_224"
    SMOOTH_OUTPUT = args.output_root.resolve() / "act4_224_action_smoothed_v1"
    started = time.perf_counter()
    split, audit = audit_source()
    print(json.dumps({"phase": "source_audit_passed", "episodes": 60, "frames": 32283,
                      "train_frames": split["train_frames"], "eval_frames": split["eval_frames"]}), flush=True)
    image_manifest = prepare_images(split)
    smooth_manifest = prepare_smoothed(split)
    samples = sample_contact_sheets()
    if inventory(SOURCE) != audit["source_files"]:
        raise ValueError("The original recorded data changed during preparation.")
    result = {"status": "verified", "image_dataset": str(IMAGE_OUTPUT), "smoothed_dataset": str(SMOOTH_OUTPUT),
              "episodes": 60, "frames_per_camera": 32283, "pixel_checked_frames": 64566,
              "split_manifest": str(RUN / "split_manifest.json"), "split": split,
              "original_source_hashes_unchanged": True,
              "image_manifest_sha256": digest(IMAGE_OUTPUT / "image_dataset_manifest.json"),
              "smoothing_manifest_sha256": digest(SMOOTH_OUTPUT / "action_smoothing_manifest.json"),
              "image_train_stats_sha256": digest(IMAGE_OUTPUT / "train_stats54.json"),
              "smoothed_train_stats_sha256": digest(SMOOTH_OUTPUT / "train_stats54.json"),
              "geometry": "RGB uint8; 640x480 -> 224x168 bilinear -> top/bottom28 black pixels -> 224x224",
              "image_contract_sha256": contract_digest(),
              "second_difference_rms_reduction_percent": smooth_manifest["second_difference_rms_reduction_percent"],
              "sample_sheets": samples, "elapsed_seconds": time.perf_counter() - started,
              "model_loads": 0, "optimizer_steps": 0, "hardware_connections": 0}
    write_json(RUN / "data_preparation_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
