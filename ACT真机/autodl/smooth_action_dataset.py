"""Create an independent LeRobot v3 dataset with episode-local action means.

Only action labels and their statistics change. No model or device is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.utils.utils import unflatten_dict






def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unsupported symbolic link: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result[path.relative_to(root).as_posix()] = {"size_bytes": path.stat().st_size, "sha256": digest}
    return result


def centered_mean(values, window):
    """Centered mean with truncated, renormalized windows at episode edges."""
    values = np.asarray(values)
    if window not in (1, 3, 5, 7) or values.ndim != 2 or len(values) == 0:
        raise ValueError("Expected a nonempty N x D array and an odd window in 1/3/5/7.")
    if not np.isfinite(values).all():
        raise ValueError("Actions must be finite.")
    half = window // 2
    centers = np.arange(len(values))
    starts = np.maximum(centers - half, 0)
    stops = np.minimum(centers + half + 1, len(values))
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0, dtype=np.float64)))
    return ((cumulative[stops] - cumulative[starts]) / (stops - starts)[:, None]).astype(values.dtype)


def read_dataset(root):
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Expected finalized LeRobot v3.0 metadata.")
    data_tables = {path.relative_to(root): pq.read_table(path) for path in sorted((root / "data").rglob("*.parquet"))}
    meta_tables = {path.relative_to(root): pq.read_table(path) for path in sorted((root / "meta/episodes").rglob("*.parquet"))}
    if not data_tables or not meta_tables:
        raise ValueError("Dataset is missing data or episode parquet files.")
    table = pa.concat_tables(list(data_tables.values()))
    rows = sorted((row for item in meta_tables.values() for row in item.to_pylist()), key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != list(range(info["total_episodes"])):
        raise ValueError("Episode indices must be complete and contiguous from zero.")
    if table.num_rows != info["total_frames"] or not np.array_equal(table["index"].to_numpy(), np.arange(table.num_rows)):
        raise ValueError("Frame indices must retain the finalized global order.")
    episodes = table["episode_index"].to_numpy()
    frames = table["frame_index"].to_numpy()
    for row in rows:
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        if stop - start != row["length"] or not np.all(episodes[start:stop] == row["episode_index"]):
            raise ValueError("Episode metadata does not describe its data interval.")
        if not np.array_equal(frames[start:stop], np.arange(row["length"])):
            raise ValueError("Per-episode frame indices are invalid.")
    return info, data_tables, meta_tables, table, rows


def episode_statistics(rows):
    return [unflatten_dict({key.removeprefix("stats/"): np.asarray(value)
                           for key, value in row.items() if key.startswith("stats/") and value is not None})
            for row in rows]


def serializable_stats(stats):
    return {feature: {name: value.tolist() for name, value in values.items()} for feature, values in stats.items()}


def prepare(source, output, window):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output must be separate directory trees.")
    before = inventory(source)
    info, data_tables, meta_tables, table, rows = read_dataset(source)
    if len(rows) != 20:
        raise ValueError("This ACT2 derivation requires exactly 20 episodes for the fixed 18/2 split.")
    original = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    filtered = original.copy()
    for row in rows:
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        filtered[start:stop] = centered_mean(original[start:stop], window)
        for name, value in get_feature_stats(filtered[start:stop], axis=0, keepdims=False).items():
            row[f"stats/action/{name}"] = value.tolist()
    per_episode = episode_statistics(rows)
    train_stats = aggregate_stats(per_episode[:18])
    all_stats = aggregate_stats(per_episode)
    if int(train_stats["action"]["count"][0]) != 10379 or int(all_stats["action"]["count"][0]) != 11579:
        raise ValueError("Unexpected ACT2 split frame counts.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        changed = {path.as_posix() for path in (*data_tables, *meta_tables)} | {"meta/stats.json"}
        for relative in before:
            if relative in changed:
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative.startswith("videos/"):
                try:
                    os.link(source / relative, target)
                except OSError:
                    shutil.copy2(source / relative, target)
            else:
                shutil.copy2(source / relative, target)
        offset = 0
        for relative, item in data_tables.items():
            count = item.num_rows
            column = pa.array(filtered[offset:offset + count].tolist(), type=item.schema.field("action").type)
            updated = item.set_column(item.schema.get_field_index("action"), item.schema.field("action"), column)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(updated, target)
            check = pq.read_table(target)
            if not check.schema.equals(item.schema, check_metadata=True):
                raise ValueError("Data parquet schema or embedded metadata changed.")
            for name in item.column_names:
                if name != "action" and not item[name].equals(check[name]):
                    raise ValueError(f"Unexpected changed data column: {name}")
            offset += count
        by_episode = {row["episode_index"]: row for row in rows}
        for relative, item in meta_tables.items():
            updated = [by_episode[row["episode_index"]] for row in item.to_pylist()]
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(updated, schema=item.schema), target)
        # Match the established ACT2 convention: saved default normalizer stats
        # use only episodes 0-17. The separate all-episode audit is explicit.
        write_json(staging / "meta/stats.json", serializable_stats(train_stats))
        write_json(staging / "train_stats18.json", serializable_stats(train_stats))
        write_json(staging / "all_episode_stats20.json", serializable_stats(all_stats))
        after = inventory(source)
        if before != after:
            raise ValueError("Source dataset changed during preparation. Nothing was published.")
        output_hashes = inventory(staging)
        for relative, expected in before.items():
            if relative not in changed and output_hashes[relative] != expected:
                raise ValueError(f"Unexpected changed inherited payload: {relative}")
        manifest = {
            "schema_version": 1, "status": "verified", "source_dataset": str(source), "output_dataset": str(output),
            "operation": "centered_episode_local_arithmetic_mean_of_action_labels", "window_frames": window,
            "edge_rule": "Truncate the centered window at each episode edge and renormalize by available frames.",
            "accumulation_dtype": "float64", "output_dtype": "float32", "fps": info["fps"],
            "train_episode_indices": list(range(18)), "eval_episode_indices": [18, 19],
            "train_frames": 10379, "eval_frames": 1200,
            "normalization_source": "meta/stats.json and train_stats18.json aggregate only episodes 0-17; validation uses these training statistics.",
            "all_episode_statistics": "all_episode_stats20.json is an audit artifact and must not normalize validation.",
            "unchanged": ["observation.state", "images", "timestamps", "indices", "episode split", "image preprocessing contract"],
            "image_dataset_manifest_note": "Inherited image_dataset_manifest.json describes the earlier image conversion. This manifest records the subsequent action-label derivation.",
            "online_processing": "No online action-label mean filter is required. Observation preprocessing and native send_action mapping remain unchanged.",
            "future_context_frames": window // 2,
            "limitation": "Centered label smoothing uses nearby future labels, modifies supervision, and has not been trained or physically evaluated.",
            "changed_files": sorted(changed), "source_files_before": before, "source_files_after": after,
            "source_hashes_unchanged": True, "output_files_before_manifest": output_hashes,
            "action_delta_abs_p99": np.quantile(np.abs(filtered - original), 0.99, axis=0).tolist(),
        }
        write_json(staging / "action_smoothing_manifest.json", manifest)
        # A sibling staging directory ensures a successful rename publishes the
        # complete validated output. Failed preparations leave no final dataset.
        staging.rename(output)
    return manifest


def self_test():
    constant = np.full((6, 2), 7, dtype=np.float32)
    ramp = np.arange(5, dtype=np.float32)[:, None]
    results = {
        "constant_unchanged": np.array_equal(centered_mean(constant, 7), constant),
        "edge_renormalized": np.array_equal(centered_mean(ramp, 3).ravel(), np.array([0.5, 1, 2, 3, 3.5])),
        "single_frame": np.array_equal(centered_mean(constant[:1], 7), constant[:1]),
        "episode_boundary_separate": np.array_equal(np.vstack([centered_mean(constant, 3), centered_mean(constant * 10, 3)]), np.vstack([constant, constant * 10])),
    }
    stats = aggregate_stats([{"action": get_feature_stats(ramp, axis=0, keepdims=False)},
                             {"action": get_feature_stats(ramp + 10, axis=0, keepdims=False)}])
    combined = np.vstack([ramp, ramp + 10])
    results["aggregate_mean"] = bool(np.allclose(stats["action"]["mean"], combined.mean(axis=0)))
    results["aggregate_std"] = bool(np.allclose(stats["action"]["std"], combined.std(axis=0)))
    if not all(results.values()):
        raise AssertionError(results)
    return {"status": "passed", "checks": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = self_test() if args.self_test else prepare(args.dataset_root, args.output_dir, args.window)
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("files_before") and not key.endswith("files_after") and key != "output_files_before_manifest"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
