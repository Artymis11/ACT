"""Build and validate a lossless CHW uint8 mmap cache for the ACT2 videos."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading

import av
import numpy as np
import pyarrow.parquet as pq
import torch


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path):
    path = Path(path)
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path)}


def nearest_indices(timestamps, queries, tolerance_s):
    """Match the original decoder's float32 one-dimensional absolute distance."""
    times = np.asarray(timestamps, dtype=np.float32)
    values = np.asarray(queries, dtype=np.float32)
    right = np.searchsorted(times, values, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    left_distance = np.abs(values - times[left])
    right_distance = np.abs(values - times[right])
    indices = np.where(left_distance <= right_distance, left, right)
    distances = np.abs(values - times[indices])
    if not np.all(distances < tolerance_s):
        raise ValueError("Cached video query exceeds the original strict timestamp tolerance.")
    return indices


def validate_cache_manifest(manifest_path, full_hash=True):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "verified":
        raise ValueError("Video cache was not fully verified and published.")
    root = Path(manifest["dataset_root"])
    for entry in manifest["videos"]:
        paths = [(root / entry["source_video"], entry["source"]),
                 (manifest_path.parent / entry["frames_file"], entry["frames"]),
                 (manifest_path.parent / entry["timestamps_file"], entry["timestamps"])]
        for path, expected in paths:
            stat = path.stat()
            if stat.st_size != expected["size_bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError(f"Video cache file fingerprint changed: {path}")
            if full_hash and sha256_file(path) != expected["sha256"]:
                raise ValueError(f"Video cache file hash changed: {path}")
        frames = np.load(paths[1][0], mmap_mode="r", allow_pickle=False)
        timestamps = np.load(paths[2][0], mmap_mode="r", allow_pickle=False)
        if frames.dtype != np.uint8 or list(frames.shape) != entry["shape"]:
            raise ValueError("Video cache array dtype or shape changed.")
        if timestamps.shape != (entry["frame_count"],) or not np.all(np.diff(timestamps) > 0):
            raise ValueError("Video cache PTS ordering changed.")
    return manifest


def prepare_cache(dataset_root, output_dir):
    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        manifest = validate_cache_manifest(output_dir / "manifest.json", full_hash=True)
        if Path(manifest["dataset_root"]).resolve() != dataset_root:
            raise ValueError("Existing cache belongs to a different dataset.")
        return output_dir / "manifest.json"
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    video_keys = sorted(key for key, value in info["features"].items() if value.get("dtype") == "video")
    if len(video_keys) != 2 or any(info["features"][key]["shape"] != [224, 224, 3] for key in video_keys):
        raise ValueError("This cache builder requires the two ACT2 RGB224 videos.")
    episodes = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    rows = pq.read_table(dataset_root / "data/chunk-000/file-000.parquet")
    local_timestamps = rows["timestamp"].to_numpy()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)).resolve()
    manifest = {"schema_version": 1, "status": "preparing", "dataset_root": str(dataset_root),
                "total_frames_per_camera": info["total_frames"], "videos": [],
                "verification": {"all_query_indices_and_tolerances": True,
                                 "all_frame_bytes_verified": True, "official_pyav_comparisons": 0}}
    try:
        for number, key in enumerate(video_keys):
            paths = sorted((dataset_root / "videos" / key).rglob("*.mp4"))
            if len(paths) != 1:
                raise ValueError("This bounded ACT2 builder expects one file per camera.")
            source = paths[0]
            source_before = file_record(source)
            frames_path = staging / f"camera{number}_frames.npy"
            times_path = staging / f"camera{number}_timestamps.npy"
            count = int(info["total_frames"])
            cache = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.uint8,
                                               shape=(count, 3, 224, 224))
            timestamps = []
            pixel_hash = hashlib.sha256()
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                decoded_count = 0
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    if decoded_count >= count:
                        raise ValueError("Source video has more frames than dataset metadata.")
                    array = np.ascontiguousarray(frame.to_ndarray(format="rgb24").transpose(2, 0, 1))
                    if array.shape != (3, 224, 224):
                        raise ValueError("Source frame shape differs from RGB224.")
                    cache[decoded_count] = array
                    pixel_hash.update(array.tobytes())
                    timestamps.append(float(frame.pts * stream.time_base))
                    decoded_count += 1
            if decoded_count != count or not np.all(np.diff(timestamps) > 0):
                raise ValueError("Decoded frame count or PTS ordering differs from the dataset.")
            cache.flush()
            cached_hash = hashlib.sha256()
            for start in range(0, count, 128):
                cached_hash.update(cache[start:start + 128].tobytes())
            if pixel_hash.hexdigest() != cached_hash.hexdigest():
                raise ValueError("Full decoded RGB frame bytes differ from the mmap cache.")
            np.save(times_path, np.asarray(timestamps, dtype=np.float64), allow_pickle=False)
            sample_queries = []
            sample_indices = []
            for episode in episodes:
                first, last = episode["dataset_from_index"], episode["dataset_to_index"]
                # The dataset reader adds Python floats before its decoder casts
                # to float32. NumPy float32 addition loses one ULP after1024s.
                queries = (local_timestamps[first:last].astype(np.float64)
                           + float(episode[f"videos/{key}/from_timestamp"]))
                matched = nearest_indices(timestamps, queries, 1e-4)
                if not np.array_equal(matched, np.arange(first, last)):
                    raise ValueError("An episode query does not select its original global video frame.")
                for index in sorted({first, (first + last - 1) // 2, last - 1}):
                    sample_indices.append(index)
                    sample_queries.append(float(queries[index - first]))
            from lerobot.datasets.video_utils import decode_video_frames
            for index, query in zip(sample_indices, sample_queries):
                original = decode_video_frames(source, [query], 1e-4, backend="pyav", return_uint8=True)
                if not np.array_equal(original[0].numpy(), cache[index]):
                    raise ValueError(f"Official PyAV random-seek pixels differ at {key} frame {index}.")
                manifest["verification"]["official_pyav_comparisons"] += 1
            del cache
            if file_record(source) != source_before:
                raise ValueError("Source video changed during cache preparation.")
            manifest["videos"].append({"camera_key": key, "source_video": str(source.relative_to(dataset_root)),
                "source": source_before, "frames_file": frames_path.name, "timestamps_file": times_path.name,
                "frames": file_record(frames_path), "timestamps": file_record(times_path),
                "shape": [count, 3, 224, 224], "frame_count": count,
                "decoded_frame_bytes_sha256": pixel_hash.hexdigest()})
        manifest["status"] = "verified"
        manifest["cache_bytes"] = sum(entry["frames"]["size_bytes"] + entry["timestamps"]["size_bytes"]
                                      for entry in manifest["videos"])
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validate_cache_manifest(staging / "manifest.json", full_hash=True)
        staging.rename(output_dir)
        return output_dir / "manifest.json"
    finally:
        if staging.exists():
            if staging.parent != output_dir.parent or not staging.name.startswith(f".{output_dir.name}-"):
                raise RuntimeError("Refusing to clean an unexpected cache staging directory.")
            shutil.rmtree(staging)


def install_video_cache(manifest_path):
    """Install a per-process decoder hook. Invalid/missing cache uses original PyAV."""
    from lerobot.datasets import dataset_reader, video_utils
    original = video_utils.decode_video_frames
    if getattr(original, "_act_video_cache", False):
        return
    try:
        manifest = validate_cache_manifest(manifest_path, full_hash=False)
    except (OSError, ValueError, KeyError) as error:
        logging.warning("ACT video cache unavailable, retaining original decoder: %s", error)
        return
    root, cache_root = Path(manifest["dataset_root"]), Path(manifest_path).resolve().parent
    entries = {str((root / entry["source_video"]).resolve()): entry for entry in manifest["videos"]}
    opened = {}
    lock = threading.Lock()

    def cached_decode(video_path, timestamps, tolerance_s, backend=None, return_uint8=False, is_depth=False):
        entry = entries.get(str(Path(video_path).resolve()))
        if entry is None or backend not in {"pyav", "video_reader"} or is_depth or not timestamps:
            return original(video_path, timestamps, tolerance_s, backend, return_uint8, is_depth)
        try:
            for path, expected in ((Path(video_path), entry["source"]),
                                   (cache_root / entry["frames_file"], entry["frames"]),
                                   (cache_root / entry["timestamps_file"], entry["timestamps"])):
                stat = path.stat()
                if stat.st_size != expected["size_bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
                    raise ValueError("Cached video file changed after verification.")
            with lock:
                if entry["frames_file"] not in opened:
                    opened[entry["frames_file"]] = (
                        np.load(cache_root / entry["frames_file"], mmap_mode="r", allow_pickle=False),
                        np.load(cache_root / entry["timestamps_file"], mmap_mode="r", allow_pickle=False))
                frames, pts = opened[entry["frames_file"]]
            indices = nearest_indices(pts, timestamps, tolerance_s)
            result = torch.from_numpy(np.array(frames[indices], copy=True))
            return result if return_uint8 else result.to(torch.float32) / 255
        except (OSError, ValueError, IndexError) as error:
            logging.warning("ACT cache query fallback to original PyAV: %s", error)
            return original(video_path, timestamps, tolerance_s, backend, return_uint8, is_depth)

    cached_decode._act_video_cache = True
    video_utils.decode_video_frames = cached_decode
    dataset_reader.decode_video_frames = cached_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("ACT_CACHE_CPU_THREADS", "1")))
    manifest_path = prepare_cache(args.dataset_root, args.output_dir)
    print(json.dumps({"status": "verified", "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
