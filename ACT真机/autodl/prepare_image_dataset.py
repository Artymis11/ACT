"""Prepare and verify a locally resized, lossless LeRobot v3 RGB dataset.

Video pixels are compared against the shared inference preprocessing function.
State/action parquet files are copied byte-for-byte. The final directory is
published only after validation, with train-only normalization statistics.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
from lerobot.utils.utils import unflatten_dict

from image_preprocessing import CONTRACT_FILENAME, preprocess_rgb, preprocessing_contract


MANIFEST_FILENAME = "image_dataset_manifest.json"
ENCODER_OPTIONS = {"qp": "0", "preset": "fast", "crf": "0", "g": "2", "bf": "0"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Symbolic links are unsupported: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result[path.relative_to(root).as_posix()] = {"size_bytes": path.stat().st_size, "sha256": digest}
    return result


def encode_frames(path, frames, rate, time_base):
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("libx264rgb", rate=rate)
        stream.width = 224
        stream.height = 224
        stream.pix_fmt = "rgb24"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.options = dict(ENCODER_OPTIONS)
        for pixels, pts, frame_time_base in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = pts
            frame.time_base = frame_time_base
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def verify_codec(directory):
    """Reject an encoder installation that changes RGB pixels or timestamps."""
    path = directory / ".lossless_codec_probe.mp4"
    images = np.random.default_rng(224).integers(0, 256, (4, 224, 224, 3), dtype=np.uint8)
    time_base = Fraction(1, 15360)
    encode_frames(path, ((image, index * 512, time_base) for index, image in enumerate(images)), Fraction(30), time_base)
    with av.open(str(path)) as video:
        frames = list(video.decode(video=0))
        require(len(frames) == len(images), "Lossless encoder probe changed frame count.")
        for index, frame in enumerate(frames):
            require(np.array_equal(images[index], frame.to_ndarray(format="rgb24")), "Encoder probe changed RGB pixels.")
            require(frame.pts * frame.time_base == Fraction(index, 30), "Encoder probe changed timestamps.")
    path.unlink()


def video_jobs(info, rows):
    """Validate constant-rate episode ranges before touching video payloads."""
    jobs = {}
    fps = Fraction(str(info["fps"]))
    for camera, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        require(not feature.get("info", {}).get("is_depth_map", False), "Only RGB camera videos are supported.")
        require(feature["shape"][-1] == 3, f"Expected three RGB channels for {camera}.")
        for row in rows:
            relative = info["video_path"].format(
                video_key=camera, chunk_index=row[f"videos/{camera}/chunk_index"],
                file_index=row[f"videos/{camera}/file_index"],
            )
            start_time = row[f"videos/{camera}/from_timestamp"]
            start = round(start_time * float(fps))
            require(abs(start / float(fps) - start_time) < 1e-5, f"Off-grid video timestamp for {camera}.")
            stop = start + row["length"]
            require(abs(stop / float(fps) - row[f"videos/{camera}/to_timestamp"]) < 1e-5, "Episode video length differs from frame count.")
            jobs.setdefault((relative, camera), []).append((start, stop, row["episode_index"]))
    require(bool(jobs), "The dataset contains no RGB video features.")
    for (relative, _), ranges in jobs.items():
        ranges.sort()
        require(ranges[0][0] == 0, f"Video episode ranges must start at frame zero: {relative}")
        for left, right in zip(ranges, ranges[1:]):
            require(left[1] == right[0], f"Video episode ranges overlap or leave gaps: {relative}")
    return fps, jobs


def convert_video(source, destination, fps):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as video:
        require(len(video.streams.video) == 1 and not video.streams.audio, "Expected one silent video stream.")
        stream = video.streams.video[0]
        require(stream.average_rate == fps, "Source video FPS differs from dataset metadata.")

        def resized_frames():
            for index, frame in enumerate(video.decode(video=0)):
                require(frame.pts is not None and frame.pts * frame.time_base == Fraction(index, 1) / fps, "Only continuous constant-rate source video is supported.")
                yield preprocess_rgb(frame.to_ndarray(format="rgb24")), frame.pts, frame.time_base

        encode_frames(destination, resized_frames(), fps, stream.time_base)


def verify_video(source, destination, ranges, fps, camera, rows):
    """Check every RGB frame and collect LeRobot's deterministic stat samples."""
    episode_by_index = {row["episode_index"]: row for row in rows}
    probes = {}
    for start, stop, episode in ranges:
        probes[episode] = {start + index for index in sample_indices(stop - start)}
    sample_arrays = []
    range_index = 0
    count = 0
    with av.open(str(source)) as original, av.open(str(destination)) as converted:
        source_frames = original.decode(video=0)
        for index, frame in enumerate(converted.decode(video=0)):
            original_frame = next(source_frames, None)
            require(original_frame is not None, "Converted video contains extra frames.")
            expected = preprocess_rgb(original_frame.to_ndarray(format="rgb24"))
            actual = frame.to_ndarray(format="rgb24")
            require(np.array_equal(expected, actual), f"Lossless RGB verification failed at {camera}, frame {index}.")
            require(frame.pts * frame.time_base == original_frame.pts * original_frame.time_base, "Video presentation timestamps changed.")
            require(index < ranges[-1][1], "Video contains frames absent from episode metadata.")
            while index >= ranges[range_index][1]:
                range_index += 1
            start, stop, episode = ranges[range_index]
            if index in probes[episode]:
                sample_arrays.append(actual.transpose(2, 0, 1))
            if index == stop - 1:
                require(len(sample_arrays) == len(probes[episode]), "Statistics sampling missed frames.")
                statistics = get_feature_stats(np.stack(sample_arrays), axis=(0, 2, 3), keepdims=True)
                row = episode_by_index[episode]
                for name, value in statistics.items():
                    normalized = value if name == "count" else np.squeeze(value / 255.0, axis=0)
                    row[f"stats/{camera}/{name}"] = normalized.tolist()
                sample_arrays.clear()
            count += 1
        require(next(source_frames, None) is None, "Converted video is missing source frames.")
        require(count == ranges[-1][1], "Decoded frame count differs from episode metadata.")
        require(converted.streams.video[0].average_rate == fps, "Converted video FPS changed.")
        pix_fmt = converted.streams.video[0].codec_context.pix_fmt
    return {"camera": camera, "frames": count, "pixel_equality": True,
            "presentation_timestamps_equal": True, "decoded_pixel_format": pix_fmt}


def prepare(args):
    started = time.perf_counter()
    source = args.dataset_root.resolve(strict=True)
    destination = args.output_dir.resolve()
    require(not destination.exists(), f"Output already exists: {destination}")
    require(not destination.is_relative_to(source) and not source.is_relative_to(destination), "Source and output must be separate directory trees.")
    require(not (source / CONTRACT_FILENAME).exists(), "The source already has an image preprocessing contract.")
    source_files = inventory(source)
    info = read_json(source / "meta/info.json")
    require(info.get("codebase_version") == "v3.0", "Expected a finalized LeRobot v3.0 dataset.")
    episode_paths = sorted((source / "meta/episodes").rglob("*.parquet"))
    require(bool(episode_paths), "Missing finalized episode metadata.")
    tables = {path: pq.read_table(path) for path in episode_paths}
    rows = [row for table in tables.values() for row in table.to_pylist()]
    rows.sort(key=lambda row: row["episode_index"])
    require(len(rows) == info["total_episodes"] and len(rows) > 2, "Expected at least three complete episodes.")
    require([row["episode_index"] for row in rows] == list(range(len(rows))), "Episode indices must be unique and contiguous.")
    require(sum(row["length"] for row in rows) == info["total_frames"], "Episode lengths differ from total_frames.")
    fps, jobs = video_jobs(info, rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.prepare-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        verify_codec(staging)
        video_paths = {relative for relative, _ in jobs}
        for relative in source_files:
            require(not Path(relative).is_absolute() and not relative.startswith("../"), "Invalid source-relative path.")
            if relative in video_paths:
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        results = []
        for (relative, camera), ranges in jobs.items():
            print(json.dumps({"phase": "converting", "camera": camera}, ensure_ascii=False), flush=True)
            convert_video(source / relative, staging / relative, fps)
            result = verify_video(source / relative, staging / relative, ranges, fps, camera, rows)
            result["path"] = relative
            results.append(result)
            feature = info["features"][camera]
            feature["shape"] = [224, 224, 3]
            feature.setdefault("info", {}).update({
                "video.height": 224, "video.width": 224, "video.codec": "h264",
                "video.pix_fmt": "gbrp", "video.g": 2, "video.crf": 0,
                "video.preset": "fast", "video.extra_options": {"qp": "0", "bf": "0"},
                "video.encoder": "libx264rgb", "video.lossless_rgb": True,
            })
            print(json.dumps({"phase": "verified", **result}, ensure_ascii=False), flush=True)
        by_episode = {row["episode_index"]: row for row in rows}
        for path, table in tables.items():
            updated = [by_episode[row["episode_index"]] for row in table.to_pylist()]
            pq.write_table(pa.Table.from_pylist(updated, schema=table.schema), staging / path.relative_to(source))
        train_rows = rows[:-2]
        train_stats = aggregate_stats([
            unflatten_dict({key.removeprefix("stats/"): np.asarray(value)
                            for key, value in row.items() if key.startswith("stats/") and value is not None})
            for row in train_rows
        ])
        write_json(staging / "meta/stats.json", {
            feature: {key: value.tolist() for key, value in statistics.items()}
            for feature, statistics in train_stats.items()
        })
        write_json(staging / "meta/info.json", info)
        write_json(staging / CONTRACT_FILENAME, preprocessing_contract())
        output_files = inventory(staging)
        modified = {"meta/info.json", "meta/stats.json", *video_paths,
                    *(path.relative_to(source).as_posix() for path in episode_paths)}
        for relative, expected in source_files.items():
            if relative not in modified:
                require(output_files[relative] == expected, f"Non-image source payload changed: {relative}")
        require(inventory(source) == source_files, "Source files changed during conversion.")
        manifest = {
            "schema_version": 1, "status": "verified", "source_dataset": str(source),
            "output_dataset": str(destination), "episodes": len(rows), "frames_per_camera": info["total_frames"],
            "preprocessing_contract_file": CONTRACT_FILENAME, "videos": results,
            "codec_probe": "Random RGB224 frames round-trip with identical pixels and exact presentation timestamps.",
            "storage_encoding": {"encoder": "libx264rgb", "options": ENCODER_OPTIONS,
                                 "pixel_format": "rgb24 input / gbrp decoded", "lossless_rgb_verified": True},
            "statistics": {
                "method": "LeRobot sample_indices + get_feature_stats on decoded resized RGB samples, per episode; aggregate_stats on training episodes only",
                "train_episode_indices": [row["episode_index"] for row in train_rows],
                "eval_episode_indices": [row["episode_index"] for row in rows[-2:]],
                "meta_stats_scope": "training episodes only; validation episode statistics remain in episode metadata",
            },
            "data_parquet_byte_identical": True, "source_files_unchanged": True,
            "source_files": source_files, "output_files_excluding_manifest": output_files,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(staging / MANIFEST_FILENAME, manifest)
        staging.rename(destination)
    print(json.dumps({"status": "verified", "output": str(destination), "episodes": len(rows),
                      "frames_per_camera": info["total_frames"], "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
