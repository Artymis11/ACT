"""Export a 30 FPS dual-camera review from a recorded ACT4 demonstration."""

import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile

import av
import pyarrow.parquet as pq

from inference_video_layout import VideoLayout


CAMERAS = ("observation.images.front", "observation.images.gripper")
JOINTS = (
    ("shoulder_pan.pos", "J1  底座旋转"),
    ("shoulder_lift.pos", "J2  肩部俯仰"),
    ("elbow_flex.pos", "J3  肘部屈伸"),
    ("wrist_flex.pos", "J4  腕部俯仰"),
    ("wrist_yaw.pos", "J5  腕部偏航"),
    ("wrist_roll.pos", "J6  腕部滚转"),
    ("gripper.pos", "J7  夹爪电机"),
)
FPS = 30


def read_episode(dataset, episode_index):
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8-sig"))
    if info["fps"] != FPS:
        raise ValueError(f"Expected a {FPS} FPS source dataset.")
    episode_files = sorted((dataset / "meta" / "episodes").glob("chunk-*/*.parquet"))
    episodes = [
        row
        for path in episode_files
        for row in pq.read_table(path, filters=[("episode_index", "=", episode_index)]).to_pylist()
    ]
    if len(episodes) != 1:
        raise ValueError(f"Expected one metadata row for episode {episode_index}.")
    episode = episodes[0]
    samples = pq.read_table(
        dataset / info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"],
        ),
        columns=["observation.state", "timestamp", "frame_index", "index"],
        filters=[("episode_index", "=", episode_index)],
    ).to_pylist()
    if len(samples) != episode["length"] or not samples:
        raise ValueError("Joint sample count does not match the episode metadata.")
    joint_names = info["features"]["observation.state"]["names"]
    joint_indexes = [joint_names.index(name) for name, _ in JOINTS]
    first_index = episode["dataset_from_index"]
    observed = []
    for index, row in enumerate(samples):
        seconds = index / FPS
        positions = tuple(row["observation.state"][joint_index] for joint_index in joint_indexes)
        if (row["frame_index"] != index or row["index"] != first_index + index
                or not math.isclose(row["timestamp"], seconds, rel_tol=0, abs_tol=0.00001)
                or not all(math.isfinite(position) for position in positions)):
            raise ValueError(f"Invalid time or joint measurement at sample {index}.")
        observed.append((seconds, positions))
    if first_index + len(observed) != episode["dataset_to_index"]:
        raise ValueError("Episode data bounds do not match its sample count.")
    return info, episode, observed


def video_segment(dataset, info, episode, key):
    fields = f"videos/{key}"
    start = round(episode[f"{fields}/from_timestamp"] * FPS)
    stop = start + episode["length"]
    if not (
        math.isclose(episode[f"{fields}/from_timestamp"], start / FPS, rel_tol=0, abs_tol=0.000001)
        and math.isclose(episode[f"{fields}/to_timestamp"], stop / FPS, rel_tol=0, abs_tol=0.000001)
    ):
        raise ValueError(f"Invalid episode timestamps for {key}.")
    path = dataset / info["video_path"].format(
        video_key=key,
        chunk_index=episode[f"{fields}/chunk_index"],
        file_index=episode[f"{fields}/file_index"],
    )
    return path, start


def read_frames(container, start, length, key):
    stream = container.streams.video[0]
    if stream.average_rate != FPS:
        raise ValueError(f"Unexpected video sampling rate for {key}: {stream.average_rate}.")
    if start:
        container.seek(int(Fraction(start, FPS) / stream.time_base), stream=stream)
    expected = start
    for frame in container.decode(stream):
        timestamp = frame.pts * frame.time_base
        if timestamp < Fraction(start, FPS):
            continue
        if timestamp != Fraction(expected, FPS):
            raise ValueError(f"Video timestamp mismatch for {key} at frame {expected}.")
        yield frame.to_image()
        expected += 1
        if expected == start + length:
            return
    raise ValueError(f"Incomplete camera sequence for {key}.")


def verify_video(path, frame_count):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != FPS:
            raise ValueError("Exported frame rate differs from the source.")
        decoded = 0
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts * frame.time_base != Fraction(index, FPS):
                raise ValueError(f"Exported timestamp mismatch at frame {index}.")
            decoded += 1
        if decoded != frame_count:
            raise ValueError(f"Expected {frame_count} output frames, decoded {decoded}.")


def export_episode(dataset, destination, episode_index):
    info, episode, observed = read_episode(dataset, episode_index)
    frames = len(observed)
    duration = frames / FPS
    layout = VideoLayout(
        f"ACT4  训练示范 {episode_index:02d}",
        "SenseCraft  ·  30 FPS",
        duration,
        [label for _, label in JOINTS],
        observed,
    )
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".mp4", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ExitStack() as stack:
            tracks = []
            for key in CAMERAS:
                path, start = video_segment(dataset, info, episode, key)
                camera = stack.enter_context(av.open(str(path)))
                tracks.append(read_frames(camera, start, frames, key))
            output = stack.enter_context(av.open(str(temporary), "w", options={"movflags": "+faststart"}))
            stream = output.add_stream("libx264", rate=FPS)
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "17", "preset": "fast"}
            for index in range(frames):
                picture = layout.render(index, tuple(next(track) for track in tracks))
                if index == 0:
                    stream.width, stream.height = picture.size
                for packet in stream.encode(av.VideoFrame.from_image(picture)):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        verify_video(temporary, frames)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return frames, duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    count, duration = export_episode(args.dataset.resolve(strict=True), args.destination, args.episode)
    print(f"Verified {count} synchronized frames at {FPS} FPS, {duration:.3f} seconds.")
