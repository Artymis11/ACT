"""Export the original continuous cameras and recorded seven-axis samples of one trial."""

import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile

import av

from inference_video_layout import VideoLayout
from render_recorded_demo_video import FPS, JOINTS, verify_video


ROLES = ("front", "gripper")
VIDEO_CLOCK = 90_000
NANOSECONDS = 1_000_000_000


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def angle_series(archive, filename, field, trial, origin, clip_end):
    samples = {}
    for row in read_rows(archive / filename):
        stamp = row["monotonic_ns"]
        if row.get("trial_id") == trial and origin <= stamp <= clip_end:
            # Pre-policy holding may send only the six arm axes.
            if row.get("timestep") == -1:
                continue
            values = tuple(row[field][key] for key, _ in JOINTS)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite joint values in {filename}")
            samples[stamp] = values
    if len(samples) < 2:
        raise ValueError(f"The clip has too few joint samples in {filename}")
    return [((stamp - origin) / NANOSECONDS, samples[stamp]) for stamp in sorted(samples)]


def read_camera_times(archive, recording):
    timelines = {role: [] for role in ROLES}
    start = recording["trial_start_monotonic_ns"]
    for row in read_rows(archive / "frame_timestamps.jsonl"):
        entries = timelines[row["camera"]]
        stamp, pts = row["capture_monotonic_ns"], row["video_pts"]
        relative = stamp - start
        if (row["frame_index"] != len(entries) or relative != row["relative_ns"] or relative < 0
                or pts != (relative * VIDEO_CLOCK + NANOSECONDS // 2) // NANOSECONDS
                or (entries and (stamp <= entries[-1][0] or pts <= entries[-1][1]))):
            raise ValueError(f"Invalid camera timing in {archive / 'frame_timestamps.jsonl'}")
        entries.append((stamp, pts))
    for role, entries in timelines.items():
        if not entries or len(entries) != recording["encoded_frames"][role]:
            raise ValueError(f"Unexpected frame count in the {role} camera index")
    return timelines


class CameraCursor:
    def __init__(self, container, timestamps, role):
        self.frames = iter(container.decode(video=0))
        self.timestamps = timestamps
        self.role = role
        self.index = 0
        self.image = None
        self.max_age_ns = 0

    def at(self, stamp):
        frame = None
        while self.index < len(self.timestamps) and self.timestamps[self.index][0] <= stamp:
            frame = next(self.frames, None)
            _, pts = self.timestamps[self.index]
            if (frame is None or frame.pts is None or (frame.width, frame.height) != (640, 480)
                    or frame.pts * frame.time_base != Fraction(pts, VIDEO_CLOCK)):
                raise ValueError(f"Decoded {self.role} video disagrees with its frame index at {self.index}")
            self.index += 1
        if frame is not None:
            self.image = frame.to_image()
        if self.image is None:
            raise ValueError(f"No {self.role} camera frame at the clip start")
        self.max_age_ns = max(self.max_age_ns, stamp - self.timestamps[self.index - 1][0])
        return self.image


def export_trial(archive, destination, title, end_s, *, start_s=0.0):
    recording = json.loads((archive / "recording.json").read_text(encoding="utf-8"))
    trial = recording["trial_id"]
    events = read_rows(archive / "trial_events.jsonl")
    starts = [row for row in events if row.get("event") == "trial_start" and row.get("trial_id") == trial]
    stops = [row["monotonic_ns"] for row in events
             if row.get("event") == "trial_end" and row.get("trial_id") == trial]
    if (len(starts) != 1 or len(stops) != 1 or recording["capture_error"] is not None
            or starts[0].get("episode_id") != recording["episode_id"]
            or starts[0]["monotonic_ns"] != recording["trial_start_monotonic_ns"]):
        raise ValueError("Camera and lifecycle records do not identify one complete trial")
    trial_start = starts[0]["monotonic_ns"]
    camera_times = read_camera_times(archive, recording)
    clip_start = trial_start + round(start_s * NANOSECONDS)
    origin = max(clip_start, *(entries[0][0] for entries in camera_times.values()))
    clip_end = trial_start + round(end_s * NANOSECONDS)
    if not origin < clip_end <= min(stops[0], *(entries[-1][0] for entries in camera_times.values())):
        raise ValueError("Clip must end within the recorded inference and both camera tracks")
    observed = angle_series(archive, "observations.jsonl", "fields", trial, origin, clip_end)
    targets = angle_series(archive, "executed_targets.jsonl", "post_dispatch_target_motor_deg", trial, origin, clip_end)
    frames = math.ceil(Fraction((clip_end - origin) * FPS, NANOSECONDS))
    layout = VideoLayout(title, "原速  ·  30 FPS", frames / FPS, [label for _, label in JOINTS],
                         observed, max_observation_gap_s=0.5, targets=targets)

    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".mp4", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ExitStack() as stack:
            cameras = [CameraCursor(stack.enter_context(av.open(str(archive / f"{role}.mp4"))),
                                    camera_times[role], role) for role in ROLES]
            output = stack.enter_context(av.open(str(temporary), "w", options={"movflags": "+faststart"}))
            stream = output.add_stream("libx264", rate=FPS)
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "17", "preset": "veryfast", "threads": "3"}
            for index in range(frames):
                stamp = origin + round(Fraction(index * NANOSECONDS, FPS))
                images = tuple(camera.at(stamp) for camera in cameras)
                picture = layout.render_at(index / FPS, images)
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
    return {"output": str(destination), "frames": frames, "seconds": frames / FPS,
            "observed_joint_samples": len(observed), "dispatched_joint_samples": len(targets),
            "max_camera_frame_age_ms": {c.role: round(c.max_age_ns / 1e6, 2) for c in cameras}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--start", type=float, default=0.0, help="Clip start time relative to trial start, in seconds")
    parser.add_argument("--end", required=True, type=float, help="Clip end time relative to trial start, in seconds")
    args = parser.parse_args()
    if not (math.isfinite(args.start) and math.isfinite(args.end) and 0 <= args.start < args.end):
        parser.error("Clip start and end must be finite with 0 <= start < end")
    result = export_trial(args.archive.resolve(strict=True), args.destination, args.title, args.end,
                          start_s=args.start)
    print(json.dumps(result, ensure_ascii=False))
