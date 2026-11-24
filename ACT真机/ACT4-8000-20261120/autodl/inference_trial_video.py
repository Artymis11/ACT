"""Archive separate camera tracks and same-trial telemetry from the ACT4 client."""

from contextlib import ExitStack
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
from queue import SimpleQueue
import re
import shutil
import sys
import threading
import time

import av
import numpy as np


VIDEO_ROOT = Path(__file__).resolve().parents[3] / "推理视频"
BUCKETS = {"unknown": "待确认", "success": "成功", "failed": "未成功"}
ROLES = ("front", "gripper")
FPS = 30
VIDEO_CLOCK = 90000
TIME_BASE = Fraction(1, VIDEO_CLOCK)
TELEMETRY_FILES = ("observations.jsonl", "executed_targets.jsonl", "trial_events.jsonl")


def classify_trial_video(trace, trial_id, episode_id, outcome, *, root=VIDEO_ROOT):
    """Move this trial's source files as a group, preserving every outcome."""
    if outcome not in ("success", "failed") or type(trial_id) is not int or not 1 <= trial_id <= 100:
        raise ValueError("Expected a reviewed trial and a success/failed outcome.")
    index = Path(trace) / "trial_videos" / f"trial_{trial_id:03d}.json"
    if not index.is_file():
        return None
    record = json.loads(index.read_text(encoding="utf-8"))
    if record["trial_id"] != trial_id or record["episode_id"] != episode_id:
        raise ValueError("The outcome belongs to another trial recording.")
    name = record["folder"]
    if not re.fullmatch(r"\d{8}_\d{6}_trial\d{3}_[0-9a-f]{32}", name):
        raise ValueError("Unexpected trial recording folder name.")
    root = Path(root).resolve()
    folders = {key: root / bucket / name for key, bucket in BUCKETS.items()}
    present = [folder for folder in folders.values() if folder.is_dir()]
    destination = folders[outcome]
    if present == [destination]:
        return destination
    if len(present) != 1 or not all(folder.resolve().is_relative_to(root) for folder in folders.values()):
        raise RuntimeError("Could not identify a unique local trial recording.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        present[0].rename(destination)
    except FileNotFoundError:
        if not destination.is_dir():
            raise
    return destination


class CameraTrack:
    def __init__(self, path):
        self.last_stamp = None
        self.count = 0
        self.container = av.open(
            str(path), mode="w", options={"movflags": "+faststart", "movie_timescale": str(VIDEO_CLOCK)},
        )
        try:
            self.stream = self.container.add_stream("libx264", rate=FPS)
            self.stream.width, self.stream.height = 640, 480
            self.stream.pix_fmt = "yuv420p"
            self.stream.time_base = TIME_BASE
            self.stream.codec_context.time_base = TIME_BASE
            self.stream.options = {"preset": "ultrafast", "crf": "19", "threads": "1"}
        except BaseException:
            self.container.close()
            raise

    def capture(self, camera, root_ns):
        if not camera.frame_lock.acquire(blocking=False):
            return None
        try:
            rgb, stamp = camera.latest_color_frame, camera.latest_timestamp
        finally:
            camera.frame_lock.release()
        if (not isinstance(rgb, np.ndarray) or rgb.shape != (480, 640, 3)
                or rgb.dtype != np.uint8 or stamp is None
                or (self.last_stamp is not None and stamp <= self.last_stamp)):
            return None
        # RealSense replaces the RGB array on each update. The frame encoder
        # can therefore copy it after releasing the driver's cache lock.
        stamp_ns = round(stamp * 1e9)
        relative_ns = stamp_ns - root_ns
        if relative_ns < 0:
            return None
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        pts = (relative_ns * VIDEO_CLOCK + 500_000_000) // 1_000_000_000
        frame.pts = pts
        frame.time_base = TIME_BASE
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.last_stamp = stamp
        self.count += 1
        return {"capture_monotonic_ns": stamp_ns, "relative_ns": relative_ns, "video_pts": pts}

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class TrialSources:
    def __init__(self, root, event):
        self.start_ns = event["monotonic_ns"]
        self.trial_id = event["trial_id"]
        self.episode_id = event["episode_id"]
        if type(self.trial_id) is not int or not 1 <= self.trial_id <= 100 or not re.fullmatch(
                r"[0-9a-f]{32}", self.episode_id):
            raise ValueError("Unexpected trial identity.")
        stamp = datetime.fromtimestamp(event["wall_time_ns"] / 1e9).strftime("%Y%m%d_%H%M%S")
        self.name = f"{stamp}_trial{self.trial_id:03d}_{self.episode_id}"
        self.pending = root / BUCKETS["unknown"] / self.name
        self.pending.mkdir(parents=True)
        self.outcome = "unknown"
        self.error = None
        self.resources = ExitStack()
        try:
            self.frame_log = self.resources.enter_context(
                (self.pending / "frame_timestamps.jsonl").open("x", encoding="utf-8", buffering=1),
            )
            self.tracks = {}
            for role in ROLES:
                track = CameraTrack(self.pending / f"{role}.mp4")
                self.resources.callback(track.close)
                self.tracks[role] = track
        except BaseException:
            self.resources.close()
            raise

    def capture(self, cameras):
        for role, track in self.tracks.items():
            timing = track.capture(cameras[role], self.start_ns)
            if timing:
                row = {"camera": role, "frame_index": track.count - 1, **timing}
                self.frame_log.write(json.dumps(row) + "\n")

    def finish(self):
        try:
            self.resources.close()
        except Exception as error:
            self.error = self.error or str(error)
        return {
            "trial_id": self.trial_id, "episode_id": self.episode_id, "folder": self.name,
            "trial_start_monotonic_ns": self.start_ns, "encoded_frames": {
                role: track.count for role, track in self.tracks.items()
            },
            "video_time_base": str(TIME_BASE), "nominal_camera_fps": FPS, "image_size": [640, 480],
            "camera_time_basis": "RealSense cache receipt on local perf_counter; not hardware exposure time",
            "outcome_from_events": self.outcome,
            "capture_error": self.error,
        }


class TrialVideoRecorder:
    """A camera-cache consumer. Trial callbacks only enqueue lifecycle records."""

    EVENTS = {"trial_start", "trial_end", "retry_zero_verified",
              "zero_state_disabled_closed", "session_finished"}

    def __init__(self, trace, cameras, *, root=VIDEO_ROOT):
        self.trace = Path(trace)
        self.root = Path(root)
        self.cameras = {role: cameras[role] for role in ROLES}
        for camera in self.cameras.values():
            if not all(hasattr(camera, name) for name in (
                    "frame_lock", "latest_color_frame", "latest_timestamp")):
                raise ValueError("An existing RealSense RGB cache is unavailable.")
        self.events = SimpleQueue()
        self.wake = threading.Event()
        self.finished = {}
        self.error = None
        self.worker = threading.Thread(target=self._run, name="act-trial-camera-recording")
        self.worker.start()

    def on_record(self, record):
        if record.get("event") in self.EVENTS:
            self.events.put(record)
            self.wake.set()

    def close(self, timeout=10.0):
        self.events.put(None)
        self.wake.set()
        self.worker.join(timeout)
        return not self.worker.is_alive()

    def _finish(self, sources):
        self.finished[sources.trial_id] = sources.finish()

    def _archive_telemetry(self):
        if not self.finished:
            return
        from inference_outcomes import summarize

        directories = {trial: self.root / BUCKETS["unknown"] / record["folder"]
                       for trial, record in self.finished.items()}
        for filename in TELEMETRY_FILES:
            if not (self.trace / filename).is_file():
                continue
            with ExitStack() as resources:
                source = resources.enter_context((self.trace / filename).open(encoding="utf-8"))
                outputs = {trial: resources.enter_context((directory / filename).open("x", encoding="utf-8"))
                           for trial, directory in directories.items()}
                for line in source:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        print(f"ACT4 camera recording: incomplete line in {filename}", file=sys.stderr)
                        continue
                    trial = row.get("trial_id")
                    if trial in outputs:
                        outputs[trial].write(line)
        index = self.trace / "trial_videos"
        index.mkdir(exist_ok=True)
        for trial, record in self.finished.items():
            record["source_feedback_directory"] = str(self.trace)
            manifest = self.trace / "manifest.json"
            if manifest.is_file():
                shutil.copyfile(manifest, directories[trial] / "feedback_manifest.json")
            text = json.dumps(record, ensure_ascii=False, indent=2)
            (directories[trial] / "recording.json").write_text(text, encoding="utf-8")
            index_path = index / f"trial_{trial:03d}.json"
            pending_index = index_path.with_suffix(".tmp")
            pending_index.write_text(text, encoding="utf-8")
            pending_index.replace(index_path)
        try:
            reviewed = {item["trial_id"]: item["outcome"]
                        for item in summarize(self.trace / "trial_events.jsonl")["trials"]}
        except (OSError, ValueError, RuntimeError):
            reviewed = {}
        for trial, record in self.finished.items():
            outcome = reviewed.get(trial, record["outcome_from_events"])
            if outcome != "unknown":
                classify_trial_video(self.trace, trial, record["episode_id"], outcome, root=self.root)

    def _run(self):
        sources = None
        try:
            while True:
                self.wake.clear()
                while not self.events.empty():
                    event = self.events.get()
                    if event is None:
                        if sources is not None:
                            previous, sources = sources, None
                            self._finish(previous)
                        self._archive_telemetry()
                        return
                    name = event["event"]
                    if name == "trial_start":
                        if sources is not None:
                            previous, sources = sources, None
                            self._finish(previous)
                        try:
                            sources = TrialSources(self.root, event)
                        except Exception as error:
                            print(f"ACT4 trial recording could not start: {error}", file=sys.stderr, flush=True)
                    elif sources is not None and name == "trial_end":
                        if event.get("outcome") == "failed" and event.get("reason") == "operator_failed":
                            sources.outcome = "failed"
                    elif sources is not None:
                        previous, sources = sources, None
                        self._finish(previous)
                if sources is not None:
                    at = time.perf_counter()
                    try:
                        sources.capture(self.cameras)
                    except Exception as error:
                        print(f"ACT4 trial recording interrupted: {error}", file=sys.stderr, flush=True)
                        sources.error = str(error)
                        previous, sources = sources, None
                        self._finish(previous)
                    self.wake.wait(max(0.001, 1 / 90 - (time.perf_counter() - at)))
                else:
                    self.wake.wait(0.5)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            print(f"ACT4 camera recording: {self.error}", file=sys.stderr, flush=True)
        finally:
            if sources is not None:
                try:
                    self._finish(sources)
                except Exception as error:
                    self.error = self.error or f"{type(error).__name__}: {error}"
