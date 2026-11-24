"""Bounded, asynchronous feedback recording from SenseCraft's existing hooks.

Set WEBUI_PIPELINE_OBSERVER=inference_feedback_observer:create_observer.
No robot, camera or policy objects are created by this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import threading
import time

import numpy as np
from PIL import Image

from image_preprocessing import preprocess_rgb, preprocessing_contract


JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_yaw", "wrist_roll", "gripper",
]
DEFAULT_ROOT = Path(__file__).resolve().parent / "runs/act2_224_20261108/feedback"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class InferenceFeedbackObserver:
    """Copy selected frames in the callback, then perform all I/O on workers."""

    def __init__(self, root=None, trace_tag=None, *, text_budget_bytes=None,
                 image_budget_bytes=None, telemetry_queue_size=4096, image_queue_size=8):
        self.trace_tag = trace_tag or os.environ.get("WEBUI_RUN_TRACE_TAG", "unassigned")
        run_name = re.sub(r"[^A-Za-z0-9_-]", "_", self.trace_tag).strip("_")[:120] or "unassigned"
        base = Path(root or os.environ.get("ACT_FEEDBACK_ROOT", str(DEFAULT_ROOT))).resolve()
        if not base.is_relative_to(PROJECT_ROOT):
            raise ValueError("Feedback output must stay inside the ACT project.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory = base / run_name / f"observer_{stamp}_{os.getpid()}"
        self.directory.mkdir(parents=True)
        self.text_budget = text_budget_bytes if text_budget_bytes is not None else 128 * 1024 * 1024
        self.image_budget = image_budget_bytes if image_budget_bytes is not None else 512 * 1024 * 1024
        if min(self.text_budget, self.image_budget, telemetry_queue_size, image_queue_size) <= 0:
            raise ValueError("Feedback budgets and queue sizes must be positive.")
        self.telemetry = Queue(maxsize=telemetry_queue_size)
        self.images = Queue(maxsize=image_queue_size)
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.sequence = 0
        self.last_image224 = -math.inf
        self.last_image640 = -math.inf
        self.stats = {
            "observations_seen": 0, "actions_seen": 0, "events_seen": 0,
            "telemetry_written": 0, "telemetry_queue_dropped": 0,
            "telemetry_budget_dropped": 0, "telemetry_write_failed": 0,
            "image_jobs_queued": 0, "image_queue_dropped": 0,
            "image_budget_dropped": 0, "image_write_failed": 0,
            "images224_written": 0, "images640_written": 0,
            "invalid_images": 0, "text_bytes": 0, "image_bytes": 0,
            "errors": [],
        }
        self.trial_id = None
        metadata = {
            "schema_version": 1, "trace_tag": self.trace_tag,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "joint_order": JOINT_NAMES,
            "model_directory": os.environ.get("ACT_FEEDBACK_MODEL_DIRECTORY") or None,
            "model_weights_sha256": os.environ.get("ACT_FEEDBACK_MODEL_SHA256") or None,
            "image_preprocessing": preprocessing_contract(),
            "observed_position_semantics": "Values from the existing robot.get_observation callback, in degrees. Per-motor receipt age and validity are not supplied by this hook.",
            "performed_action_semantics": "Post-dispatch goal returned by robot.send_action, after direction/scale/limit handling. These are commanded motor targets, not measured positions.",
            "callback_time_semantics": "Local observer receipt time. Camera exposure time and cloud source-observation linkage are not supplied by this hook.",
            "policy_upload_identity_semantics": "Prepared RPC inputs carry trial, episode, TimedObservation timestep/timestamp and exact224 pixel hashes. Returned chunk source identities follow the pinned server timing contract. Preparation alone does not verify upload or model consumption.",
            "image224_max_hz": 2.0, "image640_max_hz": 0.5,
            "text_budget_bytes": self.text_budget, "image_budget_bytes": self.image_budget,
            "telemetry_queue_capacity": telemetry_queue_size, "image_queue_capacity": image_queue_size,
            "image_storage": "PNG RGB, lossless. Image budget includes PNG bytes and image index records.",
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        self.telemetry_thread = threading.Thread(target=self._write_telemetry, name="act-feedback-state", daemon=True)
        self.image_thread = threading.Thread(target=self._write_images, name="act-feedback-images", daemon=True)
        self.telemetry_thread.start()
        self.image_thread.start()

    def _increment(self, name, amount=1):
        with self.lock:
            self.stats[name] += amount

    def _error(self, stage, error):
        with self.lock:
            self.stats["errors"] = (self.stats["errors"] + [f"{stage}: {type(error).__name__}: {error}"])[-8:]

    def _enqueue(self, record):
        video = getattr(self, "trial_video", None)
        if video is not None:
            try:
                video.on_record(record)
            except Exception as error:
                self._error("trial_video", error)
        try:
            self.telemetry.put_nowait(record)
        except Full:
            self._increment("telemetry_queue_dropped")

    def on_event(self, record):
        """Record adapter lifecycle events alongside the callback trace."""
        if self.stopping.is_set():
            return
        if not isinstance(record, dict) or not isinstance(record.get("event"), str):
            raise ValueError("A feedback event requires a dict with a string event.")
        with self.lock:
            self.stats["events_seen"] += 1
            if "trial_id" in record:
                self.trial_id = record["trial_id"]
        self._enqueue({**record, "wall_time_ns": time.time_ns(),
                       "monotonic_ns": time.perf_counter_ns(), "record_kind": "lifecycle"})

    def _current_trial(self):
        with self.lock:
            return self.trial_id

    def on_observation(self, observation, *, task):
        if self.stopping.is_set():
            return
        now = time.perf_counter()
        self.sequence += 1
        sequence = self.sequence
        self._increment("observations_seen")
        fields = {f"{joint}.{kind}": finite_number(observation.get(f"{joint}.{kind}"))
                  for joint in JOINT_NAMES for kind in ("pos", "vel", "torque")}
        record = {
            "event": "observation", "sequence": sequence,
            "trial_id": self._current_trial(),
            "wall_time_ns": time.time_ns(), "monotonic_ns": time.perf_counter_ns(),
            "task": str(task), "fields": fields,
            "policy_timestep": None, "motor_feedback_freshness_verified": None,
            "image_manifest_sequence": sequence,
        }
        self._enqueue(record)
        take224 = now - self.last_image224 >= 0.5
        take640 = now - self.last_image640 >= 2.0
        if not (take224 or take640):
            return
        if take224:
            self.last_image224 = now
        if take640:
            self.last_image640 = now
        if self.images.full():
            self._increment("image_queue_dropped")
            return
        frames = {}
        for camera in ("front", "gripper"):
            image = observation.get(camera)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (480, 640, 3):
                self._increment("invalid_images")
                continue
            frames[camera] = image.copy()
        if not frames:
            return
        try:
            self.images.put_nowait({"sequence": sequence, "wall_time_ns": record["wall_time_ns"],
                                    "trial_id": record["trial_id"],
                                    "take224": take224, "take640": take640, "frames": frames})
            self._increment("image_jobs_queued")
        except Full:
            self._increment("image_queue_dropped")

    def on_policy_observation(self, observation, *, episode_id, trial_id):
        """Record the exact prepared RPC input identity, before its upload.

        The timestamp belongs to TimedObservation. It does not establish camera
        exposure time, motor RX age, upload success or server model consumption.
        A returned chunk can be joined using its first action's source identity.
        """
        if self.stopping.is_set():
            return
        values = observation.get_observation()
        images = {}
        for camera in ("front", "gripper"):
            frame = values.get(camera)
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.shape != (224, 224, 3)):
                raise ValueError(f"Prepared {camera} image must be RGB uint8 HWC224.")
            images[camera] = hashlib.sha256(frame.tobytes()).hexdigest()
        self.on_event({
            "event": "observation_upload_prepared", "trial_id": trial_id,
            "source_observation": {
                "episode_id": episode_id,
                "timestep": int(observation.get_timestep()),
                "timestamp": finite_number(observation.get_timestamp()),
            },
            "image224_rgb_uint8_sha256": images,
            "state_native_motor_deg": {
                f"{joint}.pos": finite_number(values.get(f"{joint}.pos"))
                for joint in JOINT_NAMES
            },
            "upload_success_verified": None,
            "model_consumption_verified": None,
        })

    def on_action(self, performed, *, timestep):
        if self.stopping.is_set():
            return
        self._increment("actions_seen")
        targets = {f"{joint}.pos": finite_number(performed.get(f"{joint}.pos"))
                   for joint in JOINT_NAMES} if isinstance(performed, dict) else None
        self._enqueue({
            "event": "performed_target", "wall_time_ns": time.time_ns(),
            "trial_id": self._current_trial(),
            "monotonic_ns": time.perf_counter_ns(), "timestep": timestep,
            "post_dispatch_target_motor_deg": targets,
            "measured_position_deg": None,
        })

    def _write_telemetry(self):
        last_status = 0.0
        while not self.stopping.is_set() or not self.telemetry.empty():
            try:
                record = self.telemetry.get(timeout=0.1)
            except Empty:
                continue
            try:
                encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                with self.lock:
                    budget_available = self.stats["text_bytes"] + len(encoded) <= self.text_budget
                if budget_available:
                    if record.get("record_kind") == "lifecycle":
                        filename = "trial_events.jsonl"
                    elif record["event"] == "observation":
                        filename = "observations.jsonl"
                    else:
                        filename = "executed_targets.jsonl"
                    with (self.directory / filename).open("ab") as stream:
                        stream.write(encoded)
                    self._increment("text_bytes", len(encoded))
                    self._increment("telemetry_written")
                else:
                    self._increment("telemetry_budget_dropped")
            except Exception as error:
                self._increment("telemetry_write_failed")
                self._error("telemetry", error)
            finally:
                self.telemetry.task_done()
            if time.monotonic() - last_status >= 1.0:
                self._write_status()
                last_status = time.monotonic()
        self._write_status()

    def _save_png(self, job, camera, kind, image):
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
        encoded = buffer.getvalue()
        relative = Path(f"images{kind}") / camera / f"observation_{job['sequence']:08d}.png"
        item = {"sequence": job["sequence"], "wall_time_ns": job["wall_time_ns"],
                "trial_id": job.get("trial_id"),
                "camera": camera, "kind": kind, "shape_hwc": list(image.shape),
                "file": relative.as_posix(), "rgb_uint8_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        index_bytes = (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
        byte_count = len(encoded) + len(index_bytes)
        with self.lock:
            budget_available = self.stats["image_bytes"] + byte_count <= self.image_budget
        if not budget_available:
            self._increment("image_budget_dropped")
            return
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".png.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)
        self._increment("image_bytes", byte_count)
        with (self.directory / "image_frames.jsonl").open("ab") as stream:
            stream.write(index_bytes)
        self._increment(f"images{kind}_written")

    def _write_images(self):
        while not self.stopping.is_set() or not self.images.empty():
            try:
                job = self.images.get(timeout=0.1)
            except Empty:
                continue
            try:
                for camera, frame in job["frames"].items():
                    if job["take224"]:
                        self._save_png(job, camera, "224", preprocess_rgb(frame))
                    if job["take640"]:
                        self._save_png(job, camera, "640", frame)
            except Exception as error:
                self._increment("image_write_failed")
                self._error("images", error)
            finally:
                self.images.task_done()
        self._write_status()

    def snapshot(self):
        with self.lock:
            stats = {**self.stats, "errors": list(self.stats["errors"])}
        return {
            "schema_version": 1, "trace_tag": self.trace_tag, "directory": str(self.directory),
            "stopping": self.stopping.is_set(),
            "telemetry_writer_alive": self.telemetry_thread.is_alive(),
            "image_writer_alive": self.image_thread.is_alive(),
            "telemetry_pending": self.telemetry.qsize(), "images_pending": self.images.qsize(),
            **stats,
        }

    def _write_status(self):
        try:
            # The status writer is serialized because both worker threads publish it.
            with self.status_lock:
                with self.lock:
                    record = {**self.stats, "errors": list(self.stats["errors"]),
                              "stopping": self.stopping.is_set(),
                              "telemetry_pending": self.telemetry.qsize(), "images_pending": self.images.qsize()}
                temporary = self.directory / "status.json.tmp"
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary.replace(self.directory / "status.json")
        except Exception as error:
            self._error("status", error)

    def close(self, timeout=2.0):
        self.stopping.set()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in (self.telemetry_thread, self.image_thread):
            thread.join(max(0.0, deadline - time.monotonic()))
        return self.snapshot()


def create_observer():
    return InferenceFeedbackObserver()


class _ComposedObserver:
    """Preserve vendor preview hooks while isolating each observer's failures."""

    def __init__(self, existing, feedback, *, on_observation=None):
        self.existing = existing
        self.feedback = feedback
        self.observation_callback = on_observation

    def _call(self, method, *args, **kwargs):
        for observer in (self.existing, self.feedback):
            handler = getattr(observer, method, None)
            if callable(handler):
                try:
                    handler(*args, **kwargs)
                except Exception as error:
                    self.feedback._error(f"{type(observer).__name__}.{method}", error)

    def on_observation(self, observation, *, task):
        self._call("on_observation", observation, task=task)
        if self.observation_callback is not None:
            try:
                self.observation_callback(observation, task=task)
            except Exception as error:
                self.feedback._error("return_phase_monitor", error)

    def on_policy_observation(self, observation, *, episode_id, trial_id):
        self._call("on_policy_observation", observation,
                   episode_id=episode_id, trial_id=trial_id)

    def on_action(self, performed, *, timestep):
        self._call("on_action", performed, timestep=timestep)

    def on_event(self, record):
        self._call("on_event", record)

    def close(self, timeout=2.0):
        # The vendor owns the original preview observer and its socket lifecycle.
        return self.feedback.close(timeout=timeout)

    def snapshot(self):
        return self.feedback.snapshot()


def compose_observer(existing, feedback, *, on_observation=None):
    return _ComposedObserver(existing, feedback, on_observation=on_observation)
