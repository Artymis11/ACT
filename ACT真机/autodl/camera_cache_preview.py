"""Publish wrist previews from the existing RealSense cache off the policy path."""

import math
import threading
import time

import numpy as np


class WristCameraCachePreview:
    """Copy only fresh wrist cache entries into the installed preview publisher."""

    def __init__(self, camera, publisher, shutdown_event, *, poll_hz=60.0,
                 max_age_ms=200.0, clock=time.monotonic, capture_clock=time.perf_counter):
        if not math.isfinite(poll_hz) or not 20 <= poll_hz <= 60:
            raise ValueError("Preview cache polling must be between 20 and 60 Hz.")
        if not math.isfinite(max_age_ms) or not 0 < max_age_ms <= 200:
            raise ValueError("Preview cache age must be positive and at most 200 ms.")
        self.camera = camera
        self.publisher = publisher
        self.shutdown_event = shutdown_event
        self.poll_hz = float(poll_hz)
        self.max_age_ms = float(max_age_ms)
        self.clock = clock
        # The installed RealSense loop stores latest_timestamp using perf_counter.
        self.capture_clock = capture_clock
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._last_timestamp = None
        self._counts = {"forwarded_cache_entries": 0, "duplicate_entries": 0,
                        "stale_entries": 0, "busy_cache": 0, "invalid_entries": 0,
                        "publisher_errors": 0}

    def start(self):
        """Restart after a trial shutdown clears, until explicit stop is called."""
        with self._lock:
            if self._stop.is_set() or self.shutdown_event.is_set():
                return False
            if self._thread is not None and self._thread.is_alive():
                return True
            self._thread = threading.Thread(target=self._run, name="act224-wrist-cache-preview", daemon=True)
            self._thread.start()
            return True

    def stop(self, timeout_s=1.0):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_s)))
        return thread is None or not thread.is_alive()

    def snapshot(self):
        with self._lock:
            return {**self._counts, "running": self._thread is not None and self._thread.is_alive(),
                    "last_forwarded_capture_timestamp": self._last_timestamp}

    def receipt(self, status="installed"):
        return {"status": status, "source": "existing gripper RealSense latest_color_frame cache",
                "published_role": "wrist", "source_poll_hz": self.poll_hz,
                "wrist_preview_fps_limit": 20, "max_cache_age_ms": self.max_age_ms,
                "front_cache_read": False, "hardware_reads": 0,
                "policy_observation_calls": 0,
                "shutdown": "Observe client shutdown_event and explicit stop(timeout_s=1)."}

    def publish_latest(self):
        """Attempt one non-blocking cache snapshot without invoking camera APIs."""
        if self._stop.is_set() or self.shutdown_event.is_set():
            return False
        frame_lock = self.camera.frame_lock
        if not frame_lock.acquire(blocking=False):
            with self._lock:
                self._counts["busy_cache"] += 1
            return False
        try:
            frame = self.camera.latest_color_frame
            timestamp = self.camera.latest_timestamp
            valid = (isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                     and frame.ndim == 3 and frame.shape[2] == 3
                     and frame.shape[0] > 0 and frame.shape[1] > 0)
            if valid and not isinstance(timestamp, bool):
                try:
                    timestamp = float(timestamp)
                    valid = math.isfinite(timestamp)
                except (TypeError, ValueError):
                    valid = False
            else:
                valid = False
            image = frame.copy() if valid else None
        finally:
            frame_lock.release()
        if not valid:
            with self._lock:
                self._counts["invalid_entries"] += 1
            return False
        age_ms = (self.capture_clock() - timestamp) * 1000
        with self._lock:
            if self._stop.is_set() or self.shutdown_event.is_set():
                return False
            if not 0 <= age_ms <= self.max_age_ms:
                self._counts["stale_entries"] += 1
                return False
            if self._last_timestamp is not None and timestamp <= self._last_timestamp:
                self._counts["duplicate_entries"] += 1
                return False
            self._last_timestamp = timestamp
        try:
            # PerRolePreviewPublisher retains the wrist20/front10 admission
            # clocks and replaces pending images by role. This adds no queue.
            self.publisher.on_observation({"wrist": image})
        except Exception:
            with self._lock:
                self._counts["publisher_errors"] += 1
            return False
        with self._lock:
            self._counts["forwarded_cache_entries"] += 1
        return True

    def _run(self):
        period = 1.0 / self.poll_hz
        next_tick = self.clock()
        while not self._stop.is_set() and not self.shutdown_event.is_set():
            now = self.clock()
            if now < next_tick:
                self._stop.wait(min(next_tick - now, 0.02))
                continue
            self.publish_latest()
            next_tick += period
            now = self.clock()
            if next_tick <= now:
                next_tick = now + period


def install_camera_cache_preview(client, *, poll_hz=60.0, max_age_ms=200.0,
                                 clock=time.monotonic, capture_clock=time.perf_counter):
    """Prepare after wrist pacing and before feedback observer composition.

    Return an unstarted worker and a JSON-compatible receipt. The client starts
    it after a successful cloud handshake and restarts it between managed trials.
    """
    existing = getattr(client, "_camera_cache_preview_worker", None)
    if isinstance(existing, WristCameraCachePreview):
        return existing, existing.receipt("already_installed")
    from wrist_preview import PerRolePreviewPublisher

    publisher = getattr(getattr(client, "_pipeline_observer", None), "_publisher", None)
    if not isinstance(publisher, PerRolePreviewPublisher):
        return None, {"status": "inactive", "reason": "Per-role preview pacing is not installed."}
    cameras = getattr(getattr(client, "robot", None), "cameras", {})
    camera = cameras.get("gripper")
    if camera is None or not all(hasattr(camera, name) for name in
                                 ("frame_lock", "latest_color_frame", "latest_timestamp")):
        return None, {"status": "inactive", "reason": "The existing gripper camera cache is unavailable."}
    shutdown_event = getattr(client, "shutdown_event", None)
    if shutdown_event is None or not callable(getattr(shutdown_event, "is_set", None)):
        return None, {"status": "inactive", "reason": "The client shutdown event is unavailable."}
    worker = WristCameraCachePreview(camera, publisher, shutdown_event, poll_hz=poll_hz,
                                    max_age_ms=max_age_ms, clock=clock, capture_clock=capture_clock)
    client._camera_cache_preview_worker = worker
    return worker, worker.receipt()
