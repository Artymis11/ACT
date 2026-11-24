"""Process-local wrist preview pacing for the installed SenseCraft observer."""

import asyncio
from contextlib import aclosing
from functools import wraps
import json
import queue
import struct
import threading
import time

import numpy as np


class PerRolePreviewPublisher:
    """Reuse the vendor sender with a single pending packet of latest images."""

    def __init__(self, publisher, *, clock=time.monotonic, telemetry_factory=None):
        self.publisher = publisher
        self.clock = clock
        self._lock = threading.Lock()
        self._next_due = {"front": None, "wrist": None}
        self._period = {"front": float(publisher._enqueue_interval_s), "wrist": 0.05}
        self._admitted = {"front": 0, "wrist": 0}
        self._replaced_pending = 0
        self._telemetry_factory = telemetry_factory

    def on_observation(self, observation):
        now = self.clock()
        with self._lock:
            sources = {}
            for role in ("front", "wrist"):
                if role == "wrist" and not self.publisher._wrist:
                    continue
                due = self._next_due[role]
                if due is not None and now + 1e-9 < due:
                    continue
                frame = observation.get(role)
                if role == "wrist" and frame is None:
                    for alias in ("side", "gripper"):
                        frame = observation.get(alias)
                        if frame is not None:
                            break
                if frame is None:
                    continue
                try:
                    frame = np.asarray(frame)
                    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                        continue
                    sources[role] = frame.copy()
                except (TypeError, ValueError):
                    continue
                period = self._period[role]
                # Accumulate deadlines so a 30 Hz caller admits 20 wrist frames
                # per second instead of falling to 15 Hz at a strict 50 ms gap.
                self._next_due[role] = (now + period if due is None or now - due > period * 4
                                        else due + period)
                self._admitted[role] += 1
            if not sources:
                return
            if self._telemetry_factory is not None:
                telemetry = self._telemetry_factory(observation)
                if telemetry is not None:
                    sources["_legacy_telemetry"] = telemetry
            pending_queue = self.publisher._queue
            try:
                pending = pending_queue.get_nowait()
            except queue.Empty:
                pending = None
            if pending is not None:
                # A wrist-only update retains an unsent front update. New values
                # replace matching roles, so slow sends cannot build a frame FIFO.
                sources = {**pending, **sources}
                self._replaced_pending += 1
            pending_queue.put_nowait(sources)

    def snapshot(self):
        with self._lock:
            return {"admitted_frames": dict(self._admitted),
                    "replaced_pending_packets": self._replaced_pending,
                    "pending_packets": self.publisher._queue.qsize()}


def install_wrist_preview(client, *, clock=time.monotonic):
    """Call immediately after the vendor client constructor, before feedback composition.

    Only the original run-preview observer is changed. Its existing sender, raw
    preview packet codec, transport destination and telemetry callback are reused.
    The legacy direct JPEG observer is upgraded to the same vendor queued sender.
    """
    observer = getattr(client, "_pipeline_observer", None)
    publisher = getattr(observer, "_publisher", None)
    if isinstance(publisher, PerRolePreviewPublisher):
        return {"status": "already_installed", **_receipt(publisher, "shared_sender")}
    path = "shared_sender"
    telemetry_factory = None
    if publisher is None and all(hasattr(observer, name) for name in
                                 ("_host", "_port", "_wrist", "_robot_type", "_close_socket")):
        from backend.run.preview import _RunPreviewObserver, _extract_run_sim_telemetry

        robot_type = observer._robot_type
        publisher = _legacy_queued_publisher(observer)
        telemetry_factory = lambda observation: _extract_run_sim_telemetry(observation, robot_type=robot_type)
        observer._close_socket()
        observer = _RunPreviewObserver(publisher=publisher, robot_type=observer._robot_type)
        client._pipeline_observer = observer
        path = "legacy_direct_sender_upgraded"
    if publisher is None or not all(hasattr(publisher, name) for name in
                                    ("_queue", "_wrist", "_enqueue_interval_s")):
        return {"status": "inactive", "reason": "No supported run-preview publisher is active."}
    if publisher._queue.maxsize != 1 or publisher._enqueue_interval_s <= 0:
        raise ValueError("The reviewed SenseCraft preview sender must have one pending packet and a positive interval.")
    paced = PerRolePreviewPublisher(publisher, clock=clock, telemetry_factory=telemetry_factory)
    observer._publisher = paced
    return {"status": "installed", **_receipt(paced, path)}


def _receipt(publisher, path):
    return {"path": path, "wrist_preview_fps_limit": 20,
            "front_preview_fps_limit": 1.0 / publisher._period["front"],
            "pending_packet_capacity": 1, "pending_policy": "latest_per_role",
            "hardware_profile_changed": False, "model_observation_changed": False}


def _legacy_queued_publisher(observer):
    from backend.recording.preview import build_raw_preview_packet
    from backend.recording.preview_publisher import CollectionPreviewPublisher

    class LegacyQueuedPublisher(CollectionPreviewPublisher):
        def _run_worker(self):
            while not self._stopping.is_set():
                try:
                    sources = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    packet = build_raw_preview_packet(sources)
                    telemetry = sources.get("_legacy_telemetry")
                    if telemetry is not None:
                        encoded = json.dumps(telemetry, ensure_ascii=False, separators=(",", ":"),
                                             default=str).encode("utf-8")
                        packet += b"WPV2" + struct.pack("!III", 0, 0, len(encoded)) + encoded
                except (TypeError, ValueError):
                    continue
                if self._send_packet(packet):
                    self._last_sent = time.monotonic()

    return LegacyQueuedPublisher(host=observer._host, port=observer._port,
                                 wrist_configured=observer._wrist)


def install_wrist_collection_preview(app, *, current_setup, wrist_source, preview_module=None,
                                     sleep=asyncio.sleep, clock=time.monotonic):
    """Pace only the bound wrist's idle stream through the vendor lease owner."""
    if preview_module is None:
        from backend.routers import preview as preview_module
    path = "/api/collection/camera-preview-source.mjpeg"
    route = next((route for route in app.routes
                  if getattr(route, "path", None) == path and "GET" in getattr(route, "methods", ())), None)
    if route is None or not hasattr(route, "dependant"):
        raise ValueError("The original SenseCraft collection preview route is missing.")
    original = route.dependant.call
    if getattr(original, "_act224_wrist_preview", False):
        return {"status": "already_installed", "path": path, "wrist_preview_fps_limit": 20}
    source = str(wrist_source).strip()

    @wraps(original)
    def wrist_preview(**kwargs):
        # Preserve all native validation, preview-disabled checks and role/source
        # precedence before replacing the still-unstarted original generator.
        response = original(**kwargs)
        role = str(kwargs.get("role") or "").strip().lower()
        requested = str(kwargs.get("source") or "").strip()
        is_wrist = role == "secondary" if role else requested == source
        if not is_wrist or str(current_setup().get("camera_wrist_index") or "").strip() != source:
            return response
        camera_index = preview_module.opencv_camera_source(source)
        if preview_module.is_virtual_camera_source(camera_index):
            return response
        options = {key: kwargs[key] for key in ("width", "height")
                   if isinstance(kwargs.get(key), (int, float)) and not isinstance(kwargs[key], bool)}
        options["fps"] = 30

        async def generate():
            frame_boundary = b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n"
            deadline = None
            try:
                try:
                    # This is the original subprocess owner. It performs the
                    # native lease check/registration and closes on aclose.
                    async with aclosing(preview_module._iter_subprocess_mjpeg_stream_frames(
                            camera_index, **options)) as stream:
                        async for frame in stream:
                            now = clock()
                            deadline = now if deadline is None or now - deadline > 0.2 else deadline
                            yield frame_boundary + frame + b"\r\n"
                            deadline += 0.05
                            delay = deadline - clock()
                            if delay > 0:
                                await sleep(delay)
                except preview_module.CameraPreviewSuspended:
                    preview_module.logger.info("collection wrist preview suspended source=%r", camera_index)
                except RuntimeError as error:
                    preview_module.logger.warning("collection wrist preview: %s source=%r", error, camera_index)
            finally:
                try:
                    await preview_module._release_source_camera_preview_capture(camera_index)
                except Exception as error:
                    preview_module.logger.warning("collection wrist preview release failed source=%r: %s",
                                                  camera_index, error)

        response.body_iterator = generate()
        return response

    wrist_preview._act224_wrist_preview = True
    route.dependant.call = wrist_preview
    return {"status": "installed", "path": path, "wrist_preview_fps_limit": 20,
            "camera_hardware_fps": 30, "lease_owner": "Original SenseCraft preview subprocess",
            "scope": "Bound wrist only"}
