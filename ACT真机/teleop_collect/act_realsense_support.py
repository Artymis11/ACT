"""RealSense serial sources for SenseCraft camera discovery, preview and config."""

from __future__ import annotations

import base64
import logging
import math


PREFIX = "intelrealsense://"
logger = logging.getLogger("webui-backend.realsense")


def is_realsense_source(source):
    return isinstance(source, str) and source.startswith(PREFIX)


def serial_from_source(source):
    if not is_realsense_source(source):
        raise ValueError("Expected an intelrealsense:// serial source")
    serial = source[len(PREFIX):]
    if not serial or any(not (char.isascii() and (char.isalnum() or char in "-_.")) for char in serial):
        raise ValueError("Invalid RealSense serial number")
    return serial


def _modes_for_device(device):
    import pyrealsense2 as rs

    modes = {}
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            if profile.format() not in (rs.format.rgb8, rs.format.bgr8):
                continue
            video = profile.as_video_stream_profile()
            key = (video.width(), video.height(), profile.fps())
            modes[key] = {"width": key[0], "height": key[1], "fps": key[2], "max_fps": key[2]}
    return sorted(modes.values(), key=lambda item: (item["width"], item["height"], item["fps"]))


def _preferred_mode(modes):
    if not modes:
        raise RuntimeError("RealSense device has no RGB8/BGR8 color stream")
    return min(modes, key=lambda item: (
        abs(item["width"] - 640) + abs(item["height"] - 480),
        abs(item["fps"] - 30),
    ))


def list_camera_options(*, probe=False):
    """Enumerate SDK devices, optionally obtaining a real preview frame."""
    try:
        import pyrealsense2 as rs

        devices = rs.context().query_devices()
        options = []
        for device in devices:
            try:
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                modes = _modes_for_device(device)
                preferred = _preferred_mode(modes)
                option = {
                    "index_or_path": PREFIX + serial,
                    "display_name": f"{name} · {serial}",
                    "device_path": PREFIX + serial,
                    "stable_id": "intelrealsense:" + serial,
                    "backend": "intelrealsense",
                    "width": preferred["width"], "height": preferred["height"],
                    "fps": preferred["fps"], "modes": modes,
                    "is_usb": True, "auto_bind_eligible": True,
                    "detail": "RealSense SDK color profiles; serial-bound source",
                }
                if probe:
                    preview = probe_camera_source(option["index_or_path"])
                    if preview is None:
                        continue
                    option.update(preview)
                    option["display_name"] = f"{name} · {serial}"
                options.append(option)
            except Exception as exc:
                logger.warning("RealSense device enumeration unavailable: %s", exc)
        return options
    except ImportError:
        return []
    except Exception as exc:
        logger.warning("RealSense enumeration unavailable: %s", exc)
        return []


def camera_modes(source):
    serial = serial_from_source(source)
    import pyrealsense2 as rs

    for device in rs.context().query_devices():
        if device.get_info(rs.camera_info.serial_number) == serial:
            return _modes_for_device(device)
    return []


class RealSenseCapture:
    """Small VideoCapture-compatible color reader, with deferred mode application."""

    def __init__(self, source):
        self.serial = serial_from_source(source)
        self.pipeline = None
        self.actual_profile = None
        self.frame_info = None
        self.last_error = None
        self._released = False
        self._dirty = True
        self._present = False
        self._requested = {"width": 640, "height": 480, "fps": 30}
        try:
            modes = camera_modes(source)
            preferred = _preferred_mode(modes)
            self._requested = {key: preferred[key] for key in self._requested}
            self._present = True
        except Exception as exc:
            self.last_error = str(exc)

    def isOpened(self):
        return self._present and not self._released

    def _ensure_started(self):
        if not self.isOpened():
            raise RuntimeError(self.last_error or "RealSense capture is closed")
        if self.pipeline is not None and not self._dirty:
            return
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self._requested["width"],
                             self._requested["height"], rs.format.bgr8, self._requested["fps"])
        profile = pipeline.start(config)
        video = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.pipeline = pipeline
        self.actual_profile = {"width": video.width(), "height": video.height(),
                               "fps": video.fps(), "format": str(video.format()), "color_mode": "BGR"}
        self._dirty = False

    def read(self):
        try:
            import numpy as np

            self._ensure_started()
            frames = self.pipeline.wait_for_frames(2000)
            frame = frames.get_color_frame()
            if not frame:
                return False, None
            self.frame_info = {"frame_number": frame.get_frame_number(),
                               "timestamp_ms": frame.get_timestamp(),
                               "timestamp_domain": str(frame.get_frame_timestamp_domain())}
            return True, np.asanyarray(frame.get_data()).copy()
        except Exception as exc:
            self.last_error = str(exc)
            return False, None

    def set(self, property_id, value):
        key = {3: "width", 4: "height", 5: "fps"}.get(property_id)
        if key is None or not self.isOpened() or not math.isfinite(value) or value <= 0:
            return False
        parsed = int(round(value))
        if self._requested[key] != parsed:
            self._requested[key] = parsed
            self._dirty = True
        return True

    def get(self, property_id):
        key = {3: "width", 4: "height", 5: "fps"}.get(property_id)
        if key is None:
            return 0.0
        try:
            self._ensure_started()
            return float(self.actual_profile[key])
        except Exception as exc:
            self.last_error = str(exc)
            return 0.0

    def getBackendName(self):
        return "INTELREALSENSE"

    def release(self):
        pipeline = self.pipeline
        self.pipeline = None
        self._released = True
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                self.last_error = str(exc)
                logger.warning("RealSense release unavailable serial=%s error=%s", self.serial, exc)


def read_preview_frame(capture):
    """Warm the SDK color stream without importing the backend camera stack."""
    frame = None
    for _ in range(4):
        ok, frame = capture.read()
        if not ok:
            return False, None
    return True, frame


def probe_camera_source(source, *, width=None, height=None, fps=None):
    """Return a frame-backed option with actual SDK profile values."""
    import cv2

    capture = RealSenseCapture(source)
    try:
        for prop, value in ((3, width), (4, height), (5, fps)):
            if value is not None:
                capture.set(prop, float(value))
        ok, frame = read_preview_frame(capture)
        if not ok:
            logger.warning("RealSense preview unavailable serial=%s error=%s", capture.serial, capture.last_error)
            return None
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            return None
        profile = capture.actual_profile
        return {
            "index_or_path": source, "display_name": f"RealSense {capture.serial}",
            "device_path": source, "stable_id": "intelrealsense:" + capture.serial,
            "backend": "intelrealsense", "width": profile["width"],
            "height": profile["height"], "fps": profile["fps"],
            "modes": camera_modes(source), "is_usb": True, "auto_bind_eligible": True,
            "detail": f"RealSense SDK color BGR8 · {profile['width']}x{profile['height']} @ {profile['fps']} FPS",
            "preview_image_data_url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
        }
    finally:
        capture.release()
