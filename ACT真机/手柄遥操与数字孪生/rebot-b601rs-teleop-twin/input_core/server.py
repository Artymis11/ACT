"""Local offline gamepad preview. Only XInput reads and an in-memory fake backend.

This program has no CAN, motor, robot, ACT, device-write or vibration API.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from control_core import Engine, Frame

ROOT = Path(__file__).resolve().parent
MODE = "offline_fake_backend"
BUILD_FILES = ("server.py", "control_core.py", "web/index.html", "web/styles.css", "web/app.js")


def build_revision():
    content = "".join(f"{name}={hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}\n" for name in BUILD_FILES)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class XInputReader:
    """Fixed slot, read-only; never switches to another controller on loss."""

    class Gamepad(ctypes.Structure):
        _fields_ = [
            ("buttons", ctypes.c_uint16),
            ("lt", ctypes.c_uint8), ("rt", ctypes.c_uint8),
            ("lx", ctypes.c_int16), ("ly", ctypes.c_int16),
            ("rx", ctypes.c_int16), ("ry", ctypes.c_int16),
        ]

    def __init__(self, slot):
        self.slot = slot
        self.error = ""
        self.read_fn = None
        gamepad_type = self.Gamepad

        class State(ctypes.Structure):
            _fields_ = [("packet", ctypes.c_uint32), ("gamepad", gamepad_type)]

        self.state_type = State
        if os.name != "nt":
            self.error = "实时手柄读取需要 Windows XInput"
            return
        try:
            dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "xinput1_4.dll"
            self.library = ctypes.WinDLL(str(dll))
            self.read_fn = self.library.XInputGetState
            self.read_fn.argtypes = [ctypes.c_uint32, ctypes.POINTER(State)]
            self.read_fn.restype = ctypes.c_uint32
            assert ctypes.sizeof(gamepad_type) == 12 and ctypes.sizeof(State) == 16
        except (OSError, AssertionError) as exc:
            self.error = f"XInput 读取不可用：{exc}"
            self.read_fn = None

    @staticmethod
    def normalize(value):
        return value / (32767.0 if value >= 0 else 32768.0)

    def read(self):
        if self.read_fn is None:
            return Frame(connected=False, acquired_at=time.monotonic())
        state = self.state_type()
        result = self.read_fn(self.slot, ctypes.byref(state))
        acquired = time.monotonic()
        if result != 0:
            self.error = f"手柄槽位 {self.slot} 未连接（XInput {result}）"
            return Frame(connected=False, acquired_at=acquired)
        self.error = ""
        pad = state.gamepad
        return Frame(
            connected=True, packet=int(state.packet), buttons=int(pad.buttons),
            axes={"LX": self.normalize(pad.lx), "LY": self.normalize(pad.ly),
                  "RX": self.normalize(pad.rx), "RY": self.normalize(pad.ry),
                  "LT": pad.lt / 255.0, "RT": pad.rt / 255.0},
            acquired_at=acquired,
        )


class PreviewRuntime:
    def __init__(self, slot, runtime_dir):
        self.session_id = secrets.token_hex(16)
        self.started_at = utc_now()
        self.started = time.monotonic()
        self.build_revision = build_revision()
        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.reader = XInputReader(slot)
        self.engine = Engine()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.poll_error = ""
        self.log_path = self.runtime_dir / f"session_{self.session_id}.jsonl"
        self.log_handler = RotatingFileHandler(self.log_path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
        self.logger = logging.getLogger(f"offline_preview.{self.session_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.addHandler(self.log_handler)
        self.evidence = {"source": "Windows XInputGetState", "successful_reads": 0,
                         "disconnected_reads": 0, "observed_button_mask": 0,
                         "axis_ranges": {}, "intent_frames": 0,
                         "selected_joints_seen": [], "gates_seen": []}
        self.thread = threading.Thread(target=self.poll, name="xinput-read-only", daemon=True)
        self.url = ""

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            state = self.engine.snapshot(now)
            state.update(session_id=self.session_id, mode=MODE, can_opened=False,
                         started_at=self.started_at, uptime_seconds=round(now-self.started, 2),
                         log_file=str(self.log_path), input_error=self.reader.error,
                         poll_error=self.poll_error, real_input_evidence=copy.deepcopy(self.evidence),
                         build_revision=self.build_revision)
            state["input"]["slot"] = self.reader.slot
            state["events"] = [dict(event, time=round(max(0.0, event["time"]-self.started), 3))
                               for event in state["events"]]
            return state

    def poll(self):
        previous = None
        last_log = 0.0
        while not self.stop_event.is_set():
            try:
                frame = self.reader.read()
                now = time.monotonic()
                with self.lock:
                    self.engine.update(frame, now)
                    self.poll_error = ""
                    state = self.engine.snapshot(now)
                    evidence = self.evidence
                    evidence["successful_reads" if frame.connected else "disconnected_reads"] += 1
                    if frame.connected:
                        evidence["observed_button_mask"] |= frame.buttons
                        for axis, value in frame.axes.items():
                            bounds = evidence["axis_ranges"].setdefault(axis, {"min": value, "max": value})
                            bounds["min"] = min(bounds["min"], value)
                            bounds["max"] = max(bounds["max"], value)
                    control = state["control"]
                    if any(control["joint_velocity_rad_s"]) or control["gripper_intent"] != "hold":
                        evidence["intent_frames"] += 1
                    if control["selected_joint"] not in evidence["selected_joints_seen"]:
                        evidence["selected_joints_seen"].append(control["selected_joint"])
                    if control["gate"] not in evidence["gates_seen"]:
                        evidence["gates_seen"].append(control["gate"])
                    state = self.snapshot()
                signature = json.dumps({"input": {k:v for k,v in state["input"].items() if k not in ("age_ms", "packet")},
                                        "control": state["control"]}, sort_keys=True)
                if signature != previous or now-last_log >= 1.0:
                    self.logger.info(json.dumps({"at": utc_now(), "elapsed": round(now-self.started, 3),
                                                 "state": state}, ensure_ascii=False, allow_nan=False))
                    previous = signature
                    last_log = now
            except Exception as exc:
                with self.lock:
                    self.poll_error = f"读取循环异常：{type(exc).__name__}: {exc}"
                    self.engine.update(Frame(connected=False, acquired_at=time.monotonic()), time.monotonic())
                    self.engine.set_paused(True)
            self.stop_event.wait(0.02)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        with self.lock:
            self.engine.set_paused(True)
            state = self.snapshot()
        (self.runtime_dir / f"final_{self.session_id}.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        self.log_handler.close()
        self.logger.removeHandler(self.log_handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "OfflineGamepadPreview/1.0"

    @property
    def runtime(self):
        return self.server.runtime

    def log_message(self, *_args):
        pass

    def host_allowed(self):
        return self.headers.get("Host", "") == self.server.server_address[0]+":"+str(self.server.server_address[1])

    def reply(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        if not self.host_allowed():
            return self.reply(403, {"error": "仅接受本机预览地址"})
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            return self.reply(200, self.runtime.snapshot())
        if path == "/api/health":
            return self.reply(200, {"mode": MODE, "session_id": self.runtime.session_id,
                                    "can_opened": False, "url": self.runtime.url,
                                    "build_revision": self.runtime.build_revision})
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/index.html": ("index.html", "text/html; charset=utf-8"),
                  "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if path in assets:
            name, mime = assets[path]
            try:
                return self.reply(200, (ROOT / "web" / name).read_bytes(), mime)
            except FileNotFoundError:
                return self.reply(503, {"error": "预览页面文件尚未就绪"})
        return self.reply(404, {"error": "不存在的离线预览接口"})

    def do_POST(self):
        if not self.host_allowed():
            return self.reply(403, {"error": "仅接受本机预览地址"})
        if self.path != "/api/action":
            return self.reply(404, {"error": "不存在的离线预览接口"})
        if self.headers.get("Origin") not in (None, self.runtime.url.rstrip("/")):
            return self.reply(403, {"error": "请求来源不属于本次预览"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                return self.reply(400, {"error": "请求长度无效"})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求必须是对象")
        except (ValueError, UnicodeDecodeError):
            return self.reply(400, {"error": "请求格式无效"})
        if data.get("session_id") != self.runtime.session_id:
            return self.reply(409, {"error": "预览会话已更换，请刷新页面"})
        action = data.get("action")
        if action not in ("pause", "resume", "reset", "shutdown"):
            return self.reply(400, {"error": "仅支持暂停、继续、清空预览和退出"})
        with self.runtime.lock:
            if action == "pause" or action == "shutdown":
                self.runtime.engine.set_paused(True)
            elif action == "resume":
                self.runtime.engine.set_paused(False)
            elif action == "reset":
                if self.runtime.snapshot()["control"]["lb_held"]:
                    return self.reply(409, {"error": "请先松开 LB，再清空虚拟位移"})
                self.runtime.engine.reset_preview()
            self.runtime.logger.info(json.dumps({"at": utc_now(), "ui_action": action,
                                                 "session_id": self.runtime.session_id}, ensure_ascii=False))
            state = self.runtime.snapshot()
        self.reply(200, state)
        if action == "shutdown":
            threading.Thread(target=self.server.shutdown, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Offline XInput preview; no robot connection")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--slot", type=int, choices=range(4), default=0)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "runtime")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("invalid port")
    runtime = PreviewRuntime(args.slot, args.runtime_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.runtime = runtime
    runtime.url = f"http://127.0.0.1:{server.server_address[1]}/"
    active = {"mode": MODE, "url": runtime.url, "session_id": runtime.session_id,
              "pid": os.getpid(), "slot": args.slot, "started_at": runtime.started_at,
              "can_opened": False, "log_file": str(runtime.log_path),
              "build_revision": runtime.build_revision}
    (args.runtime_dir / "active.json").write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime.thread.start()
    print(json.dumps(active, ensure_ascii=False), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()
