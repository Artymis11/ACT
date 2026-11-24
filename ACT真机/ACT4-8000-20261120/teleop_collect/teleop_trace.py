"""Observe the installed 102/RS teleoperation chain without issuing device calls.

The writer records Python method inputs and returns. A successful send_mit
return does not establish a CAN acknowledgement or physical motor movement.
"""
from __future__ import annotations

import atexit
from datetime import datetime
from functools import wraps
from importlib import metadata
import itertools
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "local" / "teleop-reports"
MAX_TRACE_BYTES = 128 * 1024 * 1024
_CLOSE_RECORD_RESERVE_BYTES = 512
_local = threading.local()
_cycles = itertools.count(1)
_calls = itertools.count(1)
_installed = None


class _Recorder:
    def __init__(self, output_dir, capacity=4096, max_bytes=MAX_TRACE_BYTES):
        if max_bytes < 1024:
            raise ValueError("trace size limit must be at least 1024 bytes")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = output_dir / f"teleop_trace_{stamp}_{os.getpid()}.jsonl"
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        self._queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.written = 0
        self.bytes_written = 0
        self.max_bytes = max_bytes
        self._close_reason = "normal_shutdown"
        self.error = None
        self._closed = False
        self._thread = threading.Thread(target=self._write, name="act-teleop-trace", daemon=True)
        self._thread.start()

    def emit(self, event):
        if self._closed or self.error is not None:
            self.dropped += 1
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        last_flush = time.perf_counter()
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._closed:
                        break
                    self._handle.flush()
                    continue
                if item is None:
                    break
                line = json.dumps(item, ensure_ascii=False, default=str) + "\n"
                line_bytes = len(line.encode("utf-8"))
                if self.bytes_written + line_bytes + _CLOSE_RECORD_RESERVE_BYTES > self.max_bytes:
                    self._close_reason = "size_limit"
                    self._closed = True
                    self.dropped += 1
                    while True:
                        try:
                            pending = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if pending is not None:
                            self.dropped += 1
                    break
                self._handle.write(line)
                self.bytes_written += line_bytes
                self.written += 1
                if time.perf_counter() - last_flush >= 0.5:
                    self._handle.flush()
                    last_flush = time.perf_counter()
            closing_line = json.dumps({
                "event": "trace_closed", "written_events": self.written,
                "dropped_events": self.dropped, "writer_error": self.error,
                "reason": self._close_reason,
                "utf8_bytes_before_close": self.bytes_written,
                "max_utf8_bytes": self.max_bytes,
                "drop_count_scope": "through trace close",
            }) + "\n"
            self._handle.write(closing_line)
            self.bytes_written += len(closing_line.encode("utf-8"))
            self._handle.flush()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                print(f"[ACT_TELEOP_TRACE_WRITE_FAILED] {self.error}", file=sys.stderr, flush=True)
            except Exception:
                pass
        finally:
            self._handle.close()

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=2.0)


def _emit(recorder, event, **fields):
    """Keep observation failures outside the original method's control flow."""
    try:
        recorder.emit({
            "event": event,
            "qpc_ns": time.perf_counter_ns(),
            "thread_id": threading.get_ident(),
            "cycle_id": getattr(_local, "cycle_id", None),
            "follower_call_id": getattr(_local, "follower_call_id", None),
            **fields,
        })
    except Exception:
        pass


def _mapping(value):
    try:
        return dict(value)
    except Exception:
        return {"snapshot_unavailable": True}


def _error(exc):
    try:
        return {"type": type(exc).__name__, "message": str(exc)}
    except Exception:
        return {"type": "exception_description_unavailable"}


def _install_on_classes(leader_class, follower_class, motor_class, recorder):
    original_get_action = leader_class.get_action
    original_read_positions = leader_class._read_raw_positions
    original_send_action = follower_class.send_action
    original_send_mit = motor_class.send_mit
    original_get_state = motor_class.get_state

    @wraps(original_read_positions)
    def read_positions(self, *args, **kwargs):
        if getattr(self, "name", None) != "rebot_arm_102_leader":
            return original_read_positions(self, *args, **kwargs)
        try:
            result = original_read_positions(self, *args, **kwargs)
        except BaseException as exc:
            _emit(recorder, "leader_positions_error", error=_error(exc))
            raise
        _emit(recorder, "leader_positions_deg", positions_deg=_mapping(result),
              source="SDK monitor angles after SDK reliability filtering")
        return result

    @wraps(original_get_action)
    def get_action(self, *args, **kwargs):
        if getattr(self, "name", None) != "rebot_arm_102_leader":
            return original_get_action(self, *args, **kwargs)
        _local.cycle_id = next(_cycles)
        _emit(recorder, "leader_get_action_enter")
        try:
            result = original_get_action(self, *args, **kwargs)
        except BaseException as exc:
            _emit(recorder, "leader_action_error", error=_error(exc))
            raise
        _emit(recorder, "leader_action_deg", action_deg=_mapping(result))
        return result

    @wraps(original_send_action)
    def send_action(self, action, *args, **kwargs):
        if getattr(self, "name", None) != "seeed_b601_rs_follower":
            return original_send_action(self, action, *args, **kwargs)
        previous_call = getattr(_local, "follower_call_id", None)
        previous_names = getattr(_local, "motor_names", None)
        _local.follower_call_id = next(_calls)
        try:
            _local.motor_names = {id(motor): name for name, motor in self.motors.items()}
        except Exception:
            _local.motor_names = {}
        _emit(recorder, "follower_send_action_enter", action_deg=_mapping(action))
        try:
            result = original_send_action(self, action, *args, **kwargs)
        except BaseException as exc:
            _emit(recorder, "follower_send_action_error", error=_error(exc))
            raise
        else:
            _emit(recorder, "follower_send_action_return", mapped_action_deg=_mapping(result))
            return result
        finally:
            _local.follower_call_id = previous_call
            _local.motor_names = previous_names

    def active_motor_name(motor):
        names = getattr(_local, "motor_names", None)
        return names.get(id(motor)) if names else None

    @wraps(original_send_mit)
    def send_mit(self, *args, **kwargs):
        motor_name = active_motor_name(self)
        if motor_name is None:
            return original_send_mit(self, *args, **kwargs)
        parameters = dict(zip(("pos_rad", "vel_rad_s", "kp", "kd", "tau_nm"), args))
        parameters.update(kwargs)
        _emit(recorder, "motor_send_mit_enter", motor=motor_name, parameters=parameters)
        try:
            result = original_send_mit(self, *args, **kwargs)
        except BaseException as exc:
            _emit(recorder, "motor_send_mit_error", motor=motor_name, error=_error(exc))
            raise
        _emit(recorder, "motor_send_mit_return", motor=motor_name,
              evidence="Python SDK method returned normally; hardware receipt unverified")
        return result

    @wraps(original_get_state)
    def get_state(self, *args, **kwargs):
        result = original_get_state(self, *args, **kwargs)
        motor_name = active_motor_name(self)
        if motor_name is not None:
            try:
                snapshot = None if result is None else {
                    key: getattr(result, key, None)
                    for key in ("can_id", "arbitration_id", "status_code", "pos", "vel", "torq", "t_mos", "t_rotor")
                }
                _emit(recorder, "motor_cached_state_return", motor=motor_name, state=snapshot,
                      evidence="Existing SDK get_state call; RX freshness unavailable")
            except Exception:
                pass
        return result

    leader_class._read_raw_positions = read_positions
    leader_class.get_action = get_action
    follower_class.send_action = send_action
    motor_class.send_mit = send_mit
    motor_class.get_state = get_state


def install(output_dir=None):
    """Install process-local observers once and return their preparation status."""
    global _installed
    if _installed is not None:
        return dict(_installed)
    try:
        from lerobot_teleoperator_rebot_arm_102.rebot_arm_102_leader import RebotArm102Leader
        from lerobot_robot_seeed_b601.seeed_b601_follower import SeeedB601FollowerBase
        from motorbridge.core import Motor

        recorder = _Recorder(output_dir or DEFAULT_OUTPUT_DIR)
        atexit.register(recorder.close)
        _install_on_classes(RebotArm102Leader, SeeedB601FollowerBase, Motor, recorder)
        versions = {}
        for distribution in ("lerobot-teleoperator-rebot-arm-102", "lerobot-robot-seeed-b601", "motorbridge", "motorbridge-smart-servo"):
            try:
                versions[distribution] = metadata.version(distribution)
            except Exception:
                versions[distribution] = None
        _emit(recorder, "trace_metadata", pid=os.getpid(), versions=versions,
              executable=sys.executable, argv=list(sys.argv),
              scope="102 get_action to RS send_action and existing Motor method calls",
              max_utf8_bytes=recorder.max_bytes,
              motor_receipt_verified=False, additional_hardware_calls=0)
        _installed = {"installed": True, "path": str(recorder.path)}
        print(f"[ACT_TELEOP_TRACE] {recorder.path}", flush=True)
    except Exception as exc:
        _installed = {"installed": False, "error": _error(exc)}
        try:
            print(f"[ACT_TELEOP_TRACE_UNAVAILABLE] {_installed['error']}", file=sys.stderr, flush=True)
        except Exception:
            pass
    return dict(_installed)


def _self_test():
    class Sink:
        def __init__(self):
            self.events = []
            self.fail = False
        def emit(self, event):
            if self.fail:
                raise OSError("simulated log failure")
            self.events.append(event)

    sentinel = object()
    failure = RuntimeError("original failure")

    class Leader:
        name = "rebot_arm_102_leader"
        def _read_raw_positions(self):
            return {"joint": 12.0}
        def get_action(self):
            self._read_raw_positions()
            return sentinel

    class Motor:
        calls = 0
        fail = False
        def send_mit(self, *args, **kwargs):
            self.calls += 1
            if self.fail:
                raise failure
            return sentinel
        def get_state(self):
            return None

    motor = Motor()

    class Follower:
        name = "seeed_b601_rs_follower"
        motors = {"joint": motor}
        def send_action(self, action):
            assert action is sentinel
            motor.get_state()
            return motor.send_mit(1, 2, 3, 4, 5)

    sink = Sink()
    _install_on_classes(Leader, Follower, Motor, sink)
    assert Leader().get_action() is sentinel
    assert Follower().send_action(sentinel) is sentinel
    assert motor.calls == 1
    cycle_ids = {event["cycle_id"] for event in sink.events}
    assert len(cycle_ids) == 1 and None not in cycle_ids
    sink.fail = True
    assert Follower().send_action(sentinel) is sentinel
    assert motor.calls == 2
    motor.fail = True
    try:
        Follower().send_action(sentinel)
    except RuntimeError as exc:
        assert exc is failure
    else:
        raise AssertionError("original exception was swallowed")
    assert motor.calls == 3
    recorder = object.__new__(_Recorder)
    recorder._queue = queue.Queue(maxsize=1)
    recorder._closed = False
    recorder.error = None
    recorder.dropped = 0
    recorder.emit({"first": True})
    recorder.emit({"second": True})
    assert recorder.dropped == 1 and recorder._queue.qsize() == 1

    class MemoryHandle:
        def __init__(self):
            self.lines = []
            self.closed = False
        def write(self, line):
            self.lines.append(line)
        def flush(self):
            pass
        def close(self):
            self.closed = True

    bounded = object.__new__(_Recorder)
    bounded._queue = queue.Queue(maxsize=12)
    bounded._handle = MemoryHandle()
    bounded._closed = False
    bounded.error = None
    bounded.dropped = 0
    bounded.written = 0
    bounded.bytes_written = 0
    bounded.max_bytes = 1024
    bounded._close_reason = "normal_shutdown"
    for index in range(12):
        bounded.emit({"index": index, "payload": "界" * 60})
    bounded._write()
    contents = "".join(bounded._handle.lines)
    assert len(contents.encode("utf-8")) == bounded.bytes_written <= 1024
    assert json.loads(bounded._handle.lines[-1])["reason"] == "size_limit"
    assert bounded._closed and bounded._handle.closed and bounded._queue.empty()
    previous_dropped = bounded.dropped
    bounded.emit({"after_limit": True})
    assert bounded.dropped == previous_dropped + 1
    assert "".join(bounded._handle.lines) == contents
    print(json.dumps({"self_test": "passed", "hardware_opened": False,
                      "checks": ["return_identity", "single_original_call", "cycle_correlation", "logging_failure_isolation", "exception_identity", "bounded_queue_drop", "utf8_file_size_limit", "size_limit_close_record", "reject_after_size_limit"]}))


if __name__ == "__main__":
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("Run --self-test for the in-memory diagnostic checks.")
    _self_test()
