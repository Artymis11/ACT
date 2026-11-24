"""Read-only, six-axis MotorBridge history for the explicitly selected history ACT."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import time

from timed_joint_history import ARM_KEYS, TimedJointHistory


ARM_AXES = tuple(key.removesuffix(".pos") for key in ARM_KEYS)
HISTORY_FIELDS = tuple(f"{key}.delta_6_frames" for key in ARM_KEYS)
POSITION_TIMEOUT_MS = 20
MAX_CURRENT_AGE_NS = 55_000_000
MAX_BRACKET_NS = 95_000_000
MAX_SAMPLE_WIDTH_NS = 15_000_000


class NativeHistoryReader:
    """One bounded batch at a time on the existing client's six distinct handles."""

    def __init__(self, robot, *, on_failure):
        self._motors = {axis: robot.motors[axis] for axis in ARM_AXES}
        self._hosts = {axis: int(robot.config.motor_can_ids[axis][1]) for axis in ARM_AXES}
        self._history = TimedJointHistory(started_ns=time.monotonic_ns(),
                                          source="MotorBridge 0x7019 request/reply interval")
        self._completed_sweeps = deque(maxlen=24)
        self._on_failure = on_failure
        self._samples_lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="act-native-history", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self, *, timeout_s):
        self._stop.set()
        if self._thread.ident is None:
            return True
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def _read_axis(self, axis):
        before = time.monotonic_ns()
        angle = float(self._motors[axis].robstride_get_param_f32_host_id(
            0x7019, self._hosts[axis], timeout_ms=POSITION_TIMEOUT_MS))
        after = time.monotonic_ns()
        if not math.isfinite(angle):
            raise ValueError(f"{axis} returned a non-finite native position.")
        return f"{axis}.pos", math.degrees(angle), before, after

    def _run(self):
        period = 1 / 30
        next_sample = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=len(ARM_AXES), thread_name_prefix="act-history-axis") as pool:
                while not self._stop.wait(max(0.0, next_sample - time.monotonic())):
                    sweep = tuple(pool.map(self._read_axis, ARM_AXES))
                    with self._samples_lock:
                        for sample in sweep:
                            self._history.record(*sample)
                        self._completed_sweeps.append(max(sample[3] for sample in sweep))
                    next_sample += period
                    now = time.monotonic()
                    if next_sample <= now:
                        next_sample = now + period
        except Exception as error:
            with self._samples_lock:
                self._error = error
            if not self._stop.is_set():
                self._on_failure(error)

    def observation_at(self, captured_at):
        with self._samples_lock:
            if self._error is not None:
                raise RuntimeError(f"Original MotorBridge history read failed: {self._error}") from self._error
            latest_sweep = next((ended for ended in reversed(self._completed_sweeps)
                                 if ended <= captured_at), None)
            if latest_sweep is None:
                raise ValueError("No complete six-axis read finished before image capture.")
            receipt = self._history.snapshot(
                latest_sweep, max_current_age_ns=MAX_CURRENT_AGE_NS - (captured_at - latest_sweep),
                max_bracket_ns=MAX_BRACKET_NS,
                max_sample_width_ns=MAX_SAMPLE_WIDTH_NS)
        receipt["capture_ns"] = captured_at
        receipt["capture_age_from_sweep_end_ns"] = captured_at - latest_sweep
        if receipt["status"] != "ready":
            reasons = {axis: evidence["reason"] for axis, evidence in receipt["axes"].items()
                       if "reason" in evidence}
            raise ValueError(f"Native 0.20 s history unavailable: {reasons}")
        values = dict(zip(ARM_KEYS + HISTORY_FIELDS, receipt["current_deg"] + receipt["deltas_deg"], strict=True))
        return values, receipt
