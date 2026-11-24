"""Same-owner ACT224 stop coordination. This module imports only the standard library."""

import json
import math
from contextlib import contextmanager
from dataclasses import replace
import sys
import threading
import time


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_yaw", "wrist_roll")
ZERO_TOLERANCE_DEG = 1.0


def read_fresh_native_radians(robot, joint):
    _, host_id = robot.config.motor_can_ids[joint]
    read = getattr(robot.motors[joint], "robstride_get_param_f32_host_id", None)
    if not callable(read):
        raise RuntimeError("Fresh native mechanical-position reads are unavailable.")
    value = float(read(0x7019, int(host_id), timeout_ms=250))
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite native position for {joint}.")
    return value


@contextmanager
def fresh_native_homing_origins(robot):
    """Keep pinned native safe_zero while replacing its private cached start read.

    The installed 1.0.3 safe_zero local _read_motor_deg is the sole target.
    Normal temperature and gravity reads retain their original fast path.
    All policy writers are already joined before entering this context.
    """
    originals = {}
    reads = []
    try:
        for joint, motor in robot.motors.items():
            original = motor.get_state
            originals[joint] = original

            def get_state(joint=joint, original=original):
                state = original()
                if sys._getframe(1).f_code.co_name == "_read_motor_deg":
                    if state is None:
                        raise RuntimeError(f"Native feedback state is missing for {joint}.")
                    state = replace(state, pos=read_fresh_native_radians(robot, joint))
                    reads.append(joint)
                return state
            motor.get_state = get_state
        yield reads
    finally:
        for joint, original in originals.items():
            robot.motors[joint].get_state = original


def verify_native_zero(robot, *, cancelled=lambda: False, tolerance_deg=ZERO_TOLERANCE_DEG,
                       stable_s=1.0, timeout_s=5.0, clock=time.monotonic, sleep=time.sleep):
    """Require repeated fresh READ_PARAMETER replies, never cached get_state values.

    MotorBridge v0.5.1 get_parameter_with_host_id clears the parameter cache,
    sends READ_PARAMETER, and waits for a reply or raises on timeout.
    Source: motorbridge/motorbridge v0.5.1 motor_vendors/robstride/src/motor.rs:658-695.
    Register 0x7019 is native mechanical position in radians.
    """
    started = clock()
    stable_since = None
    sweeps = 0
    positions = {}
    while clock() - started < timeout_s:
        if cancelled() or getattr(robot, "_emergency_disable_requested", False):
            raise RuntimeError("Native zero verification was cancelled or emergency-disabled.")
        positions = {}
        for can_id, joint in enumerate(ARM_JOINTS, 1):
            mapped_id, host_id = robot.config.motor_can_ids[joint]
            if mapped_id != can_id:
                raise RuntimeError("Native homing verification requires the recorded CAN 1..6 mapping.")
            value = math.degrees(read_fresh_native_radians(robot, joint))
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite native zero feedback for {joint}.")
            positions[joint] = value
        sweeps += 1
        now = clock()
        if all(abs(value) <= tolerance_deg for value in positions.values()):
            if stable_since is None:
                stable_since = now
            if sweeps >= 3 and now - stable_since >= stable_s and now - started < timeout_s:
                return {"native_positions_deg": positions, "fresh_sweeps": sweeps,
                        "stable_seconds": now - stable_since, "tolerance_deg": tolerance_deg,
                        "feedback_source": "READ_PARAMETER 0x7019, fresh response per call"}
        else:
            stable_since = None
        sleep(0.02)
    raise RuntimeError(f"Native zero feedback did not stay within {tolerance_deg} degrees: {positions}")


def release_verified_robot(robot):
    """Disable through the native SDK before closing this owner's transport."""
    bus = robot.bus
    robot.disable_torque()  # Failure must propagate while the owner still has its handles.
    bus.close_bus()
    for motor in robot.motors.values():
        motor.close()
    bus.close()
    robot.bus = None
    camera_errors = []
    for name, camera in robot.cameras.items():
        try:
            camera.disconnect()
        except Exception as error:
            camera_errors.append(f"{name}: {error}")
    return camera_errors


class StopController:
    """Latch stop intent and finish on the existing control owner's thread."""

    def __init__(self, vendor, *, emit=print, verify=verify_native_zero,
                 release=release_verified_robot):
        self.vendor = vendor
        self.emit = emit
        self.verify = verify
        self.release = release
        self.client = None
        self.stop_requested = threading.Event()
        self.release_confirmed = threading.Event()
        self.action_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._intent = None
        self._phase = "running"
        self._hold_armed = False
        self._release_started = False
        self._finished = False
        self.released = False
        self.failure = ""

    def log(self, event, **values):
        self.emit("[act224-stop] " + json.dumps({"event": event, **values}, sort_keys=True))

    def attach(self, client):
        self.client = client

    def normal_stop_requested(self):
        with self._state_lock:
            return self.stop_requested.is_set() and self._intent == "home"

    def _interrupt(self):
        client = self.client
        if client is None:
            return
        event = getattr(client, "shutdown_event", None)
        if event is not None:
            event.set()
        sender = getattr(client, "_observation_sender", None)
        if sender is not None:
            sender.stop(timeout_s=0.0)
        trajectory = getattr(client, "_action_trajectory", None)
        stop_trajectory = getattr(trajectory, "stop", None)
        if callable(stop_trajectory):
            stop_trajectory()
        channel = getattr(client, "channel", None)
        if channel is not None:
            try:
                channel.close()
            except Exception as error:
                self.log("rpc_close_failed", error=str(error))

    def request_home(self):
        with self._state_lock:
            if self._phase in {"releasing", "finished"}:
                return
            if self._intent is None:
                self._intent = "home"
            self.stop_requested.set()
        self._interrupt()

    def request_hold(self, reason="Explicit damped hold or stdin EOF"):
        with self._state_lock:
            if self._phase in {"releasing", "finished"}:
                return
            self._intent = "hold"
            self.failure = self.failure or str(reason)
            self.stop_requested.set()
        self._interrupt()

    def confirm_release(self):
        with self._state_lock:
            if self._phase not in {"fault_hold", "fault_hold_failed"}:
                self.log("release_confirmation_ignored", phase=self._phase)
                return
            self.release_confirmed.set()
            if self._hold_armed:
                self.vendor.confirm_b601_rs_release()

    def dispatch(self, send, action):
        with self.action_lock:
            if self.stop_requested.is_set():
                return None
            return send(action)

    def _quiesce(self, client):
        self._interrupt()
        while not client._join_actuator_thread(timeout_s=2.0):
            self.request_hold("Actuator thread did not stop. Homing is deferred.")
            self.log("waiting_for_actuator_stop")
            time.sleep(0.05)
        sender = getattr(client, "_observation_sender", None)
        while sender is not None and not sender.stop(timeout_s=2.0):
            self.request_hold("Observation sender did not stop. Homing is deferred.")
            self.log("waiting_for_observation_sender_stop")
            time.sleep(0.05)
        trajectory = getattr(client, "_action_trajectory", None)
        if trajectory is not None:
            with client._action_trajectory_lock:
                stop_trajectory = getattr(trajectory, "stop", None)
                if callable(stop_trajectory):
                    stop_trajectory()
                else:
                    trajectory.reset()
        queue = getattr(client, "action_queue", None)
        if queue is not None:
            with client.action_queue_lock:
                while not queue.empty():
                    queue.get_nowait()
        failure = str(getattr(client, "_transport_failure", "") or "")
        if failure or getattr(client, "_actuator_failed", False):
            self.request_hold(failure or str(getattr(client, "_actuator_error", "Actuator failed")))

    def _cancelled(self):
        with self._state_lock:
            return self._intent != "home"

    def _home_and_release(self, robot):
        with self._state_lock:
            self._phase = "homing"
        original_send = robot.send_action

        def home_send(action):
            if self._cancelled():
                raise RuntimeError("Native homing cancelled for fault hold.")
            return original_send(action)

        robot.send_action = home_send
        try:
            self.log("native_home_started", arm_can_ids=[1, 2, 3, 4, 5, 6],
                     gripper="native safe_zero rule retained")
            with fresh_native_homing_origins(robot) as origins:
                robot.safe_zero(exit_on_complete=False)
            if not set(ARM_JOINTS).issubset(origins):
                raise RuntimeError("Native safe_zero did not use fresh reads for all six arm origins.")
            receipt = self.verify(robot, cancelled=self._cancelled)
        finally:
            robot.send_action = original_send
        with self._state_lock:
            if self._intent != "home":
                raise RuntimeError("Fault hold requested before release.")
            self._phase = "releasing"
        self.log("native_home_feedback_verified", **receipt)
        # A disable operation can fail after changing only some motors. Once
        # release begins, cleanup errors must never re-enable them for damping.
        self._release_started = True
        camera_errors = self.release(robot)
        self.released = True
        self.log("native_home_disabled_closed", camera_cleanup_errors=camera_errors)

    def _fault_hold(self, robot):
        with self._state_lock:
            self._phase = "fault_hold_failed" if self._release_started else "fault_hold"
            if not self._release_started and not self._hold_armed:
                self.vendor.request_b601_rs_damped_hold()
                self._hold_armed = True
        self.log("fault_hold_required", error=self.failure)
        if self._release_started:
            self.log("fault_hold_failed", error=self.failure, ownership_retained=True,
                     torque_state="Disable/close failed, possibly partial. No automatic re-enable.")
            self.release_confirmed.wait()
        else:
            while not self.release_confirmed.is_set():
                try:
                    self.vendor.perform_b601_rs_persistent_damped_hold(robot)
                    if not self.release_confirmed.is_set():
                        raise RuntimeError("Damped hold ended without this session's operator confirmation.")
                except Exception as error:
                    with self._state_lock:
                        self._phase = "fault_hold_failed"
                    self.log("fault_hold_failed", error=str(error), ownership_retained=True)
                    self.release_confirmed.wait(0.25)
        # This is an explicit operator release, independent of zero verification.
        self.vendor.disconnect_device_safely(robot, label="act224-confirmed-release",
                                              rs_damped_shutdown=False)
        self.released = True
        self.log("operator_confirmed_release", home_verified=False)

    def finish(self, client):
        with self._cleanup_lock:
            if self._finished:
                return
            with self._state_lock:
                self._phase = "quiescing"
            if not self.stop_requested.is_set():
                self.request_hold("Client ended without an explicit normal home_stop.")
            self._quiesce(client)
            robot = getattr(client, "robot", None)
            with self.action_lock:
                if robot is not None and getattr(robot, "bus", None) is not None:
                    try:
                        if self._cancelled():
                            raise RuntimeError(self.failure or "Fault hold requested.")
                        self._home_and_release(robot)
                    except Exception as error:
                        # A failed home must retain this owner until operator confirmation.
                        with self._state_lock:
                            self._phase = "quiescing"
                        self.request_hold(f"{type(error).__name__}: {error}")
                        if getattr(robot, "bus", None) is not None:
                            self._fault_hold(robot)
                with self._state_lock:
                    self._phase = "finished"
                self._finished = True
            if self.failure and not self.released:
                raise RuntimeError(self.failure)
