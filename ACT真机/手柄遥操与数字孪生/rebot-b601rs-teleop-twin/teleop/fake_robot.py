"""Synthetic seven-axis plant and SDK-shaped objects. No hardware imports.

Every read is labelled synthetic_fixture. Numeric limits, gains and generated
status below are simulation fixtures, never a site configuration or calibration.
"""
from __future__ import annotations

import math
from session_core import NativeFeedback

DEFAULT_FAKE_ORIGIN = (.31, -.22, .30, .11, -.18, .27, 1.1)


def synthetic_config():
    """Fresh explicit fake-only fixture; not loaded from site_config.template."""
    return {
        "synthetic_only": True, "hardware_writes_supported": False,
        "provenance": "generated simulation fixture; NOT physical measurements",
        "native_coordinate": "motor_native_rad", "physical_parameters_verified": True,
        "site_path_verified": True, "return_path_verified": True,
        "j3_reverse_motion_verified": True, "feedback_max_age_s": .25,
        "max_control_interval_s": .1, "tracking_error_rad": .3,
        "position_tolerance_rad": .002, "velocity_tolerance_rad_s": .005,
        "stable_dwell_s": .1,
        "joints": [
            {"name": f"J{i+1}", "motor_id": i+1, "host_id": 253,
             "model": "rs-06" if i < 3 else "rs-00",
             "native_min_rad": 0.0 if i == 6 else -2.0,
             "native_max_rad": 2.0, "max_speed_rad_s": .4,
             "max_accel_rad_s2": .8, "kp": 1.0, "kd": .2}
            for i in range(7)
        ],
    }


class FakeMotor:
    def __init__(self, robot, index):
        self.robot, self.index = robot, index
        self.fail_send = False
        self.fail_next_send = False
        self.closed = False

    def send_mit(self, **values):
        if self.closed or self.robot.bus_closed:
            raise RuntimeError("synthetic motor/bus is closed")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in values.values()):
            raise ValueError("non-finite synthetic command")
        failure = self.fail_send or self.fail_next_send
        self.fail_next_send = False
        self.robot.calls.append({"kind": "mit", "axis": self.index+1,
                                 "at": self.robot.now, "values": dict(values),
                                 "success": not failure})
        if failure:
            raise RuntimeError(f"injected J{self.index+1} synthetic send failure")
        self.robot.targets[self.index] = float(values["pos"])
        self.robot.command_velocities[self.index] = float(values["vel"])

    def close(self):
        if not self.robot.bus_closed:
            raise RuntimeError("close synthetic bus before freeing motors")
        self.robot.calls.append({"kind": "motor_close", "axis": self.index+1})
        self.closed = True


class FakeRobot:
    """One-cycle sampled plant, not a physics or hardware safety model.

    send_mit changes only a queued synthetic target. read_feedback advances the
    synthetic plant toward it, with measured velocity computed from q delta / dt.
    A command is thus never used directly as a same-cycle measured position.
    """
    synthetic_only = True
    hardware_writes_supported = False

    def __init__(self, origin=DEFAULT_FAKE_ORIGIN):
        if len(origin) != 7 or not all(math.isfinite(v) for v in origin):
            raise ValueError("seven finite synthetic initial positions required")
        self.positions = list(origin)
        self.targets = list(origin)
        self.velocities = [0.0]*7
        self.command_velocities = [0.0]*7
        self.fault_codes = (0,)*7
        self.control_modes = ("MIT",)*7
        self.axis_age_s = [0.0]*7
        self.status_age_s = [0.0]*7
        self.track_targets = True
        self.fail_read = False
        self.replay_feedback = False
        self.fail_enable = False
        self.fail_disable = False
        self.fail_close = False
        self.enabled = False
        self.bus_closed = False
        self.closed = False
        self.calls = []
        self.sequence = 0
        self.now = 0.0
        self._read_at = None
        self._last_feedback = None
        self.motors = tuple(FakeMotor(self, i) for i in range(7))
        self.controller = self

    def read_feedback(self, session_id, now):
        if self.bus_closed:
            raise RuntimeError("synthetic bus is closed")
        self.now = float(now)
        if self.fail_read:
            raise RuntimeError("injected synthetic feedback failure")
        if self.replay_feedback and self._last_feedback is not None:
            return self._last_feedback
        dt = 0.0 if self._read_at is None else now-self._read_at
        if self.enabled and self.track_targets and dt > 0:
            previous = self.positions[:]
            # Fake plant speed is limited independently of target generation.
            limit = .4*dt
            self.positions = [q+max(-limit, min(limit, target-q))
                              for q, target in zip(previous, self.targets)]
            self.velocities = [(new-old)/dt for old, new in zip(previous, self.positions)]
        elif self.track_targets:
            self.velocities = [0.0]*7
        self._read_at = now
        self.sequence += 1
        self._last_feedback = NativeFeedback(
            session_id, "synthetic_fixture", now, self.sequence,
            tuple(self.positions), tuple(self.velocities),
            tuple(now-age for age in self.axis_age_s), tuple(self.fault_codes),
            tuple(self.control_modes), tuple(now-age for age in self.status_age_s),
        )
        return self._last_feedback

    def enable_all(self):
        self.calls.append({"kind": "enable_all", "at": self.now})
        if self.fail_enable:
            raise RuntimeError("injected synthetic partial enable")
        self.enabled = True

    def disable_all(self):
        self.calls.append({"kind": "disable_all", "at": self.now})
        if self.fail_disable:
            raise RuntimeError("injected synthetic partial disable")
        self.enabled = False

    def close_bus(self):
        self.calls.append({"kind": "close_bus", "at": self.now})
        if self.enabled or self.fail_close:
            raise RuntimeError("synthetic bus cannot be released")
        self.bus_closed = True

    def close(self):
        if not self.bus_closed:
            raise RuntimeError("close synthetic bus before controller")
        self.calls.append({"kind": "controller_close", "at": self.now})
        self.closed = True
