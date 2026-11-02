"""Pure input mapping and an in-memory preview backend.

There is deliberately no transport/controller abstraction in this module: it
cannot connect to motors. Displacements start at a synthetic preview origin.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Any


AXES = ("LX", "LY", "RX", "RY", "LT", "RT")
BUTTONS = {
    "DPadUp": 0x0001, "DPadDown": 0x0002,
    "DPadLeft": 0x0004, "DPadRight": 0x0008,
    "Start": 0x0010, "Back": 0x0020,
    "LeftStick": 0x0040, "RightStick": 0x0080,
    "LB": 0x0100, "RB": 0x0200,
    "A": 0x1000, "B": 0x2000, "X": 0x4000, "Y": 0x8000,
}


@dataclass
class Frame:
    connected: bool
    packet: int = 0
    buttons: int = 0
    axes: dict[str, float] = field(default_factory=lambda: dict.fromkeys(AXES, 0.0))
    acquired_at: float = field(default_factory=time.monotonic)


class FakeBackend:
    """Accumulate synthetic displacement, never actual robot joint angles."""

    def __init__(self) -> None:
        self.displacement_rad = [0.0] * 6
        self.applied_samples = 0

    def apply(self, velocity_rad_s: list[float], dt: float) -> None:
        if len(velocity_rad_s) != 6 or not 0.0 <= dt <= 0.05:
            raise ValueError("Invalid synthetic backend step")
        if not all(math.isfinite(value) for value in velocity_rad_s):
            raise ValueError("Non-finite synthetic velocity")
        self.displacement_rad = [
            position + velocity * dt
            for position, velocity in zip(self.displacement_rad, velocity_rad_s)
        ]
        self.applied_samples += 1

    def reset(self) -> None:
        self.displacement_rad = [0.0] * 6
        self.applied_samples = 0


class Engine:
    def __init__(self, deadzone: float = 0.12, max_speed_rad_s: float = 0.08,
                 stale_timeout: float = 0.25) -> None:
        if not 0 < deadzone < 1 or not 0 < max_speed_rad_s <= 1 or not stale_timeout > 0:
            raise ValueError("Invalid preview configuration")
        self.deadzone = deadzone
        self.max_speed_rad_s = max_speed_rad_s
        self.stale_timeout = stale_timeout
        self.backend = FakeBackend()
        self.selected_joint = 1
        self.paused = False
        self.needs_rearm = True
        self._frame: Frame | None = None
        self._axes = dict.fromkeys(AXES, 0.0)
        self._buttons = 0
        self._previous_buttons = 0
        self._valid = False
        self._invalid_reason = ""
        self._last_update: float | None = None
        self._velocities = [0.0] * 6
        self._gripper = "hold"
        self._gate = "disconnected"
        self._reason = "等待手柄输入"
        self._events: deque[dict[str, Any]] = deque(maxlen=40)
        self._last_input_signature: Any = None
        self._stats = {
            "samples": 0, "state_changes": 0, "read_errors": 0,
            "disconnects": 0, "stale_events": 0, "invalid_inputs": 0,
            "blocked_selections": 0, "neutral_rearms": 0,
        }

    def _event(self, now: float, kind: str, message: str) -> None:
        self._events.append({"time": round(now, 3), "kind": kind, "message": message})

    def _set_gate(self, gate: str, reason: str, now: float) -> None:
        if gate != self._gate or reason != self._reason:
            if gate == "input_stale" and self._gate != "input_stale":
                self._stats["stale_events"] += 1
            self._event(now, "gate", reason)
        self._gate, self._reason = gate, reason

    def _zero(self) -> None:
        self._velocities = [0.0] * 6
        self._gripper = "hold"

    def _neutral(self) -> bool:
        return (
            all(abs(self._axes[axis]) <= self.deadzone for axis in AXES[:4])
            and all(self._axes[axis] <= self.deadzone for axis in AXES[4:])
            and not self._buttons & (BUTTONS["A"] | BUTTONS["X"])
        )

    def _age(self, now: float) -> float | None:
        if self._frame is None:
            return None
        try:
            acquired = float(self._frame.acquired_at)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(acquired):
            return None
        return now - acquired

    def _guard(self, now: float) -> bool:
        """Check freshness on both sampling and display/API reads."""
        if self._frame is None or not self._frame.connected:
            self.needs_rearm = True
            self._zero()
            self._set_gate("disconnected", "手柄未连接，所有预览意图已归零", now)
            return False
        age = self._age(now)
        if not self._valid or age is None or age < -0.000001 or age > self.stale_timeout:
            self.needs_rearm = True
            self._zero()
            reason = self._invalid_reason or "输入已过期，松开 LB 并回中后重新允许"
            self._set_gate("input_stale", reason, now)
            return False
        if self.paused:
            self.needs_rearm = True
            self._zero()
            self._set_gate("paused", "预览已暂停；点击继续后，松开 LB 并回中", now)
            return False
        if self.needs_rearm:
            self._zero()
            self._set_gate("neutral_required", "请松开 LB、摇杆回中并释放扳机及 A/X", now)
            return False
        return True

    def update(self, frame: Frame, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else float(now)
        if not math.isfinite(now):
            self.needs_rearm = True
            self._zero()
            raise ValueError("A finite monotonic time is required")
        previous_connected = bool(self._frame and self._frame.connected)
        dt = 0.0 if self._last_update is None else now - self._last_update
        gap = self._last_update is not None and (dt < 0 or dt > self.stale_timeout)
        self._last_update = now
        self._frame = frame
        self._stats["samples"] += 1
        self._valid = False
        self._invalid_reason = ""
        self._zero()

        if not frame.connected:
            if previous_connected:
                self._stats["disconnects"] += 1
                self._event(now, "input", "手柄连接中断，预览输出归零")
            self._stats["read_errors"] += 1
            self._axes = dict.fromkeys(AXES, 0.0)
            self._buttons = 0
            self._previous_buttons = 0
            self._guard(now)
            return self.snapshot(now)

        # A fresh, valid B bit must still latch pause if an axis is corrupt.
        # Validate the independent button and axis channels separately.
        buttons_valid = False
        axes_valid = False
        try:
            self._buttons = int(frame.buttons)
            if not 0 <= self._buttons <= 0xFFFF:
                raise ValueError("button bitmask outside uint16")
            buttons_valid = True
        except (TypeError, ValueError, OverflowError):
            self._buttons = 0
        try:
            self._axes = {axis: float(frame.axes.get(axis, 0.0)) for axis in AXES}
            for axis, value in self._axes.items():
                lower = 0 if axis in ("LT", "RT") else -1
                if not math.isfinite(value) or not lower <= value <= 1:
                    raise ValueError("axis outside normalized range")
            axes_valid = True
        except (TypeError, ValueError, AttributeError, OverflowError):
            self._axes = dict.fromkeys(AXES, 0.0)
        self._valid = buttons_valid and axes_valid
        if not self._valid:
            self._stats["invalid_inputs"] += 1
            self._invalid_reason = "输入值无效，所有预览意图已归零；请回中重试"

        signature = (bool(frame.connected), self._buttons, tuple(self._axes.values()))
        if self._last_input_signature is not None and signature != self._last_input_signature:
            self._stats["state_changes"] += 1
        self._last_input_signature = signature
        if not previous_connected:
            self.needs_rearm = True
            self._previous_buttons = 0
            self._event(now, "input", "已读取手柄；先松开 LB 并回中")
        if gap:
            self.needs_rearm = True
            self._event(now, "guard", "采样间隔异常，丢弃该时间段的位移并重新回中")

        age = self._age(now)
        timestamp_fresh = age is not None and -0.000001 <= age <= self.stale_timeout
        fresh = self._valid and timestamp_fresh
        lb_held = bool(self._buttons & BUTTONS["LB"])
        if timestamp_fresh and buttons_valid and self._buttons & BUTTONS["B"]:
            if not self.paused:
                self._event(now, "pause", "B 键暂停：所有预览意图已归零")
            self.paused = True
            self.needs_rearm = True

        if fresh and not self.paused:
            navigation = self._buttons & ~self._previous_buttons
            left = bool(navigation & BUTTONS["DPadLeft"])
            right = bool(navigation & BUTTONS["DPadRight"])
            contradictory_navigation = bool(self._buttons & BUTTONS["DPadLeft"]) and bool(self._buttons & BUTTONS["DPadRight"])
            if contradictory_navigation and (left or right):
                self._stats["blocked_selections"] += 1
                self._event(now, "selection_blocked", "左右方向键同时按下，忽略切换关节")
            elif left != right:
                if not lb_held and self._neutral():
                    self.selected_joint = (self.selected_joint - 1 + (1 if right else -1)) % 6 + 1
                    self._event(now, "selection", f"选择 J{self.selected_joint}（仅离线预览）")
                else:
                    self._stats["blocked_selections"] += 1
                    self._event(now, "selection_blocked", "切换关节已忽略：请松开 LB 并回中后再按方向键")
            if self.needs_rearm and not lb_held and self._neutral():
                self.needs_rearm = False
                self._stats["neutral_rearms"] += 1
                self._event(now, "rearm", "已回中，按住 LB 后可输出预览意图")

        can_output = self._guard(now)
        if can_output and lb_held:
            value = self._axes["LY"]
            if abs(value) > self.deadzone:
                normalized = math.copysign((abs(value) - self.deadzone) / (1 - self.deadzone), value)
                self._velocities[self.selected_joint - 1] = normalized * self.max_speed_rad_s
            open_held = bool(self._buttons & BUTTONS["A"])
            close_held = bool(self._buttons & BUTTONS["X"])
            if open_held != close_held:
                self._gripper = "open" if open_held else "close"
            if any(self._velocities) or self._gripper != "hold":
                self._set_gate("active", "正在输出到假后端；松开 LB 即停止意图", now)
            elif open_held and close_held:
                self._set_gate("ready", "A/X 同时按下，夹爪意图保持；关节按左摇杆独立预览", now)
            else:
                self._set_gate("ready", "LB 已按住，推动左摇杆或按 A/X 预览", now)
        elif can_output:
            self._set_gate("ready", "已就绪，按住 LB 后推动左摇杆；方向键左右选关节", now)

        # Never compensate for a sampling gap with a larger synthetic step.
        if can_output and not gap:
            self.backend.apply(self._velocities, min(max(dt, 0.0), 0.05))
        self._previous_buttons = self._buttons
        return self.snapshot(now)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else float(now)
        if not math.isfinite(now):
            self.needs_rearm = True
            self._zero()
            raise ValueError("A finite monotonic time is required")
        self._guard(now)
        age = self._age(now)
        return {
            "input": {
                "connected": bool(self._frame and self._frame.connected),
                "packet": int(self._frame.packet) if self._frame else 0,
                "buttons": [name for name, mask in BUTTONS.items() if self._buttons & mask],
                "axes": dict(self._axes),
                "age_ms": round(max(0.0, age) * 1000, 1) if age is not None else None,
            },
            "control": {
                "selected_joint": self.selected_joint,
                "paused": self.paused,
                "gate": self._gate,
                "reason": self._reason,
                "lb_held": bool(self._buttons & BUTTONS["LB"]),
                "armed": not self.needs_rearm,
                "joint_velocity_rad_s": list(self._velocities),
                "gripper_intent": self._gripper,
                "max_speed_rad_s": self.max_speed_rad_s,
                "deadzone": self.deadzone,
            },
            "fake_backend": {
                "displacement_rad": list(self.backend.displacement_rad),
                "applied_samples": self.backend.applied_samples,
            },
            "events": list(self._events),
            "stats": dict(self._stats),
        }

    def set_paused(self, paused: bool) -> None:
        if not isinstance(paused, bool):
            raise ValueError("paused must be a bool")
        if self.paused == paused:
            return
        self.paused = paused
        self.needs_rearm = True
        self._zero()
        now = self._last_update if self._last_update is not None else time.monotonic()
        self._event(now, "pause", "界面暂停预览" if paused else "界面继续预览；先松开 LB 并回中")
        self._guard(now)

    def reset_preview(self) -> None:
        self.backend.reset()
        self.needs_rearm = True
        self._zero()
        now = self._last_update if self._last_update is not None else time.monotonic()
        self._event(now, "reset", "已清空虚拟累计位移；零仅为预览参考")
        self._guard(now)
