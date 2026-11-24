"""Measure what the existing motor cache can support without issuing CAN requests.

The installed motorbridge cache has no receive clock or receive counter. A
changed full fingerprint bounds when Python could first observe a new cached
value. This proves a cache change, not when the motor produced or the driver
received the frame. It is an observation diagnostic, not an authoritative
source of fresh live history for an ACT policy.
"""

from collections import deque
from dataclasses import dataclass
import math
import struct


STATE_KEYS = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
    "wrist_yaw.pos", "wrist_roll.pos", "gripper.pos",
)
HISTORY_KEYS = tuple(f"{key}.delta_6_frames" for key in STATE_KEYS[:6])
PERIOD_S = 1 / 30
HISTORY_S = 6 * PERIOD_S


@dataclass(frozen=True)
class CacheChange:
    value_deg: float
    first_possible: float
    last_possible: float


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class MotorCacheHistory:
    """Track certain cache changes from externally timed, passive cache reads."""

    def __init__(self):
        self._last_reads = {}
        self._changes = {key: deque(maxlen=128) for key in STATE_KEYS}

    def record(self, key, state, started, ended):
        if not (math.isfinite(started) and math.isfinite(ended)) or started > ended:
            raise ValueError("Expected a monotonic interval for each cache read.")
        if state is None:
            self._last_reads.pop(key, None)
            self._changes[key].clear()
            return
        marker = (state.can_id, state.arbitration_id, state.status_code, state.pos, state.vel,
                  state.torq, state.t_mos, state.t_rotor)
        if not all(map(math.isfinite, marker)):
            raise ValueError(f"Nonfinite motor cache state for {key}.")
        last = self._last_reads.get(key)
        if last is not None:
            old_marker, old_start, old_end = last
            if started < old_end:
                raise ValueError(f"Overlapping cache reads for {key}.")
            if marker != old_marker:
                self._changes[key].append(CacheChange(float32(math.degrees(state.pos)), old_start, ended))
        self._last_reads[key] = (marker, started, ended)

    def diagnose(self, capture_at, observation):
        """Return conservative cache-clock history only where change evidence exists.

        Each six-axis endpoint must fit entirely in the same half-period
        window. The supplied current joint values must be the exact float32
        values observed near capture. The gripper contributes current position
        to the existing model input but has no historical delta channel.
        """
        names = STATE_KEYS[:6]
        deltas = {}
        windows = []
        for key, history_key in zip(names, HISTORY_KEYS, strict=True):
            current = float32(observation[key])
            fresh = [sample for sample in self._changes[key]
                     if sample.value_deg == current and sample.last_possible <= capture_at
                     and sample.first_possible >= capture_at - PERIOD_S / 2]
            midpoint = capture_at - HISTORY_S
            past = [sample for sample in self._changes[key]
                    if sample.first_possible >= midpoint - PERIOD_S / 2
                    and sample.last_possible <= midpoint + PERIOD_S / 2]
            if not fresh or not past or key not in self._last_reads:
                return {"status": "insufficient_cache_change_evidence", "missing_key": key}
            new, old = fresh[-1], min(past, key=lambda sample: abs(
                (sample.first_possible + sample.last_possible) / 2 - midpoint))
            deltas[history_key] = float32(new.value_deg - old.value_deg)
            windows.append((new, old))
        return {"status": "cache_clock_only", "deltas": deltas, "windows": windows}
