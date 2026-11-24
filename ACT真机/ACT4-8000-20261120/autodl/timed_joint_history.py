"""Construct ACT six-axis motion history from individually timed position reads."""

from collections import deque
from dataclasses import asdict, dataclass
import math
import struct

from motion_history_cache import STATE_KEYS


HISTORY_NS = 200_000_000
ARM_KEYS = STATE_KEYS[:6]


@dataclass(frozen=True)
class TimedJointSample:
    position_deg: float
    started_ns: int
    ended_ns: int

    @property
    def midpoint_ns(self):
        return (self.started_ns + self.ended_ns) // 2

    def evidence(self, observation_ns):
        return {**asdict(self), "midpoint_ns": self.midpoint_ns,
                "age_ns": observation_ns - self.midpoint_ns,
                "minimum_age_ns": observation_ns - self.ended_ns,
                "maximum_age_ns": observation_ns - self.started_ns,
                "width_ns": self.ended_ns - self.started_ns}


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class TimedJointHistory:
    """The caller supplies fresh reply-backed reads on one monotonic clock.

    Use nanoseconds for all times and degrees for positions. The caller
    serializes record and snapshot if acquiring feedback on another thread.
    """

    def __init__(self, *, started_ns, source):
        self.started_ns = started_ns
        self.source = source
        self._samples = {key: deque() for key in ARM_KEYS}
        self._first = {}

    def record(self, key, position_deg, started_ns, ended_ns):
        samples = self._samples[key]
        if started_ns < self.started_ns or ended_ns < started_ns:
            raise ValueError(f"Invalid feedback interval for {key}.")
        if samples and (started_ns < samples[-1].ended_ns or ended_ns <= samples[-1].ended_ns):
            raise ValueError(f"Feedback intervals must advance without overlap for {key}.")
        if not math.isfinite(position_deg):
            raise ValueError(f"Nonfinite feedback for {key}.")
        sample = TimedJointSample(_float32(position_deg), started_ns, ended_ns)
        samples.append(sample)
        self._first.setdefault(key, sample)
        # Keep one earlier sample for a bracket straddling the retention edge.
        while len(samples) > 2 and samples[1].ended_ns < ended_ns - 2 * HISTORY_NS:
            samples.popleft()

    def snapshot(self, observation_ns, *, max_current_age_ns, max_bracket_ns, max_sample_width_ns,
                 allow_initial_clamp=False, latest_sample_ns=None):
        """Return one causal six-axis vector or identify the unavailable axis."""
        cutoff_ns = observation_ns if latest_sample_ns is None else min(observation_ns, latest_sample_ns)
        axes = {}
        for key in ARM_KEYS:
            axis = self._axis(key, observation_ns, max_current_age_ns,
                              max_bracket_ns, max_sample_width_ns, allow_initial_clamp, cutoff_ns)
            axes[key] = axis
        ready = all("delta_deg" in axis for axis in axes.values())
        return {"status": "ready" if ready else "insufficient_timed_feedback",
                "source": self.source, "observation_ns": observation_ns, "history_ns": HISTORY_NS,
                "axes": axes,
                "current_deg": [axes[key]["current"]["position_deg"] for key in ARM_KEYS] if ready else None,
                "deltas_deg": [axes[key]["delta_deg"] for key in ARM_KEYS] if ready else None}

    def _axis(self, key, observation_ns, max_age, max_bracket, max_width, allow_initial_clamp, cutoff_ns):
        samples = [sample for sample in self._samples[key] if sample.ended_ns <= cutoff_ns]
        if not samples:
            return {"reason": "missing_current_sample"}
        current = samples[-1]
        proof = {"current": current.evidence(observation_ns)}
        if observation_ns - current.started_ns > max_age:
            return {**proof, "reason": "current_sample_too_old"}
        if current.ended_ns - current.started_ns > max_width:
            return {**proof, "reason": "current_time_interval_too_wide"}

        target_start = current.started_ns - HISTORY_NS
        target_end = current.ended_ns - HISTORY_NS
        target_ns = current.midpoint_ns - HISTORY_NS
        proof.update(target_ns=target_ns, target_earliest_ns=target_start, target_latest_ns=target_end)
        first = self._first[key]
        if (allow_initial_clamp and observation_ns - self.started_ns < HISTORY_NS
                and first.ended_ns - first.started_ns <= max_width and target_end <= first.started_ns):
            endpoint = first.evidence(observation_ns)
            return {**proof, "method": "first_sample_within_initial_200ms", "left": endpoint,
                    "right": endpoint, "bracket_width_ns": 0, "past_deg": first.position_deg,
                    "delta_deg": _float32(current.position_deg - first.position_deg)}

        usable = [sample for sample in samples if sample.ended_ns - sample.started_ns <= max_width]
        left = next((sample for sample in reversed(usable) if sample.ended_ns <= target_start), None)
        right = next((sample for sample in usable if sample.started_ns >= target_end), None)
        if left is None or right is None:
            return {**proof, "reason": "target_not_bracketed_by_measured_samples"}
        width_ns = right.ended_ns - left.started_ns
        proof.update(left=left.evidence(observation_ns), right=right.evidence(observation_ns),
                     bracket_width_ns=width_ns)
        if width_ns > max_bracket:
            return {**proof, "reason": "measured_bracket_too_wide"}
        if left == right:
            past = left.position_deg
            method = "measured_at_target"
        else:
            fraction = (target_ns - left.midpoint_ns) / (right.midpoint_ns - left.midpoint_ns)
            past = left.position_deg + fraction * (right.position_deg - left.position_deg)
            method = "interpolated_between_measured_samples"
        past = _float32(past)
        return {**proof, "method": method, "past_deg": past,
                "delta_deg": _float32(current.position_deg - past)}
