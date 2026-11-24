"""Detect a candidate learned return from the existing native observation hook.

This monitor observes J2/J3 degrees, without hardware access or motion commands.
Its event describes a return candidate. Fresh seven-axis saved-pose verification
and task outcome recording remain responsibilities of the trial supervisor.
Call reset after the prior trial's observation writer has been quiesced.
"""

from collections.abc import Mapping
import math
from numbers import Real


def _finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


class LearnedReturnMonitor:
    def __init__(self, *, enabled=False, departure_deg=80.0, terminal_abs_deg=5.0,
                 dwell_seconds=0.5, max_sample_gap=0.2, on_returned=None):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean.")
        parameters = (departure_deg, terminal_abs_deg, dwell_seconds, max_sample_gap)
        if not all(_finite_number(value) for value in parameters):
            raise ValueError("Return-monitor thresholds must be finite numbers.")
        if terminal_abs_deg < 0 or departure_deg <= terminal_abs_deg:
            raise ValueError("Departure must exceed the nonnegative terminal threshold.")
        if dwell_seconds <= 0 or max_sample_gap <= 0:
            raise ValueError("Dwell duration and maximum sample gap must be positive.")
        if on_returned is not None and not callable(on_returned):
            raise ValueError("on_returned must be callable when supplied.")
        self.enabled = enabled
        self.departure_deg = float(departure_deg)
        self.terminal_abs_deg = float(terminal_abs_deg)
        self.dwell_seconds = float(dwell_seconds)
        self.max_sample_gap = float(max_sample_gap)
        self.on_returned = on_returned
        self.reset()

    def reset(self):
        self._departed = False
        self._near_since = None
        self._last_timestamp = None
        self._returned = False

    def observe(self, observation, *, timestamp):
        """Return True once when departed J2/J3 remain near zero continuously.

        timestamp is a caller-supplied monotonic time in seconds. Position fields
        are motor-native degrees from the existing get_observation callback.
        This hook does not establish each motor receipt's freshness.
        """
        if not self.enabled or self._returned:
            return False
        if not _finite_number(timestamp):
            self._near_since = None
            return False
        timestamp = float(timestamp)
        previous_timestamp = self._last_timestamp
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            self._near_since = None
            return False
        self._last_timestamp = timestamp
        if not isinstance(observation, Mapping):
            self._near_since = None
            return False
        positions = (observation.get("shoulder_lift.pos"), observation.get("elbow_flex.pos"))
        if not all(_finite_number(value) for value in positions):
            self._near_since = None
            return False
        if previous_timestamp is not None and timestamp - previous_timestamp > self.max_sample_gap + 1e-9:
            self._near_since = None
        maximum_distance = max(abs(float(value)) for value in positions)
        if maximum_distance >= self.departure_deg:
            self._departed = True
        if not self._departed or maximum_distance > self.terminal_abs_deg:
            self._near_since = None
            return False
        if self._near_since is None:
            self._near_since = timestamp
        if timestamp - self._near_since + 1e-9 < self.dwell_seconds:
            return False
        self._returned = True
        if self.on_returned is not None:
            self.on_returned()
        return True
