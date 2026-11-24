"""Linear replanning around SenseCraft's existing trajectory and actuator.

No controller, robot or model is created here. References are seven-dimensional
CPU tensors in dataset action coordinates. Native send_action results must be
inverse-mapped by the caller solely to record the actual dispatched anchor.
"""

from collections import deque
from dataclasses import dataclass, replace
import math
import sys
import threading
import time

import torch


@dataclass(frozen=True)
class _Reference:
    timestep: int
    action: torch.Tensor


@dataclass(frozen=True)
class _ChunkAction:
    original: object
    action: torch.Tensor
    prediction_id: tuple | None = None
    observation_monotonic: float | None = None

    def get_timestep(self):
        return self.original.get_timestep()

    def get_timestamp(self):
        return self.original.get_timestamp()

    def get_action(self):
        return self.action


def _reference(value):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("A dispatched reference must be a CPU tensor.")
    if tuple(value.shape) != (7,) or not torch.isfinite(value).all():
        raise ValueError("A dispatched reference must contain seven finite coordinates.")
    return value.detach().clone()


class TemporalChunkEnsembler:
    """ACT exponential weights over predictions for the same global timestep.

    The earliest contributing prediction has index zero. Positive coefficients
    prefer older predictions, matching the corrected upstream ACT convention.
    A full chunk's first source timestep/timestamp identifies its prediction.
    Individual target timestamps can coincide across different predictions.
    """

    def __init__(self, coefficient):
        if isinstance(coefficient, bool) or not math.isfinite(coefficient) or abs(coefficient) > 1:
            raise ValueError("Temporal ensemble coefficient must be finite and within [-1, 1].")
        self.coefficient = float(coefficient)
        self.predictions = {}

    def merge(self, action):
        prediction_id = action.prediction_id
        timestep = int(action.get_timestep())
        value = _reference(action.get_action())
        history = self.predictions.setdefault(timestep, {})
        previous = history.get(prediction_id)
        if previous is not None and not torch.equal(previous, value):
            raise ValueError("A repeated prediction identity contains conflicting values.")
        history[prediction_id] = value
        ordered = sorted(history, key=lambda key: (key[1], key[0]))
        values = torch.stack([history[key] for key in ordered])
        weights = torch.softmax(-self.coefficient * torch.arange(len(values), dtype=values.dtype), dim=0)
        return replace(action, action=(values * weights[:, None]).sum(dim=0))

    def discard_through(self, timestep):
        self.predictions = {key: value for key, value in self.predictions.items() if key > timestep}

    def reset(self):
        self.predictions.clear()


class _RecentShoulderElbowMotion:
    motion_window_s = 0.10

    def __init__(self, output_hz, directions):
        self.output_hz = output_hz
        self.directions = directions[1:3]
        self.reset()

    def reset(self):
        self.history = deque()

    def record(self, target, now, *, stale):
        if self.history and not 0 < now - self.history[-1][0] <= 3 / self.output_hz:
            self.history.clear()
        while self.history and now - self.history[0][0] > self.motion_window_s:
            self.history.popleft()
        if stale:
            self.history.clear()
        else:
            self.history.append((now, target[1:3] * self.directions))

    def recent_delta(self, now):
        while self.history and now - self.history[0][0] > self.motion_window_s:
            self.history.popleft()
        if len(self.history) < 3 or not 0 <= now - self.history[-1][0] <= 3 / self.output_hz:
            return None
        return self.history[-1][1] - self.history[0][1]


class ShoulderElbowReplanGuard(_RecentShoulderElbowMotion):
    """Bias a short J2/J3 overlap toward the old plan during a contradictory replan."""

    overlap_s = 0.20
    max_bias_deg = 6.0

    def __init__(self, source_hz, output_hz, directions):
        self.source_hz = source_hz
        super().__init__(output_hz, directions)

    def reset(self):
        super().reset()
        self.hold_until = self.rearm_at = -math.inf
        self.last_guarded_step = -1
        self.direction = None
        self.holds = self.retained_targets = 0

    def retain_prefix(self, actions, old_targets, *, anchor, cursor, now):
        recent_delta = self.recent_delta(now)
        if recent_delta is None:
            return actions
        if self.hold_until <= now < self.rearm_at:
            return actions

        incoming = {int(action.get_timestep()): action for action in actions}
        step_limit = cursor + math.floor(self.overlap_s * self.source_hz)
        if now < self.hold_until:
            step_limit = min(step_limit, self.last_guarded_step)
        overlap = []
        for step in range(cursor + 1, step_limit + 1):
            if step not in old_targets or step not in incoming:
                break
            overlap.append(step)
        if not overlap or (now >= self.hold_until and len(overlap) < 3):
            return actions

        native_anchor = anchor[1:3] * self.directions
        old = torch.stack([old_targets[step][1:3] for step in overlap]) * self.directions
        new = torch.stack([incoming[step].get_action()[1:3] for step in overlap]) * self.directions
        if now >= self.hold_until:
            direction = recent_delta.sign()
            old_advance = (old[:3].mean(0) - native_anchor) * direction
            new_advance = (new[:3].mean(0) - native_anchor) * direction
            reverses = ((recent_delta.abs() >= 2.0) & (old_advance >= 1.0)
                        & (new_advance <= -2.0) & (old_advance - new_advance >= 5.0))
            if not reverses.any():
                return actions
            self.direction = direction * reverses
            self.hold_until = now + len(overlap) / self.source_hz
            self.rearm_at = self.hold_until + self.overlap_s
            self.last_guarded_step = overlap[-1]
            self.holds += 1

        discrepancy = old - new
        retain = ((discrepancy * self.direction >= 2.0)
                  & ((new - native_anchor) * self.direction <= -1.0)
                  & ((old - native_anchor) * self.direction >= -0.5))
        # A short prefix can still contain a necessary course correction. Favor
        # the old targets without substituting an arbitrarily different plan.
        bias = discrepancy.clamp(-self.max_bias_deg, self.max_bias_deg) / self.directions
        replacements = {}
        for index, step in enumerate(overlap):
            mask = retain[index]
            if mask.any():
                value = incoming[step].get_action().clone()
                value[1:3] += torch.where(mask, bias[index], 0.0)
                replacements[step] = _ChunkAction(incoming[step], value)
        self.retained_targets += len(replacements)
        return [replacements.get(int(action.get_timestep()), action) for action in actions]


class StaleOverlapReversalRateCap(_RecentShoulderElbowMotion):
    """Limit only opposite motion briefly when a delayed plan sharply disagrees."""

    old_age_s = 0.30
    duration_s = 0.12
    rearm_delay_s = 0.20
    reverse_rate_deg_s = 60.0

    def reset(self):
        super().reset()
        self.until = self.rearm_at = -math.inf
        self.direction = None
        self.triggers = self.limited_samples = 0

    def consider(self, actions, old_targets, *, anchor, cursor, now):
        source = actions[0].observation_monotonic
        if source is None or not 0.0 <= now - source:
            return
        recent_delta = self.recent_delta(now)
        if recent_delta is None:
            return
        steps = range(cursor + 1, cursor + 4)
        incoming = {int(action.get_timestep()): action.get_action()[1:3] for action in actions}
        if not all(step in incoming for step in steps):
            return
        age = now - source
        if now < self.until:
            if age < self.old_age_s:
                self.until = now
            return
        if now < self.rearm_at or age < self.old_age_s or not all(step in old_targets for step in steps):
            return

        old = torch.stack([old_targets[step][1:3] for step in steps]).mean(0) * self.directions
        new = torch.stack([incoming[step] for step in steps]).mean(0) * self.directions
        native_anchor = anchor[1:3] * self.directions
        direction = recent_delta.sign()
        old_advance = (old - native_anchor) * direction
        new_advance = (new - native_anchor) * direction
        reverses = ((recent_delta.abs() >= 2.0) & (old_advance >= 1.0)
                    & (new_advance <= -3.0) & (old_advance - new_advance >= 10.0))
        if reverses.any():
            self.direction = direction * reverses
            self.until = now + self.duration_s
            self.rearm_at = self.until + self.rearm_delay_s
            self.triggers += 1

    def limit(self, sample, *, anchor, now, elapsed):
        if sample.stale or now >= self.until or not self.history:
            return sample
        if not 0 <= now - self.history[-1][0] <= 3 / self.output_hz:
            return sample
        native_delta = (sample.action[1:3] - anchor[1:3]) * self.directions
        budget = self.reverse_rate_deg_s * elapsed
        exceeds = native_delta * self.direction < -budget
        if not exceeds.any():
            return sample
        output = sample.action.clone()
        limited_delta = torch.where(exceeds, -budget * self.direction, native_delta)
        output[1:3] = anchor[1:3] + limited_delta / self.directions
        self.limited_samples += 1
        return replace(sample, action=output)


def linear_trajectory_class(base_class):
    """Retain the pinned SenseCraft scheduler while replacing its interpolation."""
    class LinearReanchoredTrajectory(base_class):
        def __init__(self, *, temporal_ensemble_coeff=None, native_speed_limits=None,
                     arm_ema_time_constant_s=None, arm_transition_rate_deg_s=None,
                     shoulder_elbow_reversal_guard=False, stale_overlap_reversal_rate_cap=False,
                     joint_directions=(1, 1, -1, -1, -1, 1, 6), **kwargs):
            if arm_ema_time_constant_s is not None:
                if (isinstance(arm_ema_time_constant_s, bool)
                        or not isinstance(arm_ema_time_constant_s, (int, float))):
                    raise ValueError("arm_ema_time_constant_s must be null or finite and positive.")
                try:
                    arm_ema_time_constant_s = float(arm_ema_time_constant_s)
                except OverflowError as error:
                    raise ValueError("arm_ema_time_constant_s must be null or finite and positive.") from error
                if not math.isfinite(arm_ema_time_constant_s) or arm_ema_time_constant_s <= 0:
                    raise ValueError("arm_ema_time_constant_s must be null or finite and positive.")
            self._arm_ema_time_constant_s = arm_ema_time_constant_s
            if arm_transition_rate_deg_s is not None and (
                    type(arm_transition_rate_deg_s) not in (int, float)
                    or not 0 < arm_transition_rate_deg_s <= sys.float_info.max):
                raise ValueError("arm_transition_rate_deg_s must be null or finite and positive.")
            self._arm_transition_rate_deg_s = arm_transition_rate_deg_s
            self._reference_lock = threading.RLock()
            self._ensemble = (TemporalChunkEnsembler(temporal_ensemble_coeff)
                              if temporal_ensemble_coeff is not None else None)
            self._native_directions = _reference(torch.tensor(joint_directions, dtype=torch.float32))
            if (self._native_directions == 0).any():
                raise ValueError("Joint directions must be nonzero.")
            if not isinstance(shoulder_elbow_reversal_guard, bool):
                raise ValueError("shoulder_elbow_reversal_guard must be a boolean.")
            if shoulder_elbow_reversal_guard and self._ensemble is not None:
                raise ValueError("The shoulder/elbow replan guard requires temporal ensembling to be off.")
            if not isinstance(stale_overlap_reversal_rate_cap, bool):
                raise ValueError("stale_overlap_reversal_rate_cap must be a boolean.")
            if stale_overlap_reversal_rate_cap and (arm_transition_rate_deg_s is None
                    or shoulder_elbow_reversal_guard or self._ensemble is not None):
                raise ValueError("The stale overlap cap requires the arm transition with no other replan filter.")
            self._native_speed_limits = None
            if native_speed_limits is not None:
                self._native_speed_limits = _reference(torch.tensor(native_speed_limits, dtype=torch.float32))
                if (self._native_speed_limits <= 0).any():
                    raise ValueError("Native target speed limits must be positive.")
            if self._ensemble is not None:
                kwargs["merge_fn"] = None
            super().__init__(**kwargs)
            self._reversal_guard = (ShoulderElbowReplanGuard(self.source_hz, self.output_hz,
                                                             self._native_directions)
                                    if shoulder_elbow_reversal_guard else None)
            self._stale_overlap_cap = (StaleOverlapReversalRateCap(self.output_hz, self._native_directions)
                                       if stale_overlap_reversal_rate_cap else None)
            self._dispatched_reference = None
            self._dispatched_timestep = -1
            self._last_incoming = {}
            self._generation = 0
            self._sample_generation = -1
            self._blocked = False
            self._dispatched_at = None
            self._sample_at = self._committed_sample_at = None
            self._limited_samples = 0

        @staticmethod
        def _smoothstep(alpha):
            return max(0.0, min(1.0, alpha))

        def reset(self):
            with self._reference_lock:
                super().reset()
                self._dispatched_reference = None
                self._dispatched_timestep = -1
                self._last_incoming = {}
                self._generation = 0
                self._sample_generation = -1
                self._blocked = False
                self._dispatched_at = None
                self._sample_at = self._committed_sample_at = None
                self._limited_samples = 0
                if self._ensemble is not None:
                    self._ensemble.reset()
                if self._reversal_guard is not None:
                    self._reversal_guard.reset()
                if self._stale_overlap_cap is not None:
                    self._stale_overlap_cap.reset()

        def seed(self, reference, *, timestep=-1, now):
            with self._reference_lock:
                if self._blocked or self._dispatched_reference is not None:
                    raise RuntimeError("Seed once after reset and before receiving actions.")
                if not math.isfinite(now):
                    raise ValueError("A finite monotonic timestamp is required.")
                self._dispatched_reference = _reference(reference)
                self._dispatched_timestep = int(timestep)
                self._current = _Reference(int(timestep), self._dispatched_reference.clone())
                self._segment_started_at = float(now)
                self._dispatched_at = float(now)
                self._committed_sample_at = float(now)

        def _reanchor(self, now):
            future = {target.timestep: target for target in
                      [self._current, self._next, *self._pending]
                      if target is not None and target.timestep > self._dispatched_timestep}
            self._current = _Reference(self._dispatched_timestep, self._dispatched_reference.clone())
            self._next = None
            self._pending = deque(future[key] for key in sorted(future))
            self._segment_started_at = float(now)
            self._segment_duration_s = 1.0 / self.source_hz
            self._hold_started_at = None

        def prepare_chunk(self, actions, *, receipt_monotonic=None, receipt_wall_time=None):
            """Tag the complete RPC response before the parent filters stale targets."""
            actions = list(actions)
            if not actions or isinstance(actions[0], _ChunkAction):
                return actions
            if self._ensemble is None and self._stale_overlap_cap is None:
                return actions
            source_time = float(actions[0].get_timestamp())
            if self._ensemble is not None:
                if not math.isfinite(source_time):
                    raise ValueError("Temporal predictions require finite source timestamps.")
                prediction_id = (int(actions[0].get_timestep()), source_time)
                return [_ChunkAction(action, action.get_action(), prediction_id) for action in actions]
            # The server echoes the local observation's wall-clock timestamp.
            # Capture its age before target filtering, then use monotonic time.
            if receipt_monotonic is None:
                receipt_monotonic = time.monotonic()
            if receipt_wall_time is None:
                receipt_wall_time = time.time()
            age = receipt_wall_time - source_time
            observation_monotonic = (
                receipt_monotonic - age if math.isfinite(receipt_monotonic) and 0 <= age < math.inf else None)
            return [_ChunkAction(action, action.get_action(), observation_monotonic=observation_monotonic)
                    for action in actions]

        def push_many(self, actions, *, now=None):
            actions = self.prepare_chunk(actions)
            with self._reference_lock:
                if self._blocked:
                    return {"received": len(actions), "stale": len(actions), "merged": 0}
                if self._dispatched_reference is None or now is None or not math.isfinite(now):
                    raise RuntimeError("Seed the dispatched reference and provide monotonic now before replanning.")
                changed = []
                stale = duplicate = 0
                for action in actions:
                    timestep = int(action.get_timestep())
                    if timestep <= self._dispatched_timestep:
                        stale += 1
                        continue
                    value = _reference(action.get_action())
                    if self._ensemble is not None:
                        action = self._ensemble.merge(action)
                        value = action.get_action()
                    previous = self._last_incoming.get(timestep)
                    if previous is not None and torch.equal(previous, value):
                        duplicate += 1
                        continue
                    self._last_incoming[timestep] = value
                    changed.append(action)
                if not changed:
                    return {"received": len(actions), "stale": stale, "merged": duplicate}
                # step() can be sampled before a concurrent chunk arrives, while
                # the single actuator is still sending that sampled reference.
                if self._current is not None and self._current.timestep > self._dispatched_timestep:
                    self._reanchor(now)
                if self._reversal_guard is not None or self._stale_overlap_cap is not None:
                    old_targets = {target.timestep: target.action for target in (self._next, *self._pending)
                                   if target is not None}
                if self._reversal_guard is not None:
                    changed = self._reversal_guard.retain_prefix(
                        changed, old_targets, anchor=self._dispatched_reference,
                        cursor=self._dispatched_timestep, now=now)
                if self._stale_overlap_cap is not None:
                    self._stale_overlap_cap.consider(
                        changed, old_targets, anchor=self._dispatched_reference,
                        cursor=self._dispatched_timestep, now=now)
                previous_next = self._next
                previous_value = previous_next.action.clone() if previous_next is not None else None
                stats = super().push_many(changed, now=now)
                next_changed = (self._next is not None and previous_next is not None
                                and not torch.equal(self._next.action, previous_value))
                if self._next is None or next_changed:
                    self._reanchor(now)
                self._generation += 1
                return {"received": len(actions), "stale": stale + stats["stale"],
                        "merged": duplicate + stats["merged"]}

        def step(self, *, now):
            with self._reference_lock:
                if self._blocked:
                    return None
                if self._dispatched_reference is None:
                    raise RuntimeError("Seed before sampling the trajectory.")
                sample = super().step(now=now)
                if sample is not None and (self._native_speed_limits is not None
                                           or self._arm_ema_time_constant_s is not None):
                    elapsed = max(0.0, now - self._dispatched_at)
                    delta = (sample.action - self._dispatched_reference) * self._native_directions
                    if self._arm_ema_time_constant_s is not None:
                        alpha = -math.expm1(-elapsed / self._arm_ema_time_constant_s)
                        delta[:6] *= alpha
                    # Bound each successfully dispatched native target advance.
                    # A delayed caller receives at most one output tick of travel.
                    if self._native_speed_limits is not None:
                        dt = min(elapsed, 1.0 / self.output_hz)
                        maximum = self._native_speed_limits * dt
                        limited = torch.clamp(delta, -maximum, maximum)
                        if not torch.equal(delta, limited):
                            self._limited_samples += 1
                        output = self._dispatched_reference + limited / self._native_directions
                    else:
                        output = sample.action.clone()
                        output[:6] = (self._dispatched_reference[:6]
                                      + delta[:6] / self._native_directions[:6])
                    sample = replace(sample, action=output)
                    self._last_sample = sample
                if sample is not None and self._arm_transition_rate_deg_s is not None:
                    # Use consecutive successful control ticks, including the send time
                    # between them. Angular anchor comes from the actual vendor result.
                    self._sample_at = float(now)
                    elapsed = max(0.0, self._sample_at - self._committed_sample_at)
                    anchor = self._dispatched_reference[:6]
                    delta = sample.action[:6] - anchor
                    largest_gap = float((delta * self._native_directions[:6]).abs().max())
                    budget = self._arm_transition_rate_deg_s * elapsed
                    if largest_gap > budget:
                        output = sample.action.clone()
                        output[:6] = anchor + delta * (budget / largest_gap)
                        sample = replace(sample, action=output)
                        self._last_sample = sample
                    if self._stale_overlap_cap is not None:
                        sample = self._stale_overlap_cap.limit(
                            sample, anchor=self._dispatched_reference, now=now, elapsed=elapsed)
                        self._last_sample = sample
                self._sample_generation = self._generation
                return sample

        def mark_dispatched(self, reference, *, now):
            with self._reference_lock:
                if self._blocked:
                    return False
                if self._last_sample is None or not math.isfinite(now):
                    raise RuntimeError("Record only a successful sampled dispatch with monotonic now.")
                self._dispatched_reference = _reference(reference)
                self._dispatched_at = float(now)
                if self._arm_transition_rate_deg_s is not None:
                    self._committed_sample_at = self._sample_at
                self._dispatched_timestep = max(self._dispatched_timestep, int(self._last_sample.timestep))
                self._last_incoming = {key: value for key, value in self._last_incoming.items()
                                       if key > self._dispatched_timestep}
                if self._ensemble is not None:
                    self._ensemble.discard_through(self._dispatched_timestep)
                if self._reversal_guard is not None:
                    self._reversal_guard.record(self._dispatched_reference, now, stale=self._last_sample.stale)
                if self._stale_overlap_cap is not None:
                    self._stale_overlap_cap.record(self._dispatched_reference, now, stale=self._last_sample.stale)
                if self._sample_generation != self._generation:
                    self._reanchor(now)
                return True

        def stop(self):
            with self._reference_lock:
                self._blocked = True
                super().reset()
                self._last_incoming.clear()
                if self._ensemble is not None:
                    self._ensemble.reset()
                if self._reversal_guard is not None:
                    self._reversal_guard.reset()
                if self._stale_overlap_cap is not None:
                    self._stale_overlap_cap.reset()

        def execution_snapshot(self):
            with self._reference_lock:
                snapshot = {
                    "temporal_ensemble_coeff": self._ensemble.coefficient if self._ensemble else None,
                    "native_target_speed_limits_deg_s": (self._native_speed_limits.tolist()
                                                          if self._native_speed_limits is not None else None),
                    "speed_limited_samples": self._limited_samples,
                }
                if self._arm_ema_time_constant_s is not None:
                    snapshot["arm_ema_time_constant_s"] = self._arm_ema_time_constant_s
                    snapshot["processing_order"] = (
                        "linear_sample, arm6_ema, optional_native_cap, caller_gripper_guard, successful_dispatch_anchor")
                if self._arm_transition_rate_deg_s is not None:
                    snapshot["arm_transition_rate_deg_s"] = self._arm_transition_rate_deg_s
                if self._reversal_guard is not None:
                    snapshot["shoulder_elbow_reversal_guard"] = True
                    snapshot["shoulder_elbow_reversal_holds"] = self._reversal_guard.holds
                    snapshot["shoulder_elbow_reversal_retained_targets"] = self._reversal_guard.retained_targets
                if self._stale_overlap_cap is not None:
                    snapshot["stale_overlap_reversal_rate_cap"] = True
                    snapshot["stale_overlap_reversal_triggers"] = self._stale_overlap_cap.triggers
                    snapshot["stale_overlap_reversal_limited_samples"] = self._stale_overlap_cap.limited_samples
                return snapshot

    return LinearReanchoredTrajectory


class ArmOnlyTemporalTrajectory:
    """One dispatched action. Independent queues preserve the gripper's original raw-packet semantics."""

    def __init__(self, weighted, arm):
        self.weighted = weighted
        self.arm = arm
        self._lock = threading.RLock()

    def __getattr__(self, name):
        # Refill, coverage and queue diagnostics follow the original client clock.
        return getattr(self.weighted, name)

    @property
    def output_hz(self):
        return self.weighted.output_hz

    @output_hz.setter
    def output_hz(self, hz):
        with self._lock:
            self.weighted.output_hz = self.arm.output_hz = hz

    def prepare_chunk(self, actions):
        return self.arm.prepare_chunk(actions)

    def push_many(self, actions, *, now):
        packet = self.prepare_chunk(actions)
        with self._lock:
            self.arm.push_many(packet, now=now)
            return self.weighted.push_many([a.original for a in packet], now=now)

    def seed(self, reference, *, timestep=-1, now):
        with self._lock:
            self.weighted.seed(reference, timestep=timestep, now=now)
            self.arm.seed(reference, timestep=timestep, now=now)

    def step(self, *, now):
        with self._lock:
            original = self.weighted.step(now=now)
            arm = self.arm.step(now=now)
            if arm is None or original is None:
                return None
            action = arm.action.clone()
            action[6] = original.action[6]
            return replace(original, action=action)

    def mark_dispatched(self, reference, *, now):
        with self._lock:
            self.arm.mark_dispatched(reference, now=now)
            return self.weighted.mark_dispatched(reference, now=now)

    def reset(self):
        with self._lock:
            self.arm.reset()
            self.weighted.reset()

    def stop(self):
        with self._lock:
            self.arm.stop()
            self.weighted.stop()

    def execution_snapshot(self):
        with self._lock:
            return {**self.arm.execution_snapshot(), "temporal_ensemble_scope": "arm6"}


def install_linear_trajectory(client, *, temporal_ensemble_coeff=None, native_speed_limits=None,
                              arm_ema_time_constant_s=None, arm_transition_rate_deg_s=None,
                              temporal_ensemble_scope="all", shoulder_elbow_reversal_guard=False,
                              stale_overlap_reversal_rate_cap=False):
    """Install before client.start, then seed after its reset and before threads."""
    if not client._action_smoothing_enabled:
        raise ValueError("Enable the existing SenseCraft action_smoothing path first.")
    original = client._action_trajectory
    factory = linear_trajectory_class(type(original))
    options = dict(
        source_hz=original.source_hz, output_hz=original.output_hz,
        max_hold_s=original.max_hold_s, low_water_s=original.low_water_s,
        merge_fn=original.merge_fn,
        native_speed_limits=native_speed_limits,
        arm_ema_time_constant_s=arm_ema_time_constant_s,
        arm_transition_rate_deg_s=arm_transition_rate_deg_s,
        shoulder_elbow_reversal_guard=shoulder_elbow_reversal_guard,
        stale_overlap_reversal_rate_cap=stale_overlap_reversal_rate_cap,
    )
    if temporal_ensemble_scope not in ("all", "arm6") or (temporal_ensemble_scope == "arm6" and temporal_ensemble_coeff is None):
        raise ValueError("Select an enabled coefficient for arm6 temporal ensembling.")
    trajectory = factory(temporal_ensemble_coeff=temporal_ensemble_coeff, **options)
    if temporal_ensemble_scope == "arm6":
        if not callable(original.merge_fn):
            raise ValueError("Arm-only temporal ensembling requires the original gripper queue merge.")
        trajectory = ArmOnlyTemporalTrajectory(factory(**options), trajectory)
    client._action_trajectory = trajectory
    return trajectory
