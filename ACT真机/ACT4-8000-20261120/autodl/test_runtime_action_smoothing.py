"""Chunk-boundary regressions using CPU tensors and the pinned scheduler only."""

from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch

from runtime_action_smoothing import _ChunkAction, TemporalChunkEnsembler, install_linear_trajectory, linear_trajectory_class

VENDOR = (
    Path(os.environ["LOCALAPPDATA"]) / "SenseCraft Robotics/product-store/blobs"
    / "645ca8d6ca04a3c48a5dce7b0d141e15d6555d7e2ad7f70c17ce725268ee9948/expanded"
)
spec = importlib.util.spec_from_file_location("pinned_action_trajectory", VENDOR / "backend/run/action_trajectory.py")
scheduler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scheduler
spec.loader.exec_module(scheduler)


@dataclass
class Action:
    timestamp: float
    timestep: int
    action: torch.Tensor

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep

    def get_action(self):
        return self.action


def action(timestep, value, timestamp=1.0):
    return Action(timestamp, timestep, torch.full((7,), float(value)))


def prediction(timestep, value, timestamp=1.0):
    original = action(timestep, value, timestamp)
    if not math.isfinite(timestamp):
        raise ValueError("A finite prediction source timestamp is required.")
    return _ChunkAction(original, original.action, (timestep, timestamp))


class TemporalTests(unittest.TestCase):
    def test_all_predictions_have_official_exponential_weights(self):
        ensemble = TemporalChunkEnsembler(0.01)
        for index, value in enumerate((0, 10, 100)):
            result = ensemble.merge(prediction(12, value, timestamp=index + 1))
        weights = torch.exp(-0.01 * torch.arange(3))
        expected = float((weights * torch.tensor([0, 10, 100])).sum() / weights.sum())
        self.assertAlmostEqual(float(result.action[0]), expected, places=5)
        self.assertNotAlmostEqual(float(result.action[0]), 0.3 * (0.3 * 0 + 0.7 * 10) + 0.7 * 100)

    def test_equal_weight_mode_and_zero_values_are_valid_predictions(self):
        ensemble = TemporalChunkEnsembler(0)
        ensemble.merge(prediction(1, 0, 1))
        result = ensemble.merge(prediction(1, 10, 2))
        self.assertEqual(result.action.tolist(), [5.0] * 7)

    def test_repeated_delivery_does_not_increase_its_weight(self):
        ensemble = TemporalChunkEnsembler(0)
        ensemble.merge(prediction(1, 0, 1))
        ensemble.merge(prediction(1, 10, 2))
        result = ensemble.merge(prediction(1, 10, 2))
        self.assertEqual(result.action.tolist(), [5.0] * 7)

    def test_source_timestamp_orders_predictions_despite_late_delivery(self):
        ensemble = TemporalChunkEnsembler(1)
        ensemble.merge(prediction(1, 10, 2))
        result = ensemble.merge(prediction(1, 0, 1))
        self.assertAlmostEqual(float(result.action[0]), 10 / (math.e + 1), places=5)

    def test_conflicting_duplicate_source_is_rejected(self):
        ensemble = TemporalChunkEnsembler(0)
        ensemble.merge(prediction(1, 0, 1))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            ensemble.merge(prediction(1, 10, 1))

    def test_timesteps_are_separate_and_history_is_pruned(self):
        ensemble = TemporalChunkEnsembler(0)
        ensemble.merge(prediction(1, 10))
        self.assertEqual(ensemble.merge(prediction(2, 20)).action.tolist(), [20.0] * 7)
        ensemble.discard_through(1)
        self.assertEqual(list(ensemble.predictions), [2])
        ensemble.reset()
        self.assertEqual(ensemble.predictions, {})

    def test_invalid_parameters_and_values_are_rejected(self):
        for coefficient in (True, math.nan, math.inf, 1.1):
            with self.assertRaises(ValueError):
                TemporalChunkEnsembler(coefficient)
        with self.assertRaises(ValueError):
            TemporalChunkEnsembler(0).merge(prediction(1, 1, math.nan))
        with self.assertRaises(ValueError):
            TemporalChunkEnsembler(0).merge(prediction(1, math.nan))


class TrajectoryTests(unittest.TestCase):
    @staticmethod
    def make_trajectory(**options):
        trajectory = linear_trajectory_class(scheduler.TimedActionTrajectory)(
            source_hz=30, output_hz=60, merge_fn=lambda old, new: 0.3 * old + 0.7 * new, **options)
        trajectory.seed(torch.zeros(7), now=100.0)
        return trajectory

    def test_existing_profiles_preserve_recursive_merging(self):
        trajectory = self.make_trajectory()
        trajectory.push_many([action(0, 10, 1)], now=100)
        trajectory.push_many([action(0, 20, 2)], now=100)
        self.assertAlmostEqual(float(trajectory._pending[0].action[0]), 17)
        self.assertIsNone(trajectory.execution_snapshot()["native_target_speed_limits_deg_s"])

    def test_temporal_option_avoids_double_blending(self):
        trajectory = self.make_trajectory(temporal_ensemble_coeff=0)
        trajectory.push_many([action(0, 0, 1)], now=100)
        trajectory.push_many([action(0, 10, 2)], now=100)
        self.assertEqual(trajectory._pending[0].action.tolist(), [5.0] * 7)

    def test_large_replan_obeys_native_bounds_with_gripper_scaling(self):
        trajectory = self.make_trajectory(native_speed_limits=[300] * 6 + [720])
        trajectory.push_many([action(index, 40) for index in range(10)], now=100)
        previous = torch.zeros(7)
        for tick in range(1, 20):
            now = 100 + tick / 60
            sample = trajectory.step(now=now)
            delta = (sample.action - previous) * trajectory._native_directions
            self.assertTrue(bool((delta.abs() <= torch.tensor([5] * 6 + [12]) + 1e-5).all()))
            trajectory.mark_dispatched(sample.action, now=now)
            previous = sample.action.clone()
            if tick == 5:
                trajectory.push_many([action(index, -40, 2) for index in range(2, 15)], now=now)
        self.assertGreater(trajectory.execution_snapshot()["speed_limited_samples"], 0)

    def test_delayed_or_immediate_sampling_cannot_accumulate_a_catchup_jump(self):
        trajectory = self.make_trajectory(native_speed_limits=[300] * 6 + [720])
        trajectory.push_many([action(index, 80) for index in range(100)], now=100)
        self.assertEqual(trajectory.step(now=100).action.tolist(), [0.0] * 7)
        sample = trajectory.step(now=101)
        self.assertAlmostEqual(float(sample.action[0]), 5, places=5)
        self.assertAlmostEqual(float(sample.action[6]), 2, places=5)

    def test_same_target_timestamp_from_different_source_chunks_is_combined(self):
        trajectory = self.make_trajectory(temporal_ensemble_coeff=0)
        first = [action(0, 0, 10), action(1, 0, 10 + 1 / 30), action(2, 0, 10 + 2 / 30)]
        second = [action(1, 10, 10 + 1 / 30), action(2, 10, 10 + 2 / 30)]
        trajectory.push_many(first, now=100)
        trajectory.push_many(second, now=100)
        self.assertEqual(trajectory._pending[-1].action.tolist(), [5.0] * 7)

    def test_parent_stale_filter_keeps_full_response_identity(self):
        trajectory = self.make_trajectory(temporal_ensemble_coeff=0)
        prepared = trajectory.prepare_chunk([action(0, 10, 10), action(1, 10, 10 + 1 / 30)])
        trajectory.push_many(prepared[1:], now=100)
        trajectory.push_many(prepared, now=100)
        self.assertEqual(len(trajectory._ensemble.predictions[1]), 1)

    def test_successful_dispatch_result_is_the_next_anchor(self):
        trajectory = self.make_trajectory(native_speed_limits=[300] * 6 + [720])
        trajectory.push_many([action(index, 80) for index in range(100)], now=100)
        sample = trajectory.step(now=100.1)
        actual_sent = sample.action.clone()
        actual_sent[6] = 0.1
        trajectory.mark_dispatched(actual_sent, now=100.1)
        next_sample = trajectory.step(now=100.1 + 1 / 60)
        self.assertLessEqual(float(next_sample.action[6]), 2.10001)

    def test_stale_updates_cannot_move_policy_time_backwards(self):
        trajectory = self.make_trajectory(temporal_ensemble_coeff=0)
        trajectory.push_many([action(index, index) for index in range(10)], now=100)
        sample = trajectory.step(now=100.2)
        trajectory.mark_dispatched(sample.action, now=100.2)
        last = trajectory._dispatched_timestep
        stats = trajectory.push_many([action(last, -80, 2)], now=100.2)
        self.assertEqual(stats["stale"], 1)
        self.assertTrue(all(key > last for key in trajectory._ensemble.predictions))

    def test_chunk_arriving_between_sample_and_send_reanchors_to_actual_send(self):
        trajectory = self.make_trajectory(native_speed_limits=[300] * 6 + [720])
        trajectory.push_many([action(index, 10) for index in range(10)], now=100)
        sample = trajectory.step(now=100.02)
        trajectory.push_many([action(index, -40, 2) for index in range(10)], now=100.021)
        trajectory.mark_dispatched(sample.action, now=100.022)
        self.assertTrue(torch.equal(trajectory._current.action, sample.action))

    def test_stop_and_reset_clear_ensemble_and_dispatch_state(self):
        trajectory = self.make_trajectory(temporal_ensemble_coeff=0)
        trajectory.push_many([action(0, 10)], now=100)
        trajectory.stop()
        self.assertIsNone(trajectory.step(now=101))
        self.assertEqual(trajectory._ensemble.predictions, {})
        trajectory.reset()
        trajectory.seed(torch.ones(7), now=102)
        self.assertEqual(trajectory.step(now=102).action.tolist(), [1.0] * 7)

    def test_install_keeps_the_same_scheduler_configuration(self):
        original = scheduler.TimedActionTrajectory(source_hz=30, output_hz=60, low_water_s=0.7)
        client = SimpleNamespace(_action_smoothing_enabled=True, _action_trajectory=original)
        result = install_linear_trajectory(client, temporal_ensemble_coeff=0.01)
        self.assertIs(client._action_trajectory, result)
        self.assertEqual(result.low_water_s, 0.7)
        self.assertEqual(result.output_hz, 60)


class ShoulderElbowReplanTests(unittest.TestCase):
    @staticmethod
    def make_trajectory(guard=True):
        return TrajectoryTests.make_trajectory(shoulder_elbow_reversal_guard=guard)

    @staticmethod
    def first_chunk():
        return [Action(1.0, step, torch.tensor([
            4.0, 1.5 * (step + 1), 2.0 * (step + 1), -3.0, 5.0, 7.0, 2.0,
        ])) for step in range(24)]

    @staticmethod
    def conflicting_chunk(start, gripper=6.0):
        value = torch.tensor([3.0, -8.0, -10.0, 2.0, 4.0, 5.0, gripper])
        return [Action(2.0, step, value) for step in range(start, 24)]

    @staticmethod
    def dispatch_ticks(trajectory, ticks, *, hold_shoulder_elbow=False):
        for tick in ticks:
            now = 100.0 + tick / 60
            sent = trajectory.step(now=now).action.clone()
            if hold_shoulder_elbow:
                sent[1:3] = 0.0
            trajectory.mark_dispatched(sent, now=now)

    @staticmethod
    def queued(trajectory):
        return {target.timestep: target.action.clone()
                for target in (trajectory._next, *trajectory._pending) if target is not None}

    def test_conflicting_overlap_biases_only_j2_j3_for_a_bounded_prefix(self):
        guarded = self.make_trajectory()
        baseline = self.make_trajectory(guard=False)
        for trajectory in (guarded, baseline):
            trajectory.push_many(self.first_chunk(), now=100.0)
            self.dispatch_ticks(trajectory, range(1, 7))
        start = guarded._dispatched_timestep + 1
        incoming = self.conflicting_chunk(start)
        for trajectory in (guarded, baseline):
            trajectory.push_many(incoming, now=100.1)
        held, normal = self.queued(guarded), self.queued(baseline)
        for step in range(start, start + 6):
            # Each raw conflict exceeds 6°; the scheduler weights the capped bias by 0.7.
            torch.testing.assert_close(held[step][1:3] - normal[step][1:3], torch.full((2,), 4.2))
            torch.testing.assert_close(held[step][[0, 3, 4, 5, 6]], normal[step][[0, 3, 4, 5, 6]])
        torch.testing.assert_close(held[start + 6], normal[start + 6])
        torch.testing.assert_close(incoming[0].action, torch.tensor([3., -8., -10., 2., 4., 5., 6.]))
        self.assertEqual(guarded.execution_snapshot()["shoulder_elbow_reversal_holds"], 1)
        self.assertEqual(guarded.execution_snapshot()["shoulder_elbow_reversal_retained_targets"], 6)

        self.dispatch_ticks(guarded, range(7, 14))
        guarded.push_many(self.conflicting_chunk(guarded._dispatched_timestep + 1, gripper=7.0), now=100.22)
        retained = guarded.execution_snapshot()["shoulder_elbow_reversal_retained_targets"]
        self.assertGreater(retained, 6)
        self.dispatch_ticks(guarded, range(14, 19))
        previous = self.queued(guarded)
        incoming = self.conflicting_chunk(guarded._dispatched_timestep + 1, gripper=8.0)
        guarded.push_many(incoming, now=100.31)
        step = incoming[0].timestep
        expected = 0.3 * previous[step] + 0.7 * incoming[0].action
        torch.testing.assert_close(self.queued(guarded)[step], expected)
        self.assertEqual(guarded.execution_snapshot()["shoulder_elbow_reversal_holds"], 1)
        self.assertEqual(guarded.execution_snapshot()["shoulder_elbow_reversal_retained_targets"], retained)

    def test_first_chunk_and_steady_arm_leave_overlapping_replans_unchanged(self):
        guarded = self.make_trajectory()
        baseline = self.make_trajectory(guard=False)
        for trajectory in (guarded, baseline):
            trajectory.push_many(self.first_chunk(), now=100.0)
        torch.testing.assert_close(self.queued(guarded), self.queued(baseline))
        for trajectory in (guarded, baseline):
            self.dispatch_ticks(trajectory, range(1, 7), hold_shoulder_elbow=True)
            trajectory.push_many(self.conflicting_chunk(trajectory._dispatched_timestep + 1), now=100.1)
        torch.testing.assert_close(self.queued(guarded), self.queued(baseline))
        self.assertEqual(guarded.execution_snapshot()["shoulder_elbow_reversal_holds"], 0)

    def test_reset_clears_existing_hold_and_recent_dispatch_history(self):
        trajectory = self.make_trajectory()
        trajectory.push_many(self.first_chunk(), now=100.0)
        self.dispatch_ticks(trajectory, range(1, 7))
        trajectory.push_many(self.conflicting_chunk(trajectory._dispatched_timestep + 1), now=100.1)
        self.assertEqual(trajectory.execution_snapshot()["shoulder_elbow_reversal_holds"], 1)

        trajectory.reset()
        trajectory.seed(torch.zeros(7), now=100.11)
        first = self.first_chunk()
        incoming = self.conflicting_chunk(0)
        trajectory.push_many(first, now=100.11)
        trajectory.push_many(incoming, now=100.12)
        torch.testing.assert_close(self.queued(trajectory)[0], 0.3 * first[0].action + 0.7 * incoming[0].action)
        self.assertEqual(trajectory.execution_snapshot()["shoulder_elbow_reversal_holds"], 0)
        self.assertEqual(trajectory.execution_snapshot()["shoulder_elbow_reversal_retained_targets"], 0)


class StaleOverlapReversalTests(unittest.TestCase):
    def moving_trajectory(self, enabled=True):
        trajectory = TrajectoryTests.make_trajectory(
            arm_transition_rate_deg_s=165, stale_overlap_reversal_rate_cap=enabled)
        trajectory.push_many(ShoulderElbowReplanTests.first_chunk(), now=100.0)
        ShoulderElbowReplanTests.dispatch_ticks(trajectory, range(1, 7))
        return trajectory

    @staticmethod
    def merge_conflict(trajectory, *, now, age, gripper=6.0):
        start = trajectory._dispatched_timestep - 1
        actions = ShoulderElbowReplanTests.conflicting_chunk(start, gripper=gripper)
        for index, action in enumerate(actions):
            action.timestamp = 200.0 - age + index / 30
        prepared = trajectory.prepare_chunk(actions, receipt_monotonic=now, receipt_wall_time=200.0)
        trajectory.push_many(prepared[2:], now=now)

    def test_caps_only_the_opposed_axes_with_old_overlapping_input(self):
        guarded = self.moving_trajectory()
        baseline = self.moving_trajectory(enabled=False)
        for trajectory in (guarded, baseline):
            self.merge_conflict(trajectory, now=100.1, age=.35)
        guarded_queue = ShoulderElbowReplanTests.queued(guarded)
        baseline_queue = ShoulderElbowReplanTests.queued(baseline)
        for step, target in guarded_queue.items():
            torch.testing.assert_close(target, baseline_queue[step])

        anchor = guarded._dispatched_reference.clone()
        now = 100.0 + 7 / 60
        limited = guarded.step(now=now)
        normal = baseline.step(now=now)
        torch.testing.assert_close(limited.action[1:3] - anchor[1:3], -torch.ones(2))
        self.assertTrue(((normal.action[1:3] - anchor[1:3]) < -1.0).all())
        torch.testing.assert_close(limited.action[[0, 3, 4, 5, 6]], normal.action[[0, 3, 4, 5, 6]])
        self.assertEqual(guarded.execution_snapshot()["stale_overlap_reversal_triggers"], 1)
        self.assertEqual(guarded.execution_snapshot()["stale_overlap_reversal_limited_samples"], 1)

    def test_fresh_conflict_and_unusable_timestamp_keep_standard_path(self):
        for age in (.29, -.1, math.nan):
            with self.subTest(age=age):
                guarded = self.moving_trajectory()
                baseline = self.moving_trajectory(enabled=False)
                for trajectory in (guarded, baseline):
                    self.merge_conflict(trajectory, now=100.1, age=age)
                now = 100.0 + 7 / 60
                torch.testing.assert_close(guarded.step(now=now).action, baseline.step(now=now).action)
                self.assertEqual(guarded.execution_snapshot()["stale_overlap_reversal_triggers"], 0)

    def test_fresh_chunk_clears_cap_and_reset_clears_history(self):
        guarded = self.moving_trajectory()
        baseline = self.moving_trajectory(enabled=False)
        for trajectory in (guarded, baseline):
            self.merge_conflict(trajectory, now=100.1, age=.4)
            self.merge_conflict(trajectory, now=100.11, age=.09, gripper=7.0)
        now = 100.0 + 7 / 60
        torch.testing.assert_close(guarded.step(now=now).action, baseline.step(now=now).action)
        self.assertEqual(guarded.execution_snapshot()["stale_overlap_reversal_triggers"], 1)
        self.assertEqual(guarded.execution_snapshot()["stale_overlap_reversal_limited_samples"], 0)
        guarded.reset()
        guarded.seed(torch.zeros(7), now=100.11)
        self.merge_conflict(guarded, now=100.12, age=.4)
        self.assertEqual(guarded.execution_snapshot()["stale_overlap_reversal_triggers"], 0)

    def test_stale_repeated_update_does_not_extend_deadline(self):
        guarded = self.moving_trajectory()
        self.merge_conflict(guarded, now=100.1, age=.4)
        ShoulderElbowReplanTests.dispatch_ticks(guarded, range(7, 12))
        self.merge_conflict(guarded, now=100.19, age=.41, gripper=7.0)
        ShoulderElbowReplanTests.dispatch_ticks(guarded, range(12, 14))
        limited = guarded.execution_snapshot()["stale_overlap_reversal_limited_samples"]
        ShoulderElbowReplanTests.dispatch_ticks(guarded, range(14, 18))
        self.assertEqual(guarded.execution_snapshot()["stale_overlap_reversal_limited_samples"], limited)
        self.assertEqual(guarded.execution_snapshot()["stale_overlap_reversal_triggers"], 1)


if __name__ == "__main__":
    unittest.main()
