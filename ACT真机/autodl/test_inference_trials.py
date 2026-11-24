"""Inference lifecycle regressions with fake clients, no CAN, RPC or hotkeys."""

import queue
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import inference_trials as target


class Channel:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Sender:
    def __init__(self):
        self.stopped = False
        self.can_stop = True

    def stop(self, timeout_s):
        self.stopped = True
        return self.can_stop


class Hotkeys:
    def __init__(self, on_retry, on_finish):
        self.on_retry = on_retry
        self.on_finish = on_finish
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self, timeout):
        self.closed = True

    def raise_if_failed(self):
        pass


class Client:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.channel = Channel()
        self._observation_sender = Sender()
        self.action_queue = queue.Queue()
        self.action_queue_lock = threading.Lock()
        self._actuator_failed = False
        self._transport_failure = None
        self.actuator_can_stop = True
        self.robot = object()
        self.attempts = 0
        self.on_control = None

    def request_stop(self):
        self.shutdown_event.set()
        self.channel.close()

    def _join_actuator_thread(self, timeout_s):
        return self.actuator_can_stop

    def start(self):
        self.shutdown_event.clear()
        return True

    def receive_actions(self):
        self.shutdown_event.wait(1)

    def control_loop(self, task):
        self.attempts += 1
        self.action_queue.put("old action")
        self.on_control(self.attempts)


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.profile = {**target.DEFAULT_PROFILE, "max_trials": 2, "rest_seconds": 1}
        self.stop = target.TrialStopController(SimpleNamespace(), self.profile,
                                               emit=lambda line: None)
        self.stop.event = lambda **event: self.events.append(event)
        self.stop.target_native_rad = {joint: 0 for joint in target.JOINTS}
        self.client = Client()
        self.stop.attach(self.client)
        self.returns = []
        self.stop.return_saved = lambda robot: self.returns.append(robot) or {"verified": True}
        self.keyboards = []
        self.rebuilds = []

    def keyboard(self, on_retry, on_finish):
        keys = Hotkeys(on_retry, on_finish)
        self.keyboards.append(keys)
        return keys

    def rebuild(self, client, episode_id):
        self.assertTrue(client.action_queue.empty(), "Old actions survived the trial boundary")
        self.rebuilds.append(episode_id)
        client.channel = Channel()
        client._observation_sender = Sender()
        client.stub = SimpleNamespace(episode_id=episode_id)

    def run_trials(self, rest=lambda *args, **kwargs: None, *, seed=lambda client: None):
        with patch.object(target, "rebuild_transport", side_effect=self.rebuild), \
             patch.object(target, "start_trial_with_hold", side_effect=lambda client, stop: client.start()), \
             patch("inference_safe_return.hold_saved_pose", side_effect=rest):
            target.run_trials(self.client, SimpleNamespace(task="test task"), self.stop,
                              seed=seed, hotkeys_factory=self.keyboard)

    def test_failed_attempt_returns_rests_and_starts_fresh_epoch(self):
        self.client.on_control = lambda attempt: (
            self.stop.request_retry() if attempt == 1 else self.stop.request_home())
        self.run_trials()
        self.assertEqual(self.client.attempts, 2)
        self.assertEqual(len(set(self.rebuilds)), 2)
        self.assertEqual(len(self.returns), 1)
        outcomes = [event["outcome"] for event in self.events if event["event"] == "trial_end"]
        self.assertEqual(outcomes, ["failed", "unknown"])
        self.assertEqual(self.stop._intent, "home")
        self.assertTrue(self.keyboards[0].started)
        self.assertTrue(self.keyboards[0].closed)

    def test_time_limit_remains_unknown(self):
        self.profile["max_trials"] = 1
        self.client.on_control = lambda attempt: self.stop.request_retry("time_limit_unknown")
        self.run_trials()
        end = next(event for event in self.events if event["event"] == "trial_end")
        self.assertEqual(end["outcome"], "unknown")
        self.assertEqual(self.stop._intent, "home")

    def test_finish_during_rest_prevents_next_attempt(self):
        self.client.on_control = lambda attempt: self.stop.request_retry()
        self.run_trials(rest=lambda *args, **kwargs: self.stop.request_home())
        self.assertEqual(self.client.attempts, 1)
        self.assertEqual(self.stop._intent, "home")

    def test_manual_active_control_c_returns_then_continues_once(self):
        self.profile["manual_advance"] = True
        self.client.on_control = lambda attempt: (
            self.keyboards[0].on_retry() if attempt == 1 else self.stop.request_home())
        rests = []

        def hold(robot, pose, duration, **kwargs):
            self.assertEqual(self.client.attempts, 1)
            self.assertEqual(duration, self.profile["rest_seconds"])
            self.assertFalse(self.stop.waiting_for_continue)
            self.assertTrue(self.stop.continue_requested.is_set())
            self.stop.request_advance()
            rests.append(duration)

        self.run_trials(rest=hold)
        self.assertEqual(self.client.attempts, 2)
        self.assertEqual(rests, [self.profile["rest_seconds"]])
        self.assertEqual(len(self.returns), 1)
        self.assertFalse(self.stop.waiting_for_continue)
        self.assertFalse(self.stop.continue_requested.is_set())
        self.assertEqual(len(set(self.rebuilds)), 2)
        endings = [event for event in self.events if event["event"] == "trial_end"]
        self.assertEqual(endings[0]["outcome"], "unknown")
        self.assertEqual(endings[0]["reason"], "operator_continue")

    def test_manual_control_v_while_waiting_prevents_next_attempt(self):
        self.profile["manual_advance"] = True
        self.client.on_control = lambda attempt: self.stop.request_retry("time_limit_unknown")

        def hold(robot, pose, duration, **kwargs):
            if self.stop.waiting_for_continue:
                self.keyboards[0].on_finish()

        self.run_trials(rest=hold)
        self.assertEqual(self.client.attempts, 1)
        self.assertEqual(self.stop._intent, "home")
        self.assertFalse(self.stop.waiting_for_continue)

    def test_require_fresh_control_c_after_each_verified_return(self):
        self.profile.update(operator_controlled=True, manual_advance=True, max_trials=None,
                            trial_seconds=None, rest_seconds=0, require_continue_after_return=True)
        home_checks = []
        waits = {}

        def action(attempt):
            self.keyboards[0].on_retry()
            self.assertFalse(self.stop.continue_requested.is_set())

        def home(robot):
            self.keyboards[0].on_retry()
            self.assertFalse(self.stop.continue_requested.is_set())
            home_checks.append(self.client.attempts)
            return {"verified": True}

        def hold(robot, pose, duration, **kwargs):
            attempt = self.client.attempts
            self.assertTrue(self.stop.waiting_for_continue)
            self.assertEqual(home_checks[-1], attempt)
            self.assertFalse(self.stop.continue_requested.is_set())
            waits[attempt] = waits.get(attempt, 0) + 1
            if waits[attempt] == 3:
                if attempt < 2:
                    self.keyboards[0].on_retry()
                else:
                    self.keyboards[0].on_finish()

        self.client.on_control = action
        self.stop.return_saved = home
        self.run_trials(rest=hold)
        self.assertEqual(waits, {1: 3, 2: 3})
        self.assertEqual(len(set(self.rebuilds)), 2)
        self.assertEqual(self.stop._intent, "home")
        self.assertEqual(len([e for e in self.events if e["event"] == "operator_continue_ignored"]), 3)

    def test_manual_control_v_after_continue_has_priority(self):
        self.profile["manual_advance"] = True
        self.client.on_control = lambda attempt: self.stop.request_retry()

        def hold(robot, pose, duration, **kwargs):
            if self.stop.waiting_for_continue:
                self.stop.request_advance()
                self.stop.request_home()

        self.run_trials(rest=hold)
        self.assertEqual(self.client.attempts, 1)
        self.assertEqual(self.stop._intent, "home")

    def test_manual_control_c_during_return_is_queued_once(self):
        self.profile["manual_advance"] = True
        self.client.on_control = lambda attempt: (
            self.stop.request_retry("time_limit_unknown") if attempt == 1 else self.stop.request_home())
        homes = []

        def home(robot):
            self.stop.request_advance()
            self.stop.request_advance()
            self.assertTrue(self.stop.continue_requested.is_set())
            homes.append(robot)
            return {"verified": True}

        def hold(robot, pose, duration, **kwargs):
            self.assertEqual(duration, self.profile["rest_seconds"])

        self.stop.return_saved = home
        self.run_trials(rest=hold)
        self.assertEqual(self.client.attempts, 2)
        self.assertEqual(len(homes), 1)

    def test_timeout_waits_for_control_c_and_returns_to_saved_pose_again(self):
        self.profile["manual_advance"] = True
        self.client.on_control = lambda attempt: (
            self.stop.request_retry("time_limit_unknown") if attempt == 1 else self.stop.request_home())
        waits = []

        def hold(robot, pose, duration, **kwargs):
            self.assertEqual(self.client.attempts, 1)
            self.assertFalse(self.stop.continue_requested.is_set())
            if self.stop.waiting_for_continue:
                waits.append(duration)
                if len(waits) == 3:
                    self.keyboards[0].on_retry()

        self.run_trials(rest=hold)
        self.assertEqual(waits, [0.2, 0.2, 0.2])
        self.assertEqual(len(self.returns), 2)
        self.assertEqual(self.client.attempts, 2)

    def test_continue_latch_is_consumed_before_next_trial_timeout(self):
        self.profile.update(manual_advance=True, max_trials=3)
        self.client.on_control = lambda attempt: (
            self.stop.request_advance() if attempt == 1 else self.stop.request_retry("time_limit_unknown"))

        def hold(robot, pose, duration, **kwargs):
            if self.stop.waiting_for_continue:
                self.assertEqual(self.client.attempts, 2)
                self.assertFalse(self.stop.continue_requested.is_set())
                self.stop.request_home()

        self.run_trials(rest=hold)
        self.assertEqual(self.client.attempts, 2)

    def test_control_v_during_queued_continue_return_wins(self):
        self.profile["manual_advance"] = True
        self.client.on_control = lambda attempt: self.stop.request_advance()
        self.stop.return_saved = lambda robot: self.stop.request_home() or {"verified": True}
        self.run_trials(rest=lambda *args, **kwargs: self.fail("Stop must prevent rest and next trial"))
        self.assertEqual(self.client.attempts, 1)
        self.assertEqual(self.stop._intent, "home")

    def test_profile_without_manual_field_preserves_automatic_mode(self):
        profile = dict(target.DEFAULT_PROFILE)
        profile.pop("manual_advance")
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "read_text", return_value=json.dumps(profile)):
            loaded = target.load_profile(target.PROFILE_PATH)
        self.assertFalse(loaded["manual_advance"])

    def test_profile_rejects_non_boolean_manual_flag(self):
        profile = {**target.DEFAULT_PROFILE, "manual_advance": "true"}
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "read_text", return_value=json.dumps(profile)):
            with self.assertRaisesRegex(ValueError, "manual_advance"):
                target.load_profile(target.PROFILE_PATH)

    def test_profile_accepts_explicit_candidate_and_preserves_old_defaults(self):
        processing = {"temporal_ensemble_coeff": 0.01, "limit_native_target_speed": True}
        profile = {**target.DEFAULT_PROFILE, "action_processing": processing,
                   "return_monitor": {"enabled": True}}
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "read_text", return_value=json.dumps(profile)):
            loaded = target.load_profile(target.PROFILE_PATH)
        self.assertEqual(loaded["action_processing"], processing)
        self.assertTrue(loaded["return_monitor"]["enabled"])

    def test_profile_rejects_invalid_execution_options_before_connect(self):
        for processing in ({"temporal_ensemble_coeff": True}, {"temporal_ensemble_coeff": float("nan")},
                           {"limit_native_target_speed": "true"}, {"unknown_option": True},
                           {"stale_overlap_reversal_rate_cap": "true"},
                           {"shoulder_elbow_reversal_guard": True, "stale_overlap_reversal_rate_cap": True}):
            profile = {**target.DEFAULT_PROFILE, "action_processing": processing}
            with patch.object(Path, "exists", return_value=True), \
                    patch.object(Path, "read_text", return_value=json.dumps(profile)):
                with self.assertRaises(ValueError):
                    target.load_profile(target.PROFILE_PATH)

    def test_return_phase_monitor_is_advisory_and_ignores_nonpolicy_observations(self):
        profile = {**self.profile, "return_monitor": {"enabled": True}}
        stop = target.TrialStopController(SimpleNamespace(), profile, emit=lambda line: None)
        events = []
        stop.event = lambda **event: events.append(event)
        away = {"shoulder_lift.pos": 130, "elbow_flex.pos": 120}
        near = {"shoulder_lift.pos": 1, "elbow_flex.pos": 1}
        stop.observe_return_phase(away, task="test")
        self.assertFalse(stop.return_monitor._departed)
        stop.policy_active = True
        with patch.object(target.time, "monotonic", side_effect=[0, 0.1, 0.3, 0.5, 0.7]):
            for observation in (away, near, near, near, near):
                stop.observe_return_phase(observation, task="test")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "learned_return_phase_detected")
        self.assertTrue(events[0]["advisory_only"])
        self.assertFalse(stop.retry_requested.is_set())
        self.assertFalse(stop.pause_requested.is_set())

    def test_environment_selects_the_same_profile_for_adapter_and_cli_default(self):
        profile = {**target.DEFAULT_PROFILE, "manual_advance": True}
        selected = target.ROOT / "runs/act4_224_scratch_20261109/inference_20261110/trials_profile.json"
        with patch.dict(target.os.environ, {"ACT_INFERENCE_TRIALS_PROFILE": str(selected)}), \
                patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "read_text", autospec=True, return_value=json.dumps(profile)) as read:
            loaded = target.load_profile()
        self.assertEqual(read.call_args.args[0], selected)
        self.assertTrue(loaded["manual_advance"])

    def test_hold_cancels_home_and_retry(self):
        self.stop.request_retry()
        self.stop.request_home()
        self.stop.request_hold("fresh fault")
        self.stop.request_retry()
        self.assertEqual(self.stop._intent, "hold")
        self.assertTrue(self.stop.homing_cancelled())
        self.assertEqual(self.stop.failure, "fresh fault")

    def test_pause_blocks_policy_dispatch(self):
        sent = []
        self.stop.request_retry()
        result = self.stop.dispatch(lambda action: sent.append(action), {"a.pos": 1})
        self.assertIsNone(result)
        self.assertEqual(sent, [])

    def test_unexpected_loop_end_refuses_automatic_resume(self):
        self.client.on_control = lambda attempt: None
        with self.assertRaisesRegex(RuntimeError, "without retry/finish"):
            self.run_trials()
        self.assertEqual(self.client.attempts, 1)
        self.assertEqual(self.stop._intent, "hold")
        self.assertEqual(self.returns, [])

    def test_actuator_failure_refuses_home_retry(self):
        def fail(attempt):
            self.stop.request_retry()
            self.client._actuator_failed = True
            self.client._actuator_error = "fake actuator failure"
        self.client.on_control = fail
        with self.assertRaisesRegex(RuntimeError, "fake actuator failure"):
            self.run_trials()
        self.assertEqual(self.stop._intent, "hold")
        self.assertEqual(self.returns, [])

    def test_live_actuator_refuses_trial_handoff(self):
        self.client.actuator_can_stop = False
        self.client.on_control = lambda attempt: self.stop.request_retry()
        with self.assertRaisesRegex(RuntimeError, "Old actuator"):
            self.run_trials()
        self.assertEqual(self.returns, [])

    def test_live_uploader_refuses_trial_handoff(self):
        def fail(attempt):
            self.stop.request_retry()
            self.client._observation_sender.can_stop = False
        self.client.on_control = fail
        with self.assertRaisesRegex(RuntimeError, "Old observation"):
            self.run_trials()
        self.assertEqual(self.returns, [])

    def test_finish_only_releases_after_saved_pose_verification(self):
        sequence = []
        self.stop.return_saved = lambda robot: sequence.append("verified") or {}
        self.stop.release = lambda robot: sequence.append("release") or []
        self.stop.request_home()
        self.stop._home_and_release(self.client.robot)
        self.assertEqual(sequence, ["verified", "release"])
        self.assertTrue(self.stop.released)

    def test_failed_return_does_not_call_release(self):
        def fail(robot):
            raise RuntimeError("fake return fault")
        self.stop.return_saved = fail
        release_calls = []
        self.stop.release = lambda robot: release_calls.append(robot)
        self.stop.request_home()
        with self.assertRaisesRegex(RuntimeError, "fake return fault"):
            self.stop._home_and_release(self.client.robot)
        self.assertEqual(release_calls, [])


if __name__ == "__main__":
    unittest.main()
