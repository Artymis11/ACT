"""Exercise episode races with fake policies and RPC contexts only."""

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
import threading
import time
import unittest
from unittest import mock

import episode_policy_guard
from episode_policy_guard import install_episode_guard


class Aborted(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


class FakeContext:
    def __init__(self, episode_id="one", peer="client", metadata=None):
        self.metadata = ([("x-act-episode-id", episode_id)] if episode_id is not None else [])
        if metadata is not None:
            self.metadata = metadata
        self.client_peer = peer
        self.active = True
        self.initial_metadata = []

    def invocation_metadata(self):
        return self.metadata

    def peer(self):
        return self.client_peer

    def is_active(self):
        return self.active

    def send_initial_metadata(self, metadata):
        self.initial_metadata.extend(metadata)

    def abort(self, code, detail):
        raise Aborted(code, detail)


class Resettable:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def guarded_server_class():
    class FakeServer:
        def __init__(self):
            self.config = SimpleNamespace(obs_queue_timeout=0.1, inference_latency=0)
            self.contract_handshake_complete = False
            self.observation_queue = Queue(maxsize=1)
            self._predicted_timesteps = set()
            self._predicted_timesteps_lock = threading.Lock()
            self.last_processed_obs = None
            self.fps_tracker = Resettable()
            self.policy = None
            self.preprocessor = None
            self.postprocessor = None
            self.loads = 0
            self.predictions = []
            self.prediction_started = threading.Event()
            self.prediction_continue = threading.Event()
            self.prediction_continue.set()
            self.stopped = False

        def Ready(self, request, context):
            self.contract_handshake_complete = False
            self.observation_queue = Queue(maxsize=1)
            self._predicted_timesteps = set()
            return SimpleNamespace(data=b"")

        def SendPolicyInstructions(self, request, context):
            if request.data == b"invalid":
                context.abort("INVALID_ARGUMENT", "validation failed")
            self.loads += 1
            self.policy = Resettable()
            self.preprocessor = Resettable()
            self.postprocessor = Resettable()
            self.contract_handshake_complete = True
            return SimpleNamespace(data=b"")

        def SendObservations(self, request_iterator, context):
            if not self.contract_handshake_complete:
                context.abort("FAILED_PRECONDITION", "handshake required")
            for observation in request_iterator:
                self._enqueue_observation(observation)
            return SimpleNamespace(data=b"")

        def GetActions(self, request, context):
            raise AssertionError("Guard must replace unsafe original GetActions.")

        def _enqueue_observation(self, observation):
            self.observation_queue.put_nowait(observation)
            return True

        def _predict_action_chunk(self, observation):
            self.prediction_started.set()
            if not self.prediction_continue.wait(2):
                raise AssertionError("Test did not release prediction.")
            self.last_processed_obs = observation
            self.predictions.append(observation.get_timestep())
            return [observation.get_timestep()]

        def stop(self):
            self.stopped = True

    codes = SimpleNamespace(INVALID_ARGUMENT="INVALID_ARGUMENT", CANCELLED="CANCELLED",
                            FAILED_PRECONDITION="FAILED_PRECONDITION", RESOURCE_EXHAUSTED="RESOURCE_EXHAUSTED")
    return install_episode_guard(FakeServer, status_codes=codes,
                                 empty_factory=lambda: SimpleNamespace(data=b""),
                                 actions_factory=lambda data: SimpleNamespace(data=data))


class EpisodePolicyGuardTests(unittest.TestCase):
    def setUp(self):
        self.server = guarded_server_class()()
        self.request = SimpleNamespace(data=b"valid-contract")

    def start(self, episode_id="one", peer="client"):
        context = FakeContext(episode_id, peer)
        self.server.Ready(None, context)
        self.server.SendPolicyInstructions(self.request, context)
        return context

    def observation(self, timestep=0):
        return SimpleNamespace(get_timestep=lambda: timestep)

    def assert_aborted(self, code, callback):
        with self.assertRaises(Aborted) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def test_tagged_ready_advertises_capability_and_requires_handshake(self):
        context = FakeContext()
        self.server.Ready(None, context)
        self.assertEqual(context.initial_metadata, [("act-episode-reset", "v1")])
        self.assert_aborted("FAILED_PRECONDITION", lambda: self.server.GetActions(None, context))

    def test_legacy_client_remains_single_session_compatible(self):
        context = self.start(None)
        self.assertEqual(context.initial_metadata, [])
        self.server.SendObservations(iter([self.observation()]), context)
        self.assertTrue(self.server.GetActions(None, context).data)
        other_peer = FakeContext(None, peer="other")
        self.assert_aborted("FAILED_PRECONDITION", lambda: self.server.GetActions(None, other_peer))

    def test_legacy_ready_cannot_replace_tagged_episode(self):
        self.start()
        self.assert_aborted("FAILED_PRECONDITION", lambda: self.server.Ready(None, FakeContext(None)))

    def test_metadata_validation_rejects_duplicates_and_invalid_identifiers(self):
        for metadata in [[("x-act-episode-id", "a"), ("x-act-episode-id", "b")],
                         [("x-act-episode-id", "")], [("x-act-episode-id", "with space")]]:
            self.assert_aborted("INVALID_ARGUMENT", lambda: self.server.Ready(None, FakeContext(metadata=metadata)))

    def test_repeated_specs_reuse_model_and_reset_all_state(self):
        self.start()
        policy = self.server.policy
        self.server.last_processed_obs = self.observation(50)
        self.server._predicted_timesteps.add(50)
        self.server.observation_queue.put(self.observation(51))
        second = self.start("two")
        self.assertEqual(self.server.loads, 1)
        self.assertIs(self.server.policy, policy)
        self.assertEqual(self.server.policy.resets, 2)
        self.assertIsNone(self.server.last_processed_obs)
        self.assertEqual(self.server._predicted_timesteps, set())
        self.assertTrue(self.server.observation_queue.empty())
        self.assertEqual(second.initial_metadata, [("act-episode-reset", "v1")])

    def test_changed_specs_revalidate_and_failed_load_invalidates_cache(self):
        self.start()
        second = FakeContext("two")
        self.server.Ready(None, second)
        self.assert_aborted("INVALID_ARGUMENT", lambda: self.server.SendPolicyInstructions(
            SimpleNamespace(data=b"invalid"), second))
        self.assertFalse(self.server.contract_handshake_complete)
        self.server.SendPolicyInstructions(self.request, second)
        self.assertEqual(self.server.loads, 2)

    def test_old_episode_calls_and_replayed_ready_are_rejected(self):
        old_context = self.start()
        self.start("two")
        self.assert_aborted("FAILED_PRECONDITION", lambda: self.server.SendObservations(
            iter([self.observation()]), old_context))
        self.assert_aborted("FAILED_PRECONDITION", lambda: self.server.GetActions(None, old_context))
        self.assert_aborted("FAILED_PRECONDITION", lambda: self.server.Ready(None, old_context))

    def test_ready_waits_for_inflight_prediction_then_clears_its_state(self):
        old_context = self.start()
        self.server.SendObservations(iter([self.observation(7)]), old_context)
        self.server.prediction_continue.clear()
        with ThreadPoolExecutor(max_workers=2) as pool:
            prediction = pool.submit(self.server.GetActions, None, old_context)
            self.assertTrue(self.server.prediction_started.wait(1))
            next_context = FakeContext("two")
            ready = pool.submit(self.server.Ready, None, next_context)
            time.sleep(0.02)
            self.assertFalse(ready.done())
            self.server.prediction_continue.set()
            try:
                prediction.result(1)
            except Aborted as error:
                self.assertEqual(error.code, "FAILED_PRECONDITION")
            ready.result(1)
        self.assertIsNone(self.server.last_processed_obs)
        self.assertEqual(self.server._predicted_timesteps, set())

    def test_waiting_getactions_does_not_block_observation_writer(self):
        context = self.start()
        with ThreadPoolExecutor(max_workers=2) as pool:
            action = pool.submit(self.server.GetActions, None, context)
            self.server.SendObservations(iter([self.observation(4)]), context)
            self.assertTrue(action.result(1).data)

    def test_old_waiter_cannot_consume_next_episode_queue(self):
        old_context = self.start()
        with ThreadPoolExecutor(max_workers=1) as pool:
            old_action = pool.submit(self.server.GetActions, None, old_context)
            time.sleep(0.02)
            new_context = self.start("two")
            self.server.SendObservations(iter([self.observation(9)]), new_context)
            self.assert_aborted("FAILED_PRECONDITION", lambda: old_action.result(1))
        self.assertTrue(self.server.GetActions(None, new_context).data)
        self.assertEqual(self.server.predictions, [9])

    def test_old_stream_finishing_after_ready_cannot_enqueue(self):
        old_context = self.start()
        stream_started = threading.Event()
        stream_continue = threading.Event()

        def delayed_stream():
            stream_started.set()
            self.assertTrue(stream_continue.wait(1))
            yield self.observation(8)

        with ThreadPoolExecutor(max_workers=1) as pool:
            upload = pool.submit(self.server.SendObservations, delayed_stream(), old_context)
            self.assertTrue(stream_started.wait(1))
            self.start("two")
            stream_continue.set()
            self.assert_aborted("FAILED_PRECONDITION", lambda: upload.result(1))
        self.assertTrue(self.server.observation_queue.empty())

    def test_cancelled_context_cannot_reset_or_predict(self):
        context = FakeContext()
        context.active = False
        self.assert_aborted("CANCELLED", lambda: self.server.Ready(None, context))
        context = self.start()
        context.active = False
        self.assert_aborted("CANCELLED", lambda: self.server.GetActions(None, context))

    def test_stop_invalidates_current_episode(self):
        context = self.start()
        self.server.stop()
        self.assertTrue(self.server.stopped)
        self.assert_aborted("FAILED_PRECONDITION", lambda: self.server.GetActions(None, context))

    def test_private_calls_require_an_active_rpc(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError, "active rollout RPC"):
            self.server._predict_action_chunk(self.observation())
        with self.assertRaisesRegex(RuntimeError, "active rollout RPC"):
            self.server._enqueue_observation(self.observation())

    def test_staged_observation_payload_bytes_are_bounded(self):
        context = self.start()
        message = SimpleNamespace(data=b"123456789")
        with mock.patch.object(episode_policy_guard, "MAX_OBSERVATION_BYTES", 8):
            self.assert_aborted("RESOURCE_EXHAUSTED", lambda: self.server.SendObservations(iter([message]), context))
        self.assertTrue(self.server.observation_queue.empty())

    def test_staged_empty_chunk_count_is_bounded(self):
        context = self.start()
        messages = [SimpleNamespace(data=b"")] * 3
        with mock.patch.object(episode_policy_guard, "MAX_OBSERVATION_MESSAGES", 2):
            self.assert_aborted("RESOURCE_EXHAUSTED", lambda: self.server.SendObservations(iter(messages), context))
        self.assertTrue(self.server.observation_queue.empty())


if __name__ == "__main__":
    unittest.main()
