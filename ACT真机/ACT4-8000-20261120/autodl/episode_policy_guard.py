"""Serialize cloud policy resets and isolate asynchronous rollout episodes.

The guard stages observation RPC messages before taking the policy lock, and
waits for observations outside that lock. Network waits therefore cannot block
the writer that makes an action request ready. No model code is imported here.
"""

from contextlib import contextmanager
import hashlib
import pickle
from queue import Empty
import re
import threading
import time


EPISODE_METADATA_KEY = "x-act-episode-id"
CAPABILITY_METADATA = ("act-episode-reset", "v1")
MAX_OBSERVATION_BYTES = 16 * 1024 * 1024
MAX_OBSERVATION_MESSAGES = 128
_VALID_EPISODE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


class EpisodePolicyGuard:
    def __init__(self, server, status_codes, empty_factory, actions_factory):
        self.server = server
        self.status_codes = status_codes
        self.empty_factory = empty_factory
        self.actions_factory = actions_factory
        self.lock = threading.RLock()
        self.local = threading.local()
        self.generation = 0
        self.episode_id = None
        self.legacy_peer = None
        self.retired_ids = set()
        self.policy_fingerprint = None
        self.ready_complete = False

    def abort(self, context, code, message):
        context.abort(getattr(self.status_codes, code), message)
        raise RuntimeError(message)  # A test context must also terminate the call.

    def read_episode_id(self, context):
        values = [value for key, value in context.invocation_metadata()
                  if key.lower() == EPISODE_METADATA_KEY]
        if len(values) > 1:
            self.abort(context, "INVALID_ARGUMENT", "Supply one x-act-episode-id value.")
        if not values:
            return None
        if not isinstance(values[0], str) or _VALID_EPISODE_ID.fullmatch(values[0]) is None:
            self.abort(context, "INVALID_ARGUMENT", "Invalid x-act-episode-id value.")
        return values[0]

    def ensure_active(self, context):
        if not context.is_active():
            self.abort(context, "CANCELLED", "The rollout RPC was cancelled.")

    def capture(self, context):
        self.ensure_active(context)
        episode_id = self.read_episode_id(context)
        if self.generation == 0 or not self.ready_complete:
            self.abort(context, "FAILED_PRECONDITION", "Complete Ready before rollout RPCs.")
        if episode_id != self.episode_id:
            self.abort(context, "FAILED_PRECONDITION", "The rollout episode has been superseded.")
        if episode_id is None and context.peer() != self.legacy_peer:
            self.abort(context, "FAILED_PRECONDITION", "This legacy client does not own the active rollout.")
        return self.generation

    def ensure_current(self, generation, context):
        self.ensure_active(context)
        if generation != self.generation:
            self.abort(context, "FAILED_PRECONDITION", "The rollout episode has been superseded.")
        self.capture(context)

    @contextmanager
    def request_scope(self, generation, context):
        previous = getattr(self.local, "request", None)
        self.local.request = (generation, context)
        try:
            yield
        finally:
            self.local.request = previous

    def ensure_private_request(self):
        request = getattr(self.local, "request", None)
        if request is None:
            raise RuntimeError("Guarded policy operations require an active rollout RPC.")
        generation, context = request
        self.ensure_current(generation, context)

    def reset_policy_state(self):
        for component in (self.server.policy, self.server.preprocessor, self.server.postprocessor):
            reset = getattr(component, "reset", None)
            if reset is not None:
                reset()

    def ready(self, original, request, context):
        episode_id = self.read_episode_id(context)
        with self.lock:
            self.ensure_active(context)
            if episode_id is not None and episode_id in self.retired_ids:
                self.abort(context, "FAILED_PRECONDITION", "A retired rollout episode cannot reconnect.")
            if episode_id is None and self.episode_id is not None:
                self.abort(context, "FAILED_PRECONDITION", "Legacy Ready cannot replace a tagged rollout.")
            if self.episode_id is not None and self.episode_id != episode_id:
                self.retired_ids.add(self.episode_id)
            # Invalidate even duplicate Ready calls, so pending RPCs cannot cross
            # a handshake boundary. The next handshake must validate the model.
            self.generation += 1
            self.ready_complete = False
            self.episode_id = episode_id
            self.legacy_peer = context.peer() if episode_id is None else None
            self.server.contract_handshake_complete = False
            response = original(self.server, request, context)
            self.server.last_processed_obs = None
            self.server.fps_tracker.reset()
            self.reset_policy_state()
            self.ensure_active(context)
            self.ready_complete = True
            if episode_id is not None:
                context.send_initial_metadata((CAPABILITY_METADATA,))
            return response

    def instructions(self, original, request, context):
        with self.lock:
            generation = self.capture(context)
            self.server.contract_handshake_complete = False
            fingerprint = hashlib.sha256(bytes(request.data)).digest()
            if fingerprint == self.policy_fingerprint and self.server.policy is not None:
                # This exact payload already passed ImagePolicyServer validation.
                self.reset_policy_state()
                self.server.contract_handshake_complete = True
                response = self.empty_factory()
            else:
                # A failed replacement may leave partially initialized policy
                # objects. It must invalidate the preceding cache identity.
                self.policy_fingerprint = None
                with self.request_scope(generation, context):
                    response = original(self.server, request, context)
                if self.server.policy is not None and self.server.contract_handshake_complete:
                    self.policy_fingerprint = fingerprint
            self.ensure_current(generation, context)
            return response

    def observations(self, original, request_iterator, context):
        with self.lock:
            generation = self.capture(context)
        # Original SendObservations updates shared FPS and queue state. Consume
        # the network stream first so its remaining work can run under the lock.
        messages = []
        payload_bytes = 0
        for message in request_iterator:
            self.ensure_active(context)
            payload_bytes += len(getattr(message, "data", b""))
            if payload_bytes > MAX_OBSERVATION_BYTES or len(messages) >= MAX_OBSERVATION_MESSAGES:
                self.abort(context, "RESOURCE_EXHAUSTED", "The staged observation exceeds the rollout RPC limit.")
            messages.append(message)
        with self.lock:
            self.ensure_current(generation, context)
            with self.request_scope(generation, context):
                return original(self.server, iter(messages), context)

    def actions(self, request, context):
        started = time.perf_counter()
        with self.lock:
            generation = self.capture(context)
            if not self.server.contract_handshake_complete:
                self.abort(context, "FAILED_PRECONDITION", "Complete the matching policy handshake first.")
            observation_queue = self.server.observation_queue
        try:
            observation = observation_queue.get(timeout=self.server.config.obs_queue_timeout)
        except Empty:
            with self.lock:
                self.ensure_current(generation, context)
            return self.actions_factory(data=b"")

        with self.lock:
            self.ensure_current(generation, context)
            with self.server._predicted_timesteps_lock:
                self.server._predicted_timesteps.add(observation.get_timestep())
            with self.request_scope(generation, context):
                action_chunk = self.server._predict_action_chunk(observation)
            self.ensure_current(generation, context)
            actions = self.actions_factory(data=pickle.dumps(action_chunk))
        # Preserve upstream pacing while allowing Ready and observation uploads.
        delay = max(0.0, self.server.config.inference_latency - (time.perf_counter() - started))
        if delay:
            time.sleep(delay)
        with self.lock:
            self.ensure_current(generation, context)
        return actions


def install_episode_guard(server_class, *, status_codes=None, empty_factory=None, actions_factory=None):
    """Install once on ImagePolicyServer before any instance is constructed."""
    if server_class.__dict__.get("_episode_guard_installed", False):
        return server_class
    if status_codes is None:
        import grpc
        status_codes = grpc.StatusCode
    if empty_factory is None or actions_factory is None:
        from lerobot.transport import services_pb2
        empty_factory = empty_factory or services_pb2.Empty
        actions_factory = actions_factory or services_pb2.Actions
    original_init = server_class.__init__
    original_ready = server_class.Ready
    original_instructions = server_class.SendPolicyInstructions
    original_observations = server_class.SendObservations
    original_enqueue = server_class._enqueue_observation
    original_predict = server_class._predict_action_chunk
    original_stop = server_class.stop

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._episode_guard = EpisodePolicyGuard(self, status_codes, empty_factory, actions_factory)

    def ready(self, request, context):
        return self._episode_guard.ready(original_ready, request, context)

    def instructions(self, request, context):
        return self._episode_guard.instructions(original_instructions, request, context)

    def observations(self, request_iterator, context):
        return self._episode_guard.observations(original_observations, request_iterator, context)

    def actions(self, request, context):
        return self._episode_guard.actions(request, context)

    def enqueue(self, observation):
        with self._episode_guard.lock:
            self._episode_guard.ensure_private_request()
            try:
                return original_enqueue(self, observation)
            except Empty:
                # Upstream tests full() before get_nowait(). An action waiter
                # may consume that item between those two queue operations.
                # With this producer lock held, retrying sees an empty queue.
                return original_enqueue(self, observation)

    def predict(self, observation):
        with self._episode_guard.lock:
            self._episode_guard.ensure_private_request()
            return original_predict(self, observation)

    def stop(self):
        with self._episode_guard.lock:
            self._episode_guard.ready_complete = False
            self._episode_guard.generation += 1
            return original_stop(self)

    server_class.__init__ = initialize
    server_class.Ready = ready
    server_class.SendPolicyInstructions = instructions
    server_class.SendObservations = observations
    server_class.GetActions = actions
    server_class._enqueue_observation = enqueue
    server_class._predict_action_chunk = predict
    server_class.stop = stop
    server_class._episode_guard_installed = True
    return server_class
