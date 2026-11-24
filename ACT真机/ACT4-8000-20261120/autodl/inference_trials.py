"""Inference-only episodic lifecycle. No hardware or model is created here.

In operator-controlled sessions Ctrl+C returns an active trial to the saved pose.
A new Ctrl+C after the verified return starts the next trial.
Ctrl+V ends the run. The existing client owns home/rest/retry.
"""

import argparse
import hashlib
from itertools import count
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

from sensecraft_224_stop import StopController

ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "inference_trials.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_yaw", "wrist_roll", "gripper")
DEFAULT_PROFILE = {
    "schema": "act-inference-trials-v1", "enabled": True,
    "max_trials": 20, "trial_seconds": 30.0, "rest_seconds": 5.0,
    "tolerance_deg": 2.0, "return_timeout_seconds": 45.0,
    "return_speed_deg_s": 25.0,
    "zero_state_path": str(ROOT / "zero_state.json"),
    "capture_request_path": str(ROOT / "zero_state_capture_request.json"),
    "require_cloud_episode_guard": True, "manual_advance": False, "operator_saved_pose": False,
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_profile(path=None):
    path = Path(path if path is not None else os.environ.get("ACT_INFERENCE_TRIALS_PROFILE", PROFILE_PATH))
    profile = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else dict(DEFAULT_PROFILE)
    profile.setdefault("manual_advance", False)
    if type(profile["manual_advance"]) is not bool:
        raise ValueError("manual_advance must be a boolean.")
    profile.setdefault("operator_saved_pose", False)
    if type(profile["operator_saved_pose"]) is not bool:
        raise ValueError("operator_saved_pose must be a boolean.")
    processing = profile.get("action_processing", {})
    if not isinstance(processing, dict) or set(processing) - {
            "temporal_ensemble_coeff", "limit_native_target_speed", "native_gripper_speed_deg_s",
            "action_output_hz", "arm_ema_time_constant_s", "arm_transition_rate_deg_s", "temporal_ensemble_scope",
            "shoulder_elbow_reversal_guard", "stale_overlap_reversal_rate_cap"}:
        raise ValueError("Invalid action_processing configuration.")
    output_hz = processing.get("action_output_hz", 60)
    if type(output_hz) is not int or output_hz not in (30, 60):
        raise ValueError("action_output_hz must be the reviewed integer cadence30 or60.")
    if type(processing.get("limit_native_target_speed", False)) is not bool:
        raise ValueError("limit_native_target_speed must be a boolean.")
    if type(processing.get("shoulder_elbow_reversal_guard", False)) is not bool:
        raise ValueError("shoulder_elbow_reversal_guard must be a boolean.")
    if type(processing.get("stale_overlap_reversal_rate_cap", False)) is not bool:
        raise ValueError("stale_overlap_reversal_rate_cap must be a boolean.")
    if processing.get("shoulder_elbow_reversal_guard") and processing.get("stale_overlap_reversal_rate_cap"):
        raise ValueError("Select one overlap processing method.")
    coefficient = processing.get("temporal_ensemble_coeff")
    scope = processing.get("temporal_ensemble_scope", "all")
    if scope not in ("all", "arm6") or (scope == "arm6" and coefficient is None):
        raise ValueError("Select an enabled coefficient for arm6 temporal ensembling.")
    if coefficient is not None and (isinstance(coefficient, bool) or not isinstance(coefficient, (int, float))
                                    or not math.isfinite(coefficient) or abs(coefficient) > 1):
        raise ValueError("temporal_ensemble_coeff must be null or finite within [-1, 1].")
    arm_ema = processing.get("arm_ema_time_constant_s")
    arm_rate = processing.get("arm_transition_rate_deg_s")
    if arm_rate is not None and (
            type(arm_rate) not in (int, float) or not 0 < arm_rate <= sys.float_info.max):
        raise ValueError("arm_transition_rate_deg_s must be null or finite and positive.")
    if arm_ema is not None:
        if isinstance(arm_ema, bool) or not isinstance(arm_ema, (int, float)):
            raise ValueError("arm_ema_time_constant_s must be null or finite and positive.")
        try:
            arm_ema = float(arm_ema)
        except OverflowError as error:
            raise ValueError("arm_ema_time_constant_s must be null or finite and positive.") from error
        if not math.isfinite(arm_ema) or arm_ema <= 0:
            raise ValueError("arm_ema_time_constant_s must be null or finite and positive.")
    if "native_gripper_speed_deg_s" in processing:
        speed = processing["native_gripper_speed_deg_s"]
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not math.isfinite(speed) or not 0 < speed <= 720):
            raise ValueError("native_gripper_speed_deg_s must be finite and within (0, 720].")
    monitor = profile.get("return_monitor", {})
    if not isinstance(monitor, dict) or set(monitor) - {
            "enabled", "departure_deg", "terminal_abs_deg", "dwell_seconds", "max_sample_gap"}:
        raise ValueError("Invalid return_monitor configuration.")
    from learned_return_monitor import LearnedReturnMonitor
    LearnedReturnMonitor(**monitor)
    if profile.get("schema") != DEFAULT_PROFILE["schema"] or type(profile.get("enabled")) is not bool:
        raise ValueError("Invalid inference trial profile schema/enabled flag.")
    manual = profile.get("operator_controlled", False)
    if type(manual) is not bool:
        raise ValueError("operator_controlled must be a boolean.")
    require_continue = profile.get("require_continue_after_return", False)
    if type(require_continue) is not bool or (require_continue and not manual):
        raise ValueError("require_continue_after_return requires an operator-controlled trial profile.")
    if manual:
        profile["require_continue_after_return"] = True
    park_gripper = profile.get("park_gripper_after_return", False)
    if type(park_gripper) is not bool or (park_gripper and not (manual and profile["operator_saved_pose"])):
        raise ValueError("park_gripper_after_return requires a saved pose and manual continuation after return.")
    session_reference = profile.get("session_start_reference", False)
    if type(session_reference) is not bool or (session_reference and not park_gripper):
        raise ValueError("session_start_reference requires manual arm return with separate gripper parking.")
    manual_limit = profile.get("max_trials")
    if manual and (not profile["manual_advance"]
                   or (manual_limit is not None and (type(manual_limit) is not int or manual_limit != 1))
                   or profile.get("trial_seconds") is not None or profile.get("rest_seconds") != 0):
        raise ValueError("Operator-controlled trials require manual advance, an optional one-trial limit, "
                         "no timer, and no scheduled rest.")
    if not manual and (type(profile.get("max_trials")) is not int or not 1 <= profile["max_trials"] <= 100):
        raise ValueError("max_trials must be an integer from 1 to 100.")
    for key, minimum, maximum in (("trial_seconds", 1, 600), ("rest_seconds", 1, 600),
                                  ("tolerance_deg", 0.1, 2), ("return_timeout_seconds", 5, 180),
                                  ("return_speed_deg_s", 1, 25)):
        value = profile.get(key)
        if manual and key in ("trial_seconds", "rest_seconds"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {key} in inference trial profile.")
    if profile.get("require_cloud_episode_guard") is not True:
        raise ValueError("Cloud episode isolation cannot be disabled for sequential trials.")
    for key in ("zero_state_path", "capture_request_path"):
        target = (path.parent / Path(profile[key])).resolve()
        if not target.is_relative_to(ROOT.parent.resolve()):
            raise ValueError(f"{key} must stay inside the ACT project.")
        profile[key] = str(target)
    return profile


def robot_identity(report):
    path = Path(report["calibration_dir"]) / (report["robot_id"] + ".json")
    return {"robot_type": report["robot_type"], "robot_id": report["robot_id"],
            "robot_port": report["robot_port"],
            "calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "joint_directions": report["joint_directions_in_robot_send_action"]}


def load_zero(profile, identity):
    saved = json.loads(Path(profile["zero_state_path"]).read_text(encoding="utf-8-sig"))
    if saved.get("schema") != "act-native-zero-state-v1" or saved.get("identity") != identity:
        raise ValueError("zero_state robot/calibration identity does not match this run.")
    target = saved.get("native_rad", {})
    if set(target) != set(JOINTS) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                        or not math.isfinite(x) for x in target.values()):
        raise ValueError("zero_state must contain seven finite motor-native radian positions.")
    return saved


def valid_capture_request(profile, now=None):
    path = Path(profile["capture_request_path"])
    if not path.exists():
        return False
    request = json.loads(path.read_text(encoding="utf-8-sig"))
    age = (time.time() if now is None else now) - float(request["created_unix_s"])
    if request.get("schema") != "act-zero-capture-request-v1" or not 0 <= age <= 3600:
        raise ValueError("Saved-pose capture request expired. Re-arm it after positioning the arm.")
    return True


def prepare_zero(robot, profile, identity, *, audit=lambda **values: None):
    if valid_capture_request(profile):
        from inference_safe_return import capture_zero_state
        # Consume before attempting a read: a failed startup cannot silently
        # capture a different posture during a later unrelated run.
        Path(profile["capture_request_path"]).unlink()
        positions = capture_zero_state(robot, operator_saved_pose=profile.get("operator_saved_pose", False),
                                       audit=lambda event, **values: audit(event=event, **values))
        saved = {"schema": "act-native-zero-state-v1", "identity": identity,
                 "created_unix_s": time.time(), "native_rad": positions,
                 "source": "explicit operator one-shot request; fresh native motor reads"}
        saved["operator_saved_pose"] = profile.get("operator_saved_pose", False)
        atomic_json(profile["zero_state_path"], saved)
        audit(event="zero_state_saved", native_rad=positions)
        return saved
    return load_zero(profile, identity)


class EpisodeStub:
    """Tag every RPC with one trial ID; require an advertised reset barrier."""
    def __init__(self, stub, episode_id):
        self.stub = stub
        self.episode_id = episode_id
        self.guard_confirmed = False

    def __getattr__(self, name):
        method = getattr(self.stub, name)
        def call(*args, **kwargs):
            metadata = list(kwargs.pop("metadata", ()) or ())
            metadata.append(("x-act-episode-id", self.episode_id))
            kwargs["metadata"] = metadata
            if name == "Ready":
                kwargs.setdefault("timeout", 15.0)
                response, rpc = method.with_call(*args, **kwargs)
                metadata_received = dict(rpc.initial_metadata() or ())
                if metadata_received.get("act-episode-reset") != "v1":
                    raise RuntimeError("Cloud service lacks act-episode-reset=v1. Deploy the episode guard before robot inference.")
                self.guard_confirmed = True
                return response
            if name == "SendPolicyInstructions":
                if not self.guard_confirmed:
                    raise RuntimeError("Cloud reset capability has not been verified.")
                kwargs.setdefault("timeout", 60.0)
            elif name in {"GetActions", "SendObservations"}:
                kwargs.setdefault("timeout", 5.0)
            return method(*args, **kwargs)
        return call


class TrialStopController(StopController):
    """Permanent stop retains the existing fault/confirmed-release lifecycle."""
    def __init__(self, vendor, profile, **kwargs):
        super().__init__(vendor, **kwargs)
        self.profile = profile
        self.pause_requested = threading.Event()
        self.retry_requested = threading.Event()
        self.continue_requested = threading.Event()
        self.continue_needs_home = False
        self._retry_lock = threading.Lock()
        self.waiting_for_continue = False
        self.retry_reason = None
        self.target_native_rad = None
        self.session_start_native_rad = None
        self.observer = None
        self.trial_id = None
        self.policy_active = False
        self.gripper_parked = False
        from learned_return_monitor import LearnedReturnMonitor
        self.return_monitor = LearnedReturnMonitor(
            **profile.get("return_monitor", {}), on_returned=self._return_phase_detected)

    def _return_phase_detected(self):
        self.event(event="learned_return_phase_detected", advisory_only=True,
                   meaning="J2/J3 return phase candidate; gripper and full task completion unverified")

    def observe_return_phase(self, observation, *, task):
        if self.policy_active:
            self.return_monitor.observe(observation, timestamp=time.monotonic())

    def event(self, **values):
        record = {"monotonic_s": time.monotonic(), "trial_id": self.trial_id, **values}
        self.emit("[act224-trial] " + json.dumps(record, sort_keys=True, allow_nan=False))
        observer = self.observer
        if observer is not None:
            observer.on_event(record)

    def request_advance(self):
        with self._retry_lock:
            if self.stop_requested.is_set():
                return
            waiting = self.waiting_for_continue
            manual = self.profile.get("operator_controlled", False)
            ignored = manual and not waiting and self.pause_requested.is_set()
            if waiting or not manual:
                self.continue_requested.set()
                self.continue_needs_home = self.continue_needs_home or waiting
        if ignored:
            self.event(event="operator_continue_ignored", key="Ctrl+C", reason="return_in_progress")
            return
        self.event(event="operator_continue_requested", key="Ctrl+C", waiting=waiting)
        if not waiting:
            self.request_retry("operator_continue")

    def request_retry(self, reason="operator_failed", *, expected_trial=None):
        with self._retry_lock:
            if expected_trial is not None and expected_trial != self.trial_id:
                return
            if self.stop_requested.is_set() or self.pause_requested.is_set():
                return
            self.retry_reason = reason
            self.pause_requested.set()
            self.retry_requested.set()
        self._interrupt()

    def request_home(self):
        self.pause_requested.set()
        super().request_home()

    def request_hold(self, reason="Explicit damped hold or stdin EOF"):
        self.pause_requested.set()
        super().request_hold(reason)

    def dispatch(self, send, action):
        with self.action_lock:
            if self.pause_requested.is_set() or self.stop_requested.is_set():
                return None
            if self.gripper_parked and "gripper.pos" in action:
                self.gripper_parked = False
            return super().dispatch(send, action)

    def homing_cancelled(self):
        with self._state_lock:
            return self._intent == "hold"

    def normal_stop_requested(self):
        return self.retry_requested.is_set() or super().normal_stop_requested()

    def allow_transport_cancellation(self, error):
        if not self.normal_stop_requested() or getattr(self.client, "_transport_failure", None):
            return False
        import grpc
        if isinstance(error, grpc.RpcError):
            return error.code() == grpc.StatusCode.CANCELLED
        return str(error) == "observation transport returned unsuccessful result"

    def return_saved(self, robot):
        from inference_safe_return import run_return
        if self.target_native_rad is None:
            raise RuntimeError("No operator-saved zero_state is available. Retain fault hold.")
        controller = self
        class Cancel:
            def is_set(self):
                return controller.homing_cancelled()
        receipt = run_return(
            robot, self.target_native_rad,
            operator_saved_pose=self.profile.get("operator_saved_pose", False),
            tolerance_deg=self.profile["tolerance_deg"],
            timeout_s=self.profile["return_timeout_seconds"],
            speed_deg_s=self.profile["return_speed_deg_s"],
            paced=self.profile.get("operator_controlled", False),
            park_gripper=self.profile.get("park_gripper_after_return", False),
            gripper_parked=self.gripper_parked,
            cancel_event=Cancel(), audit=lambda event, **kw: self.event(event=event, **kw))
        self.gripper_parked = receipt.get("gripper_parked", False)
        return receipt

    def _home_and_release(self, robot):
        with self._state_lock:
            self._phase = "homing"
        receipt = self.return_saved(robot)
        with self._state_lock:
            if self._intent != "home":
                raise RuntimeError("Fault hold requested before release.")
            self._phase = "releasing"
        self.event(event="zero_state_verified", receipt=receipt)
        cleanup = self.release(robot)
        self.released = True
        self.event(event="zero_state_disabled_closed", camera_cleanup_errors=cleanup)


def quiesce_trial(client, receiver):
    """No homing or new channel while any old action/RPC writer is alive."""
    client.request_stop()
    stop_history = getattr(client, "stop_observation_history", None)
    if callable(stop_history):
        stop_history()
    if not client._join_actuator_thread(timeout_s=2.0):
        raise RuntimeError("Old actuator still owns the action path; retry refused.")
    if not client._observation_sender.stop(timeout_s=2.0):
        raise RuntimeError("Old observation uploader still running; retry refused.")
    if receiver is not None:
        receiver.join(timeout=2.0)
        if receiver.is_alive():
            raise RuntimeError("Old action receiver still running; retry refused.")
    if getattr(client, "_actuator_failed", False) or getattr(client, "_transport_failure", None):
        raise RuntimeError(str(getattr(client, "_actuator_error", None)
                               or getattr(client, "_transport_failure", None)))
    with client.action_queue_lock:
        while True:
            try:
                client.action_queue.get_nowait()
            except queue.Empty:
                break


def rebuild_transport(client, episode_id):
    """Rebuild RPC/sender state only; preserve the one robot connection."""
    from backend.run.observation_sender import LatestObservationSender
    from backend.run.remote_pipeline_diagnostics import RemotePipelineDiagnostics
    from lerobot.async_inference import robot_client as official
    client.channel = official.grpc.insecure_channel(
        client.server_address,
        official.grpc_channel_options(initial_backoff=f"{client.config.environment_dt:.4f}s"))
    client.stub = EpisodeStub(official.services_pb2_grpc.AsyncInferenceStub(client.channel), episode_id)
    # A watchdog rebuild would create an untagged stub and erase the epoch
    # guarantee. Sequential trials use bounded calls instead (EpisodeStub).
    client._rpc_watchdog = None
    client._observation_sender = LatestObservationSender(
        client._send_observation_sync, key_fn=lambda obs: obs.get_timestep(),
        force_fn=lambda obs: bool(getattr(obs, "must_go", False)),
        on_failure=client._mark_transport_failure,
        thread_name="act224-trial-observation-sender")
    client.start_barrier = threading.Barrier(2, timeout=10.0)
    with client.latest_action_lock:
        client.latest_action = -1
    client.action_chunk_size = -1
    client.action_queue_size = []
    client.must_go.set()
    client._pipeline_diagnostics = RemotePipelineDiagnostics()
    client._next_observation_capture_at = 0.0
    client._actuator_thread = None


def start_trial_with_hold(client, stop):
    """Cloud handshake runs without CAN I/O; this owner maintains its posture."""
    from inference_safe_return import capture_zero_state, hold_saved_pose
    pose = stop.session_start_native_rad
    hold_options = {"arm_only": True} if pose is not None else {}
    if pose is None:
        pose = capture_zero_state(client.robot, operator_saved_pose=stop.profile.get("operator_saved_pose", False))
    done = threading.Event()
    result = []
    errors = []
    def handshake():
        try:
            result.append(client.start())
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    worker = threading.Thread(target=handshake, name="act224-cloud-handshake", daemon=True)
    worker.start()
    try:
        while not done.is_set():
            if stop.stop_requested.is_set() or stop.pause_requested.is_set():
                client.request_stop()
                if not done.wait(2.0):
                    raise RuntimeError("Cloud handshake did not cancel; retry refused.")
                break
            hold_saved_pose(client.robot, pose, 0.2, cancel_event=stop.pause_requested,
                            operator_saved_pose=stop.profile.get("operator_saved_pose", False),
                            gripper_parked=stop.gripper_parked, **hold_options)
    except BaseException:
        client.request_stop()
        raise
    finally:
        worker.join(timeout=2.0)
    if worker.is_alive():
        raise RuntimeError("Cloud handshake still running; retry refused.")
    if errors:
        raise errors[0]
    return bool(result and result[0])


def wait_for_continue(client, stop):
    """Hold this pose until the latched Ctrl+C continuation or Ctrl+V finish."""
    from inference_safe_return import hold_saved_pose
    with stop._retry_lock:
        stop.waiting_for_continue = True
    stop.event(event="waiting_for_continue", next_trial_id=stop.trial_id + 1,
               torque="enabled_hold", continue_key="Ctrl+C", finish_key="Ctrl+V")
    try:
        while not stop.stop_requested.is_set() and not stop.continue_requested.is_set():
            hold_saved_pose(client.robot, stop.target_native_rad, 0.2,
                            operator_saved_pose=stop.profile.get("operator_saved_pose", False),
                            gripper_parked=stop.gripper_parked,
                            cancel_event=stop.stop_requested,
                            tolerance_deg=stop.profile["tolerance_deg"],
                            audit=lambda event, **values: stop.event(event=event, **values))
    finally:
        with stop._retry_lock:
            needs_home = stop.continue_needs_home
            stop.waiting_for_continue = False
            stop.continue_requested.clear()
            stop.continue_needs_home = False
    if needs_home and not stop.stop_requested.is_set():
        with stop.action_lock:
            receipt = stop.return_saved(client.robot)
            stop.event(event="continue_zero_verified", receipt=receipt)


def run_trials(client, config, stop, *, seed, hotkeys_factory=None):
    """Same process, one hardware owner, fresh cloud epoch for each full trial."""
    if hotkeys_factory is None:
        from inference_hotkeys import InferenceHotkeys
        hotkeys_factory = InferenceHotkeys
    hotkeys = hotkeys_factory(stop.request_advance, stop.request_home)
    receiver = None
    monitor_done = threading.Event()
    deadline = [None]
    def monitor():
        while not monitor_done.wait(0.02):
            try:
                hotkeys.raise_if_failed()
            except Exception as error:
                stop.request_hold(f"Inference keyboard listener failed: {error}")
                return
            expires = deadline[0]
            if expires is not None and time.monotonic() >= expires[1]:
                stop.request_retry("time_limit_unknown", expected_trial=expires[0])
    watcher = None
    try:
        hotkeys.start()  # Register before cloud actions can execute.
        watcher = threading.Thread(target=monitor, name="act224-trial-supervisor", daemon=True)
        watcher.start()
        manual = stop.profile.get("operator_controlled", False)
        for trial_id in (count(1) if manual else range(1, stop.profile["max_trials"] + 1)):
            if stop.stop_requested.is_set():
                break
            stop.trial_id = trial_id
            stop.return_monitor.reset()
            with stop._retry_lock:
                if not (manual and trial_id == 1):
                    stop.retry_requested.clear()
                    stop.retry_reason = None
                    stop.pause_requested.clear()
                    stop.continue_requested.clear()
                    stop.continue_needs_home = False
            # Fresh UUID prevents cached/in-flight prior cloud work from being
            # interpreted as part of this attempt even when timestep resets.
            if trial_id == 1:
                client.channel.close()
            rebuild_transport(client, uuid.uuid4().hex)
            stop.event(event="trial_start", episode_id=client.stub.episode_id)
            if stop.stop_requested.is_set():
                break
            started = start_trial_with_hold(client, stop)
            if not started and not (stop.stop_requested.is_set() or stop.pause_requested.is_set()):
                raise RuntimeError("The cloud trial handshake failed.")
            if stop.stop_requested.is_set() or stop.pause_requested.is_set():
                quiesce_trial(client, None)
            else:
                seed(client)
                if not stop.stop_requested.is_set() and not stop.pause_requested.is_set():
                    if not manual:
                        deadline[0] = (trial_id, time.monotonic() + stop.profile["trial_seconds"])
                    receiver = threading.Thread(target=client.receive_actions,
                                                name="act224-trial-action-receiver", daemon=True)
                    receiver.start()
                    stop.policy_active = True
                    client.control_loop(task=config.task)
            stop.policy_active = False
            deadline[0] = None
            quiesce_trial(client, receiver)
            receiver = None
            reason = stop.retry_reason
            outcome = "failed" if reason == "operator_failed" else "unknown"
            stop.event(event="trial_end", outcome=outcome,
                       reason=reason or ("operator_finish" if stop.stop_requested.is_set() else "unexpected_stop"))
            if stop.stop_requested.is_set():
                break
            if not stop.retry_requested.is_set():
                raise RuntimeError("Trial ended without retry/finish intent; automatic resume refused.")
            if trial_id == stop.profile["max_trials"]:
                stop.request_home()
                break
            with stop.action_lock:
                stop.event(event="return_for_retry")
                receipt = stop.return_saved(client.robot)
                stop.event(event="retry_zero_verified", receipt=receipt)
            if stop.stop_requested.is_set():
                break
            if manual:
                wait_for_continue(client, stop)
                continue
            stop.event(event="rest_started", seconds=stop.profile["rest_seconds"], torque="enabled_hold")
            from inference_safe_return import hold_saved_pose
            hold_saved_pose(client.robot, stop.target_native_rad, stop.profile["rest_seconds"],
                            operator_saved_pose=stop.profile.get("operator_saved_pose", False),
                            cancel_event=stop.stop_requested,
                            tolerance_deg=stop.profile["tolerance_deg"],
                            audit=lambda event, **values: stop.event(event=event, **values))
            if stop.stop_requested.is_set():
                break
            stop.event(event="rest_ended")
            if stop.profile.get("manual_advance", False):
                wait_for_continue(client, stop)
                if stop.stop_requested.is_set():
                    break
    except BaseException as error:
        stop.request_hold(f"{type(error).__name__}: {error}")
        raise
    finally:
        stop.policy_active = False
        deadline[0] = None
        monitor_done.set()
        if watcher is not None:
            watcher.join(timeout=1.0)
        hotkeys.close(timeout=2.0)
        if receiver is not None:
            receiver.join(timeout=2.0)


def check_cloud_guard(config):
    """Verify the deployed cloud reset protocol before touching the robot."""
    from lerobot.async_inference import robot_client as official
    channel = official.grpc.insecure_channel(config.server_address,
                  official.grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"))
    try:
        stub = EpisodeStub(official.services_pb2_grpc.AsyncInferenceStub(channel), uuid.uuid4().hex)
        stub.Ready(official.services_pb2.Empty())
    finally:
        channel.close()


def release_saved_robot(robot):
    from backend.common.b601_rs_mit import stop_b601_rs_mit_controller
    from sensecraft_224_stop import release_verified_robot
    # Native RS mode may not have this optional controller. The helper returns
    # false in that case, and otherwise joins its sole writer before disable.
    stop_b601_rs_mit_controller(robot)
    return release_verified_robot(robot)


def wait_for_first_start(stop, keys):
    """Latch a start key with no connected client or motor owner yet."""
    with stop._retry_lock:
        stop.waiting_for_continue = True
    try:
        keys.start()
        stop.event(event="waiting_for_first_start", start_key="Ctrl+C", finish_key="Ctrl+V",
                   hardware_connected=False)
        while not stop.stop_requested.is_set() and not stop.continue_requested.wait(.05):
            keys.raise_if_failed()
        return not stop.stop_requested.is_set()
    finally:
        with stop._retry_lock:
            stop.waiting_for_continue = False
            stop.continue_needs_home = False
            stop.continue_requested.clear()


def run_managed_session(vendor, config, contract, profile, report, client_factory):
    """Adapter entry. Missing target/server capability blocks before connect."""
    identity = robot_identity(report)
    capture_pending = valid_capture_request(profile)
    session_reference = profile.get("session_start_reference", False)
    if session_reference and (capture_pending or not report.get("live_history_feedback_verified")):
        raise ValueError("Session start reference requires verified live history and no pending saved-zero capture.")
    initial_zero = None if capture_pending else load_zero(profile, identity)
    check_cloud_guard(config)
    os.environ["ACT_FEEDBACK_MODEL_DIRECTORY"] = report["contract_model_path"]
    weights_path = Path(report["contract_model_path"]) / "model.safetensors"
    if weights_path.is_file():
        with weights_path.open("rb") as source:
            os.environ["ACT_FEEDBACK_MODEL_SHA256"] = hashlib.file_digest(source, "sha256").hexdigest()
    stop = TrialStopController(vendor, profile, release=release_saved_robot,
                              emit=lambda text: print(text, file=sys.stderr, flush=True))
    if profile.get("operator_controlled", False) and initial_zero is not None and not session_reference:
        stop.target_native_rad = initial_zero["native_rad"]
    client_class = client_factory(vendor.InternalRobotClient, contract, stop)
    holder = vendor.TaskTextHolder(config.task)
    client = None
    feedback = None
    manual_keys = None
    caught = None
    stdin_controls = {"stop": stop.request_home, "home_stop": stop.request_home,
                      "trial_retry": stop.request_retry, "damped_hold": stop.request_hold,
                      "confirm_release": stop.confirm_release}
    if not profile.get("operator_controlled", False):
        stdin_controls["trial_continue"] = stop.request_advance
    vendor.start_stdin_control_listener(
        stdin_controls,
        prefix_handlers={"task": holder.set}, on_eof=stop.request_hold,
        thread_name="act224-inference-stdin")
    try:
        if profile.get("operator_controlled", False):
            from inference_hotkeys import InferenceHotkeys
            manual_keys = InferenceHotkeys(stop.request_advance, stop.request_home)
            if not wait_for_first_start(stop, manual_keys):
                return
        if stop.stop_requested.is_set():
            return
        client = client_class(config)
        client._task_text_holder = holder
        from inference_feedback_observer import InferenceFeedbackObserver, compose_observer
        feedback = InferenceFeedbackObserver()
        if os.environ.get("ACT_TRIAL_VIDEO") == "1":
            try:
                from inference_trial_video import TrialVideoRecorder
                feedback.trial_video = TrialVideoRecorder(
                    feedback.directory, client.robot.cameras,
                )
            except Exception as error:
                feedback._error("trial_video", error)
                print(f"ACT4 video recorder unavailable: {error}", file=sys.stderr, flush=True)
        client._pipeline_observer = compose_observer(
            client._pipeline_observer, feedback, on_observation=stop.observe_return_phase)
        stop.observer = feedback
        if session_reference:
            from inference_safe_return import capture_session_start_pose
            pose = capture_session_start_pose(client.robot, cancel_event=stop.stop_requested,
                                              audit=lambda event, **values: stop.event(event=event, **values))
            if pose is None:
                return
            stop.session_start_native_rad = pose
            saved = initial_zero
            stop.target_native_rad = {**pose, "gripper": saved["native_rad"]["gripper"]}
        else:
            saved = prepare_zero(client.robot, profile, identity, audit=stop.event)
            stop.target_native_rad = saved["native_rad"]
        reference_trace = ({"session_start_native_rad": stop.session_start_native_rad,
                            "return_target_native_rad": stop.target_native_rad} if session_reference else {})
        stop.event(event="session_prepared", profile=profile, zero_state=saved,
                   feedback_directory=str(feedback.directory), model_directory=report["contract_model_path"],
                   image_contract_sha256=report["image_contract_sha256"], **reference_trace)
        def seed(owned_client):
            import torch
            from sensecraft_224_stop import read_fresh_native_radians
            session_pose = stop.session_start_native_rad
            reference = []
            for joint in JOINTS:
                if stop.stop_requested.is_set() or stop.pause_requested.is_set():
                    return
                if session_pose is not None and joint != "gripper":
                    radians = session_pose[joint]
                else:
                    radians = read_fresh_native_radians(owned_client.robot, joint)
                    if session_pose is not None:
                        native_deg = math.degrees(radians)
                        owned_client._session_start_gripper_native_deg = native_deg
                        # The arm-only pre-roll never initializes the vendor gripper guard.
                        owned_client._native_rs_gripper_prev_action_deg = native_deg / identity["joint_directions"][joint]
                reference.append(math.degrees(radians) / identity["joint_directions"][joint])
            owned_client._runtime_trajectory.seed(torch.tensor(reference, dtype=torch.float32),
                                                  timestep=-1, now=time.monotonic())
            if (report.get("live_history_feedback_verified") and not stop.pause_requested.is_set()
                    and not stop.stop_requested.is_set()):
                owned_client.start_observation_history()
        if manual_keys is None:
            run_trials(client, config, stop, seed=seed)
        else:
            run_trials(client, config, stop, seed=seed, hotkeys_factory=lambda *_: manual_keys)
    except BaseException as error:
        stop.request_hold(f"{type(error).__name__}: {error}")
        caught = error
    finally:
        owned = client if client is not None else stop.client
        try:
            if owned is not None:
                owned.stop()
        finally:
            if manual_keys is not None:
                manual_keys.close()
            if feedback is not None:
                stop.event(event="session_finished", released=stop.released, error=stop.failure or None)
                video = getattr(feedback, "trial_video", None)
                feedback.close(timeout=8.0 if video is not None else 2.0)
                if video is not None:
                    if not video.close():
                        print("ACT4 video recorder is still finalizing.", file=sys.stderr, flush=True)
    if stop.released:
        return
    if caught is not None:
        raise caught


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "arm-capture", "cancel-capture"])
    parser.add_argument("--profile", type=Path,
                        default=Path(os.environ.get("ACT_INFERENCE_TRIALS_PROFILE", PROFILE_PATH)))
    parser.add_argument("--trials", type=int, choices=[10, 20])
    args = parser.parse_args()
    profile = load_profile(args.profile)
    if args.trials:
        profile["max_trials"] = args.trials
    if args.command == "arm-capture":
        atomic_json(args.profile, profile)
        atomic_json(profile["capture_request_path"], {"schema": "act-zero-capture-request-v1",
                                                     "created_unix_s": time.time()})
        print("One-shot saved-pose capture armed for next operator-started inference. No hardware connected. Expires in 1 hour.")
    elif args.command == "cancel-capture":
        Path(profile["capture_request_path"]).unlink(missing_ok=True)
        print("Capture request removed. No hardware connected.")
    else:
        print(json.dumps({**profile, "zero_state_exists": Path(profile["zero_state_path"]).exists(),
                          "capture_armed": valid_capture_request(profile),
                          "hardware_connected": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
