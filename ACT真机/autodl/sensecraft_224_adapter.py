"""Standalone 224 observation adapter around the installed SenseCraft RS client.

The default operation is metadata-only CheckOnly. Hardware execution requires
--launch and the original SenseCraft Python runtime. The adapter adds shared
224 images, wrist-only 20 Hz preview, reanchored linear action interpolation,
and same-owner native homing before torque disable on a normal Stop.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
from functools import partial
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time

sys.dont_write_bytecode = True
if os.name == "nt":
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from image_preprocessing import preprocess_rgb, validate_model_contract, contract_digest
from sensecraft_224_stop import StopController, read_fresh_native_radians


PACKAGE_ROOT = Path(__file__).resolve().parent
JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_yaw.pos", "wrist_roll.pos", "gripper.pos"]
JOINT_DIRECTIONS = dict(zip((name.removesuffix(".pos") for name in JOINT_NAMES),
                            [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 6.0]))
CAMERAS = {"observation.images.front", "observation.images.gripper"}
POLICY_FPS = 30
CAMERA_HARDWARE_FPS = 30
OBSERVATION_FPS = POLICY_FPS
PREVIEW_FPS_LIMITS = {"front": None, "gripper": 20}
SESSION_START_WARMUP_NS = 2_000_000_000
SESSION_START_HISTORY_DELTA_DEG = 0.5


def prepare_observation(observation):
    values = dict(observation.get_observation())
    for camera in ("front", "gripper"):
        values[camera] = preprocess_rgb(values[camera])
    return replace(observation, observation=values)


def prepare_raw_transport_observation(observation):
    import numpy as np

    values = observation.get_observation()
    for camera in ("front", "gripper"):
        image = values[camera]
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (480, 640, 3):
            raise ValueError(f"Raw transport requires a 640×480 RGB uint8 frame for {camera}.")
    return replace(observation)


def image_client_class(base_class, contract, stop_controller=None, *, history_feedback=False,
                       observation_transport_ab=False):
    """Extend the installed client without copying or changing its actuator."""
    if history_feedback and stop_controller is None:
        raise ValueError("Live history requires the operator-owned trial stop controller.")
    if observation_transport_ab and (stop_controller is None or history_feedback):
        raise ValueError("Camera transport A/B requires managed trials and the original seven-axis state.")
    startup_feedback = getattr(stop_controller, "profile", {}).get("session_start_reference", False)
    startup_only = startup_feedback and not history_feedback

    class SenseCraft224Client(base_class):
        def __init__(self, config):
            self._native_history = None
            self._history_snapshot = None
            self._history_warmup_deadline = None
            self._history_warmed = False
            self._history_missing_since = None
            self._session_start_verified = False
            self._session_policy_dispatched = False
            self._sampled_timestep = None
            if stop_controller is not None:
                stop_controller.attach(self)
            try:
                super().__init__(config)
                from wrist_preview import install_wrist_preview
                self._wrist_preview_rate = install_wrist_preview(self)
                features = deepcopy(self.policy_config.lerobot_features)
                image_keys = {key for key in features if key.startswith("observation.images.")}
                if image_keys != CAMERAS or features["observation.state"]["names"] != JOINT_NAMES:
                    raise ValueError("Live observation features differ from the ACT2 contract.")
                if history_feedback:
                    from live_native_history import HISTORY_FIELDS
                    features["observation.state"].update(
                        shape=(13,), names=[*JOINT_NAMES, *HISTORY_FIELDS])
                if self._observation_codec != "raw":
                    raise ValueError("The 224 contract requires raw uint8 observation transport.")
                for key in image_keys:
                    features[key].update(shape=(224, 224, 3), names=["height", "width", "channels"])
                self.policy_config.lerobot_features = features
                setattr(self.policy_config, "image_preprocessing_contract", contract)
                if stop_controller is not None:
                    from runtime_action_smoothing import install_linear_trajectory
                    processing = getattr(stop_controller, "profile", {}).get("action_processing", {})
                    speed_limits = None
                    if processing.get("limit_native_target_speed", False):
                        from backend.common.b601_rs_mit import resolve_rs_mit_config
                        mit_config = resolve_rs_mit_config()
                        speed_limits = [mit_config.arm_speed_deg_s] * 6 + [mit_config.gripper_speed_deg_s]
                    reversal_options = {
                        flag: True for flag in ("shoulder_elbow_reversal_guard", "stale_overlap_reversal_rate_cap")
                        if processing.get(flag)
                    }
                    self._runtime_trajectory = install_linear_trajectory(
                        self, temporal_ensemble_coeff=processing.get("temporal_ensemble_coeff"),
                        temporal_ensemble_scope=processing.get("temporal_ensemble_scope", "all"),
                        native_speed_limits=speed_limits,
                        arm_ema_time_constant_s=processing.get("arm_ema_time_constant_s"),
                        arm_transition_rate_deg_s=processing.get("arm_transition_rate_deg_s"),
                        **reversal_options)
                from camera_cache_preview import install_camera_cache_preview
                self._camera_cache_preview_worker, self._camera_cache_preview_receipt = (
                    install_camera_cache_preview(self))
            except BaseException as error:
                if stop_controller is not None:
                    stop_controller.request_hold(f"Client construction failed: {error}")
                self.stop()
                raise

        def start(self):
            started = super().start()
            if started:
                worker = getattr(self, "_camera_cache_preview_worker", None)
                if worker is not None:
                    worker.start()
            return started

        def start_observation_history(self):
            if not (history_feedback or startup_only):
                return None
            from live_native_history import NativeHistoryReader

            def end_on_read_failure(error):
                if startup_only and self._session_start_verified:
                    return
                stop_controller.event(event="history_read_failed", reason=str(error))
                stop_controller.request_home()

            reader = NativeHistoryReader(self.robot, on_failure=end_on_read_failure)
            self._native_history = reader
            self._history_warmed = False
            self._history_missing_since = None
            self._session_start_verified = False
            self._session_policy_dispatched = False
            self._sampled_timestep = None
            session_start = getattr(stop_controller, "session_start_native_rad", None)
            wait_ns = SESSION_START_WARMUP_NS if session_start is not None else 600_000_000
            self._history_warmup_deadline = time.monotonic_ns() + wait_ns
            reader.start()

        def stop_observation_history(self):
            reader = self._native_history
            if reader is not None:
                while not reader.stop(timeout_s=1.0):
                    stop_controller.request_hold("Original MotorBridge history read still in flight.")
                    stop_controller.log("waiting_for_observation_history_stop")
                self._native_history = None

        def _capture_smoothed_observation(self, task, verbose=False):
            if startup_only and self._session_start_verified:
                self.stop_observation_history()
                if stop_controller.stop_requested.is_set():
                    return None
                return super()._capture_smoothed_observation(task, verbose=verbose)
            if not (history_feedback or startup_only):
                return super()._capture_smoothed_observation(task, verbose=verbose)
            captured_at = time.monotonic_ns()
            try:
                if self._native_history is None:
                    raise RuntimeError("No native history reader owns the current trial.")
                self._history_snapshot = self._native_history.observation_at(captured_at)
            except ValueError as error:
                if startup_only or not self._history_warmed:
                    deadline = self._history_warmup_deadline
                else:
                    if self._history_missing_since is None:
                        self._history_missing_since = captured_at
                        stop_controller.event(event="history_capture_deferred", reason=str(error))
                    # A short scheduler stall can invalidate both the current read
                    # and, 200 ms later, its interpolation bracket. Wait only for
                    # a valid window, within the existing 0.5 s queue low-water period.
                    deadline = self._history_missing_since + 300_000_000
                if captured_at < deadline:
                    # Defer the entire capture and preserve the vendor's must-go flag.
                    return None
                stop_controller.event(event="history_observation_rejected", reason=str(error))
                stop_controller.request_home()
                return None
            except RuntimeError as error:
                stop_controller.event(event="history_observation_rejected", reason=str(error))
                stop_controller.request_home()
                return None
            session_start = getattr(stop_controller, "session_start_native_rad", None)
            if session_start is not None and not self._session_start_verified:
                history, _ = self._history_snapshot
                errors = {key: history[key] - math.degrees(session_start[key.removesuffix(".pos")])
                          for key in JOINT_NAMES[:6]}
                deltas = {key: history[f"{key}.delta_6_frames"] for key in JOINT_NAMES[:6]}
                if (max(map(abs, errors.values())) > stop_controller.profile["tolerance_deg"]
                        or max(map(abs, deltas.values())) > SESSION_START_HISTORY_DELTA_DEG):
                    self._history_snapshot = None
                    if captured_at >= self._history_warmup_deadline:
                        stop_controller.event(event="session_start_not_settled",
                                              arm_errors_deg=errors, arm_history_deltas_deg=deltas)
                        stop_controller.request_retry("session_start_not_settled")
                    return None
            if not self._history_warmed:
                self._history_warmed = True
                stop_controller.event(event="history_warmup_complete", timing=self._history_snapshot[1])
            self._history_missing_since = None
            try:
                return super()._capture_smoothed_observation(task, verbose=verbose)
            finally:
                self._history_snapshot = None

        def send_observation(self, observation):
            if not history_feedback and (not startup_only or self._session_start_verified):
                return super().send_observation(observation)
            if self._history_snapshot is None:
                stop_controller.event(event="history_observation_rejected",
                                      reason="No native history is associated with the current image capture.")
                stop_controller.request_home()
                return False
            history, receipt = self._history_snapshot
            if history_feedback:
                values = dict(observation.get_observation())
                values.update(history)
                prepared = replace(observation, observation=values)
            else:
                # Keep the proof paired with this queued frame. The RGB224
                # replacement in _send_observation_sync keeps dataclass fields
                # only, so this attribute never reaches the vendor uploader.
                prepared = replace(observation)
                prepared._session_start_history = history
            accepted = super().send_observation(prepared)
            observer = getattr(stop_controller, "observer", None)
            if accepted and observer is not None:
                observer.on_event({"event": ("session_start_observation_prepared" if startup_only
                                             else "history_observation_prepared"),
                    "trial_id": stop_controller.trial_id, "timestep": int(prepared.get_timestep()),
                    "observation_timestamp": float(prepared.get_timestamp()),
                    "episode_id": getattr(self.stub, "episode_id", None),
                    "state_native_motor_deg": history, "timing": receipt})
            return accepted

        def _send_observation_sync(self, observation):
            # The parent retains diagnostics, raw RPC, queuing and failure handling.
            startup_history = getattr(observation, "_session_start_history", None)
            if startup_only and not self._session_start_verified and startup_history is None:
                stop_controller.event(event="history_observation_rejected",
                                      reason="Startup frame is missing verified local feedback.")
                stop_controller.request_home()
                return False
            transport_mode = None
            if observation_transport_ab:
                trial_id = stop_controller.trial_id
                if not isinstance(trial_id, int) or trial_id < 1:
                    raise RuntimeError("Transport A/B requires an active trial ID.")
                transport_mode = "local224" if trial_id % 2 else "raw640"
                preparation_start = time.monotonic_ns()
            prepared = (prepare_raw_transport_observation(observation) if transport_mode == "raw640"
                        else prepare_observation(observation))
            observer = getattr(self, "_pipeline_observer", None)
            record_prepared = getattr(observer, "on_policy_observation", None)
            if transport_mode is None and callable(record_prepared):
                try:
                    record_prepared(prepared, episode_id=getattr(getattr(self, "stub", None), "episode_id", None),
                                    trial_id=getattr(stop_controller, "trial_id", None))
                except Exception as error:
                    logging.getLogger("act224.feedback").warning(
                        "Prepared-observation feedback failed: %s", error, exc_info=True)
            if transport_mode is not None:
                upload_start = time.monotonic_ns()
            sent = super()._send_observation_sync(prepared)
            if transport_mode is not None and stop_controller.observer is not None:
                upload_end = time.monotonic_ns()
                try:
                    stop_controller.observer.on_event({
                        "event": "trial_observation_transport", "trial_id": trial_id,
                        "episode_id": getattr(self.stub, "episode_id", None), "mode": transport_mode,
                        "timestep": int(prepared.get_timestep()), "capture_time_s": float(prepared.get_timestamp()),
                        "preparation_start_ns": preparation_start, "upload_start_ns": upload_start,
                        "upload_end_ns": upload_end, "sent": bool(sent),
                    })
                except Exception as error:
                    logging.getLogger("act224.feedback").warning("A/B transport feedback failed: %s", error)
            if (sent and getattr(stop_controller, "session_start_native_rad", None) is not None
                    and not self._session_start_verified):
                self._session_start_verified = True
                if startup_only:
                    reader = self._native_history
                    if reader is not None:
                        # This sender only signals the dedicated reader. The capture
                        # loop or managed teardown joins it before releasing CAN.
                        reader.stop(timeout_s=0)
                values = prepared.get_observation()
                history = startup_history if startup_only else values
                feedback = ({"arm_feedback_native_deg": {key: history[key] for key in JOINT_NAMES[:6]}}
                            if startup_only else {})
                stop_controller.event(event="session_start_observation_sent",
                                      timestep=int(prepared.get_timestep()),
                                      arm_native_deg={key: values[key] for key in JOINT_NAMES[:6]},
                                      arm_history_deltas_deg={key: history[f"{key}.delta_6_frames"]
                                                              for key in JOINT_NAMES[:6]}, **feedback)
            return sent

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
            observer = getattr(stop_controller, "observer", None)
            source_observation = None
            if incoming_actions and observer is not None:
                source_observation = {
                    "episode_id": getattr(getattr(self, "stub", None), "episode_id", None),
                    "timestep": int(incoming_actions[0].get_timestep()),
                    "timestamp": float(incoming_actions[0].get_timestamp()),
                }
            trajectory = getattr(self, "_runtime_trajectory", None)
            prepared = trajectory.prepare_chunk(incoming_actions) if trajectory is not None else incoming_actions
            result = super()._aggregate_action_queues(prepared, aggregate_fn)
            if observer is not None:
                observer.on_event({
                    "event": "cloud_predicted_chunk", "trial_id": stop_controller.trial_id,
                    "monotonic_s": time.monotonic(),
                    "source_observation": source_observation,
                    "coordinate_system": "dataset action coordinates, before interpolation and motor direction/scaling",
                    "actions": [{"timestep": int(action.get_timestep()),
                                 "server_timestamp": float(action.get_timestamp()),
                                 "values": action.get_action().detach().cpu().tolist()}
                                for action in incoming_actions],
                })
                trajectory = getattr(self, "_runtime_trajectory", None)
                if trajectory is not None:
                    observer.on_event({"event": "action_processing", "trial_id": stop_controller.trial_id,
                                       "monotonic_s": time.monotonic(),
                                       "action_output_hz": getattr(self, "_action_output_hz", None),
                                       "native_gripper_guard_deg_s": getattr(self, "_native_rs_gripper_speed_deg_s", None),
                                       **trajectory.execution_snapshot()})
            return result

        def _mark_transport_failure(self, error):
            filter_cancellation = getattr(stop_controller, "allow_transport_cancellation", None)
            if callable(filter_cancellation):
                if filter_cancellation(error):
                    stop_controller.log("transport_cancelled_between_trials", error=str(error))
                    return
                return super()._mark_transport_failure(error)
            if (stop_controller is not None and stop_controller.normal_stop_requested()
                    and not getattr(self, "_transport_failure", None)):
                stop_controller.log("transport_cancelled_after_normal_stop", error=str(error))
                return
            return super()._mark_transport_failure(error)

        def _record_trajectory_timestep(self, timestep, *, stale):
            if history_feedback or startup_only:
                self._sampled_timestep = timestep
            return super()._record_trajectory_timestep(timestep, stale=stale)

        def _dispatch_action(self, action):
            if stop_controller is None:
                return super()._dispatch_action(action)
            session_start = getattr(stop_controller, "session_start_native_rad", None)
            startup_hold = (session_start is not None and not self._session_policy_dispatched
                            and self._sampled_timestep == -1)
            send = super()._dispatch_action
            if startup_hold:
                # The vendor still emits its seed between reads and the first policy step.
                # Maintain the arm setpoint and leave the already closed gripper at rest.
                from inference_safe_return import ARM_JOINTS, _operator_pose_limits, _send_native_pose

                def send(_action):
                    with _operator_pose_limits(self.robot, session_start, enabled=True):
                        goals = _send_native_pose(self.robot, session_start, joints=ARM_JOINTS)
                    return {f"{joint}.pos": value for joint, value in goals.items()}

                action = {key: action[key] for key in JOINT_NAMES[:6]}
            result = stop_controller.dispatch(send, action)
            trajectory = getattr(self, "_runtime_trajectory", None)
            if result is not None and trajectory is not None:
                import torch
                if (session_start is not None and self._sampled_timestep is not None
                        and self._sampled_timestep >= 0):
                    self._session_policy_dispatched = True
                native_reference = ({**result, "gripper.pos": self._session_start_gripper_native_deg}
                                    if startup_hold else result)
                reference = torch.tensor([
                    float(native_reference[key]) / JOINT_DIRECTIONS[key.removesuffix(".pos")]
                    for key in JOINT_NAMES], dtype=torch.float32)
                trajectory.mark_dispatched(reference, now=time.monotonic())
            return result

        def stop(self):
            self.stop_observation_history()
            worker = getattr(self, "_camera_cache_preview_worker", None)
            if worker is not None:
                worker.stop(timeout_s=1.0)
            if stop_controller is None:
                return super().stop()
            return stop_controller.finish(self)

    return SenseCraft224Client


def preflight(args, *, runtime_root, calibration_dir, trial_profile_path=None):
    """Inspect files and package metadata without importing a robot client."""
    backend_root = args.backend_root.resolve()
    if not (backend_root / "backend/run/lerobot_async_robot_client.py").is_file():
        raise ValueError("backend-root must contain the installed backend package.")
    if Path(sys.executable).resolve() != (runtime_root / "python.exe").resolve():
        raise ValueError("Use the pinned SenseCraft Python runtime through START_SENSECRAFT_224.ps1.")
    expected = {"lerobot": "0.6.0", "lerobot_robot_seeed_b601": "1.0.3",
                "motorbridge": "0.5.1", "pillow": "12.3.0"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise ValueError(f"Runtime packages differ from the reviewed setup: {versions}; expected {expected}")
    contract = validate_model_contract(args.contract_model_path)
    config = json.loads((args.contract_model_path / "config.json").read_text(encoding="utf-8-sig"))
    history = getattr(args, "history_observation", False)
    state_shape = [13] if history else [7]
    image_keys = {key for key, feature in config["input_features"].items() if feature["type"] == "VISUAL"}
    if (image_keys != CAMERAS or config["input_features"]["observation.state"]["shape"] != state_shape
            or config["output_features"]["action"]["shape"] != [7] or config["chunk_size"] != 100):
        raise ValueError("Local metadata must describe the selected ACT dual-camera, state and action contract.")
    host, separator, port = args.server_address.rpartition(":")
    if (not separator or host not in {"127.0.0.1", "localhost", "[::1]"}
            or not port.isdigit() or not 1 <= int(port) <= 65535):
        raise ValueError("server-address must be the local SSH-forwarded loopback port.")
    if not args.remote_model_path.startswith("/"):
        raise ValueError("remote-model-path must be the absolute serving path printed by serve.sh.")
    if not 1 <= args.actions_per_chunk <= 100 or not 0 <= args.chunk_size_threshold <= 1:
        raise ValueError("Use 1..100 actions per chunk and a threshold in 0..1.")
    recording = json.loads(args.recording_state.read_text(encoding="utf-8-sig"))
    setup = recording["setup"]
    if setup["robot_type"] != "seeed_b601_rs_follower" or setup["camera_wrist_role"] != "gripper":
        raise ValueError("Recording setup does not describe the reviewed physical B601RS pair.")
    if not (calibration_dir / f"{setup['robot_id']}.json").is_file():
        raise ValueError("The existing follower calibration file is missing.")
    gravity_root = backend_root / "rebotarm-control-py"
    if not (gravity_root / "urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf").is_file():
        raise ValueError("The installed gravity URDF is missing.")
    cameras = {}
    for role, prefix in (("front", "camera_front"), ("gripper", "camera_wrist")):
        source = str(setup[f"{prefix}_index"])
        if not source.startswith("intelrealsense://"):
            raise ValueError("This reviewed setup requires the two recorded RealSense sources.")
        dimensions = [int(setup[f"{prefix}_{key}"]) for key in ("width", "height", "fps")]
        if dimensions[:2] != [640, 480] or dimensions[2] not in {20, 30}:
            raise ValueError("Camera dimensions or FPS differ from the reviewed collection.")
        cameras[role] = {"serial_number_or_name": source.split("://", 1)[1],
                         "width": 640, "height": 480, "fps": CAMERA_HARDWARE_FPS,
                         "use_rgb": True, "use_depth": False}
    report = {
        "status": "check_only_passed", "hardware_connected": False,
        "robot_client_constructed": False, "local_model_forward_calls": 0,
        "runtime_python": sys.executable, "versions": versions,
        "backend_root": str(backend_root), "recording_state": str(args.recording_state),
        "server_address": args.server_address, "remote_model_path": args.remote_model_path,
        "contract_model_path": str(args.contract_model_path),
        "image_contract_sha256": contract_digest(contract), "cameras": cameras,
        "robot_type": setup["robot_type"], "robot_id": setup["robot_id"],
        "robot_port": setup["robot_port"], "joint_directions_in_robot_send_action": JOINT_DIRECTIONS,
        "gravity_compensation": True, "calibration_dir": str(calibration_dir),
        "task": args.task if args.task is not None else recording["task_text"],
        "actions_per_chunk": args.actions_per_chunk, "fps": POLICY_FPS,
        "observation_fps_limit": OBSERVATION_FPS,
        "preview_fps_limits": PREVIEW_FPS_LIMITS,
        "preview_fps_scope": "Wrist D405 display only; front preview keeps the original cadence. Native RGB profiles and model observation/action timebase remain 30 Hz.",
        "preview_pending_policy": "One pending packet, latest frame per camera role; original front sender interval retained, wrist target 20 Hz.",
        "action_smoothing_enabled": True, "action_output_hz": 60,
        "atomic_action_chunks": False,
        "home_verification_tolerance_deg": 1.0, "home_verification_stable_seconds": 1.0,
        "home_joint_scope": "Six arm joints to native zero, original gripper safe_zero semantics",
        "camera_hardware_fps": CAMERA_HARDWARE_FPS,
        "recording_camera_fps": {prefix: setup[f"{prefix}_fps"]
                                 for prefix in ("camera_front", "camera_wrist")},
        "chunk_size_threshold": args.chunk_size_threshold, "aggregate_fn_name": args.aggregate_fn_name,
        "stop_behavior": "stop/home_stop quiesces model writers, then native safe_zero, fresh six-joint feedback verification, torque disable and CAN close. Fault/EOF requests persistent damped hold until explicit operator confirm_release.",
        "limits": ["Physical availability and exclusive CAN/camera ownership are not checked.",
                   "Remote checkpoint weights and RPC connectivity require separate replay/hash verification.",
                   "SenseCraft's existing native gripper speed limiter remains active."],
        "source_sha256": {},
    }
    for relative in ("backend/run/lerobot_async_robot_client.py",
                     "backend/run/async_robot_client_pipeline.py", "backend/run/async_robot_client_actuator.py",
                     "backend/common/b601_rs_shutdown.py", "backend/common/lerobot_types.py"):
        report["source_sha256"][relative] = hashlib.sha256((backend_root / relative).read_bytes()).hexdigest()
    from inference_trials import load_profile, valid_capture_request
    trial_profile = load_profile(trial_profile_path) if trial_profile_path is not None else load_profile()
    report["action_output_hz"] = trial_profile.get("action_processing", {}).get("action_output_hz", 60)
    if history:
        report["state_input_width"] = state_shape[0]
        report["state_history_frames"] = 6
        report["state_history_seconds"] = 0.2
    report["inference_trials"] = {**trial_profile,
        "zero_state_exists": Path(trial_profile["zero_state_path"]).is_file(),
        "explicit_capture_armed": valid_capture_request(trial_profile),
        "cloud_episode_guard_verified": False,
        "physical_homing_verified": False}
    if trial_profile["enabled"]:
        report["home_verification_tolerance_deg"] = trial_profile["tolerance_deg"]
        report["home_verification_stable_seconds"] = 0.5
        report["home_joint_scope"] = "All seven motor-native angles to operator-saved zero_state"
        report["operator_saved_pose"] = trial_profile.get("operator_saved_pose", False)
        report["home_limit_scope"] = (
            "Capture/return/rest use the envelope of configured limits, fresh current and operator-saved target. "
            "Original policy limits are restored before policy execution."
            if trial_profile.get("operator_saved_pose", False)
            else "Configured command limits with at most0.5 degree native feedback boundary allowance.")
        report["stop_behavior"] = (
            "Ctrl+C interrupts the current trial, returns to operator-saved zero_state, verifies all seven native motor angles within 2 degrees, "
            + ("holds/rests at least5 seconds and consumes one latched Ctrl+C continuation, or waits for Ctrl+C after a timeout. "
               if trial_profile.get("manual_advance", False)
               else "holds/rests then starts a fresh isolated cloud episode. ")
            + "Ctrl+V or UI Stop homes to the same saved pose then disables. "
            "Missing zero_state/one-shot capture or cloud act-episode-reset=v1 capability blocks before hardware connection. Faults retain operator-confirmed hold.")
        if trial_profile.get("operator_controlled", False):
            report["stop_behavior"] = ("Wait for the first Ctrl+C before connecting hardware. Ctrl+C during a trial "
                "stops the trial and returns to the verified saved pose. The original owner holds position with "
                "torque enabled until a new Ctrl+C starts a fresh cloud episode. Ctrl+V returns to the saved "
                "pose, verifies all seven axes and then disables. No trial timer or trial-count limit.")
            if trial_profile["max_trials"] == 1:
                report["stop_behavior"] = ("Wait for the first Ctrl+C before connecting hardware for one trial. "
                    "Ctrl+C during that trial or Ctrl+V ends it, returns to the saved pose, "
                    "verifies its seven native motor angles, then disables.")
        if trial_profile.get("park_gripper_after_return", False):
            report["home_joint_scope"] = (
                "Six arm motor-native angles to the operator-saved pose, then one native J7 closure.")
            report["stop_behavior"] = (
                "Wait for the first Ctrl+C before connecting hardware. Ctrl+C during a trial "
                "stops it, returns the arm to the saved pose, and closes J7 once. "
                "Wait for a new Ctrl+C before the next trial. Ctrl+V verifies the returned "
                "arm and closed gripper before disabling. No trial timer or trial-count limit.")
        if trial_profile.get("session_start_reference", False):
            report["home_joint_scope"] = (
                "Six arm angles from the operator's stable first-trial pose, then one native J7 closure.")
            report["stop_behavior"] = (
                "The first Ctrl+C connects hardware and reads a stable session pose. "
                "Ctrl+C during a trial returns to that pose, closes J7 once, and waits for a new Ctrl+C. "
                "Each first cloud observation needs fresh six-axis feedback and stable 0.20-second movement. "
                "Ctrl+V verifies the returned arm and closed gripper before disabling.")
    return report, contract, setup


def run_client(args, report, contract, setup):
    history_feedback = getattr(args, "history_observation", False)
    observation_transport_ab = getattr(args, "observation_transport_ab", False)
    # All installed-package changes below live only in this explicitly launched process.
    from inference_trials import load_profile, run_managed_session
    trial_profile = load_profile()
    if observation_transport_ab and (not trial_profile.get("enabled")
                                     or not trial_profile.get("operator_controlled")):
        raise ValueError("Camera transport A/B requires operator-controlled manual trials.")
    if ((history_feedback or trial_profile.get("session_start_reference", False))
            and (not trial_profile.get("enabled")
                 or not trial_profile.get("operator_controlled")
                 or trial_profile.get("max_trials") is not None
                 or not trial_profile.get("require_continue_after_return")
                 or not report.get("live_history_feedback_verified"))):
        raise RuntimeError("Native joint feedback requires a verified read path and operator-controlled trials.")
    processing = trial_profile.get("action_processing", {})
    if "native_gripper_speed_deg_s" in processing:
        # Reuse the vendor's supported per-process setting before constructing
        # its client. This selects the existing guard, including its validation.
        os.environ["WEBUI_RS_NATIVE_GRIPPER_SPEED_DEG_S"] = str(float(processing["native_gripper_speed_deg_s"]))
    os.environ["WEBUI_BACKEND_REBOTARM_CONTROL_PY_ROOT"] = str(args.backend_root / "rebotarm-control-py")
    os.environ["WEBUI_RS_ROLLOUT_AUTO_HOME"] = "0"
    os.environ["WEBUI_CLOUD_OBSERVATION_TRANSFER_OPTIMIZATION"] = "0"
    os.environ["WEBUI_CLOUD_OBSERVATION_CODEC"] = "raw"
    os.environ["WEBUI_CLOUD_RPC_WATCHDOG"] = "1"
    os.environ["WEBUI_ACTION_SMOOTHING"] = "1"
    os.environ["WEBUI_ACTION_OUTPUT_HZ"] = str(processing.get("action_output_hz", 60))
    os.environ["WEBUI_RS_ATOMIC_ACTION_CHUNKS"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.backend_root))

    from backend.run import lerobot_async_robot_client as vendor
    from backend.common.b601_rs_shutdown import install_seeed_b601_persistent_hold_disconnect
    from backend.common.b601_rs_shutdown import perform_b601_rs_persistent_damped_hold
    from backend.common.lerobot_types import make_follower_config, suppress_device_calibration
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras = {role: RealSenseCameraConfig(**values) for role, values in report["cameras"].items()}
    robot_config = make_follower_config(
        follower_type=setup["robot_type"], follower_id=setup["robot_id"], follower_port=setup["robot_port"],
        calibration_dir=Path(report["calibration_dir"]), cameras=cameras,
    )
    robot_config.gravity_compensation = True
    if robot_config.joint_directions != JOINT_DIRECTIONS:
        raise ValueError("Robot send_action directions differ from the actual collection config.")
    config = vendor.RobotClientConfig(
        robot=robot_config, policy_type="act", pretrained_name_or_path=args.remote_model_path,
        server_address=args.server_address, policy_device="cuda", client_device="cpu",
        actions_per_chunk=report["actions_per_chunk"], task=report["task"], fps=POLICY_FPS,
        chunk_size_threshold=report["chunk_size_threshold"], aggregate_fn_name=report["aggregate_fn_name"],
    )
    vendor._install_robot_lifecycle_patches(robot_config)
    install_seeed_b601_persistent_hold_disconnect()
    vendor.reset_b601_rs_damped_hold()
    vendor.perform_b601_rs_persistent_damped_hold = perform_b601_rs_persistent_damped_hold
    # A failed constructor is finalized by our same-owner stop controller.
    # Suppress the vendor helper's automatic reconnect-after-cleanup retries.
    def connect_once_without_calibration(device):
        with suppress_device_calibration(device):
            device.connect(calibrate=False)
    vendor.connect_device_without_calibration = connect_once_without_calibration
    if trial_profile["enabled"]:
        factory = partial(image_client_class, history_feedback=history_feedback,
                          observation_transport_ab=observation_transport_ab)
        run_managed_session(vendor, config, contract, trial_profile, report, factory)
    else:
        run_session(vendor, config, contract)


def run_session(vendor, config, contract):
    """Keep the operator channel available throughout connect and disconnect."""
    stop = StopController(vendor, emit=lambda message: print(message, file=sys.stderr, flush=True))
    client_class = image_client_class(vendor.InternalRobotClient, contract, stop)
    client = None
    receiver = None
    holder = vendor.TaskTextHolder(config.task)
    caught_error = None
    # Establish the operator's release channel before any connect/cleanup path.
    vendor.start_stdin_control_listener(
        {"stop": stop.request_home, "home_stop": stop.request_home,
         "damped_hold": stop.request_hold, "confirm_release": stop.confirm_release},
        prefix_handlers={"task": holder.set}, on_eof=stop.request_hold,
        thread_name="act224-sensecraft-stdin",
    )
    try:
        if stop.stop_requested.is_set():
            return
        client = client_class(config)
        client._task_text_holder = holder
        if stop.stop_requested.is_set():
            return
        print("[act224] stop/home_stop homes and verifies before release. Fault/EOF holds for operator confirmation.",
              file=sys.stderr, flush=True)
        if not client.start():
            if not stop.stop_requested.is_set():
                raise RuntimeError("The cloud policy handshake failed.")
            return
        if stop.stop_requested.is_set():
            return
        # Seed the trajectory from active native reads before either worker can
        # produce or execute a policy action. This inverse map records an anchor.
        import torch
        reference = []
        for key in JOINT_NAMES:
            if stop.stop_requested.is_set():
                return
            joint = key.removesuffix(".pos")
            radians = read_fresh_native_radians(client.robot, joint)
            degrees = math.degrees(float(radians))
            if not math.isfinite(degrees):
                raise RuntimeError(f"Non-finite native trajectory anchor for {joint}.")
            reference.append(degrees / JOINT_DIRECTIONS[joint])
        if stop.stop_requested.is_set():
            return
        client._runtime_trajectory.seed(torch.tensor(reference, dtype=torch.float32),
                                        timestep=-1, now=time.monotonic())
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        client.control_loop(task=config.task)
    except BaseException as error:
        stop.request_hold(f"{type(error).__name__}: {error}")
        caught_error = error
    finally:
        owned_client = client if client is not None else stop.client
        if owned_client is not None:
            owned_client.stop()
        if receiver is not None:
            receiver.join(timeout=2.0)
    if stop.released:
        if stop.failure:
            stop.log("session_fault_released", error=stop.failure)
        return
    if caught_error is not None:
        raise caught_error
    failure = stop.failure or str(getattr(client, "_transport_failure", "") or "").strip()
    if failure:
        raise RuntimeError(f"Cloud transport failure: {failure}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--recording-state", type=Path, required=True)
    parser.add_argument("--contract-model-path", type=Path, required=True)
    parser.add_argument("--remote-model-path", required=True)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--task")
    parser.add_argument("--actions-per-chunk", type=int, default=100)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument("--aggregate-fn-name", default="weighted_average",
                        choices=["weighted_average", "latest_only", "average", "conservative"])
    parser.add_argument("--check-report", type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    try:
        report, contract, setup = preflight(
            args, runtime_root=args.runtime_root, calibration_dir=args.calibration_dir,
        )
    except Exception as error:
        report = {"status": "check_only_blocked", "error": f"{type(error).__name__}: {error}",
                  "hardware_connected": False, "robot_client_constructed": False}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    if args.check_report:
        args.check_report.parent.mkdir(parents=True, exist_ok=True)
        args.check_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.launch:
        run_client(args, report, contract, setup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
