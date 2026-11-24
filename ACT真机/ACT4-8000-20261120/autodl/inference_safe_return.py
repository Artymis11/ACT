"""Return an existing B601-RS owner's MIT controller to a saved native pose.

The caller must stop and join every policy writer before calling run_return.
This module keeps the connected robot and never enables or disables a motor.
Fresh READ_PARAMETER replies are required for position, mode and fault state.
Register definitions and read-cache invalidation follow MotorBridge v0.5.1.
"""

from dataclasses import replace
from contextlib import contextmanager, nullcontext
import math
import threading
import time

from sensecraft_224_stop import read_fresh_native_radians


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_yaw", "wrist_roll", "gripper")
ARM_JOINTS = JOINTS[:-1]
STAGES = (("shoulder_pan", "wrist_flex", "wrist_yaw", "wrist_roll"),
          ("shoulder_lift", "elbow_flex", "gripper"))
CONTROLLER_ATTRIBUTE = "_webui_rs_mit_controller"
POSE_BOUNDARY_TOLERANCE_DEG = 0.5


class ReturnFailed(RuntimeError):
    """The caller must enter its existing hold lifecycle and retain ownership."""


def _controller(robot):
    controller = getattr(robot, CONTROLLER_ATTRIBUTE, None)
    if getattr(robot, "bus", None) is None:
        raise ReturnFailed("The existing connected RS owner is unavailable.")
    if controller is not None:
        controller.raise_if_failed()
        if controller._stop.is_set():
            raise ReturnFailed("The existing RS MIT controller has stopped.")
    elif getattr(robot, "motor_type", None) != "rs":
        raise ReturnFailed("Saved-pose return requires the existing RS native driver.")
    return controller


def _emit(audit, event, **values):
    if audit is not None:
        audit(event, **values)


def _check_cancel(robot, cancelled, deadline, clock):
    if cancelled() or getattr(robot, "_emergency_disable_requested", False):
        raise ReturnFailed("Saved-pose return was cancelled or emergency-disabled.")
    if clock() >= deadline:
        raise ReturnFailed("Saved-pose return exceeded its deadline.")


def _freeze_reference(controller):
    """Stop advancing toward a previously submitted policy target."""
    if controller is None:
        return
    with controller._state_lock:
        if set(controller.command_positions) != set(JOINTS):
            raise ReturnFailed("The MIT command reference is incomplete.")
        reference = dict(controller.command_positions)
        if not all(math.isfinite(float(value)) for value in reference.values()):
            raise ReturnFailed("The MIT command reference is non-finite.")
        controller.target_positions = reference


def _read_pose(robot, controller, *, cancelled, deadline, clock, joints=JOINTS):
    positions = {}
    for index, joint in enumerate(JOINTS, 1):
        if joint not in joints:
            continue
        _check_cancel(robot, cancelled, deadline, clock)
        if controller is not None:
            controller.raise_if_failed()
        motor = robot.motors[joint]
        motor_id, host_id = robot.config.motor_can_ids[joint]
        if int(motor_id) != index:
            raise ReturnFailed("Saved-pose return requires the recorded CAN 1..7 joint mapping.")
        # Release the shared CAN lock between each request so the existing
        # 200 Hz MIT owner can continue sending while register replies arrive.
        with controller.can_lock if controller is not None else nullcontext():
            fault = int(motor.robstride_get_param_u32(0x3022, timeout_ms=150))
        if fault:
            raise ReturnFailed(f"{joint} reports native faultSta 0x{fault:08x}.")
        with controller.can_lock if controller is not None else nullcontext():
            mode = int(motor.robstride_get_param_i8(0x7005, timeout_ms=150))
        if mode != 0:
            raise ReturnFailed(f"{joint} is in native run_mode {mode}, expected MIT 0.")
        with controller.can_lock if controller is not None else nullcontext():
            position = float(motor.robstride_get_param_f32_host_id(
                0x7019, int(host_id), timeout_ms=250))
        if not math.isfinite(position):
            raise ReturnFailed(f"{joint} returned a non-finite native position.")
        positions[joint] = position
        _check_cancel(robot, cancelled, deadline, clock)
    if controller is not None:
        controller.raise_if_failed()
    return positions


def _configured_command_pose(robot, pose):
    """Project return references into the existing native-degree goal limits."""
    command_pose = {}
    for joint in JOINTS:
        lower, upper = map(float, robot.config.joint_limits[joint])
        degrees = math.degrees(pose[joint])
        command_pose[joint] = math.radians(max(lower, min(upper, degrees)))
    return command_pose


def _send_native_pose(robot, pose, *, joints=JOINTS):
    command_pose = _configured_command_pose(robot, pose)
    action = {f"{joint}.pos": math.degrees(command_pose[joint]) /
              float(robot.config.joint_directions[joint]) for joint in joints}
    cap = getattr(robot.config, "max_relative_target", None)
    if joints != JOINTS and isinstance(cap, dict):
        # LeRobot requires the cap keys to match the subset being sent.
        robot.config.max_relative_target = {joint: cap[joint] for joint in joints}
        try:
            performed = robot.send_action(action)
        finally:
            robot.config.max_relative_target = cap
    else:
        performed = robot.send_action(action)
    if not isinstance(performed, dict) or any(
            f"{joint}.pos" not in performed or
            not math.isfinite(float(performed[f"{joint}.pos"])) for joint in joints):
        raise ReturnFailed("Native send_action did not return all requested finite motor goals.")
    return {joint: float(performed[f"{joint}.pos"]) for joint in joints}


def _damp_gripper(robot, position):
    from backend.common.b601_rs_shutdown import _motor_shutdown_kd, resolve_rs_shutdown_damping_kd

    kd = _motor_shutdown_kd("gripper", base_kd=resolve_rs_shutdown_damping_kd())
    if not math.isfinite(kd) or kd <= 0:
        raise ReturnFailed("Native gripper damping is unavailable.")
    robot.motors["gripper"].send_mit(position, 0.0, 0.0, kd, 0.0)


def _native_gripper_reference(robot, target, position):
    cap = getattr(robot.config, "max_relative_target", None)
    if cap is None:
        return target
    from lerobot.robots.utils import ensure_safe_goal_position

    if isinstance(cap, dict):
        cap = {"gripper": cap["gripper"]}
    safe = ensure_safe_goal_position({"gripper": (math.degrees(target), math.degrees(position))}, cap)
    return math.radians(safe["gripper"])


def _close_gripper(robot, pose, current, *, goal, lower, parked, cancelled, deadline, clock,
                   sleep, speed_deg_s, tolerance_deg, audit):
    """Close toward the configured goal and leave one zero-drive damping command."""
    joint = "gripper"
    motor = robot.motors[joint]
    position = current[joint]
    close_min = lower - math.radians(tolerance_deg)
    close_max = goal + math.radians(tolerance_deg)
    step = math.radians(speed_deg_s * 0.02)
    reference = position
    closing = False
    next_health_check = clock()
    settled_reference_deadline = deadline
    try:
        while position > close_max and not parked:
            _check_cancel(robot, cancelled, deadline, clock)
            started = clock()
            if started >= settled_reference_deadline:
                raise ReturnFailed("Gripper did not reach the configured closed position.")
            # Preserve the installed send path for arm temperature checks and holding.
            _send_native_pose(robot, pose, joints=ARM_JOINTS)
            if reference > goal:
                reference = max(goal, reference - step)
                if reference == goal:
                    settled_reference_deadline = min(deadline, started + 2.0)
            native_reference = _native_gripper_reference(robot, reference, position)
            torque = robot.mit_output_torque_limit(motor, native_reference)
            if torque is None or not math.isfinite(torque):
                raise ReturnFailed("Native gripper return torque is unavailable.")
            closing = True
            # Negative torque closes this native axis. Do not drive it back open.
            motor.send_mit(0.0, 0.0, 0.0, 1.5, min(0.0, torque))
            position = read_fresh_native_radians(robot, joint)
            if position <= close_max:
                break
            if clock() >= next_health_check:
                position = _read_pose(robot, None, cancelled=cancelled, deadline=deadline,
                                      clock=clock, joints=(joint,))[joint]
                next_health_check = clock() + 0.5
            if position > close_max:
                sleep(max(0.0, started + 0.02 - clock()))
    finally:
        if closing and not getattr(robot, "_emergency_disable_requested", False):
            _damp_gripper(robot, position)

    current = _read_pose(robot, None, cancelled=cancelled, deadline=deadline, clock=clock)
    position = current[joint]
    if not close_min <= position <= close_max:
        raise ReturnFailed(f"Gripper did not verify closed after return: {math.degrees(position):.2f}°.")
    arm_errors = [abs(math.degrees(current[name] - pose[name])) for name in ARM_JOINTS]
    if any(error > tolerance_deg for error in arm_errors):
        raise ReturnFailed("Arm drifted outside saved-pose tolerance during gripper return.")
    _emit(audit, "saved_pose_gripper_parked", closed_this_return=closing,
          already_parked=parked, native_position_deg=math.degrees(position),
          command_target_deg=math.degrees(goal),
          verified_interval_deg=[math.degrees(close_min), math.degrees(close_max)])
    return current


def _native_return(robot, target, current, *, cancelled, deadline, clock, sleep,
                   speed_deg_s, tolerance_deg, stable_s, release_after, audit, paced=False,
                   gripper_goal=None, gripper_parked=False):
    """Use the installed native send_action, preserving its gains and torque rules."""
    stage_target = dict(current)
    total_sweeps = 0
    joints = ARM_JOINTS if gripper_goal is not None else JOINTS
    if gripper_goal is not None and not gripper_parked:
        _check_cancel(robot, cancelled, deadline, clock)
        _send_native_pose(robot, current, joints=joints)
        _damp_gripper(robot, current["gripper"])
    already_home = all(abs(math.degrees(current[j] - target[j])) <= tolerance_deg for j in joints)
    for number, stage in enumerate(STAGES, 1):
        # Carry forward the last commanded reference at a stage boundary.
        # The measured pose may lag it by the accepted arrival tolerance.
        origin = dict(stage_target)
        for joint in stage:
            if joint in joints:
                stage_target[joint] = target[joint]
        largest_delta = max(abs(math.degrees(stage_target[joint] - origin[joint]))
                            for joint in JOINTS)
        # smoothstep's maximum derivative is 1.5. Fixed frames and sleeps
        # avoid a delayed iteration catching up with a large reference jump.
        # The installed safe_zero uses a linear shared fraction at 0.5°/frame.
        # At 25°/s and 20 ms this is exactly its normal per-frame advance.
        scale = 1.0 if paced else 1.5
        frames = max(1, math.ceil(scale * largest_delta / (speed_deg_s * 0.02)))
        next_health_check = clock()
        for frame in range(1, frames + 1):
            _check_cancel(robot, cancelled, deadline, clock)
            started = clock()
            ratio = frame / frames
            if not paced:
                ratio = ratio * ratio * (3 - 2 * ratio)
            reference = {joint: origin[joint] +
                         (stage_target[joint] - origin[joint]) * ratio for joint in JOINTS}
            _send_native_pose(robot, reference, joints=joints)
            if clock() >= next_health_check:
                _read_pose(robot, None, cancelled=cancelled, deadline=deadline, clock=clock)
                total_sweeps += 1
                next_health_check = clock() + 0.5
            # Charge synchronous send and health checks to this cycle. Late
            # checks still advance only one bounded frame on the next send.
            sleep(max(0.0, started + .02 - clock()) if paced else .02)
        stable_since = None
        stage_sweeps = 0
        command_target = _configured_command_pose(robot, stage_target)
        while True:
            _check_cancel(robot, cancelled, deadline, clock)
            command = _send_native_pose(robot, stage_target, joints=joints)
            current = _read_pose(robot, None, cancelled=cancelled, deadline=deadline, clock=clock)
            stage_sweeps += 1
            total_sweeps += 1
            errors = {joint: math.degrees(current[joint] - stage_target[joint]) for joint in joints}
            now = clock()
            command_complete = all(abs(command[joint] - math.degrees(command_target[joint]))
                                   <= 0.05 for joint in joints)
            if command_complete and all(abs(error) <= tolerance_deg for error in errors.values()):
                stable_since = now if stable_since is None else stable_since
                if (paced and already_home and number == 1) or (stage_sweeps >= 3 and now - stable_since >= stable_s):
                    break
            else:
                stable_since = None
            sleep(0.02)
        _emit(audit, "saved_pose_return_stage_verified", stage=number,
              native_positions_rad=current, errors_deg=errors, fresh_sweeps=stage_sweeps)
    command_target = _configured_command_pose(robot, target)
    receipt_fields = {}
    if gripper_goal is not None:
        command_target["gripper"] = gripper_goal
        current = _close_gripper(
            robot, target, current, goal=gripper_goal, lower=min(gripper_goal, target["gripper"]),
            parked=gripper_parked, cancelled=cancelled, deadline=deadline, clock=clock,
            sleep=sleep, speed_deg_s=speed_deg_s, tolerance_deg=tolerance_deg, audit=audit)
        total_sweeps += 1
        errors = {joint: math.degrees(current[joint] - target[joint]) for joint in JOINTS}
        receipt_fields["gripper_parked"] = True
    return {"native_positions_rad": current, "native_target_rad": target,
            "native_command_target_rad": command_target,
            "boundary_feedback_tolerance_deg": POSE_BOUNDARY_TOLERANCE_DEG,
            "errors_deg": errors, "tolerance_deg": tolerance_deg,
            "stable_seconds": clock() - stable_since, "fresh_sweeps": total_sweeps,
            "release_after": bool(release_after),
            "feedback_source": "fresh READ_PARAMETER 0x7019 / 0x7005 / 0x3022",
            "control_path": "existing native RS send_action", **receipt_fields}


def _validate_pose(robot, target, *, operator_saved_pose=False):
    if set(target) != set(JOINTS):
        raise ReturnFailed("A saved native pose must contain exactly all seven named joints.")
    result = {}
    for joint in JOINTS:
        radians = float(target[joint])
        direction = float(robot.config.joint_directions[joint])
        if not math.isfinite(radians) or not math.isfinite(direction) or direction == 0:
            raise ReturnFailed(f"Invalid native position or action scale for {joint}.")
        lower, upper = map(float, robot.config.joint_limits[joint])
        degrees = math.degrees(radians)
        if (not math.isfinite(lower + upper) or lower > upper
                or (not operator_saved_pose and not lower - POSE_BOUNDARY_TOLERANCE_DEG
                    <= degrees <= upper + POSE_BOUNDARY_TOLERANCE_DEG)):
            raise ReturnFailed(
                f"{joint} native pose is outside configured limits. "
                f"native_deg={degrees:.9f}, native_rad={radians:.12f}, "
                f"configured_limits_deg=[{lower:.9f}, {upper:.9f}], "
                f"boundary_feedback_tolerance_deg={POSE_BOUNDARY_TOLERANCE_DEG}")
        result[joint] = radians
    return result


@contextmanager
def _operator_pose_limits(robot, *poses, enabled=False, audit=None):
    """Limit a quiesced owner's saved-pose operation to its occupied envelope."""
    if type(enabled) is not bool:
        raise ValueError("operator_saved_pose must be a boolean.")
    if not enabled:
        yield
        return
    original = robot.config.joint_limits
    scoped = {joint: tuple(map(float, original[joint])) for joint in JOINTS}
    for pose in poses:
        validated = _validate_pose(robot, pose, operator_saved_pose=True)
        for joint in JOINTS:
            lower, upper = scoped[joint]
            degrees = math.degrees(validated[joint])
            scoped[joint] = (min(lower, degrees), max(upper, degrees))
    robot.config.joint_limits = scoped
    try:
        _emit(audit, "operator_saved_pose_limits", original_limits_deg=original,
              scoped_limits_deg=scoped, native_reference_poses_rad=poses)
        yield
    finally:
        robot.config.joint_limits = original
        _emit(audit, "operator_saved_pose_limits_restored", configured_limits_deg=original)


def capture_zero_state(robot, *, audit=None, timeout_s=15.0,
                       clock=time.monotonic, sleep=time.sleep, operator_saved_pose=False):
    """Read an explicitly requested manually positioned pose on the same owner.

    Call only before policy writers start. The caller is responsible for
    persisting robot identity, calibration identity and this operator intent.
    """
    controller = _controller(robot)
    _freeze_reference(controller)
    deadline = clock() + timeout_s
    pose = _read_pose(robot, controller, cancelled=lambda: False, deadline=deadline, clock=clock)
    native_deg = {joint: math.degrees(pose[joint]) for joint in JOINTS}
    limits = {joint: list(map(float, robot.config.joint_limits[joint])) for joint in JOINTS}
    clipped = {joint: max(limits[joint][0], min(limits[joint][1], native_deg[joint]))
               for joint in JOINTS}
    _emit(audit, "zero_state_capture_read", native_positions_rad=pose,
          native_positions_deg=native_deg, configured_limits_deg=limits,
          projected_send_action_native_goal_deg=clipped,
          projected_goal_clip_delta_deg={joint: clipped[joint] - native_deg[joint] for joint in JOINTS},
          joint_directions={joint: float(robot.config.joint_directions[joint]) for joint in JOINTS},
          feedback_source="fresh READ_PARAMETER 0x7019 / 0x7005 / 0x3022",
          projected_goal_scope="Configured clipping only, before relative-target or gripper-speed limiting; no command sent by this diagnostic")
    pose = _validate_pose(robot, pose, operator_saved_pose=operator_saved_pose)
    if operator_saved_pose:
        _emit(audit, "zero_state_captured", native_positions_rad=pose,
              fresh_sweeps=1, capture_method="current fresh native snapshot",
              feedback_source="fresh READ_PARAMETER 0x7019 / 0x7005 / 0x3022")
        return pose
    with _operator_pose_limits(robot, pose, enabled=operator_saved_pose, audit=audit):
        reference = dict(pose)
        stable_since = clock()
        sweeps = 0
        while True:
            if controller is None:
                _send_native_pose(robot, reference)
            pose = _read_pose(robot, controller, cancelled=lambda: False,
                              deadline=deadline, clock=clock)
            sweeps += 1
            if any(abs(math.degrees(pose[joint] - reference[joint])) > 0.5 for joint in JOINTS):
                raise ReturnFailed("Manually positioned zero_state moved by more than 0.5 degrees during capture.")
            if sweeps >= 3 and clock() - stable_since >= 0.5:
                break
            sleep(0.02)
        _validate_pose(robot, pose)
        _emit(audit, "zero_state_captured", native_positions_rad=reference,
              verified_native_positions_rad=pose,
              native_command_target_rad=_configured_command_pose(robot, reference),
              boundary_feedback_tolerance_deg=POSE_BOUNDARY_TOLERANCE_DEG, fresh_sweeps=sweeps,
              feedback_source="fresh READ_PARAMETER 0x7019 / 0x7005 / 0x3022")
        return reference


def capture_session_start_pose(robot, *, cancel_event=None, audit=None,
                               clock=time.monotonic, sleep=time.sleep):
    """Sample the operator's current pose without sending a position target."""
    controller = _controller(robot)
    if controller is not None:
        raise ReturnFailed("Session start reference requires the native RS owner.")
    cancelled = cancel_event.is_set if cancel_event is not None else lambda: False
    deadline = clock() + 2.0
    stable_since = None
    stable_sweeps = 0
    minima = {}
    maxima = {}
    try:
        while not cancelled():
            pose = _read_pose(robot, controller, cancelled=cancelled, deadline=deadline, clock=clock)
            _validate_pose(robot, pose, operator_saved_pose=True)
            now = clock()
            if stable_since is not None:
                minima = {joint: min(minima[joint], pose[joint]) for joint in ARM_JOINTS}
                maxima = {joint: max(maxima[joint], pose[joint]) for joint in ARM_JOINTS}
            if stable_since is None or any(
                    math.degrees(maxima[joint] - minima[joint]) > POSE_BOUNDARY_TOLERANCE_DEG
                    for joint in ARM_JOINTS):
                stable_since = now
                stable_sweeps = 0
                minima = {joint: pose[joint] for joint in ARM_JOINTS}
                maxima = minima.copy()
            stable_sweeps += 1
            if stable_sweeps >= 3 and now - stable_since >= 0.2:
                _emit(audit, "session_start_pose_captured", native_positions_rad=pose,
                      stable_seconds=now - stable_since, fresh_sweeps=stable_sweeps,
                      arm_span_deg={joint: math.degrees(maxima[joint] - minima[joint])
                                    for joint in ARM_JOINTS})
                return pose
            sleep(0.02)
    except ReturnFailed:
        if not cancelled():
            raise
    return None


def run_return(robot, target_native_rad, *, tolerance_deg=2.0, stable_s=0.5,
               timeout_s=45.0, speed_deg_s=25.0, cancel_event=None,
               audit=None, release_after=False, clock=time.monotonic,
               sleep=time.sleep, operator_saved_pose=False, paced=False,
               park_gripper=False, gripper_parked=False):
    """Return and verify all seven motor-native angles, retaining torque.

    release_after records the caller's requested next action in the receipt.
    The caller performs its verified release lifecycle after this succeeds.
    Cancellation or any read/control failure raises ReturnFailed. The existing
    command reference is frozen for the caller's persistent damped-hold path.
    """
    for name, value, upper in (("tolerance_deg", tolerance_deg, 2.0),
                               ("stable_s", stable_s, 10.0),
                               ("timeout_s", timeout_s, 300.0),
                               ("speed_deg_s", speed_deg_s, 25.0)):
        if not math.isfinite(float(value)) or not 0 < value <= upper:
            raise ValueError(f"{name} must be positive and at most {upper}.")
    target = _validate_pose(robot, target_native_rad, operator_saved_pose=operator_saved_pose)
    controller = _controller(robot)
    if park_gripper and (controller is not None or not operator_saved_pose or not paced):
        raise ReturnFailed("One-close gripper parking requires the native paced operator return.")
    gripper_goal = _configured_command_pose(robot, target)["gripper"] if park_gripper else None
    cancel_event = cancel_event or threading.Event()
    cancelled = cancel_event.is_set
    deadline = clock() + timeout_s
    previous_config = controller.config if controller is not None else None
    try:
        _freeze_reference(controller)
        current = _read_pose(robot, controller, cancelled=cancelled,
                             deadline=deadline, clock=clock)
        with _operator_pose_limits(robot, current, target, enabled=operator_saved_pose, audit=audit):
            _validate_pose(robot, current)
            if controller is None:
                _emit(audit, "saved_pose_return_started", native_target_rad=target,
                      tolerance_deg=tolerance_deg, max_speed_deg_s=speed_deg_s)
                receipt = _native_return(
                    robot, target, current, cancelled=cancelled, deadline=deadline, clock=clock,
                    sleep=sleep, speed_deg_s=speed_deg_s, tolerance_deg=tolerance_deg,
                    stable_s=stable_s, release_after=release_after, audit=audit, paced=paced,
                    gripper_goal=gripper_goal, gripper_parked=gripper_parked)
                _emit(audit, "saved_pose_return_verified", **receipt)
                return receipt
            controller.config = replace(
                previous_config,
                arm_speed_deg_s=min(previous_config.arm_speed_deg_s, speed_deg_s),
                gripper_speed_deg_s=min(previous_config.gripper_speed_deg_s, speed_deg_s))
            _emit(audit, "saved_pose_return_started", native_target_rad=target,
                  tolerance_deg=tolerance_deg, max_speed_deg_s=speed_deg_s)
            stage_target = dict(current)
            total_sweeps = 0
            for number, stage in enumerate(STAGES, 1):
                for joint in stage:
                    stage_target[joint] = target[joint]
                command_target = _configured_command_pose(robot, stage_target)
                action = {f"{joint}.pos": math.degrees(command_target[joint]) /
                          float(robot.config.joint_directions[joint]) for joint in JOINTS}
                _check_cancel(robot, cancelled, deadline, clock)
                controller.publish_action(action)
                stable_since = None
                stage_sweeps = 0
                while True:
                    current = _read_pose(robot, controller, cancelled=cancelled,
                                         deadline=deadline, clock=clock)
                    total_sweeps += 1
                    stage_sweeps += 1
                    errors = {joint: math.degrees(current[joint] - stage_target[joint])
                              for joint in JOINTS}
                    with controller._state_lock:
                        command_complete = all(
                            abs(float(controller.command_positions[joint]) -
                                math.degrees(command_target[joint])) <= 0.05 for joint in JOINTS)
                    now = clock()
                    if command_complete and all(abs(error) <= tolerance_deg
                                                for error in errors.values()):
                        stable_since = now if stable_since is None else stable_since
                        if stage_sweeps >= 3 and now - stable_since >= stable_s:
                            break
                    else:
                        stable_since = None
                    sleep(0.02)
                _emit(audit, "saved_pose_return_stage_verified", stage=number,
                      native_positions_rad=current, errors_deg=errors,
                      fresh_sweeps=stage_sweeps)
            receipt = {"native_positions_rad": current, "native_target_rad": target,
                       "native_command_target_rad": _configured_command_pose(robot, target),
                       "boundary_feedback_tolerance_deg": POSE_BOUNDARY_TOLERANCE_DEG,
                       "errors_deg": errors, "tolerance_deg": tolerance_deg,
                       "stable_seconds": clock() - stable_since,
                       "fresh_sweeps": total_sweeps, "release_after": bool(release_after),
                       "feedback_source": "fresh READ_PARAMETER 0x7019 / 0x7005 / 0x3022"}
            _emit(audit, "saved_pose_return_verified", **receipt)
            return receipt
    except BaseException as error:
        try:
            _freeze_reference(controller)
        except Exception as freeze_error:
            _emit(audit, "saved_pose_reference_freeze_failed", error=str(freeze_error))
        _emit(audit, "saved_pose_return_failed", error=str(error))
        if isinstance(error, ReturnFailed):
            raise
        raise ReturnFailed(f"Saved-pose return failed: {error}") from error
    finally:
        if controller is not None:
            controller.config = previous_config


def hold_saved_pose(robot, target_native_rad, duration_s, *, cancel_event=None,
                    tolerance_deg=2.0, audit=None, clock=time.monotonic, sleep=time.sleep,
                    operator_saved_pose=False, gripper_parked=False, arm_only=False):
    """Maintain the verified pose during rest on this same control owner."""
    if not math.isfinite(duration_s) or not 0 < duration_s <= 600:
        raise ValueError("Rest duration must be positive and at most 600 seconds.")
    if not math.isfinite(tolerance_deg) or not 0 < tolerance_deg <= 2:
        raise ValueError("Rest tolerance must be positive and at most 2 degrees.")
    target = _validate_pose(robot, target_native_rad, operator_saved_pose=operator_saved_pose)
    controller = _controller(robot)
    joints = ARM_JOINTS if gripper_parked or arm_only else JOINTS
    if joints == ARM_JOINTS and (controller is not None or not operator_saved_pose):
        raise ReturnFailed("Arm-only hold requires the native operator return.")
    cancel_event = cancel_event or threading.Event()
    cancelled = cancel_event.is_set
    started = clock()
    deadline = started + duration_s + 10.0
    next_check = started
    checks = 0
    try:
        with _operator_pose_limits(robot, target, enabled=operator_saved_pose, audit=audit):
            while clock() - started < duration_s:
                if cancelled():
                    return {"cancelled": True, "fresh_sweeps": checks}
                if controller is None:
                    _send_native_pose(robot, target, joints=joints)
                else:
                    controller.raise_if_failed()
                if not operator_saved_pose and clock() >= next_check:
                    pose = _read_pose(robot, controller, cancelled=lambda: False,
                                      deadline=deadline, clock=clock)
                    checks += 1
                    if any(abs(math.degrees(pose[joint] - target[joint])) > tolerance_deg
                           for joint in JOINTS):
                        raise ReturnFailed("The robot drifted outside saved-pose tolerance during rest.")
                    next_check = clock() + 0.5
                sleep(0.02)
            command_target = _configured_command_pose(robot, target)
            if joints == ARM_JOINTS:
                del command_target["gripper"]
            receipt = {"cancelled": False, "fresh_sweeps": checks,
                       "native_target_rad": target,
                       "native_command_target_rad": command_target,
                       "boundary_feedback_tolerance_deg": POSE_BOUNDARY_TOLERANCE_DEG,
                       "hold_path": "official_owner_hold" if operator_saved_pose else "fresh_feedback_hold",
                       "rest_seconds": clock() - started}
            if gripper_parked:
                receipt["gripper_parked"] = True
            _emit(audit, "saved_pose_rest_complete", **receipt)
            return receipt
    except BaseException as error:
        _freeze_reference(controller)
        if isinstance(error, ReturnFailed):
            raise
        raise ReturnFailed(f"Saved-pose rest failed: {error}") from error
