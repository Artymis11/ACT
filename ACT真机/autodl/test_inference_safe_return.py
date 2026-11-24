"""Fake-owner regressions. These tests never import a hardware driver."""

from dataclasses import dataclass
import math
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import inference_safe_return as target


@dataclass(frozen=True)
class ControlConfig:
    arm_speed_deg_s: float = 300.0
    gripper_speed_deg_s: float = 720.0


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class Motor:
    def __init__(self, robot, name):
        self.robot = robot
        self.name = name
        self.fault = 0
        self.mode = 0
        self.reads = []
        self.timeout = False

    def _read(self, register, timeout_ms):
        self.reads.append(register)
        self.robot.clock.now += 0.002
        if self.timeout:
            raise TimeoutError("fake read timeout")

    def robstride_get_param_u32(self, register, timeout_ms):
        self._read(register, timeout_ms)
        return self.fault

    def robstride_get_param_i8(self, register, timeout_ms):
        self._read(register, timeout_ms)
        return self.mode

    def robstride_get_param_f32_host_id(self, register, host_id, timeout_ms):
        self._read(register, timeout_ms)
        if host_id != 0xFD:
            raise AssertionError("Host identity was not preserved")
        return self.robot.positions[self.name]


class Controller:
    def __init__(self, robot):
        self.robot = robot
        self.config = ControlConfig()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self.can_lock = threading.RLock()
        self.command_positions = {name: math.degrees(robot.positions[name])
                                  for name in target.JOINTS}
        self.target_positions = dict(self.command_positions)
        self.published = []
        self.failed = False
        self.move = True
        self.on_publish = None

    def raise_if_failed(self):
        if self.failed:
            raise RuntimeError("fake retained MIT fault")

    def publish_action(self, action):
        if self.config.arm_speed_deg_s > 25 or self.config.gripper_speed_deg_s > 25:
            raise AssertionError("Homing speed was not lowered")
        self.published.append(dict(action))
        mapped = {name: value * self.robot.config.joint_directions[name]
                  for key, value in action.items() for name in [key.removesuffix(".pos")]}
        self.target_positions = dict(mapped)
        if self.move:
            self.command_positions = dict(mapped)
            self.robot.positions = {name: math.radians(value) for name, value in mapped.items()}
        if self.on_publish:
            self.on_publish()


class Robot:
    def __init__(self):
        self.clock = Clock()
        self.bus = object()
        self.positions = {name: math.radians(10) for name in target.JOINTS}
        self.config = SimpleNamespace(
            motor_can_ids={name: (index, 0xFD) for index, name in enumerate(target.JOINTS, 1)},
            joint_directions=dict(zip(target.JOINTS, (1, 1, -1, -1, -1, 1, 6))),
            joint_limits={name: (-180, 270) for name in target.JOINTS})
        self.motors = {name: Motor(self, name) for name in target.JOINTS}
        self.controller = Controller(self)
        setattr(self, target.CONTROLLER_ATTRIBUTE, self.controller)
        self.disable_called = False

    def disable_torque(self):
        self.disable_called = True
        raise AssertionError("Return module must retain caller lifecycle ownership")


class NativeRobot(Robot):
    def __init__(self):
        super().__init__()
        delattr(self, target.CONTROLLER_ATTRIBUTE)
        self.motor_type = "rs"
        self.commands = []
        self.on_send = None
        self.send_error = None
        self.move = True

    def send_action(self, action):
        if self.send_error is not None:
            raise self.send_error
        mapped = {key: value * self.config.joint_directions[key.removesuffix(".pos")]
                  for key, value in action.items()}
        self.commands.append((self.clock(), mapped))
        if self.move:
            self.positions.update({key.removesuffix(".pos"): math.radians(value)
                                   for key, value in mapped.items()})
        if self.on_send:
            self.on_send()
        return mapped


class SessionStartCaptureTests(unittest.TestCase):
    def test_reads_stable_session_start_without_actuating(self):
        robot = NativeRobot()
        start = dict(robot.positions)
        robot.send_error = AssertionError("Session start capture must only read the arm")

        pose = target.capture_session_start_pose(robot, clock=robot.clock, sleep=robot.clock.sleep)

        self.assertEqual(pose, start)
        self.assertGreaterEqual(robot.clock(), 0.2)
        for motor in robot.motors.values():
            self.assertEqual(set(motor.reads), {0x3022, 0x7005, 0x7019})
            self.assertGreaterEqual(motor.reads.count(0x7019), 3)
        self.assertFalse(robot.disable_called)

    def test_rejects_moving_arm_until_cancelled(self):
        robot = NativeRobot()
        cancelled = threading.Event()
        robot.send_error = AssertionError("Session start capture must only read the arm")
        joint = "shoulder_lift"

        def move(duration):
            robot.clock.sleep(duration)
            robot.positions[joint] += math.radians(0.6)
            if robot.clock() >= 0.5:
                cancelled.set()

        pose = target.capture_session_start_pose(robot, cancel_event=cancelled, clock=robot.clock, sleep=move)

        self.assertIsNone(pose)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(robot.disable_called)


class ReturnTests(unittest.TestCase):
    def setUp(self):
        self.robot = Robot()
        self.goal = {name: math.radians(20) for name in target.JOINTS}

    def run_return(self, **options):
        return target.run_return(self.robot, self.goal, clock=self.robot.clock,
                                 sleep=self.robot.clock.sleep, **options)

    def test_two_stages_and_native_direction_mapping(self):
        receipt = self.run_return(release_after=True)
        published = self.robot.controller.published
        self.assertEqual(len(published), 2)
        self.assertAlmostEqual(published[0]["elbow_flex.pos"], -10)
        self.assertAlmostEqual(published[1]["elbow_flex.pos"], -20)
        self.assertAlmostEqual(published[1]["gripper.pos"], 20 / 6)
        self.assertTrue(receipt["release_after"])
        self.assertGreaterEqual(receipt["fresh_sweeps"], 6)
        self.assertGreaterEqual(receipt["stable_seconds"], 0.5)
        self.assertEqual(set(receipt["errors_deg"]), set(target.JOINTS))
        self.assertFalse(self.robot.disable_called)
        self.assertEqual(self.robot.controller.config, ControlConfig())
        for motor in self.robot.motors.values():
            self.assertEqual(set(motor.reads), {0x3022, 0x7005, 0x7019})

    def test_explicit_capture_reads_every_joint(self):
        pose = target.capture_zero_state(self.robot, clock=self.robot.clock,
                                         sleep=self.robot.clock.sleep)
        self.assertEqual(pose, self.robot.positions)
        self.assertFalse(self.robot.controller.published)

    def test_capture_rejection_reports_seven_fresh_angles_and_exact_bounds(self):
        self.robot.config.joint_limits["shoulder_lift"] = (0.0, 170.0)
        self.robot.positions["shoulder_lift"] = math.radians(-0.75)
        events = []
        with self.assertRaisesRegex(target.ReturnFailed, "native_deg=-0.750000000.*170.000000000"):
            target.capture_zero_state(self.robot, clock=self.robot.clock,
                                      sleep=self.robot.clock.sleep,
                                      audit=lambda event, **values: events.append((event, values)))
        self.assertEqual(len(events), 1)
        name, payload = events[0]
        self.assertEqual(name, "zero_state_capture_read")
        self.assertEqual(set(payload["native_positions_deg"]), set(target.JOINTS))
        self.assertEqual(payload["configured_limits_deg"]["shoulder_lift"], [0.0, 170.0])
        self.assertAlmostEqual(payload["native_positions_deg"]["shoulder_lift"], -0.75)
        self.assertEqual(payload["projected_send_action_native_goal_deg"]["shoulder_lift"], 0.0)
        self.assertAlmostEqual(payload["projected_goal_clip_delta_deg"]["shoulder_lift"], 0.75)
        self.assertEqual(self.robot.config.joint_limits["shoulder_lift"], (0.0, 170.0))
        self.assertFalse(self.robot.controller.published)
        self.assertFalse(self.robot.disable_called)

    def test_gripper_is_verified_as_motor_native_angle(self):
        self.robot.controller.move = False
        self.goal = dict(self.robot.positions)
        self.goal["gripper"] += math.radians(2.1)
        with self.assertRaisesRegex(target.ReturnFailed, "deadline"):
            self.run_return(timeout_s=1)
        self.assertFalse(self.robot.disable_called)

    def test_fault_refuses_any_return_target(self):
        self.robot.motors["wrist_roll"].fault = 2
        with self.assertRaisesRegex(target.ReturnFailed, "faultSta"):
            self.run_return()
        self.assertFalse(self.robot.controller.published)

    def test_wrong_mode_refuses_any_return_target(self):
        self.robot.motors["gripper"].mode = 1
        with self.assertRaisesRegex(target.ReturnFailed, "run_mode"):
            self.run_return()
        self.assertFalse(self.robot.controller.published)

    def test_late_fault_freezes_and_restores_speed(self):
        self.robot.controller.on_publish = lambda: setattr(
            self.robot.motors["shoulder_pan"], "fault", 1)
        with self.assertRaisesRegex(target.ReturnFailed, "faultSta"):
            self.run_return()
        self.assertEqual(self.robot.controller.target_positions,
                         self.robot.controller.command_positions)
        self.assertEqual(self.robot.controller.config, ControlConfig())
        self.assertFalse(self.robot.disable_called)

    def test_cancel_during_return_freezes(self):
        cancel = threading.Event()
        self.robot.controller.on_publish = cancel.set
        with self.assertRaisesRegex(target.ReturnFailed, "cancelled"):
            self.run_return(cancel_event=cancel)
        self.assertEqual(self.robot.controller.config, ControlConfig())
        self.assertFalse(self.robot.disable_called)

    def test_read_timeout_keeps_owner_and_propagates(self):
        self.robot.motors["shoulder_pan"].timeout = True
        with self.assertRaisesRegex(target.ReturnFailed, "fake read timeout"):
            self.run_return()
        self.assertIsNotNone(self.robot.bus)
        self.assertFalse(self.robot.controller.published)

    def test_cached_command_is_not_accepted_as_measured(self):
        self.robot.controller.move = False
        with self.assertRaisesRegex(target.ReturnFailed, "deadline"):
            self.run_return(timeout_s=1)
        self.assertTrue(self.robot.motors["shoulder_pan"].reads)

    def test_nan_measurement_fails(self):
        self.robot.positions["gripper"] = float("nan")
        with self.assertRaisesRegex(target.ReturnFailed, "non-finite"):
            self.run_return()

    def test_invalid_target_rejected_before_motion(self):
        del self.goal["gripper"]
        with self.assertRaisesRegex(target.ReturnFailed, "exactly all seven"):
            self.run_return()
        self.assertFalse(self.robot.controller.published)

    def test_out_of_range_target_is_not_silently_clipped(self):
        self.goal["shoulder_pan"] = math.radians(271)
        with self.assertRaisesRegex(target.ReturnFailed, "outside"):
            self.run_return()
        self.assertFalse(self.robot.controller.published)

    def test_current_out_of_range_fails_before_policy_clipping(self):
        self.robot.positions["shoulder_pan"] = math.radians(271)
        with self.assertRaisesRegex(target.ReturnFailed, "outside"):
            self.run_return()
        self.assertFalse(self.robot.controller.published)

    def test_mapping_change_refused(self):
        self.robot.config.motor_can_ids["gripper"] = (6, 0xFD)
        with self.assertRaisesRegex(target.ReturnFailed, "CAN 1..7"):
            self.run_return()

    def test_stopped_controller_refused(self):
        self.robot.controller._stop.set()
        with self.assertRaisesRegex(target.ReturnFailed, "has stopped"):
            self.run_return()

    def test_tolerance_cannot_exceed_user_bound(self):
        with self.assertRaises(ValueError):
            self.run_return(tolerance_deg=2.1)
        self.assertFalse(self.robot.controller.published)

    def test_small_boundary_feedback_keeps_saved_pose_and_clips_commands_in_both_paths(self):
        for factory in (Robot, NativeRobot):
            with self.subTest(path=factory.__name__):
                robot = factory()
                robot.config.joint_limits.update(shoulder_lift=(0.0, 170.0),
                                                 elbow_flex=(0.0, 200.0), gripper=(0.0, 270.0))
                robot.positions.update(shoulder_lift=math.radians(-0.122070314785),
                                       elbow_flex=math.radians(200.25), gripper=math.radians(270.25))
                original = dict(robot.positions)
                limits = dict(robot.config.joint_limits)
                saved = target.capture_zero_state(robot, clock=robot.clock, sleep=robot.clock.sleep)
                self.assertEqual(saved, original)
                receipt = target.run_return(robot, saved, clock=robot.clock, sleep=robot.clock.sleep)
                self.assertEqual(receipt["native_target_rad"], original)
                self.assertAlmostEqual(math.degrees(receipt["native_command_target_rad"]["shoulder_lift"]), 0.0)
                self.assertAlmostEqual(receipt["errors_deg"]["shoulder_lift"], 0.122070314785)
                self.assertAlmostEqual(receipt["errors_deg"]["elbow_flex"], -0.25)
                self.assertAlmostEqual(receipt["errors_deg"]["gripper"], -0.25)
                target.hold_saved_pose(robot, saved, 0.5, clock=robot.clock, sleep=robot.clock.sleep)
                if factory is NativeRobot:
                    commands = [command for timestamp, command in robot.commands]
                else:
                    commands = [{key: value * robot.config.joint_directions[key.removesuffix(".pos")]
                                 for key, value in action.items()} for action in robot.controller.published]
                for command in commands:
                    for joint, (lower, upper) in limits.items():
                        self.assertGreaterEqual(command[f"{joint}.pos"], lower - 1e-10)
                        self.assertLessEqual(command[f"{joint}.pos"], upper + 1e-10)
                self.assertEqual(robot.config.joint_limits, limits)
                self.assertFalse(robot.disable_called)

    def test_boundary_allowance_cannot_accept_large_or_inverted_limits(self):
        for factory in (Robot, NativeRobot):
            for degrees in (-0.500001, 170.500001):
                with self.subTest(path=factory.__name__, degrees=degrees):
                    robot = factory()
                    robot.config.joint_limits["shoulder_lift"] = (0.0, 170.0)
                    robot.positions["shoulder_lift"] = math.radians(degrees)
                    with self.assertRaisesRegex(target.ReturnFailed, "outside configured"):
                        target.capture_zero_state(robot, clock=robot.clock, sleep=robot.clock.sleep)
            robot.config.joint_limits["shoulder_lift"] = (0.0, -0.1)
            with self.assertRaisesRegex(target.ReturnFailed, "outside configured"):
                target._validate_pose(robot, {joint: 0.0 for joint in target.JOINTS})

    def test_small_boundary_allowance_preserves_requested_arrival_tolerance(self):
        robot = NativeRobot()
        robot.config.joint_limits["shoulder_lift"] = (0.0, 170.0)
        saved = dict(robot.positions, shoulder_lift=math.radians(-0.122070314785))
        with self.assertRaisesRegex(target.ReturnFailed, "deadline"):
            target.run_return(robot, saved, tolerance_deg=0.1, timeout_s=1.0,
                              clock=robot.clock, sleep=robot.clock.sleep)
        with self.assertRaisesRegex(target.ReturnFailed, "drifted"):
            target.hold_saved_pose(robot, saved, 0.5, tolerance_deg=0.1,
                                   clock=robot.clock, sleep=robot.clock.sleep)
        self.assertFalse(robot.disable_called)


class NativeReturnTests(unittest.TestCase):
    def setUp(self):
        self.robot = NativeRobot()
        self.goal = {name: math.radians(20) for name in target.JOINTS}

    def run_return(self, **options):
        return target.run_return(self.robot, self.goal, clock=self.robot.clock,
                                 sleep=self.robot.clock.sleep, **options)

    def test_actual_native_driver_path_smoothstep_and_speed(self):
        receipt = self.run_return()
        self.assertEqual(receipt["control_path"], "existing native RS send_action")
        self.assertTrue(all(abs(value) <= 2 for value in receipt["errors_deg"].values()))
        previous_time = 0
        previous = {f"{name}.pos": 10 for name in target.JOINTS}
        for timestamp, command in self.robot.commands:
            if timestamp > previous_time:
                self.assertTrue(all(abs(command[key] - previous[key]) <=
                                    25 * (timestamp - previous_time) + 0.001
                                    for key in command))
            previous_time, previous = timestamp, command
        self.assertFalse(self.robot.disable_called)

    def test_native_capture_holds_and_checks_stability(self):
        pose = target.capture_zero_state(self.robot, clock=self.robot.clock,
                                         sleep=self.robot.clock.sleep)
        self.assertEqual(pose, self.robot.positions)
        self.assertGreaterEqual(len(self.robot.commands), 3)
        self.assertGreaterEqual(self.robot.clock(), 0.5)
        self.assertTrue(all(set(motor.reads) == {0x3022, 0x7005, 0x7019}
                            for motor in self.robot.motors.values()))

    def test_stage_boundary_keeps_last_command_despite_arrival_error(self):
        def measured_lag():
            self.robot.positions["wrist_flex"] -= math.radians(1.99)
        self.robot.on_send = measured_lag
        self.run_return()
        for (previous_time, previous), (timestamp, command) in zip(
                self.robot.commands, self.robot.commands[1:]):
            self.assertLessEqual(abs(command["wrist_flex.pos"] - previous["wrist_flex.pos"]),
                                 25 * (timestamp - previous_time) + 0.001)

    def test_native_overheat_interrupt_enters_failure_path(self):
        self.robot.send_error = KeyboardInterrupt("Overheat detected")
        with self.assertRaisesRegex(target.ReturnFailed, "Overheat detected"):
            self.run_return()
        self.assertIsNotNone(self.robot.bus)
        self.assertFalse(self.robot.disable_called)

    def test_native_capture_limit_diagnostic_precedes_any_send(self):
        self.robot.config.joint_limits["shoulder_lift"] = (0.0, 170.0)
        self.robot.positions["shoulder_lift"] = math.radians(-0.75)
        events = []
        with self.assertRaisesRegex(target.ReturnFailed, "native_deg=-0.750000000"):
            target.capture_zero_state(self.robot, clock=self.robot.clock,
                                      sleep=self.robot.clock.sleep,
                                      audit=lambda event, **values: events.append((event, values)))
        self.assertEqual(events[0][0], "zero_state_capture_read")
        self.assertEqual(set(events[0][1]["native_positions_rad"]), set(target.JOINTS))
        self.assertEqual(self.robot.commands, [])
        self.assertFalse(self.robot.disable_called)

    def test_native_cancel_mid_motion(self):
        cancel = threading.Event()
        self.robot.on_send = cancel.set
        with self.assertRaisesRegex(target.ReturnFailed, "cancelled"):
            self.run_return(cancel_event=cancel)
        self.assertEqual(len(self.robot.commands), 1)
        self.assertFalse(self.robot.disable_called)

    def test_native_fault_mid_motion(self):
        self.robot.on_send = lambda: setattr(self.robot.motors["gripper"], "fault", 2)
        with self.assertRaisesRegex(target.ReturnFailed, "faultSta"):
            self.run_return()
        self.assertFalse(self.robot.disable_called)

    def test_native_rest_repeatedly_sends_hold_reference(self):
        pose = dict(self.robot.positions)
        receipt = target.hold_saved_pose(self.robot, pose, 1,
                                        clock=self.robot.clock, sleep=self.robot.clock.sleep)
        self.assertFalse(receipt["cancelled"])
        self.assertGreaterEqual(receipt["fresh_sweeps"], 2)
        self.assertGreaterEqual(len(self.robot.commands), 30)
        self.assertFalse(self.robot.disable_called)

    def test_native_rest_cancel_is_normal_completion(self):
        cancel = threading.Event()
        self.robot.on_send = cancel.set
        receipt = target.hold_saved_pose(self.robot, self.robot.positions, 1,
                                        cancel_event=cancel, clock=self.robot.clock,
                                        sleep=self.robot.clock.sleep)
        self.assertTrue(receipt["cancelled"])
        self.assertFalse(self.robot.disable_called)

    def test_native_read_failure_holds_last_command_without_disable(self):
        self.robot.on_send = lambda: setattr(self.robot.motors["gripper"], "timeout", True)
        with self.assertRaisesRegex(target.ReturnFailed, "fake read timeout"):
            self.run_return()
        self.assertEqual(len(self.robot.commands), 1)
        self.assertFalse(self.robot.disable_called)


class ParkGripperTests(unittest.TestCase):
    def setUp(self):
        self.robot = NativeRobot()
        self.robot.config.joint_limits["gripper"] = (0, 270)
        self.goal = {**self.robot.positions, "gripper": math.radians(-4.02)}
        self.j7_writes = []
        self.references = []
        self.gripper = self.robot.motors["gripper"]
        self.gripper.send_mit = self.send_mit
        self.robot.mit_output_torque_limit = self.limit_torque
        self.damping = self.enterContext(patch.object(target, "_damp_gripper", side_effect=self.damp))

    def limit_torque(self, motor, reference):
        self.assertIs(motor, self.gripper)
        self.references.append(math.degrees(reference))
        return 0.5 if len(self.references) == 1 else -0.5

    def send_mit(self, position, velocity, kp, kd, torque):
        self.j7_writes.append(torque)
        if torque < 0:
            self.robot.positions["gripper"] -= math.radians(0.5)

    def damp(self, robot, position):
        self.assertIs(robot, self.robot)
        self.gripper.send_mit(position, 0.0, 0.0, 2.5, 0.0)

    def run_return(self, **options):
        return target.run_return(self.robot, self.goal, operator_saved_pose=True, paced=True,
                                 park_gripper=True, clock=self.robot.clock, sleep=self.robot.clock.sleep,
                                 **options)

    def test_one_closing_pass_then_six_axis_wait_and_recheck(self):
        receipt = self.run_return()
        self.assertTrue(receipt["gripper_parked"])
        self.assertEqual(receipt["native_command_target_rad"]["gripper"], 0.0)
        self.assertLessEqual(math.degrees(receipt["native_positions_rad"]["gripper"]), 2.0)
        self.assertTrue(any(torque < 0 for torque in self.j7_writes))
        self.assertTrue(all(torque <= 0 for torque in self.j7_writes))
        self.assertGreaterEqual(min(self.references), 0.0)
        self.assertEqual(self.j7_writes[-1], 0.0)
        self.assertEqual(self.damping.call_count, 2)
        writes = len(self.j7_writes)
        hold = target.hold_saved_pose(self.robot, self.goal, 1.0, operator_saved_pose=True,
                                     gripper_parked=True, clock=self.robot.clock,
                                     sleep=self.robot.clock.sleep)
        self.assertNotIn("gripper", hold["native_command_target_rad"])
        self.assertTrue(self.run_return(gripper_parked=True)["gripper_parked"])
        self.assertEqual(len(self.j7_writes), writes)
        self.assertTrue(all("gripper.pos" not in action for _, action in self.robot.commands))

    def test_saved_closed_angle_below_configured_goal_does_not_reopen(self):
        self.robot.positions["gripper"] = self.goal["gripper"]
        self.assertTrue(self.run_return()["gripper_parked"])
        self.assertEqual(self.j7_writes, [0.0])
        self.assertEqual(self.references, [])

    def test_stuck_gripper_cuts_off_torque_and_fails_return(self):
        def stuck(*args):
            self.j7_writes.append(args[-1])
        self.gripper.send_mit = stuck
        with self.assertRaisesRegex(target.ReturnFailed, "did not reach"):
            self.run_return()
        self.assertEqual(self.j7_writes[-1], 0.0)
        self.assertFalse(self.robot.disable_called)

    def test_parked_but_drifted_away_fails_without_an_extra_close(self):
        with self.assertRaisesRegex(target.ReturnFailed, "did not verify closed"):
            self.run_return(gripper_parked=True)
        self.assertEqual(self.j7_writes, [])

    def test_no_gripper_write_if_native_send_rejects_temperature(self):
        self.robot.send_error = RuntimeError("overheat")
        with self.assertRaisesRegex(target.ReturnFailed, "overheat"):
            self.run_return()
        self.assertEqual(self.j7_writes, [])


class OperatorPoseTests(unittest.TestCase):
    def test_snapshot_captures_current_pose_without_sending(self):
        robot = NativeRobot()
        robot.config.joint_limits["gripper"] = (0.0, 270.0)
        robot.positions["gripper"] = math.radians(-4.04)
        robot.send_error = AssertionError("Snapshot capture must not send motion")
        saved = target.capture_zero_state(
            robot, operator_saved_pose=True, clock=robot.clock, sleep=robot.clock.sleep)
        self.assertEqual(saved, robot.positions)
        self.assertEqual(robot.commands, [])
        self.assertTrue(all(motor.reads == [0x3022, 0x7005, 0x7019]
                            for motor in robot.motors.values()))

    def test_operator_hold_uses_existing_owner_without_register_polling(self):
        for factory in (NativeRobot, Robot):
            with self.subTest(path=factory.__name__):
                robot = factory()
                robot.config.joint_limits["gripper"] = (0.0, 270.0)
                robot.positions["gripper"] = math.radians(-4.04)
                limits = robot.config.joint_limits
                for motor in robot.motors.values():
                    motor.timeout = True
                receipt = target.hold_saved_pose(
                    robot, dict(robot.positions), 0.3, operator_saved_pose=True,
                    clock=robot.clock, sleep=robot.clock.sleep)
                self.assertEqual(receipt["fresh_sweeps"], 0)
                self.assertEqual(receipt["hold_path"], "official_owner_hold")
                self.assertTrue(all(not motor.reads for motor in robot.motors.values()))
                self.assertIs(robot.config.joint_limits, limits)
                self.assertFalse(robot.disable_called)
                if factory is NativeRobot:
                    self.assertTrue(robot.commands)
                    robot.send_error = RuntimeError("native send failed")
                else:
                    robot.controller.failed = True
                with self.assertRaises(RuntimeError):
                    target.hold_saved_pose(
                        robot, dict(robot.positions), 0.3, operator_saved_pose=True,
                        clock=robot.clock, sleep=robot.clock.sleep)
                self.assertIs(robot.config.joint_limits, limits)


if __name__ == "__main__":
    unittest.main()
