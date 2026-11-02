"""Injected runner tests. No PCAN DLL, XInput read, or physical motion.

FakeRobot is admitted only through the explicit test-only injection option;
its fixture is not a production configuration or physical acceptance.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fake_robot import DEFAULT_FAKE_ORIGIN, FakeRobot, synthetic_config
from session_core import SessionError
from unified_runner import BUTTONS, input_frame

import live_runner
from live_runner import NativeGamepadRunner, main
from native_pcan_backend import NativePcanBackend


ROOT = Path(__file__).resolve().parent


class Clock:
    def __init__(self, now=10.0):
        self.now=now

    def __call__(self):
        return self.now


class DelayedRobot(FakeRobot):
    def __init__(self, clock, origin=DEFAULT_FAKE_ORIGIN):
        super().__init__(origin)
        self.clock=clock
        self.read_delay=0.0

    def read_feedback(self, session_id, now):
        self.clock.now+=self.read_delay
        return super().read_feedback(session_id,self.clock.now)


class Rig:
    def __init__(self, *, robot=None, config=None, allow_motion=True):
        self.clock=Clock()
        self.robot=robot if robot is not None else FakeRobot()
        self.config=synthetic_config() if config is None else config
        self.runner=NativeGamepadRunner(self.clock.now,self.robot,self.config,
            allow_motion=allow_motion,clock=self.clock,allow_synthetic_backend=True)
        self.runner.start(self.clock.now)

    @property
    def now(self):
        return self.clock.now

    def tick(self, buttons=0, ly=0.0, *, action=None, session_id=None,
             supported=False, connected=True, age=0.0, dt=.02,motor_power_off_confirmed=False):
        self.clock.now+=dt
        return self.runner.step(input_frame(self.now,buttons,ly,
            connected=connected,acquired_at=self.now-age),self.now,
            action=action,
            session_id=self.runner.session_id if action and session_id is None else session_id,
            supported=supported,motor_power_off_confirmed=motor_power_off_confirmed)

    def ready(self):
        return self.tick(action="resume")

    def jog(self,count=8):
        for _ in range(count):
            state=self.tick(BUTTONS["LB"],.7)
        return state

    def calls(self,kind):
        return [call for call in self.robot.calls if call["kind"]==kind]

    def begin_return(self):
        self.tick(action="pause")
        self.tick()
        return self.tick(action="return_origin")

    def settle(self):
        for _ in range(500):
            state=self.tick()
            if state["phase"]=="at_origin":
                return state
            if state["phase"] in ("fault","fault_holding","cleanup_fault"):
                raise AssertionError(state)
        raise AssertionError("fake measured return did not settle")


class NativeShapeSpy(NativePcanBackend):
    """Native-class call-order spy whose every actuator is a FakeRobot method.

    Deliberately skip the real backend constructor. No transport, DLL, or CAN
    object exists. This tests runner wiring, not production site admission.
    """
    def __init__(self,clock):
        self.robot=FakeRobot()
        self.clock=clock
        self.controller=self
        self.motors=self.robot.motors
        self.support_authorized=False
        self.authorized_session=None

    def read_feedback(self,session_id,now=None):
        return replace(self.robot.read_feedback(session_id,self.clock()),source="live_native")

    def authorize_motion(self,session_id,feedback,now=None,*,allow_motion=False):
        if allow_motion is not True or session_id!=feedback.session_id:
            raise AssertionError("runner must explicitly authorize this exact session")
        self.authorized_session=session_id
        self.robot.calls.append({"kind":"authorize_motion","session_id":session_id})

    def enable_all(self):
        if self.authorized_session is None:
            raise AssertionError("enable before native backend authorization")
        self.robot.enable_all()

    def authorize_supported_disable(self,session_id,feedback,now=None,*,supported=False):
        if supported is not True or session_id!=self.authorized_session or feedback.session_id!=session_id:
            raise AssertionError("fresh supported-disable authorization required")
        self.support_authorized=True
        self.robot.calls.append({"kind":"authorize_supported_disable","session_id":session_id})

    def disable_all(self):
        if not self.support_authorized:
            raise AssertionError("normal disable before backend support authorization")
        self.robot.disable_all()

    def close_bus(self):
        self.robot.close_bus()

    def acknowledge_external_power_off(self,session_id,*,supported=False,motor_power_off_confirmed=False):
        if session_id!=self.authorized_session or supported is not True or motor_power_off_confirmed is not True:
            raise AssertionError("wrong or incomplete explicit external power-off acknowledgement")
        self.robot.enabled=False
        self.robot.calls.append({"kind":"acknowledge_external_power_off","session_id":session_id})

    def close(self):
        self.robot.close()


class NativeRunnerTests(unittest.TestCase):
    def assert_owned_hold(self,rig,*,allow_disable_attempt=False):
        state=rig.runner.snapshot()
        self.assertTrue(state["transport"]["possibly_enabled"])
        self.assertFalse(state["closed"])
        if not allow_disable_attempt:
            self.assertFalse(rig.calls("disable_all"))
        self.assertFalse(rig.calls("close_bus"))
        last=rig.calls("mit")[-7:]
        self.assertEqual(len(last),7)
        self.assertTrue(all(call["values"]["vel"]==0 for call in last))
        self.assertEqual(tuple(call["values"]["pos"] for call in last),rig.runner.transport.last_targets)

    def test_read_only_start_and_input_never_send_or_enable(self):
        rig=Rig(allow_motion=False)
        self.assertEqual(rig.runner.planner.origin_rad,DEFAULT_FAKE_ORIGIN)
        rig.ready()
        rig.jog()
        rig.tick(BUTTONS["LB"]|BUTTONS["A"])
        rig.tick(action="return_origin")
        self.assertFalse(rig.calls("mit"))
        self.assertFalse(rig.calls("enable_all"))
        self.assertFalse(rig.calls("disable_all"))
        self.assertEqual(rig.runner.planner.origin_rad,DEFAULT_FAKE_ORIGIN)

    def test_file_motion_boolean_cannot_promote_readonly_session(self):
        config=synthetic_config()
        config["allow_motion"]=True
        config["motion_authorized"]=True
        rig=Rig(config=config,allow_motion=False)
        rig.ready();rig.jog()
        self.assertFalse(rig.runner.transport.armed)
        self.assertFalse(rig.calls("mit"))
        self.assertFalse(rig.calls("enable_all"))

    def test_runtime_motion_flag_cannot_enable_an_existing_observer(self):
        rig=Rig(allow_motion=False)
        try:
            rig.runner.allow_motion=True
        except (AttributeError,ValueError,PermissionError):
            pass
        rig.ready();rig.jog()
        self.assertFalse(rig.calls("enable_all"))
        self.assertFalse(rig.calls("mit"))

    def test_start_uses_same_session_fresh_nonzero_origin_and_exact_hold(self):
        rig=Rig()
        self.assertEqual(rig.runner.planner.origin_rad,DEFAULT_FAKE_ORIGIN)
        self.assertGreater(rig.runner.planner.origin_rad[2],.093)
        self.assertEqual(tuple(call["values"]["pos"] for call in rig.calls("mit")[:7]),DEFAULT_FAKE_ORIGIN)
        self.assertTrue(all(call["values"]["vel"]==0 for call in rig.calls("mit")))
        self.assertEqual(len(rig.calls("enable_all")),1)

    def test_new_sessions_capture_independent_origin_and_identifier(self):
        first=Rig()
        second_origin=(.4,-.3,.5,.2,-.1,.1,1.2)
        second=Rig(robot=FakeRobot(second_origin))
        self.assertNotEqual(first.runner.session_id,second.runner.session_id)
        self.assertEqual(second.runner.planner.origin_rad,second_origin)
        self.assertEqual(first.runner.planner.origin_rad,DEFAULT_FAKE_ORIGIN)

    def test_repeated_start_never_recaptures_or_enables_again(self):
        rig=Rig();rig.ready();rig.jog()
        with self.assertRaises((SessionError,ValueError,RuntimeError)):
            rig.runner.start(rig.now+.02)
        self.assertEqual(rig.runner.planner.origin_rad,DEFAULT_FAKE_ORIGIN)
        self.assertEqual(len(rig.calls("enable_all")),1)
        self.assert_owned_hold(rig)

    def test_synthetic_backend_requires_explicit_test_injection(self):
        with self.assertRaises((ValueError,TypeError,PermissionError)):
            NativeGamepadRunner(10.0,FakeRobot(),synthetic_config(),allow_motion=True)

    def test_current_site_candidate_remains_blocked_even_with_motion_flag(self):
        config=json.loads((ROOT/"site_config.candidate_20261030.json").read_text(encoding="utf-8"))
        robot=FakeRobot()
        try:
            rig=Rig(robot=robot,config=config)
        except (ValueError,PermissionError):
            pass  # Refusal before session startup is also a valid safe result.
        else:
            rig.ready();rig.jog()
        self.assertFalse(any(call["kind"] in ("enable_all","mit") for call in robot.calls))

    def test_j3_first_negative_input_is_not_reinterpreted_as_positive_absolute_reference(self):
        rig=Rig();rig.ready()
        for _ in range(2):
            rig.tick(BUTTONS["DPadRight"]);rig.tick()
        target=rig.runner.transport.last_targets[2]
        state=rig.tick(BUTTONS["LB"],-.8)
        self.assertIn(state["phase"],("fault","fault_holding"))
        self.assertEqual(rig.runner.transport.last_targets[2],target)
        self.assert_owned_hold(rig)

    def test_nonzero_motion_requires_lb_and_release_or_center_holds_immediately(self):
        rig=Rig();rig.ready()
        target=rig.runner.transport.last_targets
        rig.tick(0,.8)
        self.assertEqual(rig.runner.transport.last_targets,target)
        rig.jog()
        self.assertTrue(any(call["values"]["vel"]!=0 for call in rig.calls("mit")))
        rig.tick(BUTTONS["LB"],0)
        self.assertTrue(all(call["values"]["vel"]==0 for call in rig.calls("mit")[-7:]))
        target=rig.runner.transport.last_targets
        rig.tick(0,.8)
        self.assertEqual(rig.runner.transport.last_targets,target)
        self.assert_owned_hold(rig)

    def test_b_priority_latches_and_held_input_cannot_resume(self):
        rig=Rig();rig.ready();rig.jog()
        held=BUTTONS["LB"]|BUTTONS["B"]|BUTTONS["A"]
        rig.tick(held,.8)
        target=rig.runner.transport.last_targets
        for _ in range(3):rig.tick(BUTTONS["LB"],.8)
        self.assertEqual(rig.runner.transport.last_targets,target)
        state=rig.tick(BUTTONS["LB"],.8,action="resume")
        self.assertIsNotNone(state["action_error"])
        rig.tick()
        self.assertNotEqual(rig.runner.planner.phase,"ready")
        self.assertEqual(rig.ready()["phase"],"ready")
        self.assert_owned_hold(rig)

    def test_disconnect_and_stale_input_cancel_nonzero_output(self):
        for kind in ("disconnect","stale"):
            with self.subTest(kind=kind):
                rig=Rig();rig.ready();rig.jog()
                target=rig.runner.transport.last_targets
                rig.tick(connected=False) if kind=="disconnect" else rig.tick(age=.4)
                self.assertEqual(rig.runner.transport.last_targets,target)
                self.assert_owned_hold(rig)
                rig.tick(BUTTONS["LB"],.8)
                self.assertEqual(rig.runner.transport.last_targets,target)

    def test_reconnect_requires_neutral_and_new_explicit_resume(self):
        rig=Rig();rig.ready();rig.jog()
        rig.tick(connected=False)
        target=rig.runner.transport.last_targets
        rig.tick(BUTTONS["LB"],.8)
        rig.tick()
        self.assertEqual(rig.runner.transport.last_targets,target)
        self.assertNotEqual(rig.runner.planner.phase,"ready")
        self.assertEqual(rig.ready()["phase"],"ready")

    def test_foreign_or_unknown_action_fails_to_owned_hold(self):
        for action,session in (("resume","old"),("return_origin","old"),("confirm_support","old"),("home_zero",None)):
            with self.subTest(action=action,session=session):
                rig=Rig();rig.ready();rig.jog()
                state=rig.tick(BUTTONS["LB"],.8,action=action,session_id=session)
                self.assertIn(state["phase"],("fault","fault_holding"))
                self.assertTrue(all(value==0 for value in state["input"]["control"]["joint_velocity_rad_s"]))
                self.assert_owned_hold(rig)

    def test_motor_fault_is_latched_and_resume_does_not_clear_it(self):
        rig=Rig();rig.ready();rig.jog()
        rig.robot.fault_codes=(0,0,4,0,0,0,0)
        state=rig.tick()
        self.assertIn(state["phase"],("fault","fault_holding"))
        rig.robot.fault_codes=(0,)*7
        rig.tick(action="resume")
        self.assertIn(rig.runner.snapshot()["phase"],("fault","fault_holding"))
        self.assert_owned_hold(rig)

    def test_stale_axis_or_status_cannot_keep_jogging(self):
        for attribute in ("axis_age_s","status_age_s"):
            with self.subTest(attribute=attribute):
                rig=Rig();rig.ready();rig.jog()
                getattr(rig.robot,attribute)[4]=.5
                state=rig.tick(BUTTONS["LB"],.8)
                self.assertNotEqual(state["phase"],"ready")
                self.assert_owned_hold(rig)

    def test_partial_enable_preserves_owned_hold(self):
        robot=FakeRobot();robot.fail_enable=True
        rig=Rig(robot=robot)
        rig.tick()
        self.assert_owned_hold(rig)

    def test_partial_send_keeps_successful_axis_targets(self):
        rig=Rig();rig.ready();rig.jog()
        previous=rig.runner.transport.last_targets
        rig.robot.motors[4].fail_next_send=True
        rig.tick(BUTTONS["LB"],.8)
        self.assertGreater(rig.runner.transport.last_targets[0],previous[0])
        self.assertEqual(rig.runner.transport.last_targets[4:],previous[4:])
        self.assert_owned_hold(rig)

    def test_b_disconnect_stale_or_input_intervention_cancels_return(self):
        for kind in ("b","disconnect","stale","manual"):
            with self.subTest(kind=kind):
                rig=Rig();rig.ready();rig.jog(12)
                self.assertEqual(rig.begin_return()["phase"],"returning")
                rig.tick();target=rig.runner.transport.last_targets
                if kind=="b":rig.tick(BUTTONS["B"])
                elif kind=="disconnect":rig.tick(connected=False)
                elif kind=="stale":rig.tick(age=.4)
                else:rig.tick(BUTTONS["LB"],.8)
                self.assertIsNone(rig.runner.planner._return)
                self.assertEqual(rig.runner.transport.last_targets,target)
                self.assert_owned_hold(rig)

    def test_measured_return_then_support_is_required_before_disable_and_close(self):
        rig=Rig();rig.ready();rig.jog();rig.begin_return();rig.settle()
        self.assertFalse(rig.calls("disable_all"))
        state=rig.tick(action="confirm_support",supported=True)
        self.assertTrue(state["closed"])
        self.assertEqual(state["phase"],"disabled")
        names=[call["kind"] for call in rig.robot.calls]
        self.assertLess(names.index("disable_all"),names.index("close_bus"))
        self.assertLess(names.index("close_bus"),names.index("motor_close"))
        self.assertEqual(names[-1],"controller_close")
        self.assertEqual(rig.runner.planner.origin_rad,DEFAULT_FAKE_ORIGIN)

    def test_early_support_cannot_disable(self):
        rig=Rig();rig.ready();rig.jog()
        rig.tick(action="confirm_support",supported=True)
        self.assert_owned_hold(rig)

    def test_partial_disable_does_not_free_controller_and_future_ticks_hold(self):
        rig=Rig();rig.ready();rig.jog();rig.begin_return();rig.settle()
        rig.robot.fail_disable=True
        state=rig.tick(action="confirm_support",supported=True)
        self.assertEqual(state["phase"],"fault_holding")
        before=len(rig.calls("mit"))
        for _ in range(3):rig.tick()
        self.assertEqual(len(rig.calls("mit")),before+21)
        self.assert_owned_hold(rig,allow_disable_attempt=True)

    def test_disable_success_but_close_failure_is_not_closed_success(self):
        rig=Rig();rig.ready();rig.jog();rig.begin_return();rig.settle()
        rig.robot.fail_close=True
        state=rig.tick(action="confirm_support",supported=True)
        self.assertEqual(state["phase"],"cleanup_fault")
        self.assertFalse(state["closed"])
        self.assertFalse(state["transport"]["possibly_enabled"])
        self.assertFalse(rig.calls("motor_close"))

    def test_interrupt_or_feedback_exception_never_returns_disables_or_releases(self):
        for kind in ("interrupt","read_error"):
            with self.subTest(kind=kind):
                rig=Rig();rig.ready();rig.jog()
                if kind=="interrupt":rig.runner.interrupt("test-only EOF")
                else:
                    rig.robot.fail_read=True
                    rig.tick()
                self.assertIsNone(rig.runner.planner._return)
                self.assert_owned_hold(rig)

    def test_hold_send_failure_reports_unverified_hold_without_freeing(self):
        rig=Rig();rig.ready();rig.jog()
        rig.robot.motors[2].fail_send=True
        state=rig.tick(BUTTONS["B"])
        self.assertTrue(state["transport"]["hold_send_errors"])
        self.assertTrue(state["transport"]["possibly_enabled"])
        self.assertFalse(rig.calls("disable_all"))
        self.assertFalse(rig.calls("close_bus"))

    def test_confirmed_external_poweroff_closes_without_normal_disable_return_or_more_mit(self):
        rig=Rig();rig.ready();rig.jog()
        rig.runner.interrupt("injected feedback fault")
        before=len(rig.calls("mit"))
        origin=rig.runner.planner.origin_rad
        state=rig.tick(action="confirm_external_power_off",supported=True,motor_power_off_confirmed=True)
        self.assertTrue(state["closed"])
        self.assertEqual(state["exit_mode"],"external_poweroff_handover")
        self.assertEqual(state["phase"],"external_poweroff_handover")
        self.assertFalse(state["normal_disable_verified"])
        self.assertFalse(state["transport"]["possibly_enabled"])
        self.assertEqual(len(rig.calls("mit")),before)
        self.assertFalse(rig.calls("disable_all"))
        self.assertIsNone(rig.runner.planner._return)
        self.assertEqual(rig.runner.planner.origin_rad,origin)
        self.assertTrue(rig.calls("acknowledge_external_power_off"))
        self.assertEqual(rig.robot.calls[-1]["kind"],"controller_close")

    def test_external_poweroff_requires_both_explicit_true_facts_and_current_session(self):
        for supported,power,session in ((False,True,None),(True,False,None),(1,True,None),
                                         (True,1,None),(True,True,"old-session")):
            with self.subTest(supported=supported,power=power,session=session):
                rig=Rig();rig.ready();rig.jog();rig.runner.interrupt("test fault")
                state=rig.tick(action="confirm_external_power_off",supported=supported,
                    motor_power_off_confirmed=power,session_id=session)
                self.assertFalse(state["closed"])
                self.assertFalse(rig.calls("acknowledge_external_power_off"))
                self.assertFalse(rig.calls("close_bus"))
                self.assert_owned_hold(rig)

    def test_external_poweroff_cleanup_failure_is_not_closed_success_or_disable(self):
        rig=Rig();rig.ready();rig.jog();rig.runner.interrupt("test fault")
        before=len(rig.calls("mit"))
        rig.robot.fail_close=True
        state=rig.tick(action="confirm_external_power_off",supported=True,motor_power_off_confirmed=True)
        self.assertFalse(state["closed"])
        self.assertEqual(state["phase"],"cleanup_fault")
        self.assertEqual(state["exit_mode"],"external_poweroff_handover")
        self.assertFalse(state["normal_disable_verified"])
        self.assertFalse(state["transport"]["possibly_enabled"])
        self.assertFalse(rig.calls("disable_all"))
        self.assertEqual(len(rig.calls("mit")),before)

    def test_delayed_io_uses_post_receive_clock_for_feedback_and_planning(self):
        clock=Clock()
        robot=DelayedRobot(clock)
        runner=NativeGamepadRunner(clock.now,robot,synthetic_config(),allow_motion=True,
            clock=clock,allow_synthetic_backend=True)
        runner.start(clock.now)
        clock.now+=.02
        robot.read_delay=.04
        acquired=clock.now
        state=runner.step(input_frame(acquired),acquired,action="resume",session_id=runner.session_id)
        self.assertEqual(state["phase"],"ready",state)
        self.assertEqual(runner.planner.last_sample_time,clock.now)
        self.assertEqual(runner.last_feedback.acquired_at,clock.now)

    def test_input_that_expires_during_io_cannot_send_nonzero_jog(self):
        clock=Clock()
        robot=DelayedRobot(clock)
        runner=NativeGamepadRunner(clock.now,robot,synthetic_config(),allow_motion=True,
            clock=clock,allow_synthetic_backend=True)
        runner.start(clock.now)
        clock.now+=.02
        runner.step(input_frame(clock.now),clock.now,action="resume",session_id=runner.session_id)
        robot.read_delay=.3
        clock.now+=.02
        before=runner.transport.last_targets
        acquired=clock.now
        state=runner.step(input_frame(acquired,BUTTONS["LB"],.8),acquired)
        self.assertNotEqual(state["phase"],"ready")
        self.assertEqual(runner.transport.last_targets,before)
        self.assertTrue(all(call["values"]["vel"]==0 for call in robot.calls if call["kind"]=="mit") )
        self.assertFalse(any(call["kind"] in ("disable_all","close_bus") for call in robot.calls))


class NativeRunnerCliTests(unittest.TestCase):
    def test_default_cli_describes_without_constructing_backend_or_input_reader(self):
        with patch.object(live_runner,"NativePcanBackend",side_effect=AssertionError("no backend")), \
             patch.object(live_runner,"load_xinput_reader",side_effect=AssertionError("no real input"),create=True), \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([]),0)
        result=json.loads(output.getvalue())
        self.assertEqual(result["mode"],"description_only")
        self.assertFalse(result["can_opened"])

    def test_control_requires_allow_enable_even_if_other_admission_is_ready(self):
        backend=SimpleNamespace(admission_errors=lambda:[],open=lambda **_k:(_ for _ in ()).throw(AssertionError("must not open")))
        with patch.object(live_runner,"NativePcanBackend",return_value=backend), \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--control"]),2)
        result=json.loads(output.getvalue())
        self.assertEqual(result["mode"],"control_startup_blocked")
        self.assertFalse(result["can_opened"])
        self.assertFalse(result["motion_enable_requested"])

    def test_allow_enable_cannot_override_uncommissioned_site_before_can_open(self):
        backend=SimpleNamespace(admission_errors=lambda:["native bounds unverified"],open=lambda **_k:(_ for _ in ()).throw(AssertionError("must not open")))
        with patch.object(live_runner,"NativePcanBackend",return_value=backend), \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--control","--allow-enable"]),2)
        result=json.loads(output.getvalue())
        self.assertFalse(result["can_opened"])
        self.assertIn("native bounds unverified",result["admission_errors"])

    def test_observe_with_enable_option_is_rejected_before_open(self):
        backend=SimpleNamespace(open=lambda **_k:(_ for _ in ()).throw(AssertionError("must not open")))
        with patch.object(live_runner,"NativePcanBackend",return_value=backend), \
             patch("sys.stderr",io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["--observe","--allow-enable"])
        self.assertEqual(raised.exception.code,2)


class NativeAuthorizationWiringTests(unittest.TestCase):
    def test_native_backend_authorized_before_enable_and_support_before_normal_disable(self):
        clock=Clock()
        backend=NativeShapeSpy(clock)
        config=synthetic_config()
        # These are generated numbers for a fake actuator spy, not a site file.
        runner=NativeGamepadRunner(clock(),backend,config,allow_motion=True,clock=clock)
        runner.start(clock())
        def tick(*,action=None,supported=False,buttons=0,ly=0):
            clock.now+=.02
            return runner.step(input_frame(clock(),buttons,ly),clock(),action=action,
                session_id=runner.session_id if action else None,supported=supported)
        tick(action="resume")
        for _ in range(8):tick(buttons=BUTTONS["LB"],ly=.7)
        tick(action="pause");tick();tick(action="return_origin")
        for _ in range(500):
            if tick()["phase"]=="at_origin":break
        self.assertEqual(runner.planner.phase,"at_origin")
        self.assertFalse(backend.support_authorized)
        self.assertTrue(tick(action="confirm_support",supported=True)["closed"])
        kinds=[call["kind"] for call in backend.robot.calls]
        self.assertLess(kinds.index("authorize_motion"),kinds.index("mit"))
        self.assertLess(kinds.index("authorize_motion"),kinds.index("enable_all"))
        self.assertLess(kinds.index("authorize_supported_disable"),kinds.index("disable_all"))
        self.assertLess(kinds.index("disable_all"),kinds.index("close_bus"))

    def test_native_backend_external_poweroff_acknowledged_before_resource_close(self):
        clock=Clock();backend=NativeShapeSpy(clock)
        runner=NativeGamepadRunner(clock(),backend,synthetic_config(),allow_motion=True,clock=clock)
        runner.start(clock());runner.interrupt("test owned failure")
        before=sum(call["kind"]=="mit" for call in backend.robot.calls)
        state=runner.step(input_frame(clock()),action="confirm_external_power_off",
            session_id=runner.session_id,supported=True,motor_power_off_confirmed=True)
        self.assertTrue(state["closed"])
        kinds=[call["kind"] for call in backend.robot.calls]
        self.assertLess(kinds.index("acknowledge_external_power_off"),kinds.index("close_bus"))
        self.assertEqual(sum(kind=="mit" for kind in kinds),before)
        self.assertNotIn("disable_all",kinds)
        self.assertFalse(state["normal_disable_verified"])


class FileCommandTests(unittest.TestCase):
    def command(self,value,seen=None):
        with tempfile.TemporaryDirectory(dir=ROOT/"reports") as folder:
            path=Path(folder)/"command.json"
            path.write_text(json.dumps(value),encoding="utf-8")
            return live_runner.poll_command(path,"current-session",set() if seen is None else seen)

    def test_missing_file_has_no_action_or_implied_support(self):
        self.assertEqual(live_runner.poll_command(ROOT/"reports"/"missing-never-issued-command.json","current-session",set()),(None,False))

    def test_old_missing_or_wrong_session_never_accepted(self):
        for value in ({"command_id":"1","action":"resume"},
                      {"session_id":"old","command_id":"1","action":"return_origin"},
                      ["current-session","1","resume"]):
            with self.subTest(value=value),self.assertRaises(SessionError):
                self.command(value)

    def test_nonempty_string_command_id_is_required(self):
        for command_id in (None,"",1,True):
            with self.subTest(command_id=command_id),self.assertRaises(SessionError):
                self.command({"session_id":"current-session","command_id":command_id,"action":"pause"})

    def test_command_id_replay_is_ignored(self):
        seen=set()
        value={"session_id":"current-session","command_id":"one","action":"resume"}
        self.assertEqual(self.command(value,seen),("resume",False))
        value["action"]="return_origin"
        self.assertEqual(self.command(value,seen),(None,False))

    def test_unknown_action_cannot_become_home_zero(self):
        with self.assertRaises(SessionError):
            self.command({"session_id":"current-session","command_id":"1","action":"home_zero"})

    def test_support_requires_literal_true_not_truthy_json_values(self):
        for supported in (1,"true","yes",None,False):
            with self.subTest(supported=supported):
                action,approved=self.command({"session_id":"current-session","command_id":"1",
                    "action":"confirm_support","supported":supported})
                self.assertEqual(action,"confirm_support")
                self.assertFalse(approved)
        self.assertEqual(self.command({"session_id":"current-session","command_id":"1",
            "action":"confirm_support","supported":True}),("confirm_support",True))

    def test_external_poweroff_command_needs_both_facts_before_it_can_reach_step(self):
        base={"session_id":"current-session","command_id":"power-1","action":"confirm_external_power_off"}
        for fields in ({},{"supported":True},{"motor_power_off_confirmed":True},
                       {"supported":1,"motor_power_off_confirmed":True},
                       {"supported":True,"motor_power_off_confirmed":"true"}):
            with self.subTest(fields=fields),self.assertRaises(SessionError):
                self.command({**base,**fields})
        seen=set()
        full={**base,"supported":True,"motor_power_off_confirmed":True}
        self.assertEqual(self.command(full,seen),("confirm_external_power_off",True))
        self.assertEqual(self.command(full,seen),(None,False))


class LoopSpy:
    """Bounded CLI control-flow spy: there are no actuators or hardware."""
    def __init__(self,*,owned=True,read_only=False,backend=None):
        self.session_id="unit-loop-session"
        self.transport=SimpleNamespace(possibly_enabled=owned)
        self.closed=False
        self.allow_motion=not read_only
        self.last_error=None
        self.actions=[]
        self.interrupts=[]
        self.backend=backend
        self.observer_cleanup_error=None

    def snapshot(self):
        return {"phase":"external_poweroff_handover" if self.closed else "fault" if self.last_error else "paused",
            "last_error":self.last_error,"closed":self.closed,"exit_mode":"external_poweroff_handover" if self.closed else None,
            "normal_disable_verified":False}

    def start(self,*_args):
        return self.snapshot()

    def interrupt(self,reason):
        self.interrupts.append(reason)
        self.last_error=str(reason)
        return self.snapshot()

    def step(self,frame,*,action=None,session_id=None,supported=False,motor_power_off_confirmed=False):
        self.actions.append((action,session_id,supported,motor_power_off_confirmed))
        if len(self.actions)>5:
            raise SystemExit("test loop failed to process bounded handover")
        if action=="confirm_external_power_off":
            if session_id!=self.session_id or supported is not True or motor_power_off_confirmed is not True:
                raise AssertionError("CLI handed unverified power-off facts to runner")
            self.transport.possibly_enabled=False
            self.closed=True
            if self.backend is not None:
                self.backend.possibly_enabled=False
                self.backend.opened=False
        return self.snapshot()

    def close_observer(self):
        if self.observer_cleanup_error:
            raise self.observer_cleanup_error
        self.closed=True


class CliOwnershipLoopTests(unittest.TestCase):
    def run_loop(self,runner,commands,*,reader=None,save_effect=None,expired=True,sleep_effect=None):
        trace=[]
        with patch.object(live_runner,"poll_command",side_effect=commands), \
             patch.object(live_runner,"save_atomic",side_effect=save_effect) as save, \
             patch.object(live_runner.time,"sleep",side_effect=sleep_effect or ([None]*5+[SystemExit("loop exceeded test bound")])), \
             patch.object(live_runner.time,"monotonic",return_value=10.0 if expired else 0.0):
            result=live_runner.run_session_loop(runner,reader,Path("command"),Path("status"),0,1,set(),trace)
        return result,trace,save.call_count

    def test_duration_interrupts_once_and_never_overwrites_later_handover(self):
        runner=LoopSpy()
        (fatal,triggered),trace,_=self.run_loop(runner,[(None,False),(None,False),("confirm_external_power_off",True)])
        self.assertTrue(triggered)
        self.assertEqual(len(runner.interrupts),1)
        self.assertEqual([action[0] for action in runner.actions],[None,None,"confirm_external_power_off"])
        self.assertTrue(runner.closed)
        self.assertFalse(runner.transport.possibly_enabled)
        self.assertEqual(trace[-1]["exit_mode"],"external_poweroff_handover")

    def test_invalid_file_keeps_processing_same_channel_until_explicit_handover(self):
        runner=LoopSpy()
        self.run_loop(runner,[SessionError("old session"),("confirm_external_power_off",True)],expired=False)
        self.assertIn("old session",runner.interrupts[0])
        self.assertTrue(runner.closed)

    def test_reader_failure_cannot_block_external_poweroff_command(self):
        runner=LoopSpy()
        reader=SimpleNamespace(read=lambda:(_ for _ in ()).throw(RuntimeError("input unavailable")))
        self.run_loop(runner,[(None,False),("confirm_external_power_off",True)],reader=reader,expired=False)
        self.assertIn("input unavailable",runner.interrupts[0])
        self.assertTrue(runner.closed)

    def test_status_save_failure_keeps_commands_and_records_hold(self):
        runner=LoopSpy()
        self.run_loop(runner,[(None,False),("confirm_external_power_off",True)],
            save_effect=[OSError("status blocked"),None],expired=False)
        self.assertIn("status blocked",runner.interrupts[0])
        self.assertTrue(runner.closed)

    def test_keyboard_interrupt_does_not_exit_owned_loop(self):
        runner=LoopSpy()
        self.run_loop(runner,[("confirm_external_power_off",True)],expired=False,
            sleep_effect=[KeyboardInterrupt(),None,SystemExit("loop exceeded test bound")])
        self.assertIn("KeyboardInterrupt",runner.interrupts[0])
        self.assertTrue(runner.closed)

    def test_cleanup_failure_is_archived_separately_from_original_fatal_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"reports") as folder:
            output=Path(folder)/"result.json"
            backend=SimpleNamespace(opened=False,possibly_enabled=False,admission_errors=lambda:[],
                diagnostics=lambda:{"fake_backend":True})
            backend.open=lambda **_kwargs:setattr(backend,"opened",True)
            runner=LoopSpy(owned=False,read_only=True)
            runner.observer_cleanup_error=RuntimeError("independent cleanup failure")
            def fail_start(*_args):raise RuntimeError("original startup failure")
            runner.start=fail_start
            with patch.object(live_runner,"ROOT",Path(folder)), \
                 patch.object(live_runner,"NativePcanBackend",return_value=backend), \
                 patch.object(live_runner,"NativeGamepadRunner",return_value=runner), \
                 redirect_stdout(io.StringIO()):
                code=main(["--observe","--config",str(ROOT/"site_config.candidate_20261030.json"),"--out",str(output)])
            self.assertEqual(code,2)
            result=json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("original startup failure",result["fatal_error"])
            self.assertIn("independent cleanup failure",result["cleanup_error"])
            self.assertFalse(result["final"]["normal_disable_verified"])

    def test_outer_startup_status_failure_uses_same_command_polling_fallback(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"reports") as folder:
            backend=SimpleNamespace(opened=False,possibly_enabled=True,admission_errors=lambda:[],
                diagnostics=lambda:{"fake_backend":True})
            backend.open=lambda **_kwargs:setattr(backend,"opened",True)
            runner=LoopSpy(backend=backend)
            calls=[]
            def save(path,value):
                calls.append(str(path))
                if len(calls)==1:raise OSError("startup status blocked")
            with patch.object(live_runner,"ROOT",Path(folder)), \
                 patch.object(live_runner,"NativePcanBackend",return_value=backend), \
                 patch.object(live_runner,"NativeGamepadRunner",return_value=runner), \
                 patch.object(live_runner,"save_atomic",side_effect=save), \
                 patch.object(live_runner,"poll_command",return_value=("confirm_external_power_off",True)) as poll, \
                 patch.object(live_runner.time,"monotonic",return_value=0.0), \
                 patch.object(live_runner.time,"sleep",side_effect=[None,SystemExit("loop exceeded test bound")]), \
                 redirect_stdout(io.StringIO()):
                code=main(["--control","--allow-enable","--config",str(ROOT/"site_config.candidate_20261030.json")])
            self.assertEqual(code,2)
            self.assertTrue(runner.closed)
            self.assertTrue(poll.called)
            self.assertIn("startup status blocked",runner.interrupts[0])


if __name__ == "__main__":
    unittest.main()
