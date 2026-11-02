"""Tony profile input and injected-loop regressions; no DLL, CAN or XInput.

All numeric site values below are intent/plant fixtures, never physical limits.
The tests keep independent multi-axis intent separate from motion admission.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"input_core"))
from control_core import AXES, BUTTONS, Frame
from tony_engine import TonyEngine


SPEEDS_DEG=(30.,20.,20.,30.,40.,60.,60.)
DPAD=(BUTTONS["DPadUp"]|BUTTONS["DPadDown"]|BUTTONS["DPadLeft"]|BUTTONS["DPadRight"])


def curve(value):
    if abs(value)<=.12:
        return 0.
    shifted=math.copysign((abs(value)-.12)/.88,value)
    return shifted**3


def frame(now,buttons=0,*,connected=True,age=0.,**axes):
    values=dict.fromkeys(AXES,0.)
    values.update(axes)
    return Frame(connected=connected,packet=1,buttons=buttons,axes=values,acquired_at=now-age)


class Rig:
    def __init__(self,config=None):
        self.now=10.
        self.engine=TonyEngine(site_config=config)

    def tick(self,buttons=0,*,dt=.02,connected=True,age=0.,**axes):
        self.now+=dt
        return self.engine.update(frame(self.now,buttons,connected=connected,age=age,**axes),self.now)

    def ready(self):
        self.tick()
        self.tick(BUTTONS["Start"])
        self.engine.consume_action()
        return self.tick()


class TonyEngineTests(unittest.TestCase):
    def assert_zero(self,state):
        ctrl=state["control"]
        self.assertEqual(tuple(ctrl["joint_velocity_rad_s"]),(0.,)*6)
        self.assertEqual(ctrl["gripper_velocity_rad_s"],0.)
        self.assertEqual(ctrl["gripper_intent"],"hold")

    def test_profile_has_no_fake_position_backend_in_real_input_snapshot(self):
        rig=Rig();state=rig.ready()
        self.assertNotIn("fake_backend",state)
        self.assertIn("neutral_confirmed",state["control"])
        self.assertTrue(rig.engine.is_neutral(state))

    def test_lb_required_for_all_six_joints_and_analog_gripper(self):
        rig=Rig();rig.ready()
        state=rig.tick(DPAD,LX=.8,LY=.7,RX=.6,RY=.9,RT=.8)
        self.assert_zero(state)
        self.assertIsNone(rig.engine.consume_action())

    def test_multi_axis_and_trigger_mapping_match_upstream_cubic_profile(self):
        rig=Rig();rig.ready()
        state=rig.tick(BUTTONS["LB"]|BUTTONS["DPadUp"]|BUTTONS["DPadLeft"],
                       LX=.7,LY=.8,RX=-.9,RY=.75,RT=.65)
        expected=(curve(.7),curve(.8),curve(.75),curve(-.9),1.,1.,curve(.65))
        ctrl=state["control"]
        for index,(intent,degrees) in enumerate(zip(expected,SPEEDS_DEG)):
            velocity=(ctrl["joint_velocity_rad_s"][index] if index<6 else ctrl["gripper_velocity_rad_s"])
            self.assertAlmostEqual(velocity,intent*math.radians(degrees)*.25,places=11)
        self.assertGreater(ctrl["joint_velocity_rad_s"][2],0.)
        self.assertEqual(ctrl["gripper_intent"],"open")

    def test_site_limits_cap_each_profile_velocity_including_gripper(self):
        config={"joints":[{"max_speed_rad_s":v} for v in (.01,.02,.03,.04,.05,.06,.07)]}
        rig=Rig(config);rig.ready()
        state=rig.tick(BUTTONS["LB"]|BUTTONS["DPadUp"]|BUTTONS["DPadLeft"],
                       LX=1.,LY=1.,RX=1.,RY=1.,RT=1.)
        velocities=tuple(state["control"]["joint_velocity_rad_s"])+(state["control"]["gripper_velocity_rad_s"],)
        self.assertEqual(tuple(round(value,10) for value in velocities),(.01,.02,.03,.04,.05,.06,.07))
        self.assertEqual(config["joints"][0]["max_speed_rad_s"],.01)

    def test_deadzone_and_release_zero_individual_controls(self):
        rig=Rig();rig.ready()
        self.assert_zero(rig.tick(BUTTONS["LB"],LX=.12,LY=-.12,RX=.05,RY=-.05,RT=.1))
        active=rig.tick(BUTTONS["LB"],LX=.8,RY=.7,RT=.6)
        self.assertNotEqual(active["control"]["joint_velocity_rad_s"][0],0.)
        held=rig.tick(BUTTONS["LB"],RY=.7)
        self.assertEqual(held["control"]["joint_velocity_rad_s"][0],0.)
        self.assertGreater(held["control"]["joint_velocity_rad_s"][2],0.)
        self.assertEqual(held["control"]["gripper_velocity_rad_s"],0.)
        self.assert_zero(rig.tick())

    def test_contradictory_dpad_and_triggers_are_zero(self):
        rig=Rig();rig.ready()
        state=rig.tick(BUTTONS["LB"]|DPAD,LT=.35,RT=.8)
        self.assert_zero(state)

    def test_positive_and_negative_trigger_intents_retain_analog_magnitude(self):
        rig=Rig();rig.ready()
        opening=rig.tick(BUTTONS["LB"],RT=.5)["control"]
        closing=rig.tick(BUTTONS["LB"],LT=.5)["control"]
        full=rig.tick(BUTTONS["LB"],RT=1.)["control"]
        self.assertGreater(opening["gripper_velocity_rad_s"],0.)
        self.assertLess(closing["gripper_velocity_rad_s"],0.)
        self.assertEqual(closing["gripper_intent"],"close")
        self.assertAlmostEqual(opening["gripper_velocity_rad_s"],-closing["gripper_velocity_rad_s"])
        self.assertLess(opening["gripper_velocity_rad_s"],full["gripper_velocity_rad_s"]*.5)

    def test_fresh_b_pauses_even_when_axes_are_invalid(self):
        rig=Rig();rig.ready()
        rig.tick(BUTTONS["LB"],LX=.8)
        state=rig.tick(BUTTONS["B"]|BUTTONS["Y"]|BUTTONS["Start"],LX=math.nan)
        self.assertTrue(rig.engine.paused)
        self.assert_zero(state)
        self.assertIsNone(rig.engine.consume_action())

    def test_b_remains_latched_until_new_neutral_start_edge(self):
        rig=Rig();rig.ready()
        rig.tick(BUTTONS["B"])
        self.assert_zero(rig.tick(BUTTONS["LB"],LX=1.))
        self.assertTrue(rig.engine.paused)
        self.assertIsNone(rig.engine.consume_action())
        rig.tick()
        resumed=rig.tick(BUTTONS["Start"])
        self.assertFalse(rig.engine.paused)
        self.assertEqual(rig.engine.consume_action(),"resume")
        self.assertIsNone(rig.engine.consume_action())
        self.assert_zero(resumed)
        rig.tick()
        self.assertGreater(rig.tick(BUTTONS["LB"],LX=.7)["control"]["joint_velocity_rad_s"][0],0.)

    def test_start_held_with_lb_or_motion_cannot_resume(self):
        for buttons,axes in ((BUTTONS["Start"]|BUTTONS["LB"],{}),
                             (BUTTONS["Start"],{"LY":.5}),
                             (BUTTONS["Start"]|BUTTONS["DPadUp"],{}),
                             (BUTTONS["Start"],{"RT":.4})):
            with self.subTest(buttons=buttons,axes=axes):
                rig=Rig();rig.ready();rig.tick(BUTTONS["B"]);rig.tick()
                self.assert_zero(rig.tick(buttons,**axes))
                self.assertTrue(rig.engine.paused)
                self.assertIsNone(rig.engine.consume_action())

    def test_b_priority_discards_simultaneous_y_start_and_speed(self):
        rig=Rig();rig.ready()
        before=rig.engine.snapshot(rig.now)["control"]["speed_level"]
        state=rig.tick(BUTTONS["B"]|BUTTONS["Y"]|BUTTONS["Start"]|BUTTONS["RB"])
        self.assert_zero(state)
        self.assertTrue(rig.engine.paused)
        self.assertEqual(state["control"]["speed_level"],before)
        self.assertIsNone(rig.engine.consume_action())

    def test_operator_action_conflict_pauses_without_selecting_one(self):
        rig=Rig();rig.ready()
        state=rig.tick(BUTTONS["Y"]|BUTTONS["RB"])
        self.assert_zero(state)
        self.assertTrue(rig.engine.paused)
        self.assertIsNone(rig.engine.consume_action())

    def test_y_emits_return_request_once_and_does_not_expose_zero_target(self):
        rig=Rig();rig.ready()
        state=rig.tick(BUTTONS["Y"])
        self.assertEqual(rig.engine.consume_action(),"return_origin")
        self.assertIsNone(rig.engine.consume_action())
        self.assert_zero(state)
        self.assertNotIn("targets_rad",state["control"])
        rig.tick(BUTTONS["Y"])
        self.assertIsNone(rig.engine.consume_action())
        rig.tick();rig.tick(BUTTONS["Y"])
        self.assertEqual(rig.engine.consume_action(),"return_origin")

    def test_y_requires_lb_release_and_full_motion_neutral(self):
        for buttons,axes in ((BUTTONS["Y"]|BUTTONS["LB"],{}),
                             (BUTTONS["Y"]|BUTTONS["DPadLeft"],{}),
                             (BUTTONS["Y"],{"RX":.8}),
                             (BUTTONS["Y"],{"LT":.4})):
            with self.subTest(buttons=buttons,axes=axes):
                rig=Rig();rig.ready()
                rig.tick(buttons,**axes)
                self.assertIsNone(rig.engine.consume_action())

    def test_y_does_not_unlock_a_b_latched_pause(self):
        rig=Rig();rig.ready();rig.tick(BUTTONS["B"]);rig.tick()
        rig.tick(BUTTONS["Y"])
        self.assertTrue(rig.engine.paused)
        self.assertIsNone(rig.engine.consume_action())

    def test_speed_cycles_only_on_fresh_neutral_released_lb_edges(self):
        rig=Rig();state=rig.ready()
        self.assertEqual(state["control"]["speed_scale"],.25)
        for expected in (.5,1.,.25):
            state=rig.tick(BUTTONS["RB"])
            self.assertEqual(state["control"]["speed_scale"],expected)
            rig.tick(BUTTONS["RB"])
            self.assertEqual(rig.engine.snapshot(rig.now)["control"]["speed_scale"],expected)
            rig.tick()

    def test_speed_edge_while_moving_or_with_dpad_is_discarded_not_deferred(self):
        rig=Rig();rig.ready()
        for buttons,axes in ((BUTTONS["LB"]|BUTTONS["RB"],{"LX":.7}),
                             (BUTTONS["RB"]|BUTTONS["DPadDown"],{}),
                             (BUTTONS["RB"],{"RT":.4})):
            rig.tick()
            state=rig.tick(buttons,**axes)
            self.assertEqual(state["control"]["speed_scale"],.25)
            rig.tick(BUTTONS["RB"])
            self.assertEqual(rig.engine.snapshot(rig.now)["control"]["speed_scale"],.25)

    def test_initial_action_keys_held_before_connection_are_not_operator_edges(self):
        for key in ("Y","Start","RB"):
            with self.subTest(key=key):
                rig=Rig()
                state=rig.tick(BUTTONS[key])
                self.assert_zero(state)
                self.assertIsNone(rig.engine.consume_action())
                self.assertEqual(state["control"]["speed_scale"],.25)
                rig.tick(BUTTONS[key])
                self.assertIsNone(rig.engine.consume_action())
                rig.tick();rig.tick(BUTTONS[key])
                if key=="RB":
                    self.assertEqual(rig.engine.snapshot(rig.now)["control"]["speed_scale"],.5)
                else:
                    self.assertEqual(rig.engine.consume_action(),"resume" if key=="Start" else "return_origin")

    def test_startup_or_reconnect_dpad_is_not_neutral_for_rearming(self):
        for reconnect in (False,True):
            with self.subTest(reconnect=reconnect):
                rig=Rig()
                if reconnect:
                    rig.ready();rig.tick(connected=False)
                state=rig.tick(BUTTONS["DPadUp"])
                self.assertFalse(state["control"]["neutral_confirmed"])
                self.assert_zero(rig.tick(BUTTONS["DPadUp"]|BUTTONS["LB"]))
                rig.tick()
                active=rig.tick(BUTTONS["DPadUp"]|BUTTONS["LB"])
                self.assertGreater(active["control"]["joint_velocity_rad_s"][4],0.)

    def test_disconnect_and_stale_or_future_input_zero_every_axis(self):
        for changes in ({"connected":False},{"age":.3},{"age":-.1}):
            with self.subTest(changes=changes):
                rig=Rig();rig.ready();rig.tick(BUTTONS["LB"],LX=.8,RY=.8,RT=.8)
                self.assert_zero(rig.tick(BUTTONS["LB"],LX=.8,RY=.8,RT=.8,**changes))
                self.assertIsNone(rig.engine.consume_action())

    def test_reconnect_held_input_and_held_action_needs_release_and_new_edge(self):
        rig=Rig();rig.ready();rig.tick(connected=False)
        state=rig.tick(BUTTONS["LB"]|BUTTONS["Y"],LX=1.)
        self.assert_zero(state)
        self.assertIsNone(rig.engine.consume_action())
        self.assert_zero(rig.tick(BUTTONS["LB"]|BUTTONS["Y"],LX=1.))
        rig.tick()
        self.assertGreater(rig.tick(BUTTONS["LB"],LX=1.)["control"]["joint_velocity_rad_s"][0],0.)

    def test_gap_discards_old_action_and_requires_neutral(self):
        rig=Rig();rig.ready();rig.tick(BUTTONS["LB"],LX=.8)
        state=rig.tick(BUTTONS["LB"]|BUTTONS["Y"],dt=.4,LX=.8)
        self.assert_zero(state)
        self.assertIsNone(rig.engine.consume_action())
        self.assert_zero(rig.tick(BUTTONS["LB"],LX=.8))
        rig.tick()
        self.assertGreater(rig.tick(BUTTONS["LB"],LX=.8)["control"]["joint_velocity_rad_s"][0],0.)

    def test_invalid_axes_cannot_drive_profile_or_operator_actions(self):
        for value in (math.nan,math.inf,-math.inf,1.1):
            with self.subTest(value=value):
                rig=Rig();rig.ready()
                state=rig.tick(BUTTONS["LB"]|BUTTONS["Y"],LX=value,RT=.8)
                self.assert_zero(state)
                self.assertIsNone(rig.engine.consume_action())

    def test_snapshot_expiry_does_not_replay_nonzero_velocity_or_pending_action(self):
        rig=Rig();rig.ready();rig.tick(BUTTONS["LB"],LX=.8,RT=.8)
        self.assert_zero(rig.engine.snapshot(rig.now+.3))
        rig.tick();rig.tick(BUTTONS["Y"])
        rig.engine.snapshot(rig.now+.3)
        self.assertIsNone(rig.engine.consume_action())

    def test_explicit_pause_keeps_profile_zero_after_prior_nonzero_input(self):
        rig=Rig();rig.ready();rig.tick(BUTTONS["LB"],LX=.8,RY=.8,RT=.8)
        rig.engine.set_paused(True)
        state=rig.engine.snapshot(rig.now)
        self.assert_zero(state)
        self.assertTrue(state["control"]["paused"])


if __name__=="__main__":
    unittest.main()
