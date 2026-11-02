"""Whole-loop regressions using only a labelled synthetic plant."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch
from fake_robot import FakeRobot, synthetic_config
from live_runner import NativeGamepadRunner, Frame, AXES, BUTTONS
from tony_engine import TonyEngine
import tony_teleop

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now=10.
        self.robot=FakeRobot()
        self.config=synthetic_config()
        self.runner=NativeGamepadRunner(self.now,self.robot,self.config,allow_motion=True,
            clock=lambda:self.now,allow_synthetic_backend=True,input_engine=TonyEngine(self.config))
        self.runner.start()
        self.tick();self.tick(BUTTONS['Start']);self.tick()

    def tick(self,buttons=0,**axes):
        self.now+=.02
        values=dict.fromkeys(AXES,0.);values.update(axes)
        return self.runner.step(Frame(True,buttons=buttons,axes=values,acquired_at=self.now),self.now)

    def test_seven_axis_intent_reaches_actual_fake_transport(self):
        for _ in range(6):
            state=self.tick(BUTTONS['LB']|BUTTONS['DPadUp']|BUTTONS['DPadLeft'],LX=.6,LY=.6,RX=.6,RY=.6,RT=.4)
        self.assertIsNone(state['last_error'])
        self.assertTrue(all(a>b for a,b in zip(self.robot.targets,self.runner.planner.origin_rad)))
        self.assertGreater(self.robot.command_velocities[6],0)
        self.assertLess(self.robot.command_velocities[6],.02)

    def test_b_latches_and_y_cannot_resume_it(self):
        self.tick(BUTTONS['LB'],LX=.7)
        self.tick(BUTTONS['B']);self.tick();state=self.tick(BUTTONS['Y'])
        self.assertEqual(state['planner_phase'],'paused')
        self.assertTrue(self.runner.engine.paused)
        self.tick();state=self.tick(BUTTONS['Start'])
        self.assertEqual(state['planner_phase'],'ready')

    def test_y_returns_immutable_session_origin_then_requires_support(self):
        origin=self.runner.planner.origin_rad
        for _ in range(10):self.tick(BUTTONS['LB'],LX=.7)
        self.tick();state=self.tick(BUTTONS['Y'])
        self.assertEqual(state['planner_phase'],'returning')
        for _ in range(300):
            state=self.tick()
            if state['planner_phase']=='at_origin':break
        self.assertEqual(state['planner_phase'],'at_origin')
        self.assertTrue(self.runner.transport.possibly_enabled)
        self.assertEqual(self.runner.planner.origin_rad,origin)
        self.now+=.02
        state=self.runner.step(Frame(True,acquired_at=self.now),self.now,action='confirm_support',
            session_id=self.runner.session_id,supported=True)
        self.assertTrue(state['normal_disable_verified'])
        self.assertTrue(state['closed'])

    def test_default_and_blocked_control_never_open_can(self):
        with patch('live_runner.NativePcanBackend.open',side_effect=AssertionError('CAN opened')):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(tony_teleop.main([]),0)
                self.assertEqual(tony_teleop.main(['--control','--allow-enable','--config','site_config.candidate_20261030.json']),2)

    def test_scripted_dry_run_does_not_construct_can_backend(self):
        with patch('live_runner.NativePcanBackend',side_effect=AssertionError('backend constructed')):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(tony_teleop.main(['--dry-run','--scripted-fixture','--duration','.8']),0)

if __name__=='__main__':unittest.main()
