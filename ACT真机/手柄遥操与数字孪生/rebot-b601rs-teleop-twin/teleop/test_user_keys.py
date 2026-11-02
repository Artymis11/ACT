"""Requested X/A(3s)/B/Y contract; fake plant and input only."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch
from enable_latch import EnableLatch
from fake_robot import FakeRobot
from live_runner import NativeGamepadRunner,Frame,AXES,BUTTONS
from tony_engine import TonyEngine
from tony_github_config import github_config
import tony_teleop

class KeyTests(unittest.TestCase):
    def setUp(self):
        self.now=10.
        self.config=github_config()
        self.robot=FakeRobot()
        self.runner=NativeGamepadRunner(self.now,self.robot,self.config,allow_motion=True,clock=lambda:self.now,
            allow_synthetic_backend=True,input_engine=TonyEngine(self.config))
        self.runner.start();self.tick()

    def frame(self,buttons=0,**axes):
        values=dict.fromkeys(AXES,0.);values.update(axes)
        return Frame(True,buttons=buttons,axes=values,acquired_at=self.now)

    def tick(self,buttons=0,**axes):
        self.now+=.02
        return self.runner.step(self.frame(buttons,**axes),self.now)

    def test_x_resumes_and_legacy_start_does_not(self):
        self.runner.engine.set_paused(True)
        self.runner.planner.pause(self.runner.session_id)
        self.tick();state=self.tick(BUTTONS['Start'])
        self.assertEqual(state['phase'],'paused')
        self.tick();state=self.tick(BUTTONS['X'])
        self.assertEqual(state['phase'],'ready',state)

    def test_a_short_press_does_not_stop_and_three_seconds_returns_origin_then_disables(self):
        origin=self.runner.planner.origin_rad
        for _ in range(15):self.tick(BUTTONS['LB'],LX=.6)
        self.tick()
        for _ in range(100):state=self.tick(BUTTONS['A'])
        self.assertIsNone(state['shutdown_reason'])
        self.tick()
        for _ in range(149):state=self.tick(BUTTONS['A'])
        self.assertIsNone(state['shutdown_reason'])
        for _ in range(3):state=self.tick(BUTTONS['A'])
        self.assertEqual(state['shutdown_destination'],'origin')
        self.assertEqual(state['phase'],'returning')
        self.assertTrue(self.robot.enabled)
        for _ in range(400):
            state=self.tick()
            if state['closed']:break
        self.assertTrue(state['closed'],state)
        self.assertEqual(state['exit_mode'],'github_origin_disable')
        self.assertEqual(tuple(self.robot.targets),origin)
        self.assertFalse(self.robot.enabled)

    def test_a_counts_while_lb_held_and_suppresses_jog_during_hold(self):
        for _ in range(100):state=self.tick(BUTTONS['A']|BUTTONS['LB'],LX=.6)
        self.assertIsNone(state['shutdown_reason'])
        self.assertEqual(state['input']['control']['joint_velocity_rad_s'],[0.]*6)
        for _ in range(60):state=self.tick(BUTTONS['A']|BUTTONS['LB'],LX=.6)
        self.assertEqual(state['shutdown_destination'],'origin')
        self.assertEqual(state['phase'],'returning')

    def test_y_returns_origin_without_disable(self):
        for _ in range(10):self.tick(BUTTONS['LB'],LX=.5)
        self.tick();self.tick(BUTTONS['Y'])
        for _ in range(200):state=self.tick()
        self.assertEqual(state['phase'],'at_origin',state)
        self.assertTrue(self.robot.enabled)
        self.assertFalse(state['closed'])

    def test_b_overrides_a_return_and_uses_all_zero_finish(self):
        for _ in range(153):state=self.tick(BUTTONS['A'])
        self.assertEqual(state['shutdown_destination'],'origin')
        state=self.tick(BUTTONS['B'])
        self.assertEqual(state['shutdown_destination'],'zero')
        for _ in range(400):
            state=self.tick()
            if state['closed']:break
        self.assertTrue(state['closed'],state)
        self.assertEqual(self.robot.targets,[0.]*7)

    def test_x_enable_latch_needs_new_neutral_press(self):
        latch=EnableLatch()
        self.assertFalse(latch.update(self.frame(BUTTONS['X']),self.now))
        self.now+=.02
        self.assertFalse(latch.update(self.frame(),self.now))
        self.now+=.02
        self.assertFalse(latch.update(self.frame(BUTTONS['X']|BUTTONS['LB']),self.now))
        self.now+=.02;latch.update(self.frame(),self.now)
        self.now+=.02
        self.assertTrue(latch.update(self.frame(BUTTONS['X']),self.now))

    def test_supervisor_waits_for_x_before_any_can_open(self):
        with patch.object(tony_teleop,'wait_for_x',return_value=False), \
             patch('native_pcan_backend.NativePcanBackend.open',side_effect=AssertionError('CAN opened')), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(tony_teleop.main(['--control','--allow-enable']),0)

    def test_supervisor_reopens_only_after_next_x_following_completed_stop(self):
        calls=[]
        def session(*args,**kwargs):
            calls.append(args)
            kwargs['result_callback']({'final':{'closed':True,'exit_mode':'github_origin_disable'}})
            return 0
        with patch.object(tony_teleop,'wait_for_x',side_effect=[True,True,False]), \
             patch.object(tony_teleop.live_runner,'main',side_effect=session),redirect_stdout(io.StringIO()):
            self.assertEqual(tony_teleop.main(['--control','--allow-enable']),0)
        self.assertEqual(len(calls),2)
        forwarded=calls[0][0]
        self.assertEqual(float(forwarded[forwarded.index('--duration')+1]),0.)

    def test_explicit_handover_skips_only_first_x_wait(self):
        completed=[]
        def session(*args,**kwargs):
            completed.append(True)
            kwargs['result_callback']({'final':{'closed':True,'exit_mode':'github_origin_disable'}})
            return 0
        with patch.object(tony_teleop,'wait_for_x',return_value=False) as wait, \
             patch.object(tony_teleop.live_runner,'main',side_effect=session),redirect_stdout(io.StringIO()):
            self.assertEqual(tony_teleop.main(['--control','--allow-enable','--take-over-running']),0)
        self.assertEqual(len(completed),1)
        self.assertEqual(wait.call_count,1)

if __name__=='__main__':unittest.main()
