"""Boundary behavior with fake native observations. No physical motion."""
import math
import unittest
from fake_robot import FakeRobot
from live_runner import NativeGamepadRunner,Frame,AXES,BUTTONS
from tony_engine import TonyEngine
from tony_github_config import github_config

class Rig:
    def __init__(self,origin):
        self.now=10.
        self.config=github_config()
        self.robot=FakeRobot(origin)
        self.runner=NativeGamepadRunner(self.now,self.robot,self.config,clock=lambda:self.now,
            allow_motion=True,allow_synthetic_backend=True,input_engine=TonyEngine(self.config))
        self.runner.start();self.tick()

    def tick(self,buttons=0,**axes):
        self.now+=.02
        values=dict.fromkeys(AXES,0.);values.update(axes)
        return self.runner.step(Frame(True,buttons=buttons,axes=values,acquired_at=self.now),self.now)

    def jog(self,axis,direction):
        buttons=BUTTONS['LB'];axes={}
        if axis<4:axes[('LX','LY','RY','RX')[axis]]=direction
        elif axis==4:buttons|=BUTTONS['DPadUp' if direction>0 else 'DPadDown']
        elif axis==5:buttons|=BUTTONS['DPadLeft' if direction>0 else 'DPadRight']
        else:axes['RT' if direction>0 else 'LT']=1.
        return self.tick(buttons,**axes)

class DirectionalLimitTests(unittest.TestCase):
    def origins(self):
        return [(j['native_min_rad']+j['native_max_rad'])/2 for j in github_config()['joints']]

    def test_only_j2_j3_intervals_changed(self):
        expected=[(-112.28,112.28),(0.,126.),(0.,126.),(-63.,63.),(-63.,63.),(-126.,126.),(0.,170.)]
        for joint,(lo,hi) in zip(github_config()['joints'],expected):
            self.assertAlmostEqual(joint['native_min_rad'],math.radians(lo))
            self.assertAlmostEqual(joint['native_max_rad'],math.radians(hi))

    def test_every_axis_at_both_bounds_blocks_only_outward_and_accepts_reverse(self):
        for axis,joint in enumerate(github_config()['joints']):
            for direction,key in ((1,'native_max_rad'),(-1,'native_min_rad')):
                with self.subTest(axis=axis+1,direction=direction):
                    origin=self.origins();origin[axis]=joint[key]
                    rig=Rig(origin);state=rig.jog(axis,direction)
                    self.assertIsNone(state['last_error'],state)
                    self.assertTrue(rig.robot.enabled)
                    self.assertFalse(state['closed'])
                    self.assertEqual(rig.robot.targets[axis],origin[axis])
                    self.assertEqual(state['soft_limit_blocks'][axis],'positive' if direction>0 else 'negative')
                    state=rig.jog(axis,-direction)
                    self.assertIsNone(state['last_error'],state)
                    self.assertGreater((rig.robot.targets[axis]-origin[axis])*(-direction),0.)

    def test_measured_overshoot_blocks_outward_without_disabling_other_axes(self):
        origin=self.origins();rig=Rig(origin)
        rig.robot.track_targets=False
        rig.robot.positions[1]=math.radians(127.)
        state=rig.tick(BUTTONS['LB'],LX=.7,LY=1.)
        self.assertIsNone(state['last_error'],state)
        self.assertEqual(rig.robot.targets[1],origin[1])
        self.assertGreater(rig.robot.targets[0],origin[0])
        self.assertEqual(state['soft_limit_blocks'][1],'positive')
        state=rig.jog(1,-1)
        self.assertIsNone(state['last_error'],state)
        self.assertLess(rig.robot.targets[1],origin[1])

    def test_starting_outside_boundary_does_not_snap_on_outward_request(self):
        origin=self.origins();origin[2]=math.radians(128.)
        rig=Rig(origin);state=rig.jog(2,1)
        self.assertIsNone(state['last_error'],state)
        self.assertEqual(rig.robot.targets[2],origin[2])
        state=rig.jog(2,-1)
        self.assertGreater(rig.robot.targets[2],math.radians(126.))
        self.assertLess(rig.robot.targets[2],origin[2])

    def test_y_and_a_return_to_immutable_origin_even_outside_jog_margin(self):
        origin=self.origins();origin[1]=math.radians(128.)
        rig=Rig(origin)
        for _ in range(20):rig.jog(1,-1)
        rig.tick();rig.tick(BUTTONS['Y'])
        for _ in range(500):state=rig.tick()
        self.assertEqual(state['phase'],'at_origin',state)
        self.assertTrue(rig.robot.enabled)
        self.assertAlmostEqual(rig.robot.targets[1],origin[1])
        for _ in range(154):state=rig.tick(BUTTONS['A'])
        for _ in range(500):
            state=rig.tick()
            if state['closed']:break
        self.assertTrue(state['closed'],state)
        self.assertEqual(state['exit_mode'],'github_origin_disable')
        self.assertEqual(rig.runner.planner.origin_rad,tuple(origin))

    def test_b_returns_all_axes_to_zero_from_soft_boundary(self):
        origin=[j['native_max_rad'] for j in github_config()['joints']]
        rig=Rig(origin);rig.tick(BUTTONS['B'])
        for _ in range(1500):
            state=rig.tick()
            if state['closed']:break
        self.assertTrue(state['closed'],state)
        self.assertEqual(state['exit_mode'],'github_smooth_zero_disable')
        self.assertEqual(rig.robot.targets,[0.]*7)

if __name__=='__main__':unittest.main()
