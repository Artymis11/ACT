"""Source-derived configuration and fake execution; no hardware checks."""
import copy
import math
import unittest
from fake_robot import FakeRobot
from live_runner import NativeGamepadRunner,Frame,AXES,BUTTONS
from native_pcan_backend import NativePcanBackend
from session_core import config_errors
from tony_engine import TonyEngine
from tony_github_config import github_config

class GitHubConfigTests(unittest.TestCase):
    def test_upstream_defaults_have_no_invented_acceptance_flags(self):
        config=github_config()
        for field in ('physical_parameters_verified','site_path_verified','return_path_verified','holding_gains_verified','hardware_loss_response_verified'):
            self.assertNotIn(field,config)
        self.assertEqual(config_errors(config),[])
        self.assertEqual(NativePcanBackend(config).admission_errors(),[])
        self.assertIsNone(config['tracking_error_rad'])
        self.assertIsNone(config['position_tolerance_rad'])
        self.assertIsNone(config['joints'][5]['max_accel_rad_s2'])
        self.assertEqual(config['joints'][5]['kp'],50.)
        self.assertAlmostEqual(config['joints'][5]['native_max_rad'],math.radians(126))
        self.assertAlmostEqual(config['joints'][6]['native_max_rad'],math.radians(170))

    def test_label_alone_cannot_authorize_a_modified_or_fake_profile(self):
        for field,value in (('joints',[]),('source_commit','other'),('synthetic_only',True)):
            config=github_config();config[field]=value
            self.assertTrue(config_errors(config))
            self.assertTrue(NativePcanBackend(config).admission_errors())

    def rig(self):
        self.now=10.
        self.config=github_config()
        self.robot=FakeRobot()
        self.runner=NativeGamepadRunner(self.now,self.robot,self.config,clock=lambda:self.now,
            allow_motion=True,allow_synthetic_backend=True,input_engine=TonyEngine(self.config))
        self.runner.start()
        self.tick()

    def tick(self,buttons=0,**axes):
        self.now+=.02
        values=dict.fromkeys(AXES,0.);values.update(axes)
        return self.runner.step(Frame(True,buttons=buttons,axes=values,acquired_at=self.now),self.now)

    def test_initial_neutral_readies_direct_lb_jog_and_keeps_zero_mit_velocity(self):
        self.rig()
        self.assertEqual(self.runner.planner.phase,'ready')
        for _ in range(3):state=self.tick(BUTTONS['LB'],LX=.5,RY=-.5)
        self.assertIsNone(state['last_error'])
        self.assertGreater(self.robot.targets[0],self.runner.planner.origin_rad[0])
        self.assertLess(self.robot.targets[2],self.runner.planner.origin_rad[2])
        self.assertTrue(all(v==0 for v in self.robot.command_velocities))

    def test_b_returns_zero_then_disables_instead_of_pausing_or_unloading_bent_arm(self):
        self.rig()
        origin=self.runner.planner.origin_rad
        state=self.tick(BUTTONS['B'])
        self.assertEqual(state['planner_phase'],'returning')
        self.assertTrue(self.robot.enabled)
        for _ in range(400):
            state=self.tick(BUTTONS['LB'],LX=.5)
            if state['closed']:break
        self.assertTrue(state['closed'],state)
        self.assertEqual(state['exit_mode'],'github_smooth_zero_disable')
        self.assertEqual(self.robot.targets,[0.]*7)
        self.assertFalse(self.robot.enabled)
        self.assertEqual(self.runner.planner.origin_rad,origin)

    def test_inactivity_waits_120_seconds_then_slowly_stops(self):
        self.rig();self.now=129.9
        state=self.tick()
        self.assertIsNone(state['shutdown_reason'])
        self.now=130.01
        state=self.tick()
        self.assertIn('120 seconds',state['shutdown_reason'])
        self.assertFalse(state['closed'])
        self.assertTrue(self.robot.enabled)
        self.assertEqual(state['phase'],'returning')

    def test_valid_operator_command_resets_idle_timer(self):
        self.rig();self.now=129.9
        self.tick(BUTTONS['LB'],LX=.5)
        self.now=131.
        state=self.tick()
        self.assertIsNone(state['shutdown_reason'])
        self.assertLess(state['idle_seconds'],2)

    def test_disconnected_gamepad_is_not_an_immediate_disable(self):
        self.rig();self.now+=1.
        state=self.runner.step(Frame(False,acquired_at=self.now),self.now)
        self.assertIsNone(state['shutdown_reason'])
        self.assertTrue(self.robot.enabled)
        self.assertFalse(state['closed'])

    def test_scheduler_delay_discards_backlog_without_an_extra_commissioning_fault(self):
        self.rig()
        self.now+=.15
        state=self.tick(BUTTONS['LB'],LX=.5)
        self.assertIsNone(state['last_error'],state)
        expected=self.runner.planner.origin_rad[0]+state['input']['control']['joint_velocity_rad_s'][0]*.02
        self.assertAlmostEqual(self.robot.targets[0],expected)

    def test_windows_status_file_lock_does_not_interrupt_github_control(self):
        from types import SimpleNamespace
        from test_live_runner import LoopSpy,CliOwnershipLoopTests
        runner=LoopSpy()
        runner.planner=SimpleNamespace(config=github_config())
        (fatal,_),_,_=CliOwnershipLoopTests.run_loop(self,runner,
            [(None,False),('confirm_external_power_off',True)],
            save_effect=[PermissionError('Windows reader denies rename'),None],expired=False)
        self.assertEqual(runner.interrupts,[])
        self.assertIsNone(fatal)
        self.assertTrue(runner.closed)

    def test_native_wire_profile_starts_without_added_flags_and_discloses_j3_quantization(self):
        from test_native_pcan_backend import Clock,WirePlant
        clock=Clock();wire=WirePlant(clock)
        wire.positions[2]=.00017
        config=github_config()
        backend=NativePcanBackend(config,transport=wire,clock=clock,sleep=clock.sleep)
        backend.open(allow_can_open=True)
        runner=NativeGamepadRunner(clock.now,backend,config,clock=clock,input_engine=TonyEngine(config),allow_motion=True)
        state=runner.start()
        self.assertIsNone(state['last_error'],state)
        self.assertTrue(backend.enabled)
        self.assertLess(backend.startup_quantization_preview[2]['position_quantization_error_rad'],0)
        self.assertEqual(sum(packet.arbitration_id>>24==3 for packet in wire.commands),7)
        backend.thermal_disconnect_requested=True
        backend.github_thermal_emergency_disconnect()
        self.assertFalse(backend.possibly_enabled)
        self.assertEqual(sum(packet.arbitration_id>>24==4 for packet in wire.commands),7)
        backend.close_bus();backend.close()

    def test_jog_soft_limit_clamps_as_upstream_instead_of_crossing(self):
        self.rig()
        for _ in range(1000):state=self.tick(BUTTONS['LB'],LX=1.)
        self.assertIsNone(state['last_error'])
        self.assertAlmostEqual(self.robot.targets[0],self.config['joints'][0]['native_max_rad'])

    def test_return_uses_fresh_origin_and_upstream_settle_without_faked_measured_tolerance(self):
        self.rig()
        origin=self.runner.planner.origin_rad
        for _ in range(10):self.tick(BUTTONS['LB'],LX=.7)
        self.tick();self.tick(BUTTONS['Y'])
        for _ in range(200):
            state=self.tick()
            if state['planner_phase']=='at_origin':break
        self.assertEqual(state['planner_phase'],'at_origin')
        self.assertEqual(self.runner.planner.origin_rad,origin)
        self.assertTrue(self.runner.transport.possibly_enabled)
        self.now+=.02
        state=self.runner.step(Frame(True,acquired_at=self.now),self.now,action='confirm_support',
            session_id=self.runner.session_id,supported=True)
        self.assertTrue(state['closed'],state)
        self.assertTrue(state['normal_disable_verified'])

if __name__=='__main__':unittest.main()
