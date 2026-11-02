"""Upstream soft command limits must not masquerade as measured hard limits."""
import math
import struct
import unittest
from fake_robot import FakeRobot,DEFAULT_FAKE_ORIGIN
from live_runner import NativeGamepadRunner,Frame,AXES,BUTTONS
from native_pcan_backend import NativePcanBackend
from tony_engine import TonyEngine
from tony_github_config import github_config
from test_native_pcan_backend import Clock,WirePlant
from pcan_readonly import decode_packet,Packet

class LimitSemanticsTests(unittest.TestCase):
    def test_software_handover_does_not_force_enable_reset_motors(self):
        from high_rate_backend import HighRateBackend
        clock=Clock();wire=WirePlant(clock);config=github_config()
        backend=HighRateBackend(config,transport=wire,clock=clock,sleep=clock.sleep,require_running_handover=True)
        backend.open(allow_can_open=True)
        runner=NativeGamepadRunner(clock(),backend,config,allow_motion=True,clock=clock,input_engine=TonyEngine(config))
        state=runner.start()
        self.assertIn('already-running',state['last_error'])
        self.assertEqual(wire.commands,[])
        self.assertFalse(backend.enabled)

    def test_native_startup_can_hold_measured_angle_beyond_soft_command_limit(self):
        clock=Clock();wire=WirePlant(clock);config=github_config()
        wire.positions[1]=math.radians(127.04)
        backend=NativePcanBackend(config,transport=wire,clock=clock,sleep=clock.sleep)
        backend.open(allow_can_open=True)
        runner=NativeGamepadRunner(clock(),backend,config,allow_motion=True,clock=clock,input_engine=TonyEngine(config))
        state=runner.start()
        self.assertIsNone(state['last_error'],state)
        self.assertTrue(backend.enabled)
        self.assertGreater(backend.origin_rad[1],config['joints'][1]['native_max_rad'])

    def test_jog_targets_still_clamp_while_feedback_overshoot_does_not_lock_arm(self):
        now=[10.];config=github_config();origin=list(DEFAULT_FAKE_ORIGIN);origin[1]=math.radians(127.04)
        robot=FakeRobot(origin)
        runner=NativeGamepadRunner(now[0],robot,config,allow_motion=True,clock=lambda:now[0],
            allow_synthetic_backend=True,input_engine=TonyEngine(config))
        runner.start()
        for buttons in (0,BUTTONS['LB']):
            now[0]+=.02
            values=dict.fromkeys(AXES,0.);values['LY']=1. if buttons else 0.
            state=runner.step(Frame(True,buttons=buttons,axes=values,acquired_at=now[0]),now[0])
        self.assertIsNone(state['last_error'],state)
        self.assertAlmostEqual(robot.targets[1],math.radians(127.04))
        self.assertEqual(state['soft_limit_blocks'][1],'positive')
        now[0]+=.02;values['LY']=-1.
        state=runner.step(Frame(True,buttons=BUTTONS['LB'],axes=values,acquired_at=now[0]),now[0])
        self.assertIsNone(state['last_error'],state)
        self.assertLess(robot.targets[1],math.radians(127.04))
        self.assertEqual(config['joints'][1]['native_max_rad'],math.radians(126))

    def test_status_bit_two_and_full_fault_word_two_are_not_the_same_alarm(self):
        status=decode_packet(Packet((24<<24)|(2<<16)|(5<<8)|253,bytes(8)))
        fault=decode_packet(Packet((21<<24)|(5<<8)|253,struct.pack('<II',2,0)))
        self.assertTrue(status['fault_flags']['overcurrent'])
        self.assertEqual(fault['fault_namespace'],'fault_word_32')
        self.assertTrue(fault['fault_flags']['driver_fault'])
        self.assertFalse(fault['fault_flags']['phase_b_overcurrent'])

    def test_4030_decodes_stall_and_phase_b_c_current_reports(self):
        fault=decode_packet(Packet((21<<24)|(4<<8)|253,struct.pack('<II',0x4030,0)))
        for name in ('stall_overload','phase_b_overcurrent','phase_c_overcurrent'):
            self.assertTrue(fault['fault_flags'][name])

if __name__=='__main__':unittest.main()
