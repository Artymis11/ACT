"""Deterministic fake wire timing. No DLL or hardware is opened."""
import unittest
from unittest.mock import patch
from high_rate_backend import HighRateBackend
from live_runner import NativeGamepadRunner
from tony_engine import TonyEngine
from tony_github_config import github_config
from test_native_pcan_backend import Clock,WirePlant

class StreamTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.wire=WirePlant(self.clock)
        self.config=github_config()
        self.backend=HighRateBackend(self.config,transport=self.wire,clock=self.clock,sleep=self.clock.sleep)
        self.backend.open(allow_can_open=True)
        self.runner=NativeGamepadRunner(self.clock.now,self.backend,self.config,allow_motion=True,
            clock=self.clock,input_engine=TonyEngine(self.config))
        self.runner.start()
        self.assertIsNone(self.runner.last_error)
        with patch('high_rate_backend.threading.Thread') as thread:
            thread.return_value.is_alive.return_value=False
            self.backend.start_streaming()

    def queue(self,positions):
        for motor,q,joint in zip(self.backend.motors,positions,self.config['joints']):
            motor.send_mit(pos=q,vel=0.,kp=joint['kp'],kd=joint['kd'],tau=0.)

    def test_seven_target_commit_is_atomic_and_main_caller_never_writes_can(self):
        before=len(self.wire.commands)
        original=self.backend.desired
        joint=self.config['joints'][0]
        self.backend.motors[0].send_mit(pos=.3,vel=0.,kp=joint['kp'],kd=joint['kd'],tau=0.)
        self.assertEqual(self.backend.desired,original)
        self.assertEqual(len(self.wire.commands),before)
        self.queue([.3]+list(original[1:]))
        self.assertEqual(self.backend.desired[0],.3)
        self.assertEqual(len(self.wire.commands),before)

    def test_200hz_interpolation_has_source_speed_and_skips_send_bursts(self):
        original=self.backend.command[0]
        targets=list(self.backend.desired);targets[0]+=.1
        self.queue(targets);self.backend.set_speed_scale(.5)
        self.backend._pump()
        self.assertAlmostEqual(self.backend.command[0]-original,self.config['joints'][0]['max_speed_rad_s']*.5*.005)
        count=self.backend.send_batches
        self.backend._pump()
        self.assertEqual(count,self.backend.send_batches)
        self.clock.sleep(.015)
        self.backend._pump()
        self.assertEqual(self.backend.send_batches,count+1)

    def test_parameter_waits_keep_servicing_200hz_transmit(self):
        self.wire.delay_per_query=.0025
        before=self.backend.send_batches
        self.backend.wire.read_feedback(self.runner.session_id)
        self.assertGreaterEqual(self.backend.send_batches-before,9)

    def test_cached_publications_keep_original_rx_time_and_do_not_issue_queries(self):
        before=len(self.wire.queries)
        first=self.backend.read_feedback(self.runner.session_id)
        self.clock.sleep(.02)
        second=self.backend.read_feedback(self.runner.session_id)
        self.assertGreater(second.sequence,first.sequence)
        self.assertEqual(second.acquired_at,first.acquired_at)
        self.assertEqual(second.axis_timestamps,first.axis_timestamps)
        self.assertEqual(before,len(self.wire.queries))

    def test_fault_holds_last_sent_command_without_advancing_to_queued_target(self):
        self.backend._pump()
        previous=self.backend.command[:]
        target=previous[:];target[0]+=.1
        self.queue(target)
        self.backend.wire._latch('injected motor communication failure')
        self.clock.sleep(.005);self.backend._pump()
        self.assertEqual(self.backend.command,previous)
        self.assertTrue(self.backend.possibly_enabled)

    def stale_fault(self):
        self.backend.worker_error='NativeBackendError: feedback, fault or mode receipt is unknown/stale'
        self.backend.wire._latch(self.backend.worker_error)
        self.backend.suspended=True

    def test_recovered_data_remains_frozen_until_explicit_resume(self):
        self.backend._pump();self.stale_fault()
        self.backend._recover_in_owner()
        self.assertIsNone(self.backend.recovery_error)
        self.assertTrue(self.backend.suspended)
        targets=list(self.backend.command);targets[0]+=.1
        self.queue(targets);previous=self.backend.command[:]
        self.clock.sleep(.005);self.backend._pump()
        self.assertEqual(previous,self.backend.command)

    def test_recovery_cannot_refresh_expired_receipts_or_clear_motor_faults(self):
        self.stale_fault();self.clock.sleep(.4)
        self.backend._recover_in_owner()
        self.assertIsNotNone(self.backend.recovery_error)
        self.assertIsNotNone(self.backend.worker_error)
        self.backend.wire.fault_codes[0]=16
        self.backend._recover_in_owner()
        self.assertIsNotNone(self.backend.recovery_error)

    def test_x_recovers_in_same_session_without_enable_or_origin_recapture(self):
        from live_runner import Frame,BUTTONS
        self.backend._pump();self.stale_fault()
        self.runner.interrupt('injected expired feedback')
        origin=self.runner.planner.origin_rad
        enables=sum(packet.arbitration_id>>24==3 for packet in self.wire.commands)
        self.clock.sleep(.02)
        self.runner.step(Frame(True,acquired_at=self.clock()),self.clock())
        def recover(_sid):
            self.backend._recover_in_owner()
            if self.backend.recovery_error:raise ValueError(self.backend.recovery_error)
        self.clock.sleep(.02)
        with patch.object(self.backend,'recover_feedback_fault',side_effect=recover):
            state=self.runner.step(Frame(True,buttons=BUTTONS['X'],acquired_at=self.clock()),self.clock())
        self.assertEqual(state['phase'],'ready',state)
        self.assertIsNone(state['last_error'])
        self.assertEqual(origin,self.runner.planner.origin_rad)
        self.assertEqual(enables,sum(packet.arbitration_id>>24==3 for packet in self.wire.commands))

if __name__=='__main__':unittest.main()
