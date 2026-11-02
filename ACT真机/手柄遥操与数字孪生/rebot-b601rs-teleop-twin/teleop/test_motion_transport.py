from dataclasses import replace
import unittest
from motion_transport import MotionTransport, TransportError
from session_core import NativeFeedback, SessionPlanner
from test_session_core import fixture_config


class Motor:
    def __init__(self,i,calls): self.i,self.calls,self.fail=i,calls,False
    def send_mit(self,**values):
        self.calls.append(("mit",self.i,values))
        if self.fail: raise RuntimeError("synthetic send failure")
    def close(self): self.calls.append(("motor_close",self.i))
class Controller:
    def __init__(self,calls): self.calls=calls; self.fail_enable=False; self.fail_disable=False
    def enable_all(self):
        self.calls.append(("enable_all",))
        if self.fail_enable: raise RuntimeError("synthetic partial enable")
    def disable_all(self):
        self.calls.append(("disable_all",))
        if self.fail_disable: raise RuntimeError("synthetic partial disable")
    def close_bus(self): self.calls.append(("close_bus",))
    def close(self): self.calls.append(("controller_close",))


class MotionTransportTests(unittest.TestCase):
    def setUp(self):
        self.now=10.; self.sequence=0; self.calls=[]
        self.q=(.1,.2,.3,-.1,.2,.1,1.)
        self.planner=SessionPlanner(fixture_config(),self.now,allow_synthetic_feedback=True)
        self.controller=Controller(self.calls)
        self.motors=[Motor(i,self.calls) for i in range(1,8)]
        self.adapter=MotionTransport(self.controller,self.motors,self.planner,allow_synthetic_feedback=True)
        self.planner.capture_origin(self.feedback(),self.now)
    def feedback(self):
        self.now+=.02; self.sequence+=1
        return NativeFeedback(self.planner.session_id,"synthetic_fixture",self.now,self.sequence,self.q,(0.,)*7,(self.now,)*7,(0,)*7,("MIT",)*7,(self.now,)*7)
    def arm(self):
        f=self.feedback()
        self.planner.resume(self.planner.session_id,f,self.now,neutral_confirmed=True)
        self.adapter.arm(f,self.now,allow_motion=True)
    def test_default_does_not_enable_or_send(self):
        with self.assertRaises(TransportError): self.adapter.arm(self.feedback(),self.now)
        self.assertEqual(self.calls,[])
    def test_start_holds_nonzero_origin_and_no_mode_zero_calls(self):
        self.arm()
        self.assertTrue(self.adapter.armed)
        self.assertEqual(sum(c[0]=="mit" for c in self.calls),14)
        first=[c for c in self.calls if c[0]=="mit"][:7]
        self.assertEqual(tuple(c[2]["pos"] for c in first),self.q)
        self.assertEqual({c[0] for c in self.calls},{"mit","enable_all"})
    def test_incomplete_site_config_blocks_arm(self):
        self.planner.config['physical_parameters_verified']=False
        with self.assertRaises(TransportError): self.adapter.arm(self.feedback(),self.now,allow_motion=True)
        self.assertEqual(self.calls,[])
    def test_unknown_stale_fault_or_wrong_mode_blocks_enable(self):
        for changes in ({'fault_codes':(None,)*7},{'fault_codes':(1,)*7},{'control_modes':('PP',)*7},{'status_timestamps':(1.,)*7}):
            f=replace(self.feedback(),**changes)
            with self.assertRaises(TransportError): self.adapter.arm(f,self.now,allow_motion=True)
        self.assertEqual(self.calls,[])
    def test_synthetic_feedback_denied_without_explicit_test_mode(self):
        a=MotionTransport(self.controller,self.motors,self.planner)
        with self.assertRaises(TransportError): a.arm(self.feedback(),self.now,allow_motion=True)
        self.assertEqual(self.calls,[])
    def test_replayed_plan_faults_without_disable_or_close(self):
        self.arm()
        f=self.feedback()
        old=self.planner.step(self.planner.session_id,f,self.now,(.1,0,0,0,0,0),permit=True)
        latest=self.planner.pause(self.planner.session_id)
        with self.assertRaises(TransportError): self.adapter.apply(old,self.feedback(),self.now)
        self.assertTrue(self.adapter.possibly_enabled)
        self.assertFalse(any(c[0] in ('disable_all','controller_close','close_bus') for c in self.calls))
    def test_partial_enable_retains_ownership(self):
        self.controller.fail_enable=True
        with self.assertRaises(TransportError): self.arm()
        self.assertTrue(self.adapter.possibly_enabled)
        with self.assertRaises(TransportError): self.adapter.close_after_disable()
        self.assertFalse(any(c[0]=='disable_all' for c in self.calls))
    def test_send_failure_is_fault_not_success_cleanup(self):
        self.arm()
        f=self.feedback()
        cmd=self.planner.step(self.planner.session_id,f,self.now,(.1,0,0,0,0,0),permit=True)
        self.motors[2].fail=True
        with self.assertRaises(TransportError): self.adapter.apply(cmd,f,self.now)
        self.assertTrue(self.adapter.possibly_enabled)
        self.assertTrue(self.adapter.last_send_errors)
        self.assertFalse(any(c[0] in ('disable_all','controller_close','close_bus') for c in self.calls))
    def test_pause_hold_is_zero_velocity_without_disable(self):
        self.arm()
        cmd=self.planner.pause(self.planner.session_id)
        self.adapter.apply(cmd,self.feedback(),self.now)
        self.assertTrue(all(c[2]['vel']==0 for c in self.calls if c[0]=='mit'))
        self.assertFalse(any(c[0]=='disable_all' for c in self.calls))
    def test_close_refused_while_armed(self):
        self.arm()
        with self.assertRaises(TransportError): self.adapter.close_after_disable()
        self.assertFalse(any(c[0]=='close_bus' for c in self.calls))
    def test_same_latest_plan_with_new_feedback_after_long_gap_rejected(self):
        self.arm()
        f=self.feedback()
        cmd=self.planner.step(self.planner.session_id,f,self.now,(.1,0,0,0,0,0),permit=True)
        self.now+=5
        fresh=self.feedback()
        with self.assertRaises(TransportError): self.adapter.apply(cmd,fresh,self.now)
        self.assertTrue(all(c[2]['vel']==0 for c in self.calls if c[0]=='mit'))
    def test_nonhold_plan_cannot_be_consumed_twice(self):
        self.arm()
        f=self.feedback()
        cmd=self.planner.step(self.planner.session_id,f,self.now,(.1,0,0,0,0,0),permit=True)
        self.adapter.apply(cmd,f,self.now)
        positive_before=sum(c[0]=='mit' and c[2]['vel']>0 for c in self.calls)
        with self.assertRaises(TransportError): self.adapter.apply(cmd,f,self.now)
        self.assertEqual(sum(c[0]=='mit' and c[2]['vel']>0 for c in self.calls),positive_before)
    def test_unpaired_fresh_feedback_cannot_execute_plan(self):
        self.arm()
        f=self.feedback()
        cmd=self.planner.step(self.planner.session_id,f,self.now,(.1,0,0,0,0,0),permit=True)
        with self.assertRaises(TransportError): self.adapter.apply(cmd,self.feedback(),self.now)
        self.assertTrue(all(c[2]['vel']==0 for c in self.calls if c[0]=='mit'))
    def test_partial_failure_never_rolls_successful_j3_target_back(self):
        self.arm()
        f=self.feedback()
        cmd=self.planner.step(self.planner.session_id,f,self.now,(0,0,.1,0,0,0),permit=True)
        self.motors[3].fail=True
        with self.assertRaises(TransportError): self.adapter.apply(cmd,f,self.now)
        j3_targets=[c[2]['pos'] for c in self.calls if c[0]=='mit' and c[1]==3]
        self.assertGreater(max(j3_targets),self.q[2])
        self.assertEqual(j3_targets[-1],max(j3_targets))
        self.assertEqual(self.adapter.last_targets[2],cmd.targets_rad[2])
    def prepare_verified_disable(self):
        self.arm()
        for _ in range(8):
            f=self.feedback()
            cmd=self.planner.step(self.planner.session_id,f,self.now,(.1,0,0,0,0,0),permit=True)
            self.adapter.apply(cmd,f,self.now)
            self.q=cmd.targets_rad
        cmd=self.planner.pause(self.planner.session_id)
        self.adapter.apply(cmd,self.feedback(),self.now)
        f=self.feedback()
        cmd=self.planner.request_return_origin(self.planner.session_id,f,self.now)
        self.adapter.apply(cmd,f,self.now)
        for _ in range(400):
            f=self.feedback()
            cmd=self.planner.step(self.planner.session_id,f,self.now,permit=False)
            self.adapter.apply(cmd,f,self.now)
            self.q=cmd.targets_rad
            if cmd.phase=='at_origin': break
        self.assertEqual(cmd.phase,'at_origin')
        self.assertFalse(any(c[0]=='disable_all' for c in self.calls))
        f=self.feedback()
        cmd=self.planner.confirm_support(self.planner.session_id,f,self.now,supported=True)
        return cmd,f
    def test_normal_return_support_disable_close_order(self):
        cmd,f=self.prepare_verified_disable()
        self.adapter.apply(cmd,f,self.now)
        self.assertFalse(self.adapter.possibly_enabled)
        self.adapter.close_after_disable()
        names=[c[0] for c in self.calls]
        self.assertLess(names.index('disable_all'),names.index('close_bus'))
        self.assertLess(names.index('close_bus'),names.index('motor_close'))
        self.assertEqual(names[-1],'controller_close')
        self.assertEqual(sum(n=='motor_close' for n in names),7)
    def test_partial_disable_never_releases_controller(self):
        cmd,f=self.prepare_verified_disable()
        self.controller.fail_disable=True
        with self.assertRaises(TransportError): self.adapter.apply(cmd,f,self.now)
        self.assertTrue(self.adapter.possibly_enabled)
        with self.assertRaises(TransportError): self.adapter.close_after_disable()
        self.assertFalse(any(c[0] in ('close_bus','controller_close','motor_close') for c in self.calls))


if __name__ == '__main__': unittest.main()
