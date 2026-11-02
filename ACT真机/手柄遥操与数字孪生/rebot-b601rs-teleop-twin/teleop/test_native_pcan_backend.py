"""Pure wire/plant fixtures; no DLL loading, PCAN opening, or physical motion."""
import copy
import json
import math
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import native_pcan_backend as native
import pcan_readonly as pcan
from fake_robot import synthetic_config
from motion_transport import MotionTransport
from session_core import SessionPlanner


class Clock:
    def __init__(self): self.now=100.0
    def __call__(self): return self.now
    def sleep(self,seconds): self.now+=seconds


def accepted_fixture():
    """An explicit generated site fixture, never saved as physical settings."""
    result=synthetic_config()
    result.update(synthetic_only=False,hardware_writes_supported=True,
                  holding_gains_verified=True,hardware_loss_response_verified=True,
                  process_exit_ownership_policy="retain_until_supported_disable",
                  provenance="test fixture only; NOT physical acceptance")
    return result


class WirePlant:
    def __init__(self,clock):
        self.clock=clock
        self.opened=False
        self.capability=None
        self.queue=[]
        self.driver_us=1000
        self.positions=[.2,-.1,0.0,.1,-.1,.1,.1]
        self.velocities=[0.0]*7
        self.modes=[0]*7
        self.states=[0]*7
        self.faults=[0]*7
        self.queries=[]
        self.commands=[]
        self.events=[]
        self.respond=True
        self.emit_status=True
        self.fail_command=None
        self.fail_close=False
        self.delay_per_query=0.0

    def open(self):
        self.events.append("open")
        self.opened=True

    def read(self):
        return self.queue.pop(0) if self.queue else None

    def _stamp(self):
        self.driver_us+=200
        return self.driver_us

    def status(self,axis,*,faults=None,state=None,stamp=None):
        faults=self.faults[axis-1] if faults is None else faults
        state=self.states[axis-1] if state is None else state
        return pcan.Packet((24<<24)|(state<<22)|(faults<<16)|(axis<<8)|253,
            bytes.fromhex("7fff7fff7fff00c8"),driver_timestamp_us=self._stamp() if stamp is None else stamp,dlc=8)

    def write_read_request(self,packet):
        pcan.validate_request(packet)
        self.queries.append(packet)
        if not self.respond:
            return
        if self.delay_per_query:
            self.clock.sleep(self.delay_per_query)
        axis=packet.arbitration_id&255
        parameter=int.from_bytes(packet.data[:2],"little")
        value={0x7019:self.positions[axis-1],0x701B:self.velocities[axis-1],0x7005:self.modes[axis-1]}[parameter]
        raw=struct.pack(pcan.FORMATS[parameter],value).ljust(4,b"\0")
        self.queue.append(pcan.Packet((17<<24)|(axis<<8)|253,
            struct.pack("<H",parameter)+bytes(2)+raw,driver_timestamp_us=self._stamp(),dlc=8))
        if self.emit_status:
            self.queue.append(self.status(axis))

    def _bind_motion_capability(self,capability):
        if self.capability is not None: raise RuntimeError("already bound")
        self.capability=capability

    def write_actuator_packet(self,packet,*,authorization):
        if self.capability is None or authorization is not self.capability:
            raise RuntimeError("foreign actuator capability")
        native.validate_actuator_packet(packet)
        comm=(packet.arbitration_id>>24)&31
        axis=packet.arbitration_id&255
        self.commands.append(packet)
        if self.fail_command==(comm,axis):
            raise RuntimeError("injected command submission failure")
        if comm in (3,4):
            self.states[axis-1]=2 if comm==3 else 0
        if self.emit_status:
            self.queue.append(self.status(axis))

    def close(self):
        self.events.append("close")
        if self.fail_close: raise RuntimeError("injected close failure")
        self.opened=False


class ProtocolTests(unittest.TestCase):
    def test_zero_and_original_gains_have_fixed_golden_bytes(self):
        zero=native.encode_mit(3,pos=0,vel=0,kp=0,kd=0,tau=0)
        self.assertEqual(zero.arbitration_id,0x017FFF03)
        self.assertEqual(zero.data.hex(),"7fff7fff00000000")
        rs06=native.encode_mit(2,pos=0,vel=0,kp=150,kd=10,tau=0)
        rs00=native.encode_mit(6,pos=0,vel=0,kp=50,kd=4,tau=0)
        self.assertEqual(rs06.data.hex(),"7fff7fff07ae1999")
        self.assertEqual(rs00.data.hex(),"7fff7fff1999cccc")

    def test_limits_are_model_specific_protocol_ranges_not_site_bounds(self):
        for axis in range(1,8):
            limits=native.WIRE_LIMITS["rs-06" if axis<=3 else "rs-00"]
            packet=native.encode_mit(axis,pos=limits[0],vel=limits[1],kp=limits[3],kd=limits[4],tau=limits[2])
            self.assertEqual(packet.data,struct.pack(">HHHH",65534,65534,65535,65535))
            self.assertEqual((packet.arbitration_id>>8)&65535,65534)
        with self.assertRaises(native.NativeBackendError):
            native.encode_mit(6,pos=0,vel=34,kp=1,kd=1,tau=0)
        native.encode_mit(3,pos=0,vel=34,kp=501,kd=6,tau=15)

    def test_nonfinite_boolean_and_out_of_range_fields_never_clip(self):
        defaults=dict(pos=0,vel=0,kp=1,kd=1,tau=0)
        for field,value in (("pos",13),("vel",51),("kp",5001),("kd",101),("tau",37),
                            ("kp",-1),("pos",math.nan),("vel",math.inf),("tau",True)):
            with self.subTest(field=field,value=value):
                with self.assertRaises(native.NativeBackendError):
                    native.encode_mit(3,**{**defaults,field:value})

    def test_enable_disable_do_not_clear_fault_or_save_parameters(self):
        self.assertEqual(native.control_packet(7,enable=True).arbitration_id,0x0300FD07)
        self.assertEqual(native.control_packet(7,enable=False).arbitration_id,0x0400FD07)
        for enable in (True,False): self.assertEqual(native.control_packet(7,enable=enable).data,bytes(8))
        for comm in (0,2,6,7,17,18,21,22,23,24,25):
            with self.assertRaises(native.NativeBackendError):
                native.validate_actuator_packet(pcan.Packet((comm<<24)|0xFD01,bytes(8)))
        with self.assertRaises(native.NativeBackendError):
            native.validate_actuator_packet(pcan.Packet(0x0400FD01,bytes([1])+bytes(7)))

    def test_effective_wire_target_exposes_j3_reverse_quantization(self):
        q=.00017044279957190156
        packet=native.encode_mit(3,pos=q,vel=0,kp=150,kd=10,tau=0)
        effective=native.decode_mit_effective(packet)
        self.assertEqual(effective["effective_position_rad"],0.0)
        self.assertLess(effective["effective_position_rad"],q)
        self.assertEqual(effective["effective_velocity_rad_s"],0.0)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock()
        self.plant=WirePlant(self.clock)
        self.backend=native.NativePcanBackend(accepted_fixture(),transport=self.plant,
                                              clock=self.clock,sleep=self.clock.sleep)

    def open_read(self):
        self.backend.open(allow_can_open=True)
        return self.backend.read_feedback("current-session",self.clock())

    def authorize(self):
        feedback=self.open_read()
        token=self.backend.authorize_motion("current-session",feedback,self.clock(),allow_motion=True)
        return feedback,token

    def prime(self):
        for motor,q,joint in zip(self.backend.motors,self.backend.origin_rad,self.backend.config["joints"]):
            motor.send_mit(pos=q,vel=0,kp=joint["kp"],kd=joint["kd"],tau=0)

    def enable(self):
        result=self.authorize()
        self.prime()
        self.backend.enable_all()
        return result

    def stable_origin(self):
        for _ in range(8):
            self.clock.sleep(.02)
            feedback=self.backend.read_feedback("current-session")
        return feedback

    def test_construction_import_and_default_open_never_load_dll_or_open_can(self):
        with patch.object(native,"NativeMotionPcanTransport",side_effect=AssertionError("must not load")):
            backend=native.NativePcanBackend(accepted_fixture())
            with self.assertRaises(PermissionError): backend.open()
        self.assertEqual(self.plant.events,[])
        self.assertEqual(self.plant.commands,[])

    def test_feedback_comes_from_new_parameter_and_status_receipts_not_targets(self):
        feedback=self.open_read()
        self.assertEqual(feedback.source,"live_native")
        self.assertEqual(feedback.session_id,"current-session")
        self.assertEqual(feedback.fault_codes,(0,)*7)
        self.assertEqual(feedback.control_modes,("MIT",)*7)
        self.assertEqual(len(self.plant.queries),21)
        self.assertEqual(feedback.acquired_at,self.clock())
        self.assertEqual(self.plant.commands,[])
        self.plant.positions[3]=.33
        second=self.backend.read_feedback("current-session")
        self.assertGreater(second.sequence,feedback.sequence)
        self.assertAlmostEqual(second.positions_rad[3],.33,places=6)

    def test_feedback_cannot_switch_session_or_use_caller_time_as_receive_time(self):
        first=self.open_read()
        with self.assertRaises(native.NativeBackendError): self.backend.read_feedback("old-session")
        self.plant.delay_per_query=.001
        second=self.backend.read_feedback("current-session",first.acquired_at)
        self.assertGreater(second.acquired_at,first.acquired_at)

    def test_one_global_batch_deadline_bounds_missing_responses(self):
        self.backend.open(allow_can_open=True)
        self.plant.respond=False
        before=self.clock()
        with self.assertRaises(native.NativeBackendError): self.backend.read_feedback("current-session")
        self.assertLessEqual(self.clock()-before,.100001)
        self.assertEqual(len(self.plant.queries),1)
        self.assertIsNotNone(self.backend.fault)
        self.assertEqual(self.plant.commands,[])

    def test_completed_but_slow_batch_also_faults(self):
        self.backend.open(allow_can_open=True)
        self.plant.delay_per_query=.02
        with self.assertRaises(native.NativeBackendError): self.backend.read_feedback("current-session")
        self.assertLess(len(self.plant.queries),21)

    def test_absent_status_stays_unknown_and_blocks_authorization(self):
        self.plant.emit_status=False
        feedback=self.open_read()
        self.assertEqual(feedback.fault_codes,(None,)*7)
        self.assertEqual(feedback.status_timestamps,(None,)*7)
        with self.assertRaises(native.NativeBackendError):
            self.backend.authorize_motion("current-session",feedback,allow_motion=True)
        self.assertEqual(self.plant.commands,[])

    def test_nonzero_fault_latches_across_later_health(self):
        self.plant.faults[4]=4
        feedback=self.open_read()
        self.assertEqual(feedback.fault_codes[4],4)
        self.plant.faults[4]=0
        next_feedback=self.backend.read_feedback("current-session")
        self.assertEqual(next_feedback.fault_codes[4],4)
        with self.assertRaises(native.NativeBackendError):
            self.backend.authorize_motion("current-session",next_feedback,allow_motion=True)

    def test_driver_clock_regression_latches_even_if_parameter_response_would_match(self):
        self.open_read()
        self.plant.queue.append(self.plant.status(1,stamp=1))
        with self.assertRaises(native.NativeBackendError): self.backend.read_feedback("current-session")
        self.assertIn("regressed",self.backend.fault)

    def test_candidate_synthetic_and_unaccepted_loss_or_gains_cannot_authorize(self):
        feedback=self.open_read()
        for changes in ({"synthetic_only":True},{"status":"MOTION_BLOCKED_CANDIDATES_ONLY"},
                        {"hardware_loss_response_verified":False},{"holding_gains_verified":False},
                        {"process_exit_ownership_policy":"close_on_exception"}):
            with self.subTest(changes=changes):
                previous=copy.deepcopy(self.backend.config)
                self.backend.config.update(changes)
                with self.assertRaises(native.NativeBackendError):
                    self.backend.authorize_motion("current-session",feedback,allow_motion=True)
                self.backend.config=previous
        self.assertEqual(self.plant.commands,[])

    def test_current_candidate_file_stays_blocked_without_hardware(self):
        config=json.loads((Path(__file__).parent/"site_config.candidate_20261030.json").read_text(encoding="utf-8"))
        backend=native.NativePcanBackend(config)
        self.assertTrue(backend.admission_errors())
        self.assertFalse(backend.opened)

    def test_explicit_authorization_and_exact_latest_feedback_required(self):
        first=self.open_read()
        with self.assertRaises(PermissionError): self.backend.authorize_motion("current-session",first)
        second=self.backend.read_feedback("current-session")
        with self.assertRaises(native.NativeBackendError):
            self.backend.authorize_motion("current-session",first,allow_motion=True)
        self.backend.authorize_motion("current-session",second,allow_motion=True)
        self.assertEqual(self.plant.commands,[])

    def test_active_existing_motor_and_non_mit_modes_cannot_be_taken_over(self):
        self.plant.states[2]=2
        feedback=self.open_read()
        with self.assertRaises(native.NativeBackendError):
            self.backend.authorize_motion("current-session",feedback,allow_motion=True)
        self.assertEqual(self.plant.commands,[])

    def test_j3_wire_reverse_rejects_whole_grant_before_any_other_axis_priming(self):
        self.plant.positions[2]=.00017044279957190156
        feedback=self.open_read()
        with self.assertRaisesRegex(native.NativeBackendError,"J3 startup hold"):
            self.backend.authorize_motion("current-session",feedback,allow_motion=True)
        self.assertEqual(self.plant.commands,[])
        self.assertIsNone(self.backend.authorization)
        self.assertEqual(self.backend.startup_quantization_preview[2]["effective_position_rad"],0)

    def test_enable_requires_all_exact_measured_holds_then_fresh_ack(self):
        self.authorize()
        with self.assertRaises(native.NativeBackendError): self.backend.enable_all()
        self.prime()
        self.backend.enable_all()
        self.assertTrue(self.backend.enabled)
        self.assertTrue(self.backend.possibly_enabled)
        self.assertEqual([(p.arbitration_id>>24)&31 for p in self.plant.commands],[1]*7+[3]*7)
        for record in self.backend.submissions[:7]:
            self.assertIn("effective_wire",record)
            self.assertIn("position_quantization_error_rad",record)

    def test_target_or_gain_cannot_change_during_unenabled_priming(self):
        self.authorize()
        with self.assertRaises(native.NativeBackendError):
            self.backend.motors[0].send_mit(pos=.3,vel=0,kp=1,kd=.2,tau=0)
        with self.assertRaises(native.NativeBackendError):
            self.backend.motors[0].send_mit(pos=self.backend.origin_rad[0],vel=0,kp=2,kd=.2,tau=0)
        self.assertEqual(self.plant.commands,[])

    def test_partial_enable_failure_keeps_owned_target_and_never_auto_disables_or_closes(self):
        self.authorize(); self.prime()
        self.plant.fail_command=(3,3)
        with self.assertRaises(native.NativeBackendError): self.backend.enable_all()
        self.assertTrue(self.backend.possibly_enabled)
        with self.assertRaises(native.NativeBackendError): self.backend.close_bus()
        self.assertNotIn("close",self.plant.events)
        self.assertNotIn(4,[(p.arbitration_id>>24)&31 for p in self.plant.commands])
        self.plant.fail_command=None
        q=self.backend.last_targets[0]
        self.backend.motors[0].send_mit(pos=q,vel=0,kp=1,kd=.2,tau=0)
        with self.assertRaises(native.NativeBackendError):
            self.backend.motors[0].send_mit(pos=q+.01,vel=0,kp=1,kd=.2,tau=0)

    def test_fault_allows_owned_hold_after_feedback_expires_but_not_new_target_or_enable(self):
        self.enable()
        self.plant.respond=False
        with self.assertRaises(native.NativeBackendError): self.backend.read_feedback("current-session")
        self.clock.sleep(.5)
        self.backend.motors[1].send_mit(pos=self.backend.last_targets[1],vel=0,kp=1,kd=.2,tau=0)
        with self.assertRaises(native.NativeBackendError): self.backend.enable_all()
        with self.assertRaises(native.NativeBackendError): self.backend.disable_all()

    def test_configuration_change_faults_new_motion_but_preserves_cached_hold(self):
        self.enable()
        self.backend.config["joints"][0]["kp"]=2
        with self.assertRaises(native.NativeBackendError):
            self.backend.motors[0].send_mit(pos=.3,vel=0,kp=2,kd=.2,tau=0)
        self.backend.motors[0].send_mit(pos=self.backend.last_targets[0],vel=0,kp=1,kd=.2,tau=0)

    def test_normal_disable_needs_own_stability_and_explicit_support_then_new_state_ack(self):
        self.enable()
        with self.assertRaises(native.NativeBackendError): self.backend.disable_all()
        feedback=self.stable_origin()
        with self.assertRaises(native.NativeBackendError):
            self.backend.authorize_supported_disable("current-session",feedback,supported=False)
        self.backend.authorize_supported_disable("current-session",feedback,supported=True)
        self.backend.disable_all()
        self.assertFalse(self.backend.possibly_enabled)
        self.assertFalse(self.backend.enabled)
        self.backend.close_bus()
        for motor in self.backend.motors: motor.close()
        self.backend.close()
        self.assertTrue(self.backend.closed)
        self.assertEqual(self.plant.events,["open","close"])

    def test_sampling_gap_invalidates_normal_disable_even_with_later_one_good_read(self):
        self.enable()
        self.stable_origin()
        self.clock.sleep(.3)
        feedback=self.backend.read_feedback("current-session")
        with self.assertRaises(native.NativeBackendError):
            self.backend.authorize_supported_disable("current-session",feedback,supported=True)

    def test_partial_disable_failure_retains_ownership_and_no_automatic_close(self):
        self.enable()
        feedback=self.stable_origin()
        self.backend.authorize_supported_disable("current-session",feedback,supported=True)
        self.plant.fail_command=(4,4)
        with self.assertRaises(native.NativeBackendError): self.backend.disable_all()
        self.assertTrue(self.backend.possibly_enabled)
        with self.assertRaises(native.NativeBackendError): self.backend.close_bus()
        self.assertNotIn("close",self.plant.events)

    def test_motion_transport_uses_same_backend_and_single_control_session(self):
        self.backend.open(allow_can_open=True)
        planner=SessionPlanner(self.backend.config,self.clock())
        feedback=self.backend.read_feedback(planner.session_id)
        planner.capture_origin(feedback,self.clock())
        self.backend.authorize_motion(planner.session_id,feedback,allow_motion=True)
        transport=MotionTransport(self.backend,self.backend.motors,planner)
        transport.arm(feedback,self.clock(),allow_motion=True)
        self.assertTrue(transport.armed)
        self.assertTrue(self.backend.enabled)
        self.assertEqual(self.plant.events,["open"])

    def test_submission_tail_is_bounded_without_resetting_success_sequence(self):
        self.authorize(); self.prime()
        self.backend.submissions=[{} for _ in range(4096)]
        previous=self.backend.submission_sequence
        q=self.backend.last_targets[0]
        self.backend.motors[0].send_mit(pos=q,vel=0,kp=1,kd=.2,tau=0)
        self.assertEqual(len(self.backend.submissions),4096)
        self.assertEqual(self.backend.submission_sequence,previous+1)
        self.assertTrue(self.backend.submissions[-1]["submitted"])

    def test_external_power_off_handoff_needs_both_manual_facts_and_current_session(self):
        self.enable()
        count=len(self.plant.commands)
        for session,support,power in (("old-session",True,True),("current-session",False,True),
                                      ("current-session",True,False),("current-session",True,1)):
            with self.subTest(session=session,support=support,power=power):
                with self.assertRaises(native.NativeBackendError):
                    self.backend.acknowledge_external_power_off(session,supported=support,motor_power_off_confirmed=power)
                self.assertTrue(self.backend.possibly_enabled)
                self.assertEqual(len(self.plant.commands),count)
        self.backend.acknowledge_external_power_off("current-session",supported=True,motor_power_off_confirmed=True)
        self.assertEqual(len(self.plant.commands),count)
        self.assertFalse(self.backend.possibly_enabled)
        self.assertFalse(self.backend.enabled)
        self.assertFalse(self.backend.external_poweroff_acknowledgement["normal_disable_verified"])
        self.backend.close_bus()
        self.backend.close()
        self.assertEqual(self.plant.events,["open","close"])

    def test_enable_ack_timeout_does_not_claim_enabled_or_release_bus(self):
        self.authorize(); self.prime()
        self.plant.queue.clear()
        self.plant.emit_status=False
        with self.assertRaisesRegex(native.NativeBackendError,"not acknowledged"):
            self.backend.enable_all()
        self.assertTrue(self.backend.possibly_enabled)
        self.assertFalse(self.backend.enabled)
        with self.assertRaises(native.NativeBackendError): self.backend.close_bus()


if __name__=="__main__": unittest.main()
