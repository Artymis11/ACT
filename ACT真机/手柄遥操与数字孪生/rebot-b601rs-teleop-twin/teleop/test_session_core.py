"""Synthetic-only planning tests. Fixture values are never site calibration."""
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from session_core import NativeFeedback, PlannedCommand, SessionError, SessionPlanner, config_errors


def fixture_config():
    return {
        "native_coordinate":"motor_native_rad", "physical_parameters_verified":True,
        "site_path_verified":True, "return_path_verified":True, "j3_reverse_motion_verified":True,
        "feedback_max_age_s":.25, "max_control_interval_s":.1,
        "tracking_error_rad":.3, "position_tolerance_rad":.002,
        "velocity_tolerance_rad_s":.005, "stable_dwell_s":.1,
        "joints":[{"name":f"J{i+1}","motor_id":i+1,"host_id":253,
                   "model":"rs-06" if i < 3 else "rs-00", "native_min_rad":-2.,
                   "native_max_rad":2.,"max_speed_rad_s":.4,"max_accel_rad_s2":.8,
                   "kp":1.,"kd":.2} for i in range(7)]
    }


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.config = fixture_config()
        self.planner = SessionPlanner(self.config, self.now, allow_synthetic_feedback=True)
        self.q = (.1,.2,.3,-.1,.2,.1,1.)
        self.v = (0.,)*7
        self.seq = 0

    def feedback(self, dt=.02, **changes):
        self.now += dt
        self.seq += 1
        data = NativeFeedback(self.planner.session_id,"synthetic_fixture",self.now,self.seq,
                              self.q,self.v,(self.now,)*7,(0,)*7,("MIT",)*7,(self.now,)*7)
        return replace(data, **changes)

    def capture(self):
        cmd = self.planner.capture_origin(self.feedback(), self.now)
        self.assertEqual(cmd.phase,"paused")
        self.assertEqual(cmd.kind,"hold_current")
        return cmd

    def ready(self):
        self.capture()
        cmd = self.planner.resume(self.planner.session_id,self.feedback(),self.now,neutral_confirmed=True)
        self.assertEqual(cmd.phase,"ready")

    def tick(self, velocity=(0.,)*6, gripper="hold", permit=True, dt=.02, feedback_changes=None):
        feedback = self.feedback(dt=dt, **(feedback_changes or {}))
        cmd = self.planner.step(self.planner.session_id,feedback,self.now,velocity,gripper,permit)
        if cmd.targets_rad is not None:
            self.q = cmd.targets_rad
        self.v = cmd.target_velocities_rad_s
        return cmd

    def assert_hold(self, cmd):
        self.assertEqual(cmd.kind,"hold_current")
        self.assertEqual(cmd.target_velocities_rad_s,(0.,)*7)
        self.assertFalse(cmd.hardware_writes_supported)

    def start_return(self):
        self.ready()
        for _ in range(15):
            self.tick((.2,0,0,0,0,0))
        self.planner.pause(self.planner.session_id)
        self.v = (0.,)*7  # synthetic measured stop, not an assumption in production
        cmd = self.planner.request_return_origin(self.planner.session_id,self.feedback(),self.now)
        self.assertEqual(cmd.kind,"request_return_session_origin")
        return cmd

    def finish_return(self):
        self.start_return()
        for _ in range(250):
            cmd = self.tick()
            if cmd.phase == "at_origin":
                return cmd
            self.assertNotEqual(cmd.phase,"fault",cmd.reason)
        self.fail("return did not settle")

    def test_nonzero_fresh_origin_is_frozen_and_first_output_holds(self):
        cmd = self.capture()
        self.assertEqual(cmd.origin_rad,self.q)
        self.assertNotEqual(cmd.targets_rad,(0.,)*7)
        with self.assertRaises(SessionError):
            self.planner.capture_origin(self.feedback(),self.now)
        self.assertEqual(self.planner.origin_rad,(.1,.2,.3,-.1,.2,.1,1.))

    def test_two_sessions_cannot_reuse_feedback_or_origin(self):
        self.capture()
        old = self.feedback()
        other = SessionPlanner(self.config,self.now,allow_synthetic_feedback=True)
        cmd = other.capture_origin(old,self.now+.01)
        self.assertEqual(cmd.phase,"fault")
        self.assertIsNone(cmd.origin_rad)
        self.assertNotEqual(other.session_id,self.planner.session_id)

    def test_no_model_preview_command_or_log_source(self):
        for source in ("model_b","preview","old_log","command"):
            p = SessionPlanner(self.config,10,allow_synthetic_feedback=True)
            f = NativeFeedback(p.session_id,source,10.01,1,self.q,self.v,(10.01,)*7)
            self.assertEqual(p.capture_origin(f,10.01).phase,"fault")

    def test_synthetic_source_denied_by_default(self):
        p = SessionPlanner(self.config,10)
        f = NativeFeedback(p.session_id,"synthetic_fixture",10.01,1,self.q,self.v,(10.01,)*7)
        self.assertEqual(p.capture_origin(f,10.01).phase,"fault")

    def test_feedback_before_session_start_denied(self):
        cmd = self.planner.capture_origin(self.feedback(axis_timestamps=(9.99,)*7),self.now)
        self.assertEqual(cmd.phase,"fault")
        self.assertIsNone(cmd.origin_rad)

    def test_one_of_seven_stale_axes_denies_origin(self):
        cmd = self.planner.capture_origin(self.feedback(axis_timestamps=(10.02,)*6+(9.,)),self.now)
        self.assertEqual(cmd.phase,"fault")

    def test_origin_capture_fault_does_not_implicitly_reset(self):
        self.planner.capture_origin(self.feedback(source="preview"),self.now)
        cmd = self.planner.capture_origin(self.feedback(),self.now)
        self.assertEqual(cmd.phase,"fault")
        self.assertIsNone(cmd.origin_rad)

    def test_template_incomplete_blocks_all_motion(self):
        template = json.loads((Path(__file__).parent/'site_config.template.json').read_text(encoding='utf-8'))
        self.assertGreater(len(config_errors(template)),35)
        self.planner = SessionPlanner(template,10,allow_synthetic_feedback=True)
        self.capture()
        cmd = self.planner.resume(self.planner.session_id,self.feedback(),self.now,neutral_confirmed=True)
        self.assert_hold(cmd)
        self.assertEqual(cmd.phase,"paused")
        cmd = self.tick((.2,)*6,"open")
        self.assert_hold(cmd)
        self.assertTrue(cmd.admission_errors)

    def test_external_configuration_changes_do_not_change_session(self):
        self.config['joints'][0]['native_max_rad'] = 999
        self.assertEqual(self.planner.config['joints'][0]['native_max_rad'],2.)

    def test_wrong_identity_and_native_units_denied(self):
        for mutation in (lambda c:c.update(native_coordinate='training_deg'),
                         lambda c:c['joints'][0].update(motor_id=4),
                         lambda c:c['joints'][6].update(model='rs-06')):
            config=fixture_config(); mutation(config)
            self.assertTrue(config_errors(config))

    def test_resume_requires_neutral_confirmation(self):
        self.capture()
        with self.assertRaises(SessionError):
            self.planner.resume(self.planner.session_id,self.feedback(),self.now)

    def test_resume_with_unknown_status_clears_prior_motion_before_rejection(self):
        self.ready()
        moving=self.tick((.2,0,0,0,0,0))
        self.assertGreater(moving.target_velocities_rad_s[0],0)
        cmd=self.planner.resume(self.planner.session_id,self.feedback(fault_codes=(None,)*7),self.now,neutral_confirmed=True)
        self.assert_hold(cmd)
        self.assertEqual(cmd.phase,'paused')
        self.assertIsNone(self.planner._return)
        self.assertIsNone(self.planner._stable_since)
        self.assertIs(cmd,self.planner.latest_command)

    def test_resume_with_invalid_site_config_clears_prior_motion(self):
        self.ready(); self.tick((.2,0,0,0,0,0))
        self.planner.config['physical_parameters_verified']=False
        cmd=self.planner.resume(self.planner.session_id,self.feedback(),self.now,neutral_confirmed=True)
        self.assert_hold(cmd)
        self.assertEqual(cmd.phase,'paused')

    def test_resume_without_neutral_confirmation_does_not_retain_movement(self):
        self.ready(); self.tick((.2,0,0,0,0,0))
        with self.assertRaises(SessionError):
            self.planner.resume(self.planner.session_id,self.feedback(),self.now,neutral_confirmed=False)
        self.assertEqual(self.planner.phase,'paused')
        self.assertEqual(self.planner._velocity,(0.,)*7)

    def test_resume_rejects_measured_drift_or_motion(self):
        self.capture()
        self.v=(.01,)+(0.,)*6
        cmd=self.planner.resume(self.planner.session_id,self.feedback(),self.now,neutral_confirmed=True)
        self.assertEqual(cmd.phase,'fault')
        self.setUp(); self.capture()
        self.q=(self.q[0]+.31,)+self.q[1:]
        cmd=self.planner.resume(self.planner.session_id,self.feedback(),self.now,neutral_confirmed=True)
        self.assertEqual(cmd.phase,'fault')

    def test_control_interval_configuration_cannot_exceed_freshness(self):
        config=fixture_config(); config['max_control_interval_s']=.3
        self.assertTrue(config_errors(config))

    def test_no_permit_means_hold_all_seven_axes(self):
        self.ready()
        cmd = self.tick((.2,)*6,"open",False)
        self.assert_hold(cmd)

    def test_nonzero_motor_fault_latches_hold(self):
        self.ready()
        cmd=self.tick(feedback_changes={'fault_codes':(0,0,4,0,0,0,0)})
        self.assertEqual(cmd.phase,'fault')
        self.assertIn('J3 motor reports fault code 4',cmd.reason)
        self.assert_hold(cmd)
        self.assertEqual(self.tick().phase,'fault')

    def test_unknown_motor_fault_status_blocks_motion_and_requires_resume(self):
        self.ready()
        cmd=self.tick((.1,0,0,0,0,0),feedback_changes={'fault_codes':(None,)+(0,)*6})
        self.assert_hold(cmd)
        self.assertEqual(cmd.phase,'paused')
        self.assertTrue(any('fault status unknown' in e for e in cmd.admission_errors))
        self.assert_hold(self.tick((.1,0,0,0,0,0)))

    def test_unknown_or_wrong_control_mode_blocks_motion(self):
        for mode in (None,'POS_VEL','VEL'):
            self.setUp(); self.ready()
            cmd=self.tick((.1,0,0,0,0,0),feedback_changes={'control_modes':(mode,)+('MIT',)*6})
            self.assert_hold(cmd)
            self.assertEqual(cmd.phase,'paused')

    def test_unknown_fault_status_cannot_be_filled_by_default(self):
        f=self.feedback()
        f=replace(f,fault_codes=(None,)*7,control_modes=(None,)*7)
        cmd=self.planner.capture_origin(f,self.now)
        self.assertEqual(cmd.phase,'paused')
        self.assertEqual(cmd.origin_rad,self.q)
        self.assertTrue(cmd.admission_errors)
        cmd=self.planner.resume(self.planner.session_id,self.feedback(fault_codes=(None,)*7),self.now,neutral_confirmed=True)
        self.assertEqual(cmd.phase,'paused')
        self.assert_hold(cmd)

    def test_nonzero_fault_during_support_confirmation_cannot_disable(self):
        self.finish_return()
        cmd=self.planner.confirm_support(self.planner.session_id,self.feedback(fault_codes=(1,)+(0,)*6),self.now,supported=True)
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)

    def test_unknown_fault_during_support_confirmation_cannot_disable(self):
        self.finish_return()
        cmd=self.planner.confirm_support(self.planner.session_id,self.feedback(fault_codes=(None,)*7),self.now,supported=True)
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)

    def test_stale_fault_or_mode_status_is_not_fresh_position_feedback(self):
        self.ready()
        cmd=self.tick((.1,0,0,0,0,0),feedback_changes={'status_timestamps':(self.now-1,)*7})
        self.assertEqual(cmd.phase,'paused')
        self.assert_hold(cmd)
        self.assertTrue(any('status freshness' in e for e in cmd.admission_errors))

    def test_latest_command_replaces_earlier_plan_object(self):
        self.ready()
        first=self.tick((.1,0,0,0,0,0))
        self.assertIs(first,self.planner.latest_command)
        second=self.tick(permit=False)
        self.assertIs(second,self.planner.latest_command)
        self.assertIsNot(first,self.planner.latest_command)
        with self.assertRaises(AttributeError):
            self.planner.latest_command=first

    def test_last_sample_time_is_read_only_and_updates_with_step(self):
        self.assertIsNone(self.planner.last_sample_time)
        self.ready()
        before=self.planner.last_sample_time
        self.tick((.1,0,0,0,0,0))
        self.assertGreater(self.planner.last_sample_time,before)
        self.assertEqual(self.planner.last_sample_time,self.now)
        with self.assertRaises(AttributeError):
            self.planner.last_sample_time=0.

    def test_latest_feedback_is_read_only_and_tracks_exact_new_frame(self):
        self.assertIsNone(self.planner.latest_feedback)
        self.ready()
        previous=self.planner.latest_feedback
        current=self.feedback()
        self.assertIs(self.planner.latest_feedback,previous)
        self.planner.step(self.planner.session_id,current,self.now,permit=False)
        self.assertIs(self.planner.latest_feedback,current)
        self.assertIsNot(self.planner.latest_feedback,previous)
        with self.assertRaises(AttributeError):
            self.planner.latest_feedback=previous

    def test_speed_and_acceleration_are_bounded(self):
        self.ready()
        last = (0.,)*7
        for _ in range(40):
            cmd = self.tick((9.,0,0,0,0,0))
            self.assertEqual(cmd.kind,"bounded_native_targets")
            self.assertLessEqual(abs(cmd.target_velocities_rad_s[0]),.4+1e-10)
            self.assertLessEqual(abs(cmd.target_velocities_rad_s[0]-last[0]),.8*.02+1e-10)
            last = cmd.target_velocities_rad_s

    def test_gripper_is_native_rad_and_open_increases(self):
        self.ready()
        cmd = self.tick(gripper="open")
        self.assertGreater(cmd.targets_rad[6],1.)
        self.assertGreater(cmd.target_velocities_rad_s[6],0.)

    def test_centered_joint_immediately_holds_while_gripper_continues(self):
        self.ready()
        moving = self.tick((.1,0,0,0,0,0),gripper="open")
        centered = self.tick((0,)*6,gripper="open")
        self.assertEqual(centered.phase,"ready")
        self.assertEqual(centered.targets_rad[0],moving.targets_rad[0])
        self.assertEqual(centered.target_velocities_rad_s[0],0.)
        self.assertGreater(centered.targets_rad[6],moving.targets_rad[6])

    def test_released_gripper_immediately_holds_while_joint_continues(self):
        self.ready()
        moving = self.tick((.1,0,0,0,0,0),gripper="open")
        released = self.tick((.1,0,0,0,0,0),gripper="hold")
        self.assertEqual(released.targets_rad[6],moving.targets_rad[6])
        self.assertEqual(released.target_velocities_rad_s[6],0.)
        self.assertGreater(released.targets_rad[0],moving.targets_rad[0])

    def test_all_zero_request_never_integrates_a_stopping_tail(self):
        self.ready()
        moving = self.tick((.1,0,0,0,0,0),gripper="open")
        stopped = self.tick((0,)*6,gripper="hold")
        self.assertEqual(stopped.targets_rad,moving.targets_rad)
        self.assertEqual(stopped.target_velocities_rad_s,(0.,)*7)

    def test_invalid_gripper_intent_faults(self):
        self.ready()
        self.assertEqual(self.tick(gripper="calibrate").phase,"fault")

    def test_initial_j3_negative_faults_without_automatic_reverse(self):
        self.ready()
        cmd = self.tick((0,0,-.1,0,0,0))
        self.assertEqual(cmd.phase,"fault")
        self.assert_hold(cmd)
        self.assertEqual(cmd.targets_rad[2],.3)

    def test_positive_j3_then_unverified_reverse_denied(self):
        self.planner.config['j3_reverse_motion_verified']=False
        self.ready()
        self.tick((0,0,.1,0,0,0))
        cmd=self.tick((0,0,-.1,0,0,0))
        self.assertEqual(cmd.phase,'fault')

    def test_positive_j3_plan_without_measured_movement_never_unlocks_reverse(self):
        self.ready()
        origin=self.planner.origin_rad
        cmd=self.tick((0,0,.1,0,0,0))
        self.assertGreater(cmd.targets_rad[2],origin[2])
        self.assertFalse(self.planner._j3_increased)
        self.q=origin
        self.v=(0.,)*7
        cmd=self.tick((0,0,-.1,0,0,0))
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)

    def test_only_fresh_measured_j3_increase_can_establish_positive_motion(self):
        self.ready()
        self.tick((0,0,.1,0,0,0))
        measured=list(self.planner.origin_rad); measured[2]+=.001
        self.q=tuple(measured)
        self.v=(0.,)*7
        cmd=self.tick((0,0,-.1,0,0,0))
        self.assertTrue(self.planner._j3_increased)
        self.assertEqual(cmd.kind,'bounded_native_targets')
        self.assertNotEqual(cmd.phase,'fault')

    def test_out_of_bounds_feedback_faults(self):
        self.ready()
        cmd=self.tick(feedback_changes={'positions_rad':(2.1,)+self.q[1:]})
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)

    def test_candidate_never_crosses_native_limit(self):
        self.q=(1.9999,)+self.q[1:]
        self.ready()
        cmd=self.tick((.4,0,0,0,0,0))
        self.assertEqual(cmd.phase,'fault')
        self.assertLessEqual(cmd.targets_rad[0],2.)

    def test_tracking_error_on_uncommanded_axis_faults(self):
        self.ready()
        q=list(self.q); q[5]+=.31
        cmd=self.tick(feedback_changes={'positions_rad':tuple(q)})
        self.assertEqual(cmd.phase,'fault')
        self.assertIn('J6',cmd.reason)

    def test_measured_excess_speed_faults(self):
        self.ready()
        cmd=self.tick(feedback_changes={'velocities_rad_s':(.5,)+(0.,)*6})
        self.assertEqual(cmd.phase,'fault')

    def test_nonfinite_control_faults(self):
        for value in (math.nan,math.inf,-math.inf,True):
            self.setUp(); self.ready()
            cmd=self.tick((value,0,0,0,0,0))
            self.assertEqual(cmd.phase,'fault')
            self.assert_hold(cmd)

    def test_missing_feedback_faults_without_disabling(self):
        self.ready()
        cmd=self.planner.step(self.planner.session_id,None,self.now+.02)
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)

    def test_feedback_gap_and_repeated_sequence_fault(self):
        self.ready()
        cmd=self.tick(dt=.3)
        self.assertEqual(cmd.phase,'fault')
        self.setUp(); self.ready()
        cmd=self.tick(feedback_changes={'sequence':self.seq})
        self.assertEqual(cmd.phase,'fault')

    def test_future_acquisition_time_faults(self):
        self.ready()
        cmd=self.tick(feedback_changes={'acquired_at':self.now+1})
        self.assertEqual(cmd.phase,'fault')

    def test_wrong_command_session_rejected_without_mutating_current(self):
        self.ready()
        with self.assertRaises(SessionError):
            self.planner.step('old',self.feedback(),self.now,(.2,)*6,'open',True)
        self.assertEqual(self.planner.phase,'ready')

    def test_interrupt_and_eof_latch_hold_never_home_or_disable(self):
        self.ready()
        self.tick((.1,0,0,0,0,0))
        cmd=self.planner.interrupt(self.planner.session_id,'EOF')
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)
        self.assertNotEqual(cmd.targets_rad,(0.,)*7)
        self.assertEqual(self.tick().phase,'fault')
        self.assertEqual(self.planner.resume(self.planner.session_id,self.feedback(),self.now,neutral_confirmed=True).phase,'fault')

    def test_fault_reset_needs_correct_session_fresh_data_and_keeps_origin(self):
        self.ready(); origin=self.planner.origin_rad
        self.planner.interrupt(self.planner.session_id)
        with self.assertRaises(SessionError):
            self.planner.reset_fault('old',self.feedback(),self.now)
        cmd=self.planner.reset_fault(self.planner.session_id,self.feedback(),self.now)
        self.assertEqual(cmd.phase,'paused')
        self.assertEqual(cmd.origin_rad,origin)
        self.assert_hold(cmd)

    def test_pause_keeps_pose_without_return(self):
        self.ready(); self.tick((.1,0,0,0,0,0))
        before=self.q
        cmd=self.planner.pause(self.planner.session_id)
        self.assert_hold(cmd)
        self.assertEqual(cmd.targets_rad,before)
        self.assertNotEqual(cmd.targets_rad,cmd.origin_rad)

    def test_unverified_return_path_denied(self):
        self.ready(); self.planner.pause(self.planner.session_id)
        self.planner.config['return_path_verified']=False
        cmd=self.planner.request_return_origin(self.planner.session_id,self.feedback(),self.now)
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)

    def test_return_request_requires_measured_standstill(self):
        self.ready(); self.planner.pause(self.planner.session_id)
        self.v=(.01,)+(0.,)*6
        cmd=self.planner.request_return_origin(self.planner.session_id,self.feedback(),self.now)
        self.assertEqual(cmd.phase,'fault')

    def test_return_request_checks_all_axis_tracking(self):
        self.ready(); self.planner.pause(self.planner.session_id)
        self.q=self.q[:4]+(self.q[4]+.31,)+self.q[5:]
        cmd=self.planner.request_return_origin(self.planner.session_id,self.feedback(),self.now)
        self.assertEqual(cmd.phase,'fault')

    def test_return_request_does_not_jump_to_origin(self):
        cmd=self.start_return()
        self.assertNotEqual(cmd.targets_rad,cmd.origin_rad)
        self.assertEqual(cmd.target_velocities_rad_s,(0.,)*7)

    def test_return_is_velocity_acceleration_bounded_and_not_zero_home(self):
        self.start_return(); last=(0.,)*7
        for _ in range(250):
            cmd=self.tick()
            self.assertNotEqual(cmd.phase,'fault',cmd.reason)
            for v,prev in zip(cmd.target_velocities_rad_s,last):
                self.assertLessEqual(abs(v),.4+1e-9)
                self.assertLessEqual(abs(v-prev),.8*.02+1e-8)
            last=cmd.target_velocities_rad_s
            if cmd.phase=='at_origin': break
        self.assertEqual(cmd.phase,'at_origin')
        self.assertEqual(cmd.targets_rad,(.1,.2,.3,-.1,.2,.1,1.))
        self.assertEqual(cmd.kind,'hold_current')

    def test_return_target_without_measured_settling_cannot_disable(self):
        self.start_return()
        for _ in range(90):
            self.v=(.01,)+(0.,)*6
            cmd=self.tick()
        self.assertEqual(cmd.phase,'returning')
        with self.assertRaises(SessionError):
            self.planner.confirm_support(self.planner.session_id,self.feedback(),self.now,supported=True)

    def test_only_stable_measured_origin_plus_support_can_disable(self):
        cmd=self.finish_return()
        with self.assertRaises(SessionError):
            self.planner.confirm_support(self.planner.session_id,self.feedback(),self.now,supported=False)
        cmd=self.planner.confirm_support(self.planner.session_id,self.feedback(),self.now,supported=True)
        self.assertEqual(cmd.kind,'disable_after_verified_support')
        self.assertEqual(cmd.phase,'disabled')
        self.assertFalse(cmd.hardware_writes_supported)
        with self.assertRaises(SessionError):
            self.planner.step(self.planner.session_id,self.feedback(),self.now)

    def test_drift_after_stability_prevents_disable(self):
        self.finish_return()
        self.q=(self.q[0]+.01,)+self.q[1:]
        cmd=self.planner.confirm_support(self.planner.session_id,self.feedback(),self.now,supported=True)
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)

    def test_support_needs_new_fresh_feedback(self):
        self.finish_return()
        f=self.feedback(axis_timestamps=(self.now-1,)*7)
        cmd=self.planner.confirm_support(self.planner.session_id,f,self.now,supported=True)
        self.assertEqual(cmd.phase,'fault')

    def test_new_fresh_feedback_after_long_gap_cannot_preserve_old_stability(self):
        self.finish_return()
        cmd=self.planner.confirm_support(self.planner.session_id,self.feedback(dt=5.),self.now,supported=True)
        self.assertEqual(cmd.phase,'fault')
        self.assert_hold(cmd)
        self.assertIsNone(self.planner._stable_since)
        self.assertIn('continuous stable feedback was interrupted',cmd.reason)

    def test_commands_are_immutable_json_serializable_data_only(self):
        cmd=self.capture()
        self.assertIsInstance(cmd,PlannedCommand)
        json.dumps(cmd.to_dict(),allow_nan=False)
        with self.assertRaises(AttributeError):
            cmd.kind='enable'


if __name__=='__main__':
    unittest.main(verbosity=2)
