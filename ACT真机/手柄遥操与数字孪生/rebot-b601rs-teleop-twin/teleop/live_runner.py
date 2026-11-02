"""Native gamepad session loop with a single live PCAN owner.

Default execution opens nothing. --observe only reads; --control additionally
requires --allow-enable and a commissioned production configuration. The small
J6 commissioning trial is a separate entry and cannot promote site candidates.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"input_core"))
from control_core import AXES, BUTTONS, Engine, Frame
from motion_transport import MotionTransport, TransportError
from native_pcan_backend import NativePcanBackend
from session_core import SessionError, SessionPlanner, config_errors


class NativeGamepadRunner:
    ACTIONS=frozenset(("pause","resume","return_origin","confirm_support","interrupt",
                       "confirm_external_power_off","shutdown","shutdown_origin"))

    def __init__(self,now,backend,config,*,allow_motion=False,clock=None,allow_synthetic_backend=False,input_engine=None):
        if type(allow_motion) is not bool:
            raise ValueError("motion authorization must be explicit bool")
        self.synthetic=bool(getattr(backend,"synthetic_only",False))
        if self.synthetic:
            from fake_robot import FakeRobot
            if allow_synthetic_backend is not True or not isinstance(backend,FakeRobot):
                raise ValueError("synthetic backend requires explicit test injection")
        elif not isinstance(backend,NativePcanBackend):
            raise ValueError("one NativePcanBackend required")
        self.backend=backend
        self.clock=clock or getattr(backend,"clock",time.monotonic)
        self.allow_motion=allow_motion
        self.engine=input_engine if input_engine is not None else Engine()
        self.planner=SessionPlanner(config,now,allow_synthetic_feedback=self.synthetic)
        self.transport=MotionTransport(backend.controller,backend.motors,self.planner,
                                      allow_synthetic_feedback=self.synthetic)
        self.started=self.closed=False
        self.last_tick=float(now)
        self.last_feedback=None
        self.last_input=self.engine.snapshot(now)
        self.last_error=self.action_error=None
        self.tick_count=0
        self.read_durations=[]
        self.exit_mode=None
        self.normal_disable_verified=False
        self.external_poweroff_acknowledgement=None
        self.status_write_error=None
        self.shutdown_reason=None
        self.shutdown_plan=None
        self.shutdown_destination='zero'
        self.last_user_activity=float(now)
        self.previous_activity_buttons=0
        self.inactivity_timeout_s=120.

    @property
    def session_id(self):
        return self.planner.session_id

    def _now(self):
        now=self.clock()
        if type(now) not in (int,float) or not math.isfinite(now):
            raise ValueError("finite monotonic time required")
        return float(now)

    def request_shutdown(self,reason='B/Esc/Ctrl+C',*,destination='zero'):
        if self.allow_motion and self.planner.config.get('admission_policy')=='tony-github-defaults-v1':
            if self.shutdown_reason is None:self.shutdown_destination=destination
            self.shutdown_reason=self.shutdown_reason or reason
        else:
            self.interrupt(reason)

    def _shutdown_tick(self,feedback,now,dt):
        from graceful_stop import GracefulStop
        self.planner.soft_limit_blocks=[None]*7
        if self.shutdown_plan is None:
            start=getattr(self.backend,'command',None) or self.transport.last_targets
            finish=self.planner.origin_rad if self.shutdown_destination=='origin' else None
            self.shutdown_plan=GracefulStop(start,self.planner.config,finish)
        positions,velocities=self.planner._feedback(feedback,now)
        targets,speeds=self.shutdown_plan.step(dt)
        self.planner._targets=targets
        self.planner._velocity=speeds
        self.planner._last_step=now
        self.planner.phase='returning'
        command=self.planner._command('bounded_native_targets','GitHub smooth zero shutdown in progress')
        if hasattr(self.backend,'set_speed_scale'):self.backend.set_speed_scale(1.)
        self.transport.apply(command,feedback,self._now())
        streamed=getattr(self.backend,'command',None) or self.transport.last_targets
        if self.shutdown_plan.ready_to_disable(now,streamed):
            if not self.synthetic:
                newest=self.backend.authorize_github_finish_disable(self.session_id,feedback,self.shutdown_plan.finish)
                if newest is not None:
                    feedback=newest
                    now=self._now()
                    self.planner._feedback(feedback,now)
                    self.planner._last_step=now
            self.planner.phase='disabled'
            self.planner._velocity=(0.,)*7
            command=self.planner._command('disable_after_github_finish','selected finish trajectory and settle completed; disable')
            self.transport.apply(command,feedback,self._now())
            self.transport.close_after_disable()
            self.closed=True
            self.normal_disable_verified=True
            self.exit_mode='github_origin_disable' if self.shutdown_destination=='origin' else 'github_smooth_zero_disable'
        return self.snapshot()

    def _sample(self,requested):
        before=self._now()
        feedback=self.backend.read_feedback(self.session_id,requested)
        # Real requests take time. Validate and send against the time AFTER RX.
        now=self._now()
        self.read_durations.append(now-before)
        self.read_durations=self.read_durations[-256:]
        self.last_feedback=feedback
        return feedback,now

    def start(self,now=None):
        if self.started or self.closed:
            if self.started and not self.closed:
                self._fault("a running session cannot recapture its origin")
            raise SessionError("new session required for another startup")
        self.started=True
        requested=self._now() if now is None else now
        try:
            feedback,actual=self._sample(requested)
            self.last_tick=actual
            command=self.planner.capture_origin(feedback,actual)
            if command.phase=="fault":
                self._fault(command.reason)
            elif self.allow_motion:
                if command.admission_errors:
                    raise TransportError("motion startup blocked: "+"; ".join(command.admission_errors))
                if not self.synthetic:
                    self.backend.authorize_motion(self.session_id,feedback,self._now(),allow_motion=True)
                self.transport.arm(feedback,self._now(),allow_motion=True)
        except Exception as exc:
            self._fault(f"startup: {type(exc).__name__}: {exc}")
        return self.snapshot()

    @staticmethod
    def _neutral(snapshot):
        inp,ctrl=snapshot["input"],snapshot["control"]
        axes=inp["axes"]
        return (inp["connected"] and inp["age_ms"] is not None and inp["age_ms"]<=250
            and ctrl["gate"] not in ("disconnected","input_stale","neutral_required")
            and all(math.isfinite(axes[a]) and abs(axes[a])<=ctrl["deadzone"] for a in AXES[:4])
            and all(math.isfinite(axes[a]) and 0<=axes[a]<=ctrl["deadzone"] for a in AXES[4:])
            and not set(inp["buttons"])&({"LB","B"} if "a_hold_seconds" in ctrl and ctrl.get('upstream_user_keys',False) else {"LB","A","X","B"})
            and ctrl.get("neutral_confirmed",True) is True)

    def _owned_hold(self):
        if self.transport.possibly_enabled:
            try:
                self.transport.heartbeat_hold()
            except Exception as exc:
                self.last_error=f"{self.last_error or 'hold'}; physical hold unverified: {type(exc).__name__}: {exc}"

    def _fault(self,reason):
        self.last_error=str(reason)
        self.engine.set_paused(True)
        self.last_input=self.engine.snapshot(self.last_tick)
        if self.planner.phase!="disabled":
            self.planner.interrupt(self.session_id,str(reason))
        self._owned_hold()

    def interrupt(self,reason="EOF/interrupt/exception"):
        if not self.closed:
            self._fault(reason)
        return self.snapshot()

    def _confirm_external_power_off(self,session_id,*,supported=False,motor_power_off_confirmed=False):
        if (session_id!=self.session_id or supported is not True
                or motor_power_off_confirmed is not True):
            raise SessionError("current session plus explicit support AND external motor-power-off confirmation required")
        # This is an acknowledgement of a physical action already performed by
        # the operator, never an instruction to switch power or disable motors.
        if self.synthetic:
            self.backend.enabled=False
            self.backend.calls.append({"kind":"acknowledge_external_power_off",
                "source":"explicit_test_external_power_off_confirmation",
                "session_id":session_id,"supported":True,"motor_power_off_confirmed":True,
                "normal_disable_verified":False})
        else:
            self.backend.acknowledge_external_power_off(session_id,supported=True,
                motor_power_off_confirmed=True)
        self.external_poweroff_acknowledgement={"session_id":session_id,"supported":True,
            "motor_power_off_confirmed":True,"normal_disable_verified":False,
            "source":"explicit_current_session_human_confirmation"}
        self.transport.armed=False
        self.transport.possibly_enabled=False
        self.engine.set_paused(True)
        self.last_input=self.engine.snapshot(self.last_tick)
        self.exit_mode="external_poweroff_handover"
        self.normal_disable_verified=False
        self.transport.close_after_disable()
        self.closed=True

    def step(self,frame,now=None,*,action=None,session_id=None,supported=False,
             motor_power_off_confirmed=False):
        if not self.started or self.closed:
            raise SessionError("start a fresh non-closed session before stepping")
        self.tick_count+=1
        self.action_error=None
        try:
            if action is not None and (session_id!=self.session_id or action not in self.ACTIONS):
                raise SessionError("unknown action or wrong current session_id")
            if action=="confirm_external_power_off":
                self._confirm_external_power_off(session_id,supported=supported,
                    motor_power_off_confirmed=motor_power_off_confirmed)
                return self.snapshot()
            if self.planner.phase=="fault" or self.transport.fault:
                if hasattr(self.backend,'recover_feedback_fault'):
                    actual=self._now()
                    self.last_tick=actual
                    if actual-self.last_user_activity>=self.inactivity_timeout_s:
                        self.request_shutdown('120 seconds idle; awaiting fault resolution before zero return')
                    self.last_input=self.engine.update(frame,actual)
                    requested_action=self.engine.consume_action()
                    if (action=='resume' or requested_action=='resume') and self._neutral(self.last_input):
                        try:
                            self.backend.recover_feedback_fault(self.session_id)
                            feedback,actual=self._sample(self._now())
                            reset=self.planner.reset_fault(self.session_id,feedback,actual)
                            if reset.phase=='fault':raise SessionError(reset.reason)
                            self.transport.fault=None
                            self.last_error=None
                            self.last_tick=actual
                            # Reset only to paused. Resume with a separate fresh
                            # observation; do not re-enable or recapture origin.
                            feedback,actual=self._sample(self._now())
                            self.planner.resume(self.session_id,feedback,actual,neutral_confirmed=True)
                            self.last_tick=actual
                            if self.planner.phase=='fault':raise SessionError(self.planner.latest_command.reason)
                            self.transport.last_targets=tuple(self.backend.command)
                            self.backend.resume_stream(self.session_id)
                        except Exception as exc:
                            self.action_error=f'resume rejected: {exc}'
                            self.engine.set_paused(True)
                    else:
                        self.engine.set_paused(True)
                else:
                    self.engine.set_paused(True)
                    self.last_input=self.engine.snapshot(self._now())
                self._owned_hold()
                return self.snapshot()
            requested=self._now() if now is None else now
            feedback,actual=self._sample(requested)
            gap=actual-self.last_tick
            self.last_tick=actual
            if action=='shutdown':self.request_shutdown('operator shutdown')
            if action=='shutdown_origin':self.request_shutdown('A held for 3 seconds',destination='origin')
            if (self.shutdown_reason and self.shutdown_destination=='origin'
                    and frame.connected and 0<=actual-frame.acquired_at<=.25
                    and type(frame.buttons) is int and frame.buttons&BUTTONS['B']):
                self.shutdown_destination='zero'
                self.shutdown_reason='B superseded origin shutdown'
                self.shutdown_plan=None
            if self.shutdown_reason:
                return self._shutdown_tick(feedback,actual,gap)
            if not self.synthetic and self.allow_motion and self.planner.config.get('admission_policy')=='tony-github-defaults-v1':
                if self.backend.thermal_disconnect_requested:
                    self.backend.github_thermal_emergency_disconnect()
                    self.transport.armed=self.transport.possibly_enabled=False
                    self.engine.set_paused(True)
                    self.normal_disable_verified=False
                    self.exit_mode='github_thermal_emergency_disable'
                    self.transport.close_after_disable()
                    self.closed=True
                    return self.snapshot()
                if self.backend.thermal_return_requested:
                    # Upstream high-temperature stop takes priority over handpad
                    # input. Return destination remains the user's session origin.
                    frame=Frame(True,acquired_at=actual)
                    self.engine.set_paused(False)
                    action='return_origin' if self.planner.phase not in ('returning','at_origin') else None
            if action in ("resume","return_origin"):
                self.engine.set_paused(False)
            if action in ("pause","interrupt"):
                self.engine.set_paused(True)
            # The input frame may have expired while the feedback read blocked.
            self.last_input=self.engine.update(frame,actual)
            if hasattr(self.engine,"consume_action"):
                requested_action=self.engine.consume_action()
                if action is None and requested_action in ("pause","resume","return_origin","shutdown","shutdown_origin"):
                    action=requested_action
            if gap<=0 or (self.planner.config.get('admission_policy')!='tony-github-defaults-v1' and gap>self.planner.config["max_control_interval_s"]):
                raise SessionError("control interval invalid; backlog discarded")
            if action=="interrupt":
                self._fault("operator interrupt; retain controller-owned hold")
                return self.snapshot()
            ctrl=self.last_input["control"]
            if self.allow_motion and self.planner.config.get('admission_policy')=='tony-github-defaults-v1':
                inp=self.last_input['input']
                buttons=frame.buttons if type(frame.buttons) is int else 0
                fresh=inp['connected'] and inp['age_ms'] is not None and 0<=inp['age_ms']<=250
                activity=any(ctrl['joint_velocity_rad_s']) or ctrl['gripper_intent']!='hold' or bool(buttons&~self.previous_activity_buttons)
                if fresh and activity:self.last_user_activity=actual
                self.previous_activity_buttons=buttons
                if action=='shutdown':self.request_shutdown('B stop button')
                elif action=='shutdown_origin':self.request_shutdown('A held for 3 seconds',destination='origin')
                elif actual-self.last_user_activity>=self.inactivity_timeout_s:self.request_shutdown('120 seconds without operator commands')
                if self.shutdown_reason:return self._shutdown_tick(feedback,actual,gap)
            if hasattr(self.backend,'set_speed_scale'):
                self.backend.set_speed_scale(ctrl.get('speed_scale',.25))
            gate=ctrl["gate"]
            if not self.allow_motion:
                # This session cannot be promoted through a file or button.
                self.planner.pause(self.session_id,"read-only observation; motion disabled for this entire session")
                self.planner.step(self.session_id,feedback,actual,(0.0,)*6,"hold",False)
                return self.snapshot()
            if action in ("resume","return_origin","confirm_support") and not self._neutral(self.last_input):
                self.action_error="release LB/B/A/X and return sticks/triggers to neutral"
                self.engine.set_paused(True)
                self.planner.pause(self.session_id,self.action_error)
                action=None
            if self.engine.paused or gate in ("disconnected","input_stale","neutral_required"):
                self.planner.pause(self.session_id,f"input gate {gate}; return cancelled")
            if self.planner.phase in ("returning","at_origin") and not self._neutral(self.last_input):
                self.planner.pause(self.session_id,"input intervention cancelled return")
                self.engine.set_paused(True)
            if action=="resume":
                command=self.planner.resume(self.session_id,feedback,actual,neutral_confirmed=True)
            elif action=="return_origin":
                command=self.planner.request_return_origin(self.session_id,feedback,actual)
            elif action=="confirm_support":
                command=self.planner.confirm_support(self.session_id,feedback,actual,supported=supported)
                if command.kind=="disable_after_verified_support" and not self.synthetic:
                    self.backend.authorize_supported_disable(self.session_id,feedback,self._now(),supported=supported)
            else:
                velocity=tuple(ctrl["joint_velocity_rad_s"])
                gripper=ctrl["gripper_intent"]
                permit=(ctrl["lb_held"] and ctrl["armed"] and not self.engine.paused
                        and gate in ("ready","active") and (any(velocity) or gripper!="hold"))
                command=self.planner.step(self.session_id,feedback,actual,velocity,gripper,permit,
                    gripper_velocity_rad_s=ctrl.get("gripper_velocity_rad_s"))
            if not self.transport.armed:
                raise TransportError("startup did not acquire execution ownership; new session required")
            self.transport.apply(command,feedback,self._now())
            if command.phase=="fault":
                self._fault(command.reason)
            elif command.phase=="paused" and command.admission_errors:
                self.engine.set_paused(True)
            self.last_input=self.engine.snapshot(self._now())
            if command.kind=="disable_after_verified_support" and not self.transport.possibly_enabled:
                self.exit_mode="normal_supported_disable"
                self.normal_disable_verified=True
                self.transport.close_after_disable()
                self.closed=True
        except Exception as exc:
            self._fault(f"loop: {type(exc).__name__}: {exc}")
        return self.snapshot()

    def close_observer(self):
        if self.allow_motion or self.transport.possibly_enabled:
            raise SessionError("execution ownership cannot use observation cleanup")
        self.backend.close_bus()
        for motor in self.backend.motors:
            motor.close()
        self.backend.close()
        self.closed=True
        self.exit_mode="observation_closed"

    def snapshot(self):
        command=self.planner.latest_command
        phase=self.planner.phase
        if phase=="disabled" and self.transport.possibly_enabled:
            phase="fault_holding"
        elif phase=="disabled" and not self.closed and self.last_error:
            phase="cleanup_fault"
        if self.exit_mode=="external_poweroff_handover":
            phase="external_poweroff_handover" if self.closed else "cleanup_fault"
        return {"kind":"native_gamepad_session_loop","session_id":self.session_id,
            "admission_policy":self.planner.config.get('admission_policy','commissioned_site'),
            "measured_return_tolerance_verified":(self.normal_disable_verified and self.planner.config.get('admission_policy')!='tony-github-defaults-v1'),
            "hottest_temperature_c":getattr(self.backend,'hottest_temperature_c',None),
            "thermal_return_requested":getattr(self.backend,'thermal_return_requested',False),
            "read_only":not self.allow_motion,"synthetic_test_backend":self.synthetic,
            "hardware_motion_supported":self.allow_motion and not self.synthetic,
            "tick_count":self.tick_count,"phase":phase,"planner_phase":self.planner.phase,"closed":self.closed,
            "origin_rad":self.planner.origin_rad,
            "origin_scope":"this running control session" if self.allow_motion else "observation only; never reuse for future control",
            "positions_native_rad":self.last_feedback.positions_rad if self.last_feedback else None,
            "velocities_native_rad_s":self.last_feedback.velocities_rad_s if self.last_feedback else None,
            "command":command.to_dict() if command else None,"input":copy.deepcopy(self.last_input),
            "transport":{"armed":self.transport.armed,"possibly_enabled":self.transport.possibly_enabled,
                "fault":self.transport.fault,"last_targets_rad":self.transport.last_targets,
                "hold_send_errors":self.transport.last_send_errors[:]},
            "last_error":self.last_error,"action_error":self.action_error,
            "status_write_error":self.status_write_error,
            "shutdown_reason":self.shutdown_reason,
            "shutdown_destination":self.shutdown_destination,
            "idle_seconds":max(0.,self.last_tick-self.last_user_activity),
            "inactivity_timeout_s":self.inactivity_timeout_s,
            "soft_limit_blocks":getattr(self.planner,'soft_limit_blocks',[None]*7),
            "streaming":self.backend.timing() if hasattr(self.backend,'timing') else None,
            "exit_mode":self.exit_mode,"normal_disable_verified":self.normal_disable_verified,
            "external_poweroff_acknowledgement":copy.deepcopy(self.external_poweroff_acknowledgement),
            "max_recent_read_duration_s":max(self.read_durations,default=0.0),
            "process_death_hold_guaranteed":False}


def load_xinput_reader(slot):
    # Reuse only the existing read-only input factory, not its fake robot loop.
    from unified_runner import load_xinput_reader as factory
    return factory(slot)


def save_atomic(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def poll_command(path,session_id,seen):
    if not path.exists():
        return None,False
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict) or value.get("session_id")!=session_id:
        raise SessionError("command requires current console session_id")
    command_id=value.get("command_id")
    if not isinstance(command_id,str) or not command_id:
        raise SessionError("command requires a unique nonempty command_id")
    if command_id in seen:
        return None,False
    if value.get("action") not in NativeGamepadRunner.ACTIONS:
        raise SessionError("unsupported operator action")
    if value["action"]=="confirm_external_power_off" and (
            value.get("supported") is not True or value.get("motor_power_off_confirmed") is not True):
        raise SessionError("external power-off handover requires explicit supported=true AND motor_power_off_confirmed=true")
    seen.add(command_id)
    return value["action"],value.get("supported") is True


def run_session_loop(runner,reader,command_path,status_path,started,duration,seen,trace,
                     *,fatal_error=None,duration_triggered=False):
    """Keep processing the same command/status files while ownership remains.

    Duration, bad files, input failures and Ctrl+C request a latched hold. They
    never infer physical power removal or make an owned process return.
    """
    fatal=fatal_error
    schedule_clock=time.perf_counter if getattr(getattr(runner,'backend',None),'high_rate',False) else time.monotonic
    next_input=schedule_clock()
    while not runner.closed:
        state=None
        try:
            if getattr(getattr(runner,'backend',None),'high_rate',False):
                next_input+=.02
                current=schedule_clock()
                if next_input<current:next_input=current
                time.sleep(max(0.,next_input-current))
            else:
                time.sleep(.02)
            action,supported=poll_command(command_path,runner.session_id,seen)
            if not duration_triggered and time.monotonic()-started>=duration:
                duration_triggered=True
                if not runner.transport.possibly_enabled:
                    break
                # Trigger once; a valid operator command in this or a later
                # iteration must not be overwritten by another timeout.
                if action!="confirm_external_power_off":
                    if getattr(getattr(runner,'planner',None),'config',{}).get('admission_policy')=='tony-github-defaults-v1':
                        runner.request_shutdown('session duration elapsed')
                    else:
                        runner.interrupt("session duration elapsed; retain controller-owned hold")
            # The confirmed external handover must work even if input or motor
            # feedback can no longer be read after power was removed.
            if action=="confirm_external_power_off":
                frame=Frame(connected=False,acquired_at=time.monotonic())
            else:
                frame=reader.read() if reader else Frame(connected=False,acquired_at=time.monotonic())
                if getattr(runner.backend,'high_rate',False):
                    frame.acquired_at=runner._now()  # XInput read completed now, in the QPC clock domain.
                    import ctypes
                    if hasattr(ctypes,'windll') and ctypes.windll.user32.GetAsyncKeyState(0x1B)&0x8000:
                        action='shutdown'
            state=runner.step(frame,action=action,
                session_id=runner.session_id if action else None,supported=supported,
                motor_power_off_confirmed=action=="confirm_external_power_off")
        except (KeyboardInterrupt,EOFError) as exc:
            if getattr(getattr(runner,'planner',None),'config',{}).get('admission_policy')=='tony-github-defaults-v1':
                runner.request_shutdown(type(exc).__name__)
                state=runner.snapshot()
            else:
                fatal=f"{type(exc).__name__}; retain owned hold"
                state=runner.interrupt(fatal)
        except Exception as exc:
            fatal=f"{type(exc).__name__}: {exc}"
            state=runner.interrupt(fatal)
        if state is not None:
            trace.append(state)
            trace[:]=trace[-300:]
            try:
                save_atomic(status_path,state)
                runner.status_write_error=None
            except OSError as exc:
                # Windows readers/antivirus may briefly deny rename. UI log IO
                # is not actuator IO: keep servicing the same controller and
                # retry the next snapshot instead of faulting the robot loop.
                if getattr(getattr(runner,'planner',None),'config',{}).get('admission_policy')=='tony-github-defaults-v1':
                    runner.status_write_error=f'{type(exc).__name__}: {exc}'
                else:
                    fatal=f"status update failed: {type(exc).__name__}: {exc}"
                    runner.interrupt(fatal)
            except (Exception,KeyboardInterrupt) as exc:
                fatal=f"status update failed: {type(exc).__name__}: {exc}"
                runner.interrupt(fatal)
        if state and state.get("last_error") and not runner.transport.possibly_enabled:
            fatal=fatal or state["last_error"]
        if not runner.transport.possibly_enabled and (fatal or runner.closed):
            break
    return fatal,duration_triggered


def main(argv=None,*,input_engine_factory=None,backend_factory=None,result_callback=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--observe",action="store_true")
    mode.add_argument("--control",action="store_true")
    parser.add_argument("--allow-enable",action="store_true")
    parser.add_argument("--xinput",action="store_true")
    parser.add_argument("--slot",type=int,choices=range(4),default=0)
    parser.add_argument("--config",type=Path,default=ROOT/"site_config.candidate_20261030.json")
    parser.add_argument("--duration",type=float,default=10.0)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args(argv)
    if not args.observe and not args.control:
        print(json.dumps({"mode":"description_only","can_opened":False,
            "message":"--observe reads real state only; --control --allow-enable requires commissioned site admission. J6 first-hold trial is separate."}))
        return 0
    if not math.isfinite(args.duration) or not (0<args.duration<=3600 or (args.control and args.duration==0)):
        parser.error("duration must be finite in (0,3600]; control duration=0 disables the fixed session deadline")
    duration_limit=args.duration if args.duration>0 else float('inf')
    config=json.loads(args.config.read_text(encoding="utf-8"))
    backend=(backend_factory or NativePcanBackend)(config)
    if args.control:
        errors=backend.admission_errors()
        if not args.allow_enable or errors:
            print(json.dumps({"mode":"control_startup_blocked","can_opened":False,
                "motion_enable_requested":False,"admission_errors":errors,
                "allow_enable_supplied":args.allow_enable},ensure_ascii=False))
            return 2
    elif args.allow_enable:
        parser.error("observation cannot be promoted by --allow-enable")
    reader=load_xinput_reader(args.slot) if args.xinput else None
    runner=None
    trace=[]
    fatal=None
    cleanup_error=None
    duration_triggered=False
    seen=set()
    status_path=command_path=None
    try:
        backend.open(allow_can_open=True)
        duration_started=time.monotonic()
        started=backend.clock() if getattr(backend,'high_rate',False) else time.monotonic()
        runner=NativeGamepadRunner(started,backend,config,allow_motion=args.control,
            input_engine=input_engine_factory(config) if input_engine_factory else None)
        runtime=ROOT/"runtime"/("live_"+runner.session_id)
        status_path=runtime/"status.json"
        command_path=runtime/"command.json"
        print(json.dumps({"session_id":runner.session_id,"status_file":str(status_path),
                          "command_file":str(command_path),"read_only":not args.control},ensure_ascii=False),flush=True)
        # Publish the handover address before startup can acquire ownership.
        state=runner.start(started)
        if hasattr(backend,'start_streaming') and not runner.last_error and runner.transport.armed:
            backend.set_speed_scale(runner.engine.snapshot(backend.clock())['control'].get('speed_scale',.25))
            backend.start_streaming()
        try:
            save_atomic(status_path,state)
        except OSError as exc:
            if config.get('admission_policy')!='tony-github-defaults-v1':
                raise
            runner.status_write_error=f'{type(exc).__name__}: {exc}'
        fatal,duration_triggered=run_session_loop(runner,reader,command_path,status_path,
            duration_started,duration_limit,seen,trace,duration_triggered=duration_triggered)
        # Returning from this function while owning an enabled arm is forbidden.
    except (Exception,KeyboardInterrupt) as exc:
        fatal=f"{type(exc).__name__}: {exc}"
        if runner and runner.transport.possibly_enabled:
            runner.interrupt(fatal)
            # Preserve the same operator command/status paths even after an
            # outer failure. A bare hold loop would strand manual handover.
            fatal,duration_triggered=run_session_loop(runner,reader,command_path,status_path,
                duration_started,duration_limit,seen,trace,fatal_error=fatal,
                duration_triggered=duration_triggered)
    finally:
        try:
            if runner and not runner.closed and not runner.allow_motion:
                runner.close_observer()
            elif backend.opened and not backend.possibly_enabled:
                backend.close_bus()
                for motor in backend.motors:
                    motor.close()
                backend.close()
        except (Exception,KeyboardInterrupt) as exc:
            cleanup_error=f"{type(exc).__name__}: {exc}"
    result={"kind":"native_gamepad_loop_observation" if args.observe else "native_gamepad_control_session",
        "fatal_error":fatal,"cleanup_error":cleanup_error,
        "physical_motion_acceptance_verified":False,
        "final":runner.snapshot() if runner else None,"trace":trace,
        "backend":backend.diagnostics()}
    if args.out:
        save_atomic(args.out,result)
    if result_callback is not None:result_callback(result)
    print(json.dumps({"out":str(args.out) if args.out else None,"fatal_error":fatal,
                      "cleanup_error":cleanup_error,
                      "final_phase":result["final"]["phase"] if result["final"] else None},ensure_ascii=False))
    return 0 if not fatal and not cleanup_error and runner and not runner.last_error else 2


if __name__=="__main__":
    raise SystemExit(main())
