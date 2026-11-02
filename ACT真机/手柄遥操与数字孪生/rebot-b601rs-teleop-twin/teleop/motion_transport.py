"""Explicit actuator-method adapter, tested with injected fake SDK objects only.

It cannot open CAN and is never loaded by the offline preview. A future caller
must supply a single controller, seven motor handles, fully verified site config,
session-bound fresh physical feedback/status, and explicit motion authorization.
This is NOT an accepted hardware control loop or a hardware safety watchdog.
"""
import math
from session_core import SessionError


class TransportError(RuntimeError):
    pass


class MotionTransport:
    def __init__(self, controller, motors, planner, *, allow_synthetic_feedback=False):
        if len(motors) != 7:
            raise TransportError("exactly seven handles from one controller required")
        self.controller, self.motors, self.planner = controller, tuple(motors), planner
        self.allow_synthetic_feedback = allow_synthetic_feedback
        self.armed = False
        self.possibly_enabled = False
        self.fault = None
        self.last_targets = None
        self.last_send_errors = []
        self._consumed_command = None

    def _feedback_check(self, feedback, now):
        if feedback.session_id != self.planner.session_id:
            raise TransportError("feedback belongs to another session")
        valid_sources = {"live_native"}
        if self.allow_synthetic_feedback:
            valid_sources.add("synthetic_fixture")
        if feedback.source not in valid_sources:
            raise TransportError("only this session's live native reads are accepted")
        vectors = (feedback.positions_rad, feedback.velocities_rad_s, feedback.axis_timestamps,
                   feedback.fault_codes, feedback.control_modes, feedback.status_timestamps)
        if any(len(v) != 7 for v in vectors):
            raise TransportError("seven fresh native position/velocity/status values required")
        timeout = self.planner.config["feedback_max_age_s"]
        timestamps = tuple(feedback.axis_timestamps)+tuple(feedback.status_timestamps)+(feedback.acquired_at,)
        for stamp in timestamps:
            if not isinstance(stamp, (int,float)) or isinstance(stamp,bool) or not math.isfinite(stamp):
                raise TransportError("unknown physical status timestamp")
            if not self.planner.opened_at <= stamp <= now or now-stamp > timeout:
                raise TransportError("physical feedback/status is stale or pre-session")
        if any(code != 0 or isinstance(code,bool) for code in feedback.fault_codes):
            raise TransportError("motor faults are nonzero or unknown")
        if any(mode != "MIT" for mode in feedback.control_modes):
            raise TransportError("MIT mode must be freshly verified; this adapter never switches modes")
        for i,(q,v,joint) in enumerate(zip(feedback.positions_rad, feedback.velocities_rad_s, self.planner.config["joints"])):
            if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (q,v)):
                raise TransportError("non-finite physical feedback")
            if self.planner.config.get('admission_policy')!='tony-github-defaults-v1' and not joint["native_min_rad"] <= q <= joint["native_max_rad"]:
                raise TransportError(f"J{i+1} physical feedback outside verified range")
            if self.planner.config['velocity_tolerance_rad_s'] is not None and abs(v) > joint["max_speed_rad_s"]+self.planner.config["velocity_tolerance_rad_s"]:
                raise TransportError(f"J{i+1} physical velocity outside verified range")

    def _send(self, targets, velocities):
        # A failed/partial send is never turned into success, automatic return,
        # automatic disable, controller release, zeroing, or mode changes.
        sent_targets = list(self.last_targets if self.last_targets is not None else targets)
        for index, (motor, q, v, joint) in enumerate(zip(self.motors, targets, velocities, self.planner.config["joints"])):
            wire_velocity=0.0 if self.planner.config.get('admission_policy')=='tony-github-defaults-v1' else v
            motor.send_mit(pos=q, vel=wire_velocity, kp=joint["kp"], kd=joint["kd"], tau=0.0)
            # Preserve each successful submission after a later-axis failure.
            # Rolling the whole vector back can reverse an already submitted J3.
            sent_targets[index] = q
            self.last_targets = tuple(sent_targets)

    def _latch(self, reason):
        self.fault = str(reason)
        try:
            self.planner.interrupt(self.planner.session_id, f"transport: {reason}")
        except SessionError:
            pass
        if self.possibly_enabled and self.last_targets is not None:
            self.heartbeat_hold()
        raise TransportError(self.fault)

    def arm(self, feedback, now, *, allow_motion=False):
        if allow_motion is not True:
            raise TransportError("explicit motion authorization is required")
        if self.armed or self.possibly_enabled or self.fault:
            raise TransportError("transport is already owned or faulted")
        if self.planner.admission_errors() or self.planner.origin_rad is None:
            raise TransportError("site values and a fresh immutable session origin are required")
        command = self.planner.latest_command
        if command is None or command.phase not in ("paused", "ready") or any(command.target_velocities_rad_s):
            raise TransportError("arm only from current stationary hold plan")
        self._feedback_check(feedback, now)
        if feedback is not self.planner.latest_feedback:
            raise TransportError("startup must use the exact freshly validated planning feedback")
        if self.planner.config['velocity_tolerance_rad_s'] is not None and any(abs(v) > self.planner.config["velocity_tolerance_rad_s"] for v in feedback.velocities_rad_s):
            raise TransportError("fresh measured standstill is required before enable")
        if command.targets_rad != tuple(feedback.positions_rad):
            raise TransportError("startup hold must exactly match this fresh measured pose")
        self.last_targets = tuple(feedback.positions_rad)
        # Ownership remains conservative even if enable_all only partially succeeds.
        self.possibly_enabled = True
        try:
            self._send(self.last_targets, (0.0,)*7)
            self.controller.enable_all()
            self._send(self.last_targets, (0.0,)*7)
        except Exception as exc:
            self._latch(f"startup/enable failed: {exc}")
        self.armed = True

    def apply(self, command, feedback, now):
        if not self.armed or self.fault:
            raise TransportError("transport is not armed or has a latched fault")
        if command is not self.planner.latest_command or command.session_id != self.planner.session_id or command.phase != self.planner.phase:
            self._latch("old, foreign, or replayed plan")
        if command.kind == "hold_current":
            self.last_targets = self.last_targets or command.targets_rad
            self.heartbeat_hold()
            if command.phase == "fault":
                self.fault = command.reason
            return
        try:
            if command is self._consumed_command:
                raise TransportError("non-hold plan has already been consumed")
            sampled = self.planner.last_sample_time
            if sampled is None or not 0 <= now-sampled <= self.planner.config["max_control_interval_s"]:
                raise TransportError("non-hold plan is too old; acquire and plan a new sample")
            if feedback is not self.planner.latest_feedback:
                raise TransportError("feedback and non-hold plan must be from the same validated sample")
            self._feedback_check(feedback, now)
            self._consumed_command = command
            if command.kind == "request_return_session_origin":
                # The request never jumps to the origin; subsequent bounded plans do the work.
                self.heartbeat_hold()
            elif command.kind == "bounded_native_targets":
                github=self.planner.config.get('admission_policy')=='tony-github-defaults-v1'
                for q,v,joint in zip(command.targets_rad,command.target_velocities_rad_s,self.planner.config["joints"]):
                    if not math.isfinite(q) or not math.isfinite(v) or (not github and not joint["native_min_rad"] <= q <= joint["native_max_rad"]) or abs(v) > joint["max_speed_rad_s"]+1e-9:
                        raise TransportError("planned target outside verified site limits")
                self._send(command.targets_rad, command.target_velocities_rad_s)
            elif command.kind in ('disable_after_github_zero','disable_after_github_finish'):
                if (self.planner.config.get('admission_policy')!='tony-github-defaults-v1'
                        or command.phase!='disabled' or tuple(command.targets_rad) not in ((0.,)*7,tuple(self.planner.origin_rad))
                        or any(abs(q-f)>1e-12 for q,f in zip(self.last_targets,command.targets_rad))):
                    raise TransportError('GitHub zero trajectory must complete before disable')
                self.controller.disable_all()
                self.armed=False
                self.possibly_enabled=False
            elif command.kind == "disable_after_verified_support":
                github=self.planner.config.get('admission_policy')=='tony-github-defaults-v1'
                if command.phase != "disabled" or (not github and (any(abs(a-b)>self.planner.config["position_tolerance_rad"] for a,b in zip(feedback.positions_rad,self.planner.origin_rad)) or any(abs(v)>self.planner.config["velocity_tolerance_rad_s"] for v in feedback.velocities_rad_s))):
                    raise TransportError("fresh stable session origin required at disable")
                self.controller.disable_all()
                self.armed = False
                self.possibly_enabled = False
            else:
                raise TransportError("unsupported planned action")
        except Exception as exc:
            self._latch(str(exc))

    def heartbeat_hold(self):
        if not self.possibly_enabled or self.last_targets is None:
            raise TransportError("no controller-owned target to hold")
        errors = []
        for i,(motor,q,joint) in enumerate(zip(self.motors,self.last_targets,self.planner.config["joints"]),1):
            try:
                motor.send_mit(pos=q, vel=0.0, kp=joint["kp"], kd=joint["kd"], tau=0.0)
            except Exception as exc:
                errors.append(f"J{i}: {type(exc).__name__}: {exc}")
        self.last_send_errors = errors
        if errors:
            self.fault = "; ".join(errors)
            raise TransportError("hold send failed; ownership retained, physical hold unverified: "+self.fault)

    def close_after_disable(self):
        if self.possibly_enabled or self.armed:
            raise TransportError("retain controller ownership until verified normal disable")
        self.controller.close_bus()
        for motor in self.motors:
            motor.close()
        self.controller.close()
