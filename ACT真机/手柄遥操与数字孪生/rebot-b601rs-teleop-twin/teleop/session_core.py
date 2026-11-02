"""Offline-only native-radian session planner, with no transport or motor API.

Outputs are inert descriptions for review/tests. This module is NOT a verified
hardware controller, and `hold_current` cannot keep hardware alive after a driver
or process failure. A real adapter, watchdog and physical acceptance remain due.
No training-coordinate mapping, zeroing, automatic zero-home or old runner is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import secrets

AXIS_COUNT = 7
HARDWARE_WRITES_SUPPORTED = False


class SessionError(ValueError):
    """Rejected request; not permission to send a hardware command."""


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _vector(values, size=AXIS_COUNT):
    try:
        values = tuple(values)
    except TypeError:
        raise SessionError("expected a finite numeric vector") from None
    if len(values) != size or not all(_finite(v) for v in values):
        raise SessionError(f"expected {size} finite numeric values")
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class NativeFeedback:
    session_id: str
    source: str
    acquired_at: float
    sequence: int
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    axis_timestamps: tuple[float, ...]
    fault_codes: tuple[int | None, ...] = (None,) * AXIS_COUNT
    control_modes: tuple[str | None, ...] = (None,) * AXIS_COUNT
    status_timestamps: tuple[float | None, ...] = (None,) * AXIS_COUNT


@dataclass(frozen=True)
class PlannedCommand:
    kind: str
    session_id: str
    phase: str
    reason: str
    targets_rad: tuple[float, ...] | None
    target_velocities_rad_s: tuple[float, ...]
    origin_rad: tuple[float, ...] | None
    faults: tuple[str, ...]
    admission_errors: tuple[str, ...]
    hardware_writes_supported: bool = False

    def to_dict(self):
        return asdict(self)


def config_errors(config):
    """Template unknowns remain unknown; this never manufactures site values."""
    if config.get('admission_policy')=='tony-github-defaults-v1':
        from tony_github_config import github_errors
        return github_errors(config)
    errors = []
    joints = config.get("joints")
    if not isinstance(joints, list) or len(joints) != AXIS_COUNT:
        return ["joints: seven configured native axes required"]
    for i, joint in enumerate(joints):
        label = f"J{i+1}"
        if not isinstance(joint, dict):
            errors.append(f"{label}: configuration missing")
            continue
        if joint.get("motor_id") != i+1 or joint.get("host_id") != 253:
            errors.append(f"{label}: motor/host identity mismatch")
        if joint.get("model") != ("rs-06" if i < 3 else "rs-00"):
            errors.append(f"{label}: model mismatch")
        low, high = joint.get("native_min_rad"), joint.get("native_max_rad")
        if not _finite(low) or not _finite(high) or not low < high:
            errors.append(f"{label}: native physical bounds unverified")
        for key in ("max_speed_rad_s", "max_accel_rad_s2", "kp"):
            if not _finite(joint.get(key)) or joint[key] <= 0:
                errors.append(f"{label}: {key} missing or invalid")
        if not _finite(joint.get("kd")) or joint["kd"] < 0:
            errors.append(f"{label}: kd missing or invalid")
    for key in ("feedback_max_age_s", "max_control_interval_s", "tracking_error_rad",
                "position_tolerance_rad", "velocity_tolerance_rad_s", "stable_dwell_s"):
        if not _finite(config.get(key)) or config[key] <= 0:
            errors.append(f"{key}: missing or invalid")
    if (_finite(config.get("max_control_interval_s")) and _finite(config.get("feedback_max_age_s"))
            and config["max_control_interval_s"] > config["feedback_max_age_s"]):
        errors.append("control interval must not exceed feedback freshness limit")
    if config.get("physical_parameters_verified") is not True:
        errors.append("physical parameters have not been verified")
    if config.get("site_path_verified") is not True:
        errors.append("native jogging path has not been verified")
    if config.get("native_coordinate") != "motor_native_rad":
        errors.append("only motor native rad is supported")
    return errors


class SessionPlanner:
    """State machine producing data only. A fresh origin is never overwritten."""

    def __init__(self, config, now, *, allow_synthetic_feedback=False):
        if not _finite(now):
            raise SessionError("session start time must be finite")
        self.config = copy.deepcopy(config)
        self.soft_limit_blocks=[None]*AXIS_COUNT
        self.opened_at = float(now)
        self.session_id = secrets.token_hex(16)
        self.allow_synthetic_feedback = bool(allow_synthetic_feedback)
        self.phase = "awaiting_origin"
        self._origin = None
        self._targets = None
        self._velocity = (0.0,) * AXIS_COUNT
        self._sequence = -1
        self._last_step = None
        self._faults = []
        self._return = None
        self._stable_since = None
        self._j3_increased = False
        self._status_errors = ("fresh per-axis fault status and control mode are unavailable",)
        self._latest_command = None
        self._latest_feedback = None

    @property
    def origin_rad(self):
        return self._origin

    @property
    def latest_command(self):
        """Adapter may only consume this exact newest object, never a saved plan."""
        return self._latest_command

    @property
    def last_sample_time(self):
        """Last accepted planning-cycle time for external command-age checks."""
        return self._last_step

    @property
    def latest_feedback(self):
        """The exact newest accepted frame; an adapter must not substitute one."""
        return self._latest_feedback

    def admission_errors(self):
        return tuple(config_errors(self.config)) + tuple(self._status_errors)

    def _session(self, session_id):
        if session_id != self.session_id:
            raise SessionError("wrong or missing current session_id")
        if self.phase == "disabled":
            raise SessionError("session is closed; a new session must acquire a new origin")

    def _command(self, kind="hold_current", reason=""):
        command = PlannedCommand(kind, self.session_id, self.phase, reason, self._targets,
                                 self._velocity, self._origin, tuple(self._faults),
                                 self.admission_errors())
        self._latest_command = command
        return command

    def _fault(self, reason):
        if reason not in self._faults:
            self._faults.append(reason)
        self.phase = "fault"
        self._velocity = (0.0,) * AXIS_COUNT
        self._return = None
        self._stable_since = None
        return self._command(reason=reason+"; retain controller-owned hold, do not auto-disable")

    def _feedback(self, feedback, now, *, require_bounds=True):
        if not _finite(now) or now < self.opened_at:
            raise SessionError("invalid or backwards session time")
        if not isinstance(feedback, NativeFeedback) or feedback.session_id != self.session_id:
            raise SessionError("feedback is not from this session")
        sources = {"live_native"}
        if self.allow_synthetic_feedback:
            sources.add("synthetic_fixture")
        if feedback.source not in sources:
            raise SessionError("log/preview/model state cannot be used as live feedback")
        if isinstance(feedback.sequence, bool) or not isinstance(feedback.sequence, int) or feedback.sequence <= self._sequence:
            raise SessionError("feedback must have a new read sequence")
        positions = _vector(feedback.positions_rad)
        velocities = _vector(feedback.velocities_rad_s)
        timestamps = _vector(feedback.axis_timestamps)
        timeout = self.config.get("feedback_max_age_s", .25)
        if not _finite(timeout) or timeout <= 0:
            timeout = .25  # read freshness only; never a motion authorization default
        if not _finite(feedback.acquired_at):
            raise SessionError("invalid feedback acquisition time")
        for stamp in timestamps + (float(feedback.acquired_at),):
            if stamp < self.opened_at or stamp > now+1e-6 or now-stamp > timeout:
                raise SessionError("all seven axes need fresh feedback acquired after session start")
        if (len(feedback.fault_codes) != AXIS_COUNT or len(feedback.control_modes) != AXIS_COUNT
                or len(feedback.status_timestamps) != AXIS_COUNT):
            raise SessionError("seven per-axis fault statuses and modes required; unknown must stay unknown")
        status_errors = []
        for i, (fault, mode, stamp) in enumerate(zip(feedback.fault_codes, feedback.control_modes, feedback.status_timestamps)):
            if fault is not None and (isinstance(fault, bool) or not isinstance(fault, int) or fault < 0):
                raise SessionError(f"J{i+1} invalid fault status")
            if fault is not None and fault != 0:
                raise SessionError(f"J{i+1} motor reports fault code {fault}")
            if fault is None:
                status_errors.append(f"J{i+1}: fresh motor fault status unknown")
            if mode != "MIT":
                status_errors.append(f"J{i+1}: fresh MIT control mode unverified")
            if not _finite(stamp) or stamp < self.opened_at or stamp > now+1e-6 or now-stamp > timeout:
                status_errors.append(f"J{i+1}: fault/mode status freshness unverified")
        self._status_errors = tuple(status_errors)
        if require_bounds and self.config.get('admission_policy')!='tony-github-defaults-v1':
            joints = self.config.get("joints", [])
            for i, (position, joint) in enumerate(zip(positions, joints)):
                low, high = joint.get("native_min_rad"), joint.get("native_max_rad")
                if _finite(low) and _finite(high) and not low <= position <= high:
                    raise SessionError(f"J{i+1} feedback outside verified native bounds")
        self._sequence = feedback.sequence
        self._latest_feedback = feedback
        return positions, velocities

    def capture_origin(self, feedback, now):
        """Read-only planning capture, with no enable or start-target jump."""
        if self._origin is not None:
            raise SessionError("origin is immutable; start a genuinely new session to recapture")
        if self.phase == "fault":
            return self._command(reason="initial capture fault is latched; explicit fault reset required")
        try:
            positions, _ = self._feedback(feedback, now)
        except (SessionError, TypeError, AttributeError) as exc:
            return self._fault(str(exc))
        self._origin = positions
        self._targets = positions
        self._velocity = (0.0,) * AXIS_COUNT
        self.phase = "paused"
        self._last_step = float(now)
        return self._command(reason="fresh seven-axis origin captured; hold current, explicit neutral resume required")

    def pause(self, session_id, reason="operator pause"):
        self._session(session_id)
        if self.phase == "fault":
            return self._command(reason="fault is latched; pause cannot reset it")
        self.phase = "paused"
        self._velocity = (0.0,) * AXIS_COUNT
        self._return = None
        self._stable_since = None
        return self._command(reason=reason)

    def interrupt(self, session_id, reason="interrupt/EOF/exception"):
        self._session(session_id)
        return self._fault(reason)

    def reset_fault(self, session_id, feedback, now):
        self._session(session_id)
        if self.phase != "fault":
            raise SessionError("no latched fault to reset")
        try:
            positions, _ = self._feedback(feedback, now)
        except (SessionError, TypeError, AttributeError) as exc:
            return self._fault(str(exc))
        self._faults.clear()
        self._targets = positions  # hold newly measured pose; never recapture origin
        self._velocity = (0.0,) * AXIS_COUNT
        self.phase = "paused"
        self._last_step = float(now)
        return self._command(reason="fault reset to paused hold; explicit neutral resume required")

    def resume(self, session_id, feedback, now, *, neutral_confirmed=False):
        self._session(session_id)
        if self.phase == "fault":
            return self._command(reason="fault reset is required before resume")
        # A valid-session resume request first cancels every old motion intent.
        # No subsequent rejection may retain a ready phase or nonzero velocity.
        self._velocity = (0.0,) * AXIS_COUNT
        self._return = None
        self._stable_since = None
        self.phase = "paused"
        if self._origin is None:
            raise SessionError("fresh origin has not been captured")
        if neutral_confirmed is not True:
            raise SessionError("LB released and controls neutral must be confirmed")
        try:
            positions, measured_velocity = self._feedback(feedback, now)
        except (SessionError, TypeError, AttributeError) as exc:
            return self._fault(str(exc))
        if self.admission_errors():
            return self._command(reason="motion admission blocked by unverified site configuration")
        try:
            self._tracking(positions, measured_velocity)
            if self.config["velocity_tolerance_rad_s"] is not None and any(abs(v) > self.config["velocity_tolerance_rad_s"] for v in measured_velocity):
                raise SessionError("resume requires measured standstill")
        except (SessionError, KeyError) as exc:
            return self._fault(str(exc))
        self._targets = positions
        self._velocity = (0.0,) * AXIS_COUNT
        self._last_step = float(now)
        self.phase = "ready"
        return self._command(reason="native jog planning ready; physical hold permit still required")

    def _tracking(self, positions, velocities):
        if self.config.get('admission_policy')=='tony-github-defaults-v1':
            return  # Upstream does not define a tracking-error commissioning gate.
        threshold = self.config["tracking_error_rad"]
        for i, (actual, target) in enumerate(zip(positions, self._targets)):
            if abs(actual-target) > threshold:
                raise SessionError(f"J{i+1} tracking error exceeds verified threshold")
        for i, (velocity, joint) in enumerate(zip(velocities, self.config["joints"])):
            if abs(velocity) > joint["max_speed_rad_s"] + self.config["velocity_tolerance_rad_s"]:
                raise SessionError(f"J{i+1} measured speed exceeds verified threshold")

    def step(self, session_id, feedback, now, joint_velocity_rad_s=(0.0,)*6,
             gripper_intent="hold", permit=False, *, gripper_velocity_rad_s=None):
        self._session(session_id)
        self.soft_limit_blocks=[None]*AXIS_COUNT
        try:
            positions, measured_velocity = self._feedback(feedback, now)
            if self._origin is None:
                raise SessionError("fresh origin has not been captured")
            if self.phase == "fault":
                return self._command(reason="fault latched; fresh input does not clear it")
            if self.admission_errors():
                self._velocity = (0.0,) * AXIS_COUNT
                self.phase = "paused"
                self._return = None
                self._stable_since = None
                return self._command(reason="motion admission blocked by unverified site configuration")
            self._tracking(positions, measured_velocity)
            # A requested/sent target is not evidence of physical movement.
            # Only a newly validated measured native angle can establish this.
            if positions[2] > self._origin[2]+1e-9:
                self._j3_increased = True
            dt = float(now)-self._last_step
            self._last_step = float(now)
            github=self.config.get('admission_policy')=='tony-github-defaults-v1'
            if dt<=0 or (not github and dt>self.config["max_control_interval_s"]):
                raise SessionError("control interval invalid; do not integrate a backlog")
            if github:
                dt=min(dt,self.config['input_period_s'])
                if self._return is not None:
                    self._return['logical_elapsed']=self._return.get('logical_elapsed',0.)+dt
            if self.phase in ("returning", "at_origin"):
                return self._return_step(positions, measured_velocity, float(now))
            if self.phase == "paused" or permit is not True:
                self._velocity = (0.0,) * AXIS_COUNT
                return self._command(reason="paused or permit released; keep hold, no return or disable")
            requested = list(_vector(joint_velocity_rad_s, 6))
            if gripper_intent not in ("open", "close", "hold"):
                raise SessionError("invalid gripper intent")
            grip={"open": self.config["joints"][6]["max_speed_rad_s"],
                  "close": -self.config["joints"][6]["max_speed_rad_s"], "hold": 0.0}[gripper_intent]
            if gripper_velocity_rad_s is not None:
                value=gripper_velocity_rad_s
                if (type(value) not in (int,float) or not math.isfinite(value)
                        or abs(value)>self.config["joints"][6]["max_speed_rad_s"]
                        or (gripper_intent=="hold" and value!=0)
                        or (gripper_intent=="open" and value<=0)
                        or (gripper_intent=="close" and value>=0)):
                    raise SessionError("invalid analog gripper velocity or intent mismatch")
                grip=value
            requested.append(grip)
            if self.config.get('admission_policy')!='tony-github-defaults-v1' and requested[2] < 0 and (not self._j3_increased or self.config.get("j3_reverse_motion_verified") is not True):
                raise SessionError("J3 initial native motion must increase; reverse needs explicit site verification")
            next_targets, next_velocity = [], []
            self.soft_limit_blocks=[None]*AXIS_COUNT
            for i, (want, last_v, target, joint) in enumerate(zip(requested, self._velocity, self._targets, self.config["joints"])):
                vmax, accel = joint["max_speed_rad_s"], joint["max_accel_rad_s2"]
                want = max(-vmax, min(vmax, want))
                # Releasing this individual control cancels its intent in this
                # sample, even while another joint/gripper control stays held.
                # Acceleration shaping applies to active requests; zero is a
                # hold request, not an instruction to integrate a stopping tail.
                velocity = want if accel is None else (0.0 if want == 0.0 else max(last_v-accel*dt, min(last_v+accel*dt, want)))
                candidate = target+velocity*dt
                if github:
                    low,high=joint['native_min_rad'],joint['native_max_rad']
                    # A soft boundary rejects only the outward direction. An
                    # existing overshoot/start pose is held, never snapped to
                    # the boundary merely because the operator pushes outward.
                    if want>0 and (target>=high or positions[i]>=high):
                        candidate,velocity=target,0.
                        self.soft_limit_blocks[i]='positive'
                    elif want<0 and (target<=low or positions[i]<=low):
                        candidate,velocity=target,0.
                        self.soft_limit_blocks[i]='negative'
                    elif want>0:
                        candidate=min(high,candidate)
                        velocity=(candidate-target)/dt
                    elif want<0:
                        candidate=max(low,candidate)
                        velocity=(candidate-target)/dt
                    else:
                        candidate,velocity=target,0.
                elif not joint['native_min_rad']<=candidate<=joint['native_max_rad']:
                    raise SessionError(f"J{i+1} candidate would cross verified native bound")
                next_targets.append(candidate)
                next_velocity.append(velocity)
            self._targets, self._velocity = tuple(next_targets), tuple(next_velocity)
            return self._command("bounded_native_targets", "bounded native jog plan only; no hardware send")
        except (SessionError, TypeError, AttributeError, KeyError, OverflowError) as exc:
            return self._fault(str(exc))

    def request_return_origin(self, session_id, feedback, now):
        self._session(session_id)
        self.soft_limit_blocks=[None]*AXIS_COUNT
        if self.phase == "fault":
            return self._command(reason="fault must be explicitly resolved before a return request")
        try:
            positions, velocities = self._feedback(feedback, now)
            if self._origin is None or self.admission_errors():
                raise SessionError("origin or verified site parameters unavailable")
            self._tracking(positions, velocities)
            github=self.config.get('admission_policy')=='tony-github-defaults-v1'
            if not github and self.config.get("return_path_verified") is not True:
                raise SessionError("return path has not been verified; an origin is not a safe path")
            if not github and positions[2] > self._origin[2]+1e-12 and self.config.get("j3_reverse_motion_verified") is not True:
                raise SessionError("J3 reverse return is not verified")
            tolerance = self.config["velocity_tolerance_rad_s"]
            if (tolerance is not None and any(abs(v) > tolerance for v in velocities)) or any(abs(v) > 1e-12 for v in self._velocity):
                raise SessionError("pause and verify measured standstill before requesting return")
            duration = 0.0
            for start, finish, joint in zip(positions, self._origin, self.config["joints"]):
                distance, vmax, accel = abs(finish-start), joint["max_speed_rad_s"], joint["max_accel_rad_s2"]
                if github:
                    min_time=max(self.config['return_min_time_s'],1.5*distance/min(vmax,self.config['return_max_speed_rad_s']))
                else:
                    min_time = 2*math.sqrt(distance/accel) if distance <= vmax*vmax/accel else distance/vmax+vmax/accel
                duration = max(duration, min_time)
            self._targets = positions
            self._velocity = (0.0,) * AXIS_COUNT
            self._return = {"start": positions, "started_at": float(now), "duration": duration}
            self._last_step = float(now)
            self._stable_since = None
            self.phase = "returning"
            return self._command("request_return_session_origin", "explicit return requested; use subsequent bounded samples, never jump to origin")
        except (SessionError, TypeError, AttributeError, KeyError) as exc:
            return self._fault(str(exc))

    def _return_step(self, positions, measured_velocity, now):
        plan = self._return
        if plan is None:
            raise SessionError("return plan missing")
        duration = plan["duration"]
        elapsed = max(0.0, min(duration, now-plan["started_at"]))
        if self.config.get('admission_policy')=='tony-github-defaults-v1':
            elapsed=min(duration,plan.get('logical_elapsed',0.))
        targets, velocities = [], []
        for start, finish, joint in zip(plan["start"], self._origin, self.config["joints"]):
            signed_distance = finish-start
            distance, accel = abs(signed_distance), joint["max_accel_rad_s2"]
            direction = 1.0 if signed_distance >= 0 else -1.0
            if distance == 0 or elapsed >= duration:
                position, velocity = finish, 0.0
            elif accel is None:
                alpha=elapsed/duration
                position=start+signed_distance*(alpha*alpha*(3-2*alpha))
                velocity=signed_distance*6*alpha*(1-alpha)/duration
            else:
                disc = max(0.0, duration*duration-4*distance/accel)
                peak = 2*distance/(duration+math.sqrt(disc))
                ramp = peak/accel
                if elapsed < ramp:
                    traveled, speed = .5*accel*elapsed*elapsed, accel*elapsed
                elif elapsed <= duration-ramp:
                    traveled, speed = .5*accel*ramp*ramp+peak*(elapsed-ramp), peak
                else:
                    remaining = duration-elapsed
                    traveled, speed = distance-.5*accel*remaining*remaining, accel*remaining
                position, velocity = start+direction*traveled, direction*speed
            if self.config.get('admission_policy')!='tony-github-defaults-v1' and not joint["native_min_rad"] <= position <= joint["native_max_rad"]:
                raise SessionError("return candidate outside verified bounds")
            if abs(velocity) > joint["max_speed_rad_s"]+1e-9:
                raise SessionError("return candidate exceeds verified speed")
            targets.append(position)
            velocities.append(velocity)
        self._targets, self._velocity = tuple(targets), tuple(velocities)
        stable = elapsed >= duration and self._at_origin(positions, measured_velocity)
        if stable:
            if self._stable_since is None:
                self._stable_since = now
            if now-self._stable_since >= self.config["stable_dwell_s"]:
                self.phase = "at_origin"
                if self.config.get('admission_policy')=='tony-github-defaults-v1':
                    return self._command(reason='GitHub return trajectory and settle time completed; operator must review actual origin readings and confirm support')
                return self._command(reason="measured origin position and velocity stable; await explicit load support confirmation")
        else:
            self._stable_since = None
            self.phase = "returning"
        return self._command("bounded_native_targets", "bounded return plan; measured settling and support still required")

    def _at_origin(self, positions, velocities):
        if self.config.get('admission_policy')=='tony-github-defaults-v1':
            # No invented measurement tolerance. Keep live readings visible;
            # the operator explicitly accepts arrival and support at finish.
            return self._return is not None and all(abs(a-b)<1e-12 for a,b in zip(self._targets,self._origin))
        return (all(abs(a-b) <= self.config["position_tolerance_rad"] for a,b in zip(positions,self._origin))
                and all(abs(v) <= self.config["velocity_tolerance_rad_s"] for v in velocities))

    def confirm_support(self, session_id, feedback, now, *, supported=False):
        self._session(session_id)
        if supported is not True or self.phase != "at_origin":
            raise SessionError("measured stable origin and explicit human support confirmation required")
        try:
            positions, velocities = self._feedback(feedback, now)
            if self.admission_errors():
                raise SessionError("fresh verified fault/mode status and site configuration required before disable")
            if (self._last_step is None
                    or not 0.0 <= float(now)-self._last_step <= self.config["max_control_interval_s"]
                    or self._stable_since is None
                    or float(now)-self._stable_since < self.config["stable_dwell_s"]):
                raise SessionError("continuous stable feedback was interrupted; settle again before disable")
            if not self._at_origin(positions, velocities):
                raise SessionError("origin no longer measurably stable; do not disable")
        except (SessionError, TypeError, AttributeError) as exc:
            return self._fault(str(exc))
        self.phase = "disabled"
        self._velocity = (0.0,) * AXIS_COUNT
        return self._command("disable_after_verified_support", "data only: stable session origin and explicit human support confirmed")
