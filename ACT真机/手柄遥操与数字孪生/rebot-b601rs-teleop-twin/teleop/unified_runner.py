"""Unified input/planner/transport loop, restricted to a synthetic robot.

Default invocation opens nothing. --scripted-demo uses synthetic input as well.
--xinput reads a real Windows controller but still sends only to FakeRobot.
There is deliberately no hardware, CAN, site-config, enable-real or ACT option.
An in-process heartbeat cannot preserve a physical robot after process failure.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIEW = ROOT.parent / "input_core"
sys.path.insert(0, str(PREVIEW))
from control_core import AXES, BUTTONS, Engine, Frame
from fake_robot import DEFAULT_FAKE_ORIGIN, FakeRobot, synthetic_config
from motion_transport import MotionTransport, TransportError
from session_core import SessionError, SessionPlanner


class UnifiedRunner:
    """Deterministic tick API; all output methods target the same FakeRobot.

    step accepts one input frame and at most one session-bound operator action.
    It reads seven fresh synthetic feedback values once, plans once, and applies
    only that newest command with exactly the paired feedback object.
    """
    ACTIONS = frozenset(("pause", "resume", "return_origin", "confirm_support", "interrupt"))

    def __init__(self, now, *, robot=None, config=None):
        self.robot = FakeRobot() if robot is None else robot
        if not isinstance(self.robot, FakeRobot) or self.robot.synthetic_only is not True:
            raise ValueError("This runner accepts only the synthetic FakeRobot; real hardware is unavailable")
        config = synthetic_config() if config is None else copy.deepcopy(config)
        if config.get("synthetic_only") is not True or config.get("hardware_writes_supported") is not False:
            raise ValueError("Explicit synthetic-only configuration required; physical site files are rejected")
        self.engine = Engine()
        self.planner = SessionPlanner(config, now, allow_synthetic_feedback=True)
        self.transport = MotionTransport(self.robot.controller, self.robot.motors,
                                         self.planner, allow_synthetic_feedback=True)
        self.started = False
        self.closed = False
        self.last_tick = None
        self.last_feedback = None
        self.last_input = self.engine.snapshot(now)
        self.last_error = None
        self.action_error = None
        self.tick_count = 0

    @property
    def session_id(self):
        return self.planner.session_id

    def start(self, now):
        """Capture this session's fresh fake observation and hold its exact q."""
        if self.started or self.closed:
            if self.started and not self.closed:
                self._fault("running session cannot recapture its immutable origin")
            raise SessionError("a running/closed session cannot recapture its origin")
        self.started = True
        self.last_tick = float(now)
        try:
            feedback = self.robot.read_feedback(self.session_id, now)
            self.last_feedback = feedback
            command = self.planner.capture_origin(feedback, now)
            if command.phase == "fault" or command.admission_errors:
                self.last_error = command.reason
            else:
                self.transport.arm(feedback, now, allow_motion=True)
        except Exception as exc:
            self._fault(f"startup: {type(exc).__name__}: {exc}")
        return self.snapshot()

    @staticmethod
    def _neutral(snapshot):
        inp, ctrl = snapshot["input"], snapshot["control"]
        axes = inp["axes"]
        return (inp["connected"] and inp["age_ms"] is not None
                and inp["age_ms"] <= 250
                and ctrl["gate"] not in ("disconnected", "input_stale", "neutral_required")
                and all(math.isfinite(axes[a]) and abs(axes[a]) <= ctrl["deadzone"] for a in AXES[:4])
                and all(math.isfinite(axes[a]) and 0 <= axes[a] <= ctrl["deadzone"] for a in AXES[4:])
                and not set(inp["buttons"]) & {"LB", "A", "X", "B"})

    def _owned_hold(self):
        if self.transport.possibly_enabled:
            try:
                self.transport.heartbeat_hold()
            except Exception as exc:
                self.last_error = f"{self.last_error or 'hold failed'}; {type(exc).__name__}: {exc}"

    def _fault(self, reason):
        self.last_error = str(reason)
        self.engine.set_paused(True)
        # The reported mapping must also stop; retaining the old pre-fault
        # snapshot could display a nonzero intent after the transport holds.
        try:
            self.last_input = self.engine.snapshot(self.last_tick)
        except Exception:
            self.last_input = copy.deepcopy(self.last_input)
            self.last_input["control"].update(paused=True, armed=False,
                gate="paused", reason=str(reason),
                joint_velocity_rad_s=[0.0]*6, gripper_intent="hold")
        if self.planner.phase != "disabled":
            command = self.planner.interrupt(self.session_id, str(reason))
            if self.transport.armed and not self.transport.fault:
                try:
                    self.transport.apply(command, self.last_feedback, self.last_tick)
                    return
                except Exception as exc:
                    self.last_error += f"; {type(exc).__name__}: {exc}"
        self._owned_hold()

    def interrupt(self, reason="EOF/interrupt/exception"):
        """Best-effort fake hold; never automatic return, disable or release."""
        if not self.closed:
            self._fault(reason)
        return self.snapshot()

    def step(self, frame, now, *, action=None, session_id=None, supported=False):
        if not self.started:
            raise SessionError("start a new session before ticking")
        if self.closed or (self.planner.phase == "disabled" and not self.transport.possibly_enabled):
            raise SessionError("session is closed; create a new runner and capture a fresh origin")
        self.action_error = None
        self.tick_count += 1
        try:
            # A rejected command must cancel a prior active output, rather
            # than escape before the owned transport receives a zero-v hold.
            if action is not None:
                if session_id != self.session_id:
                    raise SessionError("operator action requires this current session_id")
                if action not in self.ACTIONS:
                    raise SessionError("unsupported operator action")
            if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
                raise ValueError("finite monotonic tick time required")
            gap = now-self.last_tick
            self.last_tick = now
            if action in ("resume", "return_origin"):
                self.engine.set_paused(False)
            if action in ("pause", "interrupt"):
                self.engine.set_paused(True)
            self.last_input = self.engine.update(frame, now)
            # Do not integrate a backlog, even if input and feedback are fresh.
            if not 0 < gap <= self.planner.config["max_control_interval_s"]:
                self._fault("control loop interval invalid; backlog discarded")
                return self.snapshot()
            if self.transport.fault or self.planner.phase == "fault":
                # A resume request must never clear the visible/input gate of
                # a latched execution fault, even though it still cannot send.
                self.engine.set_paused(True)
                self.last_input = self.engine.snapshot(now)
                self._owned_hold()
                return self.snapshot()
            feedback = self.robot.read_feedback(self.session_id, now)
            self.last_feedback = feedback
            ctrl = self.last_input["control"]
            gate = ctrl["gate"]
            if action == "interrupt":
                self._fault("operator interrupt; retain owned hold")
                return self.snapshot()
            if action in ("resume", "return_origin", "confirm_support") and not self._neutral(self.last_input):
                self.action_error = "action rejected: release LB/B/A/X and return sticks/triggers to neutral"
                self.engine.set_paused(True)
                self.planner.pause(self.session_id, self.action_error)
                action = None
            # A fresh B, input loss, stale frame or required rearm cancels a
            # return explicitly. permit=False alone cannot cancel core._return.
            if (self.engine.paused or gate in ("disconnected", "input_stale", "neutral_required")):
                if self.planner.phase != "paused":
                    self.planner.pause(self.session_id, f"input gate {gate}; return cancelled")
            if self.planner.phase in ("returning", "at_origin") and not self._neutral(self.last_input):
                self.planner.pause(self.session_id, "input intervention cancelled explicit return")
                self.engine.set_paused(True)
            if action == "resume":
                command = self.planner.resume(self.session_id, feedback, now, neutral_confirmed=True)
            elif action == "return_origin":
                command = self.planner.request_return_origin(self.session_id, feedback, now)
            elif action == "confirm_support":
                command = self.planner.confirm_support(self.session_id, feedback, now, supported=supported)
            else:
                velocity = tuple(ctrl["joint_velocity_rad_s"])
                gripper = ctrl["gripper_intent"]
                # A centred stick with LB held must hold immediately, not let a
                # previously moving target coast through acceleration limiting.
                nonzero_intent = any(velocity) or gripper != "hold"
                permit = (ctrl["lb_held"] and ctrl["armed"] and not self.engine.paused
                          and gate in ("ready", "active") and nonzero_intent)
                command = self.planner.step(self.session_id, feedback, now, velocity, gripper, permit)
            if not self.transport.armed:
                # A blocked startup cannot acquire execution ownership later by
                # receiving a good frame: start a genuinely new session instead.
                if command.kind != "hold_current":
                    raise TransportError("startup was not armed; new synthetic session required")
            else:
                self.transport.apply(command, feedback, now)
            if command.phase == "paused" and command.admission_errors:
                self.engine.set_paused(True)
            if command.phase == "fault":
                self.last_error = command.reason
                self.engine.set_paused(True)
            # Expose the current stopped gate, not the pre-fault/pre-admission
            # input mapping that was sampled earlier in this same cycle.
            self.last_input = self.engine.snapshot(now)
            if command.kind == "disable_after_verified_support" and not self.transport.possibly_enabled:
                self.transport.close_after_disable()
                self.closed = True
        except Exception as exc:
            self._fault(f"loop: {type(exc).__name__}: {exc}")
        return self.snapshot()

    def snapshot(self):
        command = self.planner.latest_command
        # A disable *plan* is not evidence that a partially failing SDK call
        # disabled all axes. Keep ticking the owned hold until handled later.
        phase = self.planner.phase
        if phase == "disabled" and self.transport.possibly_enabled:
            phase = "fault_holding"
        elif phase == "disabled" and not self.closed and self.last_error:
            phase = "cleanup_fault"
        return {
            "kind": "unified_synthetic_robot_loop", "synthetic_only": True,
            "can_opened": False, "hardware_motion_supported": False,
            "session_id": self.session_id, "tick_count": self.tick_count,
            "phase": phase, "planner_phase": self.planner.phase, "closed": self.closed,
            "origin_rad": self.planner.origin_rad,
            "fake_positions_rad": tuple(self.robot.positions),
            "fake_velocities_rad_s": tuple(self.robot.velocities),
            "command": command.to_dict() if command else None,
            "input": copy.deepcopy(self.last_input),
            "transport": {"armed": self.transport.armed,
                          "possibly_enabled": self.transport.possibly_enabled,
                          "fault": self.transport.fault,
                          "last_targets_rad": self.transport.last_targets,
                          "hold_send_errors": self.transport.last_send_errors[:]},
            "last_error": self.last_error, "action_error": self.action_error,
            "watchdog_scope": "in-process synthetic heartbeat only; no physical watchdog or process-death hold guarantee",
        }


def input_frame(now, buttons=0, ly=0.0, *, connected=True, acquired_at=None):
    axes = dict.fromkeys(AXES, 0.0)
    axes["LY"] = ly
    return Frame(connected, buttons=buttons, axes=axes,
                 acquired_at=now if acquired_at is None else acquired_at)


def scripted_demo():
    """Explicit synthetic scenario. No user input or human support is claimed."""
    now = 10.0
    runner = UnifiedRunner(now)
    trace = [runner.start(now)]

    def tick(buttons=0, ly=0.0, *, action=None, connected=True, count=1, supported=False):
        nonlocal now
        for _ in range(count):
            now += .02
            state = runner.step(input_frame(now, buttons, ly, connected=connected), now,
                                action=action, session_id=runner.session_id if action else None,
                                supported=supported)
            trace.append(state)
            if state["phase"] in ("fault", "fault_holding", "cleanup_fault"):
                raise RuntimeError(state["last_error"] or state["command"]["reason"])
        return state

    tick(action="resume")
    for axis in range(1, 7):
        if axis > 1:
            tick(BUTTONS["DPadRight"])
            tick()
        tick(BUTTONS["LB"], .7, count=8)
        tick(BUTTONS["LB"], 0.0)  # centred while permit held
        tick()
    tick(BUTTONS["LB"] | BUTTONS["A"], count=5)
    tick(BUTTONS["LB"] | BUTTONS["X"], count=5)
    tick()
    tick(BUTTONS["B"])
    tick()
    tick(action="resume")
    tick(connected=False)
    tick()  # reconnect alone does not resume
    tick(action="resume")
    tick(action="pause")
    tick()
    tick(action="return_origin")
    for _ in range(500):
        if tick()["phase"] == "at_origin":
            break
    if runner.planner.phase != "at_origin":
        raise RuntimeError("synthetic return did not measurably settle")
    tick(action="confirm_support", supported=True)
    names = [call["kind"] for call in runner.robot.calls]
    success = (runner.closed and runner.planner.phase == "disabled"
               and names.index("disable_all") < names.index("close_bus") < names.index("motor_close")
               and names[-1] == "controller_close")
    nonzero_axes = sorted({c["axis"] for c in runner.robot.calls if c["kind"] == "mit"
                           and c["success"] and c["values"]["vel"] != 0})
    return {"kind": "scripted_synthetic_end_to_end", "passed": success,
            "can_opened": False, "real_input_used": False, "hardware_motion_supported": False,
            "support_confirmation": "scripted synthetic support flag; NOT human physical support evidence",
            "config_provenance": "synthetic fixtures, not site configuration",
            "nonzero_command_axes": nonzero_axes, "final": runner.snapshot(),
            "sdk_calls": runner.robot.calls, "trace": trace}


def load_xinput_reader(slot):
    # Importing server.py defines a read-only reader; its __main__ is not run.
    spec = importlib.util.spec_from_file_location("_offline_xinput_reader", PREVIEW / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.XInputReader(slot)


def start_resume_action(frame, previous_buttons):
    """A physical START edge explicitly requests resume; B retains priority.

    This helper never acknowledges neutral. UnifiedRunner still validates the
    full current input before accepting the returned action.
    """
    if (frame.connected and isinstance(frame.buttons, int)
            and not isinstance(frame.buttons, bool)
            and frame.buttons & BUTTONS["Start"]
            and not previous_buttons & BUTTONS["Start"]
            and not frame.buttons & BUTTONS["B"]):
        return "resume"
    return None


def xinput_demo(duration, slot, *, evidence_source="Windows_XInputGetState_current_session"):
    """Real gamepad -> synthetic robot only. No automatic return or support."""
    if evidence_source not in ("Windows_XInputGetState_current_session", "synthetic_fixture"):
        raise ValueError("unknown input evidence source")
    if sys.platform != "win32":
        raise RuntimeError("XInput demonstration requires Windows")
    import msvcrt
    reader = load_xinput_reader(slot)
    started = time.monotonic()
    runner = UnifiedRunner(started)
    runner.start(started)
    trace = []
    print(f"OFFLINE / FAKE ROBOT / NO CAN. session_id={runner.session_id}", flush=True)
    print("START or R=explicit resume after neutral; P=pause; T=explicit return; S=explicit synthetic support/disable; Q=interrupt.", flush=True)
    print("DPad left/right=axis, LB+LY=jog; LB+A/X=gripper; B=latched pause.", flush=True)
    exit_reason = "duration_elapsed"
    previous_buttons = 0
    connected_reads = 0
    disconnected_reads = 0
    mapped_intent_frames = 0
    observed_joint_intents = set()
    observed_gripper_intents = set()
    fatal_error = None
    try:
        while time.monotonic()-started < duration and not runner.closed:
            # Windows monotonic time may have coarser resolution than the
            # first read. Wait a sampling period before the first planning
            # tick as well, rather than weakening the strict positive-dt gate.
            time.sleep(.02)
            if time.monotonic()-started >= duration:
                break
            action, supported = None, False
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                action = {"r": "resume", "p": "pause", "t": "return_origin",
                          "s": "confirm_support", "q": "interrupt"}.get(key)
                supported = key == "s"
            frame = reader.read()
            if action is None:
                action = start_resume_action(frame, previous_buttons)
            previous_buttons = frame.buttons if frame.connected else 0
            connected_reads += bool(frame.connected)
            disconnected_reads += not frame.connected
            now = time.monotonic()
            state = runner.step(frame, now, action=action,
                                session_id=runner.session_id if action else None, supported=supported)
            trace.append(state)
            ctrl = state["input"]["control"]
            if any(ctrl["joint_velocity_rad_s"]) or ctrl["gripper_intent"] != "hold":
                mapped_intent_frames += 1
            observed_joint_intents.update(i+1 for i, v in enumerate(ctrl["joint_velocity_rad_s"]) if v)
            if ctrl["gripper_intent"] != "hold":
                observed_gripper_intents.add(ctrl["gripper_intent"])
            if state["phase"] in ("fault", "fault_holding", "cleanup_fault") and action != "interrupt":
                fatal_error = state["last_error"] or state["command"]["reason"]
                exit_reason = "loop_fault"
                break
            if action == "interrupt":
                exit_reason = "operator_interrupt"
                break
    except (KeyboardInterrupt, EOFError):
        exit_reason = "interrupt_or_eof"
    except Exception as exc:
        exit_reason = f"exception: {type(exc).__name__}: {exc}"
        fatal_error = exit_reason
    finally:
        if not runner.closed:
            runner.interrupt(f"{exit_reason}; no automatic return/disable on exit")
    real_input = evidence_source == "Windows_XInputGetState_current_session"
    return {"kind": "real_xinput_to_synthetic_robot" if real_input else "synthetic_input_to_synthetic_robot",
            "source": evidence_source, "can_opened": False,
            "real_input_used": real_input, "hardware_motion_supported": False,
            "sampling_ok": bool(connected_reads and not fatal_error),
            "physical_input_verified": False, "physical_robot_verified": False,
            "connected_reads": connected_reads, "disconnected_reads": disconnected_reads,
            "mapped_intent_frames": mapped_intent_frames,
            "joint_intents_observed": sorted(observed_joint_intents),
            "gripper_intents_observed": sorted(observed_gripper_intents),
            "successful_nonzero_fake_command_axes": sorted({c["axis"] for c in runner.robot.calls
                if c["kind"] == "mit" and c["success"] and c["values"]["vel"] != 0}),
            "fatal_error": fatal_error,
            "slot": slot, "duration_s": time.monotonic()-started,
            "reader_error": reader.error, "exit_reason": exit_reason,
            "support_confirmation": "keyboard S is synthetic demonstration only; never hardware support evidence",
            "final": runner.snapshot(), "trace": trace,
            "sdk_calls": runner.robot.calls,
            "physical_acceptance": "NOT PERFORMED; inspect trace, do not infer pass from program exit"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scripted-demo", action="store_true")
    group.add_argument("--xinput", action="store_true")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--slot", type=int, choices=range(4), default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.scripted_demo and not args.xinput:
        print(json.dumps({"mode": "description_only", "can_opened": False,
                          "message": "Use --scripted-demo for full synthetic regression or --xinput for real input to FakeRobot only. No hardware mode exists."}))
        return 0
    if not math.isfinite(args.duration) or not 0 < args.duration <= 3600:
        parser.error("duration must be finite and between 0 and 3600 seconds")
    result = scripted_demo() if args.scripted_demo else xinput_demo(args.duration, args.slot)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"kind": result["kind"], "passed": result.get("passed"),
                      "sampling_ok": result.get("sampling_ok"),
                      "physical_input_verified": False,
                      "can_opened": False, "final_phase": result["final"]["phase"],
                      "out": str(args.out) if args.out else None}, ensure_ascii=False))
    return 0 if result.get("passed", result.get("sampling_ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
