"""Session-bound native PCAN backend; construction and import open no hardware.

One PCAN client supplies actual query/status receipts and tightly constrained
MIT/enable/normal-disable operations. Site admission and a separately issued
authorization are required before actuator writes. No ACT, mode/zero/parameter
write, automatic homing, automatic disable, or process-death hold is provided.

Wire scaling follows MotorBridge 0.5.6 robstride motor/protocol source, with
out-of-range values rejected instead of the upstream encoder's clamping.
RS00: p=4*pi, v=33, t=14, kp=500, kd=5.
RS06: p=4*pi, v=50, t=36, kp=5000, kd=100.
These are protocol coding ranges, NEVER mechanical limits or accepted gains.
"""
from __future__ import annotations

import copy
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import secrets
import struct
import time

import pcan_readonly as pcan
from session_core import NativeFeedback, config_errors


class NativeBackendError(RuntimeError):
    pass


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


PMAX = _f32(4.0 * _f32(math.pi))
WIRE_LIMITS = {"rs-00": (PMAX, 33.0, 14.0, 500.0, 5.0),
               "rs-06": (PMAX, 50.0, 36.0, 5000.0, 100.0)}
PROTOCOL_SOURCES = {
    "motorbridge_version": "0.5.6",
    "reference_motor": "https://github.com/motorbridge/motorbridge (v0.5.6 RobStride motor codec)",
    "reference_protocol": "https://github.com/motorbridge/motorbridge (v0.5.6 RobStride protocol)",
    "official_guide": "https://wiki.seeedstudio.com/robstride_control/",
    "reference_use": "static wire specification only; no historical controller or site configuration loaded",
}


def _axis(motor_id):
    if type(motor_id) is not int or motor_id not in range(1, 8):
        raise NativeBackendError("only arm motor IDs 1..7 are supported")
    return "rs-06" if motor_id <= 3 else "rs-00"


def encode_mit(motor_id, *, pos, vel, kp, kd, tau):
    """Exact SDK float32 wire arithmetic after strict domain validation."""
    model = _axis(motor_id)
    pmax, vmax, tmax, kpmax, kdmax = WIRE_LIMITS[model]
    values = (pos, vel, kp, kd, tau)
    if not all(_finite(value) for value in values):
        raise NativeBackendError("MIT fields must be finite real numbers")
    for name,value,low,high in (("pos",pos,-pmax,pmax),("vel",vel,-vmax,vmax),
                               ("kp",kp,0,kpmax),("kd",kd,0,kdmax),("tau",tau,-tmax,tmax)):
        if not low <= value <= high:
            raise NativeBackendError(f"{model} {name} exceeds protocol range; no clipping")
    def signed(value, maximum):
        return int(_f32(_f32(_f32(_f32(value)/maximum)+1.0)*32767.0))
    def positive(value, maximum):
        return int(_f32(_f32(_f32(value)/maximum)*65535.0))
    data=struct.pack(">HHHH",signed(pos,pmax),signed(vel,vmax),positive(kp,kpmax),positive(kd,kdmax))
    return pcan.Packet((1<<24)|(signed(tau,tmax)<<8)|motor_id,data,dlc=8)


def control_packet(motor_id, *, enable):
    _axis(motor_id)
    if type(enable) is not bool:
        raise NativeBackendError("enable must be an explicit boolean")
    return pcan.Packet(((3 if enable else 4)<<24)|(0xFD<<8)|motor_id,bytes(8),dlc=8)


def decode_mit_effective(packet):
    """Actual quantized wire values, distinct from requested measured holds."""
    validate_actuator_packet(packet)
    if (packet.arbitration_id>>24)&31 != 1:
        raise NativeBackendError("an MIT packet is required for quantized values")
    model=_axis(packet.arbitration_id&255)
    pmax,vmax,tmax,kpmax,kdmax=WIRE_LIMITS[model]
    q,v,kp,kd=struct.unpack(">HHHH",packet.data)
    torque=(packet.arbitration_id>>8)&65535
    signed=lambda value,limit:_f32(_f32(_f32(value/32767.0)-1.0)*limit)
    positive=lambda value,limit:_f32(_f32(value/65535.0)*limit)
    return {"effective_position_rad":signed(q,pmax),"effective_velocity_rad_s":signed(v,vmax),
            "effective_kp":positive(kp,kpmax),"effective_kd":positive(kd,kdmax),
            "effective_torque_nm":signed(torque,tmax)}


def validate_actuator_packet(packet):
    if (packet.message_type != pcan.PCAN_MESSAGE_EXTENDED or len(packet.data)!=8
            or packet.dlc not in (None,8) or not 0 <= packet.arbitration_id <= 0x1FFFFFFF):
        raise NativeBackendError("only eight-byte extended actuator frames are permitted")
    motor_id=packet.arbitration_id&255
    _axis(motor_id)
    comm=(packet.arbitration_id>>24)&31
    if comm in (3,4):
        expected=control_packet(motor_id,enable=comm==3)
        if packet.arbitration_id != expected.arbitration_id or packet.data != expected.data:
            raise NativeBackendError("enable/disable must have host FD and all-zero data; fault reset forbidden")
    elif comm == 1:
        if any(value>65534 for value in (*struct.unpack(">HH",packet.data[:4]),(packet.arbitration_id>>8)&65535)):
            raise NativeBackendError("signed MIT wire field exceeds the SDK coding domain")
    else:
        raise NativeBackendError("only MIT, enable and normal disable actuator operations are permitted")


class NativeMotionPcanTransport(pcan.WindowsPcanTransport):
    """A private writer capability bound by NativePcanBackend after admission."""
    def __init__(self, *, allow_can_open=False):
        super().__init__(allow_can_open=allow_can_open)
        self._motion_capability=None

    def _bind_motion_capability(self, capability):
        if self._motion_capability is not None:
            raise NativeBackendError("motion writer already belongs to a session")
        self._motion_capability=capability

    def configure_report_enable(self,motor_id):
        # The same Windows handle supplies firmware state reporting after a
        # power cycle; this is communication configuration, never motor enable.
        from status_reporting import ReportingTransport
        ReportingTransport.configure_report_enable(self,motor_id,allow_reporting_change=True)

    def write_actuator_packet(self, packet, *, authorization):
        if self._motion_capability is None or authorization is not self._motion_capability:
            raise NativeBackendError("no current backend-issued actuator capability")
        validate_actuator_packet(packet)
        if not self.opened:
            raise NativeBackendError("CAN is closed")
        message=pcan._TPCANMsg()
        message.ID,message.MSGTYPE,message.LEN=packet.arbitration_id,packet.message_type,8
        message.DATA[:]=packet.data
        self._check(self.lib.CAN_Write(pcan.PCAN_USBBUS1,ctypes.byref(message)),"native actuator submission")


@dataclass(frozen=True)
class MotionAuthorization:
    session_id: str
    config_sha256: str
    issued_at_monotonic: float
    feedback_sequence: int
    backend_id: str


class _ReceiptCapture(pcan.PcanReadOnlyCapture):
    def __init__(self, transport, owner, **kwargs):
        super().__init__(transport,**kwargs)
        self.owner=owner

    def _receive(self, context, *, accept_status=True):
        record=super()._receive(context,accept_status=accept_status)
        if record is not None:
            self.owner._observe_record(record)
        return record


class NativeMotor:
    def __init__(self, backend, index):
        self.backend,self.index=backend,index
        self.closed=False

    def send_mit(self, *, pos, vel, kp, kd, tau):
        if self.closed:
            raise NativeBackendError("motor handle is closed")
        self.backend._send_mit(self.index,pos=pos,vel=vel,kp=kp,kd=kd,tau=tau)

    def close(self):
        if self.backend.opened or self.backend.possibly_enabled:
            raise NativeBackendError("close the safely released bus before motor handles")
        self.closed=True


class NativePcanBackend:
    synthetic_only=False
    hardware_writes_supported=True

    def __init__(self, config, *, transport=None, clock=time.monotonic, sleep=time.sleep):
        self.config=copy.deepcopy(config)
        self.clock,self.sleep=clock,sleep
        self._transport=transport
        self.capture=None
        self.controller=self
        self.motors=tuple(NativeMotor(self,i) for i in range(7))
        self.backend_id=secrets.token_hex(16)
        self.session_id=None
        self.last_feedback=None
        self.feedback_sequence=0
        self.last_batch=[]
        self.fault=None
        self.fault_codes=[0]*7
        self.fault_receipts=[None]*7
        self.fault_sources=[{} for _ in range(7)]
        self.authorization=None
        self.origin_rad=None
        self.possibly_enabled=False
        self.enabled=False
        self.closed=False
        self.phase="not_open"
        self.last_targets=[None]*7
        self._hold_targets=[None]*7
        self._hold_gains=None
        self._last_command_velocities=[0.0]*7
        self._effective_targets=[None]*7
        self._primed=set()
        self._disable_permission=None
        self._stable_origin_since=None
        self._last_feedback_time=None
        self._j3_increased=False
        self.startup_quantization_preview=[]
        self._read_deadline=None
        self.release_allowed_after_external_poweroff=False
        self.external_poweroff_acknowledgement=None
        self.submissions=[]
        self.submission_sequence=0
        self.hottest_temperature_c=None
        self.thermal_return_requested=False
        self.thermal_disconnect_requested=False
        self.communication_configuration_submissions=[]

    @property
    def opened(self):
        return self.capture is not None and self.capture.opened

    def open(self, *, allow_can_open=False):
        if allow_can_open is not True:
            raise PermissionError("explicit CAN opening is required")
        if self.capture is not None or self.closed:
            raise NativeBackendError("a genuinely new backend is required to open another session")
        if self._transport is None:
            self._transport=NativeMotionPcanTransport(allow_can_open=True)
        self.capture=_ReceiptCapture(self._transport,self,clock=self._capture_clock,sleep=self.sleep)
        try:
            self.capture.open(allow_can_open=True)
            if self.config.get('admission_policy')=='tony-github-defaults-v1' and hasattr(self._transport,'configure_report_enable'):
                for axis in range(1,8):
                    self._transport.configure_report_enable(axis)
                    self.communication_configuration_submissions.append({'motor_id':axis,'communication_type':24,
                        'setting':'active_reporting_enabled','at_monotonic':self.clock()})
                self.capture.listen(.02)
            self.phase="observing"
        except Exception as exc:
            self._latch(f"open: {exc}")
            raise

    def _latch(self, reason):
        if self.fault is None:
            self.fault=str(reason)
        self.phase="fault_owned" if self.possibly_enabled else "fault_unarmed"
        self._disable_permission=None

    def _observe_record(self, record):
        if not record["driver_timestamp_not_regressed"]:
            self._latch("driver RX timestamp regressed or became unknown")
            raise NativeBackendError(self.fault)
        decoded=record["decoded"]
        if (self.config.get('admission_policy')=='tony-github-defaults-v1'
                and record['context']!='initial_queue_drain' and record['driver_timestamp_advanced']
                and decoded.get('kind')=='status'):
            temp=decoded.get('temperature_c')
            if _finite(temp):
                self.hottest_temperature_c=temp if self.hottest_temperature_c is None else max(self.hottest_temperature_c,temp)
                thresholds=self.config['temperatures']
                self.thermal_return_requested |= temp>=thresholds['return_zero_c']
                self.thermal_disconnect_requested |= temp>=thresholds['disconnect_c']
        if (record["context"]!="initial_queue_drain" and record["driver_timestamp_advanced"]
                and decoded["kind"] in ("status","fault")):
            code=decoded.get("fault_flags_raw",decoded.get("fault_raw",0))
            if code:
                i=decoded["motor_id"]-1
                self.fault_codes[i] |= code
                self.fault_receipts[i]=record
                self.fault_sources[i][decoded['kind']]={'communication_type':decoded['communication_type'],
                    'namespace':decoded.get('fault_namespace'),'flags':decoded.get('fault_flags'),
                    'code':code,'receive_sequence':record['receive_sequence'],
                    'driver_timestamp_us':record['driver_timestamp_us'],
                    'data_hex':record['data_hex'],'arbitration_id_hex':record['arbitration_id_hex']}
                self._latch(f"J{i+1} motor reported nonzero fault {code}")

    def _clock_budget(self):
        value=self.config.get("max_control_interval_s",.1)
        return value if _finite(value) and 0<value<=1 else .1

    def _maximum_age(self):
        value=self.config.get("feedback_max_age_s",.25)
        return value if _finite(value) and 0<value<=1 else .25

    def _capture_clock(self):
        now=self.clock()
        if self._read_deadline is not None and now>self._read_deadline:
            raise NativeBackendError("whole seven-axis read deadline expired inside RX/query loop")
        return now

    def read_feedback(self, session_id, now=None):
        """Actual requests only; caller time is never substituted for RX time.

        One global budget bounds the whole batch, not each of 21 requests.
        Acquired_at is sampled AFTER I/O; callers must plan with clock() then.
        """
        if not self.opened or self.closed:
            raise NativeBackendError("backend is not open")
        if not isinstance(session_id,str) or not session_id:
            raise NativeBackendError("a nonempty current session_id is required")
        if self.session_id is None:
            self.session_id=session_id
        if session_id != self.session_id:
            raise NativeBackendError("backend feedback is bound to one control session")
        if now is not None and not _finite(now):
            raise NativeBackendError("caller time must be finite")
        deadline=self.clock()+self._clock_budget()
        self._read_deadline=deadline
        batch=[]
        try:
            for motor_id in range(1,8):
                axis={}
                for parameter in pcan.CORE_PARAMETERS:
                    remaining=deadline-self.clock()
                    if remaining<=0:
                        raise NativeBackendError("whole seven-axis read deadline expired")
                    query=self.capture.query(motor_id,parameter,timeout_s=min(remaining,self._maximum_age()))
                    axis[pcan.PARAMETERS[parameter]]=query
                    if not query["ok"]:
                        raise NativeBackendError(f"J{motor_id} {pcan.PARAMETERS[parameter]} read failed: {query.get('error')}")
                    if self.clock()>deadline:
                        raise NativeBackendError("whole seven-axis read deadline expired")
                batch.append(axis)
            self.capture._drain("finish_live_feedback")
            finished=self.clock()
            if finished>deadline:
                raise NativeBackendError("whole seven-axis read deadline expired")
            positions,velocities,axis_times,modes,faults,status_times=[],[],[],[],[],[]
            for i,axis in enumerate(batch):
                q,v,mode=(axis[name] for name in ("position","velocity","mode"))
                qtime,vtime,mtime=(r["rx"]["local_rx_monotonic"] for r in (q,v,mode))
                if any(not 0<=finished-stamp<=self._maximum_age() for stamp in (qtime,vtime,mtime)):
                    raise NativeBackendError(f"J{i+1} parameter receipt expired within batch")
                state=self.capture.statuses.get(i+1)
                report=self.capture.faults.get(i+1)
                fresh=lambda record: record is not None and 0<=finished-record["local_rx_monotonic"]<=self._maximum_age()
                valid=[r for r in (state,report) if fresh(r)]
                code=self.fault_codes[i] or (0 if valid else None)
                stamp=min([mtime]+[r["local_rx_monotonic"] for r in valid]) if valid else None
                if self.fault_codes[i] and self.fault_receipts[i] is not None:
                    stamp=min(mtime,self.fault_receipts[i]["local_rx_monotonic"])
                positions.append(q["value"]); velocities.append(v["value"])
                axis_times.append(min(qtime,vtime)); modes.append("MIT" if mode["value"]==0 else f"native_mode_{mode['value']}")
                faults.append(code); status_times.append(stamp)
            self.feedback_sequence+=1
            feedback=NativeFeedback(session_id,"live_native",finished,self.feedback_sequence,
                tuple(positions),tuple(velocities),tuple(axis_times),tuple(faults),tuple(modes),tuple(status_times))
            self.last_batch=batch
            self.last_feedback=feedback
            if self.origin_rad is not None and positions[2]>self.origin_rad[2]+1e-9:
                self._j3_increased=True
            self._update_stability(feedback,finished)
            # Keep bounded in-process diagnostics; sequence numbers never reset.
            self.capture.frames[:]=self.capture.frames[-4096:]
            self.capture.queries[:]=self.capture.queries[-128:]
            return feedback
        except Exception as exc:
            self._latch(f"feedback: {exc}")
            raise NativeBackendError(self.fault) from exc
        finally:
            self._read_deadline=None

    def _config_digest(self):
        return hashlib.sha256(json.dumps(self.config,sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()

    def admission_errors(self):
        errors=list(config_errors(self.config))
        if self.config.get('admission_policy')=='tony-github-defaults-v1':
            return errors
        if self.config.get("synthetic_only") is True or self.config.get("hardware_writes_supported") is False:
            errors.append("synthetic or explicitly non-hardware configuration")
        if any(word in str(self.config.get("status","")).upper() for word in ("CANDIDATE","BLOCKED","INCOMPLETE")):
            errors.append("candidate or incomplete configuration cannot authorize hardware")
        if self.config.get("holding_gains_verified") is not True:
            errors.append("holding_gains_verified must be true after physical acceptance")
        if self.config.get("hardware_loss_response_verified") is not True:
            errors.append("hardware_loss_response_verified must be true; timeout=0 is not a watchdog")
        if self.config.get("process_exit_ownership_policy") != "retain_until_supported_disable":
            errors.append("process_exit_ownership_policy must retain_until_supported_disable")
        return errors

    def _feedback_check(self, feedback, now):
        if feedback is not self.last_feedback or feedback.session_id != self.session_id or feedback.source!="live_native":
            raise NativeBackendError("exact latest feedback from this backend/session required")
        stamps=tuple(feedback.axis_timestamps)+tuple(feedback.status_timestamps)+(feedback.acquired_at,)
        if any(not _finite(stamp) or not self.capture.opened_at<=stamp<=now or now-stamp>self._maximum_age() for stamp in stamps):
            raise NativeBackendError("feedback, fault or mode receipt is unknown/stale")
        if any(type(code) is not int or code!=0 for code in feedback.fault_codes) or any(mode!="MIT" for mode in feedback.control_modes):
            raise NativeBackendError("fresh zero fault and MIT mode required for all axes")
        for i,(q,v,joint) in enumerate(zip(feedback.positions_rad,feedback.velocities_rad_s,self.config["joints"])):
            if self.config.get('admission_policy')!='tony-github-defaults-v1' and not joint["native_min_rad"]<=q<=joint["native_max_rad"]:
                raise NativeBackendError(f"J{i+1} actual angle outside accepted native bounds")
            if self.config['velocity_tolerance_rad_s'] is not None and abs(v)>joint["max_speed_rad_s"]+self.config["velocity_tolerance_rad_s"]:
                raise NativeBackendError(f"J{i+1} actual speed outside accepted bounds")

    def authorize_motion(self, session_id, feedback, now=None, *, allow_motion=False):
        if allow_motion is not True:
            raise PermissionError("explicit motion authorization is required")
        if not self.opened or self.authorization is not None or self.possibly_enabled or self.fault:
            raise NativeBackendError("backend is closed, already owned, or faulted")
        errors=self.admission_errors()
        if errors:
            raise NativeBackendError("; ".join(errors))
        actual_now=self.clock()
        if now is not None and (not _finite(now) or now>actual_now):
            raise NativeBackendError("invalid authorization time")
        if session_id != self.session_id:
            raise NativeBackendError("wrong current session_id")
        self._feedback_check(feedback,actual_now)
        if self.config.get('admission_policy')=='tony-github-defaults-v1' and self.thermal_return_requested:
            raise NativeBackendError('upstream temperature stop threshold is active before enable')
        if self.config['velocity_tolerance_rad_s'] is not None and any(abs(v)>self.config["velocity_tolerance_rad_s"] for v in feedback.velocities_rad_s):
            raise NativeBackendError("measured standstill required before ownership")
        for i in range(7):
            state=self.capture.statuses.get(i+1)
            if (state is None or actual_now-state["local_rx_monotonic"]>self._maximum_age()
                    or (self.config.get('admission_policy')!='tony-github-defaults-v1' and state["decoded"]["operation_state_bits"]!=0)):
                raise NativeBackendError("fresh reset/disabled state required before taking motor ownership")
        # Quantization can turn a nominal measured-position hold into an
        # actual negative J3 target. Reject the whole grant BEFORE any frame,
        # rather than discovering that after priming other axes.
        self.startup_quantization_preview=[]
        for i,joint in enumerate(self.config["joints"]):
            packet=encode_mit(i+1,pos=feedback.positions_rad[i],vel=0.0,
                              kp=joint["kp"],kd=joint["kd"],tau=0.0)
            effective=decode_mit_effective(packet)["effective_position_rad"]
            self.startup_quantization_preview.append({"motor_id":i+1,
                "requested_position_rad":feedback.positions_rad[i],"effective_position_rad":effective,
                "position_quantization_error_rad":effective-feedback.positions_rad[i]})
            if self.config.get('admission_policy')!='tony-github-defaults-v1' and not joint["native_min_rad"]<=effective<=joint["native_max_rad"]:
                raise NativeBackendError(f"J{i+1} quantized startup hold exceeds accepted native bounds")
            if self.config['position_tolerance_rad'] is not None and abs(effective-feedback.positions_rad[i])>self.config["position_tolerance_rad"]:
                raise NativeBackendError(f"J{i+1} wire quantization exceeds accepted hold tolerance")
            if self.config.get('admission_policy')!='tony-github-defaults-v1' and i==2 and effective<feedback.positions_rad[i]-1e-12:
                raise NativeBackendError("J3 startup hold would decrease after wire quantization; no silent rounding-up or motion")
        token=MotionAuthorization(session_id,self._config_digest(),actual_now,feedback.sequence,self.backend_id)
        self.authorization=token
        if hasattr(self._transport,"_bind_motion_capability"):
            self._transport._bind_motion_capability(token)
        self.origin_rad=tuple(feedback.positions_rad)
        self._hold_targets=list(self.origin_rad)
        self._hold_gains=tuple((j["kp"],j["kd"]) for j in self.config["joints"])
        self.phase="authorized_unenabled"
        self._stable_origin_since=None
        self._update_stability(feedback,actual_now)
        return token

    def _authorization_check(self):
        token=self.authorization
        if not isinstance(token,MotionAuthorization) or token.backend_id!=self.backend_id or token.session_id!=self.session_id:
            raise NativeBackendError("no current backend-issued motion authorization")
        if self._config_digest()!=token.config_sha256:
            self._latch("configuration changed after authorization")
            raise NativeBackendError(self.fault)
        if self.fault:
            raise NativeBackendError(self.fault)
        self._feedback_check(self.last_feedback,self.clock())

    def _submit(self, packet, kind, motor_id, *, values=None):
        validate_actuator_packet(packet)
        record={"kind":kind,"motor_id":motor_id,"requested_at_monotonic":self.clock(),
            "arbitration_id":packet.arbitration_id,"data_hex":packet.data.hex(),"values":values,"submitted":False}
        self.submissions.append(record)
        self.submissions[:]=self.submissions[-4096:]
        try:
            self._transport.write_actuator_packet(packet,authorization=self.authorization)
        except Exception as exc:
            record.update(returned_at_monotonic=self.clock(),error=f"{type(exc).__name__}: {exc}")
            self._latch(f"{kind} submission J{motor_id}: {exc}")
            raise NativeBackendError(self.fault) from exc
        self.submission_sequence+=1
        record.update(submitted=True,submission_sequence=self.submission_sequence,returned_at_monotonic=self.clock())
        # Submission proves CAN_Write only, not physical target attainment.

    def _send_mit(self, index, *, pos, vel, kp, kd, tau):
        if not self.opened or self.authorization is None:
            raise NativeBackendError("CAN and backend-issued motion authorization required")
        packet=encode_mit(index+1,pos=pos,vel=vel,kp=kp,kd=kd,tau=tau)
        expected_gains=self._hold_gains[index]
        is_hold=(self.possibly_enabled and pos==self._hold_targets[index] and vel==0.0
                 and (kp,kd)==expected_gains and tau==0.0)
        if self.fault and not is_hold:
            raise NativeBackendError("faulted backend permits only its already owned zero-velocity hold")
        if not self.fault:
            try:
                self._authorization_check()
                joint=self.config["joints"][index]
                effective=decode_mit_effective(packet)
                github=self.config.get('admission_policy')=='tony-github-defaults-v1'
                if (not github and not joint["native_min_rad"]<=pos<=joint["native_max_rad"]) or abs(vel)>joint["max_speed_rad_s"]:
                    raise NativeBackendError("MIT target exceeds accepted site bounds/speed")
                if ((not github and not joint["native_min_rad"]<=effective["effective_position_rad"]<=joint["native_max_rad"])
                        or abs(effective["effective_velocity_rad_s"])>joint["max_speed_rad_s"]):
                    raise NativeBackendError("quantized MIT target exceeds accepted site bounds/speed")
                if (kp,kd)!=expected_gains or tau!=0.0:
                    raise NativeBackendError("only accepted fixed holding gains and zero feed-forward torque are supported")
                if not self.enabled and (pos!=self.origin_rad[index] or vel!=0):
                    raise NativeBackendError("unenabled priming must exactly hold this session's measured origin")
                if (self.config.get('admission_policy')!='tony-github-defaults-v1' and self.enabled and index==2 and pos<self._hold_targets[2]-1e-12
                        and (not self._j3_increased or self.config.get("j3_reverse_motion_verified") is not True)):
                    raise NativeBackendError("J3 reverse motion is not verified")
            except Exception as exc:
                if self.possibly_enabled:
                    self._latch(exc)
                raise
        # Ownership is conservative from the first actuator frame, including a
        # partial priming failure. Never release the bus on uncertain ownership.
        self.possibly_enabled=True
        self._submit(packet,"mit",index+1,values={"pos":pos,"vel":vel,"kp":kp,"kd":kd,"tau":tau})
        self.submissions[-1]["effective_wire"]=decode_mit_effective(packet)
        self.submissions[-1]["position_quantization_error_rad"]=self.submissions[-1]["effective_wire"]["effective_position_rad"]-pos
        self.last_targets[index]=pos
        self._hold_targets[index]=pos
        self._last_command_velocities[index]=vel
        self._effective_targets[index]=self.submissions[-1]["effective_wire"]["effective_position_rad"]
        if not self.enabled:
            self._primed.add(index)

    def _wait_operation_state(self, expected, sequence_before, driver_before):
        deadline=self.clock()+self._clock_budget()
        self._read_deadline=deadline
        try:
            while self.clock()<=deadline:
                self.capture._drain("actuator_operation_ack")
                if self.fault:
                    raise NativeBackendError(self.fault)
                states=[self.capture.statuses.get(i) for i in range(1,8)]
                if all(record is not None and record["receive_sequence"]>sequence_before
                       and (driver_before is None or record["driver_timestamp_us"]>driver_before)
                       and record["decoded"]["operation_state_bits"]==expected
                       and record["decoded"]["fault_flags_raw"]==0 for record in states):
                    return
                remaining=deadline-self.clock()
                if remaining<=0:
                    break
                self.sleep(min(.002,remaining))
        finally:
            self._read_deadline=None
        self._latch(f"new seven-axis operation state {expected} was not acknowledged")
        raise NativeBackendError(self.fault)

    def enable_all(self):
        self._authorization_check()
        if self.enabled or len(self._primed)!=7:
            raise NativeBackendError("seven exact startup-hold submissions required before one enable")
        before,driver=self.capture.sequence,self.capture.driver_watermark
        self.possibly_enabled=True
        self.phase="enabling"
        try:
            for i in range(7):
                self._submit(control_packet(i+1,enable=True),"enable",i+1)
            self._wait_operation_state(2,before,driver)
        except Exception as exc:
            self._latch(f"enable/acknowledgement: {exc}")
            raise NativeBackendError(self.fault) from exc
        self.enabled=True
        self.phase="owned"

    def _update_stability(self, feedback, now):
        if self.origin_rad is None:
            return
        if self.config.get('admission_policy')=='tony-github-defaults-v1':
            self._last_feedback_time=now
            return
        continuous=self._last_feedback_time is None or 0<now-self._last_feedback_time<=self._clock_budget()
        self._last_feedback_time=now
        stable=(continuous and all(abs(q-origin)<=self.config["position_tolerance_rad"] for q,origin in zip(feedback.positions_rad,self.origin_rad))
                and all(abs(v)<=self.config["velocity_tolerance_rad_s"] for v in feedback.velocities_rad_s)
                and all(code==0 for code in feedback.fault_codes) and all(mode=="MIT" for mode in feedback.control_modes))
        if stable:
            if self._stable_origin_since is None:
                self._stable_origin_since=now
        else:
            self._stable_origin_since=None

    def authorize_supported_disable(self, session_id, feedback, now=None, *, supported=False):
        if supported is not True or session_id!=self.session_id:
            raise NativeBackendError("current session and explicit physical support confirmation required")
        self._authorization_check()
        actual_now=self.clock()
        if feedback is not self.last_feedback or now is not None and (not _finite(now) or now>actual_now):
            raise NativeBackendError("exact latest measured feedback required for normal disable")
        if self.config.get('admission_policy')=='tony-github-defaults-v1':
            if (not self.enabled or any(abs(target-origin)>1e-12 for target,origin in zip(self._hold_targets,self.origin_rad))
                    or any(self._last_command_velocities)):
                raise NativeBackendError('return command must finish at this session origin before operator-supported disable')
            self._disable_permission=(feedback.sequence,actual_now)
            return
        if not self.enabled or self._stable_origin_since is None or actual_now-self._stable_origin_since<self.config["stable_dwell_s"]:
            raise NativeBackendError("continuous measured stable session origin required before normal disable")
        if self._last_feedback_time is None or actual_now-self._last_feedback_time>self._clock_budget():
            raise NativeBackendError("measured stability sampling was interrupted")
        if any(abs(target-origin)>self.config["position_tolerance_rad"] for target,origin in zip(self._hold_targets,self.origin_rad)) or any(self._last_command_velocities):
            raise NativeBackendError("zero-velocity target at this session origin required before disable")
        self._disable_permission=(feedback.sequence,actual_now)

    def authorize_github_zero_disable(self,session_id,feedback):
        return self.authorize_github_finish_disable(session_id,feedback,(0.,)*7)

    def authorize_github_finish_disable(self,session_id,feedback,finish):
        if self.config.get('admission_policy')!='tony-github-defaults-v1' or session_id!=self.session_id:
            raise NativeBackendError('GitHub current-session shutdown required')
        self._authorization_check()
        if feedback is not self.last_feedback or not self.enabled:
            raise NativeBackendError('latest live feedback and owned motors required')
        finish=tuple(finish)
        if finish not in ((0.,)*7,tuple(self.origin_rad)):
            raise NativeBackendError('only GitHub zero or immutable session origin may finish')
        if any(abs(q-f)>1e-12 for q,f in zip(self._hold_targets,finish)) or any(self._last_command_velocities):
            raise NativeBackendError('finish command must be sent before shutdown disable')
        self._disable_permission=(feedback.sequence,self.clock())

    def disable_all(self):
        self._authorization_check()
        now=self.clock()
        if self._disable_permission is None or self._disable_permission[0]!=self.last_feedback.sequence or now-self._disable_permission[1]>self._clock_budget():
            raise NativeBackendError("normal disable needs a fresh measured support authorization")
        self._disable_permission=None
        before,driver=self.capture.sequence,self.capture.driver_watermark
        self.phase="disabling"
        try:
            for i in range(7):
                self._submit(control_packet(i+1,enable=False),"disable",i+1)
            self._wait_operation_state(0,before,driver)
        except Exception as exc:
            self._latch(f"disable/acknowledgement: {exc}")
            raise NativeBackendError(self.fault) from exc
        self.enabled=False
        self.possibly_enabled=False
        self.phase="disabled_verified"

    def github_thermal_emergency_disconnect(self):
        if (self.config.get('admission_policy')!='tony-github-defaults-v1'
                or not self.thermal_disconnect_requested or self.authorization is None):
            raise NativeBackendError('only the upstream thermal emergency may use this path')
        before,driver=self.capture.sequence,self.capture.driver_watermark
        for i in range(7):
            self._submit(control_packet(i+1,enable=False),'upstream_thermal_emergency_disable',i+1)
        # The firmware may also report an overtemperature fault; reset-state
        # acknowledgement cannot use the normal zero-fault return protocol.
        deadline=self.clock()+self._clock_budget()
        self._read_deadline=deadline
        try:
            while self.clock()<deadline:
                self.capture._drain('github_thermal_reset_ack')
                states=[self.capture.statuses.get(i) for i in range(1,8)]
                if all(row is not None and row['receive_sequence']>before
                       and (driver is None or row['driver_timestamp_us']>driver)
                       and row['decoded']['state_bits']==0 for row in states):
                    break
                self.sleep(min(.002,max(0,deadline-self.clock())))
            else:
                raise NativeBackendError('thermal disable reset acknowledgement missing; retain controller')
        finally:
            self._read_deadline=None
        self.enabled=False
        self.possibly_enabled=False
        self.phase='github_thermal_emergency_disable_submitted'

    def close_bus(self):
        if self.possibly_enabled or self.enabled:
            raise NativeBackendError("controller ownership must remain until verified supported disable")
        if self.capture is not None:
            self.capture.close()
        self.phase="bus_closed"

    def acknowledge_external_power_off(self, session_id, *, supported=False, motor_power_off_confirmed=False):
        """Manual hardware handoff, never a normal-disable or motion result.

        The caller must carry the current user's explicit physical support and
        externally removed motor power facts. This only relinquishes software
        ownership after those facts; it sends no CAN command and cannot infer
        power-off from a timeout or a stale/empty status cache.
        """
        if (session_id!=self.session_id or not isinstance(session_id,str)
                or supported is not True or motor_power_off_confirmed is not True):
            raise NativeBackendError("current session plus explicit support AND external motor-power-off confirmation required")
        if self.closed:
            raise NativeBackendError("backend is already closed")
        self.external_poweroff_acknowledgement={"session_id":session_id,
            "supported":True,"motor_power_off_confirmed":True,
            "confirmed_at_monotonic":self.clock(),"source":"explicit_current_session_human_confirmation",
            "normal_disable_verified":False,"can_command_sent":False}
        self.release_allowed_after_external_poweroff=True
        self.possibly_enabled=False
        self.enabled=False
        self.authorization=None
        self._disable_permission=None
        self.phase="released_after_external_poweroff"

    def close(self):
        if self.opened or self.possibly_enabled or self.enabled:
            raise NativeBackendError("close the safely released bus before controller")
        self.closed=True
        self.phase="closed"

    def diagnostics(self):
        return {"backend_id":self.backend_id,"session_id":self.session_id,"phase":self.phase,
                "opened":self.opened,"possibly_enabled":self.possibly_enabled,"enabled_acknowledged":self.enabled,
                "fault":self.fault,"latched_fault_codes":self.fault_codes[:],
                "feedback_sequence":self.feedback_sequence,"rx_sequence":self.capture.sequence if self.capture else 0,
                "origin_rad":self.origin_rad,"last_submitted_targets_rad":self.last_targets[:],
                "effective_submitted_targets_rad":self._effective_targets[:],
                "startup_quantization_preview":self.startup_quantization_preview[:],
                "release_allowed_after_external_poweroff":self.release_allowed_after_external_poweroff,
                "external_poweroff_acknowledgement":self.external_poweroff_acknowledgement,
                "successful_submission_sequence":self.submission_sequence,
                "admission_errors":self.admission_errors(),
                "process_death_hold_guaranteed":False,"can_timeout_changed":False}
