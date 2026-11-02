"""One-owner PCAN/RobStride read-only capture with driver RX evidence.

Default execution describes the tool and never loads a DLL or opens CAN.
The only permitted transmissions are type-17 reads of named parameters. Three
core reads are default; optional compatibility probes remain explicit.
No SDK controller, active-report change, status request, enable, MIT command,
disable, mode write, zero, save, reset, or flash operation is provided.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
HOST_ID = 0xFD
PCAN_USBBUS1, PCAN_BAUD_1M = 0x51, 0x0014
PCAN_MESSAGE_EXTENDED = 0x02
PCAN_ERROR_QRCVEMPTY = 0x20
PCAN_CHANNEL_CONDITION, PCAN_CHANNEL_AVAILABLE = 0x0D, 0x01
PARAMETERS = {0x7019: "position", 0x701B: "velocity", 0x7005: "mode"}
CORE_PARAMETERS = tuple(PARAMETERS)
PARAMETERS.update({0x3022: "fault_status_register", 0x3023: "warning_status_register",
                   0x7028: "can_timeout"})
FORMATS = {0x7019: "<f", 0x701B: "<f", 0x7005: "<b",
           0x3022: "<I", 0x3023: "<I", 0x7028: "<I"}
_ACTIVE_CHANNELS = set()


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Packet:
    arbitration_id: int
    data: bytes
    message_type: int = PCAN_MESSAGE_EXTENDED
    driver_timestamp_us: int = 0
    dlc: int | None = None


def parameter_read_request(motor_id, parameter_id, host_id=HOST_ID):
    """An allowlist, rather than a generic CAN writer exposed to callers."""
    if type(motor_id) is not int or motor_id not in range(1, 8):
        raise CaptureError("only this arm's motor IDs 1..7 may be queried")
    if parameter_id not in PARAMETERS or type(parameter_id) is not int:
        raise CaptureError("parameter is outside the explicit read-only allowlist")
    if type(host_id) is not int or host_id != HOST_ID:
        raise CaptureError("only host 0xFD is permitted; no host fallback")
    return Packet((17 << 24) | (host_id << 8) | motor_id,
                  struct.pack("<H", parameter_id) + bytes(6))


def validate_request(packet):
    if (packet.message_type != PCAN_MESSAGE_EXTENDED or len(packet.data) != 8
            or packet.dlc not in (None, 8)):
        raise CaptureError("only eight-byte extended parameter reads may be sent")
    parameter_id = int.from_bytes(packet.data[:2], "little")
    expected = parameter_read_request(packet.arbitration_id & 0xFF, parameter_id)
    if packet.arbitration_id != expected.arbitration_id or packet.data != expected.data:
        raise CaptureError("transmission rejected by read-only frame allowlist")


def decode_packet(packet):
    """Decode known RX shapes; no status q/v model range is guessed."""
    if (packet.message_type != PCAN_MESSAGE_EXTENDED or len(packet.data) != 8
            or packet.dlc not in (None, 8)
            or not 0 <= packet.arbitration_id <= 0x1FFFFFFF):
        return {"kind": "rejected", "reason": "non-data/echo/RTR/error/FD or malformed frame"}
    comm = (packet.arbitration_id >> 24) & 0x1F
    extra = (packet.arbitration_id >> 8) & 0xFFFF
    motor_id, host = extra & 0xFF, packet.arbitration_id & 0xFF
    if host != HOST_ID or motor_id not in range(1, 8):
        return {"kind": "unrelated", "reason": "different host or motor", "communication_type": comm}
    base = {"motor_id": motor_id, "host_id": host, "communication_type": comm}
    if comm == 17:
        parameter = int.from_bytes(packet.data[:2], "little")
        read_result = extra >> 8
        if read_result:
            return {**base, "kind": "parameter_error", "parameter_id": parameter,
                    "read_result": read_result, "reason": "motor returned a nonzero parameter read result"}
        if parameter not in PARAMETERS or packet.data[2:4] != bytes(2):
            return {**base, "kind": "rejected", "reason": "unknown parameter or nonzero reserved/error bytes"}
        value = struct.unpack(FORMATS[parameter], packet.data[4:5] if parameter == 0x7005 else packet.data[4:8])[0]
        if not math.isfinite(value):
            return {**base, "kind": "rejected", "reason": "non-finite parameter value"}
        return {**base, "kind": "parameter", "parameter_id": parameter,
                "parameter": PARAMETERS[parameter], "value": value}
    if comm in (2, 24):
        # Firmware-version replies also use type 2, and are not motor status.
        if comm == 2 and packet.data[:3] == bytes((0, 0xC4, 0x56)):
            return {**base, "kind": "firmware_version", "status_usable": False}
        fault_flags = (extra >> 8) & 0x3F
        names = ("undervoltage", "overcurrent", "overtemperature",
                 "magnetic_encoder_fault", "stall", "uncalibrated")
        return {**base, "kind": "status", "fault_flags_raw": fault_flags, "fault_bits": fault_flags,
                "fault_namespace":"status_bits_6",
                "status_format": "operation_status_type2" if comm == 2 else "active_report_response_type24",
                "fault_flags": {name: bool(fault_flags & (1 << i)) for i, name in enumerate(names)},
                "operation_state_bits": (extra >> 14) & 3, "state_bits": (extra >> 14) & 3,
                "operation_state_is_control_mode": False,
                "raw_position_u16": int.from_bytes(packet.data[:2], "big"),
                "raw_velocity_u16": int.from_bytes(packet.data[2:4], "big"),
                "raw_torque_u16": int.from_bytes(packet.data[4:6], "big"),
                "temperature_c": int.from_bytes(packet.data[6:8], "big") / 10.0}
    if comm == 21:
        raw=int.from_bytes(packet.data[:4],'little')
        flags={16:'phase_a_overcurrent',14:'stall_overload',9:'position_init_fault',8:'hardware_id_fault',
            7:'encoder_uncalibrated',5:'phase_c_overcurrent',4:'phase_b_overcurrent',3:'overvoltage',
            2:'undervoltage',1:'driver_fault',0:'overtemperature'}
        return {**base, "kind": "fault", "fault_raw":raw,
                "fault_namespace":"fault_word_32",'fault_flags':{name:bool(raw&(1<<bit)) for bit,name in flags.items()},
                "warning_raw": int.from_bytes(packet.data[4:], "little")}
    return {**base, "kind": "unrelated", "reason": "other communication type"}


class _TPCANMsg(ctypes.Structure):
    _fields_ = [("ID", ctypes.c_uint32), ("MSGTYPE", ctypes.c_uint8),
                ("LEN", ctypes.c_uint8), ("DATA", ctypes.c_uint8 * 8)]


class _TPCANTimestamp(ctypes.Structure):
    _fields_ = [("millis", ctypes.c_uint32), ("millis_overflow", ctypes.c_uint16),
                ("micros", ctypes.c_uint16)]


class WindowsPcanTransport:
    """Loads only the isolated PCANBasic DLL after explicit authorization."""
    def __init__(self, *, allow_can_open=False):
        if allow_can_open is not True:
            raise PermissionError("explicit read-only CAN opening is required")
        if os.name != "nt":
            raise CaptureError("Windows PCAN only")
        if any(name == "motorbridge" or name.startswith("motorbridge.") for name in sys.modules):
            raise CaptureError("a fresh process without an SDK controller is required")
        manifest = json.loads((ROOT / "vendor_manifest.json").read_text(encoding="utf-8"))
        relative = "vendor/pcan/PCANBasic.dll"
        entry = next(r for r in manifest["files"] if r["path"].replace("\\", "/") == relative)
        dll_path = ROOT / relative
        if hashlib.sha256(dll_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise CaptureError("isolated PCAN DLL hash mismatch")
        self.dll_path, self.dll_sha256 = str(dll_path), entry["sha256"]
        self.lib = ctypes.WinDLL(str(dll_path))
        prototypes = {
            "CAN_GetValue": ([ctypes.c_uint16, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_uint32),
            "CAN_Initialize": ([ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint16], ctypes.c_uint32),
            "CAN_Uninitialize": ([ctypes.c_uint16], ctypes.c_uint32),
            "CAN_Read": ([ctypes.c_uint16, ctypes.POINTER(_TPCANMsg), ctypes.POINTER(_TPCANTimestamp)], ctypes.c_uint32),
            "CAN_Write": ([ctypes.c_uint16, ctypes.POINTER(_TPCANMsg)], ctypes.c_uint32),
        }
        for name, (argtypes, restype) in prototypes.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = argtypes, restype
        self.opened = False

    @staticmethod
    def _check(status, operation):
        if status:
            raise CaptureError(f"{operation} returned PCAN status 0x{status:08X}")

    def open(self):
        if self.opened or PCAN_USBBUS1 in _ACTIVE_CHANNELS:
            raise CaptureError("this process already owns can0")
        condition = ctypes.c_uint32()
        self._check(self.lib.CAN_GetValue(PCAN_USBBUS1, PCAN_CHANNEL_CONDITION,
                                        ctypes.byref(condition), ctypes.sizeof(condition)), "channel condition")
        if condition.value != PCAN_CHANNEL_AVAILABLE:
            raise CaptureError(f"can0 must be available before capture, condition=0x{condition.value:X}")
        result = self.lib.CAN_Initialize(PCAN_USBBUS1, PCAN_BAUD_1M, 0, 0, 0)
        # CAUTION can report initialized-at-another-bitrate: release our client,
        # then fail, never accept a successful-looking connection at a wrong rate.
        if result == 0x02000000:
            self.lib.CAN_Uninitialize(PCAN_USBBUS1)
        self._check(result, "initialize can0 at 1Mbps")
        self.opened = True
        _ACTIVE_CHANNELS.add(PCAN_USBBUS1)

    def read(self):
        if not self.opened:
            raise CaptureError("CAN transport is closed")
        message, stamp = _TPCANMsg(), _TPCANTimestamp()
        status = self.lib.CAN_Read(PCAN_USBBUS1, ctypes.byref(message), ctypes.byref(stamp))
        if status == PCAN_ERROR_QRCVEMPTY:
            return None
        self._check(status, "read")
        timestamp_us = stamp.micros + 1000 * stamp.millis + (1 << 32) * 1000 * stamp.millis_overflow
        return Packet(message.ID, bytes(message.DATA[:message.LEN]), message.MSGTYPE, timestamp_us, message.LEN)

    def write_read_request(self, packet):
        validate_request(packet)
        if not self.opened:
            raise CaptureError("CAN transport is closed")
        message = _TPCANMsg()
        message.ID, message.MSGTYPE, message.LEN = packet.arbitration_id, packet.message_type, 8
        message.DATA[:] = packet.data
        self._check(self.lib.CAN_Write(PCAN_USBBUS1, ctypes.byref(message)), "parameter read request")

    def close(self):
        if not self.opened:
            return
        self._check(self.lib.CAN_Uninitialize(PCAN_USBBUS1), "uninitialize this diagnostic channel")
        self.opened = False
        _ACTIVE_CHANNELS.discard(PCAN_USBBUS1)


class PcanReadOnlyCapture:
    def __init__(self, transport, *, clock=time.monotonic, sleep=time.sleep):
        self.transport, self.clock, self.sleep = transport, clock, sleep
        self.opened = False
        self.session_id = uuid.uuid4().hex
        self.opened_at = None
        self.sequence = 0
        self.request_sequence = 0
        self.frames, self.queries = [], []
        self.statuses, self.faults = {}, {}
        self.driver_watermark = None
        self.capture_started = False

    def open(self, *, allow_can_open=False):
        if allow_can_open is not True:
            raise PermissionError("explicit read-only CAN opening is required")
        if self.opened or self.opened_at is not None:
            raise CaptureError("each capture object is a single new session")
        self.transport.open()
        self.opened, self.opened_at = True, self.clock()
        self._drain("initial_queue_drain", accept_status=False)

    def _receive(self, context, *, accept_status=True):
        packet = self.transport.read()
        if packet is None:
            return None
        self.sequence += 1
        now = self.clock()
        decoded = decode_packet(packet)
        stamp = packet.driver_timestamp_us
        timestamp_valid = type(stamp) is int and stamp >= 0
        not_regressed = timestamp_valid and (self.driver_watermark is None or stamp >= self.driver_watermark)
        advanced = timestamp_valid and (self.driver_watermark is None or stamp > self.driver_watermark)
        record = {"receive_sequence": self.sequence, "local_rx_monotonic": now,
                  "driver_timestamp_us": stamp, "driver_timestamp_not_regressed": not_regressed,
                  "driver_timestamp_advanced": advanced,
                  "arbitration_id": packet.arbitration_id, "arbitration_id_hex": f"0x{packet.arbitration_id:08X}",
                  "message_type": packet.message_type,
                  "dlc": len(packet.data) if packet.dlc is None else packet.dlc,
                  "data_hex": packet.data.hex(),
                  "context": context, "decoded": decoded}
        self.frames.append(record)
        if timestamp_valid:
            self.driver_watermark = max(stamp, self.driver_watermark if self.driver_watermark is not None else stamp)
        # Equal driver timestamps can be a replay of one recorded RX. They may
        # not refresh a status timestamp or replace a reported fault with zero.
        if accept_status and advanced and decoded["kind"] in ("status", "fault"):
            target = self.statuses if decoded["kind"] == "status" else self.faults
            target[decoded["motor_id"]] = record
        return record

    def _drain(self, context, *, accept_status=True, maximum=4096):
        # A flooded channel is a failed capture, not an unbounded blocking drain.
        for _ in range(maximum):
            if self._receive(context, accept_status=accept_status) is None:
                return
        raise CaptureError("RX queue did not drain within 4096 frames")

    def query(self, motor_id, parameter_id, *, timeout_s=0.18):
        packet = parameter_read_request(motor_id, parameter_id)
        if not self.opened:
            raise CaptureError("capture is closed")
        if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 2:
            raise CaptureError("query timeout must be in (0, 2] seconds")
        self._drain("before_parameter_request")
        self.request_sequence += 1
        before_rx, before_driver = self.sequence, self.driver_watermark
        requested = self.clock()
        request = {"request_sequence": self.request_sequence, "motor_id": motor_id,
                   "parameter_id": parameter_id, "parameter": PARAMETERS[parameter_id],
                   "requested_at_monotonic": requested, "rx_sequence_before_request": before_rx,
                   "driver_watermark_before_request_us": before_driver,
                   "request_arbitration_id": packet.arbitration_id, "request_data_hex": packet.data.hex(),
                   "ok": False, "value": None, "rx": None}
        self.queries.append(request)
        self.transport.write_read_request(packet)
        deadline = requested + timeout_s
        while self.clock() < deadline:
            record = self._receive("parameter_response_window")
            if record is None:
                self.sleep(min(0.002, max(0, deadline - self.clock())))
                continue
            decoded = record["decoded"]
            if (decoded["kind"] in ("parameter", "parameter_error") and decoded["motor_id"] == motor_id
                    and decoded["parameter_id"] == parameter_id
                    and record["receive_sequence"] > before_rx
                    and requested <= record["local_rx_monotonic"] <= deadline
                    and record["driver_timestamp_not_regressed"]
                    and (before_driver is None or record["driver_timestamp_us"] > before_driver)):
                if decoded["kind"] == "parameter_error":
                    request.update(error=f"motor read_result=0x{decoded['read_result']:02X}",
                                   read_result=decoded["read_result"], rx=record)
                else:
                    request.update(ok=True, value=decoded["value"], rx=record)
                return request
        request["error"] = "no matching post-drain response within request window"
        return request

    def listen(self, duration_s):
        if not self.opened:
            raise CaptureError("capture is closed")
        if not math.isfinite(duration_s) or not 0 <= duration_s <= 30:
            raise CaptureError("passive duration must be in [0, 30] seconds")
        deadline = self.clock() + duration_s
        while self.clock() < deadline:
            if self._receive("passive_listener") is None:
                self.sleep(min(0.002, max(0, deadline - self.clock())))

    @staticmethod
    def _observed_status(record, finished, maximum_age):
        if record is None:
            return {"known": False, "observed_this_session": False,
                    "fresh_at_finish": False, "rx": None}
        age = finished - record["local_rx_monotonic"]
        return {**record["decoded"], "known": True, "observed_this_session": True,
                "fresh_at_finish": 0 <= age <= maximum_age,
                "age_at_finish_s": age, "rx": record}

    def report(self, *, maximum_age_s=0.25):
        finished = self.clock()
        axes = []
        for motor_id in range(1, 8):
            axis = {"motor_id": motor_id, "model": "rs-06" if motor_id <= 3 else "rs-00"}
            for parameter, name in PARAMETERS.items():
                matches = [r for r in self.queries if r["motor_id"] == motor_id and r["parameter_id"] == parameter]
                result = dict(matches[-1]) if matches else {"ok": False, "value": None, "rx": None}
                result["requested"] = bool(matches)
                result["compatibility_probe"] = parameter not in CORE_PARAMETERS
                rx = result.get("rx")
                result["response_observed_fresh_at_finish"] = bool(rx and 0 <= finished-rx["local_rx_monotonic"] <= maximum_age_s)
                result["fresh_at_finish"] = result["ok"] is True and result["response_observed_fresh_at_finish"]
                if rx:
                    result["age_at_finish_s"] = finished-rx["local_rx_monotonic"]
                axis[name] = result
            axis["status_frame"] = self._observed_status(self.statuses.get(motor_id), finished, maximum_age_s)
            axis["fault_report"] = self._observed_status(self.faults.get(motor_id), finished, maximum_age_s)
            axes.append(axis)
        return {"schema": "pcan-read-only-receipt-v1", "kind": "physical_read_only_driver_capture",
                "session_id": self.session_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "channel": "can0@1000000", "opened_at_monotonic": self.opened_at,
                "capture_started_monotonic": self.opened_at,
                "finished_at_monotonic": finished, "freshness_max_age_s": maximum_age_s,
                "axis_count": 7, "axes": axes, "frames": self.frames, "queries": self.queries,
                "all_parameter_queries_ok": bool(self.queries) and all(r["ok"] for r in self.queries),
                "all_parameter_reads_ok": bool(self.queries) and all(r["ok"] for r in self.queries),
                "all_seven_parameters_read": all(all(a[PARAMETERS[p]]["ok"] for p in CORE_PARAMETERS) for a in axes),
                "only_parameter_read_transmissions": True, "allowed_transmit_communication_types": [17],
                "allowed_transmit_parameters": [f"0x{p:04X}" for p in PARAMETERS],
                "sdk_controller_imported": "motorbridge" in sys.modules,
                "motion_admitted": False, "usable_as_next_control_session_origin": False,
                "notes": [
                    "local_rx_monotonic is CAN_Read observation time; driver_timestamp_us is the driver's reception timestamp, with a different epoch.",
                    "RX sequence counts actual CAN_Read returns within this new session, not SDK cache calls.",
                    "Parameter matching requires a drained RX queue, exact host/axis/parameter and a new driver timestamp after the previous watermark.",
                    "The motor protocol has no request nonce. Queue-drain correlation cannot prove that an arbitrarily delayed old matching response is impossible.",
                    "Absent type-2/type-24 status and type-21 fault packets remain unknown. Type-24 RX uses the official active-report response table; original communication type is retained. Operation-state bits are not MIT control mode.",
                    "Diagnostic capture neither enables motors nor verifies mechanical limits, hold, return, or support."]}

    def capture(self, *, cycles=3, passive_s=0.2, timeout_s=0.18,
                probe_fault_registers=False, read_timeout_config=False):
        if self.capture_started:
            raise CaptureError("capture may run only once per new session")
        if type(cycles) is not int or not 1 <= cycles <= 20:
            raise CaptureError("cycles must be in 1..20")
        self.capture_started = True
        self.listen(passive_s)
        parameters = list(CORE_PARAMETERS)
        if probe_fault_registers:
            parameters.extend((0x3022, 0x3023))
        if read_timeout_config:
            parameters.append(0x7028)
        for _ in range(cycles):
            for motor_id in range(1, 8):
                for parameter in parameters:
                    self.query(motor_id, parameter, timeout_s=timeout_s)
        return self.report()

    def close(self):
        if self.opened:
            self.transport.close()
            self.opened = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-hardware", action="store_true")
    parser.add_argument("--cycles", type=int, choices=range(1, 21), default=3)
    parser.add_argument("--passive-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=0.18)
    parser.add_argument("--probe-fault-registers", action="store_true")
    parser.add_argument("--read-timeout-config", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.read_hardware:
        print(json.dumps({"mode": "description_only", "can_opened": False, "dll_loaded": False,
                          "message": "Explicit --read-hardware captures only 0x7019/0x701B/0x7005 reads and passive RX."}))
        return 0
    if not math.isfinite(args.passive_seconds) or not 0 <= args.passive_seconds <= 30:
        parser.error("passive seconds must be in [0,30]")
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 2:
        parser.error("timeout seconds must be in (0,2]")
    output = args.out or ROOT / "reports" / ("pcan_readonly_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    if output.exists():
        parser.error("refusing to overwrite an existing capture")
    capture, report, fatal, cleanup = None, None, None, None
    try:
        capture = PcanReadOnlyCapture(WindowsPcanTransport(allow_can_open=True))
        capture.open(allow_can_open=True)
        report = capture.capture(cycles=args.cycles, passive_s=args.passive_seconds, timeout_s=args.timeout_seconds,
                                 probe_fault_registers=args.probe_fault_registers,
                                 read_timeout_config=args.read_timeout_config)
    except Exception as exc:
        fatal = f"{type(exc).__name__}: {exc}"
        report = capture.report() if capture is not None else {"schema": "pcan-read-only-receipt-v1", "axes": [], "motion_admitted": False}
    finally:
        if capture is not None:
            try:
                capture.close()
            except Exception as exc:
                cleanup = f"{type(exc).__name__}: {exc}"
    report.update(fatal_error=fatal, cleanup_error=cleanup, diagnostic_channel_closed=bool(capture and not capture.opened))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous physical evidence even on failure.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(output), "all_seven_parameters_read": report.get("all_seven_parameters_read", False),
                      "all_parameter_reads_ok": report.get("all_parameter_reads_ok", False),
                      "fatal_error": fatal, "cleanup_error": cleanup, "motion_admitted": False}, ensure_ascii=False))
    return 0 if report.get("all_seven_parameters_read") and report.get("all_parameter_reads_ok") and not fatal and not cleanup else 2


if __name__ == "__main__":
    raise SystemExit(main())
