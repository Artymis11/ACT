"""Explicit non-motion active-report configuration and telemetry capture.

This is not a strictly read-only operation: the optional command changes each
motor's active-reporting switch. It cannot enable motors, send MIT commands,
switch control modes, write a register, disable, zero, save, or flash. Default
execution loads no DLL and opens no hardware. Both --read-hardware and
--enable-reporting are required for the configuration/capture operation.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import pcan_readonly as readonly

ROOT = Path(__file__).resolve().parent
ENABLE_PAYLOAD = bytes((1, 2, 3, 4, 5, 6, 1, 0))


def reporting_enable_packet(motor_id):
    if type(motor_id) is not int or motor_id not in range(1, 8):
        raise readonly.CaptureError("reporting may be enabled only for motor IDs 1..7")
    return readonly.Packet(0x1800FD00 | motor_id, ENABLE_PAYLOAD)


def validate_reporting_enable(packet):
    expected = reporting_enable_packet(packet.arbitration_id & 0xFF)
    if (packet.arbitration_id != expected.arbitration_id or packet.data != ENABLE_PAYLOAD
            or packet.message_type != readonly.PCAN_MESSAGE_EXTENDED
            or packet.dlc not in (None, 8)):
        raise readonly.CaptureError("only the exact active-report-enable command may be submitted")


class ReportingTransport(readonly.WindowsPcanTransport):
    """Adds one tightly constrained command to an owned PCAN transport."""
    def configure_report_enable(self, motor_id, *, allow_reporting_change=False):
        if allow_reporting_change is not True:
            raise PermissionError("explicit active-reporting configuration is required")
        packet = reporting_enable_packet(motor_id)
        validate_reporting_enable(packet)
        if not self.opened:
            raise readonly.CaptureError("the same diagnostic CAN channel must already be open")
        message = readonly._TPCANMsg()
        message.ID, message.MSGTYPE, message.LEN = packet.arbitration_id, packet.message_type, 8
        message.DATA[:] = packet.data
        self._check(self.lib.CAN_Write(readonly.PCAN_USBBUS1, ctypes.byref(message)),
                    f"J{motor_id} active-report-enable submission")


class StatusReportingSession:
    """One diagnostic channel for configuration and subsequent reception."""
    def __init__(self, transport, *, clock=time.monotonic, sleep=time.sleep):
        self.transport, self.clock = transport, clock
        self.capture = readonly.PcanReadOnlyCapture(transport, clock=clock, sleep=sleep)
        self.actions = []
        self.preflight_queries = []
        self.started = False

    def _decorate(self, telemetry, fatal_error, cleanup_error):
        for action in self.actions:
            latest = self.capture.statuses.get(action["motor_id"])
            rx_before = action.get("rx_sequence_before_submit")
            driver_before = action.get("driver_watermark_before_submit_us")
            action["status_observed_this_session"] = latest is not None
            action["status_observed_after_submit"] = bool(
                action["submitted"] and latest is not None
                and latest["receive_sequence"] > rx_before
                and (driver_before is None or latest["driver_timestamp_us"] > driver_before)
                and latest["local_rx_monotonic"] >= action["requested_at_monotonic"])
            action["status_rx"] = latest
        telemetry.update(
            parent_schema_ref="active-reporting-configuration-v1",
            communication_setting_write_requested=any(a["attempted"] for a in self.actions),
            communication_setting_actions=self.actions,
            only_parameter_read_transmissions=False,
            allowed_transmit_communication_types=[17, 24],
            original_active_reporting_switch_state="unknown",
            automatic_reporting_rollback_attempted=False,
            fatal_error=fatal_error, cleanup_error=cleanup_error,
            diagnostic_channel_closed=not self.capture.opened)
        telemetry["notes"].insert(0, "This telemetry belongs to a session that may submit type-24 active-report configuration before type-17 read-only sampling; retain communication_setting_actions and parent provenance.")
        # Preserve the standard telemetry subreport for the existing diagnostic
        # evidence checker. The outer report explicitly identifies the write.
        return {
            "schema": "active-reporting-configuration-v1",
            "kind": "non_motion_communication_configuration_and_capture",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.capture.session_id,
            "channel": "can0@1000000",
            "configuration_setting": "active_reporting",
            "preflight_parameter_queries": self.preflight_queries,
            "configuration_change_requested": True,
            "configuration_commands": self.actions,
            "configuration_submitted_motor_ids": [a["motor_id"] for a in self.actions if a["submitted"]],
            "configuration_change_submitted": any(a["submitted"] for a in self.actions),
            "configuration_attempted_motor_ids": [a["motor_id"] for a in self.actions if a["attempted"]],
            "original_active_reporting_switch_state": "unknown",
            "automatic_reporting_rollback_attempted": False,
            "type24_submission_is_motor_enable": False,
            "motor_enable_or_disable_requested": False,
            "mit_motion_or_mode_write_requested": False,
            "register_or_flash_write_requested": False,
            "allowed_transmit_communication_types": [17, 24],
            "only_parameter_read_transmissions": False,
            "motion_admitted": False,
            "usable_as_next_control_session_origin": False,
            "all_seven_parameters_read": telemetry.get("all_seven_parameters_read", False),
            "all_parameter_reads_ok": telemetry.get("all_parameter_reads_ok", False),
            "all_seven_status_observed": all(a["status_frame"]["known"] for a in telemetry["axes"]),
            "all_seven_status_fresh_at_finish": all(a["status_frame"]["fresh_at_finish"] for a in telemetry["axes"]),
            "fatal_error": fatal_error,
            "cleanup_error": cleanup_error,
            "diagnostic_channel_closed": not self.capture.opened,
            "telemetry": telemetry,
            "notes": [
                "This operation changes active reporting; it is not a strictly read-only diagnostic.",
                "A submitted type-24 command proves CAN_Write success, not motor acknowledgement or application.",
                "Prior switch states are unknown. The tool never disables reporting to pretend it restored them.",
                "No nonvolatile save/Flash command is sent; persistence across reset or power loss is not assumed.",
                "Status reporting while motor-disabled is firmware-dependent and is not promised.",
                "Type-24 receive status follows the official active-report response table; its original communication type and raw frame are retained.",
                "Only the diagnostic CAN client is closed on completion; no motor disable or motion is sent."]}

    def run(self, *, allow_can_open=False, allow_reporting_change=False,
            cycles=3, passive_before_s=0.1, passive_after_s=0.15,
            timeout_s=0.18, read_timeout_config=False):
        if allow_can_open is not True or allow_reporting_change is not True:
            raise PermissionError("both hardware access and active-reporting change must be explicit")
        if self.started:
            raise readonly.CaptureError("a new session is required for another reporting operation")
        if type(cycles) is not int or not 1 <= cycles <= 20:
            raise readonly.CaptureError("cycles must be in 1..20")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 2
               for v in (passive_before_s, passive_after_s)):
            raise readonly.CaptureError("passive observation durations must be in [0,2] seconds")
        if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 2:
            raise readonly.CaptureError("query timeout must be in (0,2] seconds")
        self.started = True
        fatal_error = cleanup_error = None
        try:
            self.capture.open(allow_can_open=True)
            self.capture.listen(passive_before_s)
            # Confirm current arm communication before submitting any setting.
            # A previously powered-off arm can leave pending driver TX; new
            # query receipts must satisfy the same post-drain matching rules.
            for motor_id in range(1,8):
                for parameter in readonly.CORE_PARAMETERS:
                    query=self.capture.query(motor_id,parameter,timeout_s=timeout_s)
                    self.preflight_queries.append(query)
                    if not query["ok"]:
                        raise readonly.CaptureError(f"J{motor_id} current communication preflight failed; no reporting configuration submitted")
            for motor_id in range(1, 8):
                latest = self.capture.statuses.get(motor_id)
                now = self.clock()
                fresh = latest is not None and 0 <= now-latest["local_rx_monotonic"] <= 0.25
                packet = reporting_enable_packet(motor_id)
                action = {"motor_id": motor_id,
                          "action": "skip_existing_fresh_status" if fresh else "enable_active_reporting",
                          "prior_switch_state": "unknown", "attempted": False, "submitted": False,
                          "requested_at_monotonic": None, "returned_at_monotonic": None,
                          "rx_sequence_before_submit": self.capture.sequence,
                          "driver_watermark_before_submit_us": self.capture.driver_watermark,
                          "arbitration_id": packet.arbitration_id,
                          "arbitration_id_hex": f"0x{packet.arbitration_id:08X}",
                          "data_hex": packet.data.hex()}
                self.actions.append(action)
                if fresh:
                    continue
                action.update(attempted=True, requested_at_monotonic=self.clock())
                try:
                    self.transport.configure_report_enable(motor_id, allow_reporting_change=True)
                except Exception as exc:
                    action["returned_at_monotonic"] = self.clock()
                    action["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                action.update(submitted=True, returned_at_monotonic=self.clock())
            self.capture.capture(cycles=cycles, passive_s=passive_after_s,
                                 timeout_s=timeout_s, probe_fault_registers=False,
                                 read_timeout_config=read_timeout_config)
        except Exception as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"
        finally:
            # A failed later submission never triggers retry, rollback,
            # enable/disable, or reopening a second controller.
            telemetry = self.capture.report()
            try:
                self.capture.close()
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
        return self._decorate(telemetry, fatal_error, cleanup_error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-hardware", action="store_true")
    parser.add_argument("--enable-reporting", action="store_true")
    parser.add_argument("--cycles", type=int, choices=range(1, 21), default=3)
    parser.add_argument("--read-timeout-config", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.read_hardware or not args.enable_reporting:
        print(json.dumps({"mode": "description_only", "can_opened": False, "dll_loaded": False,
                          "configuration_change_requested": False,
                          "message": "Both --read-hardware and --enable-reporting are required. Active reporting is a non-motion configuration change, not motor enable."}))
        return 0 if not args.read_hardware and not args.enable_reporting else 2
    output = args.out or ROOT / "reports" / ("active_reporting_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    if output.exists():
        parser.error("refusing to overwrite an existing hardware report")
    try:
        session = StatusReportingSession(ReportingTransport(allow_can_open=True))
        report = session.run(allow_can_open=True, allow_reporting_change=True,
                             cycles=args.cycles, read_timeout_config=args.read_timeout_config)
    except Exception as exc:
        report = {"schema": "active-reporting-configuration-v1", "configuration_change_requested": True,
                  "motion_admitted": False, "fatal_error": f"{type(exc).__name__}: {exc}",
                  "configuration_commands": [], "configuration_submitted_motor_ids": [],
                  "configuration_attempted_motor_ids": [], "cleanup_error": None,
                  "diagnostic_channel_closed": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(output), "configuration_submitted_motor_ids": report["configuration_submitted_motor_ids"],
                      "all_seven_parameters_read": report.get("all_seven_parameters_read", False),
                      "all_seven_status_observed": report.get("all_seven_status_observed", False),
                      "all_seven_status_fresh_at_finish": report.get("all_seven_status_fresh_at_finish", False),
                      "fatal_error": report.get("fatal_error"), "cleanup_error": report.get("cleanup_error"),
                      "motion_admitted": False}, ensure_ascii=False))
    return 0 if (report.get("all_parameter_reads_ok") and report.get("all_seven_status_fresh_at_finish")
                 and not report.get("fatal_error") and not report.get("cleanup_error")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
