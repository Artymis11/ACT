"""Active-report-only allowlist and injected session tests; no real DLL/CAN."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pcan_readonly as p
import status_reporting as reporting
from test_pcan_readonly import Clock, FakeDll, FakePcan, status


class FakeReportingTransport(FakePcan):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.reporting_sent = []
        self.reporting_attempts = []
        self.fail_axis = None
        self.emit_status = True
        self.initial_status_axes = []
        self.emitted_initial = False
        self.raw_type24 = False

    def read(self):
        if self.initial_status_axes and not self.emitted_initial and self.clock.now > 100:
            self.emitted_initial = True
            for axis in self.initial_status_axes:
                self.next_stamp += 200
                self.queue.append(status(axis=axis, stamp=self.next_stamp))
        return super().read()

    def configure_report_enable(self, axis, *, allow_reporting_change=False):
        if allow_reporting_change is not True:
            raise PermissionError("no reporting permission")
        packet = reporting.reporting_enable_packet(axis)
        reporting.validate_reporting_enable(packet)
        self.reporting_attempts.append(axis)
        if self.fail_axis == axis:
            raise p.CaptureError("simulated type24 send failure")
        self.reporting_sent.append(packet)
        if self.emit_status:
            self.next_stamp += 200
            packet = status(axis=axis, stamp=self.next_stamp)
            if self.raw_type24:
                packet = p.Packet((24 << 24) | (axis << 8) | p.HOST_ID,
                                  packet.data, driver_timestamp_us=packet.driver_timestamp_us)
            self.queue.append(packet)


class ReportingWireTests(unittest.TestCase):
    def test_fixed_enable_reporting_command_for_all_axes(self):
        for axis in range(1, 8):
            packet = reporting.reporting_enable_packet(axis)
            self.assertEqual(packet.arbitration_id, 0x1800FD00 | axis)
            self.assertEqual(packet.data, bytes.fromhex("0102030405060100"))
            reporting.validate_reporting_enable(packet)

    def test_no_motor_enable_disable_mit_zero_or_other_configuration_command_allowed(self):
        for command in (0, 1, 2, 3, 4, 6, 7, 17, 18, 21, 22, 23, 25):
            with self.assertRaises(p.CaptureError):
                reporting.validate_reporting_enable(p.Packet((command << 24) | 0xFD01, reporting.ENABLE_PAYLOAD))
        for axis in (0, 8, True, "1"):
            with self.assertRaises(p.CaptureError):
                reporting.reporting_enable_packet(axis)

    def test_reporting_disable_and_payload_changes_rejected(self):
        packet = reporting.reporting_enable_packet(1)
        for index in range(8):
            data = bytearray(packet.data)
            data[index] ^= 1
            with self.assertRaises(p.CaptureError):
                reporting.validate_reporting_enable(p.Packet(packet.arbitration_id, bytes(data)))
        for altered in (p.Packet(packet.arbitration_id, packet.data, message_type=0),
                        p.Packet(packet.arbitration_id, packet.data, message_type=0x22),
                        p.Packet(packet.arbitration_id, packet.data, dlc=7),
                        p.Packet(0x1800FE01, packet.data),
                        p.Packet(0x180001FD, packet.data)):
            with self.assertRaises(p.CaptureError):
                reporting.validate_reporting_enable(altered)

    def test_readonly_writer_still_rejects_type24(self):
        with self.assertRaises(p.CaptureError):
            p.validate_request(reporting.reporting_enable_packet(1))


class ReportingTransportTests(unittest.TestCase):
    def setUp(self):
        self.transport = reporting.ReportingTransport.__new__(reporting.ReportingTransport)
        self.transport.lib, self.transport.opened = FakeDll(), False
        p._ACTIVE_CHANNELS.clear()

    def tearDown(self):
        p._ACTIVE_CHANNELS.clear()

    def test_reporting_method_requires_explicit_permission_and_existing_channel(self):
        with self.assertRaises(PermissionError):
            self.transport.configure_report_enable(1)
        with self.assertRaises(p.CaptureError):
            self.transport.configure_report_enable(1, allow_reporting_change=True)
        self.assertEqual(self.transport.lib.calls, [])

    def test_same_channel_gets_only_exact_type24_frame(self):
        self.transport.open()
        self.transport.configure_report_enable(3, allow_reporting_change=True)
        self.assertEqual(self.transport.lib.calls[-1],
                         ("write", 0x51, 0x1800FD03, reporting.ENABLE_PAYLOAD, 2, 8))
        self.assertEqual(sum(c[0] == "initialize" for c in self.transport.lib.calls), 1)
        self.transport.close()
        self.assertEqual(self.transport.lib.calls[-1], ("uninitialize", 0x51))


class ReportingSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.transport = FakeReportingTransport(self.clock)
        self.session = reporting.StatusReportingSession(self.transport, clock=self.clock, sleep=self.clock.sleep)

    def run_session(self, **kwargs):
        return self.session.run(allow_can_open=True, allow_reporting_change=True,
                                cycles=1, passive_before_s=.01, passive_after_s=.01, **kwargs)

    def test_explicit_flags_required_before_opening_or_configuration(self):
        for hardware, change in ((False, False), (True, False), (False, True)):
            with self.assertRaises(PermissionError):
                self.session.run(allow_can_open=hardware, allow_reporting_change=change)
        self.assertEqual(self.transport.events, [])
        self.assertEqual(self.transport.reporting_attempts, [])

    def test_one_channel_configures_then_reads_and_preserves_setting_change_provenance(self):
        result = self.run_session()
        self.assertEqual(self.transport.events, ["open", "close"])
        self.assertEqual(self.transport.reporting_attempts, list(range(1, 8)))
        self.assertEqual(len(self.transport.sent), 42)
        self.assertTrue(result["all_seven_parameters_read"])
        self.assertTrue(result["all_seven_status_fresh_at_finish"])
        self.assertFalse(result["motion_admitted"])
        self.assertFalse(result["motor_enable_or_disable_requested"])
        self.assertFalse(result["mit_motion_or_mode_write_requested"])
        self.assertFalse(result["register_or_flash_write_requested"])
        self.assertEqual(result["original_active_reporting_switch_state"], "unknown")
        self.assertFalse(result["automatic_reporting_rollback_attempted"])
        for action in result["configuration_commands"]:
            self.assertTrue(action["submitted"])
            self.assertTrue(action["status_observed_after_submit"])
        telemetry = result["telemetry"]
        self.assertFalse(telemetry["only_parameter_read_transmissions"])
        self.assertTrue(telemetry["communication_setting_write_requested"])
        self.assertEqual(telemetry["allowed_transmit_communication_types"], [17, 24])
        self.assertEqual(telemetry["communication_setting_actions"], result["configuration_commands"])

    def test_fresh_existing_status_is_not_reconfigured(self):
        self.transport.initial_status_axes = list(range(1, 8))
        result = self.run_session()
        self.assertEqual(self.transport.reporting_sent, [])
        self.assertEqual(result["configuration_submitted_motor_ids"], [])
        self.assertFalse(result["configuration_change_submitted"])
        self.assertFalse(result["telemetry"]["communication_setting_write_requested"])
        self.assertTrue(all(a["action"] == "skip_existing_fresh_status" for a in result["configuration_commands"]))

    def test_only_missing_axes_are_configured(self):
        self.transport.initial_status_axes = [1, 4]
        result = self.run_session()
        self.assertEqual(self.transport.reporting_attempts, [2, 3, 5, 6, 7])
        self.assertTrue(result["all_seven_status_observed"])

    def test_old_queued_status_does_not_skip_configuration(self):
        self.transport.queue = [status(axis=1, stamp=500)]
        result = self.run_session()
        self.assertIn(1, self.transport.reporting_attempts)
        self.assertEqual(result["telemetry"]["frames"][0]["context"], "initial_queue_drain")

    def test_partial_submission_failure_stops_later_sends_without_rollback(self):
        self.transport.fail_axis = 3
        result = self.run_session()
        self.assertEqual(self.transport.reporting_attempts, [1, 2, 3])
        self.assertEqual(result["configuration_submitted_motor_ids"], [1, 2])
        self.assertIn("type24 send failure", result["fatal_error"])
        self.assertEqual(result["telemetry"]["fatal_error"],result["fatal_error"])
        self.assertEqual(len(self.transport.sent),21)
        self.assertEqual(self.transport.events, ["open", "close"])
        self.assertFalse(result["automatic_reporting_rollback_attempted"])
        self.assertTrue(result["diagnostic_channel_closed"])

    def test_successful_send_without_status_remains_unverified(self):
        self.transport.emit_status = False
        result = self.run_session()
        self.assertEqual(len(result["configuration_submitted_motor_ids"]), 7)
        self.assertFalse(result["all_seven_status_observed"])
        self.assertTrue(result["all_seven_parameters_read"])
        self.assertTrue(all(not a["status_observed_after_submit"] for a in result["configuration_commands"]))

    def test_type24_receive_uses_official_response_format_and_retains_original_type(self):
        self.transport.raw_type24 = True
        result = self.run_session()
        received = [f for f in result["telemetry"]["frames"] if f["decoded"].get("communication_type") == 24]
        self.assertEqual(len(received), 7)
        self.assertTrue(all(f["decoded"]["kind"] == "status" for f in received))
        self.assertTrue(all(f["decoded"]["status_format"] == "active_report_response_type24" for f in received))
        self.assertTrue(result["all_seven_status_observed"])

    def test_timeout_read_is_optional_and_unsupported_fault_registers_are_never_queried(self):
        result = self.run_session(read_timeout_config=True)
        parameters = [int.from_bytes(packet.data[:2], "little") for packet in self.transport.sent]
        self.assertEqual(len(parameters), 49)
        self.assertEqual(set(parameters), {0x7019, 0x701B, 0x7005, 0x7028})
        self.assertTrue(result["all_parameter_reads_ok"])

    def test_cleanup_failure_preserved_separately(self):
        self.transport.close_error = p.CaptureError("close failed")
        result = self.run_session()
        self.assertIsNone(result["fatal_error"])
        self.assertIn("close failed", result["cleanup_error"])
        self.assertEqual(result["telemetry"]["cleanup_error"],result["cleanup_error"])
        self.assertFalse(result["diagnostic_channel_closed"])

    def test_no_response_preflight_never_submits_reporting_configuration(self):
        self.transport.on_write=lambda *_args:None
        result=self.run_session()
        self.assertIn("preflight failed",result["fatal_error"])
        self.assertEqual(self.transport.reporting_attempts,[])
        self.assertEqual(result["configuration_submitted_motor_ids"],[])

    def test_invalid_options_do_not_open_hardware_or_mark_session_used(self):
        for kwargs in ({"cycles":0}, {"cycles":True}, {"passive_before_s":-1},
                       {"passive_after_s":3}, {"timeout_s":0}):
            with self.assertRaises(p.CaptureError):
                self.session.run(allow_can_open=True, allow_reporting_change=True, **kwargs)
        self.assertEqual(self.transport.events, [])
        self.assertFalse(self.session.started)

    def test_session_cannot_be_reused_for_more_configuration(self):
        self.run_session()
        with self.assertRaises(p.CaptureError):
            self.run_session()
        self.assertEqual(self.transport.events, ["open", "close"])


class ReportingCliTests(unittest.TestCase):
    def test_default_and_incomplete_flags_never_construct_transport(self):
        with patch.object(reporting, "ReportingTransport", side_effect=AssertionError("hardware access")):
            for arguments, expected in (([], 0), (["--read-hardware"], 2), (["--enable-reporting"], 2)):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(reporting.main(arguments), expected)
                self.assertFalse(json.loads(output.getvalue())["can_opened"])

    def test_cli_saves_partial_evidence_and_does_not_overwrite(self):
        clock = Clock()
        transport = FakeReportingTransport(clock)
        transport.emit_status = False
        with tempfile.TemporaryDirectory(dir=ROOT_REPORTS) as directory:
            output = Path(directory) / "report.json"
            args = ["--read-hardware", "--enable-reporting", "--cycles", "1", "--out", str(output)]
            session = reporting.StatusReportingSession(transport, clock=clock, sleep=clock.sleep)
            with patch.object(reporting, "ReportingTransport", return_value=transport):
                with patch.object(reporting, "StatusReportingSession", return_value=session):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(reporting.main(args), 2)
                    saved = json.loads(output.read_text(encoding="utf-8"))
                    self.assertFalse(saved["all_seven_status_observed"])
                    self.assertEqual(saved["configuration_submitted_motor_ids"], list(range(1, 8)))
                    before = output.read_bytes()
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            reporting.main(args)
                    self.assertEqual(output.read_bytes(), before)


ROOT_REPORTS = Path(__file__).resolve().parent / "reports"


if __name__ == "__main__":
    unittest.main()
