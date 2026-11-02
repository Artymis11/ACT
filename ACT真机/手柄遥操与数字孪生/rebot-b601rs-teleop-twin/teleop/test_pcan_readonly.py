"""Injected PCAN tests: no DLL load, CAN connection, or real transmission."""
import contextlib
import ctypes
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import pcan_readonly as p


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def reply(axis=1, parameter=0x7019, value=0.1, stamp=1000, host=p.HOST_ID, result=0):
    raw = struct.pack(p.FORMATS[parameter], value)
    data = struct.pack("<H", parameter) + bytes(2) + raw.ljust(4, b"\0")
    return p.Packet((17 << 24) | (result << 16) | (axis << 8) | host, data, driver_timestamp_us=stamp)


def status(axis=1, faults=0, state=0, stamp=1000, host=p.HOST_ID):
    return p.Packet((2 << 24) | (state << 22) | (faults << 16) | (axis << 8) | host,
                    bytes((0x80, 0, 0x7F, 0xFF, 0x80, 0, 1, 44)), driver_timestamp_us=stamp)


class FakePcan:
    def __init__(self):
        self.queue = []
        self.sent = []
        self.events = []
        self.next_stamp = 1000
        self.on_write = self.answer
        self.read_error = None
        self.write_error = None
        self.close_error = None

    def open(self):
        self.events.append("open")

    def read(self):
        if self.read_error:
            raise self.read_error
        return self.queue.pop(0) if self.queue else None

    def answer(self, packet):
        axis = packet.arbitration_id & 0xFF
        parameter = int.from_bytes(packet.data[:2], "little")
        value = 0 if parameter == 0x7005 else (0xABCD if parameter in (0x3022, 0x3023, 0x7028) else axis / 10)
        self.next_stamp += 200
        self.queue.append(reply(axis, parameter, value, self.next_stamp))

    def write_read_request(self, packet):
        p.validate_request(packet)
        if self.write_error:
            raise self.write_error
        self.sent.append(packet)
        self.on_write(packet)

    def close(self):
        self.events.append("close")
        if self.close_error:
            raise self.close_error


class WireTests(unittest.TestCase):
    def test_active_report_response_type24_preserves_faults_and_original_type(self):
        source=status(axis=3,faults=5,state=2,stamp=1234)
        packet=p.Packet((24<<24)|(source.arbitration_id&0xFFFFFF),source.data,
                        driver_timestamp_us=1234)
        decoded=p.decode_packet(packet)
        self.assertEqual(decoded["kind"],"status")
        self.assertEqual(decoded["communication_type"],24)
        self.assertEqual(decoded["fault_bits"],5)
        self.assertEqual(decoded["state_bits"],2)
        self.assertFalse(decoded["operation_state_is_control_mode"])

    def test_active_report_tx_echo_and_wrong_host_cannot_be_health_evidence(self):
        for packet in (p.Packet(0x1800FD03,bytes.fromhex("0102030405060100")),
                       p.Packet(0x180003FE,bytes(8)),
                       p.Packet(0x180003FD,bytes(8),message_type=0x22)):
            self.assertNotEqual(p.decode_packet(packet)["kind"],"status")

    def test_exact_type17_wire_and_allowlist(self):
        for axis in range(1, 8):
            for parameter in p.PARAMETERS:
                request = p.parameter_read_request(axis, parameter)
                self.assertEqual(request.arbitration_id, 0x1100FD00 | axis)
                self.assertEqual(request.data, struct.pack("<H", parameter) + bytes(6))
                p.validate_request(request)

    def test_forbidden_commands_and_parameters_cannot_cross_writer(self):
        for comm in (0, 1, 2, 3, 4, 6, 7, 18, 21, 22, 23, 24, 25):
            with self.assertRaises(p.CaptureError):
                p.validate_request(p.Packet((comm << 24) | 0xFD01, bytes.fromhex("1970000000000000")))
        for axis, param, host in ((0, 0x7019, 0xFD), (8, 0x7019, 0xFD),
                                   (1, 0x7006, 0xFD), (1, 0x7019, 0xFF), (True, 0x7019, 0xFD)):
            with self.assertRaises(p.CaptureError):
                p.parameter_read_request(axis, param, host)
        valid = p.parameter_read_request(1, 0x7019)
        for packet in (p.Packet(valid.arbitration_id, valid.data[:-1]),
                       p.Packet(valid.arbitration_id, valid.data[:-1] + b"\1"),
                       p.Packet(valid.arbitration_id, valid.data, 0),
                       p.Packet(valid.arbitration_id, valid.data, 0x22)):
            with self.assertRaises(p.CaptureError):
                p.validate_request(packet)

    def test_non_finite_parameter_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            self.assertEqual(p.decode_packet(reply(value=value))["kind"], "rejected")

    def test_integer_modes_and_optional_u32_do_not_use_float_decode(self):
        self.assertEqual(p.decode_packet(reply(parameter=0x7005, value=-1))["value"], -1)
        for parameter in (0x3022, 0x3023, 0x7028):
            self.assertEqual(p.decode_packet(reply(parameter=parameter, value=0xFFFFFFFF))["value"], 0xFFFFFFFF)

    def test_error_result_is_not_a_successful_parameter(self):
        decoded = p.decode_packet(reply(value=0, result=1))
        self.assertEqual(decoded["kind"], "parameter_error")
        self.assertEqual(decoded["read_result"], 1)
        self.assertNotIn("value", decoded)

    def test_status_bits_mapped_but_not_misread_as_mit_mode(self):
        decoded = p.decode_packet(status(axis=3, faults=0b101101, state=2))
        self.assertEqual(decoded["motor_id"], 3)
        self.assertEqual(decoded["fault_bits"], 0b101101)
        self.assertEqual(decoded["state_bits"], 2)
        self.assertTrue(decoded["fault_flags"]["undervoltage"])
        self.assertTrue(decoded["fault_flags"]["uncalibrated"])
        self.assertFalse(decoded["fault_flags"]["overcurrent"])
        self.assertFalse(decoded["operation_state_is_control_mode"])
        self.assertNotIn("position_rad", decoded)
        self.assertNotIn("mode", decoded)

    def test_fault_report_u32_words_and_host_filter(self):
        packet = p.Packet((21 << 24) | (7 << 8) | p.HOST_ID, struct.pack("<II", 0x10001, 1))
        self.assertEqual(p.decode_packet(packet)["fault_raw"], 0x10001)
        self.assertEqual(p.decode_packet(packet)["warning_raw"], 1)
        wrong_host = p.Packet(packet.arbitration_id - 1, packet.data)
        self.assertEqual(p.decode_packet(wrong_host)["kind"], "unrelated")

    def test_firmware_reply_does_not_become_good_status(self):
        packet = p.Packet(0x020001FD, bytes((0, 0xC4, 0x56, 0, 3, 1, 41, 7)))
        self.assertEqual(p.decode_packet(packet)["kind"], "firmware_version")

    def test_bad_frame_shapes_and_non_data_flags_rejected(self):
        valid = reply()
        for flags in (0, 1, 0x03, 0x04, 0x20, 0x22, 0x40, 0x42, 0x80, 0x82):
            self.assertEqual(p.decode_packet(p.Packet(valid.arbitration_id, valid.data, flags))["kind"], "rejected")
        for packet in (p.Packet(0x20000000, valid.data), p.Packet(-1, valid.data),
                       p.Packet(valid.arbitration_id, valid.data, dlc=9),
                       p.Packet(valid.arbitration_id, valid.data[:7]),
                       p.Packet(valid.arbitration_id, valid.data + b"\0"),
                       p.Packet(valid.arbitration_id, valid.data[:2] + b"\1\0" + valid.data[4:])):
            self.assertEqual(p.decode_packet(packet)["kind"], "rejected")


class FakeDll:
    def __init__(self):
        self.condition = p.PCAN_CHANNEL_AVAILABLE
        self.initialize_result, self.read_result, self.close_result = 0, p.PCAN_ERROR_QRCVEMPTY, 0
        self.calls = []

    def CAN_GetValue(self, channel, parameter, buffer, size):
        self.calls.append(("get_value", channel, parameter, size))
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint32)).contents.value = self.condition
        return 0

    def CAN_Initialize(self, *args):
        self.calls.append(("initialize",) + args)
        return self.initialize_result

    def CAN_Uninitialize(self, channel):
        self.calls.append(("uninitialize", channel))
        return self.close_result

    def CAN_Read(self, channel, message_ptr, timestamp_ptr):
        self.calls.append(("read", channel))
        if self.read_result == 0:
            message = ctypes.cast(message_ptr, ctypes.POINTER(p._TPCANMsg)).contents
            message.ID, message.MSGTYPE, message.LEN = 0x110001FD, p.PCAN_MESSAGE_EXTENDED, 8
            message.DATA[:] = reply(value=1.25).data
            stamp = ctypes.cast(timestamp_ptr, ctypes.POINTER(p._TPCANTimestamp)).contents
            stamp.millis, stamp.millis_overflow, stamp.micros = 123, 2, 456
        return self.read_result

    def CAN_Write(self, channel, message_ptr):
        message = ctypes.cast(message_ptr, ctypes.POINTER(p._TPCANMsg)).contents
        self.calls.append(("write", channel, message.ID, bytes(message.DATA), message.MSGTYPE, message.LEN))
        return 0


class WindowsAdapterTests(unittest.TestCase):
    """Exercise ctypes-facing calls with a Python fake DLL only."""
    def setUp(self):
        self.transport = p.WindowsPcanTransport.__new__(p.WindowsPcanTransport)
        self.transport.lib, self.transport.opened = FakeDll(), False
        p._ACTIVE_CHANNELS.clear()

    def tearDown(self):
        p._ACTIVE_CHANNELS.clear()

    def test_transport_constructor_default_denies_before_loading_dll(self):
        with self.assertRaises(PermissionError):
            p.WindowsPcanTransport()

    def test_occupied_channel_never_initializes_or_changes_settings(self):
        self.transport.lib.condition = 2
        with self.assertRaisesRegex(p.CaptureError, "available"):
            self.transport.open()
        self.assertEqual([call[0] for call in self.transport.lib.calls], ["get_value"])

    def test_initialize_uses_exact_channel_and_1m_and_closes_same_channel(self):
        self.transport.open()
        self.assertEqual(self.transport.lib.calls[-1], ("initialize", 0x51, 0x14, 0, 0, 0))
        self.transport.close()
        self.assertEqual(self.transport.lib.calls[-1], ("uninitialize", 0x51))
        self.assertFalse(self.transport.opened)

    def test_wrong_bitrate_caution_releases_diagnostic_client_and_fails(self):
        self.transport.lib.initialize_result = 0x02000000
        with self.assertRaises(p.CaptureError):
            self.transport.open()
        self.assertFalse(self.transport.opened)
        self.assertEqual(self.transport.lib.calls[-1], ("uninitialize", 0x51))

    def test_driver_rx_timestamp_combines_overflow_without_substituting_local_clock(self):
        self.transport.open()
        self.transport.lib.read_result = 0
        packet = self.transport.read()
        self.assertEqual(packet.driver_timestamp_us, 456 + 123000 + 2 * (1 << 32) * 1000)
        self.assertEqual(packet.dlc, 8)
        self.assertEqual(p.decode_packet(packet)["value"], 1.25)

    def test_empty_queue_is_distinguished_from_bus_error(self):
        self.transport.open()
        self.assertIsNone(self.transport.read())
        self.transport.lib.read_result = 0x10
        with self.assertRaisesRegex(p.CaptureError, "0x00000010"):
            self.transport.read()

    def test_ctypes_writer_applies_allowlist_before_any_write(self):
        self.transport.open()
        with self.assertRaises(p.CaptureError):
            self.transport.write_read_request(p.Packet(0x0300FD01, bytes(8)))
        self.assertNotIn("write", [call[0] for call in self.transport.lib.calls])
        request = p.parameter_read_request(7, 0x7005)
        self.transport.write_read_request(request)
        self.assertEqual(self.transport.lib.calls[-1], ("write", 0x51, request.arbitration_id, request.data, 2, 8))

    def test_failed_uninitialize_does_not_report_released_channel(self):
        self.transport.open()
        self.transport.lib.close_result = 0x200
        with self.assertRaises(p.CaptureError):
            self.transport.close()
        self.assertTrue(self.transport.opened)
        self.assertIn(p.PCAN_USBBUS1, p._ACTIVE_CHANNELS)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.transport = Clock(), FakePcan()
        self.capture = p.PcanReadOnlyCapture(self.transport, clock=self.clock, sleep=self.clock.sleep)

    def open(self):
        self.capture.open(allow_can_open=True)

    def test_default_mode_never_constructs_transport(self):
        with patch.object(p, "WindowsPcanTransport", side_effect=AssertionError("hardware construction")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(p.main([]), 0)
        self.assertFalse(__import__("json").loads(output.getvalue())["can_opened"])

    def test_explicit_open_and_new_single_session_required(self):
        with self.assertRaises(PermissionError):
            self.capture.open()
        self.assertEqual(self.transport.events, [])
        self.open()
        with self.assertRaises(p.CaptureError):
            self.capture.open(allow_can_open=True)
        self.capture.close()
        self.capture.close()
        self.assertEqual(self.transport.events, ["open", "close"])
        with self.assertRaises(p.CaptureError):
            self.capture.open(allow_can_open=True)

    def test_seven_axes_have_independent_real_read_records(self):
        self.open()
        report = self.capture.capture(cycles=1, passive_s=0)
        self.assertTrue(report["all_seven_parameters_read"])
        self.assertTrue(report["all_parameter_reads_ok"])
        self.assertEqual(len(self.transport.sent), 21)
        self.assertEqual(len(report["frames"]), 21)
        for axis in report["axes"]:
            self.assertAlmostEqual(axis["position"]["value"], axis["motor_id"] / 10)
            self.assertEqual(axis["mode"]["value"], 0)
            self.assertTrue(axis["position"]["fresh_at_finish"])
            self.assertEqual(axis["position"]["rx"]["local_rx_monotonic"], self.clock.now)
            self.assertGreater(axis["position"]["rx"]["driver_timestamp_us"], 0)
            self.assertFalse(axis["fault_status_register"]["requested"])
        self.assertFalse(report["motion_admitted"])
        with self.assertRaises(p.CaptureError):
            self.capture.capture(cycles=1, passive_s=0)

    def test_optional_fault_and_timeout_probes_stay_explicit(self):
        self.open()
        report = self.capture.capture(cycles=1, passive_s=0, probe_fault_registers=True, read_timeout_config=True)
        self.assertEqual(len(self.transport.sent), 42)
        for axis in report["axes"]:
            for field in ("fault_status_register", "warning_status_register", "can_timeout"):
                self.assertTrue(axis[field]["compatibility_probe"])
                self.assertTrue(axis[field]["ok"])
                self.assertEqual(axis[field]["value"], 0xABCD)

    def test_missing_status_and_fault_never_become_zero(self):
        self.open()
        report = self.capture.capture(cycles=1, passive_s=0)
        for axis in report["axes"]:
            for name in ("status_frame", "fault_report"):
                self.assertFalse(axis[name]["known"])
                self.assertFalse(axis[name]["fresh_at_finish"])
                self.assertNotIn("fault_raw", axis[name])
                self.assertNotIn("fault_bits", axis[name])

    def test_cached_matching_frame_is_drained_and_cannot_satisfy_query(self):
        self.transport.queue = [reply(stamp=900)]
        self.open()
        self.transport.on_write = lambda packet: None
        result = self.capture.query(1, 0x7019, timeout_s=0.01)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["value"])
        self.assertEqual(self.capture.frames[0]["context"], "initial_queue_drain")

    def test_queued_status_is_discarded_as_session_status(self):
        self.transport.queue = [status(stamp=900)]
        self.open()
        self.assertFalse(self.capture.report()["axes"][0]["status_frame"]["known"])

    def test_wrong_axis_host_parameter_and_echo_do_not_match(self):
        self.open()
        def answer(packet):
            self.transport.queue.extend([reply(axis=2, stamp=1001), reply(host=0xFE, stamp=1002),
                                         reply(parameter=0x701B, stamp=1003),
                                         p.Packet(reply(stamp=1004).arbitration_id, reply().data, 0x22, 1004),
                                         reply(value=.7, stamp=1005)])
        self.transport.on_write = answer
        result = self.capture.query(1, 0x7019)
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(result["value"], .7)
        self.assertEqual(result["rx"]["receive_sequence"], 5)

    def test_driver_timestamp_older_than_pre_request_watermark_is_rejected(self):
        self.transport.queue = [reply(axis=2, stamp=2000)]
        self.open()
        self.transport.on_write = lambda packet: self.transport.queue.append(reply(stamp=1999))
        result = self.capture.query(1, 0x7019, timeout_s=.01)
        self.assertFalse(result["ok"])
        self.assertFalse(self.capture.frames[-1]["driver_timestamp_not_regressed"])

    def test_equal_driver_timestamp_cannot_replay_pre_request_sample(self):
        self.transport.queue = [reply(stamp=2000)]
        self.open()
        self.transport.on_write = lambda packet: self.transport.queue.append(reply(stamp=2000))
        self.assertFalse(self.capture.query(1, 0x7019, timeout_s=.01)["ok"])

    def test_stale_status_does_not_overwrite_newer_status(self):
        self.open()
        self.transport.queue = [status(faults=1, stamp=2000), status(faults=0, stamp=1999)]
        self.capture.listen(.01)
        result = self.capture.report()["axes"][0]["status_frame"]
        self.assertEqual(result["fault_bits"], 1)
        self.assertEqual(result["rx"]["receive_sequence"], 1)

    def test_equal_driver_timestamp_cannot_clear_status_fault_or_renew_freshness(self):
        self.open()
        self.transport.queue = [status(faults=1, stamp=2000)]
        self.capture.listen(.01)
        first_time = self.capture.report()["axes"][0]["status_frame"]["rx"]["local_rx_monotonic"]
        self.clock.now += .3
        self.transport.queue = [status(faults=0, stamp=2000)]
        self.capture.listen(.01)
        result = self.capture.report()["axes"][0]["status_frame"]
        self.assertEqual(result["fault_bits"], 1)
        self.assertEqual(result["rx"]["local_rx_monotonic"], first_time)
        self.assertFalse(result["fresh_at_finish"])
        self.assertFalse(self.capture.frames[-1]["driver_timestamp_advanced"])

    def test_equal_driver_timestamp_cannot_clear_fault_report(self):
        self.open()
        self.transport.queue = [p.Packet(0x150001FD, struct.pack("<II", 3, 1), driver_timestamp_us=2000)]
        self.capture.listen(.01)
        self.clock.now += .3
        self.transport.queue = [p.Packet(0x150001FD, bytes(8), driver_timestamp_us=2000)]
        self.capture.listen(.01)
        result = self.capture.report()["axes"][0]["fault_report"]
        self.assertEqual(result["fault_raw"], 3)
        self.assertFalse(result["fresh_at_finish"])

    def test_query_error_code_is_preserved_without_zero_filling(self):
        self.open()
        self.transport.on_write = lambda packet: self.transport.queue.append(reply(parameter=0x3022, value=0, result=2, stamp=2000))
        result = self.capture.query(1, 0x3022)
        self.assertFalse(result["ok"])
        self.assertEqual(result["read_result"], 2)
        self.assertIsNone(result["value"])
        self.assertIsNotNone(result["rx"])
        saved = self.capture.report()["axes"][0]["fault_status_register"]
        self.assertFalse(saved["fresh_at_finish"])
        self.assertTrue(saved["response_observed_fresh_at_finish"])

    def test_cli_partial_optional_probe_never_exits_success_or_overwrites_evidence(self):
        def answer(packet):
            parameter = int.from_bytes(packet.data[:2], "little")
            if parameter in (0x3022, 0x3023):
                self.transport.next_stamp += 200
                self.transport.queue.append(reply(axis=packet.arbitration_id & 0xFF,
                                                  parameter=parameter, value=0, result=1,
                                                  stamp=self.transport.next_stamp))
            else:
                self.transport.answer(packet)
        self.transport.on_write = answer
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent / "reports") as directory:
            path = Path(directory) / "optional.json"
            args = ["--read-hardware", "--cycles", "1", "--passive-seconds", "0",
                    "--probe-fault-registers", "--out", str(path)]
            with patch.object(p, "WindowsPcanTransport", return_value=self.transport):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(p.main(args), 2)
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(saved["all_seven_parameters_read"])
                self.assertFalse(saved["all_parameter_reads_ok"])
                self.assertTrue(saved["diagnostic_channel_closed"])
                self.assertIsNone(saved["fatal_error"])
                self.assertFalse(json.loads(output.getvalue())["all_parameter_reads_ok"])
                original = path.read_bytes()
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        p.main(args)
                self.assertEqual(path.read_bytes(), original)

    def test_current_status_and_fault_keep_raw_rx_identity(self):
        self.open()
        self.transport.queue = [status(axis=3, faults=2, state=2, stamp=2000),
                                p.Packet(0x150003FD, struct.pack("<II", 3, 1), driver_timestamp_us=2200)]
        self.capture.listen(.01)
        result = self.capture.report()["axes"][2]
        self.assertTrue(result["status_frame"]["known"])
        self.assertTrue(result["status_frame"]["fresh_at_finish"])
        self.assertEqual(result["status_frame"]["fault_bits"], 2)
        self.assertEqual(result["fault_report"]["fault_raw"], 3)
        self.assertEqual(result["fault_report"]["rx"]["driver_timestamp_us"], 2200)
        self.clock.now += .3
        after = self.capture.report()["axes"][2]
        self.assertTrue(after["status_frame"]["known"])
        self.assertFalse(after["status_frame"]["fresh_at_finish"])

    def test_parameter_and_mode_freshness_expires_without_more_rx(self):
        self.open()
        self.capture.query(1, 0x7005)
        self.clock.now += .26
        result = self.capture.report()["axes"][0]["mode"]
        self.assertTrue(result["ok"])
        self.assertFalse(result["fresh_at_finish"])

    def test_channel_read_failure_does_not_become_empty_queue_success(self):
        self.open()
        self.transport.read_error = p.CaptureError("bus off")
        with self.assertRaisesRegex(p.CaptureError, "bus off"):
            self.capture.query(1, 0x7019)
        self.transport.read_error = None
        self.capture.close()
        self.assertEqual(self.transport.events[-1], "close")

    def test_write_failure_preserves_failed_query_and_does_not_retry_dangerous_commands(self):
        self.open()
        self.transport.write_error = p.CaptureError("write failed")
        with self.assertRaisesRegex(p.CaptureError, "write failed"):
            self.capture.query(1, 0x7019)
        self.assertEqual(self.transport.sent, [])
        self.assertFalse(self.capture.queries[0]["ok"])
        self.capture.close()

    def test_close_failure_is_not_claimed_as_closed(self):
        self.open()
        self.transport.close_error = p.CaptureError("uninitialize failed")
        with self.assertRaises(p.CaptureError):
            self.capture.close()
        self.assertTrue(self.capture.opened)

    def test_flooded_queue_is_bounded(self):
        self.transport.queue = [reply(stamp=i) for i in range(4097)]
        with self.assertRaisesRegex(p.CaptureError, "4096"):
            self.open()
        self.assertTrue(self.capture.opened)
        self.capture.close()

    def test_invalid_options_and_closed_query_are_rejected(self):
        with self.assertRaises(p.CaptureError):
            self.capture.query(1, 0x7019)
        self.open()
        for timeout in (0, -1, 3, math.nan, math.inf):
            with self.assertRaises(p.CaptureError):
                self.capture.query(1, 0x7019, timeout_s=timeout)
        for duration in (-1, 31, math.nan):
            with self.assertRaises(p.CaptureError):
                self.capture.listen(duration)
        for cycles in (0, 21, True):
            with self.assertRaises(p.CaptureError):
                self.capture.capture(cycles=cycles)


if __name__ == "__main__":
    unittest.main()
