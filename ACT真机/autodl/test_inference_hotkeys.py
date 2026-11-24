"""Fake Windows API regressions. These tests never register physical hotkeys."""

import queue
import threading
import unittest
from unittest import mock

import inference_hotkeys as hotkeys


class FakeWindowsAPI:
    def __init__(self, fail_key=None):
        self.messages = queue.Queue()
        self.registered = []
        self.unregistered = []
        self.fail_key = fail_key
        self.queue_created = False

    def GetCurrentThreadId(self):
        return threading.get_ident()

    def PeekMessageW(self, message, hwnd, minimum, maximum, remove):
        self.queue_created = True
        return 0

    def RegisterHotKey(self, hwnd, identifier, modifiers, key):
        if not self.queue_created:
            raise AssertionError("Message queue must exist before registration.")
        if key == self.fail_key:
            return 0
        self.registered.append((identifier, modifiers, key))
        return 1

    def UnregisterHotKey(self, hwnd, identifier):
        self.unregistered.append(identifier)
        return 1

    def GetMessageW(self, message, hwnd, minimum, maximum):
        kind, identifier = self.messages.get(timeout=5)
        if kind == -1:
            return -1
        if kind == hotkeys.WM_QUIT:
            return 0
        message._obj.message = kind
        message._obj.wParam = identifier
        return 1

    def PostThreadMessageW(self, thread_id, kind, wparam, lparam):
        if not self.queue_created:
            return 0
        self.messages.put((kind, wparam))
        return 1


class InferenceHotkeyTests(unittest.TestCase):
    def make_listener(self, api=None, on_retry=None, on_finish=None):
        api = api or FakeWindowsAPI()
        listener = hotkeys.InferenceHotkeys(on_retry or (lambda: None),
                                            on_finish or (lambda: None),
                                            _user32=api, _kernel32=api)
        self.addCleanup(listener.close)
        return listener, api

    def test_constructor_has_no_registration_side_effect(self):
        listener, api = self.make_listener()
        self.assertEqual(api.registered, [])
        self.assertIsNone(listener._thread)

    def test_registers_control_c_v_with_no_repeat_and_cleans_up(self):
        listener, api = self.make_listener()
        self.assertIs(listener.start(), listener)
        listener.start()
        modifiers = hotkeys.MOD_CONTROL | hotkeys.MOD_NOREPEAT
        self.assertEqual(api.registered, [(hotkeys.RETRY_ID, modifiers, hotkeys.VK_C),
                                          (hotkeys.FINISH_ID, modifiers, hotkeys.VK_V)])
        listener.close()
        self.assertEqual(api.unregistered, [hotkeys.FINISH_ID, hotkeys.RETRY_ID])
        self.assertFalse(listener._thread.is_alive())
        listener.close()
        with self.assertRaisesRegex(RuntimeError, "cannot restart"):
            listener.start()

    def test_callbacks_are_mapped_and_execute_on_listener_thread(self):
        retry = threading.Event()
        finish = threading.Event()
        callback_threads = []

        def on_retry():
            callback_threads.append(threading.get_ident())
            retry.set()

        listener, api = self.make_listener(on_retry=on_retry, on_finish=finish.set)
        listener.start()
        api.messages.put((hotkeys.WM_HOTKEY, 999))
        api.messages.put((hotkeys.WM_HOTKEY, hotkeys.RETRY_ID))
        api.messages.put((hotkeys.WM_HOTKEY, hotkeys.FINISH_ID))
        self.assertTrue(retry.wait(1))
        self.assertTrue(finish.wait(1))
        self.assertEqual(callback_threads, [listener._thread_id])
        self.assertNotEqual(callback_threads[0], threading.get_ident())

    def test_first_key_conflict_fails_without_partial_registration(self):
        listener, api = self.make_listener(FakeWindowsAPI(fail_key=hotkeys.VK_C))
        with self.assertRaisesRegex(RuntimeError, r"Ctrl\+C is already in use"):
            listener.start()
        self.assertEqual(api.registered, [])
        self.assertEqual(api.unregistered, [])

    def test_second_key_conflict_releases_first_key(self):
        listener, api = self.make_listener(FakeWindowsAPI(fail_key=hotkeys.VK_V))
        with self.assertRaisesRegex(RuntimeError, r"Ctrl\+V is already in use"):
            listener.start()
        self.assertEqual(api.unregistered, [hotkeys.RETRY_ID])

    def test_windows_message_failure_is_exposed_and_unregisters(self):
        listener, api = self.make_listener()
        listener.start()
        api.messages.put((-1, 0))
        listener._thread.join(1)
        with self.assertRaisesRegex(RuntimeError, "GetMessageW failed"):
            listener.raise_if_failed()
        self.assertEqual(api.unregistered, [hotkeys.FINISH_ID, hotkeys.RETRY_ID])

    def test_callback_failure_is_exposed_and_unregisters(self):
        def failing_callback():
            raise ValueError("supervisor intent callback failed")

        listener, api = self.make_listener(on_retry=failing_callback)
        listener.start()
        api.messages.put((hotkeys.WM_HOTKEY, hotkeys.RETRY_ID))
        listener._thread.join(1)
        with self.assertRaisesRegex(RuntimeError, "intent callback failed"):
            listener.raise_if_failed()
        self.assertEqual(api.unregistered, [hotkeys.FINISH_ID, hotkeys.RETRY_ID])

    def test_quit_without_supervisor_close_is_failure(self):
        listener, api = self.make_listener()
        listener.start()
        api.messages.put((hotkeys.WM_QUIT, 0))
        listener._thread.join(1)
        with self.assertRaisesRegex(RuntimeError, "ended unexpectedly"):
            listener.raise_if_failed()

    def test_closing_before_start_is_harmless(self):
        listener, api = self.make_listener()
        listener.close()
        self.assertEqual(api.registered, [])

    def test_unsupported_platform_fails_before_creating_thread(self):
        listener = hotkeys.InferenceHotkeys(lambda: None, lambda: None)
        with mock.patch.object(hotkeys.os, "name", "posix"):
            with self.assertRaisesRegex(OSError, "require Windows"):
                listener.start()
        self.assertIsNone(listener._thread)


if __name__ == "__main__":
    unittest.main()
