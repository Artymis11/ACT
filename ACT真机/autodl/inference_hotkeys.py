"""Process-scoped inference hotkeys using the Windows message queue.

Constructing this class has no keyboard side effects. Only ``start`` registers
Ctrl+C and Ctrl+V. Callbacks on the message thread only signal intent.
Robot I/O belongs to the inference supervisor thread.
"""

import ctypes
from ctypes import wintypes
import os
import threading


WM_QUIT = 0x0012
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
MOD_CONTROL = 0x0002
VK_C = 0x43
VK_V = 0x56
RETRY_ID = 0x4AC8
FINISH_ID = 0x4AC9


class InferenceHotkeys:
    def __init__(self, on_retry, on_finish, *, _user32=None, _kernel32=None):
        if not callable(on_retry) or not callable(on_finish):
            raise TypeError("Hotkey callbacks must be callable.")
        if (_user32 is None) != (_kernel32 is None):
            raise ValueError("Supply both Windows API fakes together.")
        self._callbacks = {RETRY_ID: on_retry, FINISH_ID: on_finish}
        self._user32 = _user32
        self._kernel32 = _kernel32
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._thread_id = None
        self._closed = False
        self.error = None

    def _load_windows_api(self):
        if self._user32 is not None:
            return
        if os.name != "nt":
            raise OSError("Inference Ctrl+C/Ctrl+V hotkeys require Windows.")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int,
                                               wintypes.UINT, wintypes.UINT]
        self._user32.RegisterHotKey.restype = wintypes.BOOL
        self._user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.UnregisterHotKey.restype = wintypes.BOOL
        self._user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG),
                                             wintypes.HWND, wintypes.UINT,
                                             wintypes.UINT, wintypes.UINT]
        self._user32.PeekMessageW.restype = wintypes.BOOL
        self._user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG),
                                            wintypes.HWND, wintypes.UINT, wintypes.UINT]
        self._user32.GetMessageW.restype = wintypes.BOOL
        self._user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT,
                                                   wintypes.WPARAM, wintypes.LPARAM]
        self._user32.PostThreadMessageW.restype = wintypes.BOOL
        self._kernel32.GetCurrentThreadId.argtypes = []
        self._kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    def start(self):
        """Register both keys, raising if either key is already reserved."""
        with self._lock:
            if self._closed:
                raise RuntimeError("A closed inference hotkey listener cannot restart.")
            if self._thread is not None:
                self.raise_if_failed()
                return self
            self._load_windows_api()
            self._thread = threading.Thread(target=self._run,
                                            name="act-inference-hotkeys", daemon=True)
            self._thread.start()
            if not self._ready.wait(2.0):
                self._closing.set()
                self._post_quit()
                raise TimeoutError("Inference hotkey registration timed out.")
            if self.error is not None:
                self._thread.join(2.0)
                self.raise_if_failed()
            return self

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError(f"Inference Ctrl+C/Ctrl+V listener failed: {self.error}") from self.error

    def _post_quit(self):
        if self._thread_id is not None:
            if not self._user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0):
                # GetMessage may already have returned and destroyed the queue.
                if self._thread is not None and self._thread.is_alive():
                    raise RuntimeError("Could not stop the inference hotkey message thread.")

    def close(self, timeout=2):
        """Unregister both keys on their owning thread and wait for shutdown."""
        if timeout < 0:
            raise ValueError("Hotkey close timeout must be nonnegative.")
        with self._lock:
            if self._closed:
                return
            self._closing.set()
            if self._thread is not None:
                if threading.current_thread() is self._thread:
                    raise RuntimeError("Close inference hotkeys from the supervisor thread.")
                if self._thread.is_alive():
                    self._post_quit()
                    self._thread.join(timeout)
                if self._thread.is_alive():
                    raise TimeoutError("Inference hotkey message thread did not stop.")
            self._closed = True

    def _run(self):
        registered = []
        try:
            self._thread_id = int(self._kernel32.GetCurrentThreadId())
            message = wintypes.MSG()
            # This creates this thread's message queue before close can post to it.
            self._user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
            for hotkey_id, virtual_key, label in ((RETRY_ID, VK_C, "Ctrl+C"),
                                                   (FINISH_ID, VK_V, "Ctrl+V")):
                if self._closing.is_set():
                    return
                if not self._user32.RegisterHotKey(None, hotkey_id, MOD_CONTROL | MOD_NOREPEAT, virtual_key):
                    raise RuntimeError(f"{label} is already in use or unavailable.")
                registered.append(hotkey_id)
            self._ready.set()
            while not self._closing.is_set():
                result = self._user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result == -1:
                    raise RuntimeError("Windows GetMessageW failed.")
                if result == 0:
                    if not self._closing.is_set():
                        raise RuntimeError("Inference hotkey message queue ended unexpectedly.")
                    return
                if message.message == WM_HOTKEY and not self._closing.is_set():
                    callback = self._callbacks.get(int(message.wParam))
                    if callback is not None:
                        callback()
        except Exception as error:
            self.error = error
        finally:
            for hotkey_id in reversed(registered):
                if not self._user32.UnregisterHotKey(None, hotkey_id) and self.error is None:
                    self.error = RuntimeError(f"Could not unregister inference hotkey {hotkey_id}.")
            self._ready.set()
