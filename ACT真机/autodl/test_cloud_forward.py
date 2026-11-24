"""SSH forwarding CLI checks without opening a socket or starting a service."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import paramiko

import cloud_forward


class ForwardOptionsTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.legacy = Path(directory.name) / "legacy_hosts"
        self.pinned = Path(directory.name) / "known_hosts"
        self.key = paramiko.RSAKey.generate(1024)
        self.endpoint = "[example.test]:52795"
        keys = paramiko.HostKeys()
        keys.add(self.endpoint, self.key.get_name(), self.key)
        keys.save(str(self.pinned))
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys = Mock()
        self.client.save_host_keys = Mock()
        self.client.connect = Mock()
        self.transport = Mock(local_compression="zlib@openssh.com", remote_compression="zlib@openssh.com")
        self.client.get_transport = Mock(return_value=self.transport)

    def run_cli(self, *arguments):
        argv = ["forward", "--host", "example.test", "--port", "52795", *arguments]
        output = io.StringIO()
        with patch("sys.argv", argv), patch.dict(os.environ, ACT_SSH_PASSWORD="synthetic-test"), \
             patch.object(cloud_forward, "KNOWN_HOSTS", self.legacy), \
             patch.object(cloud_forward.paramiko, "SSHClient", return_value=self.client), \
             patch.object(cloud_forward, "ForwardServer") as server, \
             patch.object(cloud_forward.threading, "Thread"), patch.object(cloud_forward, "monitor_connection"), \
             redirect_stdout(output), redirect_stderr(output):
            status = cloud_forward.main()
            self.assertNotIn("ACT_SSH_PASSWORD", os.environ)
        self.assertNotIn("synthetic-test", output.getvalue())
        return status, server, output.getvalue()

    def test_compressed_route_uses_only_specified_host_keys(self):
        previous = self.pinned.read_bytes()
        status, server, out = self.run_cli("--compress", "--known-hosts", str(self.pinned),
                                           "--local-port", "8163", "--remote-port", "8161")
        self.assertEqual(status, 0)
        self.assertTrue(self.client.connect.call_args.kwargs["compress"])
        self.assertEqual(server.call_args.args[0], ("127.0.0.1", 8163))
        self.assertEqual(server.call_args.kwargs["remote_address"], ("127.0.0.1", 8161))
        self.client.load_system_host_keys.assert_not_called()
        self.client.save_host_keys.assert_not_called()
        self.assertIsInstance(self.client._policy, paramiko.RejectPolicy)
        self.assertEqual(self.client.get_host_keys().lookup(self.endpoint)[self.key.get_name()], self.key)
        self.assertEqual(previous, self.pinned.read_bytes())
        self.assertIn("upload=zlib@openssh.com", out)

    def test_missing_pin_fails_before_connect(self):
        self.pinned.write_text("", encoding="ascii")
        status, server, _ = self.run_cli("--known-hosts", str(self.pinned))
        self.assertEqual(status, 1)
        self.client.connect.assert_not_called()
        server.assert_not_called()
        self.client.save_host_keys.assert_not_called()

    def test_legacy_defaults(self):
        status, server, _ = self.run_cli()
        self.assertEqual(status, 0)
        self.assertFalse(self.client.connect.call_args.kwargs["compress"])
        self.assertEqual(server.call_args.kwargs["remote_address"], ("127.0.0.1", 8080))
        self.client.load_system_host_keys.assert_called_once_with()
        self.client.save_host_keys.assert_called_once_with(str(self.legacy))
        self.assertIsInstance(self.client._policy, paramiko.AutoAddPolicy)


if __name__ == "__main__":
    unittest.main()
