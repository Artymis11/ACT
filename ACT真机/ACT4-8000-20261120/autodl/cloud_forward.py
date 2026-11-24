"""Forward a local loopback port to the cloud policy server over SSH."""

import argparse
import os
from pathlib import Path
import re
import socket
import socketserver
import sys
import threading
import time

import paramiko


REMOTE_ADDRESS = ("127.0.0.1", 8080)
KNOWN_HOSTS = Path(__file__).resolve().parent / "runs" / "act2_20261108" / "ssh_known_hosts"


def close_pair(pair):
    for endpoint in pair:
        try:
            endpoint.shutdown(socket.SHUT_RDWR)
        except (OSError, EOFError, paramiko.SSHException):
            pass
        try:
            endpoint.close()
        except (OSError, EOFError, paramiko.SSHException):
            pass


def forward_pair(local_socket, channel, stopping):
    pair = (local_socket, channel)
    failed = threading.Event()
    for endpoint in pair:
        endpoint.settimeout(5)

    def relay(source, destination):
        try:
            while not stopping.is_set() and not failed.is_set():
                try:
                    payload = source.recv(65536)
                except socket.timeout:
                    continue
                if not payload:
                    destination.shutdown(socket.SHUT_WR)
                    return
                destination.sendall(payload)
        except (OSError, EOFError, paramiko.SSHException):
            failed.set()
            close_pair(pair)

    upstream = threading.Thread(target=relay, args=pair, daemon=True)
    upstream.start()
    try:
        relay(channel, local_socket)
    finally:
        failed.set()
        close_pair(pair)
        upstream.join(timeout=6)


class ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self):
        server = self.server
        try:
            channel = server.transport.open_channel(
                "direct-tcpip", server.remote_address, self.client_address, timeout=15,
            )
        except (paramiko.SSHException, OSError, EOFError):
            print("A forwarding connection could not be opened.", file=sys.stderr, flush=True)
            return
        if channel is None:
            return
        pair = (self.request, channel)
        with server.connection_lock:
            if server.stopping.is_set():
                close_pair(pair)
                return
            server.connections.add(pair)
        try:
            forward_pair(self.request, channel, server.stopping)
        finally:
            with server.connection_lock:
                server.connections.discard(pair)


class ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False

    def __init__(self, address, transport, stopping, *, remote_address=None):
        self.transport = transport
        self.stopping = stopping
        self.remote_address = REMOTE_ADDRESS if remote_address is None else remote_address
        self.connections = set()
        self.connection_lock = threading.Lock()
        super().__init__(address, ForwardHandler)

    def close_connections(self):
        with self.connection_lock:
            pairs = list(self.connections)
        for pair in pairs:
            close_pair(pair)

    def handle_error(self, request, client_address):
        print("A forwarding connection ended with an error.", file=sys.stderr, flush=True)


def monitor_connection(transport, stopping, heartbeat_seconds=20, response_seconds=30):
    pending = threading.Event()
    disconnected = threading.Event()
    probe_started = [0.0]

    def heartbeat():
        while not stopping.wait(heartbeat_seconds):
            probe_started[0] = time.monotonic()
            pending.set()
            try:
                # A success or failure reply confirms that the SSH peer responded.
                transport.global_request("keepalive@openssh.com", wait=True)
            except (paramiko.SSHException, OSError, EOFError):
                disconnected.set()
                return
            finally:
                pending.clear()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        while not stopping.wait(0.5):
            if not transport.is_active() or disconnected.is_set():
                raise ConnectionError("SSH transport disconnected.")
            if pending.is_set() and time.monotonic() - probe_started[0] > response_seconds:
                raise ConnectionError("SSH peer did not answer the keepalive request.")
    finally:
        stopping.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--local-port", type=int, default=8080)
    parser.add_argument("--remote-port", type=int, default=REMOTE_ADDRESS[1],
                        help="Destination loopback port on the SSH host (default: 8080).")
    parser.add_argument("--compress", action="store_true",
                        help="Negotiate lossless SSH transport compression.")
    parser.add_argument("--known-hosts", type=Path,
                        help="Trust only keys in this existing OpenSSH known_hosts file; never update it.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9.-]+", args.host):
        parser.error("Invalid SSH host name.")
    if not all(1 <= port <= 65535 for port in (args.port, args.local_port, args.remote_port)):
        parser.error("Ports must be between 1 and 65535.")
    password = os.environ.pop("ACT_SSH_PASSWORD", None)
    if not password:
        parser.error("ACT_SSH_PASSWORD is required in the process environment.")

    stopping = threading.Event()
    client = paramiko.SSHClient()
    server = None
    serving = None
    exit_code = 1
    try:
        if args.known_hosts is None:
            client.load_system_host_keys()
            if KNOWN_HOSTS.is_file():
                client.load_host_keys(str(KNOWN_HOSTS))
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        else:
            client.load_host_keys(str(args.known_hosts))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            key_name = args.host if args.port == 22 else f"[{args.host}]:{args.port}"
            if client.get_host_keys().lookup(key_name) is None:
                raise ValueError("No pinned host key for the requested SSH endpoint.")
        try:
            client.connect(
                args.host, port=args.port, username="root", password=password,
                allow_agent=False, look_for_keys=False, compress=args.compress,
                timeout=20, banner_timeout=20, auth_timeout=20,
            )
        finally:
            password = None
        if args.known_hosts is None:
            KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
            client.save_host_keys(str(KNOWN_HOSTS))
        transport = client.get_transport()
        if transport is None or not transport.is_authenticated():
            raise ConnectionError("SSH authentication did not complete.")
        remote_address = ("127.0.0.1", args.remote_port)
        server = ForwardServer(("127.0.0.1", args.local_port), transport, stopping,
                               remote_address=remote_address)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        print(f"Forwarding 127.0.0.1:{args.local_port} to cloud 127.0.0.1:{args.remote_port}.", flush=True)
        if args.compress:
            print(f"SSH compression: upload={transport.local_compression}, "
                  f"download={transport.remote_compression}.", flush=True)
        monitor_connection(transport, stopping)
        exit_code = 0
    except KeyboardInterrupt:
        print("Forwarding stopped.", flush=True)
        exit_code = 0
    except Exception as error:
        # Exception text and tracebacks are omitted from authentication output.
        print(f"SSH forwarding failed ({type(error).__name__}).", file=sys.stderr, flush=True)
    finally:
        password = None
        stopping.set()
        if server is not None:
            server.close_connections()
            if serving is not None and serving.is_alive():
                server.shutdown()
            server.server_close()
        client.close()
        if serving is not None:
            serving.join(timeout=5)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
