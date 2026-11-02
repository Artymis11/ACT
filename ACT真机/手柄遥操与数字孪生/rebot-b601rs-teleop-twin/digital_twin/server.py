"""Loopback, GET-only mirror of existing telemetry. Never imports robot code."""
import argparse
import json
import math
import mimetypes
from pathlib import Path
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'teleop' / 'runtime'
APP_ID = 'b601-feedback-twin-public-v1'


def vector(value):
    return (isinstance(value, (list, tuple)) and len(value) == 7
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


class StateReader:
    def __init__(self, runtime=RUNTIME, clock=time.time):
        self.runtime, self.clock = Path(runtime), clock
        self.lock = threading.Lock()
        self.cached_at = 0.
        self.cached = None

    def read(self):
        with self.lock:
            now = self.clock()
            if self.cached is not None and 0 <= now-self.cached_at < .04:
                return self.cached
            result = self._read(now)
            self.cached_at, self.cached = now, result
            return result

    def _read(self, now):
        empty = {'app_id': APP_ID, 'server_time': now, 'state': 'waiting',
                 'message': '等待遥操反馈', 'positions': None, 'targets': None,
                 'session_id': None, 'age_ms': None, 'live': False}
        try:
            candidates = []
            for p in self.runtime.glob('live_*/status.json'):
                try:
                    candidates.append((p.stat().st_mtime_ns, p))
                except OSError:
                    continue
            if not candidates:
                return empty
            stamp_ns, path = max(candidates)
            # Atomic producer writes. Read without a persistent Windows file lock.
            raw = path.read_bytes()
            if len(raw) > 2_000_000:
                raise ValueError('status too large')
            s = json.loads(raw)
            if s.get('kind') != 'native_gamepad_session_loop' or s.get('synthetic_test_backend') is not False:
                raise ValueError('not real native feedback')
            positions = s.get('positions_native_rad')
            if not vector(positions):
                raise ValueError('seven finite actual positions required')
            age = max(0., now-stamp_ns/1e9)
            closed = s.get('closed') is True
            phase = s.get('phase', 'unknown')
            error = s.get('last_error') or s.get('transport', {}).get('fault')
            stale = age > .75 or stamp_ns/1e9 > now+1
            state = 'ended' if closed else 'fault' if error else 'stale' if stale else 'live'
            messages = {'ended': '会话已结束 · 最后姿态', 'fault': '控制器故障 · 最后姿态',
                        'stale': '数据已过期 · 最后姿态', 'live': '实测反馈同步中'}
            targets = s.get('transport', {}).get('last_targets_rad')
            if not vector(targets):
                targets = None
            inp = s.get('input') or {}
            control = inp.get('control') or {}
            stream = s.get('streaming') or {}
            return {'app_id': APP_ID, 'server_time': now, 'state': state, 'message': messages[state],
                    'live': state == 'live', 'age_ms': round(age*1000),
                    'stamp': stamp_ns, 'session_id': s.get('session_id'), 'phase': phase,
                    'positions': positions, 'targets': targets,
                    'velocities': s.get('velocities_native_rad_s') if vector(s.get('velocities_native_rad_s')) else None,
                    'origin': s.get('origin_rad') if vector(s.get('origin_rad')) else None,
                    'gamepad_connected': (inp.get('input') or {}).get('connected'),
                    'lb_held': control.get('lb_held') is True,
                    'speed_scale': number(control.get('speed_scale')), 'send_hz': number(stream.get('measured_send_hz')),
                    'temperature_c': number(s.get('hottest_temperature_c')),
                    'soft_limit_blocks': s.get('soft_limit_blocks'), 'error': error,
                    'shutdown_reason': s.get('shutdown_reason'),
                    'source': str(path.relative_to(self.runtime.parent)).replace('\\', '/')}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            empty.update(state='unavailable', message='暂时无法读取反馈，正在重试', error=str(exc))
            return empty


class Handler(SimpleHTTPRequestHandler):
    reader = StateReader()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        super().end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/state':
            data = json.dumps(self.reader.read(), ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        # No general workspace file server, directory listing, or write API.
        if path == '/':
            self.path = '/index.html'
        elif path not in ('/index.html', '/app.js', '/pose_state.js', '/styles.css', '/model_mapping.json') and not path.startswith('/vendor/description/') and not path.startswith('/vendor/public/lib/'):
            self.send_error(404)
            return
        if '..' in path or '%2e' in path.lower():
            self.send_error(404)
            return
        super().do_GET()

    def list_directory(self, path):
        self.send_error(404)

    def log_message(self, format, *args):
        if args and str(args[1] if len(args) > 1 else '') not in ('200', '304'):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    mimetypes.add_type('text/javascript', '.js')
    httpd = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    httpd.daemon_threads = True
    print(json.dumps({'app_id': APP_ID, 'url': f'http://127.0.0.1:{httpd.server_port}',
                      'mode': 'read_only_feedback', 'can_opened': False}), flush=True)
    try:
        httpd.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main()
