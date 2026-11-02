"""Telemetry semantics tests in temporary folders; never touches real runtime."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from server import StateReader, Handler


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.now = 1000.
        self.reader = StateReader(self.root, clock=lambda:self.now)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, sid='one', stamp=999.9, **changes):
        state = {'kind':'native_gamepad_session_loop','synthetic_test_backend':False,
                 'session_id':sid,'closed':False,'phase':'ready',
                 'positions_native_rad':[.1,.2,.3,.4,.5,.6,.7],
                 'transport':{'last_targets_rad':[1.,2.,3.,4.,5.,6.,7.]}}
        state.update(changes)
        path = self.root / ('live_'+sid) / 'status.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(state), encoding='utf-8')
        os.utime(path,(stamp,stamp))
        return path

    def test_empty_does_not_fabricate_pose(self):
        s=self.reader.read();self.assertFalse(s['live']);self.assertIsNone(s['positions'])

    def test_actual_and_control_target_stay_separate(self):
        self.write();s=self.reader.read()
        self.assertTrue(s['live']);self.assertEqual(s['positions'][0],.1)
        self.assertEqual(s['targets'][0],1.)

    def test_closed_fresh_file_is_not_live(self):
        self.write(closed=True);s=self.reader.read();self.assertEqual(s['state'],'ended');self.assertFalse(s['live'])

    def test_stale_data_retains_last_pose_and_marks_stale(self):
        self.write(stamp=990.);s=self.reader.read()
        self.assertFalse(s['live']);self.assertEqual(s['state'],'stale');self.assertEqual(s['positions'][2],.3)

    def test_fresh_fault_is_not_fresh_robot_pose(self):
        self.write(last_error='motor fault');s=self.reader.read()
        self.assertFalse(s['live']);self.assertEqual(s['state'],'fault')

    def test_newest_session_wins_not_old_fault(self):
        self.write('old',stamp=980.,last_error='old fault')
        self.write('new',stamp=999.9)
        self.assertEqual(self.reader.read()['session_id'],'new')

    def test_synthetic_nan_and_short_feedback_are_rejected(self):
        for changes in ({'synthetic_test_backend':True},{'positions_native_rad':[0.]},
                        {'positions_native_rad':[float('nan')]*7}):
            self.write(**changes);self.reader.cached=None
            self.assertFalse(self.reader.read()['live'])
            self.assertIsNone(self.reader.read()['positions'])

    def test_invalid_optional_number_does_not_break_api_json(self):
        self.write(streaming={'measured_send_hz':float('nan')})
        s=self.reader.read();self.assertIsNone(s['send_hz'])
        json.dumps(s,allow_nan=False)

    def test_http_has_no_write_or_workspace_file_route(self):
        class TestHandler(Handler):
            reader=self.reader
            def log_message(self,*args): pass
        httpd=ThreadingHTTPServer(('127.0.0.1',0),TestHandler)
        thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{httpd.server_port}'
            self.assertEqual(json.load(urllib.request.urlopen(url+'/api/state'))['state'],'waiting')
            for request,expected in ((url+'/server.py',404),(urllib.request.Request(url+'/api/state',data=b'{}',method='POST'),501)):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code,expected)
        finally:
            httpd.shutdown();httpd.server_close();thread.join()


if __name__=='__main__':
    unittest.main()
