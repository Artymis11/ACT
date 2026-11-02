"""One CAN owner: interleave 200Hz MIT writes with real parameter replies.

The 50Hz caller sees immutable cached native observations with their original
RX times. Publication sequence is separate from actual query batch sequence.
No reader thread holds a bus lock while waiting for a whole seven-axis batch.
"""
from dataclasses import replace
import threading
import time
import math
from native_pcan_backend import NativePcanBackend, NativeBackendError
from session_core import NativeFeedback
import pcan_readonly as pcan


class PumpedWire(NativePcanBackend):
    pump=None
    def _capture_clock(self):
        if self.pump is not None:
            self.pump()
        return super()._capture_clock()


class StreamMotor:
    def __init__(self,owner,index): self.owner,self.index=owner,index
    def send_mit(self,**values): self.owner.queue(self.index,values)
    def close(self): self.owner.wire.motors[self.index].close()


class HighRateBackend(NativePcanBackend):
    high_rate=True
    def __init__(self,config,*,require_running_handover=False,**kwargs):
        # This Python's Windows monotonic clock is GetTickCount64 (15.625ms).
        # QPC is necessary for 5ms deadlines and distinct 20ms input samples.
        kwargs.setdefault('clock',time.perf_counter)
        self.wire=PumpedWire(config,**kwargs)
        self.require_running_handover=require_running_handover
        self.controller=self
        self.motors=tuple(StreamMotor(self,i) for i in range(7))
        self.lock=threading.RLock()
        self.stop_event=threading.Event()
        self.thread=None
        self.active=False
        self.cached=None
        self.publication_sequence=0
        self.pending={}
        self.desired=None
        self.command=None
        self.scale=.25
        self._pumping=False
        self.next_send=0.
        self.send_batches=0
        self.query_batches=0
        self.worker_error=None
        self.send_times=[]
        self.last_publication=None
        self.receipts=None
        self.poll_index=0
        self.failed_queries={}
        self.recovery_requested=threading.Event()
        self.recovery_done=threading.Event()
        self.recovery_error=None
        self.suspended=False

    def __getattr__(self,name): return getattr(self.wire,name)
    def open(self,**kwargs): return self.wire.open(**kwargs)
    def authorize_motion(self,*args,**kwargs):
        if self.require_running_handover:
            now=self.clock()
            states=[self.wire.capture.statuses.get(i) for i in range(1,8)]
            if any(row is None or not 0<=now-row['local_rx_monotonic']<=self.wire._maximum_age()
                   or row['decoded']['state_bits']!=2 for row in states):
                raise NativeBackendError('software handover requires fresh already-running motors; use X for a normal enable')
        return self.wire.authorize_motion(*args,**kwargs)
    def enable_all(self): return self.wire.enable_all()

    def queue(self,index,values):
        if not self.active:
            return self.wire.motors[index].send_mit(**values)
        if values['vel']!=0 or values['tau']!=0:
            raise NativeBackendError('upstream streamed MIT requires zero velocity and feed-forward torque')
        joint=self.config['joints'][index]
        if (values['kp'],values['kd'])!=(joint['kp'],joint['kd']):
            raise NativeBackendError('upstream fixed gains required')
        q=values['pos']
        if not math.isfinite(q):raise NativeBackendError('finite target required')
        if self.config.get('admission_policy')!='tony-github-defaults-v1' and not joint['native_min_rad']<=q<=joint['native_max_rad']:
            raise NativeBackendError('streamed target exceeds upstream software bounds')
        with self.lock:
            self.pending[index]=q
            if len(self.pending)==7:
                self.desired=tuple(self.pending[i] for i in range(7))
                self.pending.clear()

    def set_speed_scale(self,scale):
        if scale not in (.25,.5,1.):raise ValueError('upstream speed level required')
        with self.lock:self.scale=scale

    def start_streaming(self):
        if not self.wire.enabled or self.active:raise NativeBackendError('one enabled startup required')
        self.cached=self.wire.last_feedback
        self.receipts=[dict(row) for row in self.wire.last_batch]
        self.publication_sequence=self.cached.sequence
        self.desired=tuple(self.wire.last_targets)
        self.command=list(self.desired)
        self.stop_event.clear()
        self.next_send=self.clock()
        self.active=True
        self.wire.pump=self._pump
        self.thread=threading.Thread(target=self._work,name='PCAN-200Hz-owner',daemon=True)
        self.thread.start()

    def _pump(self):
        now=self.clock()
        if not self.active or self._pumping or now<self.next_send:return
        self._pumping=True
        try:
            with self.lock: desired,scale=self.desired,self.scale
            if self.wire.fault or self.suspended:
                candidate=list(self.wire._hold_targets)
            else:
                candidate=[q+max(-j['max_speed_rad_s']*scale*.005,
                    min(j['max_speed_rad_s']*scale*.005,target-q))
                    for q,target,j in zip(self.command,desired,self.config['joints'])]
            for i,(q,joint) in enumerate(zip(candidate,self.config['joints'])):
                self.wire.motors[i].send_mit(pos=q,vel=0.,kp=joint['kp'],kd=joint['kd'],tau=0.)
                self.command[i]=q
            self.send_batches+=1
            self.send_times.append(now)
            self.send_times[:]=self.send_times[-2000:]
            # Skip missed deadlines without sending bursts of accumulated work.
            self.next_send+=.005
            if self.next_send<=self.clock():self.next_send=self.clock()+.005
        finally:self._pumping=False

    def _work(self):
        while not self.stop_event.is_set():
            try:
                self._pump()
                self._poll_one()
                if self.recovery_requested.is_set():
                    self._recover_in_owner()
                self.stop_event.wait(.001)
            except Exception as exc:
                self.suspended=True
                self.worker_error=f'{type(exc).__name__}: {exc}'
                self.wire._latch(self.worker_error)
                # Continue the same owned fixed hold; no automatic release.
                try:self._pump()
                except Exception:pass
                self.stop_event.wait(.002)

    def _poll_one(self):
        # One correlated real reply at a time, instead of a 21-request blocking
        # batch. Untouched channels retain their ORIGINAL receipt timestamps.
        now=self.clock()
        candidates=[(row[pcan.PARAMETERS[param]]['rx']['local_rx_monotonic'],i,param)
            for i,row in enumerate(self.receipts) for param in pcan.CORE_PARAMETERS
            if now>=self.failed_queries.get((i,param),0.)]
        if not candidates:return
        _,axis,parameter=min(candidates)
        answer=self.wire.capture.query(axis+1,parameter,timeout_s=.02)
        if not answer['ok']:
            self.failed_queries[axis,parameter]=self.clock()+.02
            return  # Old receipts expire normally; no synthetic refresh.
        self.receipts[axis][pcan.PARAMETERS[parameter]]=answer
        now=self.clock()
        positions=[];velocities=[];axis_times=[];modes=[];faults=[];status_times=[]
        for i,row in enumerate(self.receipts):
            q,v,m=(row[name] for name in ('position','velocity','mode'))
            qt,vt,mt=(item['rx']['local_rx_monotonic'] for item in (q,v,m))
            status=self.wire.capture.statuses.get(i+1)
            known=status is not None and 0<=now-status['local_rx_monotonic']<=self.wire._maximum_age()
            positions.append(q['value']);velocities.append(v['value']);axis_times.append(min(qt,vt))
            modes.append('MIT' if m['value']==0 else f"native_mode_{m['value']}")
            faults.append(self.wire.fault_codes[i] or (0 if known else None))
            status_times.append(min(mt,status['local_rx_monotonic']) if known else None)
        self.wire.feedback_sequence+=1
        feedback=NativeFeedback(self.wire.session_id,'live_native',now,self.wire.feedback_sequence,
            tuple(positions),tuple(velocities),tuple(axis_times),tuple(faults),tuple(modes),tuple(status_times))
        self.wire.last_feedback=feedback
        with self.lock:self.cached=feedback
        self.query_batches+=1
        self.wire.capture.frames[:]=self.wire.capture.frames[-4096:]
        self.wire.capture.queries[:]=self.wire.capture.queries[-128:]

    def _recover_in_owner(self):
        try:
            expected='feedback, fault or mode receipt is unknown/stale'
            if (not self.worker_error or expected not in self.worker_error
                    or self.wire.fault is None or expected not in self.wire.fault
                    or any(self.wire.fault_codes) or self.wire.thermal_return_requested):
                raise NativeBackendError('only expired-feedback faults may be resumed by Start')
            self.wire._feedback_check(self.wire.last_feedback,self.clock())
            if not self.wire.enabled:
                raise NativeBackendError('motor ownership is not active')
            if any(self.wire.capture.statuses.get(i,{}).get('decoded',{}).get('state_bits')!=2 for i in range(1,8)):
                raise NativeBackendError('fresh running motor states required for resume')
            with self.lock:
                self.pending.clear()
                self.desired=tuple(self.command)
            self.wire.fault=None
            self.wire.phase='owned'
            self.worker_error=None
            self.recovery_error=None
        except Exception as exc:
            self.recovery_error=str(exc)
        finally:
            self.recovery_requested.clear()
            self.recovery_done.set()

    def recover_feedback_fault(self,session_id):
        if session_id!=self.session_id:raise NativeBackendError('wrong recovery session')
        self.recovery_done.clear()
        self.recovery_requested.set()
        if not self.recovery_done.wait(.3):raise NativeBackendError('fresh recovery still pending; retry Start after neutral')
        if self.recovery_error:raise NativeBackendError(self.recovery_error)

    def resume_stream(self,session_id):
        if session_id!=self.session_id or self.worker_error or self.wire.fault:
            raise NativeBackendError('feedback recovery not complete')
        with self.lock:
            self.pending.clear()
            self.desired=tuple(self.command)
            self.suspended=False

    def read_feedback(self,session_id,now=None):
        if not self.active:return self.wire.read_feedback(session_id,now)
        if self.worker_error:raise NativeBackendError(self.worker_error)
        if session_id!=self.wire.session_id:raise NativeBackendError('wrong stream session')
        with self.lock:
            sample=self.cached
            self.publication_sequence+=1
            # This is a new publication, never a fabricated CAN receipt.
            self.last_publication=replace(sample,sequence=self.publication_sequence)
            return self.last_publication

    def _stop_worker(self):
        if self.thread:
            self.stop_event.set()
            self.thread.join(1.)
            if self.thread.is_alive():raise NativeBackendError('retain owner: CAN worker has not stopped')
        self.active=False
        self.wire.pump=None

    def authorize_supported_disable(self,session_id,feedback,now=None,*,supported=False):
        if feedback is not self.last_publication or session_id!=self.session_id or supported is not True:
            raise NativeBackendError('current published feedback and explicit support required')
        self._stop_worker()
        fresh=self.wire.read_feedback(session_id)
        self.wire.authorize_supported_disable(session_id,fresh,self.clock(),supported=True)

    def disable_all(self): return self.wire.disable_all()
    def authorize_github_zero_disable(self,session_id,feedback):
        return self.authorize_github_finish_disable(session_id,feedback,(0.,)*7)
    def authorize_github_finish_disable(self,session_id,feedback,finish):
        if feedback is not self.last_publication or session_id!=self.session_id:
            raise NativeBackendError('latest published observation required for shutdown')
        if any(abs(q-f)>1e-12 for q,f in zip(self.command,finish)):
            raise NativeBackendError('stream must actually send finish target before disable')
        self._stop_worker()
        fresh=self.wire.read_feedback(session_id)
        self.wire.authorize_github_finish_disable(session_id,fresh,finish)
        self.publication_sequence+=1
        self.last_publication=replace(fresh,sequence=self.publication_sequence)
        return self.last_publication
    def github_thermal_emergency_disconnect(self):
        self._stop_worker()
        return self.wire.github_thermal_emergency_disconnect()
    def acknowledge_external_power_off(self,*args,**kwargs):
        self._stop_worker()
        return self.wire.acknowledge_external_power_off(*args,**kwargs)
    def close_bus(self):
        self._stop_worker()
        return self.wire.close_bus()
    def close(self):return self.wire.close()
    def diagnostics(self):
        result=self.wire.diagnostics()
        result['streaming']=self.timing()
        return result
    def timing(self):
        times=self.send_times[:]
        return {'target_hz':200,'send_batches':self.send_batches,'parameter_updates':self.query_batches,
            'fault_sources':self.wire.fault_sources,
            'measured_send_hz':(len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None,
            'actual_query_sequence':self.cached.sequence if self.cached else None,
            'publication_sequence':self.publication_sequence,'worker_error':self.worker_error}
