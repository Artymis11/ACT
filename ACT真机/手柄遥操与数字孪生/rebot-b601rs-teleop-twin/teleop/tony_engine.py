"""Pinned Tony input profile behind local freshness and operator interlocks."""
import math
from dataclasses import replace
from tony_input import TonyInputMapper
from live_runner import Engine, AXES, BUTTONS

DPAD=15
ACTION_MASK=BUTTONS['Start']|BUTTONS['Y']|BUTTONS['RB']

class TonyEngine:
    def __init__(self,site_config=None,mapper=None,initial_speed_level=0):
        self.mapper=mapper or TonyInputMapper()
        self.guard=Engine()
        self.config=site_config or {}
        if initial_speed_level not in (0,1,2):raise ValueError('invalid initial speed level')
        self.level=initial_speed_level
        self.pending=None
        self.previous=0
        self.connected=False
        self.last_time=None
        self.intent=None
        self.full_neutral=False
        self.initial_resume_sent=False
        self.a_hold_started=None
        self.a_hold_sent=False

    @property
    def paused(self): return self.guard.paused

    def set_paused(self,value):
        self.guard.set_paused(value)
        if value: self.pending=None

    def consume_action(self):
        result,self.pending=self.pending,None
        return result

    def is_neutral(self,snapshot=None):
        return (snapshot or self.guard.snapshot(self.last_time))['control'].get('neutral_confirmed',self.full_neutral)

    def update(self,frame,now):
        self.pending=None
        valid=False
        try:
            intent=self.mapper.map(frame)
            valid=bool(intent.connected and 0<=now-frame.acquired_at<=.25)
        except (ValueError,TypeError,AttributeError):
            intent=None
        buttons=frame.buttons if type(frame.buttons) is int and 0<=frame.buttons<=65535 else 0
        custom=self.config.get('admission_policy')=='tony-github-defaults-v1'
        blocked=DPAD|BUTTONS['LB']|BUTTONS['B']
        if not custom:blocked|=BUTTONS['A']|BUTTONS['X']
        neutral=valid and all(abs(frame.axes[a])<=.12 for a in AXES) and not buttons&blocked
        gap=self.last_time is not None and not 0<now-self.last_time<=.25
        discontinuity=not self.connected or gap or not valid
        action_mask=(BUTTONS['X']|BUTTONS['Y']|BUTTONS['RB']) if custom else ACTION_MASK
        edges=0 if discontinuity else buttons&~self.previous&action_mask
        a_held=custom and bool(buttons&BUTTONS['A'])
        if not a_held:
            self.a_hold_started=None
            self.a_hold_sent=False
        elif not valid or discontinuity or buttons&(BUTTONS['X']|BUTTONS['Y']|BUTTONS['RB']):
            self.a_hold_started=None
        elif not self.previous&BUTTONS['A']:
            self.a_hold_started=now
        fresh_b=bool(frame.connected and 0<=now-frame.acquired_at<=.25 and buttons&BUTTONS['B'])
        if fresh_b:
            self.a_hold_started=None
            self.set_paused(True)
            if self.config.get('admission_policy')=='tony-github-defaults-v1':
                self.pending='shutdown'
        elif edges.bit_count()>1 or (a_held and edges):
            self.set_paused(True)
        elif a_held:
            if self.a_hold_started is not None and now-self.a_hold_started>=3. and not self.a_hold_sent:
                self.set_paused(True)
                self.pending='shutdown_origin'
                self.a_hold_sent=True
        elif neutral:
            if edges&(BUTTONS['X'] if custom else BUTTONS['Start']):
                self.guard.set_paused(False)
                self.pending='resume'
            elif edges&BUTTONS['Y'] and not self.paused:
                self.pending='return_origin'
            elif edges&BUTTONS['RB'] and not self.paused:
                self.level=(self.level+1)%3
            elif self.config.get('admission_policy')=='tony-github-defaults-v1' and not self.initial_resume_sent and not self.paused:
                self.pending='resume'
                self.initial_resume_sent=True
        must_rearm=self.guard.needs_rearm or discontinuity
        hidden=DPAD|(BUTTONS['A']|BUTTONS['X'] if custom else 0)
        self.guard.update(replace(frame,buttons=buttons&~hidden),now)
        if must_rearm and not neutral:
            self.guard.needs_rearm=True
        self.previous=buttons
        self.connected=bool(valid)
        self.last_time=now
        self.full_neutral=bool(neutral)
        self.intent=intent if valid else None
        return self.snapshot(now)

    def snapshot(self,now):
        state=self.guard.snapshot(now)
        state.pop('fake_backend',None)
        ctrl=state['control']
        if ctrl['gate'] in ('input_stale','disconnected'):
            self.pending=None
        levels=self.mapper.upstream_constants['SPEED_LEVELS']
        scale=levels[self.level]
        velocities=[0.]*7
        holding_stop=self.config.get('admission_policy')=='tony-github-defaults-v1' and self.previous&BUTTONS['A']
        if self.intent and ctrl['gate'] in ('ready','active') and ctrl['lb_held'] and not self.paused and not holding_stop:
            for i,(jog,speed) in enumerate(zip(self.intent.jog,self.mapper.upstream_constants['JOG_SPEEDS_DEG_S'])):
                value=jog*math.radians(speed)*scale
                joints=self.config.get('joints',[])
                limit=joints[i].get('max_speed_rad_s') if len(joints)>i else None
                if type(limit) in (int,float) and math.isfinite(limit) and limit>0:
                    value=max(-limit,min(limit,value))
                velocities[i]=value
        ctrl.update(joint_velocity_rad_s=velocities[:6],gripper_velocity_rad_s=velocities[6],
            gripper_intent='open' if velocities[6]>0 else 'close' if velocities[6]<0 else 'hold',
            neutral_confirmed=self.full_neutral and ctrl['gate'] not in ('input_stale','disconnected'),
            speed_level=self.level,speed_scale=scale,operator_action=self.pending)
        ctrl['a_hold_seconds']=0. if self.a_hold_started is None else max(0.,now-self.a_hold_started)
        ctrl['upstream_user_keys']=self.config.get('admission_policy')=='tony-github-defaults-v1'
        # Preserve the raw D-pad names for diagnostics and full-neutral checks.
        state['input']['buttons'] += [name for name in ('DPadUp','DPadDown','DPadLeft','DPadRight') if self.previous&BUTTONS[name]]
        if self.config.get('admission_policy')=='tony-github-defaults-v1':
            state['input']['buttons'] += [name for name in ('A','X') if self.previous&BUTTONS[name]]
        state['upstream']=self.mapper.provenance
        return state
