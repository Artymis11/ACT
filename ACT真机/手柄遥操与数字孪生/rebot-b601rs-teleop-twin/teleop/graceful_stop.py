"""GitHub-style smooth zero trajectory; no device IO or fabricated arrival."""
import math

class GracefulStop:
    def __init__(self,start,config,finish=None):
        self.start=tuple(start)
        self.finish=tuple(finish) if finish is not None else (0.,)*7
        if len(self.start)!=7 or not all(math.isfinite(q) for q in self.start):
            raise ValueError('seven finite commanded start angles required')
        if len(self.finish)!=7 or not all(math.isfinite(q) for q in self.finish):raise ValueError('seven finite finish angles required')
        self.duration=max(config['return_min_time_s'],*(1.5*abs(q-f)/min(j['max_speed_rad_s'],config['return_max_speed_rad_s'])
            for q,f,j in zip(self.start,self.finish,config['joints'])))
        self.elapsed=0.
        self.settle_s=config['stable_dwell_s']
        self.settling_since=None

    def step(self,dt):
        if dt<=0:raise ValueError('positive stop interval required')
        self.elapsed=min(self.duration,self.elapsed+min(dt,.02))
        alpha=self.elapsed/self.duration
        if self.elapsed>=self.duration:return self.finish,(0.,)*7
        remaining=1-alpha*alpha*(3-2*alpha)
        slope=6*alpha*(1-alpha)/self.duration
        return tuple(f+(q-f)*remaining for q,f in zip(self.start,self.finish)),tuple((f-q)*slope for q,f in zip(self.start,self.finish))

    def ready_to_disable(self,now,streamed):
        at_target=self.elapsed>=self.duration and all(abs(q-f)<=1e-12 for q,f in zip(streamed,self.finish))
        if not at_target:self.settling_since=None;return False
        if self.settling_since is None:self.settling_since=now
        return now-self.settling_since>=self.settle_s
