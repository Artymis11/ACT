"""X-enable edge detection while CAN is closed; no device SDK imports."""
from tony_input import TonyInputMapper, BUTTON_NAMES, BUTTON_MASKS

MASK=dict(zip(BUTTON_NAMES,BUTTON_MASKS))

class EnableLatch:
    def __init__(self,mapper=None):
        self.mapper=mapper or TonyInputMapper()
        self.connected=False
        self.previous=0

    def update(self,frame,now):
        buttons=frame.buttons if type(frame.buttons) is int else 0
        try:
            valid=self.mapper.map(frame).connected and 0<=now-frame.acquired_at<=.25
        except (ValueError,TypeError,AttributeError):valid=False
        edge=valid and self.connected and bool(buttons&MASK['X']) and not self.previous&MASK['X']
        self.connected=bool(valid)
        self.previous=buttons
        if not edge:return False
        return (all(abs(value)<=.12 for value in frame.axes.values())
            and not buttons&sum(MASK[name] for name in ('A','B','Y','LB','RB','DPadUp','DPadDown','DPadLeft','DPadRight')))
