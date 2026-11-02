"""Windows integration of Tony's pinned input helpers. Default opens nothing."""
import argparse
import json
import math
from pathlib import Path
import time
import live_runner
from live_runner import Frame, AXES, BUTTONS, NativeGamepadRunner, save_atomic
from tony_engine import TonyEngine
from tony_input import TonyInputMapper

def wait_for_x(reader,mapper):
    from enable_latch import EnableLatch
    import ctypes
    latch=EnableLatch(mapper)
    print(json.dumps({'mode':'disabled_waiting_for_X','can_opened':False,
        'message':'Release LB and center sticks, then press X to enable a fresh session.'}),flush=True)
    while True:
        try:
            if hasattr(ctypes,'windll') and ctypes.windll.user32.GetAsyncKeyState(0x1B)&0x8000:return False
            frame=reader.read()
            if latch.update(frame,time.monotonic()):return True
            time.sleep(.02)
        except (KeyboardInterrupt,EOFError):return False

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--observe',action='store_true')
    modes.add_argument('--control',action='store_true')
    parser.add_argument('--scripted-fixture',action='store_true')
    parser.add_argument('--allow-enable',action='store_true')
    parser.add_argument('--take-over-running',action='store_true',help='explicit software handover of already-enabled motors; first session only')
    parser.add_argument('--slot',type=int,choices=range(4),default=0)
    parser.add_argument('--initial-speed-level',type=int,choices=range(3),default=0)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if args.duration is None:args.duration=0. if args.control else 10.
    if not math.isfinite(args.duration) or not (0<args.duration<=3600 or (args.control and args.duration==0)):
        parser.error('duration must be in (0,3600]; control duration=0 means no fixed session timeout')
    if args.scripted_fixture and not args.dry_run:
        parser.error('--scripted-fixture requires --dry-run')
    if args.allow_enable and not args.control:
        parser.error('--allow-enable requires --control')
    if args.take_over_running and not (args.control and args.allow_enable):
        parser.error('--take-over-running requires --control --allow-enable')
    # Verify upstream bytes before creating any device reader or controller.
    mapper=TonyInputMapper()
    if not (args.dry_run or args.observe or args.control):
        print(json.dumps({'mode':'description_only','can_opened':False,'upstream':mapper.provenance},ensure_ascii=False))
        return 0
    if args.observe or args.control:
        config_path=args.config
        if config_path is None:
            from tony_github_config import github_config
            config_path=live_runner.ROOT/'site_config.tony_github.json'
            save_atomic(config_path,github_config())
        forwarded=['--control' if args.control else '--observe','--xinput','--slot',str(args.slot),
                   '--duration',str(args.duration),'--config',str(config_path)]
        if args.allow_enable: forwarded+=['--allow-enable']
        if args.out: forwarded+=['--out',str(args.out)]
        from high_rate_backend import HighRateBackend
        factory=lambda config:TonyEngine(config,mapper,args.initial_speed_level)
        # Validate before waiting for X so invalid configurations never appear
        # ready. Waiting and re-arming hold no CAN handle.
        config=json.loads(config_path.read_text(encoding='utf-8'))
        from session_core import config_errors
        if not args.control or not args.allow_enable or config_errors(config):
            return live_runner.main(forwarded,input_engine_factory=factory,
                backend_factory=HighRateBackend if args.control else None)
        reader=live_runner.load_xinput_reader(args.slot)
        cycle=0
        take_over=args.take_over_running
        while take_over or wait_for_x(reader,mapper):
            handover_this=take_over
            take_over=False
            cycle+=1
            completed=[]
            current=forwarded[:]
            if args.out and cycle>1:
                current[current.index('--out')+1]=str(args.out.with_name(args.out.stem+f'_session_{cycle}'+args.out.suffix))
            code=live_runner.main(current,input_engine_factory=factory,
                backend_factory=lambda config:HighRateBackend(config,require_running_handover=handover_this),
                result_callback=completed.append)
            if code or not completed:return code
            final=completed[-1].get('final') or {}
            if not final.get('closed') or final.get('exit_mode') not in ('github_smooth_zero_disable','github_origin_disable'):
                return code
        return 0
    # Explicit fake plant only. No PCAN backend is constructed or opened.
    from fake_robot import FakeRobot, synthetic_config
    config=synthetic_config()
    robot=FakeRobot()
    now=[10.]
    runner=NativeGamepadRunner(now[0],robot,config,allow_motion=True,clock=lambda:now[0],
        allow_synthetic_backend=True,input_engine=TonyEngine(config,mapper))
    reader=None if args.scripted_fixture else live_runner.load_xinput_reader(args.slot)
    trace=[]
    fatal=None
    runner.start()
    try:
        for index in range(max(1,int(args.duration/.02))):
            now[0]+=.02
            if args.scripted_fixture:
                buttons=BUTTONS['Start'] if index==1 else BUTTONS['B'] if index==25 else 0
                axes=dict.fromkeys(AXES,0.)
                if 3<=index<20:
                    buttons=BUTTONS['LB']|BUTTONS['DPadUp']|BUTTONS['DPadLeft']
                    axes.update(LX=.6,LY=.6,RX=.6,RY=.6,RT=.4)
                frame=Frame(True,buttons=buttons,axes=axes,acquired_at=now[0])
            else:
                time.sleep(.02)
                frame=reader.read()
                # Simulation clock is separate from input acquisition clock.
                frame.acquired_at=now[0]-(time.monotonic()-frame.acquired_at)
            state=runner.step(frame,now[0])
            trace.append(state)
            trace[:]=trace[-300:]
            if state['last_error']:
                fatal=state['last_error'];break
    except (Exception,KeyboardInterrupt) as exc:
        fatal=f'{type(exc).__name__}: {exc}'
    finally:
        # This handover is a fixture fact about FakeRobot, never a human or
        # physical power-off claim. Real sessions use the retained live loop.
        now[0]+=.02
        runner.step(Frame(False,acquired_at=now[0]),now[0],action='confirm_external_power_off',
            session_id=runner.session_id,supported=True,motor_power_off_confirmed=True)
    report={'kind':'tony_fake_plant_dry_run','synthetic_only':True,'can_opened':False,
        'scripted_fixture':args.scripted_fixture,'physical_motion_acceptance_verified':False,
        'upstream':mapper.upstream_metadata,'fatal_error':fatal,'trace':trace,'final':runner.snapshot()}
    if args.out: save_atomic(args.out,report)
    print(json.dumps({'mode':'dry_run','can_opened':False,'ticks':len(trace),'fatal_error':fatal,
        'out':str(args.out) if args.out else None},ensure_ascii=False))
    return 2 if fatal else 0

if __name__=='__main__': raise SystemExit(main())
