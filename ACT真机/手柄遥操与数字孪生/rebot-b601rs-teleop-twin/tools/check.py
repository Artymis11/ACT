"""Offline-only regression suite; fake transports, no hardware controller."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TESTS=['test_directional_soft_limits','test_soft_limit_semantics','test_user_keys',
       'test_tony_github_config','test_high_rate_backend','test_live_runner',
       'test_tony_engine','test_tony_integration','test_tony_input','test_native_pcan_backend',
       'test_session_core','test_motion_transport','test_pcan_readonly','test_status_reporting']


def main():
    started=time.time()
    for relative in ('teleop/reports','teleop/runtime','digital_twin/logs'):(ROOT/relative).mkdir(parents=True,exist_ok=True)
    commands=[([sys.executable,'-B','-X','utf8','-m','unittest',*TESTS],ROOT/'teleop'),
              ([sys.executable,'-B','-X','utf8','-m','unittest','test_server.py'],ROOT/'digital_twin')]
    node=shutil.which('node')
    if node:commands.extend([([node,'test_pose_state.js'],ROOT/'digital_twin'),([node,'--check','app.js'],ROOT/'digital_twin')])
    results=[]
    for cmd,cwd in commands:
        run=subprocess.run(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        print(run.stdout)
        results.append({'command':cmd[1:],'directory':cwd.relative_to(ROOT).as_posix(),'exit_code':run.returncode,'output':run.stdout})
    # Reports intentionally contain only paths relative to this repository.
    report={'can_opened':False,'physical_motion_test':False,'elapsed_s':round(time.time()-started,2),
            'node_checks':'run' if node else 'skipped: Node.js not installed','passed':all(r['exit_code']==0 for r in results),'results':results}
    out=ROOT/'teleop/reports/offline_checks.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
