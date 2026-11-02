"""GitHub values, without fabricated physical-acceptance flags."""
import ast
import copy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
from tony_input import ROOT, UPSTREAM_COMMIT, verify_upstream_snapshot

POLICY='tony-github-defaults-v1'

@lru_cache(maxsize=1)
def _github_config_cached():
    verify_upstream_snapshot()
    base=ROOT/'vendor'/'tony_xbox_teleop'
    tree=ast.parse((base/'xbox_teleop.py').read_text(encoding='utf-8'))
    names={'JOG_SPEEDS_DEG_S','URDF_LIMITS_DEG','LIMIT_SAFE_SCALE','GRIPPER_MIN_DEG','GRIPPER_MAX_DEG','LOOP_HZ'}
    constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names}
    yaml=(base/'config'/'rebotarm_rs.yaml').read_text(encoding='utf-8')
    motors=re.findall(r'- motor_id:\s*(\d+)\s+model:\s*(rs-\d+)\s+kp:\s*([\d.]+)\s+kd:\s*([\d.]+)',yaml)
    if len(motors)!=7:raise ValueError('pinned upstream seven-motor YAML could not be read')
    bounds=[(lo*constants['LIMIT_SAFE_SCALE'],hi*constants['LIMIT_SAFE_SCALE']) for lo,hi in constants['URDF_LIMITS_DEG']]
    # User confirms native zero is the one-sided travel origin on J2/J3.
    # Keep 70% of their full 180-degree span without recentering native zero.
    for index in (1,2):
        lo,hi=constants['URDF_LIMITS_DEG'][index]
        bounds[index]=(0.,(hi-lo)*constants['LIMIT_SAFE_SCALE'])
    bounds.append((constants['GRIPPER_MIN_DEG'],constants['GRIPPER_MAX_DEG']))
    config={'admission_policy':POLICY,'status':'GITHUB_DEFAULTS_WITH_USER_J2_J3_NATIVE_ZERO','native_coordinate':'motor_native_rad',
        'coordinate_override_source':'user confirms J2/J3 native zero is one-sided travel start; software span 0..126 degrees; not measured mechanical limits',
        'source_commit':UPSTREAM_COMMIT,'source_files':{name:hashlib.sha256((base/name).read_bytes()).hexdigest()
            for name in ('xbox_teleop.py','config/rebotarm_rs.yaml')},
        'feedback_max_age_s':.25,'max_control_interval_s':.1,
        'input_period_s':1./constants['LOOP_HZ'],
        'tracking_error_rad':None,'position_tolerance_rad':None,'velocity_tolerance_rad_s':None,
        'stable_dwell_s':float(re.search(r'settle_time_s:\s*([\d.]+)',yaml)[1]),
        'return_min_time_s':float(re.search(r'min_time_s:\s*([\d.]+)',yaml)[1]),
        'return_max_speed_rad_s':math.radians(float(re.search(r'(?m)^  max_speed_deg_s:\s*([\d.]+)',yaml)[1])),
        'return_acceptance':'operator_confirms_displayed_actual_origin_and_support',
        'temperatures':{name:float(re.search(r'(?m)^  '+name+r':\s*([\d.]+)',yaml)[1])
            for name in ('alarm_c','return_zero_c','disconnect_c')},
        'process_exit_ownership_policy':'retain_until_supported_disable','joints':[]}
    for index,((axis,model,kp,kd),(lo,hi),speed) in enumerate(zip(motors,bounds,constants['JOG_SPEEDS_DEG_S'])):
        if int(axis)!=index+1:raise ValueError('upstream axis order differs')
        config['joints'].append({'name':f'J{axis}','motor_id':int(axis),'host_id':253,'model':model,
            'kp':float(kp),'kd':float(kd),'native_min_rad':math.radians(lo),'native_max_rad':math.radians(hi),
            'max_speed_rad_s':math.radians(speed),'max_accel_rad_s2':None,
            'bounds_source':('user native-zero correction: one-sided 0..126 degrees, 70% of upstream 180-degree span'
                if index in (1,2) else 'upstream software limits, not measured mechanical limits')})
    return config

def github_config():
    # Verify and bind source once per process, then compare immutable values in
    # the control loop without repeatedly reading the source tree during CAN IO.
    return copy.deepcopy(_github_config_cached())

def is_github(config):return config.get('admission_policy')==POLICY

def github_errors(config):
    try:
        return [] if config==github_config() else ['GitHub profile differs from pinned upstream defaults; no physical acceptance is inferred']
    except (OSError,ValueError,KeyError) as exc:return [f'GitHub profile source error: {exc}']
