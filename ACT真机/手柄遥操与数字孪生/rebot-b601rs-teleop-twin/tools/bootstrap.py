"""Fetch version-pinned runtime resources and optionally register a local DLL.

No device libraries are imported or loaded; no CAN handle is created.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def resource_plan(teleop=True,twin=True):
    rows=[]
    if teleop:
        manifest=json.loads((ROOT/'teleop/tony_source_manifest.json').read_text(encoding='utf-8'))
        for row in manifest['files']:
            suffix=row['path'].removeprefix('vendor/tony_xbox_teleop/')
            rows.append((ROOT/'teleop'/row['path'],row['sha256'],
                f"https://raw.githubusercontent.com/TonyWeyyy/reBot_B601_rs_xbox_teleop/{manifest['commit']}/{suffix}"))
    if twin:
        manifest=json.loads((ROOT/'digital_twin/vendor_manifest.json').read_text(encoding='utf-8'))
        rows.extend((ROOT/'digital_twin'/r['file'],r['sha256'],r['url']) for r in manifest['files'])
    return rows


def valid(path,digest):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest


def get_resource(row,verify_only=False):
    path,digest,url=row
    if valid(path,digest):return str(path.relative_to(ROOT))
    if verify_only:raise RuntimeError(f'Missing or changed resource: {path.relative_to(ROOT)}; run bootstrap first.')
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'B601RS-Teleop-Twin-Setup'})
            with urllib.request.urlopen(request,timeout=60) as response:data=response.read()
            if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError(f'Hash mismatch: {path.name}')
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.download')
            temporary.write_bytes(data);temporary.replace(path)
            return str(path.relative_to(ROOT))
        except Exception:
            if attempt==2:raise
            time.sleep(1)


def register_pcan(path):
    source=Path(path).expanduser().resolve()
    data=source.read_bytes()
    if len(data)<256 or data[:2]!=b'MZ':raise ValueError('Expected a Windows x64 PCANBasic.dll')
    offset=struct.unpack_from('<I',data,0x3c)[0]
    if data[offset:offset+4]!=b'PE\0\0' or struct.unpack_from('<H',data,offset+4)[0]!=0x8664:
        raise ValueError('PCANBasic.dll must be x64 (AMD64), matching x64 Python')
    dest=ROOT/'teleop/vendor/pcan/PCANBasic.dll';dest.parent.mkdir(parents=True,exist_ok=True)
    if source!=dest.resolve():shutil.copyfile(source,dest)
    manifest={'description':'Local user-provided PCANBasic x64 runtime. Not part of the source distribution.',
              'files':[{'path':'vendor/pcan/PCANBasic.dll','sha256':hashlib.sha256(data).hexdigest()}]}
    (ROOT/'teleop/vendor_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--teleop-only',action='store_true')
    group.add_argument('--twin-only',action='store_true')
    parser.add_argument('--verify',action='store_true',help='verify local bytes only; no network')
    parser.add_argument('--pcan-dll',type=Path,help='copy and hash an official locally installed x64 PCANBasic.dll')
    args=parser.parse_args(argv)
    rows=resource_plan(not args.twin_only,not args.teleop_only)
    with ThreadPoolExecutor(max_workers=6) as pool:
        completed=list(pool.map(lambda row:get_resource(row,args.verify),rows))
    if args.pcan_dll:register_pcan(args.pcan_dll)
    elif not args.verify:
        candidate=os.environ.get('REBOT_PCAN_DLL')
        if candidate:register_pcan(candidate)
    for name in ['teleop/reports','teleop/runtime','digital_twin/logs']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    print(f'已校验 {len(completed)} 件固定版本资源；没有打开CAN。')
    if not args.twin_only and not (ROOT/'teleop/vendor/pcan/PCANBasic.dll').is_file():
        print('真实遥操还需安装PEAK驱动，并注册本机x64 PCANBasic.dll：')
        print('  python tools/bootstrap.py --pcan-dll "<官方安装目录>/x64/PCANBasic.dll"')
        print('离线测试和数字孪生不需要DLL。')
    return 0


if __name__=='__main__':raise SystemExit(main())
