"""Build a public source ZIP. Excludes drivers, fetched caches and live files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SKIP_PREFIXES=('teleop/vendor/','teleop/runtime/','teleop/reports/',
               'digital_twin/vendor/','digital_twin/logs/','digital_twin/output/','.git/','.venv/')
SKIP_FILES={'teleop/vendor_manifest.json','teleop/site_config.tony_github.json','SOURCE_MANIFEST.json'}


def public_files():
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(ROOT).as_posix()
        if relative in SKIP_FILES or relative.startswith(SKIP_PREFIXES):continue
        if any(p in ('__pycache__','node_modules') for p in path.parts):continue
        if path.suffix.lower() in ('.dll','.pyc','.pyo','.log','.zip','.tmp','.download'):continue
        if path.name.startswith('.env'):continue
        yield path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT.parent/'B601RS_GitHub_Source_20261102.zip')
    args=parser.parse_args()
    files=list(public_files())
    manifest={'release':'2026-11-02','scope':'Public archive files; excludes this manifest itself. Runtime resources fetched separately.',
              'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    mp=ROOT/'SOURCE_MANIFEST.json';mp.write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in files+[mp]:archive.write(path,ROOT.name+'/'+path.relative_to(ROOT).as_posix())
    digest=hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({'archive':str(args.out),'files':len(files)+1,'bytes':args.out.stat().st_size,'sha256':digest},ensure_ascii=False))


if __name__=='__main__':main()
