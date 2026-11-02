"""Check release hygiene and local documentation links, without device access."""
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from make_source_zip import ROOT,public_files


def main():
    files=list(public_files());issues=[]
    private=re.compile(r'(?:[A-Za-z]:[/\\]+Users[/\\]+(?:888|[^/\\\s"<>]+)[/\\]+|D:[/\\]+act[/\\]+|wxid_[A-Za-z0-9]+|sk-[A-Za-z0-9_-]{24,})')
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in ('.jpg','.png'):continue
        text=p.read_text(encoding='utf-8-sig')
        if p.name!='audit_source.py' and private.search(text):issues.append({'path':rel,'reason':'possible personal path or secret'})
        if p.suffix=='.svg':ET.fromstring(text)
        if p.suffix=='.md':
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                target=link.strip('<>').split('#')[0]
                if not target or '://' in target or target.startswith('mailto:'):continue
                if not (p.parent/target).exists():issues.append({'path':rel,'reason':'missing local link: '+target})
    print(json.dumps({'public_files':len(files),'issues':issues},ensure_ascii=False))
    return 1 if issues else 0


if __name__=='__main__':raise SystemExit(main())
