"""Export read-only experiment views for the legacy frontend, never source code.

The canonical experiment stays in official_run/agent. No environment or scorer
files are touched. Session credentials, dependency caches and source snapshots
are excluded. Hard links are refreshed atomically for changed source inodes.
"""
from pathlib import Path
import os,json,sys,time
src=Path(sys.argv[1]).resolve();dst=Path(sys.argv[2]).resolve()
assert src.name==dst.name and src.parent.name==dst.parent.name=='_experiments'
assert src.parent.parent.name==dst.parent.parent.name=='agent'
assert src!=dst
skip={'submission','pysr-service','session','tmp','bridge-cache','__pycache__','model-logs','integration-preflight'}
dst.mkdir(parents=True,exist_ok=True)
marker=dst/'frontend-export-origin.json'
if marker.exists():assert json.loads(marker.read_text())['canonical_directory']==str(src)
else:marker.write_text(json.dumps({'canonical_directory':str(src),'purpose':'read-only legacy frontend projection'},indent=2))
while True:
 for root,dirs,files in os.walk(src):
  dirs[:]=[x for x in dirs if x not in skip]
  for name in files:
   p=Path(root)/name
   if p.is_symlink() or name.endswith('.tmp'):continue
   q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True)
   try:
    if q.exists() and os.path.samestat(p.stat(),q.stat()):continue
    temp=q.with_name(q.name+'.export-tmp')
    if temp.exists():temp.unlink()
    os.link(p,temp);os.replace(temp,q)
   except FileNotFoundError:continue
 if (src/'report.json').exists():break
 time.sleep(15)
print('FRONTEND_EXPORT_COMPLETE',src.name,flush=True)
