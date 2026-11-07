"""Development-only deployment admission; no scientific Agent modification."""
import fcntl,json,os,time
from pathlib import Path

def acquire(agent_dir,output,args):
    config_path=Path(agent_dir)/'resource-control.json'
    if not config_path.exists():return None
    config=json.loads(config_path.read_text())
    if config.get('mode')=='drain':
        (Path(output)/'deployment-deferred.json').write_text(json.dumps({'reason':'User requested newer GitHub Agent; no mission started'}))
        raise SystemExit(75)
    lock_path=Path(config['lock_file']);lock_path.parent.mkdir(parents=True,exist_ok=True)
    state=Path(output)/'resource-admission.json'
    state.write_text(json.dumps({'status':'queued_before_mission','actual_model_port':8000,'pid':os.getpid()}))
    lease=lock_path.open('a')
    fcntl.flock(lease.fileno(),fcntl.LOCK_EX)
    for pid in config.get('legacy_pids',[]):
        while True:
            try:os.kill(pid,0)
            except ProcessLookupError:break
            time.sleep(.2)
    args.base_url='http://127.0.0.1:8000/v1'
    state.write_text(json.dumps({'status':'admitted','actual_model_port':8000,'pid':os.getpid()}))
    return lease
