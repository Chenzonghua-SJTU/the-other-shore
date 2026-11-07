"""Three registered repeats; unchanged original GitHub Agent through official pipeline."""
from pathlib import Path
import argparse,copy,hashlib,json,os,queue,shutil,sys,time
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor,as_completed
from development_worlds.runner import run_episode
from .pipeline_bridge.bootstrap import repository
repository()
from .pipeline_bridge.service import symbolic_service
A=Path(__file__).resolve().parent;ROOT=A.parent

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def fixed_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0]!='agent'}
def freeze(target):
    target.mkdir(parents=True)
    for name in ['__init__.py','run.py','models.py','models.json','strategy.json','repository-config.json','repository-source.json','github-parity.json']:
        shutil.copy2(A/name,target/name)
    for name in ['repository','pipeline_bridge']:shutil.copytree(A/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(A/'vendor_runtime',target/'vendor_runtime',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc','lock.pid','*.pidfile'))
    return {str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file() and 'vendor_runtime' not in p.parts}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=1);parser.add_argument('--parallel',type=int,default=1);parser.add_argument('--label',default='OT02');args=parser.parse_args()
    if args.parallel!=1:parser.error('Current user resource allocation permits only model port 8000')
    output=A/'_experiments'/(args.label+'-'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S'));output.mkdir(parents=True)
    frozen=output/'submission/agent';hashes=freeze(frozen);fixed=fixed_hashes()
    save(output/'source-hashes.json',hashes);save(output/'official-files-before.json',fixed)
    registration={'experiment':args.label,'commit':json.loads((frozen/'repository-source.json').read_text())['github_commit'],'repeats':args.repeats,'worlds':list(range(1,7)),
        'seed':42,'parallel':args.parallel,'allowed_model_ports':[8000],'success_definition':'score >= 90 and complete public fixed-case evaluation',
        'sampling':'model seed unchanged from original config; no session retries or best-of selection',
        'canary':'R01 world 2 retained; scientific incomplete/submission failure remains data, abort only on entrypoint error',
        'actual_environment':'Unmodified official Console determines observation noise and budgets; config.environment describes repository reference run only',
        'settings':json.loads((frozen/'repository-config.json').read_text()),'scope':'official public development pipeline, not private organizer leaderboard'}
    save(output/'registration.json',registration)
    latest={'directory':str(output),'frozen_agent':str(frozen),'status':'warming_pysr','registered_attempts':6*args.repeats}
    save(A/'official-latest.json',latest);print('OUTPUT',output,flush=True)
    ports=queue.Queue()
    for port in range(8000,8000+args.parallel):ports.put(port)
    def one(repeat,world,endpoint):
        port=ports.get();folder=output/f'R{repeat:02d}'/f'world-{world}'
        try:
            command=[sys.executable,'-I','-B',str(ROOT/'scripts/run_strategy.py'),'--agent-dir',str(frozen),'--models-file',str(frozen/'models.json'),'--',
                '--base-url',f'http://127.0.0.1:{port}/v1','--model-id','qwen3.8-27b','--symbolic-endpoint',endpoint,'--max-tokens','4096']
            print('START',repeat,world,port,flush=True)
            result=run_episode(str(world),folder,command,seed=42,startup_seconds=900,evaluate=True)
            result.update(repeat=repeat,directory=str(folder))
            print('DONE',repeat,world,result.get('has_valid_submission'),(result.get('score') or {}).get('total'),flush=True)
            return result
        finally:ports.put(port)
    results=[];error=None
    try:
        with symbolic_service(output,size=args.parallel,timeout=900) as endpoint:
            latest['status']='running';save(A/'official-latest.json',latest)
            canary=one(1,2,endpoint);results.append(canary);save(output/'progress.json',results)
            if canary.get('agent_exit_code') not in (0,2) or not (output/'R01/world-2/agent-output/runtime-provenance.json').exists():
                raise RuntimeError('Canary official entrypoint/submission did not complete; retained its logs and stopped remaining registered attempts.')
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                jobs={pool.submit(one,r,w,endpoint):(r,w) for r in range(1,args.repeats+1) for w in range(1,7) if (r,w)!=(1,2)}
                for job in as_completed(jobs):
                    r,w=jobs[job]
                    try:results.append(job.result())
                    except Exception as exc:results.append({'world':str(w),'repeat':r,'harness_error':type(exc).__name__+': '+str(exc)[:1000]})
                    save(output/'progress.json',results)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)[:1500]
    rows=[]
    for r in sorted(results,key=lambda x:(x['repeat'],int(x['world']))):
        s=r.get('score') or {};complete=bool(s) and not s.get('invalid_cases') and not r.get('evaluation_error')
        rows.append({'repeat':r['repeat'],'world':r['world'],'accepted':r.get('has_valid_submission',False),'exit_code':r.get('agent_exit_code'),
            'total':s.get('total'),'prediction':s.get('prediction'),'law':s.get('law'),'saving':s.get('saving'),
            'complete_prediction_evaluation':complete and s.get('prediction') is not None,'success':complete and (s.get('total') or 0)>=90,
            'directory':r.get('directory'),'error':r.get('harness_error') or r.get('evaluation_error')})
    report={'registration':registration,'results':results,'summary_rows':rows,'error':error,
        'source_integrity':all(hashlib.sha256((frozen/k).read_bytes()).hexdigest()==v for k,v in hashes.items()),
        'official_files_unchanged':fixed==fixed_hashes(),'official_fixed_file_count':len(fixed),
        'accepted':sum(r['accepted'] for r in rows),'success_count':sum(r['success'] for r in rows),'attempted':len(rows),
        'mean_total':sum(r['total'] or 0 for r in rows)/len(rows) if rows else None}
    save(output/'report.json',report)
    latest.update(status='failed' if error else 'finished',attempted=len(rows),accepted=report['accepted'],success_count=report['success_count'],mean_total=report['mean_total'],error=error)
    save(A/'official-latest.json',latest);print('FINAL',json.dumps(latest),flush=True)
if __name__=='__main__':main()
