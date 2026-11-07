"""Registered BF2 noisy baseline: five repeats per world, no selection/retries."""
import argparse,hashlib,json,os,queue,sys,time,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from .benchmark_official import freeze,fixed_hashes
from .pipeline_bridge.service import symbolic_service
from .e10_episode import run_episode
A=Path(__file__).resolve().parent;ROOT=A.parent

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-pid',type=int);parser.add_argument('--round-start',type=int,default=1);parser.add_argument('--bugfix',default='');args=parser.parse_args()
    arms=[(f'R{args.round_start+i:02d}',True) for i in range(5)]
    seeds={r:42+i for i,(r,_) in enumerate(arms)}
    if args.wait_for_pid:
        print('WAIT_FOR_PREVIOUS',args.wait_for_pid,flush=True)
        while Path('/proc',str(args.wait_for_pid)).exists():time.sleep(1)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    output=A/'_experiments'/('E10-'+stamp);output.mkdir(parents=True)
    frozen=output/'submission/agent';hashes=freeze(frozen);fixed=fixed_hashes()
    save(output/'source-hashes.json',hashes);save(output/'official-files-before.json',fixed)
    config=json.loads((frozen/'repository-config.json').read_text())
    registration={'experiment':'E10','title':'正式测评版六宇宙','commit':json.loads((frozen/'repository-source.json').read_text())['github_commit'],
        'arms':{r:{'name':'带噪声五次baseline','noise':n,'environment_seed':seeds[r]} for r,n in arms},'bugfix':args.bugfix,'worlds':list(range(1,7)),'repeats_per_arm':1,'repeats_per_world':5,'registered_attempts':30,
        'seed':None,'environment_seeds':seeds,'model_seed_fixed':False,'parallel':4,'allowed_model_ports':[8000,8001,8002,8003],
        'settings_shared':config,'scientific_agent_same_for_both_arms':True,'frozen_agent':str(frozen),
        'only_independent_variable':'Registered environment noise seeds 42..46; stochastic model sampling; frozen BF2 Agent',
        'official_default_noise_arm':arms[0][0],'clean_arm_scope':'No clean arm in this baseline',
        'scorer':'Unchanged development_worlds.evaluation.evaluate_law, full public fixed case suite',
        'success_definition':'unrounded official development total >=90 and complete fixed-case prediction',
        'no_retries_or_selection':'One scientific session per world per arm; retain failures; original model transport retries unchanged',
        'scope':'Formal SDK public development pipeline, not private organizer leaderboard'}
    save(output/'registration.json',registration)
    latest={'directory':str(output),'frozen_agent':str(frozen),'experiment':'E10','title':'正式测评版六宇宙','status':'warming_pysr','registered_attempts':30}
    save(A/'e10-latest.json',latest);save(A/'official-latest.json',latest)
    print('OUTPUT',output,flush=True)
    ports=queue.Queue()
    for port in [8000,8001,8002,8003]:ports.put(port)
    results=[];error=None
    def one(round_id,world,noise,endpoint):
        port=ports.get();folder=output/round_id/f'world-{world}'
        try:
            save(output/(round_id+f'-world-{world}-state.json'),{'state':'running','port':port,'noise':noise,'started_at':datetime.now().astimezone().isoformat()})
            command=[sys.executable,'-I','-B',str(ROOT/'scripts/run_strategy.py'),'--agent-dir',str(frozen),'--models-file',str(frozen/'models.json'),'--',
                '--base-url',f'http://127.0.0.1:{port}/v1','--model-id','qwen3.8-27b','--symbolic-endpoint',endpoint,'--max-tokens','4096']
            print('START',round_id,world,'noise='+str(noise),'port='+str(port),flush=True)
            result=run_episode(str(world),folder,command,seed=seeds[round_id],startup_seconds=900,evaluate=True,noise=noise)
            result.update(repeat=round_id,environment_seed=seeds[round_id],noise=noise,directory=str(folder),model_port=port)
            save(output/(round_id+f'-world-{world}-state.json'),{'state':'finished','port':port,'noise':noise})
            print('DONE',round_id,world,result.get('has_valid_submission'),(result.get('score') or {}).get('total'),flush=True)
            return result
        finally:ports.put(port)
    try:
        with symbolic_service(output,size=4,timeout=900) as endpoint:
            latest['status']='running';save(A/'e10-latest.json',latest);save(A/'official-latest.json',latest)
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs={pool.submit(one,r,w,n,endpoint):(r,w,n) for r,n in arms for w in range(1,7)}
                for job in as_completed(jobs):
                    r,w,n=jobs[job]
                    try:results.append(job.result())
                    except Exception as exc:results.append({'world':str(w),'repeat':r,'noise':n,'harness_error':type(exc).__name__,'directory':str(output/r/f'world-{w}')})
                    save(output/'progress.json',results)
                    latest['completed']=len(results);save(A/'e10-latest.json',latest);save(A/'official-latest.json',latest)
    except Exception as exc:error=type(exc).__name__
    rows=[]
    for r in sorted(results,key=lambda x:(x['repeat'],int(x['world']))):
        s=r.get('score') or {};complete=bool(s.get('N_probes')) and not s.get('invalid_cases') and not r.get('evaluation_error')
        rows.append({'repeat':r['repeat'],'world':r['world'],'environment_seed':seeds[r['repeat']],'noise':r['noise'],'accepted':r.get('has_valid_submission',False),'exit_code':r.get('agent_exit_code'),
            'total':s.get('total'),'prediction':s.get('prediction'),'law':s.get('law'),'saving':s.get('saving'),'complete_prediction_evaluation':complete,
            'success':complete and s.get('total',0)>=90,'directory':r.get('directory'),'error':r.get('harness_error') or r.get('evaluation_error')})
    report={'registration':registration,'results':results,'summary_rows':rows,'error':error,
        'source_integrity':all(hashlib.sha256((frozen/k).read_bytes()).hexdigest()==v for k,v in hashes.items()),
        'official_files_unchanged':fixed==fixed_hashes(),'official_fixed_file_count':len(fixed),'arms':{}}
    for r,_ in arms:
        subset=[x for x in rows if x['repeat']==r];report['arms'][r]={'attempted':len(subset),'accepted':sum(x['accepted'] for x in subset),'success_count':sum(x['success'] for x in subset),'mean_total':sum(x['total'] or 0 for x in subset)/len(subset) if subset else None}
    report['per_world']={}
    for w in range(1,7):
        subset=[x for x in rows if int(x['world'])==w]
        values=[x['total'] if x['total'] is not None else 0. for x in subset]
        report['per_world'][str(w)]={'attempted':len(subset),'scores':values,'mean':statistics.mean(values) if values else None,'sample_std':statistics.stdev(values) if len(values)>1 else None,'minimum':min(values) if values else None,'maximum':max(values) if values else None,'success_count':sum(x['success'] for x in subset),'accepted':sum(x['accepted'] for x in subset),'infrastructure_errors':sum(bool(x['error']) for x in subset)}
    report['baseline_complete']=len(rows)==30 and all(v['attempted']==5 for v in report['per_world'].values()) and not error and not any(x['error'] for x in rows)
    report['overall_mean']=statistics.mean([x['total'] or 0. for x in rows]) if rows else None
    report['aggregation_policy']='Keep all registered attempts; invalid scientific submissions score zero; infrastructure exceptions are flagged and baseline_complete=false, never silently discarded.'
    save(output/'report.json',report)
    latest.update(status='failed' if error else 'finished',completed=len(rows),arms=report['arms'],error=error)
    save(A/'e10-latest.json',latest);save(A/'official-latest.json',latest);print('FINAL',json.dumps(latest,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
