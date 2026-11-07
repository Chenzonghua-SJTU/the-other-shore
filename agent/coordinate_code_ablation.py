"""Keep authenticated Docker RPC and SSH forwarding alive for an experiment."""
import base64,json,os,secrets,shlex,subprocess,sys,time,urllib.request,tempfile
from pathlib import Path
A=Path(__file__).resolve().parent;ROOT=A.parent;BASE=A/'_analysis/code-ablation';OUT=BASE/('session-'+secrets.token_hex(4));OUT.mkdir(parents=True,exist_ok=True)
token=secrets.token_urlsafe(40);os.environ['SHORE_CODE_SERVICE_TOKEN']=token
log=(OUT/'broker.log').open('a')
broker=subprocess.Popen([sys.executable,'-B','-m','agent.code_broker'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=log)
forward=None
try:
    from agent.pipeline_bridge.code_transport import install
    for i in range(60):
        try:receipt=install('http://127.0.0.1:18993');break
        except OSError:time.sleep(.25)
    else:raise RuntimeError('Broker startup failed')
    from agent.repository.shore_agent.code_execution import execute_code
    tests=[]
    cases=[('numeric',"import numpy as np, scipy, sympy, matplotlib\nx=read_json('observations/test.json')\nresult={'sum':float(np.sum(x['values']))}\nfrom pathlib import Path\nPath('output/check.txt').write_text('ok')",30,{'observations/test.json':base64.b64encode(b'{"values":[1,2,3]}').decode()},'success'),
      ('exception',"raise ValueError('intentional preflight error')",10,{},'error'),
      ('isolation',"import os,socket\nfrom pathlib import Path\ns=socket.socket();s.settimeout(1)\ntry:\n s.connect(('1.1.1.1',53));blocked=False\nexcept OSError:blocked=True\nresult={'network_blocked':blocked,'host_project_visible':Path('/inspire/hdd/project').exists(),'credential_visible':'SHORE_CODE_SERVICE_TOKEN' in os.environ}",10,{},'success'),
      ('timeout','while True: pass',1,{},'timeout')]
    for name,code,timeout,files,status in cases:
        r=execute_code(code,{}, {},OUT/('test-'+name),{'timeout_seconds':timeout},timeout,files=files)
        assert r['status']==status,(name,r['status'])
        if name=='numeric':assert r['result']['sum']==6 and (OUT/'test-numeric/check.txt').read_text()=='ok'
        if name=='isolation':assert r['result']=={'network_blocked':True,'host_project_visible':False,'credential_visible':False}
        tests.append({'name':name,'status':r['status'],'elapsed_seconds':r['elapsed_seconds']})
    forward=subprocess.Popen(['ssh','-N','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-R','127.0.0.1:18993:127.0.0.1:18993','qz-infer'],stdout=log,stderr=log)
    time.sleep(2);assert forward.poll() is None,'SSH reverse forwarding failed'
    check="import json,os,sys;from pathlib import Path;d=json.load(sys.stdin);os.environ['SHORE_CODE_SERVICE_TOKEN']=d['token'];sys.path.insert(0,d['root']);from agent.pipeline_bridge.code_transport import install;r=install(d['endpoint']);from agent.repository.shore_agent.code_execution import execute_code;x=execute_code('result=6*7',{}, {},Path(d['root'])/('agent/_analysis/code-remote-smoke-'+d['smoke_id']),{'timeout_seconds':10},10);assert x['status']=='success' and x['result']==42;print(json.dumps({'remote_execution':True,'receipt':r}))"
    data={'token':token,'root':'/inspire/hdd/project/26-the-other-shore/263140062/official_run','endpoint':'http://127.0.0.1:18993','smoke_id':secrets.token_hex(6)}
    r=subprocess.run(['ssh','qz-infer','/opt/shore-test5-forge2/bin/python -B -c '+shlex.quote(check)],input=json.dumps(data),text=True,capture_output=True,timeout=45)
    assert r.returncode==0,r.stderr[-1000:]
    evidence={'passed':True,'tests':tests,'sandbox':receipt,'remote_smoke':json.loads(r.stdout)}
    (BASE/'preflight.json').write_text(json.dumps(evidence,indent=2));print('CODE_PREFLIGHT_PASS',json.dumps(evidence),flush=True)
    while not (BASE/'LAUNCH_READY').exists():
        assert broker.poll() is None and forward.poll() is None;time.sleep(2)
    launch="import json,os,sys,subprocess;from pathlib import Path;d=json.load(sys.stdin);os.environ['SHORE_CODE_SERVICE_TOKEN']=d['token'];a=Path(d['root'])/'agent';f=(a/'E15-code3-launcher.log').open('w');p=subprocess.Popen(['/opt/shore-test5-forge2/bin/python','-B',str(a/'launch_code3.py')],cwd=d['root'],env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid}))"
    r=subprocess.run(['ssh','qz-infer','/opt/shore-test5-forge2/bin/python -B -c '+shlex.quote(launch)],input=json.dumps(data),text=True,capture_output=True,timeout=30)
    assert r.returncode==0,r.stderr[-1000:];print('CODE_EXPERIMENT_LAUNCHED',r.stdout.strip(),flush=True)
    old=None
    while True:
        assert broker.poll() is None and forward.poll() is None,'Code transport stopped'
        cmd="python3 -c "+shlex.quote("from pathlib import Path;p=Path('/inspire/hdd/project/26-the-other-shore/263140062/official_run/agent/e15-latest.json');print(p.read_text() if p.exists() else '{}')")
        r=subprocess.run(['ssh','qz-infer',cmd],capture_output=True,text=True,timeout=30)
        if r.returncode==0:
            state=json.loads(r.stdout)
            if state!=old:print('CODE_PROGRESS',json.dumps(state),flush=True);old=state
            if state.get('status') in ('finished','failed'):break
        time.sleep(20)
finally:
    if forward:forward.terminate()
    broker.terminate();log.close()
