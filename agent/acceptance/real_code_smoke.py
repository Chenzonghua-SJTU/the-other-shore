"""Original ToolRuntime file snapshots -> original Docker code -> tool record."""
import copy,json,os
from pathlib import Path
from .real_filter_smoke import (Console,DevelopmentConsole,DevelopmentSession,serve_session,catalog,Environment,install_backend,configure,ToolRuntime)
from ..pipeline_bridge.code_transport import install

def run(output,endpoint):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    os.environ['RIFT_OUTPUT_DIR']=str(output);receipt=install(endpoint)
    session=DevelopmentSession(DevelopmentConsole('2',catalog.noise_sigma('2'),731,catalog.config('2')),output/'session')
    with serve_session(session) as credentials:
        env=Environment(Console(credentials['endpoint'],credentials['token']));env.ruleset='e5'
        backend=install_backend(env,output/'runtime')
        rt=ToolRuntime(env,output/'runtime',config={'symbolic_regression_enabled':False},code_config={'enabled':True,'timeout_seconds':90,'memory_mb':1024})
        backend.attach(rt);configure(rt);assert rt.mission()['ok']
        before=copy.deepcopy(rt.store.records['probe']);pid=next(iter(before));path='observations/'+pid+'.json'
        result=rt.execute('run_code',{'code':f"import numpy as np\np=read_json({path!r})\nresult={{'samples':len(p['times']),'finite':bool(np.isfinite(p['r_obs']).all())}}",'files':[path]})
        assert result['ok'] and result['data']['status']=='success',result
        assert result['data']['result']=={'samples':8,'finite':True},result
        assert rt.store.records['probe']==before
        denied=rt.execute('run_code',{'code':'result=1','files':['observations/not_acquired.json']});assert not denied['ok']
        assert rt.store.records['probe']==before
        report={'passed':True,'scope':'Integration only; excluded from scientific trials','receipt':receipt,'original_tool_runtime':True,'snapshot_samples':8,'raw_records_unchanged':True,'unacquired_files_rejected':True}
    (output/'report.json').write_text(json.dumps(report,indent=2));print('CODE_INTEGRATION_PASS',json.dumps(report),flush=True)
    return report
