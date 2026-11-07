"""Real Console -> filter -> warmed PySR -> raw fit -> official submission.

An integration fixture, not an Agent trial and not included in scientific scores.
"""
from pathlib import Path
import copy,hashlib,json,os
from rift import Console
from development_worlds.console import DevelopmentConsole
from development_worlds.service import DevelopmentSession,serve_session
from development_worlds import catalog
from ..pipeline_bridge.environment import Environment
from ..pipeline_bridge.backend import install as install_backend
from ..pipeline_bridge.reliability import configure
from ..pipeline_bridge.filter_tool import install as install_filter
from ..pipeline_bridge.filter_capability import verify
from ..repository.shore_agent.runtime import ToolRuntime


def run(output, endpoint):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    os.environ['RIFT_OUTPUT_DIR']=str(output)
    capability=verify(endpoint,wait_timeout=60.)
    session=DevelopmentSession(DevelopmentConsole('2',catalog.noise_sigma('2'),731,catalog.config('2')),output/'session')
    with serve_session(session) as credentials:
        env=Environment(Console(credentials['endpoint'],credentials['token']));env.ruleset='e5'
        backend=install_backend(env,output/'runtime')
        rt=ToolRuntime(env,output/'runtime',config={'symbolic_regression_enabled':True,'symbolic_regression_endpoint':endpoint,'symbolic_regression_seed':42})
        backend.attach(rt);configure(rt);install_filter(rt,enabled=True)
        assert rt.mission()['ok']
        probes=[{'r0':[1.5,0.],'v0':[0.,.3],'properties':{'mass':1.,'charge':1.}},
                {'r0':[1.5,0.],'v0':[0.,.3],'properties':{'mass':2.,'charge':1.}},
                {'r0':[1.5,0.],'v0':[0.,.3],'properties':{'mass':1.,'charge':2.}},
                {'r0':[-1.5,.5],'v0':[.2,0.],'properties':{'mass':1.,'charge':1.}}]
        inserted=rt.execute('insert_probe',{'probes':probes,'t0':0.,'dt':.2,'n_samples':50});assert inserted['ok'],inserted
        ids=[p['probe_id'] for p in inserted['data']['probes']]
        before=copy.deepcopy(rt.store.records['probe'])
        filtered=rt.execute('denoise_trajectories',{'train_probe_ids':ids,'time_budget_seconds':2.});assert filtered['ok'],filtered
        fid=filtered['data']['preprocessing_id']
        assert any(r['status']=='filtered' for r in filtered['data']['per_probe'])
        searched=rt.execute('symbolic_regression',{'train_probe_ids':ids,'time_budget_seconds':45.,'preprocessing_id':fid});assert searched['ok'],searched
        assert searched['data']['preprocessing']['worker_consumption_verified']
        assert rt.store.records['probe']==before,'search mutated acquired observations'
        # A generic zero hypothesis exercises the full raw-data rollout path;
        # its accuracy is deliberately not a pass criterion or reported score.
        fitted=rt.execute('fit_law_and_rollout',{'law_spec':{'ax':'0','ay':'0'},'parameter_spec':[],'train_probe_ids':ids});assert fitted['ok'],fitted
        lid=fitted['data']['law_id']
        validated=rt.execute('validate_law',{'law_id':lid,'test_probe_ids':['probe_0002']});assert validated['ok'],validated
        assert rt.store.records['probe']==before
        args={'law_id':lid,'claims':{'dynamics_order':2,'hidden_matter':False,'time_dependence':False},'physical_explanation':'滤波接口集成回归检查，不是科研测评成绩。'}
        assert rt.execute('save_draft',args)['ok']
        submitted=rt.execute('submit',args)
        if not submitted['ok'] and submitted['error']['code']=='scientific_review_needed':submitted=rt.execute('submit',args)
        assert submitted['ok'] and session.snapshot()['closed'],submitted
        report={'passed':True,'scope':'Integration only, excluded from 30 scientific trials','real_pysr_search':True,'worker_capability':capability['verified_capability'],'filtered_probes':sum(r['status']=='filtered' for r in filtered['data']['per_probe']),'worker_consumption_verified':True,'raw_probe_records_unchanged':True,'official_rollouts':backend.rollout_calls,'accepted':True,'search_id':searched['data']['preprocessing']['search_id']}
        assert backend.rollout_calls>0
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('FILTER_INTEGRATION_PASS',json.dumps(report,ensure_ascii=False),flush=True)
    return report
