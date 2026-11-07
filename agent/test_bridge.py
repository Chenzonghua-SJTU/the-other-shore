"""Original repository tools through real official Console and official rollout."""
import json,os,tempfile,hashlib
from pathlib import Path
from rift import Console
from development_worlds.console import DevelopmentConsole
from development_worlds.service import DevelopmentSession,serve_session
from development_worlds import catalog
from .pipeline_bridge.bootstrap import repository
repository()
from .pipeline_bridge.environment import Environment
from .pipeline_bridge.backend import install
from .repository.shore_agent.runtime import ToolRuntime

def main():
    A=Path(__file__).parent;root=A/'_tests';root.mkdir(exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='bridge-',dir=root)).resolve();os.environ['RIFT_OUTPUT_DIR']=str(out)
    reports=[]
    for world in ['2','3']:
        session=DevelopmentSession(DevelopmentConsole(world,catalog.noise_sigma(world),42,catalog.config(world)),out/world/'session')
        with serve_session(session) as credentials:
            env=Environment(Console(credentials['endpoint'],credentials['token']))
            backend=install(env,out/world/'runtime')
            runtime=ToolRuntime(env,out/world/'runtime',config={'symbolic_regression_enabled':False})
            backend.attach(runtime)
            mission=runtime.mission();assert mission['ok'],mission
            assert len(mission['data']['first_light']['probes'][0]['r_obs'])==8
            assert '_official_record' not in mission['data']['first_light']['probes'][0]
            assert abs(runtime.task['position_noise_sigma']-catalog.noise_sigma(world))<1e-12
            if world=='2':
                launch=runtime.execute('insert_probe',{'probes':[{'r0':[1.5,0.],'v0':[0.,.3],'properties':{'mass':1.,'charge':2.}}, {'r0':[1.5,0.],'v0':[0.,.3],'properties':{'mass':2.,'charge':1.}}], 't0':0.,'dt':.05,'n_samples':4})
                assert launch['ok'],launch
                assert env.budget()['energy_used']==2
            fit=runtime.execute('fit_law_and_rollout',{'law_spec':{'ax':'0','ay':'0'},'parameter_spec':[],'train_probe_ids':['probe_0001']})
            assert fit['ok'] and fit['data']['fit_status']!='failed',fit
            lid=fit['data']['law_id']
            valid=runtime.execute('validate_law',{'law_id':lid,'test_probe_ids':['probe_0002']})
            assert valid['ok'],valid
            residual=runtime.execute('analyze_residual',{'law_id':lid,'probe_ids':['probe_0002']})
            assert residual['ok'],residual
            draft=runtime.execute('save_draft',{'law_id':lid,'claims':{'dynamics_order':2,'hidden_matter':False,'time_dependence':False},'physical_explanation':'接口回归检查，不代表科学判断。'})
            assert draft['ok'],draft
            submitted=runtime.finalize_draft('integration_test')
            assert submitted and submitted['status']=='accepted',submitted
            assert session.snapshot()['closed']
            assert backend.rollout_calls>0
            reports.append({'world':world,'official_rollouts':backend.rollout_calls,'console_closed':True,'original_runtime':type(runtime).__module__})
    repository()
    (out/'report.json').write_text(json.dumps(reports,indent=2))
    print('PASS',out,json.dumps(reports),flush=True)
if __name__=='__main__':main()
