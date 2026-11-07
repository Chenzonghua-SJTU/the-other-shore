"""Late failed/exported candidates cannot replace the original Agent-selected pair."""
import json,os,tempfile
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

class Spy:
    def __init__(self,console):self.console=console;self.reject=False;self.saved=[];self.submitted=[]
    def __getattr__(self,k):return getattr(self.console,k)
    def save_draft(self,**args):
        if self.reject:return {'accepted':False,'submitted':False}
        reply=self.console.save_draft(**args)
        if reply['accepted']:self.saved.append(json.loads(Path(args['claims']).read_text()))
        return reply
    def submit(self,**args):
        reply=self.console.submit(**args)
        if reply['accepted']:self.submitted.append(json.loads(Path(args['claims']).read_text()))
        return reply

def main():
    A=Path(__file__).parent;out=Path(tempfile.mkdtemp(prefix='selected-pair-',dir=A/'_tests')).resolve();os.environ['RIFT_OUTPUT_DIR']=str(out)
    session=DevelopmentSession(DevelopmentConsole('2',catalog.noise_sigma('2'),42,catalog.config('2')),out/'session')
    with serve_session(session) as credentials:
        spy=Spy(Console(credentials['endpoint'],credentials['token']));env=Environment(spy);backend=install(env,out/'runtime');rt=ToolRuntime(env,out/'runtime',config={'symbolic_regression_enabled':False});backend.attach(rt)
        m=rt.mission();assert len(m['data']['first_light']['probes'][0]['times'])==8
        assert 'dt' not in m['data']['first_light']['probes'][0]['initial_conditions']
        assert m['data']['first_light']['probes'][0]['a_est_method']=='velocity_nonuniform_centered_difference'
        fitargs={'law_spec':{'ax':'0','ay':'0'},'parameter_spec':[],'train_probe_ids':['probe_0001']}
        f=rt.execute('fit_law_and_rollout',fitargs);assert f['ok'],f;l1=f['data']['law_id']
        assert rt.execute('validate_law',{'law_id':l1,'test_probe_ids':['probe_0002']})['ok']
        claims1={'dynamics_order':2,'hidden_matter':False,'time_dependence':False};claims2={**claims1,'hidden_matter':True}
        assert rt.execute('save_draft',{'law_id':l1,'claims':claims1,'physical_explanation':'selected first pair'})['ok']
        f2=rt.execute('fit_law_and_rollout',fitargs);assert f2['ok'];l2=f2['data']['law_id']
        spy.reject=True
        rejected=rt.execute('save_draft',{'law_id':l2,'claims':claims2,'physical_explanation':'rejected later pair'})
        assert not rejected['ok'] and rt.last_valid_draft['law_id']==l1,rejected
        spy.reject=False
        backend.export(out/'abandoned-export',rt._law(l2),claims2,rt._probes(['probe_0001']),'exported but never selected')
        assert spy.saved==[claims1],spy.saved
        result=rt.finalize_draft('selection_regression')
        assert result and result['law_id']==l1,result
        assert spy.submitted==[claims1],spy.submitted
        assert session.snapshot()['closed']
    repository();report={'original_selected_law':l1,'abandoned_law':l2,'rejected_draft_rolled_back':True,'export_did_not_commit':True,'formal_pair_matches_original_agent':True}
    (out/'report.json').write_text(json.dumps(report,indent=2));print('PASS',out,json.dumps(report))
if __name__=='__main__':main()
